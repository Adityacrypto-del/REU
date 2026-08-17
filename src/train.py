import os
import time
import json
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

from dataset import get_dataloaders
from model import EarlyExitResNet
from nc_metrics import compute_all_metrics
from feature_extractor import extract_all_features, get_exit_classifier_weights


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct_exits = [0, 0, 0, 0]
    total_samples = 0
    ce_losses = [[], [], [], []]
    correct_preds_batch = []

    batch_idx = 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        logits, _ = model(inputs)

        loss1 = criterion(logits[0], labels)
        loss2 = criterion(logits[1], labels)
        loss3 = criterion(logits[2], labels)
        loss4 = criterion(logits[3], labels)

        loss = loss1 + loss2 + loss3 + loss4
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)

        for i, logit in enumerate(logits):
            _, preds = torch.max(logit, 1)
            correct_exits[i] += torch.sum(preds == labels).item()
            ce_losses[i].append([loss1.item(), loss2.item(), loss3.item(), loss4.item()][i])

        batch_correct = []
        for logit in logits:
            _, preds = torch.max(logit, 1)
            batch_correct.append(int(torch.sum(preds == labels).item()))
        correct_preds_batch.append([batch_idx, batch_correct])
        batch_idx += 1

    epoch_loss = total_loss / total_samples
    epoch_accs = [c / total_samples for c in correct_exits]
    return epoch_loss, epoch_accs, ce_losses, correct_preds_batch


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct_exits = [0, 0, 0, 0]
    total_samples = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        logits, _ = model(inputs)

        loss = sum(criterion(logit, labels) for logit in logits)
        total_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)

        for i, logit in enumerate(logits):
            _, preds = torch.max(logit, 1)
            correct_exits[i] += int((preds == labels).sum().item())

    val_loss = total_loss / total_samples
    val_accs = [c / total_samples for c in correct_exits]
    return val_loss, val_accs


def compute_nc_metrics_all_layers(model, dataloader, device, num_classes=102):
    features_per_layer, all_labels = extract_all_features(model, dataloader, device, num_classes)
    classifier_weights = get_exit_classifier_weights(model)

    metrics_per_layer = []
    kde_densities = []

    for layer_idx in range(4):
        feats = features_per_layer[layer_idx]
        labs = all_labels
        w = classifier_weights[layer_idx].cpu()  # Ensure same device as features

        metrics = compute_all_metrics(feats, labs, w, num_classes)
        metrics_per_layer.append(metrics)

        feats_np = feats.cpu().numpy()
        kde = gaussian_kde(feats_np.T)
        sample_count = min(100, feats_np.shape[0])
        density_sample = kde(feats_np[:sample_count].T)
        kde_densities.append(density_sample)

    return metrics_per_layer, kde_densities


def save_checkpoint(model, optimizer, scheduler, epoch, history, save_dir='./results/checkpoints'):
    os.makedirs(save_dir, exist_ok=True)

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    }, os.path.join(save_dir, f'model_epoch_{epoch}.pt'))

    weights_np = {name: param.detach().cpu().numpy() for name, param in model.named_parameters()}
    np.savez(os.path.join(save_dir, f'weights_epoch_{epoch}.npz'), **weights_np)

    for key, value in history.items():
        try:
            np.save(os.path.join(save_dir, f'{key}_epoch_{epoch}.npy'), np.array(value, dtype=object))
        except Exception:
            pass

    with open(os.path.join(save_dir, f'history_epoch_{epoch}.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f"    💾 Checkpoint saved: epoch {epoch}")


def get_device():
    """Select the best available device: MPS (Apple Silicon) > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description="Train EarlyExitResNet with NC metrics")
    parser.add_argument('--resume', type=str, default=None,
                        help="Path to checkpoint .pt file to resume training from")
    args = parser.parse_args()

    NUM_CLASSES = 102
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LR = 1e-3
    CHECKPOINT_EVERY = 1
    NC_METRICS_EVERY = 1
    RESULTS_DIR = './results'

    device = get_device()
    print(f"Using device: {device}")
    print(f"Configuration: epochs={NUM_EPOCHS}, batch_size={BATCH_SIZE}, lr={LR}")
    print(f"NC metrics computed every {NC_METRICS_EVERY} epochs\n")

    train_loader, val_loader, class_priors = get_dataloaders(batch_size=BATCH_SIZE)
    print(f"  Train batches: {len(train_loader)} | Val batches: {len(val_loader)}\n")

    model = EarlyExitResNet(num_classes=NUM_CLASSES).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # --- Resume from checkpoint ---
    start_epoch = 1
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_accs': [],
        'val_accs': [],
        'nc_metrics': [],
        'learning_rates': [],
    }

    all_weights = []
    all_ce_losses = []
    all_correct_preds = []
    all_kde_densities = []

    if args.resume and os.path.exists(args.resume):
        print(f"\n🔄 Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        remaining_epochs = NUM_EPOCHS - checkpoint['epoch']

        if remaining_epochs > 0 and remaining_epochs != (NUM_EPOCHS - 1):
            # Training was extended (e.g. 50→100). The old scheduler completed
            # its full cosine cycle (LR≈0). Create a fresh cosine schedule for
            # the remaining epochs so LR ramps from LR_max → 0 again.
            # First, reset optimizer LR (it's ≈0 from the old schedule)
            for param_group in optimizer.param_groups:
                param_group['lr'] = LR
                param_group['initial_lr'] = LR
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=remaining_epochs
            )
            print(f"   🔄 Fresh cosine LR schedule: {remaining_epochs} epochs (LR: {LR} → 0)")
        elif checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"   Loaded model from epoch {checkpoint['epoch']}")
        print(f"   Resuming training from epoch {start_epoch}\n")

        # Load existing history from the corresponding history JSON
        history_file = os.path.join(
            os.path.dirname(args.resume),
            f"history_epoch_{checkpoint['epoch']}.json"
        )
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                history = json.load(f)
            print(f"   Loaded history ({len(history['train_loss'])} epochs of prior data)")
        else:
            print(f"   ⚠️  History file not found at {history_file}, starting fresh history")

    print("=" * 70)
    print(f"{'EPOCH':>5} | {'LOSS':>8} | {'Exit1':>7} {'Exit2':>7} {'Exit3':>7} {'Exit4':>7} | {'LR':>10}")
    print("-" * 70)

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        epoch_start = time.time()

        train_loss, train_accs, ce_losses, correct_preds_batch = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        val_loss, val_accs = validate(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_accs'].append(train_accs)
        history['val_accs'].append(val_accs)
        history['learning_rates'].append(current_lr)

        all_ce_losses.append(ce_losses)
        all_correct_preds.append(correct_preds_batch)
        weights_dict = {name: param.detach().cpu().numpy() for name, param in model.named_parameters()}
        all_weights.append(weights_dict)

        epoch_time = time.time() - epoch_start

        print(
            f"  {epoch:3d}   | {train_loss:8.4f} | "
            f"{train_accs[0]:6.2%} {train_accs[1]:6.2%} {train_accs[2]:6.2%} {train_accs[3]:6.2%} | "
            f"{current_lr:.2e} | {epoch_time:.1f}s"
        )
        print(
            f"   val  | {val_loss:8.4f} | "
            f"{val_accs[0]:6.2%} {val_accs[1]:6.2%} {val_accs[2]:6.2%} {val_accs[3]:6.2%}"
        )

        if epoch % NC_METRICS_EVERY == 0 or epoch == 1:
            print(f"\n    [NC] Computing NC metrics on validation set...")
            nc_start = time.time()
            metrics_per_layer, kde_densities = compute_nc_metrics_all_layers(
                model, val_loader, device, NUM_CLASSES
            )

            history['nc_metrics'].append({'epoch': epoch, 'layers': metrics_per_layer})
            all_kde_densities.append(kde_densities)

            for li, m in enumerate(metrics_per_layer):
                print(
                    f"    Layer {li+1}: "
                    f"NC1={m['nc1']:.4f} | "
                    f"NC2={m['nc2_mean']:.4f}±{m['nc2_std']:.4f} (target {m['nc2_target']:.4f}) | "
                    f"NC3={m['nc3']:.4f} | "
                    f"NC4={m['nc4']:.4f}"
                )

            nc_time = time.time() - nc_start
            print(f"    NC metrics computed in {nc_time:.1f}s\n")

        if epoch % CHECKPOINT_EVERY == 0:
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                history,
                os.path.join(RESULTS_DIR, 'checkpoints'),
            )

        print()

    print("Performing post-training analysis...")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.save(os.path.join(RESULTS_DIR, 'all_weights.npy'), np.array(all_weights, dtype=object))
    np.save(os.path.join(RESULTS_DIR, 'all_ce_losses.npy'), np.array(all_ce_losses, dtype=object))
    np.save(os.path.join(RESULTS_DIR, 'all_correct_preds.npy'), np.array(all_correct_preds, dtype=object))
    np.save(os.path.join(RESULTS_DIR, 'all_kde_densities.npy'), np.array(all_kde_densities, dtype=object))

    for label in range(min(10, NUM_CLASSES)):
        plt.figure()
        for epoch_idx in range(len(all_weights)):
            weights = all_weights[epoch_idx].get('classifier.weight')
            if weights is None:
                continue
            epoch_num = start_epoch + epoch_idx
            plt.plot(weights[label], label=f'Epoch {epoch_num}')
        plt.xlabel('Weight Dimension')
        plt.ylabel('Weight Value')
        plt.title(f'Weight Evolution for Label {label}')
        plt.legend(fontsize='small', ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f'weight_evolution_label_{label}.png'))
        plt.close()

    for layer in range(4):
        plt.figure()
        for epoch_idx in range(len(all_kde_densities)):
            epoch_num = start_epoch + epoch_idx
            plt.hist(
                all_kde_densities[epoch_idx][layer],
                bins=50,
                alpha=0.4,
                label=f'Epoch {epoch_num}',
                density=True,
            )
        plt.xlabel('Density Value')
        plt.ylabel('Frequency')
        plt.title(f'Probability Distribution (KDE) for Layer {layer+1}')
        plt.legend(fontsize='small')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f'prob_dist_layer_{layer+1}.png'))
        plt.close()

    save_checkpoint(
        model,
        optimizer,
        scheduler,
        NUM_EPOCHS,
        history,
        os.path.join(RESULTS_DIR, 'checkpoints'),
    )

    os.makedirs(os.path.join(RESULTS_DIR, 'metrics'), exist_ok=True)
    history_path = os.path.join(RESULTS_DIR, 'metrics', 'full_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Full history saved to: {history_path}")

    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    best_accs = [
        max(history['val_accs'][e][i] for e in range(len(history['val_accs'])))
        for i in range(4)
    ]
    for i, acc in enumerate(best_accs):
        print(f"  Exit {i+1}: {acc:.2%}")

    if history['nc_metrics']:
        last_nc = history['nc_metrics'][-1]
        print(f"\nFinal NC metrics (epoch {last_nc['epoch']}):")
        for li, m in enumerate(last_nc['layers']):
            print(f"  Layer {li+1}: NC1={m['nc1']:.4f}  NC3={m['nc3']:.4f}  NC4={m['nc4']:.4f}")

    print(f"Cross-entropy losses saved. Sample: {all_ce_losses[-1]}")
    print(f"Batch-wise correct predictions saved. Sample: {all_correct_preds[-1][:5]}")


if __name__ == "__main__":
    main()