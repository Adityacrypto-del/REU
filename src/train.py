import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
import time

from dataset import get_dataloaders
from model import EarlyExitResNet
from nc_metrics import compute_all_metrics
from feature_extractor import extract_all_features, get_exit_classifier_weights


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct_exits = [0, 0, 0, 0] # Track accuracy for each of the 4 exits
    total_samples = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward pass: get predictions from all 4 layers
        logits, features = model(inputs)

        # Compute loss for all exits
        # This trains the backbone and all branch classifiers simultaneously
        loss1 = criterion(logits[0], labels)
        loss2 = criterion(logits[1], labels)
        loss3 = criterion(logits[2], labels)
        loss4 = criterion(logits[3], labels)

        # Simple unweighted sum of losses.
        # (Later we could try weighting deeper layers higher)
        loss = loss1 + loss2 + loss3 + loss4

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)

        # Calculate accuracy for each exit cleanly
        for i in range(4):
            _, preds = torch.max(logits[i], 1)
            correct_exits[i] += torch.sum(preds == labels.data).item()

    epoch_loss = total_loss / total_samples
    epoch_accs = [c / total_samples for c in correct_exits]

    return epoch_loss, epoch_accs


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """Run validation and return loss + per-exit accuracies."""
    model.eval()
    total_loss = 0
    correct_exits = [0, 0, 0, 0]
    total_samples = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        logits, _ = model(inputs)

        loss = sum(criterion(logits[i], labels) for i in range(4))
        total_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)

        for i in range(4):
            _, preds = torch.max(logits[i], 1)
            correct_exits[i] += (preds == labels).sum().item()

    val_loss = total_loss / total_samples
    val_accs = [c / total_samples for c in correct_exits]
    return val_loss, val_accs


def compute_nc_metrics_all_layers(model, dataloader, device, num_classes=102):
    """
    Extract features from all 4 layers and compute NC1-NC4 at each.
    
    Returns:
        list of 4 dicts, one per layer, each containing:
        {'nc1', 'nc2_mean', 'nc2_std', 'nc2_target', 'nc3', 'nc4'}
    """
    features_per_layer, all_labels = extract_all_features(model, dataloader, device, num_classes)
    classifier_weights = get_exit_classifier_weights(model)

    metrics_per_layer = []
    for layer_idx in range(4):
        feats = features_per_layer[layer_idx].to(device)
        labs = all_labels.to(device)
        w = classifier_weights[layer_idx].to(device)

        metrics = compute_all_metrics(feats, labs, w, num_classes)
        metrics_per_layer.append(metrics)

    return metrics_per_layer


def save_checkpoint(model, optimizer, scheduler, epoch, history, save_dir='./results/checkpoints'):
    """Save model weights + full training history."""
    os.makedirs(save_dir, exist_ok=True)

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    }, os.path.join(save_dir, f'model_epoch_{epoch}.pt'))

    with open(os.path.join(save_dir, f'history_epoch_{epoch}.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f"    💾 Checkpoint saved: epoch {epoch}")


def main():
    # ===================== Configuration =====================
    NUM_CLASSES = 102
    BATCH_SIZE = 32
    NUM_EPOCHS = 2
    LR = 1e-3
    CHECKPOINT_EVERY = 1        # Save model every N epochs
    NC_METRICS_EVERY = 1        # Compute NC metrics every N epochs
    RESULTS_DIR = './results'
    # =========================================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Configuration: {NUM_EPOCHS} epochs, batch_size={BATCH_SIZE}, lr={LR}")
    print(f"NC metrics computed every {NC_METRICS_EVERY} epochs\n")

    # 1. Load Data
    print("Loading Oxford Flowers 102...")
    train_loader, val_loader, class_priors = get_dataloaders(batch_size=BATCH_SIZE)
    print(f"  Train batches: {len(train_loader)} | Val batches: {len(val_loader)}\n")

    # 2. Setup Model
    print("Initializing EarlyExitResNet (pretrained=True)...")
    model = EarlyExitResNet(num_classes=NUM_CLASSES).to(device)

    # 3. Optimizer, Loss, Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # 4. History tracker - stores everything for later analysis
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_accs': [],       # list of [exit1_acc, exit2_acc, exit3_acc, exit4_acc]
        'val_accs': [],
        'nc_metrics': [],       # list of dicts: {epoch, layers: [{nc1, nc2_mean, ...}, ...]}
        'learning_rates': [],
    }

    # ===================== Training Loop =====================
    print("=" * 70)
    print(f"{'EPOCH':>5} | {'LOSS':>8} | {'Exit1':>7} {'Exit2':>7} {'Exit3':>7} {'Exit4':>7} | {'LR':>10}")
    print("-" * 70)

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()

        # ---- Train ----
        train_loss, train_accs = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # ---- Validate ----
        val_loss, val_accs = validate(model, val_loader, criterion, device)

        # ---- Step scheduler ----
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        # ---- Log ----
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_accs'].append(train_accs)
        history['val_accs'].append(val_accs)
        history['learning_rates'].append(current_lr)

        epoch_time = time.time() - epoch_start

        print(f"  {epoch:3d}   | {train_loss:8.4f} | "
              f"{train_accs[0]:6.2%} {train_accs[1]:6.2%} {train_accs[2]:6.2%} {train_accs[3]:6.2%} | "
              f"{current_lr:.2e} | {epoch_time:.1f}s")
        print(f"   val  | {val_loss:8.4f} | "
              f"{val_accs[0]:6.2%} {val_accs[1]:6.2%} {val_accs[2]:6.2%} {val_accs[3]:6.2%}")

        # ---- NC Metrics (every N epochs) ----
        if epoch % NC_METRICS_EVERY == 0 or epoch == 1:
            print(f"\n    [NC] Computing NC metrics on validation set...")
            nc_start = time.time()
            metrics_per_layer = compute_nc_metrics_all_layers(
                model, val_loader, device, NUM_CLASSES
            )

            nc_record = {'epoch': epoch, 'layers': metrics_per_layer}
            history['nc_metrics'].append(nc_record)

            for li, m in enumerate(metrics_per_layer):
                print(f"    Layer {li+1}: "
                      f"NC1={m['nc1']:.4f} | "
                      f"NC2={m['nc2_mean']:.4f}±{m['nc2_std']:.4f} (target {m['nc2_target']:.4f}) | "
                      f"NC3={m['nc3']:.4f} | "
                      f"NC4={m['nc4']:.4f}")

            nc_time = time.time() - nc_start
            print(f"    NC metrics computed in {nc_time:.1f}s\n")

        # ---- Checkpoint ----
        if epoch % CHECKPOINT_EVERY == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, history, 
                           os.path.join(RESULTS_DIR, 'checkpoints'))

        print()

    # ===================== Final Save =====================
    print("=" * 70)
    print("Training complete!")
    save_checkpoint(model, optimizer, scheduler, NUM_EPOCHS, history,
                   os.path.join(RESULTS_DIR, 'checkpoints'))

    # Save full history as standalone JSON
    os.makedirs(os.path.join(RESULTS_DIR, 'metrics'), exist_ok=True)
    history_path = os.path.join(RESULTS_DIR, 'metrics', 'full_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Full history saved to: {history_path}")

    # ---- Print Final Summary ----
    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nBest validation accuracies (per exit):")
    best_accs = [max(history['val_accs'][e][i] for e in range(len(history['val_accs']))) 
                 for i in range(4)]
    for i, acc in enumerate(best_accs):
        print(f"  Exit {i+1}: {acc:.2%}")

    if history['nc_metrics']:
        last_nc = history['nc_metrics'][-1]
        print(f"\nFinal NC metrics (epoch {last_nc['epoch']}):")
        for li, m in enumerate(last_nc['layers']):
            print(f"  Layer {li+1}: NC1={m['nc1']:.4f}  NC3={m['nc3']:.4f}  NC4={m['nc4']:.2%}")


if __name__ == "__main__":
    main()
