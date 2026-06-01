import torch
import torch.nn.functional as F
import numpy as np

# =========================================================
# Evaluation Module: Per-Exit Accuracy, Confidence Analysis,
# and Early Exit Simulation
#
# This answers the core practical question:
# "If we deploy early exiting at threshold tau, what accuracy
#  and computational savings do we get?"
# =========================================================


@torch.no_grad()
def evaluate(model, dataloader, device, class_priors=None, mla_tau=1.0):
    """
    Compute per-exit accuracy and collect per-sample confidence data.

    Returns:
        accuracies:     list of 4 floats (per-exit accuracy on full val set)
        all_confidences: list of 4 tensors, each (N,) - max softmax confidence
        all_correct:     list of 4 tensors, each (N,) bool - whether prediction was correct
        all_preds:       list of 4 tensors, each (N,) - predicted class
        all_labels:      (N,) tensor - ground truth labels
        class_wise_acc:  list of 4 arrays - accuracy per class (for Bias analysis - Wang et al.)
    """
    model.eval()
    num_exits = 4
    num_classes = len(class_priors) if class_priors is not None else 102

    correct = [0] * num_exits
    total = 0
    all_confidences = [[] for _ in range(num_exits)]
    all_correct = [[] for _ in range(num_exits)]
    all_preds = [[] for _ in range(num_exits)]
    all_labels = []

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        logits, _ = model(inputs)
        total += labels.size(0)

        for i in range(num_exits):
            # --- Paper 4: Multiplicative Logit Adjustment (Hasegawa & Sato) ---
            # Adjusts logits based on class priors to repair geometry distortion caused by class imbalance
            if class_priors is not None:
                priors_tensor = class_priors.to(device)
                logits[i] = logits[i] + mla_tau * torch.log(priors_tensor + 1e-8)

            probs = F.softmax(logits[i], dim=1)
            max_conf, preds = probs.max(dim=1)
            is_correct = (preds == labels)

            correct[i] += is_correct.sum().item()
            all_confidences[i].append(max_conf.cpu())
            all_correct[i].append(is_correct.cpu())
            all_preds[i].append(preds.cpu())

        all_labels.append(labels.cpu())

    accuracies = [c / total for c in correct]
    all_confidences = [torch.cat(c) for c in all_confidences]
    all_correct = [torch.cat(c) for c in all_correct]
    all_preds = [torch.cat(c) for c in all_preds]
    all_labels = torch.cat(all_labels)

    # --- Paper 2: Debiased Learning (Wang et al) ---
    # By analyzing the variance in class-wise accuracy, we can detect shortcut learning/bias
    class_wise_acc = []
    for i in range(num_exits):
        c_acc = np.zeros(num_classes)
        for c in range(num_classes):
            mask = (all_labels == c)
            if mask.sum() > 0:
                c_acc[c] = all_correct[i][mask].float().mean().item()
        class_wise_acc.append(c_acc)

    return accuracies, all_confidences, all_correct, all_preds, all_labels, class_wise_acc


def simulate_early_exit(all_confidences, all_correct, thresholds):
    """
    Simulate the early exiting strategy at various confidence thresholds.

    For each threshold tau:
      - Process each sample through exits 1 -> 2 -> 3 -> 4 sequentially
      - If max_softmax(exit_i) >= tau, the sample exits at layer i
      - Samples that don't exit at any layer use the final exit (exit 4)

    This models real-world deployment where you want to save compute
    by skipping deeper layers when the network is already confident.

    Args:
        all_confidences: list of 4 (N,) tensors - max softmax per sample per exit
        all_correct:     list of 4 (N,) bool tensors - correctness per sample per exit
        thresholds:      list of float values to sweep

    Returns:
        results: dict mapping threshold -> {
            'per_layer_exits': [count_L1, count_L2, count_L3, count_L4],
            'per_layer_exit_acc': [acc_L1, acc_L2, acc_L3, acc_L4],
            'overall_accuracy': float,
            'avg_layers_used': float,   # proxy for computational cost
            'speedup_ratio': float,     # 4.0 / avg_layers_used
        }
    """
    N = all_confidences[0].shape[0]
    results = {}

    for tau in thresholds:
        remaining = torch.ones(N, dtype=torch.bool)
        layer_exits = [0, 0, 0, 0]
        layer_correct = [0, 0, 0, 0]
        total_correct = 0
        total_layers_used = 0

        for i in range(4):
            if i < 3:
                # Exits 1-3: exit if confidence >= threshold
                exit_mask = remaining & (all_confidences[i] >= tau)
            else:
                # Exit 4 (final): all remaining samples must exit here
                exit_mask = remaining

            n_exited = exit_mask.sum().item()
            n_correct = (exit_mask & all_correct[i]).sum().item()

            layer_exits[i] = n_exited
            layer_correct[i] = n_correct
            total_correct += n_correct
            total_layers_used += n_exited * (i + 1)  # layer index is cost proxy

            remaining = remaining & ~exit_mask

        # Compute per-layer exit accuracy (avoid div by zero)
        layer_exit_acc = []
        for i in range(4):
            if layer_exits[i] > 0:
                layer_exit_acc.append(layer_correct[i] / layer_exits[i])
            else:
                layer_exit_acc.append(0.0)

        avg_layers = total_layers_used / N
        results[tau] = {
            'per_layer_exits': layer_exits,
            'per_layer_exit_acc': layer_exit_acc,
            'overall_accuracy': total_correct / N,
            'avg_layers_used': avg_layers,
            'speedup_ratio': 4.0 / avg_layers if avg_layers > 0 else 1.0,
        }

    return results


def compute_entropy(model, dataloader, device):
    """
    Compute Shannon entropy at each exit for all samples.
    H(p) = -sum(p * log(p))

    Low entropy = high confidence (good candidate for early exit)
    High entropy = uncertain (should continue to deeper layers)

    Returns:
        entropies: list of 4 (N,) tensors
    """
    model.eval()
    all_entropies = [[] for _ in range(4)]

    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            logits, _ = model(inputs)

            for i in range(4):
                probs = F.softmax(logits[i], dim=1)
                # Clamp to avoid log(0)
                log_probs = torch.log(probs.clamp(min=1e-10))
                entropy = -(probs * log_probs).sum(dim=1)  # (B,)
                all_entropies[i].append(entropy.cpu())

    entropies = [torch.cat(e) for e in all_entropies]
    return entropies


@torch.no_grad()
def evaluate_ood_detection(model, in_loader, out_loader, device):
    """
    Paper 1: OOD Detection via Neural Collapse (Liu & Qin 2025)
    Calculates the feature distance to the nearest class center.
    In a strong NC state, ID data is very close to centers, while OOD is far.
    """
    from feature_extractor import extract_all_features, get_exit_classifier_weights
    
    # 1. Get class centers from the classifier weights (NC3 alignment)
    classifier_weights = get_exit_classifier_weights(model)
    
    # 2. Extract features for ID and OOD data
    in_features, _ = extract_all_features(model, in_loader, device, 102)
    out_features, _ = extract_all_features(model, out_loader, device, 10) # assuming CIFAR-10 is OOD
    
    ood_results = []
    
    for i in range(4):
        # We use cosine distance to class centers as the OOD score
        w = classifier_weights[i].to(device) # (num_classes, feature_dim)
        w_norm = F.normalize(w, p=2, dim=1)
        
        # ID distances
        f_in = F.normalize(in_features[i].to(device), p=2, dim=1)
        # Cosine similarity to all centers: (N, C)
        sim_in = torch.matmul(f_in, w_norm.T)
        # Max similarity = min distance to nearest center
        score_in, _ = sim_in.max(dim=1)
        
        # OOD distances
        f_out = F.normalize(out_features[i].to(device), p=2, dim=1)
        sim_out = torch.matmul(f_out, w_norm.T)
        score_out, _ = sim_out.max(dim=1)
        
        ood_results.append({
            'layer': i + 1,
            'id_mean_sim': score_in.mean().item(),
            'ood_mean_sim': score_out.mean().item(),
            'gap': score_in.mean().item() - score_out.mean().item()
        })
        
    return ood_results


def print_evaluation_report(model, dataloader, device, class_priors=None):
    """
    Run full evaluation and print a formatted report.
    """
    print("=" * 70)
    print("EVALUATION REPORT")
    print("=" * 70)

    # 1. Per-exit accuracy
    accuracies, confidences, corrects, preds, labels, class_wise_acc = evaluate(model, dataloader, device, class_priors=class_priors)
    print(f"\nPer-Exit Accuracy (all samples go through each exit):")
    for i, acc in enumerate(accuracies):
        avg_conf = confidences[i].mean().item()
        print(f"  Exit {i+1}: {acc:.2%}  (avg confidence: {avg_conf:.4f})")

    # 2. Early exit simulation
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    exit_results = simulate_early_exit(confidences, corrects, thresholds)

    print(f"\nEarly Exit Simulation:")
    print(f"{'Threshold':>10} | {'Accuracy':>9} | {'Avg Layers':>10} | {'Speedup':>8} | {'L1':>5} {'L2':>5} {'L3':>5} {'L4':>5}")
    print("-" * 70)

    for tau in thresholds:
        r = exit_results[tau]
        exits = r['per_layer_exits']
        print(f"  {tau:>8.2f} | {r['overall_accuracy']:>8.2%} | "
              f"{r['avg_layers_used']:>10.2f} | {r['speedup_ratio']:>7.2f}x | "
              f"{exits[0]:>5d} {exits[1]:>5d} {exits[2]:>5d} {exits[3]:>5d}")

    # 3. Entropy stats
    entropies = compute_entropy(model, dataloader, device)
    print(f"\nShannon Entropy per Exit:")
    for i, ent in enumerate(entropies):
        print(f"  Exit {i+1}: mean={ent.mean():.4f}  std={ent.std():.4f}  "
              f"min={ent.min():.4f}  max={ent.max():.4f}")

    # 4. Class-wise Variance (Bias indicator - Paper 2)
    print(f"\nClass-wise Accuracy Variance (Bias Indicator):")
    for i, c_acc in enumerate(class_wise_acc):
        valid_accs = c_acc[c_acc > 0] # ignore unseen classes in test set
        if len(valid_accs) > 0:
            print(f"  Exit {i+1}: Variance = {np.var(valid_accs):.4f}  |  Min Acc = {np.min(valid_accs):.1%}  |  Max Acc = {np.max(valid_accs):.1%}")

    print("=" * 70)

    return accuracies, exit_results, entropies, class_wise_acc


if __name__ == "__main__":
    from model import EarlyExitResNet
    from dataset import get_dataloaders

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    model = EarlyExitResNet(num_classes=102).to(device)
    _, val_loader, class_priors = get_dataloaders(batch_size=32)

    print_evaluation_report(model, val_loader, device, class_priors=class_priors)
