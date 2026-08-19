import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt
import numpy as np
import os
import json

# =========================================================
# Visualization Module: Paper-Quality Plots
#
# Generates all the figures needed for the research paper:
# 1. NC metric evolution (NC1-NC4 across epochs, per layer)
# 2. Exit accuracy comparison
# 3. Confidence histograms per exit
# 4. Early exit threshold sweep
# 5. NC strength vs exit accuracy correlation
# =========================================================

# Use a clean, publication-quality style
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

LAYER_COLORS = ['#e74c3c', '#e67e22', '#2ecc71', '#3498db']
LAYER_LABELS = ['Layer 1 (64-d)', 'Layer 2 (128-d)', 'Layer 3 (256-d)', 'Layer 4 (512-d)']


def plot_nc_evolution(nc_history, save_dir='./results/plots'):
    """
    Plot NC1-NC4 evolution across training epochs for all 4 layers.

    This is the CENTRAL figure of the research - it shows how Neural Collapse
    develops progressively from shallow to deep layers.

    Args:
        nc_history: list of dicts with keys {'epoch', 'layers': [{'nc1', 'nc2_mean', ...}, ...]}
        save_dir: directory to save the plot
    """
    os.makedirs(save_dir, exist_ok=True)

    epochs = [rec['epoch'] for rec in nc_history]

    # Extract per-layer metric trajectories
    nc1_data = [[rec['layers'][l]['nc1'] for rec in nc_history] for l in range(4)]
    nc2_data = [[rec['layers'][l]['nc2_mean'] for rec in nc_history] for l in range(4)]
    nc3_data = [[rec['layers'][l]['nc3'] for rec in nc_history] for l in range(4)]
    nc4_data = [[rec['layers'][l]['nc4'] for rec in nc_history] for l in range(4)]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Layer-wise Neural Collapse Evolution During Training',
                 fontsize=16, fontweight='bold', y=1.02)

    # NC1: Within-class collapse (lower = better)
    ax = axes[0, 0]
    for l in range(4):
        ax.plot(epochs, nc1_data[l], marker='o', markersize=4,
                color=LAYER_COLORS[l], label=LAYER_LABELS[l], linewidth=2)
    ax.set_title('NC1: Within-Class Collapse (Sw/Sb)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Tr(Sw) / Tr(Sb)')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.annotate('Lower = Better', xy=(0.95, 0.95), xycoords='axes fraction',
                ha='right', va='top', fontsize=9, style='italic', color='gray')

    # NC2: Cosine similarity (closer to -1/(C-1) = better)
    ax = axes[0, 1]
    target = nc_history[0]['layers'][0]['nc2_target']
    for l in range(4):
        ax.plot(epochs, nc2_data[l], marker='s', markersize=4,
                color=LAYER_COLORS[l], label=LAYER_LABELS[l], linewidth=2)
    ax.axhline(y=target, color='black', linestyle='--', alpha=0.5,
               label=f'Target: {target:.4f}')
    ax.set_title('NC2: Mean Pairwise Cosine Similarity')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean Cosine Similarity')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # NC3: Classifier alignment (higher = better)
    ax = axes[1, 0]
    for l in range(4):
        ax.plot(epochs, nc3_data[l], marker='^', markersize=4,
                color=LAYER_COLORS[l], label=LAYER_LABELS[l], linewidth=2)
    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Perfect (1.0)')
    ax.set_title('NC3: Classifier-Feature Alignment')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean Cosine Similarity')
    ax.set_ylim(-0.1, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.annotate('Higher = Better', xy=(0.95, 0.05), xycoords='axes fraction',
                ha='right', va='bottom', fontsize=9, style='italic', color='gray')

    # NC4: NCC accuracy (higher = better)
    ax = axes[1, 1]
    for l in range(4):
        ax.plot(epochs, nc4_data[l], marker='D', markersize=4,
                color=LAYER_COLORS[l], label=LAYER_LABELS[l], linewidth=2)
    ax.set_title('NC4: Nearest-Class-Center Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('NCC Accuracy')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.annotate('Higher = Better', xy=(0.95, 0.05), xycoords='axes fraction',
                ha='right', va='bottom', fontsize=9, style='italic', color='gray')

    plt.tight_layout()
    path = os.path.join(save_dir, 'nc_evolution.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_accuracy_curves(history, save_dir='./results/plots'):
    """
    Plot training and validation accuracy curves for all 4 exits.
    """
    os.makedirs(save_dir, exist_ok=True)

    epochs = list(range(1, len(history['train_accs']) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Training & Validation Accuracy per Exit', fontsize=14, fontweight='bold')

    # Training accuracy
    ax = axes[0]
    for i in range(4):
        train_accs = [history['train_accs'][e][i] for e in range(len(epochs))]
        ax.plot(epochs, train_accs, color=LAYER_COLORS[i], label=LAYER_LABELS[i], linewidth=1.5)
    ax.set_title('Training Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Validation accuracy
    ax = axes[1]
    for i in range(4):
        val_accs = [history['val_accs'][e][i] for e in range(len(epochs))]
        ax.plot(epochs, val_accs, color=LAYER_COLORS[i], label=LAYER_LABELS[i], linewidth=1.5)
    ax.set_title('Validation Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, 'accuracy_curves.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_loss_curve(history, save_dir='./results/plots'):
    """Plot training and validation loss."""
    os.makedirs(save_dir, exist_ok=True)

    epochs = list(range(1, len(history['train_loss']) + 1))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history['train_loss'], label='Train Loss', color='#e74c3c', linewidth=2)
    ax.plot(epochs, history['val_loss'], label='Val Loss', color='#3498db', linewidth=2)
    ax.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (sum of 4 exits)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, 'loss_curve.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_confidence_histograms(all_confidences, all_correct, save_dir='./results/plots'):
    """
    Plot confidence distribution per exit, colored by correct vs incorrect.

    This reveals whether confidence is well-calibrated at each layer:
    - Good calibration: correct predictions have high confidence, incorrect have low
    - Poor calibration: confidence doesn't distinguish correct from incorrect
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Confidence Distribution per Exit (Correct vs Incorrect)',
                 fontsize=14, fontweight='bold')

    for i in range(4):
        ax = axes[i // 2][i % 2]
        conf = all_confidences[i].numpy()
        corr = all_correct[i].numpy()

        ax.hist(conf[corr == True], bins=50, alpha=0.7, color='#2ecc71',
                label='Correct', density=True)
        ax.hist(conf[corr == False], bins=50, alpha=0.7, color='#e74c3c',
                label='Incorrect', density=True)
        ax.set_title(f'Exit {i+1} ({LAYER_LABELS[i]})')
        ax.set_xlabel('Max Softmax Confidence')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, 'confidence_histograms.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_exit_sweep(exit_results, save_dir='./results/plots'):
    """
    Plot accuracy vs computational cost for different early exit thresholds.

    This is the key practical figure: it shows the accuracy-efficiency tradeoff.
    """
    os.makedirs(save_dir, exist_ok=True)

    thresholds = sorted(exit_results.keys())
    accuracies = [exit_results[t]['overall_accuracy'] for t in thresholds]
    avg_layers = [exit_results[t]['avg_layers_used'] for t in thresholds]
    speedups = [exit_results[t]['speedup_ratio'] for t in thresholds]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Early Exit: Accuracy vs Efficiency Tradeoff',
                 fontsize=14, fontweight='bold')

    # Accuracy vs Threshold
    ax = axes[0]
    ax.plot(thresholds, accuracies, 'o-', color='#3498db', linewidth=2, markersize=6)
    ax.set_xlabel('Confidence Threshold')
    ax.set_ylabel('Overall Accuracy')
    ax.set_title('Accuracy vs Threshold')
    ax.grid(True, alpha=0.3)

    # Accuracy vs Speedup (Pareto front)
    ax = axes[1]
    scatter = ax.scatter(speedups, accuracies, c=thresholds, cmap='viridis',
                         s=80, edgecolors='black', linewidth=0.5)
    ax.plot(speedups, accuracies, '--', color='gray', alpha=0.5)
    for t, s, a in zip(thresholds, speedups, accuracies):
        ax.annotate(f'{t:.2f}', (s, a), textcoords="offset points",
                    xytext=(5, 5), fontsize=8)
    ax.set_xlabel('Speedup Ratio (x)')
    ax.set_ylabel('Overall Accuracy')
    ax.set_title('Accuracy vs Speedup (Pareto Front)')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Threshold')

    plt.tight_layout()
    path = os.path.join(save_dir, 'exit_sweep.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_nc_vs_exit_accuracy(nc_history, val_accs_history, save_dir='./results/plots'):
    """
    Scatter plot: NC metric strength at each layer vs exit accuracy at that layer.

    This directly tests the core hypothesis:
    "Stronger Neural Collapse at layer L => more reliable early exit at layer L"
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('NC Strength vs Exit Accuracy (Core Hypothesis)',
                 fontsize=14, fontweight='bold')

    # Collect (nc1, accuracy) and (nc4, accuracy) pairs across all recorded epochs and layers
    nc1_points = []
    nc4_points = []

    for rec in nc_history:
        epoch = rec['epoch']
        # Find matching val accuracy for this epoch
        if epoch - 1 < len(val_accs_history):
            for l in range(4):
                nc1_val = rec['layers'][l]['nc1']
                nc4_val = rec['layers'][l]['nc4']
                exit_acc = val_accs_history[epoch - 1][l]
                nc1_points.append((nc1_val, exit_acc, l))
                nc4_points.append((nc4_val, exit_acc, l))

    # NC1 vs Exit Accuracy (expect negative correlation)
    ax = axes[0]
    for l in range(4):
        layer_nc1 = [p[0] for p in nc1_points if p[2] == l]
        layer_acc = [p[1] for p in nc1_points if p[2] == l]
        ax.scatter(layer_nc1, layer_acc, color=LAYER_COLORS[l],
                   label=LAYER_LABELS[l], s=50, alpha=0.8)
    ax.set_xlabel('NC1 (Sw/Sb) - lower = more collapsed')
    ax.set_ylabel('Exit Accuracy')
    ax.set_title('NC1 vs Exit Accuracy (expect negative correlation)')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # NC4 vs Exit Accuracy (expect positive correlation)
    ax = axes[1]
    for l in range(4):
        layer_nc4 = [p[0] for p in nc4_points if p[2] == l]
        layer_acc = [p[1] for p in nc4_points if p[2] == l]
        ax.scatter(layer_nc4, layer_acc, color=LAYER_COLORS[l],
                   label=LAYER_LABELS[l], s=50, alpha=0.8)
    ax.set_xlabel('NC4 (NCC Accuracy) - higher = more collapsed')
    ax.set_ylabel('Exit Accuracy')
    ax.set_title('NC4 vs Exit Accuracy (expect positive correlation)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, 'nc_vs_accuracy.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_geometric_complexity_comparison(flowers_history, cifar_history, save_dir='./results/plots'):
    """
    Paper 4 (Munn et al. 2024): Geometric Complexity in Transfer Learning.
    Compares NC1 and NC4 evolution between Flowers-102 (high complexity, 102 classes)
    and CIFAR-10 (low complexity, 10 classes).
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Geometric Complexity Impact on Neural Collapse (Flowers-102 vs CIFAR-10)',
                 fontsize=14, fontweight='bold')

    fl_nc = flowers_history.get('nc_metrics', [])
    cf_nc = cifar_history.get('nc_metrics', [])

    if not fl_nc or not cf_nc:
        print("  ⚠️ Skipping geometric complexity plot: NC history missing for one or both datasets")
        return

    fl_epochs = [rec['epoch'] for rec in fl_nc]
    fl_nc1_l4 = [rec['layers'][3]['nc1'] for rec in fl_nc]
    fl_nc4_l4 = [rec['layers'][3]['nc4'] for rec in fl_nc]

    cf_epochs = [rec['epoch'] for rec in cf_nc]
    cf_nc1_l4 = [rec['layers'][3]['nc1'] for rec in cf_nc]
    cf_nc4_l4 = [rec['layers'][3]['nc4'] for rec in cf_nc]

    # NC1 comparison
    ax = axes[0]
    ax.plot(fl_epochs, fl_nc1_l4, 'o-', color='#e74c3c', label='Flowers-102 (Complex, 102 classes)', linewidth=2)
    ax.plot(cf_epochs, cf_nc1_l4, 's-', color='#3498db', label='CIFAR-10 (Simple, 10 classes)', linewidth=2)
    ax.set_title('NC1 Within-Class Variability (Layer 4)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Tr(Sw) / Tr(Sb) (log scale)')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # NC4 comparison
    ax = axes[1]
    ax.plot(fl_epochs, fl_nc4_l4, 'o-', color='#e74c3c', label='Flowers-102 (Complex, 102 classes)', linewidth=2)
    ax.plot(cf_epochs, cf_nc4_l4, 's-', color='#3498db', label='CIFAR-10 (Simple, 10 classes)', linewidth=2)
    ax.set_title('NC4 Nearest-Class-Center Accuracy (Layer 4)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('NCC Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, 'geometric_complexity_comparison.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_ood_auroc_by_layer(ood_results, save_dir='./results/plots'):
    """
    Paper 1 (Liu & Qin 2025): Distance-based OOD Detection across Exits.
    Plots AUROC and FPR@95 across layer depth (Exits 1-4).
    """
    os.makedirs(save_dir, exist_ok=True)

    layers = [res['layer'] for res in ood_results]
    aurocs = [res['auroc'] for res in ood_results]
    fpr95s = [res['fpr95'] for res in ood_results]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = '#2ecc71'
    ax1.set_xlabel('Exit Layer Depth')
    ax1.set_ylabel('OOD AUROC (Higher = Better)', color=color)
    ax1.plot(layers, aurocs, 'o-', color=color, linewidth=2.5, markersize=8, label='AUROC')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0.4, 1.02)
    ax1.set_xticks(layers)
    ax1.set_xticklabels([f'Exit {l} (Layer {l})' for l in layers])
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color = '#e74c3c'
    ax2.set_ylabel('FPR@95 (Lower = Better)', color=color)
    ax2.plot(layers, fpr95s, 's--', color=color, linewidth=2.5, markersize=8, label='FPR@95')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(-0.02, 1.02)

    plt.title('OOD Detection Performance across Exit Layers (Liu & Qin 2025)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(save_dir, 'ood_auroc_layerwise.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_mla_sweep(mla_sweep_results, save_dir='./results/plots'):
    """
    Paper 4 (Hasegawa & Sato 2024): Multiplicative Logit Adjustment.
    Plots validation accuracy and class-wise accuracy variance across tau values per exit.
    """
    os.makedirs(save_dir, exist_ok=True)

    taus = sorted(mla_sweep_results.keys())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Multiplicative Logit Adjustment (MLA) Sweep across Exits', fontsize=14, fontweight='bold')

    # 1. Accuracy vs Tau per exit
    ax = axes[0]
    for i in range(4):
        accs = [mla_sweep_results[t]['accuracies'][i] for t in taus]
        ax.plot(taus, accs, 'o-', color=LAYER_COLORS[i], label=LAYER_LABELS[i], linewidth=2)
    ax.set_xlabel('MLA Tau (Logit Scaling Factor)')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('Exit Accuracy vs Logit Adjustment Tau')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Class Variance vs Tau per exit (Bias indicator)
    ax = axes[1]
    for i in range(4):
        variances = [mla_sweep_results[t]['class_variances'][i] for t in taus]
        ax.plot(taus, variances, 's--', color=LAYER_COLORS[i], label=LAYER_LABELS[i], linewidth=2)
    ax.set_xlabel('MLA Tau (Logit Scaling Factor)')
    ax.set_ylabel('Class-wise Accuracy Variance (Lower = Fairer)')
    ax.set_title('Class Accuracy Variance vs Tau')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, 'mla_tau_sweep.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def generate_all_plots_from_history(history_path, save_dir='./results/plots'):
    """
    Load a saved history JSON and generate all plots.
    Can be run standalone after training is complete.
    """
    with open(history_path, 'r') as f:
        history = json.load(f)

    print("Generating plots from saved history...")
    print(f"  Epochs trained: {len(history['train_loss'])}")
    print(f"  NC metric snapshots: {len(history['nc_metrics'])}")

    plot_loss_curve(history, save_dir)
    plot_accuracy_curves(history, save_dir)

    if history['nc_metrics']:
        plot_nc_evolution(history['nc_metrics'], save_dir)
        plot_nc_vs_exit_accuracy(history['nc_metrics'], history['val_accs'], save_dir)

    print("\nAll plots generated!")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # Usage: python visualize.py path/to/full_history.json
        generate_all_plots_from_history(sys.argv[1])
    else:
        print("Usage: python visualize.py <path_to_history.json>")
        print("  e.g.: python visualize.py ./results/metrics/full_history.json")
        print("\nOr import and call individual plot functions from your training script.")
