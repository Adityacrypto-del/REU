import torch
import os
import json
import argparse
import numpy as np
from model import EarlyExitResNet
from dataset import get_dataloaders
from evaluate import evaluate, print_evaluation_report, evaluate_ood_detection, simulate_early_exit
from visualize import (
    generate_all_plots_from_history,
    plot_confidence_histograms,
    plot_exit_sweep,
    plot_ood_auroc_by_layer,
    plot_mla_sweep,
    plot_geometric_complexity_comparison,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Run Neural Collapse Experiments")
    parser.add_argument('--dataset', type=str, default='flowers102', choices=['flowers102', 'cifar10'],
                        help="Dataset to evaluate on")
    parser.add_argument('--checkpoint', type=str, default=None,
                        help="Path to a trained model checkpoint (.pt)")
    return parser.parse_args()

def main():
    args = parse_args()
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("=" * 75)
    print(f"Starting Complete Experiment Suite on Device: {device}")
    print(f"Target Dataset: {args.dataset}")
    print("=" * 75)

    # 1. Load Dataset
    print(f"\n[1] Loading target dataset: {args.dataset}...")
    train_loader, val_loader, class_priors = get_dataloaders(dataset_name=args.dataset, batch_size=32)
    num_classes = len(class_priors)

    # 2. Load Model
    print("\n[2] Initializing model...")
    model = EarlyExitResNet(num_classes=num_classes).to(device)

    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"  Loading weights from {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("  ⚠️ WARNING: No checkpoint provided or found. Evaluating with ImageNet pretrained weights.")

    # 3. Main Evaluation & Calibration
    print("\n[3] Running Standard Evaluation & Calibration Analysis...")
    print_evaluation_report(model, val_loader, device, class_priors=None)

    accuracies, confidences, corrects, preds, labels, class_wise_acc = evaluate(
        model, val_loader, device, class_priors=None
    )
    plot_confidence_histograms(confidences, corrects, save_dir='./results/plots')
    thresholds_sweep = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
    exit_results_sweep = simulate_early_exit(confidences, corrects, thresholds_sweep)
    plot_exit_sweep(exit_results_sweep, save_dir='./results/plots')

    # 4. Multiplicative Logit Adjustment (MLA) Tau Sweep (Hasegawa & Sato 2024)
    print("\n[4] Running Multiplicative Logit Adjustment (MLA) Sweep...")
    tau_values = [0.0, 0.5, 1.0, 1.5, 2.0]
    mla_sweep_results = {}

    for tau in tau_values:
        accs, _, _, _, _, c_accs = evaluate(
            model, val_loader, device, class_priors=class_priors, mla_tau=tau
        )
        variances = [np.var(ca[ca > 0]) if len(ca[ca > 0]) > 0 else 0.0 for ca in c_accs]
        mla_sweep_results[tau] = {
            'accuracies': accs,
            'class_variances': variances
        }
        print(f"  Tau = {tau:<4.1f} | Accuracies: Exit1={accs[0]:.2%} Exit2={accs[1]:.2%} Exit3={accs[2]:.2%} Exit4={accs[3]:.2%}")

    plot_mla_sweep(mla_sweep_results, save_dir='./results/plots')

    # 5. Distance-Based OOD Detection (Liu & Qin 2025)
    print("\n[5] Running Layer-wise OOD Detection (Liu & Qin 2025)...")
    ood_dataset = 'cifar10' if args.dataset == 'flowers102' else 'flowers102'
    print(f"  ID Dataset: {args.dataset}")

    try:
        print(f"  Attempting to load OOD dataset: {ood_dataset}...")
        _, ood_loader, _ = get_dataloaders(dataset_name=ood_dataset, batch_size=32)
    except Exception as e:
        print(f"  ℹ️ {ood_dataset} not found locally ({e}). Creating synthetic Gaussian Noise OOD dataset baseline...")
        import torch.utils.data as data_utils
        # Create 500 synthetic Gaussian noise images as standard baseline OOD
        dummy_images = torch.randn(500, 3, 224, 224)
        dummy_labels = torch.zeros(500, dtype=torch.long)
        ood_loader = data_utils.DataLoader(
            data_utils.TensorDataset(dummy_images, dummy_labels),
            batch_size=32, shuffle=False
        )

    try:
        ood_results = evaluate_ood_detection(model, val_loader, ood_loader, device)

        print("\nOOD Detection Results (Distance to NC Class Centers):")
        print(f"{'Layer':<7} | {'ID Sim':<10} | {'OOD Sim':<10} | {'Gap':<10} | {'AUROC':<10} | {'FPR@95':<10}")
        print("-" * 75)
        for res in ood_results:
            print(f"Exit {res['layer']:<2} | {res['id_mean_sim']:<10.4f} | {res['ood_mean_sim']:<10.4f} | {res['gap']:<10.4f} | {res['auroc']:<10.4f} | {res['fpr95']:<10.4f}")

        plot_ood_auroc_by_layer(ood_results, save_dir='./results/plots')
    except Exception as e:
        print(f"  ⚠️ Could not run OOD detection: {e}")

    # 6. Geometric Complexity Comparison (Munn et al. 2024)
    print("\n[6] Checking Geometric Complexity History Files...")
    fl_hist_path = './results/flowers102/metrics/full_history.json'
    if not os.path.exists(fl_hist_path) and os.path.exists('./results/metrics/full_history.json'):
        fl_hist_path = './results/metrics/full_history.json'
    cf_hist_path = './results/cifar10/metrics/full_history.json'

    if os.path.exists(fl_hist_path):
        generate_all_plots_from_history(fl_hist_path, save_dir='./results/plots')

    if os.path.exists(fl_hist_path) and os.path.exists(cf_hist_path):
        print("  Generating Flowers-102 vs CIFAR-10 Geometric Complexity Comparison Plot...")
        with open(fl_hist_path) as f1, open(cf_hist_path) as f2:
            fl_h = json.load(f1)
            cf_h = json.load(f2)
            plot_geometric_complexity_comparison(fl_h, cf_h, save_dir='./results/plots')
    else:
        print("  ℹ️ CIFAR-10 training history not found yet. Train CIFAR-10 baseline using train.py --dataset cifar10 to generate comparison plot.")

    print("\n" + "=" * 75)
    print("Experiment Suite Execution Completed!")
    print("=" * 75)

if __name__ == "__main__":
    main()
