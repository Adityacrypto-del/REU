import torch
import os
import argparse
from model import EarlyExitResNet
from dataset import get_dataloaders
from evaluate import print_evaluation_report, evaluate_ood_detection
from visualize import generate_all_plots_from_history

def parse_args():
    parser = argparse.ArgumentParser(description="Run Neural Collapse Experiments")
    parser.add_argument('--dataset', type=str, default='flowers102', choices=['flowers102', 'cifar10'],
                        help="Dataset to test geometric complexity (Munn et al.)")
    parser.add_argument('--mla_tau', type=float, default=1.0,
                        help="Multiplicative Logit Adjustment strength (Hasegawa & Sato)")
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
    print("=" * 70)
    print(f"Starting Experiments on Device: {device}")
    print(f"Target Dataset: {args.dataset}")
    print(f"Logit Adjustment (MLA) Tau: {args.mla_tau}")
    print("=" * 70)

    # 1. Load the Target Dataset
    print(f"\n[1] Loading target dataset: {args.dataset}...")
    train_loader, val_loader, class_priors = get_dataloaders(dataset_name=args.dataset, batch_size=32)
    num_classes = len(class_priors)

    # 2. Load Model
    print("\n[2] Initializing model...")
    model = EarlyExitResNet(num_classes=num_classes).to(device)
    
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading weights from {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("WARNING: No checkpoint provided or found. Evaluating with untrained/ImageNet weights.")
        print("Please train the model first using train.py if you expect meaningful results.")

    # 3. Evaluate and Measure Bias & Early Exit Confidence
    print("\n[3] Running Main Evaluation (Bias, MLA, Confidence)...")
    print_evaluation_report(model, val_loader, device, class_priors=class_priors if args.mla_tau > 0 else None)

    # 4. OOD Detection Evaluation
    print("\n[4] Running OOD Detection (Liu & Qin 2025)...")
    ood_dataset = 'cifar10' if args.dataset == 'flowers102' else 'flowers102'
    print(f"Loading OOD dataset: {ood_dataset}...")
    try:
        _, ood_loader, _ = get_dataloaders(dataset_name=ood_dataset, batch_size=32)
        ood_results = evaluate_ood_detection(model, val_loader, ood_loader, device)
        
        print("\nOOD Detection Results (Distance to NC Class Centers):")
        print(f"{'Layer':<7} | {'ID Similarity':<15} | {'OOD Similarity':<15} | {'Gap (ID - OOD)':<15}")
        print("-" * 65)
        for res in ood_results:
            print(f"Exit {res['layer']:<2} | {res['id_mean_sim']:<15.4f} | {res['ood_mean_sim']:<15.4f} | {res['gap']:<15.4f}")
    except Exception as e:
        print(f"Could not run OOD detection: {e}")

    # 5. Visualizations
    print("\n[5] Generating Visualizations...")
    history_path = './results/metrics/full_history.json'
    if os.path.exists(history_path):
        generate_all_plots_from_history(history_path, save_dir='./results/plots')
    else:
        print(f"No training history found at {history_path}. Run train.py first to generate plots.")

    print("\nExperiment run complete!")

if __name__ == "__main__":
    main()
