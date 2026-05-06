import torch
import torch.nn as nn

# =========================================================
# Feature Extractor: Bridges Model → NC Metrics
#
# The model outputs raw spatial feature maps (B, C, H, W).
# NC metrics need flattened, pooled vectors (N, d).
# This module runs the full dataset through the model and
# collects pooled features + labels for metric computation.
# =========================================================


@torch.no_grad()
def extract_all_features(model, dataloader, device, num_classes=102):
    """
    Run the entire dataset through the model and collect:
      - Pooled feature vectors at each of the 4 exit points
      - Corresponding labels

    The model's forward() returns raw feature maps (x1, x2, x3, x4).
    We apply AdaptiveAvgPool2d → Flatten to get (B, d_l) vectors,
    matching what the exit classifiers see internally.

    Args:
        model:      EarlyExitResNet instance
        dataloader: DataLoader for the dataset (train or val)
        device:     torch.device
        num_classes: int (unused here, kept for API consistency)

    Returns:
        features_per_layer: list of 4 tensors
            [0] Layer1 features: (N, 64)
            [1] Layer2 features: (N, 128)
            [2] Layer3 features: (N, 256)
            [3] Layer4 features: (N, 512)
        all_labels: (N,) tensor of integer labels
    """
    model.eval()
    pool = nn.AdaptiveAvgPool2d((1, 1))

    # Accumulators: one list per layer, each list will hold (B, d) chunks
    all_features = [[], [], [], []]
    all_labels = []

    for inputs, labels in dataloader:
        inputs = inputs.to(device)

        # Forward pass — we only need the features, not logits
        _, raw_features = model(inputs)

        for i, feat_map in enumerate(raw_features):
            # feat_map: (B, C_l, H_l, W_l) → pool → (B, C_l, 1, 1) → flatten → (B, C_l)
            pooled = pool(feat_map).flatten(1)
            all_features[i].append(pooled.cpu())

        all_labels.append(labels.cpu())

    # Concatenate all batches into single tensors
    features_per_layer = [torch.cat(f, dim=0) for f in all_features]
    all_labels = torch.cat(all_labels, dim=0)

    return features_per_layer, all_labels


def get_exit_classifier_weights(model):
    """
    Extract the Linear layer weight matrix from each exit classifier.
    
    Each exit is: Sequential(AdaptiveAvgPool2d, Flatten, Linear)
    The Linear layer is always the last module (index [-1]).

    Returns:
        list of 4 tensors: [(C, 64), (C, 128), (C, 256), (C, 512)]
    """
    exits = [model.exit1, model.exit2, model.exit3, model.exit4]
    weights = []
    for exit_head in exits:
        # The Linear layer is the last in the Sequential
        linear_layer = exit_head[-1]
        weights.append(linear_layer.weight.data.clone())
    return weights


if __name__ == "__main__":
    # Quick sanity check with dummy data
    from model import EarlyExitResNet
    from torch.utils.data import DataLoader, TensorDataset

    print("Testing feature extractor...")

    model = EarlyExitResNet(num_classes=10, pretrained=False)
    device = torch.device("cpu")

    # Create a tiny dummy dataset: 20 images, 3x224x224
    dummy_images = torch.randn(20, 3, 224, 224)
    dummy_labels = torch.randint(0, 10, (20,))
    dummy_loader = DataLoader(
        TensorDataset(dummy_images, dummy_labels),
        batch_size=4
    )

    features, labels = extract_all_features(model, dummy_loader, device, num_classes=10)

    print(f"\nExtracted features:")
    for i, f in enumerate(features):
        print(f"  Layer {i+1}: {f.shape}")    # Expect (20, 64), (20, 128), (20, 256), (20, 512)
    print(f"  Labels: {labels.shape}")         # Expect (20,)

    weights = get_exit_classifier_weights(model)
    print(f"\nClassifier weights:")
    for i, w in enumerate(weights):
        print(f"  Exit {i+1}: {w.shape}")      # Expect (10, 64), (10, 128), (10, 256), (10, 512)

    print("\nFeature extractor test PASSED!")
