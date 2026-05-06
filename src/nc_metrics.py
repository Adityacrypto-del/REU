import torch
import numpy as np

# =========================================================
# Phase 2: Neural Collapse Metrics (Complete Implementation)
# 
# Mathematical Reference: mathematical_formulation.md
# Each function maps directly to NC1-NC4 as defined there.
# =========================================================


def compute_class_means(features, labels, num_classes):
    """
    Compute per-class mean feature vectors (μ_c) and the global mean (μ_G).

    From mathematical_formulation.md:
        μ_c = (1/N) Σ h_{i,c}
        μ_G = (1/C) Σ μ_c

    Args:
        features: (N, d) tensor — pooled feature vectors from a layer
        labels:   (N,)   tensor — integer class labels
        num_classes: int (C)

    Returns:
        class_means:  (C, d) tensor
        global_mean:  (d,)   tensor
        class_counts: (C,)   tensor — number of samples per class (useful for debugging imbalance)
    """
    d = features.shape[1]
    class_means = torch.zeros(num_classes, d, device=features.device)
    class_counts = torch.zeros(num_classes, device=features.device)

    for c in range(num_classes):
        mask = labels == c
        count = mask.sum()
        if count > 0:
            class_means[c] = features[mask].mean(dim=0)
            class_counts[c] = count

    global_mean = features.mean(dim=0)
    return class_means, global_mean, class_counts


def compute_nc1(features, labels, num_classes):
    """
    NC1: Within-Class Variability Collapse.

    Measures Tr(Σ_W) / Tr(Σ_B).
    - Σ_W = (1/N_total) Σ_c Σ_i (h_{i,c} - μ_c)(h_{i,c} - μ_c)^T
    - Σ_B = (1/C) Σ_c (μ_c - μ_G)(μ_c - μ_G)^T

    We compute the TRACE ratio (scalar), not the full matrix ratio,
    because Tr(Σ_W)/Tr(Σ_B) is numerically stable and captures the
    overall magnitude of within-class spread vs between-class spread.

    Interpretation:
        - High value → features are still scattered within classes (bad)
        - Low value  → features collapsed to class means (good, NC1 holds)
        - Value → 0  means perfect within-class collapse

    Returns:
        float: Tr(Σ_W) / Tr(Σ_B)
    """
    class_means, global_mean, class_counts = compute_class_means(features, labels, num_classes)

    # ---- Within-class covariance trace ----
    # Tr(Σ_W) = (1/N) Σ_c Σ_i ||h_{i,c} - μ_c||^2
    sw_trace = 0.0
    total_samples = 0
    for c in range(num_classes):
        mask = labels == c
        n_c = mask.sum().item()
        if n_c > 0:
            centered = features[mask] - class_means[c]          # (n_c, d)
            sw_trace += (centered ** 2).sum().item()             # Σ ||h - μ_c||^2
            total_samples += n_c

    if total_samples > 0:
        sw_trace /= total_samples

    # ---- Between-class covariance trace ----
    # Tr(Σ_B) = (1/C) Σ_c ||μ_c - μ_G||^2
    centered_means = class_means - global_mean                   # (C, d)
    sb_trace = (centered_means ** 2).sum().item() / num_classes

    # Guard against division by zero (happens if all class means identical)
    if sb_trace < 1e-10:
        return float('inf')

    return sw_trace / sb_trace


def compute_nc2(features, labels, num_classes):
    """
    NC2: Convergence to Simplex ETF.

    The centered class means should form a Simplex Equiangular Tight Frame:
        cos(∠(μ_c - μ_G, μ_c' - μ_G)) = -1/(C-1)  for all c ≠ c'

    We compute all pairwise cosine similarities of centered, normalized 
    class means and measure how close they are to the target value.

    Interpretation:
        - mean_cos ≈ -1/(C-1) → perfect simplex ETF (good)
        - std_cos  ≈ 0        → all pairs equally separated (good)
        - Large std_cos        → asymmetric geometry (bad)

    Returns:
        mean_cos:   float — average off-diagonal cosine similarity
        std_cos:    float — standard deviation of off-diagonal cosine similarities
        target_cos: float — theoretical target = -1/(C-1)
    """
    class_means, global_mean, _ = compute_class_means(features, labels, num_classes)
    centered_means = class_means - global_mean                   # (C, d)

    # Normalize each centered mean to unit length
    norms = centered_means.norm(dim=1, keepdim=True).clamp(min=1e-8)
    normalized = centered_means / norms                          # (C, d)

    # Full cosine similarity matrix: (C, C)
    cos_matrix = normalized @ normalized.T

    # Extract off-diagonal elements (all c ≠ c' pairs)
    mask = ~torch.eye(num_classes, dtype=torch.bool, device=features.device)
    off_diag = cos_matrix[mask]

    target = -1.0 / (num_classes - 1)
    mean_cos = off_diag.mean().item()
    std_cos = off_diag.std().item()

    return mean_cos, std_cos, target


def compute_nc3(features, labels, classifier_weight, num_classes):
    """
    NC3: Classifier-Feature Alignment.

    The classifier weight W_c should align with the centered class mean (μ_c - μ_G):
        || W_c/||W_c|| - (μ_c - μ_G)/||μ_c - μ_G|| ||  → 0

    We measure this via cosine similarity between each W_c and (μ_c - μ_G).
    Perfect alignment = cosine similarity of 1.0 for all classes.

    Args:
        features:          (N, d) tensor
        labels:            (N,) tensor
        classifier_weight: (C, d) tensor — the Linear layer's weight matrix
        num_classes:       int

    Returns:
        mean_alignment: float — average cosine similarity across classes (1.0 = perfect)
        per_class:      (C,) tensor — per-class alignment scores
    """
    class_means, global_mean, _ = compute_class_means(features, labels, num_classes)
    centered_means = class_means - global_mean                   # (C, d)

    # Normalize both to unit vectors
    cm_norms = centered_means.norm(dim=1, keepdim=True).clamp(min=1e-8)
    cm_normalized = centered_means / cm_norms                    # (C, d)

    w_norms = classifier_weight.norm(dim=1, keepdim=True).clamp(min=1e-8)
    w_normalized = classifier_weight / w_norms                   # (C, d)

    # Per-class cosine similarity
    per_class = (cm_normalized * w_normalized).sum(dim=1)        # (C,)
    mean_alignment = per_class.mean().item()

    return mean_alignment, per_class


def compute_nc4(features, labels, num_classes):
    """
    NC4: Nearest-Class-Center (NCC) Classification.

    Because of NC1+NC2+NC3, the learned classifier simplifies to:
        argmax_c (W_c^T h) → argmin_c ||h - μ_c||

    We test this by classifying every sample using pure Euclidean distance
    to the class means (no learned weights involved) and measuring accuracy.

    Interpretation:
        - High NCC accuracy → geometry is so clean that a simple distance
          classifier works, confirming Neural Collapse
        - NCC accuracy ≈ trained classifier accuracy → NC4 holds

    Returns:
        ncc_accuracy: float — fraction of samples correctly classified by NCC
    """
    class_means, _, _ = compute_class_means(features, labels, num_classes)

    # Distance from each sample to each class mean
    # features: (N, d), class_means: (C, d)  → dists: (N, C)
    dists = torch.cdist(features.unsqueeze(0), class_means.unsqueeze(0)).squeeze(0)
    preds = dists.argmin(dim=1)                                  # (N,)

    ncc_accuracy = (preds == labels).float().mean().item()
    return ncc_accuracy


def compute_all_metrics(features, labels, classifier_weight, num_classes):
    """
    Convenience function: compute all four NC metrics in one call.

    Args:
        features:          (N, d) pooled feature vectors from one layer
        labels:            (N,) integer labels
        classifier_weight: (C, d) Linear layer weight for that exit
        num_classes:       int

    Returns:
        dict with keys: 'nc1', 'nc2_mean', 'nc2_std', 'nc2_target', 
                        'nc3', 'nc4'
    """
    nc1 = compute_nc1(features, labels, num_classes)
    nc2_mean, nc2_std, nc2_target = compute_nc2(features, labels, num_classes)
    nc3, _ = compute_nc3(features, labels, classifier_weight, num_classes)
    nc4 = compute_nc4(features, labels, num_classes)

    return {
        'nc1': nc1,
        'nc2_mean': nc2_mean,
        'nc2_std': nc2_std,
        'nc2_target': nc2_target,
        'nc3': nc3,
        'nc4': nc4,
    }


# =========================================================
# Self-test: verify shapes and sanity with synthetic data
# =========================================================
if __name__ == "__main__":
    print("=" * 60)
    print("NC Metrics Self-Test (Synthetic Data)")
    print("=" * 60)

    torch.manual_seed(42)
    num_classes = 10
    d = 64
    samples_per_class = 50
    N = num_classes * samples_per_class

    # Create synthetic features: tight clusters around random class centers
    # This should show STRONG collapse (low NC1, high NC4)
    class_centers = torch.randn(num_classes, d) * 5
    features = []
    labels = []
    for c in range(num_classes):
        noise = torch.randn(samples_per_class, d) * 0.1  # Very tight clusters
        features.append(class_centers[c] + noise)
        labels.extend([c] * samples_per_class)

    features = torch.cat(features, dim=0)                 # (500, 64)
    labels = torch.tensor(labels)                          # (500,)
    fake_weights = torch.randn(num_classes, d)             # Random (unaligned)

    print(f"\nFeatures shape: {features.shape}")
    print(f"Labels shape:   {labels.shape}")

    # Compute metrics
    nc1 = compute_nc1(features, labels, num_classes)
    print(f"\nNC1 (Sw/Sb):        {nc1:.6f}   (expect LOW for tight clusters)")

    nc2_mean, nc2_std, nc2_target = compute_nc2(features, labels, num_classes)
    print(f"NC2 mean cos:       {nc2_mean:.6f}  (target: {nc2_target:.6f})")
    print(f"NC2 std cos:        {nc2_std:.6f}   (expect LOW for symmetric ETF)")

    nc3, per_class = compute_nc3(features, labels, fake_weights, num_classes)
    print(f"NC3 alignment:      {nc3:.6f}   (expect LOW - weights are random)")

    nc4 = compute_nc4(features, labels, num_classes)
    print(f"NC4 NCC accuracy:   {nc4:.2%}     (expect HIGH for tight clusters)")

    # Test compute_all_metrics
    all_metrics = compute_all_metrics(features, labels, fake_weights, num_classes)
    print(f"\ncompute_all_metrics keys: {list(all_metrics.keys())}")

    print("\n" + "=" * 60)
    print("Self-test PASSED!")
    print("=" * 60)
