# Research Project: Problem Statement, Motivation & Objectives

---

## Problem Statement

**Title:** *Investigating Neural Collapse Geometry and Its Implications for Early Exiting in Convolutional Neural Networks on Fine-Grained Visual Classification*

**Statement:**

Neural Collapse (NC) describes a highly structured geometric state that emerges in the penultimate-layer feature space of deep neural networks during the terminal phase of training: within-class features converge to their class means (NC1), class means form a Simplex Equiangular Tight Frame (NC2), classifier weights align with class means (NC3), and classification simplifies to a nearest-class-center rule (NC4). While this phenomenon is well-documented for the **final** layer under ideal conditions, its behavior across **intermediate layers** — and consequently its impact on **Early Exiting** strategies — remains poorly understood, particularly for **fine-grained** classification tasks where inter-class similarity is high.

This project investigates:

1. **How does Neural Collapse evolve layer-by-layer** through a ResNet architecture trained on the Oxford Flowers 102 dataset (a fine-grained, moderate-scale benchmark)?
2. **How does the strength of collapse at intermediate layers affect the reliability and accuracy of Early Exit classifiers?**
3. **What is the interplay between geometric complexity (high inter-class similarity), class imbalance, and the completeness of Neural Collapse at both early and final layers?**

---

## Motivation

### Why This Problem Matters

The four reference papers in this project collectively reveal a critical gap:

| Paper | What It Shows | The Gap It Leaves |
|-------|---------------|-------------------|
| **Liu & Qin (CVPR 2025)** — OOD Detection via NC | Strong collapse geometry enables reliable distance-based OOD detection | Assumes collapse is strong; does not study what happens at *early/intermediate layers* where collapse is weak |
| **Wang et al. (CVPR 2024)** — Debiased Learning via NC | Shortcut learning and bias *distort* the ideal simplex geometry | Does not study how bias propagates through intermediate layers or affects early exit confidence |
| **Munn et al. (arXiv 2024)** — Geometric Complexity in Transfer Learning | Collapse strength is *not universal* — it depends heavily on dataset geometric complexity | Does not connect this finding to early exiting or inference-time computational savings |
| **Hasegawa & Sato (arXiv 2024)** — Multiplicative Logit Adjustment | Class imbalance distorts collapse geometry; simple logit scaling can repair it | Only studies the final classifier; does not explore whether such corrections help intermediate/early exit classifiers |

**The common thread:** All four papers treat Neural Collapse as a **final-layer** phenomenon. None of them systematically study how collapse *develops progressively* through intermediate convolutional layers, and none connect this progression to Early Exiting — a strategy of enormous practical importance for edge deployment, energy-efficient AI, and real-time inference.

### Why Oxford Flowers 102?

| Property | Why It Matters for This Research |
|----------|--------------------------------|
| **102 fine-grained classes** | Many flower species share very similar petal shapes, colors, and textures. This creates *high geometric complexity* (per Munn et al.), which means collapse is *not* trivially achieved — making the layer-by-layer progression more informative and measurable |
| **~8,189 total images** | A moderate, realistic dataset scale. Unlike massive datasets (ImageNet), the limited data per class means the network cannot simply memorize — the quality of the learned geometry genuinely matters. This also makes imbalance effects (per Hasegawa & Sato) observable without being masked by over-parameterization |
| **Rich visual hierarchy** | Flowers exhibit strong low-level features (edges, color gradients) in early layers and increasingly abstract semantic features (species identity) in deeper layers. This makes it ideal for studying how the NC properties (NC1–NC4) strengthen layer-by-layer |
| **Availability** | Built directly into `torchvision.datasets.Flowers102` — no broken URLs, no manual extraction needed |

### Why ResNet First? (Over VGG, DenseNet, EfficientNet, ViTs)

This is a deliberate, carefully reasoned choice:

**1. ResNet is the canonical architecture for studying Neural Collapse.**
>The original Neural Collapse paper (Papyan, Han & Donoho, 2020) and all four of our reference papers use ResNet variants (ResNet-18, ResNet-50) as primary or baseline architectures. Starting with ResNet ensures our results are **directly comparable** to the existing literature.

**2. Residual connections create a clean layer-wise hierarchy.**
>ResNet is structured into clearly defined *residual blocks* (Layer1 → Layer2 → Layer3 → Layer4 in ResNet-18). Each block produces a distinct feature map at a specific spatial resolution and semantic abstraction level. This makes it natural to:
>- Attach early exit classifiers at each block boundary
>- Extract and analyze features independently at each depth
>- Track the *progression* of NC1–NC4 metrics from shallow to deep
>
>**In contrast:**
>- **VGG** has no skip connections — training is less stable for deep variants, and feature quality degrades more sharply in early layers
>- **DenseNet** concatenates all previous features — this creates dense, entangled representations that make it harder to isolate layer-wise collapse behavior
>- **EfficientNet** uses compound scaling (width × depth × resolution) — conflates multiple dimensions, making it unclear which factor drives collapse changes
>- **ViTs (Vision Transformers)** use self-attention, not convolutions — a fundamentally different mechanism. Studying them is valuable *later*, but starting with them would conflate architectural differences with the collapse phenomenon itself

**3. ResNet is computationally efficient for iterative research.**
>ResNet-18 (11.7M parameters) trains quickly on a single GPU, enabling rapid experimentation with hyperparameters, thresholds, and metrics. This is critical for a research project that requires many training runs to characterize collapse behavior under different conditions.

**4. The residual connections provide a theoretical advantage.**
>Skip connections in ResNet ensure that gradient flow is healthy even in early layers. This means early-layer features are *well-trained* (not starved of gradients), giving us a fair assessment of whether collapse can form at intermediate depths — rather than conflating the absence of collapse with poor training.

**5. Future extensibility.**
>Starting with ResNet establishes a clean baseline. Subsequent experiments can then extend to DenseNet (to study the effect of feature reuse), EfficientNet (to study scaling), and ViTs (to contrast convolutional vs attention-based representations) — with ResNet results as the comparative anchor.

---

## Research Objectives

### Primary Objectives

1. **Quantify layer-wise Neural Collapse in ResNet trained on Oxford Flowers 102**
   - Compute NC1 (within-class covariance), NC2 (cosine similarity deviation from -1/(C-1)), NC3 (classifier-feature alignment), and NC4 (NCC accuracy) at each residual block
   - Track these metrics across training epochs to observe the *temporal and spatial* evolution of collapse

2. **Implement and evaluate Early Exit classifiers at intermediate layers**
   - Attach lightweight classifiers after each residual block (Layer1, Layer2, Layer3) and the penultimate layer (Layer4)
   - Measure accuracy, confidence calibration, and exit rate at each exit point under varying confidence thresholds

3. **Establish the quantitative relationship between collapse strength and early exit reliability**
   - Correlate NC1–NC4 metrics at each layer with the corresponding early exit classifier's accuracy and calibration
   - Test hypothesis: *layers exhibiting stronger collapse produce more accurate and better-calibrated early exits*

### Secondary Objectives

4. **Evaluate the impact of fine-grained geometric complexity on collapse progression**
   - Compare collapse metrics on Oxford Flowers 102 (fine-grained, 102 classes) against a simpler baseline (e.g., CIFAR-10, 10 classes) to validate Munn et al.'s findings that geometric complexity weakens collapse

5. **Investigate logit adjustment at early exits under class imbalance**
   - Apply Hasegawa & Sato's multiplicative logit adjustment to intermediate classifiers and measure whether it stabilizes early exit confidence for underrepresented classes

6. **Assess OOD detection reliability at intermediate layers**
   - Use Liu & Qin's distance-based OOD scoring at each early exit layer and quantify how detection degrades when collapse is partial

---

## Expected Contributions

1. **Layer-wise Neural Collapse characterization** — First systematic measurement of NC1–NC4 progression through intermediate residual blocks on a fine-grained dataset
2. **Collapse-aware Early Exiting** — Empirical evidence linking collapse strength to early exit reliability, providing guidelines for when early exits can be trusted
3. **Cross-paper synthesis** — Unified experimental framework connecting OOD detection, debiasing, geometric complexity, and logit adjustment through the lens of layer-wise representation geometry

---

## Experimental Pipeline (High-Level)

```
Oxford Flowers 102
        |
        v
  ResNet-18 (pretrained on ImageNet, fine-tuned)
        |
        +-- Layer1 output --> Exit 1 classifier --> NC metrics + confidence
        +-- Layer2 output --> Exit 2 classifier --> NC metrics + confidence
        +-- Layer3 output --> Exit 3 classifier --> NC metrics + confidence
        +-- Layer4 output --> Final classifier  --> NC metrics + confidence
                                                       |
                                                       v
                                              Compare NC strength
                                              vs Exit accuracy
                                              vs OOD detection
                                              vs Logit adjustment effect
```
