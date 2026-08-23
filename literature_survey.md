# 📚 Concept-Wise Literature Survey

## 1. The Geometry of Deep Representations (Neural Collapse)
Neural Collapse (NC), first identified by **Papyan, Han, & Donoho (2020)** [1], characterizes a universal geometric phase transition that occurs in the penultimate layer of deep neural networks during the terminal phase of training (when training error reaches zero). Rather than continuously scattered features, representations collapse into a highly structured mathematical state defined by four core properties:
*   **NC1 (Within-Class Variability Collapse):** The intra-class features cluster tightly around their respective class means ($\mu_c$), causing the ratio of within-class covariance to between-class covariance ($Tr(\Sigma_W)/Tr(\Sigma_B)$) to approach zero.
*   **NC2 (Simplex Equiangular Tight Frame):** The class means center themselves relative to the global mean ($\mu_G$) and arrange as a Simplex Equiangular Tight Frame (ETF). In this configuration, all class means have equal norm, and the pairwise cosine similarity between any two class means is exactly $-1/(C-1)$ (where $C$ is the number of classes).
*   **NC3 (Classifier-Feature Alignment):** The classifier weight vectors ($W_c$) align perfectly with the centered class means, meaning $W_c \propto (\mu_c - \mu_G)$.
*   **NC4 (Nearest-Center Classification):** The network's learned classifier becomes equivalent to a simple Euclidean distance-based Nearest-Class-Center (NCC) rule, rendering learned classifier boundaries redundant.

---

## 2. Layer-wise Representation Dynamics
Traditional CNN theory describes representation learning as a hierarchical progression, where shallow layers extract visual primitives (edges, color blobs) and deep layers assemble them into complex semantic parts (**Zeiler & Fergus, 2014** [2]). 
However, standard CNN training evaluates representations only at the final layer, leaving intermediate feature geometry unconstrained. 
In multi-exit networks, the pressure to solve classification at intermediate points forces these layers to form their own geometric representations. Under this setup, the transition from local spatial representations to class-collapsed representations happens progressively:
*   **Arora et al. (2018)** [3] demonstrate that ReLU activations create piecewise-linear decision boundaries whose complexity increases exponentially with depth, directly linking activation sparsity to boundary sharpness.
*   By monitoring NC1–NC4 layer-by-layer, we observe that within-class collapse (NC1) and distance-based decision capability (NC4) develop continuously through the network, validating that Neural Collapse is a progressive layer-wise refinement rather than an abrupt final-layer event.

---

## 3. Adaptive Inference and Early Exiting
Early Exiting is a key method for reducing inference latency by attaching intermediate classifiers to shallow layers of a deep network. The decision to halt inference is typically governed by confidence thresholds (e.g., maximum softmax probability or Shannon entropy) (**Hendrycks & Gimpel, 2017** [4]).
The reliability of these shallow exit decisions is fundamentally bounded by the geometric quality of their underlying features:
*   At shallow layers, weak Neural Collapse (high NC1 variance and low NC2 symmetry) implies that class clusters overlap significantly in the representation space.
*   Consequently, simple confidence thresholds at early exits are prone to overconfidence on incorrect predictions. Only as the features undergo progressive collapse in deeper layers do confidence scores align reliably with classification accuracy.

---

## 4. Boundary Distortion under Class Imbalanced Data
In practical applications, datasets are rarely balanced. Class imbalance distorts the ideal simplex ETF geometry of Neural Collapse:
*   **Hasegawa & Sato (2024)** [5] demonstrate that majority classes compress class means closer to the global mean and push decision boundaries into the territory of minority classes, violating NC2 and NC3.
*   To resolve this without retraining, **Multiplicative Logit Adjustment (MLA)** scales the raw output logits by the training class priors ($p_c^\tau$). This post-hoc adjustment mathematically restores boundary symmetry at both intermediate and final exits, significantly improving calibration and accuracy for minority classes.

---

## 5. Distance-Based Out-of-Distribution (OOD) Detection
A key utility of structured representation spaces is the ability to detect inputs that do not belong to any training category (OOD detection).
*   **Liu & Qin (2025)** [6] show that under strong Neural Collapse, in-distribution (ID) samples are tightly bound to their class centers, while OOD samples fall into the low-density space between centers.
*   By calculating the distance to class means, models can identify OOD inputs. However, this capability is highly depth-dependent: at early layers, shared visual features (e.g., textures) cause ID and OOD representations to overlap completely (resulting in a low OOD AUROC). A distinct separation gap only emerges at deeper layers where semantic collapse is complete.

---

## 6. Geometric Complexity and Bias in Representations
The completeness of Neural Collapse is heavily influenced by dataset difficulty and training dynamics:
*   **Munn et al. (2024)** [7] show that the strength of representation collapse depends on the *geometric complexity* of the dataset. Simpler datasets with distinct boundaries (e.g., CIFAR-10) collapse quickly and completely, while fine-grained datasets (e.g., Oxford Flowers 102) retain significant intra-class variance due to high inter-class visual similarity.
*   Furthermore, **Wang et al. (2024)** [8] show that "shortcut learning" (relying on spurious background correlations rather than object geometry) distorts the collapse geometry, leading to biased, asymmetric representations. Minimizing this bias is critical to achieving symmetric, generalizable collapse.

---

## 📄 References

1. **Papyan, V., Han, X., & Donoho, D. L. (2020).** *Prevalence of Neural Collapse during the terminal phase of deep learning training.* Proceedings of the National Academy of Sciences (PNAS).
2. **Zeiler, M. D., & Fergus, R. (2014).** *Visualizing and understanding convolutional networks.* European Conference on Computer Vision (ECCV).
3. **Arora, R., Basu, A., Mianjy, P., & Mukherjee, A. (2018).** *Understanding deep neural networks with Rectified Linear Units.* International Conference on Learning Representations (ICLR).
4. **Hendrycks, D., & Gimpel, K. (2017).** *A baseline for detecting misclassified and out-of-distribution examples in neural networks.* International Conference on Learning Representations (ICLR).
5. **Hasegawa, T., & Sato, I. (2024).** *Multiplicative Logit Adjustment for Neural Collapse.* arXiv preprint arXiv:2404.14816.
6. **Liu, Y., & Qin, Y. (2025).** *Detecting Out-of-Distribution through the Lens of Neural Collapse.* Computer Vision and Pattern Recognition (CVPR).
7. **Munn, J., et al. (2024).** *Geometric Complexity in Transfer Learning.* arXiv preprint arXiv:2409.08312.
8. **Wang, Z., et al. (2024).** *Debiased Learning via Neural Collapse.* Computer Vision and Pattern Recognition (CVPR).
