# Mathematical Formulation of Neural Collapse & Early Exiting

For a classification problem with $C$ classes and $N$ samples per class, let:
*   $x_{i,c}$ be the $i$-th training sample of class $c$.
*   $h_{i,c} = f(x_{i,c}) \in \mathbb{R}^d$ be the penultimate layer feature vector (embedding) of the sample.
*   $W \in \mathbb{R}^{C \times d}$ be the weight matrix of the final linear classifier, where $W_c \in \mathbb{R}^d$ is the weight vector for class $c$.
*   $\mu_c = \frac{1}{N} \sum_{i=1}^N h_{i,c}$ be the class mean feature vector.
*   $\mu_G = \frac{1}{C} \sum_{c=1}^C \mu_c$ be the global mean feature vector.

---

## 1. The Four Properties of Neural Collapse

Theoretical formulation defines Terminal Phase Training (when training loss $\to 0$) as satisfying four exact mathematical states:

### NC1: Within-Class Variability Collapse
The features of all samples in the same class collapse to the class mean. Specifically, the within-class covariance matrix $\Sigma_W$ converges to zero.

$$ \Sigma_W = \frac{1}{C N} \sum_{c=1}^C \sum_{i=1}^N (h_{i,c} - \mu_c)(h_{i,c} - \mu_c)^T \to 0 $$

Alternatively stated, for any sample $i$:
$$ ||h_{i,c} - \mu_c||_2 \to 0 $$

### NC2: Convergence to Simplex ETF (Equiangular Tight Frame)
The class means geometrically distance themselves symmetrically around the global mean. 
Let $M = [\mu_1 - \mu_G, \mu_2 - \mu_G, \dots, \mu_C - \mu_G] \in \mathbb{R}^{d \times C}$ be the matrix of centered class means. 

M evolves to form a Simplex ETF:
$$ M^T M = \frac{C}{C-1} \left( I_C - \frac{1}{C} \mathbf{1}_C \mathbf{1}_C^T \right) ||M||^2 $$

For any two distinct classes $c \neq c'$, the cosine similarity of their centered means is perfectly bounded:
$$ \cos(\angle(\mu_c - \mu_G, \mu_{c'} - \mu_G)) = \frac{\langle \mu_c - \mu_G, \mu_{c'} - \mu_G \rangle}{||\mu_c - \mu_G|| \cdot ||\mu_{c'} - \mu_G||} = -\frac{1}{C-1} $$

### NC3: Classifier Alignment to Class Means
The classifier weight vectors $W_c$ perfectly align with the centered class means $(\mu_c - \mu_G)$. 
$$ W_c \propto (\mu_c - \mu_G) $$

Measured mathematically by tracking the distance between normalized matrices:
$$ \left\| \frac{W_c}{||W_c||} - \frac{\mu_c - \mu_G}{||\mu_c - \mu_G||} \right\|_2 \to 0 $$

### NC4: Simplification to Nearest-Class-Center (NCC)
Because of NC1, NC2, and NC3, the learned linear classifier decision rule fundamentally simplifies. Instead of passing through a fully connected layer $W \cdot h$, the prediction is purely based on the closest Euclidean feature distance:

$$ \arg\max_{c} (W_c^T h_{i,c'} + b_c) \quad \to \quad \arg\min_{c} ||h_{i,c'} - \mu_c||_2 $$

---

## 2. Mathematics of OOD Detection via Neural Collapse (Liu & Qin)

Out-of-Distribution detection leverages NC4. An ID (In-Distribution) sample $x_{ID}$ will tightly cluster to $\mu_c$, while an OOD sample $x_{OOD}$ will lie in the empty space between vertices of the Simplex ETF.

**Distance-based Scoring Formulation:**
$$ S(x) = \min_{c} ||h(x) - \mu_c||_2 $$
If $S(x) > \tau_{OOD}$, the sample is flagged as Out-of-Distribution.

---

## 3. Logit Adjustment for Imbalance (Hasegawa & Sato)

When class sizes $N_c$ are imbalanced, weight magnitudes become skewed (distorting NC3 and the Simplex ETF). 

Standard Additive Logit Adjustment (assuming prior $\pi_c$):
$$ z_c' = W_c^T h(x) - \log(\pi_c) $$

**Multiplicative Logit Adjustment (NC-Aware Correction):**
$$ z_c' = \alpha_c \left( W_c^T h(x) \right) $$
Where $\alpha_c$ scales the logit relative to the distorted weight norm $||W_c||$. This successfully acts as a proxy for repairing the geometric distortion of the bounding hyperplanes.

---

## 4. Mathematics of Early Exiting
At intermediate layer $l \in \{1, 2, \dots, L-1\}$, an early exit classifier $f_l$ generates a probability distribution $p^{(l)}(x) = \text{Softmax}(W_l^T h_l(x))$.

The exit gate compares confidence against a predefined strict threshold $\tau$. If the threshold is met, computation halts.

**Criteria 1: Max Softmax Probability**
Exit if:
$$ \max_{c} p^{(l)}_c(x) > \tau $$

**Criteria 2: Shannon Entropy (More robust indicator of uncertainty)**
$$ H(p^{(l)}) = - \sum_{c=1}^C p^{(l)}_c(x) \log\left( p^{(l)}_c(x) \right) $$
Exit if entropy is sufficiently low:
$$ H(p^{(l)}) < \tau_{entropy} $$

---

## 5. Mathematical Justification for ResNet over Standard CNNs
When defending the choice of **ResNet** over standard sequential networks (like VGG) or entangled feature networks (like DenseNet) for **layer-wise feature evolution**, the mathematical backing rests on the gradient preservation and the identity mapping.

Standard networks learn a direct complex mapping:
$$ y = \mathcal{F}(x, W) $$

If the model is deep, gradients diminish mathematically through repeated chain-rule multiplication: 
$$ \frac{\partial L}{\partial x_1} = \frac{\partial L}{\partial y} \cdot \prod_{i=1}^{L} \frac{\partial x_{i+1}}{\partial x_i} \to 0 $$

**ResNet's Residual Block:**
ResNet forces the layer to explicitly learn a residual function mapping $\mathcal{F}(x)$ rather than the full mapping. The block output is:
$$ \mathcal{H}(x) = \mathcal{F}(x, \{W_i\}) + x $$

From a derivative standpoint, the gradient with respect to a lower layer $x_l$ becomes:
$$ \frac{\partial L}{\partial x_l} = \frac{\partial L}{\partial x_L} \left( 1 + \frac{\partial}{\partial x_l} \sum_{i=l}^{L-1} \mathcal{F}(x_i, W_i) \right) $$

**Why this is crucial for Neural Collapse and Early Exiting:**
1.  **The "$+ 1$" Term:** The gradient skips the internal weight matrix $\mathcal{F}$ entirely and propagates cleanly back to early layers. This guarantees that **meaningful feature geometry forms at intermediate layers**, making it possible for Early Exiting classifiers to function reliably. 
2.  **Identity Conservation:** If a feature representation reaches an optimal geometry (partial Neural Collapse) at an early layer, the residual function $\mathcal{F}(x)$ can simply drive its weights mathematically to zero ($\mathcal{F}(x) \to 0$). The identity mapping ensures the collapsed geometry is smoothly carried forward without being destroyed by non-linear destruction, which happens in VGG or DenseNet.
