# E(2) Steerable CNN: Complete Mathematical Derivation from Scratch

## Table of Contents
1. Group Theory Foundations
2. Representation Theory
3. Kernel Constraint Derivation
4. Irrep Decomposition Strategy
5. Fourier Basis Solution for O(2)
6. Implementation Details
7. Equivariance Testing

---

## 1. GROUP THEORY FOUNDATIONS

### 1.1 The Euclidean Group E(2)

The Euclidean group is the set of all isometries (distance-preserving transformations) of ℝ²:

$$E(2) = \{(t, O) \mid t \in \mathbb{R}^2, O \in O(2)\}$$

where:
- **t** is a translation vector
- **O(2)** is the orthogonal group (rotations and reflections)

**Semidirect Product Structure:**
$$E(2) = (\mathbb{R}^2, +) \rtimes O(2)$$

Any element acts on a point as: $(t, O) \cdot x = Ox + t$

### 1.2 Subgroups of O(2)

For this work, we focus on practical subgroups:

| Group | Order | Elements | Use Case |
|-------|-------|----------|----------|
| **SO(2)** | ∞ | Continuous rotations | Exact rotation equivariance |
| **C_N** | N | N discrete rotations by 2π/N | Efficient finite rotation |
| **D_N** | 2N | N rotations + N reflections | Include reflection symmetry |
| **O(2)** | ∞ | SO(2) + reflections | Maximum symmetry |

### 1.3 Group Action on Images

For a scalar field $f: \mathbb{R}^2 \to \mathbb{R}$ (e.g., grayscale image):

$$(L_g f)(x) = f(g^{-1}x)$$

where $g = (t, O) \in E(2)$.

---

## 2. REPRESENTATION THEORY

### 2.1 Group Representations

A **representation** $\rho: G \to GL(\mathbb{R}^c)$ is a homomorphism that maps group elements to invertible matrices.

**Key Property:** $\rho(g_1 g_2) = \rho(g_1)\rho(g_2)$ and $\rho(e) = I$

### 2.2 Irreducible Representations (Irreps)

An irrep is a representation that cannot be decomposed into smaller representations. For **C_N**, the irreps are 1D:

$$\psi_m: C_N \to \mathbb{C}^* \quad \psi_m(g_k) = e^{2\pi i m k/N}$$

where $m \in \{0, 1, ..., N-1\}$ and $k$ is the rotation index.

### 2.3 Regular Representation

The **regular representation** $\rho_{\text{reg}}: G \to GL(\mathbb{R}^{|G|})$ acts by permuting basis vectors indexed by group elements:

$$\rho_{\text{reg}}(h) e_g = e_{hg}$$

For **C_N**, this is cyclic permutation of N channels.

**Decomposition:** For finite groups,
$$\rho_{\text{reg}} = \bigoplus_{i=1}^k m_i \psi_i$$

where $m_i$ is the multiplicity of irrep $\psi_i$.

### 2.4 Induced Representations

Feature fields on $\mathbb{R}^2$ transform under the **induced representation** of $E(2)$:

$$\text{Ind}^{E(2)}_G \rho: E(2) \to GL(\mathbb{R}^c)$$

$$((\text{Ind}\rho)(t,g) \cdot f)(x) = \rho(g) \cdot f(g^{-1}(x-t))$$

This combines translation (via convolution) with group action.

---

## 3. KERNEL CONSTRAINT DERIVATION

### 3.1 Equivariance Requirement

For a convolution layer to be equivariant:

**Input:** $f_{\text{in}}: \mathbb{R}^2 \to \mathbb{R}^{c_{\text{in}}}$ transforming under $\rho_{\text{in}}$

**Output:** $f_{\text{out}}: \mathbb{R}^2 \to \mathbb{R}^{c_{\text{out}}}$ transforming under $\rho_{\text{out}}$

**Convolution:**
$$(f_{\text{out}})(x) = \int_{\mathbb{R}^2} k(y) f_{\text{in}}(x-y) dy$$

**Equivariance Condition:**
$$((\text{Ind}\rho_{\text{out}})(g) \cdot f_{\text{out}})(x) = (f_{\text{out}})(g^{-1}(x-t))$$

when input is transformed: $f_{\text{in}} \to \rho_{\text{in}}(g) \cdot f_{\text{in}}(g^{-1}(x-t))$

### 3.2 Kernel Constraint

For the convolution to preserve equivariance, the kernel must satisfy:

$$\boxed{k(gx) = \rho_{\text{out}}(g) \, k(x) \, \rho_{\text{in}}(g^{-1}) \quad \forall g \in G, x \in \mathbb{R}^2}$$

**Intuition:** Rotating the input location must transform the kernel consistently with output representation.

### 3.3 Solution Space

The constraint is **linear** in $k$, so solutions form a linear subspace:

$$\mathcal{K} = \{k: \mathbb{R}^2 \to \mathbb{R}^{c_{\text{out}} \times c_{\text{in}}} \mid \text{satisfies constraint}\}$$

We parameterize $\mathcal{K}$ by a basis: $k(x) = \sum_{i=1}^d w_i \phi_i(x)$ where $\{\phi_i\}$ are basis kernels.

---

## 4. IRREP DECOMPOSITION STRATEGY

### 4.1 Change of Basis

Any representation decomposes into irreps:

$$\rho_{\text{in}} = Q_{\text{in}}^{-1} \bigoplus_{j \in I_{\text{in}}} \psi_j \, Q_{\text{in}}$$

where $Q_{\text{in}}$ is the change-of-basis matrix and $\psi_j$ are irreps with multiplicities encoded in $I_{\text{in}}$.

### 4.2 Reduced Kernel Constraint

Define $\kappa = Q_{\text{out}} k Q_{\text{in}}^{-1}$. Then:

$$k(gx) = \rho_{\text{out}}(g) k(x) \rho_{\text{in}}(g^{-1})$$

becomes:

$$\kappa(gx) = \bigoplus_{i \in I_{\text{out}}} \psi_i(g) \bigoplus_{j \in I_{\text{in}}} \psi_j(g)^{-1} \, \kappa(x)$$

This decomposes into **independent block constraints**:

$$\boxed{\kappa_{ij}(gx) = \psi_i(g) \, \kappa_{ij}(x) \, \psi_j(g)^{-1}}$$

### 4.3 Solving Block Constraints

For each block, we need to find kernels satisfying the constraint between two irreps $\psi_i$ and $\psi_j$.

**Key Insight:** We can solve each $\psi_i \to \psi_j$ constraint independently, then combine via:

$$k = Q_{\text{out}}^{-1} \left(\bigoplus_{i,j} \kappa_{ij}^{(1)}, \ldots, \kappa_{ij}^{(d_{ij})}\right) Q_{\text{in}}$$

---

## 5. FOURIER BASIS SOLUTION FOR O(2)

### 5.1 Polar Coordinate Expansion

Since the group action is norm-preserving ($|gx| = |x|$), we expand kernels in polar coordinates $(r, \varphi)$:

$$\kappa_{ij}(r, \varphi) = \sum_{\mu} \left[A_\mu(r) \cos(\mu\varphi) + B_\mu(r) \sin(\mu\varphi)\right]$$

### 5.2 Fourier Coefficient Constraints

Substituting into the block constraint $\kappa_{ij}(g(r,\varphi)) = \psi_i(g) \kappa_{ij}(r,\varphi) \psi_j(g)^{-1}$ and comparing Fourier coefficients, we get constraints on $A_\mu(r), B_\mu(r)$.

### 5.3 Analytical Solution for Common Cases

**Table: Steerable Kernel Bases for O(2)**

For irreps labeled by frequency $m \in \mathbb{Z}$ (where $\psi_m(g_\theta) = e^{im\theta}$):

| Input | Output | Basis Functions |
|-------|--------|-----------------|
| $\psi_0$ (trivial) | $\psi_0$ (trivial) | $\{1\}$ (constant) |
| $\psi_0$ | $\psi_m$ | $\{\sin(m\varphi), \cos(m\varphi)\}$ |
| $\psi_m$ | $\psi_n$ | $\{\cos((m-n)\varphi), \sin((m-n)\varphi), \cos((m+n)\varphi), \sin((m+n)\varphi)\}$ |

**Radial Part:** All coefficients $A_\mu(r), B_\mu(r)$ are **free** to be learned!

This allows:
$$\kappa(r,\varphi) = R(r) \cdot \text{(angular harmonic)}$$

### 5.4 Construction Algorithm

```
For each pair (freq_out, freq_in):
    diff = freq_out - freq_in
    sum = freq_out + freq_in
    
    For each radial basis function r_i(r):
        If diff >= 0:
            basis.append(r_i(r) * cos(diff*φ))
            basis.append(r_i(r) * sin(diff*φ))
        If sum > 0:
            basis.append(r_i(r) * cos(sum*φ))
            basis.append(r_i(r) * sin(sum*φ))
    End
End
```

---

## 6. IMPLEMENTATION DETAILS

### 6.1 Discrete Sampling

Kernels are sampled on discrete pixel grids. To avoid aliasing:

1. **Grid Definition:** Sample at pixel positions $(i, j)$ for $i,j \in \{-K, ..., K\}$
2. **Polar Conversion:** $r_{ij} = \sqrt{i^2 + j^2}$, $\varphi_{ij} = \text{atan2}(j,i)$
3. **Radial Basis:** Use Gaussian: $\phi_\ell(r) = e^{-(r/\sigma_\ell)^2}$ with multiple scales

### 6.2 Kernel Construction

**Algorithm:**
```
1. Create basis tensors: basis[b] ∈ ℝ^{K×K} for b=1..B
2. Learn weights: w ∈ ℝ^{C_out × C_in × B}
3. Construct kernel: k = Σ_b w_{:,:,b} ⊗ basis[b]
4. Apply conv2d(input, k, ...)
```

**Memory Note:** This is more efficient than learning $C_{\text{out}} \times C_{\text{in}} \times K \times K$ free parameters!

### 6.3 Forward Pass Pseudocode

```python
def steerable_conv_forward(x, basis, weights, padding):
    B, C_in, H, W = x.shape
    C_out, _, num_basis = weights.shape
    
    # Expand basis: (num_basis, 1, K, K) with weights
    kernels = torch.zeros(C_out, C_in, K, K)
    
    for c_out in range(C_out):
        for c_in in range(C_in):
            for b in range(num_basis):
                kernels[c_out, c_in] += \
                    weights[c_out, c_in, b] * basis[b, 0]
    
    # Standard convolution
    output = F.conv2d(x, kernels, padding=padding)
    return output
```

---

## 7. EQUIVARIANCE TESTING

### 7.1 Testing Rotation Equivariance

**Property:** For a classification model (invariant), outputs should be **invariant** to input rotations.

```python
def test_invariance(model, x, group_order=4):
    # Original output
    y_orig = model(x)
    
    # Rotated input (discrete rotations)
    for k in range(1, group_order):
        angle = 2*np.pi*k / group_order
        x_rot = rotate_image(x, angle)
        y_rot = model(x_rot)
        error = (y_orig - y_rot).abs().max()
        print(f"Rotation {k}: error = {error:.6f}")
```

### 7.2 Testing Feature Equivariance

For intermediate feature maps with regular representation:

```python
def test_feature_equivariance(model, x, k, group_order=4):
    # Get feature before group pooling
    x_rot = rotate_image(x, 2*np.pi*k/group_order)
    
    # Features should cyclically permute
    # channels according to rotation group action
```

---

## 8. ADVANTAGES & GUARANTEES

### 8.1 Equivariance Guarantees

✓ **Exact equivariance:** For finite groups (C_N, D_N), guaranteed by construction  
✓ **Data efficiency:** Learn transformations once, use for all orientations  
✓ **Parameter efficiency:** Kernel bases reduce effective parameters

### 8.2 Computational Costs

- **Training:** Kernel basis expansion ~10-20% overhead
- **Inference:** Standard kernels after training, no overhead
- **Memory:** Can export to pure PyTorch after training

---

## 9. EXTENSION: CONTINUOUS SO(2)

For **exact continuous rotation equivariance**, use SO(2) with band-limited representations:

```
ψ_m(θ) = e^{imθ}  for m ∈ {-M, ..., M}
```

Limit to finite frequency $M$ for computational tractability. The Fourier basis approach extends naturally to SO(2) with arbitrary frequencies.

---

## References

1. **General E(2)-Equivariant Steerable CNNs** (Weiler & Cesa, NeurIPS 2019)
   - Complete mathematical framework
   - Analytical solutions for all E(2) subgroups

2. **Harmonic Networks: Deep Translation and Rotation Equivariance** (Weiler et al., ICML 2017)
   - First practical steerable CNN implementation

3. **Group Equivariant Convolutional Networks** (Cohen & Welling, ICML 2016)
   - Foundation for discrete group equivariance

4. **Steerable CNNs** (Weiler et al., ICLR 2019)
   - General framework for homogeneous spaces

---

## Appendix: Character Theory for Irreps

For finite groups, compute irrep multiplicities using **character orthogonality**:

$$m_i = \frac{1}{|G|} \sum_{g \in G} \chi_\rho(g) \overline{\chi_{\psi_i}(g)}$$

where $\chi$ is the character (trace) of the representation.

For **C_N**: $\chi_{\psi_m}(g_k) = e^{2\pi i m k/N}$
