---
title: "Cross-Currency Curve Analysis with Tensor PCA"
subtitle: "A Quantum Computing Use-Case Test Proposal for USD-Collateralized OIS Curves"
author: "Quantitative Analytics"
date: "2026-08-31"
---

# Executive Summary

This report documents an end-to-end workflow for constructing USD-collateralized
overnight-index-swap (OIS) discount curves across four currencies (USD, EUR,
GBP, JPY), generating a scenario tensor of zero rates, and comparing **classic
(flattened) PCA** against **structure-preserving Tensor PCA (Tucker / HOSVD)**.

Beyond the classical comparison, the report **positions the tensor structure as
a natural test case for quantum computing**. Financial curve scenarios live in a
tensor-product Hilbert space $\mathbb{R}^{C}\otimes\mathbb{R}^{T}$ whose
dimension grows *multiplicatively* with the number of currencies and tenors —
exactly the regime where quantum state encodings and quantum linear-algebra
primitives promise an exponential representational advantage. Tucker
decomposition is the classical analogue of finding a **low-entanglement
(low-bond-dimension) approximation** of a quantum state, making this problem an
ideal, well-understood benchmark for near-term quantum algorithms.

# 1. Curve Construction

## 1.1 Market Data and Pillars

Synthetic but realistic market data were quoted at nine standard pillars.
The OIS par rates used are:

| Currency | 1Y   | 2Y   | 3Y   | 5Y   | 7Y   | 10Y  | 15Y  | 20Y  | 30Y  |
|----------|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
| USD      | 4.00 | 3.95 | 3.90 | 3.85 | 3.85 | 3.90 | 4.00 | 4.05 | 4.10 |
| EUR      | 2.50 | 2.55 | 2.60 | 2.70 | 2.80 | 2.90 | 3.00 | 3.05 | 3.10 |
| GBP      | 4.50 | 4.45 | 4.40 | 4.35 | 4.35 | 4.40 | 4.45 | 4.50 | 4.55 |
| JPY      | 0.50 | 0.60 | 0.70 | 0.85 | 1.00 | 1.20 | 1.40 | 1.50 | 1.60 |

*(values in %)*

The cross-currency basis spreads (bps, on the foreign leg) are:

| Pair    | 1Y  | 2Y  | 3Y  | 5Y  | 7Y  | 10Y | 15Y | 20Y | 30Y |
|---------|----:|----:|----:|----:|----:|----:|----:|----:|----:|
| EUR/USD | -15 | -18 | -20 | -22 | -24 | -25 | -26 | -27 | -28 |
| GBP/USD |  -8 |  -9 | -10 | -11 | -12 | -13 | -14 | -15 | -16 |
| JPY/USD | -30 | -35 | -40 | -45 | -50 | -55 | -58 | -60 | -62 |

## 1.2 Bootstrapping Methodology

1. **OIS discount curves.** For each currency, OISRateHelper instruments were
   bootstrapped with a PiecewiseLogLinearDiscount curve. A discount factor
   $P(t)$ relates to the continuously-compounded zero rate $z(t)$ by

$$
P(t) = e^{-z(t)\,t}, \qquad z(t) = -\frac{1}{t}\ln P(t).
$$

2. **Index re-linking.** Each overnight index was re-linked to its bootstrapped
   forecast curve, since the cross-currency helper forecasts the USD (base) leg.

3. **USD-collateralized foreign curves.** The foreign discount curves were
   re-bootstrapped under USD collateral using a constant-notional cross-currency
   basis swap. At par,

$$
\text{PV}_{\text{USD}} \;=\; \text{PV}_{\text{FOR}} + b \cdot A_{\text{FOR}},
$$

where $b$ is the quoted basis spread and $A_{\text{FOR}}$ is the foreign-leg
annuity discounted on the collateralized curve.

# 2. Scenario Generation

To provide meaningful input for dimensionality reduction, 200 scenarios were
generated with a **low-rank factor model** that reproduces the shared structure
of real curve movements.

## 2.1 Generative Model

Let $\mathbf{Z}_0 \in \mathbb{R}^{C \times T}$ be the base zero-rate surface
($C=4$ currencies, $T=9$ tenors). For scenario $s$,

$$
\mathbf{Z}^{(s)} \;=\; \mathbf{Z}_0
\;+\; f^{(s)}_{L}\,\mathbf{c}\otimes\boldsymbol{\ell}
\;+\; f^{(s)}_{S}\,\mathbf{c}\otimes\mathbf{s}
\;+\; f^{(s)}_{K}\,\mathbf{c}\otimes\mathbf{k}
\;+\; f^{(s)}_{B}\,\mathbf{d}\otimes\boldsymbol{\ell},
$$

where $\otimes$ is the **outer (tensor) product**:
$\mathbf{c}\otimes\boldsymbol{\ell}$ is the rank-1 currency-by-tenor surface
with entries $c_i\,\ell_j$. The shapes
$\boldsymbol{\ell}, \mathbf{s}, \mathbf{k} \in \mathbb{R}^{T}$ are level, slope,
and curvature; $\mathbf{c}, \mathbf{d} \in \mathbb{R}^{C}$ are the common and
basis-divergence currency loadings; and $f^{(s)}_{L},\dots \sim \mathcal{N}$ are
independent Gaussian factors.

Crucially, **each shock is a sum of tensor-product (rank-1) terms** — precisely
the separable structure a quantum register would encode as a low-entanglement
state.

| Factor    | Volatility $\sigma$ | USD | EUR | GBP | JPY |
|-----------|--------------------:|----:|----:|----:|----:|
| Level     | 0.0010              | 1.00 | 0.90 | 0.85 | 0.70 |
| Slope     | 0.0006              | 1.00 | 0.90 | 0.85 | 0.70 |
| Curvature | 0.0003              | 1.00 | 0.90 | 0.85 | 0.70 |
| Basis     | 0.0004              | 0.00 | 1.00 | 0.60 | 1.40 |

Stacking gives a tensor
$\mathcal{X} \in \mathbb{R}^{200 \times 4 \times 9}$ of true multilinear rank 4.

# 3. Analysis: Classic PCA vs Tensor PCA

## 3.1 Classic (Flattened) PCA

Each centred surface is flattened to
$\mathbf{x}^{(s)} = \operatorname{vec}(\mathbf{Z}^{(s)} - \bar{\mathbf{Z}})
\in \mathbb{R}^{CT}$. The flattening is itself a tensor-product identification,
$\mathbb{R}^{C}\otimes\mathbb{R}^{T}\cong\mathbb{R}^{CT}$, but PCA then treats
the 36-vector as unstructured. PCA diagonalizes the covariance

$$
\boldsymbol{\Sigma} = \frac{1}{N-1}\sum_{s}
\mathbf{x}^{(s)}\mathbf{x}^{(s)\top}
= \mathbf{U}\,\boldsymbol{\Lambda}\,\mathbf{U}^{\top}.
$$

| Component  | PC1   | PC2   | PC3   | PC4    | PC5+ |
|------------|------:|------:|------:|-------:|-----:|
| Variance   | 92.3% | 4.6%  | 3.0%  | 0.06%  | ~0%  |
| Cumulative | 92.3% | 96.9% | 99.9% | 100.0% | 100% |

![Classic PCA scree plot](figures/scree_plot.png)

Each component is a full $4\times 9$ surface, **entangling** the currency and
tenor axes:

![Classic PCA components (flattened, reshaped to currency x tenor)](figures/classic_pca_components.png)

## 3.2 Tensor PCA (Tucker / HOSVD)

Arrange the centred deviations as a third-order tensor
$\mathcal{X} \in \mathbb{R}^{N \times C \times T}$ living in the tensor-product
space $\mathbb{R}^{N}\otimes\mathbb{R}^{C}\otimes\mathbb{R}^{T}$. The mode-$n$
unfolding matricizes along one axis; classic PCA is exactly the SVD of the
mode-1 unfolding $X_{(1)} \in \mathbb{R}^{N \times CT}$, so it never resolves the
$\mathbb{R}^{C}\otimes\mathbb{R}^{T}$ sub-structure.

Tucker factorizes across **all** modes simultaneously:

$$
\mathcal{X} \;\approx\;
\mathcal{G} \times_1 U^{(1)}
            \times_2 U^{(2)}
            \times_3 U^{(3)},
$$

equivalently, in vectorized (Kronecker) form,

$$
\operatorname{vec}(\mathcal{X}) \;\approx\;
\big(U^{(1)} \otimes U^{(2)} \otimes U^{(3)}\big)\,
\operatorname{vec}(\mathcal{G}),
$$

where $\otimes$ is the Kronecker product, $\mathcal{G} \in
\mathbb{R}^{R_1 \times R_2 \times R_3}$ is the **core**, and each factor
$U^{(n)}$ has orthonormal columns. HOSVD initializes each factor from the
leading left singular vectors of $X_{(n)} = U^{(n)}\Sigma^{(n)}V^{(n)\top}$, with
the core $\mathcal{G} = \mathcal{X} \times_1 U^{(1)\top} \times_2 U^{(2)\top}
\times_3 U^{(3)\top}$, refined by ALS to minimize
$\lVert \mathcal{X} - \hat{\mathcal{X}} \rVert_F$.

With ranks $[4, 2, 3]$ the reconstruction reached machine precision:

$$
\frac{\lVert \mathcal{X} - \hat{\mathcal{X}} \rVert_F}
     {\lVert \mathcal{X} \rVert_F} \;\approx\; 1.0\times 10^{-14}.
$$

![Tucker mode factors: tenor (left) and currency (right)](figures/tucker_mode_factors.png)

## 3.3 Comparison

Rank-$r$ flattened PCA stores $r\,CT$ loadings (the **product** of mode sizes);
Tucker stores

$$
\underbrace{C R_2}_{\text{ccy factors}}
+ \underbrace{T R_3}_{\text{tenor factors}}
+ \underbrace{R_2 R_3}_{\text{core}}
= 4\cdot 2 + 9\cdot 3 + 2\cdot 3 = 41,
$$

the **sum** of mode sizes — the same separation of scales that lets a quantum
register store an $N$-mode product state in $\mathcal{O}(\log)$ qubits.

| Method              | Relative Error | Loading Parameters | Interpretable?                  |
|---------------------|---------------:|-------------------:|---------------------------------|
| Classic PCA (r = 1) | 2.77e-01       | 36                 | No — entangled 4x9 surfaces     |
| Classic PCA (r = 2) | 1.75e-01       | 72                 | No — entangled 4x9 surfaces     |
| Classic PCA (r = 3) | 2.36e-02       | 108                | No — entangled 4x9 surfaces     |
| Classic PCA (r = 4) | 1.53e-15       | 144                | No — entangled 4x9 surfaces     |
| **Tucker [4, 2, 3]**| **1.01e-14**   | **41**             | **Yes — separable ccy x tenor** |

# 4. Quantum Computing Use-Case Rationale

This problem is a deliberately-chosen, tractable proxy for the structures where
quantum computing is expected to help. The connections are direct:

## 4.1 Tensor-Product State Space

A curve-scenario surface lives in
$\mathcal{H} = \mathbb{R}^{C}\otimes\mathbb{R}^{T}$. Adding currencies and
tenors grows $\dim\mathcal{H} = C\cdot T$ **multiplicatively**. A quantum
register encodes such a space as a product of qubit sub-registers,
$\mathcal{H}_Q = (\mathbb{C}^{2})^{\otimes n}$, storing an amplitude vector of
dimension $2^n$ in only $n$ qubits — the canonical exponential encoding
advantage.

## 4.2 Low Rank = Low Entanglement

The Tucker/HOSVD approximation is the classical twin of finding a
**low-bond-dimension (low-entanglement) approximation** of a quantum state, as in
matrix-product-state / tensor-network methods. A surface that is a sum of a few
$\mathbf{c}\otimes\boldsymbol{\ell}$ terms is a **near-product state**: its
Schmidt rank across the currency|tenor cut is small. Our data are engineered to
have exactly this property, giving a controlled benchmark with a known answer.

## 4.3 Candidate Quantum Primitives to Benchmark

- **Quantum PCA (qPCA).** Exponentiating the density-matrix analogue of
  $\boldsymbol{\Sigma}$ to extract dominant eigenvectors — testable here against
  the exact classical spectrum $\{\lambda_i\}$.
- **Variational quantum eigensolver / tensor-network ansatz.** Fit a
  low-bond-dimension state to $\operatorname{vec}(\mathcal{X})$ and compare its
  Schmidt spectrum to the classical Tucker core.
- **Quantum linear algebra (HHL / block-encoding).** Solve the bootstrapping or
  covariance systems as a stress test of quantum linear solvers on a problem with
  a verifiable classical solution.

## 4.4 Why It Is a Good Test Case

- **Known ground truth.** Machine-precision classical Tucker gives an exact
  reference for validating any quantum approximation.
- **Tunable difficulty.** Increasing $C$, $T$, or the number of factors scales
  the state dimension and entanglement smoothly, mapping out where quantum
  methods begin to pay off.
- **Financial relevance.** The pipeline is a real risk/scenario workflow, so
  positive results transfer directly to production use cases.

# 5. Conclusions

- **Flattening destroys structure.** Vectorizing forces PCA to entangle the
  currency and tenor axes into every component.
- **Tucker/HOSVD preserves structure**, yielding interpretable level/slope/
  curvature and common/basis-divergence factors at one-third the parameter cost
  (41 vs 144).
- **The tensor-product structure makes this an ideal quantum benchmark** —
  multiplicative state growth, tunable entanglement, and an exact classical
  reference.

# 6. Next Steps

- Scale $C$ and $T$ to map the classical-vs-quantum crossover in state dimension.
- Prototype a **tensor-network (MPS) ansatz** and compare its bond dimension to
  the classical Tucker ranks.
- Benchmark a **qPCA / VQE** implementation against the exact spectrum on
  quantum simulators, then on hardware.
- Replace synthetic shocks with **historical curve moves** to test robustness of
  the low-entanglement assumption on real data.
