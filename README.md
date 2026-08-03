# Graph-Aware EigenCleaning Optimizer (GAECO-Net)

A PyTorch and VectorBT quantitative framework implementing the **Graph-Aware EigenCleaning Optimizer (GAECO-Net)** for portfolio allocation and variance reduction.

GAECO-Net combines **Spectral Graph Neural Networks (ChebNet)** with **Random Matrix Theory (RMT)** denoising to construct robust, noise-resilient asset covariance matrices and portfolio allocations.

---

## Paper Abstract

Mean-variance portfolio optimization is fundamentally limited by
estimation error in the sample covariance matrix, an effect that intensifies as the number of assets approaches the length of the estimation window. Methods like linear shrinkage and Marchenko-
Pastur eigenvalue filtering apply a single, static correction uniformly across the universe. We introduce GAECO-Net (Graph-Aware
EigenCleaning Optimizer), an architecture that instead learns a
continuous, asset-specific shrinkage intensity: a spatial graph-convolutional
branch and a spectral eigenvalue branch are fused through an additive attention gate that determines, per asset and per rebalance
date, how strongly each eigenvalue is pulled toward the cross-sectional
noise-band mean. The cleaned covariance estimate and a jointly-
learned expected-return vector feed a closed-form mean-variance
decoder, trained end-to-end with a turnover-penalized Sortino objective and a two-phase, five-agent ensemble procedure combining synthetic-regime pre-training with real-data fine-tuning. We
evaluate the architecture out-of-sample universe from 2020 through 
2024 under realistic execution frictions, benchmarked against Ledoit-Wolf shrinkage, Marchenko-Pastur
denoising, and sample-covariance baselines. We evaluate on a suite of statistical tests including a Memmel-
corrected Jobson-Korkie Sharpe test, a paired mean-lift test and a
Levene variance test. GAECO-Net achieves a structurally and statistically significant reduction in realized return variance relative
to every baseline while achieving similar to slightly better Sharpe and returns – a gap traceable
to a multi-year sub-period where the strategy trails benchmarks
before decisively overtaking them from 2023 onward. A post-hoc explainability layer produces auditable,
per-decision subgraph attributions whose own variance-reduction
profile is statistically indistinguishable from, and in several comparisons stronger than, the full model’s – indicating that interpretability can be layered onto this architecture at negligible additional statistical cost.

---

## Technical Architecture & Methodology

```text
+-----------------------------------------------------------------------------------+
|                                 INPUT PIPELINE                                    |
| Asset Daily Returns  -->  Empirical Correlation Graph  -->  Graph Laplacian (L)   |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                           SPECTRAL GNN CORE (ChebNet)                             |
|  - Chebyshev Polynomial Convolutions over Graph Laplacian L                       |
|  - Node Features: Rolling Volatility, Momentum, Return Skewness                   |
|  - Spectral Eigen-filtering: Denoising Empirical Eigenvalues via RMT               |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                        POST-HOC EXPLAINABILITY LAYER                              |
|  - Continuous Soft Masking on Laplacian Edges & Node Features                     |
|  - Preserves Graph Connectivity (Prevents Laplacian Singularities)                |
|  - Active Asset Universe Pruning (Noise Asset Removal)                           |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                             BACKTESTING & EVALUATION                              |
|  - VectorBT Execution (Rebalancing, Transaction Fee Modeling)                    |
|  - Hypotheses Testing: Jobson-Korkie (1981) + Memmel (2002) Correction            |
|  - Benchmark Comparison: Ledoit-Wolf, Marchenko-Pastur, Sample Covariance        |
+-----------------------------------------------------------------------------------+
