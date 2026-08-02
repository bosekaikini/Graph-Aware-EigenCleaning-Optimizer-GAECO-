# Graph-Aware EigenCleaning Optimizer (GAECO-Net)

A PyTorch and VectorBT quantitative framework implementing the **Graph-Aware EigenCleaning Optimizer (GAECO-Net)** for portfolio allocation and risk management, based on *Karzanov et al. (2025)* ([arXiv:2408.01387](https://arxiv.org/abs/2408.01387)).

GAECO-Net combines **Spectral Graph Neural Networks (ChebNet)** with **Random Matrix Theory (RMT)** denoising to construct robust, noise-resilient asset covariance matrices and portfolio allocations.

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



Core ConceptsGraph Laplacian Signal Processing:The asset universe is modeled as a connected graph $G = (V, E)$, where edge weights correspond to empirical correlations. The normalized Graph Laplacian is defined as $\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}$. ChebNet layers apply $K$-th order Chebyshev polynomials to approximate spectral graph convolutions without explicit eigendecomposition on every pass.Random Matrix Theory (RMT) Eigencleaning:Empirical correlation matrices suffer from sample noise in high dimensions ($N \approx T$). GAECO-Net cleans noisy eigenvalues outside the Marchenko-Pastur bounds ($\lambda \in [\lambda_{-}, \lambda_{+}]$), preserving informative market factors while shrinking noise modes.Asymmetric Downside (Sortino) Optimization:The model is optimized using an asymmetric downside return loss combined with an $L_1$ turnover penalty:$$\mathcal{L} = -\left( \beta \cdot \bar{R}_p + \frac{\bar{R}_p}{\sigma_{\text{down}}} \right) + \lambda_{\text{turnover}} \cdot \Vert{}\mathbf{w}_t - \mathbf{w}_{t-1}\Vert{}_1$$This penalizes downside volatility without restricting upside variance, mitigating transaction cost drag.Continuous Soft Masking (GAECO-Explained):Instead of hard binary edge truncations (which disconnect the Graph Laplacian and induce numerical instability), GAECO-Explained applies smooth sigmoidal soft masks:$$\mathbf{L}_{\text{masked}} = \mathbf{L} \odot (1 - \alpha + \alpha \cdot \mathbf{M}_{\text{edge}})$$This preserves spectral topology while suppressing noisy cross-asset correlations.Project StructurePlaintextGraph-Aware-EigenCleaning-Optimizer-GAECO-/
├── data/
│   └── price_loader.py         # Downloads/formats asset price series and computes log returns.
├── pipeline/
│   ├── model.py                # ChebNet spectral GNN & Markowitz allocation layers.
│   ├── train.py                # Training loop, loss functions (Sortino), feature extractors.
│   └── rmt.py                  # Marchenko-Pastur eigenvalue cleaning & spectral filters.
├── explainability/
│   └── explainer.py            # GAECOExplainer module, soft masking engine, networkx visualizer.
├── backtest/
│   ├── engine.py               # VectorBT execution wrapper & transaction cost engine.
│   └── covariance.py           # Benchmark estimators (Ledoit-Wolf, Sample Covariance).
├── main.py                     # Primary pipeline entry point: data load, train, eval, export.
├── benchmark_results.txt       # Out-of-sample backtest report & statistical significance tests.
├── requirements.txt            # Environment dependencies.
└── README.md                   # System documentation.
Detailed File Specifications1. pipeline/model.pyGAECO_Net(nn.Module): Main network combining Chebyshev graph convolutions, RMT eigenvalue cleaning, and softmax/quadratic constrained weight generation.Input Tensors: Node Features (B, N, F), Graph Laplacian (B, N, N), Eigenvalues (B, N), Eigenvectors (B, N, N).2. explainability/explainer.pyGAECOExplainer(nn.Module): Post-hoc optimization layer learning continuous edge masks $\mathbf{M}_{\text{edge}} \in [0, 1]^{N \times N}$ and node feature masks $\mathbf{M}_{\text{feat}} \in [0, 1]^F$.explain_allocation(): Optimizes fidelity, sparsity, and entropy loss terms. Critical Implementation Detail: Uses with torch.enable_grad(): internally to ensure autograd functions during backtest/evaluation sweeps.generate_explained_allocations(): Generates out-of-sample allocations on soft-masked graphs ($\alpha = 0.5$).3. backtest/engine.py & main.pyStatistical Verification: Computes two-tailed paired $t$-tests and Jobson-Korkie (1981) tests with Memmel (2002) correction to evaluate Sharpe ratio differences:$$Z = \frac{\hat{SR}_A - \hat{SR}_B}{\sqrt{\frac{1}{T} \left[ 2 - 2\rho + \frac{1}{2}(\hat{SR}_A^2 + \hat{SR}_B^2 - 2\hat{SR}_A\hat{SR}_B\rho^2) \right]}}$$Type annotations explicitly cast return values to native Python float primitives (float(z_stat), float(p_value)) to avoid NumPy array type mismatch issues in Pylance/MyPy static analysis.Key Technical Decisions & Gotchas for AI AgentsIf you are an automated agent or developer extending this codebase, keep the following constraints in mind:PyTorch Gradient Scope in Explainer:Always maintain with torch.enable_grad(): inside explainability/explainer.py -> explain_allocation(). Calling the explainer inside an evaluation loop (which is wrapped in with torch.no_grad():) will cause PyTorch to raise RuntimeError: element 0 of tensors does not require grad if explicit gradient scope override is omitted.Detached Target Tensors:When calculating post-hoc explainer loss, ensure target portfolio weights are explicitly detached from the base computational graph (target_weights.detach()).Graph Laplacian Stability:Do not apply hard binary thresholding (e.g., edge_mask > percentile) to the Laplacian matrix during inference. Hard thresholding breaks graph connectivity, leading to singular degree matrices $\mathbf{D}$ and failing spectral convolutions. Always use the continuous soft masking implementation defined in generate_explained_allocations().Type Checking Annotations:SciPy distribution functions (stats.norm.cdf) return 0-dimensional NumPy NDArray[float64] objects. Always explicitly convert scalar statistics using float(...) prior to returning from functions annotated with tuple[float, float].