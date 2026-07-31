# Graph-Aware-EigenCleaning-Optimizer-GAECO-

A complete differentiable portfolio optimization framework integrating spectral graph theory, deep learning, and dynamic optimization to denoise empirical financial correlation matrices during rapid market regime shifts.

Shorter timeframes during these regime shifts will highlight GAECO's performance gain over lagging standard indicators.

___
## GAECO V2
# Graph-Aware Eigen-Cleaning Optimizer (GAECO-Net)

GAECO-Net is an end-to-end differentiable portfolio optimization framework. It integrates **Spectral Graph Neural Networks (GCNs)** with **Random Matrix Theory (RMT)** cleanings to parameterize and optimize mean-variance allocations directly under transactional and non-convex constraints.

Rather than relying on classic empirical covariance estimates—which suffer from severe sample noise ($N \approx T$) and ill-conditioning—GAECO-Net dynamically denoises asset covariance matrices in the spectral domain while simultaneously estimating cross-sectional expected returns ($\boldsymbol{\mu}$) via localized spatial graph signals.

---

## 1. Mathematical Basis & Formulation

### Empirical Covariance & Spectral Decomposition
Given a matrix of rolling asset returns $\mathbf{R} \in \mathbb{R}^{T \times N}$ across $N$ assets over lookback window $T$:
$$\mathbf{\Sigma}_{\text{emp}} = \frac{1}{T-1} \mathbf{R}^T \mathbf{R}$$

We perform spectral decomposition on $\mathbf{\Sigma}_{\text{emp}}$ to yield eigenvalues and eigenvectors:
$$\mathbf{\Sigma}_{\text{emp}} = \mathbf{V} \mathbf{\Lambda} \mathbf{V}^T, \quad \mathbf{\Lambda} = \text{diag}(\lambda_1, \lambda_2, \dots, \lambda_N)$$

### Graph Representation & Laplacian
Asset relationships are modeled as an undirected weighted graph $\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathbf{W})$, where:
- $\mathcal{V}$ represents the $N$ asset nodes.
- $\mathbf{W} \in \mathbb{R}^{N \times N}$ is the empirical correlation matrix $\mathbf{C}_{\text{emp}}$.
- $\mathbf{D} = \text{diag}\left(\sum_j W_{ij}\right)$ is the degree matrix.
- Unnormalized Graph Laplacian: $\mathbf{L} = \mathbf{D} - \mathbf{W}$.

### Spectral Eigen-Cleaning & Return Prediction
To eliminate noise eigenvalues (the "Marcenko-Pastur bulk"), GAECO-Net uses a spatial-spectral cross-attention mechanism to output a node-level shrinkage operator $\boldsymbol{\gamma} \in (0, 1)^N$:

$$\boldsymbol{\lambda}_{\text{clean}} = \text{Softplus}\left(\boldsymbol{\gamma} \odot \boldsymbol{\lambda} + (\mathbf{1} - \boldsymbol{\gamma}) \odot \bar{\lambda}\right) + \epsilon$$
$$\mathbf{\Sigma}_{\text{clean}} = \mathbf{V} \cdot \text{diag}(\boldsymbol{\lambda}_{\text{clean}}) \cdot \mathbf{V}^T$$

Where:
- $\bar{\lambda} = \frac{1}{N} \sum_{i=1}^N \lambda_i$ is the empirical bulk mean.
- $\mathbf{\Sigma}_{\text{clean}}$ is guaranteed to be symmetric positive definite (SPD).

Concurrently, a graph-convolutional return prediction head estimates forward expected returns:
$$\boldsymbol{\mu} = f_{\theta}(\mathbf{H}^{(L)}) \in \mathbb{R}^N$$

### Analytical Differentiable Portfolio Layer
The optimal mean-variance portfolio allocation $\mathbf{w}^*$ under risk aversion parameter $\gamma_{\text{risk}}$ is solved analytically:
$$\mathbf{w}_{\text{unconstrained}} = \frac{1}{\gamma_{\text{risk}}} \mathbf{\Sigma}_{\text{clean}}^{-1} \boldsymbol{\mu}$$
$$\mathbf{w}^* = \text{Softmax}(\mathbf{w}_{\text{unconstrained}})$$

This ensures non-negative long-only weights ($\sum_i w_i = 1, w_i \ge 0$) while preserving explicit gradients with respect to both $\mathbf{\Sigma}_{\text{clean}}$ and $\boldsymbol{\mu}$.

---

## 2. Dynamic 6-Feature Node Matrix ($\mathbf{X} \in \mathbb{R}^{N \times 6}$)

At each timestep $t$, each node (asset) is initialized with a rolling 6-dimensional feature vector $\mathbf{x}_i \in \mathbb{R}^6$ capturing multi-scale volatility, momentum, tail risk, and drawdown dynamics:

| Feature Index | Signal Name | Formulation | Microstructural Purpose |
| :--- | :--- | :--- | :--- |
| **0** | **20D Volatility** | $\sigma_{i, 20} = \text{std}(R_{i, t-20:t})$ | Short-term asset variance scale |
| **1** | **10D Momentum** | $\mu_{i, 10} = \text{mean}(R_{i, t-10:t})$ | Short-term directional momentum |
| **2** | **60D Momentum** | $\mu_{i, 60} = \text{mean}(R_{i, t-60:t})$ | Intermediate trend strength |
| **3** | **30D Skewness** | $S_{i, 30} = \mathbb{E}\left[\left(\frac{R - \mu_{10}}{\sigma_{20}}\right)^3\right]$ | Left/right tail asymmetry & crash risk |
| **4** | **Downside Risk** | $\sigma_{i, \text{down}} = \text{std}(\min(R_{i, t-20:t}, 0))$ | Downside semivariance / downside deviation |
| **5** | **Max Drawdown** | $DD_{i} = \frac{\max(R_{t-20:t}) - R_{t-1}}{\max(R_{t-20:t}) + \epsilon}$ | Peak-to-trough distance under short window |

*Note: Standard deviation calculations enforce sample size constraint $N_{\text{samples}} \ge 2$ to prevent zero-degree-of-freedom execution warnings.*

---

## 3. Computational Network Architecture (`GAECONetPipeline`)

Node Features X [1, N, 6] ────────► Spatial GCN ────┐
├─► Cross-Attention Fusion (gamma) ─► Clean Eigenvalues ─► Clean Covariance Σ
Graph Laplacian L [1, N, N] ────────┤                │                                                                  │
│                                                                  ▼
Empirical Eigenvalues λ [1, N] ───► Spectral MLP ───┘                                                          Differentiable MV Layer ──► Portfolio Weights w
Empirical Eigenvectors V [1, N, N] ─────────────────────────────────────────────────────────────────────────────┤
│
Predicted Returns μ [1, N] ◄────── Return Head ◄────────────────────────────────────────────────────────────────┘


### Core Sub-modules (`models/gaeco_network.py`):
1. **Spatial Branch (`SpatialGCN`)**: Processes node attributes $\mathbf{X}$ using Graph Convolutional operators parameterized by the normalized Laplacian $\mathbf{L}$:
   $$\mathbf{H}^{(l+1)} = \sigma \left( \tilde{\mathbf{D}}^{-\frac{1}{2}} \tilde{\mathbf{A}} \tilde{\mathbf{D}}^{-\frac{1}{2}} \mathbf{H}^{(l)} \mathbf{W}^{(l)} \right)$$
2. **Spectral Branch (`SpectralMLP`)**: Processes raw empirical eigenvalues $\boldsymbol{\lambda}$ into dense spectral embeddings.
3. **Fusion Attention (`FusionAttention`)**: Cross-attends spatial node representations with spectral global information to generate asset-specific shrinkage rates $\boldsymbol{\gamma} \in (0, 1)^N$.
4. **Return Head (`return_head`)**: A multi-layer perceptron (MLP) mapping spatial GCN node representations $\mathbf{H}^{(L)}$ directly to predicted expected returns $\boldsymbol{\mu}$.
5. **Portfolio Allocation Layer (`DifferentiableMeanVariance`)**: Implements closed-form differentiable linear system solvers (`torch.linalg.solve`) to produce final weights $\mathbf{w}^*$.

---

## 4. End-to-End Objective Function

The model is trained end-to-end directly on out-of-sample portfolio Sharpe Ratio optimization, augmented by a quadratic turnover penalty to limit execution drag:

$$\mathcal{L}_{\text{GAECO}}(\theta) = - \text{Sharpe}(\mathbf{w}_t, \mathbf{r}_{t+1}) + \lambda_{\text{turnover}} \sum_{i=1}^N \vert{}w_{i, t} - w_{i, t-1}\vert{}$$

Where the realized out-of-sample Sharpe Ratio over batch step $t$ is calculated as:
$$\text{Sharpe}(\mathbf{w}_t, \mathbf{r}_{t+1}) = \frac{\mathbf{w}_t^T \mathbf{r}_{t+1}}{\sqrt{\mathbf{w}_t^T \mathbf{\Sigma}_{t+1}^{\text{realized}} \mathbf{w}_t} + \epsilon}$$

---

## 5. System Architecture & Directory Layout
│
├── data/
│   └── loader.py              # WRDS CRSP API Data Fetcher & Empirical Matrix Generators
├── models/
│   ├── gaeco_network.py       # GAECONetPipeline & Core Graph Architecture
│   ├── portfolio_layer.py     # Differentiable Mean-Variance Analytical Solver Layer
│   └── layers/
│       ├── dynamic_gcn.py     # Spatial Laplacian Graph Convolutions
│       ├── spectral_mlp.py    # Eigenvalue Spectral Representation Net
│       └── attention.py       # Spatial-Spectral Cross-Attention Module
├── loss/
│   └── risk_loss.py           # Negative Sharpe Ratio Loss with Turnover Penalty
├── pipeline/
│   └── train.py               # Rolling Epoch Loop, Slicing, & Model Checkpointing
├── backtester/
│   └── engine.py              # VectorBT Wrapper Functions
├── main.py                    # Complete Orchestration Entrypoint
└── requirements.txt           # Dependencies (torch, vectorbt, wrds, pandas, matplotlib)
---

## 6. Execution & Backtesting Workflow

### Data Ingestion & Preprocessing
`WRDSDataLoader` connects to the WRDS CRSP database, extracts adjusted daily security returns across a universe of $N=30$ assets, and precomputes rolling $T=60$ window empirical correlation matrices $\mathbf{C}_{\text{emp}}$, eigenvalues $\boldsymbol{\lambda}$, and eigenvectors $\mathbf{V}$.

### Out-of-Sample Backtesting (`main.py`)
1. **Training Pass**: `train_gaeco_net()` iterates across all rolling windows, producing optimal weight predictions $\mathbf{w}_t$ while preserving temporal ordering.
2. **Weekly Resampling & Regularization**: Weights generated daily are resampled to a **Weekly (Friday)** frequency (`W-FRI`), normalized ($\sum_i w_{i} = 1.0$), and held constant over the week to minimize daily portfolio turnover friction.
3. **VectorBT Execution**:
   ```python
   portfolio = vbt.Portfolio.from_orders(
       close=price_df,
       size=weekly_weights,
       size_type='targetpercent', # VectorBT exact string match
       fees=0.001,                # 10 bps transaction fee
       freq='1D',
       cash_sharing=True
   )
   Metrics Evaluated:Total Cumulative Portfolio Return ($\%$)Annualized Sharpe RatioMaximum Drawdown ($\%$)Full Portfolio Value Equity Curve (portfolio.value().vbt.plot())