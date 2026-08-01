# GAECO-NET

**GAECO-Net (Graph-Aware EigenCleaning Optimizer)** is an end-to-end, graph-informed deep learning framework for robust portfolio optimization. It combines Graph Neural Networks (GNNs), Random Matrix Theory (RMT) spectral regularization, and realistic trade execution constraints to produce low-turnover, risk-managed portfolios.

---

## 🎯 Key Innovation & Methodology

Classical mean-variance optimization and neural portfolio allocation models often underperform in live trading environments due to two primary challenges:

1. **Sample Covariance Noise:** In high-dimensional regimes where the asset count $N$ is large relative to historical lookback $T$, empirical correlation matrices exhibit extreme spectral distortion and noise, leading to unstable allocations.
2. **Transaction Cost Drag:** Unconstrained neural models frequently propose daily micro-allocations. Processing continuous floating-point adjustments generates excessive turnover, causing transaction fees to drain strategy returns over time.

### Quantitative Pillars

* **Graph Structure Learning:** Uses dynamic correlation thresholding to construct normalized Graph Laplacians ($\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}$), allowing GNN message passing to learn structural dependencies and asset relationships.
* **Spectral Regularization (EigenCleaning):** Isolates informative signal from high-frequency noise in empirical correlation matrices using Random Matrix Theory (RMT) principles.
* **Out-of-Sample Discipline:** Maintains strict temporal separation between training windows and evaluation windows to eliminate lookahead bias and data leakage.
* **Execution Constraints:** Integrates periodic rebalancing schedules, turnover penalties during training, and target allocation deadbands to minimize trading churn and manage friction costs.

---

## 📂 Repository & File Structure

```text
Graph-Aware-EigenCleaning-Optimizer-GAECO-/
│
├── data/
│   ├── loader.py              # WRDS/CRSP Data loader & rolling matrix calculator
│   └── graph_builder.py       # Graph construction & normalized Laplacian builder
│
├── models/
│   └── gaeco_network.py       # Core GNN & Spectral Cleaning neural architecture
│
├── pipeline/
│   └── train.py               # Multi-agent ensemble training pipeline
│
├── benchmarks/
│   └── estimators.py          # Baseline estimators (1/N, Min-Var, Shrinkage)
│
├── backtester/
│   └── engine.py              # VectorBT backtesting engine & execution constraints
│
├── main.py                    # Master execution workflow script
├── README.md                  # Project documentation
└── requirements.txt           # Environment dependencies
🧩 Module Roles & Component Responsibilities
1. data/loader.py — Data Acquisition & Rolling Matrices
CRSP Data Ingestion: Pulls historical daily asset returns over designated date ranges via the WRDS API.

Rolling Estimations: Pre-computes rolling empirical correlation matrices over a configurable historical lookback window (default: 60 days).

Spectral Decomposition: Extracts daily rolling eigenvalues and eigenvectors required for spectral regularization.

2. data/graph_builder.py — Graph Construction
Adjacency Estimation: Applies correlation thresholds to build dynamic asset connectivity graphs.

Laplacian Normalization: Computes normalized Graph Laplacians to enable graph convolutions across network nodes.

3. models/gaeco_network.py — Neural Architecture
Node Feature Processing: Formulates asset-level input features combining historical returns and topological graph metrics.

Graph Convolutional Network (GCN): Passes messages across the asset connectivity graph to capture non-linear relational features.

Spectral Cleaning Head: Combines GNN embeddings with filtered spectral decompositions to output normalized, long-only portfolio weights (∑w 
i
​
 =1,w 
i
​
 ≥0).

4. pipeline/train.py — Multi-Agent Ensemble & Loss Function
Ensemble Learning: Trains multiple independent neural agents with distinct random initialization seeds to average out individual model variance.

Turnover Penalty Optimization: Trains the network using a joint loss function that balances risk-adjusted return against allocation stability:

L 
total
​
 =L 
risk
​
 +λ 
turnover
​
 ⋅∥w 
t
​
 −w 
t−1
​
 ∥ 
1
​
 
5. benchmarks/estimators.py — Comparative Baselines
Provides standard non-neural benchmark strategies for baseline comparison:

Equal Weight (1/N): Static uniform allocation across the asset universe.

Global Minimum Variance (GMV): Allocation minimizing empirical portfolio variance.

Shrinkage Estimators: Covariance estimation adjusted via Ledoit-Wolf shrinkage.

6. backtester/engine.py — Backtesting & Execution Engine
Execution Constraints:

Periodic Resampling: Resamples raw neural weight outputs to execute strictly on fixed schedules (e.g., bi-weekly Fridays: 2W-FRI).

Deadband Filter: Holds current asset weights static if target allocation drift falls below a specified deadband threshold (e.g., 2%), ignoring non-essential micro-trades.

Vectorized Backtesting: Runs vectorized backtests via VectorBT using price series, accounting for explicit transaction fees (default: 10 bps).

Comparative Summary: Prints side-by-side performance metrics (Total Return, Sharpe Ratio, Sortino Ratio, Max Drawdown, Total Trades, Fees Paid) comparing GAECO-Net to benchmark strategies under identical friction rules.

7. main.py — Master Workflow Script
Orchestrates data pipeline execution, train/test splitting, ensemble training on historical data, out-of-sample backtesting, and performance visualization.

⚙️ Key Engineering & Execution Logic
Temporal Separation & Zero-Leakage Policy
To prevent lookahead bias, training returns and empirical matrices are strictly isolated to the training date window. Model inference and weight generation run out-of-sample on unseen test periods.

Transaction Cost Reduction Pipeline
Strategy turnover and fee impact are managed across three distinct layers:

Model Layer: Turnover penalty added to the loss function during training.

Scheduling Layer: Execution weights strictly resampled to periodic schedules (e.g., 2W-FRI).

Execution Layer: Allocation deadband threshold suppresses minor drift adjustments.

Index Alignment Standards
All dataset indices, matrix structures, and backtest inputs are normalized to unified date-time structures (pd.DatetimeIndex) to ensure alignment across pandas operations and VectorBT backtests.