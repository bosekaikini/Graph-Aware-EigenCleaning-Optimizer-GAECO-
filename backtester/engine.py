#vectorbt implementation exclusively
#month vs 2week vs week test periods
import pandas as pd
import numpy as np
import torch
import vectorbt as vbt

from data.graph_builder import GraphBuilder
from models.gaeco_network import GAECONetPipeline

# backtester/engine.py

import numpy as np
import pandas as pd

def apply_rebalance_deadband(weights_df: pd.DataFrame, threshold: float = 0.02) -> pd.DataFrame:
    """
    Suppresses trades if the allocation shift from current portfolio drift weight is < threshold.
    """
    clean_weights = weights_df.astype('float64').copy()    
    current_w = clean_weights.iloc[0].to_numpy(dtype=np.float64).copy()
    
    for t in range(1, len(clean_weights)):
        target_w = clean_weights.iloc[t].to_numpy(dtype=np.float64)
        delta = np.abs(target_w - current_w)
        
        # Only adjust weight if drift exceeds threshold
        mask = delta > threshold
        new_w = np.where(mask, target_w, current_w)
        
        # Renormalize to maintain 100% allocation
        if np.sum(new_w) > 0:
            new_w = new_w / np.sum(new_w)
            
        clean_weights.iloc[t] = new_w
        current_w = new_w
        
    return clean_weights

def process_execution_weights(
    weights_df: pd.DataFrame, 
    rebalance_freq: str = '2W-FRI', 
    deadband_threshold: float = 0.02
) -> pd.DataFrame:
    """
    Resamples weights to rebalance frequency and applies deadband filtering.
    """
    # Resample daily outputs to weekly/bi-weekly target dates
    resampled_weights = weights_df.resample(rebalance_freq).last().reindex(weights_df.index, method='ffill')
    
    # Apply deadband filter
    filtered_weights = apply_rebalance_deadband(resampled_weights, threshold=deadband_threshold)
    return filtered_weights

def run_vectorbt_backtest(
    model_checkpoint: str,
    returns_df: pd.DataFrame,
    data_dict: dict,
    num_assets: int = 50,
    fee_rate: float = 0.001, # 10 bps transaction cost
    freq: str = '2W-FRI'
):
    """
    Generates out-of-sample portfolio allocation weights using GAEC-Net
    and executes a vectorized backtest via VectorBT.
    """
    device = torch.device("cpu")
    
    # 1. Load Trained Core Model
    model = GAECONetPipeline(num_assets=num_assets, in_features=2, hidden_dim=64).to(device)
    model.load_state_dict(torch.load(model_checkpoint, map_location=device))
    model.eval()

    corr_emp = data_dict["corr_emp"].to(device)
    eigenvals = data_dict["eigenvals"].to(device)
    eigenvecs = data_dict["eigenvecs"].to(device)
    dates = data_dict["dates"]

    graph_builder = GraphBuilder(num_assets=num_assets, alpha=0.0, threshold=0.2)
    weights_list = []

    print("Generating out-of-sample portfolio weights...")
    with torch.no_grad():
        for t in range(len(dates)):
            c_emp_t = corr_emp[t:t+1]
            evals_t = eigenvals[t:t+1]
            evecs_t = eigenvecs[t:t+1]

            adj_t = graph_builder.build_hybrid_adjacency(c_emp_t)
            lap_t = graph_builder.compute_normalized_laplacian(adj_t)

            # Node features
            node_features = torch.ones((1, num_assets, 2), device=device)

            weights, _, _ = model(node_features, lap_t, evals_t, evecs_t)
            weights_list.append(weights.squeeze(0).cpu().numpy())

    # 2. Build Allocation Weights DataFrame [Dates x Assets]
    weights_matrix = np.array(weights_list)
    aligned_returns = returns_df.loc[dates]
    
    weights_df = pd.DataFrame(
        weights_matrix, 
        index=aligned_returns.index, 
        columns=aligned_returns.columns
    )

    # Convert returns into cumulative price index for VectorBT
    price_df = (1 + aligned_returns).cumprod()

    # 3. VectorBT Backtest Execution
    print("Running VectorBT Backtest...")
    portfolio = vbt.Portfolio.from_orders(
        close=price_df,
        size=weights_df,
        size_type='targetpercent',  # Target portfolio weight allocation
        fees=fee_rate,              # Apply transaction costs on rebalancing
        freq=freq
    )

    # 4. Print & Plot Institutional Performance
    print("\n================ GAEC-Net Performance Stats ================")
    print(portfolio.stats())
    
    return portfolio