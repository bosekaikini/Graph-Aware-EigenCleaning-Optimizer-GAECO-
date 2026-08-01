# backtester/engine.py

import pandas as pd
import numpy as np
import torch
import vectorbt as vbt

from data.graph_builder import GraphBuilder
from models.gaeco_network import GAECONetPipeline

def apply_rebalance_deadband(weights_df: pd.DataFrame, threshold: float = 0.02) -> pd.DataFrame:
    """
    Suppresses trades if allocation shift from currently held weight is < threshold.
    """
    if weights_df.empty:
        return weights_df

    clean_weights = weights_df.astype('float64').copy()    
    current_w = clean_weights.iloc[0].to_numpy(dtype=np.float64).copy()
    
    for t in range(1, len(clean_weights)):
        target_w = clean_weights.iloc[t].to_numpy(dtype=np.float64)
        delta = np.abs(target_w - current_w)
        
        # Shift only if drift > deadband threshold
        mask = delta > threshold
        new_w = np.where(mask, target_w, current_w)
        
        if np.sum(new_w) > 0:
            new_w = new_w / np.sum(new_w)
            
        clean_weights.iloc[t] = new_w
        current_w = new_w.copy()
        
    return clean_weights


def process_execution_weights(
    weights_df: pd.DataFrame, 
    rebalance_freq: str = '2W-FRI', 
    deadband_threshold: float = 0.02
) -> pd.DataFrame:
    """
    Resamples weights to rebalance frequency and applies deadband filtering safely.
    """
    if weights_df.empty:
        raise ValueError("process_execution_weights received an empty weights_df DataFrame.")

    clean_df = weights_df.copy()
    
    # Ensure index is DatetimeIndex before resampling
    if not isinstance(clean_df.index, pd.DatetimeIndex):
        clean_df.index = pd.to_datetime(clean_df.index)

    # 1. Resample strictly to target schedule
    resampled_df = clean_df.resample(rebalance_freq).last().dropna(how='all')

    # Fallback if resampling produces empty DataFrame (e.g., date range shorter than rebalance_freq)
    if resampled_df.empty:
        resampled_df = clean_df.copy()

    # 2. Apply deadband filter to resampled rebalance snapshots
    filtered_resampled = apply_rebalance_deadband(resampled_df, threshold=deadband_threshold)

    # 3. Forward-fill daily weights back across the full backtest calendar
    daily_weights = filtered_resampled.reindex(clean_df.index, method='ffill').bfill()
    return daily_weights


# backtester/engine.py

def run_vectorbt_backtest(
    model_checkpoint: str | None = None,
    weights_df: pd.DataFrame | None = None,
    returns_df: pd.DataFrame | None = None,
    data_dict: dict | None = None,
    num_assets: int = 50,
    fee_rate: float = 0.001,           # 10 bps transaction fee
    rebalance_freq: str = '2W-FRI',     # 'W-FRI', '2W-FRI', 'M'
    deadband_threshold: float = 0.02   # 2% deadband
) -> vbt.Portfolio:
    """
    Executes a fee-adjusted vectorized backtest via VectorBT with zero lookahead bias
    and frequency-aligned price/weight matching to eliminate daily drift trades.
    """
    device = torch.device("cpu")
    
    if returns_df is None:
        raise ValueError("returns_df must be provided.")

    if weights_df is None:
        if model_checkpoint is None or data_dict is None:
            raise ValueError("Must provide weights_df OR model_checkpoint & data_dict.")
        
        model = GAECONetPipeline(num_assets=num_assets, in_features=2, hidden_dim=64).to(device)
        model.load_state_dict(torch.load(model_checkpoint, map_location=device))
        model.eval()

        corr_emp = data_dict["corr_emp"].to(device)
        eigenvals = data_dict["eigenvals"].to(device)
        eigenvecs = data_dict["eigenvecs"].to(device)
        dates = data_dict["dates"]

        graph_builder = GraphBuilder(num_assets=num_assets, alpha=0.0, threshold=0.2)
        weights_list = []

        with torch.no_grad():
            for t in range(len(dates)):
                c_emp_t = corr_emp[t:t+1]
                evals_t = eigenvals[t:t+1]
                evecs_t = eigenvecs[t:t+1]

                adj_t = graph_builder.build_hybrid_adjacency(c_emp_t)
                lap_t = graph_builder.compute_normalized_laplacian(adj_t)
                node_features = torch.ones((1, num_assets, 2), device=device)

                weights, _, _, _ = model(node_features, lap_t, evals_t, evecs_t)
                weights_list.append(weights.squeeze(0).cpu().numpy())

        raw_weights_df = pd.DataFrame(
            np.array(weights_list), 
            index=pd.to_datetime(dates), 
            columns=returns_df.columns
        )
    else:
        raw_weights_df = weights_df.copy()

    # Align dates cleanly
    returns_df = returns_df.copy()
    returns_df.index = pd.to_datetime(returns_df.index)
    raw_weights_df.index = pd.to_datetime(raw_weights_df.index)

    # -------------------------------------------------------------------------
    # FIX 1: Strict Shift to Eliminate Lookahead Bias (w_t executes on r_{t+1})
    # -------------------------------------------------------------------------
    raw_weights_df = raw_weights_df.shift(1).fillna(1.0 / returns_df.shape[1])

    common_dates = returns_df.index.intersection(raw_weights_df.index)
    aligned_returns = returns_df.loc[common_dates]
    raw_weights_df = raw_weights_df.loc[common_dates]

    # Process periodic execution weights & apply deadband filter
    clean_weights_df = process_execution_weights(
        raw_weights_df, 
        rebalance_freq=rebalance_freq, 
        deadband_threshold=deadband_threshold
    )

    # Calculate daily cumulative price index
    daily_price_df = (1.0 + aligned_returns).cumprod()

    # -------------------------------------------------------------------------
    # FIX 2: Resample Prices to Rebalancing Schedule
    # Running from_orders on daily prices causes passive position drift to 
    # trigger orders daily. Resampling both prices & weights to the execution 
    # schedule ensures orders only execute strictly on rebalance dates (e.g. 2W-FRI).
    # -------------------------------------------------------------------------
    exec_prices = daily_price_df.resample(rebalance_freq).last().dropna()
    exec_weights = clean_weights_df.reindex(exec_prices.index, method='ffill')

    print(f"Running VectorBT Strategy Backtest (Execution points: {len(exec_prices)} cycles)...")
    portfolio = vbt.Portfolio.from_orders(
        close=exec_prices,
        size=exec_weights,
        size_type='targetpercent',
        fees=fee_rate,
        freq='1D',
        cash_sharing=True
    )

    return portfolio

def run_covariance_benchmark_backtest(
    returns_df: pd.DataFrame,
    estimator_func,
    lookback: int = 60,
    fee_rate: float = 0.001,
    rebalance_freq: str = '14D',
    deadband_threshold: float = 0.02
) -> vbt.Portfolio:
    """
    Transforms a baseline matrix estimator (Ledoit-Wolf / Marchenko-Pastur) into 
    a Global Minimum Variance (GMV) tracking portfolio matching your main execution path.
    """
    print(f"Generating weights utilizing estimator: {estimator_func.__name__}...")
    weights_list = []
    dates = returns_df.index
    num_assets = returns_df.shape[1]
    
    # Fill target lookback window with an equal weight matrix
    for _ in range(lookback):
        weights_list.append(np.ones(num_assets) / num_assets)
        
    # Generate historical out-of-sample GMV weight tracks
    for i in range(lookback, len(returns_df)):
        window_returns = returns_df.iloc[i - lookback:i].to_numpy()
        
        try:
            # Generate your custom filtered covariance matrix structure
            cov_matrix = estimator_func(window_returns)
            
            # Formulate the Global Minimum Variance optimization weights vector
            inv_cov = np.linalg.pinv(cov_matrix)
            ones = np.ones(num_assets)
            raw_weights = inv_cov @ ones
            
            # Apply standard non-negative (long-only) constraints and scale to 1.0
            raw_weights = np.clip(raw_weights, 0, None)
            if raw_weights.sum() > 0:
                weights = raw_weights / raw_weights.sum()
            else:
                weights = np.ones(num_assets) / num_assets
        except Exception:
            weights = np.ones(num_assets) / num_assets
            
        weights_list.append(weights)
        
    # Reconstruct into a clean pandas data container
    raw_bench_weights = pd.DataFrame(weights_list, index=dates, columns=returns_df.columns)
    
    # Shift weights by 1 trading day to strictly eliminate target lookahead bias
    raw_bench_weights = raw_bench_weights.shift(1).fillna(1.0 / num_assets)
    
    # Process weights using your exact engine rebalance layout
    clean_bench_weights = process_execution_weights(
        raw_bench_weights, 
        rebalance_freq=rebalance_freq, 
        deadband_threshold=deadband_threshold
    )
    
    price_df = (1.0 + returns_df).cumprod()
    
    # Run through your unified vectorbt instance path
    bench_portfolio = vbt.Portfolio.from_orders(
        close=price_df,
        size=clean_bench_weights,
        size_type='targetpercent',
        fees=fee_rate,
        freq='1D',
        cash_sharing=True
    )
    return bench_portfolio