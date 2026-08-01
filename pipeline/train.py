import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from models.gaeco_network import GAECONetPipeline
from loss.risk_loss import NetSortinoLoss


def _extract_node_features(window_ret: torch.Tensor, curr_idx_val: torch.Tensor, num_assets: int) -> torch.Tensor:
    """
    Computes the 6 node features for a given lookback window of returns.

    `window_ret` must contain ONLY data strictly before the decision day
    (i.e. it must not include the return of the day being decided about).
    `curr_idx_val` is the most recent known observation (typically
    `window_ret[-1]`) and is used as the reference point for the drawdown
    feature.
    """
    vol_20 = torch.std(window_ret[-20:], dim=0, correction=1) if window_ret[-20:].shape[0] > 1 else torch.zeros(num_assets)
    mom_10 = torch.mean(window_ret[-10:], dim=0)
    mom_60 = torch.mean(window_ret[-60:], dim=0)
    skew_30 = torch.mean(((window_ret[-30:] - mom_10) / (vol_20 + 1e-8)) ** 3, dim=0) if window_ret[-30:].shape[0] > 0 else torch.zeros(num_assets)
    downside_ret = torch.clamp(window_ret[-20:], max=0.0)
    downside_vol = torch.std(downside_ret, dim=0, correction=1) if downside_ret.shape[0] > 1 else torch.zeros(num_assets)
    win_20 = window_ret[-20:]
    peak = torch.max(win_20, dim=0)[0] if win_20.shape[0] > 0 else torch.zeros(num_assets)
    drawdown = (peak - curr_idx_val) / (peak + 1e-8)

    return torch.stack([vol_20, mom_10, mom_60, skew_30, downside_vol, drawdown], dim=-1).unsqueeze(0)


def generate_cbb_synthetic_data(returns_tensor: torch.Tensor, block_length: int = 20, target_length: int = 252, lookback: int = 60):
    """
    Generates a Circular Block Bootstrap (CBB) sequence of returns and precomputes
    all graph and spectral inputs required by GAECO-Net. Based on Karzanov et al. (2025).
    """
    T, N = returns_tensor.shape
    num_blocks = int(np.ceil((target_length + lookback) / block_length))
    
    # Circular extension along time dimension
    circular_returns = torch.cat([returns_tensor, returns_tensor[:block_length]], dim=0)
    
    blocks = []
    for _ in range(num_blocks):
        start_idx = np.random.randint(0, T)
        block = circular_returns[start_idx : start_idx + block_length]
        blocks.append(block)
        
    synth_returns = torch.cat(blocks, dim=0)[:target_length + lookback]
    
    synth_features, synth_laplacians, synth_evals, synth_evecs = [], [], [], []
    
    for t in range(lookback, len(synth_returns)):
        # window_ret = synth_returns[t-lookback : t] uses data strictly
        # BEFORE day t (does not include synth_returns[t]) -- causal by
        # construction, already correct in the original file.
        window_ret = synth_returns[t - lookback : t]
        
        # 1. Feature extraction (curr_idx_val = window_ret[-1] = synth_returns[t-1],
        #    the most recent observation strictly before day t)
        node_feats = _extract_node_features(window_ret, synth_returns[t - 1], N)
        
        # 2. Covariance and Correlation
        cov_mat = torch.cov(window_ret.T)
        std_devs = torch.sqrt(torch.diag(cov_mat)).unsqueeze(1) + 1e-8
        corr_mat = cov_mat / (std_devs @ std_devs.T)
        corr_mat = torch.nan_to_num(corr_mat, nan=0.0)
        corr_mat.fill_diagonal_(1.0)
        
        # 3. Graph Laplacian
        deg = torch.diag(torch.sum(corr_mat, dim=-1))
        laplacian = (deg - corr_mat).unsqueeze(0)
        
        # 4. Spectral Decomposition
        evals, evecs = torch.linalg.eigh(corr_mat)
        idx = torch.argsort(evals, descending=True)
        evals = evals[idx].unsqueeze(0)
        evecs = evecs[:, idx].unsqueeze(0)
        
        synth_features.append(node_feats)
        synth_laplacians.append(laplacian)
        synth_evals.append(evals)
        synth_evecs.append(evecs)
        
    # synth_features[i] / synth_laplacians[i] / ... were built from
    # synth_returns[i : i+lookback], i.e. strictly before day (i + lookback).
    # synth_returns[lookback:][i] == synth_returns[i + lookback] is therefore
    # exactly the "next day" return relative to synth_features[i] -- this is
    # what training should score each step against (see FIX below).
    return synth_returns[lookback:], synth_features, synth_laplacians, synth_evals, synth_evecs


def train_gaeco_net(
    returns_df: pd.DataFrame,
    empirical_data: dict,
    epochs: int = 20,
    synthetic_epochs: int = 10,
    block_length:int = 10,
    lookback: int = 60,
    horizon: int = 10,           # 10-day forward horizon matching 2W rebalance schedule
    risk_aversion: float = 0.5,
    lambda_turnover: float = 0.01,
    lr: float = 1e-3,
    model: torch.nn.Module | None = None
):
    """
    Trains GAECO-Net using a multi-day forward cumulative return target (H = horizon).
    """
    num_assets = returns_df.shape[1]
    
    if model is None:
        model = GAECONetPipeline(num_assets=num_assets, in_features=6, risk_aversion=risk_aversion)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = NetSortinoLoss(fee_rate=0.0010, return_weight=2.0)

    returns_tensor = torch.tensor(returns_df.values, dtype=torch.float32)
    num_timesteps = len(returns_df)
    num_windows = len(empirical_data["corr_emp"])

    # Prevent boundary overflow when slicing target returns [t + lookback : t + lookback + horizon]
    last_valid_idx = min(num_timesteps - lookback - horizon, num_windows - 1)

    # -------------------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------------------
    if epochs > 0:
        model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            prev_weights = None

            for t in range(last_valid_idx + 1):
                # 1. Slice node features & empirical tensors for window t
                window_ret = returns_tensor[t : t + lookback]
                curr_idx_val = returns_tensor[t + lookback - 1]
                node_feat = _extract_node_features(window_ret, curr_idx_val, num_assets)

                sample_corr = empirical_data["corr_emp"][t]
                sample_lap = (torch.eye(num_assets) - sample_corr).unsqueeze(0)
                eigenvals = empirical_data["eigenvals"][t:t+1]
                eigenvecs = empirical_data["eigenvecs"][t:t+1]

                # 2. Slice 10-day Forward Cumulative Target Returns
                forward_returns = returns_tensor[t + lookback : t + lookback + horizon].sum(dim=0, keepdim=True)

                # 3. Model Forward Pass
                weights, C_clean, mu, gamma = model(node_feat, sample_lap, eigenvals, eigenvecs)

                # 4. Compute Loss over Forward Horizon
                loss = criterion(
                    weights=weights,
                    next_returns=forward_returns,
                    realized_cov=C_clean,
                    prev_weights=prev_weights
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                prev_weights = weights.detach()
                total_loss += loss.item()

            if (epoch + 1) % 5 == 0 or epoch == 0:
                avg_epoch_loss = total_loss / (last_valid_idx + 1)
                print(f"Epoch [{epoch+1}/{epochs}] - Avg Loss: {avg_epoch_loss:.6f}")

    # -------------------------------------------------------------------------
    # Out-of-Sample Prediction / Allocation Mode
    # -------------------------------------------------------------------------
    model.eval()
    agent_weights = []

    with torch.no_grad():
        for t in range(len(empirical_data["corr_emp"])):
            if t + lookback > num_timesteps:
                break
                
            window_ret = returns_tensor[t : t + lookback]
            curr_idx_val = returns_tensor[t + lookback - 1]
            node_feat = _extract_node_features(window_ret, curr_idx_val, num_assets)

            sample_corr = empirical_data["corr_emp"][t]
            sample_lap = (torch.eye(num_assets) - sample_corr).unsqueeze(0)
            eigenvals = empirical_data["eigenvals"][t:t+1]
            eigenvecs = empirical_data["eigenvecs"][t:t+1]

            weights, _, _, _ = model(node_feat, sample_lap, eigenvals, eigenvecs)
            agent_weights.append(weights.squeeze(0).cpu().numpy())

    # Align allocations with exact empirical matrix evaluation dates
    weights_array = np.array(agent_weights)
    eval_dates = empirical_data["dates"][:len(weights_array)]
    
    weights_df = pd.DataFrame(
        weights_array,
        index=eval_dates,
        columns=returns_df.columns
    )

    return weights_df, model

def train_gaeco_net_ensemble(
    returns_df: pd.DataFrame,
    empirical_data: dict,
    n_agents: int = 5,
    base_seed: int = 42,
    epochs: int = 10,
    synthetic_epochs: int = 10,
    block_length: int = 20,
    lr: float = 1e-3,
    lookback: int = 60,
    risk_aversion: float = 1.0,
    lambda_turnover: float = 0.005
) -> pd.DataFrame:
    """
    Trains an ensemble of N independent GAECO-Net agents across different synthetic seeds,
    and returns the averaged allocation weight DataFrame.
    """
    all_agent_weights = []

    for agent_id in range(n_agents):
        seed = base_seed + agent_id
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f"\n================ Train Agent {agent_id + 1}/{n_agents} (Seed: {seed}) ================")

        weights_df, _ = train_gaeco_net(
            returns_df=returns_df,
            empirical_data=empirical_data,
            epochs=epochs,
            synthetic_epochs=synthetic_epochs,
            block_length=block_length,
            lr=lr,
            lookback=lookback,
            risk_aversion=risk_aversion,
            lambda_turnover=lambda_turnover,
            model=None
        )
        all_agent_weights.append(weights_df)

    ensemble_weights_array = np.mean([df.values for df in all_agent_weights], axis=0)
    
    ensemble_weights_df = pd.DataFrame(
        ensemble_weights_array,
        index=all_agent_weights[0].index,
        columns=all_agent_weights[0].columns
    )

    # Renormalize rows to sum to 1.0
    ensemble_weights_df = ensemble_weights_df.div(ensemble_weights_df.sum(axis=1), axis=0)
    
    print("\n Ensembling Complete! Averaged portfolio weights across all agents.")
    return ensemble_weights_df