import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from models.gaeco_network import GAECONetPipeline
from loss.risk_loss import NetSharpeLoss


def _extract_node_features(window_ret: torch.Tensor, curr_idx_val: torch.Tensor, num_assets: int) -> torch.Tensor:
    """
    Computes the 6 node features for a given lookback window of returns.
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
        window_ret = synth_returns[t - lookback : t]
        
        # 1. Feature extraction
        node_feats = _extract_node_features(window_ret, synth_returns[t-1], N)
        
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
        
    return synth_returns[lookback:], synth_features, synth_laplacians, synth_evals, synth_evecs


def train_gaeco_net(
    returns_df: pd.DataFrame,
    empirical_data: dict,
    epochs: int = 20,
    synthetic_epochs: int = 10,
    block_length: int = 20,
    lr: float = 1e-3,
    lookback: int = 60,
    risk_aversion: float = 1.0,
    lambda_turnover: float = 0.005,
    model: nn.Module | None = None
) -> tuple[pd.DataFrame, nn.Module]:
    """
    Trains GAECO-Net with optional Synthetic CBB Pre-training before fine-tuning on real data.
    """
    returns_tensor = torch.tensor(returns_df.values, dtype=torch.float32)
    num_timesteps, num_assets = returns_tensor.shape
    
    corr_emp = empirical_data["corr_emp"]
    eigenvals = empirical_data["eigenvals"]
    eigenvecs = empirical_data["eigenvecs"]

    # Instantiate model if none is provided
    if model is None:
        model = GAECONetPipeline(
            num_assets=num_assets, 
            in_features=6, 
            hidden_dim=64, 
            risk_aversion=risk_aversion
        )
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = NetSharpeLoss(fee_rate=lambda_turnover)
    
    num_windows = len(corr_emp)
    weights_over_time = []
    
    is_eval_only = (epochs == 0)

    # -------------------------------------------------------------------------
    # PHASE 1: SYNTHETIC BLOCK BOOTSTRAP PRE-TRAINING
    # -------------------------------------------------------------------------
    if not is_eval_only and synthetic_epochs > 0:
        model.train()
        print(f"--- Starting Phase 1: Synthetic CBB Pre-training ({synthetic_epochs} Epochs) ---")
        
        for synth_epoch in range(synthetic_epochs):
            synth_loss = 0.0
            prev_w_synth = None
            
            # Generate 1-year equivalent synthetic market sequence
            synth_R, synth_X, synth_L, synth_evals, synth_evecs = generate_cbb_synthetic_data(
                returns_tensor=returns_tensor,
                block_length=block_length,
                target_length=252,
                lookback=lookback
            )
            
            for t in range(len(synth_R) - 1):
                optimizer.zero_grad()
                
                weights, _, _, _ = model(synth_X[t], synth_L[t], synth_evals[t], synth_evecs[t])
                
                next_ret = synth_R[t + 1].unsqueeze(0)
                realized_cov = (next_ret.T @ next_ret).unsqueeze(0)
                
                loss = loss_fn(
                    weights=weights,
                    next_returns=next_ret,
                    realized_cov=realized_cov,
                    prev_weights=prev_w_synth
                )
                
                loss.backward()
                optimizer.step()
                
                synth_loss += loss.item()
                prev_w_synth = weights.detach()
                
            if (synth_epoch + 1) % 2 == 0 or synth_epoch == 0:
                print(f"Synth Epoch {synth_epoch + 1}/{synthetic_epochs} - Avg Loss: {synth_loss / len(synth_R):.4f}")
        
        print("--- Synthetic Pre-training Complete! ---")

    # -------------------------------------------------------------------------
    # PHASE 2: REAL DATA TRAINING / INFERENCE
    # -------------------------------------------------------------------------
    num_epochs = max(1, epochs)

    if is_eval_only:
        model.eval()
        print(f"Starting Inference Pass over {num_windows} rolling windows...")
    else:
        model.train()
        print(f"--- Starting Phase 2: Fine-tuning on Real Data ({epochs} Epochs) ---")

    max_steps = min(num_windows, num_timesteps - lookback)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        prev_w = None

        for t in range(max_steps):
            curr_idx = min(t + lookback, num_timesteps - 1)
            
            start_idx = max(0, curr_idx - lookback + 1)
            window_ret = returns_tensor[start_idx : curr_idx + 1]
            if window_ret.shape[0] == 0:
                continue

            node_features = _extract_node_features(window_ret, returns_tensor[curr_idx], num_assets)

            corr_mat = corr_emp[t].unsqueeze(0)
            evals = eigenvals[t].unsqueeze(0)
            evecs = eigenvecs[t].unsqueeze(0)
            
            deg = torch.diag_embed(torch.sum(corr_mat, dim=-1))
            laplacian = deg - corr_mat

            next_idx = min(curr_idx + 1, num_timesteps - 1)
            next_returns = returns_tensor[next_idx].unsqueeze(0)
            next_realized_cov = torch.cov(window_ret.T).unsqueeze(0)

            if is_eval_only:
                with torch.no_grad():
                    weights, C_clean, mu, gamma = model(node_features, laplacian, evals, evecs)
            else:
                optimizer.zero_grad()
                weights, C_clean, mu, gamma = model(node_features, laplacian, evals, evecs)
                loss = loss_fn(
                    weights=weights, 
                    next_returns=next_returns, 
                    realized_cov=next_realized_cov, 
                    prev_weights=prev_w
                )
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            prev_w = weights.detach()

            if epoch == num_epochs - 1:
                weights_over_time.append(weights.squeeze(0).detach().cpu().numpy())

        if not is_eval_only and ((epoch + 1) % 5 == 0 or epoch == 0):
            print(f"Real Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss / max_steps:.4f}")

    if len(weights_over_time) == 0:
        weights_df = pd.DataFrame(0.0, index=returns_df.index, columns=returns_df.columns)
    else:
        weights_array = np.array(weights_over_time)
        weights_df = pd.DataFrame(
            weights_array,
            index=returns_df.index[lookback - 1 : lookback - 1 + len(weights_array)],
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