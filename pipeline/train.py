import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from models.gaeco_network import GAECONetPipeline
from loss.risk_loss import NetSharpeLoss

def train_gaeco_net(
    returns_df: pd.DataFrame,
    empirical_data: dict,
    epochs: int = 20,
    lr: float = 1e-3,
    lookback: int = 60,
    risk_aversion: float = 1.0,
    lambda_turnover: float = 0.005,
    model: nn.Module | None=None
) -> tuple[pd.DataFrame, nn.Module]:
    """
    Trains GAECO-Net with Mean-Variance utility optimization or evaluates an existing model (epochs=0).
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
    
    # Run at least 1 epoch for inference if epochs == 0
    num_epochs = max(1, epochs)
    is_eval_only = (epochs == 0)

    if is_eval_only:
        model.eval()
        print(f"Starting Inference Pass over {num_windows} rolling windows...")
    else:
        model.train()
        print(f"Starting Training: {epochs} Epochs over {num_windows} rolling windows...")

    max_steps = min(num_windows, num_timesteps - lookback)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        prev_w = None

        for t in range(max_steps):
            curr_idx = min(t + lookback, num_timesteps - 1)
            
            # Safe Feature Slicing
            start_idx = max(0, curr_idx - lookback + 1)
            window_ret = returns_tensor[start_idx : curr_idx + 1]
            if window_ret.shape[0] == 0:
                continue

            # Feature calculations
            vol_20 = torch.std(window_ret[-20:], dim=0, correction=1) if window_ret[-20:].shape[0] > 1 else torch.zeros(num_assets)
            mom_10 = torch.mean(window_ret[-10:], dim=0)
            mom_60 = torch.mean(window_ret[-60:], dim=0)
            skew_30 = torch.mean(((window_ret[-30:] - mom_10) / (vol_20 + 1e-8)) ** 3, dim=0) if window_ret[-30:].shape[0] > 0 else torch.zeros(num_assets)
            downside_ret = torch.clamp(window_ret[-20:], max=0.0)
            downside_vol = torch.std(downside_ret, dim=0, correction=1) if downside_ret.shape[0] > 1 else torch.zeros(num_assets)
            win_20 = window_ret[-20:]
            peak = torch.max(win_20, dim=0)[0] if win_20.shape[0] > 0 else torch.zeros(num_assets)
            drawdown = (peak - returns_tensor[curr_idx]) / (peak + 1e-8)

            node_features = torch.stack(
                [vol_20, mom_10, mom_60, skew_30, downside_vol, drawdown], 
                dim=-1
            ).unsqueeze(0)

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

            # Record weights on the final epoch or during inference
            if epoch == num_epochs - 1:
                weights_over_time.append(weights.squeeze(0).detach().cpu().numpy())

        if not is_eval_only and ((epoch + 1) % 5 == 0 or epoch == 0):
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss / max_steps:.4f}")

    # Build DataFrame safely
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