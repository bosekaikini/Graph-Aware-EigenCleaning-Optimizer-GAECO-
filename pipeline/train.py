import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from models.gaeco_network import GAECONetPipeline
from loss.risk_loss import NetSortinoLoss


def _extract_node_features(window_ret: torch.Tensor, curr_idx_val: torch.Tensor, num_assets: int) -> torch.Tensor:
    """
    Computes the 8 node features for a given lookback window of returns.

    FIX (see "GAECO statistically doesn't have better Sharpe"): the
    previous 6 features (vol_20, mom_10, mom_60, skew_30, downside_vol,
    drawdown) are all risk/volatility descriptors -- mom_10 and mom_60 are
    the only remotely directional signals, and they're both noisy,
    unsmoothed point momentum. A Jobson-Korkie z-statistic near 0 (as
    observed) is consistent with the model having little genuine
    directional signal to allocate on, so it ends up close to the
    equal-weight/benchmark return stream. Two additional features are
    added here, both computable from the same `window_ret` slice already
    being passed in (no signature or lookback change required):

      - reversal_5: 5-day mean return, a proxy for the well-documented
        short-horizon equity reversal effect (recent losers/winners tend
        to partially revert over the next few days). The return head is
        free to learn either sign.
      - mom_accel: mom_10 - mom_60, a trend-acceleration feature (is
        recent momentum speeding up or decelerating relative to the
        medium-term trend), distinct information from either momentum
        term alone.

    A longer 6-12 month momentum feature (the classic 12-1 month
    cross-sectional momentum factor) is NOT added here because it would
    require a window longer than the 60-day `lookback` this function is
    always called with; adding it properly requires threading a separate,
    longer historical slice through the call sites in this file and
    `explainability/explainer.py`, which is a larger change than a
    same-window feature addition and is left as a follow-up.

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
    reversal_5 = torch.mean(window_ret[-5:], dim=0) if window_ret[-5:].shape[0] > 0 else torch.zeros(num_assets)
    mom_accel = mom_10 - mom_60

    raw_features = torch.stack(
        [vol_20, mom_10, mom_60, skew_30, downside_vol, drawdown, reversal_5, mom_accel], dim=-1
    )  # [N, 8]

    # FIX (targets Sharpe/return quality directly, without touching
    # risk_aversion/top_k/turnover_weight -- the three levers that
    # previously caused regressions when tuned): standardize each feature
    # cross-sectionally (across the N assets, at this rebalance date) to
    # zero mean / unit variance. Raw feature levels shift with market-wide
    # regime (e.g. every asset's vol_20 rises together in a high-vol
    # regime), which swamps the cross-sectional signal that actually
    # determines relative over/underweighting -- this is standard
    # practice in cross-sectional equity models and is the most direct
    # lever available on return-signal quality that hasn't already been
    # tried and rolled back.
    cs_mean = raw_features.mean(dim=0, keepdim=True)
    cs_std = raw_features.std(dim=0, keepdim=True, correction=1) + 1e-6
    z_features = (raw_features - cs_mean) / cs_std

    return z_features.unsqueeze(0)


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
    risk_aversion: float = 0.5,   # FIX (reverted): an earlier patch changed this
                                  # default to 1.0 to match train_gaeco_net_ensemble,
                                  # but main.py's call site never passes risk_aversion
                                  # explicitly and was implicitly relying on 0.5 --
                                  # that "reconciliation" was a silent behavior change
                                  # (flatter softmax -> more uniform weights) and is
                                  # very likely a primary cause of the regression you
                                  # saw. Reverted to 0.5; if you want the two entry
                                  # points to genuinely share one value going forward,
                                  # pass risk_aversion explicitly at both call sites
                                  # instead of relying on the default.
    lambda_turnover: float = 0.01,
    return_weight: float = 5.0,  # FIX (targets return/Sharpe directly): was hardcoded
                                  # at 2.0 below. With the loss now numerically stable
                                  # (EMA downside variance) and features now
                                  # cross-sectionally z-scored, the return head has
                                  # real signal to learn from -- raising return_weight
                                  # gives it a stronger incentive to actually use it,
                                  # rather than defaulting toward the variance-
                                  # minimizing behavior visible in the equity curve
                                  # (low realized vol, but consistently trailing the
                                  # baselines' cumulative return). Unlike top_k/
                                  # risk_aversion/turnover_weight, this hasn't been
                                  # tried and tuned down before.
    top_k: int | None = None,    # FIX: threads through to DifferentiableMeanVariance;
                                  # None preserves old dense-softmax behavior, pass e.g.
                                  # top_k=15 (for a 30-asset universe) to enable
                                  # conviction-based concentration.
    lr: float = 1e-3,
    model: torch.nn.Module | None = None
):
    """
    Trains GAECO-Net using a multi-day forward cumulative return target (H = horizon).
    """
    num_assets = returns_df.shape[1]
    
    if model is None:
        model = GAECONetPipeline(num_assets=num_assets, in_features=8, risk_aversion=risk_aversion, top_k=top_k)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = NetSortinoLoss(fee_rate=0.0010, return_weight=return_weight, turnover_weight=lambda_turnover)

    returns_tensor = torch.tensor(returns_df.values, dtype=torch.float32)
    num_timesteps = len(returns_df)
    num_windows = len(empirical_data["corr_emp"])

    # Prevent boundary overflow when slicing target returns [t + lookback : t + lookback + horizon]
    last_valid_idx = min(num_timesteps - lookback - horizon, num_windows - 1)

    # -------------------------------------------------------------------------
    # Phase 1: Synthetic Circular Block Bootstrap (CBB) Pre-Training
    # -------------------------------------------------------------------------
    # FIX (see GAECO-Net paper, Limitations #2 "Unwired synthetic
    # pre-training phase"): `generate_cbb_synthetic_data` previously existed
    # as a standalone utility but was never called from this function, so
    # only Phase 2 (real-data fine-tuning) ever ran despite `synthetic_epochs`
    # being accepted as an argument. This block wires it in: a fresh CBB
    # resample is drawn each epoch (so the agent sees a different synthetic
    # path every pass rather than overfitting to one static resample), and
    # is trained on with the identical model/criterion/optimizer used for
    # Phase 2, so the two phases are directly comparable and gradients
    # accumulate into the same weights before real-data fine-tuning begins.
    if synthetic_epochs > 0:
        print(f"--- Phase 1: Synthetic CBB Pre-Training ({synthetic_epochs} epochs) ---")
        model.train()
        for epoch in range(synthetic_epochs):
            synth_returns, synth_features, synth_laplacians, synth_evals, synth_evecs = generate_cbb_synthetic_data(
                returns_tensor, block_length=block_length, target_length=252, lookback=lookback
            )

            num_synth_steps = len(synth_features)
            last_synth_idx = num_synth_steps - horizon
            if last_synth_idx <= 0:
                print(
                    "  Skipping synthetic pre-training: target_length too short "
                    f"relative to horizon ({num_synth_steps} steps, horizon={horizon})."
                )
                break

            total_loss = 0.0
            prev_weights = None

            for t in range(last_synth_idx):
                node_feat = synth_features[t]
                sample_lap = synth_laplacians[t]
                eigenvals = synth_evals[t]
                eigenvecs = synth_evecs[t]

                # Forward horizon target, drawn from the same synthetic path
                # (synth_returns[i] is already aligned to synth_features[i]'s
                # "next day", see generate_cbb_synthetic_data's docstring).
                forward_returns = synth_returns[t : t + horizon].sum(dim=0, keepdim=True)

                weights, C_clean, mu, gamma = model(node_feat, sample_lap, eigenvals, eigenvecs)

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
                avg_epoch_loss = total_loss / last_synth_idx
                print(f"  [Synthetic] Epoch [{epoch+1}/{synthetic_epochs}] - Avg Loss: {avg_epoch_loss:.6f}")

    # -------------------------------------------------------------------------
    # Phase 2: Real-Data Fine-Tuning
    # -------------------------------------------------------------------------
    if epochs > 0:
        print(f"--- Phase 2: Real-Data Fine-Tuning ({epochs} epochs) ---")
        # FIX: don't carry a downside-variance EMA calibrated to the
        # synthetic CBB panel's volatility scale into real-data fine-tuning
        # -- reset it so Phase 2 builds its own running estimate from real
        # returns instead.
        criterion.reset_downside_ema()
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
    risk_aversion: float = 0.5,
    lambda_turnover: float = 0.005,
    top_k: int | None = None,
    return_weight: float = 5.0,
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
            top_k=top_k,
            return_weight=return_weight,
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