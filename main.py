import os
import torch
import pandas as pd
import numpy as np
import vectorbt as vbt
from pipeline.train import train_gaeco_net, _extract_node_features
from backtester.engine import run_vectorbt_backtest, run_covariance_benchmark_backtest
from data.loader import WRDSDataLoader
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from explainability.explainer import explain_allocation, plot_and_save_subgraph_attribution, generate_explained_allocations
from pipeline.stat_tests import run_academic_statistical_viability
from backtester.benchmarks.estimators import ledoit_wolf_shrinkage, marchenko_pastur_denoise, sample_covariance

load_dotenv()


def main():    
    print("=== Step 1: Loading CRSP Data ===")
    train_start_date = "2010-01-01"
    train_end_date = "2019-12-31"
    test_start_date = "2020-01-01"
    test_end_date = "2024-12-31"
    num_assets = 30
    lookback = 60

    # Calculate fetch start date to include lookback history
    fetch_start_date = str(
        (pd.to_datetime(train_start_date) - pd.Timedelta(days=lookback * 2)).date()
    )

    loader = WRDSDataLoader()
    
    returns_df = loader.fetch_crsp_returns(
        start_date=fetch_start_date,
        end_date=test_end_date,
        num_assets=num_assets,
        universe_asof_end_date=train_end_date
    )
    
    returns_df.index = pd.to_datetime(returns_df.index)
    returns_df.sort_index(inplace=True)

    print("\n=== Step 2: Computing Rolling Empirical Matrices ===")
    empirical_data = loader.compute_rolling_empirical(returns_df, lookback=lookback)
    
    dates = pd.to_datetime(empirical_data["dates"])
    empirical_data["dates"] = dates
    
    # 1. Filter Empirical Data by Target Date Windows
    train_mask = (dates >= pd.to_datetime(train_start_date)) & (dates <= pd.to_datetime(train_end_date))
    test_mask = (dates >= pd.to_datetime(test_start_date)) & (dates <= pd.to_datetime(test_end_date))
    
    train_empirical_data = {
        "corr_emp": empirical_data["corr_emp"][train_mask],
        "eigenvals": empirical_data["eigenvals"][train_mask],
        "eigenvecs": empirical_data["eigenvecs"][train_mask],
        "dates": dates[train_mask]
    }
    
    test_empirical_data = {
        "corr_emp": empirical_data["corr_emp"][test_mask],
        "eigenvals": empirical_data["eigenvals"][test_mask],
        "eigenvecs": empirical_data["eigenvecs"][test_mask],
        "dates": dates[test_mask]
    }
    
    # 2. Slice Returns Data aligned with Empirical Dates (Including Lookback Buffer)
    train_start_pos = returns_df.index.searchsorted(train_empirical_data["dates"].min())
    train_end_pos = returns_df.index.searchsorted(train_empirical_data["dates"].max(), side='right')
    train_returns_df = returns_df.iloc[max(0, train_start_pos - lookback) : train_end_pos]

    # For Test: Ensure returns_df includes exact lookback buffer before test_start_date
    test_start_pos = returns_df.index.searchsorted(test_empirical_data["dates"].min())
    test_end_pos = returns_df.index.searchsorted(test_empirical_data["dates"].max(), side='right')
    test_returns_df = returns_df.iloc[max(0, test_start_pos - lookback) : test_end_pos]

    print(f"Train dates loaded: {len(train_returns_df)} rows (includes lookback buffer)")
    print(f"Test dates loaded:  {len(test_returns_df)} rows (includes lookback buffer)")

    if test_returns_df.empty:
        raise ValueError("test_returns_df is empty! Check your date strings and filters.")

    print("\n=== Step 3: Training GAECO-Net Ensemble on TRAIN Data ===")
    n_agents = 5
    base_seed = 42
    agent_test_weights = []

    for agent_id in range(n_agents):
        seed = base_seed + agent_id
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f"\n--- Training Agent {agent_id + 1}/{n_agents} (Seed: {seed}) ---")

        # 1. Train agent on TRAIN data
        _, trained_agent = train_gaeco_net(
            returns_df=train_returns_df,
            empirical_data=train_empirical_data,
            epochs=10,
            synthetic_epochs=10,
            lookback=lookback,
            lambda_turnover=0.01
        )

        # 2. Generate out-of-sample test allocations
        test_weights_agent, _ = train_gaeco_net(
            returns_df=test_returns_df,
            empirical_data=test_empirical_data,
            epochs=0,             # Eval-only mode
            synthetic_epochs=0,
            lookback=lookback,
            model=trained_agent
        )
        agent_test_weights.append(test_weights_agent)

    print("\n=== Ensembling Out-of-Sample Agent Weights ===")
    ensemble_values = np.mean([df.values for df in agent_test_weights], axis=0)
    test_weights_df = pd.DataFrame(
        ensemble_values,
        index=agent_test_weights[0].index,
        columns=agent_test_weights[0].columns
    )
    # Renormalize weights (Long-only)
    test_weights_df = test_weights_df.div(test_weights_df.sum(axis=1), axis=0)

    print("\n=== Step 4: Running Out-of-Sample Backtest ===")
    
    # Align evaluation returns to the EXACT dates predicted by GAECO-Net
    eval_returns_df = returns_df.loc[test_weights_df.index]

    print(f"Running Backtest from {eval_returns_df.index.min().date()} to {eval_returns_df.index.max().date()} ({len(eval_returns_df)} trading days)")

    portfolio = run_vectorbt_backtest(
        weights_df=test_weights_df,
        returns_df=eval_returns_df,
        fee_rate=0.001,           # 10 bps fee
        rebalance_freq='2W-FRI',   # Bi-weekly schedule
        deadband_threshold=0.02   # 2% deadband threshold
    )

    portfolio_explained = run_vectorbt_backtest(
        weights_df=generate_explained_allocations(
            model=trained_agent,
            returns_df=eval_returns_df,
            empirical_data=test_empirical_data,
            lookback=lookback
        ),
        returns_df=eval_returns_df,
        fee_rate=0.001,           # 10 bps fee
        rebalance_freq='2W-FRI',   # Bi-weekly schedule
        deadband_threshold=0.02   # 2% deadband threshold
    )

    print("\n=== Step 4.5: Running Baseline Matrix Benchmarks ===")

    lw_portfolio = run_covariance_benchmark_backtest(
        returns_df=eval_returns_df,
        estimator_func=ledoit_wolf_shrinkage,
        lookback=lookback,
        fee_rate=0.001,
        rebalance_freq='2W-FRI',
        deadband_threshold=0.02
    )

    mp_portfolio = run_covariance_benchmark_backtest(
        returns_df=eval_returns_df,
        estimator_func=marchenko_pastur_denoise,
        lookback=lookback,
        fee_rate=0.001,
        rebalance_freq='2W-FRI',
        deadband_threshold=0.02
    )

    sv_portfolio = run_covariance_benchmark_backtest(
        returns_df=eval_returns_df,
        estimator_func=sample_covariance,
        lookback=lookback,
        fee_rate=0.001,
        rebalance_freq='2W-FRI',
        deadband_threshold=0.02
    )

    # Output stats to text file
    output_filename = "benchmark_results.txt"
    print(f"\nWriting benchmark portfolio stats to '{output_filename}'...")

    with open(output_filename, "w") as f:
        f.write("=====================================================\n")
        f.write("             GAECO-NET BENCHMARK RESULTS             \n")
        f.write("=====================================================\n\n")

        f.write("--- 1. GAECO-Net Ensemble Portfolio Stats ---\n")
        f.write(str(portfolio.stats()))
        f.write("\n\n" + "="*53 + "\n\n")

        f.write("--- 2. Ledoit-Wolf Shrinkage Stats ---\n")
        f.write(str(lw_portfolio.stats()))
        f.write("\n\n" + "="*53 + "\n\n")

        f.write("--- 3. Marchenko-Pastur Denoising Stats ---\n")
        f.write(str(mp_portfolio.stats()))
        f.write("\n\n" + "="*53 + "\n\n")

        f.write("--- 4. Sample Covariance Stats ---\n")
        f.write(str(sv_portfolio.stats()))
        f.write("\n\n" + "="*53 + "\n")

        f.write("--- 5. GAECO-Net Explained (Pruned Graph) Stats ---\n")
        f.write(str(portfolio_explained.stats()))
        f.write("\n\n" + "="*53 + "\n\n")

    print(f"Successfully saved all benchmark statistics to {output_filename}!")

    print(portfolio.stats())
    ensemble_stat_results = run_academic_statistical_viability(
        portfolio,
        eval_returns_df,
        benchmark_portfolios={
            "Ledoit-Wolf": lw_portfolio,
            "Marchenko-Pastur": mp_portfolio,
            "Sample-Covariance": sv_portfolio,
        },
        strategy_label="GAECO-Net (Ensemble)",
    )

    # Before we had only the full ensemble's returns were ever run through
    # the significance suite -- the Explained/pruned-subgraph portfolio's
    # 2.99 Sharpe / 114% return numbers were never checked against the same
    # benchmarks, so there was no way to tell whether that gap is real or
    # just a smaller, noisier sample (portfolio_explained's window is ~126
    # "days"/bi-weekly cycles vs. the ensemble's ~132 -- see the paper's
    # Period-field caveat -- and it trades far less often, both of which
    # inflate variance around any point estimate). Run it through the exact
    # same tests now.
    explained_stat_results = run_academic_statistical_viability(
        portfolio_explained,
        eval_returns_df,
        benchmark_portfolios={
            "Ledoit-Wolf": lw_portfolio,
            "Marchenko-Pastur": mp_portfolio,
            "Sample-Covariance": sv_portfolio,
        },
        strategy_label="GAECO-Net (Explained)",
    )

    # =========================================================================
    # Step 5: Post-Hoc Model Explainability Analysis
    # =========================================================================
    print("\n=== Step 5: Explainability Analysis on GAECO-Net Allocations ===")

    try:
        stress_idx = 50 if len(eval_returns_df) > 50 else (len(eval_returns_df) // 2)
        stress_date = str(eval_returns_df.index[stress_idx].date())

        def to_clean_tensor(data_slice):
            if torch.is_tensor(data_slice):
                return data_slice.detach().clone().float()
            return torch.tensor(data_slice, dtype=torch.float32)

        sample_corr  = to_clean_tensor(test_empirical_data["corr_emp"][stress_idx : stress_idx + 1])
        sample_evals = to_clean_tensor(test_empirical_data["eigenvals"][stress_idx : stress_idx + 1])
        sample_evecs = to_clean_tensor(test_empirical_data["eigenvecs"][stress_idx : stress_idx + 1])

        I = torch.eye(sample_corr.shape[-1], device=sample_corr.device).unsqueeze(0)
        sample_lap = I - sample_corr

        window_ret = torch.tensor(
            eval_returns_df.iloc[max(0, stress_idx - lookback) : stress_idx].values,
            dtype=torch.float32
        )
        curr_idx_val = torch.tensor(
            eval_returns_df.iloc[stress_idx - 1].values,
            dtype=torch.float32
        )
        
        sample_feat = _extract_node_features(window_ret, curr_idx_val, num_assets)

        sample_target = torch.tensor(
            test_weights_df.iloc[stress_idx : stress_idx + 1].values, 
            dtype=torch.float32
        )

        print(f"Running explainer optimization for stress date: {stress_date}...")

        edge_mask, feat_mask = explain_allocation(
            model=trained_agent,
            node_features=sample_feat,
            laplacian=sample_lap,
            eigenvals=sample_evals,
            eigenvecs=sample_evecs,
            target_weights=sample_target,
            epochs=200,
            lr=0.01
        )

        asset_names = list(eval_returns_df.columns)
        plot_and_save_subgraph_attribution(
            edge_mask=edge_mask,
            asset_names=asset_names,
            date_str=stress_date,
            top_k_edges=15,
            output_filename="gaeco_net_attribution_covid2020.png"
        )
        print("Explainability analysis finished successfully!")

    except Exception as e:
        import traceback
        print(f"Explainer warning: {e}")
        traceback.print_exc()

    # =========================================================================
    # Render and Save Cumulative Returns Chart
    # =========================================================================
    print("\n=== Generating Out-of-Sample Performance Plot ===")

    gaeco_returns = portfolio.returns().vbt.returns.cumulative()
    lw_returns = lw_portfolio.returns().vbt.returns.cumulative()
    mp_returns = mp_portfolio.returns().vbt.returns.cumulative()
    sv_returns = sv_portfolio.returns().vbt.returns.cumulative()

    plt.figure(figsize=(12, 6))
    plt.plot(gaeco_returns, label="GAECO-Net (Proposed)", color="#1f77b4", linewidth=2.5)
    plt.plot(lw_returns, label="Ledoit-Wolf Shrinkage", color="#ff7f0e", linestyle="--", alpha=0.8)
    plt.plot(mp_returns, label="Marchenko-Pastur Denoise", color="#2ca02c", linestyle="--", alpha=0.8)
    plt.plot(sv_returns, label="Sample Covariance", color="#d62728", linestyle=":", alpha=0.7)

    plt.title("Out-of-Sample Cumulative Returns Comparison (2020–2024)", fontsize=13, fontweight="bold")
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Cumulative Return (1.0 = 100%)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper left", fontsize=10)
    plt.tight_layout()

    chart_filename = "gaeco_net_equity_curve.png"
    plt.savefig(chart_filename, dpi=300)
    plt.close()

    print(f"Successfully generated and saved graph to '{chart_filename}'!")

if __name__ == "__main__":
    main()