# pipeline/stat_tests.py

import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox


def jobson_korkie_sharpe_test(strat_rets: pd.Series, bench_rets: pd.Series) -> tuple[float, float]:
    """
    Computes the Jobson-Korkie (1981) test with Memmel (2002) correction 
    for the statistical significance of the difference between two Sharpe ratios.
    
    H0: Sharpe(Strategy) == Sharpe(Benchmark)
    H1: Sharpe(Strategy) != Sharpe(Benchmark)
    """
    n = len(strat_rets)
    if n < 2:
        return 0.0, 1.0

    # Sample means and variances
    mu_a = strat_rets.mean()
    mu_b = bench_rets.mean()
    var_a = strat_rets.var(ddof=1)
    var_b = bench_rets.var(ddof=1)
    sd_a = np.sqrt(var_a)
    sd_b = np.sqrt(var_b)

    # Sample correlation and Sharpe ratios
    rho = strat_rets.corr(bench_rets)
    sharpe_a = mu_a / sd_a if sd_a > 0 else 0.0
    sharpe_b = mu_b / sd_b if sd_b > 0 else 0.0

    # Memmel (2002) asymptotic variance formula of Sharpe ratio difference
    var_diff = (1 / n) * (
        2 
        - 2 * rho 
        + 0.5 * (sharpe_a**2 + sharpe_b**2 - 2 * sharpe_a * sharpe_b * (rho**2))
    )

    if var_diff <= 0 or np.isnan(var_diff):
        return 0.0, 1.0

    z_stat = (sharpe_a - sharpe_b) / np.sqrt(var_diff)
    # Two-tailed p-value calculation
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z_stat)))

    # Explicitly cast to Python floats to satisfy type checkers (Pylance / MyPy)
    return float(z_stat), float(p_value)


def _run_pillar2_vs_one_benchmark(strat_vector: pd.Series, bench_vector: pd.Series, bench_name: str, strat_label: str) -> dict:
    """
    Runs the three Pillar-2 significance tests (Jobson-Korkie Sharpe test,
    paired t-test, Levene's variance test) for the strategy against a
    single named benchmark return series, printing a labeled report and
    returning the raw statistics/p-values for programmatic use.
    """
    combined = pd.concat([strat_vector, bench_vector], axis=1).dropna()
    combined.columns = [strat_label, bench_name]
    s = combined[strat_label]
    b = combined[bench_name]

    print(f"\n>>> {strat_label} vs. {bench_name} (n={len(combined)} aligned obs.) <<<")

    jk_z, jk_p_val = jobson_korkie_sharpe_test(s, b)
    print(f"[Sharpe Ratio Outperformance Test (Jobson-Korkie, Memmel-corrected)]")
    print(f"  - Z-statistic: {jk_z:.4f}")
    print(f"  - p-value: {jk_p_val:.4e}")
    if jk_p_val < 0.05 and jk_z > 0:
        print(f"  - Conclusion: Sharpe outperformance vs {bench_name} is STATISTICALLY SIGNIFICANT (p < 0.05).")
    elif jk_p_val < 0.05 and jk_z < 0:
        print(f"  - Conclusion: {bench_name}'s Sharpe ratio significantly beats {strat_label}.")
    else:
        print(f"  - Conclusion: Sharpe difference vs {bench_name} is statistically indistinguishable from noise.")

    t_stat, t_p_val = stats.ttest_rel(s, b)
    print(f"[Mean Performance Lift (Paired t-test)]")
    print(f"  - t-statistic: {t_stat:.4f}")
    print(f"  - p-value: {t_p_val:.4f}")
    if t_p_val < 0.05:
        print(f"  - Conclusion: Mean outperformance vs {bench_name} is STATISTICALLY SIGNIFICANT at alpha=5% (t-stat positive: {t_stat > 0}).")
    else:
        print(f"  - Conclusion: Mean difference vs {bench_name} is statistically indistinguishable from zero (noise).")

    levene_stat, levene_p_val = stats.levene(s, b)
    print(f"[Risk Reduction Significance (Levene's Variance Test)]")
    print(f"  - Levene Statistic: {levene_stat:.4f}")
    print(f"  - p-value: {levene_p_val:.4e}")
    if levene_p_val < 0.05:
        print(f"  - Conclusion: Variance difference vs {bench_name} is STATISTICALLY SIGNIFICANT (structural risk reduction).")
    else:
        print(f"  - Conclusion: Variance difference vs {bench_name} is not statistically distinguishable from noise.")

    return {
        "n_obs": len(combined),
        "jobson_korkie_z": jk_z, "jobson_korkie_p": jk_p_val,
        "paired_t_stat": t_stat, "paired_t_p": t_p_val,
        "levene_stat": levene_stat, "levene_p": levene_p_val,
    }


def run_academic_statistical_viability(
    portfolio,
    test_returns_df,
    benchmark_portfolios: dict | None = None,
    strategy_label: str = "GAECO_Net",
):
    """
    Computes rigorous statistical tests on empirical backtest outputs
    to validate model assumptions and quantify outperformance significance.

    FIX (previously): Pillar 2 only ever compared the strategy against a
    raw Equal-Weighted (1/N) portfolio built directly from
    `test_returns_df`. That is a materially different, and in a strong
    bull-market window like 2020-2024 often *harder*, benchmark than the
    Ledoit-Wolf / Marchenko-Pastur / Sample-Covariance baselines the
    headline results table (and this paper's actual claim) is measured
    against -- so a strategy could show no significant edge vs 1/N while
    still clearly, and significantly, beating the covariance-estimation
    baselines it is meant to be compared to. "Improving the statistics"
    without also fixing the underlying model means picking the right
    comparison, not just a better one: this version tests against
    Equal-Weight AND every baseline portfolio you actually have.

    strategy_label: identifies which GAECO-Net variant is being tested in
    the printed report and the returned dict (e.g. "GAECO-Net (Ensemble)"
    vs "GAECO-Net (Explained)"), so you can call this function once per
    variant and tell the two reports apart -- rather than only ever
    testing the full ensemble and eyeballing the explained variant's
    Sharpe from the raw vectorbt stats.

    benchmark_portfolios: optional dict of {name: vbt.Portfolio}, e.g.
        {"Ledoit-Wolf": lw_portfolio, "Marchenko-Pastur": mp_portfolio,
         "Sample-Covariance": sv_portfolio}
    from main.py. If omitted, only the Equal-Weighted (1/N) comparison is
    run (old behavior). Returns a dict of results per benchmark for
    programmatic inspection instead of only printing.
    """
    print("\n========================================================")
    print(f"=== STEP 5: EMPIRICAL STATISTICAL VIABILITY SUITE -- {strategy_label} ===")
    print("========================================================")
    
    # Extract Strategy Returns from VectorBT Portfolio object
    strategy_returns = portfolio.returns()
    
    # Establish Equal-Weighted Portfolio (1/N) Baseline Return Stream
    benchmark_returns = test_returns_df.mean(axis=1)
    
    # Intersect and align dates to drop any NaN fields across streams cleanly
    combined = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
    combined.columns = [strategy_label, 'Equal_Weighted']
    
    strat_vector = combined[strategy_label]
    bench_vector = combined['Equal_Weighted']

    # --- PILLAR 1: MODEL ASSUMPTION & STYLIZED FACTS TESTING ---
    print("\n--- Pillar 1: Return Distribution & Time Series Foundations ---")
    
    # 1. Jarque-Bera Normality Test
    jb_stat, jb_p_val = stats.jarque_bera(strat_vector)
    print(f"[Normality Test (Jarque-Bera)]")
    print(f"  - JB Statistic: {jb_stat:.4f}")
    print(f"  - p-value: {jb_p_val:.4e}")
    if jb_p_val < 0.05:
        print("  - Conclusion: Rejects normality (p < 0.05). Heavy-tails/skewness exist, justifying nonlinear GNN filters.")
    else:
        print("  - Conclusion: Fails to reject normality. Asset returns follow standard Gaussian parameters.")

    # 2. Augmented Dickey-Fuller Stationarity Test
    adf_result = adfuller(strat_vector)
    print(f"\n[Stationarity Test (ADF)]")
    print(f"  - ADF Statistic: {adf_result[0]:.4f}")
    print(f"  - p-value: {adf_result[1]:.4e}")
    if adf_result[1] < 0.05:
        print("  - Conclusion: Stationary sequence (p < 0.05). Time series assumptions are valid for mathematical forecasting.")
    else:
        print("  - Conclusion: NON-STATIONARY (p >= 0.05). WARNING: Model features run the risk of spurious structural breaks.")

    # 3. Ljung-Box Test on Squared Returns (Testing for Volatility Clustering / Heteroskedasticity)
    lb_df = acorr_ljungbox(strat_vector**2, lags=[5, 10], return_df=True)
    print(f"\n[ARCH Effects / Volatility Clustering (Ljung-Box on Squared Returns)]")
    for lag in lb_df.index:
        p_val = lb_df.loc[lag, 'lb_pvalue']
        print(f"  - Lag {lag} p-value: {p_val:.4e}")
    if (lb_df['lb_pvalue'] < 0.05).any():
        print("  - Conclusion: Significant volatility clustering detected (p < 0.05). Structural ARCH features are confirmed.")
    else:
        print("  - Conclusion: No significant volatility clustering detected.")

    # --- PILLAR 2: EMPIRICAL PERFORMANCE SIGNIFICANCE TESTING ---
    print("\n--- Pillar 2: Performance Significance vs. Every Available Benchmark ---")

    results = {
        "Equal_Weighted": _run_pillar2_vs_one_benchmark(strat_vector, bench_vector, "Equal-Weighted (1/N)", strategy_label)
    }

    if benchmark_portfolios:
        for name, bench_portfolio in benchmark_portfolios.items():
            bench_ret = bench_portfolio.returns()
            results[name] = _run_pillar2_vs_one_benchmark(strat_vector, bench_ret, name, strategy_label)
    else:
        print(
            "\n(No additional benchmark_portfolios were passed in -- only tested "
            "against raw Equal-Weight. Pass your LW/MP/SV vbt.Portfolio objects "
            "in to also test against the baselines your headline table actually "
            "reports against; see main.py wiring below.)"
        )

    print("========================================================\n")
    return results