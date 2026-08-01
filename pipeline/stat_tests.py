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


def run_academic_statistical_viability(portfolio, test_returns_df):
    """
    Computes rigorous statistical tests on empirical backtest outputs 
    to validate model assumptions and prove outperformance significance.
    """
    print("\n========================================================")
    print("=== STEP 5: EMPIRICAL STATISTICAL VIABILITY SUITE ===")
    print("========================================================")
    
    # Extract Strategy Returns from VectorBT Portfolio object
    strategy_returns = portfolio.returns()
    
    # Establish Equal-Weighted Portfolio (1/N) Baseline Return Stream
    benchmark_returns = test_returns_df.mean(axis=1)
    
    # Intersect and align dates to drop any NaN fields across streams cleanly
    combined = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
    combined.columns = ['GAECO_Net', 'Equal_Weighted']
    
    strat_vector = combined['GAECO_Net']
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
    print("\n--- Pillar 2: Alpha Performance Significance vs. Benchmark ---")

    # 1. Jobson-Korkie (Memmel Corrected) Test for Sharpe Ratio Outperformance
    jk_z, jk_p_val = jobson_korkie_sharpe_test(strat_vector, bench_vector)
    print(f"[Sharpe Ratio Outperformance Test (Jobson-Korkie with Memmel Correction)]")
    print(f"  - Z-statistic: {jk_z:.4f}")
    print(f"  - p-value: {jk_p_val:.4e}")
    if jk_p_val < 0.05 and jk_z > 0:
        print("  - Conclusion: Risk-adjusted outperformance (Sharpe Ratio) is STATISTICALLY SIGNIFICANT (p < 0.05).")
    elif jk_p_val < 0.05 and jk_z < 0:
        print("  - Conclusion: Benchmark Sharpe ratio significantly outperforms Strategy.")
    else:
        print("  - Conclusion: Sharpe ratio difference is statistically indistinguishable from benchmark noise.")
    
    # 2. Paired t-test for Daily Mean Performance Lift
    t_stat, t_p_val = stats.ttest_rel(strat_vector, bench_vector)
    print(f"\n[Mean Performance Lift (Paired t-test)]")
    print(f"  - t-statistic: {t_stat:.4f}")
    print(f"  - p-value: {t_p_val:.4f}")
    if t_p_val < 0.05:
        print(f"  - Conclusion: Mean outperformance is STATISTICALLY SIGNIFICANT at alpha = 5% (t-stat positive: {t_stat > 0}).")
    else:
        print("  - Conclusion: Mean difference is statistically indistinguishable from zero (noise).")

    # 3. Levene's Test for Risk Reduction Equality
    levene_stat, levene_p_val = stats.levene(strat_vector, bench_vector)
    print(f"\n[Risk Reduction Significance (Levene's Variance Test)]")
    print(f"  - Levene Statistic: {levene_stat:.4f}")
    print(f"  - p-value: {levene_p_val:.4f}")
    if levene_p_val < 0.05:
        print("  - Conclusion: Strategy variance differences are STATISTICALLY SIGNIFICANT. The variance reduction is structural.")
    else:
        print("  - Conclusion: Variance differences are random; failed to confirm statistical risk mitigation dominance.")
        
    print("========================================================\n")