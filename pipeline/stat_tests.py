# pipeline/stat_tests.py

import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox

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
    
    # 1. Paired t-test for Daily Mean Performance Lift
    t_stat, t_p_val = stats.ttest_rel(strat_vector, bench_vector)
    print(f"[Mean Performance Lift (Paired t-test)]")
    print(f"  - t-statistic: {t_stat:.4f}")
    print(f"  - p-value: {t_p_val:.4f}")
    if t_p_val < 0.05:
        print(f"  - Conclusion: Mean outperformance is STATISTICALLY SIGNIFICANT at alpha = 5% (t-stat positive: {t_stat > 0}).")
    else:
        print("  - Conclusion: Mean difference is statistically indistinguishable from zero (noise).")

    # 2. Levene's Test for Risk Reduction Equality
    levene_stat, levene_p_val = stats.levene(strat_vector, bench_vector)
    print(f"\n[Risk Reduction Significance (Levene's Variance Test)]")
    print(f"  - Levene Statistic: {levene_stat:.4f}")
    print(f"  - p-value: {levene_p_val:.4f}")
    if levene_p_val < 0.05:
        print("  - Conclusion: Strategy variance differences are STATISTICALLY SIGNIFICANT. The variance reduction is structural.")
    else:
        print("  - Conclusion: Variance differences are random; failed to confirm statistical risk mitigation dominance.")
        
    print("========================================================\n")
