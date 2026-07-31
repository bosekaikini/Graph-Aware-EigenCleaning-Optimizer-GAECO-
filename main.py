import os
import torch
import vectorbt as vbt
from pipeline.train import train_gaeco_net
from backtester.engine import run_vectorbt_backtest, process_execution_weights 
from data.loader import WRDSDataLoader
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()


def main():    
    print("=== Step 1: Loading CRSP Data ===")
    train_start_date = "2010-01-01"
    train_end_date = "2019-12-31"
    test_start_date = "2020-01-01"
    test_end_date = "2023-12-31"
    num_assets = 30
    lookback = 60
    epochs = 10
    save_path = "gaeco_net_model.pt"

    loader = WRDSDataLoader()
    returns_df = loader.fetch_crsp_returns(start_date=train_start_date, end_date=test_end_date, num_assets=num_assets)
    
    print("\n=== Step 2: Computing Rolling Empirical Matrices ===")
    empirical_data = loader.compute_rolling_empirical(returns_df, lookback=lookback)
    
    # --- FIX 1: Correctly slice PyTorch Tensors by date mask ---
    dates = empirical_data["dates"]
    
    train_mask = (dates >= train_start_date) & (dates <= train_end_date)
    test_mask = (dates >= test_start_date) & (dates <= test_end_date)
    
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
    
    # Align returns DataFrames to exact lookback dates
    train_returns_df = returns_df.loc[train_empirical_data["dates"]]
    test_returns_df = returns_df.loc[test_empirical_data["dates"]]

    print("\n=== Step 3: Training GAECO-Net ===")
    weights_df, model = train_gaeco_net(
        returns_df=train_returns_df,
        empirical_data=train_empirical_data,
        epochs=epochs,
        lr=1e-3,
        lookback=lookback,
        risk_aversion=1.0,
        lambda_turnover=0.005
    )

    torch.save(model.state_dict(), save_path)

    print("\n=== Step 4: Running Out-of-Sample Backtest ===")
    model.eval()

    with torch.no_grad():
        test_weights_df, _ = train_gaeco_net(
            returns_df=test_returns_df,
            empirical_data=test_empirical_data,
            epochs=0,  # Pure inference pass
            lr=1e-3,
            lookback=lookback,
            risk_aversion=1.0,
            lambda_turnover=0.005
        )

    processed_test_weights = process_execution_weights(
        test_weights_df,
        rebalance_freq='2W-FRI',
        deadband_threshold=0.02
    )

    aligned_returns = test_returns_df.loc[processed_test_weights.index]
    price_df = (1.0 + aligned_returns).cumprod()

    portfolio = vbt.Portfolio.from_orders(
        close=price_df,
        size=processed_test_weights, # Passed deadband-filtered bi-weekly weights
        size_type='targetpercent',
        fees=0.001,  # 10 bps fee
        freq='1D',   # Daily price updates with bi-weekly weight changes
        cash_sharing=True
    )

    # Display portfolio stats and equity curve
    print(portfolio.stats())
    portfolio.value().vbt.plot()
    plt.show()

if __name__ == "__main__":
    main()