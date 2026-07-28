import pandas as pd
import numpy as np
import torch
import wrds

class WRDSDataLoader:
    #takes daily s&p
    def __init__(self, wrds_username: str = None):
        self.conn = wrds.Connection(wrds_username=wrds_username)

    def fetch_crsp_returns(
        self, 
        start_date: str, 
        end_date: str, 
        num_assets: int = 100
    ) -> pd.DataFrame:
        
        query = f"""
            WITH top_stocks AS (
                SELECT permno, AVG(ABS(prc) * shrout) as avg_mktcap
                FROM crsp.dsf
                WHERE date BETWEEN '{start_date}' AND '{end_date}'
                  AND prc IS NOT NULL AND ret IS NOT NULL
                GROUP BY permno
                ORDER BY avg_mktcap DESC
                LIMIT {num_assets}
            )
            SELECT d.date, d.permno, d.ret
            FROM crsp.dsf d
            INNER JOIN top_stocks t ON d.permno = t.permno
            WHERE d.date BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY d.date, d.permno;
        """
        raw_df = self.conn.raw_sql(query, date_cols=['date'])
        
        # Pivot into a wide time-series panel [Dates x Assets]
        returns_df = raw_df.pivot(index='date', columns='permno', values='ret')
        
        # Clean missing values: Forward-fill short gaps, drop remaining NaNs
        returns_df = returns_df.ffill().fillna(0.0)
        return returns_df

    @staticmethod
    def compute_rolling_empirical(
        returns_df: pd.DataFrame, 
        lookback: int = 60
    ) -> dict:
        """
        Computes empirical rolling correlation matrices and eigendecompositions.
        
        Returns a dictionary containing PyTorch Tensors for the batch:
          - corr_emp: [Batch, N, N]
          - eigenvals: [Batch, N]
          - eigenvecs: [Batch, N, N]
        """
        num_timesteps, num_assets = returns_df.shape
        corrs, eigenvals_list, eigenvecs_list = [], [], []

        returns_matrix = returns_df.to_numpy()

        for t in range(lookback, num_timesteps):
            window_returns = returns_matrix[t - lookback : t]
            
            # Compute Empirical Correlation Matrix
            cov = np.cov(window_returns, rowvar=False)
            std = np.sqrt(np.diag(cov))
            std_outer = np.outer(std, std)
            # Avoid division by zero
            std_outer[std_outer == 0] = 1e-8
            corr = cov / std_outer
            np.fill_diagonal(corr, 1.0)

            # Eigendecomposition: C_emp = U * Lambda * U^T
            # eigh guarantees sorted eigenvalues for real symmetric matrices
            eigenvals, eigenvecs = np.linalg.eigh(corr)

            corrs.append(corr)
            eigenvals_list.append(eigenvals)
            eigenvecs_list.append(eigenvecs)

        return {
            "corr_emp": torch.tensor(np.array(corrs), dtype=torch.float32),
            "eigenvals": torch.tensor(np.array(eigenvals_list), dtype=torch.float32),
            "eigenvecs": torch.tensor(np.array(eigenvecs_list), dtype=torch.float32),
            "dates": returns_df.index[lookback:]
        }