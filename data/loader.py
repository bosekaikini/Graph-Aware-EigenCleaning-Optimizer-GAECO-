import os
import hashlib
import pandas as pd
import numpy as np
import torch
import wrds
from dotenv import load_dotenv


class WRDSDataLoader:
    # takes daily s&p
    def __init__(self, wrds_username: str | None = None, wrds_password: str | None = None):
        load_dotenv()
        self.username = wrds_username or os.environ.get("WRDS_USERNAME")
        self.password = wrds_password or os.environ.get("WRDS_PASSWORD")
        self.conn = wrds.Connection(wrds_username=self.username)

    def fetch_crsp_returns(
        self,
        start_date: str,
        end_date: str,
        num_assets: int = 100,
        universe_asof_end_date: str | None = None,
        cache_dir: str | None = ".cache/crsp",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Pulls a wide [Dates x Assets] daily-return panel for a fixed universe
        of `num_assets` names from CRSP.

        universe_asof_end_date:
            The universe (which `num_assets` permnos are selected) is chosen
            by average market cap computed ONLY over `start_date` ..
            `universe_asof_end_date`. This must be <= your training/OOS split
            date -- selecting the universe using average market cap over a
            window that includes your out-of-sample period is a lookahead/
            survivorship leak (you're implicitly using knowledge of which
            names survived and stayed large through the test period).

            If not provided, this defaults to `end_date` and a loud warning
            is printed, since that reproduces the old (leaky) behavior. Pass
            this explicitly -- e.g. your train_end_date -- to fix the leak.

        cache_dir / use_cache:
            CRSP data is periodically revised (delisting returns, restated
            prc/shrout, etc.), and the underlying SQL has no guaranteed row
            order on ties. To keep repeated runs of the same experiment
            reproducible, the first successful pull for a given
            (universe_asof_end_date, start_date, end_date, num_assets) is
            cached to disk as parquet; subsequent calls with the same
            arguments load from that cache instead of re-querying WRDS. Set
            use_cache=False to force a fresh pull (e.g. when you deliberately
            want to refresh the cache).
        """
        if universe_asof_end_date is None:
            universe_asof_end_date = end_date
            print(
                "WARNING: fetch_crsp_returns() called without "
                "universe_asof_end_date -- selecting the universe using "
                f"market cap through {end_date} (i.e. the full requested "
                "range). If end_date extends past your OOS split, this is a "
                "lookahead/survivorship leak in universe construction. Pass "
                "universe_asof_end_date=<your train_end_date> to fix this."
            )

        cache_path = None
        if cache_dir is not None:
            cache_key = "_".join(
                [
                    "crsp",
                    str(start_date),
                    str(end_date),
                    f"n{num_assets}",
                    f"asof{universe_asof_end_date}",
                ]
            )
            cache_key = hashlib.sha256(cache_key.encode()).hexdigest()[:16] + "_" + cache_key
            cache_path = os.path.join(cache_dir, f"{cache_key}.parquet")

            if use_cache and os.path.exists(cache_path):
                print(f"Loading cached CRSP pull from '{cache_path}' (skipping WRDS query for reproducibility).")
                returns_df = pd.read_parquet(cache_path)
                returns_df.index = pd.to_datetime(returns_df.index)
                return returns_df

        query = f"""
            WITH top_stocks AS (
                SELECT permno, AVG(ABS(prc) * shrout) as avg_mktcap
                FROM crsp.dsf
                WHERE date BETWEEN '{start_date}' AND '{universe_asof_end_date}'
                  AND prc IS NOT NULL AND ret IS NOT NULL
                GROUP BY permno
                ORDER BY avg_mktcap DESC, permno ASC
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
        returns_df = returns_df.ffill().fillna(0.0).astype(np.float64)

        if cache_path is not None:
            cache_path_dir = os.path.dirname(cache_path)
            if cache_path_dir:
                os.makedirs(cache_path_dir, exist_ok=True)
            # Parquet requires string column names; permno columns are ints.
            returns_df.columns = returns_df.columns.astype(str)
            returns_df.to_parquet(cache_path)
            print(f"Cached CRSP pull to '{cache_path}'.")

        return returns_df

    @staticmethod
    def compute_rolling_empirical(
        returns_df: pd.DataFrame,
        lookback: int = 60
    ) -> dict:
        """
        Computes empirical rolling correlation matrices and eigendecompositions.

        For each output index i, the correlation/eigendecomposition is
        computed strictly from returns_matrix[i : i+lookback] and is labeled
        with date returns_df.index[i+lookback] -- i.e. it represents
        information available strictly BEFORE that date's own return, and is
        safe to use for a trading decision made on that date. (The previous
        version labeled this same window with returns_df.index[i+lookback-1],
        which included that date's own return in what was supposed to be a
        pre-decision snapshot -- a one-day lookahead leak.)

        Returns a dictionary containing PyTorch Tensors for the batch:
          - corr_emp: [Batch, N, N]
          - eigenvals: [Batch, N]
          - eigenvecs: [Batch, N, N]
          - dates: [Batch] -- aligned 1:1 with the tensors above
        """
        returns_matrix = returns_df.to_numpy(dtype=np.float64)
        returns_matrix = np.nan_to_num(returns_matrix, nan=0.0)

        num_timesteps, num_assets = returns_matrix.shape
        corrs, eigenvals_list, eigenvecs_list = [], [], []

        for t in range(lookback, num_timesteps):
            window_returns = returns_matrix[t - lookback: t]

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
            # FIX: was returns_df.index[lookback-1:-1] (one-day-early / leaky label)
            "dates": returns_df.index[lookback:]
        }