import numpy as np
import torch
from sklearn.covariance import LedoitWolf

def sample_covariance(returns: np.ndarray) -> np.ndarray:
    """
    Standard Sample Covariance Matrix estimator.
    """
    return np.cov(returns, rowvar=False)

def ledoit_wolf_shrinkage(returns: np.ndarray) -> np.ndarray:
    """
    Analytical Ledoit-Wolf Linear Shrinkage Estimator.
    """
    lw = LedoitWolf()
    lw.fit(returns)
    return lw.covariance_

def marchenko_pastur_denoise(returns: np.ndarray, b_factor: float = 1.0) -> np.ndarray:
    """
    Random Matrix Theory (RMT) Marchenko-Pastur Eigenvalue Clipping.
    Replaces noisy eigenvalues below the theoretical threshold lambda_max with their average.
    """
    T, N = returns.shape
    q = T / N  # Aspect ratio T/N
    
    # 1. Standardize returns to compute correlation matrix
    std_devs = np.std(returns, axis=0, ddof=1)
    std_devs[std_devs == 0] = 1e-8
    corr = np.corrcoef(returns, rowvar=False)
    
    # 2. Eigenvalue Decomposition
    evals, evecs = np.linalg.eigh(corr)
    
    # 3. Marchenko-Pastur Theoretical Upper Bound
    sigma2 = 1.0  # Normalized variance
    lambda_max = sigma2 * (1 + (1 / q) + 2 * np.sqrt(1 / q)) * b_factor
    
    # 4. Filter Noise Eigenvalues
    noise_evals = evals[evals <= lambda_max]
    if len(noise_evals) > 0:
        avg_noise = np.mean(noise_evals)
        evals[evals <= lambda_max] = avg_noise

    # 5. Reconstruct Cleaned Correlation & Covariance Matrix
    corr_clean = evecs @ np.diag(evals) @ evecs.T
    
    # Guarantee unit diagonal for correlation
    d = np.sqrt(np.diag(corr_clean))
    corr_clean = corr_clean / np.outer(d, d)
    
    # Scale back to Covariance
    cov_clean = corr_clean * np.outer(std_devs, std_devs)
    return cov_clean