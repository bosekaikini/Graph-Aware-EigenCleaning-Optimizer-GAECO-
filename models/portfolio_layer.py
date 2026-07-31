#adjusted to allow for None or batchsize to stop  linalg solve error

import torch
import torch.nn as nn

class DifferentiableMeanVariance(nn.Module):
    def __init__(self, k_assets: int = 10, risk_aversion: float = 1.0, ema_alpha: float = 0.25):
        super().__init__()
        self.gamma_risk = risk_aversion
        self.k_assets = k_assets
        self.ema_alpha = ema_alpha

    def top_k_softmax(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Keeps top K assets with highest weight proposals and sets non-conviction assets to 0.
        """
        topk_vals, topk_indices = torch.topk(logits, k=self.k_assets, dim=-1)
        topk_weights = torch.softmax(topk_vals, dim=-1)

        zeros = torch.zeros_like(logits)
        sparse_weights = zeros.scatter(-1, topk_indices, topk_weights)
        return sparse_weights

    def forward(self, mu: torch.Tensor, cov_clean: torch.Tensor) -> torch.Tensor:
        # Swap inputs if passed as (cov_clean, mu) instead of (mu, cov_clean)
        if mu.dim() >= cov_clean.dim() and mu.shape[-1] == mu.shape[-2]:
            mu, cov_clean = cov_clean, mu

        # Ensure mu is shaped as a column vector for linalg.solve: [..., N, 1]
        mu_col = mu.unsqueeze(-1) if mu.shape[-1] != 1 else mu

        # Solve unconstrained weights: w_unconstrained = (1 / gamma) * Sigma_clean^(-1) * mu
        w_unconstrained = torch.linalg.solve(cov_clean, mu_col).squeeze(-1) / self.gamma_risk

        # Apply Top-K Sparsity instead of standard full Softmax
        w_sparse = self.top_k_softmax(w_unconstrained)

        # Temporal EMA smoothing over time steps to prevent allocation flutter
        if w_sparse.dim() == 3 and w_sparse.size(1) > 1:
            w_smooth = torch.zeros_like(w_sparse)
            w_smooth[:, 0, :] = w_sparse[:, 0, :]
            for t in range(1, w_sparse.size(1)):
                w_smooth[:, t, :] = (self.ema_alpha * w_sparse[:, t, :]) + ((1 - self.ema_alpha) * w_smooth[:, t-1, :])
            return w_smooth
        elif w_sparse.dim() == 2 and w_sparse.size(0) > 1:
            w_smooth = torch.zeros_like(w_sparse)
            w_smooth[0, :] = w_sparse[0, :]
            for t in range(1, w_sparse.size(0)):
                w_smooth[t, :] = (self.ema_alpha * w_sparse[t, :]) + ((1 - self.ema_alpha) * w_smooth[t-1, :])
            return w_smooth
        else:
            return w_sparse