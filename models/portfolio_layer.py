#adjusted to allow for None or batchsize to stop  linalg solve error

import torch
import torch.nn as nn

class DifferentiableMeanVariance(nn.Module):
    """
    FIX (see GAECO-Net paper, Limitations #5 / "Inactive top-k sparsity
    lever"): the previous version only exposed a single `k_assets`
    argument that was simultaneously (a) the universe size used by
    calling code and (b) the number of names kept by top-k selection.
    Every call site therefore passed `k_assets=num_assets`, which made
    `top_k_softmax` mathematically a no-op dense softmax over the full
    universe -- there was no way to ask for a genuinely sparse,
    conviction-ranked portfolio.

    This version separates the two concepts:
      - `num_assets`: the size of the investable universe (N).
      - `top_k`: how many of those N names to actually hold; defaults
        to `num_assets` (i.e. the old, non-sparse behavior) so existing
        callers are unaffected unless they explicitly opt in to
        concentration by passing `top_k < num_assets`.
    """
    def __init__(
        self,
        num_assets: int | None = None,
        risk_aversion: float = 1.0,
        ema_alpha: float = 0.25,
        top_k: int | None = None,
        k_assets: int | None = None,  # deprecated alias, see below
    ):
        super().__init__()
        # Backward compatibility: the old constructor argument was
        # `k_assets`, used as the universe size. If a caller still
        # passes it that way (and doesn't pass num_assets), honor it as
        # num_assets rather than silently breaking existing checkpoints
        # / call sites.
        if num_assets is None:
            if k_assets is None:
                raise ValueError("DifferentiableMeanVariance requires num_assets (or legacy k_assets).")
            num_assets = k_assets

        self.gamma_risk = risk_aversion
        self.num_assets = num_assets
        self.ema_alpha = ema_alpha
        # Default preserves old behavior (dense softmax over all assets)
        # unless the caller explicitly requests concentration.
        self.top_k = top_k if top_k is not None else num_assets
        if not (1 <= self.top_k <= num_assets):
            raise ValueError(f"top_k must satisfy 1 <= top_k <= num_assets ({num_assets}); got {self.top_k}.")

    def top_k_softmax(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Keeps the top `self.top_k` highest-conviction assets and sets all
        other assets' weights to 0. When `self.top_k == num_assets` this
        is exactly equivalent to a dense softmax over all assets (the
        previous, always-active behavior).
        """
        topk_vals, topk_indices = torch.topk(logits, k=self.top_k, dim=-1)
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