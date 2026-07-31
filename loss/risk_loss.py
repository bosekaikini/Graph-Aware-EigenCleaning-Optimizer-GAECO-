# loss/risk_loss.py

import torch
import torch.nn as nn

class NetSharpeLoss(nn.Module):
    """
    Differentiable loss function penalizing turnover using real transaction fee rates.
    Designed for single-step rolling updates during model training.
    """
    def __init__(self, fee_rate: float = 0.0010, eps: float = 1e-6):
        super().__init__()
        self.fee_rate = fee_rate
        self.eps = eps

    def forward(
        self, 
        weights: torch.Tensor, 
        next_returns: torch.Tensor, 
        realized_cov: torch.Tensor, 
        prev_weights: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        weights: [1, Assets] or [Batch, Assets]
        next_returns: [1, Assets] or [Batch, Assets]
        realized_cov: [1, Assets, Assets] or [Batch, Assets, Assets]
        prev_weights: [1, Assets] or None
        """
        # Ensure 2D tensor for vectors: [Batch, Assets]
        if weights.dim() == 1:
            weights = weights.unsqueeze(0)
        if next_returns.dim() == 1:
            next_returns = next_returns.unsqueeze(0)
        if realized_cov.dim() == 2:
            realized_cov = realized_cov.unsqueeze(0)

        # 1. Compute turnover against previous portfolio weights
        if prev_weights is not None:
            if prev_weights.dim() == 1:
                prev_weights = prev_weights.unsqueeze(0)
            turnover = torch.sum(torch.abs(weights - prev_weights), dim=-1)
        else:
            turnover = torch.zeros(weights.size(0), device=weights.device)

        # 2. Compute gross portfolio returns
        gross_returns = torch.sum(weights * next_returns, dim=-1)

        # 3. Compute net returns after transaction costs
        net_returns = gross_returns - (self.fee_rate * turnover)

        # 4. Compute expected portfolio variance: w^T * Sigma * w
        w = weights.unsqueeze(-2)       # [B, 1, N]
        w_t = weights.unsqueeze(-1)     # [B, N, 1]
        
        port_var = torch.matmul(torch.matmul(w, realized_cov), w_t).squeeze(-1).squeeze(-1)
        port_std = torch.sqrt(torch.clamp(port_var, min=self.eps))

        # 5. Compute Net Sharpe Ratio
        net_sharpe = net_returns / (port_std + self.eps)

        # Return negative Net Sharpe for gradient minimization
        return -torch.mean(net_sharpe)