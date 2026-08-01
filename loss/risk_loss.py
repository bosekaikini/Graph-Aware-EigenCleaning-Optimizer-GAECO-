# loss/risk_loss.py

import torch
import torch.nn as nn

class NetSharpeLoss(nn.Module):
    """
    Differentiable Net Sortino / Downside Risk Loss function.
    Penalizes turnover via real transaction fee rates and optimizes 
    downside semi-variance to improve directional return signals (p-value).
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
            turnover = torch.sum(torch.abs(weights - prev_weights), dim=-1) #[cite: 1]
        else:
            turnover = torch.zeros(weights.size(0), device=weights.device) #[cite: 1]

        # 2. Compute gross portfolio returns[cite: 1]
        gross_returns = torch.sum(weights * next_returns, dim=-1) #[cite: 1]

        # 3. Compute net returns after transaction costs[cite: 1]
        net_returns = gross_returns - (self.fee_rate * turnover) #[cite: 1]

        # 4. Compute Downside Semi-Variance (Sortino Risk Adjustment)
        # Isolate negative asset returns to penalize drawdowns specifically
        negative_returns = torch.clamp(next_returns, max=0.0)
        
        # Portfolio downside return component: sum(w_i * min(0, r_i))
        port_downside_returns = torch.sum(weights * negative_returns, dim=-1)
        
        # Estimate expected portfolio volatility using realized_cov for stability[cite: 1]
        w = weights.unsqueeze(-2)       # [B, 1, N][cite: 1]
        w_t = weights.unsqueeze(-1)     # [B, N, 1][cite: 1]
        port_var = torch.matmul(torch.matmul(w, realized_cov), w_t).squeeze(-1).squeeze(-1) #[cite: 1]
        port_std = torch.sqrt(torch.clamp(port_var, min=self.eps)) #[cite: 1]
        
        # Combine realized covariance with downside penalty
        downside_std = port_std + torch.abs(port_downside_returns)

        # 5. Compute Net Downside Sharpe / Sortino Ratio
        net_sortino = net_returns / (downside_std + self.eps)

        # Return negative ratio for gradient minimization
        return -torch.mean(net_sortino)