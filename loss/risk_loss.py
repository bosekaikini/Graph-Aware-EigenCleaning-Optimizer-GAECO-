import torch
import torch.nn as nn

class NetSortinoLoss(nn.Module):
    def __init__(self, fee_rate: float = 0.001, return_weight: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.fee_rate = fee_rate
        self.return_weight = return_weight
        self.eps = eps

    def forward(self, weights: torch.Tensor, asset_returns: torch.Tensor) -> torch.Tensor:
        # Portfolio daily returns: (B, T)
        port_returns = torch.sum(weights.unsqueeze(1) * asset_returns, dim=-1)
        
        mean_ret = torch.mean(port_returns)
        
        # Downside Deviation (only penalize negative returns)
        negative_returns = torch.clamp(port_returns, max=0.0)
        downside_std = torch.sqrt(torch.mean(negative_returns ** 2) + self.eps)
        
        sortino_ratio = mean_ret / downside_std
        
        # Penalize turnover / transaction costs
        weight_diff = torch.abs(weights[1:] - weights[:-1])
        turnover_penalty = torch.mean(weight_diff) * self.fee_rate
        
        # Return negative loss for gradient descent (maximizing Return + Sortino)
        loss = -(self.return_weight * mean_ret + sortino_ratio) + turnover_penalty
        return loss