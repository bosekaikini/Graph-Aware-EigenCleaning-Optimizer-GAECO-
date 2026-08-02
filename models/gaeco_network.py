import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers.dynamic_gcn import SpatialGCN
from models.layers.spectral_mlp import SpectralMLP
from models.layers.attention import FusionAttention
from models.portfolio_layer import DifferentiableMeanVariance

class GAECONetCore(nn.Module):
    """
    GAECO-Net Denoising Architecture.
    """
    def __init__(self, num_assets: int, in_features: int = 2, hidden_dim: int = 64, epsilon: float = 1e-5):
        super().__init__()
        self.num_assets = num_assets
        self.epsilon = epsilon
        
        self.gcn_branch = SpatialGCN(in_features=in_features, hidden_dim=hidden_dim)
        self.mlp_branch = SpectralMLP(num_assets=num_assets, hidden_dim=hidden_dim)
        self.attention = FusionAttention(hidden_dim=hidden_dim)

        self.return_head=nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.GELU(),
            nn.Linear(hidden_dim//2, 1)
        )

    def forward(
        self, 
        node_features: torch.Tensor, 
        laplacian: torch.Tensor, 
        eigenvals: torch.Tensor, 
        eigenvecs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        node_features: [Batch, N, In_Features]
        laplacian:     [Batch, N, N]
        eigenvals:     [Batch, N] Empirical eigenvalues (Lambda)
        eigenvecs:     [Batch, N, N] Empirical eigenvectors (U)
        
        Returns: C_clean [Batch, N, N], gamma [Batch, N]
        """
        # 1. Feature Extraction
        spatial_emb = self.gcn_branch(node_features, laplacian)   # [Batch, N, H]
        spectral_emb = self.mlp_branch(eigenvals)                 # [Batch, H]

        # 2. Spectral Attention Shrinkage
        gamma = self.attention(spatial_emb, spectral_emb)         # [Batch, N]

        # 3. Adaptive Shrinkage of Eigenvalue Spectrum
        # Mean eigenvalue target (shrinkage towards identity baseline)
        target_lambda = torch.mean(eigenvals, dim=-1, keepdim=True)
        lambda_shrunk = gamma * eigenvals + (1.0 - gamma) * target_lambda

        # Enforce PSD (lambda > 0) via Softplus numerical floor
        lambda_clean = F.softplus(lambda_shrunk) + self.epsilon   # [Batch, N]

        # 4. Reconstruct Correlation Matrix: C_clean = U * diag(lambda_clean) * U^T
        Lambda_mat = torch.diag_embed(lambda_clean)               # [Batch, N, N]
        C_clean = torch.bmm(eigenvecs, torch.bmm(Lambda_mat, eigenvecs.transpose(1, 2)))
        mu=self.return_head(spatial_emb).squeeze(-1)  # [Batch, N]

        return C_clean, mu, gamma


class GAECONetPipeline(nn.Module):
    """
    End-to-End Pipeline joining Matrix Denoising with Markowitz Optimization.
    """
    def __init__(
        self,
        num_assets: int,
        in_features: int = 8,
        hidden_dim: int = 64,
        risk_aversion: float = 0.5,
        top_k: int | None = None,
    ):
        """
        top_k: number of highest-conviction assets to actually hold
        (see models/portfolio_layer.py fix). Defaults to `num_assets`
        (dense softmax, i.e. the previous behavior); pass e.g.
        `top_k=15` for a 30-name universe to enable real concentration.
        """
        super().__init__()
        self.core = GAECONetCore(num_assets=num_assets, in_features=in_features, hidden_dim=hidden_dim)
        self.portfolio_layer = DifferentiableMeanVariance(
            num_assets=num_assets, risk_aversion=risk_aversion, top_k=top_k
        )

    def forward(
        self, 
        node_features: torch.Tensor, 
        laplacian: torch.Tensor, 
        eigenvals: torch.Tensor, 
        eigenvecs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns: portfolio_weights [Batch, N], C_clean [Batch, N, N], gamma [Batch, N], mu [Batch, N]
        """
        C_clean, mu, gamma = self.core(node_features, laplacian, eigenvals, eigenvecs)
        weights = self.portfolio_layer(mu, C_clean)
        return weights, C_clean, mu, gamma