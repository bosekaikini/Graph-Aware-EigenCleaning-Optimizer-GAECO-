import torch
import torch.nn as nn

class FusionAttention(nn.Module):
    """
    Spectral Attention Module merging Spatial GCN features [Batch, N, H] 
    and Spectral MLP features [Batch, H] to output asset shrinkage parameters gamma.
    """
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.spatial_proj = nn.Linear(hidden_dim, hidden_dim)
        self.spectral_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attention_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Outputs gamma in (0, 1)
        )

    def forward(self, spatial_emb: torch.Tensor, spectral_emb: torch.Tensor) -> torch.Tensor:
        """
        spatial_emb:  [Batch, N, Hidden_Dim]
        spectral_emb: [Batch, Hidden_Dim]
        Returns: Shrinkage Vector gamma [Batch, N]
        """
        batch_size, num_assets, _ = spatial_emb.shape
        
        # Expand global spectral embedding to all assets: [Batch, N, Hidden_Dim]
        spectral_expanded = spectral_emb.unsqueeze(1).repeat(1, num_assets, 1)
        
        # Additive Attention Fusion
        fused = self.spatial_proj(spatial_emb) + self.spectral_proj(spectral_expanded)
        
        # Predict asset-specific shrinkage vector gamma
        gamma = self.attention_head(fused).squeeze(-1) # [Batch, N]
        return gamma