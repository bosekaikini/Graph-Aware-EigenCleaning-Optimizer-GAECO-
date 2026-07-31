import torch
import torch.nn as nn

class SpectralMLP(nn.Module):
    """
    Processes decomposed empirical eigenvalue spectrum Lambda [Batch, N]
    to extract global market mode representations.
    """
    def __init__(self, num_assets: int, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_assets, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )

    def forward(self, eigenvals: torch.Tensor) -> torch.Tensor:
        """
        eigenvals: [Batch, N] Decomposed eigenvalues of C_emp
        Returns: Spectral Embeddings [Batch, Hidden_Dim]
        """
        return self.mlp(eigenvals)