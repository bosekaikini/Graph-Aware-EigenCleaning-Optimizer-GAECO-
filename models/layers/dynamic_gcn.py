import torch
import torch.nn as nn

class SpatialGCN(nn.Module):
    """
    Spatial Graph Convolutional branch.
    Processes node features X [Batch, N, F] and Laplacian L [Batch, N, N]
    using Spectral Graph Convolution: H = L * X * W
    """
    def __init__(self, in_features: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, laplacian: torch.Tensor) -> torch.Tensor:
        """
        x: [Batch, N, In_Features] (e.g., rolling asset vol, returns momentum)
        laplacian: [Batch, N, N] Normalized Graph Laplacian
        Returns: Spatial Embeddings [Batch, N, Hidden_Dim]
        """
        # Layer 1: H1 = Act( L * X * W1 )
        h = torch.bmm(laplacian, x)          # Graph diffusion aggregation
        h = self.act(self.fc1(h))             # Linear feature transformation
        
        # Layer 2: H2 = Act( L * H1 * W2 )
        h = torch.bmm(laplacian, h)
        h = self.act(self.fc2(h))
        return h