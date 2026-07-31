import torch
import torch.nn as nn

class GraphBuilder:
    """
    Constructs dynamic dynamic adjacency matrices and Normalized Graph Laplacians 
    for PyTorch / PyG batch processing.
    """
    def __init__(self, num_assets: int, alpha: float = 0.5, threshold: float = 0.2):
        """
        alpha: Weight ratio between sector structural prior and dynamic correlation [0, 1].
        threshold: Absolute correlation cut-off to remove weak noise edges.
        """
        self.num_assets = num_assets
        self.alpha = alpha
        self.threshold = threshold

    def build_hybrid_adjacency(
        self, 
        corr_emp: torch.Tensor, 
        sector_adjacency: torch.Tensor | None=None
    ) -> torch.Tensor:
        """
        Computes dynamic adjacency matrix:
        A_t = alpha * A_sector + (1 - alpha) * Threshold( |C_emp| )
        
        corr_emp: Tensor [Batch, N, N]
        sector_adjacency: Tensor [N, N] (Optional binary matrix matching GICS sectors)
        """
        batch_size = corr_emp.size(0)

        # 1. Threshold empirical correlation to filter weak noise edges
        abs_corr = torch.abs(corr_emp)
        dynamic_adj = torch.where(abs_corr > self.threshold, abs_corr, torch.zeros_like(abs_corr))
        
        # Zero-out self-loops for adjacency calculation
        mask = torch.eye(self.num_assets, device=corr_emp.device).unsqueeze(0)
        dynamic_adj = dynamic_adj * (1.0 - mask)

        # 2. Combine with Sector Prior if available
        if sector_adjacency is not None:
            sector_prior = sector_adjacency.unsqueeze(0).repeat(batch_size, 1, 1)
            A_t = self.alpha * sector_prior + (1 - self.alpha) * dynamic_adj
        else:
            A_t = dynamic_adj

        return A_t

    def compute_normalized_laplacian(self, adjacency: torch.Tensor) -> torch.Tensor:
        """
        Computes symmetric normalized Graph Laplacian for GCN processing:
        L_norm = I - D^(-1/2) * A * D^(-1/2)
        
        adjacency: Tensor [Batch, N, N]
        """
        batch_size, N, _ = adjacency.shape
        device = adjacency.device

        # Degree vector D_i = sum_j A_ij
        deg = torch.sum(adjacency, dim=2)  # [Batch, N]
        
        # D^(-1/2), replacing infinity/zeros with 0
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt) | torch.isnan(deg_inv_sqrt)] = 0.0

        # Construct diagonal degree matrix D^(-1/2)
        D_mat = torch.diag_embed(deg_inv_sqrt)  # [Batch, N, N]

        # L_norm = I - D^(-1/2) * A * D^(-1/2)
        I = torch.eye(N, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        normalized_laplacian = I - torch.bmm(torch.bmm(D_mat, adjacency), D_mat)

        return normalized_laplacian