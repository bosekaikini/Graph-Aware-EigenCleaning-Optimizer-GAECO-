import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

class GAECOExplainer(nn.Module):
    """
    Post-hoc GNN Explainer for GAECO-Net.
    Learns continuous edge and feature masks to identify top edge correlations
    and node features driving target portfolio allocations.
    """
    def __init__(self, model: nn.Module, num_nodes: int, in_features: int):
        super().__init__()
        self.model = model
        # Freeze base model parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.num_nodes = num_nodes
        self.in_features = in_features

        # Initialize mask parameters in logit space
        self.edge_mask_param = nn.Parameter(torch.randn(num_nodes, num_nodes) * 0.1)
        self.node_feat_mask_param = nn.Parameter(torch.randn(in_features) * 0.1)

    def forward(self, node_features: torch.Tensor, laplacian: torch.Tensor, 
                eigenvals: torch.Tensor, eigenvecs: torch.Tensor):
        # Apply sigmoid to constrain masks to [0, 1]
        edge_mask = torch.sigmoid(self.edge_mask_param)
        node_feat_mask = torch.sigmoid(self.node_feat_mask_param)

        # Symmetric edge masking
        edge_mask = (edge_mask + edge_mask.T) / 2.0
        masked_laplacian = laplacian * edge_mask.unsqueeze(0)
        
        # Feature masking
        masked_features = node_features * node_feat_mask.view(1, 1, -1)

        # Forward pass through frozen base model
        predicted_weights, _, _, _ = self.model(
            masked_features, masked_laplacian, eigenvals, eigenvecs
        )
        return predicted_weights, edge_mask, node_feat_mask


def explain_allocation(
    model: nn.Module,
    node_features: torch.Tensor,
    laplacian: torch.Tensor,
    eigenvals: torch.Tensor,
    eigenvecs: torch.Tensor,
    target_weights: torch.Tensor,
    epochs: int = 200,
    lr: float = 0.01,
    lambda_edge: float = 0.005,
    lambda_feat: float = 0.005
) -> tuple[np.ndarray, np.ndarray]:
    """
    Optimizes edge and feature masks to explain a specific allocation decision.
    """
    model.eval()
    num_nodes = node_features.shape[1]
    in_features = node_features.shape[2]

    explainer = GAECOExplainer(model, num_nodes, in_features)
    optimizer = torch.optim.Adam(explainer.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        pred_weights, edge_mask, feat_mask = explainer(
            node_features, laplacian, eigenvals, eigenvecs
        )

        # Fidelity Loss: MSE between masked predictions and original target weights
        loss_fidelity = nn.functional.mse_loss(pred_weights, target_weights)

        # Regularization Losses: Sparsity & Entropy on masks
        loss_edge_sparse = lambda_edge * torch.mean(edge_mask)
        loss_feat_sparse = lambda_feat * torch.mean(feat_mask)
        
        loss_edge_ent = -0.001 * torch.mean(
            edge_mask * torch.log(edge_mask + 1e-8) + (1 - edge_mask) * torch.log(1 - edge_mask + 1e-8)
        )

        total_loss = loss_fidelity + loss_edge_sparse + loss_feat_sparse + loss_edge_ent
        total_loss.backward()
        optimizer.step()

    # Extract optimized mask values
    with torch.no_grad():
        final_edge_mask = torch.sigmoid(explainer.edge_mask_param).cpu().numpy()
        final_edge_mask = (final_edge_mask + final_edge_mask.T) / 2.0
        final_feat_mask = torch.sigmoid(explainer.node_feat_mask_param).cpu().numpy()

    return final_edge_mask, final_feat_mask


def plot_and_save_subgraph_attribution(
    edge_mask: np.ndarray,
    asset_names: list[str],
    date_str: str,
    top_k_edges: int = 15,
    output_filename: str = "attribution_graph.png"
):
    """
    Visualizes top influential graph edges driving the model's allocation decision.
    """
    G = nx.Graph()
    num_assets = len(asset_names)

    # Find indices of top K edge weights
    triu_indices = np.triu_indices(num_assets, k=1)
    edge_weights = edge_mask[triu_indices]
    top_indices = np.argsort(edge_weights)[-top_k_edges:]

    for idx in top_indices:
        u_idx = triu_indices[0][idx]
        v_idx = triu_indices[1][idx]
        w = edge_weights[idx]
        G.add_edge(asset_names[u_idx], asset_names[v_idx], weight=w)

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    weights = [G[u][v]['weight'] * 4 for u, v in G.edges()]

    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue', edgecolors='black')
    nx.draw_networkx_edges(G, pos, width=weights, edge_color='darkblue', alpha=0.7)
    nx.draw_networkx_labels(G, pos, font_size=9, font_family='sans-serif', font_weight='bold')

    plt.title(f"GAECO-Net Topological Attribution Map ({date_str})", fontsize=12, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"Successfully generated and saved attribution graph to '{output_filename}'")