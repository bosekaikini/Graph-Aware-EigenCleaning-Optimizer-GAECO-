# explainability/explainer.py

import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pipeline.train import _extract_node_features


class GAECOExplainer(nn.Module):
    """
    Post-hoc GNN Explainer for GAECO-Net matching Karzanov et al. (2025).
    Learns continuous edge and node feature masks to explain allocation outputs.
    """
    def __init__(self, model: nn.Module, num_nodes: int, in_features: int):
        super().__init__()
        self.model = model
        
        # Freeze base model parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.num_nodes = num_nodes
        self.in_features = in_features

        # Mask parameters initialized in logit space
        self.edge_mask_param = nn.Parameter(torch.randn(num_nodes, num_nodes) * 0.1)
        self.node_feat_mask_param = nn.Parameter(torch.randn(in_features) * 0.1)

    def forward(
        self, 
        node_features: torch.Tensor, 
        laplacian: torch.Tensor, 
        eigenvals: torch.Tensor, 
        eigenvecs: torch.Tensor
    ):
        # Sigmoidal mapping to [0, 1]
        edge_mask = torch.sigmoid(self.edge_mask_param)
        node_feat_mask = torch.sigmoid(self.node_feat_mask_param)

        # Enforce symmetric graph structure
        edge_mask = (edge_mask + edge_mask.T) / 2.0
        
        # Mask Laplacian and Node Features
        masked_laplacian = laplacian * edge_mask.unsqueeze(0)
        masked_features = node_features * node_feat_mask.view(1, 1, -1)

        # Pass through frozen pipeline
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
    lambda_feat: float = 0.005,
    lambda_ent: float = 0.001
) -> tuple[np.ndarray, np.ndarray]:
    """
    Optimizes edge and feature masks to explain allocation decisions.
    Explicitly enables gradients so backward passes work during evaluation loops.
    """
    model.eval()
    
    # Enable gradient calculation for mask optimization, 
    # even when called inside a global torch.no_grad() block
    with torch.enable_grad():
        num_nodes = node_features.shape[1]
        in_features = node_features.shape[2]

        explainer = GAECOExplainer(model, num_nodes, in_features)
        optimizer = torch.optim.Adam(explainer.parameters(), lr=lr)
        
        # Ensure target_weights are detached from autograd history
        target_weights_detached = target_weights.detach()

        for epoch in range(epochs):
            optimizer.zero_grad()
            pred_weights, edge_mask, feat_mask = explainer(
                node_features, laplacian, eigenvals, eigenvecs
            )

            # 1. Fidelity Loss (MSE against base model target allocation)
            loss_fidelity = F.mse_loss(pred_weights, target_weights_detached)

            # 2. Sparsity Regularization (L1 norm on edge and feature masks)
            loss_edge_sparse = lambda_edge * torch.sum(edge_mask)
            loss_feat_sparse = lambda_feat * torch.sum(feat_mask)
            
            # 3. Entropy Regularization (pushes values closer to 0 or 1)
            eps = 1e-8
            edge_ent = - (edge_mask * torch.log(edge_mask + eps) + (1 - edge_mask) * torch.log(1 - edge_mask + eps))
            feat_ent = - (feat_mask * torch.log(feat_mask + eps) + (1 - feat_mask) * torch.log(1 - feat_mask + eps))
            loss_entropy = lambda_ent * (torch.mean(edge_ent) + torch.mean(feat_ent))

            total_loss = loss_fidelity + loss_edge_sparse + loss_feat_sparse + loss_entropy
            total_loss.backward()
            optimizer.step()

        with torch.no_grad():
            final_edge_mask = torch.sigmoid(explainer.edge_mask_param).detach().cpu().numpy()
            final_edge_mask = (final_edge_mask + final_edge_mask.T) / 2.0
            final_feat_mask = torch.sigmoid(explainer.node_feat_mask_param).detach().cpu().numpy()

    return final_edge_mask, final_feat_mask


def plot_and_save_subgraph_attribution(
    edge_mask: np.ndarray,
    asset_names: list[str],
    date_str: str,
    top_k_edges: int = 15,
    output_filename: str = "attribution_graph.png"
):
    """
    Generates topological attribution map for top influential asset edges.
    """
    G = nx.Graph()
    num_assets = len(asset_names)

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


def generate_explained_allocations(
    model: nn.Module,
    returns_df: pd.DataFrame,
    empirical_data: dict,
    lookback: int = 60
) -> pd.DataFrame:
    """
    Generates allocations where graph inputs are masked/filtered using 
    the GAECO-Explainer to prune noisy edge and feature dependencies.
    """
    model.eval()
    num_assets = returns_df.shape[1]
    returns_tensor = torch.tensor(returns_df.values, dtype=torch.float32)
    num_timesteps = len(returns_df)
    
    explained_weights = []

    with torch.no_grad():
        for t in range(len(empirical_data["corr_emp"])):
            if t + lookback > num_timesteps:
                break

            window_ret = returns_tensor[t : t + lookback]
            curr_idx_val = returns_tensor[t + lookback - 1]
            node_feat = _extract_node_features(window_ret, curr_idx_val, num_assets)

            sample_corr = empirical_data["corr_emp"][t]
            sample_lap = (torch.eye(num_assets) - sample_corr).unsqueeze(0)
            eigenvals = empirical_data["eigenvals"][t:t+1]
            eigenvecs = empirical_data["eigenvecs"][t:t+1]

            # Pass raw forward prediction explicitly detached as baseline target
            raw_weights, _, _, _ = model(node_feat, sample_lap, eigenvals, eigenvecs)

            # 1. Generate post-hoc masks via Explainer
            edge_mask, feat_mask = explain_allocation(
                model=model,
                node_features=node_feat,
                laplacian=sample_lap,
                eigenvals=eigenvals,
                eigenvecs=eigenvecs,
                target_weights=raw_weights.detach(),
                epochs=30,  # Fast inference pass
                lr=0.01
            )

            # 2. Mask noisy Laplacian edges & weak node features
            edge_threshold = np.percentile(edge_mask, 30)  # Keep top 70%
            filtered_edge_mask = torch.tensor(edge_mask > edge_threshold, dtype=torch.float32)
            
            masked_lap = sample_lap * filtered_edge_mask.unsqueeze(0)
            masked_feat = node_feat * torch.tensor(feat_mask, dtype=torch.float32).view(1, 1, -1)

            # 3. Predict weights on pruned graph
            weights, _, _, _ = model(masked_feat, masked_lap, eigenvals, eigenvecs)
            explained_weights.append(weights.squeeze(0).cpu().numpy())

    weights_array = np.array(explained_weights)
    eval_dates = empirical_data["dates"][:len(weights_array)]

    return pd.DataFrame(weights_array, index=eval_dates, columns=returns_df.columns)