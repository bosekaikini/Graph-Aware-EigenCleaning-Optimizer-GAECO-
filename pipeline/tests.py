import pytest
import torch
import numpy as np

from models.gaeco_network import GAECONetPipeline, GAECONetCore
from models.portfolio_layer import DifferentiableMeanVariance
from backtester.benchmarks.estimators import sample_covariance, ledoit_wolf_shrinkage, marchenko_pastur_denoise

@pytest.fixture
def mock_financial_data():
    """Generates synthetic portfolio batch dimensions."""
    batch_size = 2
    num_assets = 10
    in_features = 2
    
    torch.manual_seed(42)
    node_features = torch.randn(batch_size, num_assets, in_features)
    laplacian = torch.eye(num_assets).unsqueeze(0).repeat(batch_size, 1, 1)
    
    # Generate valid positive eigenvalues and orthogonal eigenvectors
    evals = torch.sort(torch.rand(batch_size, num_assets) + 0.1, descending=True)[0]
    raw_mat = torch.randn(batch_size, num_assets, num_assets)
    q, _ = torch.linalg.qr(raw_mat)
    
    return node_features, laplacian, evals, q, num_assets

def test_gaec_net_dimensions_and_psd(mock_financial_data):
    """Checks output dimensions and positive semi-definiteness of cleaned matrix."""
    node_features, laplacian, evals, evecs, num_assets = mock_financial_data
    batch_size = node_features.size(0)
    
    pipeline = GAECONetPipeline(num_assets=num_assets, in_features=2, hidden_dim=32)
    weights, C_clean, gamma, mu = pipeline(node_features, laplacian, evals, evecs)
    
    # Shape Checks
    assert weights.shape == (batch_size, num_assets)
    assert C_clean.shape == (batch_size, num_assets, num_assets)
    assert gamma.shape == (batch_size, num_assets)
    assert mu.shape == (batch_size, num_assets)

    # Convexity / Long-Only Constraint Check (weights sum to 1)
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(batch_size))
    assert (weights >= 0).all()

    # PSD Check: All eigenvalues of C_clean must be strictly positive (> 0)
    clean_evals = torch.linalg.eigvalsh(C_clean)
    assert (clean_evals > 0).all()

def test_autograd_gradients_flow(mock_financial_data):
    """Verifies end-to-end backpropagation through Markowitz portfolio solver."""
    node_features, laplacian, evals, evecs, num_assets = mock_financial_data
    
    pipeline = GAECONetPipeline(num_assets=num_assets, in_features=2, hidden_dim=32)
    weights, C_clean, _, mu = pipeline(node_features, laplacian, evals, evecs)
    
    # Compute dummy scalar loss and trigger backward pass
    loss = weights.pow(2).sum() + C_clean.pow(2).sum()
    loss.backward()
    
    # Ensure gradients reach all network branches
    for name, param in pipeline.named_parameters():
        assert param.grad is not None, f"Gradient failed to flow to {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient detected in {name}"

def test_benchmark_estimators():
    """Validates classical covariance matrix estimators."""
    np.random.seed(42)
    synthetic_returns = np.random.normal(0, 0.02, (100, 15))
    
    cov_sample = sample_covariance(synthetic_returns)
    cov_lw = ledoit_wolf_shrinkage(synthetic_returns)
    cov_mp = marchenko_pastur_denoise(synthetic_returns)
    
    assert cov_sample.shape == (15, 15)
    assert cov_lw.shape == (15, 15)
    assert cov_mp.shape == (15, 15)
    
    # Check PSD for numpy estimators
    assert np.all(np.linalg.eigvalsh(cov_lw) > 0)
    assert np.all(np.linalg.eigvalsh(cov_mp) > 0)