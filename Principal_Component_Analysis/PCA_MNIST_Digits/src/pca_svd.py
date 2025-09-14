# src/pca_svd.py

# IMPORTS
from dataclasses import dataclass
import torch
from typing import Tuple

# Use float64 for numerical stability throughout the PCA math
DTYPE = torch.float64

# X: (N, D) -> returns Xc: (N, D), mean: (D,)
def center_data(X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Upcast X to float64 to keep mean/centering precise
    X64 = X.to(DTYPE)
    # Per-feature mean (D,)
    mean = X64.mean(dim=0)
    # Centered data (N, D)
    Xc = X64 - mean
    return Xc, mean

# Xc: (N, D) -> returns U: (N, r), S: (r,), V: (D, r) with r = min(N, D)
def svd_thin(Xc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Thin SVD decomposition: Xc = U @ diag(S) @ V.T
    U, S, Vh = torch.linalg.svd(Xc.to(DTYPE), full_matrices=False)
    # Convert V^T to V
    V = Vh.T
    return U, S, V

# Slice the top-K singular triplets: returns U_k: (N, K), S_k: (K,), V_k: (D, K)
def select_top_k(U: torch.Tensor, S: torch.Tensor, V: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Rank r = number of singular values
    r = S.shape[0]
    # Bound check for k
    if not (1 <= k <= r):
        raise ValueError(f"k={k} must be in [1, {r}]")
    # Take leading K columns/elements
    U_k, S_k, V_k = U[:, :k], S[:k], V[:, :k]
    return U_k, S_k, V_k

# Project centered data onto the top-K principal directions: Z: (N, K)
def encode(Xc: torch.Tensor, V_k: torch.Tensor) -> torch.Tensor:
    # Shape checks
    assert Xc.ndim == 2 and V_k.ndim == 2, "encode expects 2D tensors"
    assert Xc.shape[1] == V_k.shape[0], "Feature dim mismatch: Xc (N,D) vs V_k (D,K)"
    # Projection
    Z = Xc @ V_k
    return Z

# Reconstruct from latent Z and add back the mean: X_hat: (N, D)
def decode(Z: torch.Tensor, V_k: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
    # Shape checks
    assert Z.ndim == 2 and V_k.ndim == 2, "decode expects 2D tensors for Z and V_k"
    assert V_k.shape[0] == mean.shape[0], "Mean dimension must match feature dimension D"
    # Reconstruction
    X_hat = Z @ V_k.T + mean
    return X_hat

# From singular values S and sample count N: ev_i = S_i^2 / (N - 1), returns ev: (len(S),)
def explained_variance(S: torch.Tensor, n_samples: int) -> torch.Tensor:
    # Guard against division by zero
    if n_samples <= 1:
        raise ValueError("n_samples must be > 1 to compute sample variance.")
    # Explained variance (eigenvalues of sample covariance)
    ev = (S**2) / (n_samples - 1)
    return ev

# Explained variance ratio evr_i = ev_i / sum_j ev_j, returns evr: (len(S),)
def explained_variance_ratio(S: torch.Tensor, n_samples: int) -> torch.Tensor:
    # Compute explained variance
    ev = explained_variance(S, n_samples)
    # Handle degenerate zero-variance case
    denom = ev.sum()
    if denom == 0:
        return torch.zeros_like(ev)
    # Normalise to ratios
    evr = ev / denom
    return evr

# Holds fitted PCA state and provides transform/inverse_transform convenience
@dataclass
class PCAModel:
    # mean: (D,)
    mean: torch.Tensor
    # components (V_k): (D, K)
    components: torch.Tensor
    # top-K singular values: (K,)
    S: torch.Tensor
    # number of samples used to fit
    n_samples: int

    # Transform = center with fitted mean then encode: Z: (N, K)
    def transform(self, X: torch.Tensor) -> torch.Tensor:
        return encode(X - self.mean, self.components)

    # Inverse transform = decode with fitted components and mean: X_hat: (N, D)
    def inverse_transform(self, Z: torch.Tensor) -> torch.Tensor:
        return decode(Z, self.components, self.mean)
