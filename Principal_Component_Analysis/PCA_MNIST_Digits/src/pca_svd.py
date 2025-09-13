# src/pca_svd.py

# IMPORTS
from dataclasses import dataclass
import torch
from typing import Tuple

# Use float64 for numerical stability
DTYPE = torch.float64

# CENTER DATA
def center_data(X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Upcast X
    X64 = X.to(DTYPE)
    # Get the per-feature mean of the data (D, )
    mean = X64.mean(dim=0)
    # Subtract the mean (N, D)
    Xc = X64 - mean
    return Xc, mean

# SVD (THIN)
def svd_thin(Xc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Thin SVD
    U, S, Vh = torch.linalg.svd(Xc.to(DTYPE), full_matrices = False)
    # Transpose of transpose
    V = Vh.T
    return U, S, V

# SELECT K TOP COMPONENTS
def select_top_k(U, S, V, k: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Rank
    r = S.shape[0]
    # Bound k
    if not (1 <= k <= r): raise ValueError(f"k={k} must be in [1, {r}]")
    # Reduce matrices
    U_k, S_k, V_k = U[:, :k], S[:k], V[:, :k]
    return U_k, S_k, V_k

# ENCODE
def encode(Xc: torch.Tensor, V_k: torch.Tensor) -> torch.Tensor:
    # Multiply by V_k (N, K)
    Z = Xc @ V_k
    return Z

# DECODE
def decode(Z: torch.Tensor, V_k: torch.Tensor, mean: torch.Tensor) -> torch.Tensor: 
    # Multiply by V_k transpose and add back the mean
    X_hat = Z @ V_k.T + mean
    return X_hat

# EXPLAINED VARIANCE
def explained_variance(S: torch.Tensor, n_samples: int) -> torch.Tensor:
    # Variance per principal component (same length as S)
    var_explained = (S**2) / (n_samples - 1)
    return var_explained

# EXPLAINED VARIANCE RATIO
def explained_variance_ratio(S: torch.Tensor, n_samples: int) -> torch.Tensor: 
    # Explained variance
    var_explained = explained_variance(S, n_samples) 
    # Ratio of explained variance per principal component to total explained variance
    var_explained_ratio = var_explained / var_explained.sum()
    return var_explained_ratio

# PCA MODEL DATACLASS
@dataclass
class PCAModel:
    # Mean, shape (D, )
    mean: torch.Tensor
    # Components V_k = V[:, :K], shape (D, K)
    components: torch.Tensor
    # S = (K, )
    S: torch.Tensor
    # Number of samples (N)
    n_samples: int

    # Transform
    def transform(self, X: torch.Tensor) -> torch.Tensor:
        return encode(X - self.mean, self.components)
        
    # Inverse Transform
    def inverse_transform(self, Z: torch.Tensor) -> torch.Tensor:
        return decode(Z, self.components, self.mean)
