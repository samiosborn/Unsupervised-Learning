# src/plotting.py

import matplotlib.pyplot as plt
import torch
from typing import List, Tuple
from pathlib import Path

# Scree plot of eigenvalues (explained variances)
def plot_scree(lam: torch.Tensor, outpath: Path) -> None:
    lam_np = lam.detach().cpu().numpy()
    xs = range(1, lam_np.shape[0] + 1)
    plt.figure()
    plt.plot(xs, lam_np, marker = "o")
    plt.xlabel("Component Index")
    plt.ylabel("Explained Variance")
    plt.title("Scree Plot")
    plt.tight_layout()
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=150)
    plt.close()

# Cumulative explained variance (ratio)
def plot_cum_evr(evr: torch.Tensor, outpath: Path) -> None:
    c = torch.cumsum(evr, dim=0).detach().cpu().numpy()
    xs = range(1, len(c) + 1)
    plt.figure()
    plt.plot(xs, c, marker = "o")
    plt.xlabel("Number of Components (K)")
    plt.ylabel("Cumulative EVR")
    plt.ylim(0, 1.01)
    plt.title("Cumulative Explained Variance Ratio (EVR)")
    plt.tight_layout()
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=150)
    plt.close()

# Plot of reconstruction error loss (MSE)
def plot_recon_error(pairs: List[Tuple[int, float]], outpath: Path) -> None: 
    # pairs as a list of (K, MSE)
    if not pairs:
        raise ValueError("plot_recon_error: empty (K, MSE) list.")
    ks = [k for k,_ in pairs]
    mses = [m for _,m in pairs]
    plt.figure()
    plt.plot(ks, mses, marker = "o")
    plt.xlabel("Number of Components (K)")
    plt.ylabel("Reconstruction MSE")
    plt.title("Reconstruction MSE vs. No. Components (K)")
    plt.tight_layout()
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=150)
    plt.close()
