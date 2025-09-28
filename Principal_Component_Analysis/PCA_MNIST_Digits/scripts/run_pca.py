# scripts/run_pca.py

# IMPORTS
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from src.image_loader import LoadConfig, load_images_from_csv
from src.pca_svd import (
    PCAModel,
    center_data, svd_thin, select_top_k,
    encode, decode,
    explained_variance, explained_variance_ratio
)
from src.plotting import plot_scree, plot_recon_error, plot_cum_evr

# CLI ARGUMENTS
def parse_args():
    # Build argument parser
    p = argparse.ArgumentParser(description="Run PCA (SVD) on a CSV and plot diagnostics")

    # Path to CSV
    p.add_argument("--csv", required=True, help="Path to CSV, e.g. data/raw/train.csv")

    # Label column, set to empty string to disable
    p.add_argument("--label-col", default="label", help="Column name for labels, set '' to disable")

    # Normalisation method
    p.add_argument("--normalise", default="auto", choices=["auto", "none"], help="auto rescales 0..255 to 0..1")

    # Number of components to keep before saving model
    p.add_argument("--k", type=int, default=50, help="Components to keep in the saved PCA model")

    # List of K values to scan for reconstruction error
    p.add_argument("--k-grid", type=int, nargs="+", default=[10, 25, 50, 100], help="K values to scan for reconstruction MSE plot")

    # Directory for plots
    p.add_argument("--plots-dir", default="figures", help="Directory to save plots")

    # Path to save fitted PCA state
    p.add_argument("--save-model", default="models/pca.pt", help="Path to save PCA state dict")

    # Disable loader cache
    p.add_argument("--no-cache", action="store_true", help="Disable caching in image loader")

    # Random seed (for future randomness)
    p.add_argument("--seed", type=int, default=0, help="Random seed")

    # K to use for the recon grid; defaults to --k if not set
    p.add_argument("--vis-k", type=int, default=None, help="Components to use for the reconstruction grid")

    # Output path for the 2x10 orig/recon grid image
    p.add_argument("--recon-out", default="figures/recon_digits_grid.png", help="Path to save orig/recon grid image")

    return p.parse_args()

# RECONSTRUCTION MSE (BATCHED)
def _recon_mse(X: torch.Tensor, mean: torch.Tensor, V_k: torch.Tensor, batch: int = 4096) -> float:
    # Number of samples
    N = X.shape[0]
    # Accumulator for total per-sample MSE
    acc = 0.0
    # Iterate over batches (to control memory)
    for i in range(0, N, batch):
        # Slice batch
        xb = X[i:i+batch]
        # Center
        Xc_b = xb - mean
        # Encode
        Z_b = Xc_b @ V_k
        # Decode
        Xh_b = Z_b @ V_k.T + mean
        # Accumulate per-sample MSE
        acc += torch.mean((xb - Xh_b).pow(2), dim=1).sum().item()
    # Return average
    return acc / N

# PICK INDICES FOR DIGITS 0..9
def _pick_digit_indices(y: torch.Tensor) -> list[int]:
    # If labels are missing, use first 10 rows
    if y is None:
        return list(range(10))
    # Collect one index per digit 0..9 where available
    out = []
    for d in range(10):
        idxs = (y == d).nonzero(as_tuple=False)
        if idxs.numel() > 0:
            out.append(int(idxs[0].item()))
    # Fallback if fewer than 10 found
    if not out:
        out = list(range(10))
    return out[:10]

# SAVE 2x10 ORIG/RECON GRID
def _save_recon_grid(X: torch.Tensor, mean: torch.Tensor, V_k: torch.Tensor, y: torch.Tensor | None, outpath: Path) -> None:
    # Pick indices
    indices = _pick_digit_indices(y)
    # Prepare containers
    orig_imgs, recon_imgs = [], []
    # Generate reconstructions
    with torch.no_grad():
        for idx in indices:
            xb = X[idx:idx+1]
            Xc_b = xb - mean
            Z_b = encode(Xc_b, V_k)
            xhb = decode(Z_b, V_k, mean)
            orig_imgs.append(xb.view(28, 28).detach().cpu().numpy())
            recon_imgs.append(xhb.view(28, 28).detach().cpu().numpy())
    # Create figure
    cols = len(indices)
    fig, axes = plt.subplots(2, cols, figsize=(1.6 * cols, 3.4))
    # Draw originals
    for j in range(cols):
        axes[0, j].imshow(orig_imgs[j], cmap="gray", vmin=0.0, vmax=1.0)
        axes[0, j].axis("off")
    # Draw reconstructions
    for j in range(cols):
        axes[1, j].imshow(recon_imgs[j], cmap="gray", vmin=0.0, vmax=1.0)
        axes[1, j].axis("off")
    # Row labels
    axes[0, 0].set_ylabel("orig")
    axes[1, 0].set_ylabel("recon")
    # Save
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

# MAIN
def main():
    # Parse CLI
    args = parse_args()

    # Set seed
    if args.seed is not None:
        torch.manual_seed(args.seed)

    # Resolve label column
    label_col = None if args.label_col == "" else args.label_col

    # Build LoadConfig
    cfg = LoadConfig(
        csv_path=args.csv,
        label_col=label_col,
        normalise=args.normalise,
        reshape=None,
        allow_cached=not args.no_cache,
        save_cache=not args.no_cache,
    )

    # Load data
    X, y = load_images_from_csv(cfg)
    N, D = X.shape
    print(f"Loaded X: {X.shape}, dtype={X.dtype}")

    # Center
    Xc, mean = center_data(X)

    # SVD (thin)
    U, S, V = svd_thin(Xc)

    # Explained Variance (EV) and Ratio (EVR)
    ev = explained_variance(S, N)
    evr = explained_variance_ratio(S, N)

    # Diagnostics
    print(f"Total variance: {ev.sum().item():.6f}")
    top5 = [round(v, 6) for v in evr[:5].tolist()]
    print(f"Top-5 Components EVR: {top5}")

    # Select K and save model
    _, S_k, V_k = select_top_k(U, S, V, args.k)
    model = PCAModel(mean=mean, components=V_k, S=S_k, n_samples=N)
    model_path = Path(args.save_model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "mean": model.mean,
            "components": model.components,
            "S": model.S,
            "n_samples": model.n_samples,
        },
        model_path,
    )
    print(f"Saved PCA model (K={args.k}) to {model_path}")

    # Prepare plots directory
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Plot for Scree and cumulative EVR
    plot_scree(ev, plots_dir / "scree.png")
    plot_cum_evr(evr, plots_dir / "cum_evr.png")

    # Reconstruction Error MSE vs K
    pairs = []
    with torch.no_grad():
        for k in args.k_grid:
            _, _, V_k_loop = select_top_k(U, S, V, k)
            mse = _recon_mse(X, mean, V_k_loop, batch=4096)
            pairs.append((k, mse))
            print(f"K={k:4d}  MSE={mse:.6f}")
    plot_recon_error(pairs, plots_dir / "recon_error.png")

    # Reconstruction grid (orig vs recon) using vis_k or fallback to k
    vis_k = args.vis_k if args.vis_k is not None else args.k
    _, _, V_k_vis = select_top_k(U, S, V, vis_k)
    recon_out = Path(args.recon_out)
    _save_recon_grid(X, mean, V_k_vis, y, recon_out)
    print(f"Saved orig/recon grid to {recon_out} (K={vis_k})")

# ENTRY
if __name__ == "__main__":
    main()
