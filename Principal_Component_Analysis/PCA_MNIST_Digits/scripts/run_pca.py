# scripts/run_pca.py

# IMPORTS
import argparse
from pathlib import Path
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
            # Slice basis for this K
            _, S_k, V_k = select_top_k(U, S, V, k)
            # Compute MSE
            mse = _recon_mse(X, mean, V_k, batch=4096)
            # Collect for plot
            pairs.append((k, mse))
            # Print
            print(f"K={k:4d}  MSE={mse:.6f}")

    # Plot reconstruction error
    plot_recon_error(pairs, plots_dir / "recon_error.png")

# ENTRY
if __name__ == "__main__":
    main()
