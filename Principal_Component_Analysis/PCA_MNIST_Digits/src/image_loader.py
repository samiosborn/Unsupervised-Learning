# src/image_loader.py

# Dataclass to turn a class into a configuration container
from dataclasses import dataclass
from pathlib import Path
# Typing imports to make function signatures clear
from typing import Optional, Tuple, Union
import pandas as pd
import torch
# DataLoader utilities for mini-batches
from torch.utils.data import DataLoader, TensorDataset

# LOAD CONFIG
# Configuration object describes how to read and process a CSV file
@dataclass
class LoadConfig:
    # Path to the CSV
    csv_path: Union[str, Path]
    # Optional column name to drop as labels
    label_col: Optional[str] = "label"
    # Normalise the data (either auto, a numeric divisor, or none)
    normalise: Union[str, float] = "auto"
    # Target PyTorch data-type for the features
    dtype: torch.dtype = torch.float32
    # Optionally reshape flat rows to (C, H, W) i.e. for plotting (keep None for PCA)
    reshape: Optional[Tuple[int, int, int]] = None
    # Cache directory: where to save a processed .pt cache
    cache_dir: Optional[Union[str, Path]] = "data/processed"
    # Name override for the cached file
    cache_tag: Optional[str] = None 
    # Load from cache when possible
    allow_cached: bool = True
    # Save processed tensors to cache path
    save_cache: bool = True

    # The cache key function makes a unique and descriptive filename
    def cache_key(self) -> str:
        p = Path(self.csv_path)
        base = self.cache_tag or p.stem
        norm = str(self.normalise)
        shape = f"{self.reshape}" if self.reshape else "flat"
        lbl = self.label_col if self.label_col else "nolabel"
        return f"{base}_norm-{norm}_shape-{shape}_lbl-{lbl}"

# UTILITY
# Helps compute the cache file path based on the config
def _likely_cache_path(cfg: LoadConfig) -> Optional[Path]:
    # If caching is disabled, return None
    if not cfg.cache_dir:
        return None
    # Get the filename and directory using the cache directory and config cache key
    cache_dir = Path(cfg.cache_dir)
    key = cfg.cache_key()
    return cache_dir / f"{key}.pt"

# LOAD IMAGE FROM CSV
# Returns: 
# - X: a float tensor of shape (N, D) / or (N, C, H, W) if reshape
# - y: Optional: int64 label tensor of shape (N,) if label column
def load_images_from_csv(cfg: LoadConfig) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    # If caching is enabled and the file exists, load and return it
    cache_path = _likely_cache_path(cfg)
    if cfg.allow_cached and cache_path and cache_path.exists():
        blob = torch.load(cache_path)
        return blob["X"], blob.get("y", None)
    # Read the CSV into a pandas DataFrame (reminder: Pandas infers headers automatically)
    df = pd.read_csv(cfg.csv_path)
    # Guard against empty files
    if df.empty:
        raise ValueError(f"CSV is empty: {cfg.csv_path}")
    
    # Keep labels as a separate tensor
    y = None
    if cfg.label_col and cfg.label_col in df.columns:
        # If a label column is configured and present, pop it off the table
        y = torch.as_tensor(df.pop(cfg.label_col).to_numpy(), dtype=torch.long)
    # The remaining columns are features, so convert to a NumPy array, size: (N, D)
    X_np = df.to_numpy()
    # Double-check that the table is 2D
    if X_np.ndim != 2:
        raise ValueError(f"Expected a 2D table, got shape {X_np.shape}")
    
    # Convert to a Torch tensor with the requested datatype
    X = torch.as_tensor(X_np, dtype=cfg.dtype)
    # If normalise is "auto", presume values are 0 to 255 and scale to 0 to 1
    if cfg.normalise == "auto":
        x_max = float(X.max().item())
        x_min = float(X.min().item())
        if 1.0 < x_max <= 255.0 and x_min >= 0.0:
            X = X / 255.0
    elif cfg.normalise == "none":
        pass
    else:
        # Divide by divisor otherwise
        divisor = float(cfg.normalise)
        X = X / divisor

    # Reshape to plot images to size: (N, C, H, W)
    if cfg.reshape is not None:
        # PyTorch reshape
        C, H, W = cfg.reshape
        # Size of reshape (total)
        D_expected = C * H * W
        # If reshape does not maintain total information
        if X.shape[1] != D_expected:
            raise ValueError(
                f"Cannot reshape: features D={X.shape[1]} != C*H*W={D_expected} for shape {cfg.reshape}"
            )
        X = X.view(X.shape[0], C, H, W)

    # Cache the processed tensors so subsequent runs are instant
    if cfg.save_cache and cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"X": X}
        if y is not None:
            payload["y"] = y
        torch.save(payload, cache_path)

    return X, y


# DATALOADER
# Wraps tensors into a PyTorch DataLoader (useful for batch iteration)
def make_dataloader(
    X: torch.Tensor,
    y: Optional[torch.Tensor] = None,
    *,
    # Number of samples per batch
    batch_size: int = 256,
    # Random order per epoch
    shuffle: bool = False,
    # Number of separate worker processes for loading batches in parallel (0 = load in the main process)
    num_workers: int = 0,
    # Copies tensors into pinned host memory before yielding
    pin_memory: bool = False,
) -> DataLoader:
    # If labels are missing, we build a dataset that yields only X
    if y is None:
        ds = TensorDataset(X)
    else:
        # If y is present, do consistency check: same number of samples in X and y
        assert len(X) == len(y), "X and y length mismatch."
        # Build a dataset including labels
        ds = TensorDataset(X, y)
    # Construct and return a DataLoader with the requested options
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
