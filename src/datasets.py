"""PyTorch Dataset for time-series windows."""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import SEQUENCE_LENGTH, STRIDE, TARGET_COL, TIMESTAMP_COL


class WaterSensorDataset(Dataset):
    """
    Sliding-window time-series dataset.

    Each sample is a window of shape (seq_len, n_features).
    The label is the majority label within the window
    (anomaly if any step in the window is anomalous).
    """

    def __init__(self, df: pd.DataFrame, feature_cols: list,
                 seq_len: int = SEQUENCE_LENGTH,
                 stride: int = STRIDE,
                 label_strategy: str = 'last'):
        """
        Args:
            df:              preprocessed DataFrame (sorted chronologically)
            feature_cols:    list of feature column names
            seq_len:         window length in timesteps
            stride:          step between consecutive windows
            label_strategy:  'last'  → label of the last timestep in window
                             'any'   → 1 if any timestep is anomalous
                             'majority' → majority vote
        """
        self.seq_len  = seq_len
        self.stride   = stride
        self.strategy = label_strategy

        X = df[feature_cols].values.astype(np.float32)
        y = df[TARGET_COL].values.astype(np.int64)

        self.windows = []
        self.labels  = []

        for start in range(0, len(X) - seq_len + 1, stride):
            end = start + seq_len
            self.windows.append(X[start:end])
            window_labels = y[start:end]
            if label_strategy == 'last':
                lbl = int(window_labels[-1])
            elif label_strategy == 'any':
                lbl = int(window_labels.any())
            else:  # majority
                lbl = int(window_labels.mean() >= 0.5)
            self.labels.append(lbl)

        self.windows = np.array(self.windows, dtype=np.float32)
        self.labels  = np.array(self.labels,  dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (torch.tensor(self.windows[idx], dtype=torch.float32),
                torch.tensor(self.labels[idx],  dtype=torch.long))

    @property
    def n_features(self):
        return self.windows.shape[2]

    @property
    def class_weights(self):
        """Inverse-frequency class weights for imbalanced training."""
        n_total   = len(self.labels)
        n_anomaly = self.labels.sum()
        n_normal  = n_total - n_anomaly
        w_normal  = n_total / (2 * n_normal)
        w_anomaly = n_total / (2 * n_anomaly)
        return torch.tensor([w_normal, w_anomaly], dtype=torch.float32)

    def summary(self):
        n_anom = self.labels.sum()
        print(f"  Windows: {len(self.labels)}")
        print(f"  Shape:   {self.windows.shape}")
        print(f"  Normal:  {len(self.labels)-n_anom} "
              f"({(1-self.labels.mean())*100:.1f}%)")
        print(f"  Anomaly: {n_anom} ({self.labels.mean()*100:.1f}%)")


class MaskedSensorDataset(Dataset):
    """
    Dataset for self-supervised masked reconstruction pretraining.
    Returns (masked_window, original_window, mask) tuples.
    Uses only normal (label=0) samples.
    """

    def __init__(self, df: pd.DataFrame, feature_cols: list,
                 seq_len: int = SEQUENCE_LENGTH,
                 stride: int = STRIDE,
                 mask_ratio: float = 0.15):
        self.mask_ratio = mask_ratio
        self.feature_cols = feature_cols

        X = df[df[TARGET_COL] == 0][feature_cols].values.astype(np.float32)
        y = df[df[TARGET_COL] == 0][TARGET_COL].values

        self.windows = []
        for start in range(0, len(X) - seq_len + 1, stride):
            self.windows.append(X[start:start + seq_len])
        self.windows = np.array(self.windows, dtype=np.float32)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        original = self.windows[idx].copy()
        masked   = original.copy()
        mask     = np.zeros_like(original, dtype=np.float32)

        # Random masking: mask_ratio of (timestep, feature) positions
        n_mask = max(1, int(original.size * self.mask_ratio))
        flat_indices = np.random.choice(original.size, n_mask, replace=False)
        rows, cols = np.unravel_index(flat_indices, original.shape)
        masked[rows, cols] = 0.0
        mask[rows, cols]   = 1.0

        return (torch.tensor(masked,   dtype=torch.float32),
                torch.tensor(original, dtype=torch.float32),
                torch.tensor(mask,     dtype=torch.float32))


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from data_loader import load_raw
    from preprocessing import run_preprocessing

    df = load_raw()
    train, val, test, scaler, feat_cols = run_preprocessing(df, save=False)

    print("=== WaterSensorDataset ===")
    for name, split in [('Train', train), ('Val', val), ('Test', test)]:
        ds = WaterSensorDataset(split, feat_cols)
        print(f"\n{name}:")
        ds.summary()
        x, y = ds[0]
        print(f"  Sample x shape: {x.shape}, y: {y}")
        print(f"  Class weights: {ds.class_weights}")

    print("\n=== MaskedSensorDataset ===")
    mds = MaskedSensorDataset(train, feat_cols)
    print(f"  Windows: {len(mds)}")
    mx, orig, mask = mds[0]
    print(f"  Masked shape: {mx.shape}")
    print(f"  Mask ratio actual: {mask.mean():.3f}")
