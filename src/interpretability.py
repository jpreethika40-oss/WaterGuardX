"""
Phase 11 — Model Interpretability Utilities for Temporal Transformer.

Provides:
1. extract_attention_weights: Extracts self-attention matrices from TransformerEncoder layers.
2. temporal_occlusion_sensitivity: Sliding temporal mask to measure temporal importance.
3. sensor_permutation_importance: Feature sensitivity analysis by perturbing individual sensor channels.
4. compute_calibration_curve: Expected Calibration Error (ECE) and reliability diagram data.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader, Dataset


def extract_attention_weights(model: nn.Module, x: torch.Tensor, device: str = 'cpu') -> List[np.ndarray]:
    """
    Extract multi-head self-attention weight matrices for each encoder layer.

    Args:
        model: Trained SSLTransformer model.
        x: Input tensor of shape (batch_size, seq_len, n_features).
        device: 'cpu' or 'cuda'.

    Returns:
        List of numpy arrays of shape (batch_size, seq_len, seq_len), one per layer.
    """
    model.eval()
    x = x.to(device)
    layer_attns = []

    with torch.no_grad():
        # Step 1: Input projection & positional encoding
        h = model.input_proj(x)
        h = model.pos_enc(h)

        # Step 2: Layer-by-layer forward pass extracting attention weights
        for layer in model.encoder.layers:
            # Self-attention forward with need_weights=True
            attn_out, attn_weights = layer.self_attn(
                h, h, h, need_weights=True, average_attn_weights=True
            )
            # Run residual + norm + feedforward as standard in TransformerEncoderLayer
            h = layer.norm1(h + layer.dropout1(attn_out))
            ff_out = layer.linear2(layer.dropout(layer.activation(layer.linear1(h))))
            h = layer.norm2(h + layer.dropout2(ff_out))
            
            layer_attns.append(attn_weights.cpu().numpy())

    return layer_attns


def temporal_occlusion_sensitivity(
    model: nn.Module,
    x: torch.Tensor,
    window_size: int = 5,
    stride: int = 1,
    device: str = 'cpu'
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute temporal importance via sliding temporal occlusion.
    Measures the absolute change in predicted anomaly probability |P_base - P_occluded|.

    Args:
        model: Trained classifier model.
        x: Single sequence tensor of shape (1, seq_len, n_features).
        window_size: Width of temporal occlusion window.
        stride: Stride of occlusion window.
        device: 'cpu' or 'cuda'.

    Returns:
        time_steps: Array of center time steps for each window.
        delta_probs: Array of absolute probability deltas |P_base - P_occluded|.
        base_prob: Baseline predicted anomaly probability.
    """
    model.eval()
    x = x.to(device)
    seq_len = x.shape[1]

    with torch.no_grad():
        base_logit = model.forward_classify(x)
        base_prob = float(torch.sigmoid(base_logit).cpu().item())

    time_steps = []
    delta_probs = []

    for start in range(0, seq_len - window_size + 1, stride):
        end = start + window_size
        x_occ = x.clone()
        # Zero-mask the temporal window
        x_occ[:, start:end, :] = 0.0

        with torch.no_grad():
            occ_logit = model.forward_classify(x_occ)
            occ_prob = float(torch.sigmoid(occ_logit).cpu().item())

        center = (start + end - 1) / 2.0
        time_steps.append(center)
        delta_probs.append(abs(base_prob - occ_prob))

    return np.array(time_steps), np.array(delta_probs), base_prob


def sensor_permutation_importance(
    model: nn.Module,
    dataset: Dataset,
    feat_cols: List[str],
    n_samples: int = 500,
    seed: int = 42,
    device: str = 'cpu'
) -> pd.DataFrame:
    """
    Estimate sensor-level sensitivity by permuting (shuffling) individual feature columns
    across sample sequences and measuring the mean shift in anomaly prediction probability.

    Args:
        model: Trained classifier model.
        dataset: PyTorch Dataset yielding (x, y) tuples.
        feat_cols: Names of features corresponding to channels.
        n_samples: Number of sample sequences to evaluate.
        seed: Random seed.
        device: 'cpu' or 'cuda'.

    Returns:
        DataFrame with sensor, mean_abs_prob_delta, mean_signed_delta, rank.
    """
    import pandas as pd
    from sklearn.metrics import average_precision_score

    model.eval()
    rng = np.random.default_rng(seed)
    n_total = len(dataset)
    indices = rng.choice(n_total, size=min(n_samples, n_total), replace=False)

    # Collect sample batch
    samples_x, samples_y = [], []
    for idx in indices:
        x_i, y_i = dataset[idx]
        samples_x.append(x_i.numpy())
        samples_y.append(y_i.item() if hasattr(y_i, 'item') else y_i)

    X_orig = np.stack(samples_x) # (N, seq_len, n_features)
    y_orig = np.array(samples_y).astype(int)

    # Baseline probabilities
    with torch.no_grad():
        t_X = torch.tensor(X_orig, dtype=torch.float32).to(device)
        base_logits = model.forward_classify(t_X)
        base_probs = torch.sigmoid(base_logits).cpu().numpy()

    base_ap = average_precision_score(y_orig, base_probs) if len(np.unique(y_orig)) > 1 else float('nan')

    records = []
    n_features = X_orig.shape[2]

    for fi in range(n_features):
        fname = feat_cols[fi] if fi < len(feat_cols) else f"feature_{fi}"
        X_perm = X_orig.copy()
        
        # Shuffle this feature across the batch sequences
        perm_idx = rng.permutation(len(X_orig))
        X_perm[:, :, fi] = X_orig[perm_idx, :, fi]

        with torch.no_grad():
            t_perm = torch.tensor(X_perm, dtype=torch.float32).to(device)
            perm_logits = model.forward_classify(t_perm)
            perm_probs = torch.sigmoid(perm_logits).cpu().numpy()

        abs_delta = np.mean(np.abs(base_probs - perm_probs))
        signed_delta = np.mean(perm_probs - base_probs)
        perm_ap = average_precision_score(y_orig, perm_probs) if len(np.unique(y_orig)) > 1 else float('nan')
        ap_drop = (base_ap - perm_ap) if not np.isnan(base_ap) else float('nan')

        records.append({
            "feature": fname,
            "mean_abs_prob_delta": round(float(abs_delta), 5),
            "mean_signed_delta": round(float(signed_delta), 5),
            "ap_drop": round(float(ap_drop), 5) if not np.isnan(ap_drop) else 0.0,
        })

    df_imp = pd.DataFrame(records).sort_values("mean_abs_prob_delta", ascending=False).reset_index(drop=True)
    df_imp["rank"] = df_imp.index + 1
    return df_imp


def compute_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> Dict[str, any]:
    """
    Compute reliability diagram points and Expected Calibration Error (ECE).

    Args:
        y_true: Ground truth binary labels (0/1).
        y_prob: Predicted probabilities.
        n_bins: Number of confidence bins.

    Returns:
        dict containing bin_centers, bin_accuracies, bin_counts, ece.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = []
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    ece = 0.0
    total_samples = len(y_true)

    for i in range(n_bins):
        low, high = bins[i], bins[i+1]
        in_bin = (y_prob >= low) & (y_prob < high if i < n_bins - 1 else y_prob <= high)
        n_in_bin = int(np.sum(in_bin))

        if n_in_bin > 0:
            bin_acc = float(np.mean(y_true[in_bin]))
            bin_conf = float(np.mean(y_prob[in_bin]))
            ece += (n_in_bin / total_samples) * abs(bin_acc - bin_conf)
        else:
            bin_acc = float('nan')
            bin_conf = (low + high) / 2.0

        bin_centers.append((low + high) / 2.0)
        bin_accuracies.append(bin_acc)
        bin_confidences.append(bin_conf)
        bin_counts.append(n_in_bin)

    return {
        "bin_centers": bin_centers,
        "bin_accuracies": bin_accuracies,
        "bin_confidences": bin_confidences,
        "bin_counts": bin_counts,
        "ece": round(float(ece), 5)
    }
