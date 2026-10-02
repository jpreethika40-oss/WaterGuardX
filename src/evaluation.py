"""Shared evaluation utilities used across all phases."""
import numpy as np
import time
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_prob: np.ndarray = None) -> dict:
    """
    Compute the full evaluation metric set for binary classification.

    Args:
        y_true: ground-truth labels (0/1)
        y_pred: predicted labels (0/1) at chosen threshold
        y_prob: predicted probability for class 1 (optional)

    Returns:
        dict with all metrics
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr_val = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    metrics = {
        'accuracy':  float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall':    float(recall_score(y_true, y_pred, zero_division=0)),
        'f1':        float(f1_score(y_true, y_pred, zero_division=0)),
        'fpr':       float(fpr_val),
        'fnr':       float(fnr_val),
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
    }

    if y_prob is not None:
        metrics['roc_auc'] = float(roc_auc_score(y_true, y_prob))
        metrics['pr_auc']  = float(average_precision_score(y_true, y_prob))
    else:
        metrics['roc_auc'] = float('nan')
        metrics['pr_auc']  = float('nan')

    return metrics


def select_threshold_f1(y_true: np.ndarray, y_prob: np.ndarray,
                         n_thresholds: int = 200) -> float:
    """
    Select the probability threshold that maximises F1 on the provided set.
    Must be called on validation data only — never on test data.
    """
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    best_t, best_f1 = 0.5, 0.0
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        f = f1_score(y_true, pred, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return float(best_t)


def measure_inference_time(predict_fn, X: np.ndarray,
                            n_repeats: int = 3) -> dict:
    """
    Measure inference latency.

    Args:
        predict_fn: callable that takes X and returns predictions
        X:          input array
        n_repeats:  number of timing repetitions (use median)

    Returns:
        dict with total_s, per_sample_ms
    """
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        predict_fn(X)
        times.append(time.perf_counter() - t0)
    total_s = float(np.median(times))
    return {
        'total_s':       round(total_s, 6),
        'per_sample_ms': round(total_s / len(X) * 1000, 6),
        'n_samples':     len(X),
    }


def curves(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Return ROC and PR curve arrays for plotting."""
    fpr, tpr, roc_t = roc_curve(y_true, y_prob)
    prec, rec, pr_t = precision_recall_curve(y_true, y_prob)
    return {
        'roc': {'fpr': fpr, 'tpr': tpr, 'thresholds': roc_t},
        'pr':  {'precision': prec, 'recall': rec, 'thresholds': pr_t},
    }
