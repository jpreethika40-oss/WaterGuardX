"""
Phase 7 — Distribution Shift Detection.

Method: Kolmogorov-Smirnov (KS) test per feature, aggregated as mean KS statistic.

Rationale:
- Sensor readings are continuous, univariate per channel.
- KS is non-parametric: no distributional assumption required.
- KS is interpretable: the statistic is the maximum absolute difference
  between two empirical CDFs, bounded in [0, 1].
- KS is well-suited to detecting both location and shape changes.
- Wasserstein distance was considered but requires normalisation choices;
  KS is directly comparable across features already on the same scale
  (StandardScaler applied in preprocessing).

Threshold derivation:
- Compute KS(train_bootstrap_A, train_bootstrap_B) over many bootstrap pairs
  drawn from the training set.
- Use the 99th percentile of those bootstrap KS values as the threshold.
- This gives a data-driven threshold: any KS score above it is unlikely
  to arise from random sampling of the same distribution.

Usage:
    detector = DriftDetector()
    detector.fit(train_df, feature_cols)          # build reference
    result   = detector.detect(incoming_df)        # test incoming data
    print(result['shift_detected'], result['shift_score'])
"""
import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Optional


class DriftDetector:
    """
    KS-based distribution shift detector.

    Attributes:
        reference_stats : dict  — per-feature statistics from training data
        threshold       : float — KS score above which shift is declared
        feature_cols    : list  — feature names used
    """

    def __init__(self, n_bootstrap: int = 500, bootstrap_frac: float = 0.3,
                 percentile: float = 99.0, seed: int = 42):
        """
        Args:
            n_bootstrap    : number of bootstrap pairs for threshold estimation
            bootstrap_frac : fraction of training data per bootstrap sample
            percentile     : percentile of bootstrap distribution used as threshold
            seed           : random seed for reproducibility
        """
        self.n_bootstrap    = n_bootstrap
        self.bootstrap_frac = bootstrap_frac
        self.percentile     = percentile
        self.seed           = seed

        self.reference_data  = None   # stored as numpy array (n_rows, n_features)
        self.feature_cols    = None
        self.threshold       = None
        self.reference_stats = None   # per-feature summary statistics

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, train_df: pd.DataFrame, feature_cols: List[str]) -> 'DriftDetector':
        """
        Build reference distribution from training data.

        Args:
            train_df     : training DataFrame (already scaled)
            feature_cols : list of feature column names to monitor
        """
        self.feature_cols   = list(feature_cols)
        self.reference_data = train_df[feature_cols].values.astype(np.float32)

        # Per-feature summary statistics
        self.reference_stats = {}
        for i, col in enumerate(feature_cols):
            vals = self.reference_data[:, i]
            self.reference_stats[col] = {
                'mean':   float(np.mean(vals)),
                'std':    float(np.std(vals)),
                'median': float(np.median(vals)),
                'min':    float(np.min(vals)),
                'max':    float(np.max(vals)),
                'p5':     float(np.percentile(vals, 5)),
                'p25':    float(np.percentile(vals, 25)),
                'p75':    float(np.percentile(vals, 75)),
                'p95':    float(np.percentile(vals, 95)),
                'missing_pct': float(np.isnan(vals).mean() * 100),
            }

        # Bootstrap threshold
        self.threshold = self._bootstrap_threshold()
        return self

    def _bootstrap_threshold(self) -> float:
        """
        Estimate threshold as the 99th percentile of KS scores computed
        between pairs of bootstrap samples drawn from the reference data.
        """
        rng  = np.random.default_rng(self.seed)
        n    = len(self.reference_data)
        k    = max(1, int(n * self.bootstrap_frac))
        scores = []

        for _ in range(self.n_bootstrap):
            idx_a = rng.choice(n, size=k, replace=False)
            idx_b = rng.choice(n, size=k, replace=False)
            a = self.reference_data[idx_a]
            b = self.reference_data[idx_b]
            ks_vals = [
                stats.ks_2samp(a[:, fi], b[:, fi]).statistic
                for fi in range(len(self.feature_cols))
            ]
            scores.append(float(np.mean(ks_vals)))

        return float(np.percentile(scores, self.percentile))

    # ── Detect ────────────────────────────────────────────────────────────────

    def detect(self, incoming_df: pd.DataFrame,
               feature_cols: Optional[List[str]] = None) -> dict:
        """
        Test whether incoming data has shifted from the reference distribution.

        Args:
            incoming_df  : new data DataFrame (same scaling as training)
            feature_cols : override feature list (must match fit columns if provided)

        Returns:
            dict with keys:
                shift_score     : mean KS statistic across features
                threshold       : detection threshold
                shift_detected  : bool
                per_feature     : dict of per-feature KS statistics and p-values
        """
        if self.reference_data is None:
            raise RuntimeError("Call fit() before detect().")

        cols = feature_cols if feature_cols is not None else self.feature_cols
        incoming = incoming_df[cols].values.astype(np.float32)

        per_feature = {}
        ks_vals     = []
        for fi, col in enumerate(cols):
            ks_stat, p_val = stats.ks_2samp(
                self.reference_data[:, fi], incoming[:, fi])
            per_feature[col] = {
                'ks_statistic': round(float(ks_stat), 6),
                'p_value':      round(float(p_val),   8),
                'shift_detected': bool(ks_stat > self.threshold),
            }
            ks_vals.append(float(ks_stat))

        shift_score = float(np.mean(ks_vals))
        return {
            'shift_score':    round(shift_score, 6),
            'threshold':      round(self.threshold, 6),
            'shift_detected': bool(shift_score > self.threshold),
            'per_feature':    per_feature,
            'n_features_shifted': sum(
                v['shift_detected'] for v in per_feature.values()),
            'n_features_total': len(cols),
        }

    # ── Utilities ─────────────────────────────────────────────────────────────

    def reference_stats_df(self) -> pd.DataFrame:
        """Return reference statistics as a tidy DataFrame."""
        rows = []
        for col, s in self.reference_stats.items():
            rows.append({'feature': col, **s})
        return pd.DataFrame(rows)

    def per_feature_df(self, detect_result: dict) -> pd.DataFrame:
        """Convert per_feature dict from detect() to a sorted DataFrame."""
        rows = []
        for col, v in detect_result['per_feature'].items():
            rows.append({'feature': col, **v})
        df = pd.DataFrame(rows).sort_values('ks_statistic', ascending=False)
        return df.reset_index(drop=True)

    def detect_rolling(self, df: pd.DataFrame, window_size: int = 2016,
                       step_size: int = 288, timestamp_col: str = 'datetime',
                       feature_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compute rolling window distribution shift over time.

        Args:
            df            : DataFrame with timestamp and features
            window_size   : number of timesteps per window (e.g., 2016 = 7 days @ 5-min)
            step_size     : stride between windows (e.g., 288 = 1 day @ 5-min)
            timestamp_col : column name of timestamp
            feature_cols  : features to evaluate

        Returns:
            DataFrame with window timestamps, shift scores, and detection flags.
        """
        cols = feature_cols if feature_cols is not None else self.feature_cols
        records = []
        n_rows = len(df)
        
        if n_rows < window_size:
            # If dataset is smaller than window_size, evaluate the entire df once
            res = self.detect(df, feature_cols=cols)
            records.append({
                'window_start': df[timestamp_col].iloc[0] if timestamp_col in df.columns else 0,
                'window_end':   df[timestamp_col].iloc[-1] if timestamp_col in df.columns else n_rows,
                'timestamp':    df[timestamp_col].iloc[-1] if timestamp_col in df.columns else n_rows,
                'shift_score':  res['shift_score'],
                'threshold':    res['threshold'],
                'shift_detected': res['shift_detected'],
                'n_features_shifted': res['n_features_shifted'],
            })
            return pd.DataFrame(records)

        for start_idx in range(0, n_rows - window_size + 1, step_size):
            end_idx = start_idx + window_size
            window_df = df.iloc[start_idx:end_idx]
            res = self.detect(window_df, feature_cols=cols)
            records.append({
                'window_start': window_df[timestamp_col].iloc[0] if timestamp_col in df.columns else start_idx,
                'window_end':   window_df[timestamp_col].iloc[-1] if timestamp_col in df.columns else end_idx,
                'timestamp':    window_df[timestamp_col].iloc[-1] if timestamp_col in df.columns else end_idx,
                'shift_score':  res['shift_score'],
                'threshold':    res['threshold'],
                'shift_detected': res['shift_detected'],
                'n_features_shifted': res['n_features_shifted'],
            })

        return pd.DataFrame(records)
