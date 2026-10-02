"""
Classical ML baseline models for water-system anomaly detection.

Representation choice
---------------------
Classical models receive a per-timestep tabular feature vector, NOT a raw
3D window tensor.  For each timestep t we use:

  1. Current scaled sensor values (12 features)          — same as deep models
  2. Cyclical temporal features (4 features)             — same as deep models
  3. Rolling statistics over the past 48 steps (4 hours)
       - rolling mean  per sensor (12 features)
       - rolling std   per sensor (12 features)
       - rolling range per sensor (12 features)          — 36 features
  4. First difference (rate of change) per sensor (12)   — 12 features

Total: 64 features per timestep.

Rationale: this gives classical models access to the same temporal context
that the Transformer sees via its 48-step window, but in a tabular form that
is natural for tree-based and linear models.  Rolling statistics are computed
strictly from past data (no look-ahead), so there is no leakage.
"""
import numpy as np
import pandas as pd
import joblib
import json
import time
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import xgboost as xgb

from config import SENSOR_COLS, TARGET_COL, TIMESTAMP_COL, SEED, MODELS_DIR


# ── Tabular feature engineering ───────────────────────────────────────────────

ROLL_W = 48   # 4-hour rolling window (matches Transformer sequence length)

def build_tabular_features(df: pd.DataFrame,
                            feature_cols: list) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a 2-D tabular feature matrix from the preprocessed DataFrame.

    Rolling statistics are computed with min_periods=1 so the first rows
    are not dropped.  All operations are strictly backward-looking.

    Returns:
        X: (n_rows, n_features) float32 array
        y: (n_rows,) int array
    """
    out = df[feature_cols].copy()   # scaled sensors + temporal features

    # Rolling mean, std, range per sensor (backward-looking)
    for col in SENSOR_COLS:
        roll = df[col].rolling(ROLL_W, min_periods=1)
        out[f'{col}_rmean'] = roll.mean()
        out[f'{col}_rstd']  = roll.std().fillna(0)
        out[f'{col}_rng']   = roll.max() - roll.min()

    # First difference (rate of change)
    for col in SENSOR_COLS:
        out[f'{col}_diff'] = df[col].diff().fillna(0)

    X = out.values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.int32)
    return X, y


def feature_names(base_feature_cols: list) -> list:
    names = list(base_feature_cols)
    for col in SENSOR_COLS:
        names += [f'{col}_rmean', f'{col}_rstd', f'{col}_rng']
    for col in SENSOR_COLS:
        names += [f'{col}_diff']
    return names


# ── Threshold selection ───────────────────────────────────────────────────────

def select_threshold(y_val: np.ndarray, prob_val: np.ndarray,
                     n_steps: int = 200) -> float:
    """Return threshold maximising F1 on validation set."""
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.01, 0.99, n_steps):
        pred = (prob_val >= t).astype(int)
        f = f1_score(y_val, pred, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, float(t)
    return best_t


# ── Model builders ────────────────────────────────────────────────────────────

def build_logistic_regression() -> LogisticRegression:
    return LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight='balanced',
        solver='lbfgs',
        random_state=SEED,
        n_jobs=-1,
    )


def build_random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        class_weight='balanced',
        random_state=SEED,
        n_jobs=-1,
    )


def build_xgboost(scale_pos_weight: float = 1.0) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=SEED,
        n_jobs=-1,
        verbosity=0,
    )


# ── Train / predict helpers ───────────────────────────────────────────────────

def train_model(model, X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray = None, y_val: np.ndarray = None):
    """Fit model; XGBoost uses early stopping if val data provided."""
    t0 = time.perf_counter()
    if isinstance(model, xgb.XGBClassifier) and X_val is not None:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
    else:
        model.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0
    return model, elapsed


def predict_proba(model, X: np.ndarray) -> np.ndarray:
    """Return probability of class 1."""
    return model.predict_proba(X)[:, 1]


# ── Save / load ───────────────────────────────────────────────────────────────

def save_model(model, name: str, config: dict, threshold: float):
    """Save model file + config JSON."""
    dest = MODELS_DIR / 'baselines' / name
    dest.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, dest / 'model.pkl')
    config['threshold'] = threshold
    with open(dest / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)


def load_model(name: str):
    """Load saved model and its config."""
    dest = MODELS_DIR / 'baselines' / name
    model = joblib.load(dest / 'model.pkl')
    with open(dest / 'config.json') as f:
        config = json.load(f)
    return model, config
