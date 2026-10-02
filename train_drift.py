"""
Phase 7 — Distribution Shift Detection and Robustness Evaluation.
Run from project root: python train_drift.py

Natural shift used:
  Training data  : Jan–Aug 2018  (summer/spring operating regime)
  Test data      : Nov–Dec 2018  (winter operating regime)
  The KS test confirms a measurable natural seasonal shift (KS ≈ 0.06–0.12)
  across all 12 sensor channels, roughly 2× the train→val baseline.

Synthetic severity experiment:
  Applied on top of the natural test data to study graded shift magnitudes.
  Transformation: additive mean shift on sensor columns only (not temporal
  features), scaled by multiples of the per-sensor training standard deviation.
  This represents a plausible sensor-calibration drift or operating-point change.
"""
import sys, json, time
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from data_loader import load_raw
from preprocessing import run_preprocessing
from datasets import WaterSensorDataset
from transformer import TemporalTransformer
from self_supervised import SSLTransformer
from lstm import LSTMClassifier
from drift_detection import DriftDetector
from evaluation import compute_metrics, select_threshold_f1
from training import set_seed
from config import (SEED, SEQUENCE_LENGTH, SENSOR_COLS,
                    MODELS_DIR, RESULTS_DIR, FIGURES_DIR, DATA_PROC)

set_seed(SEED)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
STRIDE = 12
BATCH  = 256
print(f"Device: {DEVICE}")

DRIFT_RES  = RESULTS_DIR / 'drift';  DRIFT_RES.mkdir(parents=True, exist_ok=True)
DRIFT_FIGS = FIGURES_DIR / 'drift';  DRIFT_FIGS.mkdir(parents=True, exist_ok=True)
EXP_DIR    = RESULTS_DIR / 'experiments'; EXP_DIR.mkdir(parents=True, exist_ok=True)
SHIFTED_DIR = DATA_PROC / 'shifted'

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("\nLoading data...")
df = load_raw()
train_df, val_df, test_df, scaler, feat_cols = run_preprocessing(df, save=False)
N_FEATURES = len(feat_cols)
print(f"Features: {N_FEATURES}  |  Sensor cols: {len(SENSOR_COLS)}")
print(f"Train: {len(train_df):,} rows  |  Val: {len(val_df):,}  |  Test: {len(test_df):,}")

# ── 2. Build reference distribution from training data ────────────────────────
print("\n" + "="*60)
print("REFERENCE DISTRIBUTION (training data only)")
print("="*60)

detector = DriftDetector(n_bootstrap=500, bootstrap_frac=0.3,
                         percentile=99.0, seed=SEED)
detector.fit(train_df, feat_cols)

ref_df = detector.reference_stats_df()
ref_df.to_csv(DRIFT_RES / 'reference_distribution.csv', index=False)
ref_df.to_csv(RESULTS_DIR / 'reference_distribution.csv', index=False)

# Also save JSON format for easy loading in production / deployment
with open(DRIFT_RES / 'reference_distribution.json', 'w') as f:
    json.dump(detector.reference_stats, f, indent=2)
with open(RESULTS_DIR / 'reference_distribution.json', 'w') as f:
    json.dump(detector.reference_stats, f, indent=2)

print(f"Bootstrap threshold (99th pct): {detector.threshold:.6f}")
print("\nReference stats (first 6 features):")
print(ref_df[['feature','mean','std','p25','p75']].head(6).to_string(index=False))

# Save detector config
det_cfg = {
    'method':          'KolmogorovSmirnov',
    'aggregation':     'mean_across_features',
    'n_bootstrap':     detector.n_bootstrap,
    'bootstrap_frac':  detector.bootstrap_frac,
    'percentile':      detector.percentile,
    'threshold':       round(detector.threshold, 6),
    'feature_cols':    feat_cols,
    'sensor_cols':     SENSOR_COLS,
    'reference_rows':  len(train_df),
    'seed':            SEED,
}
with open(DRIFT_RES / 'detector_config.json', 'w') as f:
    json.dump(det_cfg, f, indent=2)

# ── 3. Detect shift: train→val (baseline) and train→test (natural shift) ──────
print("\n" + "="*60)
print("SHIFT DETECTION: NATURAL CONDITIONS")
print("="*60)

result_val  = detector.detect(val_df)
result_test = detector.detect(test_df)

print(f"\nTrain -> Val  (same season, expected stable):")
print(f"  Shift score : {result_val['shift_score']:.6f}")
print(f"  Threshold   : {result_val['threshold']:.6f}")
print(f"  Status      : {'SHIFT DETECTED' if result_val['shift_detected'] else 'STABLE'}")
print(f"  Features shifted: {result_val['n_features_shifted']}/{result_val['n_features_total']}")

print(f"\nTrain -> Test (Nov-Dec vs Jan-Aug, seasonal shift):")
print(f"  Shift score : {result_test['shift_score']:.6f}")
print(f"  Threshold   : {result_test['threshold']:.6f}")
print(f"  Status      : {'SHIFT DETECTED' if result_test['shift_detected'] else 'STABLE'}")
print(f"  Features shifted: {result_test['n_features_shifted']}/{result_test['n_features_total']}")

# Per-feature table
pf_val  = detector.per_feature_df(result_val)
pf_test = detector.per_feature_df(result_test)
pf_val.to_csv(DRIFT_RES  / 'per_feature_ks_val.csv',  index=False)
pf_test.to_csv(DRIFT_RES / 'per_feature_ks_test.csv', index=False)

print("\nPer-feature KS (train->test, top 6):")
print(pf_test[['feature','ks_statistic','p_value','shift_detected']].head(6).to_string(index=False))

# ── 4. Synthetic shift generation ─────────────────────────────────────────────
print("\n" + "="*60)
print("SYNTHETIC SHIFT GENERATION")
print("="*60)
print("Transformation: additive mean shift on sensor columns only.")
print("Magnitude: k × per-sensor training std  (k = 0, 0.3, 0.6, 1.0)")
print("Rationale: represents sensor calibration drift or operating-point change.")
print("Applied to the natural test set to create graded severity levels.")

# Per-sensor training std (in scaled space, std≈1 by construction)
train_stds = {col: float(train_df[col].std()) for col in SENSOR_COLS}

SHIFT_LEVELS = {
    'none':     0.0,
    'mild':     0.3,
    'moderate': 0.6,
    'strong':   1.0,
}

shifted_dfs = {}
shift_configs = {}

for level_name, k in SHIFT_LEVELS.items():
    shifted = test_df.copy()
    if k > 0:
        for col in SENSOR_COLS:
            shifted[col] = shifted[col] + k * train_stds[col]
    shifted_dfs[level_name] = shifted

    # Save shifted data
    save_dir = SHIFTED_DIR / level_name
    save_dir.mkdir(parents=True, exist_ok=True)
    shifted.to_csv(save_dir / 'test_shifted.csv', index=False)

    cfg = {
        'level': level_name,
        'shift_multiplier': k,
        'affected_cols': SENSOR_COLS,
        'transformation': 'additive_mean_shift',
        'magnitude_description': f'{k} × per-sensor training std',
        'synthetic': k > 0,
    }
    shift_configs[level_name] = cfg
    with open(save_dir / 'shift_config.json', 'w') as f:
        json.dump(cfg, f, indent=2)
    print(f"  {level_name:10s} (k={k:.1f}): saved to data/processed/shifted/{level_name}/")

# Detect shift for each synthetic level
print("\nShift detection per severity level:")
shift_scores = {}
for level_name, shifted in shifted_dfs.items():
    res = detector.detect(shifted)
    shift_scores[level_name] = res
    status = 'SHIFT DETECTED' if res['shift_detected'] else 'STABLE'
    print(f"  {level_name:10s}  score={res['shift_score']:.6f}  "
          f"threshold={res['threshold']:.6f}  {status}  "
          f"({res['n_features_shifted']}/{res['n_features_total']} features)")

# ── 5. Load trained models ────────────────────────────────────────────────────
print("\n" + "="*60)
print("LOADING TRAINED MODELS")
print("="*60)

# SSL Transformer (Phase 6) — primary model
ssl_cfg = json.load(open(MODELS_DIR / 'ssl' / 'ssl_config.json'))
ssl_model = SSLTransformer(
    n_features=ssl_cfg['n_features'],
    d_model=ssl_cfg['d_model'], nhead=ssl_cfg['nhead'],
    num_layers=ssl_cfg['num_layers'],
    dim_feedforward=ssl_cfg['dim_feedforward'],
    dropout=ssl_cfg['dropout'],
    max_seq_len=SEQUENCE_LENGTH + 10,
).to(DEVICE)
ssl_model.load_state_dict(
    torch.load(MODELS_DIR / 'ssl' / 'ssl_finetuned.pt', map_location=DEVICE))
ssl_model.eval()
SSL_THRESHOLD = ssl_cfg.get('threshold', 0.0297)
# Re-read threshold from results
ssl_metrics_df = pd.read_csv(RESULTS_DIR / 'ssl' / 'final_metrics.csv')
SSL_THRESHOLD  = float(ssl_metrics_df['threshold'].iloc[0])
print(f"SSL Transformer loaded  | params={ssl_model.n_params:,} | threshold={SSL_THRESHOLD:.4f}")

# Standard Transformer (Phase 5)
tr_cfg = json.load(open(MODELS_DIR / 'transformer' / 'config.json'))
std_model = TemporalTransformer(
    n_features=tr_cfg['n_features'],
    d_model=tr_cfg['d_model'], nhead=tr_cfg['nhead'],
    num_layers=tr_cfg['num_layers'],
    dim_feedforward=tr_cfg['dim_feedforward'],
    dropout=tr_cfg['dropout'],
    max_seq_len=SEQUENCE_LENGTH + 10,
).to(DEVICE)
std_model.load_state_dict(
    torch.load(MODELS_DIR / 'transformer' / 'best_transformer.pt', map_location=DEVICE))
std_model.eval()
STD_THRESHOLD = float(tr_cfg['threshold'])
print(f"Standard Transformer loaded | params={std_model.n_params:,} | threshold={STD_THRESHOLD:.4f}")

# LSTM (Phase 4)
lstm_cfg = json.load(open(MODELS_DIR / 'lstm' / 'config.json'))
lstm_model = LSTMClassifier(
    input_size=lstm_cfg['input_size'],
    hidden_size=lstm_cfg['hidden_size'],
    num_layers=lstm_cfg['num_layers'],
    dropout=lstm_cfg['dropout'],
    bidirectional=lstm_cfg['bidirectional'],
).to(DEVICE)
lstm_model.load_state_dict(
    torch.load(MODELS_DIR / 'lstm' / 'best_lstm.pt', map_location=DEVICE))
lstm_model.eval()
lstm_metrics_df = pd.read_csv(RESULTS_DIR / 'lstm' / 'final_metrics.csv')
LSTM_THRESHOLD  = float(lstm_metrics_df['threshold'].iloc[0])
print(f"LSTM loaded             | params={lstm_model.n_params:,} | threshold={LSTM_THRESHOLD:.4f}")


def evaluate_on_data(model, df_split, threshold, model_type='ssl'):
    """Run model on a DataFrame split, return metrics dict."""
    ds     = WaterSensorDataset(df_split, feat_cols, label_strategy='last', stride=STRIDE)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=0)
    all_probs, all_labels = [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            X_b = X_b.to(DEVICE)
            if model_type == 'ssl':
                logits = model.forward_classify(X_b)
            else:
                logits = model(X_b)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(y_b.numpy())
    probs  = np.concatenate(all_probs)
    labels = np.concatenate(all_labels).astype(int)
    preds  = (probs >= threshold).astype(int)
    return compute_metrics(labels, preds, probs), probs, labels

# ── 6. Robustness evaluation: normal vs natural shift ─────────────────────────
print("\n" + "="*60)
print("ROBUSTNESS EVALUATION: NORMAL vs NATURAL SHIFT (SSL Transformer)")
print("="*60)

# Condition A: original test (natural shift already present)
met_normal, p_normal, y_normal = evaluate_on_data(
    ssl_model, test_df, SSL_THRESHOLD, model_type='ssl')
print("\nCondition A — Original test (Nov–Dec, natural seasonal shift):")
print(f"  F1={met_normal['f1']:.4f}  Precision={met_normal['precision']:.4f}  "
      f"Recall={met_normal['recall']:.4f}")
print(f"  ROC-AUC={met_normal['roc_auc']:.4f}  PR-AUC={met_normal['pr_auc']:.4f}  "
      f"FPR={met_normal['fpr']:.4f}  FNR={met_normal['fnr']:.4f}")

# Condition B: strong synthetic shift
met_strong, p_strong, y_strong = evaluate_on_data(
    ssl_model, shifted_dfs['strong'], SSL_THRESHOLD, model_type='ssl')
print("\nCondition B — Strong synthetic shift (k=1.0 × std):")
print(f"  F1={met_strong['f1']:.4f}  Precision={met_strong['precision']:.4f}  "
      f"Recall={met_strong['recall']:.4f}")
print(f"  ROC-AUC={met_strong['roc_auc']:.4f}  PR-AUC={met_strong['pr_auc']:.4f}  "
      f"FPR={met_strong['fpr']:.4f}  FNR={met_strong['fnr']:.4f}")

# Save explicit Condition A vs Condition B comparison table
rob_df = pd.DataFrame([
    {
        'Condition': 'Condition A (Normal Test)',
        'Accuracy': round(met_normal['accuracy'], 4),
        'Precision': round(met_normal['precision'], 4),
        'Recall': round(met_normal['recall'], 4),
        'F1': round(met_normal['f1'], 4),
        'ROC-AUC': round(met_normal['roc_auc'], 4),
        'PR-AUC': round(met_normal['pr_auc'], 4),
        'FPR': round(met_normal['fpr'], 4),
        'FNR': round(met_normal['fnr'], 4),
    },
    {
        'Condition': 'Condition B (Shifted Test k=1.0)',
        'Accuracy': round(met_strong['accuracy'], 4),
        'Precision': round(met_strong['precision'], 4),
        'Recall': round(met_strong['recall'], 4),
        'F1': round(met_strong['f1'], 4),
        'ROC-AUC': round(met_strong['roc_auc'], 4),
        'PR-AUC': round(met_strong['pr_auc'], 4),
        'FPR': round(met_strong['fpr'], 4),
        'FNR': round(met_strong['fnr'], 4),
    },
    {
        'Condition': 'Degradation (Delta B - A)',
        'Accuracy': round(met_strong['accuracy'] - met_normal['accuracy'], 4),
        'Precision': round(met_strong['precision'] - met_normal['precision'], 4),
        'Recall': round(met_strong['recall'] - met_normal['recall'], 4),
        'F1': round(met_strong['f1'] - met_normal['f1'], 4),
        'ROC-AUC': round(met_strong['roc_auc'] - met_normal['roc_auc'], 4),
        'PR-AUC': round(met_strong['pr_auc'] - met_normal['pr_auc'], 4),
        'FPR': round(met_strong['fpr'] - met_normal['fpr'], 4),
        'FNR': round(met_strong['fnr'] - met_normal['fnr'], 4),
    }
])
rob_df.to_csv(DRIFT_RES / 'robustness_comparison.csv', index=False)

# ── 7. Severity experiment: all 4 levels ─────────────────────────────────────
print("\n" + "="*60)
print("SEVERITY EXPERIMENT (SSL Transformer, all shift levels)")
print("="*60)

severity_rows = []
for level_name, k in SHIFT_LEVELS.items():
    shifted = shifted_dfs[level_name]
    res_det = shift_scores[level_name]
    met, probs, labels = evaluate_on_data(
        ssl_model, shifted, SSL_THRESHOLD, model_type='ssl')
    row = {
        'level':          level_name,
        'shift_k':        k,
        'shift_score':    res_det['shift_score'],
        'shift_detected': res_det['shift_detected'],
        'n_features_shifted': res_det['n_features_shifted'],
        'accuracy':       round(met['accuracy'],  4),
        'precision':      round(met['precision'], 4),
        'recall':         round(met['recall'],    4),
        'f1':             round(met['f1'],        4),
        'roc_auc':        round(met['roc_auc'],   4),
        'pr_auc':         round(met['pr_auc'],    4),
        'fpr':            round(met['fpr'],        4),
        'fnr':            round(met['fnr'],        4),
        'tp': met['tp'], 'tn': met['tn'], 'fp': met['fp'], 'fn': met['fn'],
    }
    severity_rows.append(row)
    print(f"  {level_name:10s} k={k:.1f}  shift={res_det['shift_score']:.4f}  "
          f"F1={met['f1']:.4f}  Recall={met['recall']:.4f}  "
          f"FPR={met['fpr']:.4f}  FNR={met['fnr']:.4f}")

sev_df = pd.DataFrame(severity_rows)
sev_df.to_csv(DRIFT_RES / 'severity_experiment.csv', index=False)

# ── 8. Model comparison under shift ──────────────────────────────────────────
print("\n" + "="*60)
print("MODEL COMPARISON UNDER SHIFT (strong shift, k=1.0)")
print("="*60)

strong_shifted = shifted_dfs['strong']
comp_rows = []

for model_name, model, threshold, mtype in [
    ('SSL_Transformer',      ssl_model,  SSL_THRESHOLD,  'ssl'),
    ('Standard_Transformer', std_model,  STD_THRESHOLD,  'std'),
    ('LSTM',                 lstm_model, LSTM_THRESHOLD, 'std'),
]:
    # Normal test
    m_norm, _, _ = evaluate_on_data(model, test_df,      threshold, mtype)
    # Shifted test
    m_shft, _, _ = evaluate_on_data(model, strong_shifted, threshold, mtype)

    comp_rows.append({
        'model':          model_name,
        'condition':      'normal',
        'f1':             round(m_norm['f1'],        4),
        'precision':      round(m_norm['precision'], 4),
        'recall':         round(m_norm['recall'],    4),
        'roc_auc':        round(m_norm['roc_auc'],   4),
        'pr_auc':         round(m_norm['pr_auc'],    4),
        'fpr':            round(m_norm['fpr'],        4),
        'fnr':            round(m_norm['fnr'],        4),
    })
    comp_rows.append({
        'model':          model_name,
        'condition':      'shifted',
        'f1':             round(m_shft['f1'],        4),
        'precision':      round(m_shft['precision'], 4),
        'recall':         round(m_shft['recall'],    4),
        'roc_auc':        round(m_shft['roc_auc'],   4),
        'pr_auc':         round(m_shft['pr_auc'],    4),
        'fpr':            round(m_shft['fpr'],        4),
        'fnr':            round(m_shft['fnr'],        4),
    })
    delta_f1 = m_shft['f1'] - m_norm['f1']
    print(f"  {model_name:25s}  "
          f"normal F1={m_norm['f1']:.4f}  shifted F1={m_shft['f1']:.4f}  "
          f"Delta F1={delta_f1:+.4f}")

comp_df = pd.DataFrame(comp_rows)
comp_df.to_csv(DRIFT_RES / 'model_comparison_under_shift.csv', index=False)

# ── 9. Error analysis under shift ─────────────────────────────────────────────
print("\n" + "="*60)
print("ERROR ANALYSIS UNDER SHIFT")
print("="*60)

met_norm_ssl, p_norm, y_norm = evaluate_on_data(
    ssl_model, test_df, SSL_THRESHOLD, 'ssl')
met_shft_ssl, p_shft, y_shft = evaluate_on_data(
    ssl_model, shifted_dfs['strong'], SSL_THRESHOLD, 'ssl')

pred_norm = (p_norm >= SSL_THRESHOLD).astype(int)
pred_shft = (p_shft >= SSL_THRESHOLD).astype(int)

fp_norm = np.where((pred_norm==1) & (y_norm==0))[0]
fn_norm = np.where((pred_norm==0) & (y_norm==1))[0]
fp_shft = np.where((pred_shft==1) & (y_shft==0))[0]
fn_shft = np.where((pred_shft==0) & (y_shft==1))[0]

print(f"Normal  condition: FP={len(fp_norm)}  FN={len(fn_norm)}")
print(f"Shifted condition: FP={len(fp_shft)}  FN={len(fn_shft)}")

err_rows = []
for idx in fp_norm[:5]:
    err_rows.append({'condition':'normal','error_type':'FP','window_idx':int(idx),
                     'prob':round(float(p_norm[idx]),4)})
for idx in fn_norm[:5]:
    err_rows.append({'condition':'normal','error_type':'FN','window_idx':int(idx),
                     'prob':round(float(p_norm[idx]),4)})
for idx in fp_shft[:5]:
    err_rows.append({'condition':'shifted','error_type':'FP','window_idx':int(idx),
                     'prob':round(float(p_shft[idx]),4)})
for idx in fn_shft[:5]:
    err_rows.append({'condition':'shifted','error_type':'FN','window_idx':int(idx),
                     'prob':round(float(p_shft[idx]),4)})
pd.DataFrame(err_rows).to_csv(DRIFT_RES / 'error_analysis.csv', index=False)

# ── 10. Visualizations ────────────────────────────────────────────────────────
print("\nGenerating figures...")

# Fig 1: Reference vs test distribution for 4 representative sensors
SHOW_SENSORS = ['n1', 'p227', 'T1', 'PUMP_1']
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, col in zip(axes.flat, SHOW_SENSORS):
    ax.hist(train_df[col].values, bins=60, alpha=0.6,
            color='#4C72B0', label='Train (reference)', density=True)
    ax.hist(test_df[col].values,  bins=60, alpha=0.6,
            color='#DD8452', label='Test (natural shift)', density=True)
    ks = result_test['per_feature'][col]['ks_statistic']
    ax.set_title(f'{col}  KS={ks:.4f}', fontsize=10)
    ax.set_xlabel('Scaled value'); ax.set_ylabel('Density')
    ax.legend(fontsize=8)
fig.suptitle('Reference vs Natural Shift Distribution (4 sensors)', fontsize=12)
fig.tight_layout()
fig.savefig(DRIFT_FIGS / 'distribution_comparison.png', dpi=120, bbox_inches='tight')
plt.close(fig)

# Fig 2: Per-feature KS score bar chart (train→test)
fig, ax = plt.subplots(figsize=(12, 4))
colors = ['#C44E52' if v['shift_detected'] else '#4C72B0'
          for v in result_test['per_feature'].values()]
ax.bar(pf_test['feature'], pf_test['ks_statistic'], color=colors)
ax.axhline(detector.threshold, color='black', ls='--', lw=1.5,
           label=f'Threshold={detector.threshold:.4f}')
ax.set_ylabel('KS Statistic'); ax.set_title('Per-Feature KS Score (Train → Test)')
ax.tick_params(axis='x', rotation=45); ax.legend()
fig.tight_layout()
fig.savefig(DRIFT_FIGS / 'per_feature_ks_score.png', dpi=120, bbox_inches='tight')
plt.close(fig)

# Fig 3: Severity experiment — F1, Recall, FPR vs shift level
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
x_labels = [f"{r['level']}\n(k={r['shift_k']})" for _, r in sev_df.iterrows()]
for ax, metric, color in zip(axes,
                              ['f1', 'recall', 'fpr'],
                              ['#4C72B0', '#55A868', '#C44E52']):
    ax.plot(range(len(sev_df)), sev_df[metric], marker='o', color=color, lw=2)
    for i, v in enumerate(sev_df[metric]):
        ax.text(i, v + 0.005, f'{v:.4f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(range(len(sev_df))); ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_ylabel(metric.upper()); ax.set_title(f'SSL Transformer — {metric.upper()} vs Shift Severity')
    ax.set_ylim(0, 1.1)
fig.suptitle('Performance Degradation vs Shift Severity', fontsize=11)
fig.tight_layout()
fig.savefig(DRIFT_FIGS / 'severity_performance.png', dpi=120, bbox_inches='tight')
plt.close(fig)

# Fig 4: Model comparison under shift (grouped bar)
models_list = ['SSL_Transformer', 'Standard_Transformer', 'LSTM']
metrics_cmp = ['f1', 'precision', 'recall', 'roc_auc']
x = np.arange(len(models_list))
width = 0.2
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, condition, title in zip(axes, ['normal', 'shifted'],
                                 ['Normal Condition', 'Shifted Condition (k=1.0)']):
    sub = comp_df[comp_df['condition'] == condition].set_index('model')
    for i, metric in enumerate(metrics_cmp):
        vals = [sub.loc[m, metric] if m in sub.index else 0 for m in models_list]
        bars = ax.bar(x + i*width, vals, width,
                      label=metric.upper(),
                      color=plt.cm.Set2(i/len(metrics_cmp)))
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=7)
    ax.set_xticks(x + width*(len(metrics_cmp)-1)/2)
    ax.set_xticklabels(models_list, fontsize=9, rotation=10)
    ax.set_ylim(0, 1.2); ax.set_ylabel('Score')
    ax.set_title(title); ax.legend(fontsize=8)
fig.suptitle('Model Comparison: Normal vs Shifted', fontsize=11)
fig.tight_layout()
fig.savefig(DRIFT_FIGS / 'model_comparison_shift.png', dpi=120, bbox_inches='tight')
plt.close(fig)

# Fig 5: Shift score vs severity level
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(sev_df['shift_k'], sev_df['shift_score'], marker='o',
        color='#4C72B0', lw=2, label='Shift score')
ax.axhline(detector.threshold, color='red', ls='--', lw=1.5,
           label=f'Threshold={detector.threshold:.4f}')
for _, row in sev_df.iterrows():
    ax.text(row['shift_k'], row['shift_score']+0.002,
            f"{row['shift_score']:.4f}", ha='center', fontsize=9)
ax.set_xlabel('Shift multiplier k'); ax.set_ylabel('Mean KS score')
ax.set_title('Shift Score vs Severity Level')
ax.legend(); fig.tight_layout()
fig.savefig(DRIFT_FIGS / 'shift_score_vs_severity.png', dpi=120, bbox_inches='tight')
plt.close(fig)

# Fig 6: Temporal drift over time (Rolling 7-day window across test period)
print("Computing rolling temporal drift across test period...")
rolling_drift_df = detector.detect_rolling(test_df, window_size=2016, step_size=288, timestamp_col='datetime')
rolling_drift_df.to_csv(DRIFT_RES / 'rolling_drift_test.csv', index=False)

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(pd.to_datetime(rolling_drift_df['timestamp']), rolling_drift_df['shift_score'],
        marker='o', color='#E67E22', lw=2, label='Rolling Shift Score (7-day window)')
ax.axhline(detector.threshold, color='red', ls='--', lw=1.5,
           label=f'Shift Threshold={detector.threshold:.4f}')
ax.fill_between(pd.to_datetime(rolling_drift_df['timestamp']),
                rolling_drift_df['shift_score'], detector.threshold,
                where=(rolling_drift_df['shift_score'] >= detector.threshold),
                color='#E74C3C', alpha=0.3, interpolate=True, label='Shift Detected Region')
ax.set_title('Temporal Drift Tracking: Test Period (Nov–Dec 2018)', fontsize=12)
ax.set_xlabel('Date')
ax.set_ylabel('Mean KS Shift Score')
ax.legend(loc='upper left')
fig.tight_layout()
fig.savefig(DRIFT_FIGS / 'temporal_drift_tracking.png', dpi=120, bbox_inches='tight')
plt.close(fig)

# Copy figures to root figures/ directory for standard reference
import shutil
for f in DRIFT_FIGS.glob('*.png'):
    shutil.copy(f, FIGURES_DIR / f.name)

print("Figures saved to figures/drift/ and mirrored to figures/")

# ── 11. Experiment tracking ───────────────────────────────────────────────────
# Benchmark inference time per sequence
dummy_input = torch.randn(1, SEQUENCE_LENGTH, N_FEATURES).to(DEVICE)
with torch.no_grad():
    for _ in range(20):
        _ = ssl_model.forward_classify(dummy_input)
    t0 = time.perf_counter()
    n_runs = 200
    for _ in range(n_runs):
        _ = ssl_model.forward_classify(dummy_input)
    t1 = time.perf_counter()
inf_ms = round((t1 - t0) / n_runs * 1000, 4)

exp_rows = []
for row in severity_rows:
    exp_rows.append({
        'experiment_id':       f"phase7_ssl_{row['level']}_{int(time.time())}",
        'model_name':          'SSL_Transformer',
        'ssl_used':            True,
        'shift_type':          'synthetic_additive' if row['shift_k'] > 0 else 'none',
        'shift_method':        'KolmogorovSmirnov',
        'shift_severity':      row['level'],
        'shift_k':             row['shift_k'],
        'shift_score':         round(row['shift_score'], 6),
        'shift_threshold':     round(detector.threshold, 6),
        'shift_detected':      row['shift_detected'],
        'n_features_shifted':  row['n_features_shifted'],
        'precision':           row['precision'],
        'recall':              row['recall'],
        'f1':                  row['f1'],
        'roc_auc':             row['roc_auc'],
        'pr_auc':              row['pr_auc'],
        'false_positive_rate': row['fpr'],
        'false_negative_rate': row['fnr'],
        'inference_time':      inf_ms,
        'model_size_kb':       round((MODELS_DIR/'ssl'/'ssl_finetuned.pt').stat().st_size/1024, 2),
    })

exp_path = EXP_DIR / 'experiments.csv'
new_df   = pd.DataFrame(exp_rows)
if exp_path.exists():
    existing = pd.read_csv(exp_path)
    new_df   = pd.concat([existing, new_df], ignore_index=True)
new_df.to_csv(exp_path, index=False)
print(f"Experiment tracking updated: {len(exp_rows)} rows added (Inference time: {inf_ms} ms/seq).")

# ── 12. Save consolidated drift results ───────────────────────────────────────
drift_summary = {
    'method':                'KolmogorovSmirnov',
    'threshold':             round(detector.threshold, 6),
    'train_val_shift_score': round(result_val['shift_score'],  6),
    'train_val_detected':    result_val['shift_detected'],
    'train_test_shift_score':round(result_test['shift_score'], 6),
    'train_test_detected':   result_test['shift_detected'],
    'severity_results':      severity_rows,
    'model_comparison':      comp_rows,
}
with open(DRIFT_RES / 'drift_summary.json', 'w') as f:
    json.dump(drift_summary, f, indent=2)

# ── 13. Final summary ─────────────────────────────────────────────────────────
print("\n" + "="*65)
print("PHASE 7 FINAL SUMMARY")
print("="*65)
print(f"\nDetection method : KS test (mean across {N_FEATURES} features)")
print(f"Bootstrap threshold (99th pct): {detector.threshold:.6f}")
print(f"\nNatural shift (train Jan-Aug -> test Nov-Dec):")
print(f"  Shift score : {result_test['shift_score']:.6f}  ->  "
      f"{'SHIFT DETECTED' if result_test['shift_detected'] else 'STABLE'}")
print(f"  Features shifted: {result_test['n_features_shifted']}/{result_test['n_features_total']}")
print(f"\nBaseline (train -> val, same season):")
print(f"  Shift score : {result_val['shift_score']:.6f}  ->  "
      f"{'SHIFT DETECTED' if result_val['shift_detected'] else 'STABLE'}")
print(f"\nSSL Transformer -- severity experiment:")
print(sev_df[['level','shift_k','shift_score','shift_detected','f1','recall','fpr','fnr']].to_string(index=False))
print(f"\nModel comparison under strong shift (k=1.0):")
print(comp_df[['model','condition','f1','recall','precision','roc_auc']].to_string(index=False))
print(f"\nMost shifted sensors (train->test):")
print(pf_test[['feature','ks_statistic']].head(5).to_string(index=False))
print(f"\nError analysis (SSL, strong shift):")
print(f"  Normal  : FP={len(fp_norm)}  FN={len(fn_norm)}")
print(f"  Shifted : FP={len(fp_shft)}  FN={len(fn_shft)}")
print(f"\nAll results saved to: results/drift/")
