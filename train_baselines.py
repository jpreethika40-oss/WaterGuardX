"""
Phase 3 — Classical ML Baselines training and evaluation script.
Run from project root: python train_baselines.py
"""
import sys, json, time
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report

from data_loader import load_raw
from preprocessing import run_preprocessing
from baselines import (
    build_tabular_features, feature_names,
    build_logistic_regression, build_random_forest, build_xgboost,
    train_model, predict_proba, save_model, select_threshold,
)
from evaluation import compute_metrics, measure_inference_time, curves
from config import TARGET_COL, SEED, RESULTS_DIR, FIGURES_DIR

np.random.seed(SEED)

RES   = RESULTS_DIR / 'baselines'
FIGS  = FIGURES_DIR / 'baselines'
RES.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("Loading and preprocessing data...")
df = load_raw()
train_df, val_df, test_df, scaler, feat_cols = run_preprocessing(df, save=False)

print("Building tabular features...")
X_train, y_train = build_tabular_features(train_df, feat_cols)
X_val,   y_val   = build_tabular_features(val_df,   feat_cols)
X_test,  y_test  = build_tabular_features(test_df,  feat_cols)
feat_names = feature_names(feat_cols)

print(f"  Train: {X_train.shape}, anomaly={y_train.mean()*100:.1f}%")
print(f"  Val:   {X_val.shape},   anomaly={y_val.mean()*100:.1f}%")
print(f"  Test:  {X_test.shape},  anomaly={y_test.mean()*100:.1f}%")
print(f"  Features: {X_train.shape[1]}")

# Class imbalance ratio for XGBoost
n_neg = (y_train == 0).sum()
n_pos = (y_train == 1).sum()
spw   = float(n_neg / n_pos)
print(f"\nClass imbalance — normal:{n_neg}, anomaly:{n_pos}, ratio:{spw:.2f}")

# ── 2. Define models ──────────────────────────────────────────────────────────
models_cfg = {
    'logistic_regression': {
        'model':  build_logistic_regression(),
        'config': {
            'C': 1.0, 'max_iter': 1000,
            'class_weight': 'balanced', 'solver': 'lbfgs',
        },
    },
    'random_forest': {
        'model':  build_random_forest(),
        'config': {
            'n_estimators': 300, 'max_depth': 20,
            'min_samples_split': 10, 'min_samples_leaf': 5,
            'max_features': 'sqrt', 'class_weight': 'balanced',
        },
    },
    'xgboost': {
        'model':  build_xgboost(scale_pos_weight=spw),
        'config': {
            'n_estimators': 300, 'learning_rate': 0.05,
            'max_depth': 6, 'subsample': 0.8,
            'colsample_bytree': 0.8, 'scale_pos_weight': round(spw, 3),
        },
    },
}

# ── 3. Train, tune threshold, evaluate ───────────────────────────────────────
all_results   = {}
all_curves    = {}
all_train_log = []

for name, cfg in models_cfg.items():
    print(f"\n{'='*55}")
    print(f"  {name.upper()}")
    print(f"{'='*55}")

    model = cfg['model']

    # Train
    print("  Training...")
    model, train_time = train_model(model, X_train, y_train, X_val, y_val)
    print(f"  Training time: {train_time:.2f}s")

    # Validation probabilities
    prob_val  = predict_proba(model, X_val)
    prob_test = predict_proba(model, X_test)

    # Threshold selection on validation set
    threshold = select_threshold(y_val, prob_val)
    print(f"  Selected threshold (val F1-max): {threshold:.3f}")

    # Validation metrics
    pred_val  = (prob_val  >= threshold).astype(int)
    pred_test = (prob_test >= threshold).astype(int)

    val_metrics  = compute_metrics(y_val,  pred_val,  prob_val)
    test_metrics = compute_metrics(y_test, pred_test, prob_test)

    print(f"  Val  — F1={val_metrics['f1']:.4f}  "
          f"ROC-AUC={val_metrics['roc_auc']:.4f}  "
          f"PR-AUC={val_metrics['pr_auc']:.4f}")
    print(f"  Test — F1={test_metrics['f1']:.4f}  "
          f"ROC-AUC={test_metrics['roc_auc']:.4f}  "
          f"PR-AUC={test_metrics['pr_auc']:.4f}")

    # Inference time (on test set)
    inf = measure_inference_time(lambda X: model.predict_proba(X), X_test)
    print(f"  Inference: {inf['total_s']*1000:.1f}ms total, "
          f"{inf['per_sample_ms']:.4f}ms/sample")

    # Model size
    model_path = Path('models') / 'baselines' / name / 'model.pkl'
    save_model(model, name, cfg['config'], threshold)
    model_size_kb = model_path.stat().st_size / 1024

    # Curves for plotting
    all_curves[name] = curves(y_test, prob_test)

    # Store
    all_results[name] = {
        'val':  val_metrics,
        'test': test_metrics,
        'threshold':     threshold,
        'train_time_s':  round(train_time, 3),
        'inference':     inf,
        'model_size_kb': round(model_size_kb, 2),
    }

    all_train_log.append({
        'model':           name,
        'features':        X_train.shape[1],
        'train_samples':   len(X_train),
        'val_samples':     len(X_val),
        'test_samples':    len(X_test),
        'train_anomaly%':  round(y_train.mean()*100, 2),
        'val_anomaly%':    round(y_val.mean()*100, 2),
        'test_anomaly%':   round(y_test.mean()*100, 2),
        'threshold':       round(threshold, 4),
        'train_time_s':    round(train_time, 3),
        **{f'test_{k}': round(v, 4) for k, v in test_metrics.items()
           if k not in ('tp','tn','fp','fn')},
    })

    # Confusion matrix figure
    cm = confusion_matrix(y_test, pred_test)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Normal','Anomaly'],
                yticklabels=['Normal','Anomaly'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title(f'{name} — Confusion Matrix (Test)')
    fig.tight_layout()
    fig.savefig(FIGS / f'confusion_matrix_{name}.png', dpi=120, bbox_inches='tight')
    plt.close(fig)

    # Classification report
    print("\n  Classification report (test):")
    print(classification_report(y_test, pred_test,
                                 target_names=['Normal','Anomaly']))

# ── 4. ROC comparison ─────────────────────────────────────────────────────────
print("\nGenerating comparison plots...")
colors = {'logistic_regression': '#4C72B0',
          'random_forest':       '#55A868',
          'xgboost':             '#DD8452'}
labels = {'logistic_regression': 'Logistic Regression',
          'random_forest':       'Random Forest',
          'xgboost':             'XGBoost'}

fig, ax = plt.subplots(figsize=(7, 6))
for name, c in all_curves.items():
    auc = all_results[name]['test']['roc_auc']
    ax.plot(c['roc']['fpr'], c['roc']['tpr'],
            color=colors[name], lw=2,
            label=f"{labels[name]} (AUC={auc:.4f})")
ax.plot([0,1],[0,1],'k--',lw=1)
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves — Baseline Comparison (Test Set)')
ax.legend(fontsize=9); fig.tight_layout()
fig.savefig(FIGS / 'roc_comparison.png', dpi=120, bbox_inches='tight')
plt.close(fig)

# ── 5. PR comparison ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
for name, c in all_curves.items():
    auc = all_results[name]['test']['pr_auc']
    ax.plot(c['pr']['recall'], c['pr']['precision'],
            color=colors[name], lw=2,
            label=f"{labels[name]} (PR-AUC={auc:.4f})")
baseline_rate = y_test.mean()
ax.axhline(baseline_rate, color='grey', linestyle='--', lw=1,
           label=f'No-skill baseline ({baseline_rate:.3f})')
ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curves — Baseline Comparison (Test Set)')
ax.legend(fontsize=9); fig.tight_layout()
fig.savefig(FIGS / 'pr_comparison.png', dpi=120, bbox_inches='tight')
plt.close(fig)

# ── 6. Feature importance ─────────────────────────────────────────────────────
for name in ['random_forest', 'xgboost']:
    model_obj = joblib.load(Path('models') / 'baselines' / name / 'model.pkl')
    importances = model_obj.feature_importances_
    top_n = 20
    idx = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh([feat_names[i] for i in idx[::-1]],
            importances[idx[::-1]], color=colors[name])
    ax.set_xlabel('Importance')
    ax.set_title(f'{labels[name]} — Top {top_n} Feature Importances')
    fig.tight_layout()
    fig.savefig(FIGS / f'feature_importance_{name}.png', dpi=120, bbox_inches='tight')
    plt.close(fig)

# LR coefficients
lr_model = joblib.load(Path('models') / 'baselines' / 'logistic_regression' / 'model.pkl')
coef_abs = np.abs(lr_model.coef_[0])
idx = np.argsort(coef_abs)[::-1][:20]
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh([feat_names[i] for i in idx[::-1]],
        coef_abs[idx[::-1]], color=colors['logistic_regression'])
ax.set_xlabel('|Coefficient|')
ax.set_title('Logistic Regression — Top 20 Feature Coefficients')
fig.tight_layout()
fig.savefig(FIGS / 'feature_importance_logistic_regression.png', dpi=120, bbox_inches='tight')
plt.close(fig)

# ── 7. Metric bar charts ──────────────────────────────────────────────────────
metrics_to_plot = ['f1', 'precision', 'recall', 'roc_auc', 'pr_auc']
model_names_short = list(labels.values())
x = np.arange(len(model_names_short))
width = 0.15

fig, ax = plt.subplots(figsize=(12, 5))
for i, metric in enumerate(metrics_to_plot):
    vals = [all_results[n]['test'][metric] for n in all_results]
    bars = ax.bar(x + i * width, vals, width, label=metric.upper(),
                  color=plt.cm.Set2(i / len(metrics_to_plot)))
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{v:.3f}', ha='center', va='bottom', fontsize=7)
ax.set_xticks(x + width * (len(metrics_to_plot)-1) / 2)
ax.set_xticklabels(model_names_short)
ax.set_ylim(0, 1.12)
ax.set_ylabel('Score')
ax.set_title('Baseline Model Comparison — Test Set Metrics')
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIGS / 'metric_comparison.png', dpi=120, bbox_inches='tight')
plt.close(fig)

# ── 8. Error analysis ─────────────────────────────────────────────────────────
print("\nRunning error analysis...")
error_rows = []
for name in all_results:
    model_obj = joblib.load(Path('models') / 'baselines' / name / 'model.pkl')
    prob  = predict_proba(model_obj, X_test)
    thresh = all_results[name]['threshold']
    pred  = (prob >= thresh).astype(int)

    fp_mask = (pred == 1) & (y_test == 0)
    fn_mask = (pred == 0) & (y_test == 1)

    # Sample up to 5 FP and 5 FN with their probabilities
    for idx in np.where(fp_mask)[0][:5]:
        error_rows.append({
            'model': name, 'error_type': 'FP',
            'true_label': 0, 'pred_label': 1,
            'probability': round(float(prob[idx]), 4),
            'test_row_idx': int(idx),
        })
    for idx in np.where(fn_mask)[0][:5]:
        error_rows.append({
            'model': name, 'error_type': 'FN',
            'true_label': 1, 'pred_label': 0,
            'probability': round(float(prob[idx]), 4),
            'test_row_idx': int(idx),
        })

pd.DataFrame(error_rows).to_csv(RES / 'error_analysis.csv', index=False)

# ── 9. Save results tables ────────────────────────────────────────────────────
print("\nSaving results...")

# Comparison CSV
rows = []
for name in all_results:
    r = all_results[name]
    t = r['test']
    rows.append({
        'model':          labels[name],
        'accuracy':       round(t['accuracy'],  4),
        'precision':      round(t['precision'], 4),
        'recall':         round(t['recall'],    4),
        'f1':             round(t['f1'],        4),
        'roc_auc':        round(t['roc_auc'],   4),
        'pr_auc':         round(t['pr_auc'],    4),
        'fpr':            round(t['fpr'],       4),
        'fnr':            round(t['fnr'],       4),
        'inference_ms':   r['inference']['per_sample_ms'],
        'model_size_kb':  r['model_size_kb'],
    })
comp_df = pd.DataFrame(rows)
comp_df.to_csv(RES / 'baseline_comparison.csv', index=False)

# Markdown table
md_lines = ['# Baseline Model Comparison (Test Set)\n',
            '| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR | Inf(ms/sample) | Size(KB) |',
            '|---|---|---|---|---|---|---|---|---|---|---|']
for _, row in comp_df.iterrows():
    md_lines.append(
        f"| {row['model']} | {row['accuracy']:.4f} | {row['precision']:.4f} | "
        f"{row['recall']:.4f} | {row['f1']:.4f} | {row['roc_auc']:.4f} | "
        f"{row['pr_auc']:.4f} | {row['fpr']:.4f} | {row['fnr']:.4f} | "
        f"{row['inference_ms']:.4f} | {row['model_size_kb']:.1f} |"
    )
with open(RES / 'baseline_comparison.md', 'w') as f:
    f.write('\n'.join(md_lines))

# Model summary
pd.DataFrame(all_train_log).to_csv(RES / 'model_summary.csv', index=False)

# Inference time
inf_rows = [{'model': n,
             'total_s': all_results[n]['inference']['total_s'],
             'per_sample_ms': all_results[n]['inference']['per_sample_ms'],
             'n_samples': all_results[n]['inference']['n_samples']}
            for n in all_results]
pd.DataFrame(inf_rows).to_csv(RES / 'inference_time.csv', index=False)

# Model size
size_rows = [{'model': n, 'size_kb': all_results[n]['model_size_kb']}
             for n in all_results]
pd.DataFrame(size_rows).to_csv(RES / 'model_size.csv', index=False)

# Full results JSON
with open(RES / 'full_results.json', 'w') as f:
    def _serial(o):
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        return str(o)
    json.dump(all_results, f, indent=2, default=_serial)

# ── 10. Print final table ─────────────────────────────────────────────────────
print("\n" + "="*70)
print("PHASE 3 RESULTS — TEST SET")
print("="*70)
print(comp_df.to_string(index=False))
print("\nFiles saved to results/baselines/ and figures/baselines/")
