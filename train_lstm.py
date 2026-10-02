"""
Phase 4 — LSTM (minimal CPU config).
hidden=64, layers=1, bidirectional=False, batch=512, 5 epochs.
Run from project root: python train_lstm.py
"""
import sys, json, time
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
from sklearn.metrics import confusion_matrix, classification_report

from data_loader import load_raw
from preprocessing import run_preprocessing
from datasets import WaterSensorDataset
from lstm import LSTMClassifier
from training import train_model, set_seed, run_epoch
from evaluation import compute_metrics, select_threshold_f1, curves
from config import SEED, SEQUENCE_LENGTH, MODELS_DIR, RESULTS_DIR, FIGURES_DIR

set_seed(SEED)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")

RES  = RESULTS_DIR / 'lstm'
FIGS = FIGURES_DIR / 'lstm'
RES.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)
(FIGS / 'error_analysis').mkdir(exist_ok=True)

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("\nLoading data...")
df = load_raw()
train_df, val_df, test_df, scaler, feat_cols = run_preprocessing(df, save=False)

# Use stride=12 (1-hour steps) to reduce window count ~12x
train_ds = WaterSensorDataset(train_df, feat_cols, label_strategy='last', stride=12)
val_ds   = WaterSensorDataset(val_df,   feat_cols, label_strategy='last', stride=12)
test_ds  = WaterSensorDataset(test_df,  feat_cols, label_strategy='last', stride=12)

INPUT_SIZE = train_ds.n_features
print(f"Sequence shape: (batch, {SEQUENCE_LENGTH}, {INPUT_SIZE})")
print(f"Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")

n_neg = (train_ds.labels == 0).sum()
n_pos = (train_ds.labels == 1).sum()
pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(DEVICE)
print(f"pos_weight: {pos_weight.item():.3f}")

# ── 2. Fixed config — minimal for CPU ────────────────────────────────────────
HP = {'hidden_size': 64, 'num_layers': 1, 'dropout': 0.2,
      'lr': 1e-3, 'batch_size': 512, 'bidirectional': False}
print(f"\nConfig: {HP}")

BATCH = HP['batch_size']
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False, num_workers=0)

# ── 3. Build model ────────────────────────────────────────────────────────────
model = LSTMClassifier(
    input_size=INPUT_SIZE,
    hidden_size=HP['hidden_size'],
    num_layers=HP['num_layers'],
    dropout=HP['dropout'],
    bidirectional=HP['bidirectional'],
).to(DEVICE)
print(f"Parameters: {model.n_params:,}")

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.AdamW(model.parameters(), lr=HP['lr'], weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6)

# ── 4. Train — 5 epochs ───────────────────────────────────────────────────────
print("\n" + "="*50)
print("TRAINING (5 epochs)")
print("="*50)

history = train_model(
    model, train_loader, val_loader,
    criterion, optimizer, scheduler,
    epochs=5, patience=3, device=DEVICE, verbose=True,
)

torch.save(model.state_dict(), MODELS_DIR / 'lstm' / 'best_lstm.pt')
print(f"\nBest epoch: {history['best_epoch']}  Best val F1: {history['best_val_f1']:.4f}")

# Save experiment log
pd.DataFrame([{
    'experiment_id': 1,
    'hidden_size': HP['hidden_size'], 'num_layers': HP['num_layers'],
    'dropout': HP['dropout'], 'learning_rate': HP['lr'],
    'batch_size': HP['batch_size'], 'bidirectional': HP['bidirectional'],
    'weight_decay': 1e-4, 'sequence_length': SEQUENCE_LENGTH,
    'training_epochs': len(history['train_loss']),
    'best_epoch': history['best_epoch'],
    'validation_loss': round(history['best_val_loss'], 6),
    'validation_f1': round(history['best_val_f1'], 4),
}]).to_csv(RES / 'lstm_experiments.csv', index=False)

# ── 5. Threshold selection on validation set ──────────────────────────────────
print("\nSelecting threshold on validation set...")
val_result = run_epoch(model, val_loader, criterion, None, DEVICE, threshold=0.5)
threshold  = select_threshold_f1(val_result['labels'], val_result['probs'])
print(f"Threshold (val F1-max): {threshold:.4f}")

# ── 6. Test evaluation ────────────────────────────────────────────────────────
print("\nEvaluating on test set...")
test_result = run_epoch(model, test_loader, criterion, None, DEVICE, threshold=threshold)
y_test    = test_result['labels']
p_test    = test_result['probs']
pred_test = (p_test >= threshold).astype(int)

test_metrics = compute_metrics(y_test, pred_test, p_test)
print("\nTest metrics:")
for k, v in test_metrics.items():
    if k not in ('tp','tn','fp','fn'):
        print(f"  {k}: {v:.4f}")
print("\nClassification report:")
print(classification_report(y_test, pred_test, target_names=['Normal','Anomaly']))

# ── 7. Save predictions and metrics ──────────────────────────────────────────
pd.DataFrame({'true_label': y_test, 'pred_label': pred_test,
              'probability': p_test.round(4)}
             ).to_csv(RES / 'test_predictions.csv', index=False)

pd.DataFrame([{'model': 'LSTM',
               **{k: round(v,4) for k,v in test_metrics.items()
                  if k not in ('tp','tn','fp','fn')},
               'threshold': round(threshold,4),
               'best_epoch': history['best_epoch'],
               'n_params': model.n_params}]
             ).to_csv(RES / 'final_metrics.csv', index=False)

# ── 8. Training curves ────────────────────────────────────────────────────────
print("\nGenerating figures...")
ep = range(1, len(history['train_loss']) + 1)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(ep, history['train_loss'], label='Train', color='#4C72B0')
axes[0].plot(ep, history['val_loss'],   label='Val',   color='#DD8452')
axes[0].axvline(history['best_epoch'], color='green', ls='--', lw=1,
                label=f"Best ep {history['best_epoch']}")
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
axes[0].set_title('LSTM — Loss'); axes[0].legend()
axes[1].plot(ep, history['train_f1'], label='Train F1', color='#4C72B0')
axes[1].plot(ep, history['val_f1'],   label='Val F1',   color='#DD8452')
axes[1].axvline(history['best_epoch'], color='green', ls='--', lw=1)
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('F1')
axes[1].set_title('LSTM — F1'); axes[1].legend()
fig.tight_layout()
fig.savefig(FIGS / 'loss_curve.png', dpi=120, bbox_inches='tight')
plt.close(fig)

# ── 9. Confusion matrix ───────────────────────────────────────────────────────
cm = confusion_matrix(y_test, pred_test)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Normal','Anomaly'], yticklabels=['Normal','Anomaly'])
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
ax.set_title('LSTM — Confusion Matrix (Test)')
fig.tight_layout()
fig.savefig(FIGS / 'confusion_matrix.png', dpi=120, bbox_inches='tight')
plt.close(fig)

# ── 10. ROC and PR curves ─────────────────────────────────────────────────────
cv = curves(y_test, p_test)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(cv['roc']['fpr'], cv['roc']['tpr'], lw=2,
        label=f"LSTM (AUC={test_metrics['roc_auc']:.4f})", color='#4C72B0')
ax.plot([0,1],[0,1],'k--',lw=1)
ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
ax.set_title('LSTM — ROC Curve (Test)'); ax.legend()
fig.tight_layout()
fig.savefig(FIGS / 'roc_curve.png', dpi=120, bbox_inches='tight')
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(cv['pr']['recall'], cv['pr']['precision'], lw=2,
        label=f"LSTM (PR-AUC={test_metrics['pr_auc']:.4f})", color='#4C72B0')
ax.axhline(y_test.mean(), color='grey', ls='--', lw=1,
           label=f'No-skill ({y_test.mean():.3f})')
ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
ax.set_title('LSTM — PR Curve (Test)'); ax.legend()
fig.tight_layout()
fig.savefig(FIGS / 'pr_curve.png', dpi=120, bbox_inches='tight')
plt.close(fig)

# ── 11. Error analysis ────────────────────────────────────────────────────────
print("Running error analysis...")
fp_idx = np.where((pred_test==1)&(y_test==0))[0]
fn_idx = np.where((pred_test==0)&(y_test==1))[0]
tp_idx = np.where((pred_test==1)&(y_test==1))[0]
tn_idx = np.where((pred_test==0)&(y_test==0))[0]
print(f"  TP={len(tp_idx)} TN={len(tn_idx)} FP={len(fp_idx)} FN={len(fn_idx)}")

rows = []
for idx in fp_idx[:10]:
    rows.append({'error_type':'FP','true':0,'pred':1,
                 'prob':round(float(p_test[idx]),4),'window_idx':int(idx)})
for idx in fn_idx[:10]:
    rows.append({'error_type':'FN','true':1,'pred':0,
                 'prob':round(float(p_test[idx]),4),'window_idx':int(idx)})
pd.DataFrame(rows).to_csv(RES / 'error_analysis.csv', index=False)

# Representative sequence plot
cases = [
    ('TP — Detected anomaly',  tp_idx[0] if len(tp_idx) else None, '#55A868'),
    ('FN — Missed anomaly',    fn_idx[0] if len(fn_idx) else None, '#DD8452'),
    ('FP — False alarm',       fp_idx[0] if len(fp_idx) else None, '#C44E52'),
    ('TN — Correct normal',    tn_idx[0] if len(tn_idx) else None, '#4C72B0'),
]
fig, axes = plt.subplots(4, 1, figsize=(14, 12))
for ax, (title, idx, color) in zip(axes, cases):
    if idx is None:
        ax.set_visible(False); continue
    seq = test_ds.windows[idx]
    for ch in range(min(4, seq.shape[1])):
        ax.plot(seq[:, ch], lw=0.8, alpha=0.7, label=feat_cols[ch])
    ax.set_title(f"{title} | prob={p_test[idx]:.3f} | "
                 f"true={y_test[idx]} pred={pred_test[idx]}", fontsize=9)
    ax.legend(fontsize=7, loc='upper right', ncol=4)
    ax.tick_params(labelsize=7)
fig.suptitle('LSTM — Representative Sequences', fontsize=12, fontweight='bold')
fig.tight_layout()
fig.savefig(FIGS / 'error_analysis' / 'representative_sequences.png',
            dpi=120, bbox_inches='tight')
plt.close(fig)

# Inference time — batched to avoid OOM
print("Measuring inference time...")
model.eval()
times = []
for _ in range(3):
    t0 = time.perf_counter()
    with torch.no_grad():
        for X_b, _ in test_loader:
            _ = torch.sigmoid(model(X_b.to(DEVICE)))
    times.append(time.perf_counter() - t0)
inf_s  = float(np.median(times))
inf_ms = inf_s / len(test_ds) * 1000
print(f"  Total: {inf_s*1000:.1f}ms | Per-sample: {inf_ms:.4f}ms")
pd.DataFrame([{'model':'LSTM','total_s':round(inf_s,6),
               'per_sample_ms':round(inf_ms,6),'n_samples':len(test_ds),
               'device':DEVICE}]
             ).to_csv(RES / 'inference_time.csv', index=False)

# ── 13. Model size ────────────────────────────────────────────────────────────
size_kb = (MODELS_DIR / 'lstm' / 'best_lstm.pt').stat().st_size / 1024
pd.DataFrame([{'model':'LSTM','size_kb':round(size_kb,2),
               'n_params':model.n_params}]
             ).to_csv(RES / 'model_size.csv', index=False)
print(f"  Model size: {size_kb:.1f} KB | Params: {model.n_params:,}")

# ── 14. Save config ───────────────────────────────────────────────────────────
with open(MODELS_DIR / 'lstm' / 'config.json', 'w') as f:
    json.dump({**HP, 'input_size': INPUT_SIZE,
               'sequence_length': SEQUENCE_LENGTH,
               'threshold': round(threshold,4),
               'best_epoch': history['best_epoch'],
               'n_params': model.n_params, 'device': DEVICE}, f, indent=2)

# ── 15. Classical vs LSTM comparison ─────────────────────────────────────────
print("\nBuilding comparison table...")
baseline_df = pd.read_csv('results/baselines/baseline_comparison.csv')
lstm_row = pd.DataFrame([{
    'model': 'LSTM',
    'accuracy':      round(test_metrics['accuracy'],  4),
    'precision':     round(test_metrics['precision'], 4),
    'recall':        round(test_metrics['recall'],    4),
    'f1':            round(test_metrics['f1'],        4),
    'roc_auc':       round(test_metrics['roc_auc'],   4),
    'pr_auc':        round(test_metrics['pr_auc'],    4),
    'fpr':           round(test_metrics['fpr'],       4),
    'fnr':           round(test_metrics['fnr'],       4),
    'inference_ms':  round(inf_ms, 6),
    'model_size_kb': round(size_kb, 2),
}])
comp_df = pd.concat([baseline_df, lstm_row], ignore_index=True)
comp_path = Path('results/model_comparison')
comp_path.mkdir(parents=True, exist_ok=True)
comp_df.to_csv(comp_path / 'classical_vs_lstm.csv', index=False)

md = ['# Classical ML vs LSTM Comparison (Test Set)\n',
      '| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR | Inf(ms) | Size(KB) |',
      '|---|---|---|---|---|---|---|---|---|---|---|']
for _, row in comp_df.iterrows():
    md.append(f"| {row['model']} | {row['accuracy']:.4f} | {row['precision']:.4f} | "
              f"{row['recall']:.4f} | {row['f1']:.4f} | {row['roc_auc']:.4f} | "
              f"{row['pr_auc']:.4f} | {row['fpr']:.4f} | {row['fnr']:.4f} | "
              f"{row['inference_ms']:.4f} | {row['model_size_kb']:.1f} |")
with open(comp_path / 'classical_vs_lstm.md', 'w') as f:
    f.write('\n'.join(md))

# Comparison bar chart
comp_figs = Path('figures/model_comparison')
comp_figs.mkdir(parents=True, exist_ok=True)
metrics_to_plot = ['f1','precision','recall','roc_auc','pr_auc']
x = np.arange(len(comp_df))
width = 0.15
fig, ax = plt.subplots(figsize=(13, 5))
for i, metric in enumerate(metrics_to_plot):
    vals = comp_df[metric].tolist()
    bars = ax.bar(x + i*width, vals, width, label=metric.upper(),
                  color=plt.cm.Set2(i/len(metrics_to_plot)))
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f'{v:.3f}', ha='center', va='bottom', fontsize=6.5)
ax.set_xticks(x + width*(len(metrics_to_plot)-1)/2)
ax.set_xticklabels(comp_df['model'].tolist(), fontsize=9)
ax.set_ylim(0, 1.15); ax.set_ylabel('Score')
ax.set_title('Classical ML vs LSTM — Test Set Metrics')
ax.legend(fontsize=8); fig.tight_layout()
fig.savefig(comp_figs / 'classical_vs_lstm_metrics.png', dpi=120, bbox_inches='tight')
plt.close(fig)

# ── 16. Final print ───────────────────────────────────────────────────────────
print("\n" + "="*65)
print("PHASE 4 RESULTS — TEST SET")
print("="*65)
print(comp_df.to_string(index=False))
print(f"\nConfig: {HP}")
print(f"Threshold: {threshold:.4f} | Best epoch: {history['best_epoch']}")
print(f"Model: {size_kb:.1f} KB | {model.n_params:,} params")
