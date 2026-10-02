"""
Phase 5 — Temporal Transformer training and evaluation.
Uses same stride=12 data as LSTM for fair comparison.
Run from project root: python train_transformer.py
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
from transformer import TemporalTransformer
from training import train_model, set_seed, run_epoch
from evaluation import compute_metrics, select_threshold_f1, curves
from config import SEED, SEQUENCE_LENGTH, MODELS_DIR, RESULTS_DIR, FIGURES_DIR

set_seed(SEED)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")

RES  = RESULTS_DIR / 'transformer'
FIGS = FIGURES_DIR / 'transformer'
RES.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)
(FIGS / 'error_analysis').mkdir(exist_ok=True)
(FIGS / 'attention').mkdir(exist_ok=True)

# ── 1. Load data — same stride=12 as LSTM for fair comparison ────────────────
print("\nLoading data...")
df = load_raw()
train_df, val_df, test_df, scaler, feat_cols = run_preprocessing(df, save=False)

STRIDE = 12   # same as LSTM Phase 4
train_ds = WaterSensorDataset(train_df, feat_cols, label_strategy='last', stride=STRIDE)
val_ds   = WaterSensorDataset(val_df,   feat_cols, label_strategy='last', stride=STRIDE)
test_ds  = WaterSensorDataset(test_df,  feat_cols, label_strategy='last', stride=STRIDE)

N_FEATURES = train_ds.n_features
print(f"Sequence shape : (batch, {SEQUENCE_LENGTH}, {N_FEATURES})")
print(f"Train windows  : {len(train_ds):,}  anomaly={train_ds.labels.mean()*100:.1f}%")
print(f"Val   windows  : {len(val_ds):,}    anomaly={val_ds.labels.mean()*100:.1f}%")
print(f"Test  windows  : {len(test_ds):,}   anomaly={test_ds.labels.mean()*100:.1f}%")

n_neg = (train_ds.labels == 0).sum()
n_pos = (train_ds.labels == 1).sum()
pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(DEVICE)
print(f"pos_weight     : {pos_weight.item():.3f}")

# ── 2. Hyperparameter search — 3 configs, 5 epochs each ──────────────────────
# Configs kept small for CPU. d_model must be divisible by nhead.
HP_GRID = [
    {'d_model': 32,  'nhead': 2, 'num_layers': 2, 'dim_feedforward': 64,
     'dropout': 0.1, 'lr': 1e-3, 'batch_size': 256},
    {'d_model': 64,  'nhead': 4, 'num_layers': 2, 'dim_feedforward': 128,
     'dropout': 0.1, 'lr': 1e-3, 'batch_size': 256},
    {'d_model': 64,  'nhead': 4, 'num_layers': 2, 'dim_feedforward': 128,
     'dropout': 0.2, 'lr': 5e-4, 'batch_size': 256},
]

print(f"\n{'='*55}")
print(f"HP SEARCH ({len(HP_GRID)} configs, 5 epochs each)")
print(f"{'='*55}")

exp_rows    = []
best_val_f1 = -1.0
best_hp     = None

for exp_id, hp in enumerate(HP_GRID, 1):
    set_seed(SEED)
    print(f"\n[{exp_id}/{len(HP_GRID)}] {hp}")

    train_loader = DataLoader(train_ds, batch_size=hp['batch_size'],
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=hp['batch_size'],
                              shuffle=False, num_workers=0)

    model = TemporalTransformer(
        n_features=N_FEATURES,
        d_model=hp['d_model'], nhead=hp['nhead'],
        num_layers=hp['num_layers'],
        dim_feedforward=hp['dim_feedforward'],
        dropout=hp['dropout'],
        max_seq_len=SEQUENCE_LENGTH + 10,
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=hp['lr'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6)

    history = train_model(
        model, train_loader, val_loader,
        criterion, optimizer, scheduler,
        epochs=5, patience=3, device=DEVICE, verbose=False,
    )

    vf1  = history['best_val_f1']
    vloss = history['best_val_loss']
    print(f"  params={model.n_params:,}  best_ep={history['best_epoch']}  "
          f"val_f1={vf1:.4f}  val_loss={vloss:.4f}")

    exp_rows.append({
        'experiment_id': exp_id,
        'd_model': hp['d_model'], 'nhead': hp['nhead'],
        'num_layers': hp['num_layers'],
        'dim_feedforward': hp['dim_feedforward'],
        'dropout': hp['dropout'], 'learning_rate': hp['lr'],
        'weight_decay': 1e-4, 'batch_size': hp['batch_size'],
        'sequence_length': SEQUENCE_LENGTH,
        'training_epochs': len(history['train_loss']),
        'best_epoch': history['best_epoch'],
        'validation_loss': round(vloss, 6),
        'validation_f1': round(vf1, 4),
        'n_params': model.n_params,
    })

    if vf1 > best_val_f1:
        best_val_f1 = vf1
        best_hp     = hp
        torch.save(model.state_dict(), MODELS_DIR / 'transformer' / 'best_transformer.pt')

pd.DataFrame(exp_rows).to_csv(RES / 'transformer_experiments.csv', index=False)
print(f"\nBest config : {best_hp}")
print(f"Best val F1 : {best_val_f1:.4f}")

# ── 3. Final model — retrain best config for 5 epochs ────────────────────────
print(f"\n{'='*55}")
print("FINAL MODEL TRAINING (best HP, 5 epochs)")
print(f"{'='*55}")
set_seed(SEED)

BATCH = best_hp['batch_size']
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False, num_workers=0)

final_model = TemporalTransformer(
    n_features=N_FEATURES,
    d_model=best_hp['d_model'], nhead=best_hp['nhead'],
    num_layers=best_hp['num_layers'],
    dim_feedforward=best_hp['dim_feedforward'],
    dropout=best_hp['dropout'],
    max_seq_len=SEQUENCE_LENGTH + 10,
).to(DEVICE)

print(f"Parameters: {final_model.n_params:,}")

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.AdamW(final_model.parameters(),
                               lr=best_hp['lr'], weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6)

final_history = train_model(
    final_model, train_loader, val_loader,
    criterion, optimizer, scheduler,
    epochs=5, patience=3, device=DEVICE, verbose=True,
)

torch.save(final_model.state_dict(), MODELS_DIR / 'transformer' / 'best_transformer.pt')
print(f"\nBest epoch : {final_history['best_epoch']}")
print(f"Best val F1: {final_history['best_val_f1']:.4f}")

# ── 4. Threshold selection on validation set ──────────────────────────────────
print("\nSelecting threshold on validation set...")
val_result = run_epoch(final_model, val_loader, criterion, None, DEVICE, threshold=0.5)
threshold  = select_threshold_f1(val_result['labels'], val_result['probs'])
print(f"Threshold (val F1-max): {threshold:.4f}")
with open(RES / 'threshold.json', 'w') as f:
    json.dump({'threshold': threshold}, f)

# ── 5. Test evaluation ────────────────────────────────────────────────────────
print("\nEvaluating on test set...")
test_result = run_epoch(final_model, test_loader, criterion, None, DEVICE, threshold=threshold)
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

# ── 6. Save predictions and metrics ──────────────────────────────────────────
pd.DataFrame({'true_label': y_test, 'pred_label': pred_test,
              'probability': p_test.round(4)}
             ).to_csv(RES / 'test_predictions.csv', index=False)

pd.DataFrame([{'model': 'Transformer',
               **{k: round(v,4) for k,v in test_metrics.items()
                  if k not in ('tp','tn','fp','fn')},
               'threshold': round(threshold,4),
               'best_epoch': final_history['best_epoch'],
               'n_params': final_model.n_params}]
             ).to_csv(RES / 'final_metrics.csv', index=False)

# ── 7. Training curves ────────────────────────────────────────────────────────
print("\nGenerating figures...")
ep = range(1, len(final_history['train_loss']) + 1)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(ep, final_history['train_loss'], label='Train', color='#4C72B0')
axes[0].plot(ep, final_history['val_loss'],   label='Val',   color='#DD8452')
axes[0].axvline(final_history['best_epoch'], color='green', ls='--', lw=1,
                label=f"Best ep {final_history['best_epoch']}")
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
axes[0].set_title('Transformer — Loss'); axes[0].legend()
axes[1].plot(ep, final_history['train_f1'], label='Train F1', color='#4C72B0')
axes[1].plot(ep, final_history['val_f1'],   label='Val F1',   color='#DD8452')
axes[1].axvline(final_history['best_epoch'], color='green', ls='--', lw=1)
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('F1')
axes[1].set_title('Transformer — F1'); axes[1].legend()
fig.tight_layout()
fig.savefig(FIGS / 'loss_curve.png', dpi=120, bbox_inches='tight')
plt.close(fig)

# ── 8. Confusion matrix ───────────────────────────────────────────────────────
cm = confusion_matrix(y_test, pred_test)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Normal','Anomaly'], yticklabels=['Normal','Anomaly'])
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
ax.set_title('Transformer — Confusion Matrix (Test)')
fig.tight_layout()
fig.savefig(FIGS / 'confusion_matrix.png', dpi=120, bbox_inches='tight')
plt.close(fig)

# ── 9. ROC and PR curves ──────────────────────────────────────────────────────
cv = curves(y_test, p_test)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(cv['roc']['fpr'], cv['roc']['tpr'], lw=2,
        label=f"Transformer (AUC={test_metrics['roc_auc']:.4f})", color='#4C72B0')
ax.plot([0,1],[0,1],'k--',lw=1)
ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
ax.set_title('Transformer — ROC Curve (Test)'); ax.legend()
fig.tight_layout()
fig.savefig(FIGS / 'roc_curve.png', dpi=120, bbox_inches='tight')
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(cv['pr']['recall'], cv['pr']['precision'], lw=2,
        label=f"Transformer (PR-AUC={test_metrics['pr_auc']:.4f})", color='#4C72B0')
ax.axhline(y_test.mean(), color='grey', ls='--', lw=1,
           label=f'No-skill ({y_test.mean():.3f})')
ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
ax.set_title('Transformer — PR Curve (Test)'); ax.legend()
fig.tight_layout()
fig.savefig(FIGS / 'pr_curve.png', dpi=120, bbox_inches='tight')
plt.close(fig)

# ── 10. Attention analysis ────────────────────────────────────────────────────
print("Extracting attention weights...")
try:
    # Use one normal and one anomaly sample from test set
    normal_idx  = int(np.where(y_test == 0)[0][0])
    anomaly_idx = int(np.where(y_test == 1)[0][0])

    for label_name, idx in [('normal', normal_idx), ('anomaly', anomaly_idx)]:
        x_sample = torch.tensor(
            test_ds.windows[idx:idx+1], dtype=torch.float32).to(DEVICE)
        attn_weights = final_model.get_attention_weights(x_sample)
        # attn_weights[layer]: (1, nhead, seq_len, seq_len)
        # Average over heads and batch for visualisation
        fig, axes = plt.subplots(1, len(attn_weights),
                                 figsize=(5 * len(attn_weights), 4))
        if len(attn_weights) == 1:
            axes = [axes]
        for layer_idx, (ax, aw) in enumerate(zip(axes, attn_weights)):
            avg_aw = aw[0].mean(dim=0).numpy()   # (seq_len, seq_len)
            im = ax.imshow(avg_aw, cmap='viridis', aspect='auto')
            ax.set_title(f'Layer {layer_idx+1}', fontsize=9)
            ax.set_xlabel('Key position'); ax.set_ylabel('Query position')
            plt.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(f'Attention Weights — {label_name} sequence '
                     f'(prob={p_test[idx]:.3f})', fontsize=10)
        fig.tight_layout()
        fig.savefig(FIGS / 'attention' / f'attention_{label_name}.png',
                    dpi=120, bbox_inches='tight')
        plt.close(fig)
    print("  Attention figures saved.")
except Exception as e:
    print(f"  Attention extraction skipped: {e}")
    with open(FIGS / 'attention' / 'attention_note.txt', 'w') as f:
        f.write(f"Attention extraction attempted but failed: {e}\n"
                "This is a known limitation of PyTorch TransformerEncoderLayer "
                "when need_weights=True interacts with batch_first=True in some versions.")

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
    ('TP — Detected anomaly', tp_idx[0] if len(tp_idx) else None),
    ('FN — Missed anomaly',   fn_idx[0] if len(fn_idx) else None),
    ('FP — False alarm',      fp_idx[0] if len(fp_idx) else None),
    ('TN — Correct normal',   tn_idx[0] if len(tn_idx) else None),
]
fig, axes = plt.subplots(4, 1, figsize=(14, 12))
for ax, (title, idx) in zip(axes, cases):
    if idx is None:
        ax.set_visible(False); continue
    seq = test_ds.windows[idx]
    for ch in range(min(4, seq.shape[1])):
        ax.plot(seq[:, ch], lw=0.8, alpha=0.7, label=feat_cols[ch])
    ax.set_title(f"{title} | prob={p_test[idx]:.3f} | "
                 f"true={y_test[idx]} pred={pred_test[idx]}", fontsize=9)
    ax.legend(fontsize=7, loc='upper right', ncol=4)
    ax.tick_params(labelsize=7)
fig.suptitle('Transformer — Representative Sequences', fontsize=12, fontweight='bold')
fig.tight_layout()
fig.savefig(FIGS / 'error_analysis' / 'representative_sequences.png',
            dpi=120, bbox_inches='tight')
plt.close(fig)

# ── 12. Inference time ────────────────────────────────────────────────────────
print("Measuring inference time...")
final_model.eval()
times = []
for _ in range(3):
    t0 = time.perf_counter()
    with torch.no_grad():
        for X_b, _ in test_loader:
            _ = torch.sigmoid(final_model(X_b.to(DEVICE)))
    times.append(time.perf_counter() - t0)
inf_s  = float(np.median(times))
inf_ms = inf_s / len(test_ds) * 1000
print(f"  Total: {inf_s*1000:.1f}ms | Per-sample: {inf_ms:.4f}ms")
pd.DataFrame([{'model':'Transformer','total_s':round(inf_s,6),
               'per_sample_ms':round(inf_ms,6),'n_samples':len(test_ds),
               'device':DEVICE}]
             ).to_csv(RES / 'inference_time.csv', index=False)

# ── 13. Model size ────────────────────────────────────────────────────────────
size_kb = (MODELS_DIR / 'transformer' / 'best_transformer.pt').stat().st_size / 1024
pd.DataFrame([{'model':'Transformer','size_kb':round(size_kb,2),
               'n_params':final_model.n_params}]
             ).to_csv(RES / 'model_size.csv', index=False)
print(f"  Model size: {size_kb:.1f} KB | Params: {final_model.n_params:,}")

# ── 14. Save model config ─────────────────────────────────────────────────────
with open(MODELS_DIR / 'transformer' / 'config.json', 'w') as f:
    json.dump({**best_hp, 'n_features': N_FEATURES,
               'sequence_length': SEQUENCE_LENGTH, 'stride': STRIDE,
               'threshold': round(threshold,4),
               'best_epoch': final_history['best_epoch'],
               'n_params': final_model.n_params, 'device': DEVICE}, f, indent=2)

# ── 15. All-models comparison ─────────────────────────────────────────────────
print("\nBuilding all-models comparison table...")
baseline_df = pd.read_csv('results/baselines/baseline_comparison.csv')
lstm_df     = pd.read_csv('results/model_comparison/classical_vs_lstm.csv')
lstm_row    = lstm_df[lstm_df['model'] == 'LSTM']

tr_row = pd.DataFrame([{
    'model':         'Transformer',
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

all_df = pd.concat([baseline_df, lstm_row, tr_row], ignore_index=True)
comp_path = Path('results/model_comparison')
comp_path.mkdir(parents=True, exist_ok=True)
all_df.to_csv(comp_path / 'all_models.csv', index=False)

md = ['# All Models Comparison (Test Set)\n',
      '| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR | Inf(ms) | Size(KB) |',
      '|---|---|---|---|---|---|---|---|---|---|---|']
for _, row in all_df.iterrows():
    md.append(f"| {row['model']} | {row['accuracy']:.4f} | {row['precision']:.4f} | "
              f"{row['recall']:.4f} | {row['f1']:.4f} | {row['roc_auc']:.4f} | "
              f"{row['pr_auc']:.4f} | {row['fpr']:.4f} | {row['fnr']:.4f} | "
              f"{row['inference_ms']:.4f} | {row['model_size_kb']:.1f} |")
with open(comp_path / 'all_models.md', 'w') as f:
    f.write('\n'.join(md))

# ── 16. Comparison bar chart ──────────────────────────────────────────────────
comp_figs = Path('figures/model_comparison')
comp_figs.mkdir(parents=True, exist_ok=True)

metrics_to_plot = ['f1','precision','recall','roc_auc','pr_auc']
x = np.arange(len(all_df))
width = 0.15
fig, ax = plt.subplots(figsize=(14, 5))
for i, metric in enumerate(metrics_to_plot):
    vals = all_df[metric].tolist()
    bars = ax.bar(x + i*width, vals, width, label=metric.upper(),
                  color=plt.cm.Set2(i/len(metrics_to_plot)))
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f'{v:.3f}', ha='center', va='bottom', fontsize=6)
ax.set_xticks(x + width*(len(metrics_to_plot)-1)/2)
ax.set_xticklabels(all_df['model'].tolist(), fontsize=8, rotation=10)
ax.set_ylim(0, 1.18); ax.set_ylabel('Score')
ax.set_title('All Models — Test Set Metrics Comparison')
ax.legend(fontsize=8); fig.tight_layout()
fig.savefig(comp_figs / 'all_models_metrics.png', dpi=120, bbox_inches='tight')
plt.close(fig)

# Inference time comparison
fig, ax = plt.subplots(figsize=(9, 4))
colors_inf = ['#4C72B0','#55A868','#DD8452','#C44E52','#8172B2']
bars = ax.bar(all_df['model'], all_df['inference_ms'],
              color=colors_inf[:len(all_df)])
for bar, v in zip(bars, all_df['inference_ms']):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.00002,
            f'{v:.4f}', ha='center', va='bottom', fontsize=8)
ax.set_ylabel('Inference time (ms/sample)')
ax.set_title('Inference Time Comparison')
ax.tick_params(axis='x', rotation=10)
fig.tight_layout()
fig.savefig(comp_figs / 'inference_time_comparison.png', dpi=120, bbox_inches='tight')
plt.close(fig)

# Model size comparison
fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.bar(all_df['model'], all_df['model_size_kb'],
              color=colors_inf[:len(all_df)])
for bar, v in zip(bars, all_df['model_size_kb']):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+10,
            f'{v:.0f}', ha='center', va='bottom', fontsize=8)
ax.set_ylabel('Model size (KB)')
ax.set_title('Model Size Comparison')
ax.tick_params(axis='x', rotation=10)
fig.tight_layout()
fig.savefig(comp_figs / 'model_size_comparison.png', dpi=120, bbox_inches='tight')
plt.close(fig)

# ── 17. Final summary ─────────────────────────────────────────────────────────
print("\n" + "="*65)
print("PHASE 5 RESULTS — ALL MODELS (TEST SET)")
print("="*65)
print(all_df.to_string(index=False))
print(f"\nBest HP : {best_hp}")
print(f"Threshold: {threshold:.4f} | Best epoch: {final_history['best_epoch']}")
print(f"Model    : {size_kb:.1f} KB | {final_model.n_params:,} params")
