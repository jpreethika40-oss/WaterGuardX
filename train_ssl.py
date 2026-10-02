"""
Phase 6 — Self-Supervised Masked Time-Series Reconstruction + Fine-tuning.
Run from project root: python train_ssl.py

Pipeline:
  1. SSL pretraining on normal-only training windows (masked reconstruction)
  2. Reconstruction quality analysis on validation set
  3. Fine-tuning pretrained encoder for anomaly classification
  4. Comparison: Standard Transformer (Phase 5) vs SSL Transformer
  5. Masking-ratio ablation (mask_ratio in {0.10, 0.15, 0.30})
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
from pathlib import Path
from sklearn.metrics import confusion_matrix, f1_score

from data_loader import load_raw
from preprocessing import run_preprocessing
from datasets import WaterSensorDataset, MaskedSensorDataset
from self_supervised import (SSLTransformer, masked_mse_loss,
                              pretrain_ssl, finetune_epoch)
from evaluation import compute_metrics, select_threshold_f1, curves
from training import set_seed
from config import SEED, SEQUENCE_LENGTH, MODELS_DIR, RESULTS_DIR, FIGURES_DIR

set_seed(SEED)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")

# ── Output directories ────────────────────────────────────────────────────────
SSL_MODELS   = MODELS_DIR   / 'ssl';          SSL_MODELS.mkdir(parents=True, exist_ok=True)
SSL_RES      = RESULTS_DIR  / 'ssl';          SSL_RES.mkdir(parents=True, exist_ok=True)
SSL_FIGS     = FIGURES_DIR  / 'ssl';          SSL_FIGS.mkdir(parents=True, exist_ok=True)
EXP_DIR      = RESULTS_DIR  / 'experiments';  EXP_DIR.mkdir(parents=True, exist_ok=True)
COMP_RES     = RESULTS_DIR  / 'model_comparison'
COMP_FIGS    = FIGURES_DIR  / 'model_comparison'

# ── SSL configuration ─────────────────────────────────────────────────────────
SSL_CFG = {
    'mask_ratio':        0.15,
    'd_model':           32,
    'nhead':             2,
    'num_layers':        2,
    'dim_feedforward':   64,
    'dropout':           0.1,
    'lr':                1e-3,
    'weight_decay':      1e-4,
    'batch_size':        256,
    'pretrain_epochs':   10,
    'finetune_epochs':   5,
    'patience_pretrain': 5,
    'patience_finetune': 3,
    'seed':              SEED,
    'stride':            12,
}

STRIDE     = SSL_CFG['stride']
BATCH      = SSL_CFG['batch_size']
N_FEATURES = 16   # confirmed from Phase 5 config

# ── 1. Load and preprocess data ───────────────────────────────────────────────
print("\nLoading data...")
df = load_raw()
train_df, val_df, test_df, scaler, feat_cols = run_preprocessing(df, save=False)
assert len(feat_cols) == N_FEATURES, f"Expected {N_FEATURES} features, got {len(feat_cols)}"

# Supervised datasets (same as Phase 5)
train_sup = WaterSensorDataset(train_df, feat_cols, label_strategy='last', stride=STRIDE)
val_sup   = WaterSensorDataset(val_df,   feat_cols, label_strategy='last', stride=STRIDE)
test_sup  = WaterSensorDataset(test_df,  feat_cols, label_strategy='last', stride=STRIDE)

# SSL dataset: normal-only training windows
train_ssl_ds = MaskedSensorDataset(train_df, feat_cols,
                                   stride=STRIDE,
                                   mask_ratio=SSL_CFG['mask_ratio'])
val_ssl_ds   = MaskedSensorDataset(val_df,   feat_cols,
                                   stride=STRIDE,
                                   mask_ratio=SSL_CFG['mask_ratio'])

print(f"Train supervised windows : {len(train_sup):,}  "
      f"anomaly={train_sup.labels.mean()*100:.1f}%")
print(f"Val   supervised windows : {len(val_sup):,}")
print(f"Test  supervised windows : {len(test_sup):,}")
print(f"Train SSL windows (normal only): {len(train_ssl_ds):,}")
print(f"Val   SSL windows (normal only): {len(val_ssl_ds):,}")

n_neg      = (train_sup.labels == 0).sum()
n_pos      = (train_sup.labels == 1).sum()
pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(DEVICE)
print(f"pos_weight: {pos_weight.item():.3f}")

train_ssl_loader = DataLoader(train_ssl_ds, batch_size=BATCH, shuffle=True,  num_workers=0)
val_ssl_loader   = DataLoader(val_ssl_ds,   batch_size=BATCH, shuffle=False, num_workers=0)
train_sup_loader = DataLoader(train_sup,    batch_size=BATCH, shuffle=True,  num_workers=0)
val_sup_loader   = DataLoader(val_sup,      batch_size=BATCH, shuffle=False, num_workers=0)
test_sup_loader  = DataLoader(test_sup,     batch_size=BATCH, shuffle=False, num_workers=0)

# ── 2. SSL Pretraining ────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("SSL PRETRAINING (masked reconstruction, normal-only windows)")
print(f"{'='*60}")
print(f"mask_ratio={SSL_CFG['mask_ratio']}  epochs={SSL_CFG['pretrain_epochs']}  "
      f"lr={SSL_CFG['lr']}  d_model={SSL_CFG['d_model']}")

set_seed(SEED)
ssl_model = SSLTransformer(
    n_features=N_FEATURES,
    d_model=SSL_CFG['d_model'],
    nhead=SSL_CFG['nhead'],
    num_layers=SSL_CFG['num_layers'],
    dim_feedforward=SSL_CFG['dim_feedforward'],
    dropout=SSL_CFG['dropout'],
    max_seq_len=SEQUENCE_LENGTH + 10,
).to(DEVICE)
print(f"SSL model parameters: {ssl_model.n_params:,}")

ssl_history = pretrain_ssl(
    ssl_model,
    train_ssl_loader, val_ssl_loader,
    epochs=SSL_CFG['pretrain_epochs'],
    lr=SSL_CFG['lr'],
    weight_decay=SSL_CFG['weight_decay'],
    patience=SSL_CFG['patience_pretrain'],
    device=DEVICE,
    verbose=True,
)

print(f"\nBest SSL epoch : {ssl_history['best_epoch']}")
print(f"Best val loss  : {ssl_history['best_val_loss']:.6f}")
print(f"Total time     : {ssl_history['total_time_s']:.1f}s")

# Save pretrained encoder checkpoint
torch.save(ssl_model.state_dict(), SSL_MODELS / 'ssl_pretrained_encoder.pt')
print("Saved: models/ssl/ssl_pretrained_encoder.pt")

# Save SSL config
with open(SSL_MODELS / 'ssl_config.json', 'w') as f:
    json.dump({**SSL_CFG,
               'n_features': N_FEATURES,
               'sequence_length': SEQUENCE_LENGTH,
               'best_pretrain_epoch': ssl_history['best_epoch'],
               'best_val_recon_loss': round(ssl_history['best_val_loss'], 8),
               'total_pretrain_time_s': ssl_history['total_time_s'],
               'device': DEVICE}, f, indent=2)

# ── 3. SSL training loss plot ─────────────────────────────────────────────────
ep = range(1, len(ssl_history['train_loss']) + 1)
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(ep, ssl_history['train_loss'], label='Train recon loss', color='#4C72B0')
ax.plot(ep, ssl_history['val_loss'],   label='Val recon loss',   color='#DD8452')
ax.axvline(ssl_history['best_epoch'], color='green', ls='--', lw=1,
           label=f"Best ep {ssl_history['best_epoch']}")
ax.set_xlabel('Epoch'); ax.set_ylabel('Masked MSE')
ax.set_title('SSL Pretraining — Reconstruction Loss')
ax.legend(); fig.tight_layout()
fig.savefig(SSL_FIGS / 'ssl_training_loss.png', dpi=120, bbox_inches='tight')
plt.close(fig)
print("Saved: figures/ssl/ssl_training_loss.png")

# ── 4. Reconstruction quality analysis on validation set ─────────────────────
print(f"\n{'='*60}")
print("RECONSTRUCTION QUALITY ANALYSIS (validation set)")
print(f"{'='*60}")

ssl_model.eval()
all_recon_errors = []
sample_originals, sample_masked, sample_recons, sample_masks = [], [], [], []

with torch.no_grad():
    for masked_x, original_x, mask in val_ssl_loader:
        masked_x   = masked_x.to(DEVICE)
        original_x = original_x.to(DEVICE)
        mask       = mask.to(DEVICE)
        pred       = ssl_model.forward_reconstruct(masked_x)
        # Per-sample masked MSE
        diff = ((pred - original_x) ** 2) * mask
        n_m  = mask.sum(dim=(1, 2)).clamp(min=1)
        err  = diff.sum(dim=(1, 2)) / n_m
        all_recon_errors.extend(err.cpu().numpy().tolist())
        # Keep first batch for visualisation
        if len(sample_originals) == 0:
            sample_originals = original_x[:8].cpu().numpy()
            sample_masked    = masked_x[:8].cpu().numpy()
            sample_recons    = pred[:8].cpu().numpy()
            sample_masks     = mask[:8].cpu().numpy()

mean_recon_err = float(np.mean(all_recon_errors))
print(f"Mean masked MSE (val): {mean_recon_err:.6f}")

# Per-sensor reconstruction error
sensor_errors = {}
ssl_model.eval()
all_pred_list, all_orig_list, all_mask_list = [], [], []
with torch.no_grad():
    for masked_x, original_x, mask in val_ssl_loader:
        pred = ssl_model.forward_reconstruct(masked_x.to(DEVICE))
        all_pred_list.append(pred.cpu().numpy())
        all_orig_list.append(original_x.numpy())
        all_mask_list.append(mask.numpy())

all_pred = np.concatenate(all_pred_list, axis=0)   # (N, seq, feat)
all_orig = np.concatenate(all_orig_list, axis=0)
all_mask = np.concatenate(all_mask_list, axis=0)

for fi, fname in enumerate(feat_cols):
    m = all_mask[:, :, fi]
    if m.sum() == 0:
        sensor_errors[fname] = float('nan')
        continue
    err = ((all_pred[:, :, fi] - all_orig[:, :, fi]) ** 2 * m).sum() / m.sum()
    sensor_errors[fname] = float(err)

sensor_err_df = pd.DataFrame(
    [{'feature': k, 'masked_mse': round(v, 6)} for k, v in sensor_errors.items()]
).sort_values('masked_mse', ascending=False)
sensor_err_df.to_csv(SSL_RES / 'sensor_reconstruction_errors.csv', index=False)
print("\nPer-sensor masked MSE (top 5 hardest):")
print(sensor_err_df.head().to_string(index=False))

# ── Reconstruction visualisation: 3 example windows ─────────────────────────
# Show 3 sensors (first pressure, first flow, tank level) for 3 windows
SHOW_SENSORS = [0, 8, 11]   # n1 (pressure), p227 (flow), T1 (level)
SHOW_WINDOWS = [0, 1, 2]

fig, axes = plt.subplots(len(SHOW_WINDOWS), len(SHOW_SENSORS),
                         figsize=(14, 9), sharex=True)
for row, wi in enumerate(SHOW_WINDOWS):
    for col, si in enumerate(SHOW_SENSORS):
        ax = axes[row][col]
        orig  = sample_originals[wi, :, si]
        mskd  = sample_masked[wi, :, si]
        recon = sample_recons[wi, :, si]
        mk    = sample_masks[wi, :, si]

        ax.plot(orig,  lw=1.5, color='#4C72B0', label='Original')
        ax.plot(recon, lw=1.5, color='#DD8452', ls='--', label='Reconstructed')
        masked_pos = np.where(mk > 0)[0]
        if len(masked_pos):
            ax.scatter(masked_pos, orig[masked_pos], color='red',
                       s=15, zorder=5, label='Masked')
        if row == 0:
            ax.set_title(feat_cols[si], fontsize=9)
        if col == 0:
            ax.set_ylabel(f'Window {wi+1}', fontsize=8)
        ax.tick_params(labelsize=7)
        if row == 0 and col == 0:
            ax.legend(fontsize=7)

fig.suptitle('SSL Reconstruction — Original vs Reconstructed (val, 3 windows × 3 sensors)',
             fontsize=10)
fig.tight_layout()
fig.savefig(SSL_FIGS / 'reconstruction_examples.png', dpi=120, bbox_inches='tight')
plt.close(fig)
print("Saved: figures/ssl/reconstruction_examples.png")

# Per-sensor error bar chart
fig, ax = plt.subplots(figsize=(12, 4))
valid = sensor_err_df.dropna()
ax.bar(valid['feature'], valid['masked_mse'], color='#4C72B0')
ax.set_ylabel('Masked MSE'); ax.set_title('Per-Sensor Reconstruction Error (val)')
ax.tick_params(axis='x', rotation=45)
fig.tight_layout()
fig.savefig(SSL_FIGS / 'per_sensor_recon_error.png', dpi=120, bbox_inches='tight')
plt.close(fig)
print("Saved: figures/ssl/per_sensor_recon_error.png")

# ── 5. Fine-tune pretrained encoder for anomaly classification ────────────────
print(f"\n{'='*60}")
print("FINE-TUNING SSL ENCODER FOR ANOMALY DETECTION")
print(f"{'='*60}")

set_seed(SEED)
# ssl_model already holds best pretrained weights
criterion_ft = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer_ft = torch.optim.AdamW(
    ssl_model.parameters(), lr=SSL_CFG['lr'], weight_decay=SSL_CFG['weight_decay'])
scheduler_ft = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_ft, mode='min', factor=0.5, patience=2, min_lr=1e-6)

ft_history = {
    'train_loss': [], 'val_loss': [],
    'train_f1':   [], 'val_f1':   [],
}
best_val_f1_ft  = -1.0
best_state_ft   = None
best_epoch_ft   = 0
no_improve_ft   = 0
ft_start        = time.perf_counter()

for epoch in range(1, SSL_CFG['finetune_epochs'] + 1):
    tr = finetune_epoch(ssl_model, train_sup_loader, criterion_ft,
                        optimizer_ft, DEVICE, training=True)
    vl = finetune_epoch(ssl_model, val_sup_loader,   criterion_ft,
                        None,         DEVICE, training=False)

    tr_f1 = f1_score(tr['labels'], (tr['probs'] >= 0.5).astype(int), zero_division=0)
    vl_f1 = f1_score(vl['labels'], (vl['probs'] >= 0.5).astype(int), zero_division=0)

    ft_history['train_loss'].append(tr['loss'])
    ft_history['val_loss'].append(vl['loss'])
    ft_history['train_f1'].append(tr_f1)
    ft_history['val_f1'].append(vl_f1)

    scheduler_ft.step(vl['loss'])
    print(f"  FT Epoch {epoch:3d} | "
          f"train_loss={tr['loss']:.4f}  val_loss={vl['loss']:.4f} | "
          f"train_f1={tr_f1:.4f}  val_f1={vl_f1:.4f}")

    if vl_f1 > best_val_f1_ft + 1e-4:
        best_val_f1_ft = vl_f1
        best_state_ft  = {k: v.cpu().clone() for k, v in ssl_model.state_dict().items()}
        best_epoch_ft  = epoch
        no_improve_ft  = 0
    else:
        no_improve_ft += 1
        if no_improve_ft >= SSL_CFG['patience_finetune']:
            print(f"  Early stopping at epoch {epoch} (best={best_epoch_ft})")
            break

ft_total_time = time.perf_counter() - ft_start
if best_state_ft:
    ssl_model.load_state_dict(best_state_ft)

print(f"\nBest FT epoch : {best_epoch_ft}")
print(f"Best val F1   : {best_val_f1_ft:.4f}")
print(f"FT time       : {ft_total_time:.1f}s")

torch.save(ssl_model.state_dict(), SSL_MODELS / 'ssl_finetuned.pt')
print("Saved: models/ssl/ssl_finetuned.pt")

# Fine-tuning loss curve
ep_ft = range(1, len(ft_history['train_loss']) + 1)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(ep_ft, ft_history['train_loss'], label='Train', color='#4C72B0')
axes[0].plot(ep_ft, ft_history['val_loss'],   label='Val',   color='#DD8452')
axes[0].axvline(best_epoch_ft, color='green', ls='--', lw=1,
                label=f'Best ep {best_epoch_ft}')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('BCE Loss')
axes[0].set_title('SSL Fine-tuning — Loss'); axes[0].legend()
axes[1].plot(ep_ft, ft_history['train_f1'], label='Train F1', color='#4C72B0')
axes[1].plot(ep_ft, ft_history['val_f1'],   label='Val F1',   color='#DD8452')
axes[1].axvline(best_epoch_ft, color='green', ls='--', lw=1)
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('F1')
axes[1].set_title('SSL Fine-tuning — F1'); axes[1].legend()
fig.tight_layout()
fig.savefig(SSL_FIGS / 'finetuning_curves.png', dpi=120, bbox_inches='tight')
plt.close(fig)
print("Saved: figures/ssl/finetuning_curves.png")

# ── 6. Test evaluation — SSL Transformer ─────────────────────────────────────
print(f"\n{'='*60}")
print("TEST EVALUATION — SSL TRANSFORMER")
print(f"{'='*60}")

# Threshold selection on validation set only
val_result = finetune_epoch(ssl_model, val_sup_loader, criterion_ft,
                            None, DEVICE, training=False)
ssl_threshold = select_threshold_f1(val_result['labels'], val_result['probs'])
print(f"SSL threshold (val F1-max): {ssl_threshold:.4f}")

# Test inference
test_result = finetune_epoch(ssl_model, test_sup_loader, criterion_ft,
                             None, DEVICE, training=False)
y_test    = test_result['labels']
p_ssl     = test_result['probs']
pred_ssl  = (p_ssl >= ssl_threshold).astype(int)

ssl_metrics = compute_metrics(y_test, pred_ssl, p_ssl)
print("\nSSL Transformer test metrics:")
for k, v in ssl_metrics.items():
    if k not in ('tp','tn','fp','fn'):
        print(f"  {k}: {v:.4f}")
print(f"  TP={ssl_metrics['tp']} TN={ssl_metrics['tn']} "
      f"FP={ssl_metrics['fp']} FN={ssl_metrics['fn']}")

# Inference time
ssl_model.eval()
inf_times = []
for _ in range(3):
    t0 = time.perf_counter()
    with torch.no_grad():
        for X_b, _ in test_sup_loader:
            ssl_model.forward_classify(X_b.to(DEVICE))
    inf_times.append(time.perf_counter() - t0)
ssl_inf_s  = float(np.median(inf_times))
ssl_inf_ms = ssl_inf_s / len(test_sup) * 1000

# Model size
ssl_size_kb = (SSL_MODELS / 'ssl_finetuned.pt').stat().st_size / 1024
print(f"\nInference: {ssl_inf_ms:.4f} ms/sample | Size: {ssl_size_kb:.1f} KB")

# Save SSL test predictions and metrics
pd.DataFrame({'true_label': y_test, 'pred_label': pred_ssl,
              'probability': p_ssl.round(4)}
             ).to_csv(SSL_RES / 'test_predictions.csv', index=False)

pd.DataFrame([{'model': 'SSL_Transformer',
               **{k: round(v, 4) for k, v in ssl_metrics.items()
                  if k not in ('tp','tn','fp','fn')},
               'threshold': round(ssl_threshold, 4),
               'best_pretrain_epoch': ssl_history['best_epoch'],
               'best_finetune_epoch': best_epoch_ft,
               'n_params': ssl_model.n_params}]
             ).to_csv(SSL_RES / 'final_metrics.csv', index=False)

# ── 7. Comparison: Standard Transformer vs SSL Transformer ───────────────────
print(f"\n{'='*60}")
print("COMPARISON: STANDARD TRANSFORMER vs SSL TRANSFORMER")
print(f"{'='*60}")

# Load Phase 5 results
phase5_df = pd.read_csv(RESULTS_DIR / 'transformer' / 'final_metrics.csv')
phase5    = phase5_df.iloc[0]

comp_rows = [
    {
        'model':         'Transformer (Phase 5)',
        'ssl_used':      False,
        'accuracy':      phase5['accuracy'],
        'precision':     phase5['precision'],
        'recall':        phase5['recall'],
        'f1':            phase5['f1'],
        'roc_auc':       phase5['roc_auc'],
        'pr_auc':        phase5['pr_auc'],
        'fpr':           phase5['fpr'],
        'fnr':           phase5['fnr'],
        'threshold':     phase5['threshold'],
        'n_params':      phase5['n_params'],
    },
    {
        'model':         'SSL Transformer (Phase 6)',
        'ssl_used':      True,
        'accuracy':      round(ssl_metrics['accuracy'],  4),
        'precision':     round(ssl_metrics['precision'], 4),
        'recall':        round(ssl_metrics['recall'],    4),
        'f1':            round(ssl_metrics['f1'],        4),
        'roc_auc':       round(ssl_metrics['roc_auc'],   4),
        'pr_auc':        round(ssl_metrics['pr_auc'],    4),
        'fpr':           round(ssl_metrics['fpr'],       4),
        'fnr':           round(ssl_metrics['fnr'],       4),
        'threshold':     round(ssl_threshold,            4),
        'n_params':      ssl_model.n_params,
    },
]
comp_df = pd.DataFrame(comp_rows)
comp_df.to_csv(SSL_RES / 'transformer_vs_ssl.csv', index=False)
print(comp_df[['model','f1','roc_auc','pr_auc','precision','recall','fpr','fnr']
              ].to_string(index=False))

# Comparison bar chart
metrics_plot = ['f1', 'precision', 'recall', 'roc_auc', 'pr_auc']
x = np.arange(len(metrics_plot))
width = 0.35
fig, ax = plt.subplots(figsize=(10, 5))
bars1 = ax.bar(x - width/2, [comp_rows[0][m] for m in metrics_plot],
               width, label='Standard Transformer', color='#4C72B0')
bars2 = ax.bar(x + width/2, [comp_rows[1][m] for m in metrics_plot],
               width, label='SSL Transformer',      color='#DD8452')
for bars in [bars1, bars2]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels([m.upper() for m in metrics_plot])
ax.set_ylim(0, 1.15); ax.set_ylabel('Score')
ax.set_title('Standard Transformer vs SSL Transformer — Test Set')
ax.legend(); fig.tight_layout()
fig.savefig(SSL_FIGS / 'transformer_vs_ssl_comparison.png', dpi=120, bbox_inches='tight')
plt.close(fig)
print("Saved: figures/ssl/transformer_vs_ssl_comparison.png")

# ── 8. Masking-ratio ablation ─────────────────────────────────────────────────
print(f"\n{'='*60}")
print("MASKING-RATIO ABLATION (mask_ratio in {0.10, 0.15, 0.30})")
print(f"{'='*60}")
print("NOTE: CPU-constrained — 5 pretrain + 5 finetune epochs per config.")

ABLATION_RATIOS = [0.10, 0.15, 0.30]
ablation_rows   = []

for mr in ABLATION_RATIOS:
    print(f"\n--- mask_ratio={mr} ---")
    set_seed(SEED)

    abl_ssl_ds  = MaskedSensorDataset(train_df, feat_cols, stride=STRIDE, mask_ratio=mr)
    abl_val_ds  = MaskedSensorDataset(val_df,   feat_cols, stride=STRIDE, mask_ratio=mr)
    abl_ssl_ldr = DataLoader(abl_ssl_ds, batch_size=BATCH, shuffle=True,  num_workers=0)
    abl_val_ldr = DataLoader(abl_val_ds, batch_size=BATCH, shuffle=False, num_workers=0)

    abl_model = SSLTransformer(
        n_features=N_FEATURES,
        d_model=SSL_CFG['d_model'], nhead=SSL_CFG['nhead'],
        num_layers=SSL_CFG['num_layers'],
        dim_feedforward=SSL_CFG['dim_feedforward'],
        dropout=SSL_CFG['dropout'],
        max_seq_len=SEQUENCE_LENGTH + 10,
    ).to(DEVICE)

    # Pretrain 5 epochs
    abl_pre_hist = pretrain_ssl(
        abl_model, abl_ssl_ldr, abl_val_ldr,
        epochs=5, lr=SSL_CFG['lr'], weight_decay=SSL_CFG['weight_decay'],
        patience=3, device=DEVICE, verbose=False,
    )
    print(f"  Pretrain best val loss: {abl_pre_hist['best_val_loss']:.6f} "
          f"(epoch {abl_pre_hist['best_epoch']})")

    # Fine-tune 5 epochs
    crit_abl = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt_abl  = torch.optim.AdamW(abl_model.parameters(),
                                  lr=SSL_CFG['lr'], weight_decay=SSL_CFG['weight_decay'])
    sch_abl  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt_abl, mode='min', factor=0.5, patience=2, min_lr=1e-6)

    best_vf1_abl, best_st_abl = -1.0, None
    for ep in range(1, 6):
        tr_abl = finetune_epoch(abl_model, train_sup_loader, crit_abl,
                                opt_abl, DEVICE, training=True)
        vl_abl = finetune_epoch(abl_model, val_sup_loader,   crit_abl,
                                None,     DEVICE, training=False)
        sch_abl.step(vl_abl['loss'])
        vf1 = f1_score(vl_abl['labels'],
                       (vl_abl['probs'] >= 0.5).astype(int), zero_division=0)
        if vf1 > best_vf1_abl:
            best_vf1_abl = vf1
            best_st_abl  = {k: v.cpu().clone() for k, v in abl_model.state_dict().items()}

    if best_st_abl:
        abl_model.load_state_dict(best_st_abl)

    # Threshold + test eval
    vl_res_abl = finetune_epoch(abl_model, val_sup_loader, crit_abl,
                                None, DEVICE, training=False)
    thr_abl    = select_threshold_f1(vl_res_abl['labels'], vl_res_abl['probs'])
    te_res_abl = finetune_epoch(abl_model, test_sup_loader, crit_abl,
                                None, DEVICE, training=False)
    pred_abl   = (te_res_abl['probs'] >= thr_abl).astype(int)
    met_abl    = compute_metrics(te_res_abl['labels'], pred_abl, te_res_abl['probs'])

    print(f"  F1={met_abl['f1']:.4f}  ROC-AUC={met_abl['roc_auc']:.4f}  "
          f"PR-AUC={met_abl['pr_auc']:.4f}")

    ablation_rows.append({
        'mask_ratio':        mr,
        'pretrain_best_val_loss': round(abl_pre_hist['best_val_loss'], 6),
        'pretrain_best_epoch':    abl_pre_hist['best_epoch'],
        'f1':        round(met_abl['f1'],        4),
        'precision': round(met_abl['precision'], 4),
        'recall':    round(met_abl['recall'],    4),
        'roc_auc':   round(met_abl['roc_auc'],   4),
        'pr_auc':    round(met_abl['pr_auc'],    4),
        'fpr':       round(met_abl['fpr'],        4),
        'fnr':       round(met_abl['fnr'],        4),
    })

abl_df = pd.DataFrame(ablation_rows)
abl_df.to_csv(SSL_RES / 'masking_ratio_ablation.csv', index=False)
print("\nAblation results:")
print(abl_df.to_string(index=False))

# Ablation plot
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, metric in zip(axes, ['f1', 'roc_auc', 'pretrain_best_val_loss']):
    ax.bar([str(r) for r in abl_df['mask_ratio']], abl_df[metric],
           color=['#4C72B0','#DD8452','#55A868'])
    ax.set_xlabel('Mask ratio'); ax.set_ylabel(metric)
    ax.set_title(f'Ablation — {metric}')
    for i, v in enumerate(abl_df[metric]):
        ax.text(i, v + 0.001, f'{v:.4f}', ha='center', va='bottom', fontsize=9)
fig.suptitle('Masking-Ratio Ablation', fontsize=11)
fig.tight_layout()
fig.savefig(SSL_FIGS / 'masking_ratio_ablation.png', dpi=120, bbox_inches='tight')
plt.close(fig)
print("Saved: figures/ssl/masking_ratio_ablation.png")

# ── 9. Error analysis — SSL vs Standard Transformer ──────────────────────────
print(f"\n{'='*60}")
print("ERROR ANALYSIS — SSL vs STANDARD TRANSFORMER")
print(f"{'='*60}")

# Load Phase 5 test predictions
p5_preds = pd.read_csv(RESULTS_DIR / 'transformer' / 'test_predictions.csv')
y_true_p5   = p5_preds['true_label'].values
pred_p5     = p5_preds['pred_label'].values

fp_p5  = np.where((pred_p5==1)  & (y_true_p5==0))[0]
fn_p5  = np.where((pred_p5==0)  & (y_true_p5==1))[0]
fp_ssl = np.where((pred_ssl==1) & (y_test==0))[0]
fn_ssl = np.where((pred_ssl==0) & (y_test==1))[0]

print(f"Standard Transformer — FP={len(fp_p5)}  FN={len(fn_p5)}")
print(f"SSL Transformer      — FP={len(fp_ssl)}  FN={len(fn_ssl)}")

# Errors unique to each model
fp_only_p5  = set(fp_p5)  - set(fp_ssl)
fp_only_ssl = set(fp_ssl) - set(fp_p5)
fn_only_p5  = set(fn_p5)  - set(fn_ssl)
fn_only_ssl = set(fn_ssl) - set(fn_p5)
print(f"\nFP only in Standard: {len(fp_only_p5)}  |  FP only in SSL: {len(fp_only_ssl)}")
print(f"FN only in Standard: {len(fn_only_p5)}  |  FN only in SSL: {len(fn_only_ssl)}")

err_rows = []
for idx in list(fp_only_p5)[:5]:
    err_rows.append({'error_type':'FP_standard_only','window_idx':int(idx),
                     'prob_standard':round(float(p5_preds['probability'].iloc[idx]),4),
                     'prob_ssl':round(float(p_ssl[idx]),4)})
for idx in list(fp_only_ssl)[:5]:
    err_rows.append({'error_type':'FP_ssl_only','window_idx':int(idx),
                     'prob_standard':round(float(p5_preds['probability'].iloc[idx]),4),
                     'prob_ssl':round(float(p_ssl[idx]),4)})
for idx in list(fn_only_p5)[:5]:
    err_rows.append({'error_type':'FN_standard_only','window_idx':int(idx),
                     'prob_standard':round(float(p5_preds['probability'].iloc[idx]),4),
                     'prob_ssl':round(float(p_ssl[idx]),4)})
for idx in list(fn_only_ssl)[:5]:
    err_rows.append({'error_type':'FN_ssl_only','window_idx':int(idx),
                     'prob_standard':round(float(p5_preds['probability'].iloc[idx]),4),
                     'prob_ssl':round(float(p_ssl[idx]),4)})
pd.DataFrame(err_rows).to_csv(SSL_RES / 'error_analysis.csv', index=False)

# ── 10. Experiment tracking ───────────────────────────────────────────────────
exp_id = f"phase6_ssl_{int(time.time())}"
exp_record = {
    'experiment_id':       exp_id,
    'model_name':          'SSL_Transformer',
    'ssl_used':            True,
    'mask_ratio':          SSL_CFG['mask_ratio'],
    'pretraining_epochs':  ssl_history['best_epoch'],
    'learning_rate':       SSL_CFG['lr'],
    'embedding_dimension': SSL_CFG['d_model'],
    'num_layers':          SSL_CFG['num_layers'],
    'num_heads':           SSL_CFG['nhead'],
    'fine_tuning_epochs':  best_epoch_ft,
    'precision':           round(ssl_metrics['precision'], 4),
    'recall':              round(ssl_metrics['recall'],    4),
    'f1':                  round(ssl_metrics['f1'],        4),
    'roc_auc':             round(ssl_metrics['roc_auc'],   4),
    'pr_auc':              round(ssl_metrics['pr_auc'],    4),
    'false_positive_rate': round(ssl_metrics['fpr'],       4),
    'false_negative_rate': round(ssl_metrics['fnr'],       4),
    'pretrain_time_s':     ssl_history['total_time_s'],
    'finetune_time_s':     round(ft_total_time, 2),
    'inference_ms':        round(ssl_inf_ms, 6),
    'model_size_kb':       round(ssl_size_kb, 2),
    'n_params':            ssl_model.n_params,
    'device':              DEVICE,
}

exp_path = EXP_DIR / 'experiments.csv'
exp_df   = pd.DataFrame([exp_record])
if exp_path.exists():
    existing = pd.read_csv(exp_path)
    exp_df   = pd.concat([existing, exp_df], ignore_index=True)
exp_df.to_csv(exp_path, index=False)
print(f"\nExperiment logged: {exp_id}")

# ── 11. Update all-models comparison table ────────────────────────────────────
all_models_path = COMP_RES / 'all_models.csv'
if all_models_path.exists():
    all_df = pd.read_csv(all_models_path)
    ssl_row = pd.DataFrame([{
        'model':         'SSL_Transformer',
        'accuracy':      round(ssl_metrics['accuracy'],  4),
        'precision':     round(ssl_metrics['precision'], 4),
        'recall':        round(ssl_metrics['recall'],    4),
        'f1':            round(ssl_metrics['f1'],        4),
        'roc_auc':       round(ssl_metrics['roc_auc'],   4),
        'pr_auc':        round(ssl_metrics['pr_auc'],    4),
        'fpr':           round(ssl_metrics['fpr'],       4),
        'fnr':           round(ssl_metrics['fnr'],       4),
        'inference_ms':  round(ssl_inf_ms,  6),
        'model_size_kb': round(ssl_size_kb, 2),
    }])
    all_df = pd.concat([all_df, ssl_row], ignore_index=True)
    all_df.to_csv(COMP_RES / 'all_models.csv', index=False)
    print("Updated: results/model_comparison/all_models.csv")

# ── 12. Final summary ─────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("PHASE 6 FINAL SUMMARY")
print(f"{'='*65}")
print(f"\nSSL Architecture  : SSLTransformer  d_model={SSL_CFG['d_model']}  "
      f"nhead={SSL_CFG['nhead']}  layers={SSL_CFG['num_layers']}  "
      f"ff={SSL_CFG['dim_feedforward']}")
print(f"Parameters        : {ssl_model.n_params:,}")
print(f"Masking strategy  : random {SSL_CFG['mask_ratio']*100:.0f}% of (timestep, feature) positions")
print(f"Pretrain epochs   : {ssl_history['best_epoch']} (best)  |  "
      f"val recon loss={ssl_history['best_val_loss']:.6f}")
print(f"Pretrain time     : {ssl_history['total_time_s']:.1f}s")
print(f"Fine-tune epochs  : {best_epoch_ft} (best)  |  val F1={best_val_f1_ft:.4f}")
print(f"Fine-tune time    : {ft_total_time:.1f}s")
print(f"\nStandard Transformer (Phase 5):")
print(f"  F1={phase5['f1']:.4f}  ROC-AUC={phase5['roc_auc']:.4f}  "
      f"PR-AUC={phase5['pr_auc']:.4f}  FPR={phase5['fpr']:.4f}  FNR={phase5['fnr']:.4f}")
print(f"\nSSL Transformer (Phase 6):")
print(f"  F1={ssl_metrics['f1']:.4f}  ROC-AUC={ssl_metrics['roc_auc']:.4f}  "
      f"PR-AUC={ssl_metrics['pr_auc']:.4f}  FPR={ssl_metrics['fpr']:.4f}  "
      f"FNR={ssl_metrics['fnr']:.4f}")
print(f"\nInference         : {ssl_inf_ms:.4f} ms/sample")
print(f"Model size        : {ssl_size_kb:.1f} KB")
print(f"\nMasking-ratio ablation:")
print(abl_df[['mask_ratio','pretrain_best_val_loss','f1','roc_auc']].to_string(index=False))
print(f"\nMean val reconstruction error (mask_ratio=0.15): {mean_recon_err:.6f}")
print(f"\nAll results saved to: results/ssl/")
print(f"Figures saved to    : figures/ssl/")
print(f"Checkpoints saved to: models/ssl/")
