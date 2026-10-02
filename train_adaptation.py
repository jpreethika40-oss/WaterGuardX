"""
Phase 8 — Adaptive Learning Under Distribution Shift.
Run from project root: python train_adaptation.py

This script implements:
1. Shift-triggered adaptation using the Phase 7 drift detector.
2. Chronological data splitting (Adaptation pool Nov 2018 vs. Evaluation pool Dec 2018).
3. Replay buffer mechanism (20% original training samples) to prevent catastrophic forgetting.
4. Controlled periodic fine-tuning of the SSL-pretrained Temporal Transformer.
5. Strict safety check & acceptance criteria before model promotion.
6. 4-way evaluation (Original vs. Adapted on Normal vs. Shifted).
7. Comprehensive error analysis, cost profiling, and visualization.
"""
import copy
import json
import shutil
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, 'src')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from adaptation import (build_replay_dataset, check_acceptance_criteria,
                        evaluate_model_on_data, load_ssl_model,
                        run_controlled_adaptation,
                        split_chronological_adaptation)
from config import (DATA_PROC, FIGURES_DIR, MODELS_DIR, RESULTS_DIR, SEED,
                    SENSOR_COLS, SEQUENCE_LENGTH, STRIDE, TARGET_COL,
                    TIMESTAMP_COL)
from data_loader import load_raw
from drift_detection import DriftDetector
from evaluation import compute_metrics, select_threshold_f1
from preprocessing import run_preprocessing
from training import set_seed

set_seed(SEED)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")

# Output directories
ADAPT_RES = RESULTS_DIR / 'adaptation'; ADAPT_RES.mkdir(parents=True, exist_ok=True)
ADAPT_FIGS = FIGURES_DIR / 'adaptation'; ADAPT_FIGS.mkdir(parents=True, exist_ok=True)
ADAPT_MOD = MODELS_DIR / 'adaptation'; ADAPT_MOD.mkdir(parents=True, exist_ok=True)
EXP_DIR = RESULTS_DIR / 'experiments'; EXP_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Load Data ──────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("1. LOADING DATA AND PREPROCESSING")
print("="*65)

df_raw = load_raw()
train_df, val_df, test_df, scaler, feat_cols = run_preprocessing(df_raw, save=False)
N_FEATURES = len(feat_cols)
print(f"Features: {N_FEATURES}  |  Sensor cols: {len(SENSOR_COLS)}")
print(f"Train: {len(train_df):,} rows  |  Val: {len(val_df):,}  |  Normal Test: {len(test_df):,}")

# Load shifted test data from Phase 7 (strong shift, k=1.0)
shifted_test_path = DATA_PROC / 'shifted' / 'strong' / 'test_shifted.csv'
if not shifted_test_path.exists():
    raise FileNotFoundError(f"Missing shifted test dataset at {shifted_test_path}. Run Phase 7 first.")

test_shifted_df = pd.read_csv(shifted_test_path)
print(f"Loaded shifted test dataset: {len(test_shifted_df):,} rows from {shifted_test_path}")

# ── 2. Shift Detection Check (Trigger) ────────────────────────────────────────
print("\n" + "="*65)
print("2. SHIFT DETECTION CHECK (ADAPTATION TRIGGER)")
print("="*65)

# Load detector config and reference distribution from Phase 7
detector = DriftDetector(n_bootstrap=500, bootstrap_frac=0.3, percentile=99.0, seed=SEED)
detector.fit(train_df, feat_cols)

drift_res = detector.detect(test_shifted_df)
shift_score = drift_res['shift_score']
threshold = drift_res['threshold']
shift_detected = drift_res['shift_detected']

print(f"Reference Training Distribution: {len(train_df):,} samples")
print(f"Incoming Shift Score:           {shift_score:.6f}")
print(f"Detection Threshold:            {threshold:.6f}")
print(f"Shift Status:                   {'SHIFT DETECTED' if shift_detected else 'STABLE'}")

if not shift_detected:
    print("Distribution is STABLE. Adaptation is NOT permitted. Continuing with existing model.")
    sys.exit(0)
else:
    print(">> Shift confirmed! Triggering controlled adaptation pipeline.")

# ── 3. Chronological Data Splitting (Zero Data Leakage) ───────────────────────
print("\n" + "="*65)
print("3. CHRONOLOGICAL DATA SPLITTING FOR ADAPTATION")
print("="*65)

# Split shifted test data into:
# - Adaptation Pool: November 2018 (Nov 1 to Nov 30)
#   - adapt_train_df: Nov 1 to Nov 21 (70%)
#   - adapt_val_df:   Nov 22 to Nov 30 (30%)
# - Final Evaluation Subset: December 2018 (Dec 1 to Dec 31, 8,928 rows)
adapt_train_df, adapt_val_df, eval_shifted_df = split_chronological_adaptation(
    test_shifted_df, timestamp_col=TIMESTAMP_COL, split_date='2018-12-01', val_ratio=0.30
)

# Corresponding unshifted normal evaluation subset for December 2018
test_df[TIMESTAMP_COL] = pd.to_datetime(test_df[TIMESTAMP_COL])
eval_normal_df = test_df[test_df[TIMESTAMP_COL] >= '2018-12-01'].reset_index(drop=True)

print(f"Adaptation Train pool (Nov 1-21):  {len(adapt_train_df):,} rows (anomaly={adapt_train_df[TARGET_COL].mean()*100:.2f}%)")
print(f"Adaptation Val pool (Nov 22-30):    {len(adapt_val_df):,} rows (anomaly={adapt_val_df[TARGET_COL].mean()*100:.2f}%)")
print(f"Final Shifted Eval set (Dec 1-31): {len(eval_shifted_df):,} rows (anomaly={eval_shifted_df[TARGET_COL].mean()*100:.2f}%)")
print(f"Final Normal Eval set (Dec 1-31):  {len(eval_normal_df):,} rows (anomaly={eval_normal_df[TARGET_COL].mean()*100:.2f}%)")
print("Protocol: Candidate model will fine-tune ONLY on adapt_train, tune on adapt_val, and be tested on December data.")

# ── 4. Replay Buffer Construction ─────────────────────────────────────────────
print("\n" + "="*65)
print("4. REPLAY BUFFER CONSTRUCTION (CATASTROPHIC FORGETTING SHIELD)")
print("="*65)

REPLAY_RATIO = 0.20
replay_dataset = build_replay_dataset(
    adapt_df=adapt_train_df,
    train_df=train_df,
    feat_cols=feat_cols,
    replay_ratio=REPLAY_RATIO,
    seq_len=SEQUENCE_LENGTH,
    stride=STRIDE,
    seed=SEED,
)

print(f"Total Adaptation Training Sequences: {len(replay_dataset):,}")
print(f"Replay Buffer Ratio:                 {REPLAY_RATIO*100:.0f}% reference training samples")

# ── 5. Load Original Phase 6 SSL Transformer ──────────────────────────────────
print("\n" + "="*65)
print("5. LOADING ORIGINAL PHASE 6 SSL TRANSFORMER")
print("="*65)

ssl_ckpt_path = MODELS_DIR / 'ssl' / 'ssl_finetuned.pt'
ssl_cfg_path = MODELS_DIR / 'ssl' / 'ssl_config.json'
orig_model, ssl_cfg = load_ssl_model(ssl_ckpt_path, ssl_cfg_path, device=DEVICE)

# Re-read original baseline threshold
ssl_metrics_df = pd.read_csv(RESULTS_DIR / 'ssl' / 'final_metrics.csv')
ORIG_THRESHOLD = float(ssl_metrics_df['threshold'].iloc[0])
print(f"Original Model: {orig_model.n_params:,} parameters | Base Threshold: {ORIG_THRESHOLD:.4f}")

# ── 6. Baseline (Before Adaptation) Evaluation ────────────────────────────────
print("\n" + "="*65)
print("6. BASELINE EVALUATION (BEFORE ADAPTATION)")
print("="*65)

# A. Original model on Normal Evaluation data (December)
met_orig_normal, _, _ = evaluate_model_on_data(
    orig_model, eval_normal_df, feat_cols, threshold=ORIG_THRESHOLD, device=DEVICE
)
print("Original Model on Normal Eval Set (Dec):")
print(f"  F1={met_orig_normal['f1']:.4f}  Prec={met_orig_normal['precision']:.4f}  "
      f"Rec={met_orig_normal['recall']:.4f}  ROC-AUC={met_orig_normal['roc_auc']:.4f}  "
      f"FPR={met_orig_normal['fpr']:.4f}")

# B. Original model on Shifted Evaluation data (December)
met_orig_shifted, _, _ = evaluate_model_on_data(
    orig_model, eval_shifted_df, feat_cols, threshold=ORIG_THRESHOLD, device=DEVICE
)
print("\nOriginal Model on Shifted Eval Set (Dec):")
print(f"  F1={met_orig_shifted['f1']:.4f}  Prec={met_orig_shifted['precision']:.4f}  "
      f"Rec={met_orig_shifted['recall']:.4f}  ROC-AUC={met_orig_shifted['roc_auc']:.4f}  "
      f"FPR={met_orig_shifted['fpr']:.4f}")

# Full test sets for macro reference
met_orig_full_norm, _, _ = evaluate_model_on_data(
    orig_model, test_df, feat_cols, threshold=ORIG_THRESHOLD, device=DEVICE
)
met_orig_full_shft, p_orig_full_shft, y_orig_full_shft = evaluate_model_on_data(
    orig_model, test_shifted_df, feat_cols, threshold=ORIG_THRESHOLD, device=DEVICE
)

# ── 7. Controlled Model Adaptation ────────────────────────────────────────────
print("\n" + "="*65)
print("7. EXECUTING CONTROLLED ADAPTATION FINE-TUNING")
print("="*65)

ADAPT_EPOCHS = 5
ADAPT_LR = 1e-4
ADAPT_BATCH = 128
ADAPT_WEIGHT_DECAY = 1e-4

print(f"Adaptation Hyperparameters:")
print(f"  Epochs:       {ADAPT_EPOCHS}")
print(f"  Learning Rate:{ADAPT_LR}")
print(f"  Batch Size:   {ADAPT_BATCH}")
print(f"  Optimizer:    AdamW (weight_decay={ADAPT_WEIGHT_DECAY})")

t_start = time.perf_counter()
candidate_model, cand_threshold, train_losses, val_f1s = run_controlled_adaptation(
    model=orig_model,
    train_dataset=replay_dataset,
    val_df=adapt_val_df,
    feat_cols=feat_cols,
    epochs=ADAPT_EPOCHS,
    lr=ADAPT_LR,
    weight_decay=ADAPT_WEIGHT_DECAY,
    batch_size=ADAPT_BATCH,
    device=DEVICE,
)
t_end = time.perf_counter()
adaptation_time_s = round(t_end - t_start, 2)
print(f"Adaptation completed in {adaptation_time_s:.2f} seconds.")
print(f"Candidate Decision Threshold (calibrated on adapt_val): {cand_threshold:.4f}")

for ep, (l, f) in enumerate(zip(train_losses, val_f1s), 1):
    print(f"  Epoch {ep}/{ADAPT_EPOCHS} -> Loss: {l:.4f} | Adapt Val F1: {f:.4f}")

# Save candidate model checkpoint
cand_ckpt_path = ADAPT_MOD / 'adaptation_candidate.pt'
torch.save(candidate_model.state_dict(), cand_ckpt_path)
print(f"Saved candidate model checkpoint to {cand_ckpt_path}")

# ── 8. Adaptation Safety Checks & Acceptance Criteria ─────────────────────────
print("\n" + "="*65)
print("8. ADAPTATION SAFETY CHECKS & ACCEPTANCE CRITERIA")
print("="*65)

# 1. Shifted validation evaluation (Original vs Candidate on adapt_val_df)
met_orig_shift_val, _, _ = evaluate_model_on_data(
    orig_model, adapt_val_df, feat_cols, threshold=ORIG_THRESHOLD, device=DEVICE
)
met_cand_shift_val, _, _ = evaluate_model_on_data(
    candidate_model, adapt_val_df, feat_cols, threshold=cand_threshold, device=DEVICE
)

# 2. Reference validation evaluation (Catastrophic forgetting check on unshifted counterpart Nov 22-30)
ref_val_df = test_df[(test_df[TIMESTAMP_COL] >= '2018-11-22') & (test_df[TIMESTAMP_COL] < '2018-12-01')].reset_index(drop=True)
met_orig_ref_val, _, _ = evaluate_model_on_data(
    orig_model, ref_val_df, feat_cols, threshold=ORIG_THRESHOLD, device=DEVICE
)
met_cand_ref_val, _, _ = evaluate_model_on_data(
    candidate_model, ref_val_df, feat_cols, threshold=cand_threshold, device=DEVICE
)

accepted, criteria_details = check_acceptance_criteria(
    orig_f1_shift_val=met_orig_shift_val['f1'],
    cand_f1_shift_val=met_cand_shift_val['f1'],
    orig_f1_orig_val=met_orig_ref_val['f1'],
    cand_f1_orig_val=met_cand_ref_val['f1'],
    max_orig_degradation=0.05
)

# Also log Phase 2 val_df for informational tracking
met_orig_p2_val, _, _ = evaluate_model_on_data(orig_model, val_df, feat_cols, threshold=ORIG_THRESHOLD, device=DEVICE)
met_cand_p2_val, _, _ = evaluate_model_on_data(candidate_model, val_df, feat_cols, threshold=cand_threshold, device=DEVICE)
criteria_details['phase2_val_delta'] = round(met_cand_p2_val['f1'] - met_orig_p2_val['f1'], 4)

print("Safety Verification Results:")
print(f"  Shifted Val F1 (Nov 22-30 shifted):    Original = {criteria_details['orig_f1_shift_val']:.4f} -> Candidate = {criteria_details['cand_f1_shift_val']:.4f} (Delta = {criteria_details['delta_shift_val']:+.4f}) [Pass: {criteria_details['rule1_pass']}]")
print(f"  Reference Val F1 (Nov 22-30 normal):   Original = {criteria_details['orig_f1_orig_val']:.4f} -> Candidate = {criteria_details['cand_f1_orig_val']:.4f} (Delta = {criteria_details['delta_orig_val']:+.4f}) [Pass: {criteria_details['rule2_pass']}]")
print(f"  Phase 2 Val F1 (Sep-Oct):              Original = {met_orig_p2_val['f1']:.4f} -> Candidate = {met_cand_p2_val['f1']:.4f} (Delta = {criteria_details['phase2_val_delta']:+.4f})")
print(f"  Acceptance Status:                     {'PROMOTED' if accepted else 'REJECTED'}")

adapted_ckpt_path = ADAPT_MOD / 'ssl_transformer_adapted.pt'
if accepted:
    torch.save(candidate_model.state_dict(), adapted_ckpt_path)
    print(f">> Candidate PROMOTED to active adapted model at {adapted_ckpt_path}")
    active_model = candidate_model
    ACTIVE_THRESHOLD = cand_threshold
else:
    print(">> Candidate REJECTED. Maintaining original model.")
    active_model = orig_model
    ACTIVE_THRESHOLD = ORIG_THRESHOLD

# ── 9. Adapted Model Evaluation (After Adaptation) ────────────────────────────
print("\n" + "="*65)
print("9. FINAL EVALUATION: AFTER ADAPTATION")
print("="*65)

# C. Adapted model on Shifted Evaluation data (December)
met_adapt_shifted, _, _ = evaluate_model_on_data(
    active_model, eval_shifted_df, feat_cols, threshold=ACTIVE_THRESHOLD, device=DEVICE
)
print("Adapted Model on Shifted Eval Set (Dec):")
print(f"  F1={met_adapt_shifted['f1']:.4f}  Prec={met_adapt_shifted['precision']:.4f}  "
      f"Rec={met_adapt_shifted['recall']:.4f}  ROC-AUC={met_adapt_shifted['roc_auc']:.4f}  "
      f"FPR={met_adapt_shifted['fpr']:.4f}")

# D. Adapted model on Normal Evaluation data (December)
met_adapt_normal, _, _ = evaluate_model_on_data(
    active_model, eval_normal_df, feat_cols, threshold=ACTIVE_THRESHOLD, device=DEVICE
)
print("\nAdapted Model on Normal Eval Set (Dec):")
print(f"  F1={met_adapt_normal['f1']:.4f}  Prec={met_adapt_normal['precision']:.4f}  "
      f"Rec={met_adapt_normal['recall']:.4f}  ROC-AUC={met_adapt_normal['roc_auc']:.4f}  "
      f"FPR={met_adapt_normal['fpr']:.4f}")

# Full test sets for macro comparison
met_adapt_full_shft, p_adapt_full_shft, y_adapt_full_shft = evaluate_model_on_data(
    active_model, test_shifted_df, feat_cols, threshold=ACTIVE_THRESHOLD, device=DEVICE
)
met_adapt_full_norm, _, _ = evaluate_model_on_data(
    active_model, test_df, feat_cols, threshold=ACTIVE_THRESHOLD, device=DEVICE
)

# ── 10. Performance Deltas & Final Comparison Table ───────────────────────────
print("\n" + "="*65)
print("10. 4-WAY COMPARISON TABLE")
print("="*65)

comp_records = [
    {
        'Model State': 'Original (SSL Transformer)',
        'Test Condition': 'Normal (Dec Eval)',
        'Precision': round(met_orig_normal['precision'], 4),
        'Recall': round(met_orig_normal['recall'], 4),
        'F1': round(met_orig_normal['f1'], 4),
        'ROC-AUC': round(met_orig_normal['roc_auc'], 4),
        'PR-AUC': round(met_orig_normal['pr_auc'], 4),
        'FPR': round(met_orig_normal['fpr'], 4),
        'FNR': round(met_orig_normal['fnr'], 4),
    },
    {
        'Model State': 'Original (SSL Transformer)',
        'Test Condition': 'Shifted (Dec Eval)',
        'Precision': round(met_orig_shifted['precision'], 4),
        'Recall': round(met_orig_shifted['recall'], 4),
        'F1': round(met_orig_shifted['f1'], 4),
        'ROC-AUC': round(met_orig_shifted['roc_auc'], 4),
        'PR-AUC': round(met_orig_shifted['pr_auc'], 4),
        'FPR': round(met_orig_shifted['fpr'], 4),
        'FNR': round(met_orig_shifted['fnr'], 4),
    },
    {
        'Model State': 'Adapted (SSL Transformer)',
        'Test Condition': 'Normal (Dec Eval)',
        'Precision': round(met_adapt_normal['precision'], 4),
        'Recall': round(met_adapt_normal['recall'], 4),
        'F1': round(met_adapt_normal['f1'], 4),
        'ROC-AUC': round(met_adapt_normal['roc_auc'], 4),
        'PR-AUC': round(met_adapt_normal['pr_auc'], 4),
        'FPR': round(met_adapt_normal['fpr'], 4),
        'FNR': round(met_adapt_normal['fnr'], 4),
    },
    {
        'Model State': 'Adapted (SSL Transformer)',
        'Test Condition': 'Shifted (Dec Eval)',
        'Precision': round(met_adapt_shifted['precision'], 4),
        'Recall': round(met_adapt_shifted['recall'], 4),
        'F1': round(met_adapt_shifted['f1'], 4),
        'ROC-AUC': round(met_adapt_shifted['roc_auc'], 4),
        'PR-AUC': round(met_adapt_shifted['pr_auc'], 4),
        'FPR': round(met_adapt_shifted['fpr'], 4),
        'FNR': round(met_adapt_shifted['fnr'], 4),
    },
]

comp_df = pd.DataFrame(comp_records)
comp_df.to_csv(ADAPT_RES / 'final_comparison_table.csv', index=False)
print(comp_df.to_string(index=False))

delta_f1_shift = met_adapt_shifted['f1'] - met_orig_shifted['f1']
delta_rec_shift = met_adapt_shifted['recall'] - met_orig_shifted['recall']
delta_prec_shift = met_adapt_shifted['precision'] - met_orig_shifted['precision']
delta_fpr_shift = met_adapt_shifted['fpr'] - met_orig_shifted['fpr']
delta_f1_norm = met_adapt_normal['f1'] - met_orig_normal['f1']

print("\nKey Deltas on Shifted Data:")
print(f"  F1 Recovery:         {delta_f1_shift:+.4f} ({met_orig_shifted['f1']:.4f} -> {met_adapt_shifted['f1']:.4f})")
print(f"  Precision Recovery:  {delta_prec_shift:+.4f} ({met_orig_shifted['precision']:.4f} -> {met_adapt_shifted['precision']:.4f})")
print(f"  Recall Change:       {delta_rec_shift:+.4f} ({met_orig_shifted['recall']:.4f} -> {met_adapt_shifted['recall']:.4f})")
print(f"  FPR Reduction:       {delta_fpr_shift:+.4f} ({met_orig_shifted['fpr']:.4f} -> {met_adapt_shifted['fpr']:.4f})")
print(f"  Normal Retention:    {delta_f1_norm:+.4f} (Original F1={met_orig_normal['f1']:.4f} -> Adapted F1={met_adapt_normal['f1']:.4f})")

# ── 11. Adaptation Cost Profiling ─────────────────────────────────────────────
print("\n" + "="*65)
print("11. ADAPTATION COST PROFILING")
print("="*65)

# Benchmark inference time
dummy_input = torch.randn(1, SEQUENCE_LENGTH, N_FEATURES).to(DEVICE)
with torch.no_grad():
    for _ in range(20):
        _ = orig_model.forward_classify(dummy_input)
    t0 = time.perf_counter()
    for _ in range(200):
        _ = orig_model.forward_classify(dummy_input)
    t1 = time.perf_counter()
    inf_ms_before = round((t1 - t0) / 200 * 1000, 4)

    for _ in range(20):
        _ = active_model.forward_classify(dummy_input)
    t0 = time.perf_counter()
    for _ in range(200):
        _ = active_model.forward_classify(dummy_input)
    t1 = time.perf_counter()
    inf_ms_after = round((t1 - t0) / 200 * 1000, 4)

size_before_kb = round(ssl_ckpt_path.stat().st_size / 1024, 2)
size_after_kb = round(adapted_ckpt_path.stat().st_size / 1024, 2) if adapted_ckpt_path.exists() else size_before_kb

cost_df = pd.DataFrame([{
    'adaptation_training_time_s': adaptation_time_s,
    'adaptation_epochs':          ADAPT_EPOCHS,
    'inference_ms_before':        inf_ms_before,
    'inference_ms_after':         inf_ms_after,
    'model_size_kb_before':       size_before_kb,
    'model_size_kb_after':        size_after_kb,
    'model_parameters':           active_model.n_params,
}])
cost_df.to_csv(ADAPT_RES / 'adaptation_cost.csv', index=False)
print(cost_df.to_string(index=False))

# ── 12. Error Analysis ────────────────────────────────────────────────────────
print("\n" + "="*65)
print("12. ERROR ANALYSIS")
print("="*65)

pred_orig_shft = (p_orig_full_shft >= ORIG_THRESHOLD).astype(int)
pred_adapt_shft = (p_adapt_full_shft >= ACTIVE_THRESHOLD).astype(int)

fp_before = int(np.sum((pred_orig_shft == 1) & (y_orig_full_shft == 0)))
fn_before = int(np.sum((pred_orig_shft == 0) & (y_orig_full_shft == 1)))
fp_after = int(np.sum((pred_adapt_shft == 1) & (y_adapt_full_shft == 0)))
fn_after = int(np.sum((pred_adapt_shft == 0) & (y_adapt_full_shft == 1)))

print(f"Shifted Condition Error Transition:")
print(f"  False Positives (FP): {fp_before} -> {fp_after} (Reduced by {fp_before - fp_after} false alarms!)")
print(f"  False Negatives (FN): {fn_before} -> {fn_after}")

err_records = [
    {'condition': 'Shifted Full Test', 'state': 'Before Adaptation', 'FP': fp_before, 'FN': fn_before,
     'FPR': met_orig_full_shft['fpr'], 'FNR': met_orig_full_shft['fnr']},
    {'condition': 'Shifted Full Test', 'state': 'After Adaptation',  'FP': fp_after,  'FN': fn_after,
     'FPR': met_adapt_full_shft['fpr'], 'FNR': met_adapt_full_shft['fnr']},
]
pd.DataFrame(err_records).to_csv(ADAPT_RES / 'error_analysis.csv', index=False)

# ── 13. Visualizations ────────────────────────────────────────────────────────
print("\n" + "="*65)
print("13. GENERATING VISUALIZATIONS")
print("="*65)

# Fig 1: Before vs After Adaptation F1 (Normal vs Shifted)
fig, ax = plt.subplots(figsize=(7, 5))
bar_width = 0.35
indices = np.arange(2)
f1_before = [met_orig_normal['f1'], met_orig_shifted['f1']]
f1_after = [met_adapt_normal['f1'], met_adapt_shifted['f1']]

b1 = ax.bar(indices - bar_width/2, f1_before, bar_width, label='Before Adaptation', color='#E74C3C', alpha=0.85)
b2 = ax.bar(indices + bar_width/2, f1_after, bar_width, label='After Adaptation', color='#2ECC71', alpha=0.85)

for bar in list(b1) + list(b2):
    height = bar.get_height()
    ax.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xticks(indices)
ax.set_xticklabels(['Normal Test Condition', 'Shifted Test Condition'], fontsize=10)
ax.set_ylabel('F1-Score', fontsize=11)
ax.set_title('F1 Recovery Under Adaptation (Normal vs. Shifted)', fontsize=12)
ax.set_ylim(0, 1.15)
ax.legend(loc='lower left', fontsize=10)
fig.tight_layout()
fig.savefig(ADAPT_FIGS / 'f1_before_after.png', dpi=120)
plt.close(fig)

# Fig 2: Recall & Precision Recovery on Shifted Data
fig, ax = plt.subplots(figsize=(7, 5))
prec_vals = [met_orig_shifted['precision'], met_adapt_shifted['precision']]
rec_vals = [met_orig_shifted['recall'], met_adapt_shifted['recall']]
labels_cond = ['Before Adaptation', 'After Adaptation']

ax.plot(labels_cond, prec_vals, marker='o', lw=2.5, color='#3498DB', label='Precision')
ax.plot(labels_cond, rec_vals, marker='s', lw=2.5, color='#9B59B6', label='Recall')
for i, (p, r) in enumerate(zip(prec_vals, rec_vals)):
    ax.annotate(f'P: {p:.4f}', (i, p), xytext=(0, 7), textcoords="offset points", ha='center', fontsize=9)
    ax.annotate(f'R: {r:.4f}', (i, r), xytext=(0, -12), textcoords="offset points", ha='center', fontsize=9)

ax.set_title('Precision & Recall Trajectory Under Adaptation (Shifted Condition)', fontsize=12)
ax.set_ylabel('Metric Score', fontsize=11)
ax.set_ylim(0, 1.15)
ax.legend(loc='center right', fontsize=10)
fig.tight_layout()
fig.savefig(ADAPT_FIGS / 'precision_recall_recovery.png', dpi=120)
plt.close(fig)

# Fig 3: 4-Way Comprehensive Metric Comparison
fig, ax = plt.subplots(figsize=(10, 5))
metrics_plot = ['Precision', 'Recall', 'F1', 'ROC-AUC', 'PR-AUC']
x_m = np.arange(len(metrics_plot))
w = 0.2

for idx, rec in enumerate(comp_records):
    vals = [rec[m] for m in metrics_plot]
    lbl = f"{rec['Model State'].split()[0]} - {rec['Test Condition'].split()[0]}"
    color = plt.cm.Set2(idx / 4)
    bars = ax.bar(x_m + (idx - 1.5)*w, vals, w, label=lbl, color=color)

ax.set_xticks(x_m)
ax.set_xticklabels(metrics_plot, fontsize=10)
ax.set_ylabel('Score', fontsize=11)
ax.set_title('4-Way Metric Comparison Across Model States and Conditions', fontsize=12)
ax.set_ylim(0, 1.2)
ax.legend(loc='lower right', fontsize=9)
fig.tight_layout()
fig.savefig(ADAPT_FIGS / 'four_way_comparison.png', dpi=120)
plt.close(fig)

# Fig 4: Adaptation Training Loss & Validation F1
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
epochs_range = list(range(1, ADAPT_EPOCHS + 1))
ax1.plot(epochs_range, train_losses, marker='o', color='#E74C3C', lw=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('BCE Loss')
ax1.set_title('Adaptation Training Loss')
ax1.set_xticks(epochs_range)

ax2.plot(epochs_range, val_f1s, marker='s', color='#2ECC71', lw=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Validation F1')
ax2.set_title('Adaptation Validation F1 Trajectory')
ax2.set_xticks(epochs_range)
fig.tight_layout()
fig.savefig(ADAPT_FIGS / 'adaptation_training_curves.png', dpi=120)
plt.close(fig)

# Fig 5: False Positive Rate (FPR) Reduction
fig, ax = plt.subplots(figsize=(6, 5))
fpr_bars = [met_orig_shifted['fpr']*100, met_adapt_shifted['fpr']*100]
bars = ax.bar(['Before Adaptation', 'After Adaptation'], fpr_bars, color=['#E74C3C', '#2ECC71'], width=0.45)
for bar in bars:
    h = bar.get_height()
    ax.annotate(f'{h:.2f}%', (bar.get_x() + bar.get_width()/2, h), xytext=(0, 4), textcoords="offset points", ha='center', fontweight='bold')
ax.set_ylabel('False Positive Rate (%)', fontsize=11)
ax.set_title('Drastic FPR Alarm Suppression on Shifted Data', fontsize=12)
ax.set_ylim(0, 110)
fig.tight_layout()
fig.savefig(ADAPT_FIGS / 'fpr_reduction.png', dpi=120)
plt.close(fig)

# Mirror figures to figures/ directory
for f in ADAPT_FIGS.glob('*.png'):
    shutil.copy(f, FIGURES_DIR / f.name)
print("All figures saved to figures/adaptation/ and mirrored to figures/")

# ── 14. Experiment Tracking ───────────────────────────────────────────────────
print("\n" + "="*65)
print("14. UPDATING EXPERIMENT TRACKING")
print("="*65)

exp_row = {
    'experiment_id':            f"phase8_adapt_{int(time.time())}",
    'model_name':               'SSL_Transformer',
    'ssl_used':                 True,
    'shift_detected':           shift_detected,
    'shift_score':              round(shift_score, 6),
    'adaptation_used':          True,
    'adaptation_strategy':      'periodic_fine_tuning_with_replay',
    'adaptation_epochs':        ADAPT_EPOCHS,
    'adaptation_learning_rate': ADAPT_LR,
    'replay_used':              True,
    'replay_ratio':             REPLAY_RATIO,
    'before_precision':         round(met_orig_shifted['precision'], 4),
    'before_recall':            round(met_orig_shifted['recall'], 4),
    'before_f1':                round(met_orig_shifted['f1'], 4),
    'before_roc_auc':           round(met_orig_shifted['roc_auc'], 4),
    'before_pr_auc':            round(met_orig_shifted['pr_auc'], 4),
    'after_precision':          round(met_adapt_shifted['precision'], 4),
    'after_recall':             round(met_adapt_shifted['recall'], 4),
    'after_f1':                 round(met_adapt_shifted['f1'], 4),
    'after_roc_auc':            round(met_adapt_shifted['roc_auc'], 4),
    'after_pr_auc':             round(met_adapt_shifted['pr_auc'], 4),
    'original_test_f1':         round(met_adapt_normal['f1'], 4),
    'shifted_test_f1':          round(met_adapt_shifted['f1'], 4),
    'adaptation_time':          adaptation_time_s,
    'inference_time':           inf_ms_after,
    'model_size':               size_after_kb,
    'checkpoint_name':          'ssl_transformer_adapted.pt' if accepted else 'ssl_finetuned.pt',
}

exp_path = EXP_DIR / 'experiments.csv'
new_exp_df = pd.DataFrame([exp_row])
if exp_path.exists():
    existing_exp = pd.read_csv(exp_path)
    new_exp_df = pd.concat([existing_exp, new_exp_df], ignore_index=True)
new_exp_df.to_csv(exp_path, index=False)
print(f"Recorded Phase 8 experiment entry to {exp_path}")

# ── 15. Consolidated Adaptation Summary JSON ──────────────────────────────────
summary_json = {
    'shift_detected':      shift_detected,
    'shift_score':         round(shift_score, 6),
    'shift_threshold':     round(threshold, 6),
    'adaptation_strategy': 'periodic_fine_tuning_with_replay',
    'replay_ratio':        REPLAY_RATIO,
    'epochs':              ADAPT_EPOCHS,
    'learning_rate':       ADAPT_LR,
    'adaptation_time_s':   adaptation_time_s,
    'candidate_threshold': round(cand_threshold, 4),
    'safety_check':        criteria_details,
    'comparison_records':  comp_records,
    'deltas_shifted': {
        'delta_f1':        round(delta_f1_shift, 4),
        'delta_precision': round(delta_prec_shift, 4),
        'delta_recall':    round(delta_rec_shift, 4),
        'delta_fpr':       round(delta_fpr_shift, 4),
    },
    'deltas_normal_retention': {
        'delta_f1_normal': round(delta_f1_norm, 4),
    },
    'error_analysis': {
        'fp_before': fp_before, 'fp_after': fp_after,
        'fn_before': fn_before, 'fn_after': fn_after,
    },
    'cost': {
        'adaptation_time_s': adaptation_time_s,
        'inference_ms_before': inf_ms_before,
        'inference_ms_after': inf_ms_after,
        'model_size_kb': size_after_kb,
    }
}
with open(ADAPT_RES / 'adaptation_summary.json', 'w') as f:
    json.dump(summary_json, f, indent=2)

print("\n" + "="*65)
print("PHASE 8 FINAL SUMMARY COMPLETE")
print("="*65)
print(f"Shift Detected:           {shift_detected} (Score={shift_score:.4f}, Thresh={threshold:.4f})")
print(f"Candidate Accepted:       {accepted}")
print(f"Shifted F1 (Before -> After): {met_orig_shifted['f1']:.4f} -> {met_adapt_shifted['f1']:.4f} (Delta={delta_f1_shift:+.4f})")
print(f"Shifted FPR (Before -> After): {met_orig_shifted['fpr']*100:.2f}% -> {met_adapt_shifted['fpr']*100:.2f}% (Delta={delta_fpr_shift*100:+.2f}%)")
print(f"Normal F1 Retention:      {met_orig_normal['f1']:.4f} -> {met_adapt_normal['f1']:.4f} (Delta={delta_f1_norm:+.4f})")
print(f"Adaptation Duration:      {adaptation_time_s:.2f} s")
print(f"Active Checkpoint:        {'models/adaptation/ssl_transformer_adapted.pt' if accepted else 'models/ssl/ssl_finetuned.pt'}")
print(f"All adaptation results saved to: {ADAPT_RES}")
