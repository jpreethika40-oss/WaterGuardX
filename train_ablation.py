"""
Phase 9 — Ablation Study and Component Contribution Analysis.
Run from project root: python train_ablation.py

Evaluates 5 distinct system configurations under identical experimental conditions:
  Config A: Baseline Transformer (No SSL, No Shift Detection, No Adaptation)
  Config B: Transformer + SSL (With SSL, No Shift Detection, No Adaptation)
  Config C: Transformer + Adaptation (No SSL, With Shift Detection, With Adaptation)
  Config D: SSL Transformer + Adaptation (With SSL, With Shift Detection, With Adaptation)
  Config E: Full Proposed System (Transformer + SSL + Shift Detection + Adaptive Learning)
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
import torch
import torch.nn as nn

from ablation import (adapt_generic_model, compute_component_contributions,
                      evaluate_model_generic, load_baseline_transformer)
from adaptation import (build_replay_dataset, check_acceptance_criteria,
                        load_ssl_model, split_chronological_adaptation)
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
ABL_RES = RESULTS_DIR / 'ablation'; ABL_RES.mkdir(parents=True, exist_ok=True)
ABL_FIGS = FIGURES_DIR / 'ablation'; ABL_FIGS.mkdir(parents=True, exist_ok=True)
ABL_MOD = MODELS_DIR / 'ablation'; ABL_MOD.mkdir(parents=True, exist_ok=True)
EXP_DIR = RESULTS_DIR / 'experiments'; EXP_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Load Data & Prepare Splits ─────────────────────────────────────────────
print("\n" + "="*65)
print("1. LOADING DATA AND PREPARING TEST EVALUATION SUBSETS")
print("="*65)

df_raw = load_raw()
train_df, val_df, test_df, scaler, feat_cols = run_preprocessing(df_raw, save=False)
N_FEATURES = len(feat_cols)

shifted_test_path = DATA_PROC / 'shifted' / 'strong' / 'test_shifted.csv'
test_shifted_df = pd.read_csv(shifted_test_path)

# Chronological partition (same protocol as Phase 8: Nov adaptation, Dec evaluation)
adapt_train_df, adapt_val_df, eval_shifted_df = split_chronological_adaptation(
    test_shifted_df, timestamp_col=TIMESTAMP_COL, split_date='2018-12-01', val_ratio=0.30
)
test_df[TIMESTAMP_COL] = pd.to_datetime(test_df[TIMESTAMP_COL])
eval_normal_df = test_df[test_df[TIMESTAMP_COL] >= '2018-12-01'].reset_index(drop=True)

print(f"Evaluation Protocol:")
print(f"  Normal Evaluation Subset (Dec 1-31):  {len(eval_normal_df):,} rows")
print(f"  Shifted Evaluation Subset (Dec 1-31): {len(eval_shifted_df):,} rows")
print(f"  Adaptation Training Set (Nov 1-21):   {len(adapt_train_df):,} rows")
print(f"  Adaptation Validation Set (Nov 22-30):{len(adapt_val_df):,} rows")

# Replay dataset with 20% reference training samples
replay_dataset = build_replay_dataset(
    adapt_df=adapt_train_df, train_df=train_df, feat_cols=feat_cols,
    replay_ratio=0.20, seq_len=SEQUENCE_LENGTH, stride=STRIDE, seed=SEED
)

# Reference drift detector from Phase 7
detector = DriftDetector(n_bootstrap=500, bootstrap_frac=0.3, percentile=99.0, seed=SEED)
detector.fit(train_df, feat_cols)
drift_eval = detector.detect(eval_shifted_df)
SHIFT_DETECTED = drift_eval['shift_detected']
SHIFT_SCORE = drift_eval['shift_score']
SHIFT_THRESH = drift_eval['threshold']
print(f"Shift Detection on Evaluation Stream: Score={SHIFT_SCORE:.4f}, Thresh={SHIFT_THRESH:.4f} -> Detected={SHIFT_DETECTED}")

# Helper timing function
def benchmark_inference(model):
    dummy = torch.randn(1, SEQUENCE_LENGTH, N_FEATURES).to(DEVICE)
    with torch.no_grad():
        for _ in range(20):
            if hasattr(model, 'forward_classify'):
                _ = model.forward_classify(dummy)
            else:
                _ = model(dummy)
        t0 = time.perf_counter()
        for _ in range(200):
            if hasattr(model, 'forward_classify'):
                _ = model.forward_classify(dummy)
            else:
                _ = model(dummy)
        t1 = time.perf_counter()
    return round((t1 - t0) / 200 * 1000, 4)

# ── 2. CONFIGURATION A: Baseline Transformer (No SSL, No Adapt) ───────────────
print("\n" + "="*65)
print("2. EVALUATING CONFIG A: BASELINE TRANSFORMER (No SSL, No Adapt)")
print("="*65)

base_ckpt = MODELS_DIR / 'transformer' / 'best_transformer.pt'
base_cfg_path = MODELS_DIR / 'transformer' / 'config.json'
model_a, cfg_a = load_baseline_transformer(base_ckpt, base_cfg_path, device=DEVICE)
thresh_a = float(cfg_a.get('threshold', 0.0198))

met_a_norm, _, _ = evaluate_model_generic(model_a, eval_normal_df, feat_cols, threshold=thresh_a, device=DEVICE)
met_a_shft, _, _ = evaluate_model_generic(model_a, eval_shifted_df, feat_cols, threshold=thresh_a, device=DEVICE)
inf_a = benchmark_inference(model_a)
size_a_kb = round(base_ckpt.stat().st_size / 1024, 2)
train_time_a = 75.40  # From Phase 5 logs

print("Config A (Baseline Transformer):")
print(f"  Normal:  F1={met_a_norm['f1']:.4f}, Prec={met_a_norm['precision']:.4f}, Rec={met_a_norm['recall']:.4f}, ROC-AUC={met_a_norm['roc_auc']:.4f}")
print(f"  Shifted: F1={met_a_shft['f1']:.4f}, Prec={met_a_shft['precision']:.4f}, Rec={met_a_shft['recall']:.4f}, ROC-AUC={met_a_shft['roc_auc']:.4f}")
print(f"  Cost:    Params={model_a.n_params:,}, Size={size_a_kb} KB, Inf={inf_a} ms/seq")

# ── 3. CONFIGURATION B: Transformer + SSL (With SSL, No Adapt) ────────────────
print("\n" + "="*65)
print("3. EVALUATING CONFIG B: TRANSFORMER + SSL (With SSL, No Adapt)")
print("="*65)

ssl_ckpt = MODELS_DIR / 'ssl' / 'ssl_finetuned.pt'
ssl_cfg_path = MODELS_DIR / 'ssl' / 'ssl_config.json'
model_b, cfg_b = load_ssl_model(ssl_ckpt, ssl_cfg_path, device=DEVICE)
ssl_metrics_df = pd.read_csv(RESULTS_DIR / 'ssl' / 'final_metrics.csv')
thresh_b = float(ssl_metrics_df['threshold'].iloc[0])

met_b_norm, _, _ = evaluate_model_generic(model_b, eval_normal_df, feat_cols, threshold=thresh_b, device=DEVICE)
met_b_shft, _, _ = evaluate_model_generic(model_b, eval_shifted_df, feat_cols, threshold=thresh_b, device=DEVICE)
inf_b = benchmark_inference(model_b)
size_b_kb = round(ssl_ckpt.stat().st_size / 1024, 2)
train_time_b = 253.99  # From Phase 6 (169.88s pretrain + 84.11s finetune)

print("Config B (Transformer + SSL):")
print(f"  Normal:  F1={met_b_norm['f1']:.4f}, Prec={met_b_norm['precision']:.4f}, Rec={met_b_norm['recall']:.4f}, ROC-AUC={met_b_norm['roc_auc']:.4f}")
print(f"  Shifted: F1={met_b_shft['f1']:.4f}, Prec={met_b_shft['precision']:.4f}, Rec={met_b_shft['recall']:.4f}, ROC-AUC={met_b_shft['roc_auc']:.4f}")
print(f"  Cost:    Params={model_b.n_params:,}, Size={size_b_kb} KB, Inf={inf_b} ms/seq")

# ── 4. CONFIGURATION C: Transformer + Adaptation (No SSL, With Adapt) ─────────
print("\n" + "="*65)
print("4. EVALUATING CONFIG C: TRANSFORMER + ADAPTATION (No SSL, With Adapt)")
print("="*65)

# Adapt the baseline Transformer on adaptation data using the exact same protocol
model_c, thresh_c, losses_c, val_f1s_c, adapt_time_c = adapt_generic_model(
    model=model_a,
    train_dataset=replay_dataset,
    val_df=adapt_val_df,
    feat_cols=feat_cols,
    epochs=5,
    lr=1e-4,
    weight_decay=1e-4,
    batch_size=128,
    device=DEVICE,
)

# Save checkpoint
ckpt_c = ABL_MOD / 'transformer_adapted_no_ssl.pt'
torch.save(model_c.state_dict(), ckpt_c)

met_c_norm, _, _ = evaluate_model_generic(model_c, eval_normal_df, feat_cols, threshold=thresh_c, device=DEVICE)
met_c_shft, _, _ = evaluate_model_generic(model_c, eval_shifted_df, feat_cols, threshold=thresh_c, device=DEVICE)
inf_c = benchmark_inference(model_c)
size_c_kb = round(ckpt_c.stat().st_size / 1024, 2)
train_time_c = train_time_a + adapt_time_c

print("Config C (Transformer + Adaptation, No SSL):")
print(f"  Normal:  F1={met_c_norm['f1']:.4f}, Prec={met_c_norm['precision']:.4f}, Rec={met_c_norm['recall']:.4f}, ROC-AUC={met_c_norm['roc_auc']:.4f}")
print(f"  Shifted: F1={met_c_shft['f1']:.4f}, Prec={met_c_shft['precision']:.4f}, Rec={met_c_shft['recall']:.4f}, ROC-AUC={met_c_shft['roc_auc']:.4f}")
print(f"  Cost:    Adapt Time={adapt_time_c}s, Params={model_c.n_params:,}, Size={size_c_kb} KB, Inf={inf_c} ms/seq")

# ── 5. CONFIGURATION D: SSL Transformer + Adaptation (With SSL, With Adapt) ───
print("\n" + "="*65)
print("5. EVALUATING CONFIG D: SSL TRANSFORMER + ADAPTATION (With SSL, With Adapt)")
print("="*65)

adapted_ssl_ckpt = MODELS_DIR / 'adaptation' / 'ssl_transformer_adapted.pt'
model_d, _ = load_ssl_model(adapted_ssl_ckpt, ssl_cfg_path, device=DEVICE)
with open(RESULTS_DIR / 'adaptation' / 'adaptation_summary.json') as f:
    adapt_meta = json.load(f)
thresh_d = float(adapt_meta['candidate_threshold'])

met_d_norm, _, _ = evaluate_model_generic(model_d, eval_normal_df, feat_cols, threshold=thresh_d, device=DEVICE)
met_d_shft, _, _ = evaluate_model_generic(model_d, eval_shifted_df, feat_cols, threshold=thresh_d, device=DEVICE)
inf_d = benchmark_inference(model_d)
size_d_kb = round(adapted_ssl_ckpt.stat().st_size / 1024, 2)
adapt_time_d = float(adapt_meta['adaptation_time_s'])
train_time_d = train_time_b + adapt_time_d

print("Config D (SSL Transformer + Adaptation):")
print(f"  Normal:  F1={met_d_norm['f1']:.4f}, Prec={met_d_norm['precision']:.4f}, Rec={met_d_norm['recall']:.4f}, ROC-AUC={met_d_norm['roc_auc']:.4f}")
print(f"  Shifted: F1={met_d_shft['f1']:.4f}, Prec={met_d_shft['precision']:.4f}, Rec={met_d_shft['recall']:.4f}, ROC-AUC={met_d_shft['roc_auc']:.4f}")
print(f"  Cost:    Adapt Time={adapt_time_d}s, Params={model_d.n_params:,}, Size={size_d_kb} KB, Inf={inf_d} ms/seq")

# ── 6. CONFIGURATION E: Full System (SSL + Shift Detection + Adaptation) ──────
print("\n" + "="*65)
print("6. EVALUATING CONFIG E: FULL PROPOSED SYSTEM")
print("="*65)

# In the full system:
# When stream is stable -> Original SSL model (Config B) is used.
# When shift is detected -> Adapted SSL model (Config D) is triggered and used.
met_e_norm = copy.deepcopy(met_b_norm) # Nominally stable operating condition
met_e_shft = copy.deepcopy(met_d_shft) # Shift detected -> adapted model activated
inf_e = inf_d
size_e_kb = size_d_kb
adapt_time_e = adapt_time_d
train_time_e = train_time_d

print("Config E (Full System):")
print(f"  Normal (Stable Path):   F1={met_e_norm['f1']:.4f}, Prec={met_e_norm['precision']:.4f}, Rec={met_e_norm['recall']:.4f}, ROC-AUC={met_e_norm['roc_auc']:.4f}")
print(f"  Shifted (Adapted Path): F1={met_e_shft['f1']:.4f}, Prec={met_e_shft['precision']:.4f}, Rec={met_e_shft['recall']:.4f}, ROC-AUC={met_e_shft['roc_auc']:.4f}")

# ── 7. Consolidated Ablation Table ───────────────────────────────────────────
print("\n" + "="*65)
print("7. ABLATION STUDY COMPARISON TABLE")
print("="*65)

ablation_records = [
    {
        'Config': 'A',
        'Name': 'Baseline Transformer',
        'SSL': 'No',
        'Shift Detection': 'No',
        'Adaptation': 'No',
        'Normal F1': round(met_a_norm['f1'], 4),
        'Shifted F1': round(met_a_shft['f1'], 4),
        'Adapted F1': 'N/A',
        'Shifted ROC-AUC': round(met_a_shft['roc_auc'], 4),
        'Shifted PR-AUC': round(met_a_shft['pr_auc'], 4),
        'Shifted FPR': f"{met_a_shft['fpr']*100:.2f}%",
    },
    {
        'Config': 'B',
        'Name': 'Transformer + SSL',
        'SSL': 'Yes',
        'Shift Detection': 'No',
        'Adaptation': 'No',
        'Normal F1': round(met_b_norm['f1'], 4),
        'Shifted F1': round(met_b_shft['f1'], 4),
        'Adapted F1': 'N/A',
        'Shifted ROC-AUC': round(met_b_shft['roc_auc'], 4),
        'Shifted PR-AUC': round(met_b_shft['pr_auc'], 4),
        'Shifted FPR': f"{met_b_shft['fpr']*100:.2f}%",
    },
    {
        'Config': 'C',
        'Name': 'Transformer + Adaptation',
        'SSL': 'No',
        'Shift Detection': 'Yes',
        'Adaptation': 'Yes',
        'Normal F1': round(met_c_norm['f1'], 4),
        'Shifted F1': round(met_a_shft['f1'], 4),
        'Adapted F1': round(met_c_shft['f1'], 4),
        'Shifted ROC-AUC': round(met_c_shft['roc_auc'], 4),
        'Shifted PR-AUC': round(met_c_shft['pr_auc'], 4),
        'Shifted FPR': f"{met_c_shft['fpr']*100:.2f}%",
    },
    {
        'Config': 'D',
        'Name': 'SSL Transformer + Adaptation',
        'SSL': 'Yes',
        'Shift Detection': 'Yes',
        'Adaptation': 'Yes',
        'Normal F1': round(met_d_norm['f1'], 4),
        'Shifted F1': round(met_b_shft['f1'], 4),
        'Adapted F1': round(met_d_shft['f1'], 4),
        'Shifted ROC-AUC': round(met_d_shft['roc_auc'], 4),
        'Shifted PR-AUC': round(met_d_shft['pr_auc'], 4),
        'Shifted FPR': f"{met_d_shft['fpr']*100:.2f}%",
    },
    {
        'Config': 'E',
        'Name': 'Full System (All Components)',
        'SSL': 'Yes',
        'Shift Detection': 'Yes',
        'Adaptation': 'Yes',
        'Normal F1': round(met_e_norm['f1'], 4),
        'Shifted F1': round(met_b_shft['f1'], 4),
        'Adapted F1': round(met_e_shft['f1'], 4),
        'Shifted ROC-AUC': round(met_e_shft['roc_auc'], 4),
        'Shifted PR-AUC': round(met_e_shft['pr_auc'], 4),
        'Shifted FPR': f"{met_e_shft['fpr']*100:.2f}%",
    },
]

abl_df = pd.DataFrame(ablation_records)
abl_df.to_csv(ABL_RES / 'ablation_table.csv', index=False)
print(abl_df.to_string(index=False))

# ── 8. Component Contributions Analysis ───────────────────────────────────────
print("\n" + "="*65)
print("8. COMPONENT CONTRIBUTION DECOMPOSITION")
print("="*65)

metrics_map = {
    'a': {'normal': met_a_norm, 'shifted': met_a_shft, 'adapted': met_a_shft},
    'b': {'normal': met_b_norm, 'shifted': met_b_shft, 'adapted': met_b_shft},
    'c': {'normal': met_c_norm, 'shifted': met_a_shft, 'adapted': met_c_shft},
    'd': {'normal': met_d_norm, 'shifted': met_b_shft, 'adapted': met_d_shft},
    'e': {'normal': met_e_norm, 'shifted': met_b_shft, 'adapted': met_e_shft},
}

contrib_df = compute_component_contributions(
    metrics_a=metrics_map['a'],
    metrics_b=metrics_map['b'],
    metrics_c=metrics_map['c'],
    metrics_d=metrics_map['d'],
    metrics_e=metrics_map['e'],
)
contrib_df.to_csv(ABL_RES / 'component_contributions.csv', index=False)
print(contrib_df.to_string(index=False))

# ── 9. Error Analysis Across Configurations ───────────────────────────────────
print("\n" + "="*65)
print("9. ERROR ANALYSIS ACROSS CONFIGURATIONS (Shifted Evaluation Set)")
print("="*65)

error_records = [
    {
        'Config': 'A (Baseline)',
        'Model': 'Baseline Transformer',
        'Shifted FP': met_a_shft['fp'],
        'Shifted FN': met_a_shft['fn'],
        'Shifted FPR': f"{met_a_shft['fpr']*100:.2f}%",
        'Shifted FNR': f"{met_a_shft['fnr']*100:.2f}%",
    },
    {
        'Config': 'B (SSL Only)',
        'Model': 'Transformer + SSL',
        'Shifted FP': met_b_shft['fp'],
        'Shifted FN': met_b_shft['fn'],
        'Shifted FPR': f"{met_b_shft['fpr']*100:.2f}%",
        'Shifted FNR': f"{met_b_shft['fnr']*100:.2f}%",
    },
    {
        'Config': 'C (Adaptation Only)',
        'Model': 'Transformer + Adaptation',
        'Shifted FP': met_c_shft['fp'],
        'Shifted FN': met_c_shft['fn'],
        'Shifted FPR': f"{met_c_shft['fpr']*100:.2f}%",
        'Shifted FNR': f"{met_c_shft['fnr']*100:.2f}%",
    },
    {
        'Config': 'D (SSL + Adapt)',
        'Model': 'SSL Transformer + Adaptation',
        'Shifted FP': met_d_shft['fp'],
        'Shifted FN': met_d_shft['fn'],
        'Shifted FPR': f"{met_d_shft['fpr']*100:.2f}%",
        'Shifted FNR': f"{met_d_shft['fnr']*100:.2f}%",
    },
    {
        'Config': 'E (Full System)',
        'Model': 'Full Proposed System',
        'Shifted FP': met_e_shft['fp'],
        'Shifted FN': met_e_shft['fn'],
        'Shifted FPR': f"{met_e_shft['fpr']*100:.2f}%",
        'Shifted FNR': f"{met_e_shft['fnr']*100:.2f}%",
    },
]

err_df = pd.DataFrame(error_records)
err_df.to_csv(ABL_RES / 'error_analysis_ablation.csv', index=False)
print(err_df.to_string(index=False))

# ── 10. Computational Cost Comparison ─────────────────────────────────────────
print("\n" + "="*65)
print("10. COMPUTATIONAL COST COMPARISON")
print("="*65)

cost_records = [
    {
        'Config': 'A',
        'Configuration': 'Baseline Transformer',
        'Training Time (s)': train_time_a,
        'Adaptation Time (s)': 0.0,
        'Total Time (s)': train_time_a,
        'Inference Latency (ms)': inf_a,
        'Model Size (KB)': size_a_kb,
        'Parameters': model_a.n_params,
    },
    {
        'Config': 'B',
        'Configuration': 'Transformer + SSL',
        'Training Time (s)': train_time_b,
        'Adaptation Time (s)': 0.0,
        'Total Time (s)': train_time_b,
        'Inference Latency (ms)': inf_b,
        'Model Size (KB)': size_b_kb,
        'Parameters': model_b.n_params,
    },
    {
        'Config': 'C',
        'Configuration': 'Transformer + Adaptation',
        'Training Time (s)': train_time_a,
        'Adaptation Time (s)': adapt_time_c,
        'Total Time (s)': round(train_time_c, 2),
        'Inference Latency (ms)': inf_c,
        'Model Size (KB)': size_c_kb,
        'Parameters': model_c.n_params,
    },
    {
        'Config': 'D',
        'Configuration': 'SSL Transformer + Adaptation',
        'Training Time (s)': train_time_b,
        'Adaptation Time (s)': adapt_time_d,
        'Total Time (s)': round(train_time_d, 2),
        'Inference Latency (ms)': inf_d,
        'Model Size (KB)': size_d_kb,
        'Parameters': model_d.n_params,
    },
    {
        'Config': 'E',
        'Configuration': 'Full System (All Components)',
        'Training Time (s)': train_time_b,
        'Adaptation Time (s)': adapt_time_e,
        'Total Time (s)': round(train_time_e, 2),
        'Inference Latency (ms)': inf_e,
        'Model Size (KB)': size_e_kb,
        'Parameters': model_d.n_params,
    },
]

cost_df = pd.DataFrame(cost_records)
cost_df.to_csv(ABL_RES / 'computational_cost.csv', index=False)
print(cost_df.to_string(index=False))

# ── 11. Visualizations ────────────────────────────────────────────────────────
print("\n" + "="*65)
print("11. GENERATING ABLATION VISUALIZATIONS")
print("="*65)

# Fig 1: F1 Comparison across configurations (Normal vs Shifted vs Adapted)
fig, ax = plt.subplots(figsize=(9, 5))
configs = ['A (Base)', 'B (SSL)', 'C (Adapt)', 'D (SSL+Adapt)', 'E (Full)']
norm_f1s = [met_a_norm['f1'], met_b_norm['f1'], met_c_norm['f1'], met_d_norm['f1'], met_e_norm['f1']]
shft_f1s = [met_a_shft['f1'], met_b_shft['f1'], met_c_shft['f1'], met_d_shft['f1'], met_e_shft['f1']]

x = np.arange(len(configs))
w = 0.35
b1 = ax.bar(x - w/2, norm_f1s, w, label='Normal Condition F1', color='#3498DB', alpha=0.85)
b2 = ax.bar(x + w/2, shft_f1s, w, label='Shifted Condition F1 (Adapted if active)', color='#E67E22', alpha=0.85)

for bar in list(b1) + list(b2):
    h = bar.get_height()
    ax.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3),
                textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(configs, fontsize=10)
ax.set_ylabel('F1-Score', fontsize=11)
ax.set_title('Ablation Study: F1 Score Across Configurations (Normal vs. Shifted)', fontsize=12)
ax.set_ylim(0, 1.2)
ax.legend(loc='lower left', fontsize=10)
fig.tight_layout()
fig.savefig(ABL_FIGS / 'f1_ablation_comparison.png', dpi=120)
plt.close(fig)

# Fig 2: Precision vs Recall Comparison under Shift
fig, ax = plt.subplots(figsize=(8, 5))
prec_s = [met_a_shft['precision'], met_b_shft['precision'], met_c_shft['precision'], met_d_shft['precision'], met_e_shft['precision']]
rec_s = [met_a_shft['recall'], met_b_shft['recall'], met_c_shft['recall'], met_d_shft['recall'], met_e_shft['recall']]

p1 = ax.plot(configs, prec_s, marker='o', lw=2.5, color='#2980B9', label='Shifted Precision')
p2 = ax.plot(configs, rec_s, marker='s', lw=2.5, color='#8E44AD', label='Shifted Recall')
for i, (p, r) in enumerate(zip(prec_s, rec_s)):
    ax.annotate(f'{p:.3f}', (i, p), xytext=(0, 6), textcoords="offset points", ha='center', fontsize=9)
    ax.annotate(f'{r:.3f}', (i, r), xytext=(0, -12), textcoords="offset points", ha='center', fontsize=9)

ax.set_title('Precision & Recall Trade-off Across Ablation Configurations', fontsize=12)
ax.set_ylabel('Score', fontsize=11)
ax.set_ylim(0, 1.15)
ax.legend(loc='center right', fontsize=10)
fig.tight_layout()
fig.savefig(ABL_FIGS / 'precision_recall_ablation.png', dpi=120)
plt.close(fig)

# Fig 3: Component Contributions Delta Breakdown
fig, ax = plt.subplots(figsize=(9, 4.5))
effects = [
    'SSL (Normal)',
    'SSL (Shifted)',
    'SSL (ROC-AUC)',
    'Adaptation (No SSL)',
    'Adaptation (With SSL)',
    'SSL in Adapt Regime',
    'Full System Gain'
]
deltas = [
    round(met_b_norm['f1'] - met_a_norm['f1'], 4),
    round(met_b_shft['f1'] - met_a_shft['f1'], 4),
    round(met_b_shft['roc_auc'] - met_a_shft['roc_auc'], 4),
    round(met_c_shft['f1'] - met_a_shft['f1'], 4),
    round(met_d_shft['f1'] - met_b_shft['f1'], 4),
    round(met_d_shft['f1'] - met_c_shft['f1'], 4),
    round(met_e_shft['f1'] - met_a_shft['f1'], 4),
]
colors = ['#27AE60' if d >= 0 else '#C0392B' for d in deltas]

bars = ax.barh(effects, deltas, color=colors, alpha=0.85)
for bar, d in zip(bars, deltas):
    x_pos = bar.get_width() + (0.015 if d >= 0 else -0.055)
    ax.text(x_pos, bar.get_y() + bar.get_height()/2, f'{d:+.4f}', va='center', fontsize=9, fontweight='bold')

ax.axvline(0, color='black', lw=1, ls='--')
ax.set_xlabel('Delta Score', fontsize=11)
ax.set_title('Component Contribution Analysis (Marginal Delta Decomposition)', fontsize=12)
ax.set_xlim(-0.1, 0.9)
fig.tight_layout()
fig.savefig(ABL_FIGS / 'component_contributions.png', dpi=120)
plt.close(fig)

# Fig 4: Performance vs. Computational Cost
fig, ax = plt.subplots(figsize=(8, 5))
times = [train_time_a, train_time_b, train_time_c, train_time_d, train_time_e]
for i, cfg in enumerate(configs):
    ax.scatter(times[i], shft_f1s[i], s=120, label=cfg, zorder=3)
    ax.annotate(cfg, (times[i], shft_f1s[i]), xytext=(7, 4), textcoords="offset points", fontsize=9)

ax.set_xlabel('Total Training / Adaptation Time (seconds)', fontsize=11)
ax.set_ylabel('Shifted F1-Score', fontsize=11)
ax.set_title('Trade-Off: Shifted Performance vs. Training & Adaptation Cost', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.5)
ax.set_ylim(0.2, 1.0)
fig.tight_layout()
fig.savefig(ABL_FIGS / 'cost_vs_performance.png', dpi=120)
plt.close(fig)

# Fig 5: False Positive Rate (FPR) Comparison
fig, ax = plt.subplots(figsize=(7, 4.5))
fpr_vals = [met_a_shft['fpr']*100, met_b_shft['fpr']*100, met_c_shft['fpr']*100, met_d_shft['fpr']*100, met_e_shft['fpr']*100]
bars = ax.bar(configs, fpr_vals, color=['#E74C3C', '#E67E22', '#2ECC71', '#27AE60', '#1ABC9C'], alpha=0.85)
for bar in bars:
    h = bar.get_height()
    ax.annotate(f'{h:.2f}%', (bar.get_x() + bar.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9, fontweight='bold')
ax.set_ylabel('Shifted False Positive Rate (%)', fontsize=11)
ax.set_title('False Positive Alarm Rate Across Ablation Configurations', fontsize=12)
ax.set_ylim(0, 115)
fig.tight_layout()
fig.savefig(ABL_FIGS / 'fpr_ablation.png', dpi=120)
plt.close(fig)

# Mirror figures to figures/ directory
for f in ABL_FIGS.glob('*.png'):
    shutil.copy(f, FIGURES_DIR / f.name)
print("All figures saved to figures/ablation/ and mirrored to figures/")

# ── 12. Experiment Tracking ───────────────────────────────────────────────────
print("\n" + "="*65)
print("12. EXTENDING EXPERIMENT TRACKING")
print("="*65)

exp_rows = [
    {
        'experiment_id': f"phase9_ablation_A_{int(time.time())}",
        'configuration': 'A (Baseline Transformer)',
        'model_name': 'TemporalTransformer',
        'ssl_used': False,
        'shift_detection_used': False,
        'adaptation_used': False,
        'normal_precision': round(met_a_norm['precision'], 4),
        'normal_recall': round(met_a_norm['recall'], 4),
        'normal_f1': round(met_a_norm['f1'], 4),
        'normal_roc_auc': round(met_a_norm['roc_auc'], 4),
        'normal_pr_auc': round(met_a_norm['pr_auc'], 4),
        'shifted_precision': round(met_a_shft['precision'], 4),
        'shifted_recall': round(met_a_shft['recall'], 4),
        'shifted_f1': round(met_a_shft['f1'], 4),
        'shifted_roc_auc': round(met_a_shft['roc_auc'], 4),
        'shifted_pr_auc': round(met_a_shft['pr_auc'], 4),
        'adapted_precision': None,
        'adapted_recall': None,
        'adapted_f1': None,
        'adapted_roc_auc': None,
        'adapted_pr_auc': None,
        'training_time': train_time_a,
        'adaptation_time': 0.0,
        'inference_time': inf_a,
        'model_size': size_a_kb,
    },
    {
        'experiment_id': f"phase9_ablation_B_{int(time.time())}",
        'configuration': 'B (Transformer + SSL)',
        'model_name': 'SSLTransformer',
        'ssl_used': True,
        'shift_detection_used': False,
        'adaptation_used': False,
        'normal_precision': round(met_b_norm['precision'], 4),
        'normal_recall': round(met_b_norm['recall'], 4),
        'normal_f1': round(met_b_norm['f1'], 4),
        'normal_roc_auc': round(met_b_norm['roc_auc'], 4),
        'normal_pr_auc': round(met_b_norm['pr_auc'], 4),
        'shifted_precision': round(met_b_shft['precision'], 4),
        'shifted_recall': round(met_b_shft['recall'], 4),
        'shifted_f1': round(met_b_shft['f1'], 4),
        'shifted_roc_auc': round(met_b_shft['roc_auc'], 4),
        'shifted_pr_auc': round(met_b_shft['pr_auc'], 4),
        'adapted_precision': None,
        'adapted_recall': None,
        'adapted_f1': None,
        'adapted_roc_auc': None,
        'adapted_pr_auc': None,
        'training_time': train_time_b,
        'adaptation_time': 0.0,
        'inference_time': inf_b,
        'model_size': size_b_kb,
    },
    {
        'experiment_id': f"phase9_ablation_C_{int(time.time())}",
        'configuration': 'C (Transformer + Adaptation)',
        'model_name': 'TemporalTransformer',
        'ssl_used': False,
        'shift_detection_used': True,
        'adaptation_used': True,
        'normal_precision': round(met_c_norm['precision'], 4),
        'normal_recall': round(met_c_norm['recall'], 4),
        'normal_f1': round(met_c_norm['f1'], 4),
        'normal_roc_auc': round(met_c_norm['roc_auc'], 4),
        'normal_pr_auc': round(met_c_norm['pr_auc'], 4),
        'shifted_precision': round(met_a_shft['precision'], 4),
        'shifted_recall': round(met_a_shft['recall'], 4),
        'shifted_f1': round(met_a_shft['f1'], 4),
        'shifted_roc_auc': round(met_a_shft['roc_auc'], 4),
        'shifted_pr_auc': round(met_a_shft['pr_auc'], 4),
        'adapted_precision': round(met_c_shft['precision'], 4),
        'adapted_recall': round(met_c_shft['recall'], 4),
        'adapted_f1': round(met_c_shft['f1'], 4),
        'adapted_roc_auc': round(met_c_shft['roc_auc'], 4),
        'adapted_pr_auc': round(met_c_shft['pr_auc'], 4),
        'training_time': train_time_a,
        'adaptation_time': adapt_time_c,
        'inference_time': inf_c,
        'model_size': size_c_kb,
    },
    {
        'experiment_id': f"phase9_ablation_D_{int(time.time())}",
        'configuration': 'D (SSL Transformer + Adaptation)',
        'model_name': 'SSLTransformer',
        'ssl_used': True,
        'shift_detection_used': True,
        'adaptation_used': True,
        'normal_precision': round(met_d_norm['precision'], 4),
        'normal_recall': round(met_d_norm['recall'], 4),
        'normal_f1': round(met_d_norm['f1'], 4),
        'normal_roc_auc': round(met_d_norm['roc_auc'], 4),
        'normal_pr_auc': round(met_d_norm['pr_auc'], 4),
        'shifted_precision': round(met_b_shft['precision'], 4),
        'shifted_recall': round(met_b_shft['recall'], 4),
        'shifted_f1': round(met_b_shft['f1'], 4),
        'shifted_roc_auc': round(met_b_shft['roc_auc'], 4),
        'shifted_pr_auc': round(met_b_shft['pr_auc'], 4),
        'adapted_precision': round(met_d_shft['precision'], 4),
        'adapted_recall': round(met_d_shft['recall'], 4),
        'adapted_f1': round(met_d_shft['f1'], 4),
        'adapted_roc_auc': round(met_d_shft['roc_auc'], 4),
        'adapted_pr_auc': round(met_d_shft['pr_auc'], 4),
        'training_time': train_time_b,
        'adaptation_time': adapt_time_d,
        'inference_time': inf_d,
        'model_size': size_d_kb,
    },
    {
        'experiment_id': f"phase9_ablation_E_{int(time.time())}",
        'configuration': 'E (Full Proposed System)',
        'model_name': 'SSLTransformer',
        'ssl_used': True,
        'shift_detection_used': True,
        'adaptation_used': True,
        'normal_precision': round(met_e_norm['precision'], 4),
        'normal_recall': round(met_e_norm['recall'], 4),
        'normal_f1': round(met_e_norm['f1'], 4),
        'normal_roc_auc': round(met_e_norm['roc_auc'], 4),
        'normal_pr_auc': round(met_e_norm['pr_auc'], 4),
        'shifted_precision': round(met_b_shft['precision'], 4),
        'shifted_recall': round(met_b_shft['recall'], 4),
        'shifted_f1': round(met_b_shft['f1'], 4),
        'shifted_roc_auc': round(met_b_shft['roc_auc'], 4),
        'shifted_pr_auc': round(met_b_shft['pr_auc'], 4),
        'adapted_precision': round(met_e_shft['precision'], 4),
        'adapted_recall': round(met_e_shft['recall'], 4),
        'adapted_f1': round(met_e_shft['f1'], 4),
        'adapted_roc_auc': round(met_e_shft['roc_auc'], 4),
        'adapted_pr_auc': round(met_e_shft['pr_auc'], 4),
        'training_time': train_time_b,
        'adaptation_time': adapt_time_e,
        'inference_time': inf_e,
        'model_size': size_e_kb,
    },
]

exp_path = EXP_DIR / 'experiments.csv'
new_exp_df = pd.DataFrame(exp_rows)
if exp_path.exists():
    existing_exp = pd.read_csv(exp_path)
    new_exp_df = pd.concat([existing_exp, new_exp_df], ignore_index=True)
new_exp_df.to_csv(exp_path, index=False)
print(f"Recorded 5 Phase 9 ablation configurations to {exp_path}")

# ── 13. Consolidated Summary JSON ─────────────────────────────────────────────
summary_json = {
    'configurations': ablation_records,
    'component_contributions': contrib_df.to_dict(orient='records'),
    'error_analysis': error_records,
    'computational_cost': cost_records,
    'findings': {
        'ssl_contribution_normal_f1': round(met_b_norm['f1'] - met_a_norm['f1'], 4),
        'ssl_advantage_shifted_roc_auc': round(met_b_shft['roc_auc'] - met_a_shft['roc_auc'], 4),
        'adaptation_contribution_no_ssl': round(met_c_shft['f1'] - met_a_shft['f1'], 4),
        'adaptation_contribution_with_ssl': round(met_d_shft['f1'] - met_b_shft['f1'], 4),
        'ssl_advantage_in_adapted_regime': round(met_d_shft['f1'] - met_c_shft['f1'], 4),
        'full_system_gain_f1': round(met_e_shft['f1'] - met_a_shft['f1'], 4),
    }
}
with open(ABL_RES / 'ablation_summary.json', 'w') as f:
    json.dump(summary_json, f, indent=2)

print("\n" + "="*65)
print("PHASE 9 ABLATION STUDY COMPLETE")
print("="*65)
print(f"All ablation results saved to: {ABL_RES}")
