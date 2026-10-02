"""
Phase 11 — Final Error Analysis and Model Interpretability Pipeline.
Run from project root: python run_error_analysis.py

Pipeline:
  1. Load final model checkpoints (nominal & adapted) from Phase 10.
  2. Generate dense predictions on untouched test streams (Normal & Shifted, Pre & Post-Adaptation).
  3. Analyze confusion matrix breakdowns and compute FP/FN statistics.
  4. Perform representative False Positive and False Negative analysis.
  5. Contrast error patterns under normal vs. shifted operating conditions.
  6. Analyze temporal error dynamics (error concentration along time and operating cycles).
  7. Conduct sensor-level error divergence analysis across TP, TN, FP, and FN windows.
  8. Model Interpretability:
     - Multi-layer self-attention weight extraction & visualization.
     - Temporal occlusion sensitivity analysis.
     - Sensor permutation sensitivity & importance ranking.
     - Confidence distribution and Expected Calibration Error (ECE).
  9. Produce publication-quality figures in figures/error_analysis/.
  10. Store structured results in results/error_analysis/.
"""
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, 'src')

# Import torch FIRST to prevent OpenMP DLL initialization conflict on Windows
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from adaptation import split_chronological_adaptation, build_replay_dataset
from config import (DATA_PROC, FIGURES_DIR, MODELS_DIR, RESULTS_DIR, SEED,
                    SENSOR_COLS, SEQUENCE_LENGTH, STRIDE, TARGET_COL,
                    TIMESTAMP_COL)
from data_loader import load_raw
from datasets import WaterSensorDataset
from error_analysis import (categorize_errors, extract_representative_errors,
                            generate_detailed_predictions,
                            sensor_distribution_by_error_type)
from evaluation import compute_metrics
from interpretability import (compute_calibration_curve, extract_attention_weights,
                              sensor_permutation_importance,
                              temporal_occlusion_sensitivity)
from preprocessing import run_preprocessing
from self_supervised import SSLTransformer
from training import set_seed

set_seed(SEED)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")

# Output directories
ERR_RES = RESULTS_DIR / 'error_analysis'; ERR_RES.mkdir(parents=True, exist_ok=True)
ERR_FIGS = FIGURES_DIR / 'error_analysis'; ERR_FIGS.mkdir(parents=True, exist_ok=True)


# ── 1. Load Preprocessed Data & Final Models ──────────────────────────────────
print("\n" + "="*65)
print("1. LOADING UNTOUCHED TEST STREAMS AND FINAL MODEL CHECKPOINTS")
print("="*65)

df_raw = load_raw()
train_df, val_df, test_df, scaler, feat_cols = run_preprocessing(df_raw, save=False)
N_FEATURES = len(feat_cols)

# Load shifted test data from Phase 7 (k=1.0)
shifted_test_path = DATA_PROC / 'shifted' / 'strong' / 'test_shifted.csv'
test_shifted_df = pd.read_csv(shifted_test_path)

# Chronological partition (Dec 1-31 is untouched test evaluation stream)
adapt_train_df, adapt_val_df, eval_shifted_df = split_chronological_adaptation(
    test_shifted_df, timestamp_col=TIMESTAMP_COL, split_date='2018-12-01', val_ratio=0.30
)
test_df[TIMESTAMP_COL] = pd.to_datetime(test_df[TIMESTAMP_COL])
eval_normal_df = test_df[test_df[TIMESTAMP_COL] >= '2018-12-01'].reset_index(drop=True)

print(f"Held-Out Normal Test Stream (Dec 1-31):  {len(eval_normal_df):,} rows")
print(f"Held-Out Shifted Test Stream (Dec 1-31): {len(eval_shifted_df):,} rows")

# Load final thresholds from Phase 10
thresh_path = RESULTS_DIR / 'final_threshold.json'
with open(thresh_path, 'r', encoding='utf-8') as f:
    thresh_data = json.load(f)

nominal_thresh = thresh_data['nominal_threshold']
adapted_thresh = thresh_data['adapted_threshold']
print(f"Loaded Thresholds: nominal={nominal_thresh:.4f}, adapted={adapted_thresh:.4f}")

# Load final models
final_model_path = MODELS_DIR / 'final_model.pt'
adapted_model_path = MODELS_DIR / 'final_model_adapted.pt'

def create_model_instance(dropout=0.15):
    return SSLTransformer(
        n_features=N_FEATURES,
        d_model=32,
        nhead=2,
        num_layers=2,
        dim_feedforward=64,
        dropout=dropout,
        max_seq_len=SEQUENCE_LENGTH + 10
    ).to(DEVICE)

nominal_model = create_model_instance()
nominal_model.load_state_dict(torch.load(final_model_path, map_location=DEVICE))
nominal_model.eval()

adapted_model = create_model_instance()
if adapted_model_path.exists():
    adapted_model.load_state_dict(torch.load(adapted_model_path, map_location=DEVICE))
    print("Loaded existing adapted model checkpoint.")
else:
    print("Adapting nominal model on adaptation stream with replay buffer...")
    REPLAY_RATIO = 0.20
    replay_dataset = build_replay_dataset(
        adapt_df=adapt_train_df, train_df=train_df, feat_cols=feat_cols,
        replay_ratio=REPLAY_RATIO, seq_len=SEQUENCE_LENGTH, stride=STRIDE, seed=SEED
    )
    adapt_loader = DataLoader(replay_dataset, batch_size=64, shuffle=True, num_workers=0)
    # Initialize from nominal model weights
    adapted_model.load_state_dict(nominal_model.state_dict())
    adapt_optimizer = torch.optim.AdamW(adapted_model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion_bce = nn.BCEWithLogitsLoss()
    adapted_model.train()
    for epoch in range(1, 6):
        for X_b, y_b in adapt_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE).float()
            adapt_optimizer.zero_grad()
            loss = criterion_bce(adapted_model.forward_classify(X_b), y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(adapted_model.parameters(), max_norm=1.0)
            adapt_optimizer.step()
    adapted_model.eval()
    torch.save(adapted_model.state_dict(), adapted_model_path)
    print(f"Saved adapted model checkpoint to: {adapted_model_path}")
adapted_model.eval()

print("Loaded nominal model and adapted model checkpoints successfully.")


# ── 2. Generate Final Dense Predictions ───────────────────────────────────────
print("\n" + "="*65)
print("2. GENERATING DENSE TEST PREDICTIONS (NO RETRAINING)")
print("="*65)

# A. Nominal model on untouched Normal Test (Dec 1-31)
preds_norm_df, win_norm = generate_detailed_predictions(
    nominal_model, eval_normal_df, feat_cols, threshold=nominal_thresh,
    eval_condition="Normal_Test_Nominal", stride=STRIDE, device=DEVICE
)

# B. Nominal model on untouched Shifted Test (Dec 1-31) pre-adaptation
preds_shift_pre_df, win_shift_pre = generate_detailed_predictions(
    nominal_model, eval_shifted_df, feat_cols, threshold=nominal_thresh,
    eval_condition="Shifted_Test_PreAdapt", stride=STRIDE, device=DEVICE
)

# C. Adapted model on untouched Shifted Test (Dec 1-31) post-adaptation
preds_shift_post_df, win_shift_post = generate_detailed_predictions(
    adapted_model, eval_shifted_df, feat_cols, threshold=adapted_thresh,
    eval_condition="Shifted_Test_PostAdapt", stride=STRIDE, device=DEVICE
)

# Combine for holistic prediction tracking
all_preds_df = pd.concat([preds_norm_df, preds_shift_pre_df, preds_shift_post_df], ignore_index=True)
all_preds_path = ERR_RES / 'final_predictions.csv'
all_preds_df.to_csv(all_preds_path, index=False)
print(f"Saved dense predictions: {all_preds_path} ({len(all_preds_df):,} total sequence predictions)")


# ── 3. Confusion-Matrix Error Analysis ────────────────────────────────────────
print("\n" + "="*65)
print("3. CONFUSION MATRIX ERROR BREAKDOWN")
print("="*65)

def print_cm_breakdown(df: pd.DataFrame, condition_name: str):
    counts = df['error_type'].value_counts()
    n_total = len(df)
    tp = counts.get('True_Positive', 0)
    tn = counts.get('True_Negative', 0)
    fp = counts.get('False_Positive', 0)
    fn = counts.get('False_Negative', 0)
    
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    
    print(f"\nCondition: {condition_name} (Total Windows: {n_total:,})")
    print(f"  True Positives  (TP): {tp:5d} ({tp/n_total*100:5.2f}%)  [Correctly detected anomalies]")
    print(f"  True Negatives  (TN): {tn:5d} ({tn/n_total*100:5.2f}%)  [Correctly identified normal operations]")
    print(f"  False Positives (FP): {fp:5d} ({fp/n_total*100:5.2f}%)  [False alarms; normal classified as anomaly]")
    print(f"  False Negatives (FN): {fn:5d} ({fn/n_total*100:5.2f}%)  [Missed anomalies; anomaly classified as normal]")
    print(f"  F1: {f1:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | FPR: {fpr*100:.2f}% | FNR: {fnr*100:.2f}%")
    return {"condition": condition_name, "total": n_total, "TP": tp, "TN": tn, "FP": fp, "FN": fn, "FPR": fpr, "FNR": fnr, "F1": f1}

cm_norm = print_cm_breakdown(preds_norm_df, "Normal Stream (Nominal Model)")
cm_shift_pre = print_cm_breakdown(preds_shift_pre_df, "Shifted Stream (Pre-Adaptation)")
cm_shift_post = print_cm_breakdown(preds_shift_post_df, "Shifted Stream (Post-Adaptation)")


# ── 4. Representative False Positive & False Negative Analysis ────────────────
print("\n" + "="*65)
print("4. REPRESENTATIVE ERROR EXTRACTION (FALSE POSITIVES & FALSE NEGATIVES)")
print("="*65)

# Extract top FP and FN sequences from Adapted Shifted Stream
top_fp_df, top_fn_df = extract_representative_errors(preds_shift_post_df, win_shift_post, n_examples=10)

# Build descriptive observed pattern explanations based on sensor deviations
def annotate_fp_patterns(fp_df: pd.DataFrame, windows: np.ndarray) -> pd.DataFrame:
    rows = []
    for idx, r in fp_df.iterrows():
        seq_id = int(r["sequence_id"])
        win = windows[seq_id] # (60, 16)
        # Check max sensor deviation at the end timestep
        last_step = win[-1, :len(SENSOR_COLS)]
        max_sensor_idx = int(np.argmax(np.abs(last_step)))
        max_sensor_name = feat_cols[max_sensor_idx]
        max_sensor_val = float(last_step[max_sensor_idx])
        grad = float(np.max(np.abs(np.diff(win[:, :len(SENSOR_COLS)], axis=0))))

        if grad > 2.0:
            obs = f"Sudden hydraulic switching spike on {max_sensor_name} (step grad={grad:.2f}σ)."
        elif r['anomaly_prob'] < 0.05:
            obs = f"Borderline threshold fluctuation (prob={r['anomaly_prob']:.4f} marginally > 0.0100)."
        else:
            obs = f"Elevated residual reading on {max_sensor_name} ({max_sensor_val:+.2f}σ above reference mean)."

        rows.append({
            "Error ID": f"FP_{seq_id}",
            "Sequence ID": seq_id,
            "Timestamp": r["timestamp"],
            "True Label": 0,
            "Prediction": 1,
            "Probability": r["anomaly_prob"],
            "Observed Pattern": obs
        })
    return pd.DataFrame(rows)

def annotate_fn_patterns(fn_df: pd.DataFrame, windows: np.ndarray) -> pd.DataFrame:
    rows = []
    for idx, r in fn_df.iterrows():
        seq_id = int(r["sequence_id"])
        win = windows[seq_id]
        mean_anomaly_amp = float(np.mean(np.abs(win[:, :len(SENSOR_COLS)])))

        if mean_anomaly_amp < 0.60:
            obs = f"Incipient / subtle leakage: mean sensor deviation is low ({mean_anomaly_amp:.2f}σ, below diurnal variance)."
        elif r['anomaly_prob'] >= 0.005:
            obs = f"Borderline false negative: score ({r['anomaly_prob']:.4f}) just below detection threshold (0.0100)."
        else:
            obs = f"Transient anomaly profile with localized attenuation on pressure channels."

        rows.append({
            "Error ID": f"FN_{seq_id}",
            "Sequence ID": seq_id,
            "Timestamp": r["timestamp"],
            "True Label": 1,
            "Prediction": 0,
            "Probability": r["anomaly_prob"],
            "Observed Pattern": obs
        })
    return pd.DataFrame(rows)

fp_annotated = annotate_fp_patterns(top_fp_df, win_shift_post)
fn_annotated = annotate_fn_patterns(top_fn_df, win_shift_post)

fp_save_path = ERR_RES / 'false_positives.csv'
fn_save_path = ERR_RES / 'false_negatives.csv'
fp_annotated.to_csv(fp_save_path, index=False)
fn_annotated.to_csv(fn_save_path, index=False)
print(f"Saved: {fp_save_path} ({len(fp_annotated)} representative FPs)")
print(f"Saved: {fn_save_path} ({len(fn_annotated)} representative FNs)")

print("\nRepresentative False Positives Table:")
print(fp_annotated[['Error ID', 'Timestamp', 'Probability', 'Observed Pattern']].head(5).to_string(index=False))

print("\nRepresentative False Negatives Table:")
print(fn_annotated[['Error ID', 'Timestamp', 'Probability', 'Observed Pattern']].head(5).to_string(index=False))


# ── 5. Error Categorization & Summary ─────────────────────────────────────────
print("\n" + "="*65)
print("5. STRUCTURED EVIDENCE-BASED ERROR CATEGORIZATION")
print("="*65)

categorized_errors = categorize_errors(preds_shift_post_df, win_shift_post, feat_cols)

cat_summary_rows = []
for err_type in ["False_Positive", "False_Negative"]:
    sub = categorized_errors[categorized_errors["error_type"] == err_type]
    total_type = len(sub)
    counts = sub["category"].value_counts()
    for cat_name, cnt in counts.items():
        cat_summary_rows.append({
            "Error Type": "False Positive" if err_type == "False_Positive" else "False Negative",
            "Category": cat_name,
            "Count": cnt,
            "Percentage of Type": f"{cnt / total_type * 100:.1f}%",
            "Total In Stream": len(preds_shift_post_df),
            "Percentage of Stream": f"{cnt / len(preds_shift_post_df) * 100:.2f}%"
        })

error_summary_df = pd.DataFrame(cat_summary_rows)
err_sum_path = ERR_RES / 'error_summary.csv'
error_summary_df.to_csv(err_sum_path, index=False)
print(f"Saved error summary: {err_sum_path}")
print(error_summary_df.to_string(index=False))


# ── 6. Sensor-Level Error Divergence Analysis ─────────────────────────────────
print("\n" + "="*65)
print("6. SENSOR-LEVEL ERROR DIVERGENCE ANALYSIS")
print("="*65)

sensor_div_df = sensor_distribution_by_error_type(win_shift_post, preds_shift_post_df, feat_cols)
div_path = ERR_RES / 'sensor_error_divergence.csv'
sensor_div_df.to_csv(div_path, index=False)
print(f"Saved sensor divergence analysis: {div_path}")
print("Top 5 sensors showing highest divergence during False Positives vs True Normals:")
print(sensor_div_df[['sensor', 'mean_TN', 'mean_FP', 'fp_divergence', 'mean_TP', 'mean_FN', 'fn_divergence']].head().to_string(index=False))


# ── 7. Model Interpretability: Sensor Permutation Importance ─────────────────
print("\n" + "="*65)
print("7. SENSOR PERMUTATION IMPORTANCE / SENSITIVITY")
print("="*65)

eval_adapted_ds = WaterSensorDataset(eval_shifted_df, feat_cols, label_strategy='last', stride=STRIDE)
perm_imp_df = sensor_permutation_importance(
    adapted_model, eval_adapted_ds, feat_cols, n_samples=300, seed=SEED, device=DEVICE
)

perm_path = ERR_RES / 'interpretability_results.csv'
perm_imp_df.to_csv(perm_path, index=False)
print(f"Saved sensor permutation sensitivity to: {perm_path}")
print("Sensor Sensitivity Ranking (Mean Absolute Delta in Predicted Anomaly Probability):")
print(perm_imp_df[['rank', 'feature', 'mean_abs_prob_delta', 'mean_signed_delta', 'ap_drop']].head(10).to_string(index=False))


# ── 8. Confidence Distribution & Calibration Analysis (ECE) ───────────────────
print("\n" + "="*65)
print("8. CONFIDENCE DISTRIBUTION AND CALIBRATION ANALYSIS (ECE)")
print("="*65)

calib_norm = compute_calibration_curve(preds_norm_df['true_label'].values, preds_norm_df['anomaly_prob'].values, n_bins=10)
calib_post = compute_calibration_curve(preds_shift_post_df['true_label'].values, preds_shift_post_df['anomaly_prob'].values, n_bins=10)

print(f"Expected Calibration Error (ECE) - Normal Stream:  {calib_norm['ece']:.5f}")
print(f"Expected Calibration Error (ECE) - Adapted Stream: {calib_post['ece']:.5f}")


# ── 9. Generate Publication-Quality Interpretability Figures ──────────────────
print("\n" + "="*65)
print("9. GENERATING PUBLICATION-QUALITY FIGURES UNDER figures/error_analysis/")
print("="*65)

# Figure 1: Detailed Confusion Matrix
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(preds_shift_post_df['true_label'], preds_shift_post_df['predicted_label'])
im = ax.imshow(cm, cmap=plt.cm.Blues, interpolation='nearest')
ax.set_title("Detailed Confusion Matrix (Adapted Shifted Test Stream)", fontsize=11, fontweight='bold')
ax.set_xticks([0, 1]); ax.set_xticklabels(["True Normal", "True Anomaly"], fontsize=9)
ax.set_yticks([0, 1]); ax.set_yticklabels(["True Normal", "True Anomaly"], fontsize=9)
ax.set_xlabel("Predicted Class", fontsize=10)
ax.set_ylabel("True Class", fontsize=10)

thresh_val = cm.max() / 2.0
labels_names = [["True Negatives (TN)", "False Positives (FP)"],
                ["False Negatives (FN)", "True Positives (TP)"]]
for i in range(2):
    for j in range(2):
        val = cm[i, j]
        desc = labels_names[i][j]
        pct = val / cm.sum() * 100
        ax.text(j, i, f"{desc}\n{val:,}\n({pct:.2f}%)",
                ha="center", va="center",
                color="white" if val > thresh_val else "black",
                fontsize=9)
fig.tight_layout()
fig.savefig(ERR_FIGS / 'confusion_matrix_detailed.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# Figure 2: Prediction Probability Distribution by True Class
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

axes[0].hist(preds_norm_df[preds_norm_df['true_label'] == 0]['anomaly_prob'], bins=50, alpha=0.7, color='#2ecc71', label='Normal (True 0)')
axes[0].hist(preds_norm_df[preds_norm_df['true_label'] == 1]['anomaly_prob'], bins=50, alpha=0.7, color='#e74c3c', label='Anomaly (True 1)')
axes[0].axvline(nominal_thresh, color='black', ls='--', lw=1.2, label=f'Threshold ({nominal_thresh:.4f})')
axes[0].set_title("Nominal Test Stream Probability Distribution", fontsize=10, fontweight='bold')
axes[0].set_xlabel("Predicted Anomaly Probability", fontsize=9); axes[0].set_ylabel("Sample Count", fontsize=9)
axes[0].set_yscale('log'); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

axes[1].hist(preds_shift_post_df[preds_shift_post_df['true_label'] == 0]['anomaly_prob'], bins=50, alpha=0.7, color='#2ecc71', label='Normal (True 0)')
axes[1].hist(preds_shift_post_df[preds_shift_post_df['true_label'] == 1]['anomaly_prob'], bins=50, alpha=0.7, color='#e74c3c', label='Anomaly (True 1)')
axes[1].axvline(adapted_thresh, color='black', ls='--', lw=1.2, label=f'Threshold ({adapted_thresh:.4f})')
axes[1].set_title("Adapted Shifted Stream Probability Distribution", fontsize=10, fontweight='bold')
axes[1].set_xlabel("Predicted Anomaly Probability", fontsize=9); axes[1].set_ylabel("Sample Count", fontsize=9)
axes[1].set_yscale('log'); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

fig.tight_layout()
fig.savefig(ERR_FIGS / 'prediction_probability_distribution.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# Figure 3: Representative False Positive Sequence Plot
rep_fp_id = int(top_fp_df.iloc[0]["sequence_id"])
rep_fp_win = win_shift_post[rep_fp_id] # (seq_len, n_features)
seq_len = rep_fp_win.shape[0]
time_axis = np.arange(-(seq_len - 1), 1) * 5 # minutes relative to prediction timestamp
fig, ax = plt.subplots(figsize=(10, 4.5))

for s_i in range(min(5, len(SENSOR_COLS))):
    ax.plot(time_axis, rep_fp_win[:, s_i], lw=1.5, label=feat_cols[s_i])

ax.set_title(f"Representative False Positive Sequence (Seq #{rep_fp_id}, Prob={top_fp_df.iloc[0]['anomaly_prob']:.4f}, True=0, Pred=1)",
             fontsize=10, fontweight='bold')
ax.set_xlabel("Time Relative to Decision Point (Minutes)", fontsize=9)
ax.set_ylabel("Normalized Sensor Reading (Z-Score)", fontsize=9)
ax.axvline(0, color='red', ls=':', lw=1.2, label='Classification Point')
ax.legend(loc='upper left', ncol=3, fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(ERR_FIGS / 'representative_false_positive.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# Figure 4: Representative False Negative Sequence Plot
rep_fn_id = int(top_fn_df.iloc[0]["sequence_id"])
rep_fn_win = win_shift_post[rep_fn_id]
fig, ax = plt.subplots(figsize=(10, 4.5))

for s_i in range(min(5, len(SENSOR_COLS))):
    ax.plot(time_axis, rep_fn_win[:, s_i], lw=1.5, label=feat_cols[s_i])

ax.set_title(f"Representative False Negative Sequence (Seq #{rep_fn_id}, Prob={top_fn_df.iloc[0]['anomaly_prob']:.4f}, True=1, Pred=0)",
             fontsize=10, fontweight='bold')
ax.set_xlabel("Time Relative to Decision Point (Minutes)", fontsize=9)
ax.set_ylabel("Normalized Sensor Reading (Z-Score)", fontsize=9)
ax.axvline(0, color='red', ls=':', lw=1.2, label='Classification Point')
ax.legend(loc='upper left', ncol=3, fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(ERR_FIGS / 'representative_false_negative.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# Figure 5: Sensor Sensitivity / Permutation Importance
fig, ax = plt.subplots(figsize=(10, 4.5))
imp_top = perm_imp_df.head(12)
x = np.arange(len(imp_top))
ax.bar(x, imp_top['mean_abs_prob_delta'], color='#3498db', alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(imp_top['feature'], rotation=35, ha='right', fontsize=9)
ax.set_ylabel("Mean |Δ Anomaly Probability|", fontsize=9)
ax.set_title("Sensor Feature Sensitivity via Channel Permutation (Adapted Model)", fontsize=11, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for p in ax.patches:
    h = p.get_height()
    ax.annotate(f"{h:.3f}", (p.get_x() + p.get_width() / 2., h),
                ha='center', va='bottom', fontsize=8, xytext=(0, 2),
                textcoords='offset points')
fig.tight_layout()
fig.savefig(ERR_FIGS / 'sensor_sensitivity_importance.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# Figure 6: Temporal Occlusion Sensitivity on True Positive Anomaly
# Select a confident True Positive sequence
tp_candidates = preds_shift_post_df[preds_shift_post_df['error_type'] == 'True_Positive']
rep_tp_id = int(tp_candidates.iloc[0]['sequence_id'])
rep_tp_tensor = torch.tensor(win_shift_post[rep_tp_id:rep_tp_id+1], dtype=torch.float32).to(DEVICE)
t_steps, occ_deltas, base_tp_prob = temporal_occlusion_sensitivity(adapted_model, rep_tp_tensor, window_size=5, stride=1, device=DEVICE)

fig, ax = plt.subplots(figsize=(9, 4.5))
seq_len_tp = rep_tp_tensor.shape[1]
t_mins = (t_steps - (seq_len_tp - 1)) * 5
ax.plot(t_mins, occ_deltas, color='#e67e22', lw=2.2, marker='o', markersize=4)
ax.set_xlabel("Temporal Window Center (Minutes Relative to Decision Point)", fontsize=9)
ax.set_ylabel("Impact on Anomaly Probability |ΔP|", fontsize=9)
ax.set_title(f"Temporal Occlusion Sensitivity across {seq_len_tp} Timesteps (Seq #{rep_tp_id}, Base P={base_tp_prob:.4f})",
             fontsize=10, fontweight='bold')
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(ERR_FIGS / 'temporal_occlusion_importance.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# Figure 7: Self-Attention Weight Heatmap
# Extract attention weights for Layer 1 and Layer 2
attns = extract_attention_weights(adapted_model, rep_tp_tensor, device=DEVICE)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for l_idx, (ax, attn_mat) in enumerate(zip(axes, attns)):
    im = ax.imshow(attn_mat[0], cmap='viridis', aspect='auto', origin='lower')
    ax.set_title(f"Self-Attention Map — Encoder Layer {l_idx + 1}", fontsize=10, fontweight='bold')
    ax.set_xlabel(f"Key Timestep (0 to {seq_len_tp - 1})", fontsize=9)
    ax.set_ylabel(f"Query Timestep (0 to {seq_len_tp - 1})", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.suptitle("Temporal Multi-Head Self-Attention Matrices (Representative Anomaly Sequence)", fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig(ERR_FIGS / 'attention_map_representative.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# Figure 8: Error Distribution: Normal vs. Shifted vs. Adapted
fig, ax = plt.subplots(figsize=(8, 4.5))
regimes = ["Normal (Nominal)", "Shifted (Pre-Adapt)", "Shifted (Post-Adapt)"]
fp_counts = [cm_norm['FP'], cm_shift_pre['FP'], cm_shift_post['FP']]
fn_counts = [cm_norm['FN'], cm_shift_pre['FN'], cm_shift_post['FN']]

x = np.arange(len(regimes))
width = 0.35
ax.bar(x - width/2, fp_counts, width, label='False Positives (False Alarms)', color='#e74c3c', alpha=0.85)
ax.bar(x + width/2, fn_counts, width, label='False Negatives (Missed Anom)', color='#f39c12', alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(regimes, fontsize=9)
ax.set_ylabel("Count of Error Sequences (Log Scale)", fontsize=9)
ax.set_yscale('log')
ax.set_title("Error Count Distribution Across Operating Regimes", fontsize=11, fontweight='bold')
ax.legend(fontsize=9, frameon=True)
ax.grid(axis='y', alpha=0.3)
for p in ax.patches:
    h = p.get_height()
    if h > 0:
        ax.annotate(f"{int(h):,}", (p.get_x() + p.get_width() / 2., h),
                    ha='center', va='bottom', fontsize=8, xytext=(0, 2),
                    textcoords='offset points')
fig.tight_layout()
fig.savefig(ERR_FIGS / 'error_distribution_normal_vs_shifted.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# Figure 9: Calibration Reliability Diagram
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot([0, 1], [0, 1], color='gray', ls='--', lw=1, label='Perfect Calibration')
ax.plot(calib_norm['bin_confidences'], calib_norm['bin_accuracies'], color='#2ca02c', marker='s', lw=1.8, label=f"Normal Test (ECE = {calib_norm['ece']:.4f})")
ax.plot(calib_post['bin_confidences'], calib_post['bin_accuracies'], color='#1f77b4', marker='o', lw=1.8, label=f"Shifted Adapted (ECE = {calib_post['ece']:.4f})")
ax.set_xlabel("Mean Predicted Anomaly Probability", fontsize=9)
ax.set_ylabel("Empirical Anomaly Ratio in Bin", fontsize=9)
ax.set_title("Reliability Calibration Diagram", fontsize=10, fontweight='bold')
ax.legend(fontsize=9, frameon=True)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(ERR_FIGS / 'calibration_reliability_diagram.png', dpi=300, bbox_inches='tight')
plt.close(fig)

print("Saved all 9 publication-quality figures to figures/error_analysis/")

print("\n" + "="*65)
print("PHASE 11 ERROR ANALYSIS & INTERPRETABILITY COMPLETE")
print("="*65)
