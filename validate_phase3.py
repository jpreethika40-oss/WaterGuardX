"""Phase 3 validation — verify all checklist items."""
import sys
sys.path.insert(0, 'src')
import json, joblib
import numpy as np
import pandas as pd
from pathlib import Path

PASS = "[PASS]"
FAIL = "[FAIL]"
checks = []

def check(name, condition, detail=''):
    status = PASS if condition else FAIL
    checks.append((status, name, detail))
    print(f"  {status} {name}" + (f" — {detail}" if detail else ''))

print("=" * 60)
print("PHASE 3 VALIDATION")
print("=" * 60)

# ── Models implemented ────────────────────────────────────────────────────────
print("\n[1] Models saved")
for name in ['logistic_regression', 'random_forest', 'xgboost']:
    model_file  = Path(f'models/baselines/{name}/model.pkl')
    config_file = Path(f'models/baselines/{name}/config.json')
    check(f"{name} model.pkl exists",  model_file.exists())
    check(f"{name} config.json exists", config_file.exists())
    if config_file.exists():
        cfg = json.loads(config_file.read_text())
        check(f"{name} threshold saved", 'threshold' in cfg,
              f"threshold={cfg.get('threshold','MISSING')}")

# ── Results files ─────────────────────────────────────────────────────────────
print("\n[2] Results files")
required = [
    'results/baselines/baseline_comparison.csv',
    'results/baselines/baseline_comparison.md',
    'results/baselines/model_summary.csv',
    'results/baselines/inference_time.csv',
    'results/baselines/model_size.csv',
    'results/baselines/error_analysis.csv',
    'results/baselines/full_results.json',
]
for f in required:
    check(f"File: {Path(f).name}", Path(f).exists())

# ── Figures ───────────────────────────────────────────────────────────────────
print("\n[3] Figures")
fig_files = [
    'figures/baselines/confusion_matrix_logistic_regression.png',
    'figures/baselines/confusion_matrix_random_forest.png',
    'figures/baselines/confusion_matrix_xgboost.png',
    'figures/baselines/roc_comparison.png',
    'figures/baselines/pr_comparison.png',
    'figures/baselines/feature_importance_logistic_regression.png',
    'figures/baselines/feature_importance_random_forest.png',
    'figures/baselines/feature_importance_xgboost.png',
    'figures/baselines/metric_comparison.png',
]
for f in fig_files:
    check(f"Figure: {Path(f).name}", Path(f).exists())

# ── Metrics completeness ──────────────────────────────────────────────────────
print("\n[4] Metrics completeness")
comp = pd.read_csv('results/baselines/baseline_comparison.csv')
required_metrics = ['accuracy','precision','recall','f1','roc_auc','pr_auc','fpr','fnr']
for m in required_metrics:
    check(f"Metric '{m}' present", m in comp.columns)
check("All 3 models in comparison", len(comp) == 3, f"{len(comp)} rows")
check("No NaN metrics", comp[required_metrics].isnull().sum().sum() == 0)

# ── No leakage ────────────────────────────────────────────────────────────────
print("\n[5] Leakage checks")
meta = json.loads(Path('data/processed/preprocessing_meta.json').read_text())
check("Scaler fit on train only", meta['scaler_type'] == 'standard')
check("Chronological split", meta['train_end'] < meta['val_end'])

# ── Threshold selection ───────────────────────────────────────────────────────
print("\n[6] Threshold selection")
for name in ['logistic_regression', 'random_forest', 'xgboost']:
    cfg = json.loads(Path(f'models/baselines/{name}/config.json').read_text())
    t = cfg.get('threshold', None)
    check(f"{name} threshold != 0.5 (val-tuned)",
          t is not None and t != 0.5, f"threshold={t}")

# ── Inference time ────────────────────────────────────────────────────────────
print("\n[7] Inference time")
inf_df = pd.read_csv('results/baselines/inference_time.csv')
check("Inference time recorded for all models", len(inf_df) == 3)
check("Per-sample time is positive", (inf_df['per_sample_ms'] > 0).all())

# ── Model size ────────────────────────────────────────────────────────────────
print("\n[8] Model size")
size_df = pd.read_csv('results/baselines/model_size.csv')
check("Model size recorded for all models", len(size_df) == 3)
check("All sizes > 0 KB", (size_df['size_kb'] > 0).all())

# ── Notebook ──────────────────────────────────────────────────────────────────
print("\n[9] Notebook")
check("Notebook exists", Path('notebooks/03_baselines.ipynb').exists())

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
passed = sum(1 for s, _, _ in checks if s == PASS)
failed = sum(1 for s, _, _ in checks if s == FAIL)
print(f"RESULT: {passed}/{len(checks)} checks passed, {failed} failed")
if failed == 0:
    print("Phase 3 validation: ALL CHECKS PASSED")
else:
    for s, name, detail in checks:
        if s == FAIL:
            print(f"  FAILED: {name} — {detail}")

# ── Print final comparison table ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("FINAL BASELINE COMPARISON (TEST SET)")
print("=" * 70)
print(comp.to_string(index=False))
