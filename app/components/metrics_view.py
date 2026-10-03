"""
WaterGuardX — Metrics, Benchmark & Interpretability View.
Renders official research evaluation results from Phase 10 and Phase 11.
Dedicated components for Tab 6 (Benchmarks) and Tab 7 (Interpretability).
"""
from pathlib import Path
import pandas as pd
import streamlit as st

from utils.app_utils import (
    load_phase10_results,
    load_phase11_results,
    FIGURES_DIR
)


def render_benchmarks_tab():
    """
    Renders dedicated Tab 6: Official Phase 10 Research Evaluation & Cross-Model Benchmarks.
    """
    data10 = load_phase10_results()
    res10 = data10.get("results", {})
    metrics_norm = res10.get("metrics_normal_test", {})
    metrics_shift_pre = res10.get("metrics_shifted_test_pre_adapt", {})
    metrics_shift_post = res10.get("metrics_shifted_test_post_adapt", {})
    metrics_retention = res10.get("metrics_normal_retention_post_adapt", {})

    st.markdown("### 🏆 6. Official Research Evaluation & Architecture Benchmark")
    st.caption("Comprehensive evaluation conducted on untouched held-out December 1–31, 2018 test data ($N = 8,881$ evaluation windows).")

    # 1. Official Phase 10 KPI Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Test F1-Score (Nominal)", f"{metrics_norm.get('f1', 0.9557):.4f}", "95.57%")
    with col2:
        st.metric("Test Recall (Sensitivity)", f"{metrics_norm.get('recall', 0.9893):.4f}", "98.93%")
    with col3:
        st.metric("Test PR-AUC", f"{metrics_norm.get('pr_auc', 0.9943):.4f}", "0.9943")
    with col4:
        st.metric("Inference Latency", f"{res10.get('inference_latency_ms', 0.1451):.4f} ms", "per window")
    with col5:
        st.metric("Active Model Size", f"{res10.get('model_size_kb', 95.15):.2f} KB", "19,313 params")

    # 2. Performance Comparison Table Under Different Operational Regimes
    st.markdown("---")
    st.markdown("#### Performance Under Normal vs. Distribution-Shifted Regimes")
    perf_data = [
        {
            "Operational Condition": "Normal Test Stream (Nominal Model)",
            "Accuracy": f"{metrics_norm.get('accuracy', 0.9827):.4f}",
            "Precision": f"{metrics_norm.get('precision', 0.9244):.4f}",
            "Recall": f"{metrics_norm.get('recall', 0.9893):.4f}",
            "Specificity": f"{metrics_norm.get('specificity', 0.9811):.4f}",
            "F1-Score": f"{metrics_norm.get('f1', 0.9557):.4f}",
            "ROC-AUC": f"{metrics_norm.get('roc_auc', 0.9979):.4f}",
            "PR-AUC": f"{metrics_norm.get('pr_auc', 0.9943):.4f}",
            "FPR": f"{metrics_norm.get('fpr', 0.0189)*100:.2f}%",
            "FNR": f"{metrics_norm.get('fnr', 0.0107)*100:.2f}%"
        },
        {
            "Operational Condition": "Shifted Test Stream (Pre-Adaptation)",
            "Accuracy": f"{metrics_shift_pre.get('accuracy', 0.1893):.4f}",
            "Precision": f"{metrics_shift_pre.get('precision', 0.1893):.4f}",
            "Recall": f"{metrics_shift_pre.get('recall', 1.0000):.4f}",
            "Specificity": f"{metrics_shift_pre.get('specificity', 0.0000):.4f}",
            "F1-Score": f"{metrics_shift_pre.get('f1', 0.3183):.4f}",
            "ROC-AUC": f"{metrics_shift_pre.get('roc_auc', 0.9824):.4f}",
            "PR-AUC": f"{metrics_shift_pre.get('pr_auc', 0.9579):.4f}",
            "FPR": f"{metrics_shift_pre.get('fpr', 1.0000)*100:.2f}%",
            "FNR": f"{metrics_shift_pre.get('fnr', 0.0000)*100:.2f}%"
        },
        {
            "Operational Condition": "Shifted Test Stream (Post-Adaptation)",
            "Accuracy": f"{metrics_shift_post.get('accuracy', 0.9896):.4f}",
            "Precision": f"{metrics_shift_post.get('precision', 0.9633):.4f}",
            "Recall": f"{metrics_shift_post.get('recall', 0.9827):.4f}",
            "Specificity": f"{metrics_shift_post.get('specificity', 0.9912):.4f}",
            "F1-Score": f"{metrics_shift_post.get('f1', 0.9729):.4f}",
            "ROC-AUC": f"{metrics_shift_post.get('roc_auc', 0.9956):.4f}",
            "PR-AUC": f"{metrics_shift_post.get('pr_auc', 0.9891):.4f}",
            "FPR": f"{metrics_shift_post.get('fpr', 0.0088)*100:.2f}%",
            "FNR": f"{metrics_shift_post.get('fnr', 0.0173)*100:.2f}%"
        },
        {
            "Operational Condition": "Normal Stream Retention (Post-Adaptation)",
            "Accuracy": f"{metrics_retention.get('accuracy', 0.9896):.4f}",
            "Precision": f"{metrics_retention.get('precision', 0.9622):.4f}",
            "Recall": f"{metrics_retention.get('recall', 0.9839):.4f}",
            "Specificity": f"{metrics_retention.get('specificity', 0.9910):.4f}",
            "F1-Score": f"{metrics_retention.get('f1', 0.9729):.4f}",
            "ROC-AUC": f"{metrics_retention.get('roc_auc', 0.9972):.4f}",
            "PR-AUC": f"{metrics_retention.get('pr_auc', 0.9926):.4f}",
            "FPR": f"{metrics_retention.get('fpr', 0.0090)*100:.2f}%",
            "FNR": f"{metrics_retention.get('fnr', 0.0161)*100:.2f}%"
        }
    ]
    st.table(pd.DataFrame(perf_data))

    # 3. Cross-Model Architectural Benchmark
    st.markdown("---")
    st.markdown("#### Comprehensive Cross-Model Project Benchmark (Phases 3–10)")
    benchmark_df = pd.DataFrame([
        {"Architecture": "Logistic Regression (Phase 3)", "F1-Score": "0.8848", "Recall": "0.8463", "PR-AUC": "0.9179", "Latency": "0.0003 ms", "Size": "1.31 KB"},
        {"Architecture": "Random Forest (Phase 3)", "F1-Score": "0.9779", "Recall": "0.9885", "PR-AUC": "0.9870", "Latency": "0.0094 ms", "Size": "8,123.7 KB"},
        {"Architecture": "XGBoost (Phase 3)", "F1-Score": "0.9777", "Recall": "0.9935", "PR-AUC": "0.9911", "Latency": "0.0020 ms", "Size": "614.6 KB"},
        {"Architecture": "LSTM Baseline (Phase 4)", "F1-Score": "0.4647", "Recall": "0.3784", "PR-AUC": "0.5806", "Latency": "1.0951 ms", "Size": "86.23 KB"},
        {"Architecture": "Standard Transformer (Phase 5)", "F1-Score": "0.9221", "Recall": "0.9392", "PR-AUC": "0.9769", "Latency": "0.6995 ms", "Size": "87.77 KB"},
        {"Architecture": "SSL Pretrained Transformer (Phase 6)", "F1-Score": "0.9320", "Recall": "0.9730", "PR-AUC": "0.9851", "Latency": "0.4257 ms", "Size": "95.23 KB"},
        {"Architecture": "WaterGuardX Nominal (Phase 10)", "F1-Score": "0.9557", "Recall": "0.9893", "PR-AUC": "0.9943", "Latency": "0.1451 ms", "Size": "95.15 KB"},
        {"Architecture": "WaterGuardX Adapted (Phase 10)", "F1-Score": "0.9729", "Recall": "0.9827", "PR-AUC": "0.9891", "Latency": "0.1451 ms", "Size": "95.15 KB"},
    ])
    st.dataframe(benchmark_df, use_container_width=True, hide_index=True)


def render_interpretability_tab():
    """
    Renders dedicated Tab 7: Phase 11 Diagnostic Error Analysis & Interpretability.
    """
    data11 = load_phase11_results()

    st.markdown("### 🔬 7. Phase 11 Diagnostic Error Analysis & Interpretability")
    st.markdown(
        "Empirical diagnostic findings conducted on the untouched December 1–31, 2018 test streams. "
        "Evaluates physical false alarm mechanisms, incipient leak attenuation, and attention sensitivity."
    )

    tab_err, tab_interp, tab_figs = st.tabs([
        "⚠️ Error Taxonomy & Physical Root Causes",
        "🧭 Permutation Sensitivity & Calibration",
        "🖼️ Publication Diagnostic Figures"
    ])

    with tab_err:
        st.markdown("#### Evidence-Based Error Taxonomy")
        if 'summary' in data11:
            st.table(data11['summary'])

        col_fp, col_fn = st.columns(2, gap="large")
        with col_fp:
            st.markdown("##### 🚨 False Positive Root Cause (0.75% – 1.53% FPR)")
            st.markdown(
                r"""
                - **Empirical Signature:** High-confidence false alarms ($\hat{p} > 0.95$) concentrate during midnight pump staging and valve actuation.
                - **Hydraulic Mechanism:** Rapid step changes ($> 4.0\sigma$ to $8.29\sigma$) on pressure junctions `n215`, `p235`, `n1`, and `n163` induce sharp hydraulic transients that temporally mimic physical pipe burst profiles.
                """
            )
            if 'false_positives' in data11:
                st.caption("Representative False Positive Cases:")
                st.dataframe(data11['false_positives'][['Error ID', 'Timestamp', 'Probability', 'Observed Pattern']].head(5), use_container_width=True)

        with col_fn:
            st.markdown("##### ⚠️ False Negative Root Cause (0.32% – 1.07% FNR)")
            st.markdown(
                r"""
                - **Empirical Signature:** Missed anomalies ($\hat{p} < 0.001$) correspond strictly to low-magnitude incipient pipe leakages.
                - **Hydraulic Mechanism:** Slow developing leaks with gradual localized pressure drops ($< 0.6\sigma$) without sharp step changes remain bounded within normal diurnal demand cycles, escaping temporal attention triggers.
                """
            )
            if 'false_negatives' in data11:
                st.caption("Representative False Negative Cases:")
                st.dataframe(data11['false_negatives'][['Error ID', 'Timestamp', 'Probability', 'Observed Pattern']].head(5), use_container_width=True)

    with tab_interp:
        st.markdown("#### Sensor Permutation Importance & Model Sensitivity")
        st.markdown(
            "Empirical sensitivity measured as mean absolute change in predicted anomaly probability "
            "($|\\Delta \\hat{p}|$) and Average Precision drop when each sensor channel is permuted on the test set."
        )
        if 'interpretability' in data11:
            st.dataframe(data11['interpretability'].head(12), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("#### Probability Calibration (Expected Calibration Error)")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Normal Stream ECE", "0.00473", "Well-Calibrated")
            st.caption("Predictions are highly aligned with empirical positive rates across nominal ranges.")
        with c2:
            st.metric("Adapted Stream ECE", "0.01252", "Mild Intermediate Under-confidence")
            st.caption("Reliable probabilities with minor under-confidence in intermediate 0.40–0.70 score bins.")

    with tab_figs:
        st.markdown("#### High-Resolution Publication Diagnostic Figures")
        err_fig_dir = FIGURES_DIR / 'error_analysis'

        fig_options = {
            "Confusion Matrix (Multi-Condition)": err_fig_dir / 'confusion_matrix_detailed.png',
            "Prediction Probability Distribution": err_fig_dir / 'prediction_probability_distribution.png',
            "Representative False Positive Sequence": err_fig_dir / 'representative_false_positive.png',
            "Representative False Negative Sequence": err_fig_dir / 'representative_false_negative.png',
            "Sensor Permutation Sensitivity Ranking": err_fig_dir / 'sensor_sensitivity_importance.png',
            "Temporal Occlusion Sensitivity Profile": err_fig_dir / 'temporal_occlusion_importance.png',
            "Transformer Attention Map": err_fig_dir / 'attention_map_representative.png',
            "Normal vs. Shifted Error Distribution": err_fig_dir / 'error_distribution_normal_vs_shifted.png',
            "Calibration & Reliability Diagram": err_fig_dir / 'calibration_reliability_diagram.png',
        }

        chosen_fig = st.selectbox("Select Diagnostic Figure to Inspect:", list(fig_options.keys()))
        fig_path = fig_options[chosen_fig]
        if fig_path.exists():
            st.image(str(fig_path), use_container_width=True, caption=f"Figure: {chosen_fig}")
        else:
            st.warning(f"Figure file not found at {fig_path}")


def render_metrics_view():
    """Backwards compatibility wrapper."""
    render_benchmarks_tab()
    st.markdown("---")
    render_interpretability_tab()
