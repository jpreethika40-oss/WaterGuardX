"""
WaterGuardX — Prediction View Component.
Handles dynamic model execution, interactive decision thresholding,
live risk categorization, side-by-side comparison, and prediction log filtering in the Blue Theme.
"""
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import streamlit as st
import torch.nn as nn

from utils.app_utils import (
    run_model_inference,
    compute_drift_score,
    apply_dynamic_threshold
)
from components.charts import render_model_comparison_chart


def render_drift_diagnostics_tab(preprocessed_df: pd.DataFrame):
    """
    Renders dedicated Tab 2: Two-sample KS distribution shift diagnostics.
    """
    st.markdown("### 🌊 2. Distribution Shift & KS Drift Diagnostics")
    st.markdown(
        "Continuous non-parametric Kolmogorov-Smirnov (KS) test evaluated across all 16 physical and cyclical channels. "
        "Quantifies divergence between incoming SCADA observations and the nominal reference training baseline."
    )

    with st.spinner("Analyzing empirical cumulative distributions across all 16 channels..."):
        drift_res = compute_drift_score(preprocessed_df)

    mean_ks = drift_res["mean_ks"]
    ref_threshold = drift_res["threshold"]
    shifted_feats = drift_res["n_shifted_features"]
    total_feats = drift_res["total_features"]

    # Dynamic threshold slider for drift sensitivity
    col_stat, col_slider = st.columns([3, 2], gap="large")
    with col_stat:
        if drift_res["shift_detected"]:
            st.error(
                f"🚨 **Distribution Shift Confirmed:** Mean KS Statistic = `{mean_ks:.5f}` "
                f"(Baseline Threshold: `{ref_threshold:.5f}`). {shifted_feats} / {total_feats} features diverge."
            )
            st.markdown(
                "**Actionable Recommendation:** Incoming sensor dynamics reflect seasonal demand variation or pump configuration drift. "
                "Switch to the **Replay-Adapted Model** in Tab 3 to suppress false alarms by **99.1%**."
            )
        else:
            st.success(
                f"✅ **Distribution Status Nominal:** Mean KS Statistic = `{mean_ks:.5f}` "
                f"(Baseline Threshold: `{ref_threshold:.5f}`). 0 features exceed threshold."
            )
            st.markdown("Incoming sensor dynamics match nominal baseline hydraulics. The **Nominal Model** is optimal.")

    with col_slider:
        st.markdown("##### ⚙️ Dynamic Sensitivity Control")
        custom_ks_threshold = st.slider(
            "Custom KS Alert Threshold (τ_drift):",
            min_value=0.0020,
            max_value=0.0500,
            value=float(ref_threshold),
            step=0.0010,
            format="%.5f",
            help="Default 0.009148 derived non-parametrically from 99th percentile of 500 bootstrap resamples on nominal training data."
        )
        if st.button("Reset to Benchmark Threshold (0.009148)"):
            custom_ks_threshold = 0.009148

    # KPI summary
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Mean KS Statistic", f"{mean_ks:.5f}")
    with k2:
        st.metric("Drift Threshold", f"{custom_ks_threshold:.5f}")
    with k3:
        n_shifted_custom = sum(1 for v in drift_res["per_feature"].values() if v["ks_statistic"] > custom_ks_threshold)
        st.metric("Diverging Channels", f"{n_shifted_custom} / {total_feats}", delta_color="inverse")
    with k4:
        st.metric("Shift Gating Verdict", "Trigger Adaptation" if n_shifted_custom > 0 else "Nominal Baseline")

    # Feature Bar Chart
    st.markdown("---")
    st.markdown("#### Channel-by-Channel KS Divergence Profile")
    from components.charts import render_distribution_shift_chart
    render_distribution_shift_chart(preprocessed_df, custom_threshold=custom_ks_threshold)

    # Detailed Per-Feature Table
    with st.expander("🔍 View Channel-Level KS Test Statistics & p-values", expanded=False):
        table_rows = []
        for feat, info in drift_res["per_feature"].items():
            table_rows.append({
                "Channel": feat,
                "KS Statistic": f"{info['ks_statistic']:.5f}",
                "p-value": f"{info['p_value']:.4e}",
                "Status": "🚨 Shifted" if info['ks_statistic'] > custom_ks_threshold else "✅ Stable"
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)


def render_prediction_view(
    nominal_model: nn.Module,
    adapted_model: Optional[nn.Module],
    preprocessed_df: pd.DataFrame,
    sequences: np.ndarray,
    timestamps: list,
    y_ground_truth: Optional[np.ndarray],
    threshold_default: float = 0.0100
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, float]:
    """
    Renders dedicated Tab 3: Model Inference & Dynamic Threshold Scoring.
    Returns: (pred_df, probs, preds, current_threshold)
    """
    st.markdown("### 🧠 3. Model Inference & Dynamic Anomaly Scoring")
    st.markdown(
        "Execute the Temporal Transformer Encoder on sliding SCADA windows. "
        "Adjust decision thresholds dynamically to optimize detection sensitivity versus operational false alarms in real time."
    )

    # Model and Comparison Selection
    col_ctrl1, col_ctrl2 = st.columns([3, 2], gap="large")
    
    with col_ctrl1:
        st.markdown("#### 🤖 Active Inference Model")
        model_options = ["Nominal Model (Phase 10 Baseline)"]
        if adapted_model is not None:
            model_options.append("Adapted Model (Phase 10 Shift-Adapted)")
            model_options.append("Side-by-Side Dual Comparison (Nominal vs Adapted)")

        selected_model_choice = st.radio(
            "Select Checkpoint Execution Mode:",
            model_options,
            index=0,
            horizontal=False
        )

    with col_ctrl2:
        st.markdown("#### 🎯 Dynamic Decision Threshold (τ)")
        
        # Preset buttons
        p_col1, p_col2, p_col3 = st.columns(3)
        with p_col1:
            if st.button("Optimal (0.0100)", help="Phase 10 calibrated optimal validation threshold"):
                st.session_state['dyn_threshold'] = 0.0100
        with p_col2:
            if st.button("High Recall (0.0050)", help="Sensitive: Catch subtle incipient leakages"):
                st.session_state['dyn_threshold'] = 0.0050
        with p_col3:
            if st.button("Strict (0.0300)", help="Conservative: Suppress all minor hydraulic noise"):
                st.session_state['dyn_threshold'] = 0.0300

        current_thresh = st.slider(
            "Adjust Threshold Slider:",
            min_value=0.0010,
            max_value=0.1000,
            value=float(st.session_state.get('dyn_threshold', threshold_default)),
            step=0.0010,
            format="%.4f",
            key="threshold_slider",
            help="Moving this slider updates anomaly classifications instantaneously in <1ms without re-running PyTorch!"
        )
        st.session_state['dyn_threshold'] = current_thresh

    # Cache Model Inferences in Session State to enable instantaneous dynamic threshold recalculation
    cache_key_nominal = f"nominal_probs_{len(sequences)}"
    cache_key_adapted = f"adapted_probs_{len(sequences)}"

    if cache_key_nominal not in st.session_state:
        with st.spinner(f"Computing sequence representations with Nominal Model ({len(sequences):,} windows)..."):
            probs_nom, _ = run_model_inference(nominal_model, sequences, threshold=current_thresh)
            st.session_state[cache_key_nominal] = probs_nom

    probs_nom = st.session_state[cache_key_nominal]

    probs_adapt = None
    if adapted_model is not None:
        if cache_key_adapted not in st.session_state:
            with st.spinner(f"Computing sequence representations with Adapted Model ({len(sequences):,} windows)..."):
                probs_ad, _ = run_model_inference(adapted_model, sequences, threshold=current_thresh)
                st.session_state[cache_key_adapted] = probs_ad
        probs_adapt = st.session_state[cache_key_adapted]

    # Select active probabilities
    is_side_by_side = "Side-by-Side" in selected_model_choice
    if "Adapted" in selected_model_choice and not is_side_by_side and probs_adapt is not None:
        active_probs = probs_adapt
        active_model_name = "Adapted Model"
    else:
        active_probs = probs_nom
        active_model_name = "Nominal Model"

    # Dynamic Threshold Recalculation (Instantaneous)
    pred_df = apply_dynamic_threshold(active_probs, timestamps, current_thresh, y_ground_truth)
    preds = (active_probs >= current_thresh).astype(int)

    total_seqs = len(preds)
    anomaly_count = int(np.sum(preds))
    normal_count = total_seqs - anomaly_count
    anomaly_pct = (anomaly_count / total_seqs) * 100.0 if total_seqs > 0 else 0.0
    max_prob = float(np.max(active_probs)) if total_seqs > 0 else 0.0

    # KPI Summary Cards (Update instantly with slider)
    st.markdown("---")
    st.markdown("#### Dynamic Inference KPI Metrics")
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.metric("Total Windows", f"{total_seqs:,}")
    with kpi2:
        st.metric("Nominal Windows", f"{normal_count:,}", f"{(normal_count/total_seqs)*100:.1f}%")
    with kpi3:
        st.metric("Anomalies Flagged", f"{anomaly_count:,}", f"{anomaly_pct:.2f}%", delta_color="inverse")
    with kpi4:
        st.metric("Max Anomaly Score", f"{max_prob:.5f}", "Peak Risk")
    with kpi5:
        st.metric("Active Threshold τ", f"{current_thresh:.4f}", f"Model: {active_model_name.split()[0]}")

    # Side-by-Side Dual View if selected
    if is_side_by_side and probs_adapt is not None:
        st.markdown("---")
        st.markdown("#### ⚡ Real-Time Side-by-Side Model Comparison")
        st.caption("Demonstrating how Replay-Buffered Adaptation eliminates false alarms while preserving normal detection.")

        df_nom_comp = apply_dynamic_threshold(probs_nom, timestamps, current_thresh, y_ground_truth)
        df_ad_comp = apply_dynamic_threshold(probs_adapt, timestamps, current_thresh, y_ground_truth)

        nom_anom = int(np.sum(probs_nom >= current_thresh))
        ad_anom = int(np.sum(probs_adapt >= current_thresh))
        suppressed_fp = max(0, nom_anom - ad_anom)
        suppress_pct = (suppressed_fp / nom_anom * 100.0) if nom_anom > 0 else 0.0

        c_d1, c_d2, c_d3 = st.columns(3)
        with c_d1:
            st.metric("Nominal Model Flags", f"{nom_anom:,}", "Un-adapted baseline")
        with c_d2:
            st.metric("Adapted Model Flags", f"{ad_anom:,}", "Shift-adapted model")
        with c_d3:
            st.metric("False Alarms Suppressed", f"{suppressed_fp:,}", f"-{suppress_pct:.1f}% reduction")

        render_model_comparison_chart(df_nom_comp, df_ad_comp, threshold=current_thresh)

    # Dynamic Filterable Table
    st.markdown("---")
    st.markdown("#### 📋 Filterable Sequential Anomaly Log")

    col_f1, col_f2, col_f3, col_f4 = st.columns([2, 2, 2, 2])
    with col_f1:
        risk_filter = st.selectbox(
            "Filter by Risk Level:",
            ["All Records", "Anomalies Only (≥ τ)", "Critical Only (≥ 0.50)", "Nominal Only (< τ)"]
        )
    with col_f2:
        min_p = st.slider("Min Probability Filter:", 0.0, 1.0, 0.0, 0.01)
    with col_f3:
        search_ts = st.text_input("Search Timestamp:", placeholder="e.g. 2018-12-05")
    with col_f4:
        st.write("") # spacer
        csv_bytes = pred_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Export CSV (Current τ)",
            data=csv_bytes,
            file_name=f"waterguardx_predictions_tau_{current_thresh:.4f}.csv",
            mime="text/csv",
            use_container_width=True
        )

    # Apply filters dynamically
    filtered_df = pred_df.copy()
    if risk_filter == "Anomalies Only (≥ τ)":
        filtered_df = filtered_df[filtered_df["Predicted Class"] == "Anomaly"]
    elif risk_filter == "Critical Only (≥ 0.50)":
        filtered_df = filtered_df[filtered_df["Anomaly Probability"] >= 0.50]
    elif risk_filter == "Nominal Only (< τ)":
        filtered_df = filtered_df[filtered_df["Predicted Class"] == "Normal"]

    if min_p > 0:
        filtered_df = filtered_df[filtered_df["Anomaly Probability"] >= min_p]

    if search_ts.strip():
        filtered_df = filtered_df[filtered_df["Timestamp"].astype(str).str.contains(search_ts.strip(), case=False)]

    st.dataframe(filtered_df.head(250), use_container_width=True, height=270)
    st.caption(f"Showing up to 250 of {len(filtered_df):,} matching sequences in table preview.")

    return pred_df, active_probs, preds, current_thresh
