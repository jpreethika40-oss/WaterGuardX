"""
WaterGuardX — Prediction View Component.
Handles model execution, decision thresholding, prediction breakdown,
distribution shift status, adaptation gating, and results export.
"""
from typing import Optional
import numpy as np
import pandas as pd
import streamlit as st
import torch.nn as nn

from utils.app_utils import (
    run_model_inference,
    compute_drift_score
)


def render_prediction_view(
    nominal_model: nn.Module,
    adapted_model: Optional[nn.Module],
    preprocessed_df: pd.DataFrame,
    sequences: np.ndarray,
    timestamps: list,
    y_ground_truth: Optional[np.ndarray],
    threshold: float = 0.0100
):
    """
    Renders the model prediction and distribution-shift assessment view.
    """
    st.markdown("---")
    st.markdown("### 🧠 2. Anomaly Detection & Distribution Shift Evaluation")

    # 1. Evaluate Distribution Shift using Kolmogorov-Smirnov Test
    with st.spinner("Analyzing temporal distribution shift against training baseline..."):
        drift_res = compute_drift_score(preprocessed_df)

    shift_detected = drift_res["shift_detected"]
    mean_ks = drift_res["mean_ks"]
    ks_threshold = drift_res["threshold"]
    shifted_feats = drift_res["n_shifted_features"]

    # Shift status banner
    col_shift1, col_shift2 = st.columns([3, 2], gap="medium")
    with col_shift1:
        if shift_detected:
            st.error(
                f"⚠️ **Distribution Shift Detected:** Mean KS Statistic = `{mean_ks:.5f}` "
                f"(Threshold: `{ks_threshold:.5f}`). {shifted_feats} / 16 features exhibit statistical divergence."
            )
            st.markdown(
                "Incoming sensor readings significantly deviate from the nominal training baseline. "
                "The **Adapted Model** is recommended to suppress distribution-shift false alarms."
            )
        else:
            st.success(
                f"✅ **Distribution Status Stable:** Mean KS Statistic = `{mean_ks:.5f}` "
                f"(Threshold: `{ks_threshold:.5f}`). 0 features exceed critical threshold."
            )
            st.markdown("Sensor distributions are consistent with the nominal training distribution.")

    with col_shift2:
        model_options = ["Nominal Model (Phase 10 Baseline)"]
        if adapted_model is not None:
            model_options.append("Adapted Model (Phase 10 Shift-Adapted)")

        default_idx = 1 if (shift_detected and adapted_model is not None) else 0
        selected_model_name = st.radio(
            "Select Inference Model Checkpoint:",
            model_options,
            index=default_idx,
            help="Switch between the Nominal and Replay-Adapted models to inspect predictive differences."
        )

    # Pick active model
    active_model = adapted_model if ("Adapted" in selected_model_name and adapted_model is not None) else nominal_model
    is_adapted_active = "Adapted" in selected_model_name

    # Safe Adaptation Policy Callout
    with st.expander("🛡️ Safe Adaptation & Reproducibility Policy", expanded=False):
        st.markdown(
            """
            **Safe Adaptation Design Rules:**
            - **No Interactive Overwrites:** The application strictly never overwrites model checkpoints during user inference.
            - **Controlled Replay Buffer:** Adaptation uses a pre-calibrated 20% training replay buffer to eliminate catastrophic forgetting on normal operations.
            - **No Uncontrolled Online Training:** Retraining is not triggered automatically per-upload to prevent drift corruption or arbitrary data poisoning.
            """
        )

    # 2. Run Inference
    with st.spinner(f"Computing predictions with {selected_model_name} over {len(sequences):,} sequences..."):
        probs, preds = run_model_inference(
            model=active_model,
            sequences=sequences,
            threshold=threshold,
            batch_size=256
        )

    total_seqs = len(preds)
    anomaly_count = int(np.sum(preds))
    normal_count = total_seqs - anomaly_count
    anomaly_pct = (anomaly_count / total_seqs) * 100.0 if total_seqs > 0 else 0.0

    # 3. KPI Summary Cards
    st.markdown("#### Inference Prediction Summary")
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.metric("Total Sequences", f"{total_seqs:,}")
    with kpi2:
        st.metric("Normal Sequences", f"{normal_count:,}", f"{(normal_count/total_seqs)*100:.1f}%")
    with kpi3:
        st.metric("Anomalous Windows", f"{anomaly_count:,}", f"{anomaly_pct:.2f}%", delta_color="inverse")
    with kpi4:
        st.metric("Decision Threshold (τ)", f"{threshold:.4f}", help="Calibrated Phase 10 optimal validation threshold.")
    with kpi5:
        st.metric("Active Model", "Adapted" if is_adapted_active else "Nominal")

    # 4. Construct Prediction DataFrame
    pred_df = pd.DataFrame({
        "Timestamp": timestamps,
        "Anomaly Probability": np.round(probs, 5),
        "Predicted Class": np.where(preds == 1, "Anomaly", "Normal"),
        "Risk Level": np.where(probs >= 0.50, "Critical (≥0.50)",
                      np.where(probs >= threshold, "Warning (≥0.01)", "Nominal (<0.01)"))
    })

    if y_ground_truth is not None and len(y_ground_truth) == len(preds):
        pred_df["Ground Truth"] = np.where(y_ground_truth == 1, "Anomaly", "Normal")
        pred_df["Correct"] = pred_df["Predicted Class"] == pred_df["Ground Truth"]

    # 5. Prediction Table & Filters
    st.markdown("#### Sequential Prediction Log")
    col_filter, col_down = st.columns([3, 1])
    with col_filter:
        show_anomalies_only = st.checkbox("Show anomalous windows only", value=False)
    with col_down:
        csv_data = pred_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download predictions.csv",
            data=csv_data,
            file_name="waterguardx_predictions.csv",
            mime="text/csv",
            help="Download full sequence-level prediction scores and timestamps."
        )

    display_df = pred_df[pred_df["Predicted Class"] == "Anomaly"] if show_anomalies_only else pred_df
    st.dataframe(display_df.head(200), use_container_width=True, height=260)
    st.caption(f"Showing up to 200 of {len(display_df):,} sequences in table preview.")

    return pred_df, probs, preds, shift_detected, mean_ks, ks_threshold
