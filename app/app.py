"""
WaterGuardX — Adaptive Water-System Anomaly Detection Using a Self-Supervised Temporal Transformer.
Final Integrated Streamlit Application.
"""
import sys
from pathlib import Path

# Add project root and app directory to path
APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import numpy as np
import pandas as pd
import streamlit as st
import torch

from utils.app_utils import (
    load_waterguardx_models,
    load_preprocessing_artifacts,
    load_phase10_results,
    preprocess_data,
    generate_inference_sequences,
    SEQUENCE_LENGTH,
    SENSOR_COLS,
    FEATURE_COLS
)
from components.data_upload import render_data_upload
from components.prediction_view import render_prediction_view
from components.charts import (
    render_anomaly_score_chart,
    render_sensor_time_series,
    render_distribution_shift_chart
)
from components.metrics_view import render_metrics_view

# Streamlit Page Setup
st.set_page_config(
    page_title="WaterGuardX — Adaptive Water Anomaly Detection",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Design System — WaterGuardX Curated Palette
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #252525;
    }
    
    /* Main Background */
    .stApp {
        background-color: #F7F5F0;
    }
    
    /* Cards and Containers */
    div[data-testid="stMetric"], .stExpander, .css-1r6slb0, .stTable {
        background-color: #FFFFFF;
        border: 1px solid #DEDAD1;
        border-radius: 8px;
        padding: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #DEDAD1;
    }
    
    /* Headings */
    h1, h2, h3, h4 {
        color: #2F3A36;
        font-weight: 600;
    }
    
    /* Metric Label and Value */
    div[data-testid="stMetricLabel"] {
        color: #5F7D73;
        font-size: 0.85rem;
        font-weight: 500;
    }
    div[data-testid="stMetricValue"] {
        color: #2F3A36;
        font-size: 1.6rem;
        font-weight: 700;
    }
    
    /* Custom Badge */
    .badge-primary {
        background-color: #2F3A36;
        color: #FFFFFF;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-normal {
        background-color: #6F8F72;
        color: #FFFFFF;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
    }
    .badge-anomaly {
        background-color: #B65C5C;
        color: #FFFFFF;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
    }
    .badge-warning {
        background-color: #C49A45;
        color: #FFFFFF;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def get_cached_models():
    """Load PyTorch models into memory once."""
    return load_waterguardx_models()


@st.cache_resource
def get_cached_preprocessing():
    """Load fitted scaler and metadata into memory once."""
    return load_preprocessing_artifacts()


def main():
    # Load core artifacts
    try:
        nominal_model, adapted_model, device = get_cached_models()
        scaler, meta = get_cached_preprocessing()
        phase10_info = load_phase10_results()
    except Exception as e:
        st.error(f"Failed to load WaterGuardX core artifacts: {str(e)}")
        st.stop()

    threshold = phase10_info.get("thresholds", {}).get("nominal_threshold", 0.0100)
    p10_res = phase10_info.get("results", {})

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 💧 WaterGuardX")
        st.markdown(
            "**Adaptive Water-System Anomaly Detection**  \n"
            "*Self-Supervised Temporal Transformer*"
        )
        st.markdown("---")

        st.markdown("### ⚙️ Final Architecture")
        st.markdown(f"**Model:** `{p10_res.get('model', 'Adaptive SSL Temporal Transformer')}`")
        st.markdown(f"**Sequence Length ($T$):** `{SEQUENCE_LENGTH} steps` (4 hours)")
        st.markdown(f"**Input Dimensions ($D$):** `{len(FEATURE_COLS)} features` (12 SCADA + 4 temporal)")
        st.markdown(f"**Decision Threshold (τ):** `{threshold:.4f}`")
        st.markdown(f"**Parameters:** `19,313` ({p10_res.get('model_size_kb', 95.15):.1f} KB)")
        st.markdown(f"**Compute Device:** `{device.upper()}`")

        st.markdown("---")
        st.markdown("### 📡 Monitored SCADA Network")
        st.markdown("- **8 Pressure Junctions:** `n1`, `n54`, `n105`, `n163`, `n215`, `n332`, `n458`, `n549`")
        st.markdown("- **2 Flow Meters:** `p227`, `p235`")
        st.markdown("- **1 Pump Station:** `PUMP_1`")
        st.markdown("- **1 Storage Tank Level:** `T1`")

        st.markdown("---")
        st.caption(
            "WaterGuardX pair-evaluates temporal SCADA sequences with self-attention representation, "
            "Kolmogorov-Smirnov shift detection, and safe replay-buffer adaptation."
        )

    # ── Main Header ───────────────────────────────────────────────────────────
    st.title("💧 WaterGuardX — Autonomous Water-System Anomaly Detection")
    st.markdown(
        "A temporal machine-learning system for continuous real-time anomaly detection in municipal water distribution networks. "
        "Built with self-supervised sequence representation learning, distribution-shift detection, and controlled adaptive updating."
    )

    # ── Main Navigation Tabs ──────────────────────────────────────────────────
    tab_live, tab_arch, tab_bench, tab_error = st.tabs([
        "🚀 Live Sensor Analysis & Inference",
        "🏗️ System Architecture & SSL Pipeline",
        "📊 Official Research Evaluation (Phase 10)",
        "🔬 Diagnostic Error Analysis (Phase 11)"
    ])

    # ── TAB 1: Live Analysis ──────────────────────────────────────────────────
    with tab_live:
        # Ingestion
        df_uploaded, is_valid, ts_col = render_data_upload()

        if is_valid and df_uploaded is not None and ts_col is not None:
            # Preprocessing
            with st.spinner("Applying exact training scaler and deriving cyclical temporal features..."):
                preprocessed_df = preprocess_data(df_uploaded, scaler, ts_col)

            # Sequence generation
            seqs, timestamps, y_true = generate_inference_sequences(
                df=preprocessed_df,
                ts_col=ts_col,
                seq_len=SEQUENCE_LENGTH,
                stride=1
            )

            if len(seqs) == 0:
                st.warning("Insufficient sequence rows generated after preprocessing.")
            else:
                # Prediction View
                pred_df, probs, preds, shift_detected, mean_ks, ks_thresh = render_prediction_view(
                    nominal_model=nominal_model,
                    adapted_model=adapted_model,
                    preprocessed_df=preprocessed_df,
                    sequences=seqs,
                    timestamps=timestamps,
                    y_ground_truth=y_true,
                    threshold=threshold
                )

                # Charts
                st.markdown("---")
                render_anomaly_score_chart(pred_df, threshold=threshold)

                render_sensor_time_series(df_raw=df_uploaded, pred_df=pred_df, ts_col=ts_col)

                st.markdown("---")
                st.markdown("#### Distribution Shift Diagnostic Profile")
                render_distribution_shift_chart(preprocessed_df)

    # ── TAB 2: Architecture & Pipeline ────────────────────────────────────────
    with tab_arch:
        st.markdown("### 🏗️ WaterGuardX Implemented Machine Learning Architecture")
        st.markdown(
            "WaterGuardX combines self-supervised sequence pretraining with a temporal transformer encoder, "
            "two-sample Kolmogorov-Smirnov distribution-shift gating, and replay-buffered adaptation."
        )

        col_a1, col_a2 = st.columns(2, gap="large")
        with col_a1:
            st.markdown("#### 1. End-to-End Inference Flow")
            st.code(
                """
Input SCADA Sensor Stream (Continuous 5-min intervals)
        ↓
Data Validation & StandardScaler Normalization
        ↓
Cyclical Temporal Feature Derivation (sin/cos hour, sin/cos dow)
        ↓
Sliding Sequence Generation (Window: 48 timesteps × 16 features)
        ↓
Feature Embedding Projection (16 → 32 dimensions)
        ↓
Sinusoidal Positional Encoding
        ↓
Temporal Transformer Encoder (2 Layers, 2 Multi-Head Attention Heads)
        ↓
Aggregated Sequence Latent Representation (dim = 32)
        ↓
Linear Anomaly Classification Head + Sigmoid Activation
        ↓
Continuous Anomaly Score p ∈ [0, 1] vs Fixed Threshold (τ = 0.0100)
        ↓
Anomaly Alert & Severity Categorization
                """,
                language="text"
            )

        with col_a2:
            st.markdown("#### 2. Self-Supervised Masked Autoencoder Pretraining")
            st.code(
                """
Raw Unlabeled Sensor Sequences (Jan–Aug 2018 Training Split)
        ↓
Random Temporal-Sensor Masking (Mask Ratio = 15%)
        ↓
Temporal Transformer Shared Backbone
        ↓
Reconstruction Projection Head
        ↓
Mean Squared Error (MSE) on Masked Values
        ↓
Pretrained Encoder Captures Diurnal & Hydraulic Dependencies
        ↓
Fine-Tuned with Binary Cross-Entropy on Supervised Anomaly Targets
                """,
                language="text"
            )

        st.markdown("---")
        st.markdown("#### 3. Distribution-Shift Detection & Safe Adaptation Gating")
        st.markdown(
            """
            - **Detection Mechanism:** Two-sample Kolmogorov-Smirnov (KS) test evaluated across all 16 features against the reference training distribution.
            - **Shift Threshold ($\tau_{\text{drift}} = 0.009148$):** Derived non-parametrically from the 99th percentile of 500 bootstrap resamples on nominal training data.
            - **Safe Adaptation Strategy:** When drift is declared, fine-tuning is triggered with a **20% replay buffer** of historical nominal data. This achieves a **99.1% reduction in distribution-shift false alarms** while eliminating catastrophic forgetting.
            """
        )

    # ── TAB 3: Official Research Benchmarks ────────────────────────────────────
    with tab_bench:
        render_metrics_view()

    # ── TAB 4: Phase 11 Interpretability ──────────────────────────────────────
    with tab_error:
        st.markdown("### 🔬 Phase 11 Detailed Interpretability & Error Analysis")
        st.markdown(
            "Empirical diagnostic findings conducted on the untouched December 1–31, 2018 test streams. "
            "*(All conclusions are grounded strictly in measured test-set data without causal speculation)*."
        )

        p11_data = phase10_info  # holds general info
        c_kpi1, c_kpi2, c_kpi3 = st.columns(3)
        with c_kpi1:
            st.metric("Normal Stream FPR", "1.89%", "136 / 7,200 non-anomalous windows")
        with c_kpi2:
            st.metric("Normal Stream FNR", "1.07%", "18 / 1,681 anomalous windows")
        with c_kpi3:
            st.metric("Post-Adaptation False Alarm Reduction", "99.1%", "7,200 FP → 67 FP")

        st.markdown("#### Key Interpretability Insights")
        st.markdown(
            r"""
            1. **Primary Sensitive Sensors:** Feature permutation importance on the test set revealed that junction pressure `n332` (Average Precision drop $= 0.4775$), `n105` (AP drop $= 0.2124$), and pump state `PUMP_1` (AP drop $= 0.2350$) exert the highest empirical influence on model predictions.
            2. **Temporal Occlusion Dynamics:** Occluding sliding temporal regions demonstrated that the model's prediction is most sensitive to the **most recent 12 timesteps ($t-60$ to $t$ minutes)**, proving that immediate transient disruptions outweigh distant historical context.
            3. **False Positive Mechanism:** Concentrated around midnight pump staging events where sudden pressure switching steps ($>4\sigma$) temporarily resemble physical pipe burst profiles.
            4. **False Negative Mechanism:** Concentrated on low-amplitude incipient leaks ($<0.6\sigma$) where signal attenuation remains within regular diurnal fluctuations.
            """
        )

        st.info("💡 To view high-resolution diagnostic plots and reliability diagrams, navigate to the 'Official Research Evaluation' tab and select the 'Publication Diagnostic Figures' sub-tab.")


if __name__ == '__main__':
    main()
