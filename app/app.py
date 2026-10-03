"""
WaterGuardX — Adaptive Water-System Anomaly Detection Using a Self-Supervised Temporal Transformer.
Final Integrated Streamlit Application with Modern Blue Theme, Full Dynamic Controls,
and Modular Multi-Tab Navigation.
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
from components.prediction_view import (
    render_drift_diagnostics_tab,
    render_prediction_view
)
from components.charts import (
    render_anomaly_score_chart,
    render_sensor_time_series
)
from components.simulator_view import render_simulator_view
from components.metrics_view import (
    render_benchmarks_tab,
    render_interpretability_tab
)

# Streamlit Page Setup
st.set_page_config(
    page_title="WaterGuardX — Dynamic Water Anomaly Detection",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Design System — WaterGuardX Curated Modern Blue Theme
CUSTOM_BLUE_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #0F172A;
    }
    
    /* Main Background — Crisp High-Tech Water Tint */
    .stApp {
        background-color: #F0F5FA;
    }
    
    /* Card & Container Styling */
    div[data-testid="stMetric"], .stExpander, div[data-testid="stTable"], .stDataFrame {
        background-color: #FFFFFF;
        border: 1px solid #DBEAFE;
        border-radius: 10px;
        padding: 12px;
        box-shadow: 0 2px 10px rgba(37, 99, 235, 0.04);
    }
    
    /* Metric Card Accent Header */
    div[data-testid="stMetric"] {
        border-top: 3.5px solid #2563EB;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.09);
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #DBEAFE;
    }
    
    /* Headings */
    h1 {
        color: #1E3A8A;
        font-weight: 800;
        letter-spacing: -0.02em;
    }
    h2, h3 {
        color: #1E40AF;
        font-weight: 700;
        letter-spacing: -0.01em;
    }
    h4, h5 {
        color: #2563EB;
        font-weight: 600;
    }
    
    /* Metric Label and Value */
    div[data-testid="stMetricLabel"] {
        color: #1E40AF;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    div[data-testid="stMetricValue"] {
        color: #0F172A;
        font-size: 1.7rem;
        font-weight: 800;
    }
    
    /* Modern Tabs Styling */
    div[data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #E2E8F0;
        padding: 6px;
        border-radius: 12px;
        border: 1px solid #CBD5E1;
    }
    button[data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 600;
        color: #475569;
        background-color: transparent;
        border: none;
        transition: all 0.2s ease;
    }
    button[data-baseweb="tab"]:hover {
        background-color: #F1F5F9;
        color: #1E40AF;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.3);
    }
    
    /* Modern Blue Buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
        color: #FFFFFF;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 8px 20px;
        box-shadow: 0 2px 6px rgba(37, 99, 235, 0.25);
        transition: all 0.15s ease-in-out;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #1D4ED8 0%, #1E3A8A 100%);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.35);
        transform: translateY(-1px);
        color: #FFFFFF;
    }
    div.stDownloadButton > button {
        background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%);
        color: #FFFFFF;
        font-weight: 600;
        border-radius: 8px;
        border: none;
    }
    
    /* Status Badges */
    .badge-blue {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 700;
        display: inline-block;
    }
    .badge-anomaly {
        background-color: #FEE2E2;
        color: #DC2626;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 700;
        display: inline-block;
    }
    .badge-warning {
        background-color: #FEF3C7;
        color: #D97706;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 700;
        display: inline-block;
    }
    
    /* Clean Radio and Checkbox */
    div[data-testid="stRadio"] label, div[data-testid="stCheckbox"] label {
        font-weight: 500;
        color: #1E293B;
    }
</style>
"""
st.markdown(CUSTOM_BLUE_CSS, unsafe_allow_html=True)


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

    default_thresh = phase10_info.get("thresholds", {}).get("nominal_threshold", 0.0100)
    p10_res = phase10_info.get("results", {})

    # Initialize threshold in session state if missing
    if 'dyn_threshold' not in st.session_state:
        st.session_state['dyn_threshold'] = default_thresh

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 💧 WaterGuardX")
        st.markdown(
            "**Adaptive Water-System Anomaly Detection**  \n"
            "*Self-Supervised Temporal Transformer*"
        )
        st.markdown('<span class="badge-blue">System Active & Operational</span>', unsafe_allow_html=True)
        st.markdown("---")

        st.markdown("### ⚙️ Deep Learning Engine")
        st.markdown(f"**Architecture:** `SSL Temporal Transformer`")
        st.markdown(f"**Window Size ($T$):** `{SEQUENCE_LENGTH} steps` (4 hours)")
        st.markdown(f"**Input Channels ($D$):** `{len(FEATURE_COLS)} features` (12 SCADA + 4 cyclical)")
        st.markdown(f"**Optimal Threshold (τ):** `{default_thresh:.4f}`")
        st.markdown(f"**Parameters:** `19,313` ({p10_res.get('model_size_kb', 95.15):.1f} KB)")
        st.markdown(f"**Compute Device:** `{device.upper()}`")

        st.markdown("---")
        st.markdown("### 📡 Municipal SCADA Channels")
        st.markdown("- **8 Pressure Junctions:** `n1`, `n54`, `n105`, `n163`, `n215`, `n332`, `n458`, `n549`")
        st.markdown("- **2 Flow Meters:** `p227`, `p235`")
        st.markdown("- **1 Pump Station:** `PUMP_1`")
        st.markdown("- **1 Storage Tank Level:** `T1`")

        st.markdown("---")
        st.caption(
            "WaterGuardX features multi-head attention sequence representation, "
            "Kolmogorov-Smirnov shift detection, and safe replay-buffered adaptation."
        )

    # ── Main Header ───────────────────────────────────────────────────────────
    col_hdr1, col_hdr2 = st.columns([4, 1])
    with col_hdr1:
        st.title("💧 WaterGuardX — Autonomous Water Anomaly Detection")
        st.markdown(
            "A real-time temporal intelligence platform for continuous pipe burst detection, "
            "incipient leakage monitoring, and operational distribution shift adaptation in municipal water networks."
        )
    with col_hdr2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div style="text-align: right;"><span class="badge-blue">⚡ Production Interface</span></div>',
            unsafe_allow_html=True
        )

    # ── Modular Multi-Tab Navigation (View One-by-One) ────────────────────────
    tab_ingest, tab_drift, tab_infer, tab_viz, tab_sim, tab_bench, tab_interp = st.tabs([
        "📥 1. SCADA Ingestion",
        "🌊 2. Drift Diagnostics",
        "🧠 3. Anomaly Inference",
        "📈 4. Interactive Visualizer",
        "🎮 5. Simulator & What-If",
        "🏆 6. Official Benchmarks",
        "🔬 7. Interpretability"
    ])

    # ── TAB 1: SCADA Data Ingestion & Profiling ────────────────────────────────
    with tab_ingest:
        df_uploaded, is_valid, ts_col = render_data_upload()

        if is_valid and df_uploaded is not None and ts_col is not None:
            # Store in session state for persistence across tab switching
            st.session_state['df_raw'] = df_uploaded
            st.session_state['ts_col'] = ts_col

            with st.spinner("Standardizing sensor features and computing cyclical temporal projections..."):
                preprocessed_df = preprocess_data(df_uploaded, scaler, ts_col)
                st.session_state['df_preprocessed'] = preprocessed_df

            seqs, timestamps, y_true = generate_inference_sequences(
                df=preprocessed_df,
                ts_col=ts_col,
                seq_len=SEQUENCE_LENGTH,
                stride=1
            )
            st.session_state['sequences'] = seqs
            st.session_state['timestamps'] = timestamps
            st.session_state['y_ground_truth'] = y_true

            st.markdown("---")
            st.info("💡 **Data Ready:** Navigate to **Tab 2 (Drift Diagnostics)** or **Tab 3 (Anomaly Inference)** to inspect results.")

    # Check if data is available for downstream dynamic tabs
    has_data = (
        'df_raw' in st.session_state and
        'df_preprocessed' in st.session_state and
        'sequences' in st.session_state and
        len(st.session_state.get('sequences', [])) > 0
    )

    # ── TAB 2: Distribution Shift & KS Drift Diagnostics ──────────────────────
    with tab_drift:
        if not has_data:
            st.warning("⚠️ No active SCADA stream loaded. Please ingest or select a sample dataset in **Tab 1 (SCADA Ingestion)**.")
        else:
            render_drift_diagnostics_tab(st.session_state['df_preprocessed'])

    # ── TAB 3: Model Inference & Dynamic Threshold Scoring ────────────────────
    with tab_infer:
        if not has_data:
            st.warning("⚠️ No active SCADA stream loaded. Please ingest or select a sample dataset in **Tab 1 (SCADA Ingestion)**.")
        else:
            pred_df, probs, preds, active_thresh = render_prediction_view(
                nominal_model=nominal_model,
                adapted_model=adapted_model,
                preprocessed_df=st.session_state['df_preprocessed'],
                sequences=st.session_state['sequences'],
                timestamps=st.session_state['timestamps'],
                y_ground_truth=st.session_state['y_ground_truth'],
                threshold_default=st.session_state['dyn_threshold']
            )
            # Save latest prediction outputs for the visualizer tab
            st.session_state['pred_df'] = pred_df
            st.session_state['probs'] = probs
            st.session_state['preds'] = preds

    # ── TAB 4: Interactive Sensor & Anomaly Visualizer ─────────────────────────
    with tab_viz:
        if not has_data or 'pred_df' not in st.session_state:
            st.warning("⚠️ Predictions not yet generated. Please visit **Tab 1 (Ingestion)** and **Tab 3 (Inference)**.")
        else:
            st.markdown("### 📈 4. Interactive Sensor Dynamics & Anomaly Timeline")
            st.markdown(
                "Analyze full-screen anomaly probability trajectories against the dynamic decision threshold, "
                "and overlay multiple physical sensor readings with custom rolling average smoothing."
            )

            # 1. Main Anomaly Score Chart
            render_anomaly_score_chart(
                st.session_state['pred_df'],
                threshold=st.session_state['dyn_threshold']
            )

            st.markdown("---")
            st.markdown("#### Multi-Sensor Physical Dynamics Inspector")

            v_col1, v_col2, v_col3 = st.columns([3, 2, 2])
            with v_col1:
                selected_sensors = st.multiselect(
                    "Select SCADA Sensors to Overlay:",
                    options=SENSOR_COLS,
                    default=[SENSOR_COLS[0], SENSOR_COLS[5]],  # n1, n332
                    help="Overlay multiple pressure, flow, and pump channels simultaneously."
                )
            with v_col2:
                smooth_window = st.slider(
                    "Smoothing Window (Rolling Mean):",
                    min_value=1,
                    max_value=24,
                    value=1,
                    help="1 = Raw physical sensor measurements, >1 = Rolling average filter."
                )
            with v_col3:
                show_markers = st.checkbox("Overlay Anomaly Markers", value=True)

            render_sensor_time_series(
                df_raw=st.session_state['df_raw'],
                pred_df=st.session_state['pred_df'],
                ts_col=st.session_state['ts_col'],
                selected_sensors=selected_sensors,
                smooth_window=smooth_window,
                show_anomaly_markers=show_markers
            )

    # ── TAB 5: Live Stream Simulator & "What-If" Stress Testing ────────────────
    with tab_sim:
        if not has_data:
            st.warning("⚠️ No active SCADA stream loaded. Please ingest a dataset in **Tab 1 (SCADA Ingestion)**.")
        else:
            render_simulator_view(
                model=nominal_model,
                sequences=st.session_state['sequences'],
                timestamps=st.session_state['timestamps'],
                threshold=st.session_state['dyn_threshold']
            )

    # ── TAB 6: Official Research Benchmarks (Phase 10) ─────────────────────────
    with tab_bench:
        render_benchmarks_tab()

    # ── TAB 7: Diagnostic Error Analysis & Interpretability (Phase 11) ─────────
    with tab_interp:
        render_interpretability_tab()


if __name__ == '__main__':
    main()
