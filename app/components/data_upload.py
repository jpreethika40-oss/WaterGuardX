"""
WaterGuardX — Data Upload & Validation Component.
Handles CSV upload, pre-packaged sample streams, validation checks, and data preview.
"""
from typing import Optional, Tuple
import pandas as pd
import streamlit as st

from utils.app_utils import (
    load_sample_stream,
    validate_dataframe,
    SENSOR_COLS,
    SEQUENCE_LENGTH
)


def render_data_upload() -> Tuple[Optional[pd.DataFrame], bool, Optional[str]]:
    """
    Renders the data ingestion and validation section.
    Returns: (dataframe, is_valid, timestamp_column_name)
    """
    st.markdown("### 📥 1. Ingest Water-System SCADA Data")
    st.markdown(
        "Upload a continuous `.csv` time-series file from water distribution network sensors, "
        "or choose a pre-loaded evaluation stream to test the end-to-end pipeline."
    )

    col_upload, col_samples = st.columns([3, 2], gap="large")

    df_loaded: Optional[pd.DataFrame] = None
    source_name = "None"

    with col_upload:
        uploaded_file = st.file_uploader(
            "Choose a SCADA Sensor CSV file",
            type=["csv"],
            help=f"Must contain 12 SCADA sensors ({', '.join(SENSOR_COLS[:4])}...) and at least {SEQUENCE_LENGTH} consecutive rows."
        )
        if uploaded_file is not None:
            try:
                df_loaded = pd.read_csv(uploaded_file)
                source_name = f"Uploaded File: `{uploaded_file.name}`"
            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")
                return None, False, None

    with col_samples:
        st.markdown("**Or load an untouched held-out evaluation stream:**")
        sample_choice = st.radio(
            "Select evaluation sample:",
            ["None", "Held-Out Normal Stream (Dec 2018)", "Held-Out Shifted Stream (Dec 2018, $k=1.0$)"],
            index=0,
            horizontal=False
        )

        if sample_choice == "Held-Out Normal Stream (Dec 2018)":
            try:
                df_loaded = load_sample_stream('normal', n_rows=1200)
                source_name = "Pre-packaged Stream: Normal Operation (Dec 2018, 1,200 timesteps)"
            except Exception as e:
                st.warning(f"Could not load normal sample: {e}")
        elif sample_choice == "Held-Out Shifted Stream (Dec 2018, $k=1.0$)":
            try:
                df_loaded = load_sample_stream('shifted', n_rows=1200)
                source_name = "Pre-packaged Stream: Distribution Shifted (Dec 2018, 1,200 timesteps)"
            except Exception as e:
                st.warning(f"Could not load shifted sample: {e}")

    if df_loaded is None:
        st.info("👆 Please upload a SCADA CSV file or select a sample stream above to begin analysis.")
        return None, False, None

    st.markdown(f"**Active Data Source:** {source_name}")

    # Validation
    is_valid, errors, ts_col = validate_dataframe(df_loaded)

    if not is_valid:
        st.error("❌ **Data Validation Failed:**")
        for err in errors:
            st.markdown(f"- {err}")
        return None, False, None

    # Quality and Summary Metrics
    st.success("✅ **Data Validation Passed:** Required 12 SCADA sensors and continuous timestamp structure verified.")

    n_rows, n_cols = df_loaded.shape
    missing_count = int(df_loaded[SENSOR_COLS].isna().sum().sum())
    dup_count = int(df_loaded.duplicated(subset=[ts_col]).sum()) if ts_col else 0

    ts_series = pd.to_datetime(df_loaded[ts_col], errors='coerce')
    ts_min_str = ts_series.min().strftime('%Y-%m-%d %H:%M') if pd.notnull(ts_series.min()) else "N/A"
    ts_max_str = ts_series.max().strftime('%Y-%m-%d %H:%M') if pd.notnull(ts_series.max()) else "N/A"

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Total Rows", f"{n_rows:,}")
    with m2:
        st.metric("Sensors Found", f"{len(SENSOR_COLS)} / 12")
    with m3:
        st.metric("Missing Values", f"{missing_count:,}")
    with m4:
        st.metric("Duplicate Timestamps", f"{dup_count:,}")
    with m5:
        st.metric("Time Span", f"{(ts_series.max() - ts_series.min()).days}d" if pd.notnull(ts_series.min()) else "N/A")

    st.caption(f"📅 **Time Range:** `{ts_min_str}` to `{ts_max_str}` (Sampling interval: ~5 minutes)")

    with st.expander("🔍 Inspect Raw Input Data Preview (First 5 Rows)", expanded=False):
        st.dataframe(df_loaded.head(5), use_container_width=True)

    return df_loaded, True, ts_col
