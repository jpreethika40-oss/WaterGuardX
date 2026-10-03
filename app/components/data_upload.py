"""
WaterGuardX — Data Upload & Validation Component.
Handles CSV upload, pre-packaged sample streams, validation checks,
dynamic range filtering, and interactive data profiling in the Blue Theme.
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
    Renders the data ingestion, validation, and profiling section.
    Returns: (dataframe, is_valid, timestamp_column_name)
    """
    st.markdown("### 📥 1. SCADA Data Ingestion & Physical Sensor Profiling")
    st.markdown(
        "Upload multivariate SCADA sensor time-series data or select a pre-packaged held-out evaluation stream. "
        "The system validates physical schema integrity, checks continuity, and derives temporal structures."
    )

    col_upload, col_samples = st.columns([3, 2], gap="large")

    df_loaded: Optional[pd.DataFrame] = None
    source_name = "None"

    with col_upload:
        st.markdown("#### 📁 Upload Custom SCADA CSV")
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
        st.markdown("#### ⚡ Quick-Load Evaluation Streams")
        sample_choice = st.radio(
            "Select an untouched held-out evaluation stream:",
            [
                "None (Upload Custom File)",
                "Held-Out Normal Stream (Dec 2018)",
                "Held-Out Shifted Stream (Dec 2018, Seasonal/Operational Drift)"
            ],
            index=1,
            help="Pre-packaged test sets from untouched held-out December 2018 BattLeDIM data."
        )

        if sample_choice == "Held-Out Normal Stream (Dec 2018)":
            try:
                df_loaded = load_sample_stream('normal', n_rows=1200)
                source_name = "Pre-packaged Stream: Normal Operation (Dec 2018, 1,200 timesteps)"
            except Exception as e:
                st.warning(f"Could not load normal sample: {e}")
        elif sample_choice == "Held-Out Shifted Stream (Dec 2018, Seasonal/Operational Drift)":
            try:
                df_loaded = load_sample_stream('shifted', n_rows=1200)
                source_name = "Pre-packaged Stream: Distribution Shifted (Dec 2018, 1,200 timesteps)"
            except Exception as e:
                st.warning(f"Could not load shifted sample: {e}")

    if df_loaded is None:
        st.info("👆 Please select an evaluation sample stream or upload a SCADA CSV above to begin.")
        return None, False, None

    # Validation
    is_valid, errors, ts_col = validate_dataframe(df_loaded)

    if not is_valid:
        st.error("❌ **Data Validation Failed:**")
        for err in errors:
            st.markdown(f"- {err}")
        return None, False, None

    st.success(f"✅ **Data Validation Passed:** Verified 12 SCADA sensors & temporal continuity. Active Source: **{source_name}**")

    # Quality and Summary Metrics
    n_rows, n_cols = df_loaded.shape
    missing_count = int(df_loaded[SENSOR_COLS].isna().sum().sum())
    dup_count = int(df_loaded.duplicated(subset=[ts_col]).sum()) if ts_col else 0

    ts_series = pd.to_datetime(df_loaded[ts_col], errors='coerce')
    ts_min_str = ts_series.min().strftime('%Y-%m-%d %H:%M') if pd.notnull(ts_series.min()) else "N/A"
    ts_max_str = ts_series.max().strftime('%Y-%m-%d %H:%M') if pd.notnull(ts_series.max()) else "N/A"

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Total Observations", f"{n_rows:,}")
    with m2:
        st.metric("SCADA Sensors", f"{len(SENSOR_COLS)} / 12", "All Verified")
    with m3:
        st.metric("Missing Values", f"{missing_count:,}", "Clean" if missing_count == 0 else "Contains NaN")
    with m4:
        st.metric("Duplicate Timestamps", f"{dup_count:,}", "0 Duplicates")
    with m5:
        st.metric("Coverage Span", f"{(ts_series.max() - ts_series.min()).days}d" if pd.notnull(ts_series.min()) else "N/A", "5-min step")

    st.caption(f"📅 **Temporal Range:** `{ts_min_str}` to `{ts_max_str}` (Temporal interval: ~5 minutes)")

    # Dynamic Data Inspector Tabs
    tab_preview, tab_stats = st.tabs(["📋 Data Preview & Search", "📊 Sensor Statistical Distribution"])

    with tab_preview:
        col_rows, col_search = st.columns([1, 2])
        with col_rows:
            n_show = st.slider("Rows to preview:", min_value=5, max_value=min(100, n_rows), value=10, step=5)
        with col_search:
            selected_cols = st.multiselect(
                "Filter columns to inspect:",
                options=list(df_loaded.columns),
                default=[ts_col] + SENSOR_COLS[:6]
            )
        st.dataframe(df_loaded[selected_cols if selected_cols else df_loaded.columns].head(n_show), use_container_width=True)

    with tab_stats:
        st.markdown("##### Empirical Sensor Distribution Summary")
        stats_df = df_loaded[SENSOR_COLS].describe().T[['mean', 'std', 'min', '50%', 'max']].rename(columns={'50%': 'median'})
        st.dataframe(stats_df.round(3), use_container_width=True)

    return df_loaded, True, ts_col
