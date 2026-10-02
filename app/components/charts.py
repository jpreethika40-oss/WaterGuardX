"""
WaterGuardX — Visualizations Component.
Implements interactive Plotly charts adhering strictly to the curated WaterGuardX palette:
- Background: #F7F5F0
- Card: #FFFFFF
- Text: #252525
- Primary: #2F3A36
- Accent: #5F7D73
- Normal: #6F8F72
- Warning: #C49A45
- Anomaly: #B65C5C
- Border: #DEDAD1
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Palette
COLOR_BG = "#F7F5F0"
COLOR_CARD = "#FFFFFF"
COLOR_TEXT = "#252525"
COLOR_PRIMARY = "#2F3A36"
COLOR_ACCENT = "#5F7D73"
COLOR_NORMAL = "#6F8F72"
COLOR_WARNING = "#C49A45"
COLOR_ANOMALY = "#B65C5C"
COLOR_BORDER = "#DEDAD1"


def render_anomaly_score_chart(pred_df: pd.DataFrame, threshold: float = 0.0100):
    """
    Renders an interactive timeline of continuous anomaly probabilities vs decision threshold.
    """
    st.markdown("### 📈 3. Anomaly Score Trajectory & Sensor Dynamics")

    fig = go.Figure()

    # Base probability curve
    fig.add_trace(go.Scatter(
        x=pred_df["Timestamp"],
        y=pred_df["Anomaly Probability"],
        mode="lines",
        name="Anomaly Probability",
        line=dict(color=COLOR_PRIMARY, width=1.5),
        hovertemplate="<b>Timestamp:</b> %{x}<br><b>Score:</b> %{y:.5f}<extra></extra>"
    ))

    # Anomalous points overlay
    anom_subset = pred_df[pred_df["Predicted Class"] == "Anomaly"]
    if len(anom_subset) > 0:
        fig.add_trace(go.Scatter(
            x=anom_subset["Timestamp"],
            y=anom_subset["Anomaly Probability"],
            mode="markers",
            name="Anomaly Detected",
            marker=dict(color=COLOR_ANOMALY, size=6, symbol="circle"),
            hovertemplate="<b>🚨 Anomaly:</b> %{y:.5f}<br><b>Time:</b> %{x}<extra></extra>"
        ))

    # Decision threshold line
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color=COLOR_ANOMALY,
        line_width=2,
        annotation_text=f"Decision Threshold τ = {threshold:.4f}",
        annotation_position="top left",
        annotation_font_color=COLOR_ANOMALY
    )

    fig.update_layout(
        title=dict(
            text="Temporal Anomaly Probability vs Fixed Decision Threshold (τ = 0.0100)",
            font=dict(color=COLOR_PRIMARY, size=15, family="Inter, sans-serif")
        ),
        xaxis=dict(
            title="Timestamp",
            gridcolor=COLOR_BORDER,
            zerolinecolor=COLOR_BORDER,
            tickfont=dict(color=COLOR_TEXT)
        ),
        yaxis=dict(
            title="Anomaly Score (Sigmoid Output)",
            range=[-0.05, 1.05],
            gridcolor=COLOR_BORDER,
            zerolinecolor=COLOR_BORDER,
            tickfont=dict(color=COLOR_TEXT)
        ),
        plot_bgcolor=COLOR_CARD,
        paper_bgcolor=COLOR_CARD,
        hovermode="x unified",
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color=COLOR_TEXT)
        ),
        height=380
    )

    st.plotly_chart(fig, use_container_width=True)


def render_sensor_time_series(df_raw: pd.DataFrame, pred_df: pd.DataFrame, ts_col: str):
    """
    Renders user-selectable sensor time series with anomalous sequence windows highlighted.
    """
    numeric_cols = [c for c in df_raw.columns if c not in [ts_col, 'label'] and np.issubdtype(df_raw[c].dtype, np.number)]

    if not numeric_cols:
        st.info("No numeric sensor columns found for time-series visualization.")
        return

    col_select, col_opt = st.columns([2, 3])
    with col_select:
        selected_sensor = st.selectbox(
            "Select SCADA Sensor to Inspect:",
            numeric_cols,
            index=0,
            help="Choose a sensor variable to inspect physical hydraulic dynamics alongside model anomaly alerts."
        )

    # Plot sensor trace
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_raw[ts_col],
        y=df_raw[selected_sensor],
        mode="lines",
        name=f"Sensor {selected_sensor}",
        line=dict(color=COLOR_ACCENT, width=1.5),
        hovertemplate=f"<b>{selected_sensor}:</b> %{{y:.2f}}<br><b>Time:</b> %{{x}}<extra></extra>"
    ))

    # Highlight anomalous intervals
    anom_times = pred_df[pred_df["Predicted Class"] == "Anomaly"]["Timestamp"]
    if len(anom_times) > 0:
        anom_sensor_vals = df_raw.set_index(ts_col).reindex(anom_times)[selected_sensor].dropna()
        fig.add_trace(go.Scatter(
            x=anom_sensor_vals.index,
            y=anom_sensor_vals.values,
            mode="markers",
            name="Anomaly Window",
            marker=dict(color=COLOR_ANOMALY, size=5, symbol="circle"),
            hovertemplate=f"<b>Anomalous State on {selected_sensor}:</b> %{{y:.2f}}<extra></extra>"
        ))

    fig.update_layout(
        title=dict(
            text=f"Sensor Physical Dynamics: {selected_sensor} (Overlaying Anomaly Detections)",
            font=dict(color=COLOR_PRIMARY, size=15, family="Inter, sans-serif")
        ),
        xaxis=dict(title="Timestamp", gridcolor=COLOR_BORDER, tickfont=dict(color=COLOR_TEXT)),
        yaxis=dict(title=f"Sensor Value ({selected_sensor})", gridcolor=COLOR_BORDER, tickfont=dict(color=COLOR_TEXT)),
        plot_bgcolor=COLOR_CARD,
        paper_bgcolor=COLOR_CARD,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=360
    )

    st.plotly_chart(fig, use_container_width=True)


def render_distribution_shift_chart(preprocessed_df: pd.DataFrame):
    """
    Renders bar chart showing feature-level Kolmogorov-Smirnov statistics vs drift threshold.
    """
    from utils.app_utils import compute_drift_score
    drift_info = compute_drift_score(preprocessed_df)
    per_feat = drift_info["per_feature"]
    threshold = drift_info["threshold"]

    feats = list(per_feat.keys())
    ks_values = [per_feat[f]["ks_statistic"] for f in feats]
    colors = [COLOR_ANOMALY if ks > threshold else COLOR_NORMAL for ks in ks_values]

    fig = go.Figure(go.Bar(
        x=feats,
        y=ks_values,
        marker_color=colors,
        hovertemplate="<b>Feature:</b> %{x}<br><b>KS Statistic:</b> %{y:.5f}<extra></extra>"
    ))

    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color=COLOR_ANOMALY,
        annotation_text=f"Shift Threshold = {threshold:.5f}",
        annotation_position="top right"
    )

    fig.update_layout(
        title=dict(
            text="Per-Feature Kolmogorov-Smirnov (KS) Shift Statistics vs Reference Baseline",
            font=dict(color=COLOR_PRIMARY, size=14, family="Inter, sans-serif")
        ),
        xaxis=dict(title="Monitored Features", tickangle=-45, tickfont=dict(color=COLOR_TEXT)),
        yaxis=dict(title="KS Statistic (Distance)", gridcolor=COLOR_BORDER, tickfont=dict(color=COLOR_TEXT)),
        plot_bgcolor=COLOR_CARD,
        paper_bgcolor=COLOR_CARD,
        margin=dict(l=40, r=40, t=50, b=50),
        height=320
    )

    st.plotly_chart(fig, use_container_width=True)
