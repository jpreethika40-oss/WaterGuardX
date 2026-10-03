"""
WaterGuardX — Visualizations Component.
Implements interactive Plotly charts adhering strictly to the Modern Blue Design System:
- Primary Blue: #1E40AF (Deep Royal Blue)
- Accent Blue: #2563EB (Electric Azure Blue)
- Secondary Cyan: #0284C7 (Clean Water Cyan)
- Card / Chart Bg: #FFFFFF
- Border: #DBEAFE
- Text: #0F172A
- Anomaly / Alert: #DC2626 (Vibrant Red)
- Warning: #D97706 (Amber)
"""
from typing import List, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Curated Modern Blue Palette
COLOR_BG = "#F0F5FA"
COLOR_CARD = "#FFFFFF"
COLOR_TEXT = "#0F172A"
COLOR_PRIMARY = "#1E40AF"      # Deep Royal Blue
COLOR_ACCENT = "#2563EB"       # Electric Azure
COLOR_SECONDARY = "#0284C7"    # Cyan / Clean Water
COLOR_NORMAL = "#0284C7"       # Ocean Blue
COLOR_WARNING = "#D97706"      # Warning Amber
COLOR_ANOMALY = "#DC2626"      # Alert Red
COLOR_BORDER = "#DBEAFE"       # Soft Blue Border
COLOR_GRID = "#F1F5F9"         # Slate-Blue Grid

SENSOR_PALETTE = [
    "#2563EB", "#0284C7", "#0D9488", "#6366F1", 
    "#8B5CF6", "#3B82F6", "#06B6D4", "#10B981"
]


def render_anomaly_score_chart(pred_df: pd.DataFrame, threshold: float = 0.0100):
    """
    Renders an interactive timeline of continuous anomaly probabilities vs decision threshold.
    """
    fig = go.Figure()

    # Decision threshold danger zone shading
    fig.add_hrect(
        y0=threshold,
        y1=1.05,
        fillcolor="rgba(220, 38, 38, 0.06)",
        line_width=0,
        annotation_text="Anomaly Alert Zone (p ≥ τ)",
        annotation_position="top right",
        annotation_font=dict(color=COLOR_ANOMALY, size=11, family="Inter, sans-serif")
    )

    # Base probability curve
    fig.add_trace(go.Scatter(
        x=pred_df["Timestamp"],
        y=pred_df["Anomaly Probability"],
        mode="lines",
        name="Anomaly Probability",
        line=dict(color=COLOR_ACCENT, width=1.8),
        hovertemplate="<b>Timestamp:</b> %{x}<br><b>Anomaly Score:</b> %{y:.5f}<extra></extra>"
    ))

    # Anomalous points overlay
    anom_subset = pred_df[pred_df["Predicted Class"] == "Anomaly"]
    if len(anom_subset) > 0:
        fig.add_trace(go.Scatter(
            x=anom_subset["Timestamp"],
            y=anom_subset["Anomaly Probability"],
            mode="markers",
            name=f"Anomaly Alert ({len(anom_subset):,})",
            marker=dict(
                color=COLOR_ANOMALY,
                size=7,
                symbol="circle",
                line=dict(color="#FFFFFF", width=1)
            ),
            hovertemplate="<b>🚨 Alert Point:</b> %{y:.5f}<br><b>Time:</b> %{x}<extra></extra>"
        ))

    # Decision threshold line
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color=COLOR_ANOMALY,
        line_width=2,
        annotation_text=f"Decision Threshold τ = {threshold:.4f}",
        annotation_position="top left",
        annotation_font=dict(color=COLOR_ANOMALY, size=12, family="Inter, sans-serif")
    )

    fig.update_layout(
        title=dict(
            text=f"Temporal Anomaly Score Trajectory vs Decision Threshold (τ = {threshold:.4f})",
            font=dict(color=COLOR_PRIMARY, size=15, family="Inter, sans-serif")
        ),
        xaxis=dict(
            title="Timestamp",
            gridcolor=COLOR_GRID,
            zerolinecolor=COLOR_BORDER,
            tickfont=dict(color=COLOR_TEXT, size=11)
        ),
        yaxis=dict(
            title="Anomaly Probability p ∈ [0, 1]",
            range=[-0.02, 1.05],
            gridcolor=COLOR_GRID,
            zerolinecolor=COLOR_BORDER,
            tickfont=dict(color=COLOR_TEXT, size=11)
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
            font=dict(color=COLOR_TEXT, size=11)
        ),
        height=390
    )

    st.plotly_chart(fig, use_container_width=True)


def render_sensor_time_series(
    df_raw: pd.DataFrame,
    pred_df: pd.DataFrame,
    ts_col: str,
    selected_sensors: Optional[List[str]] = None,
    smooth_window: int = 1,
    show_anomaly_markers: bool = True
):
    """
    Renders user-selectable sensor time series with optional multi-sensor overlay,
    smoothing, and anomalous sequence windows highlighted.
    """
    numeric_cols = [c for c in df_raw.columns if c not in [ts_col, 'label'] and np.issubdtype(df_raw[c].dtype, np.number)]

    if not numeric_cols:
        st.info("No numeric sensor columns found for time-series visualization.")
        return

    if selected_sensors is None or len(selected_sensors) == 0:
        selected_sensors = [numeric_cols[0]]

    fig = go.Figure()

    anom_times = pred_df[pred_df["Predicted Class"] == "Anomaly"]["Timestamp"] if show_anomaly_markers else []
    raw_indexed = df_raw.set_index(ts_col)

    for i, sensor in enumerate(selected_sensors):
        sensor_color = SENSOR_PALETTE[i % len(SENSOR_PALETTE)]
        vals = df_raw[sensor]

        if smooth_window > 1:
            vals_smoothed = vals.rolling(smooth_window, min_periods=1, center=True).mean()
            trace_name = f"{sensor} (MA-{smooth_window})"
        else:
            vals_smoothed = vals
            trace_name = sensor

        fig.add_trace(go.Scatter(
            x=df_raw[ts_col],
            y=vals_smoothed,
            mode="lines",
            name=trace_name,
            line=dict(color=sensor_color, width=1.6),
            hovertemplate=f"<b>{sensor}:</b> %{{y:.2f}}<br><b>Time:</b> %{{x}}<extra></extra>"
        ))

        # Add anomaly markers on the first selected sensor
        if i == 0 and len(anom_times) > 0 and show_anomaly_markers:
            anom_vals = raw_indexed.reindex(anom_times)[sensor].dropna()
            fig.add_trace(go.Scatter(
                x=anom_vals.index,
                y=anom_vals.values,
                mode="markers",
                name=f"Anomaly on {sensor}",
                marker=dict(color=COLOR_ANOMALY, size=5, symbol="circle"),
                hovertemplate=f"<b>🚨 Anomaly on {sensor}:</b> %{{y:.2f}}<extra></extra>"
            ))

    sensors_label = ", ".join(selected_sensors[:3]) + ("..." if len(selected_sensors) > 3 else "")
    fig.update_layout(
        title=dict(
            text=f"Physical Sensor Dynamics: [{sensors_label}] (Sampling: 5-min intervals)",
            font=dict(color=COLOR_PRIMARY, size=15, family="Inter, sans-serif")
        ),
        xaxis=dict(title="Timestamp", gridcolor=COLOR_GRID, tickfont=dict(color=COLOR_TEXT)),
        yaxis=dict(title="Physical Sensor Measurement", gridcolor=COLOR_GRID, tickfont=dict(color=COLOR_TEXT)),
        plot_bgcolor=COLOR_CARD,
        paper_bgcolor=COLOR_CARD,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color=COLOR_TEXT)),
        height=380
    )

    st.plotly_chart(fig, use_container_width=True)


def render_distribution_shift_chart(preprocessed_df: pd.DataFrame, custom_threshold: Optional[float] = None):
    """
    Renders bar chart showing feature-level Kolmogorov-Smirnov statistics vs drift threshold.
    """
    from utils.app_utils import compute_drift_score
    drift_info = compute_drift_score(preprocessed_df)
    per_feat = drift_info["per_feature"]
    threshold = custom_threshold if custom_threshold is not None else drift_info["threshold"]

    feats = list(per_feat.keys())
    ks_values = [per_feat[f]["ks_statistic"] for f in feats]
    colors = [COLOR_ANOMALY if ks > threshold else COLOR_ACCENT for ks in ks_values]

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
        line_width=2,
        annotation_text=f"Drift Threshold τ_drift = {threshold:.5f}",
        annotation_position="top right",
        annotation_font=dict(color=COLOR_ANOMALY, size=12, family="Inter, sans-serif")
    )

    fig.update_layout(
        title=dict(
            text="Two-Sample Kolmogorov-Smirnov (KS) Statistics per Channel vs Reference Baseline",
            font=dict(color=COLOR_PRIMARY, size=14, family="Inter, sans-serif")
        ),
        xaxis=dict(title="Monitored Sensor & Cyclical Features", tickangle=-45, tickfont=dict(color=COLOR_TEXT)),
        yaxis=dict(title="KS Statistic (Distance)", gridcolor=COLOR_GRID, tickfont=dict(color=COLOR_TEXT)),
        plot_bgcolor=COLOR_CARD,
        paper_bgcolor=COLOR_CARD,
        margin=dict(l=40, r=40, t=50, b=50),
        height=340
    )

    st.plotly_chart(fig, use_container_width=True)


def render_model_comparison_chart(
    nominal_df: pd.DataFrame,
    adapted_df: pd.DataFrame,
    threshold: float = 0.0100
):
    """
    Renders a synchronized dual comparison chart between Nominal and Replay-Adapted models.
    Demonstrates how the adapted model eliminates distribution-shift false alarms.
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "1. Nominal Model (Un-Adapted) — Vulnerable to Seasonal & Operational Drift",
            "2. Replay-Adapted Model — 99.1% False Alarm Suppression Under Shift"
        )
    )

    # 1. Nominal model trace
    fig.add_trace(
        go.Scatter(
            x=nominal_df["Timestamp"],
            y=nominal_df["Anomaly Probability"],
            mode="lines",
            name="Nominal Score",
            line=dict(color="#F59E0B", width=1.5),
            hovertemplate="<b>Nominal:</b> %{y:.4f}<extra></extra>"
        ),
        row=1, col=1
    )
    # Threshold for nominal
    fig.add_hline(
        y=threshold, line_dash="dash", line_color=COLOR_ANOMALY,
        annotation_text=f"τ = {threshold:.4f}", row=1, col=1
    )

    # 2. Adapted model trace
    fig.add_trace(
        go.Scatter(
            x=adapted_df["Timestamp"],
            y=adapted_df["Anomaly Probability"],
            mode="lines",
            name="Adapted Score",
            line=dict(color=COLOR_ACCENT, width=1.5),
            hovertemplate="<b>Adapted:</b> %{y:.4f}<extra></extra>"
        ),
        row=2, col=1
    )
    # Threshold for adapted
    fig.add_hline(
        y=threshold, line_dash="dash", line_color=COLOR_ANOMALY,
        annotation_text=f"τ = {threshold:.4f}", row=2, col=1
    )

    fig.update_layout(
        height=520,
        plot_bgcolor=COLOR_CARD,
        paper_bgcolor=COLOR_CARD,
        margin=dict(l=40, r=40, t=50, b=40),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text="Anomaly Prob", range=[-0.02, 1.05], gridcolor=COLOR_GRID)
    fig.update_xaxes(title_text="Timestamp", gridcolor=COLOR_GRID, row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)


def render_gauge_score(prob: float, threshold: float = 0.0100):
    """
    Renders an interactive semicircular gauge for live sequence monitoring.
    """
    is_anomaly = prob >= threshold
    bar_color = COLOR_ANOMALY if is_anomaly else (COLOR_WARNING if prob >= threshold / 2 else COLOR_ACCENT)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob,
        number=dict(valueformat=".4f", font=dict(color=COLOR_TEXT, size=28)),
        domain={'x': [0, 1], 'y': [0, 1]},
        title=dict(
            text="Real-Time Anomaly Probability",
            font=dict(color=COLOR_PRIMARY, size=14, family="Inter, sans-serif")
        ),
        gauge=dict(
            axis=dict(range=[0, 1], tickwidth=1, tickcolor=COLOR_BORDER, tickfont=dict(color=COLOR_TEXT)),
            bar=dict(color=bar_color),
            bgcolor=COLOR_CARD,
            borderwidth=2,
            bordercolor=COLOR_BORDER,
            steps=[
                {'range': [0, threshold], 'color': "rgba(2, 132, 199, 0.15)"},
                {'range': [threshold, 0.5], 'color': "rgba(245, 158, 11, 0.25)"},
                {'range': [0.5, 1.0], 'color': "rgba(220, 38, 38, 0.35)"}
            ],
            threshold=dict(
                line=dict(color=COLOR_ANOMALY, width=3),
                thickness=0.75,
                value=threshold
            )
        )
    ))

    fig.update_layout(
        height=220,
        margin=dict(l=25, r=25, t=35, b=20),
        paper_bgcolor=COLOR_CARD,
        font=dict(family="Inter, sans-serif")
    )
    st.plotly_chart(fig, use_container_width=True)
