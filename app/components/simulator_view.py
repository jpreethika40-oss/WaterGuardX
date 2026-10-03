"""
WaterGuardX — Live Stream Simulator & What-If Stress Testing Component.
Provides real-time sequence-by-sequence stream stepping with dynamic gauge,
and interactive hydraulic fault injection (pipe burst simulation, pump trips).
"""
import numpy as np
import pandas as pd
import streamlit as st
import torch.nn as nn
import plotly.graph_objects as go

from utils.app_utils import (
    run_single_sequence_inference,
    SENSOR_COLS,
    FEATURE_COLS,
    SEQUENCE_LENGTH
)
from components.charts import (
    render_gauge_score,
    COLOR_PRIMARY,
    COLOR_ACCENT,
    COLOR_ANOMALY,
    COLOR_BORDER,
    COLOR_GRID,
    COLOR_CARD,
    COLOR_TEXT
)


def render_simulator_view(
    model: nn.Module,
    sequences: np.ndarray,
    timestamps: list,
    threshold: float = 0.0100
):
    """
    Renders dynamic stream playback simulation and 'What-If' hydraulic fault injection.
    """
    st.markdown("### 🎮 5. Live Stream Simulator & \"What-If\" Hydraulic Stress Testing")
    st.markdown(
        "Interactively step through sequential 5-minute SCADA windows as an RTU/edge device would experience them, "
        "or inject synthetic hydraulic anomalies to evaluate model responsiveness and robustness."
    )

    tab_stream, tab_whatif = st.tabs(["⏱️ Sequential Stream Player", "⚡ 'What-If' Fault Injection Stress Test"])

    # ── TAB A: Sequential Stream Player ───────────────────────────────────────
    with tab_stream:
        st.markdown("#### Real-Time SCADA Window Stepper")
        st.caption("Inspect individual 48-timestep sliding sequence windows (4 hours of continuous telemetry).")

        n_seqs = len(sequences)
        if n_seqs == 0:
            st.warning("No sequences loaded.")
            return

        if 'stream_idx' not in st.session_state:
            st.session_state['stream_idx'] = 0

        # Stream controls
        c_btn1, c_btn2, c_btn3, c_slider = st.columns([1, 1, 1, 3])
        with c_btn1:
            if st.button("⏮️ Start"):
                st.session_state['stream_idx'] = 0
        with c_btn2:
            if st.button("▶️ Step +1"):
                st.session_state['stream_idx'] = min(n_seqs - 1, st.session_state['stream_idx'] + 1)
        with c_btn3:
            if st.button("⏩ Jump +10"):
                st.session_state['stream_idx'] = min(n_seqs - 1, st.session_state['stream_idx'] + 10)
        with c_slider:
            curr_idx = st.slider(
                "Sequence Window Index:",
                0, n_seqs - 1,
                int(st.session_state['stream_idx']),
                key="stream_slider"
            )
            st.session_state['stream_idx'] = curr_idx

        active_idx = st.session_state['stream_idx']
        active_seq = sequences[active_idx]
        active_time = timestamps[active_idx]

        # Compute single-window probability
        score = run_single_sequence_inference(model, active_seq)
        is_anom = score >= threshold

        col_g, col_details = st.columns([2, 3], gap="large")
        with col_g:
            render_gauge_score(score, threshold=threshold)
            if is_anom:
                st.error(f"🚨 **ANOMALY DETECTED** (Score: {score:.5f} ≥ τ {threshold:.4f})")
            elif score >= threshold / 2:
                st.warning(f"⚠️ **ELEVATED TRANSIENT** (Score: {score:.5f})")
            else:
                st.success(f"✅ **NOMINAL HYDRAULICS** (Score: {score:.5f} < τ {threshold:.4f})")

        with col_details:
            st.markdown(f"**Current Window Target Timestamp:** `{active_time}`")
            st.markdown(f"**Sliding Window Span:** 48 timesteps ($t-235\\text{ min}$ to $t$)")
            
            # Local sensor mini-plot for this window
            fig_mini = go.Figure()
            # Plot main pressure junction n332 and flow p227 in standardized units
            fig_mini.add_trace(go.Scatter(
                y=active_seq[:, SENSOR_COLS.index('n332')],
                mode="lines",
                name="Pressure n332 (norm)",
                line=dict(color=COLOR_ACCENT, width=1.8)
            ))
            fig_mini.add_trace(go.Scatter(
                y=active_seq[:, SENSOR_COLS.index('n105')],
                mode="lines",
                name="Pressure n105 (norm)",
                line=dict(color="#0284C7", width=1.5)
            ))
            fig_mini.add_trace(go.Scatter(
                y=active_seq[:, SENSOR_COLS.index('PUMP_1')],
                mode="lines",
                name="Pump PUMP_1 (norm)",
                line=dict(color="#6366F1", width=1.5, dash="dot")
            ))
            fig_mini.update_layout(
                title=dict(text="Standardized Sensor Trajectory for Current Window", font=dict(size=13, color=COLOR_PRIMARY)),
                xaxis=dict(title="Relative Timestep in Window (0 to 47)", gridcolor=COLOR_GRID),
                yaxis=dict(title="Standardized Value", gridcolor=COLOR_GRID),
                plot_bgcolor=COLOR_CARD,
                paper_bgcolor=COLOR_CARD,
                height=210,
                margin=dict(l=30, r=20, t=35, b=30),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_mini, use_container_width=True)

    # ── TAB B: "What-If" Hydraulic Fault Injection ────────────────────────────
    with tab_whatif:
        st.markdown("#### Interactive Hydraulic Fault Injection Simulator")
        st.caption(
            "Test model sensitivity by synthetically perturbing physical channels in the current sequence window. "
            "Simulates physical pipe bursts (sudden head loss), pump trips, or transmission valve closures."
        )

        base_seq = sequences[st.session_state['stream_idx']].copy()
        base_score = run_single_sequence_inference(model, base_seq)

        col_w1, col_w2 = st.columns(2, gap="large")
        with col_w1:
            st.markdown("##### 1. Configure Synthetic Perturbation")
            target_sensor = st.selectbox(
                "Select Target Physical Sensor:",
                SENSOR_COLS,
                index=SENSOR_COLS.index('n332') if 'n332' in SENSOR_COLS else 0,
                help="Sensor n332 is the primary downstream junction identified as highly sensitive in Phase 11."
            )
            fault_type = st.radio(
                "Fault Pattern to Inject:",
                [
                    "Sudden Pressure Drop (Simulate Major Pipe Burst)",
                    "Step Jump / Valve Surge (Water Hammer Transient)",
                    "Pump Shutoff / Trip (Switch PUMP_1 to 0)",
                    "Random Noise Attack (Sensor Degradation)"
                ]
            )

            severity = st.slider("Perturbation Magnitude (Standard Deviations σ):", 0.5, 5.0, 2.5, 0.5)
            steps_perturbed = st.slider("Timesteps Affected (Most Recent Steps):", 1, 24, 6)

        # Apply perturbation to the sequence copy
        perturbed_seq = base_seq.copy()
        feat_idx = SENSOR_COLS.index(target_sensor)

        if "Pressure Drop" in fault_type:
            perturbed_seq[-steps_perturbed:, feat_idx] -= severity
        elif "Step Jump" in fault_type:
            perturbed_seq[-steps_perturbed:, feat_idx] += severity
        elif "Pump Shutoff" in fault_type and 'PUMP_1' in SENSOR_COLS:
            p_idx = SENSOR_COLS.index('PUMP_1')
            perturbed_seq[-steps_perturbed:, p_idx] = -1.0  # off state standardized
        elif "Noise" in fault_type:
            noise = np.random.normal(0, severity, size=steps_perturbed)
            perturbed_seq[-steps_perturbed:, feat_idx] += noise

        # Run inference on perturbed sequence
        perturbed_score = run_single_sequence_inference(model, perturbed_seq)
        score_delta = perturbed_score - base_score

        with col_w2:
            st.markdown("##### 2. Real-Time Response Evaluation")
            res_c1, res_c2, res_c3 = st.columns(3)
            with res_c1:
                st.metric("Baseline Score", f"{base_score:.5f}")
            with res_c2:
                st.metric(
                    "Perturbed Score",
                    f"{perturbed_score:.5f}",
                    delta=f"{score_delta:+.5f}",
                    delta_color="inverse" if score_delta > 0 else "normal"
                )
            with res_c3:
                st.metric(
                    "Status Shift",
                    "🚨 Burst Alert!" if perturbed_score >= threshold else "Nominal",
                    "Triggered" if (perturbed_score >= threshold and base_score < threshold) else "Unchanged"
                )

            # Compare before vs after trace
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Scatter(
                y=base_seq[:, feat_idx],
                mode="lines",
                name=f"Original {target_sensor}",
                line=dict(color="#0284C7", width=1.8)
            ))
            fig_comp.add_trace(go.Scatter(
                y=perturbed_seq[:, feat_idx],
                mode="lines+markers",
                name=f"Perturbed {target_sensor}",
                line=dict(color=COLOR_ANOMALY, width=2.0, dash="dash"),
                marker=dict(size=4)
            ))
            fig_comp.update_layout(
                title=dict(text=f"Channel {target_sensor}: Original vs Injected Fault", font=dict(size=13, color=COLOR_PRIMARY)),
                xaxis=dict(title="Relative Timestep in Window (0 to 47)", gridcolor=COLOR_GRID),
                yaxis=dict(title="Standardized Head / Rate", gridcolor=COLOR_GRID),
                plot_bgcolor=COLOR_CARD,
                paper_bgcolor=COLOR_CARD,
                height=230,
                margin=dict(l=30, r=20, t=35, b=30),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_comp, use_container_width=True)
