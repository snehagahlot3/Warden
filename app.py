# -*- coding: utf-8 -*-
"""
WARDEN-T: Temporal Transformer for Multi-Step Network Attack Forecasting
Enterprise Security-Analytics Dashboard
SIH 2026 - Problem Statement 26153
"""

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
if (APP_DIR / "src").is_dir():
    PROJECT_ROOT = APP_DIR
elif (APP_DIR / "warden-t" / "src").is_dir():
    PROJECT_ROOT = APP_DIR / "warden-t"
else:
    PROJECT_ROOT = APP_DIR

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import pandas as pd
import torch
import streamlit as st
import plotly.graph_objects as go

from styles import get_custom_css
from src.explainability.attention import (
    load_artifacts,
    extract_attention,
    perturbation_importance,
)
from src.forecasting.rollout import recursive_rollout
from src.forecasting.csv_inference import process_csv
from src.models.dataset import load_raw_splits, PROCESSED_DIR, MODELS_DIR

# -----------------------------------------------------------------------------
# STREAMLIT CONFIGURATION & ENTERPRISE DARK THEME
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="WARDEN-T | Predictive Attack Forecasting",
    page_icon=":material/shield:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject single enterprise CSS block
st.markdown(get_custom_css(), unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# CACHED DATA & MODEL LOADERS
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading neural model checkpoints...")
def get_cached_artifacts(experiment_key: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return load_artifacts(device=device, experiment=experiment_key)


@st.cache_data(show_spinner="Loading test data...")
def get_cached_test_data(experiment_key: str):
    if experiment_key == "attack_day":
        from src.models.attack_day_split import (
            CANDIDATE_A,
            build_sequence_indices,
            load_states_and_targets,
            make_split_arrays,
            state_blocks,
            verify_segments,
        )

        with open(PROCESSED_DIR / "feature_names.json") as f:
            feat_names = json.load(f)
        verify_segments(verbose=False)
        n_states = len(
            pd.read_csv(PROCESSED_DIR / "network_states.csv", usecols=["is_attack"])
        )
        blocks = state_blocks(n_states)
        seq_idx = build_sequence_indices(blocks, CANDIDATE_A)
        X_all, y_all, cat_all = load_states_and_targets(feat_names)
        X_test, y_test, cat_test, _ = make_split_arrays(
            seq_idx["test"], X_all, y_all, cat_all, feat_names
        )
        return X_test, y_test, cat_test
    else:
        raw = load_raw_splits()
        return raw["X_test"], raw["y_test"], raw["cat_test"]


@st.cache_data(show_spinner=False)
def get_evaluation_metrics():
    with open(MODELS_DIR / "experiment_comparison.json") as f:
        comp = json.load(f)
    with open(MODELS_DIR / "baseline_metrics.json") as f:
        base = json.load(f)
    try:
        with open(MODELS_DIR / "test_metrics.json") as f:
            test_m = json.load(f)
    except Exception:
        test_m = {}
    return {"comparison": comp, "baseline": base, "test": test_m}


@st.cache_data(show_spinner="Computing feature sensitivities...")
def compute_evidence(experiment_key: str, sample_idx: int):
    artifacts = get_cached_artifacts(experiment_key)
    X_test, _, _ = get_cached_test_data(experiment_key)
    seq_raw = X_test[sample_idx]
    seq_norm = artifacts["scaler"].transform(seq_raw)
    pert = perturbation_importance(
        artifacts["model"],
        seq_norm,
        artifacts["feature_names"],
        mask_value=0.0,
    )
    attn = extract_attention(artifacts["model"], seq_norm)
    return pert, attn


def format_category_display(index_to_category, category_index, confidence):
    name = index_to_category.get(
        str(category_index), index_to_category.get(category_index)
    )
    if name is None:
        return "No stage predicted"
    if confidence < 0.50:
        return f"Low confidence: {name} ({confidence * 100:.0f}%)"
    return name


FEATURE_DISPLAY_NAMES = {
    "flow_duration_mean": "Flow duration (mean)",
    "flow_duration_std": "Flow duration (std dev)",
    "flow_duration_max": "Flow duration (max)",
    "flow_duration_min": "Flow duration (min)",
    "total_fwd_packets_mean": "Forward packets (mean)",
    "total_fwd_packets_std": "Forward packets (std dev)",
    "total_fwd_packets_max": "Forward packets (max)",
    "total_fwd_packets_min": "Forward packets (min)",
    "total_bwd_packets_mean": "Backward packets (mean)",
    "total_bwd_packets_std": "Backward packets (std dev)",
    "total_bwd_packets_max": "Backward packets (max)",
    "total_bwd_packets_min": "Backward packets (min)",
    "total_fwd_bytes_mean": "Forward volume (mean bytes)",
    "total_fwd_bytes_std": "Forward volume (std bytes)",
    "total_fwd_bytes_max": "Forward volume (max bytes)",
    "total_fwd_bytes_min": "Forward volume (min bytes)",
    "total_bwd_bytes_mean": "Backward volume (mean bytes)",
    "total_bwd_bytes_std": "Backward volume (std bytes)",
    "total_bwd_bytes_max": "Backward volume (max bytes)",
    "total_bwd_bytes_min": "Backward volume (min bytes)",
    "flow_iat_mean_mean": "Flow inter-arrival time (mean)",
    "flow_iat_mean_std": "Flow inter-arrival time (std)",
    "flow_iat_mean_max": "Flow inter-arrival time (max)",
    "flow_iat_mean_min": "Flow inter-arrival time (min)",
    "flow_iat_std_mean": "Flow IAT variance (mean)",
    "flow_iat_std_std": "Flow IAT variance (std)",
    "flow_iat_std_max": "Flow IAT variance (max)",
    "flow_iat_std_min": "Flow IAT variance (min)",
    "fwd_iat_mean_mean": "Forward IAT (mean)",
    "fwd_iat_mean_std": "Forward IAT (std)",
    "fwd_iat_mean_max": "Forward IAT (max)",
    "fwd_iat_mean_min": "Forward IAT (min)",
    "bwd_iat_mean_mean": "Backward IAT (mean)",
    "bwd_iat_mean_std": "Backward IAT (std)",
    "bwd_iat_mean_max": "Backward IAT (max)",
    "bwd_iat_mean_min": "Backward IAT (min)",
    "down_up_ratio_mean": "Download / upload ratio (mean)",
    "down_up_ratio_std": "Download / upload ratio (std)",
    "down_up_ratio_max": "Download / upload ratio (max)",
    "down_up_ratio_min": "Download / upload ratio (min)",
    "fwd_packets_per_sec_mean": "Forward packet rate (pkts/s)",
    "fwd_packets_per_sec_std": "Forward packet rate std",
    "fwd_packets_per_sec_max": "Forward packet rate (max)",
    "fwd_packets_per_sec_min": "Forward packet rate (min)",
    "bwd_packets_per_sec_mean": "Backward packet rate (pkts/s)",
    "bwd_packets_per_sec_std": "Backward packet rate std",
    "bwd_packets_per_sec_max": "Backward packet rate (max)",
    "bwd_packets_per_sec_min": "Backward packet rate (min)",
    "syn_flag_count_mean": "TCP SYN flags (mean count)",
    "syn_flag_count_std": "TCP SYN flags (std dev)",
    "syn_flag_count_max": "TCP SYN flags (max count)",
    "syn_flag_count_min": "TCP SYN flags (min count)",
    "rst_flag_count_mean": "TCP RST flags (mean count)",
    "rst_flag_count_std": "TCP RST flags (std dev)",
    "rst_flag_count_max": "TCP RST flags (max count)",
    "rst_flag_count_min": "TCP RST flags (min count)",
    "fin_flag_count_mean": "TCP FIN flags (mean count)",
    "fin_flag_count_std": "TCP FIN flags (std dev)",
    "fin_flag_count_max": "TCP FIN flags (max count)",
    "fin_flag_count_min": "TCP FIN flags (min count)",
    "psh_flag_count_mean": "TCP PSH flags (mean count)",
    "psh_flag_count_std": "TCP PSH flags (std dev)",
    "psh_flag_count_max": "TCP PSH flags (max count)",
    "psh_flag_count_min": "TCP PSH flags (min count)",
    "packet_length_mean_mean": "Packet payload length (mean)",
    "packet_length_mean_std": "Packet payload length (std)",
    "packet_length_mean_max": "Packet payload length (max)",
    "packet_length_mean_min": "Packet payload length (min)",
    "packet_length_std_mean": "Packet size deviation (mean)",
    "packet_length_std_std": "Packet size deviation (std)",
    "packet_length_std_max": "Packet size deviation (max)",
    "packet_length_std_min": "Packet size deviation (min)",
    "avg_packet_size_mean": "Average packet size (mean)",
    "avg_packet_size_std": "Average packet size (std)",
    "avg_packet_size_max": "Average packet size (max)",
    "avg_packet_size_min": "Average packet size (min)",
    "protocol_mode": "Transport protocol mode",
    "window_flow_count": "Window total flow count",
    "window_total_bytes": "Window aggregate bytes",
    "window_total_packets": "Window aggregate packets",
    "fwd_init_win_bytes_mean": "TCP initial window size fwd (mean)",
    "bwd_init_win_bytes_mean": "TCP initial window size bwd (mean)",
    "unique_dst_ports": "Unique destination ports",
    "dst_port_entropy": "Destination port entropy",
    "sequential_port_ratio": "Sequential port scan ratio",
    "syn_ratio": "SYN packet ratio",
    "rst_ratio": "RST packet ratio",
    "window_attack_ratio": "Window attack ratio",
}


def humanize_feature(name):
    return FEATURE_DISPLAY_NAMES.get(name, name.replace("_", " ").capitalize())


# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 6px 0 16px 0;">
            <div style="font-size: 18px; font-weight: 600; color: #F1F5F9; letter-spacing: -0.01em;">WARDEN-T</div>
            <div style="font-size: 12px; color: #94A3B8;">Predictive Attack Forecasting</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        options=[
            ":material/dashboard: Overview",
            ":material/upload_file: Analyze Traffic",
            ":material/fact_check: Model Validation",
            ":material/hub: Architecture",
        ],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown(
        """
        <div class="sidebar-footer">
            Offline inference · No network connection
        </div>
        """,
        unsafe_allow_html=True,
    )

page_name = page.split(": ")[-1] if ": " in page else page

# =============================================================================
# PAGE: OVERVIEW
# =============================================================================
if page_name == "Overview":
    # Header
    st.markdown(
        """
        <div class="enterprise-header">
            <div class="header-left">
                <div class="header-title">WARDEN-T</div>
                <div class="header-subtitle">Predictive Network Attack Forecasting</div>
            </div>
            <div>
                <div class="status-chip">
                    <span class="status-chip-dot"></span>
                    Offline · CIC-IDS2017
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Segmented Control for Demo vs Upload
    source_options = ["Demo Dataset", "Upload Network Traffic CSV"]
    if hasattr(st, "segmented_control"):
        source = st.segmented_control(
            "Traffic Source",
            options=source_options,
            default="Demo Dataset",
            label_visibility="collapsed",
        )
    else:
        source = st.radio(
            "Traffic Source",
            options=source_options,
            index=0,
            horizontal=True,
            label_visibility="collapsed",
        )

    if source == "Upload Network Traffic CSV":
        st.markdown("### Upload Traffic")
        uploaded_file = st.file_uploader(
            "Upload flow-level CSV file (minimum 5,000 flows)", type=["csv"]
        )
        if uploaded_file is None:
            st.info(
                "Upload a CIC-IDS2017-compatible CSV to analyze custom network traffic, or switch to Demo Dataset above."
            )
            st.stop()

        try:
            df_check = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            st.stop()

        from src.data.state_construction import validate_csv_columns

        missing = validate_csv_columns(df_check)
        if missing:
            st.error(f"Missing required columns: {', '.join(missing)}")
            st.stop()
        if len(df_check) < 5000:
            st.warning(
                f"Insufficient flows: {len(df_check):,} provided, 5,000 required (10 windows x 500 flows)."
            )
            st.stop()

        with st.spinner("Analyzing uploaded traffic sequence..."):
            result = process_csv(uploaded_file, experiment="original", horizon=5)
        if not result["success"]:
            st.error(f"Inference failed: {result['error']}")
            st.stop()

        steps = result["forecast"]["steps"]
        importance = result["importance"]
        attn = result["attention"]
        max_step = max(steps, key=lambda s: s["attack_probability"])
        artifacts = get_cached_artifacts("original")
        feat_names = result["feature_names"]
        seq_raw = result["sequence_raw"]
        seq_norm = result["sequence_norm"]
        s_t = seq_raw[-1]
        attack_ratio_idx = (
            feat_names.index("window_attack_ratio")
            if "window_attack_ratio" in feat_names
            else -1
        )
        curr_attack_ratio = (
            s_t[attack_ratio_idx] if attack_ratio_idx >= 0 else 0.0
        )
        source_label = "Uploaded CSV"
        n_flows = result["n_flows"]
        n_states = result["n_states"]

    else:
        # Demo Mode (Sample 0 from original test set)
        artifacts = get_cached_artifacts("original")
        X_test, y_test, cat_test = get_cached_test_data("original")
        sample_idx = 0
        seq_raw = X_test[sample_idx]
        seq_norm = artifacts["scaler"].transform(seq_raw)

        rollout_result = recursive_rollout(
            artifacts["model"],
            seq_norm,
            horizon=5,
            feature_mean=artifacts["scaler"].mean,
            feature_std=artifacts["scaler"].std,
        )
        pert_result, attn_result = compute_evidence("original", sample_idx)

        steps = rollout_result["steps"]
        importance = pert_result
        attn = attn_result
        max_step = max(steps, key=lambda s: s["attack_probability"])
        feat_names = artifacts["feature_names"]
        s_t = seq_raw[-1]
        attack_ratio_idx = (
            feat_names.index("window_attack_ratio")
            if "window_attack_ratio" in feat_names
            else -1
        )
        curr_attack_ratio = (
            s_t[attack_ratio_idx] if attack_ratio_idx >= 0 else 0.0
        )
        source_label = "CIC-IDS2017 Demo Sequence"
        n_flows = 5000
        n_states = 10

    # -------------------------------------------------------------------------
    # KPI CARDS (4)
    # -------------------------------------------------------------------------
    peak_risk = max_step["attack_probability"]
    
    # Severity & borders
    if peak_risk >= 0.70:
        threat_badge_class = "badge-critical"
        threat_border_class = "kpi-border-critical"
        threat_label = "Critical"
    elif peak_risk >= 0.50:
        threat_badge_class = "badge-high"
        threat_border_class = "kpi-border-high"
        threat_label = "High"
    elif peak_risk >= 0.35:
        threat_badge_class = "badge-medium"
        threat_border_class = "kpi-border-medium"
        threat_label = "Medium"
    else:
        threat_badge_class = "badge-low"
        threat_border_class = "kpi-border-low"
        threat_label = "Low"

    pred_cat = format_category_display(
        artifacts["index_to_category"],
        steps[0]["category_index"],
        steps[0]["category_confidence"],
    )

    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)

    with col_kpi1:
        st.markdown(
            f"""
            <div class="kpi-card {threat_border_class}">
                <div class="kpi-label">Current Threat</div>
                <div class="kpi-value-row">
                    <span class="badge-threat {threat_badge_class}">{threat_label}</span>
                </div>
                <div class="kpi-subtext">{curr_attack_ratio * 100:.0f}% observed attack ratio</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_kpi2:
        st.markdown(
            f"""
            <div class="kpi-card {threat_border_class}">
                <div class="kpi-label">Peak Forecast Risk</div>
                <div class="kpi-value-row">
                    <span class="kpi-value-num">{peak_risk * 100:.0f}%</span>
                </div>
                <div class="kpi-subtext">Predicted maximum at t+{max_step['horizon']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_kpi3:
        st.markdown(
            """
            <div class="kpi-card kpi-border-neutral">
                <div class="kpi-label">Forecast Horizon</div>
                <div class="kpi-value-row">
                    <span class="kpi-value-num">5</span>
                    <span style="font-size: 14px; color: #94A3B8; font-weight: 500;">windows</span>
                </div>
                <div class="kpi-subtext">Recursive forward rollout</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_kpi4:
        stage_border = "kpi-border-high" if peak_risk >= 0.50 else "kpi-border-neutral"
        st.markdown(
            f"""
            <div class="kpi-card {stage_border}">
                <div class="kpi-label">Predicted Attack Stage</div>
                <div class="kpi-value-row">
                    <span class="kpi-value-text" style="font-size: 16px;">{pred_cat}</span>
                </div>
                <div class="kpi-subtext">Confidence: {steps[0]['category_confidence'] * 100:.0f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # FORECAST CHART (HERO)
    # -------------------------------------------------------------------------
    st.markdown("### Attack Forecast Trajectory")
    st.markdown(
        '<div class="caption-text">Forward simulation across 5 temporal windows. Observed baseline connects to recursive rollout extrapolation.</div>',
        unsafe_allow_html=True,
    )

    chart_windows = ["NOW"] + [f"t+{s['horizon']}" for s in steps]
    chart_probs = [curr_attack_ratio] + [s["attack_probability"] for s in steps]

    # Translucent shaded band around forecast steps
    probs_upper = [
        min(1.0, p + (0.05 * (i + 1) ** 0.4 if i > 0 else 0.0))
        for i, p in enumerate(chart_probs)
    ]
    probs_lower = [
        max(0.0, p - (0.05 * (i + 1) ** 0.4 if i > 0 else 0.0))
        for i, p in enumerate(chart_probs)
    ]

    fig_hero = go.Figure()

    # Red tinted zone above threshold
    fig_hero.add_hrect(
        y0=0.50,
        y1=1.05,
        fillcolor="rgba(220, 38, 38, 0.05)",
        line_width=0,
        layer="below",
    )

    # Shaded confidence spread
    fig_hero.add_trace(
        go.Scatter(
            x=chart_windows,
            y=probs_upper,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig_hero.add_trace(
        go.Scatter(
            x=chart_windows,
            y=probs_lower,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(59, 130, 246, 0.12)",
            name="Rollout Band",
            hoverinfo="skip",
        )
    )

    # Observed segment (NOW to t+1)
    fig_hero.add_trace(
        go.Scatter(
            x=["NOW", "t+1"],
            y=[chart_probs[0], chart_probs[1]],
            mode="lines",
            line=dict(color="#3B82F6", width=2.5),
            name="Observed Transition",
            hoverinfo="skip",
        )
    )

    # Forecast dashed line (t+1 through t+5)
    fig_hero.add_trace(
        go.Scatter(
            x=chart_windows[1:],
            y=chart_probs[1:],
            mode="lines+markers",
            line=dict(color="#3B82F6", width=2.5, dash="dash"),
            marker=dict(
                size=7,
                color=[
                    "#DC2626" if p >= 0.70 else "#EA580C" if p >= 0.50 else "#16A34A"
                    for p in chart_probs[1:]
                ],
                line=dict(color="#F1F5F9", width=1),
            ),
            name="Forecast Rollout",
            hovertemplate="<b>%{x}</b><br>Attack probability: %{y:.1%}<extra></extra>",
        )
    )

    # Observed Point marker at NOW
    fig_hero.add_trace(
        go.Scatter(
            x=["NOW"],
            y=[chart_probs[0]],
            mode="markers",
            marker=dict(size=8, color="#94A3B8", line=dict(color="#F1F5F9", width=1)),
            name="Current Observed",
            hovertemplate="<b>Current (NOW)</b><br>Observed attack ratio: %{y:.1%}<extra></extra>",
        )
    )

    # Alert threshold horizontal line
    fig_hero.add_hline(
        y=0.50,
        line_dash="dash",
        line_color="#EA580C",
        line_width=1.5,
        annotation_text="Alert threshold",
        annotation_position="top right",
        annotation_font=dict(color="#EA580C", size=12),
    )

    fig_hero.add_shape(
        type="line",
        x0=0, x1=0, y0=0, y1=1,
        xref="x", yref="paper",
        line=dict(color="#94A3B8", width=1.5, dash="solid"),
    )
    fig_hero.add_annotation(
        x=0, y=1.02, xref="x", yref="paper",
        text="Now", showarrow=False,
        font=dict(color="#94A3B8", size=12),
    )

    # Annotate ONLY the Current point and the Peak point
    peak_idx = int(np.argmax(chart_probs[1:])) + 1
    fig_hero.add_annotation(
        x="NOW",
        y=chart_probs[0],
        text=f"Current: {chart_probs[0]:.0%}",
        showarrow=True,
        arrowhead=1,
        arrowcolor="#94A3B8",
        ax=0,
        ay=-28,
        font=dict(size=12, color="#F1F5F9", family="monospace"),
        bgcolor="#1E293B",
        bordercolor="#334155",
        borderwidth=1,
    )
    fig_hero.add_annotation(
        x=chart_windows[peak_idx],
        y=chart_probs[peak_idx],
        text=f"Peak: {chart_probs[peak_idx]:.0%}",
        showarrow=True,
        arrowhead=1,
        arrowcolor="#EA580C",
        ax=0,
        ay=-32,
        font=dict(size=12, color="#F1F5F9", family="monospace"),
        bgcolor="#1E293B",
        bordercolor="#334155",
        borderwidth=1,
    )

    fig_hero.update_layout(
        xaxis=dict(
            title=dict(text="Time window", font=dict(size=13, color="#94A3B8")),
            gridcolor="#334155",
            showgrid=False,
            tickfont=dict(size=13, color="#94A3B8"),
        ),
        yaxis=dict(
            title=dict(text="Attack probability", font=dict(size=13, color="#94A3B8")),
            range=[-0.02, 1.05],
            gridcolor="#334155",
            showgrid=True,
            tickfont=dict(size=13, color="#94A3B8"),
            tickformat=".0%",
        ),
        plot_bgcolor="#1E293B",
        paper_bgcolor="#1E293B",
        margin=dict(l=50, r=40, t=30, b=40),
        height=380,
        showlegend=False,
    )

    st.plotly_chart(fig_hero, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # TABBED ANALYST SECTIONS
    # -------------------------------------------------------------------------
    tab_why, tab_flagged, tab_whatif, tab_actions = st.tabs(
        [
            "Why this prediction",
            "Flagged flows",
            "What-if",
            "Recommended actions",
        ]
    )

    # --- TAB 1: WHY THIS PREDICTION ---
    with tab_why:
        st.markdown("### Forecast Drivers")
        st.markdown(
            '<div class="caption-text">Top 10 feature sensitivities determined via perturbation analysis (masked to training mean in z-space).</div>',
            unsafe_allow_html=True,
        )

        top10_records = importance["importances"][:10]
        # Sort ascending for horizontal bar chart display
        top10_sorted = sorted(top10_records, key=lambda r: r["importance_score"])
        feat_labels = [humanize_feature(r["feature"]) for r in top10_sorted]
        feat_scores = [r["importance_score"] for r in top10_sorted]
        bar_colors = ["#EA580C" if s > 0 else "#16A34A" for s in feat_scores]

        fig_drivers = go.Figure(
            go.Bar(
                x=feat_scores,
                y=feat_labels,
                orientation="h",
                marker=dict(color=bar_colors),
                text=[f"{s:+.4f}" for s in feat_scores],
                textposition="outside",
                textfont=dict(color="#F1F5F9", size=11, family="monospace"),
                hovertemplate="<b>%{y}</b><br>Sensitivity Impact: %{x:+.4f}<extra></extra>",
            )
        )
        fig_drivers.update_layout(
            xaxis=dict(
                title=dict(text="Sensitivity: P(original) - P(masked)", font=dict(size=12, color="#94A3B8")),
                gridcolor="#334155",
                showgrid=True,
                tickfont=dict(size=11, color="#94A3B8"),
            ),
            yaxis=dict(
                tickfont=dict(size=12, color="#F1F5F9"),
                showgrid=False,
            ),
            plot_bgcolor="#1E293B",
            paper_bgcolor="#1E293B",
            margin=dict(l=10, r=60, t=10, b=40),
            height=340,
        )
        st.plotly_chart(fig_drivers, use_container_width=True, config={"displayModeBar": False})

        # Temporal Attention Strip
        st.markdown("### Temporal Attention")
        attn_weights = attn["input_importance"]
        w_min = float(np.min(attn_weights))
        w_max = float(np.max(attn_weights))
        spread = max(w_max - w_min, 1e-6)

        strip_html = '<div class="attn-strip-container"><div class="attn-cells">'
        for i in range(10):
            lbl = f"t-{9 - i}" if i < 9 else "t"
            val = attn_weights[i]
            norm_val = (val - w_min) / spread
            alpha = 0.15 + 0.75 * norm_val
            cell_bg = f"rgba(59, 130, 246, {alpha:.2f})"
            strip_html += f'<div class="attn-cell" style="background-color: {cell_bg};"><span class="attn-cell-label">{lbl}</span><span class="attn-cell-val">{val:.3f}</span></div>'
        strip_html += '</div>'
        strip_html += '<div class="caption-text" style="margin-bottom:0;">Attention intensity across the 10-window receptive field (t-9 to t). Darker blue indicates greater attention mass placed by Transformer Layer 0 when generating the multi-step representation. Attention reflects internal model weighting, not causality.</div></div>'
        st.markdown(strip_html, unsafe_allow_html=True)

        st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)

        st.markdown("### Observed Traffic Window")
        st.markdown(
            """
            <div class="callout-box callout-border-info">
                The latest 500-flow traffic window is represented across 87 flow-level features
                (flow duration, packet rates, byte volume distributions, TCP flags).
                WARDEN-T conditions on the preceding 10 temporal windows to capture sequential attack progression.
                Traffic representation: <strong>500-flow aggregated network state</strong>.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- TAB 2: FLAGGED FLOWS ---
    with tab_flagged:
        st.markdown("### Observed Network Windows & Telemetry")
        st.markdown(
            '<div class="caption-text">Recent temporal windows evaluated in the receptive field. Values are extracted directly from network telemetry.</div>',
            unsafe_allow_html=True,
        )

        flow_data = []
        for i in range(10):
            window_label = f"t-{9 - i}" if i < 9 else "t (Current)"
            win_state = seq_raw[i]
            
            # Extract key features safely
            dur = win_state[feat_names.index("flow_duration_mean")] if "flow_duration_mean" in feat_names else 0.0
            bytes_tot = win_state[feat_names.index("window_total_bytes")] if "window_total_bytes" in feat_names else 0.0
            pkts_rate = win_state[feat_names.index("fwd_packets_per_sec_mean")] if "fwd_packets_per_sec_mean" in feat_names else 0.0
            syn_cnt = win_state[feat_names.index("syn_flag_count_mean")] if "syn_flag_count_mean" in feat_names else 0.0
            rst_cnt = win_state[feat_names.index("rst_flag_count_mean")] if "rst_flag_count_mean" in feat_names else 0.0
            atk_ratio = win_state[attack_ratio_idx] if attack_ratio_idx >= 0 else 0.0

            if atk_ratio >= 0.70:
                status_html = '<span class="table-pill" style="background: rgba(220,38,38,0.15); color: #DC2626;">Critical</span>'
            elif atk_ratio >= 0.35:
                status_html = '<span class="table-pill" style="background: rgba(234,88,12,0.15); color: #EA580C;">Elevated</span>'
            else:
                status_html = '<span class="table-pill" style="background: rgba(22,163,74,0.15); color: #16A34A;">Normal</span>'

            flow_data.append(
                f'<tr><td class="mono-text"><strong>{window_label}</strong></td>'
                f'<td class="mono-text">500</td>'
                f'<td class="mono-text">{dur / 1e6:.2f} s</td>'
                f'<td class="mono-text">{bytes_tot:,.0f}</td>'
                f'<td class="mono-text">{pkts_rate:.1f}</td>'
                f'<td class="mono-text">{syn_cnt:.2f}</td>'
                f'<td class="mono-text">{rst_cnt:.2f}</td>'
                f'<td class="mono-text">{atk_ratio * 100:.0f}%</td>'
                f'<td>{status_html}</td></tr>'
            )

        table_html = f"""
        <table class="enterprise-table">
            <thead>
                <tr>
                    <th>Window</th>
                    <th>Flow Count</th>
                    <th>Mean Duration</th>
                    <th>Total Bytes</th>
                    <th>Fwd Pkts/s</th>
                    <th>SYN Flags</th>
                    <th>RST Flags</th>
                    <th>Attack Ratio</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                {''.join(flow_data)}
            </tbody>
        </table>
        """
        st.markdown(table_html, unsafe_allow_html=True)
        st.caption("Host/IP binding: Aggregated 500-flow state window representation without per-state host tags.")

    # --- TAB 3: WHAT-IF ---
    with tab_whatif:
        st.markdown("### What-If Counterfactual Simulation")
        st.markdown(
            '<div class="caption-text">Adjust feature values and simulate the recursive forecast to observe mitigation impact.</div>',
            unsafe_allow_html=True,
        )

        col_wif1, col_wif2 = st.columns([1, 2])
        with col_wif1:
            available_feats = [
                humanize_feature(f)
                for f in feat_names
                if f in FEATURE_DISPLAY_NAMES
            ]
            wif_feature = st.selectbox(
                "Feature to adjust",
                available_feats,
                index=0,
            )
            wif_direction = st.radio(
                "Adjustment direction", ["Decrease", "Increase"], horizontal=True
            )
            wif_pct = st.slider("Magnitude adjustment (%)", 10, 90, 30, 5)
            wif_run = st.button("Run What-If Simulation")

        with col_wif2:
            if wif_run:
                # Map back to raw feature name
                feat_raw_name = next(
                    k for k, v in FEATURE_DISPLAY_NAMES.items() if v == wif_feature
                )
                feat_idx = feat_names.index(feat_raw_name)
                seq_wif = seq_norm.copy()
                direction = -1 if wif_direction == "Decrease" else 1
                seq_wif[:, feat_idx] += direction * (wif_pct / 100.0) * 2.0
                seq_wif = np.clip(seq_wif, -10, 10)

                wif_result = recursive_rollout(
                    artifacts["model"],
                    seq_wif,
                    horizon=5,
                    feature_mean=artifacts["scaler"].mean,
                    feature_std=artifacts["scaler"].std,
                )
                wif_steps = wif_result["steps"]
                wif_max = max(wif_steps, key=lambda s: s["attack_probability"])
                wif_peak = wif_max["attack_probability"]

                diff = peak_risk - wif_peak
                if diff > 0:
                    delta_msg = f"Risk reduced by {diff * 100:.0f} percentage points ({peak_risk * 100:.0f}% → {wif_peak * 100:.0f}%)"
                    border_style = "kpi-border-low"
                else:
                    delta_msg = f"Risk increased by {abs(diff) * 100:.0f} percentage points ({peak_risk * 100:.0f}% → {wif_peak * 100:.0f}%)"
                    border_style = "kpi-border-high"

                st.markdown(
                    f"""
                    <div class="kpi-card {border_style}" style="margin-top: 24px;">
                        <div class="kpi-label">Simulation Outcome</div>
                        <div class="kpi-value-row">
                            <span class="kpi-value-num">{wif_peak * 100:.0f}%</span>
                            <span style="font-size: 14px; color: #94A3B8;">new simulated peak</span>
                        </div>
                        <div class="kpi-subtext">{delta_msg}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.info(
                    "Select a feature, choose direction and magnitude, and run simulation to evaluate counterfactual risk trajectory."
                )

        st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)

        st.markdown("### World-Model Surprise Signal")
        st.markdown(
            '<div class="caption-text">Measures divergence between predicted next-state S_hat(t+1) and observed dynamics. High error indicates novel/anomalous behaviour.</div>',
            unsafe_allow_html=True,
        )

        surprise_scores = []
        for h in range(1, min(4, len(steps) + 1)):
            pred_state = steps[h - 1]["next_state"]
            if h < len(seq_norm):
                actual_state = seq_norm[h]
                surprise = float(np.mean(np.abs(pred_state - actual_state)))
            else:
                surprise = None
            surprise_scores.append(surprise)

        if any(s is not None for s in surprise_scores):
            valid_surprises = [s for s in surprise_scores if s is not None]
            avg_surprise = np.mean(valid_surprises)
            if avg_surprise > 2.0:
                surp_border = "kpi-border-critical"
                surp_label = "High (Novel Behaviour)"
            elif avg_surprise > 1.0:
                surp_border = "kpi-border-medium"
                surp_label = "Elevated"
            else:
                surp_border = "kpi-border-low"
                surp_label = "Normal"

            st.markdown(
                f"""
                <div class="kpi-card {surp_border}" style="max-width: 400px;">
                    <div class="kpi-label">Surprise Score</div>
                    <div class="kpi-value-row">
                        <span class="kpi-value-text">{surp_label}</span>
                    </div>
                    <div class="kpi-subtext">Mean prediction error: {avg_surprise:.3f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("Insufficient subsequent sequence windows to compute surprise signal for this sample.")

        st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)

        # Ensemble Uncertainty
        st.markdown("### Ensemble Uncertainty")
        try:
            with open(MODELS_DIR / "ensemble_results.json") as f:
                ens = json.load(f)
            mean_probs = np.array(ens.get("mean_risk_prob", []))
            std_probs = np.array(ens.get("std_risk_prob", []))
            if len(mean_probs) > 0:
                x_vals = [f"t+{i+1}" for i in range(len(mean_probs))]
                fig_ens = go.Figure()
                fig_ens.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=np.clip(mean_probs + std_probs, 0, 1),
                        mode="lines",
                        line=dict(width=0),
                        showlegend=False,
                    )
                )
                fig_ens.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=np.clip(mean_probs - std_probs, 0, 1),
                        mode="lines",
                        line=dict(width=0),
                        fill="tonexty",
                        fillcolor="rgba(59, 130, 246, 0.15)",
                        name="±1 Std Dev",
                    )
                )
                fig_ens.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=mean_probs,
                        mode="lines+markers",
                        line=dict(color="#3B82F6", width=2),
                        name="Mean Risk",
                    )
                )
                fig_ens.update_layout(
                    xaxis=dict(gridcolor="#334155", tickfont=dict(color="#94A3B8")),
                    yaxis=dict(
                        title=dict(text="Attack Probability", font=dict(size=12, color="#94A3B8")),
                        range=[-0.02, 1.05],
                        gridcolor="#334155",
                        tickfont=dict(color="#94A3B8"),
                        tickformat=".0%",
                    ),
                    plot_bgcolor="#1E293B",
                    paper_bgcolor="#1E293B",
                    margin=dict(l=40, r=40, t=20, b=40),
                    height=260,
                    showlegend=False,
                )
                st.plotly_chart(fig_ens, use_container_width=True, config={"displayModeBar": False})
                st.caption(f"Mean ensemble uncertainty: {ens.get('mean_uncertainty', 0):.4f}")
            else:
                st.info("Run `python -m src.models.evaluate_ensemble` to generate ensemble results.")
        except FileNotFoundError:
            st.info("Ensemble evaluation results not present on disk.")

    # --- TAB 4: RECOMMENDED ACTIONS ---
    with tab_actions:
        st.markdown("### Stage-Aware Incident Playbook")
        st.markdown(
            '<div class="caption-text">Recommended security response actions based on the predicted ATT&CK attack stage.</div>',
            unsafe_allow_html=True,
        )

        playbook = {
            "Reconnaissance": [
                "Rate-limit and block suspicious scanning sources at perimeter firewalls.",
                "Review external-facing services and close unauthorized ports.",
                "Enable enhanced logging on edge proxies and ingress controllers.",
            ],
            "Initial Access": [
                "Verify multi-factor authentication (MFA) enforcement on all public gateways.",
                "Inspect authentication logs for distributed brute-force patterns.",
                "Apply high-priority security patches to exposed web applications.",
            ],
            "Credential Access": [
                "Enforce immediate credential rotation for flagged accounts.",
                "Monitor for abnormal Kerberos ticket or NTLM authentication spikes.",
                "Tighten account lockout thresholds on directory services.",
            ],
            "Command & Control": [
                "Activate automated egress filtering and sinkhole flagged external domains.",
                "Inspect high-frequency beaconing traffic from internal subnets.",
                "Isolate suspicious endpoint sessions to prevent communication channels.",
            ],
            "Impact": [
                "Deploy upstream volumetric DDoS filtering and traffic scrubbers.",
                "Scale stateful firewall resources and rate-limit anomalous flow bursts.",
                "Notify SOC incident response team and initiate failover contingency plans.",
            ],
            "Lateral Movement": [
                "Enforce micro-segmentation policies between network zones.",
                "Restrict internal SMB, RDP, and SSH forwarding across departmental VLANs.",
                "Audit internal service account privileges and access logs.",
            ],
        }

        stage_key = None
        for key in playbook:
            if key.lower() in pred_cat.lower():
                stage_key = key
                break

        if stage_key:
            st.markdown(f"**Predicted Threat Stage:** `{pred_cat}`")
            for i, action in enumerate(playbook[stage_key], 1):
                st.markdown(
                    f"""
                    <div class="enterprise-card" style="padding: 14px 18px; margin-bottom: 12px; display: flex; align-items: center; gap: 12px;">
                        <span style="font-weight: 600; color: #3B82F6; font-size: 16px;">0{i}</span>
                        <span style="color: #F1F5F9; font-size: 14px;">{action}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(f"**Current Status:** `{pred_cat}`")
            st.markdown(
                """
                <div class="callout-box callout-border-info">
                    No active threat stage identified. Continue standard baseline telemetry monitoring and perimeter inspection.
                </div>
                """,
                unsafe_allow_html=True,
            )

# =============================================================================
# PAGE: ANALYZE TRAFFIC
# =============================================================================
elif page_name == "Analyze Traffic":
    st.markdown("### Analyze Network Traffic")
    st.markdown(
        '<div class="caption-text">Upload a CIC-IDS2017-compatible flow-level CSV file for automated state construction and forecast inference.</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader("Upload flow CSV", type=["csv"])
    if uploaded_file is None:
        st.info("Upload a CSV file containing flow records (minimum 5,000 flows across 21 CIC-IDS2017 features).")
        st.stop()

    try:
        df_check = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    from src.data.state_construction import validate_csv_columns, REQUIRED_COLUMNS

    missing = validate_csv_columns(df_check)
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        with st.expander("View required feature columns"):
            for col in REQUIRED_COLUMNS:
                st.code(col)
        st.stop()

    if len(df_check) < 5000:
        st.warning(f"Insufficient flows: {len(df_check):,} provided, 5,000 required (10 windows x 500 flows).")
        st.stop()

    st.success(f"File validated: {len(df_check):,} flows ready for analysis.")

    if st.button("Run Attack Forecast"):
        with st.spinner("Executing WARDEN-T temporal inference..."):
            result = process_csv(uploaded_file, experiment="original", horizon=5)

        if not result["success"]:
            st.error(f"Inference failed: {result['error']}")
            st.stop()

        steps = result["forecast"]["steps"]
        importance = result["importance"]
        max_step = max(steps, key=lambda s: s["attack_probability"])
        artifacts = get_cached_artifacts("original")
        feat_names = result["feature_names"]
        s_t = result["sequence_raw"][-1]
        attack_ratio_idx = (
            feat_names.index("window_attack_ratio")
            if "window_attack_ratio" in feat_names
            else -1
        )
        curr_attack_ratio = s_t[attack_ratio_idx] if attack_ratio_idx >= 0 else 0.0
        peak_risk = max_step["attack_probability"]

        st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
        st.markdown("### Analysis Results")

        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        with col_r1:
            st.markdown(
                f"""
                <div class="kpi-card kpi-border-neutral">
                    <div class="kpi-label">Flows Analyzed</div>
                    <div class="kpi-value-row">
                        <span class="kpi-value-num">{result['n_flows']:,}</span>
                    </div>
                    <div class="kpi-subtext">500 flows / window</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_r2:
            st.markdown(
                f"""
                <div class="kpi-card kpi-border-neutral">
                    <div class="kpi-label">Constructed States</div>
                    <div class="kpi-value-row">
                        <span class="kpi-value-num">{result['n_states']:,}</span>
                    </div>
                    <div class="kpi-subtext">Sequential time windows</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_r3:
            border_c = "kpi-border-critical" if peak_risk >= 0.70 else "kpi-border-high" if peak_risk >= 0.50 else "kpi-border-low"
            st.markdown(
                f"""
                <div class="kpi-card {border_c}">
                    <div class="kpi-label">Peak Forecast Risk</div>
                    <div class="kpi-value-row">
                        <span class="kpi-value-num">{peak_risk * 100:.0f}%</span>
                    </div>
                    <div class="kpi-subtext">Predicted maximum</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_r4:
            pred_cat = format_category_display(
                artifacts["index_to_category"],
                steps[0]["category_index"],
                steps[0]["category_confidence"],
            )
            st.markdown(
                f"""
                <div class="kpi-card kpi-border-neutral">
                    <div class="kpi-label">Predicted Stage</div>
                    <div class="kpi-value-row">
                        <span class="kpi-value-text" style="font-size: 16px;">{pred_cat}</span>
                    </div>
                    <div class="kpi-subtext">Confidence: {steps[0]['category_confidence'] * 100:.0f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
        st.markdown("### Forecast Trajectory")

        chart_windows = ["NOW"] + [f"t+{s['horizon']}" for s in steps]
        chart_probs = [curr_attack_ratio] + [s["attack_probability"] for s in steps]

        fig_ana = go.Figure()
        fig_ana.add_trace(
            go.Scatter(
                x=chart_windows,
                y=chart_probs,
                mode="lines+markers",
                line=dict(color="#3B82F6", width=2.5, dash="dash"),
                marker=dict(size=7, color="#3B82F6"),
                name="Attack Risk",
                hovertemplate="<b>%{x}</b><br>Attack probability: %{y:.1%}<extra></extra>",
            )
        )
        fig_ana.add_hline(
            y=0.50,
            line_dash="dash",
            line_color="#EA580C",
            line_width=1.5,
            annotation_text="Alert threshold",
            annotation_position="top right",
        )
        fig_ana.update_layout(
            xaxis=dict(title="Time window", gridcolor="#334155", tickfont=dict(color="#94A3B8")),
            yaxis=dict(
                title="Attack probability",
                range=[-0.02, 1.05],
                gridcolor="#334155",
                tickfont=dict(color="#94A3B8"),
                tickformat=".0%",
            ),
            plot_bgcolor="#1E293B",
            paper_bgcolor="#1E293B",
            margin=dict(l=50, r=40, t=30, b=40),
            height=320,
            showlegend=False,
        )
        st.plotly_chart(fig_ana, use_container_width=True, config={"displayModeBar": False})

# =============================================================================
# PAGE: MODEL VALIDATION
# =============================================================================
elif page_name == "Model Validation":
    st.markdown("### Model Evaluation — Measured Results")
    st.markdown(
        '<div class="caption-text">Standardized benchmark evaluation metrics recorded from saved test artifacts across evaluation protocols.</div>',
        unsafe_allow_html=True,
    )

    eval_data = get_evaluation_metrics()
    comp = eval_data["comparison"]
    orig_m = comp["original_temporal_split"]
    day_m = comp["candidate_a_attack_day_split"]
    base_m = eval_data["baseline"]

    col_v1, col_v2, col_v3 = st.columns(3)
    with col_v1:
        st.markdown(
            f"""
            <div class="enterprise-card">
                <h4 style="color: #3B82F6 !important; margin-top: 0;">Original Temporal Protocol</h4>
                <div class="text-secondary" style="margin-bottom: 12px;">60/20/20 chronological split (N={orig_m['n_test']})</div>
                <div style="border-top: 1px solid #334155; padding-top: 10px;">
                    <div style="display: flex; justify-content: space-between; margin: 4px 0;">
                        <span class="text-secondary">F1 Score:</span>
                        <span class="mono-text">{orig_m['risk_f1']:.4f}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin: 4px 0;">
                        <span class="text-secondary">ROC-AUC:</span>
                        <span class="mono-text">{orig_m['risk_roc_auc']:.4f}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin: 4px 0;">
                        <span class="text-secondary">PR-AUC:</span>
                        <span class="mono-text">{orig_m['risk_pr_auc']:.4f}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin: 4px 0;">
                        <span class="text-secondary">Next-State MAE:</span>
                        <span class="mono-text">{orig_m['next_state_mae']:.4f}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin: 4px 0;">
                        <span class="text-secondary">Next-State MSE:</span>
                        <span class="mono-text">{orig_m['next_state_mse']:.4f}</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_v2:
        st.markdown(
            f"""
            <div class="enterprise-card">
                <h4 style="color: #F1F5F9 !important; margin-top: 0;">Day-Pure Protocol</h4>
                <div class="text-secondary" style="margin-bottom: 12px;">Train Mon-Wed, Val Thu, Test Fri (N={day_m['n_test']})</div>
                <div style="border-top: 1px solid #334155; padding-top: 10px;">
                    <div style="display: flex; justify-content: space-between; margin: 4px 0;">
                        <span class="text-secondary">F1 Score:</span>
                        <span class="mono-text">{day_m['risk_f1']:.4f}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin: 4px 0;">
                        <span class="text-secondary">ROC-AUC:</span>
                        <span class="mono-text">{day_m['risk_roc_auc']:.4f}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin: 4px 0;">
                        <span class="text-secondary">PR-AUC:</span>
                        <span class="mono-text">{day_m['risk_pr_auc']:.4f}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin: 4px 0;">
                        <span class="text-secondary">Next-State MAE:</span>
                        <span class="mono-text">{day_m['next_state_mae']:.4f}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin: 4px 0;">
                        <span class="text-secondary">Next-State MSE:</span>
                        <span class="mono-text">{day_m['next_state_mse']:.4f}</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_v3:
        st.markdown(
            f"""
            <div class="enterprise-card">
                <h4 style="color: #CA8A04 !important; margin-top: 0;">Logistic Regression Baseline</h4>
                <div class="text-secondary" style="margin-bottom: 12px;">Flattened 810 features, C=1.0 (N={base_m['n_test']})</div>
                <div style="border-top: 1px solid #334155; padding-top: 10px;">
                    <div style="display: flex; justify-content: space-between; margin: 4px 0;">
                        <span class="text-secondary">F1 Score:</span>
                        <span class="mono-text">{base_m['f1']:.4f}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin: 4px 0;">
                        <span class="text-secondary">ROC-AUC:</span>
                        <span class="mono-text">{base_m['roc_auc']:.4f}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin: 4px 0;">
                        <span class="text-secondary">PR-AUC:</span>
                        <span class="mono-text">{base_m['pr_auc']:.4f}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin: 4px 0;">
                        <span class="text-secondary">Next-State MAE:</span>
                        <span class="text-secondary" style="font-style: italic;">N/A (binary only)</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin: 4px 0;">
                        <span class="text-secondary">Next-State MSE:</span>
                        <span class="text-secondary" style="font-style: italic;">N/A (binary only)</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)

    # Onset Detection
    st.markdown("### Onset Detection Performance")
    st.markdown(
        '<div class="caption-text">Measures lead-time warning capability before benign-to-attack transitions occur.</div>',
        unsafe_allow_html=True,
    )

    onset_data = eval_data.get("test", {}).get("onset_detection", {})
    n_onsets = onset_data.get("n_onsets", 0)
    n_detected = onset_data.get("n_detected", 0)
    onset_recall = onset_data.get("recall", 0) or 0
    mean_lead = onset_data.get("mean_lead_time_windows", 0) or 0
    max_lead = onset_data.get("max_lead_time_windows", 0) or 0

    col_o1, col_o2, col_o3 = st.columns(3)
    with col_o1:
        st.markdown(
            f"""
            <div class="kpi-card kpi-border-high">
                <div class="kpi-label">Onset Detection Recall</div>
                <div class="kpi-value-row">
                    <span class="kpi-value-num">{onset_recall * 100:.1f}%</span>
                </div>
                <div class="kpi-subtext">{n_detected} of {n_onsets} attack transitions caught early</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_o2:
        st.markdown(
            f"""
            <div class="kpi-card kpi-border-neutral">
                <div class="kpi-label">Mean Lead Time</div>
                <div class="kpi-value-row">
                    <span class="kpi-value-num">{mean_lead:.1f}</span>
                    <span style="font-size: 14px; color: #94A3B8;">windows</span>
                </div>
                <div class="kpi-subtext">Advance defender warning window</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_o3:
        st.markdown(
            f"""
            <div class="kpi-card kpi-border-neutral">
                <div class="kpi-label">Max Lead Time</div>
                <div class="kpi-value-row">
                    <span class="kpi-value-num">{max_lead}</span>
                    <span style="font-size: 14px; color: #94A3B8;">windows</span>
                </div>
                <div class="kpi-subtext">Peak recorded advance warning</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)

    # Dataset Limitation
    st.markdown(
        """
        <div class="callout-box callout-border-warning">
            <strong style="color: #EA580C;">Dataset Limitation Notice</strong><br>
            CIC-IDS2017 contains substantial distribution differences across capture days.
            Under a strict chronological evaluation, later attack families may be absent from the training period.
            Therefore, performance on unseen attack families can be substantially lower.
            This is an evaluation limitation, not a claim that the model can reliably predict arbitrary unseen attack types.
        </div>
        """,
        unsafe_allow_html=True,
    )

# =============================================================================
# PAGE: ARCHITECTURE
# =============================================================================
elif page_name == "Architecture":
    st.markdown("### WARDEN-T Architecture Pipeline")
    st.markdown(
        '<div class="caption-text">End-to-end dataflow from network packet capture to multi-step recursive rollout and explainability.</div>',
        unsafe_allow_html=True,
    )

    # Inline SVG horizontal flow diagram
    svg_diagram = """
    <div style="background-color: #1E293B; border: 1px solid #334155; border-radius: 10px; padding: 24px; margin-bottom: 24px; overflow-x: auto;">
        <svg width="100%" height="90" viewBox="0 0 980 90" fill="none" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <marker id="arrow" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                    <path d="M 0 1 L 8 5 L 0 9 z" fill="#3B82F6"/>
                </marker>
            </defs>
            
            <!-- 1. Traffic -->
            <rect x="0" y="15" width="110" height="56" rx="8" fill="#0F172A" stroke="#334155" stroke-width="1.5"/>
            <text x="55" y="40" fill="#F1F5F9" font-size="12" font-weight="600" text-anchor="middle" font-family="-apple-system, sans-serif">Flow Traffic</text>
            <text x="55" y="56" fill="#94A3B8" font-size="10" text-anchor="middle" font-family="-apple-system, sans-serif">500 flows / win</text>
            
            <line x1="110" y1="43" x2="135" y2="43" stroke="#3B82F6" stroke-width="1.5" marker-end="url(#arrow)"/>
            
            <!-- 2. Feature Extraction -->
            <rect x="140" y="15" width="125" height="56" rx="8" fill="#0F172A" stroke="#334155" stroke-width="1.5"/>
            <text x="202" y="40" fill="#F1F5F9" font-size="12" font-weight="600" text-anchor="middle" font-family="-apple-system, sans-serif">Feature Extract</text>
            <text x="202" y="56" fill="#94A3B8" font-size="10" text-anchor="middle" font-family="-apple-system, sans-serif">87 metrics / state</text>
            
            <line x1="265" y1="43" x2="290" y2="43" stroke="#3B82F6" stroke-width="1.5" marker-end="url(#arrow)"/>
            
            <!-- 3. Time Windows -->
            <rect x="295" y="15" width="125" height="56" rx="8" fill="#0F172A" stroke="#334155" stroke-width="1.5"/>
            <text x="357" y="40" fill="#F1F5F9" font-size="12" font-weight="600" text-anchor="middle" font-family="-apple-system, sans-serif">Time Windows</text>
            <text x="357" y="56" fill="#94A3B8" font-size="10" text-anchor="middle" font-family="-apple-system, sans-serif">S(t-9) ... S(t)</text>
            
            <line x1="420" y1="43" x2="445" y2="43" stroke="#3B82F6" stroke-width="1.5" marker-end="url(#arrow)"/>
            
            <!-- 4. Transformer Encoder -->
            <rect x="450" y="15" width="140" height="56" rx="8" fill="#0F172A" stroke="#3B82F6" stroke-width="2"/>
            <text x="520" y="40" fill="#F1F5F9" font-size="12" font-weight="600" text-anchor="middle" font-family="-apple-system, sans-serif">Transformer Enc</text>
            <text x="520" y="56" fill="#94A3B8" font-size="10" text-anchor="middle" font-family="-apple-system, sans-serif">2 layers, 4 heads</text>
            
            <line x1="590" y1="43" x2="615" y2="43" stroke="#3B82F6" stroke-width="1.5" marker-end="url(#arrow)"/>
            
            <!-- 5. Heads -->
            <rect x="620" y="15" width="120" height="56" rx="8" fill="#0F172A" stroke="#334155" stroke-width="1.5"/>
            <text x="680" y="40" fill="#F1F5F9" font-size="12" font-weight="600" text-anchor="middle" font-family="-apple-system, sans-serif">Multi-Task Heads</text>
            <text x="680" y="56" fill="#94A3B8" font-size="10" text-anchor="middle" font-family="-apple-system, sans-serif">State, Risk, Stage</text>
            
            <line x1="740" y1="43" x2="765" y2="43" stroke="#3B82F6" stroke-width="1.5" marker-end="url(#arrow)"/>
            
            <!-- 6. Rollout -->
            <rect x="770" y="15" width="105" height="56" rx="8" fill="#0F172A" stroke="#334155" stroke-width="1.5"/>
            <text x="822" y="40" fill="#F1F5F9" font-size="12" font-weight="600" text-anchor="middle" font-family="-apple-system, sans-serif">Rollout Loop</text>
            <text x="822" y="56" fill="#94A3B8" font-size="10" text-anchor="middle" font-family="-apple-system, sans-serif">t+1 → ... → t+5</text>
            
            <line x1="875" y1="43" x2="900" y2="43" stroke="#3B82F6" stroke-width="1.5" marker-end="url(#arrow)"/>
            
            <!-- 7. Explainability -->
            <rect x="905" y="15" width="75" height="56" rx="8" fill="#0F172A" stroke="#334155" stroke-width="1.5"/>
            <text x="942" y="40" fill="#F1F5F9" font-size="11" font-weight="600" text-anchor="middle" font-family="-apple-system, sans-serif">Evidence</text>
            <text x="942" y="56" fill="#94A3B8" font-size="10" text-anchor="middle" font-family="-apple-system, sans-serif">Attention</text>
        </svg>
    </div>
    """
    st.markdown(svg_diagram, unsafe_allow_html=True)

    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)

    # Two-Column Specification Grid
    st.markdown("### Technical Specifications")
    artifacts = get_cached_artifacts("original")
    param_count = artifacts["model"].num_parameters()

    st.markdown(
        f"""
        <div class="spec-grid">
            <div class="spec-item">
                <div class="spec-label">Receptive Field</div>
                <div class="spec-value">10 temporal windows (500 flows each)</div>
            </div>
            <div class="spec-item">
                <div class="spec-label">Encoder Architecture</div>
                <div class="spec-value">2 pre-norm Transformer layers, 4 heads</div>
            </div>
            <div class="spec-item">
                <div class="spec-label">Feature Dimension</div>
                <div class="spec-value">87 metrics per state vector</div>
            </div>
            <div class="spec-item">
                <div class="spec-label">Hidden Dimension (d_model)</div>
                <div class="spec-value">64 channels with sinusoidal positional encoding</div>
            </div>
            <div class="spec-item">
                <div class="spec-label">Feedforward Dimension</div>
                <div class="spec-value">128 channels (GELU non-linearity)</div>
            </div>
            <div class="spec-item">
                <div class="spec-label">Total Parameters</div>
                <div class="spec-value">{param_count:,} trainable weights</div>
            </div>
            <div class="spec-item">
                <div class="spec-label">Multi-Task Heads</div>
                <div class="spec-value">Next-State Regression, Attack Risk Logit, Stage Classification</div>
            </div>
            <div class="spec-item">
                <div class="spec-label">Rollout Mechanism</div>
                <div class="spec-value">Recursive autoregressive simulation (t+1 grounded in data)</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
