"""
SafeSpan AI — Main Streamlit Application (Enhanced v2)
AI-Powered Infrastructure Health Monitoring System
Enhanced with: Alert System, Multi-Structure Comparison,
               Improved Risk Graphs, Advanced Animations
"""

import streamlit as st
import numpy as np
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import time
import os
import random
from datetime import datetime, timedelta

# Page configuration — must be first Streamlit call
st.set_page_config(
    page_title="SafeSpan AI — Infrastructure Health Monitor",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "style.css")
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Import project modules
from model import CrackDetector
from utils import (
    generate_heatmap,
    compute_health_score,
    get_risk_level,
    generate_risk_forecast,
    generate_inspection_summary
)
from impact_agent import FailureImpactAgent, format_currency, format_population
from aging_agent import AgingClockAgent
from complaint_agent import GovernmentComplaintAgent
import math


# ─── Initialize Model (cached) ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return CrackDetector()


detector = load_model()


# ─── Demo data for multi-structure comparison ─────────────────────────────────
@st.cache_data
def get_demo_structures():
    """Generate demo infrastructure data for multi-structure comparison."""
    np.random.seed(42)
    structures = [
        {"id": "BRG-2024-A1", "name": "Highway Bridge #47", "type": "Bridge",
         "score": 82.4, "last_scan": "2 hours ago", "scans": 34},
        {"id": "BLD-2024-B3", "name": "Metro Tower East", "type": "Building",
         "score": 56.1, "last_scan": "5 hours ago", "scans": 18},
        {"id": "BRG-2024-C7", "name": "River Crossing #12", "type": "Bridge",
         "score": 28.3, "last_scan": "1 day ago", "scans": 52},
        {"id": "TNL-2024-D2", "name": "Underground Tunnel A", "type": "Tunnel",
         "score": 91.7, "last_scan": "30 min ago", "scans": 67},
        {"id": "DAM-2024-E5", "name": "Reservoir Dam #3", "type": "Dam",
         "score": 44.9, "last_scan": "12 hours ago", "scans": 41},
        {"id": "BRG-2024-F8", "name": "Overpass Section 9", "type": "Bridge",
         "score": 71.2, "last_scan": "3 hours ago", "scans": 29},
    ]
    return structures


def get_score_color(score):
    if score >= 70:
        return "#00e676"
    elif score >= 40:
        return "#ffab00"
    else:
        return "#ff1744"


def get_risk_label(score):
    if score >= 70:
        return "Safe"
    elif score >= 40:
        return "Moderate"
    else:
        return "Critical"


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem 0;">
        <div style="font-family: 'Orbitron', monospace; font-size: 1.4rem; font-weight: 800;
                    background: linear-gradient(135deg, #00d4ff, #7c4dff, #ff3366);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                    background-clip: text; letter-spacing: 2px;">
            🏗️ SAFESPAN AI
        </div>
        <div style="color: #7a8ba0; font-size: 0.7rem; margin-top: 0.3rem; letter-spacing: 2px;
                    text-transform: uppercase;">
            Infrastructure Intelligence
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div style="padding: 0.5rem 0;">
        <div class="section-header" style="font-size: 0.9rem;">📋 SYSTEM STATUS</div>
    </div>
    """, unsafe_allow_html=True)

    status_cols = st.columns(2)
    with status_cols[0]:
        st.markdown("""
        <div class="metric-card" style="padding: 0.8rem;">
            <div class="metric-label" style="font-size: 0.7rem;">AI ENGINE</div>
            <div style="color: #00e676; font-size: 0.85rem; margin-top: 0.3rem;">
                <span class="status-dot" style="background: #00e676; color: #00e676;"></span>ONLINE
            </div>
        </div>
        """, unsafe_allow_html=True)
    with status_cols[1]:
        st.markdown("""
        <div class="metric-card" style="padding: 0.8rem;">
            <div class="metric-label" style="font-size: 0.7rem;">MODEL VER.</div>
            <div style="color: #00d4ff; font-size: 0.85rem; margin-top: 0.3rem;">v2.4.1</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div style="padding: 0.5rem 0;">
        <div class="section-header" style="font-size: 0.9rem;">⚙️ ANALYSIS SETTINGS</div>
    </div>
    """, unsafe_allow_html=True)

    sensitivity = st.slider("Detection Sensitivity", 0.5, 2.0, 1.0, 0.1,
                           help="Adjust heatmap sensitivity for crack detection")
    forecast_months = st.selectbox("Forecast Period", [3, 6, 9, 12], index=1,
                                   help="Number of months for risk prediction")

    st.markdown("---")

    # Live Alerts in Sidebar
    st.markdown("""
    <div style="padding: 0.5rem 0;">
        <div class="section-header" style="font-size: 0.9rem;">🔔 LIVE ALERTS</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="alert-card alert-critical" style="padding: 0.8rem 1rem; margin: 0.4rem 0;">
        <div class="alert-title" style="color: #ff1744; font-size: 0.72rem;">🚨 CRITICAL</div>
        <div class="alert-message" style="font-size: 0.75rem;">River Crossing #12 — Score dropped below 30%</div>
        <div class="alert-timestamp">14:32 UTC</div>
    </div>

    <div class="alert-card alert-warning" style="padding: 0.8rem 1rem; margin: 0.4rem 0;">
        <div class="alert-title" style="color: #ffab00; font-size: 0.72rem;">⚠️ WARNING</div>
        <div class="alert-message" style="font-size: 0.75rem;">Reservoir Dam #3 — Accelerated deterioration</div>
        <div class="alert-timestamp">13:18 UTC</div>
    </div>

    <div class="alert-card alert-info" style="padding: 0.8rem 1rem; margin: 0.4rem 0;">
        <div class="alert-title" style="color: #00d4ff; font-size: 0.72rem;">ℹ️ INFO</div>
        <div class="alert-message" style="font-size: 0.75rem;">Underground Tunnel A — Scan complete</div>
        <div class="alert-timestamp">12:45 UTC</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="padding: 0.8rem; background: rgba(0,212,255,0.04); border-radius: 12px;
                border: 1px solid rgba(0,212,255,0.08);">
        <div style="color: #00d4ff; font-size: 0.75rem; font-weight: 600; margin-bottom: 0.3rem;">
            ℹ️ ABOUT
        </div>
        <div style="color: #7a8ba0; font-size: 0.72rem; line-height: 1.6;">
            SafeSpan AI leverages deep learning and computer vision to detect structural
            damage in infrastructure. Upload bridge or building images for instant analysis.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─── Hero Header ──────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 1.5rem 0 0.5rem 0;">
    <div class="hero-title">SAFESPAN AI</div>
    <div class="hero-subtitle">AI-Powered Infrastructure Health Monitoring System</div>
</div>
""", unsafe_allow_html=True)

# Quick status bar
status_col1, status_col2, status_col3, status_col4 = st.columns(4)

with status_col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">AI Model</div>
        <div class="metric-value" style="color: #00e676; font-size: 1.5rem;">Active</div>
        <div class="realtime-indicator" style="margin-top: 0.5rem;">
            <span class="realtime-dot"></span> LIVE
        </div>
    </div>
    """, unsafe_allow_html=True)

with status_col2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Accuracy</div>
        <div class="metric-value" style="color: #00d4ff; font-size: 1.5rem;">97.3%</div>
        <div style="color: #00e676; font-size: 0.7rem; margin-top: 0.3rem;">▲ +0.2% this week</div>
    </div>
    """, unsafe_allow_html=True)

with status_col3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Scans Today</div>
        <div class="metric-value" style="color: #7c4dff; font-size: 1.5rem;">1,247</div>
        <div style="color: #7a8ba0; font-size: 0.7rem; margin-top: 0.3rem;">↑ 12% vs yesterday</div>
    </div>
    """, unsafe_allow_html=True)

with status_col4:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Active Alerts</div>
        <div class="metric-value" style="color: #ff3366; font-size: 1.5rem;">3</div>
        <div style="color: #ff3366; font-size: 0.7rem; margin-top: 0.3rem;">1 critical</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ─── Tabbed Navigation ───────────────────────────────────────────────────────
tab_scan, tab_compare, tab_alerts, tab_impact, tab_aging, tab_govt = st.tabs([
    "🔍  SCAN & ANALYZE",
    "📊  MULTI-STRUCTURE COMPARISON",
    "🔔  ALERT CENTER",
    "💥  FAILURE IMPACT SIM",
    "⏳  AGING CLOCK",
    "🏢  GOV COMPLAINT"
])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1: SCAN & ANALYZE
# ════════════════════════════════════════════════════════════════════════════════
with tab_scan:
    st.markdown("""
    <div class="section-header">
        📷 INFRASTRUCTURE IMAGE ANALYSIS
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card" style="padding: 1rem 1.5rem; margin-bottom: 1rem;">
        <p style="color: #7a8ba0; font-size: 0.9rem; margin: 0;">
            Upload a high-resolution image of a bridge, building surface, or concrete structure
            for AI-powered crack detection and structural health assessment.
        </p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload infrastructure image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="Supported formats: JPG, JPEG, PNG, BMP, WEBP",
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        with st.spinner(""):
            st.markdown("""
            <div style="text-align: center; padding: 1.5rem;">
                <div class="loading-text" style="font-size: 1.1rem;">
                    🔍 ANALYZING STRUCTURAL INTEGRITY...
                </div>
                <div style="color: #4a5568; font-size: 0.75rem; margin-top: 0.5rem;
                            font-family: 'JetBrains Mono', monospace;">
                    Running CNN inference • Edge detection • Risk modeling
                </div>
            </div>
            """, unsafe_allow_html=True)

            crack_prob = detector.predict(image_np)
            health_score = compute_health_score(crack_prob)
            risk_info = get_risk_level(health_score)
            heatmap_img = generate_heatmap(image_np, intensity=sensitivity)
            forecast = generate_risk_forecast(health_score, months=forecast_months)
            summary = generate_inspection_summary(health_score, crack_prob, risk_info, forecast)

            time.sleep(1)

        # ── Detection Result ──────────────────────────────────────────────
        st.markdown("---")
        st.markdown("""
        <div class="section-header">
            🔬 DETECTION RESULTS
        </div>
        """, unsafe_allow_html=True)

        is_crack = crack_prob >= 0.4
        badge_class = "badge-crack" if is_crack else "badge-safe"
        badge_text = "⚠️ CRACK DETECTED" if is_crack else "✅ NO CRACK DETECTED"

        result_col1, result_col2 = st.columns([1, 1])

        with result_col1:
            st.markdown(f"""
            <div class="glass-card" style="text-align: center;">
                <div class="detection-badge {badge_class}">{badge_text}</div>
                <div style="margin-top: 1.5rem;">
                    <div class="metric-label">Crack Probability</div>
                    <div class="metric-value" style="color: {'#ff3366' if is_crack else '#00e676'};">
                        {crack_prob * 100:.1f}%
                    </div>
                    <div class="prob-bar-container">
                        <div class="prob-bar-fill" style="width: {crack_prob * 100}%;
                            background: linear-gradient(90deg,
                                {'#ffab00, #ff3366' if is_crack else '#00e676, #00d4ff'});"></div>
                    </div>
                </div>
                <div style="margin-top: 1.2rem;">
                    <div class="metric-label">Analysis Confidence</div>
                    <div style="color: #00d4ff; font-size: 1.2rem; font-weight: 600;
                                font-family: 'Orbitron', monospace;">
                        {summary['analysis_confidence']}%
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with result_col2:
            st.markdown(f"""
            <div class="glass-card" style="text-align: center;">
                <div style="font-family: 'Orbitron', monospace; font-weight: 700;
                            color: {risk_info['color']}; font-size: 1.1rem; margin-bottom: 0.5rem;">
                    {risk_info['icon']} {risk_info['level'].upper()}
                </div>
                <div class="metric-label">Structural Health Score</div>
                <div class="metric-value" style="color: {risk_info['color']}; font-size: 3.2rem;">
                    {health_score}
                </div>
                <div style="color: #4a5568; font-size: 0.8rem; margin-top: 0.3rem;
                            font-family: 'JetBrains Mono', monospace;">
                    out of 100
                </div>
                <div style="margin-top: 1rem; padding: 0.8rem; background: rgba(255,255,255,0.02);
                            border-radius: 10px; color: #7a8ba0; font-size: 0.82rem; line-height: 1.6;">
                    {risk_info['description']}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Image Comparison with corner frames ───────────────────────────
        st.markdown("""
        <div class="section-header">
            🖼️ IMAGE ANALYSIS — ORIGINAL vs DAMAGE HEATMAP
        </div>
        """, unsafe_allow_html=True)

        img_col1, img_col2 = st.columns(2)

        with img_col1:
            st.markdown("""
            <div style="text-align: center; margin-bottom: 0.5rem;">
                <span style="font-family: 'Orbitron', monospace; color: #00d4ff;
                             font-size: 0.8rem; letter-spacing: 2px;">ORIGINAL IMAGE</span>
            </div>
            <div class="image-frame">
            """, unsafe_allow_html=True)
            st.image(image, use_column_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with img_col2:
            st.markdown("""
            <div style="text-align: center; margin-bottom: 0.5rem;">
                <span style="font-family: 'Orbitron', monospace; color: #ff3366;
                             font-size: 0.8rem; letter-spacing: 2px;">DAMAGE HEATMAP</span>
            </div>
            <div class="image-frame">
            """, unsafe_allow_html=True)
            st.image(heatmap_img, use_column_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Health Score Gauge ────────────────────────────────────────────
        st.markdown("""
        <div class="section-header">
            📊 STRUCTURAL HEALTH GAUGE
        </div>
        """, unsafe_allow_html=True)

        gauge_col1, gauge_col2 = st.columns([2, 1])

        with gauge_col1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=health_score,
                number={
                    'font': {'size': 60, 'color': risk_info['color'], 'family': 'Orbitron'},
                    'suffix': '%'
                },
                title={
                    'text': "STRUCTURAL INTEGRITY INDEX",
                    'font': {'size': 13, 'color': '#7a8ba0', 'family': 'Orbitron'}
                },
                gauge={
                    'axis': {
                        'range': [0, 100],
                        'tickwidth': 1,
                        'tickcolor': '#1e293b',
                        'tickfont': {'color': '#4a5568', 'size': 10, 'family': 'JetBrains Mono'},
                        'dtick': 10
                    },
                    'bar': {'color': risk_info['color'], 'thickness': 0.25},
                    'bgcolor': 'rgba(255,255,255,0.01)',
                    'borderwidth': 0,
                    'steps': [
                        {'range': [0, 15], 'color': 'rgba(255, 23, 68, 0.18)'},
                        {'range': [15, 30], 'color': 'rgba(255, 23, 68, 0.10)'},
                        {'range': [30, 50], 'color': 'rgba(255, 171, 0, 0.10)'},
                        {'range': [50, 70], 'color': 'rgba(255, 171, 0, 0.06)'},
                        {'range': [70, 85], 'color': 'rgba(0, 230, 118, 0.06)'},
                        {'range': [85, 100], 'color': 'rgba(0, 230, 118, 0.10)'}
                    ],
                    'threshold': {
                        'line': {'color': '#ffffff', 'width': 3},
                        'thickness': 0.85,
                        'value': health_score
                    }
                }
            ))

            fig_gauge.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': '#e8edf5', 'family': 'Inter'},
                height=350,
                margin=dict(l=30, r=30, t=70, b=20)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with gauge_col2:
            st.markdown("""
            <div class="glass-card" style="height: 100%;">
                <div style="font-family: 'Orbitron', monospace; font-size: 0.8rem;
                            color: #00d4ff; margin-bottom: 1.2rem; letter-spacing: 2px;">
                    SCORE LEGEND
                </div>
            """, unsafe_allow_html=True)

            score_levels = [
                ("85–100%", "#00e676", "Excellent", "No maintenance needed"),
                ("70–84%", "#00e676", "Good", "Routine monitoring"),
                ("40–69%", "#ffab00", "Moderate", "Schedule inspection"),
                ("15–39%", "#ff1744", "Poor", "Urgent assessment"),
                ("0–14%", "#ff1744", "Critical", "Immediate action"),
            ]

            for score_range, color, label, desc in score_levels:
                st.markdown(f"""
                <div style="display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.7rem;
                            padding: 0.55rem 0.7rem; background: rgba(255,255,255,0.015); border-radius: 8px;
                            border-left: 3px solid {color}; transition: all 0.3s ease;">
                    <div style="flex: 1;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-weight: 600; color: {color}; font-size: 0.78rem;
                                        font-family: 'JetBrains Mono', monospace;">{score_range}</span>
                            <span style="color: {color}; font-size: 0.7rem; font-weight: 600;">{label}</span>
                        </div>
                        <div style="color: #4a5568; font-size: 0.7rem; margin-top: 0.15rem;">{desc}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Enhanced Risk Prediction Forecast ─────────────────────────────
        st.markdown("""
        <div class="section-header">
            📈 RISK PREDICTION FORECAST
        </div>
        """, unsafe_allow_html=True)

        colors_map = {
            'Safe': '#00e676',
            'Moderate': '#ffab00',
            'Critical': '#ff1744'
        }
        marker_colors = [colors_map.get(r, '#00d4ff') for r in forecast['risk_levels']]

        # Create enhanced dual-axis chart
        fig_forecast = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=("", "")
        )

        # -- Top: Health score line
        # Danger zone fills
        fig_forecast.add_hrect(y0=0, y1=30, fillcolor="rgba(255,23,68,0.06)",
                               line_width=0, layer="below", row=1, col=1)
        fig_forecast.add_hrect(y0=30, y1=70, fillcolor="rgba(255,171,0,0.03)",
                               line_width=0, layer="below", row=1, col=1)
        fig_forecast.add_hrect(y0=70, y1=100, fillcolor="rgba(0,230,118,0.03)",
                               line_width=0, layer="below", row=1, col=1)

        # Threshold lines
        fig_forecast.add_hline(y=70, line_dash="dot", line_color="rgba(0,230,118,0.25)",
                               annotation_text="SAFE",
                               annotation_position="top right",
                               annotation_font_color="#00e676",
                               annotation_font_size=9,
                               row=1, col=1)
        fig_forecast.add_hline(y=30, line_dash="dot", line_color="rgba(255,23,68,0.25)",
                               annotation_text="CRITICAL",
                               annotation_position="bottom right",
                               annotation_font_color="#ff1744",
                               annotation_font_size=9,
                               row=1, col=1)

        # Main health line with gradient fill
        fig_forecast.add_trace(go.Scatter(
            x=forecast['month_labels'],
            y=forecast['scores'],
            mode='lines+markers',
            name='Health Score',
            line=dict(color='#00d4ff', width=3, shape='spline'),
            marker=dict(size=10, color=marker_colors,
                        line=dict(width=2, color='#060a13'),
                        symbol='circle'),
            fill='tozeroy',
            fillcolor='rgba(0, 212, 255, 0.04)',
            hovertemplate="<b>%{x}</b><br>Health Score: <b>%{y:.1f}%</b><extra></extra>"
        ), row=1, col=1)

        # Confidence band (upper/lower)
        upper_bound = [min(s + np.random.uniform(3, 8), 100) for s in forecast['scores']]
        lower_bound = [max(s - np.random.uniform(3, 8), 0) for s in forecast['scores']]

        fig_forecast.add_trace(go.Scatter(
            x=forecast['month_labels'], y=upper_bound,
            mode='lines', line=dict(width=0),
            showlegend=False, hoverinfo='skip'
        ), row=1, col=1)

        fig_forecast.add_trace(go.Scatter(
            x=forecast['month_labels'], y=lower_bound,
            mode='lines', line=dict(width=0),
            fill='tonexty', fillcolor='rgba(0, 212, 255, 0.06)',
            showlegend=False, hoverinfo='skip',
            name='Confidence Band'
        ), row=1, col=1)

        # -- Bottom: Monthly deterioration rate as bar chart
        deterioration = [0]
        for i in range(1, len(forecast['scores'])):
            deterioration.append(round(forecast['scores'][i-1] - forecast['scores'][i], 1))

        bar_colors = ['rgba(255,51,102,0.7)' if d > 3 else 'rgba(255,171,0,0.6)' if d > 1
                      else 'rgba(0,212,255,0.5)' for d in deterioration]

        fig_forecast.add_trace(go.Bar(
            x=forecast['month_labels'],
            y=deterioration,
            name='Monthly Decline',
            marker_color=bar_colors,
            marker_line_width=0,
            hovertemplate="<b>%{x}</b><br>Decline: <b>%{y:.1f}%</b><extra></extra>"
        ), row=2, col=1)

        fig_forecast.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': '#7a8ba0', 'family': 'Inter'},
            height=500,
            margin=dict(l=50, r=30, t=20, b=40),
            showlegend=False,
            hovermode='x unified',
            hoverlabel=dict(
                bgcolor='#0f1623',
                font_size=12,
                font_family='Inter',
                bordercolor='#00d4ff'
            ),
            bargap=0.3,
        )

        fig_forecast.update_yaxes(
            title_text="Health Score (%)", row=1, col=1,
            gridcolor='rgba(255,255,255,0.03)', showgrid=True,
            range=[0, 105],
            tickfont={'color': '#4a5568', 'size': 10},
            title_font={'color': '#7a8ba0', 'size': 11},
        )
        fig_forecast.update_yaxes(
            title_text="Decline %", row=2, col=1,
            gridcolor='rgba(255,255,255,0.03)', showgrid=True,
            tickfont={'color': '#4a5568', 'size': 9},
            title_font={'color': '#7a8ba0', 'size': 10},
        )
        fig_forecast.update_xaxes(
            gridcolor='rgba(255,255,255,0.02)',
            tickfont={'color': '#4a5568', 'size': 10},
        )

        st.plotly_chart(fig_forecast, use_container_width=True)

        # Forecast insight cards
        score_drop = forecast['scores'][0] - forecast['scores'][-1]
        avg_monthly = score_drop / forecast_months if forecast_months > 0 else 0

        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            st.markdown(f"""
            <div class="glass-card" style="text-align: center; padding: 1rem;">
                <div class="metric-label">Total Projected Decline</div>
                <div style="font-family: 'Orbitron', monospace; color: #ff3366;
                            font-size: 1.6rem; font-weight: 700; margin: 0.3rem 0;">
                    -{score_drop:.1f}%
                </div>
                <div style="color: #4a5568; font-size: 0.75rem;">
                    over {forecast_months} months
                </div>
            </div>
            """, unsafe_allow_html=True)
        with fc2:
            st.markdown(f"""
            <div class="glass-card" style="text-align: center; padding: 1rem;">
                <div class="metric-label">Avg Monthly Rate</div>
                <div style="font-family: 'Orbitron', monospace; color: #ffab00;
                            font-size: 1.6rem; font-weight: 700; margin: 0.3rem 0;">
                    -{avg_monthly:.1f}%
                </div>
                <div style="color: #4a5568; font-size: 0.75rem;">
                    per month
                </div>
            </div>
            """, unsafe_allow_html=True)
        with fc3:
            end_score = forecast['scores'][-1]
            end_risk = get_risk_label(end_score)
            end_color = get_score_color(end_score)
            st.markdown(f"""
            <div class="glass-card" style="text-align: center; padding: 1rem;">
                <div class="metric-label">Projected Status</div>
                <div style="font-family: 'Orbitron', monospace; color: {end_color};
                            font-size: 1.6rem; font-weight: 700; margin: 0.3rem 0;">
                    {end_score:.0f}%
                </div>
                <div style="color: {end_color}; font-size: 0.75rem; font-weight: 600;">
                    {end_risk}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Automated Inspection Summary ──────────────────────────────────
        st.markdown("""
        <div class="section-header">
            📋 AUTOMATED INSPECTION SUMMARY
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="analysis-container">
            <div class="analysis-container-inner">
        """, unsafe_allow_html=True)

        summary_items = [
            ("Infrastructure ID", summary['infrastructure_id'], "#00d4ff"),
            ("Inspection Date", summary['inspection_date'], "#e8edf5"),
            ("Crack Probability", f"{summary['crack_probability']}%",
             "#ff3366" if summary['crack_probability'] > 40 else "#00e676"),
            ("Current Health Score", f"{summary['health_score']}%", risk_info['color']),
        ]

        for label, value, color in summary_items:
            st.markdown(f"""
            <div class="summary-row">
                <span class="summary-label">{label}</span>
                <span class="summary-value" style="color: {color};">{value}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="summary-row">
            <span class="summary-label">Risk Level</span>
            <span class="risk-badge" style="background: {risk_info['color']}18;
                  color: {risk_info['color']}; border: 1px solid {risk_info['color']};">
                {risk_info['icon']} {risk_info['level']}
            </span>
        </div>
        """, unsafe_allow_html=True)

        future_risk_color = summary['future_risk_color']
        st.markdown(f"""
        <div class="summary-row">
            <span class="summary-label">Predicted Future Risk ({forecast_months}mo)</span>
            <span class="risk-badge" style="background: {future_risk_color}18;
                  color: {future_risk_color}; border: 1px solid {future_risk_color};">
                {summary['predicted_future_risk']} ({summary['predicted_score_6mo']}%)
            </span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="summary-row">
            <span class="summary-label">Analysis Confidence</span>
            <span class="summary-value" style="color: #00d4ff;">{summary['analysis_confidence']}%</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("</div></div>", unsafe_allow_html=True)

        # Recommended Action
        action_color = risk_info['color']
        st.markdown(f"""
        <div class="glass-card" style="border-left: 4px solid {action_color}; margin-top: 1.5rem;">
            <div style="font-family: 'Orbitron', monospace; font-size: 0.8rem;
                        color: {action_color}; margin-bottom: 0.5rem; letter-spacing: 1.5px;">
                🎯 RECOMMENDED ACTION
            </div>
            <div style="color: #e8edf5; font-size: 0.92rem; line-height: 1.7;">
                {summary['recommended_action']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="glass-card" style="text-align: center; padding: 3.5rem 2rem;">
            <div style="font-size: 4rem; margin-bottom: 1rem; animation: fadeInUp 0.6s ease-out both;">🏗️</div>
            <div style="font-family: 'Orbitron', monospace; font-size: 1.2rem;
                        color: #00d4ff; margin-bottom: 0.5rem; letter-spacing: 3px;">
                AWAITING INPUT
            </div>
            <div style="color: #7a8ba0; font-size: 0.88rem; max-width: 500px; margin: 0 auto;
                        line-height: 1.8;">
                Upload a bridge or building surface image above to begin AI-powered
                structural health analysis. The system will detect cracks, assess damage
                severity, and provide predictive risk analysis.
            </div>
            <div style="display: flex; justify-content: center; gap: 2.5rem; margin-top: 2.5rem;
                        flex-wrap: wrap;">
                <div style="text-align: center; transition: transform 0.3s;">
                    <div style="font-size: 1.8rem;">🔍</div>
                    <div style="color: #4a5568; font-size: 0.72rem; margin-top: 0.4rem;
                                font-family: 'Orbitron', monospace; letter-spacing: 1px;">DETECT</div>
                </div>
                <div style="text-align: center; transition: transform 0.3s;">
                    <div style="font-size: 1.8rem;">🗺️</div>
                    <div style="color: #4a5568; font-size: 0.72rem; margin-top: 0.4rem;
                                font-family: 'Orbitron', monospace; letter-spacing: 1px;">MAP</div>
                </div>
                <div style="text-align: center; transition: transform 0.3s;">
                    <div style="font-size: 1.8rem;">📊</div>
                    <div style="color: #4a5568; font-size: 0.72rem; margin-top: 0.4rem;
                                font-family: 'Orbitron', monospace; letter-spacing: 1px;">SCORE</div>
                </div>
                <div style="text-align: center; transition: transform 0.3s;">
                    <div style="font-size: 1.8rem;">📈</div>
                    <div style="color: #4a5568; font-size: 0.72rem; margin-top: 0.4rem;
                                font-family: 'Orbitron', monospace; letter-spacing: 1px;">PREDICT</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2: MULTI-STRUCTURE COMPARISON
# ════════════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.markdown("""
    <div class="section-header">
        🏗️ MONITORED INFRASTRUCTURE
    </div>
    """, unsafe_allow_html=True)

    structures = get_demo_structures()

    st.markdown('<div class="comparison-grid">', unsafe_allow_html=True)

    comp_cols = st.columns(3)
    for idx, struct in enumerate(structures):
        col = comp_cols[idx % 3]
        score = struct['score']
        color = get_score_color(score)
        risk = get_risk_label(score)
        risk_bg = f"{color}18"

        with col:
            st.markdown(f"""
            <div class="comparison-card" style="animation-delay: {idx * 0.1}s;">
                <div class="comparison-header">
                    <span class="comparison-id">{struct['id']}</span>
                    <span class="comparison-status" style="background: {risk_bg};
                          color: {color}; border: 1px solid {color}40;">
                        {risk}
                    </span>
                </div>
                <div style="font-weight: 600; color: #e8edf5; font-size: 0.95rem; margin-bottom: 0.2rem;">
                    {struct['name']}
                </div>
                <div style="color: #4a5568; font-size: 0.75rem; margin-bottom: 1rem;">
                    {struct['type']} • Last scan: {struct['last_scan']}
                </div>
                <div class="comparison-stat">
                    <span class="comparison-stat-label">Health Score</span>
                    <span class="comparison-stat-value" style="color: {color};">{score}%</span>
                </div>
                <div class="comparison-stat">
                    <span class="comparison-stat-label">Total Scans</span>
                    <span class="comparison-stat-value" style="color: #7a8ba0;">{struct['scans']}</span>
                </div>
                <div class="mini-health-bar">
                    <div class="mini-health-fill" style="width: {score}%; background: linear-gradient(90deg, {color}, {color}88);"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Comparative bar chart
    st.markdown("""
    <div class="section-header">
        📊 HEALTH SCORE COMPARISON
    </div>
    """, unsafe_allow_html=True)

    names = [s['name'] for s in structures]
    scores = [s['score'] for s in structures]
    bar_colors_comp = [get_score_color(s) for s in scores]

    fig_compare = go.Figure()

    fig_compare.add_trace(go.Bar(
        y=names,
        x=scores,
        orientation='h',
        marker=dict(
            color=bar_colors_comp,
            line=dict(width=0),
            opacity=0.85
        ),
        text=[f"{s}%" for s in scores],
        textposition='inside',
        textfont=dict(color='white', size=12, family='Orbitron'),
        hovertemplate="<b>%{y}</b><br>Score: <b>%{x}%</b><extra></extra>"
    ))

    # Threshold lines
    fig_compare.add_vline(x=70, line_dash="dot", line_color="rgba(0,230,118,0.3)",
                          annotation_text="Safe", annotation_position="top",
                          annotation_font_color="#00e676", annotation_font_size=9)
    fig_compare.add_vline(x=30, line_dash="dot", line_color="rgba(255,23,68,0.3)",
                          annotation_text="Critical", annotation_position="top",
                          annotation_font_color="#ff1744", annotation_font_size=9)

    fig_compare.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#7a8ba0', 'family': 'Inter'},
        height=350,
        margin=dict(l=160, r=30, t=30, b=30),
        xaxis=dict(
            range=[0, 105],
            gridcolor='rgba(255,255,255,0.03)',
            showgrid=True,
            title="Health Score (%)",
            title_font={'color': '#7a8ba0', 'size': 11},
            tickfont={'color': '#4a5568', 'size': 10},
        ),
        yaxis=dict(
            tickfont={'color': '#e8edf5', 'size': 11},
            autorange='reversed',
        ),
        showlegend=False,
        hoverlabel=dict(
            bgcolor='#0f1623',
            font_size=12,
            bordercolor='#00d4ff'
        )
    )

    st.plotly_chart(fig_compare, use_container_width=True)

    # Radar chart for multi-dimensional comparison
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="section-header">
        🎯 MULTI-DIMENSIONAL ANALYSIS
    </div>
    """, unsafe_allow_html=True)

    radar_cols = st.columns([1, 1])

    with radar_cols[0]:
        categories = ['Structural', 'Surface', 'Load Bearing', 'Corrosion', 'Vibration']

        fig_radar = go.Figure()

        # Show top 3 structures on radar
        np.random.seed(42)
        radar_colors = ['#00d4ff', '#ff3366', '#00e676']
        radar_fills = ['rgba(0,212,255,0.06)', 'rgba(255,51,102,0.06)', 'rgba(0,230,118,0.06)']
        for i, struct in enumerate(structures[:3]):
            values = [max(10, min(100, struct['score'] + np.random.uniform(-20, 20))) for _ in categories]
            values.append(values[0])

            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill='toself',
                fillcolor=radar_fills[i],
                line=dict(color=radar_colors[i], width=2),
                name=struct['name'],
                marker=dict(size=5)
            ))

        fig_radar.update_layout(
            polar=dict(
                bgcolor='rgba(0,0,0,0)',
                radialaxis=dict(
                    visible=True, range=[0, 100],
                    gridcolor='rgba(255,255,255,0.04)',
                    tickfont={'color': '#4a5568', 'size': 8},
                    linecolor='rgba(255,255,255,0.04)',
                ),
                angularaxis=dict(
                    gridcolor='rgba(255,255,255,0.04)',
                    tickfont={'color': '#7a8ba0', 'size': 10},
                    linecolor='rgba(255,255,255,0.04)',
                ),
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': '#7a8ba0', 'family': 'Inter'},
            height=400,
            margin=dict(l=60, r=60, t=40, b=40),
            showlegend=True,
            legend=dict(
                font=dict(color='#7a8ba0', size=10),
                bgcolor='rgba(0,0,0,0)',
                bordercolor='rgba(255,255,255,0.04)',
                borderwidth=1,
            )
        )

        st.plotly_chart(fig_radar, use_container_width=True)

    with radar_cols[1]:
        # Trend sparklines
        st.markdown("""
        <div class="glass-card" style="padding: 1.5rem;">
            <div style="font-family: 'Orbitron', monospace; font-size: 0.8rem;
                        color: #00d4ff; margin-bottom: 1rem; letter-spacing: 2px;">
                30-DAY TREND
            </div>
        """, unsafe_allow_html=True)

        for struct in structures[:4]:
            score = struct['score']
            color = get_score_color(score)
            trend = "▲" if np.random.random() > 0.5 else "▼"
            trend_color = "#00e676" if trend == "▲" else "#ff3366"
            change = round(np.random.uniform(0.5, 5.0), 1)

            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; align-items: center;
                        padding: 0.6rem 0; border-bottom: 1px solid rgba(255,255,255,0.03);">
                <div>
                    <div style="font-weight: 600; color: #e8edf5; font-size: 0.82rem;">
                        {struct['name']}
                    </div>
                    <div style="color: #4a5568; font-size: 0.7rem;">{struct['id']}</div>
                </div>
                <div style="text-align: right;">
                    <div style="font-family: 'Orbitron', monospace; color: {color};
                                font-weight: 700; font-size: 0.9rem;">{score}%</div>
                    <div style="color: {trend_color}; font-size: 0.7rem;
                                font-family: 'JetBrains Mono', monospace;">
                        {trend} {change}%
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3: ALERT CENTER
# ════════════════════════════════════════════════════════════════════════════════
with tab_alerts:
    st.markdown("""
    <div class="section-header">
        🔔 ALERT CENTER
    </div>
    """, unsafe_allow_html=True)

    # Alert summary cards
    alert_c1, alert_c2, alert_c3, alert_c4 = st.columns(4)

    with alert_c1:
        st.markdown("""
        <div class="metric-card" style="border-left: 3px solid #ff1744;">
            <div class="metric-label">Critical</div>
            <div class="metric-value" style="color: #ff1744; font-size: 2rem;">1</div>
        </div>
        """, unsafe_allow_html=True)

    with alert_c2:
        st.markdown("""
        <div class="metric-card" style="border-left: 3px solid #ffab00;">
            <div class="metric-label">Warnings</div>
            <div class="metric-value" style="color: #ffab00; font-size: 2rem;">2</div>
        </div>
        """, unsafe_allow_html=True)

    with alert_c3:
        st.markdown("""
        <div class="metric-card" style="border-left: 3px solid #00d4ff;">
            <div class="metric-label">Info</div>
            <div class="metric-value" style="color: #00d4ff; font-size: 2rem;">5</div>
        </div>
        """, unsafe_allow_html=True)

    with alert_c4:
        st.markdown("""
        <div class="metric-card" style="border-left: 3px solid #00e676;">
            <div class="metric-label">Resolved</div>
            <div class="metric-value" style="color: #00e676; font-size: 2rem;">12</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Alert list
    alerts_data = [
        {"type": "critical", "icon": "🚨", "title": "CRITICAL THRESHOLD BREACH",
         "message": "River Crossing #12 (BRG-2024-C7) health score dropped to 28.3%. Immediate structural assessment required. Load restriction advisory issued.",
         "time": "14:32 UTC — 2 hours ago", "css": "alert-critical"},
        {"type": "warning", "icon": "⚠️", "title": "ACCELERATED DETERIORATION",
         "message": "Reservoir Dam #3 (DAM-2024-E5) showing 3.2× normal deterioration rate over the last 14 days. Monthly decline: 4.7%. Schedule priority inspection.",
         "time": "13:18 UTC — 3 hours ago", "css": "alert-warning"},
        {"type": "warning", "icon": "⚠️", "title": "APPROACHING THRESHOLD",
         "message": "Metro Tower East (BLD-2024-B3) projected to cross moderate threshold within 45 days at current deterioration rate.",
         "time": "11:05 UTC — 5 hours ago", "css": "alert-warning"},
        {"type": "info", "icon": "ℹ️", "title": "SCAN COMPLETE",
         "message": "Underground Tunnel A (TNL-2024-D2) scan completed. Health score: 91.7% (Excellent). No action required.",
         "time": "12:45 UTC — 4 hours ago", "css": "alert-info"},
        {"type": "info", "icon": "📊", "title": "WEEKLY REPORT GENERATED",
         "message": "Automated weekly health report generated for all 6 monitored infrastructure units. 2 require attention.",
         "time": "09:00 UTC — 7 hours ago", "css": "alert-info"},
        {"type": "success", "icon": "✅", "title": "MAINTENANCE VERIFIED",
         "message": "Highway Bridge #47 (BRG-2024-A1) post-maintenance scan confirms structural improvement. Score increased from 74.1% to 82.4%.",
         "time": "Yesterday — 08:30 UTC", "css": "alert-success"},
    ]

    for alert in alerts_data:
        title_color_map = {
            "alert-critical": "#ff1744",
            "alert-warning": "#ffab00",
            "alert-info": "#00d4ff",
            "alert-success": "#00e676",
        }
        title_color = title_color_map.get(alert['css'], '#00d4ff')

        st.markdown(f"""
        <div class="alert-card {alert['css']}">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div>
                    <div class="alert-title" style="color: {title_color};">
                        {alert['icon']} {alert['title']}
                    </div>
                    <div class="alert-message">{alert['message']}</div>
                </div>
            </div>
            <div class="alert-timestamp">{alert['time']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Alert timeline chart
    st.markdown("""
    <div class="section-header">
        📈 ALERT FREQUENCY — LAST 30 DAYS
    </div>
    """, unsafe_allow_html=True)

    np.random.seed(123)
    days = [(datetime.now() - timedelta(days=i)).strftime('%b %d') for i in range(29, -1, -1)]
    critical_counts = np.random.poisson(0.3, 30)
    warning_counts = np.random.poisson(1.2, 30)
    info_counts = np.random.poisson(2.5, 30)

    fig_timeline = go.Figure()

    fig_timeline.add_trace(go.Bar(
        x=days, y=critical_counts, name='Critical',
        marker_color='rgba(255, 23, 68, 0.75)',
        hovertemplate="<b>%{x}</b><br>Critical: %{y}<extra></extra>"
    ))
    fig_timeline.add_trace(go.Bar(
        x=days, y=warning_counts, name='Warning',
        marker_color='rgba(255, 171, 0, 0.65)',
        hovertemplate="<b>%{x}</b><br>Warning: %{y}<extra></extra>"
    ))
    fig_timeline.add_trace(go.Bar(
        x=days, y=info_counts, name='Info',
        marker_color='rgba(0, 212, 255, 0.45)',
        hovertemplate="<b>%{x}</b><br>Info: %{y}<extra></extra>"
    ))

    fig_timeline.update_layout(
        barmode='stack',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#7a8ba0', 'family': 'Inter'},
        height=300,
        margin=dict(l=40, r=20, t=20, b=40),
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.02)',
            tickfont={'color': '#4a5568', 'size': 9},
            tickangle=-45,
            dtick=3,
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.03)',
            tickfont={'color': '#4a5568', 'size': 10},
            title="Count",
            title_font={'color': '#7a8ba0', 'size': 11},
        ),
        legend=dict(
            font=dict(color='#7a8ba0', size=10),
            bgcolor='rgba(0,0,0,0)',
            orientation='h',
            yanchor='bottom', y=1.02,
            xanchor='right', x=1,
        ),
        hoverlabel=dict(bgcolor='#0f1623', font_size=11, bordercolor='#00d4ff'),
        bargap=0.15,
    )

    st.plotly_chart(fig_timeline, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4: FAILURE IMPACT SIMULATION
# ════════════════════════════════════════════════════════════════════════════════
with tab_impact:
    st.markdown("""
    <div class="section-header">
        💥 FAILURE IMPACT SIMULATION AGENT
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card" style="padding: 1rem 1.5rem; margin-bottom: 1rem;">
        <p style="color: #7a8ba0; font-size: 0.88rem; margin: 0; line-height: 1.7;">
            Simulates the cascading impact if a monitored infrastructure structure fails.
            Calculates blast/collapse radius, affected population, traffic disruption,
            and economic loss estimates. <strong style="color: #ff3366;">Auto-activates when
            Health Score &lt; 50 or Risk = Critical.</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Configuration
    sim_cols = st.columns([1, 1, 1])
    with sim_cols[0]:
        sim_structure_type = st.selectbox(
            "Structure Type",
            ['Bridge', 'Flyover', 'Building', 'Tunnel', 'Dam'],
            index=0,
            help="Select the type of infrastructure"
        )
    with sim_cols[1]:
        sim_health = st.slider("Health Score Override", 5.0, 100.0, 35.0, 1.0,
                               help="Override health score for simulation")
    with sim_cols[2]:
        sim_crack = st.slider("Crack Probability Override", 0.0, 1.0, 0.6, 0.05,
                              help="Override crack probability")

    # Initialize agent
    impact_agent = FailureImpactAgent()
    sim_risk_info = get_risk_level(sim_health)
    auto_activated = impact_agent.should_activate(sim_health, sim_risk_info['level'])

    # Auto-activation banner
    if auto_activated:
        st.markdown("""
        <div class="alert-card alert-critical" style="text-align: center; padding: 1rem;">
            <div class="alert-title" style="color: #ff1744; font-size: 0.85rem;">⚡ SIMULATION AUTO-ACTIVATED</div>
            <div class="alert-message">Health Score below 50% or Risk Level is Critical — Impact simulation is running automatically.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="alert-card alert-info" style="text-align: center; padding: 1rem;">
            <div class="alert-title" style="color: #00d4ff; font-size: 0.85rem;">ℹ️ MANUAL MODE</div>
            <div class="alert-message">Health Score is above threshold. Simulation running in manual/preview mode.</div>
        </div>
        """, unsafe_allow_html=True)

    run_sim = st.button("🚀 RUN IMPACT SIMULATION", use_container_width=True)

    if run_sim or auto_activated:
        sim_result = impact_agent.run_simulation(
            health_score=sim_health,
            risk_level=sim_risk_info['level'],
            structure_type=sim_structure_type,
            crack_probability=sim_crack,
        )

        st.markdown("---")

        # ── Impact Classification Banner ──────────────────────────────
        ic = sim_result['impact_class']
        ic_color = sim_result['impact_color']
        st.markdown(f"""
        <div style="text-align: center; padding: 1.5rem; margin: 1rem 0;
                    background: linear-gradient(135deg, {ic_color}15, {ic_color}08);
                    border: 1px solid {ic_color}40;
                    border-radius: 16px;">
            <div style="font-size: 2.5rem; margin-bottom: 0.3rem;">{sim_result['impact_icon']}</div>
            <div style="font-family: 'Orbitron', monospace; font-size: 1.5rem; font-weight: 800;
                        color: {ic_color}; letter-spacing: 3px;">
                {ic} IMPACT
            </div>
            <div style="color: #7a8ba0; font-size: 0.85rem; margin-top: 0.5rem; max-width: 600px;
                        margin-left: auto; margin-right: auto;">
                {sim_result['impact_description']}
            </div>
            <div style="margin-top: 0.8rem; font-family: 'JetBrains Mono', monospace;
                        color: #4a5568; font-size: 0.72rem;">
                SIM ID: {sim_result['simulation_id']} | {sim_result['timestamp']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Key Metrics Row ───────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.markdown(f"""
            <div class="metric-card" style="border-top: 3px solid #ff1744;">
                <div class="metric-label">Impact Radius</div>
                <div class="metric-value" style="color: #ff1744; font-size: 1.8rem;">
                    {sim_result['impact_radius_m']}m
                </div>
                <div style="color: #4a5568; font-size: 0.7rem; margin-top: 0.2rem;">collapse zone</div>
            </div>
            """, unsafe_allow_html=True)

        with m2:
            st.markdown(f"""
            <div class="metric-card" style="border-top: 3px solid #ff6d00;">
                <div class="metric-label">People Affected</div>
                <div class="metric-value" style="color: #ff6d00; font-size: 1.8rem;">
                    {format_population(sim_result['affected_population'])}
                </div>
                <div style="color: #4a5568; font-size: 0.7rem; margin-top: 0.2rem;">est. population</div>
            </div>
            """, unsafe_allow_html=True)

        with m3:
            st.markdown(f"""
            <div class="metric-card" style="border-top: 3px solid {sim_result['traffic_color']};">
                <div class="metric-label">Traffic Disruption</div>
                <div class="metric-value" style="color: {sim_result['traffic_color']}; font-size: 1.8rem;">
                    {sim_result['traffic_disruption']}
                </div>
                <div style="color: #4a5568; font-size: 0.7rem; margin-top: 0.2rem;">
                    {sim_result['affected_routes']} routes affected
                </div>
            </div>
            """, unsafe_allow_html=True)

        with m4:
            st.markdown(f"""
            <div class="metric-card" style="border-top: 3px solid #ffab00;">
                <div class="metric-label">Economic Loss</div>
                <div class="metric-value" style="color: #ffab00; font-size: 1.8rem;">
                    {format_currency(sim_result['total_economic_loss'])}
                </div>
                <div style="color: #4a5568; font-size: 0.7rem; margin-top: 0.2rem;">total estimated</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Impact Zone Visualization ─────────────────────────────────
        viz_col1, viz_col2 = st.columns([3, 2])

        with viz_col1:
            st.markdown("""
            <div class="section-header">
                🎯 IMPACT ZONE VISUALIZATION
            </div>
            """, unsafe_allow_html=True)

            # Create concentric impact zone chart
            r_outer = sim_result['impact_radius_m']
            r_mid = sim_result['outer_zone_m']
            r_inner = sim_result['inner_zone_m']

            # Generate circle points
            theta = np.linspace(0, 2 * np.pi, 100)

            fig_impact = go.Figure()

            # Outer advisory zone
            fig_impact.add_trace(go.Scatter(
                x=r_outer * np.cos(theta), y=r_outer * np.sin(theta),
                fill='toself', fillcolor='rgba(255,171,0,0.08)',
                line=dict(color='rgba(255,171,0,0.4)', width=1, dash='dot'),
                name=f'Advisory Zone ({r_outer}m)',
                hoverinfo='name'
            ))

            # Evacuation zone
            fig_impact.add_trace(go.Scatter(
                x=r_mid * np.cos(theta), y=r_mid * np.sin(theta),
                fill='toself', fillcolor='rgba(255,109,0,0.12)',
                line=dict(color='rgba(255,109,0,0.6)', width=1.5, dash='dash'),
                name=f'Evacuation Zone ({r_mid}m)',
                hoverinfo='name'
            ))

            # Immediate danger zone
            fig_impact.add_trace(go.Scatter(
                x=r_inner * np.cos(theta), y=r_inner * np.sin(theta),
                fill='toself', fillcolor='rgba(255,23,68,0.15)',
                line=dict(color='rgba(255,23,68,0.8)', width=2),
                name=f'Danger Zone ({r_inner}m)',
                hoverinfo='name'
            ))

            # Structure point at center
            fig_impact.add_trace(go.Scatter(
                x=[0], y=[0],
                mode='markers+text',
                marker=dict(size=14, color='#ff1744',
                            symbol='x', line=dict(width=2, color='white')),
                text=[sim_structure_type],
                textposition='top center',
                textfont=dict(color='#ff1744', size=11, family='Orbitron'),
                name='Structure',
                hoverinfo='name'
            ))

            fig_impact.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#7a8ba0', family='Inter'),
                height=420,
                margin=dict(l=40, r=40, t=20, b=40),
                xaxis=dict(
                    title='Distance (meters)',
                    gridcolor='rgba(255,255,255,0.03)',
                    zeroline=False,
                    tickfont=dict(color='#4a5568', size=9),
                    title_font=dict(color='#7a8ba0', size=10),
                    scaleanchor='y', scaleratio=1,
                ),
                yaxis=dict(
                    title='Distance (meters)',
                    gridcolor='rgba(255,255,255,0.03)',
                    zeroline=False,
                    tickfont=dict(color='#4a5568', size=9),
                    title_font=dict(color='#7a8ba0', size=10),
                ),
                showlegend=True,
                legend=dict(
                    font=dict(color='#7a8ba0', size=9),
                    bgcolor='rgba(0,0,0,0)',
                    bordercolor='rgba(255,255,255,0.04)',
                    borderwidth=1,
                    yanchor='top', y=0.99, xanchor='left', x=0.01,
                ),
            )

            st.plotly_chart(fig_impact, use_container_width=True)

        with viz_col2:
            st.markdown("""
            <div class="section-header">
                👥 POPULATION IMPACT
            </div>
            """, unsafe_allow_html=True)

            zones = [
                ('🔴 Immediate Danger', sim_result['immediate_zone_pop'], '#ff1744',
                 f'{sim_result["inner_zone_m"]}m radius'),
                ('🟠 Evacuation Zone', sim_result['evacuation_zone_pop'], '#ff6d00',
                 f'{sim_result["outer_zone_m"]}m radius'),
                ('🟡 Advisory Zone', sim_result['advisory_zone_pop'], '#ffab00',
                 f'{sim_result["impact_radius_m"]}m radius'),
            ]

            for zone_name, pop, color, radius in zones:
                st.markdown(f"""
                <div class="alert-card" style="background: {color}08; border: 1px solid {color}30;
                            border-left: 4px solid {color}; padding: 0.8rem 1rem; margin: 0.5rem 0;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <div style="font-weight: 600; color: {color}; font-size: 0.82rem;">
                                {zone_name}
                            </div>
                            <div style="color: #4a5568; font-size: 0.72rem; margin-top: 0.15rem;">{radius}</div>
                        </div>
                        <div style="font-family: 'Orbitron', monospace; color: {color};
                                    font-weight: 700; font-size: 1.2rem;">
                            {format_population(pop)}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="glass-card" style="text-align: center; padding: 1rem; margin-top: 0.5rem;">
                <div class="metric-label">Total Affected</div>
                <div class="metric-value" style="color: #ff3366; font-size: 2rem;">
                    {format_population(sim_result['affected_population'])}
                </div>
                <div style="color: #4a5568; font-size: 0.72rem;">estimated population</div>
            </div>
            """, unsafe_allow_html=True)

            # Collapse probability
            cp = sim_result['collapse_probability']
            cp_color = '#ff1744' if cp > 30 else '#ffab00' if cp > 10 else '#00e676'
            st.markdown(f"""
            <div class="glass-card" style="text-align: center; padding: 1rem; margin-top: 0.5rem;
                        border-left: 3px solid {cp_color};">
                <div class="metric-label">Collapse Probability</div>
                <div class="metric-value" style="color: {cp_color}; font-size: 1.8rem;">
                    {cp}%
                </div>
                <div class="prob-bar-container" style="margin-top: 0.3rem;">
                    <div class="prob-bar-fill" style="width: {cp}%;
                        background: linear-gradient(90deg, #ffab00, {cp_color});"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Economic Breakdown ────────────────────────────────────────
        st.markdown("""
        <div class="section-header">
            💰 ECONOMIC IMPACT BREAKDOWN
        </div>
        """, unsafe_allow_html=True)

        eco_categories = ['Repair Cost', 'Traffic Loss', 'Productivity Loss', 'Emergency Response']
        eco_values = [
            sim_result['repair_cost'],
            sim_result['traffic_loss'],
            sim_result['productivity_loss'],
            sim_result['emergency_response_cost']
        ]
        eco_colors = ['#ff3366', '#ff6d00', '#ffab00', '#7c4dff']

        eco_col1, eco_col2 = st.columns([1, 1])

        with eco_col1:
            fig_eco = go.Figure(go.Pie(
                labels=eco_categories,
                values=eco_values,
                marker=dict(colors=eco_colors, line=dict(color='#060a13', width=2)),
                hole=0.55,
                textinfo='label+percent',
                textfont=dict(color='#e8edf5', size=11),
                hovertemplate="<b>%{label}</b><br>%{value:,.0f}<br>%{percent}<extra></extra>"
            ))

            fig_eco.add_annotation(
                text=f"<b>{format_currency(sim_result['total_economic_loss'])}</b><br>" +
                     "<span style='font-size:10px;color:#7a8ba0'>TOTAL LOSS</span>",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color='#ff3366', family='Orbitron'),
            )

            fig_eco.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#7a8ba0', family='Inter'),
                height=350,
                margin=dict(l=20, r=20, t=20, b=20),
                showlegend=False,
            )
            st.plotly_chart(fig_eco, use_container_width=True)

        with eco_col2:
            for cat, val, color in zip(eco_categories, eco_values, eco_colors):
                pct = (val / sim_result['total_economic_loss']) * 100 if sim_result['total_economic_loss'] > 0 else 0
                st.markdown(f"""
                <div style="display: flex; align-items: center; gap: 1rem; padding: 0.8rem 1rem;
                            margin: 0.4rem 0; background: rgba(255,255,255,0.015); border-radius: 10px;
                            border-left: 3px solid {color};">
                    <div style="flex: 1;">
                        <div style="font-weight: 600; color: #e8edf5; font-size: 0.85rem;">{cat}</div>
                        <div style="color: #4a5568; font-size: 0.72rem; margin-top: 0.15rem;">
                            {pct:.1f}% of total
                        </div>
                    </div>
                    <div style="font-family: 'Orbitron', monospace; color: {color};
                                font-weight: 700; font-size: 1.05rem;">
                        {format_currency(val)}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Traffic details
            st.markdown(f"""
            <div class="glass-card" style="padding: 1rem; margin-top: 0.6rem;">
                <div style="font-family: 'Orbitron', monospace; font-size: 0.75rem;
                            color: #00d4ff; letter-spacing: 1.5px; margin-bottom: 0.6rem;">TRAFFIC DETAILS</div>
                <div class="comparison-stat">
                    <span class="comparison-stat-label">Affected Routes</span>
                    <span class="comparison-stat-value" style="color: #ff6d00;">{sim_result['affected_routes']}</span>
                </div>
                <div class="comparison-stat">
                    <span class="comparison-stat-label">Detour Distance</span>
                    <span class="comparison-stat-value" style="color: #ffab00;">{sim_result['detour_distance_km']} km</span>
                </div>
                <div class="comparison-stat">
                    <span class="comparison-stat-label">Recovery Time</span>
                    <span class="comparison-stat-value" style="color: #ff3366;">{sim_result['recovery_days']} days</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Government Report / Recommendations ───────────────────────
        st.markdown("""
        <div class="section-header">
            📋 FAILURE IMPACT ANALYSIS — GOVERNMENT REPORT
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="analysis-container">
            <div class="analysis-container-inner">
        """, unsafe_allow_html=True)

        report_rows = [
            ('Simulation ID', sim_result['simulation_id'], '#00d4ff'),
            ('Structure Type', sim_result['structure_type'], '#e8edf5'),
            ('Health Score', f"{sim_result['health_score']}%", sim_risk_info['color']),
            ('Risk Level', sim_result['risk_level'], sim_risk_info['color']),
            ('Impact Classification', sim_result['impact_class'], sim_result['impact_color']),
            ('Collapse Probability', f"{sim_result['collapse_probability']}%", cp_color),
            ('Impact Radius', f"{sim_result['impact_radius_m']}m", '#ff1744'),
            ('Affected Population', f"{sim_result['affected_population']:,}", '#ff6d00'),
            ('Traffic Disruption', sim_result['traffic_disruption'], sim_result['traffic_color']),
            ('Total Economic Loss', format_currency(sim_result['total_economic_loss']), '#ffab00'),
            ('Est. Recovery Time', f"{sim_result['recovery_days']} days", '#7c4dff'),
        ]

        for label, value, color in report_rows:
            st.markdown(f"""
            <div class="summary-row">
                <span class="summary-label">{label}</span>
                <span class="summary-value" style="color: {color};">{value}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div></div>", unsafe_allow_html=True)

        # Recommendations
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="section-header">
            🎯 PRIORITIZED RECOMMENDATIONS
        </div>
        """, unsafe_allow_html=True)

        for rec in sim_result['recommendations']:
            st.markdown(f"""
            <div class="alert-card" style="background: {rec['color']}08; border: 1px solid {rec['color']}25;
                        border-left: 4px solid {rec['color']}; padding: 0.8rem 1.2rem; margin: 0.4rem 0;">
                <div style="display: flex; align-items: flex-start; gap: 0.8rem;">
                    <span style="font-family: 'Orbitron', monospace; font-size: 0.68rem;
                                font-weight: 700; color: {rec['color']}; background: {rec['color']}15;
                                padding: 0.2rem 0.6rem; border-radius: 6px; white-space: nowrap;">
                        {rec['priority']}
                    </span>
                    <span style="color: #e8edf5; font-size: 0.85rem; line-height: 1.5;">
                        {rec['action']}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 5: STRUCTURAL AGING CLOCK
# ════════════════════════════════════════════════════════════════════════════════
with tab_aging:
    st.markdown("""
    <div class="section-header">
        ⏳ STRUCTURAL AGING CLOCK AGENT
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card" style="padding: 1rem 1.5rem; margin-bottom: 1rem;">
        <p style="color: #7a8ba0; font-size: 0.88rem; margin: 0; line-height: 1.7;">
            Estimates remaining structural life and visualizes degradation trajectory.
            Combines health score, crack severity, and predicted growth rate to calculate
            lifecycle intelligence. <strong style="color: #ffab00;">Automatically updates
            when a new scan is analyzed.</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Configuration
    age_cols = st.columns([1, 1, 1, 1])
    with age_cols[0]:
        age_structure_type = st.selectbox(
            "Structure Type ",  # trailing space to avoid duplicate widget ID
            ['Bridge', 'Flyover', 'Building', 'Tunnel', 'Dam'],
            index=0, key='aging_struct'
        )
    with age_cols[1]:
        age_health = st.slider("Health Score", 5.0, 100.0, 42.0, 1.0, key='aging_health')
    with age_cols[2]:
        age_crack = st.slider("Crack Probability", 0.0, 1.0, 0.55, 0.05, key='aging_crack')
    with age_cols[3]:
        age_years = st.slider("Structure Age (years)", 1, 80, 25, 1, key='aging_years')

    # Run analysis
    aging_agent = AgingClockAgent()
    aging_result = aging_agent.run_analysis(
        health_score=age_health,
        crack_probability=age_crack,
        structure_type=age_structure_type,
        structure_age_years=age_years,
    )

    # ── Aging Status Banner ───────────────────────────────────────
    ac = aging_result['aging_color']
    st.markdown(f"""
    <div style="text-align: center; padding: 1.5rem; margin: 1rem 0;
                background: linear-gradient(135deg, rgba({int(ac[1:3],16)},{int(ac[3:5],16)},{int(ac[5:7],16)},0.1),
                            rgba({int(ac[1:3],16)},{int(ac[3:5],16)},{int(ac[5:7],16)},0.03));
                border: 1px solid {ac}40; border-radius: 16px;">
        <div style="font-size: 2.5rem; margin-bottom: 0.3rem;">{aging_result['aging_icon']}</div>
        <div style="font-family: 'Orbitron', monospace; font-size: 1.4rem; font-weight: 800;
                    color: {ac}; letter-spacing: 3px;">
            {aging_result['aging_status'].upper()}
        </div>
        <div style="color: #7a8ba0; font-size: 0.85rem; margin-top: 0.5rem; max-width: 600px;
                    margin-left: auto; margin-right: auto;">
            {aging_result['aging_description']}
        </div>
        <div style="margin-top: 0.8rem; font-family: 'JetBrains Mono', monospace;
                    color: #4a5568; font-size: 0.72rem;">
            ID: {aging_result['analysis_id']} | {aging_result['timestamp']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Acceleration Warning ─────────────────────────────────────
    if aging_result['acceleration_detected']:
        alert_css = 'alert-critical' if aging_result['acceleration_severity'] == 'CRITICAL' else 'alert-warning'
        st.markdown(f"""
        <div class="alert-card {alert_css}" style="text-align: center; padding: 1rem;">
            <div class="alert-message" style="font-size: 0.88rem;">{aging_result['acceleration_message']}</div>
        </div>
        """, unsafe_allow_html=True)

    if aging_result['life_below_3_years']:
        st.markdown("""
        <div class="alert-card alert-critical" style="text-align: center; padding: 0.8rem;">
            <div class="alert-title" style="color: #ff1744;">\U0001f6a8 CRITICAL LIFE ALERT</div>
            <div class="alert-message">Remaining safe life is below 3 years. Immediate structural engineering review required.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Key Metrics ──────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)

    rl_color = '#00e676' if aging_result['remaining_life_years'] >= 10 else '#ffab00' if aging_result['remaining_life_years'] >= 3 else '#ff1744'

    with k1:
        st.markdown(f"""
        <div class="metric-card" style="border-top: 3px solid {rl_color};">
            <div class="metric-label">Remaining Life</div>
            <div class="metric-value" style="color: {rl_color}; font-size: 1.6rem;">
                {aging_result['remaining_life_years']}y
            </div>
            <div style="color: #4a5568; font-size: 0.68rem;">{aging_result['remaining_life_months']} months</div>
        </div>
        """, unsafe_allow_html=True)

    with k2:
        dr = aging_result['annual_degradation_rate']
        st.markdown(f"""
        <div class="metric-card" style="border-top: 3px solid {aging_result['degradation_trend_color']};">
            <div class="metric-label">Degradation Rate</div>
            <div class="metric-value" style="color: {aging_result['degradation_trend_color']}; font-size: 1.6rem;">
                {dr}%/yr
            </div>
            <div style="color: {aging_result['degradation_trend_color']}; font-size: 0.68rem;">
                {aging_result['degradation_trend_icon']} {aging_result['degradation_trend']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with k3:
        st.markdown(f"""
        <div class="metric-card" style="border-top: 3px solid {aging_result['criticality_color']};">
            <div class="metric-label">Criticality Score</div>
            <div class="metric-value" style="color: {aging_result['criticality_color']}; font-size: 1.6rem;">
                {aging_result['criticality_score']}
            </div>
            <div style="color: {aging_result['criticality_color']}; font-size: 0.68rem;">
                {aging_result['criticality_level']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with k4:
        st.markdown(f"""
        <div class="metric-card" style="border-top: 3px solid #7c4dff;">
            <div class="metric-label">Life Used</div>
            <div class="metric-value" style="color: #7c4dff; font-size: 1.6rem;">
                {aging_result['life_percentage_used']}%
            </div>
            <div style="color: #4a5568; font-size: 0.68rem;">
                of {aging_result['design_life_years']}yr design
            </div>
        </div>
        """, unsafe_allow_html=True)

    with k5:
        st.markdown(f"""
        <div class="metric-card" style="border-top: 3px solid #00d4ff;">
            <div class="metric-label">Critical Year</div>
            <div class="metric-value" style="color: #00d4ff; font-size: 1.1rem;">
                {aging_result['safe_operation_until']}
            </div>
            <div style="color: #4a5568; font-size: 0.68rem;">projected safe until</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Main Content: Lifecycle Chart + Aging Meter ───────────────
    chart_col, meter_col = st.columns([3, 2])

    with chart_col:
        st.markdown("""
        <div class="section-header">
            📉 LIFECYCLE DEGRADATION PROJECTION
        </div>
        """, unsafe_allow_html=True)

        proj = aging_result
        scores = proj['projection_scores']
        labels = proj['projection_labels']

        fig_lifecycle = go.Figure()

        # Danger zones
        fig_lifecycle.add_hrect(y0=0, y1=20, fillcolor='rgba(255,23,68,0.06)',
                                line_width=0, layer='below')
        fig_lifecycle.add_hrect(y0=20, y1=40, fillcolor='rgba(255,171,0,0.03)',
                                line_width=0, layer='below')

        fig_lifecycle.add_hline(y=20, line_dash='dot', line_color='rgba(255,23,68,0.3)',
                                annotation_text='CRITICAL THRESHOLD',
                                annotation_position='bottom right',
                                annotation_font_color='#ff1744',
                                annotation_font_size=9)

        # Color markers based on score
        marker_colors = []
        for s in scores:
            if s >= 60:
                marker_colors.append('#00e676')
            elif s >= 40:
                marker_colors.append('#ffab00')
            elif s >= 20:
                marker_colors.append('#ff6d00')
            else:
                marker_colors.append('#ff1744')

        # Main projection line
        fig_lifecycle.add_trace(go.Scatter(
            x=labels, y=scores,
            mode='lines+markers',
            name='Projected Health',
            line=dict(color='#00d4ff', width=3, shape='spline'),
            marker=dict(size=8, color=marker_colors,
                        line=dict(width=2, color='#060a13')),
            fill='tozeroy',
            fillcolor='rgba(0,212,255,0.04)',
            hovertemplate='<b>%{x}</b><br>Health: <b>%{y:.1f}%</b><extra></extra>'
        ))

        # Mark current year
        fig_lifecycle.add_shape(
            type='line', x0=labels[0], x1=labels[0], y0=0, y1=105,
            line=dict(color='rgba(0,212,255,0.4)', width=1, dash='solid')
        )
        fig_lifecycle.add_annotation(
            x=labels[0], y=102, text='NOW', showarrow=False,
            font=dict(color='#00d4ff', size=9, family='Orbitron')
        )

        fig_lifecycle.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#7a8ba0', family='Inter'),
            height=400,
            margin=dict(l=50, r=30, t=30, b=40),
            xaxis=dict(
                gridcolor='rgba(255,255,255,0.02)',
                tickfont=dict(color='#4a5568', size=10),
                title='Year', title_font=dict(color='#7a8ba0', size=11),
            ),
            yaxis=dict(
                range=[0, 105],
                gridcolor='rgba(255,255,255,0.03)',
                tickfont=dict(color='#4a5568', size=10),
                title='Health Score (%)', title_font=dict(color='#7a8ba0', size=11),
            ),
            showlegend=False,
            hoverlabel=dict(bgcolor='#0f1623', font_size=12, bordercolor='#00d4ff'),
        )

        st.plotly_chart(fig_lifecycle, use_container_width=True)

    with meter_col:
        st.markdown("""
        <div class="section-header">
            ⏱️ AGING COUNTDOWN
        </div>
        """, unsafe_allow_html=True)

        # Remaining life gauge
        rl = aging_result['remaining_life_years']
        fig_aging_gauge = go.Figure(go.Indicator(
            mode='gauge+number',
            value=rl,
            number=dict(
                font=dict(size=48, color=rl_color, family='Orbitron'),
                suffix=' yrs'
            ),
            title=dict(
                text='ESTIMATED REMAINING LIFE',
                font=dict(size=11, color='#7a8ba0', family='Orbitron')
            ),
            gauge=dict(
                axis=dict(range=[0, 30], tickwidth=1, tickcolor='#1e293b',
                          tickfont=dict(color='#4a5568', size=9), dtick=5),
                bar=dict(color=rl_color, thickness=0.25),
                bgcolor='rgba(255,255,255,0.01)',
                borderwidth=0,
                steps=[
                    dict(range=[0, 3], color='rgba(255,23,68,0.15)'),
                    dict(range=[3, 8], color='rgba(255,171,0,0.08)'),
                    dict(range=[8, 15], color='rgba(255,171,0,0.04)'),
                    dict(range=[15, 30], color='rgba(0,230,118,0.04)'),
                ],
                threshold=dict(
                    line=dict(color='white', width=3),
                    thickness=0.8, value=rl
                )
            )
        ))

        fig_aging_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e8edf5', family='Inter'),
            height=260,
            margin=dict(l=25, r=25, t=60, b=10),
        )
        st.plotly_chart(fig_aging_gauge, use_container_width=True)

        # Aging details
        aging_details = [
            ('Annual Degradation', f"{aging_result['annual_degradation_rate']}%/yr",
             aging_result['degradation_trend_color']),
            ('Monthly Rate', f"{aging_result['monthly_degradation_rate']}%/mo",
             aging_result['degradation_trend_color']),
            ('Degradation Trend', f"{aging_result['degradation_trend_icon']} {aging_result['degradation_trend']}",
             aging_result['degradation_trend_color']),
            ('Structure Age', f"{age_years} years", '#7c4dff'),
            ('Design Life', f"{aging_result['design_life_years']} years", '#00d4ff'),
            ('Projected Critical', aging_result['projected_critical_year_label'], '#ff1744'),
        ]

        st.markdown("""
        <div class="glass-card" style="padding: 1rem;">
            <div style="font-family: 'Orbitron', monospace; font-size: 0.75rem;
                        color: #00d4ff; letter-spacing: 1.5px; margin-bottom: 0.6rem;">
                AGING PARAMETERS
            </div>
        """, unsafe_allow_html=True)

        for label, value, color in aging_details:
            st.markdown(f"""
            <div class="comparison-stat">
                <span class="comparison-stat-label">{label}</span>
                <span class="comparison-stat-value" style="color: {color};">{value}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Criticality Gauge + Life Usage Bar ───────────────────────
    crit_col1, crit_col2 = st.columns([1, 1])

    with crit_col1:
        st.markdown("""
        <div class="section-header">
            ⚠️ CRITICALITY INDEX
        </div>
        """, unsafe_allow_html=True)

        cs = aging_result['criticality_score']
        cc = aging_result['criticality_color']

        fig_crit = go.Figure(go.Indicator(
            mode='gauge+number',
            value=cs,
            number=dict(font=dict(size=50, color=cc, family='Orbitron')),
            title=dict(
                text='COMBINED CRITICALITY',
                font=dict(size=11, color='#7a8ba0', family='Orbitron')
            ),
            gauge=dict(
                axis=dict(range=[0, 100], tickwidth=1, tickcolor='#1e293b',
                          tickfont=dict(color='#4a5568', size=9), dtick=10),
                bar=dict(color=cc, thickness=0.25),
                bgcolor='rgba(255,255,255,0.01)',
                borderwidth=0,
                steps=[
                    dict(range=[0, 30], color='rgba(0,230,118,0.06)'),
                    dict(range=[30, 50], color='rgba(255,171,0,0.06)'),
                    dict(range=[50, 75], color='rgba(255,109,0,0.08)'),
                    dict(range=[75, 100], color='rgba(255,23,68,0.12)'),
                ],
                threshold=dict(
                    line=dict(color='white', width=3),
                    thickness=0.8, value=cs
                )
            )
        ))

        fig_crit.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e8edf5', family='Inter'),
            height=300,
            margin=dict(l=25, r=25, t=60, b=20),
        )
        st.plotly_chart(fig_crit, use_container_width=True)

    with crit_col2:
        st.markdown("""
        <div class="section-header">
            📋 STRUCTURAL AGING ANALYSIS
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="analysis-container">
            <div class="analysis-container-inner">
        """, unsafe_allow_html=True)

        report_data = [
            ('Analysis ID', aging_result['analysis_id'], '#00d4ff'),
            ('Structure Type', aging_result['structure_type'], '#e8edf5'),
            ('Structure Age', f"{age_years} years", '#7c4dff'),
            ('Health Score', f"{aging_result['health_score']}%",
             '#00e676' if aging_result['health_score'] >= 60 else '#ffab00' if aging_result['health_score'] >= 30 else '#ff1744'),
            ('Remaining Life', f"{aging_result['remaining_life_years']} years", rl_color),
            ('Degradation Rate', f"{aging_result['annual_degradation_rate']}%/yr",
             aging_result['degradation_trend_color']),
            ('Aging Status', aging_result['aging_status'], aging_result['aging_color']),
            ('Criticality', f"{aging_result['criticality_score']} ({aging_result['criticality_level']})",
             aging_result['criticality_color']),
            ('Safe Until', aging_result['safe_operation_until'], '#00d4ff'),
            ('Critical Year', aging_result['projected_critical_year_label'], '#ff1744'),
        ]

        for label, value, color in report_data:
            st.markdown(f"""
            <div class="summary-row">
                <span class="summary-label">{label}</span>
                <span class="summary-value" style="color: {color};">{value}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div></div>", unsafe_allow_html=True)

    # Summary text
    st.markdown(f"""
    <div class="glass-card" style="border-left: 4px solid {aging_result['aging_color']}; margin-top: 1rem;">
        <div style="font-family: 'Orbitron', monospace; font-size: 0.8rem;
                    color: {aging_result['aging_color']}; margin-bottom: 0.5rem; letter-spacing: 1.5px;">
            📝 AGING ANALYSIS SUMMARY
        </div>
        <div style="color: #e8edf5; font-size: 0.9rem; line-height: 1.8;">
            {aging_result['summary_text']}
        </div>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 6: GOVERNMENT COMPLAINT & EMERGENCY ALERT
# ════════════════════════════════════════════════════════════════════════════════
with tab_govt:
    st.markdown("""
    <div class="section-header">
        🏢 GOVERNMENT COMPLAINT & EMERGENCY ALERT AGENT
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card" style="padding: 1rem 1.5rem; margin-bottom: 1rem;">
        <p style="color: #7a8ba0; font-size: 0.88rem; margin: 0; line-height: 1.7;">
            Automatically generates government-level inspection complaints when infrastructure
            risk reaches critical thresholds. Computes the <strong style="color: #ff3366;">Critical
            Risk Score (CRS)</strong> and triggers emergency alerts with full location intelligence
            and downloadable inspection reports.
        </p>
    </div>
    """, unsafe_allow_html=True)

    govt_agent = GovernmentComplaintAgent()

    # ── Configuration ────────────────────────────────────────────
    st.markdown("""
    <div class="section-header" style="font-size: 1rem;">
        ⚙️ CRS INPUT PARAMETERS
    </div>
    """, unsafe_allow_html=True)

    g_col1, g_col2, g_col3, g_col4 = st.columns(4)
    with g_col1:
        g_health = st.slider("Structural Health Index", 5.0, 100.0, 32.0, 1.0, key='g_health')
    with g_col2:
        g_predicted_risk = st.slider("Predicted Risk %", 0.0, 100.0, 72.0, 1.0, key='g_pred')
    with g_col3:
        g_aging_risk = st.slider("Aging Risk Factor", 0.0, 100.0, 65.0, 1.0, key='g_aging')
    with g_col4:
        g_impact_sev = st.slider("Impact Severity", 0.0, 100.0, 80.0, 1.0, key='g_impact')

    g2_col1, g2_col2, g2_col3 = st.columns(3)
    with g2_col1:
        g_structure_id = st.selectbox(
            "Select Structure",
            list(govt_agent.LOCATION_DB.keys()),
            index=2,  # River Crossing (critical)
            key='g_struct',
            format_func=lambda x: f"{x} — {govt_agent.LOCATION_DB[x]['name']}"
        )
    with g2_col2:
        g_remaining_life = st.slider("Remaining Life (years)", 0.5, 30.0, 2.5, 0.5, key='g_life')
    with g2_col3:
        g_impact_radius = st.slider("Impact Radius (m)", 50, 1000, 350, 10, key='g_radius')

    # Compute CRS
    crs = govt_agent.compute_crs(g_health, g_predicted_risk, g_aging_risk, g_impact_sev)
    trigger = govt_agent.should_trigger(crs['score'], g_health, g_remaining_life)

    st.markdown("---")

    # ── CRS Score Display ───────────────────────────────────────
    crs_col, breakdown_col = st.columns([1, 1])

    with crs_col:
        st.markdown(f"""
        <div style="text-align: center; padding: 2rem; margin: 0.5rem 0;
                    background: linear-gradient(135deg, rgba({int(crs['color'][1:3],16)},{int(crs['color'][3:5],16)},{int(crs['color'][5:7],16)},0.12),
                                rgba({int(crs['color'][1:3],16)},{int(crs['color'][3:5],16)},{int(crs['color'][5:7],16)},0.03));
                    border: 1px solid {crs['color']}40; border-radius: 16px;">
            <div style="font-family: 'Orbitron', monospace; font-size: 0.75rem;
                        color: #7a8ba0; letter-spacing: 3px; margin-bottom: 0.5rem;">CRITICAL RISK SCORE</div>
            <div style="font-family: 'Orbitron', monospace; font-size: 4rem; font-weight: 900;
                        color: {crs['color']};">{crs['score']}</div>
            <div style="font-family: 'Orbitron', monospace; font-size: 1rem; font-weight: 700;
                        color: {crs['color']}; letter-spacing: 3px; margin-top: 0.3rem;">
                {crs['level']}
            </div>
            <div style="color: #4a5568; font-size: 0.75rem; margin-top: 0.5rem;">
                Threshold: 70 | Trigger: {'YES' if trigger['triggered'] else 'NO'}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with breakdown_col:
        st.markdown("""
        <div class="section-header" style="font-size: 0.95rem;">
            📊 CRS FORMULA BREAKDOWN
        </div>
        """, unsafe_allow_html=True)

        formula_items = [
            ('(100 - Health) × 0.4', crs['components']['health'], 40, '#ff3366'),
            ('Predicted Risk × 0.3', crs['components']['predicted_risk'], 30, '#ff6d00'),
            ('Aging Risk × 0.2', crs['components']['aging_risk'], 20, '#ffab00'),
            ('Impact Severity × 0.1', crs['components']['impact_severity'], 10, '#7c4dff'),
        ]

        for label, value, max_val, color in formula_items:
            pct = (value / max_val * 100) if max_val > 0 else 0
            st.markdown(f"""
            <div style="margin: 0.5rem 0; padding: 0.6rem 1rem; background: rgba(255,255,255,0.015);
                        border-radius: 8px; border-left: 3px solid {color};">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="color: #7a8ba0; font-size: 0.82rem;">{label}</span>
                    <span style="font-family: 'Orbitron', monospace; color: {color};
                                font-weight: 700;">{value}/{max_val}</span>
                </div>
                <div class="prob-bar-container" style="height: 5px; margin-top: 0.3rem;">
                    <div class="prob-bar-fill" style="width: {pct}%; background: {color};
                         animation: none;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="text-align: right; margin-top: 0.5rem; font-family: 'Orbitron', monospace;
                    color: {crs['color']}; font-size: 0.9rem; font-weight: 700;">
            TOTAL CRS = {crs['score']} / 100
        </div>
        """, unsafe_allow_html=True)

    # ── Emergency Alert Banner ───────────────────────────────────
    if trigger['triggered']:
        st.markdown(f"""
        <div class="alert-card alert-critical" style="text-align: center; padding: 1.2rem;
                    margin: 1rem 0; animation: badgePulse 2s ease-in-out infinite;">
            <div style="font-family: 'Orbitron', monospace; font-size: 1rem; font-weight: 800;
                        color: #ff1744; letter-spacing: 2px; margin-bottom: 0.3rem;">
                \U0001f6a8 IMMEDIATE GOVERNMENT ATTENTION REQUIRED
            </div>
            <div style="color: #e8edf5; font-size: 0.88rem; margin-bottom: 0.5rem;">
                {trigger['reason_count']} trigger condition(s) met — Severity: {trigger['severity']}
            </div>
            <div style="color: #7a8ba0; font-size: 0.78rem;">
                {'  |  '.join(trigger['reasons'])}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="alert-card alert-success" style="text-align: center; padding: 1rem; margin: 1rem 0;">
            <div class="alert-title" style="color: #00e676;">\u2705 NO ALERT TRIGGERED</div>
            <div class="alert-message">All parameters within safe limits. CRS below government complaint threshold.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Infrastructure Map ───────────────────────────────────────
    st.markdown("""
    <div class="section-header">
        \U0001f5fa\ufe0f INFRASTRUCTURE LOCATION MAP
    </div>
    """, unsafe_allow_html=True)

    map_structures = govt_agent.get_all_structures_for_map()

    # Use Plotly scatter_geo for map
    map_lats = [s['lat'] for s in map_structures]
    map_lons = [s['lon'] for s in map_structures]
    map_names = [f"{s['name']} ({s['id']})" for s in map_structures]
    map_scores = [s['score'] for s in map_structures]
    map_colors = [s['color'] for s in map_structures]
    map_status = [s['status'] for s in map_structures]
    map_cities = [s['city'] for s in map_structures]

    fig_map = go.Figure()

    # Add markers by status group for legend
    for status_type, color in [('Critical', '#ff1744'), ('Moderate', '#ffab00'), ('Safe', '#00e676')]:
        indices = [i for i, s in enumerate(map_structures) if s['status'] == status_type]
        if not indices:
            continue
        fig_map.add_trace(go.Scattergeo(
            lat=[map_lats[i] for i in indices],
            lon=[map_lons[i] for i in indices],
            text=[f"<b>{map_names[i]}</b><br>{map_cities[i]}<br>Health: {map_scores[i]}%<br>Status: {map_status[i]}" for i in indices],
            marker=dict(
                size=[14 if map_status[i] == 'Critical' else 10 for i in indices],
                color=color,
                line=dict(width=1, color='white'),
                symbol='circle',
            ),
            name=status_type,
            hoverinfo='text',
        ))

    fig_map.update_layout(
        geo=dict(
            scope='asia',
            center=dict(lat=22.0, lon=79.0),
            projection_scale=4.5,
            bgcolor='rgba(0,0,0,0)',
            showland=True, landcolor='rgba(15,22,35,0.9)',
            showocean=True, oceancolor='rgba(6,10,19,0.9)',
            showlakes=True, lakecolor='rgba(6,10,19,0.5)',
            showcountries=True, countrycolor='rgba(0,212,255,0.15)',
            showsubunits=True, subunitcolor='rgba(0,212,255,0.08)',
            showframe=False,
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#7a8ba0', family='Inter'),
        height=450,
        margin=dict(l=0, r=0, t=10, b=10),
        legend=dict(
            font=dict(color='#7a8ba0', size=10),
            bgcolor='rgba(15,22,35,0.8)',
            bordercolor='rgba(0,212,255,0.15)',
            borderwidth=1,
            yanchor='top', y=0.95, xanchor='left', x=0.02,
        ),
    )

    st.plotly_chart(fig_map, use_container_width=True)

    # Location details for selected structure
    sel_loc = govt_agent.LOCATION_DB[g_structure_id]
    loc_c1, loc_c2, loc_c3 = st.columns(3)

    with loc_c1:
        st.markdown(f"""
        <div class="glass-card" style="padding: 1rem;">
            <div style="font-family: 'Orbitron', monospace; font-size: 0.72rem;
                        color: #00d4ff; letter-spacing: 1.5px; margin-bottom: 0.6rem;">LOCATION</div>
            <div class="comparison-stat">
                <span class="comparison-stat-label">Name</span>
                <span class="comparison-stat-value" style="color: #e8edf5; font-size: 0.8rem;">{sel_loc['name']}</span>
            </div>
            <div class="comparison-stat">
                <span class="comparison-stat-label">Type</span>
                <span class="comparison-stat-value" style="color: #7c4dff;">{sel_loc['type']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with loc_c2:
        st.markdown(f"""
        <div class="glass-card" style="padding: 1rem;">
            <div style="font-family: 'Orbitron', monospace; font-size: 0.72rem;
                        color: #00d4ff; letter-spacing: 1.5px; margin-bottom: 0.6rem;">COORDINATES</div>
            <div class="comparison-stat">
                <span class="comparison-stat-label">GPS</span>
                <span class="comparison-stat-value" style="color: #00d4ff; font-size: 0.8rem;">
                    {sel_loc['lat']:.4f}°N, {sel_loc['lon']:.4f}°E
                </span>
            </div>
            <div class="comparison-stat">
                <span class="comparison-stat-label">Landmark</span>
                <span class="comparison-stat-value" style="color: #7a8ba0; font-size: 0.75rem;">{sel_loc['landmark']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with loc_c3:
        st.markdown(f"""
        <div class="glass-card" style="padding: 1rem;">
            <div style="font-family: 'Orbitron', monospace; font-size: 0.72rem;
                        color: #00d4ff; letter-spacing: 1.5px; margin-bottom: 0.6rem;">JURISDICTION</div>
            <div class="comparison-stat">
                <span class="comparison-stat-label">City</span>
                <span class="comparison-stat-value" style="color: #e8edf5; font-size: 0.8rem;">{sel_loc['city']}</span>
            </div>
            <div class="comparison-stat">
                <span class="comparison-stat-label">Ward/Zone</span>
                <span class="comparison-stat-value" style="color: #7a8ba0;">{sel_loc.get('ward', 'N/A')}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Generate Complaint ──────────────────────────────────────
    gen_complaint = trigger['triggered'] or st.button("📨 GENERATE GOVERNMENT COMPLAINT", use_container_width=True)

    if gen_complaint:
        complaint = govt_agent.generate_complaint(
            structure_id=g_structure_id,
            health_score=g_health,
            remaining_life=g_remaining_life,
            impact_radius=g_impact_radius,
            affected_population=int(g_impact_radius * 15),
            crs=crs,
            trigger_info=trigger,
        )

        st.markdown("---")

        # ── Complaint Header ───────────────────────────────────────
        st.markdown(f"""
        <div style="text-align: center; padding: 1.5rem;
                    background: linear-gradient(135deg, rgba(255,23,68,0.08), rgba(255,23,68,0.02));
                    border: 1px solid rgba(255,23,68,0.2); border-radius: 16px; margin-bottom: 1rem;">
            <div style="font-family: 'Orbitron', monospace; font-size: 0.7rem;
                        color: #7a8ba0; letter-spacing: 3px;">GOVERNMENT OF INDIA • INFRASTRUCTURE SAFETY DIVISION</div>
            <div style="font-family: 'Orbitron', monospace; font-size: 1.2rem; font-weight: 800;
                        color: #ff3366; letter-spacing: 2px; margin: 0.5rem 0;">EMERGENCY INSPECTION ORDER</div>
            <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 0.5rem;">
                <div>
                    <div style="color: #4a5568; font-size: 0.68rem;">TICKET ID</div>
                    <div style="font-family: 'JetBrains Mono', monospace; color: #00d4ff;
                                font-weight: 600;">{complaint['ticket_id']}</div>
                </div>
                <div>
                    <div style="color: #4a5568; font-size: 0.68rem;">PRIORITY</div>
                    <div style="font-family: 'Orbitron', monospace; color: {complaint['priority_color']};
                                font-weight: 700; font-size: 0.85rem;">{complaint['priority']}</div>
                </div>
                <div>
                    <div style="color: #4a5568; font-size: 0.68rem;">STATUS</div>
                    <div style="color: #ff1744; font-weight: 600;">{complaint['status']}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Full Government Report ─────────────────────────────────
        st.markdown("""
        <div class="section-header">
            📋 OFFICIAL GOVERNMENT INSPECTION REPORT
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="analysis-container">
            <div class="analysis-container-inner">
        """, unsafe_allow_html=True)

        report_rows = [
            ('Ticket ID', complaint['ticket_id'], '#00d4ff'),
            ('Date', complaint['generated_at'], '#e8edf5'),
            ('Infrastructure Name', complaint['structure_name'], '#e8edf5'),
            ('Structure ID', complaint['structure_id'], '#7c4dff'),
            ('Type', complaint['structure_type'], '#7a8ba0'),
            ('City / District', complaint['city'], '#e8edf5'),
            ('GPS Coordinates', complaint['gps_string'], '#00d4ff'),
            ('Nearest Landmark', complaint['landmark'], '#7a8ba0'),
            ('Ward / Zone', complaint['ward'], '#7a8ba0'),
            ('Health Score', f"{complaint['health_score']}%", '#ff3366' if complaint['health_score'] < 50 else '#00e676'),
            ('Critical Risk Score', f"{complaint['crs_score']} ({complaint['crs_level']})", complaint['crs_color']),
            ('Remaining Life', f"{complaint['remaining_life_years']} years",
             '#ff1744' if complaint['remaining_life_years'] < 3 else '#ffab00'),
            ('Impact Radius', f"{complaint['impact_radius_m']}m", '#ff6d00'),
            ('Affected Population', f"{complaint['affected_population']:,}", '#ff6d00'),
            ('Priority', complaint['priority'], complaint['priority_color']),
        ]

        for label, value, color in report_rows:
            st.markdown(f"""
            <div class="summary-row">
                <span class="summary-label">{label}</span>
                <span class="summary-value" style="color: {color};">{value}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div></div>", unsafe_allow_html=True)

        # Recommended Action
        st.markdown(f"""
        <div class="glass-card" style="border-left: 4px solid {complaint['priority_color']}; margin-top: 1rem;">
            <div style="font-family: 'Orbitron', monospace; font-size: 0.8rem;
                        color: {complaint['priority_color']}; margin-bottom: 0.5rem; letter-spacing: 1.5px;">
                🎯 RECOMMENDED IMMEDIATE ACTION
            </div>
            <div style="color: #e8edf5; font-size: 0.92rem; line-height: 1.7;">
                {complaint['recommended_action']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Trigger conditions
        if trigger['triggered']:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div class="section-header" style="font-size: 1rem;">
                ⚡ TRIGGER CONDITIONS MET
            </div>
            """, unsafe_allow_html=True)

            for reason in trigger['reasons']:
                st.markdown(f"""
                <div class="alert-card alert-critical" style="padding: 0.7rem 1rem; margin: 0.3rem 0;">
                    <div class="alert-message" style="font-size: 0.85rem;">\u2713 {reason}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Actions Row ────────────────────────────────────────────
        st.markdown("""
        <div class="section-header" style="font-size: 1rem;">
            🛠️ COMPLAINT ACTIONS
        </div>
        """, unsafe_allow_html=True)

        act_c1, act_c2, act_c3 = st.columns(3)

        with act_c1:
            officer_name = st.text_input("Assign Field Officer", "Inspector R. Sharma", key='officer')
            if officer_name:
                st.markdown(f"""
                <div style="color: #00e676; font-size: 0.8rem; margin-top: 0.3rem;">
                    \u2713 Assigned to: {officer_name}
                </div>
                """, unsafe_allow_html=True)

        with act_c2:
            status_sel = st.selectbox("Update Status", govt_agent.STATUS_OPTIONS, index=0, key='status_sel')
            st.markdown(f"""
            <div style="color: #00d4ff; font-size: 0.8rem; margin-top: 0.3rem;">
                Current: {status_sel}
            </div>
            """, unsafe_allow_html=True)

        with act_c3:
            st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
            if st.checkbox("Mark as Inspection Scheduled", key='insp_sched'):
                st.markdown("""
                <div style="color: #00e676; font-size: 0.8rem;">
                    \u2713 Inspection Scheduled
                </div>
                """, unsafe_allow_html=True)

        # ── PDF Download ──────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        pdf_text = govt_agent.generate_pdf_text(complaint)

        st.download_button(
            label="\U0001f4e5 DOWNLOAD GOVERNMENT REPORT (TXT)",
            data=pdf_text,
            file_name=f"SafeSpan_Govt_Report_{complaint['ticket_id']}.txt",
            mime="text/plain",
            use_container_width=True,
        )

        # Preview
        with st.expander("👁️ Preview Report Content"):
            st.code(pdf_text, language='text')


# ─── Footer ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <div style="font-family: 'Orbitron', monospace; font-size: 0.75rem; color: #00d4ff;
                letter-spacing: 3px; margin-bottom: 0.3rem;">
        SAFESPAN AI
    </div>
    <div style="color: #4a5568;">Government-Ready Infrastructure Emergency Monitoring & Reporting Platform</div>
    <div style="margin-top: 0.3rem; color: #4a5568;">
        Built with TensorFlow • Streamlit • Plotly | © 2026 SafeSpan AI
    </div>
</div>
""", unsafe_allow_html=True)
