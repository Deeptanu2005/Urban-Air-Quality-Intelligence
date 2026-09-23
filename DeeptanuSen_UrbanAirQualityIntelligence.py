"""
Urban Air Quality Intelligence
================================
A premium dark-mode Streamlit dashboard for Indian urban air-quality analysis.

Dataset: city_day.csv  (29,531 rows × 16 columns)
  Cities : 26 Indian cities  |  Date range : 2015-01-01 → 2020-07-01
  Columns: City, Date, PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3,
           Benzene, Toluene, Xylene, AQI, AQI_Bucket
"""

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import re
import io
import urllib.request
import pandas as pd
import numpy as np

import os

# Force Streamlit to use Dark theme
os.environ["STREAMLIT_THEME_BASE"] = "dark"

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from scipy.stats import mannwhitneyu

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────



st.set_page_config(
    page_title="Urban Air Quality Intelligence",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
# Design system — CSS
# ─────────────────────────────────────────────────────────────────────────────



DARK_CSS = """
<style>

/* ══════════════════════════════════════════════════════
   URBAN AIR QUALITY INTELLIGENCE
   Modern Dark Intelligence Dashboard
══════════════════════════════════════════════════════ */

:root {
    /* ───────── Core palette ───────── */
    --bg-base:              #070b12;
    --bg-deep:              #05080d;
    --bg-surface:           #0b111b;
    --bg-elevated:          #101925;
    --bg-card:              rgba(16, 25, 37, 0.82);
    --bg-glass:             rgba(13, 20, 31, 0.72);

    /* ───────── Borders ───────── */
    --border:               rgba(148, 163, 184, 0.12);
    --border-strong:        rgba(148, 163, 184, 0.20);
    --border-accent:        rgba(45, 212, 191, 0.38);

    /* ───────── Brand ───────── */
    --accent-primary:       #2dd4bf;
    --accent-primary-soft:  rgba(45, 212, 191, 0.12);
    --accent-blue:          #38bdf8;
    --accent-blue-soft:     rgba(56, 189, 248, 0.12);
    --accent-purple:        #a78bfa;
    --accent-purple-soft:   rgba(167, 139, 250, 0.12);

    /* ───────── Status ───────── */
    --accent-green:         #4ade80;
    --accent-yellow:        #facc15;
    --accent-orange:        #fb923c;
    --accent-red:           #f87171;
    --accent-pink:          #f472b6;

    /* ───────── Text ───────── */
    --text-primary:         #f1f5f9;
    --text-secondary:       #94a3b8;
    --text-muted:           #64748b;
    --text-faint:            #475569;

    /* ───────── Geometry ───────── */
    --radius-xs:            5px;
    --radius-sm:            8px;
    --radius-md:            12px;
    --radius-lg:            18px;
    --radius-xl:            24px;

    /* ───────── Shadows ───────── */
    --shadow-sm:            0 4px 14px rgba(0, 0, 0, 0.22);
    --shadow-md:            0 10px 30px rgba(0, 0, 0, 0.28);
    --shadow-lg:            0 20px 50px rgba(0, 0, 0, 0.38);

    --glow-teal:            0 0 28px rgba(45, 212, 191, 0.10);
    --glow-blue:            0 0 28px rgba(56, 189, 248, 0.10);

    --transition-fast:      150ms ease;
    --transition:           220ms ease;
}


/* ══════════════════════════════════════════════════════
   GLOBAL FOUNDATION
══════════════════════════════════════════════════════ */

html,
body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main {
    background:
        radial-gradient(
            circle at 85% 5%,
            rgba(45, 212, 191, 0.045),
            transparent 28%
        ),
        radial-gradient(
            circle at 15% 90%,
            rgba(56, 189, 248, 0.035),
            transparent 30%
        ),
        var(--bg-base) !important;

    color: var(--text-primary);
    font-family:
        Inter,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        system-ui,
        sans-serif;
}


/* Subtle dashboard texture */
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;

    background-image:
        linear-gradient(
            rgba(255,255,255,0.012) 1px,
            transparent 1px
        ),
        linear-gradient(
            90deg,
            rgba(255,255,255,0.012) 1px,
            transparent 1px
        );

    background-size: 48px 48px;
    mask-image: linear-gradient(
        to bottom,
        black,
        transparent 85%
    );
}


/* Main content */
[data-testid="stMainBlockContainer"] {
    padding-top: 0.7rem !important;
    padding-bottom: 3rem !important;
}

.block-container {
    padding-top: 4.2rem !important;
    padding-bottom: 3rem !important;
}

section[data-testid="stMain"] .main .block-container {
    padding-top: 0.5rem !important;
}


/* ══════════════════════════════════════════════════════
   SIDEBAR
══════════════════════════════════════════════════════ */

[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #0a111b 0%,
            #070d15 55%,
            #060a11 100%
        ) !important;

    border-right: 1px solid var(--border) !important;
    box-shadow: 10px 0 35px rgba(0,0,0,0.18);
}

[data-testid="stSidebar"] .block-container {
    padding-top: 0.7rem !important;
}

[data-testid="stSidebarNav"] {
    display: none;
}


/* Sidebar scrollbar */
[data-testid="stSidebar"]::-webkit-scrollbar {
    width: 5px;
}

[data-testid="stSidebar"]::-webkit-scrollbar-thumb {
    background: rgba(148,163,184,0.15);
    border-radius: 10px;
}


/* Sidebar section label */
.sidebar-section-label {
    font-size: 0.66rem;
    font-weight: 800;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-muted);

    margin: 1.25rem 0 0.55rem 0;
    padding-left: 3px;
}


/* Sidebar stat */
.sidebar-stat {
    display: flex;
    justify-content: space-between;
    align-items: center;

    background: rgba(16,25,37,0.65);
    border: 1px solid var(--border);

    border-radius: var(--radius-sm);

    padding: 8px 11px;
    margin-bottom: 6px;

    font-size: 0.80rem;

    transition:
        border-color var(--transition),
        background var(--transition);
}

.sidebar-stat:hover {
    background: rgba(45,212,191,0.055);
    border-color: var(--border-accent);
}

.sidebar-stat .label {
    color: var(--text-secondary);
}

.sidebar-stat .value {
    color: var(--accent-primary);
    font-weight: 700;
}


/* ══════════════════════════════════════════════════════
   HERO
══════════════════════════════════════════════════════ */

.hero-banner {
    position: relative;
    overflow: hidden;

    background:
        radial-gradient(
            circle at 90% 20%,
            rgba(45,212,191,0.12),
            transparent 32%
        ),
        radial-gradient(
            circle at 65% 100%,
            rgba(56,189,248,0.07),
            transparent 35%
        ),
        linear-gradient(
            135deg,
            #0c1420 0%,
            #0b1520 50%,
            #09121c 100%
        );

    border: 1px solid var(--border-strong);
    border-left: 3px solid var(--accent-primary);

    border-radius: var(--radius-xl);

    padding: 30px 38px 26px;
    margin-bottom: 26px;

    box-shadow: var(--shadow-lg), var(--glow-teal);
}


/* Hero glow */
.hero-banner::before {
    content: "";

    position: absolute;
    top: -120px;
    right: -100px;

    width: 320px;
    height: 320px;

    background:
        radial-gradient(
            circle,
            rgba(45,212,191,0.12),
            transparent 68%
        );

    pointer-events: none;
}


/* Hero grid lines */
.hero-banner::after {
    content: "";

    position: absolute;
    inset: 0;

    background-image:
        linear-gradient(
            rgba(255,255,255,0.018) 1px,
            transparent 1px
        ),
        linear-gradient(
            90deg,
            rgba(255,255,255,0.018) 1px,
            transparent 1px
        );

    background-size: 34px 34px;

    mask-image: linear-gradient(
        120deg,
        transparent 15%,
        black 75%
    );

    pointer-events: none;
}


.hero-title {
    position: relative;
    z-index: 1;

    font-size: 2.05rem;
    font-weight: 850;

    color: var(--text-primary);

    letter-spacing: -0.035em;

    margin: 0 0 6px 0;

    line-height: 1.12;
}

.hero-title span {
    color: var(--accent-primary);

    text-shadow:
        0 0 24px rgba(45,212,191,0.20);
}


.hero-subtitle {
    position: relative;
    z-index: 1;

    font-size: 0.94rem;
    color: var(--text-secondary);

    margin: 0 0 17px 0;

    max-width: 680px;

    line-height: 1.55;
}


.hero-meta {
    position: relative;
    z-index: 1;

    display: flex;
    flex-wrap: wrap;
    gap: 7px;
}


.hero-badge {
    background: rgba(45,212,191,0.075);

    border: 1px solid rgba(45,212,191,0.20);

    border-radius: 999px;

    padding: 4px 12px;

    font-size: 0.74rem;

    color: #7ee7d8;

    font-weight: 600;

    backdrop-filter: blur(8px);

    transition:
        background var(--transition),
        border-color var(--transition);
}

.hero-badge:hover {
    background: rgba(45,212,191,0.13);
    border-color: rgba(45,212,191,0.38);
}


/* ══════════════════════════════════════════════════════
   KPI CARDS
══════════════════════════════════════════════════════ */

.kpi-grid {
    display: grid;

    grid-template-columns:
        repeat(3, minmax(0, 1fr));

    gap: 15px;

    margin-bottom: 22px;
}


.kpi-card {
    position: relative;
    overflow: hidden;

    background:
        linear-gradient(
            145deg,
            rgba(18,28,42,0.94),
            rgba(11,18,28,0.94)
        );

    border: 1px solid var(--border);

    border-radius: var(--radius-lg);

    padding: 21px 23px 18px;

    box-shadow: var(--shadow-sm);

    transition:
        transform var(--transition),
        border-color var(--transition),
        box-shadow var(--transition);
}

.kpi-card:hover {
    transform: translateY(-2px);

    border-color: var(--border-accent);

    box-shadow:
        var(--shadow-md),
        var(--glow-teal);
}


/* top shine */
.kpi-card::before {
    content: "";

    position: absolute;

    top: 0;
    left: 0;
    right: 0;

    height: 1px;

    background:
        linear-gradient(
            90deg,
            transparent,
            rgba(255,255,255,0.18),
            transparent
        );
}


/* bottom accent */
.kpi-card::after {
    content: "";

    position: absolute;

    bottom: 0;
    left: 7%;
    right: 7%;

    height: 2px;

    border-radius: 999px;

    background:
        linear-gradient(
            90deg,
            transparent,
            var(--kpi-accent, var(--accent-primary)),
            transparent
        );

    opacity: 0.75;
}


.kpi-label {
    font-size: 0.68rem;

    font-weight: 800;

    letter-spacing: 0.12em;

    text-transform: uppercase;

    color: var(--text-muted);

    margin-bottom: 9px;
}


.kpi-value {
    font-size: 2.25rem;

    font-weight: 850;

    color: var(--text-primary);

    line-height: 1;

    margin-bottom: 7px;

    letter-spacing: -0.035em;
}


.kpi-badge {
    display: inline-flex;

    align-items: center;

    padding: 3px 10px;

    border-radius: 999px;

    font-size: 0.70rem;

    font-weight: 700;

    margin-bottom: 8px;
}


.kpi-sub {
    font-size: 0.77rem;

    color: var(--text-secondary);

    line-height: 1.45;

    margin-top: 4px;
}


/* AQI badges */
.badge-good {
    background: rgba(74,222,128,0.10);
    color: #6ee7a0;
    border: 1px solid rgba(74,222,128,0.25);
}

.badge-satisfactory {
    background: rgba(45,212,191,0.10);
    color: #67e8d8;
    border: 1px solid rgba(45,212,191,0.25);
}

.badge-moderate {
    background: rgba(250,204,21,0.10);
    color: #fde047;
    border: 1px solid rgba(250,204,21,0.25);
}

.badge-poor {
    background: rgba(251,146,60,0.11);
    color: #fdba74;
    border: 1px solid rgba(251,146,60,0.27);
}

.badge-very-poor {
    background: rgba(248,113,113,0.11);
    color: #fca5a5;
    border: 1px solid rgba(248,113,113,0.27);
}

.badge-severe {
    background: rgba(167,139,250,0.12);
    color: #c4b5fd;
    border: 1px solid rgba(167,139,250,0.28);
}


/* Risk pills */
.risk-low {
    background: rgba(74,222,128,0.10);
    color: #6ee7a0;
    border: 1px solid rgba(74,222,128,0.25);
}

.risk-medium {
    background: rgba(250,204,21,0.10);
    color: #fde047;
    border: 1px solid rgba(250,204,21,0.25);
}

.risk-high {
    background: rgba(251,146,60,0.11);
    color: #fdba74;
    border: 1px solid rgba(251,146,60,0.27);
}

.risk-severe {
    background: rgba(248,113,113,0.11);
    color: #fca5a5;
    border: 1px solid rgba(248,113,113,0.27);
}


/* ══════════════════════════════════════════════════════
   SUPPORTING STATS
══════════════════════════════════════════════════════ */

.kpi-stat-strip {
    display: grid;

    grid-template-columns:
        repeat(4, minmax(0, 1fr));

    gap: 10px;

    margin-bottom: 28px;
}


.stat-chip {
    position: relative;

    background: rgba(13,20,31,0.72);

    border: 1px solid var(--border);

    border-radius: var(--radius-md);

    padding: 11px 14px;

    text-align: center;

    transition:
        background var(--transition),
        border-color var(--transition);
}

.stat-chip:hover {
    background: rgba(45,212,191,0.045);
    border-color: rgba(45,212,191,0.20);
}

.stat-chip .n {
    font-size: 1.2rem;

    font-weight: 800;

    color: var(--text-primary);
}

.stat-chip .lbl {
    font-size: 0.65rem;

    color: var(--text-muted);

    margin-top: 3px;

    letter-spacing: 0.08em;

    text-transform: uppercase;
}


/* ══════════════════════════════════════════════════════
   SECTION HEADERS
══════════════════════════════════════════════════════ */

.section-header {
    display: flex;

    align-items: center;

    gap: 11px;

    margin: 10px 0 17px;

    padding-bottom: 11px;

    border-bottom: 1px solid var(--border);
}


.section-icon {
    width: 34px;
    height: 34px;

    flex-shrink: 0;

    background:
        linear-gradient(
            135deg,
            rgba(45,212,191,0.13),
            rgba(56,189,248,0.08)
        );

    border: 1px solid rgba(45,212,191,0.20);

    border-radius: var(--radius-sm);

    display: flex;

    align-items: center;
    justify-content: center;

    font-size: 0.95rem;

    box-shadow:
        inset 0 0 12px rgba(45,212,191,0.035);
}


.section-title {
    font-size: 1.04rem;

    font-weight: 750;

    color: var(--text-primary);

    letter-spacing: -0.015em;
}


.section-desc {
    font-size: 0.74rem;

    color: var(--text-muted);

    margin-left: auto;
}


/* ══════════════════════════════════════════════════════
   INSIGHT CARDS
══════════════════════════════════════════════════════ */

.insight-card {
    position: relative;

    background:
        linear-gradient(
            135deg,
            rgba(16,25,37,0.92),
            rgba(11,18,28,0.92)
        );

    border: 1px solid var(--border);

    border-left: 3px solid var(--accent-blue);

    border-radius: var(--radius-md);

    padding: 14px 17px;

    margin-bottom: 10px;

    font-size: 0.84rem;

    color: var(--text-primary);

    line-height: 1.55;

    box-shadow: var(--shadow-sm);

    transition:
        transform var(--transition),
        border-color var(--transition);
}

.insight-card:hover {
    transform: translateX(2px);

    border-color: rgba(56,189,248,0.22);
    border-left-color: var(--accent-blue);
}

.insight-card.warn {
    border-left-color: var(--accent-orange);

    background:
        linear-gradient(
            135deg,
            rgba(251,146,60,0.065),
            rgba(16,25,37,0.92)
        );
}

.insight-card strong {
    color: var(--accent-blue);
}

.insight-card.warn strong {
    color: var(--accent-orange);
}


/* ══════════════════════════════════════════════════════
   MODERN INTELLIGENCE TABS
══════════════════════════════════════════════════════ */

[data-testid="stTabs"] {
    background: transparent !important;
}

/* Tab navigation row */
[data-testid="stTabs"] > div:first-child {
    border-bottom: 1px solid rgba(70, 85, 105, 0.35) !important;
    padding: 0 0 0 0 !important;
    gap: 8px !important;
    overflow: visible !important;
}

/* Individual tabs */
[data-testid="stTab"] {
    position: relative !important;
    height: 64px !important;
    min-height: 64px !important;

    display: flex !important;
    align-items: center !important;
    justify-content: center !important;

    padding: 0 24px !important;
    margin: 0 !important;

    border: 1px solid transparent !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 16px 16px 0 0 !important;

    background: transparent !important;

    color: #718096 !important;

    font-size: 0.95rem !important;
    font-weight: 600 !important;

    transition:
        color 0.2s ease,
        background 0.2s ease,
        border-color 0.2s ease,
        transform 0.2s ease !important;

    cursor: pointer !important;
}

/* Text inside tabs */
[data-testid="stTab"] p {
    margin: 0 !important;
    color: inherit !important;
    font-size: inherit !important;
    font-weight: inherit !important;
}

/* Hover */
[data-testid="stTab"]:hover {
    color: #cbd5e1 !important;

    background: rgba(34, 211, 238, 0.045) !important;

    border-color: rgba(34, 211, 238, 0.10) !important;
}

/* Active tab */
[data-testid="stTab"][aria-selected="true"] {
    color: #22d3ee !important;

    background:
        linear-gradient(
            180deg,
            rgba(34, 211, 238, 0.10) 0%,
            rgba(34, 211, 238, 0.035) 75%,
            rgba(34, 211, 238, 0.00) 100%
        ) !important;

    border: 1px solid rgba(34, 211, 238, 0.22) !important;

    border-bottom: 2px solid #22d3ee !important;

    box-shadow:
        inset 0 1px 0 rgba(255,255,255,0.025),
        0 0 22px rgba(34,211,238,0.06) !important;
}

/* Active tab text */
[data-testid="stTab"][aria-selected="true"] p {
    color: #22d3ee !important;
}

/* Remove Streamlit's default selection indicator */
[data-testid="stTab"] .react-aria-SelectionIndicator {
    display: none !important;
}

/* Icons / emoji */
[data-testid="stTab"] p::first-letter {
    filter: saturate(1.15);
}

/* Tab content */
[data-testid="stTabsContent"] {
    padding-top: 24px !important;
    background: transparent !important;
}


/* ══════════════════════════════════════════════════════
   STREAMLIT INPUT CONTROLS
══════════════════════════════════════════════════════ */

/* Selectbox */
[data-testid="stSelectbox"] > div,
[data-testid="stMultiSelect"] > div {
    background: var(--bg-elevated) !important;

    border: 1px solid var(--border) !important;

    border-radius: var(--radius-sm) !important;

    transition:
        border-color var(--transition),
        box-shadow var(--transition);
}


[data-testid="stSelectbox"] > div:hover,
[data-testid="stMultiSelect"] > div:hover {
    border-color: rgba(45,212,191,0.32) !important;

    box-shadow:
        0 0 0 2px rgba(45,212,191,0.035);
}


/* Text inside controls */
[data-baseweb="select"] {
    color: var(--text-primary) !important;
}


[data-baseweb="select"] input {
    color: var(--text-primary) !important;
}


/* Selected tags */
[data-baseweb="tag"] {
    background:
        rgba(45,212,191,0.10) !important;

    border:
        1px solid rgba(45,212,191,0.22) !important;

    color:
        #8ff1e3 !important;
}


/* Dropdown menus */
[data-baseweb="popover"] {
    background: #0e1722 !important;

    border: 1px solid var(--border-strong) !important;

    box-shadow: var(--shadow-lg) !important;

    border-radius: var(--radius-md) !important;
}


[role="option"] {
    background: transparent !important;

    color: var(--text-secondary) !important;
}


[role="option"]:hover,
[aria-selected="true"] {
    background:
        rgba(45,212,191,0.09) !important;

    color: var(--text-primary) !important;
}


/* ══════════════════════════════════════════════════════
   SLIDER
══════════════════════════════════════════════════════ */

[data-testid="stSlider"] {
    color: var(--accent-primary) !important;
}


[data-testid="stSlider"] [role="slider"] {
    background: var(--accent-primary) !important;

    border-color: var(--accent-primary) !important;

    box-shadow:
        0 0 0 4px rgba(45,212,191,0.08);
}


[data-testid="stSlider"] [data-testid="stThumbValue"] {
    color: var(--accent-primary) !important;

    font-weight: 700 !important;
}


/* ══════════════════════════════════════════════════════
   EXPANDERS
══════════════════════════════════════════════════════ */

[data-testid="stExpander"] {
    background:
        linear-gradient(
            135deg,
            rgba(16,25,37,0.88),
            rgba(10,17,27,0.88)
        ) !important;

    border:
        1px solid var(--border) !important;

    border-radius:
        var(--radius-md) !important;

    overflow: hidden;

    transition:
        border-color var(--transition),
        box-shadow var(--transition);
}


[data-testid="stExpander"]:hover {
    border-color:
        rgba(45,212,191,0.22) !important;
}


[data-testid="stExpander"] summary {
    color: var(--text-primary) !important;

    font-weight: 650 !important;
}


/* ══════════════════════════════════════════════════════
   NATIVE METRICS
══════════════════════════════════════════════════════ */

[data-testid="stMetric"] {
    background:
        linear-gradient(
            145deg,
            rgba(16,25,37,0.90),
            rgba(11,18,28,0.90)
        ) !important;

    border:
        1px solid var(--border) !important;

    border-radius:
        var(--radius-md) !important;

    padding:
        15px 18px !important;

    box-shadow:
        var(--shadow-sm);
}


[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;

    font-size: 0.70rem !important;

    font-weight: 700 !important;

    letter-spacing: 0.08em !important;

    text-transform: uppercase !important;
}


[data-testid="stMetricValue"] {
    color: var(--text-primary) !important;

    font-size: 1.65rem !important;

    font-weight: 850 !important;
}


[data-testid="stMetricDelta"] {
    font-size: 0.78rem !important;
}


/* ══════════════════════════════════════════════════════
   RADIO BUTTONS
══════════════════════════════════════════════════════ */

[data-testid="stRadio"] label {
    color: var(--text-secondary) !important;

    font-size: 0.83rem !important;

    transition: color var(--transition);
}


[data-testid="stRadio"] label:hover {
    color: var(--text-primary) !important;
}


[data-testid="stRadio"] label:has(input:checked) {
    color: var(--accent-primary) !important;
}


/* ══════════════════════════════════════════════════════
   ALERTS / MESSAGES
══════════════════════════════════════════════════════ */

.stAlert {
    background:
        var(--bg-elevated) !important;

    border:
        1px solid var(--border) !important;

    border-radius:
        var(--radius-md) !important;
}


[data-testid="stInfoMessage"] {
    background:
        rgba(56,189,248,0.065) !important;

    border-color:
        rgba(56,189,248,0.22) !important;

    color:
        var(--text-secondary) !important;

    font-size:
        0.83rem !important;
}


[data-testid="stSuccessMessage"] {
    background:
        rgba(74,222,128,0.065) !important;

    border-color:
        rgba(74,222,128,0.22) !important;

    font-size:
        0.83rem !important;
}


/* ══════════════════════════════════════════════════════
   DATAFRAME
══════════════════════════════════════════════════════ */

[data-testid="stDataFrame"] {
    border-radius:
        var(--radius-md) !important;

    border:
        1px solid var(--border) !important;

    overflow: hidden !important;
}


/* ══════════════════════════════════════════════════════
   CAPTIONS
══════════════════════════════════════════════════════ */

[data-testid="stCaptionContainer"] {
    color:
        var(--text-muted) !important;

    font-size:
        0.76rem !important;
}


/* ══════════════════════════════════════════════════════
   DIVIDERS
══════════════════════════════════════════════════════ */

hr {
    border-color:
        var(--border) !important;

    margin:
        22px 0 !important;
}


/* ══════════════════════════════════════════════════════
   STREAMLIT MAIN MENU
   Keep useful controls, remove theme switcher
══════════════════════════════════════════════════════ */

# [data-testid="stThemeSwitcher"],
# [data-testid="stMainMenuDivider"] {
#     display: none !important;
# }


/* ══════════════════════════════════════════════════════
   SCROLLBARS
══════════════════════════════════════════════════════ */

::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}

::-webkit-scrollbar-track {
    background:
        var(--bg-deep);
}

::-webkit-scrollbar-thumb {
    background:
        rgba(148,163,184,0.16);

    border-radius:
        999px;
}

::-webkit-scrollbar-thumb:hover {
    background:
        rgba(45,212,191,0.32);
}


/* ══════════════════════════════════════════════════════
   RESPONSIVE
══════════════════════════════════════════════════════ */

@media (max-width: 900px) {

    .kpi-grid {
        grid-template-columns:
            1fr;
    }

    .kpi-stat-strip {
        grid-template-columns:
            repeat(2, 1fr);
    }

    .hero-banner {
        padding:
            24px 24px 21px;
    }

    .hero-title {
        font-size:
            1.65rem;
    }

}


@media (max-width: 600px) {

    .kpi-stat-strip {
        grid-template-columns:
            1fr 1fr;
    }

    .hero-banner {
        border-radius:
            var(--radius-lg);

        padding:
            21px 19px;
    }

    .hero-title {
        font-size:
            1.45rem;
    }

}


/* ══════════════════════════════════════════════════════
   ACCESSIBILITY / FOCUS
══════════════════════════════════════════════════════ */

*:focus-visible {
    outline:
        2px solid rgba(45,212,191,0.65) !important;

    outline-offset:
        2px !important;
}


[data-tag] {
    background: #123b52 !important;
    background-color: #123b52 !important;
    color: #d9f7ff !important;
    border: 1px solid #1f6f8b !important;
    border-radius: 7px !important;
}

[data-tag] svg {
    color: #58d6ff !important;
    fill: #58d6ff !important;
}

/* ══════════════════════════════════════════════════════
   YEAR RANGE — CYAN MODERN SLIDER
══════════════════════════════════════════════════════ */

/* Slider container */
[data-testid="stSlider"] [data-rac][data-orientation="horizontal"] {
    background: transparent !important;
}

/* Actual slider track */
[data-testid="stSlider"] .efbyxod5 {
    height: 5px !important;
    background: linear-gradient(
        90deg,
        #22d3ee 0%,
        #06b6d4 100%
    ) !important;
    border-radius: 999px !important;
    box-shadow: 0 0 10px rgba(34, 211, 238, 0.18) !important;
}

/* Thumb containers */
[data-testid="stSlider"] .efbyxod3 {
    z-index: 10 !important;
}

/* Thumb values above the slider */
[data-testid="stSliderThumbValue"] {
    color: #22d3ee !important;
    font-weight: 700 !important;
}

/* Tick labels below */
[data-testid="stSliderTickBar"] p {
    color: #718096 !important;
    font-weight: 500 !important;
}

/* Slider thumbs */
[data-testid="stSlider"] .efbyxod3::before {
    content: "";
    display: block;
    width: 20px;
    height: 20px;
    background: #22d3ee !important;
    border: 3px solid #071018 !important;
    border-radius: 50%;
    box-sizing: border-box;
    box-shadow:
        0 0 0 2px rgba(34, 211, 238, 0.25),
        0 0 12px rgba(34, 211, 238, 0.35);
}

/* Thumb hover */
[data-testid="stSlider"] .efbyxod3:hover::before {
    background: #67e8f9 !important;
    box-shadow:
        0 0 0 3px rgba(34, 211, 238, 0.25),
        0 0 18px rgba(34, 211, 238, 0.55);
}

/* Date Range metric */
[data-testid="stMetric"] {
    container-type: inline-size;
}

/* Keep the value inside its viewport */
[data-testid="stMetricValue"] {
    overflow: hidden !important;
    white-space: nowrap !important;
}

/* The actual text */
[data-testid="stMetricValue"] p {
    display: inline-block !important;
    width: max-content !important;
    white-space: nowrap !important;
    transform: translateX(0);
}

/* Hover → move exactly until the tail reaches the right edge */
[data-testid="stMetric"]:hover
[data-testid="stMetricValue"] p {
    animation: date-marquee 3s linear infinite alternate !important;
}

@keyframes date-marquee {
    from {
        transform: translateX(0);
    }

    to {
        transform: translateX(calc(100cqw - 100%));
    }
}



/* ══════════════════════════════════════════════════════
   SELECTBOX — REMOVE CLICK/FOCUS TEAL BLOCK
══════════════════════════════════════════════════════ */

/* Outer selectbox */
[data-testid="stSelectbox"] [data-baseweb="select"] {
    background: #0d141c !important;
    border: 1px solid #263746 !important;
    border-radius: 16px !important;
    overflow: hidden !important;
    box-shadow: none !important;
}

/* EVERY direct child of BaseWeb select */
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    background: #0d141c !important;
}

/* Kill background on ALL nested elements */
[data-testid="stSelectbox"] [data-baseweb="select"] * {
    background-color: transparent !important;
}

/* Restore the select background itself */
[data-testid="stSelectbox"] [data-baseweb="select"] {
    background-color: #0d141c !important;
}

/* Focus / click */
[data-testid="stSelectbox"] [data-baseweb="select"]:focus-within {
    border-color: #22d3ee !important;
    outline: none !important;
    box-shadow:
        0 0 0 1px rgba(34, 211, 238, 0.35),
        0 0 20px rgba(34, 211, 238, 0.08) !important;
}

/* Text */
[data-testid="stSelectbox"] [data-baseweb="select"] span {
    color: #e6edf3 !important;
}

/* Arrow */
[data-testid="stSelectbox"] [data-baseweb="select"] svg {
    fill: #94a3b8 !important;
    color: #94a3b8 !important;
    background: transparent !important;
}

/* Arrow hover/focus */
[data-testid="stSelectbox"] [data-baseweb="select"] svg:hover {
    fill: #22d3ee !important;
}

/* Remove browser/native focus visuals */
[data-testid="stSelectbox"] input,
[data-testid="stSelectbox"] input:focus {
    outline: none !important;
    box-shadow: none !important;
    background: transparent !important;
}

/* Keep dropdown arrow area dark */
[data-testid="stSelectbox"] [role="button"],
[data-testid="stSelectbox"] [role="combobox"] {
    background-color: #0d141c !important;
}

/* Dropdown itself */
[data-baseweb="popover"] {
    background: #0d141c !important;
    border: 1px solid #263746 !important;
    border-radius: 14px !important;
}

[data-baseweb="menu"] {
    background: #0d141c !important;
}

[data-baseweb="menu"] li {
    background: transparent !important;
    color: #cbd5e1 !important;
}

[data-baseweb="menu"] li:hover {
    background: rgba(34, 211, 238, 0.08) !important;
    color: #22d3ee !important;
}
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH = Path("city_day.csv")

POLLUTANTS = [
    "PM2.5", "PM10", "NO", "NO2", "NOx",
    "NH3", "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene",
]

# Reference upper-bound concentrations (µg/m³ or mg/m³ for CO) used only to
# normalise pollutants onto a common scale when computing Dominant Pollution
# Driver.  Values are approximate Indian NAAQS / WHO guideline maxima.
POLLUTANT_REF = {
    "PM2.5":   500,  # µg/m³  (India NAAQS 24h standard: 60)
    "PM10":    600,  # µg/m³
    "NO":      400,  # µg/m³
    "NO2":     400,  # µg/m³
    "NOx":     500,  # µg/m³
    "NH3":     400,  # µg/m³
    "CO":      180,  # mg/m³
    "SO2":     200,  # µg/m³
    "O3":      300,  # µg/m³
    "Benzene": 50,   # µg/m³
    "Toluene": 50,   # µg/m³
    "Xylene":  50,   # µg/m³
}

AQI_ORDER = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]

# AQI threshold that separates "normal" from "high-risk" for diagnostic comparisons.
# India AQI > 200 = 'Poor' — the first level flagged as broadly harmful to health.
AQI_RISK_THRESHOLD = 200

# Rolling-window (days) and z-score cutoff used for spike detection.
# 30-day window gives a local seasonal baseline; z > 2 ≈ top ~2.3 % of a normal dist.
SPIKE_WINDOW = 30
SPIKE_Z_THRESHOLD = 2.0

# AQI band boundaries for drilldown binning
AQI_BINS   = [0, 50, 100, 200, 300, 400, 999]
AQI_LABELS = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]

AQI_COLORS = {
    "Good":         "#2ea043",
    "Satisfactory": "#56d364",
    "Moderate":     "#e3b341",
    "Poor":         "#f0883e",
    "Very Poor":    "#da3633",
    "Severe":       "#8957e5",
}

SEASON_MAP = {
    12: "Winter", 1: "Winter",  2: "Winter",
    3:  "Spring", 4: "Spring",  5: "Spring",
    6:  "Summer", 7: "Summer",  8: "Summer",
    9:  "Autumn", 10: "Autumn", 11: "Autumn",
}

# Base layout — no xaxis/yaxis keys so callers can safely add their own
# without triggering "multiple values for keyword argument" TypeError.
PLOTLY_DARK_LAYOUT = dict(
    paper_bgcolor="#0e1117",
    plot_bgcolor="#161b22",
    font_color="#e6edf3",
    margin=dict(l=40, r=20, t=50, b=40),
)

# Default axis styling — merged explicitly where a chart does not override axes.
_DARK_AXIS = dict(gridcolor="#30363d", linecolor="#30363d")

# ─────────────────────────────────────────────────────────────────────────────
# Data loading & cleaning
# ─────────────────────────────────────────────────────────────────────────────
from pathlib import Path
import urllib.request

import pandas as pd
import streamlit as st


# ============================================================
# DATASET CONFIGURATION
# ============================================================

DATA_PATH = Path(__file__).parent / "city_day.csv"

_GDRIVE_FILE_ID = "1ADfDemnRyd1lutyx1u0srnDBfsj1dKJz"

_GDRIVE_DOWNLOAD_URL = (
    f"https://drive.usercontent.google.com/download"
    f"?id={_GDRIVE_FILE_ID}&export=download&confirm=t"
)


# ============================================================
# GOOGLE DRIVE DOWNLOAD
# ============================================================

# ============================================================
# GOOGLE DRIVE DOWNLOAD
# ============================================================

def _fetch_csv_from_gdrive(dest: Path) -> None:
    """
    Download city_day.csv from Google Drive.

    This function is called only when the local CSV is missing
    and the cached dataset is unavailable.
    """

    request = urllib.request.Request(
        _GDRIVE_DOWNLOAD_URL,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "*/*",
        },
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=30,
        ) as response:

            content = response.read()

    except Exception as exc:
        raise RuntimeError(
            f"Google Drive download failed: {exc}"
        ) from exc

    if not content:
        raise RuntimeError(
            "Google Drive returned an empty response."
        )

    # Check that we received the CSV rather than an HTML page.
    preview = content[:2000].lower()

    if (
        b"<html" in preview
        or b"<!doctype" in preview
        or b"<head" in preview
    ):
        raise RuntimeError(
            "Google Drive returned an HTML page instead of "
            "city_day.csv. Make sure the file is publicly accessible."
        )

    # Save locally.
    dest.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dest.write_bytes(content)



# ============================================================
# CACHED DATASET ACCESS
# ============================================================

@st.cache_data(
    show_spinner=False,
    persist=True,
)
def _get_dataset_bytes() -> bytes:
    """
    Download city_day.csv from Google Drive.

    Streamlit caches the downloaded bytes, so once this succeeds,
    browser reruns do not need to contact Google Drive again.
    """

    request = urllib.request.Request(
        _GDRIVE_DOWNLOAD_URL,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "*/*",
        },
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=30,
        ) as response:

            content = response.read()

    except Exception as exc:
        raise RuntimeError(
            f"Google Drive download failed: {exc}"
        ) from exc

    if not content:
        raise RuntimeError(
            "Google Drive returned an empty response."
        )

    preview = content[:2000].lower()

    if (
        b"<html" in preview
        or b"<!doctype" in preview
        or b"<head" in preview
    ):
        raise RuntimeError(
            "Google Drive returned an HTML page instead of "
            "city_day.csv."
        )

    return content


# ============================================================
# DATASET LOADING + CLEANING
# ============================================================

@st.cache_data(
    show_spinner="Loading and cleaning dataset…",
    persist=True,
)
def load_data(
    path: Path = DATA_PATH,
) -> tuple[pd.DataFrame, dict]:

    """
    Load city_day.csv, run the full cleaning pipeline and return:

      - df_clean  : cleaned DataFrame ready for analysis
      - qa_report : dict summarising every data-quality decision taken
    """

    # ========================================================
    # 0. DATASET ACQUISITION
    # ========================================================

    if path.exists() and path.stat().st_size > 0:

        # Local CSV exists → use it.
        raw = pd.read_csv(path)

    else:

        # Local CSV doesn't exist.
        #
        # Check Streamlit's cached Google Drive download.
        try:

            cached_bytes = _get_dataset_bytes()

        except Exception as exc:

            st.error(
                f"""
                ❌ **Could not load `city_day.csv`**

                `{exc}`

                Please make sure the Google Drive file is publicly
                accessible.
                """
            )

            st.stop()

        # Save cached/downloaded dataset locally.
        try:

            path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            path.write_bytes(cached_bytes)

        except Exception:
            # Local persistence is optional.
            pass

        # Read the dataset.
        raw = pd.read_csv(
            io.BytesIO(cached_bytes)
        )

    # ========================================================
    # EXISTING CLEANING PIPELINE STARTS HERE
    # ========================================================

    qa: dict = {}

    # ── 1. Record raw shape ─────────────────────────────────

    qa["raw_rows"] = len(raw)
    qa["raw_cols"] = raw.shape[1]

    # ── 2. Parse Date ────────────────────────────────────────────────────────
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    invalid_dates = raw["Date"].isna().sum()
    qa["invalid_dates_dropped"] = int(invalid_dates)
    df = raw.dropna(subset=["Date"]).copy()

    # ── 3. Drop exact (City, Date) duplicates ───────────────────────────────
    before = len(df)
    df = df.drop_duplicates(subset=["City", "Date"])
    qa["duplicates_dropped"] = before - len(df)

    # ── 4. Clip extreme outliers ─────────────────────────────────────────────
    # AQI > 1000 are physically implausible (India AQI scale 0–500 officially;
    # values up to ~2049 observed in data for Ahmedabad are instrument spikes).
    # We cap at 999 rather than drop, retaining high-pollution signal.
    aqi_extreme = (df["AQI"] > 999).sum()
    df["AQI"] = df["AQI"].clip(upper=999)
    qa["aqi_extreme_capped"] = int(aqi_extreme)

    # Clip each pollutant at its reference max (data-quality ceiling)
    for col, ref in POLLUTANT_REF.items():
        extreme = (df[col] > ref * 2).sum()
        df[col] = df[col].clip(upper=ref * 2)

    # ── 5. Treat zero Benzene / Toluene / Xylene / CO as NaN ────────────────
    # These sensors report 0.00 when below detection limit; zeros distort means.
    btx_co_cols = ["Benzene", "Toluene", "Xylene", "CO"]
    qa["btx_co_zeros_to_nan"] = int((df[btx_co_cols] == 0).sum().sum())
    df[btx_co_cols] = df[btx_co_cols].replace(0.0, np.nan)

    # ── 6. Impute missing pollutants with city-month median ──────────────────
    # City-month median preserves seasonal and city-specific patterns without
    # introducing cross-city bias.
    df["_month"] = df["Date"].dt.month
    for col in POLLUTANTS:
        df[col] = df.groupby(["City", "_month"])[col].transform(
            lambda s: s.fillna(s.median())
        )
    # Residual NaNs (cities with no data for a month) → city-level median
    for col in POLLUTANTS:
        df[col] = df.groupby("City")[col].transform(
            lambda s: s.fillna(s.median())
        )
    df.drop(columns=["_month"], inplace=True)

    # Final fallback: cities with zero data for a pollutant (e.g. PM10 for Lucknow)
    # → global dataset median for that pollutant to avoid NaN in KPI calculations.
    for col in POLLUTANTS:
        global_med = df[col].median()
        if not pd.isna(global_med):
            df[col] = df[col].fillna(global_med)

    # ── 7. Impute missing AQI using remaining pollutant signal ───────────────
    # AQI is officially determined from sub-index maxima; here we use the
    # city-month median AQI as a conservative imputation.
    df["_month"] = df["Date"].dt.month
    df["AQI"] = df.groupby(["City", "_month"])["AQI"].transform(
        lambda s: s.fillna(s.median())
    )
    df["AQI"] = df.groupby("City")["AQI"].transform(
        lambda s: s.fillna(s.median())
    )
    df.drop(columns=["_month"], inplace=True)

    # Re-derive AQI_Bucket for any imputed AQI rows
    bucket_order = AQI_ORDER
    def aqi_to_bucket(aqi):
        if pd.isna(aqi):
            return np.nan
        if aqi <= 50:
            return "Good"
        elif aqi <= 100:
            return "Satisfactory"
        elif aqi <= 200:
            return "Moderate"
        elif aqi <= 300:
            return "Poor"
        elif aqi <= 400:
            return "Very Poor"
        else:
            return "Severe"

    df["AQI_Bucket"] = df["AQI"].apply(aqi_to_bucket)

    # ── 8. Feature engineering ───────────────────────────────────────────────
    df["Year"]   = df["Date"].dt.year
    df["Month"]  = df["Date"].dt.month
    df["Season"] = df["Month"].map(SEASON_MAP)

    # ── 9. Record final quality stats ────────────────────────────────────────
    qa["clean_rows"]         = len(df)
    qa["missing_after_clean"] = int(df[POLLUTANTS + ["AQI"]].isna().sum().sum())
    qa["date_min"]           = df["Date"].min().strftime("%Y-%m-%d")
    qa["date_max"]           = df["Date"].max().strftime("%Y-%m-%d")
    qa["n_cities"]           = int(df["City"].nunique())

    return df, qa


# ─────────────────────────────────────────────────────────────────────────────
# KPI calculations
# ─────────────────────────────────────────────────────────────────────────────
def compute_kpis(df: pd.DataFrame, filter_mask: pd.Series | None = None) -> dict:
    """
    Compute the three headline KPIs for the (optionally filtered) dataframe.

    KPI 1 — Average AQI
        Arithmetic mean of AQI across all valid rows in the selection.

    KPI 2 — Pollution Risk Rate
        Percentage of days with AQI > 200 (i.e., 'Poor' or worse on the
        Indian AQI scale), indicating days where air quality poses a
        health risk to the general population.

    KPI 3 — Dominant Pollution Driver
        Computed dynamically: each pollutant's mean concentration is
        normalised by a reference upper-bound to bring all 12 pollutants
        onto the same 0–1 scale.  The pollutant with the highest normalised
        mean is declared the dominant driver for the current selection.
        This is never hardcoded — it changes with city / date filters.
    """
    subset = df[filter_mask] if filter_mask is not None else df

    # KPI 1 — Average AQI
    avg_aqi = subset["AQI"].mean()

    # KPI 2 — Pollution Risk Rate
    total_days  = len(subset)
    risk_days   = (subset["AQI"] > 200).sum()
    risk_rate   = (risk_days / total_days * 100) if total_days > 0 else 0.0

    # KPI 3 — Dominant Pollution Driver (dynamic, normalised)
    norm_means = {}
    for col, ref in POLLUTANT_REF.items():
        col_mean = subset[col].mean()
        if not pd.isna(col_mean):
            norm_means[col] = col_mean / ref

    dominant_driver     = max(norm_means, key=norm_means.get) if norm_means else "N/A"
    dominant_norm_value = norm_means.get(dominant_driver, np.nan)

    return {
        "avg_aqi":            round(avg_aqi, 1) if not pd.isna(avg_aqi) else "N/A",
        "risk_rate":          round(risk_rate, 1),
        "risk_days":          int(risk_days),
        "total_days":         total_days,
        "dominant_driver":    dominant_driver,
        "dominant_pct":       round(dominant_norm_value * 100, 1) if not pd.isna(dominant_norm_value) else 0,
        "norm_means":         norm_means,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Descriptive analysis helpers
# ─────────────────────────────────────────────────────────────────────────────
def city_aqi_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("City")["AQI"]
        .agg(Mean="mean", Median="median", Std="std", Min="min", Max="max", Count="count")
        .round(1)
        .sort_values("Mean", ascending=False)
        .reset_index()
    )


def yearly_trend(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("Year")["AQI"]
        .agg(Mean="mean", Median="median", Count="count")
        .round(1)
        .reset_index()
    )


def monthly_pattern(df: pd.DataFrame) -> pd.DataFrame:
    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    return (
        df.groupby("Month")["AQI"]
        .agg(Mean="mean")
        .round(1)
        .reset_index()
        .assign(MonthName=lambda x: x["Month"].map(month_names))
        .sort_values("Month")
    )


def seasonal_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    season_order = ["Winter", "Spring", "Summer", "Autumn"]
    agg = df.groupby("Season")["AQI"].agg(Mean="mean", Count="count").round(1).reset_index()
    agg["Season"] = pd.Categorical(agg["Season"], categories=season_order, ordered=True)
    return agg.sort_values("Season")


def bucket_distribution(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["AQI_Bucket"].value_counts().reset_index()
    counts.columns = ["AQI_Bucket", "Count"]
    counts["AQI_Bucket"] = pd.Categorical(counts["AQI_Bucket"], categories=AQI_ORDER, ordered=True)
    counts = counts.sort_values("AQI_Bucket")
    counts["Pct"] = (counts["Count"] / counts["Count"].sum() * 100).round(1)
    return counts


def pollutant_correlation(df: pd.DataFrame) -> pd.Series:
    corr = df[POLLUTANTS + ["AQI"]].corr()["AQI"].drop("AQI").sort_values(ascending=False)
    return corr.round(3)


def top_polluted_days(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    return (
        df.nlargest(n, "AQI")[
            ["City", "Date", "AQI", "AQI_Bucket", "PM2.5", "PM10", "NO2", "SO2"]
        ]
        .reset_index(drop=True)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Plotly chart builders
# ─────────────────────────────────────────────────────────────────────────────
def _apply_dark(fig: go.Figure) -> go.Figure:
    fig.update_layout(**PLOTLY_DARK_LAYOUT)
    fig.update_xaxes(**_DARK_AXIS)
    fig.update_yaxes(**_DARK_AXIS)
    return fig


def chart_city_aqi_bar(summary: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        summary, x="City", y="Mean",
        error_y="Std",
        color="Mean",
        color_continuous_scale="RdYlGn_r",
        labels={"Mean": "Mean AQI", "City": ""},
        title="Mean AQI by City (error bars = ±1 SD)",
    )
    fig.update_layout(**PLOTLY_DARK_LAYOUT, coloraxis_showscale=False)
    fig.update_xaxes(tickangle=40, **_DARK_AXIS)
    fig.update_yaxes(**_DARK_AXIS)
    return fig


def chart_yearly_trend(trend: pd.DataFrame) -> go.Figure:
    fig = px.line(
        trend, x="Year", y="Mean",
        markers=True,
        labels={"Mean": "Mean AQI", "Year": "Year"},
        title="Yearly Average AQI Trend",
    )
    fig.update_traces(line_color="#58a6ff", marker_color="#79c0ff")
    fig.update_layout(**PLOTLY_DARK_LAYOUT)
    fig.update_xaxes(**_DARK_AXIS)
    fig.update_yaxes(**_DARK_AXIS)
    return fig


def chart_monthly_pattern(monthly: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        monthly, x="MonthName", y="Mean",
        color="Mean",
        color_continuous_scale="RdYlGn_r",
        labels={"Mean": "Mean AQI", "MonthName": "Month"},
        title="Monthly Average AQI Pattern",
    )
    fig.update_layout(**PLOTLY_DARK_LAYOUT, coloraxis_showscale=False)
    fig.update_xaxes(**_DARK_AXIS)
    fig.update_yaxes(**_DARK_AXIS)
    return fig


def chart_bucket_donut(bucket_df: pd.DataFrame) -> go.Figure:
    colors = [AQI_COLORS.get(b, "#8b949e") for b in bucket_df["AQI_Bucket"]]
    fig = go.Figure(go.Pie(
        labels=bucket_df["AQI_Bucket"],
        values=bucket_df["Count"],
        hole=0.55,
        marker_colors=colors,
        textinfo="label+percent",
        textfont_color="#e6edf3",
    ))
    fig.update_layout(
        title="AQI Category Distribution",
        **PLOTLY_DARK_LAYOUT,
        showlegend=False,
    )
    return fig


def chart_pollutant_corr_bar(corr: pd.Series) -> go.Figure:
    clrs = ["#3fb950" if v >= 0 else "#da3633" for v in corr.values]
    fig = go.Figure(go.Bar(
        x=corr.index,
        y=corr.values,
        marker_color=clrs,
    ))
    fig.update_layout(
        title="Pollutant Correlation with AQI",
        xaxis_title="Pollutant",
        yaxis_title="Pearson r",
        **PLOTLY_DARK_LAYOUT,
    )
    fig.update_xaxes(**_DARK_AXIS)
    fig.update_yaxes(**_DARK_AXIS)
    return fig


def chart_seasonal_bar(seasonal: pd.DataFrame) -> go.Figure:
    season_colors = {
        "Winter": "#79c0ff",
        "Spring": "#56d364",
        "Summer": "#f0883e",
        "Autumn": "#e3b341",
    }
    clrs = [season_colors.get(s, "#8b949e") for s in seasonal["Season"]]
    fig = go.Figure(go.Bar(
        x=seasonal["Season"],
        y=seasonal["Mean"],
        marker_color=clrs,
        text=seasonal["Mean"],
        textposition="outside",
        textfont_color="#e6edf3",
    ))
    fig.update_layout(
        title="Average AQI by Season",
        xaxis_title="Season",
        yaxis_title="Mean AQI",
        **PLOTLY_DARK_LAYOUT,
    )
    fig.update_xaxes(**_DARK_AXIS)
    fig.update_yaxes(**_DARK_AXIS)
    return fig


def chart_dominant_driver_radar(norm_means: dict) -> go.Figure:
    categories = list(norm_means.keys())
    values     = [norm_means[c] for c in categories]
    values_closed = values + [values[0]]
    cats_closed   = categories + [categories[0]]

    fig = go.Figure(go.Scatterpolar(
        r=values_closed,
        theta=cats_closed,
        fill="toself",
        fillcolor="rgba(88,166,255,0.2)",
        line_color="#58a6ff",
    ))
    fig.update_layout(
        title="Normalised Pollutant Contribution (Radar)",
        polar=dict(
            bgcolor="#161b22",
            radialaxis=dict(visible=True, gridcolor="#30363d", color="#8b949e"),
            angularaxis=dict(gridcolor="#30363d", color="#e6edf3"),
        ),
        paper_bgcolor="#0e1117",
        font_color="#e6edf3",
        margin=dict(l=60, r=60, t=60, b=40),
    )
    return fig


def chart_city_heatmap(df: pd.DataFrame) -> go.Figure:
    pivot = (
        df.groupby(["City", "Year"])["AQI"]
        .mean()
        .round(0)
        .unstack("Year")
    )
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[str(c) for c in pivot.columns],
        y=pivot.index.tolist(),
        colorscale="RdYlGn_r",
        colorbar=dict(title="AQI", tickfont_color="#e6edf3", title_font_color="#e6edf3"),
        text=pivot.values.round(0),
        texttemplate="%{text:.0f}",
        hoverongaps=False,
    ))
    fig.update_layout(
        title="City × Year AQI Heatmap",
        xaxis_title="Year",
        yaxis_title="",
        **PLOTLY_DARK_LAYOUT,
        height=600,
    )
    fig.update_xaxes(**_DARK_AXIS)
    fig.update_yaxes(**_DARK_AXIS)
    return fig


def chart_pollutant_box(df: pd.DataFrame, pollutant: str) -> go.Figure:
    city_order = (
        df.groupby("City")[pollutant]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )
    fig = px.box(
        df, x="City", y=pollutant,
        category_orders={"City": city_order},
        color="City",
        color_discrete_sequence=px.colors.qualitative.Dark24,
        title=f"{pollutant} Distribution by City",
        labels={"City": ""},
    )
    fig.update_layout(**PLOTLY_DARK_LAYOUT, showlegend=False)
    fig.update_xaxes(tickangle=40, **_DARK_AXIS)
    fig.update_yaxes(**_DARK_AXIS)
    return fig



# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic analysis — data functions
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def diag_full_corr_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return the full Pearson correlation matrix for all pollutants + AQI.
    Correlations are *associations* only — they do not imply causation.
    """
    return df[POLLUTANTS + ["AQI"]].corr().round(3)


@st.cache_data(show_spinner=False)
def diag_pollutant_aqi_corr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ranked table of each pollutant's Pearson r with AQI, plus p-value
    from a t-test on the correlation coefficient (2-tailed).
    """
    from scipy.stats import pearsonr
    rows = []
    for col in POLLUTANTS:
        valid = df[[col, "AQI"]].dropna()
        if len(valid) < 10:
            continue
        r, p = pearsonr(valid[col], valid["AQI"])
        rows.append({
            "Pollutant": col,
            "Pearson r": round(r, 3),
            "p-value": p,
            "Significant (p<0.05)": "Yes" if p < 0.05 else "No",
            "Association": (
                "Strong positive" if r >= 0.6 else
                "Moderate positive" if r >= 0.3 else
                "Weak positive" if r > 0 else
                "Weak negative" if r > -0.3 else
                "Moderate negative"
            ),
        })
    return (
        pd.DataFrame(rows)
        .sort_values("Pearson r", ascending=False)
        .reset_index(drop=True)
    )


@st.cache_data(show_spinner=False)
def diag_compute_spikes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect AQI spikes using a per-city rolling z-score.

    Method
    ------
    For each city independently, compute a {SPIKE_WINDOW}-day rolling mean
    and rolling std of AQI.  A day is flagged as a spike when its z-score
    exceeds {SPIKE_Z_THRESHOLD}.

    Assumption: the rolling window provides a local seasonal baseline, so
    summer/winter baseline differences are automatically accounted for.
    A z-score threshold of {SPIKE_Z_THRESHOLD} corresponds roughly to the
    top 2.3 % of a normal distribution — i.e., unusually elevated days
    *relative to their recent local context*, not an absolute cut-off.

    This is a statistical characterisation; the method is descriptive and
    does not determine cause.
    """.format(SPIKE_WINDOW=SPIKE_WINDOW, SPIKE_Z_THRESHOLD=SPIKE_Z_THRESHOLD)
    df_sorted = df.sort_values(["City", "Date"])
    parts = []
    for _, grp in df_sorted.groupby("City", sort=False):
        grp = grp.copy()
        roll = grp["AQI"].rolling(SPIKE_WINDOW, min_periods=5)
        roll_mean = roll.mean()
        roll_std  = roll.std().replace(0.0, np.nan)
        grp["aqi_roll_mean"] = roll_mean
        grp["aqi_roll_std"]  = roll_std
        grp["aqi_z_score"]   = (grp["AQI"] - roll_mean) / roll_std
        grp["is_spike"]      = grp["aqi_z_score"] > SPIKE_Z_THRESHOLD
        parts.append(grp)
    return pd.concat(parts, ignore_index=True)


def diag_spike_summary(df_with_z: pd.DataFrame) -> pd.DataFrame:
    """Monthly spike-day counts across the filtered data."""
    spikes = df_with_z[df_with_z["is_spike"] == True]
    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    counts = spikes.groupby("Month").size().reset_index(name="Spike Days")
    counts["MonthName"] = counts["Month"].map(month_names)
    return counts.sort_values("Month")


def diag_spike_pollutant_profile(df_with_z: pd.DataFrame) -> pd.DataFrame:
    """
    Mean pollutant concentration during spike days vs non-spike days
    (for rows where z-score is calculable, i.e., after the 30-day warm-up).
    Returns a comparison DataFrame with ratio column.
    """
    calculable = df_with_z["aqi_z_score"].notna()
    spikes     = df_with_z[calculable & (df_with_z["is_spike"] == True)]
    non_spikes = df_with_z[calculable & (df_with_z["is_spike"] == False)]
    if len(spikes) == 0 or len(non_spikes) == 0:
        return pd.DataFrame()
    result = pd.DataFrame({
        "Spike Days Mean":     spikes[POLLUTANTS].mean().round(2),
        "Non-Spike Days Mean": non_spikes[POLLUTANTS].mean().round(2),
    })
    result["Elevation Ratio"] = (
        result["Spike Days Mean"] / result["Non-Spike Days Mean"].replace(0, np.nan)
    ).round(2)
    return result.sort_values("Elevation Ratio", ascending=False).reset_index(names=["Pollutant"])


@st.cache_data(show_spinner=False)
def diag_normal_vs_highrisk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare mean pollutant concentrations between:
      - Normal days   : AQI ≤ 100  (Good + Satisfactory)
      - High-risk days: AQI > 200  (Poor + Very Poor + Severe)

    Also runs a one-sided Mann-Whitney U test to check whether high-risk
    days have statistically significantly higher concentrations.
    Statistical significance ≠ practical causation; confounding is possible.
    """
    normal    = df[df["AQI"] <= 100]
    high_risk = df[df["AQI"] >  AQI_RISK_THRESHOLD]

    rows = []
    for col in POLLUTANTS:
        n_mean = normal[col].mean()
        h_mean = high_risk[col].mean()
        ratio  = h_mean / n_mean if (n_mean and not np.isnan(n_mean) and n_mean != 0) else np.nan

        n_vals = normal[col].dropna()
        h_vals = high_risk[col].dropna()
        if len(h_vals) >= 5 and len(n_vals) >= 5:
            _, p = mannwhitneyu(h_vals, n_vals, alternative="greater")
            sig = "Yes" if p < 0.05 else "No"
        else:
            p, sig = np.nan, "N/A"

        rows.append({
            "Pollutant":             col,
            "Normal Mean (AQI≤100)": round(n_mean, 2) if not np.isnan(n_mean) else np.nan,
            "High-Risk Mean (AQI>200)": round(h_mean, 2) if not np.isnan(h_mean) else np.nan,
            "Elevation Ratio":       round(ratio, 2) if not np.isnan(ratio) else np.nan,
            "Sig. Higher? (p<0.05)": sig,
        })

    return (
        pd.DataFrame(rows)
        .sort_values("Elevation Ratio", ascending=False)
        .reset_index(drop=True)
    )


def diag_pollutant_by_aqi_band(df: pd.DataFrame, pollutant: str) -> pd.DataFrame:
    """
    Median concentration of a chosen pollutant within each AQI category band.
    Useful for visualising the monotonic (or non-monotonic) relationship
    between a pollutant and AQI — without assuming linearity.
    """
    df2 = df.copy()
    df2["AQI_Band"] = pd.cut(
        df2["AQI"], bins=AQI_BINS, labels=AQI_LABELS, include_lowest=True
    )
    result = (
        df2.groupby("AQI_Band", observed=True)[pollutant]
        .agg(Median="median", Mean="mean", Count="count")
        .round(2)
        .reset_index()
    )
    result["AQI_Band"] = pd.Categorical(result["AQI_Band"], categories=AQI_LABELS, ordered=True)
    return result.sort_values("AQI_Band")


@st.cache_data(show_spinner=False)
def diag_dominant_driver_shifts(df: pd.DataFrame) -> dict:
    """
    Compute the dominant pollution driver (normalised mean contribution)
    broken down by:
      - Season (Winter / Spring / Summer / Autumn)
      - Year
      - City
    Returns a dict of DataFrames, one per breakdown dimension.
    """
    def _dominant_table(group_col: str) -> pd.DataFrame:
        rows = []
        for val, grp in df.groupby(group_col):
            nm = {}
            for col, ref in POLLUTANT_REF.items():
                cm = grp[col].mean()
                if not pd.isna(cm):
                    nm[col] = cm / ref
            if not nm:
                continue
            ranked = sorted(nm, key=nm.get, reverse=True)
            rows.append({
                group_col:    val,
                "Dominant":   ranked[0],
                "2nd Driver": ranked[1] if len(ranked) > 1 else "",
                "3rd Driver": ranked[2] if len(ranked) > 2 else "",
                "Dom. Score": round(nm[ranked[0]] * 100, 1),
            })
        return pd.DataFrame(rows)

    season_order = ["Winter", "Spring", "Summer", "Autumn"]
    by_season = _dominant_table("Season")
    if "Season" in by_season.columns:
        by_season["Season"] = pd.Categorical(by_season["Season"], categories=season_order, ordered=True)
        by_season = by_season.sort_values("Season")

    return {
        "by_season": by_season,
        "by_year":   _dominant_table("Year").sort_values("Year"),
        "by_city":   _dominant_table("City").sort_values("City"),
    }


def diag_generate_insights(
    df: pd.DataFrame,
    corr_ranked: pd.DataFrame,
    normal_vs_hr: pd.DataFrame,
    df_with_z: pd.DataFrame,
    driver_shifts: dict,
) -> list[str]:
    """
    Generate a list of concise, data-grounded diagnostic insight strings
    from the currently filtered dataset.  All values are computed from
    the actual filtered data — nothing is hardcoded.

    Returns a list of plain-text bullet strings.
    """
    insights = []

    # ── 1. Strongest AQI-associated pollutant ────────────────────────────────
    top_corr = corr_ranked.iloc[0] if len(corr_ranked) > 0 else None
    if top_corr is not None:
        insights.append(
            f"**Strongest AQI association:** {top_corr['Pollutant']} "
            f"(Pearson r = {top_corr['Pearson r']:.3f}). "
            f"This is a statistical correlation, not a direct causal link."
        )

    # ── 2. Most elevated pollutant during high-risk days ─────────────────────
    if len(normal_vs_hr) > 0:
        top_elev = normal_vs_hr.iloc[0]
        ratio = top_elev["Elevation Ratio"]
        if not np.isnan(ratio):
            insights.append(
                f"**Most elevated during high-risk days (AQI > 200):** "
                f"{top_elev['Pollutant']} — {ratio:.1f}× higher than on normal days "
                f"(mean {top_elev['High-Risk Mean (AQI>200)']} vs {top_elev['Normal Mean (AQI≤100)']} µg/m³)."
            )

    # ── 3. Spike count and dominant spike season ─────────────────────────────
    n_spikes = df_with_z["is_spike"].sum()
    total_valid = df_with_z["aqi_z_score"].notna().sum()
    if total_valid > 0:
        spike_pct = n_spikes / total_valid * 100
        if n_spikes > 0:
            spike_season = (
                df_with_z[df_with_z["is_spike"] == True]
                .groupby("Season")["is_spike"]
                .sum()
                .idxmax()
            )
            insights.append(
                f"**AQI spikes (z > {SPIKE_Z_THRESHOLD}, 30-day rolling baseline):** "
                f"{n_spikes:,} spike days detected ({spike_pct:.1f}% of valid records). "
                f"Most spikes occur in {spike_season}."
            )

    # ── 4. Seasonal dominant driver shift ────────────────────────────────────
    by_season = driver_shifts.get("by_season", pd.DataFrame())
    if len(by_season) > 1:
        shifts = by_season[["Season", "Dominant"]].set_index("Season")["Dominant"].to_dict()
        unique_drivers = len(set(shifts.values()))
        shift_str = " → ".join(f"{s}: {d}" for s, d in shifts.items())
        if unique_drivers > 1:
            insights.append(
                f"**Seasonal dominant driver shifts:** {shift_str}. "
                f"The dominant pollutant is not constant across seasons."
            )
        else:
            single_driver = list(shifts.values())[0]
            insights.append(
                f"**Seasonal dominant driver:** {single_driver} is the leading driver "
                f"across all selected seasons in this filter context."
            )

    # ── 5. City-level driver diversity ───────────────────────────────────────
    by_city = driver_shifts.get("by_city", pd.DataFrame())
    if len(by_city) > 1 and "Dominant" in by_city.columns:
        driver_variety = by_city["Dominant"].value_counts()
        top_driver     = driver_variety.index[0]
        n_cities_top   = driver_variety.iloc[0]
        n_cities_total = len(by_city)
        insights.append(
            f"**City-level driver diversity:** Across {n_cities_total} selected cities, "
            f"{top_driver} is the dominant driver in {n_cities_top} of them. "
            f"{len(driver_variety)} distinct dominant drivers are present — "
            f"indicating city-specific pollution profiles."
        )

    # ── 6. Risk rate context ─────────────────────────────────────────────────
    risk_days = (df["AQI"] > AQI_RISK_THRESHOLD).sum()
    total_days = len(df)
    risk_pct   = risk_days / total_days * 100 if total_days > 0 else 0
    avg_aqi    = df["AQI"].mean()
    insights.append(
        f"**Overall risk context:** {risk_days:,} of {total_days:,} days "
        f"({risk_pct:.1f}%) exceed AQI 200. "
        f"Mean AQI is {avg_aqi:.0f} — "
        f"{'well above' if avg_aqi > 200 else 'above' if avg_aqi > 100 else 'within'} "
        f"the Satisfactory threshold (≤ 100)."
    )

    # ── 7. Correlation vs causation disclaimer ───────────────────────────────
    insights.append(
        "⚠️ **Caution:** All associations shown are correlational. "
        "Elevated pollutant concentrations during high-AQI days do not "
        "establish a direct causal mechanism without further controlled analysis. "
        "Confounding (e.g., shared seasonal drivers, co-emission sources) is possible."
    )

    return insights


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic analysis — chart builders
# ─────────────────────────────────────────────────────────────────────────────

def chart_corr_heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
    """Full pollutant × pollutant + AQI Pearson correlation heatmap."""
    cols = corr_matrix.columns.tolist()
    z    = corr_matrix.values

    fig = go.Figure(go.Heatmap(
        z=z,
        x=cols,
        y=cols,
        colorscale="RdBu",
        zmid=0,
        zmin=-1, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in z],
        texttemplate="%{text}",
        textfont_size=9,
        colorbar=dict(
            title="r",
            tickfont_color="#e6edf3",
            title_font_color="#e6edf3",
        ),
        hoverongaps=False,
    ))
    fig.update_layout(
        title="Full Correlation Matrix — Pollutants × Pollutants + AQI",
        **PLOTLY_DARK_LAYOUT,
        height=520,
    )
    fig.update_xaxes(tickangle=45, **_DARK_AXIS)
    fig.update_yaxes(autorange="reversed", **_DARK_AXIS)
    return fig


def chart_aqi_corr_ranked(corr_ranked: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart: each pollutant's Pearson r with AQI, colour-coded."""
    clrs = ["#3fb950" if v >= 0 else "#da3633" for v in corr_ranked["Pearson r"]]
    fig = go.Figure(go.Bar(
        x=corr_ranked["Pearson r"],
        y=corr_ranked["Pollutant"],
        orientation="h",
        marker_color=clrs,
        text=corr_ranked["Pearson r"].apply(lambda v: f"{v:.3f}"),
        textposition="outside",
        textfont_color="#e6edf3",
    ))
    fig.add_vline(x=0, line_color="#8b949e", line_width=1)
    fig.add_vline(x=0.6,  line_dash="dot", line_color="#58a6ff",
                  annotation_text="Strong (0.6)", annotation_font_color="#58a6ff",
                  annotation_position="top right")
    fig.add_vline(x=0.3,  line_dash="dot", line_color="#e3b341",
                  annotation_text="Moderate (0.3)", annotation_font_color="#e3b341",
                  annotation_position="top right")
    fig.update_layout(
        title="Pollutant Association with AQI (Pearson r) — statistical, not causal",
        xaxis_title="Pearson r",
        yaxis_title="",
        **PLOTLY_DARK_LAYOUT,
        height=400,
    )
    fig.update_xaxes(range=[-0.2, 1.0], **_DARK_AXIS)
    fig.update_yaxes(autorange="reversed", **_DARK_AXIS)
    return fig


def chart_spike_calendar(spike_counts: pd.DataFrame) -> go.Figure:
    """Bar chart of spike-day counts per month."""
    fig = go.Figure(go.Bar(
        x=spike_counts["MonthName"],
        y=spike_counts["Spike Days"],
        marker_color="#f0883e",
        text=spike_counts["Spike Days"],
        textposition="outside",
        textfont_color="#e6edf3",
    ))
    fig.update_layout(
        title=f"AQI Spike Days per Month (rolling z-score > {SPIKE_Z_THRESHOLD}, {SPIKE_WINDOW}-day window)",
        xaxis_title="Month",
        yaxis_title="Spike Days",
        **PLOTLY_DARK_LAYOUT,
    )
    fig.update_xaxes(**_DARK_AXIS)
    fig.update_yaxes(**_DARK_AXIS)
    return fig


def chart_normal_vs_highrisk(nvhr: pd.DataFrame) -> go.Figure:
    """Grouped bar chart comparing mean pollutant levels: normal vs high-risk days."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Normal Days (AQI ≤ 100)",
        x=nvhr["Pollutant"],
        y=nvhr["Normal Mean (AQI≤100)"],
        marker_color="#3fb950",
    ))
    fig.add_trace(go.Bar(
        name="High-Risk Days (AQI > 200)",
        x=nvhr["Pollutant"],
        y=nvhr["High-Risk Mean (AQI>200)"],
        marker_color="#da3633",
    ))
    fig.update_layout(
        title="Mean Pollutant Concentration: Normal vs High-Risk Days",
        barmode="group",
        xaxis_title="Pollutant",
        yaxis_title="Mean Concentration (µg/m³, CO in mg/m³)",
        legend=dict(font_color="#e6edf3", bgcolor="#161b22"),
        **PLOTLY_DARK_LAYOUT,
    )
    fig.update_xaxes(**_DARK_AXIS)
    fig.update_yaxes(**_DARK_AXIS)
    return fig


def chart_pollutant_by_aqi_band(band_df: pd.DataFrame, pollutant: str) -> go.Figure:
    """
    Line + marker chart showing median pollutant concentration per AQI category.
    A steep, monotonic rise indicates a strong positive association with AQI.
    Non-monotonic behaviour (e.g. O3 plateaus at Very Poor) reveals more complex dynamics.
    """
    clrs = [AQI_COLORS.get(b, "#8b949e") for b in band_df["AQI_Band"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=band_df["AQI_Band"],
        y=band_df["Median"],
        marker_color=clrs,
        text=band_df["Median"].round(1),
        textposition="outside",
        textfont_color="#e6edf3",
        name="Median",
    ))
    fig.add_trace(go.Scatter(
        x=band_df["AQI_Band"],
        y=band_df["Mean"],
        mode="lines+markers",
        name="Mean",
        line_color="#58a6ff",
        marker=dict(size=7, color="#58a6ff"),
    ))
    fig.update_layout(
        title=f"{pollutant} Concentration by AQI Category (Median + Mean)",
        xaxis_title="AQI Category",
        yaxis_title=f"{pollutant} (µg/m³)" + (" [mg/m³]" if pollutant == "CO" else ""),
        legend=dict(font_color="#e6edf3", bgcolor="#161b22"),
        **PLOTLY_DARK_LAYOUT,
    )
    fig.update_xaxes(**_DARK_AXIS)
    fig.update_yaxes(**_DARK_AXIS)
    return fig


def chart_dominant_driver_shifts(driver_shifts: dict, dimension: str) -> go.Figure:
    """
    Stacked-like bar chart showing the dominant driver and its normalised score
    by the chosen breakdown dimension (season / year / city).
    """
    col_map = {"season": "Season", "year": "Year", "city": "City"}
    key_map  = {"season": "by_season", "year": "by_year", "city": "by_city"}
    dim_col = col_map.get(dimension, "Season")
    df_dim  = driver_shifts.get(key_map.get(dimension, "by_season"), pd.DataFrame())

    if df_dim.empty:
        fig = go.Figure()
        fig.update_layout(title="No data", **PLOTLY_DARK_LAYOUT)
        return fig

    # Assign a colour to each unique dominant driver
    unique_drivers = df_dim["Dominant"].unique().tolist()
    palette = px.colors.qualitative.Dark24
    color_map = {d: palette[i % len(palette)] for i, d in enumerate(unique_drivers)}
    bar_colors = [color_map[d] for d in df_dim["Dominant"]]

    fig = go.Figure(go.Bar(
        x=df_dim[dim_col].astype(str),
        y=df_dim["Dom. Score"],
        marker_color=bar_colors,
        text=df_dim["Dominant"],
        textposition="inside",
        textfont_color="#ffffff",
        hovertemplate=(
            f"<b>%{{x}}</b><br>Dominant: %{{text}}<br>"
            "Normalised score: %{y:.1f}%<extra></extra>"
        ),
    ))

    # Legend items (one per unique driver)
    for driver, color in color_map.items():
        fig.add_trace(go.Bar(
            x=[None], y=[None],
            name=driver,
            marker_color=color,
            showlegend=True,
        ))

    tick_angle = 40 if dimension == "city" else 0
    fig.update_layout(
        title=f"Dominant Pollution Driver by {dim_col} (normalised concentration score)",
        xaxis_title=dim_col,
        yaxis_title="Normalised Score (%)",
        barmode="group",
        legend=dict(font_color="#e6edf3", bgcolor="#161b22"),
        **PLOTLY_DARK_LAYOUT,
        height=420 if dimension == "city" else 380,
    )
    fig.update_xaxes(tickangle=tick_angle, **_DARK_AXIS)
    fig.update_yaxes(**_DARK_AXIS)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
def build_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    """Render the premium sidebar and return the filtered dataframe."""

    # ── Branding strip ───────────────────────────────────────────────────────
    st.sidebar.markdown(
        """
        <div style="padding:14px 4px 10px 4px; border-bottom:1px solid rgba(48,54,61,0.8); margin-bottom:8px;">
          <div style="font-size:1rem; font-weight:800; color:#e6edf3; letter-spacing:-0.01em;">
            ANALYSIS<span style="color:#58a6ff;"> CONTROLS</span>
          </div>
          <div style="font-size:0.72rem; color:#484f58; margin-top:2px; letter-spacing:0.03em;">
            INDIA · 2015 – 2020
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── City filter ──────────────────────────────────────────────────────────
    st.sidebar.markdown(
        '<div class="sidebar-section-label">🏙 City Selection</div>',
        unsafe_allow_html=True,
    )
    all_cities = sorted(df["City"].unique())

    # Quick-select presets
    preset = st.sidebar.selectbox(
        "Quick preset",
        ["All cities", "North India (Delhi/Gurugram/Lucknow/Patna/Amritsar/Chandigarh)",
         "South India (Chennai/Bengaluru/Hyderabad/Kochi/Thiruvananthapuram/Coimbatore/Visakhapatnam/Amaravati)",
         "East India (Kolkata/Patna/Brajrajnagar/Jorapokhar/Talcher/Guwahati/Shillong/Aizawl)",
         "West India (Mumbai/Ahmedabad/Jaipur/Bhopal)",
         "Custom"],
        key="city_preset",
        label_visibility="collapsed",
    )
    north  = ["Delhi","Gurugram","Lucknow","Patna","Amritsar","Chandigarh"]
    south  = ["Chennai","Bengaluru","Hyderabad","Kochi","Thiruvananthapuram","Coimbatore","Visakhapatnam","Amaravati"]
    east   = ["Kolkata","Patna","Brajrajnagar","Jorapokhar","Talcher","Guwahati","Shillong","Aizawl"]
    west   = ["Mumbai","Ahmedabad","Jaipur","Bhopal"]

    if preset.startswith("North"):
        default_cities = [c for c in north if c in all_cities]
    elif preset.startswith("South"):
        default_cities = [c for c in south if c in all_cities]
    elif preset.startswith("East"):
        default_cities = [c for c in east if c in all_cities]
    elif preset.startswith("West"):
        default_cities = [c for c in west if c in all_cities]
    else:
        default_cities = all_cities

    selected_cities = st.sidebar.multiselect(
        "Cities",
        options=all_cities,
        default=default_cities,
        help="Hold Ctrl/Cmd to select multiple cities",
        label_visibility="collapsed",
    )
    if not selected_cities:
        selected_cities = all_cities

    # ── Date/year filter ─────────────────────────────────────────────────────
    st.sidebar.markdown(
        '<div class="sidebar-section-label">📅 Date Range</div>',
        unsafe_allow_html=True,
    )
    years = sorted(df["Year"].unique())
    year_range = st.sidebar.slider(
        "Year range",
        min_value=int(years[0]),
        max_value=int(years[-1]),
        value=(int(years[0]), int(years[-1])),
        label_visibility="collapsed",
    )

    # ── Season filter ────────────────────────────────────────────────────────
    st.sidebar.markdown(
        '<div class="sidebar-section-label">🌤 Seasons</div>',
        unsafe_allow_html=True,
    )
    season_icons = {"Winter": "❄️ Winter", "Spring": "🌸 Spring",
                    "Summer": "☀️ Summer", "Autumn": "🍂 Autumn"}
    seasons = st.sidebar.multiselect(
        "Seasons",
        options=list(season_icons.keys()),
        default=list(season_icons.keys()),
        format_func=lambda s: season_icons[s],
        label_visibility="collapsed",
    )
    if not seasons:
        seasons = list(season_icons.keys())

    # ── AQI category quick-filter ────────────────────────────────────────────
    st.sidebar.markdown(
        '<div class="sidebar-section-label">🎯 AQI Category Focus</div>',
        unsafe_allow_html=True,
    )
    all_buckets = AQI_ORDER
    selected_buckets = st.sidebar.multiselect(
        "AQI categories",
        options=all_buckets,
        default=all_buckets,
        help="Filter to only these AQI categories",
        label_visibility="collapsed",
    )
    if not selected_buckets:
        selected_buckets = all_buckets

    # ── Apply mask ───────────────────────────────────────────────────────────
    mask = (
        df["City"].isin(selected_cities)
        & df["Year"].between(year_range[0], year_range[1])
        & df["Season"].isin(seasons)
        & df["AQI_Bucket"].isin(selected_buckets)
    )
    df_filtered = df[mask].copy()

    # ── Live stats panel ─────────────────────────────────────────────────────
    st.sidebar.markdown("<hr style='margin:14px 0 10px 0;'>", unsafe_allow_html=True)
    st.sidebar.markdown(
        '<div class="sidebar-section-label">📊 Selection Summary</div>',
        unsafe_allow_html=True,
    )

    n_records  = len(df_filtered)
    n_cities   = df_filtered["City"].nunique() if n_records > 0 else 0
    avg_aqi_s  = f"{df_filtered['AQI'].mean():.0f}" if n_records > 0 else "–"
    date_span  = (
        f"{df_filtered['Date'].min().strftime('%b %Y')} → "
        f"{df_filtered['Date'].max().strftime('%b %Y')}"
        if n_records > 0 else "–"
    )

    st.sidebar.markdown(
        f"""
        <div class="sidebar-stat"><span class="label">Records</span><span class="value">{n_records:,}</span></div>
        <div class="sidebar-stat"><span class="label">Cities</span><span class="value">{n_cities}</span></div>
        <div class="sidebar-stat"><span class="label">Mean AQI</span><span class="value">{avg_aqi_s}</span></div>
        <div class="sidebar-stat"><span class="label">Period</span><span class="value" style="font-size:0.72rem">{date_span}</span></div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown(
        """
        <div style="margin-top:20px; padding-top:12px; border-top:1px solid rgba(48,54,61,0.8);
             font-size:0.7rem; color:#484f58; text-align:center; line-height:1.5;">
          Urban Air Quality Intelligence<br>
          CPCB India · Descriptive &amp; Diagnostic
        </div>
        """,
        unsafe_allow_html=True,
    )

    return df_filtered


# ─────────────────────────────────────────────────────────────────────────────
# Main application
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    # ── Load & filter ────────────────────────────────────────────────────────
    df_full, qa = load_data()
    df = build_sidebar(df_full)

    if df.empty:
        st.markdown(
            """<div class="hero-banner">
              <div class="hero-title">🌦️ Urban Air Quality <span>Intelligence</span></div>
              <div class="hero-subtitle" style="color:#da3633;">
                No data matches the current filter selection.<br>
                Please adjust the sidebar filters to continue.
              </div>
            </div>""",
            unsafe_allow_html=True,
        )
        return

    # ── Compute all analysis results (once, shared across all tabs) ───────────
    kpis         = compute_kpis(df)
    corr_matrix  = diag_full_corr_matrix(df)
    corr_ranked  = diag_pollutant_aqi_corr(df)
    df_with_z    = diag_compute_spikes(df)
    normal_vs_hr = diag_normal_vs_highrisk(df)
    driver_shifts= diag_dominant_driver_shifts(df)
    insights     = diag_generate_insights(df, corr_ranked, normal_vs_hr, df_with_z, driver_shifts)

    # ─────────────────────────────────────────────────────────────────────────
    # HERO BANNER
    # ─────────────────────────────────────────────────────────────────────────
    # Derive bucket label for the filtered selection
    aqi_val  = kpis["avg_aqi"]
    # Guard: aqi_val is "N/A" (str) when df is empty — use float check before comparison
    if isinstance(aqi_val, (int, float)) and not pd.isna(aqi_val):
        _av = float(aqi_val)
        bucket = (
            "Good"         if _av <= 50  else
            "Satisfactory" if _av <= 100 else
            "Moderate"     if _av <= 200 else
            "Poor"         if _av <= 300 else
            "Very Poor"    if _av <= 400 else
            "Severe"
        )
    else:
        bucket = "N/A"
    bucket_css = bucket.lower().replace(" ", "-") if bucket != "N/A" else "moderate"

    filter_city_label = (
        f"{df['City'].nunique()} cities"
        if df["City"].nunique() > 3
        else " · ".join(sorted(df["City"].unique()))
    )
    filter_year_label = (
        f"{int(df['Year'].min())}–{int(df['Year'].max())}"
    )

    st.markdown(
        f"""
        <div class="hero-banner">
          <div class="hero-title">🌦️ Urban Air Quality <span>Intelligence</span></div>
          <div class="hero-subtitle">
            Descriptive &amp; diagnostic analysis of CPCB air-quality measurements
            across Indian cities — exploring AQI patterns, pollutant associations,
            and high-risk periods.
          </div>
          <div class="hero-meta">
            <span class="hero-badge">&#9672; {filter_city_label}</span>
            <span class="hero-badge">&#9656; {filter_year_label}</span>
            <span class="hero-badge">&#9632; {len(df):,} records</span>
            <span class="hero-badge">&#9654; {qa['date_min']} &rarr; {qa['date_max']}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # KPI CARDS  (custom HTML — full control over colour and layout)
    # ─────────────────────────────────────────────────────────────────────────
    risk_rate  = kpis["risk_rate"]
    risk_css   = (
        "risk-low"    if risk_rate < 10  else
        "risk-medium" if risk_rate < 25  else
        "risk-high"   if risk_rate < 50  else
        "risk-severe"
    )
    dom_driver = kpis["dominant_driver"]
    dom_pct    = kpis["dominant_pct"]

    st.markdown(
        f"""
        <div class="kpi-grid">

          <!-- KPI 1 — Average AQI -->
          <div class="kpi-card" style="--kpi-accent: {AQI_COLORS.get(bucket, '#58a6ff')};">
            <div class="kpi-label">Average AQI</div>
            <div class="kpi-value">{aqi_val}</div>
            <span class="kpi-badge badge-{bucket_css}">{bucket}</span>
            <div class="kpi-sub">
              Mean AQI across all filtered records.<br>
              India scale: 0–50 Good → &gt;400 Severe.
            </div>
          </div>

          <!-- KPI 2 — Pollution Risk Rate -->
          <div class="kpi-card" style="--kpi-accent: #f0883e;">
            <div class="kpi-label">Pollution Risk Rate</div>
            <div class="kpi-value">{risk_rate}%</div>
            <span class="kpi-badge {risk_css}">{kpis['risk_days']:,} of {kpis['total_days']:,} days</span>
            <div class="kpi-sub">
              Days with AQI &gt; 200 ("Poor" or worse) —<br>
              threshold at which health impacts begin.
            </div>
          </div>

          <!-- KPI 3 — Dominant Pollution Driver -->
          <div class="kpi-card" style="--kpi-accent: #8957e5;">
            <div class="kpi-label">Dominant Pollution Driver</div>
            <div class="kpi-value">{dom_driver}</div>
            <span class="kpi-badge" style="background:rgba(137,87,229,0.18);color:#8957e5;border:1px solid rgba(137,87,229,0.35);">
              Norm. score: {dom_pct}%
            </span>
            <div class="kpi-sub">
              Highest normalised mean concentration<br>
              (÷ reference max). Dynamic with filters.
            </div>
          </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    # Supporting stats strip
    n_cities_sel  = df["City"].nunique()
    n_records_sel = len(df)
    good_days     = (df["AQI_Bucket"] == "Good").sum()
    severe_days   = (df["AQI_Bucket"] == "Severe").sum()

    st.markdown(
        f"""
        <div class="kpi-stat-strip">
          <div class="stat-chip">
            <div class="n">{n_cities_sel}</div>
            <div class="lbl">Cities</div>
          </div>
          <div class="stat-chip">
            <div class="n">{n_records_sel:,}</div>
            <div class="lbl">Observations</div>
          </div>
          <div class="stat-chip">
            <div class="n" style="color:#56d364;">{good_days:,}</div>
            <div class="lbl">Good Days</div>
          </div>
          <div class="stat-chip">
            <div class="n" style="color:#8957e5;">{severe_days:,}</div>
            <div class="lbl">Severe Days</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # NAVIGATION TABS
    # ─────────────────────────────────────────────────────────────────────────
    tab_aqi, tab_city, tab_poll, tab_diag, tab_data = st.tabs([
        "📈  AQI Intelligence",
        "🏙️  City Intelligence",
        "🧪  Pollution Intelligence",
        "🔬  Diagnostic Intelligence",
        "🗄️  Data Quality",
    ])

    # =========================================================================
    # TAB 1 — AQI INTELLIGENCE
    # =========================================================================
    with tab_aqi:
        # Section: AQI Distribution
        st.markdown(
            """<div class="section-header">
              <div class="section-icon">📈</div>
              <span class="section-title">AQI Distribution &amp; Categories</span>
              <span class="section-desc">How is AQI distributed across the filtered selection?</span>
            </div>""",
            unsafe_allow_html=True,
        )

        # Compute once, reuse for donut and table
        bucket_df = bucket_distribution(df)

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(chart_bucket_donut(bucket_df), width="stretch")
        with c2:
            fig_hist = px.histogram(
                df, x="AQI", nbins=60,
                color_discrete_sequence=["#58a6ff"],
                title="AQI Value Distribution",
                labels={"AQI": "AQI"},
            )
            fig_hist.update_layout(**PLOTLY_DARK_LAYOUT)
            st.plotly_chart(fig_hist, width="stretch")

        # Category table + worst days side by side
        c3, c4 = st.columns([1, 2])
        with c3:
            st.markdown(
                """<div class="section-header" style="margin-top:8px;">
                  <div class="section-icon">🗂️</div>
                  <span class="section-title">Category Breakdown</span>
                </div>""", unsafe_allow_html=True,
            )
            bdf_display = bucket_df.rename(
                columns={"AQI_Bucket": "Category", "Count": "Days", "Pct": "Share (%)"}
            )
            st.dataframe(bdf_display[["Category", "Days", "Share (%)"]], width="stretch", hide_index=True)

        with c4:
            st.markdown(
                """<div class="section-header" style="margin-top:8px;">
                  <div class="section-icon">⚠️</div>
                  <span class="section-title">Worst Pollution Days</span>
                </div>""", unsafe_allow_html=True,
            )
            st.dataframe(top_polluted_days(df, n=10), width="stretch", hide_index=True)

        # Section: AQI Trends
        st.markdown(
            """<div class="section-header">
              <div class="section-icon">📅</div>
              <span class="section-title">AQI Trends Over Time</span>
              <span class="section-desc">Annual and seasonal patterns</span>
            </div>""",
            unsafe_allow_html=True,
        )

        ca, cb = st.columns(2)
        with ca:
            trend = yearly_trend(df)
            st.plotly_chart(chart_yearly_trend(trend), width="stretch")
        with cb:
            monthly = monthly_pattern(df)
            st.plotly_chart(chart_monthly_pattern(monthly), width="stretch")

        # Seasonal bar
        seasonal = seasonal_breakdown(df)
        cs1, cs2 = st.columns([2, 1])
        with cs1:
            st.plotly_chart(chart_seasonal_bar(seasonal), width="stretch")
        with cs2:
            st.markdown(
                """<div class="section-header" style="margin-top:8px;">
                  <div class="section-icon">🌤️</div>
                  <span class="section-title">Seasonal Summary</span>
                </div>""", unsafe_allow_html=True,
            )
            st.dataframe(
                seasonal.rename(columns={"Mean": "Mean AQI", "Count": "Records"}),
                width="stretch", hide_index=True,
            )

    # =========================================================================
    # TAB 2 — CITY INTELLIGENCE
    # =========================================================================
    with tab_city:
        st.markdown(
            """<div class="section-header">
              <div class="section-icon">🏙️</div>
              <span class="section-title">City AQI Ranking</span>
              <span class="section-desc">Mean AQI with variability (±1 SD)</span>
            </div>""",
            unsafe_allow_html=True,
        )
        summary = city_aqi_summary(df)
        st.plotly_chart(chart_city_aqi_bar(summary), width="stretch")

        st.markdown(
            """<div class="section-header">
              <div class="section-icon">🗺️</div>
              <span class="section-title">City × Year AQI Heatmap</span>
              <span class="section-desc">Track how each city's AQI evolved year-by-year</span>
            </div>""",
            unsafe_allow_html=True,
        )
        st.plotly_chart(chart_city_heatmap(df), width="stretch")

        with st.expander("📋 Full City AQI Summary Table"):
            st.dataframe(summary, width="stretch", hide_index=True)

    # =========================================================================
    # TAB 3 — POLLUTION INTELLIGENCE
    # =========================================================================
    with tab_poll:
        # Radar + correlation bar side-by-side
        st.markdown(
            """<div class="section-header">
              <div class="section-icon">🧪</div>
              <span class="section-title">Pollutant Profile</span>
              <span class="section-desc">Normalised contribution radar &amp; AQI correlation</span>
            </div>""",
            unsafe_allow_html=True,
        )

        cp1, cp2 = st.columns(2)
        with cp1:
            norm_means = kpis.get("norm_means", {})
            if norm_means:
                st.plotly_chart(chart_dominant_driver_radar(norm_means), width="stretch")
        with cp2:
            corr = pollutant_correlation(df)
            st.plotly_chart(chart_pollutant_corr_bar(corr), width="stretch")

        # Pollutant distribution by city (selectbox)
        st.markdown(
            """<div class="section-header">
              <div class="section-icon">📦</div>
              <span class="section-title">Pollutant Distribution by City</span>
              <span class="section-desc">Box-plot showing spread and outliers per city</span>
            </div>""",
            unsafe_allow_html=True,
        )
        sel_pollutant = st.selectbox(
            "Select pollutant",
            POLLUTANTS,
            index=0,
            key="poll_city_select",
        )
        st.plotly_chart(chart_pollutant_box(df, sel_pollutant), width="stretch")

        # Pollutant stats table
        with st.expander("📊 Pollutant Descriptive Statistics"):
            poll_stats = (
                df[POLLUTANTS]
                .describe()
                .T[["mean", "std", "min", "50%", "max"]]
                .rename(columns={"mean": "Mean", "std": "Std Dev", "min": "Min", "50%": "Median", "max": "Max"})
                .round(2)
            )
            st.dataframe(poll_stats, width="stretch")

    # =========================================================================
    # TAB 4 — DIAGNOSTIC INTELLIGENCE
    # =========================================================================
    with tab_diag:
        # Disclaimer banner
        st.markdown(
            """
            <div style="background:rgba(88,166,255,0.05);border:1px solid rgba(88,166,255,0.18);
                 border-radius:10px;padding:12px 16px;margin-bottom:20px;font-size:0.83rem;
                 color:#8b949e;line-height:1.55;">
              <strong style="color:#58a6ff;">ℹ️ Analytical note:</strong>
              All associations shown are <em>correlational</em>.
              They describe patterns in the data and do not establish causal mechanisms.
              Confounding factors — seasonality, shared emission sources, instrument calibration
              — may influence results.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Diagnostic Insights ───────────────────────────────────────────────
        st.markdown(
            """<div class="section-header">
              <div class="section-icon">💡</div>
              <span class="section-title">Diagnostic Insights</span>
              <span class="section-desc">Auto-generated from filtered data</span>
            </div>""",
            unsafe_allow_html=True,
        )
        for ins in insights:
            is_warn = ins.startswith("⚠️")
            card_cls = "insight-card warn" if is_warn else "insight-card"
            # Convert **bold** markdown to <strong> for HTML rendering
            ins_html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", ins)
            st.markdown(f'<div class="{card_cls}">{ins_html}</div>', unsafe_allow_html=True)

        # ── Correlation Analysis ─────────────────────────────────────────────
        st.markdown(
            """<div class="section-header">
              <div class="section-icon">📐</div>
              <span class="section-title">Pollutant–AQI Correlation</span>
              <span class="section-desc">Pearson r — statistical association only</span>
            </div>""",
            unsafe_allow_html=True,
        )

        dc1, dc2 = st.columns([3, 2])
        with dc1:
            st.plotly_chart(chart_aqi_corr_ranked(corr_ranked), width="stretch")
        with dc2:
            st.markdown(
                """<div class="section-header" style="margin-top:8px;">
                  <div class="section-icon">🔲</div>
                  <span class="section-title">Full Correlation Matrix</span>
                </div>""", unsafe_allow_html=True,
            )
            st.caption("Red = positive · Blue = negative")
            st.plotly_chart(chart_corr_heatmap(corr_matrix), width="stretch")

        with st.expander("📋 Ranked correlation table with p-values"):
            display_corr = corr_ranked.copy()
            display_corr["p-value"] = display_corr["p-value"].apply(
                lambda p: f"{p:.2e}" if not pd.isna(p) else "N/A"
            )
            st.dataframe(display_corr, width="stretch", hide_index=True)

        # ── Normal vs High-Risk Comparison ───────────────────────────────────
        st.markdown(
            """<div class="section-header">
              <div class="section-icon">⚖️</div>
              <span class="section-title">Normal vs High-Risk Day Comparison</span>
              <span class="section-desc">AQI ≤ 100 vs AQI > 200 — Mann-Whitney U significance</span>
            </div>""",
            unsafe_allow_html=True,
        )
        n_normal_days   = (df["AQI"] <= 100).sum()
        n_highrisk_days = (df["AQI"] > AQI_RISK_THRESHOLD).sum()
        st.caption(
            f"Normal days (AQI ≤ 100): {n_normal_days:,}  ·  "
            f"High-risk days (AQI > 200): {n_highrisk_days:,}.  "
            "Elevation Ratio = High-risk mean ÷ Normal mean."
        )
        # Contextual warning when one group is too small for a meaningful comparison
        if n_highrisk_days < 5:
            st.info(
                "⚠️ The current filter selection contains fewer than 5 high-risk days (AQI > 200). "
                "The High-Risk comparison bars will be empty or missing — "
                "try broadening the year range, city selection, or AQI category filter."
            )
        elif n_normal_days < 5:
            st.info(
                "⚠️ The current filter selection contains fewer than 5 normal days (AQI ≤ 100). "
                "The Normal comparison bars will be empty or missing."
            )
        st.plotly_chart(chart_normal_vs_highrisk(normal_vs_hr), width="stretch")
        with st.expander("📋 Full comparison table with significance test"):
            st.dataframe(normal_vs_hr, width="stretch", hide_index=True)

        # ── Pollutant-to-AQI Drilldown ────────────────────────────────────────
        st.markdown(
            """<div class="section-header">
              <div class="section-icon">🔍</div>
              <span class="section-title">Pollutant-to-AQI Drilldown</span>
              <span class="section-desc">Median concentration within each AQI category</span>
            </div>""",
            unsafe_allow_html=True,
        )
        st.caption(
            "A monotonically rising profile = strong positive association. "
            "Non-monotonic (e.g. O3 plateaus at Very Poor) = complex dynamics."
        )
        drill_pollutant = st.selectbox(
            "Select pollutant for drilldown",
            POLLUTANTS, index=0, key="diag_drill_pollutant",
        )
        band_df = diag_pollutant_by_aqi_band(df, drill_pollutant)
        st.plotly_chart(chart_pollutant_by_aqi_band(band_df, drill_pollutant), width="stretch")
        with st.expander("📋 Drilldown data table"):
            st.dataframe(band_df, width="stretch", hide_index=True)

        # ── Spike / Anomaly Analysis ───────────────────────────────────────────
        st.markdown(
            """<div class="section-header">
              <div class="section-icon">⚡</div>
              <span class="section-title">AQI Spike &amp; Anomaly Analysis</span>
              <span class="section-desc">30-day rolling z-score · threshold z &gt; 2.0</span>
            </div>""",
            unsafe_allow_html=True,
        )
        st.caption(
            f"A spike = AQI exceeds its {SPIKE_WINDOW}-day rolling mean by more than "
            f"{SPIKE_Z_THRESHOLD} SDs. Identifies unusually bad days relative to recent "
            "local conditions — not just absolute high values. Computed per city independently."
        )

        spike_counts   = diag_spike_summary(df_with_z)
        n_total_spikes = int(df_with_z["is_spike"].sum())

        if n_total_spikes == 0:
            st.info("No spike days detected for the current filter selection.")
        else:
            sp1, sp2, sp3 = st.columns(3)
            sp1.metric("Spike Days", f"{n_total_spikes:,}")
            valid_z = int(df_with_z["aqi_z_score"].notna().sum())
            sp2.metric("Spike Rate", f"{n_total_spikes / valid_z * 100:.1f}%")
            peak_spike_city = (
                df_with_z[df_with_z["is_spike"] == True]
                .groupby("City")["is_spike"].sum().idxmax()
            )
            sp3.metric("City with Most Spikes", peak_spike_city)

            st.plotly_chart(chart_spike_calendar(spike_counts), width="stretch")

            spike_profile = diag_spike_pollutant_profile(df_with_z)
            if not spike_profile.empty:
                st.markdown("**Pollutant Elevation During Spike Days**")
                st.caption("Elevation ratio > 1 means the pollutant is higher on spike days on average.")
                fig_ratio = go.Figure(go.Bar(
                    x=spike_profile["Elevation Ratio"],
                    y=spike_profile["Pollutant"],
                    orientation="h",
                    marker_color=[
                        "#da3633" if r >= 1.5 else "#f0883e" if r >= 1.2 else "#e3b341"
                        for r in spike_profile["Elevation Ratio"]
                    ],
                    text=spike_profile["Elevation Ratio"].apply(lambda v: f"{v:.2f}×"),
                    textposition="outside",
                    textfont_color="#e6edf3",
                ))
                fig_ratio.add_vline(x=1.0, line_color="#8b949e", line_dash="dash",
                                    annotation_text="Baseline (1.0×)",
                                    annotation_font_color="#8b949e")
                fig_ratio.update_layout(
                    title="Pollutant Elevation Ratio: Spike vs Non-Spike Days",
                    xaxis_title="Elevation Ratio",
                    yaxis_title="",
                    **PLOTLY_DARK_LAYOUT,
                    height=380,
                    yaxis=dict(autorange="reversed", gridcolor="#30363d", linecolor="#30363d"),
                )
                st.plotly_chart(fig_ratio, width="stretch")
                with st.expander("📋 Spike pollutant profile table"):
                    st.dataframe(spike_profile, width="stretch", hide_index=True)

        # ── Dominant Driver Shifts ─────────────────────────────────────────────
        st.markdown(
            """<div class="section-header">
              <div class="section-icon">🔀</div>
              <span class="section-title">Dominant Driver Shifts</span>
              <span class="section-desc">How the leading pollutant changes across seasons, years, and cities</span>
            </div>""",
            unsafe_allow_html=True,
        )
        st.caption(
            "A shift in the dominant driver signals structurally different pollution profiles "
            "— not just intensity differences."
        )
        shift_dim = st.radio(
            "Break down by:",
            options=["Season", "Year", "City"],
            horizontal=True,
            key="diag_shift_dim",
        )
        dim_key = shift_dim.lower()
        st.plotly_chart(
            chart_dominant_driver_shifts(driver_shifts, dim_key),
            width="stretch",
        )
        with st.expander(f"📋 Dominant driver table by {shift_dim}"):
            tbl = driver_shifts.get(f"by_{dim_key}", pd.DataFrame())
            if not tbl.empty:
                st.dataframe(tbl, width="stretch", hide_index=True)

    # =========================================================================
    # TAB 5 — DATA QUALITY
    # =========================================================================
    with tab_data:
        st.markdown(
            """<div class="section-header">
              <div class="section-icon">🗄️</div>
              <span class="section-title">Dataset Overview</span>
            </div>""",
            unsafe_allow_html=True,
        )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Raw Rows",    f"{qa['raw_rows']:,}")
        m2.metric("Clean Rows",  f"{qa['clean_rows']:,}")
        m3.metric("Cities",      qa["n_cities"])
        m4.metric("Date Range",  f"{qa['date_min']} → {qa['date_max']}")

        st.markdown(
            """<div class="section-header">
              <div class="section-icon">🔧</div>
              <span class="section-title">Cleaning Pipeline</span>
            </div>""",
            unsafe_allow_html=True,
        )
        cleaning_log = [
            {"Step": "Parse Date column",              "Action": "Converted to datetime; coerce errors → NaN",     "Affected": str(qa["invalid_dates_dropped"])},
            {"Step": "Drop (City, Date) duplicates",   "Action": "Removed exact duplicate rows",                   "Affected": str(qa["duplicates_dropped"])},
            {"Step": "Cap extreme AQI (> 999)",         "Action": "Capped at 999 — instrument spike suppression",  "Affected": str(qa["aqi_extreme_capped"])},
            {"Step": "Zero BTX/CO → NaN",              "Action": "Below-detection-limit zeros treated as missing", "Affected": str(qa["btx_co_zeros_to_nan"])},
            {"Step": "City-month median imputation",   "Action": "Filled pollutant NaN with city-month median",    "Affected": "all rows"},
            {"Step": "City median fallback",           "Action": "Residual NaN filled with city-level median",     "Affected": "all rows"},
            {"Step": "Global median fallback",         "Action": "Cities with zero data for a pollutant",          "Affected": "all rows"},
            {"Step": "AQI imputation",                 "Action": "Missing AQI filled with city-month median",      "Affected": "all rows"},
            {"Step": "AQI_Bucket rederived",           "Action": "Bucket recomputed from final AQI values",        "Affected": "all rows"},
        ]
        st.dataframe(pd.DataFrame(cleaning_log), width="stretch", hide_index=True)

        st.markdown(
            """<div class="section-header">
              <div class="section-icon">❓</div>
              <span class="section-title">Original Missing Value Rates</span>
            </div>""",
            unsafe_allow_html=True,
        )
        orig_missing = {
            "PM2.5": 15.57, "PM10": 37.72, "NO": 12.13, "NO2": 12.14, "NOx": 14.17,
            "NH3": 34.97,   "CO": 6.97,    "SO2": 13.05, "O3": 13.62,
            "Benzene": 19.04, "Toluene": 27.23, "Xylene": 61.32,
            "AQI": 15.85,   "AQI_Bucket": 15.85,
        }
        miss_df = pd.DataFrame(list(orig_missing.items()), columns=["Column", "Missing %"])
        fig_miss = px.bar(
            miss_df, x="Column", y="Missing %",
            color="Missing %",
            color_continuous_scale="Reds",
            title="Original Missing Value Rate per Column (%)",
        )
        fig_miss.update_layout(**PLOTLY_DARK_LAYOUT, coloraxis_showscale=False)
        st.plotly_chart(fig_miss, width="stretch")

        remaining = df[POLLUTANTS + ["AQI"]].isna().sum()
        if remaining.sum() == 0:
            st.success("✅ No missing values remain in pollutant or AQI columns after cleaning.")
        else:
            st.warning(f"⚠️ {remaining.sum()} missing values remain in the current selection.")
            st.dataframe(
                remaining[remaining > 0].reset_index().rename(columns={"index": "Column", 0: "Missing Count"}),
                width="stretch", hide_index=True,
            )

    # ─────────────────────────────────────────────────────────────────────────
    # FOOTER
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown(
    """
    <div style="
        margin-top:42px;
        padding:18px 0 12px;
        border-top:1px solid rgba(48,54,61,0.65);
        text-align:center;
        font-size:0.74rem;
        color:#596575;
        line-height:1.8;">
        <strong style="color:#8b949e;">
            Urban Air Quality Intelligence
        </strong>
        <span style="color:#303945; margin:0 8px;">•</span>
        CPCB India Dataset
        <span style="color:#303945; margin:0 8px;">•</span>
        2015–2020 
        <span style="color:#303945; margin:0 8px;">•</span>
        ✦ Built by Deeptanu Sen

        
    </div>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
