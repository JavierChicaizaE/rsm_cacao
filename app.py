"""
app.py
======
Aplicativo interactivo de Metodologia de Superficie de Respuesta (RSM)
Caso: Optimizacion del tostado de cacao Nacional (Ecuador)

Ejecutar localmente:
    streamlit run app.py

Autor: Proyecto RSM (version mejorada) - 2026
"""

import io
import logging
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Silenciar logs recurrentes de health check (/_stcore/health) en consola
logging.getLogger("streamlit").setLevel(logging.WARNING)

from rsm_core import (
    generate_ccd, generate_bbd, design_to_real,
    fit_model, anova_table, lack_of_fit_test, residual_diagnostics,
    canonical_analysis, steepest_ascent, ridge_analysis,
    ResponseGoal, individual_desirability, overall_desirability,
    optimize_desirability,
)
from data_generator import generate_dataset, FACTOR_RANGES, FACTOR_NAMES

st.set_page_config(
    page_title="RSM Tostado de Cacao Nacional",
    page_icon="📊",
    layout="wide",
)

# ---------------------------------------------------------------------------
# ESTILOS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;0,900&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
    #MainMenu, footer, header { visibility: hidden; }

    /* ═══ FONDO PRINCIPAL ═══ */
    .block-container { padding-top: 2rem !important; padding-left: 2rem !important; padding-right: 2rem !important; max-width: 1300px !important; }

    div[data-testid="stAppViewContainer"] {
        background: #F8FAFC !important;
    }
    div[data-testid="stMain"] { background: transparent !important; }

    /* ═══ SIDEBAR — Notion/Clean Style ═══ */
    section[data-testid="stSidebar"] {
        background: #FFFFFF !important;
        border-right: 1px solid #E2E8F0 !important;
    }
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] h3 { color: #1E293B !important; }

    /* Sidebar Cards */
    .sidebar-card {
        background: #F8FAFC !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        margin-bottom: 0.75rem !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .sidebar-card:hover {
        border-color: #8B5E34 !important;
        box-shadow: 0 4px 12px rgba(99,102,241,0.08) !important;
    }
    .sidebar-header {
        font-size: 0.68rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
        color: #5A351F !important;
        margin-bottom: 0.5rem !important;
    }
    .sidebar-desc {
        font-size: 0.85rem !important;
        color: #0F172A !important;
        font-weight: 700 !important;
        margin-bottom: 0.6rem !important;
    }
    .sidebar-meta { display: flex !important; flex-direction: column !important; gap: 0.35rem !important; }
    .meta-item {
        display: flex !important;
        justify-content: space-between !important;
        font-size: 0.75rem !important;
        border-bottom: 1px solid #E2E8F0 !important;
        padding-bottom: 0.25rem !important;
    }
    .meta-item:last-child { border-bottom: none !important; }
    .meta-label { color: #64748B !important; }
    .meta-val { color: #0F172A !important; font-weight: 600 !important; }
    .sidebar-pill-container { display: flex !important; flex-direction: column !important; gap: 0.4rem !important; margin-top: 0.4rem !important; }
    .sidebar-pill {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px !important;
        padding: 0.5rem 0.75rem !important;
        display: flex !important;
        flex-direction: column !important;
        transition: all 0.2s ease;
    }
    .sidebar-pill:hover {
        border-color: #8B5E34 !important;
        box-shadow: 0 2px 8px rgba(99,102,241,0.08) !important;
    }
    .pill-title { font-size: 0.8rem !important; font-weight: 600 !important; color: #1E293B !important; }
    .pill-subtitle { font-size: 0.72rem !important; color: #64748B !important; margin-top: 1px; }

    /* ═══ HERO ═══ */
    .hero {
        background: linear-gradient(135deg, #2E1B10 0%, #5A351F 40%, #8B5E34 75%, #D8B25C 100%);
        border-radius: 20px;
        padding: 0;
        margin-bottom: 1.75rem;
        display: grid;
        grid-template-columns: 1fr 1fr;
        min-height: 200px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 12px 40px rgba(79,70,229,0.4), 0 4px 12px rgba(79,70,229,0.25);
    }
    .hero::before {
        content: "";
        position: absolute; inset: 0;
        background-image: radial-gradient(circle, rgba(255,255,255,0.08) 1px, transparent 1px);
        background-size: 24px 24px;
        pointer-events: none;
        z-index: 0;
    }
    .hero::after {
        content: "";
        position: absolute; top: -80px; right: -80px;
        width: 350px; height: 350px;
        background: radial-gradient(circle, rgba(165,180,252,0.25) 0%, transparent 65%);
        pointer-events: none;
        z-index: 0;
    }
    .hero-left {
        padding: 2.5rem 2.5rem 2.5rem 3rem;
        z-index: 2;
        display: flex;
        flex-direction: column;
        justify-content: center;
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    .hero-tag {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(255,255,255,0.15);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.68rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: rgba(255,255,255,0.9);
        margin-bottom: 14px;
        width: fit-content;
        backdrop-filter: blur(4px);
    }
    .hero-tag::before {
        content: '';
        width: 6px; height: 6px;
        background: #A5F3FC;
        border-radius: 50%;
        box-shadow: 0 0 6px #A5F3FC;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.6; transform: scale(0.85); }
    }
    .hero-title {
        font-weight: 900;
        font-size: 2.1rem;
        line-height: 1.15;
        color: #FFFFFF !important;
        margin: 0 0 10px 0;
        letter-spacing: -0.03em;
    }
    .hero-title span {
        display: block;
        font-size: 1.15rem;
        font-weight: 600;
        color: #F8FAFC !important;
        letter-spacing: -0.01em;
        margin-top: 6px;
    }
    .hero-desc {
        font-size: 0.9rem;
        color: #F1F5F9 !important;
        line-height: 1.65;
        margin: 0;
        max-width: 440px;
        font-weight: 400;
    }
    .hero-right {
        padding: 2rem 2.5rem 2rem 2rem;
        z-index: 2;
        display: flex;
        flex-direction: column;
        justify-content: center;
        gap: 0.85rem;
    }
    .hero-group-label {
        font-size: 0.65rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        color: rgba(255,255,255,0.5);
        margin-bottom: 2px;
    }
    .hero-members-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
    }
    .hero-member-card {
        background: rgba(255,255,255,0.1);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 10px;
        padding: 8px 12px;
        backdrop-filter: blur(6px);
        transition: background 0.2s ease, border-color 0.2s ease;
    }
    .hero-member-card:hover {
        background: rgba(255,255,255,0.18);
        border-color: rgba(255,255,255,0.3);
    }
    .hero-member-name {
        font-size: 0.82rem;
        font-weight: 600;
        color: #FFFFFF;
        line-height: 1.2;
    }
    .hero-divider {
        width: 100%;
        height: 1px;
        background: rgba(255,255,255,0.12);
        margin: 2px 0;
    }
    .hero-visual { display: none; }

    /* ═══ STEPPER — Notion Style ═══ */
    .stepper {
        display: flex; justify-content: space-between;
        margin: 0 0 1.75rem 0; padding: 1rem 1.5rem;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }
    .step-item { flex: 1; text-align: center; position: relative; font-family: 'Inter', sans-serif; }
    .step-item:not(:last-child)::after {
        content: ""; position: absolute; top: 14px; left: 55%;
        width: 90%; height: 2px;
        background: #E2E8F0; z-index: 0;
    }
    .step-item.done:not(:last-child)::after {
        background: linear-gradient(90deg, #10B981, #A7F3D0);
    }
    .step-circle {
        width: 30px; height: 30px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        margin: 0 auto 6px auto; font-weight: 700; font-size: 0.75rem;
        position: relative; z-index: 1;
        border: 2px solid #E2E8F0;
        background: #F8FAFC; color: #94A3B8;
        transition: all 0.3s ease;
    }
    .step-item.done .step-circle {
        background: #10B981; border-color: #10B981; color: white;
        box-shadow: 0 2px 10px rgba(16,185,129,0.25);
    }
    .step-item.ready .step-circle {
        background: #FFF7E9; border-color: #8B5E34; color: #8B5E34;
        box-shadow: 0 2px 10px rgba(99,102,241,0.15);
        font-weight: 800;
    }
    .step-label { font-size: 0.72rem; font-weight: 600; color: #94A3B8; }
    .step-item.done .step-label { color: #10B981; }
    .step-item.ready .step-label { color: #8B5E34; }

    /* ═══ TABS ═══ */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0; border-bottom: 2px solid #E2E8F0 !important;
        background: transparent; padding-bottom: 0;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Inter', sans-serif; font-weight: 500; font-size: 0.88rem;
        color: #64748B !important; padding: 10px 20px;
        background: transparent; border: none;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #0F172A !important;
        background: #F1F5F9;
    }
    .stTabs [aria-selected="true"] {
        color: #5A351F !important;
        background: transparent !important;
        border-bottom: 2px solid #8B5E34 !important;
        font-weight: 700 !important;
        margin-bottom: -2px;
    }

    /* ═══ HEADINGS ═══ */
    h1, h2, h3, h4 {
        font-family: 'Inter', sans-serif !important;
        color: #0F172A !important;
        font-weight: 700 !important;
        letter-spacing: -0.025em;
    }
    h2 { font-size: 1.35rem !important; margin-bottom: 0.75rem; }
    h3 { font-size: 1.05rem !important; color: #1E293B !important; }

    /* ═══ METRIC CARDS — SaaS Style (Stripe/Linear) ═══ */
    div[data-testid="stMetric"] {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 16px !important;
        padding: 1.5rem 1.75rem !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.03), 0 2px 4px -1px rgba(0, 0, 0, 0.02) !important;
        border-left: 4px solid #8B5E34 !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    div[data-testid="stMetric"]:hover {
        border-color: #8B5E34 !important;
        border-left-width: 6px !important;
        box-shadow: 0 10px 15px -3px rgba(99, 102, 241, 0.08), 0 4px 6px -2px rgba(99, 102, 241, 0.04) !important;
        transform: translateY(-3px);
    }
    div[data-testid="stMetricValue"] {
        font-weight: 800 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 2.25rem !important;
        letter-spacing: -0.05em !important;
        color: #0F172A !important;
        margin-top: 4px !important;
    }
    div[data-testid="stMetricLabel"] {
        font-weight: 700 !important;
        font-size: 0.75rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.10em !important;
        color: #5A351F !important;
    }
    div[data-testid="stMetricDelta"] {
        font-size: 0.8rem !important;
        font-weight: 600 !important;
    }

    /* ═══ BOTONES DIRECT OVERRIDES (Garantiza visibilidad de letras) ═══ */
    button[data-testid^="baseButton-"] {
        font-family: 'Inter', sans-serif !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.86rem !important;
        padding: 0.55rem 1.35rem !important;
        transition: all 0.2s ease !important;
        cursor: pointer !important;
    }
    button[data-testid="baseButton-primary"] {
        background: #8B5E34 !important;
        border: 1px solid #5A351F !important;
        color: #FFFFFF !important;
        box-shadow: 0 2px 8px rgba(99,102,241,0.3) !important;
    }
    button[data-testid="baseButton-primary"]:hover {
        background: #5A351F !important;
        box-shadow: 0 4px 16px rgba(99,102,241,0.4) !important;
        transform: translateY(-1px) !important;
    }
    button[data-testid="baseButton-secondary"] {
        background: #FFFFFF !important;
        border: 1px solid #D1D5DB !important;
        color: #374151 !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
    }
    button[data-testid="baseButton-secondary"]:hover {
        background: #F9FAFB !important;
        border-color: #8B5E34 !important;
        color: #5A351F !important;
        box-shadow: 0 2px 8px rgba(99,102,241,0.12) !important;
    }
    button[data-testid="baseButton-download"] {
        background: #F0FDF4 !important;
        border: 1px solid #86EFAC !important;
        color: #166534 !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
    }
    button[data-testid="baseButton-download"]:hover {
        background: #DCFCE7 !important;
        border-color: #22C55E !important;
        color: #15803D !important;
    }

    /* ═══ INPUTS / FORMS ═══ */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background: #FFFFFF !important;
        border: 1.5px solid #CBD5E1 !important;
        border-radius: 8px !important;
        color: #0F172A !important;
    }
    div[data-baseweb="select"]:focus-within > div,
    div[data-baseweb="input"]:focus-within > div {
        border-color: #8B5E34 !important;
        box-shadow: 0 0 0 3px rgba(99,102,241,0.12) !important;
    }
    div[data-baseweb="select"] *, div[data-baseweb="input"] input {
        color: #0F172A !important;
    }
    .stRadio label span,
    .stSlider label,
    .stSelectbox label,
    .stNumberInput label,
    .stMultiSelect label,
    .stFileUploader label {
        color: #334155 !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
    }
    /* Radio options */
    .stRadio [data-testid="stMarkdownContainer"] p {
        color: #334155 !important;
        font-size: 0.9rem !important;
    }
    .stRadio div[role="radiogroup"] label {
        background: #FFFFFF;
        border: 1.5px solid #E2E8F0;
        border-radius: 8px;
        padding: 0.45rem 1rem;
        transition: all 0.2s ease;
    }
    .stRadio div[role="radiogroup"] label:hover { border-color: #8B5E34; background: #F8FAFC; }

    /* ═══ EXPANDERS ═══ */
    div[data-testid="stExpander"] {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 12px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02) !important;
        margin-bottom: 0.5rem;
    }
    div[data-testid="stExpander"] summary {
        color: #1E293B !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }

    /* ═══ DATAFRAMES ═══ */
    div[data-testid="stDataFrame"] {
        border: 1px solid #E2E8F0 !important;
        border-radius: 12px !important;
        overflow: hidden;
        box-shadow: 0 1px 4px rgba(0,0,0,0.02) !important;
    }

    /* ═══ ALERTAS ═══ */
    div[data-testid="stAlert"] { border-radius: 12px !important; border: 1px solid #E2E8F0 !important; }

    /* ═══ HOWTO BOX ═══ */
    .howto {
        background: #FFF7E9;
        border-left: 4px solid #8B5E34;
        border-radius: 0 12px 12px 0;
        padding: 1.1rem 1.4rem;
        margin-bottom: 1.5rem;
        border-top: 1px solid #C7D2FE;
        border-right: 1px solid #C7D2FE;
        border-bottom: 1px solid #C7D2FE;
    }
    .howto-title { font-weight: 700; color: #4338CA; font-size: 0.9rem; margin-bottom: 4px; }
    .howto-body { color: #5A351F; font-size: 0.86rem; line-height: 1.6; font-weight: 500; }

    /* ═══ DIVIDERS Y TEXTO GLOBAL ═══ */
    div[data-testid="stMarkdownContainer"] hr { border-color: #E2E8F0 !important; margin: 1.5rem 0 !important; }
    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stMarkdownContainer"] li,
    div[data-testid="stMarkdownContainer"] strong,
    div[data-testid="stMarkdownContainer"] span {
        color: #334155 !important;
        line-height: 1.65;
        font-size: 0.9rem;
    }
    .stCaption p, small { color: #94A3B8 !important; font-size: 0.78rem !important; }

    /* ═══ SCROLLBAR ═══ */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #F1F5F9; }
    ::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #8B5E34; }

    /* PREMIUM CACAO THEME */
    :root {
        --cacao-ink: #26180F;
        --cacao-muted: #756250;
        --cacao-line: #E8D9C5;
        --cacao-paper: #FFFCF7;
        --cacao-cream: #F7F1E8;
        --cacao-soft: #EFE1CF;
        --cacao: #5A351F;
        --cacao-2: #8B5E34;
        --cacao-gold: #C89B3C;
        --leaf: #2F6B4F;
        --berry: #8E3E32;
        --shadow-soft: 0 18px 45px rgba(67, 42, 23, 0.10);
        --shadow-card: 0 10px 28px rgba(67, 42, 23, 0.08);
    }

    div[data-testid="stAppViewContainer"] {
        background:
            linear-gradient(180deg, rgba(255,252,247,0.96), rgba(247,241,232,0.96)),
            repeating-linear-gradient(135deg, rgba(139,94,52,0.045) 0 1px, transparent 1px 18px) !important;
    }
    .block-container {
        padding-top: 1.45rem !important;
        padding-bottom: 2.5rem !important;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2E1B10 0%, #3D2416 58%, #25170F 100%) !important;
        border-right: 1px solid rgba(255,255,255,0.10) !important;
        box-shadow: 16px 0 40px rgba(47, 28, 16, 0.16) !important;
    }
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] h3 { color: #F8EBDD !important; }
    .sidebar-card {
        background: rgba(255, 248, 236, 0.08) !important;
        border: 1px solid rgba(232, 217, 197, 0.18) !important;
        border-radius: 10px !important;
        box-shadow: none !important;
        backdrop-filter: blur(10px);
    }
    .sidebar-card:hover {
        border-color: rgba(200, 155, 60, 0.62) !important;
        box-shadow: 0 14px 30px rgba(0,0,0,0.16) !important;
    }
    .sidebar-header {
        color: #D8B25C !important;
        letter-spacing: 0.12em !important;
    }
    .sidebar-desc, .meta-val, .pill-title { color: #FFF8EC !important; }
    .meta-label, .pill-subtitle { color: #D9C4AA !important; }
    .meta-item { border-bottom-color: rgba(232, 217, 197, 0.16) !important; }
    .sidebar-pill {
        background: rgba(255, 252, 247, 0.08) !important;
        border-color: rgba(232, 217, 197, 0.14) !important;
        border-radius: 8px !important;
    }
    .sidebar-pill:hover {
        border-color: rgba(216,178,92,0.55) !important;
        box-shadow: 0 8px 18px rgba(0,0,0,0.12) !important;
    }

    .hero {
        background:
            linear-gradient(118deg, rgba(45,26,14,0.98) 0%, rgba(82,47,25,0.98) 48%, rgba(137,83,40,0.96) 100%),
            repeating-linear-gradient(45deg, rgba(255,255,255,0.08) 0 1px, transparent 1px 20px) !important;
        border: 1px solid rgba(255, 238, 203, 0.18) !important;
        border-radius: 14px !important;
        min-height: 220px !important;
        box-shadow: 0 28px 80px rgba(74, 43, 22, 0.24), inset 0 1px 0 rgba(255,255,255,0.14) !important;
    }
    .hero::before {
        background:
            linear-gradient(90deg, rgba(255,255,255,0.09) 0 1px, transparent 1px 100%),
            linear-gradient(0deg, rgba(255,255,255,0.06) 0 1px, transparent 1px 100%) !important;
        background-size: 42px 42px !important;
        opacity: 0.32;
    }
    .hero::after {
        top: auto !important; right: 0 !important; bottom: 0 !important;
        width: 42% !important; height: 100% !important;
        background:
            linear-gradient(135deg, transparent 0 42%, rgba(255,255,255,0.10) 42% 43%, transparent 43% 100%),
            linear-gradient(45deg, transparent 0 54%, rgba(216,178,92,0.22) 54% 55%, transparent 55% 100%) !important;
    }
    .hero-left { border-right-color: rgba(255, 238, 203, 0.15) !important; }
    .hero-tag {
        background: rgba(216, 178, 92, 0.16) !important;
        border-color: rgba(216, 178, 92, 0.34) !important;
        color: #F6D984 !important;
        border-radius: 999px !important;
    }
    .hero-tag::before { background: #D8B25C !important; box-shadow: 0 0 10px #D8B25C !important; }
    .hero-title {
        color: #FFF8EC !important;
        letter-spacing: 0 !important;
        text-wrap: balance;
    }
    .hero-title span, .hero-desc { color: #F5E1C4 !important; }
    .hero-group-label { color: #D8B25C !important; }
    .hero-member-card {
        background: rgba(255, 252, 247, 0.10) !important;
        border-color: rgba(255, 238, 203, 0.18) !important;
        border-radius: 8px !important;
    }
    .hero-member-card:hover {
        background: rgba(255, 252, 247, 0.16) !important;
        border-color: rgba(216, 178, 92, 0.42) !important;
    }

    .stepper {
        background: rgba(255,252,247,0.82) !important;
        border-color: var(--cacao-line) !important;
        border-radius: 12px !important;
        box-shadow: var(--shadow-card) !important;
        backdrop-filter: blur(12px);
    }
    .step-item:not(:last-child)::after { background: #E7D8C3 !important; }
    .step-item.done:not(:last-child)::after { background: linear-gradient(90deg, var(--leaf), #9DC1A8) !important; }
    .step-circle {
        background: #FFF9EF !important;
        border-color: #DEC8AA !important;
        color: #AD977C !important;
    }
    .step-item.done .step-circle {
        background: var(--leaf) !important;
        border-color: var(--leaf) !important;
        box-shadow: 0 8px 18px rgba(47,107,79,0.22) !important;
    }
    .step-item.ready .step-circle {
        background: #FFF3D8 !important;
        border-color: var(--cacao-gold) !important;
        color: #7A551F !important;
        box-shadow: 0 8px 18px rgba(200,155,60,0.22) !important;
    }
    .step-label { color: #A18B73 !important; }
    .step-item.done .step-label { color: var(--leaf) !important; }
    .step-item.ready .step-label { color: var(--cacao-2) !important; }

    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,252,247,0.72) !important;
        border: 1px solid var(--cacao-line) !important;
        border-radius: 10px !important;
        padding: 0.25rem !important;
        gap: 0.25rem !important;
        box-shadow: 0 8px 22px rgba(67,42,23,0.06) !important;
    }
    .stTabs [data-baseweb="tab"] {
        color: var(--cacao-muted) !important;
        border-radius: 8px !important;
        padding: 0.62rem 1rem !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: #F4E8D8 !important;
        color: var(--cacao-ink) !important;
    }
    .stTabs [aria-selected="true"] {
        background: #3A2215 !important;
        color: #FFF8EC !important;
        border-bottom: 0 !important;
        margin-bottom: 0 !important;
        box-shadow: 0 8px 20px rgba(58,34,21,0.18) !important;
    }

    h1, h2, h3, h4 { color: var(--cacao-ink) !important; letter-spacing: 0 !important; }
    h2, h3 { font-weight: 800 !important; }

    div[data-testid="stMetric"],
    div[data-testid="stExpander"],
    div[data-testid="stDataFrame"],
    div[data-testid="stAlert"] {
        background: rgba(255,252,247,0.92) !important;
        border-color: var(--cacao-line) !important;
        border-radius: 10px !important;
        box-shadow: var(--shadow-card) !important;
    }
    div[data-testid="stMetric"] {
        border-left: 4px solid var(--cacao-gold) !important;
        padding: 1.25rem 1.35rem !important;
    }
    div[data-testid="stMetric"]:hover {
        border-color: #D6BD8F !important;
        box-shadow: var(--shadow-soft) !important;
        transform: translateY(-2px);
    }
    div[data-testid="stMetricValue"] { color: var(--cacao-ink) !important; letter-spacing: 0 !important; }
    div[data-testid="stMetricLabel"] { color: var(--cacao-2) !important; letter-spacing: 0.08em !important; }

    button[data-testid^="baseButton-"] {
        border-radius: 8px !important;
        letter-spacing: 0 !important;
    }
    button[data-testid="baseButton-primary"] {
        background: #5A351F !important;
        border-color: #5A351F !important;
        color: #FFF8EC !important;
        box-shadow: 0 12px 24px rgba(90,53,31,0.18) !important;
    }
    button[data-testid="baseButton-primary"]:hover {
        background: #744322 !important;
        box-shadow: 0 16px 30px rgba(90,53,31,0.24) !important;
    }
    button[data-testid="baseButton-secondary"] {
        background: #FFFCF7 !important;
        border-color: #D9C4AA !important;
        color: #4B2B17 !important;
    }
    button[data-testid="baseButton-secondary"]:hover {
        background: #FFF3D8 !important;
        border-color: var(--cacao-gold) !important;
        color: #3A2215 !important;
    }
    button[data-testid="baseButton-download"] {
        background: #EDF7EF !important;
        border-color: #A7C8AD !important;
        color: #24583F !important;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    textarea {
        background: #FFFCF7 !important;
        border-color: #D9C4AA !important;
        border-radius: 8px !important;
        color: var(--cacao-ink) !important;
    }
    div[data-baseweb="select"]:focus-within > div,
    div[data-baseweb="input"]:focus-within > div,
    textarea:focus {
        border-color: var(--cacao-2) !important;
        box-shadow: 0 0 0 3px rgba(139,94,52,0.14) !important;
    }
    .stRadio label span,
    .stSlider label,
    .stSelectbox label,
    .stNumberInput label,
    .stMultiSelect label,
    .stFileUploader label { color: #4B392A !important; }
    .stRadio div[role="radiogroup"] label {
        background: #FFFCF7 !important;
        border-color: #E1CDB4 !important;
        border-radius: 8px !important;
    }
    .stRadio div[role="radiogroup"] label:hover {
        border-color: var(--cacao-gold) !important;
        background: #FFF7E9 !important;
    }

    .howto {
        background: #FFF7E9 !important;
        border-left-color: var(--cacao-gold) !important;
        border-top-color: #E7D0A1 !important;
        border-right-color: #E7D0A1 !important;
        border-bottom-color: #E7D0A1 !important;
        border-radius: 0 10px 10px 0 !important;
        box-shadow: 0 8px 18px rgba(200,155,60,0.08);
    }
    .howto-title { color: #7A551F !important; }
    .howto-body { color: #5F4A32 !important; }
    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stMarkdownContainer"] li,
    div[data-testid="stMarkdownContainer"] strong,
    div[data-testid="stMarkdownContainer"] span { color: #4B392A !important; }
    .stCaption p, small { color: #88735C !important; }
    div[data-testid="stMarkdownContainer"] hr { border-color: var(--cacao-line) !important; }
    ::-webkit-scrollbar-track { background: #EFE1CF !important; }
    ::-webkit-scrollbar-thumb { background: #B89B79 !important; }
    ::-webkit-scrollbar-thumb:hover { background: var(--cacao-2) !important; }

    /* PREMIUM CACAO V2: contraste y acabado final */
    .block-container {
        max-width: 1440px !important;
        padding-left: 2.6rem !important;
        padding-right: 2.6rem !important;
    }
    div[data-testid="stAppViewContainer"] {
        background:
            linear-gradient(180deg, #FFFDF8 0%, #F8F0E5 54%, #F2E3D1 100%) !important;
    }

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(180deg, #25150D 0%, #321D11 48%, #1A100B 100%) !important;
    }
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] div,
    section[data-testid="stSidebar"] .sidebar-card,
    section[data-testid="stSidebar"] .sidebar-card * {
        color: #FFF7EA !important;
        opacity: 1 !important;
    }
    section[data-testid="stSidebar"] .sidebar-card {
        background: rgba(255, 247, 234, 0.105) !important;
        border: 1px solid rgba(232, 202, 151, 0.30) !important;
        border-radius: 12px !important;
    }
    section[data-testid="stSidebar"] .sidebar-header {
        color: #F1C765 !important;
        font-size: 0.72rem !important;
    }
    section[data-testid="stSidebar"] .sidebar-desc,
    section[data-testid="stSidebar"] .pill-title,
    section[data-testid="stSidebar"] .meta-val {
        color: #FFFFFF !important;
        font-weight: 800 !important;
    }
    section[data-testid="stSidebar"] .meta-label,
    section[data-testid="stSidebar"] .pill-subtitle {
        color: #E8D4B8 !important;
    }
    section[data-testid="stSidebar"] .sidebar-pill {
        background: rgba(255,255,255,0.085) !important;
        border-color: rgba(241,199,101,0.25) !important;
    }
    section[data-testid="stSidebar"] .sidebar-note {
        background: rgba(241,199,101,0.12) !important;
        border-color: rgba(241,199,101,0.42) !important;
    }

    .hero {
        min-height: 245px !important;
        grid-template-columns: minmax(0, 1.05fr) minmax(360px, 0.95fr) !important;
        background:
            linear-gradient(105deg, rgba(36,20,11,0.98) 0%, rgba(61,33,17,0.98) 45%, rgba(120,70,32,0.96) 100%) !important;
        border: 1px solid rgba(209, 162, 74, 0.42) !important;
        box-shadow: 0 26px 70px rgba(53,31,17,0.22), 0 1px 0 rgba(255,255,255,0.18) inset !important;
    }
    .hero::before {
        background:
            linear-gradient(90deg, rgba(255,255,255,0.07) 1px, transparent 1px),
            linear-gradient(0deg, rgba(255,255,255,0.055) 1px, transparent 1px) !important;
        background-size: 56px 56px !important;
        opacity: 0.42 !important;
    }
    .hero::after {
        width: 46% !important;
        background:
            linear-gradient(135deg, transparent 0 44%, rgba(226,178,85,0.22) 44% 45%, transparent 45% 100%),
            radial-gradient(circle at 76% 28%, rgba(241,199,101,0.22), transparent 34%) !important;
    }
    .hero-left {
        padding: 2.65rem 3rem !important;
    }
    .hero-right {
        padding: 2.4rem 3rem !important;
    }
    .hero,
    .hero [data-testid="stMarkdownContainer"],
    .hero p,
    .hero span,
    .hero div,
    .hero strong,
    .hero-title,
    .hero-title span,
    .hero-desc,
    .hero-member-name {
        color: #FFF8EA !important;
        opacity: 1 !important;
    }
    .hero-title {
        font-size: clamp(2rem, 2.2vw, 2.85rem) !important;
        line-height: 1.05 !important;
        font-weight: 900 !important;
        max-width: 650px !important;
        text-shadow: 0 2px 18px rgba(0,0,0,0.22);
    }
    .hero-title span {
        color: #F6D384 !important;
        font-size: clamp(1rem, 1.12vw, 1.25rem) !important;
        font-weight: 750 !important;
    }
    .hero-desc {
        color: #F3E3CC !important;
        font-size: 0.98rem !important;
        line-height: 1.72 !important;
        max-width: 590px !important;
    }
    .hero-tag,
    .hero-tag * {
        color: #FFE49B !important;
        font-weight: 900 !important;
    }
    .hero-group-label {
        color: #FFE49B !important;
        font-size: 0.74rem !important;
        opacity: 1 !important;
    }
    .hero-member-card {
        background: rgba(255,255,255,0.14) !important;
        border: 1px solid rgba(255,228,155,0.26) !important;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.16) !important;
    }

    .stepper {
        border: 1px solid #E8D4B4 !important;
        background: rgba(255,253,248,0.96) !important;
        box-shadow: 0 14px 34px rgba(64,38,20,0.085) !important;
    }
    .step-circle {
        width: 34px !important;
        height: 34px !important;
        font-size: 0.82rem !important;
    }
    .step-label {
        color: #876D53 !important;
        font-weight: 800 !important;
        font-size: 0.78rem !important;
    }
    .step-item.ready .step-label,
    .step-item.ready .step-circle {
        color: #8A5B24 !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        background: #FFFDF8 !important;
        border-color: #E3CDAF !important;
        box-shadow: 0 16px 34px rgba(64,38,20,0.075) !important;
    }
    .stTabs [data-baseweb="tab"],
    .stTabs [data-baseweb="tab"] p,
    .stTabs [data-baseweb="tab"] span {
        color: #5D4734 !important;
        font-weight: 750 !important;
        opacity: 1 !important;
    }
    .stTabs [data-baseweb="tab"]:hover,
    .stTabs [data-baseweb="tab"]:hover p,
    .stTabs [data-baseweb="tab"]:hover span {
        color: #24150D !important;
        background: #F4E6D5 !important;
    }
    .stTabs [aria-selected="true"],
    .stTabs [aria-selected="true"] p,
    .stTabs [aria-selected="true"] span {
        background: #2A180E !important;
        color: #FFF8EA !important;
        opacity: 1 !important;
    }

    button[data-testid^="baseButton-"],
    button[data-testid^="baseButton-"] *,
    div[data-testid="stDownloadButton"] button,
    div[data-testid="stDownloadButton"] button * {
        opacity: 1 !important;
        text-shadow: none !important;
    }
    button[data-testid="baseButton-primary"],
    button[data-testid="baseButton-primary"] * {
        color: #FFFFFF !important;
    }
    button[data-testid="baseButton-secondary"],
    button[data-testid="baseButton-secondary"] * {
        color: #2A180E !important;
    }
    button[data-testid="baseButton-download"],
    button[data-testid="baseButton-download"] * {
        color: #12452F !important;
    }

    .howto {
        background: #FFF8EA !important;
        border: 1px solid #E5C68C !important;
        border-left: 6px solid #C7932F !important;
        border-radius: 12px !important;
        box-shadow: 0 14px 34px rgba(138,91,36,0.08) !important;
    }
    .howto,
    .howto *,
    .howto-title,
    .howto-body {
        color: #3B291B !important;
        opacity: 1 !important;
    }
    .howto-title {
        color: #7A4E16 !important;
        font-size: 0.98rem !important;
    }
    .howto-body {
        font-size: 0.96rem !important;
        line-height: 1.7 !important;
    }

    div[data-testid="stAlert"] *,
    div[data-testid="stMetric"] *,
    div[data-testid="stExpander"] *,
    div[data-testid="stDataFrame"] * {
        opacity: 1 !important;
    }

    @media (max-width: 900px) {
        .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        .hero {
            grid-template-columns: 1fr !important;
        }
        .hero-left,
        .hero-right {
            padding: 1.6rem !important;
            border-right: 0 !important;
        }
        .hero-members-grid {
            grid-template-columns: 1fr !important;
        }
        .stepper {
            overflow-x: auto !important;
            gap: 1rem !important;
        }
        .step-item {
            min-width: 110px !important;
        }
    }

    div[data-testid="stMarkdownContainer"] .hero .hero-title,
    div[data-testid="stMarkdownContainer"] .hero .hero-title *,
    div[data-testid="stMarkdownContainer"] .hero .hero-title span,
    div[data-testid="stMarkdownContainer"] .hero .hero-desc,
    div[data-testid="stMarkdownContainer"] .hero .hero-desc * {
        color: #FFFFFF !important;
        opacity: 1 !important;
        visibility: visible !important;
    }
    div[data-testid="stMarkdownContainer"] .hero .hero-title {
        text-shadow: 0 3px 18px rgba(0,0,0,0.42) !important;
    }
    div[data-testid="stMarkdownContainer"] .hero .hero-title span {
        color: #FFE39A !important;
    }
    div[data-testid="stMarkdownContainer"] .hero .hero-desc {
        color: #FFF1D7 !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# PLANTILLA DE GRAFICOS (paleta premium cacao consistente en toda la app)
# ---------------------------------------------------------------------------
import plotly.io as pio

REACT_COLORWAY = ["#8B5E34", "#2F6B4F", "#C89B3C", "#8E3E32", "#5A351F", "#6A6F3A"]
REACT_SCALE = [[0, "#FFF7E9"], [0.35, "#D8B25C"], [0.68, "#8B5E34"], [1, "#3A2215"]]

pio.templates["cacao_premium"] = go.layout.Template(
    layout=go.Layout(
        font=dict(family="Inter, sans-serif", color="#4B392A", size=12),
        title=dict(font=dict(family="Inter, sans-serif", size=15, color="#26180F", weight=700)),
        colorway=REACT_COLORWAY,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFCF7",
        xaxis=dict(
            gridcolor="#EFE1CF", zerolinecolor="#D9C4AA",
            tickfont=dict(color="#756250", size=11),
            title_font=dict(color="#4B392A", size=12)
        ),
        yaxis=dict(
            gridcolor="#EFE1CF", zerolinecolor="#D9C4AA",
            tickfont=dict(color="#756250", size=11),
            title_font=dict(color="#4B392A", size=12)
        ),
        legend=dict(
            bgcolor="rgba(255,252,247,0.92)",
            bordercolor="#E8D9C5",
            borderwidth=1,
            font=dict(color="#4B392A", size=12)
        ),
    )
)
pio.templates.default = "cacao_premium"


RESPONSE_META = {
    "Polifenoles_mgGAE_g": {"label": "Polifenoles totales (mg GAE/g)", "goal_default": "maximize"},
    "DPPH_pct_inhibicion": {"label": "Actividad antioxidante DPPH (% inhibición)", "goal_default": "maximize"},
    "Indice_pardeamiento": {"label": "Índice de pardeamiento (color)", "goal_default": "target"},
    "Puntaje_sensorial": {"label": "Puntaje sensorial (1-9)", "goal_default": "maximize"},
}

# ---------------------------------------------------------------------------
# SIDEBAR — ficha del caso
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-card">
        <div class="sidebar-header">Caso de Estudio</div>
        <div class="sidebar-desc">Tostado de Cacao Nacional (Ecuador)</div>
        <div class="sidebar-meta">
            <div class="meta-item"><span class="meta-label">Materia Prima</span><span class="meta-val">Cacao Fino de Aroma</span></div>
            <div class="meta-item"><span class="meta-label">Operación</span><span class="meta-val">Tostado térmico</span></div>
            <div class="meta-item"><span class="meta-label">Objetivo</span><span class="meta-val">Maximizar calidad</span></div>
        </div>
    </div>
    
    <div class="sidebar-card">
        <div class="sidebar-header">Factores de Proceso</div>
        <div class="sidebar-pill-container">
    """, unsafe_allow_html=True)
    
    for f, (lo, hi) in FACTOR_RANGES.items():
        unidad = "°C" if f == "Temperatura" else ("min" if f == "Tiempo" else "%")
        st.markdown(f"""
            <div class="sidebar-pill">
                <span class="pill-title">{f}</span>
                <span class="pill-subtitle">Rango: {lo} – {hi} {unidad}</span>
            </div>
        """, unsafe_allow_html=True)
        
    st.markdown("""
        </div>
    </div>
    
    <div class="sidebar-card">
        <div class="sidebar-header">Respuestas de Calidad</div>
        <div class="sidebar-pill-container">
    """, unsafe_allow_html=True)
    
    for r, meta in RESPONSE_META.items():
        st.markdown(f"""
            <div class="sidebar-pill">
                <span class="pill-title">{meta['label']}</span>
            </div>
        """, unsafe_allow_html=True)
        
    st.markdown("""
        </div>
    </div>
    
    <div class="sidebar-card sidebar-note">
        <div class="sidebar-header">Nota del Modelo</div>
        <div style="font-size: 0.74rem; line-height: 1.55;">
            Dataset sintético calibrado con tendencias científicas de tostado. No reemplaza determinaciones reales de laboratorio.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# ESTADO DE SESION
# ---------------------------------------------------------------------------
if "df" not in st.session_state:
    st.session_state.df = None
if "models" not in st.session_state:
    st.session_state.models = {}
if "factor_names" not in st.session_state:
    st.session_state.factor_names = FACTOR_NAMES
if "factor_ranges" not in st.session_state:
    st.session_state.factor_ranges = FACTOR_RANGES


def get_coded_df():
    """Devuelve el dataframe con columnas de factores en unidades codificadas
    listas para modelar (nombres = factor_names, sin sufijo _cod), eliminando
    las columnas en unidades reales (_real) para que no se confundan con
    posibles respuestas."""
    df = st.session_state.df
    fn = st.session_state.factor_names
    if df is None:
        return None
    cod_cols = {f"{f}_cod": f for f in fn if f"{f}_cod" in df.columns}
    real_cols = [f"{f}_real" for f in fn if f"{f}_real" in df.columns]
    if cod_cols:
        out = df.rename(columns=cod_cols)
        out = out.drop(columns=real_cols, errors="ignore")
        return out
    return df


def howto(title: str, body: str):
    """Caja instructiva: explica en lenguaje simple qué hace la sección y
    qué va a obtener el usuario. Se usa al inicio de cada pestaña."""
    st.markdown(
        f'<div class="howto"><div class="howto-title">{title}</div>'
        f'<div class="howto-body">{body}</div></div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# ENCABEZADO
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <!-- Panel izquierdo -->
    <div class="hero-left">
        <div class="hero-tag">RSM &middot; Proyecto Académico</div>
        <p class="hero-title">
            Metodología de Superficie
            de Respuesta
            <span>Tostado de Cacao Nacional &middot; Ecuador</span>
        </p>
        <p class="hero-desc">Diseño de experimentos, modelado estadístico y optimización simultánea multivariable para el análisis del tostado de cacao Nacional Fino de Aroma.</p>
    </div>
    <!-- Panel derecho -->
    <div class="hero-right">
        <div class="hero-group-label">Integrantes</div>
        <div class="hero-divider"></div>
        <div class="hero-members-grid">
            <div class="hero-member-card">
                <div class="hero-member-name">Chicaiza Eduardo</div>
            </div>
            <div class="hero-member-card">
                <div class="hero-member-name">Guamanarca Didier</div>
            </div>
            <div class="hero-member-card" style="grid-column: 1 / -1;">
                <div class="hero-member-name">Tamay Katherine</div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# STEPPER DE PROGRESO — muestra en qué paso vas del flujo completo
# ---------------------------------------------------------------------------
_has_data = st.session_state.df is not None
_has_models = bool(st.session_state.models)
_has_opt = "desirability_result" in st.session_state

_steps = [
    ("1", "Cargar datos", _has_data),
    ("2", "Ajustar modelo", _has_models),
    ("3", "Ver diagnóstico", _has_models),
    ("4", "Optimizar", _has_opt),
    ("5", "Visualizar", _has_models),
    ("6", "Exportar", _has_models),
]
_prereq = [True, _has_data, _has_models, _has_models, _has_models, _has_models]

step_html = '<div class="stepper">'
for (num, label, done), ready in zip(_steps, _prereq):
    cls = "done" if done else ("ready" if ready else "")
    icon = "✓" if done else num
    step_html += (
        f'<div class="step-item {cls}">'
        f'<div class="step-circle">{icon}</div>'
        f'<div class="step-label">{label}</div></div>'
    )
step_html += "</div>"
st.markdown(step_html, unsafe_allow_html=True)

tabs = st.tabs([
    "Datos y Diseño",
    "Ajuste del Modelo",
    "Diagnóstico",
    "Optimización",
    "Visualización",
    "Exportar",
])

# ===========================================================================
# TAB 1: DATOS Y DISEÑO
# ===========================================================================
with tabs[0]:
    st.subheader("Datos y Diseño Experimental")
    howto(
        "¿Qué hace este paso?",
        "Aquí defines <b>con qué datos vas a trabajar</b>. Puedes generar automáticamente un "
        "diseño experimental (CCD o Box-Behnken) con datos sintéticos ya calculados, o subir tu "
        "propio archivo CSV si tienes datos reales de laboratorio. Este es siempre el primer paso: "
        "sin datos aquí, no se puede ajustar ningún modelo en los pasos siguientes.",
    )

    modo = st.radio(
        "Fuente de datos",
        ["Generar diseño CCD sintético (cacao)", "Generar diseño Box-Behnken", "Cargar CSV propio"],
        horizontal=True,
    )

    if modo == "Generar diseño CCD sintético (cacao)":
        c1, c2, c3 = st.columns(3)
        n_center = c1.number_input("Puntos centrales", 3, 10, 6)
        alpha_opt = c2.selectbox("Tipo de alpha", ["rotatable", "orthogonal", "1.0 (cara centrada)"])
        seed = c3.number_input("Semilla aleatoria", 0, 9999, 42)
        alpha_val = "rotatable" if alpha_opt == "rotatable" else ("orthogonal" if alpha_opt == "orthogonal" else 1.0)

        if st.button("Generar dataset sintético", type="primary"):
            df = generate_dataset(n_center=n_center, alpha=alpha_val, seed=seed)
            st.session_state.df = df
            st.session_state.factor_names = FACTOR_NAMES
            st.session_state.factor_ranges = FACTOR_RANGES
            st.session_state.models = {}
            st.success(f"Diseño CCD generado: {len(df)} corridas ({df['PointType'].value_counts().to_dict()})")

        st.caption(
            "⚠️ **Transparencia:** estos datos son sintéticos, generados con funciones que respetan "
            "tendencias reportadas en literatura científica (los polifenoles y el DPPH bajan con más "
            "temperatura/tiempo; el pardeamiento sube; el sabor tiene un óptimo intermedio). No sustituyen "
            "datos experimentales reales — ver declaración completa en el reporte técnico."
        )

    elif modo == "Generar diseño Box-Behnken":
        k = st.slider("Número de factores", 3, 4, 3)
        n_center_bbd = st.number_input("Puntos centrales", 1, 10, 3)
        names = [FACTOR_NAMES[i] if i < len(FACTOR_NAMES) else f"X{i+1}" for i in range(k)]
        if st.button("Generar diseño Box-Behnken", type="primary"):
            bbd = generate_bbd(k=k, n_center=n_center_bbd, factor_names=names)
            st.session_state.df = bbd
            st.session_state.factor_names = names
            st.session_state.models = {}
            st.success(f"Diseño BBD generado: {len(bbd)} corridas")
            st.info("Este diseño solo trae las coordenadas codificadas. Añade tus respuestas experimentales "
                    "en la tabla y vuelve a esta pestaña, o exporta el CSV, complétalo y súbelo abajo.")

    else:
        up = st.file_uploader("Sube un archivo CSV con tus datos (factores + respuestas)", type=["csv"])
        if up is not None:
            df_up = pd.read_csv(up)
            st.session_state.df = df_up
            st.session_state.models = {}
            st.success(f"Archivo cargado: {df_up.shape[0]} filas, {df_up.shape[1]} columnas")
            st.info("Selecciona a continuación cuáles columnas son factores (en unidades codificadas -1 a 1).")
            cols = list(df_up.columns)
            sel_factors = st.multiselect("Columnas de factores (codificadas)", cols)
            if sel_factors:
                st.session_state.factor_names = sel_factors

    st.divider()
    if st.session_state.df is not None:
        st.subheader("Vista de la matriz de diseño / datos")
        st.dataframe(st.session_state.df, use_container_width=True, height=320)
        csv_bytes = st.session_state.df.to_csv(index=False).encode("utf-8")
        st.download_button("Descargar datos (CSV)", csv_bytes, "datos_rsm.csv", "text/csv")
    else:
        st.info("Genera un diseño o carga un CSV para comenzar.")

# ===========================================================================
# TAB 2: AJUSTE DEL MODELO
# ===========================================================================
with tabs[1]:
    st.subheader("Ajuste de Modelos")
    howto(
        "¿Qué hace este paso?",
        "Aquí el aplicativo <b>ajusta una ecuación matemática</b> (regresión) que relaciona los "
        "factores (temperatura, tiempo, humedad) con cada respuesta que elijas (polifenoles, DPPH, "
        "color, sabor). Vas a obtener: qué tan bien se ajusta el modelo (R²), la tabla ANOVA (si el "
        "modelo es estadísticamente significativo) y la prueba de falta de ajuste (si el modelo tiene "
        "la forma correcta). Necesitas haber generado datos en el paso Datos y Diseño.",
    )

    df_cod = get_coded_df()
    if df_cod is None:
        st.warning("Primero genera o carga datos en la pestaña Datos y Diseño.")
    else:
        fn = st.session_state.factor_names
        response_cols = [c for c in df_cod.columns if c not in fn + ["Run", "PointType"]
                          and pd.api.types.is_numeric_dtype(df_cod[c])]

        c1, c2 = st.columns([0.6, 0.4])
        sel_responses = c1.multiselect(
            "Respuestas a modelar", response_cols, default=response_cols[:4] if response_cols else []
        )
        order = c2.selectbox("Orden del modelo", [2, 1], format_func=lambda o: f"{o}º orden")

        if st.button("Ajustar modelo(s)", type="primary") and sel_responses:
            models = {}
            for resp in sel_responses:
                models[resp] = fit_model(df_cod, fn, resp, order=order)
            st.session_state.models = models
            st.success(f"{len(models)} modelo(s) ajustado(s) correctamente.")

        if st.session_state.models:
            resp_pick = st.selectbox("Ver resultados de la respuesta:", list(st.session_state.models.keys()))
            model = st.session_state.models[resp_pick]

            m1, m2, m3 = st.columns(3)
            m1.metric("R²", f"{model.r2:.4f}")
            m2.metric("R² ajustado", f"{model.r2_adj:.4f}")
            m3.metric("N° de corridas", f"{len(model.y)}")

            cta, ctb = st.columns(2)
            with cta:
                st.markdown("**Coeficientes del modelo**")
                coef_df = model.fitted_model.summary2().tables[1]
                st.dataframe(coef_df.round(4), use_container_width=True)

            with ctb:
                st.markdown("**Tabla ANOVA**")
                st.dataframe(anova_table(model).round(4), use_container_width=True)

            st.markdown("**Prueba de falta de ajuste** (usa réplicas en el diseño como error puro)")
            lof = lack_of_fit_test(df_cod, model)
            lof_show = {k: v for k, v in lof.items() if k != "conclusion"}
            st.dataframe(pd.DataFrame([lof_show]).round(4), use_container_width=True)
            if lof.get("p_valor") is not None:
                if lof["p_valor"] > 0.05:
                    st.success(f"{lof['conclusion']} (p = {lof['p_valor']:.4f})")
                else:
                    st.warning(f"{lof['conclusion']} (p = {lof['p_valor']:.4f})")
            else:
                st.info(lof["conclusion"])

            eq_terms = [f"{model.coefficients['const']:.3f}"]
            for name, val in model.coefficients.items():
                if name == "const":
                    continue
                sign = "+" if val >= 0 else "-"
                eq_terms.append(f"{sign} {abs(val):.3f}·{name}")
            st.markdown("**Ecuación del modelo (unidades codificadas):**")
            st.code(f"{RESPONSE_META.get(resp_pick, {}).get('label', resp_pick)} = " + " ".join(eq_terms))
        else:
            st.info("Selecciona respuestas y ajusta el/los modelo(s) para ver resultados.")

# ===========================================================================
# TAB 3: DIAGNOSTICO DE RESIDUOS
# ===========================================================================
with tabs[2]:
    st.subheader("Diagnóstico de Residuos")
    howto(
        "¿Qué hace este paso?",
        "Revisa si el modelo ajustado en el paso Ajuste del Modelo. es <b>confiable</b>. Un residuo es la diferencia "
        "entre lo que el modelo predice y el dato real. Si los residuos se ven aleatorios (sin patrón) "
        "y siguen una distribución normal, el modelo es confiable para optimizar. Si ves un patrón "
        "claro (forma de U, embudo, etc.), el modelo necesita ajustes.",
    )
    if not st.session_state.models:
        st.warning("Ajusta primero un modelo en la pestaña Ajuste del Modelo.")
    else:
        resp_pick = st.selectbox("Respuesta:", list(st.session_state.models.keys()), key="diag_resp")
        model = st.session_state.models[resp_pick]
        diag = residual_diagnostics(model)

        c1, c2 = st.columns(2)
        with c1:
            fig1 = px.scatter(
                diag, x="Predicho", y="Residuo", trendline="ols",
                title="Residuos vs. valores predichos",
                color_discrete_sequence=["#8B5E34"],
            )
            fig1.add_hline(y=0, line_dash="dash", line_color="#C89B3C")
            st.plotly_chart(fig1, use_container_width=True)

        with c2:
            fig2 = px.histogram(
                diag, x="Residuo_estandarizado", nbins=10,
                title="Histograma de residuos estandarizados",
                color_discrete_sequence=["#2F6B4F"],
            )
            st.plotly_chart(fig2, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            from scipy import stats as sstats
            osm, osr = sstats.probplot(diag["Residuo_estandarizado"], dist="norm", fit=False)
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=osm[0] if isinstance(osm, tuple) else osm, y=osr if osr is not None else osm,
                                       mode="markers", marker=dict(color="#26180F"), name="Residuos"))
            qq = sstats.probplot(diag["Residuo_estandarizado"], dist="norm")
            (theor, sample), (slope, intercept, r) = qq
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=theor, y=sample, mode="markers", marker=dict(color="#26180F"), name="Datos"))
            fig3.add_trace(go.Scatter(x=theor, y=slope * theor + intercept, mode="lines",
                                       line=dict(color="#C89B3C", dash="dash"), name="Normal teórica"))
            fig3.update_layout(title=f"Gráfico Q-Q normal (R={r:.3f})",
                                xaxis_title="Cuantiles teóricos", yaxis_title="Residuos estandarizados")
            st.plotly_chart(fig3, use_container_width=True)

        with c4:
            diag_run = diag.copy()
            diag_run["Corrida"] = range(1, len(diag_run) + 1)
            fig4 = px.scatter(
                diag_run, x="Corrida", y="Residuo", title="Residuos vs. orden de corrida",
                color_discrete_sequence=["#8B5E34"],
            )
            fig4.add_hline(y=0, line_dash="dash", line_color="#C89B3C")
            st.plotly_chart(fig4, use_container_width=True)

        st.dataframe(diag.round(4), use_container_width=True)

# ===========================================================================
# TAB 4: OPTIMIZACION
# ===========================================================================
with tabs[3]:
    st.subheader("Optimización de Procesos")
    howto(
        "¿Qué hace este paso?",
        "Aquí encuentras las <b>condiciones de proceso óptimas</b> (temperatura, tiempo, humedad). "
        "Tienes 4 métodos: <b>Análisis canónico</b> (te dice si el óptimo es un máximo, mínimo o punto "
        "de silla), <b>Ascenso más pronunciado</b> (hacia dónde moverte para mejorar rápido), "
        "<b>Análisis de cresta</b> (el mejor punto a cada distancia del centro) y <b>Deseabilidad</b> "
        "(el método más importante: optimiza las 4 respuestas al mismo tiempo, con los objetivos que "
        "tú definas para cada una).",
    )
    if not st.session_state.models or len(st.session_state.models) == 0:
        st.warning("Ajusta primero al menos un modelo en la pestaña Ajuste del Modelo.")
    else:
        fn = st.session_state.factor_names
        opt_mode = st.radio(
            "Método de optimización",
            ["Análisis canónico", "Ascenso más pronunciado", "Análisis de cresta", "Deseabilidad (multi-respuesta)"],
            horizontal=True,
        )

        if opt_mode == "Análisis canónico":
            resp_pick = st.selectbox("Respuesta:", list(st.session_state.models.keys()), key="canon_resp")
            model = st.session_state.models[resp_pick]
            if model.order != 2:
                st.error("El análisis canónico requiere un modelo de 2º orden.")
            else:
                ca = canonical_analysis(model)
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Punto estacionario (unidades codificadas)**")
                    st.dataframe(pd.DataFrame([ca["punto_estacionario"]]).round(4), use_container_width=True)
                    st.metric(f"Respuesta estimada en el punto estacionario", f"{ca['respuesta_en_punto_estacionario']:.3f}")
                    if ca["tipo_punto"] == "Maximo":
                        st.success(f"Tipo de punto: **{ca['tipo_punto']}**")
                    elif ca["tipo_punto"] == "Minimo":
                        st.info(f"Tipo de punto: **{ca['tipo_punto']}**")
                    else:
                        st.warning(f"Tipo de punto: **{ca['tipo_punto']}**")
                with c2:
                    st.markdown("**Eigenvalores de la matriz B**")
                    eig_df = pd.DataFrame({"Eigenvalor": ca["eigenvalores"]})
                    st.dataframe(eig_df.round(4), use_container_width=True)
                    st.caption(
                        "Todos negativos → máximo · Todos positivos → mínimo · "
                        "Signos mixtos → punto de silla (montura)"
                    )
                    st.markdown("**Matriz B (cuadrática)**")
                    st.dataframe(pd.DataFrame(ca["matriz_B"], columns=fn, index=fn).round(4))

                if fn == FACTOR_NAMES:
                    real_pt = {f: round(np_val, 2) for f, np_val in
                               zip(fn, [FACTOR_RANGES[f][0] + (FACTOR_RANGES[f][1]-FACTOR_RANGES[f][0]) *
                                        (ca["punto_estacionario"][f] + 1.682) / (2*1.682) for f in fn])}
                    st.caption(f"Referencia aproximada en unidades reales (asumiendo diseño rotatable α=1.682): {real_pt}")

        elif opt_mode == "Ascenso más pronunciado":
            resp_pick = st.selectbox("Respuesta a mejorar:", list(st.session_state.models.keys()), key="sa_resp")
            model = st.session_state.models[resp_pick]
            c1, c2, c3 = st.columns(3)
            direction = c1.radio("Dirección", ["Maximizar", "Minimizar"], horizontal=True)
            step_size = c2.number_input("Tamaño de paso (unid. codificadas)", 0.1, 2.0, 0.5, step=0.1)
            n_steps = c3.number_input("N° de pasos", 1, 15, 6)
            sa = steepest_ascent(model, step_size=step_size, n_steps=n_steps, maximize=(direction == "Maximizar"))
            st.dataframe(sa.round(4), use_container_width=True)
            fig = px.line(
                sa, x="Paso", y=f"{resp_pick}_predicho", markers=True,
                title=f"Trayectoria de {'ascenso' if direction=='Maximizar' else 'descenso'} más pronunciado",
                color_discrete_sequence=["#8B5E34"],
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "La trayectoria se calcula a partir de los coeficientes lineales del modelo. "
                "Es válida mientras te mantengas cerca de la región experimental (unidades codificadas ±2 aprox.)."
            )

        elif opt_mode == "Análisis de cresta":
            resp_pick = st.selectbox("Respuesta:", list(st.session_state.models.keys()), key="ridge_resp")
            model = st.session_state.models[resp_pick]
            if model.order != 2:
                st.error("El análisis de cresta requiere un modelo de 2º orden.")
            else:
                c1, c2 = st.columns(2)
                direction = c1.radio("Dirección", ["Maximizar", "Minimizar"], horizontal=True, key="ridge_dir")
                r_max = c2.slider("Radio máximo (unid. codificadas)", 0.5, 3.0, 2.0, step=0.1)
                radii = np.linspace(0.25, r_max, 8)
                ridge = ridge_analysis(model, radii=radii, maximize=(direction == "Maximizar"))
                st.dataframe(ridge.round(4), use_container_width=True)
                fig = px.line(
                    ridge, x="Radio", y=f"{resp_pick}_predicho", markers=True,
                    title="Respuesta óptima predicha por radio (análisis de cresta)",
                    color_discrete_sequence=["#2F6B4F"],
                )
                st.plotly_chart(fig, use_container_width=True)

        else:  # Deseabilidad
            st.markdown("**Configura el objetivo de cada respuesta (función de Derringer-Suich)**")
            df_cod = get_coded_df()
            goals = []
            for resp, model in st.session_state.models.items():
                meta = RESPONSE_META.get(resp, {"label": resp, "goal_default": "maximize"})
                with st.expander(f"{meta['label']}", expanded=True):
                    c1, c2, c3, c4 = st.columns(4)
                    goal = c1.selectbox("Objetivo", ["maximize", "minimize", "target"],
                                         index=["maximize", "minimize", "target"].index(meta["goal_default"]),
                                         key=f"goal_{resp}")
                    lo_default = float(df_cod[resp].min())
                    hi_default = float(df_cod[resp].max())
                    lo = c2.number_input("Límite inferior aceptable", value=lo_default, key=f"lo_{resp}")
                    hi = c3.number_input("Límite superior aceptable", value=hi_default, key=f"hi_{resp}")
                    target = None
                    if goal == "target":
                        target = c4.number_input("Valor objetivo (target)", value=(lo + hi) / 2, key=f"tg_{resp}")
                    weight = st.slider("Importancia relativa (peso)", 0.1, 3.0, 1.0, 0.1, key=f"w_{resp}")
                    goals.append(ResponseGoal(name=resp, goal=goal, low=lo, high=hi, target=target, weight=weight))

            if st.button("Optimizar deseabilidad global", type="primary"):
                fn = st.session_state.factor_names
                result = optimize_desirability(st.session_state.models, goals, fn, bounds=(-1.68, 1.68))
                st.session_state.desirability_result = result

            if "desirability_result" in st.session_state:
                result = st.session_state.desirability_result
                st.success(f"Deseabilidad global D = **{result['deseabilidad_global']:.4f}**")

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Factores óptimos (codificados)**")
                    fac_df = pd.DataFrame([result["factores_optimos_codificados"]])
                    st.dataframe(fac_df.round(4), use_container_width=True)
                    if st.session_state.factor_names == FACTOR_NAMES:
                        real_vals = {f: round(FACTOR_RANGES[f][0] + (FACTOR_RANGES[f][1] - FACTOR_RANGES[f][0]) *
                                               (result["factores_optimos_codificados"][f] + 1) / 2, 2)
                                     for f in FACTOR_NAMES}
                        st.markdown("**Factores óptimos (unidades reales)**")
                        st.dataframe(pd.DataFrame([real_vals]), use_container_width=True)
                with c2:
                    st.markdown("**Respuestas predichas en el óptimo**")
                    st.dataframe(pd.DataFrame([result["respuestas_predichas"]]).round(3), use_container_width=True)

# ===========================================================================
# TAB 5: VISUALIZACION
# ===========================================================================
with tabs[4]:
    st.subheader("Visualización Gráfica")
    howto(
        "¿Qué hace este paso?",
        "Convierte el modelo matemático en <b>imágenes que se entienden de un vistazo</b>: "
        "el <b>contorno</b> y la <b>superficie 3D</b> muestran cómo cambia la respuesta según dos "
        "factores a la vez; el <b>Pareto</b> muestra qué factor influye más; la <b>perturbación</b> "
        "muestra el efecto de mover cada factor por separado desde el centro del diseño.",
    )
    if not st.session_state.models:
        st.warning("Ajusta primero un modelo en la pestaña Ajuste del Modelo.")
    else:
        fn = st.session_state.factor_names
        viz_mode = st.radio(
            "Tipo de gráfico", ["Contorno", "Superficie 3D", "Pareto de efectos", "Perturbación"],
            horizontal=True,
        )
        resp_pick = st.selectbox("Respuesta:", list(st.session_state.models.keys()), key="viz_resp")
        model = st.session_state.models[resp_pick]

        if viz_mode in ("Contorno", "Superficie 3D"):
            if len(fn) < 2:
                st.error("Se necesitan al menos 2 factores.")
            else:
                c1, c2, c3 = st.columns(3)
                fx = c1.selectbox("Eje X", fn, index=0)
                fy = c2.selectbox("Eje Y", fn, index=1)
                other = [f for f in fn if f not in (fx, fy)]
                fixed_vals = {}
                if other:
                    st.markdown("**Valor fijo para los demás factores (codificado):**")
                    cols_fix = st.columns(len(other))
                    for i, f in enumerate(other):
                        fixed_vals[f] = cols_fix[i].slider(f, -2.0, 2.0, 0.0, 0.1, key=f"fix_{f}_{viz_mode}")

                grid_n = 40
                xs = np.linspace(-1.8, 1.8, grid_n)
                ys = np.linspace(-1.8, 1.8, grid_n)
                XX, YY = np.meshgrid(xs, ys)
                pred_df = pd.DataFrame({fx: XX.ravel(), fy: YY.ravel()})
                for f in other:
                    pred_df[f] = fixed_vals[f]
                pred_df = pred_df[fn]
                ZZ = model.predict(pred_df).reshape(XX.shape)

                if viz_mode == "Contorno":
                    fig = go.Figure(data=go.Contour(
                        x=xs, y=ys, z=ZZ,
                        colorscale=REACT_SCALE,
                        contours=dict(showlabels=True),
                    ))
                    fig.update_layout(
                        title=f"Contorno de {RESPONSE_META.get(resp_pick,{}).get('label',resp_pick)}",
                        xaxis_title=fx, yaxis_title=fy, height=550,
                    )
                else:
                    fig = go.Figure(data=[go.Surface(
                        x=xs, y=ys, z=ZZ,
                        colorscale=REACT_SCALE,
                    )])
                    fig.update_layout(
                        title=f"Superficie de respuesta: {RESPONSE_META.get(resp_pick,{}).get('label',resp_pick)}",
                        scene=dict(xaxis_title=fx, yaxis_title=fy, zaxis_title=resp_pick),
                        height=600,
                    )
                st.plotly_chart(fig, use_container_width=True)

        elif viz_mode == "Pareto de efectos":
            coefs = model.coefficients.drop("const", errors="ignore")
            pareto_df = pd.DataFrame({
                "Termino": coefs.index,
                "Efecto_abs": coefs.abs().values,
                "Signo": np.where(coefs.values >= 0, "Positivo", "Negativo"),
            }).sort_values("Efecto_abs", ascending=True)
            fig = px.bar(
                pareto_df, x="Efecto_abs", y="Termino", color="Signo", orientation="h",
                title=f"Diagrama de Pareto de efectos: {RESPONSE_META.get(resp_pick,{}).get('label',resp_pick)}",
                color_discrete_map={"Positivo": "#8B5E34", "Negativo": "#B89B79"},
            )
            st.plotly_chart(fig, use_container_width=True)

        else:  # Perturbacion
            st.caption("Muestra cómo cambia la respuesta al mover cada factor individualmente desde el centro del diseño.")
            xs = np.linspace(-1.8, 1.8, 50)
            fig = go.Figure()
            palette = ["#8B5E34", "#2F6B4F", "#C89B3C", "#8E3E32", "#5A351F"]
            for i, f in enumerate(fn):
                pred_df = pd.DataFrame({ff: np.zeros_like(xs) for ff in fn})
                pred_df[f] = xs
                y_pred = model.predict(pred_df)
                fig.add_trace(go.Scatter(x=xs, y=y_pred, mode="lines", name=f, line=dict(color=palette[i % len(palette)])))
            fig.update_layout(
                title=f"Gráfico de perturbación: {RESPONSE_META.get(resp_pick,{}).get('label',resp_pick)}",
                xaxis_title="Desviación desde el centro (unidades codificadas)",
                yaxis_title=resp_pick, height=500,
            )
            st.plotly_chart(fig, use_container_width=True)

# ===========================================================================
# TAB 6: EXPORTAR
# ===========================================================================
with tabs[5]:
    st.subheader("Exportar Resultados")
    howto(
        "¿Qué hace este paso?",
        "Descarga todo lo que hiciste en un solo archivo Excel: los datos, los coeficientes de cada "
        "modelo, las tablas ANOVA y el punto óptimo de deseabilidad (si lo calculaste). Úsalo como "
        "respaldo o para anexarlo a tu reporte técnico.",
    )
    if not st.session_state.models:
        st.warning("Ajusta al menos un modelo para poder exportar resultados.")
    else:
        st.markdown("Genera un resumen consolidado con los datos, coeficientes de los modelos y, si la calculaste, "
                     "la optimización por deseabilidad.")

        if st.button("Generar reporte resumen (Excel)", type="primary"):
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                st.session_state.df.to_excel(writer, sheet_name="Datos", index=False)
                for resp, model in st.session_state.models.items():
                    sheet = resp[:28]
                    coef_df = model.fitted_model.summary2().tables[1]
                    coef_df.to_excel(writer, sheet_name=f"Coef_{sheet}")
                    anova_table(model).to_excel(writer, sheet_name=f"ANOVA_{sheet}", index=False)
                if "desirability_result" in st.session_state:
                    res = st.session_state.desirability_result
                    pd.DataFrame([res["factores_optimos_codificados"]]).to_excel(
                        writer, sheet_name="Optimo_Deseabilidad", index=False, startrow=0)
                    pd.DataFrame([res["respuestas_predichas"]]).to_excel(
                        writer, sheet_name="Optimo_Deseabilidad", index=False, startrow=3)
            st.download_button(
                "Descargar reporte (.xlsx)", buffer.getvalue(),
                file_name="reporte_rsm_cacao.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
            st.success("Reporte generado. Usa el botón de descarga.")

        st.divider()
        st.markdown("**Modelos ajustados actualmente en la sesión:**")
        for resp, model in st.session_state.models.items():
            st.write(f"- {RESPONSE_META.get(resp, {}).get('label', resp)} — R²adj = {model.r2_adj:.4f}")

st.divider()
st.markdown(
    '<p class="footer-note">Proyecto académico RSM · Datos sintéticos con fines educativos · '
    "No reemplaza análisis experimental de laboratorio.</p>",
    unsafe_allow_html=True,
)
