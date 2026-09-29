# -*- coding: utf-8 -*-
"""
WARDEN-T Enterprise Security-Analytics Design System & Styles
Clean Slate Palette:
  Background: #0F172A
  Surface/Cards: #1E293B
  Borders: #334155
  Primary Text: #F1F5F9
  Secondary Text: #94A3B8
  Accent: #3B82F6
  Severity: Critical #DC2626 | High #EA580C | Medium #CA8A04 | Low/OK #16A34A
Offline System Sans & Monospace. No neon, no glow, no gradients.
"""

CSS = """
<style>
/* -------------------------------------------------------------------------
   GLOBAL APPLICATION STYLING & RESETS
   ------------------------------------------------------------------------- */
html, body, [class*="css"], .stApp {
    background-color: #0F172A !important;
    color: #F1F5F9 !important;
    font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    font-size: 14px;
    line-height: 1.5;
    -webkit-font-smoothing: antialiased;
}

/* Eliminate horizontal rule lines - use 24px spacing instead */
hr, .stMarkdown hr {
    display: none !important;
}

/* Section Spacing Utility */
.section-spacer {
    height: 24px;
    width: 100%;
}

/* -------------------------------------------------------------------------
   TYPOGRAPHY
   ------------------------------------------------------------------------- */
h1 {
    font-size: 28px !important;
    font-weight: 600 !important;
    color: #F1F5F9 !important;
    letter-spacing: -0.02em;
    margin: 0 0 4px 0 !important;
    line-height: 1.25 !important;
}

h2, h3 {
    font-size: 18px !important;
    font-weight: 600 !important;
    color: #F1F5F9 !important;
    letter-spacing: -0.01em;
    margin: 0 0 12px 0 !important;
    line-height: 1.35 !important;
}

h4, h5, h6 {
    font-size: 15px !important;
    font-weight: 600 !important;
    color: #F1F5F9 !important;
    margin: 0 0 8px 0 !important;
}

p, span, div, label {
    color: #F1F5F9;
    font-size: 14px;
}

.text-secondary {
    color: #94A3B8 !important;
    font-size: 13px !important;
}

.caption-text {
    font-size: 13px !important;
    color: #94A3B8 !important;
    margin-top: 4px;
    margin-bottom: 16px;
    line-height: 1.45;
}

.mono-text {
    font-family: ui-monospace, "SF Mono", Menlo, Monaco, Consolas, monospace !important;
}

/* -------------------------------------------------------------------------
   HEADER
   ------------------------------------------------------------------------- */
.enterprise-header {
    background-color: #1E293B;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 20px 24px;
    margin-bottom: 24px;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.25);
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 16px;
}

.header-left {
    display: flex;
    flex-direction: column;
}

.header-title {
    font-size: 28px;
    font-weight: 600;
    color: #F1F5F9;
    letter-spacing: -0.02em;
    line-height: 1.2;
}

.header-subtitle {
    font-size: 13px;
    color: #94A3B8;
    margin-top: 2px;
}

.status-chip {
    background-color: #0F172A;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 6px 14px;
    font-size: 12px;
    color: #94A3B8;
    font-weight: 500;
    font-family: ui-monospace, "SF Mono", Menlo, Monaco, Consolas, monospace;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}

.status-chip-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background-color: #16A34A;
    display: inline-block;
}

/* -------------------------------------------------------------------------
   CARDS & PANELS
   ------------------------------------------------------------------------- */
.enterprise-card {
    background-color: #1E293B;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 20px 24px;
    margin-bottom: 24px;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.25);
}

/* KPI CARDS (4) */
.kpi-card {
    background-color: #1E293B;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 20px 22px;
    margin-bottom: 16px;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.25);
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    min-height: 125px;
}

.kpi-border-critical { border-left: 4px solid #DC2626 !important; }
.kpi-border-high     { border-left: 4px solid #EA580C !important; }
.kpi-border-medium   { border-left: 4px solid #CA8A04 !important; }
.kpi-border-low      { border-left: 4px solid #16A34A !important; }
.kpi-border-neutral  { border-left: 4px solid #3B82F6 !important; }

.kpi-label {
    font-size: 12px;
    color: #94A3B8;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 8px;
}

.kpi-value-row {
    display: flex;
    align-items: baseline;
    gap: 10px;
    margin-bottom: 4px;
}

.kpi-value-num {
    font-size: 32px;
    font-weight: 600;
    color: #F1F5F9;
    font-family: ui-monospace, "SF Mono", Menlo, Monaco, Consolas, monospace;
    line-height: 1.1;
}

.kpi-value-text {
    font-size: 20px;
    font-weight: 600;
    color: #F1F5F9;
    line-height: 1.2;
}

.kpi-subtext {
    font-size: 13px;
    color: #94A3B8;
    line-height: 1.35;
}

/* Status Badges (Coloured, No Glow) */
.badge-threat {
    display: inline-block;
    font-size: 14px;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 6px;
    line-height: 1.3;
}
.badge-critical {
    background-color: rgba(220, 38, 38, 0.15);
    color: #DC2626;
    border: 1px solid rgba(220, 38, 38, 0.35);
}
.badge-high {
    background-color: rgba(234, 88, 12, 0.15);
    color: #EA580C;
    border: 1px solid rgba(234, 88, 12, 0.35);
}
.badge-medium {
    background-color: rgba(202, 138, 4, 0.15);
    color: #CA8A04;
    border: 1px solid rgba(202, 138, 4, 0.35);
}
.badge-low {
    background-color: rgba(22, 163, 74, 0.15);
    color: #16A34A;
    border: 1px solid rgba(22, 163, 74, 0.35);
}

/* -------------------------------------------------------------------------
   TABS STYLING
   ------------------------------------------------------------------------- */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: transparent !important;
    border-bottom: 1px solid #334155 !important;
    padding-bottom: 2px;
    margin-bottom: 20px;
}

.stTabs [data-baseweb="tab"] {
    background-color: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    color: #94A3B8 !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    padding: 10px 16px !important;
    transition: all 0.15s ease;
}

.stTabs [data-baseweb="tab"]:hover {
    color: #F1F5F9 !important;
}

.stTabs [aria-selected="true"] {
    color: #F1F5F9 !important;
    font-weight: 600 !important;
    border-bottom: 2px solid #3B82F6 !important;
}

/* -------------------------------------------------------------------------
   SIDEBAR STYLING
   ------------------------------------------------------------------------- */
section[data-testid="stSidebar"] {
    background-color: #1E293B !important;
    border-right: 1px solid #334155 !important;
    width: 260px !important;
    min-width: 260px !important;
}

section[data-testid="stSidebar"] .stMarkdown h3 {
    font-size: 16px !important;
    margin-bottom: 2px !important;
}

/* Sidebar Radio/Navigation Items */
section[data-testid="stSidebar"] div[role="radiogroup"] {
    display: flex;
    flex-direction: column;
    gap: 4px;
}

section[data-testid="stSidebar"] div[role="radiogroup"] > label {
    padding: 8px 12px !important;
    border-radius: 6px !important;
    border-left: 3px solid transparent !important;
    transition: all 0.15s ease;
    cursor: pointer;
}

section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
    background-color: rgba(51, 65, 85, 0.4) !important;
}

section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"],
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
    background-color: #0F172A !important;
    border-left: 3px solid #3B82F6 !important;
}

section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"] p,
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) p {
    color: #F1F5F9 !important;
    font-weight: 600 !important;
}

.sidebar-footer {
    position: fixed;
    bottom: 20px;
    width: 220px;
    padding-top: 12px;
    border-top: 1px solid #334155;
    font-size: 12px;
    color: #94A3B8;
    line-height: 1.4;
}

/* -------------------------------------------------------------------------
   TABLES & DATAFRAMES
   ------------------------------------------------------------------------- */
div[data-testid="stDataFrame"], div[data-testid="stTable"] {
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    overflow: hidden !important;
}

.enterprise-table {
    width: 100%;
    border-collapse: collapse;
    background-color: #1E293B;
    border-radius: 8px;
    overflow: hidden;
    font-size: 13px;
    margin: 12px 0 20px 0;
}

.enterprise-table th {
    background-color: #1E293B;
    color: #94A3B8;
    font-weight: 600;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    padding: 10px 14px;
    text-align: left;
    border-bottom: 1px solid #334155;
}

.enterprise-table td {
    padding: 10px 14px;
    border-bottom: 1px solid #334155;
    color: #F1F5F9;
}

.enterprise-table tr:last-child td {
    border-bottom: none;
}

.enterprise-table tr:hover {
    background-color: rgba(255, 255, 255, 0.02);
}

.table-pill {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
}

/* -------------------------------------------------------------------------
   SPECIFICATION GRID (ARCHITECTURE)
   ------------------------------------------------------------------------- */
.spec-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 16px;
    margin-top: 16px;
}

.spec-item {
    background-color: #1E293B;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 14px 18px;
}

.spec-label {
    font-size: 12px;
    color: #94A3B8;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 4px;
}

.spec-value {
    font-size: 14px;
    color: #F1F5F9;
    font-family: ui-monospace, "SF Mono", Menlo, Monaco, Consolas, monospace;
    font-weight: 500;
}

/* -------------------------------------------------------------------------
   ATTENTION STRIP
   ------------------------------------------------------------------------- */
.attn-strip-container {
    background-color: #1E293B;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 16px;
    margin-top: 16px;
}

.attn-cells {
    display: flex;
    gap: 6px;
    margin-bottom: 8px;
}

.attn-cell {
    flex: 1;
    height: 48px;
    border-radius: 4px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    border: 1px solid #334155;
}

.attn-cell-label {
    font-size: 11px;
    font-weight: 500;
    font-family: ui-monospace, "SF Mono", monospace;
}

.attn-cell-val {
    font-size: 10px;
    font-family: ui-monospace, "SF Mono", monospace;
    opacity: 0.85;
}

/* -------------------------------------------------------------------------
   ALERT / CALLOUT BOXES
   ------------------------------------------------------------------------- */
.callout-box {
    background-color: #1E293B;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 16px 0;
    font-size: 13px;
    line-height: 1.5;
}
.callout-border-info { border-left: 4px solid #3B82F6; color: #F1F5F9; }
.callout-border-warning { border-left: 4px solid #EA580C; color: #F1F5F9; }

/* -------------------------------------------------------------------------
   STREAMLIT FORM CONTROLS POLISH
   ------------------------------------------------------------------------- */
.stButton > button {
    background-color: #3B82F6 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    padding: 8px 16px !important;
    transition: background-color 0.15s ease;
}

.stButton > button:hover {
    background-color: #2563EB !important;
}

div[data-testid="stFileUploader"] {
    background-color: #1E293B;
    border: 1px dashed #334155;
    border-radius: 8px;
    padding: 16px;
}

/* Segmented control / radio pill */
div[data-testid="stSegmentedControl"] {
    background-color: #1E293B;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 3px;
}
</style>
"""

def get_custom_css():
    """Return the single custom CSS block."""
    return CSS
