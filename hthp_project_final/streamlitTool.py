"""
HTHP Modeling Environment – Streamlit Frontend
================================================
Pages:
  upload          → upload questionnaire, read parameters
  pinch_analysis  → pinch analysis + composite curves
  case_generation → configure simulation cases
  calculation     → run cases via case_calculator.py (no math here)
  results         → individual case views + comparative overview
"""

import os
import sys
import time
import traceback

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ---------------------------------------------------------------------------
# Project path
# ---------------------------------------------------------------------------
project_path = os.path.join(os.path.dirname(__file__), '..')
if project_path not in sys.path:
    sys.path.insert(0, project_path)

from utils.questionnaire_reader import QuestionnaireReader
from utils.pinch_params import PinchParamBuilder, _val
from utils.pinch_analyzer import PinchAnalyzer
from utils.mvr_stage_selection import determine_n_stages

# Diagram generator – optional dependency
try:
    from utils.state_diagram_generator import StateDiagramGenerator
    _DIAGRAMS_AVAILABLE = True
except ImportError:
    _DIAGRAMS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Available HTHP model names
# ---------------------------------------------------------------------------
# ── HP Model Lookup Table ─────────────────────────────────────────────────────
# Maps (category, cascade_sub, econ_type, comp_var, ihx_variant, transcritical)
# → (class_name, econ_type_for_constructor)
#
# category     : 'Simple Cycle' | 'Intercooling' | 'Economizer' |
#                'Flash Tank'   | 'Cascade Cycle'
# cascade_sub  : None | 'Base' | 'Intercooling' | 'Economizer' | 'Flash Tank'
#                (only relevant when category == 'Cascade Cycle')
# econ_type    : None | 'open' | 'closed'
# comp_var     : None | 'series' | 'parallel'
# ihx_variant  : None | 'A' | 'B' | 'both' | '2ihx'
# transcritical: False | True
HP_MODEL_LOOKUP = {
    # ── Simple Cycle ─────────────────────────────────────────────────────────
    ('Simple Cycle', None, None, None, None,  False): ('HeatPumpSimple',     None),
    ('Simple Cycle', None, None, None, None,  True):  ('HeatPumpSimpleTrans',None),
    ('Simple Cycle', None, None, None, 'ihx', False): ('HeatPumpIHX',        None),
    ('Simple Cycle', None, None, None, 'ihx', True):  ('HeatPumpIHXTrans',   None),
    # ── Intercooling ─────────────────────────────────────────────────────────
    ('Intercooling', None, None, 'series', None, False): ('HeatPumpIC',     None),
    ('Intercooling', None, None, 'series', None, True):  ('HeatPumpICTrans',None),
    # ── Flash Tank ───────────────────────────────────────────────────────────
    ('Flash Tank', None, None, None, None, False): ('HeatPumpFlash',     None),
    ('Flash Tank', None, None, None, None, True):  ('HeatPumpFlashTrans',None),
    # ── Economizer · series · closed ─────────────────────────────────────────
    ('Economizer', None, 'closed', 'series', None,   False): ('HeatPumpEcon',        'closed'),
    ('Economizer', None, 'closed', 'series', None,   True):  ('HeatPumpEconTrans',   'closed'),
    ('Economizer', None, 'closed', 'series', 'A',    False): ('HeatPumpIHXEcon',     'closed'),
    ('Economizer', None, 'closed', 'series', 'A',    True):  ('HeatPumpIHXEconTrans','closed'),
    ('Economizer', None, 'closed', 'series', 'B',    False): ('HeatPumpEconIHX',     'closed'),
    ('Economizer', None, 'closed', 'series', 'B',    True):  ('HeatPumpEconIHXTrans','closed'),
    # ── Economizer · series · open ───────────────────────────────────────────
    ('Economizer', None, 'open',   'series', None,   False): ('HeatPumpEcon',        'open'),
    ('Economizer', None, 'open',   'series', None,   True):  ('HeatPumpEconTrans',   'open'),
    ('Economizer', None, 'open',   'series', 'A',    False): ('HeatPumpIHXEcon',     'open'),
    ('Economizer', None, 'open',   'series', 'A',    True):  ('HeatPumpIHXEconTrans','open'),
    ('Economizer', None, 'open',   'series', 'B',    False): ('HeatPumpEconIHX',     'open'),
    ('Economizer', None, 'open',   'series', 'B',    True):  ('HeatPumpEconIHXTrans','open'),
    # ── Economizer · parallel (PC) · closed ──────────────────────────────────
    ('Economizer', None, 'closed', 'parallel', None,   False): ('HeatPumpPC',          'closed'),
    ('Economizer', None, 'closed', 'parallel', None,   True):  ('HeatPumpPCTrans',     'closed'),
    ('Economizer', None, 'closed', 'parallel', 'A',    False): ('HeatPumpIHXPC',       'closed'),
    ('Economizer', None, 'closed', 'parallel', 'A',    True):  ('HeatPumpIHXPCTrans',  'closed'),
    ('Economizer', None, 'closed', 'parallel', 'B',    False): ('HeatPumpPCIHX',       'closed'),
    ('Economizer', None, 'closed', 'parallel', 'B',    True):  ('HeatPumpPCIHXTrans',  'closed'),
    ('Economizer', None, 'closed', 'parallel', 'both', False): ('HeatPumpIHXPCIHX',    'closed'),
    ('Economizer', None, 'closed', 'parallel', 'both', True):  ('HeatPumpIHXPCIHXTrans','closed'),
    # ── Economizer · parallel (PC) · open ────────────────────────────────────
    ('Economizer', None, 'open',   'parallel', None,   False): ('HeatPumpPC',          'open'),
    ('Economizer', None, 'open',   'parallel', None,   True):  ('HeatPumpPCTrans',     'open'),
    ('Economizer', None, 'open',   'parallel', 'A',    False): ('HeatPumpIHXPC',       'open'),
    ('Economizer', None, 'open',   'parallel', 'A',    True):  ('HeatPumpIHXPCTrans',  'open'),
    ('Economizer', None, 'open',   'parallel', 'B',    False): ('HeatPumpPCIHX',       'open'),
    ('Economizer', None, 'open',   'parallel', 'B',    True):  ('HeatPumpPCIHXTrans',  'open'),
    ('Economizer', None, 'open',   'parallel', 'both', False): ('HeatPumpIHXPCIHX',    'open'),
    ('Economizer', None, 'open',   'parallel', 'both', True):  ('HeatPumpIHXPCIHXTrans','open'),
    # ── Cascade · Base ───────────────────────────────────────────────────────
    ('Cascade Cycle', 'Base', None, None, None,   False): ('HeatPumpCascade',     None),
    ('Cascade Cycle', 'Base', None, None, None,   True):  ('HeatPumpCascadeTrans',None),
    ('Cascade Cycle', 'Base', None, None, '2ihx', False): ('HeatPumpCascade2IHX',     None),
    ('Cascade Cycle', 'Base', None, None, '2ihx', True):  ('HeatPumpCascade2IHXTrans',None),
    # ── Cascade · Intercooling ────────────────────────────────────────────────
    ('Cascade Cycle', 'Intercooling', None, 'series', None, False): ('HeatPumpCascadeIC',     None),
    ('Cascade Cycle', 'Intercooling', None, 'series', None, True):  ('HeatPumpCascadeICTrans',None),
    # ── Cascade · Flash Tank ─────────────────────────────────────────────────
    ('Cascade Cycle', 'Flash Tank', None, None, None, False): ('HeatPumpCascadeFlash',     None),
    ('Cascade Cycle', 'Flash Tank', None, None, None, True):  ('HeatPumpCascadeFlashTrans',None),
    # ── Cascade + Economizer · series · closed ────────────────────────────────
    ('Cascade Cycle', 'Economizer', 'closed', 'series', None,   False): ('HeatPumpCascadeEcon',        'closed'),
    ('Cascade Cycle', 'Economizer', 'closed', 'series', None,   True):  ('HeatPumpCascadeEconTrans',   'closed'),
    ('Cascade Cycle', 'Economizer', 'closed', 'series', 'A',    False): ('HeatPumpCascadeIHXEcon',     'closed'),
    ('Cascade Cycle', 'Economizer', 'closed', 'series', 'A',    True):  ('HeatPumpCascadeIHXEconTrans','closed'),
    ('Cascade Cycle', 'Economizer', 'closed', 'series', 'B',    False): ('HeatPumpCascadeEconIHX',     'closed'),
    ('Cascade Cycle', 'Economizer', 'closed', 'series', 'B',    True):  ('HeatPumpCascadeEconIHXTrans','closed'),
    # ── Cascade + Economizer · series · open ──────────────────────────────────
    ('Cascade Cycle', 'Economizer', 'open',   'series', None,   False): ('HeatPumpCascadeEcon',        'open'),
    ('Cascade Cycle', 'Economizer', 'open',   'series', None,   True):  ('HeatPumpCascadeEconTrans',   'open'),
    ('Cascade Cycle', 'Economizer', 'open',   'series', 'A',    False): ('HeatPumpCascadeIHXEcon',     'open'),
    ('Cascade Cycle', 'Economizer', 'open',   'series', 'A',    True):  ('HeatPumpCascadeIHXEconTrans','open'),
    ('Cascade Cycle', 'Economizer', 'open',   'series', 'B',    False): ('HeatPumpCascadeEconIHX',     'open'),
    ('Cascade Cycle', 'Economizer', 'open',   'series', 'B',    True):  ('HeatPumpCascadeEconIHXTrans','open'),
    # ── Cascade + Economizer · parallel (PC) · closed ─────────────────────────
    ('Cascade Cycle', 'Economizer', 'closed', 'parallel', None,   False): ('HeatPumpCascadePC',          'closed'),
    ('Cascade Cycle', 'Economizer', 'closed', 'parallel', None,   True):  ('HeatPumpCascadePCTrans',     'closed'),
    ('Cascade Cycle', 'Economizer', 'closed', 'parallel', 'A',    False): ('HeatPumpCascadeIHXPC',       'closed'),
    ('Cascade Cycle', 'Economizer', 'closed', 'parallel', 'A',    True):  ('HeatPumpCascadeIHXPCTrans',  'closed'),
    ('Cascade Cycle', 'Economizer', 'closed', 'parallel', 'B',    False): ('HeatPumpCascadePCIHX',       'closed'),
    ('Cascade Cycle', 'Economizer', 'closed', 'parallel', 'B',    True):  ('HeatPumpCascadePCIHXTrans',  'closed'),
    ('Cascade Cycle', 'Economizer', 'closed', 'parallel', 'both', False): ('HeatPumpCascadeIHXPCIHX',    'closed'),
    ('Cascade Cycle', 'Economizer', 'closed', 'parallel', 'both', True):  ('HeatPumpCascadeIHXPCIHXTrans','closed'),
    # ── Cascade + Economizer · parallel (PC) · open ───────────────────────────
    ('Cascade Cycle', 'Economizer', 'open',   'parallel', None,   False): ('HeatPumpCascadePC',          'open'),
    ('Cascade Cycle', 'Economizer', 'open',   'parallel', None,   True):  ('HeatPumpCascadePCTrans',     'open'),
    ('Cascade Cycle', 'Economizer', 'open',   'parallel', 'A',    False): ('HeatPumpCascadeIHXPC',       'open'),
    ('Cascade Cycle', 'Economizer', 'open',   'parallel', 'A',    True):  ('HeatPumpCascadeIHXPCTrans',  'open'),
    ('Cascade Cycle', 'Economizer', 'open',   'parallel', 'B',    False): ('HeatPumpCascadePCIHX',       'open'),
    ('Cascade Cycle', 'Economizer', 'open',   'parallel', 'B',    True):  ('HeatPumpCascadePCIHXTrans',  'open'),
    ('Cascade Cycle', 'Economizer', 'open',   'parallel', 'both', False): ('HeatPumpCascadeIHXPCIHX',    'open'),
    ('Cascade Cycle', 'Economizer', 'open',   'parallel', 'both', True):  ('HeatPumpCascadeIHXPCIHXTrans','open'),
}


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="HTHP Modeling Environment",
    page_icon="🔥",
    layout="wide"
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
_defaults = {
    'questionnaire_data':   None,
    'questionnaire_reader': None,
    'cases':                [],
    'page':                 'upload',
    'calculation_started':  False,
    'pinch_analysis_data':  None,
    'calculation_results':  None,
    'calc_case_ids':        [],
    'editing_case_idx':     None,   # index of case being edited (None = new case)
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ---------------------------------------------------------------------------
# Global theme / styling  (visual layer only – no logic changes)
# ---------------------------------------------------------------------------
_GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root{
  --brand-900:#050544;
  --brand-700:#00008C;   /* Brillantblau */
  --brand-500:#2b3bd4;
  --brand-300:#8f97e8;
  --brand-100:#e9ebff;
  --ink:#1b2138;
  --muted:#5b6478;
  --bg:#f4f6fb;
  --card:#ffffff;
  --border:#e5e8f0;
  --ok:#1e9e6a; --warn:#c9820b; --err:#d64545;
  --radius:14px;
  --shadow:0 1px 2px rgba(16,24,40,.04), 0 4px 16px rgba(16,24,40,.05);
}

/* ---- base typography & canvas ---------------------------------------- */
html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"]{
  font-family:'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
[data-testid="stAppViewContainer"]{ background:var(--bg); }
[data-testid="stMainBlockContainer"], .main .block-container{
  max-width:1180px; padding-top:1.1rem; padding-bottom:3rem;
}
[data-testid="stHeader"]{ background:transparent; }
[data-testid="stDecoration"]{ display:none; }
footer{ visibility:hidden; height:0; }

[data-testid="stAppViewContainer"] .stMarkdown p,
[data-testid="stAppViewContainer"] .stMarkdown li{ color:var(--ink); line-height:1.6; }

h1,h2,h3,h4,h5{ color:var(--ink); font-weight:700; letter-spacing:-.01em; }
[data-testid="stMainBlockContainer"] h2{ font-size:1.45rem; margin-top:.6rem; }
[data-testid="stMainBlockContainer"] h4{ font-size:1.03rem; font-weight:700; margin-top:.4rem; }
[data-testid="stMainBlockContainer"] h4::before{
  content:""; display:inline-block; width:4px; height:.95em;
  background:var(--brand-500); border-radius:3px;
  margin-right:.55rem; vertical-align:-1px;
}
[data-testid="stMainBlockContainer"] hr{
  margin:1.1rem 0; border:none; border-top:1px solid var(--border);
}

/* ---- KPI metric cards ------------------------------------------------ */
[data-testid="stMetric"]{
  background:var(--card); border:1px solid var(--border);
  border-radius:var(--radius); padding:15px 18px; box-shadow:var(--shadow);
}
[data-testid="stMetricLabel"] p{
  color:var(--muted); font-size:.76rem; font-weight:600;
  text-transform:uppercase; letter-spacing:.04em;
}
[data-testid="stMetricValue"]{ color:var(--brand-700); font-weight:700; }

/* ---- buttons --------------------------------------------------------- */
.stButton > button{
  border-radius:10px; border:1px solid var(--border);
  background:var(--card); color:var(--ink); font-weight:600;
  padding:.5rem 1.05rem; box-shadow:0 1px 2px rgba(16,24,40,.04);
  transition:all .15s ease;
}
.stButton > button:hover{ border-color:var(--brand-300); color:var(--brand-700); }
.stButton > button[kind="primary"]{
  background:var(--brand-700); border-color:var(--brand-700); color:#fff;
}
.stButton > button[kind="primary"]:hover{
  background:var(--brand-500); border-color:var(--brand-500); color:#fff;
}
.stDownloadButton > button{ border-radius:10px; font-weight:600; }

/* ---- tabs ------------------------------------------------------------ */
[data-testid="stTabs"] [data-baseweb="tab-list"]{
  gap:2px; border-bottom:1px solid var(--border);
}
[data-testid="stTabs"] [data-baseweb="tab"]{
  padding:9px 18px; color:var(--muted); font-weight:600;
}
[data-testid="stTabs"] [aria-selected="true"]{ color:var(--brand-700); }
[data-testid="stTabs"] [data-baseweb="tab-highlight"]{ background:var(--brand-700); }

/* ---- expanders / dataframes / alerts --------------------------------- */
[data-testid="stExpander"]{
  border:1px solid var(--border); border-radius:var(--radius);
  background:var(--card); box-shadow:var(--shadow); overflow:hidden;
}
[data-testid="stExpander"] summary{ font-weight:600; }
[data-testid="stDataFrame"], [data-testid="stTable"]{
  border:1px solid var(--border); border-radius:var(--radius); overflow:hidden;
}
[data-testid="stAlert"]{ border-radius:12px; }

/* ---- app hero banner ------------------------------------------------- */
.app-hero{
  display:flex; align-items:center; justify-content:space-between; gap:16px;
  background:linear-gradient(120deg, var(--brand-700), var(--brand-500));
  color:#fff; border-radius:16px; padding:18px 24px; margin:.1rem 0 1.4rem;
  box-shadow:0 10px 26px rgba(0,0,140,.18);
}
.app-hero-left{ display:flex; align-items:center; gap:16px; }
.app-hero-icon{
  font-size:1.7rem; background:rgba(255,255,255,.16);
  width:52px; height:52px; border-radius:13px;
  display:flex; align-items:center; justify-content:center; flex:0 0 auto;
}
.app-hero-title{ font-size:1.3rem; font-weight:800; letter-spacing:-.02em; line-height:1.2; color:#fff; }
.app-hero-sub{ font-size:.84rem; opacity:.9; margin-top:3px; color:#fff; }
.app-hero-badge{
  background:rgba(255,255,255,.16); color:#fff; padding:7px 15px;
  border-radius:999px; font-size:.72rem; font-weight:700;
  letter-spacing:.06em; white-space:nowrap;
}

/* ---- sidebar (light, readable) --------------------------------------- */
[data-testid="stSidebar"]{ background:var(--card); border-right:1px solid var(--border); }
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] li,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"]{ color:var(--ink); }
[data-testid="stSidebar"] h3{ font-size:.95rem; margin:.2rem 0 .4rem; }
[data-testid="stSidebar"] .stMarkdown p{ font-size:.82rem; margin-bottom:.18rem; }

.sb-brand{
  display:flex; align-items:center; gap:11px;
  padding:2px 2px 14px; border-bottom:1px solid var(--border); margin-bottom:14px;
}
.sb-brand-mark{
  font-size:1.25rem; background:var(--brand-100); color:var(--brand-700);
  width:38px; height:38px; border-radius:11px;
  display:flex; align-items:center; justify-content:center;
}
.sb-brand-text{ font-weight:800; color:var(--brand-700); font-size:1.02rem; letter-spacing:-.01em; }
.sb-label{ font-size:.7rem; font-weight:700; letter-spacing:.09em; color:var(--muted); margin:2px 2px 8px; }
.sb-divider{ height:1px; background:var(--border); margin:16px 0 8px; }

/* sidebar nav buttons rendered as menu items */
[data-testid="stSidebar"] .stButton > button{
  width:100%; justify-content:flex-start; text-align:left;
  border:none; background:transparent; color:var(--muted);
  font-weight:600; padding:.5rem .75rem; border-radius:10px; box-shadow:none;
}
[data-testid="stSidebar"] .stButton > button p{ text-align:left; width:100%; font-size:.86rem; }
[data-testid="stSidebar"] .stButton > button:hover{ background:var(--brand-100); color:var(--brand-700); }
[data-testid="stSidebar"] .stButton > button:hover p{ color:var(--brand-700); }
[data-testid="stSidebar"] .stButton > button[kind="primary"]{ background:var(--brand-700); color:#fff; }
[data-testid="stSidebar"] .stButton > button[kind="primary"] p{ color:#fff !important; }
[data-testid="stSidebar"] .stButton > button:disabled{ color:#b6bdcd; background:transparent; }
[data-testid="stSidebar"] .stButton > button:disabled p{ color:#b6bdcd; }
</style>
"""


def _inject_global_style():
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)


def _render_app_header():
    """Styled header banner replacing the plain st.title (visual only)."""
    page = st.session_state.get('page', 'upload')
    step_map = {
        'upload':          (1, 'Upload'),
        'manual_entry':    (1, 'Upload'),
        'pinch_analysis':  (2, 'Pinch Analysis'),
        'case_generation': (3, 'Case Generation'),
        'calculation':     (4, 'Calculation'),
        'results':         (4, 'Results'),
    }
    step_no, step_name = step_map.get(page, (1, 'Upload'))
    st.markdown(
        f"""
        <div class="app-hero">
          <div class="app-hero-left">
            <div class="app-hero-icon">🔥</div>
            <div>
              <div class="app-hero-title">High-Temperature Heat Pump Modeling Environment</div>
              <div class="app-hero-sub">Pre-selection &amp; comparison of industrial heat-supply architectures · HTHP · MVR · Hybrid</div>
            </div>
          </div>
          <div class="app-hero-badge">STEP {step_no} / 4 · {step_name.upper()}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar_nav():
    """Persistent workflow navigation in the sidebar (respects gating)."""
    page        = st.session_state.get('page', 'upload')
    has_data    = st.session_state.get('questionnaire_data')    is not None
    has_results = st.session_state.get('calculation_results')   is not None

    # (number, label, target page, {pages that mark this step active}, reachable)
    steps = [
        ('1', 'Upload & Parameters', 'upload',          {'upload', 'manual_entry'}, True),
        ('2', 'Pinch Analysis',      'pinch_analysis',  {'pinch_analysis'},          has_data),
        ('3', 'Case Generation',     'case_generation', {'case_generation'},         has_data),
        ('4', 'Results',             'results',         {'calculation', 'results'},  has_results),
    ]

    with st.sidebar:
        st.markdown(
            '<div class="sb-brand">'
            '<span class="sb-brand-mark">🔥</span>'
            '<span class="sb-brand-text">HTHP&nbsp;Suite</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="sb-label">WORKFLOW</div>', unsafe_allow_html=True)

        for num, label, target, active_pages, reachable in steps:
            is_active = page in active_pages
            if st.button(
                f"{num}   {label}",
                key=f"nav_{target}",
                type=('primary' if is_active else 'secondary'),
                disabled=(not reachable and not is_active),
                use_container_width=True,
            ):
                st.session_state.page = target
                st.rerun()

        st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Apply theme + header  (order: style → sidebar nav → header)
# ---------------------------------------------------------------------------
_inject_global_style()
_render_sidebar_nav()
_render_app_header()


# ============================================================================
# SIDEBAR: BOUNDARY CONDITIONS  (always visible once questionnaire is loaded)
# ============================================================================
def _render_sidebar_boundary_conditions():
    params = st.session_state.get('questionnaire_data')
    reader = st.session_state.get('questionnaire_reader')
    if not params or not reader:
        with st.sidebar:
            st.markdown("### 📌 Boundary Conditions")
            st.caption("No questionnaire loaded yet.")
        return

    case = reader.get_application_case()

    # Exclude sections based on application case
    exclude_sections = set()
    if case == 'hot_water':
        exclude_sections.add('3b. Steam')
        exclude_sections.add('3c. Additional Consumer')
    elif case == 'steam':
        exclude_sections.add('3a. Hot Water')
        exclude_sections.add('3c. Additional Consumer')

    # Group parameters by section
    sections_data = {}
    for key, param_info in params.items():
        if param_info['value'] is None or param_info['value'] in ('', 'N/A', '-'):
            continue  # Skip empty values
        
        section = param_info.get('section', 'Other')
        
        # Skip excluded sections
        if section in exclude_sections:
            continue
        
        if section not in sections_data:
            sections_data[section] = []
        
        sections_data[section].append({
            'key': key,
            'description': param_info['description'],
            'value': param_info['value'],
            'unit': param_info.get('unit', ''),
        })

    # Sort sections in logical order
    section_order = [
        '1. Project Data',
        '2. Heat Source',
        '3. Heat Consumer',
        '3a. Hot Water',
        '3b. Steam',
        '3c. Additional Consumer',
        '4. Infrastructure',
        '5. Optimization',
        '6. Location & Environment',
        '7. Safety & Permissions',
        '8. Economic Parameters',
    ]

    with st.sidebar:
        st.markdown("### 📌 Boundary Conditions")
        
        first_section = True
        for section in section_order:
            if section not in sections_data or section in exclude_sections:
                continue
            
            if not first_section:
                st.markdown("---")
            first_section = False
            
            # Section header with emoji
            section_emoji = {
                '1. Project Data': '📋',
                '2. Heat Source': '🟦',
                '3. Heat Consumer': '🟥',
                '3a. Hot Water': '🟥',
                '3b. Steam': '🟥',
                '3c. Additional Consumer': '🟥',
                '4. Infrastructure': '⚙️',
                '5. Optimization': '⭐',
                '6. Location & Environment': '🌍',
                '7. Safety & Permissions': '🔒',
                '8. Economic Parameters': '💰',
            }.get(section, '▪️')
            
            st.markdown(f"**{section_emoji} {section}**")
            
            # Display parameters in this section
            for param in sections_data[section]:
                value_str = str(param['value'])
                unit_str = f" {param['unit']}" if param['unit'] else ""
                st.write(f"• **{param['description']}:** {value_str}{unit_str}")
        
        # ── Case Information ─────────────────────────────────────────────────
        st.markdown("---")
        st.caption(f"**Case:** {case.replace('_', ' ').title()}")

_render_sidebar_boundary_conditions()


# ============================================================================
# HELPER: convert Streamlit UI cases to case_calculator format
# ============================================================================
def _ui_cases_to_calc_format(ui_cases: list) -> list:
    """
    Converts the Streamlit UI case dictionaries to the format expected by
    case_calculator.calculate_cases(). No thermodynamic computations here -
    only data restructuring.

    UI case dict fields used:
      system_type   : 'HTHP' | 'MVR' | 'HTHP+MVR'
      hthp_model    : model class name string
      refrigerant   : comma-separated string for cascade, single otherwise
      efficiencies  : list[float] or dict
                      HTHP single  -> [eta_s]
                      HTHP cascade -> [eta_s_lt, eta_s_ht]
                      MVR          -> {'mvr_efficiencies': [...], 'n_stages': n}
                      HTHP+MVR     -> [eta_s_hthp, eta_s_mvr]
      overheats     : list[float] superheat values [K] (for IHX models)
      mvr_stages    : int
    """
    from case_calculator import CaseType

    calc_cases = []
    for idx, case in enumerate(ui_cases):
        system_type = case['system_type']
        effs        = case['efficiencies']
        overheats   = case.get('overheats') or []
        model_name  = case.get('hthp_model', 'HeatPumpSimple')
        is_cascade  = ',' in (case.get('refrigerant') or '')

        # -- Scalar eta_s (fallback / single-stage) --
        if isinstance(effs, list):
            eta_s = effs[0] if effs else 0.75
        elif isinstance(effs, dict):
            if 'mvr_efficiencies' in effs:
                mvr_effs = effs.get('mvr_efficiencies') or [0.80]
                eta_s = mvr_effs[0] if mvr_effs else 0.80
            else:
                eta_s = effs.get('hthp_efficiency', 0.75)
        else:
            eta_s = 0.75

        # -- Build case id --
        refrig_tag = (case.get('refrigerant') or '').replace(' ', '').replace(',', '_')
        case_id = (
            f"case_{idx+1:03d}_{system_type.replace('+', '_')}_{model_name}_{refrig_tag}"
            .lower().rstrip('_').replace('__', '_')
        )

        base = {'id': case_id}

        # ── HTHP ──────────────────────────────────────────────────────────
        if system_type == 'HTHP':
            base['type']  = CaseType.HTHP
            base['model'] = model_name

            refrigerant = case.get('refrigerant', 'R600a')
            if is_cascade:
                parts = [r.strip() for r in refrigerant.split(',')]
                base['refrigerant1'] = parts[0]
                base['refrigerant2'] = parts[1] if len(parts) > 1 else parts[0]
                # Cascade: LP and HP efficiencies may differ
                if isinstance(effs, list) and len(effs) >= 2:
                    base['eta_s_lt'] = effs[0]
                    base['eta_s_ht'] = effs[1]
                    base['eta_s']    = effs[0]   # fallback scalar
                else:
                    base['eta_s'] = eta_s
                # Kaskadentemperatur übergeben (T_mid des Kaskaden-HX)
                if case.get('t_cascade_hx') is not None:
                    base['t_cascade_hx'] = float(case['t_cascade_hx'])
            else:
                base['refrigerant'] = refrigerant
                base['eta_s'] = eta_s

            # Econ/PC models need econ_type (from UI selection, default 'closed')
            if any(k in model_name for k in ('Econ', 'PC')):
                base['econ_type'] = case.get('econ_type', 'closed')

            # Superheat for IHX models — keep full list so both sh_lp (index 0)
            # and sh_hp (index 1) reach _apply_hthp_superheat correctly.
            # Do NOT set scalar 'superheat': it takes priority and would mask overheats[1].
            if overheats:
                base['overheats'] = overheats

        # ── MVR ───────────────────────────────────────────────────────────
        elif system_type == 'MVR':
            base['type']     = CaseType.MVR
            base['eta_s']    = eta_s
            base['n_stages'] = case.get('mvr_stages')

            if isinstance(effs, dict) and 'mvr_efficiencies' in effs:
                for i, e in enumerate(effs['mvr_efficiencies'], start=1):
                    base[f'eta_s_{i}'] = e
            if case.get('dT_per_stage') is not None:
                base['dT_per_stage'] = float(case['dT_per_stage'])
            if case.get('mvr_sh') is not None:
                base['mvr_sh'] = float(case['mvr_sh'])

        # ── HTHP + MVR ────────────────────────────────────────────────────
        elif system_type == 'HTHP+MVR':
            base['type']  = CaseType.HTHP_MVR
            base['model'] = model_name

            refrigerant = case.get('refrigerant', 'R600a')
            if is_cascade:
                parts = [r.strip() for r in refrigerant.split(',')]
                base['refrigerant1'] = parts[0]
                base['refrigerant2'] = parts[1] if len(parts) > 1 else parts[0]
                # Kaskadentemperatur übergeben (T_mid des Kaskaden-HX)
                if case.get('t_cascade_hx') is not None:
                    base['t_cascade_hx'] = float(case['t_cascade_hx'])
            else:
                base['refrigerant'] = refrigerant.split(',')[0].strip()

            # HTHP-Wirkungsgrad kommt aus den Haupt-efficiencies
            if isinstance(effs, dict) and 'hthp_efficiency' in effs:
                base['eta_s_hthp'] = effs.get('hthp_efficiency', 0.75)
                base['eta_s_mvr']  = effs.get('mvr_efficiency',  0.80)
            elif isinstance(effs, list) and len(effs) >= 1:
                base['eta_s_hthp'] = effs[0]
                base['eta_s_mvr']  = effs[1] if len(effs) >= 2 else 0.80
            else:
                base['eta_s_hthp'] = eta_s
                base['eta_s_mvr']  = 0.80
            base['eta_s'] = base['eta_s_hthp']

            # MVR stufenweise Wirkungsgrade (aus dedizierten Schiebreglern)
            _mvr_stage_effs = case.get('hthp_mvr_efficiencies') or []
            if _mvr_stage_effs:
                for _i, _e in enumerate(_mvr_stage_effs, start=1):
                    base[f'eta_s_mvr_{_i}'] = float(_e)
                base['n_stages'] = len(_mvr_stage_effs)
            else:
                base['n_stages'] = case.get('mvr_stages')

            # Übergangsdruck und ΔT/Stufe
            if case.get('p_intermediate') is not None:
                base['p_intermediate'] = float(case['p_intermediate'])
            if case.get('dT_per_stage') is not None:
                base['dT_per_stage'] = float(case['dT_per_stage'])
            if case.get('mvr_sh') is not None:
                base['mvr_sh'] = float(case['mvr_sh'])

            if any(k in model_name for k in ('Econ', 'PC')):
                base['econ_type'] = case.get('econ_type', 'closed')

            if overheats:
                base['overheats'] = overheats
                # scalar 'superheat' deliberately omitted — see HTHP branch above

        calc_cases.append(base)

    return calc_cases


# ============================================================================
# HELPER: matplotlib figure renderer with PDF download button
# ============================================================================

def _pyplot_with_download(fig, filename: str, button_label: str = "⬇ Download PDF",
                          max_width: int = 820):
    """
    Renders a matplotlib figure as a responsive, vector (SVG) image and adds a
    small PDF download button below it.

    Instead of a fixed-size raster stretched edge-to-edge across the (wide)
    page, the chart is embedded as an SVG that:
      * is centred and constrained to `max_width` pixels, so it no longer fills
        the whole window right up to the border, and
      * uses width:100% / height:auto, so it scales cleanly with the window
        size and the browser zoom level (crisp at any zoom, being vector).

    Parameters
    ----------
    fig          : matplotlib Figure
    filename     : suggested filename for the download (without extension)
    button_label : label shown on the download button
    max_width    : maximum on-screen width of the chart in pixels
    """
    import io
    import re

    # --- SVG for crisp, resolution-independent scaling ----------------------
    sbuf = io.StringIO()
    fig.savefig(sbuf, format='svg', bbox_inches='tight')
    svg = sbuf.getvalue()
    svg = svg[svg.find('<svg'):]          # strip xml declaration / doctype

    # Exact aspect ratio from the viewBox (accounts for bbox_inches='tight').
    aspect = 0.4
    m = re.search(r'viewBox="[\d.]+ [\d.]+ ([\d.]+) ([\d.]+)"', svg)
    if m:
        vb_w, vb_h = float(m.group(1)), float(m.group(2))
        if vb_w:
            aspect = vb_h / vb_w

    box_id = f"resp_{filename}_{id(fig)}"
    # Height the iframe needs when the chart is at its maximum on-screen width.
    frame_h = int(max_width * aspect) + 6

    html = f"""
    <div style="width:100%; display:flex; justify-content:center;">
      <div id="{box_id}" style="width:100%; max-width:{max_width}px;">
        <style>
          #{box_id} svg {{ width:100% !important; height:auto !important;
                            display:block; }}
        </style>
        {svg}
      </div>
    </div>
    """
    st.components.v1.html(html, height=frame_h)

    # --- PDF download (vector, independent of on-screen scaling) ------------
    pbuf = io.BytesIO()
    fig.savefig(pbuf, format='pdf', bbox_inches='tight')
    pbuf.seek(0)
    st.download_button(
        label=button_label,
        data=pbuf.read(),
        file_name=f"{filename}.pdf",
        mime="application/pdf",
        key=f"dl_{filename}_{id(fig)}",
    )


def _pyplot_interactive(fig, filename: str, height: int = 660,
                        button_label: str = "⬇ Download PDF"):
    """
    Renders a matplotlib figure as an inline SVG that (1) scales with the
    window/container width and (2) can be zoomed with the mouse wheel and
    panned by dragging (via the svg-pan-zoom library from a CDN).

    The setup re-fits itself when the container first gains a real size, so
    figures placed in a tab that is hidden on first render (e.g. the Grand
    Composite Curve) still appear correctly once their tab is opened.

    A PDF download button (vector, unaffected by zoom) is added below.
    """
    import io

    # --- SVG for crisp, interactive display ---------------------------------
    sbuf = io.StringIO()
    fig.savefig(sbuf, format='svg', bbox_inches='tight')
    svg = sbuf.getvalue()
    svg = svg[svg.find('<svg'):]          # strip xml declaration / doctype

    box_id = f"pz_{filename}_{id(fig)}"
    html = f"""
    <div id="{box_id}_wrap"
         style="width:96%; height:{height}px; margin:0 auto;
                border:1px solid #e6e6e6; border-radius:10px;
                background:#fff; overflow:hidden;">
      {svg}
    </div>
    <script src="https://cdn.jsdelivr.net/npm/svg-pan-zoom@3.6.1/dist/svg-pan-zoom.min.js"></script>
    <script>
      (function() {{
        var wrap  = document.getElementById("{box_id}_wrap");
        var svgEl = wrap.querySelector("svg");
        svgEl.setAttribute("width", "100%");
        svgEl.setAttribute("height", "100%");
        svgEl.style.width = "100%";
        svgEl.style.height = "100%";
        var pz = null;
        function ready() {{
          return typeof svgPanZoom !== "undefined"
                 && wrap.clientWidth > 0 && wrap.clientHeight > 0;
        }}
        function setup() {{
          if (pz || !ready()) return;
          pz = svgPanZoom(svgEl, {{
            zoomEnabled: true, mouseWheelZoomEnabled: true,
            controlIconsEnabled: true, fit: true, center: true,
            minZoom: 0.5, maxZoom: 20, zoomScaleSensitivity: 0.3
          }});
        }}
        function refit() {{
          if (!pz) {{ setup(); return; }}
          pz.resize(); pz.fit(); pz.center();
        }}
        // Retry briefly to catch the async CDN script load.
        var tries = 0;
        var timer = setInterval(function() {{
          setup();
          if (pz || tries++ > 60) clearInterval(timer);
        }}, 150);
        window.addEventListener("load", setup);
        // Re-fit when the container first gains size (hidden tab -> visible).
        if (window.ResizeObserver) {{
          new ResizeObserver(function() {{
            if (!pz) setup(); else refit();
          }}).observe(wrap);
        }}
      }})();
    </script>
    """
    st.components.v1.html(html, height=height + 12)

    # --- PDF download (vector, independent of on-screen zoom) ---------------
    pbuf = io.BytesIO()
    fig.savefig(pbuf, format='pdf', bbox_inches='tight')
    pbuf.seek(0)
    st.download_button(
        label=button_label,
        data=pbuf.read(),
        file_name=f"{filename}.pdf",
        mime="application/pdf",
        key=f"dl_{filename}_{id(fig)}",
    )


# ============================================================================
# HELPER: State point tables  (after hp_dashboard.py)
# ============================================================================

def _build_state_table(nw, wf_name):
    """
    Extract TESPy Connection results for one fluid circuit, clean up, add units.

    nw      : TESPy Network object  (hp.nw  or  mvr.nw)
    wf_name : CoolProp name to filter rows by, or None (show all rows)
    """
    import numpy as np

    df = nw.results['Connection'].copy()

    # Drop unit-suffix meta columns
    df = df.loc[:, ~df.columns.str.contains('_unit', case=False, regex=False)]
    if 'Td_bp' in df.columns:
        df = df.drop(columns=['Td_bp'])

    # Filter to the requested fluid
    if wf_name and wf_name in df.columns:
        df = df[df[wf_name] == 1.0].copy()

    # Drop all fluid-fraction columns (they are all True/1.0 after the filter above)
    fluid_cols = []
    for col in df.columns:
        try:
            if set(df[col].dropna().unique()).issubset({0.0, 1.0, True, False}):
                fluid_cols.append(col)
        except Exception:
            pass
    df = df.drop(columns=fluid_cols, errors='ignore')

    # Format vapour quality: negative = superheated  →  show as "—"
    if 'x' in df.columns:
        def _fmt_x(v):
            try:
                fv = float(v)
                return '—' if fv < 0 else f'{fv:.4f}'
            except Exception:
                return str(v)
        df['x'] = df['x'].apply(_fmt_x)

    # Round all remaining numeric columns to 5 significant figures
    for col in df.columns:
        if df[col].dtype == object:
            continue
        try:
            df[col] = df[col].apply(lambda v: f'{float(v):.5g}' if pd.notna(v) else '—')
        except Exception:
            pass

    # Rename to English with units
    df = df.rename(columns={
        'm':    'm [kg/s]',
        'p':    'p [bar]',
        'h':    'h [kJ/kg]',
        'T':    'T [°C]',
        'v':    'v [m³/kg]',
        'vol':  'vol [m³/s]',
        's':    's [kJ/(kg·K)]',
        'x':    'x [-]',
        'e_ph': 'e_ph [kJ/kg]',
        'e_ch': 'e_ch [kJ/kg]',
    })
    return df


def _render_hp_state_table(hp):
    """Render state point table(s) for a heatpumps model (single or cascade)."""
    try:
        is_cascade = hasattr(hp, 'wf1') and hasattr(hp, 'wf2')
        if is_cascade:
            wf1 = hp.params['setup']['refrig1']
            wf2 = hp.params['setup']['refrig2']
            st.markdown(f"**LP circuit  ·  {wf1}**")
            df1 = _build_state_table(hp.nw, wf1)
            st.dataframe(df1, use_container_width=True)
            st.markdown(f"**HP circuit  ·  {wf2}**")
            df2 = _build_state_table(hp.nw, wf2)
            st.dataframe(df2, use_container_width=True)
        else:
            wf = hp.params['setup']['refrig']
            st.caption(f"Working fluid: **{wf}**")
            df = _build_state_table(hp.nw, wf)
            st.dataframe(df, use_container_width=True)
    except Exception as e:
        st.warning(f"State points not available: {e}")


def _render_state_points(result, ui_case: dict):
    """Render expandable state point tables for any case type."""
    from case_calculator import CaseType as _CT
    model_inst = result.model_instance
    if model_inst is None:
        return

    with st.expander("📋 State Points"):
        if isinstance(model_inst, tuple):
            # HTHP + MVR
            hp_inst, mvr_inst = model_inst
            st.markdown("##### HTHP Circuit")
            _render_hp_state_table(hp_inst)
            st.markdown("---")
            st.markdown("##### MVR Circuit  (Water / Steam)")
            try:
                df_mvr = _build_state_table(mvr_inst.nw, wf_name=None)
                st.dataframe(df_mvr, use_container_width=True)
            except Exception as e:
                st.warning(f"MVR state points not available: {e}")
        elif result.case_type == _CT.MVR:
            try:
                df_mvr = _build_state_table(model_inst.nw, wf_name=None)
                st.dataframe(df_mvr, use_container_width=True)
            except Exception as e:
                st.warning(f"State points not available: {e}")
        else:
            _render_hp_state_table(model_inst)


# ============================================================================
# HELPER: render a single case result
# ============================================================================
def _extract_compressor_table(hp):
    """
    Try to extract per-compressor pressure ratios and efficiency from TESPy network.
    Returns a list of dicts (one per compressor), or empty list on failure.
    """
    rows = []
    try:
        for label, comp in hp.comps.items():
            if not hasattr(comp, 'eta_s'):
                continue
            try:
                in_c  = comp.inl[0]
                out_c = comp.outl[0]
                p_in  = in_c.p.val
                p_out = out_c.p.val
                PR    = p_out / p_in if p_in > 0 else None
                rows.append({
                    'Stage':         label,
                    'p_in [bar]':    round(p_in,  3),
                    'p_out [bar]':   round(p_out, 3),
                    'PR [-]':        round(PR, 3) if PR else '—',
                    'T_in [°C]':     round(in_c.T.val,  2),
                    'T_out [°C]':    round(out_c.T.val, 2),
                    'h_in [kJ/kg]':  round(in_c.h.val,  1),
                    'h_out [kJ/kg]': round(out_c.h.val, 1),
                    'η_s [-]':       round(comp.eta_s.val, 4),
                    'm [kg/s]':      round(in_c.m.val,  5),
                })
            except Exception:
                pass
    except Exception:
        pass
    return rows



def _extract_cascade_per_stage(hp_model) -> list:
    """
    Extract per-circuit (LP / HP) W_comp, Q_evap, Q_cond, COP from a cascade
    heat pump model that uses a single shared TESPy network (heatpumps standard).
    """
    try:
        setup = hp_model.params.get('setup', {})
        wf1 = setup.get('refrig1') or setup.get('refrig')
        wf2 = setup.get('refrig2')
        if not wf1:
            return []

        circuits = {}
        for wf, lbl, icon in [(wf1, 'LP Circuit (Stage 1)', '🔵'),
                               (wf2, 'HP Circuit (Stage 2)', '🔴')]:
            if wf:
                circuits[wf] = {'label': f'{icon} {lbl}', 'refrig': wf,
                                 'W_comp': 0.0, 'Q_evap': 0.0, 'Q_cond': 0.0}

        def _dominant_fluid(conn):
            try:
                fv = conn.fluid.val
                best = max(fv, key=fv.get)
                return best if fv.get(best, 0) > 0.5 else None
            except Exception:
                return None

        for _comp_key, comp in hp_model.comps.items():
            if not comp.inl:
                continue
            wf = _dominant_fluid(comp.inl[0])
            if wf not in circuits:
                continue

            # Compressor
            if hasattr(comp, 'eta_s'):
                try:
                    circuits[wf]['W_comp'] += abs(comp.P.val) / 1000.0
                except Exception:
                    pass
                continue  # skip HX logic for compressors

            # Heat exchanger — try Q attribute, then enthalpy balance fallback
            q_kw = None
            if hasattr(comp, 'Q'):
                try:
                    q_kw = comp.Q.val / 1000.0
                except Exception:
                    pass
            if q_kw is None and comp.outl:
                try:
                    m  = comp.inl[0].m.val
                    dh = comp.outl[0].h.val - comp.inl[0].h.val
                    q_kw = m * dh  # kW
                except Exception:
                    pass
            if q_kw is not None:
                if q_kw > 0:
                    circuits[wf]['Q_evap'] += q_kw
                else:
                    circuits[wf]['Q_cond'] += abs(q_kw)

        rows = []
        for wf, d in circuits.items():
            W  = d['W_comp'] or None
            Qc = d['Q_cond'] or None
            Qe = d['Q_evap'] or None
            cop = (Qc / W) if (Qc and W and W > 0) else None
            rows.append({
                'Stage':       f"{d['label']} · {wf}",
                'W_comp [kW]': f'{W:.2f}'   if W   is not None else '—',
                'Q_evap [kW]': f'{Qe:.2f}'  if Qe  is not None else '—',
                'Q_cond [kW]': f'{Qc:.2f}'  if Qc  is not None else '—',
                'COP [-]':     f'{cop:.4f}' if cop is not None else '—',
            })
        return rows
    except Exception:
        return []


def _render_hthp_detail(result):
    """Detailed HTHP performance breakdown (single-fluid and cascade)."""
    ex    = result.extra
    p_use = result.params_used or {}
    is_cascade = bool(ex.get('refrigerant1'))

    # ── COP comparison table ──────────────────────────────────────────────
    st.markdown("#### COP & Efficiency Indicators")
    _cop_rows = []
    if result.COP       is not None: _cop_rows.append(('COP (simulated)',    f'{result.COP:.4f}'))
    if ex.get('cop_lorenz') is not None: _cop_rows.append(('COP Lorenz',     f'{ex["cop_lorenz"]:.4f}'))
    if ex.get('cop_carnot') is not None: _cop_rows.append(('COP Carnot',     f'{ex["cop_carnot"]:.4f}'))
    if ex.get('eta_lorenz') is not None: _cop_rows.append(('η Lorenz',       f'{ex["eta_lorenz"]:.4f}'))
    if result.epsilon   is not None: _cop_rows.append(('ε exergy',           f'{result.epsilon:.4f}'))
    if _cop_rows:
        st.dataframe(pd.DataFrame(_cop_rows, columns=['Indicator', 'Value']),
                     use_container_width=True, hide_index=True)

    # ── Heat & power ──────────────────────────────────────────────────────
    st.markdown("#### Heat & Power Balance")
    _pwr_rows = []
    if result.Q_con  is not None: _pwr_rows.append(('Q_cond [kW]',     f'{result.Q_con:.2f}'))
    if result.W_comp is not None: _pwr_rows.append(('W_comp [kW]',     f'{result.W_comp:.2f}'))
    if result.Q_con and result.W_comp:
        Q_source = result.Q_con - result.W_comp
        _pwr_rows.append(('Q_source [kW]',  f'{Q_source:.2f}'))
    if _pwr_rows:
        st.dataframe(pd.DataFrame(_pwr_rows, columns=['Quantity', 'Value']),
                     use_container_width=True, hide_index=True)

    # ── Temperature levels ────────────────────────────────────────────────
    st.markdown("#### Temperature & Pressure Levels")
    _temp_rows = []
    if result.T_cold_in is not None: _temp_rows.append(('T_source_in [°C]',  f'{result.T_cold_in:.1f}'))
    if result.T_hot_out is not None: _temp_rows.append(('T_sink_out [°C]',   f'{result.T_hot_out:.1f}'))
    for _lbl, _key in [('p_sink [bar]', 'C3'), ('p_source [bar]', 'B1')]:
        _p = p_use.get(_key, {}).get('p')
        if _p: _temp_rows.append((_lbl, f'{_p:.4f}'))
    if _temp_rows:
        st.dataframe(pd.DataFrame(_temp_rows, columns=['Quantity', 'Value']),
                     use_container_width=True, hide_index=True)

    # ── Compressor details ────────────────────────────────────────────────
    _model_inst = result.model_instance
    _rows = _extract_compressor_table(_model_inst) if _model_inst else []
    if _rows:
        st.markdown("#### Compressor Details")
        st.dataframe(pd.DataFrame(_rows), use_container_width=True, hide_index=True)

    # ── Per-stage summary (cascade only) ──────────────────────────────────
    if is_cascade and _model_inst is not None:
        st.markdown("#### Per-Stage Summary (Cascade)")
        _stage_rows = _extract_cascade_per_stage(_model_inst)
        if _stage_rows:
            st.dataframe(pd.DataFrame(_stage_rows),
                         use_container_width=True, hide_index=True)
        else:
            st.info("Per-stage metrics not available for this cascade model.")

    # ── Refrigerant info ──────────────────────────────────────────────────
    if is_cascade:
        st.caption(
            f"LP refrigerant: **{ex.get('refrigerant1', '—')}**  ·  "
            f"HP refrigerant: **{ex.get('refrigerant2', '—')}**"
        )
    elif ex.get('refrigerant'):
        st.caption(f"Refrigerant: **{ex['refrigerant']}**")


def _render_mvr_detail(result):
    """Detailed MVR performance breakdown."""
    ex = result.extra

    # ── Power balance ─────────────────────────────────────────────────────
    st.markdown("#### Power Balance")
    _pwr_rows = []
    _w_comp = ex.get('W_compressor_kW', result.W_comp)
    _w_ph   = ex.get('W_electric_preheater_kW', 0.0) or 0.0
    _pwr_rows.append(('W_compressors [kW]',  f'{_w_comp:.2f}'))
    if _w_ph > 0:
        _pwr_rows.append(('W_preheater [kW]',   f'{_w_ph:.2f}'))
    _pwr_rows.append(('W_total [kW]',         f'{result.W_comp:.2f}'))
    if result.Q_con:
        _pwr_rows.append(('Q_heat [kW]',      f'{result.Q_con:.2f}'))
    if result.COP:
        _pwr_rows.append(('COP [-]',          f'{result.COP:.4f}'))
    st.dataframe(pd.DataFrame(_pwr_rows, columns=['Quantity', 'Value']),
                 use_container_width=True, hide_index=True)

    # ── SEI breakdown ─────────────────────────────────────────────────────
    st.markdown("#### Specific Energy Input (SEI)")
    _sei_rows = []
    if ex.get('SEI_compressor_kWh_per_kg') is not None:
        _sei_rows.append(('SEI compressor only [kWh/kg]', f'{ex["SEI_compressor_kWh_per_kg"]:.5f}'))
    if ex.get('SEI_total_kWh_per_kg') is not None:
        _sei_rows.append(('SEI total [kWh/kg]',           f'{ex["SEI_total_kWh_per_kg"]:.5f}'))
    if ex.get('SEI_MJ_per_kg') is not None:
        _sei_rows.append(('SEI [MJ/kg]',                  f'{ex["SEI_MJ_per_kg"]:.4f}'))
    if ex.get('specific_work_kJ_per_kg') is not None:
        _sei_rows.append(('Specific work [kJ/kg]',        f'{ex["specific_work_kJ_per_kg"]:.3f}'))
    if _sei_rows:
        st.dataframe(pd.DataFrame(_sei_rows, columns=['Indicator', 'Value']),
                     use_container_width=True, hide_index=True)

    # ── Mass flows ────────────────────────────────────────────────────────
    st.markdown("#### Mass Flows & Compression")
    _mf_rows = []
    if ex.get('m_steam_in_kg_s')  is not None:
        _mf_rows.append(('ṁ inlet [kg/s]',           f'{ex["m_steam_in_kg_s"]:.5f}'))
        _mf_rows.append(('ṁ inlet [kg/h]',           f'{ex["m_steam_in_kg_s"]*3600:.2f}'))
    if ex.get('m_steam_out_kg_s') is not None:
        _mf_rows.append(('ṁ outlet [kg/s]',          f'{ex["m_steam_out_kg_s"]:.5f}'))
        _mf_rows.append(('ṁ outlet [kg/h]',          f'{ex["m_steam_out_kg_s"]*3600:.2f}'))
    if ex.get('total_water_injected_kg_s') is not None:
        _mf_rows.append(('ṁ water injected [kg/s]',  f'{ex["total_water_injected_kg_s"]:.5f}'))
    if ex.get('compression_ratio') is not None:
        _mf_rows.append(('PR total [-]',              f'{ex["compression_ratio"]:.4f}'))
    if ex.get('n_stages') is not None:
        _mf_rows.append(('Stages [-]',                str(ex['n_stages'])))
    if ex.get('T_steam_out') is not None:
        _mf_rows.append(('T_out last stage [°C]',    f'{ex["T_steam_out"]:.2f}'))
    if _mf_rows:
        st.dataframe(pd.DataFrame(_mf_rows, columns=['Quantity', 'Value']),
                     use_container_width=True, hide_index=True)

    # ── Per-stage injection rates ─────────────────────────────────────────
    inj = ex.get('water_injection_rates_kg_s')
    if inj and len(inj) > 0:
        st.markdown("#### Water Injection per Stage")
        _inj_rows = [
            {'Stage': f'Stage {k+1} → {k+2}',
             'ṁ_inj [kg/s]': f'{v:.5f}',
             'ṁ_inj [kg/h]': f'{v*3600:.3f}'}
            for k, v in enumerate(inj)
        ]
        st.dataframe(pd.DataFrame(_inj_rows), use_container_width=True, hide_index=True)

    _note = result.extra.get('note')
    if _note:
        st.warning(f"⚡ {_note}")

    # ── Exergy loss analysis ───────────────────────────────────────────────
    _mvr_inst = result.model_instance
    if _mvr_inst is not None and hasattr(_mvr_inst, 'calc_exergy_losses'):
        with st.expander("🔥 Exergy Loss Analysis (MVR)", expanded=False):
            try:
                _ex_res = _mvr_inst.calc_exergy_losses()
                # Overall metrics
                st.markdown(
                    f"**Reference state:** {_ex_res['T_amb_C']} °C · "
                    f"{_ex_res['p_amb_bar']} bar  |  "
                    f"h₀ = {_ex_res['h0_kJ_kg']} kJ/kg  ·  "
                    f"s₀ = {_ex_res['s0_kJ_kgK']} kJ/(kg·K)"
                )
                _ov_rows = [
                    ('E_F (electricity) [kW]',  f"{_ex_res['E_F_total_kW']:.3f}"),
                    ('E_P (net steam exergy) [kW]', f"{_ex_res['E_P_total_kW']:.3f}"),
                    ('E_D total [kW]',           f"{_ex_res['E_D_total_kW']:.3f}"),
                    ('ε overall [-]',
                     f"{_ex_res['epsilon_total']:.4f}" if _ex_res['epsilon_total'] is not None else '—'),
                ]
                st.dataframe(pd.DataFrame(_ov_rows, columns=['Quantity', 'Value']),
                             use_container_width=True, hide_index=True)

                # Per-component table
                st.markdown("##### Per-Component Breakdown")
                _comp_rows = []
                for _sd in _ex_res['stages']:
                    _comp_rows.append({
                        'Component':    _sd['component'],
                        'W_comp [kW]':  f"{_sd['W_comp [kW]']:.3f}" if _sd['W_comp [kW]'] is not None else '—',
                        'E_P [kW]':     f"{_sd['E_P [kW]']:.3f}",
                        'E_D [kW]':     f"{_sd['E_D [kW]']:.3f}",
                        'ε [-]':        f"{_sd['ε [-]']:.4f}" if _sd['ε [-]'] is not None else '—',
                        'T_in [°C]':    f"{_sd['T_in [°C]']:.1f}" if _sd['T_in [°C]'] is not None else '—',
                        'T_out [°C]':   f"{_sd['T_out [°C]']:.1f}" if _sd['T_out [°C]'] is not None else '—',
                    })
                st.dataframe(pd.DataFrame(_comp_rows),
                             use_container_width=True, hide_index=True)
            except Exception as _ex_err:
                st.warning(f"Exergy analysis failed: {_ex_err}")


def _render_hthp_mvr_detail(result):
    """Detailed HTHP+MVR hybrid performance breakdown."""
    ex = result.extra

    # ── Power & heat split ────────────────────────────────────────────────
    st.markdown("#### Power & Heat Split  (HTHP | MVR | System)")
    _split_rows = [
        ('W_HTHP [kW]',         f'{ex.get("W_hthp_kW", 0):.2f}'),
        ('W_MVR [kW]',          f'{ex.get("W_mvr_kW",  0):.2f}'),
        ('W_total [kW]',        f'{result.W_comp:.2f}'),
        ('Q_cond HTHP [kW]',    f'{ex.get("Q_cond_hthp_kW", 0):.2f}'),
        ('Q_heat MVR [kW]',     f'{ex.get("Q_heat_mvr_kW",  0):.2f}'),
        ('Q_system [kW]',       f'{result.Q_con:.2f}' if result.Q_con else '—'),
    ]
    st.dataframe(pd.DataFrame(_split_rows, columns=['Quantity', 'Value']),
                 use_container_width=True, hide_index=True)

    # ── COP comparison ────────────────────────────────────────────────────
    st.markdown("#### COP & Efficiency")
    _cop_rows = []
    if ex.get('COP_hthp')   is not None: _cop_rows.append(('COP HTHP (refrigerant cycle)',  f'{ex["COP_hthp"]:.4f}'))
    if ex.get('COP_system') is not None: _cop_rows.append(('COP System (total)',             f'{ex["COP_system"]:.4f}'))
    if ex.get('cop_lorenz_hthp') is not None: _cop_rows.append(('COP Lorenz (HTHP)',        f'{ex["cop_lorenz_hthp"]:.4f}'))
    if result.epsilon        is not None: _cop_rows.append(('ε exergy [-]',                  f'{result.epsilon:.4f}'))
    if _cop_rows:
        st.dataframe(pd.DataFrame(_cop_rows, columns=['Indicator', 'Value']),
                     use_container_width=True, hide_index=True)

    # ── SEI breakdown ─────────────────────────────────────────────────────
    st.markdown("#### SEI Breakdown")
    _sei_rows = []
    if ex.get('SEI_mvr_kWh_per_kg')    is not None: _sei_rows.append(('SEI MVR only [kWh/kg]',    f'{ex["SEI_mvr_kWh_per_kg"]:.5f}'))
    if ex.get('SEI_system_kWh_per_kg') is not None: _sei_rows.append(('SEI system [kWh/kg]',       f'{ex["SEI_system_kWh_per_kg"]:.5f}'))
    if result.SEI                      is not None: _sei_rows.append(('SEI (result) [kWh/kg]',     f'{result.SEI:.5f}'))
    if _sei_rows:
        st.dataframe(pd.DataFrame(_sei_rows, columns=['Indicator', 'Value']),
                     use_container_width=True, hide_index=True)

    # ── MVR mass flow & stage info ─────────────────────────────────────────
    st.markdown("#### MVR Stage & Mass Flow")
    _mvr_rows = []
    if ex.get('p_intermediate_bar')          is not None: _mvr_rows.append(('p_intermediate [bar]',              f'{ex["p_intermediate_bar"]:.4f}'))
    if ex.get('n_stages_mvr')                is not None: _mvr_rows.append(('MVR stages [-]',                     str(ex['n_stages_mvr'])))
    if ex.get('m_steam_int_kg_s')            is not None:
        _mvr_rows.append(('ṁ HTHP→MVR stage 1 [kg/s]',  f'{ex["m_steam_int_kg_s"]:.5f}'))
        _mvr_rows.append(('ṁ HTHP→MVR stage 1 [kg/h]',  f'{ex["m_steam_int_kg_s"]*3600:.2f}'))
    if ex.get('total_water_injected_kg_s')   is not None: _mvr_rows.append(('ṁ water injected total [kg/s]',     f'{ex["total_water_injected_kg_s"]:.5f}'))
    if ex.get('m_steam_out_kg_s')            is not None:
        _mvr_rows.append(('ṁ final steam out [kg/s]',    f'{ex["m_steam_out_kg_s"]:.5f}'))
        _mvr_rows.append(('ṁ final steam out [kg/h]',    f'{ex["m_steam_out_kg_s"]*3600:.2f}'))
    if result.T_hot_out                      is not None: _mvr_rows.append(('T_out MVR last stage [°C]',          f'{result.T_hot_out:.2f}'))
    if _mvr_rows:
        st.dataframe(pd.DataFrame(_mvr_rows, columns=['Quantity', 'Value']),
                     use_container_width=True, hide_index=True)

    # ── HTHP compressor details ────────────────────────────────────────────
    _model_inst = result.model_instance
    if isinstance(_model_inst, tuple):
        hp_inst, _ = _model_inst
        _rows = _extract_compressor_table(hp_inst)
        if _rows:
            st.markdown("#### HTHP Compressor Details")
            st.dataframe(pd.DataFrame(_rows), use_container_width=True, hide_index=True)

        # ── Per-stage summary for cascade HTHP inside HTHP+MVR ───────────
        if ex.get('refrigerant1'):
            st.markdown("#### HTHP Per-Stage Summary (Cascade)")
            _stage_rows = _extract_cascade_per_stage(hp_inst)
            if _stage_rows:
                st.dataframe(pd.DataFrame(_stage_rows),
                             use_container_width=True, hide_index=True)
            else:
                st.info("Per-stage metrics not available for this cascade model.")

    if ex.get('refrigerant1'):
        st.caption(
            f"HTHP LP refrigerant: **{ex['refrigerant1']}**  ·  "
            f"HP refrigerant: **{ex['refrigerant2']}**  ·  "
            f"Model: **{ex.get('model_hthp', '—')}**"
        )
    elif ex.get('refrigerant'):
        st.caption(f"HTHP refrigerant: **{ex['refrigerant']}**  ·  Model: **{ex.get('model_hthp', '—')}**")


def _render_case_result(result, ui_case: dict):
    from case_calculator import CaseStatus, CaseType as _CT

    if result.status == CaseStatus.SUCCESS:
        st.success("✅ Calculation completed successfully")
    elif result.status == CaseStatus.FAILED:
        st.error("❌ Calculation failed")
        with st.expander("Error details"):
            st.code(result.error or "Unknown error")
        return
    else:
        st.warning(f"⚠️ Case skipped: {result.error}")
        return

    _is_mvr_type = result.case_type in (_CT.MVR, _CT.HTHP_MVR)
    _q_label = "Q_heat [kW]" if _is_mvr_type else "Q_cond [kW]"

    # ── Top-level KPI strip ───────────────────────────────────────────────
    st.markdown("#### Key Performance Indicators")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("COP",           f"{result.COP:.3f}"      if result.COP     is not None else "—")
    c2.metric(_q_label,        f"{result.Q_con:.1f}"    if result.Q_con   is not None else "—")
    c3.metric("W_el [kW]",     f"{result.W_comp:.1f}"   if result.W_comp  is not None else "—")
    c4.metric("ε exergy",      f"{result.epsilon:.3f}"  if result.epsilon is not None else "—")
    c5.metric("SEI [kWh/kg]",  f"{result.SEI:.4f}"      if result.SEI     is not None else "—")

    st.markdown("---")

    # ── Type-specific detail section ──────────────────────────────────────
    if result.case_type == _CT.HTHP:
        _render_hthp_detail(result)
    elif result.case_type == _CT.MVR:
        _render_mvr_detail(result)
    elif result.case_type == _CT.HTHP_MVR:
        _render_hthp_mvr_detail(result)

    # ── State diagrams ────────────────────────────────────────────────────
    diagram_paths = result.extra.get('diagram_paths', {})
    if diagram_paths:
        st.markdown("---")
        st.markdown("#### State Diagrams")
        st.caption("\U0001f4a1 Scroll to zoom · drag to pan · double-click to reset")

        import plotly.graph_objects as go
        from PIL import Image as _PILImage

        def _plotly_image(img_path: str, title: str):
            img = _PILImage.open(img_path)
            w, h = img.size
            fig = go.Figure()
            fig.add_layout_image(dict(
                source=img, xref="x", yref="y",
                x=0, y=h, sizex=w, sizey=h,
                sizing="stretch", layer="below",
            ))
            fig.update_xaxes(range=[0, w], showgrid=False, zeroline=False,
                             showticklabels=False, visible=False)
            fig.update_yaxes(range=[0, h], showgrid=False, zeroline=False,
                             showticklabels=False, visible=False)
            fig.update_layout(
                title=dict(text=title, font=dict(size=13)),
                margin=dict(l=0, r=0, t=30, b=0),
                height=380, dragmode="pan",
            )
            return fig

        def _label_for_key(dtype: str) -> str:
            _map = {
                'hthp_logph': 'HTHP \u2013 log(p)-h',
                'hthp_Ts':    'HTHP \u2013 T-s',
                'mvr_logph':  'MVR \u2013 log(p)-h (Water)',
                'logph':          'log(p)-h',
                'Ts':             'T-s',
            }
            return _map.get(dtype, dtype)

        _diagram_items = []
        for dtype, path in diagram_paths.items():
            lbl = _label_for_key(dtype)
            if isinstance(path, list):
                for j, p in enumerate(path):
                    if p and os.path.exists(p):
                        suffix = f' \u2013 Cycle {j+1}' if len(path) > 1 else ''
                        _diagram_items.append((p, lbl + suffix))
            elif isinstance(path, str) and os.path.exists(path):
                _diagram_items.append((path, lbl))

        _cols = st.columns(2) if len(_diagram_items) >= 2 else st.columns(1)
        for _idx, (p, lbl) in enumerate(_diagram_items):
            with _cols[_idx % len(_cols)]:
                _chart_key = f"diagram_{result.case_id}_{_idx}_{os.path.basename(p)}"
                st.plotly_chart(
                    _plotly_image(p, lbl),
                    use_container_width=True,
                    config={"scrollZoom": True, "displayModeBar": False},
                    key=_chart_key,
                )
                # ── Download buttons for this state diagram ───────────────
                _dl_stem = os.path.splitext(os.path.basename(p))[0]
                _dl_key  = f"dl_diag_{result.case_id}_{_idx}"
                _pdf_path = os.path.join(os.path.dirname(p), _dl_stem + '.pdf')

                col_pdf, col_png = st.columns(2)
                # PDF: offer the original vector file saved alongside the PNG
                if os.path.exists(_pdf_path):
                    with open(_pdf_path, 'rb') as _fh_pdf:
                        col_pdf.download_button(
                            label="⬇ PDF",
                            data=_fh_pdf.read(),
                            file_name=f"{_dl_stem}.pdf",
                            mime="application/pdf",
                            key=_dl_key + "_pdf",
                        )
                # PNG download (always available)
                with open(p, 'rb') as _fh_png:
                    col_png.download_button(
                        label="⬇ PNG",
                        data=_fh_png.read(),
                        file_name=os.path.basename(p),
                        mime="image/png",
                        key=_dl_key + "_png",
                    )

    # ── State point table ─────────────────────────────────────────────────
    st.markdown("---")
    _render_state_points(result, ui_case)


# ============================================================================
# HELPER: comparative overview
# ============================================================================
def _render_comparative_overview(results: dict, ui_cases: list):
    from case_calculator import CaseStatus

    n_ok   = sum(1 for r in results.values() if r.status == CaseStatus.SUCCESS)
    n_fail = sum(1 for r in results.values() if r.status != CaseStatus.SUCCESS)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total cases",      len(results))
    c2.metric("Successful",       n_ok)
    c3.metric("Failed / skipped", n_fail)

    # ── Build main results DataFrame ──────────────────────────────────────
    rows = []
    for result in results.values():
        ex = result.extra
        row = {
            'Case ID':            result.case_id,
            'Type':               result.case_type.value if result.case_type else '—',
            'Status':             result.status.value,
            'COP [-]':            result.COP,
            'COP Lorenz [-]':     ex.get('cop_lorenz') or ex.get('cop_lorenz_hthp'),
            'COP HTHP [-]':       ex.get('COP_hthp'),
            'Q_heat [kW]':        result.Q_con,
            'W_el [kW]':          result.W_comp,
            'ε exergy [-]':       result.epsilon,
            'SEI [kWh/kg]':       result.SEI,
            'T_source_in [°C]':   result.T_cold_in,
            'T_sink_out [°C]':    result.T_hot_out,
            'W_HTHP [kW]':        ex.get('W_hthp_kW'),
            'W_MVR [kW]':         ex.get('W_mvr_kW'),
            'n_stages MVR':       ex.get('n_stages') or ex.get('n_stages_mvr'),
            'p_int [bar]':        ex.get('p_intermediate_bar'),
            'ṁ_w_inj [kg/s]':    ex.get('total_water_injected_kg_s'),
        }
        rows.append(row)
    df = pd.DataFrame(rows)

    # Drop columns that are entirely None/NaN (keep only populated ones)
    df_display = df.dropna(axis=1, how='all')

    st.markdown("---")
    st.markdown("#### Results Table")
    st.dataframe(df_display, use_container_width=True)

    success_df = df[df['Status'] == 'success'].set_index('Case ID')
    if success_df.empty:
        st.info("No successful cases to chart.")
        return

    # ── Performance charts ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Performance Comparison Charts")
    tab_cop, tab_power, tab_eff, tab_sei = st.tabs(
        ["COP", "Heat & Power", "Exergy Efficiency", "SEI"]
    )

    with tab_cop:
        _cop_cols = [c for c in ['COP [-]', 'COP Lorenz [-]', 'COP HTHP [-]'] if c in success_df.columns]
        d = success_df[_cop_cols].dropna(how='all')
        if not d.empty:
            st.bar_chart(d)
        else:
            st.info("COP not available for any successful case.")

    with tab_power:
        _pwr_cols = [c for c in ['Q_heat [kW]', 'W_el [kW]', 'W_HTHP [kW]', 'W_MVR [kW]'] if c in success_df.columns]
        d = success_df[_pwr_cols].dropna(how='all')
        if not d.empty:
            st.bar_chart(d)
        else:
            st.info("Heat / power data not available.")

    with tab_eff:
        d = success_df[['ε exergy [-]']].dropna() if 'ε exergy [-]' in success_df.columns else pd.DataFrame()
        if not d.empty:
            st.bar_chart(d)
        else:
            st.info("Exergy efficiency not available for any successful case.")

    with tab_sei:
        d = success_df[['SEI [kWh/kg]']].dropna() if 'SEI [kWh/kg]' in success_df.columns else pd.DataFrame()
        if not d.empty:
            st.bar_chart(d)
        else:
            st.info("SEI applies only to MVR / hybrid cases.")

    # ── Annual electricity cost table ─────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 💰 Annual Electricity Cost Estimate")

    params = st.session_state.get('questionnaire_data') or {}

    def _get_val(d, key):
        v = d.get(key)
        if isinstance(v, dict):
            return v.get('value')
        return v

    _price_raw = _get_val(params, 'electricity_price')
    _hours_raw = _get_val(params, 'annual_operating_hours')

    # Allow manual override via number_input
    col_ep, col_oh = st.columns(2)
    with col_ep:
        try:
            _price_def = float(_price_raw) if _price_raw not in (None, '', 'N/A') else 0.12
        except Exception:
            _price_def = 0.12
        elec_price = st.number_input(
            "Electricity price [€/kWh]",
            min_value=0.0, max_value=2.0,
            value=_price_def, step=0.005, format="%.4f",
            key="cost_elec_price",
            help="Read from questionnaire Section 8 – override here if needed."
        )

    with col_oh:
        try:
            _hours_def = float(_hours_raw) if _hours_raw not in (None, '', 'N/A') else 8000.0
        except Exception:
            _hours_def = 8000.0
        op_hours = st.number_input(
            "Annual operating hours [h/yr]",
            min_value=0.0, max_value=8760.0,
            value=_hours_def, step=100.0,
            key="cost_op_hours",
            help="Read from questionnaire Section 8 – override here if needed."
        )

    if elec_price > 0 and op_hours > 0:
        cost_rows = []
        for result in results.values():
            if result.status != CaseStatus.SUCCESS or result.W_comp is None:
                continue
            w_kw = result.W_comp
            cost_yr    = w_kw * op_hours * elec_price          # €/yr
            cost_day   = w_kw * (op_hours / 365.0) * elec_price # €/day  (approx)
            cost_mwh   = elec_price * 1000.0                    # €/MWh
            cost_rows.append({
                'Case ID':               result.case_id,
                'Type':                  result.case_type.value if result.case_type else '—',
                'W_el [kW]':             f'{w_kw:.1f}',
                'Annual energy [MWh/yr]':f'{w_kw * op_hours / 1000:.1f}',
                'Cost/yr [k€/yr]':       f'{cost_yr / 1000:.1f}',
                'Cost/yr [€/yr]':        f'{cost_yr:,.0f}',
                'Cost/day [€/day]':      f'{cost_day:,.0f}',
            })

        if cost_rows:
            df_cost = pd.DataFrame(cost_rows)
            st.dataframe(df_cost, use_container_width=True, hide_index=True)

            # Bar chart: annual cost
            df_cost_chart = pd.DataFrame({
                r['Case ID']: [float(r['Annual energy [MWh/yr]'])]
                for r in cost_rows
            }).T
            df_cost_chart.columns = ['Annual energy [MWh/yr]']
            st.caption(
                f"Basis: {elec_price:.4f} €/kWh  ·  {op_hours:.0f} h/yr  ·  "
                f"equivalent tariff {elec_price * 1000:.1f} €/MWh"
            )
            st.bar_chart(df_cost_chart)
        else:
            st.info("No successful cases to calculate costs for.")
    else:
        st.info("Enter a valid electricity price and operating hours above to calculate annual costs.")


# ============================================================================
# REFRIGERANT CATALOGUE & OVERVIEW HELPER
# ============================================================================

# Extended catalogue: GWP, safety class, chemical family
# Sources: Annex 58 (industrial HPs > 100 °C), Annex 68 (thermodynamic screening),
#          EU F-Gas Regulation 2024/573 – only refrigerants without phase-out ban included
_REFRIG_CATALOGUE = {
    # ── Hydrocarbons (HC) ───────────────────────────────────────────────────
    'R290':        {'name': 'Propane',            'coolprop': 'Propane',      'GWP100': 3,    'safety': 'A3',  'family': 'HC',   'annex': 'A58/A68'},
    'R600':        {'name': 'n-Butane',           'coolprop': 'n-Butane',     'GWP100': 4,    'safety': 'A3',  'family': 'HC',   'annex': 'A58/A68'},
    'R600a':       {'name': 'Isobutane',          'coolprop': 'IsoButane',    'GWP100': 3,    'safety': 'A3',  'family': 'HC',   'annex': 'A58/A68'},
    'R601':        {'name': 'n-Pentane',          'coolprop': 'n-Pentane',    'GWP100': 5,    'safety': 'A3',  'family': 'HC',   'annex': 'A58/A68'},
    'R601a':       {'name': 'Isopentane',         'coolprop': 'Isopentane',   'GWP100': 5,    'safety': 'A3',  'family': 'HC',   'annex': 'A58'},
    'RC270':       {'name': 'Cyclopropane',       'coolprop': 'CycloPropane', 'GWP100': 3,    'safety': 'A3',  'family': 'HC',   'annex': '—'},
    # ── Natural refrigerants ─────────────────────────────────────────────────
    'R717':        {'name': 'Ammonia',            'coolprop': 'Ammonia',      'GWP100': 0,    'safety': 'B2L', 'family': 'Natural', 'annex': 'A58/A68'},
    'R744':        {'name': 'Carbon dioxide',     'coolprop': 'CO2',          'GWP100': 1,    'safety': 'A1',  'family': 'Natural', 'annex': 'A58/A68'},
    'R718':        {'name': 'Water (steam cycle)','coolprop': 'Water',        'GWP100': 0,    'safety': 'A1',  'family': 'Natural', 'annex': 'A58'},
    # ── HFOs ─────────────────────────────────────────────────────────────────
    'R1234yf':     {'name': 'HFO-1234yf',         'coolprop': 'R1234yf',      'GWP100': 4,    'safety': 'A2L', 'family': 'HFO',  'annex': '—'},
    'R1234ze(E)':  {'name': 'HFO-1234ze(E)',      'coolprop': 'R1234ze(E)',   'GWP100': 6,    'safety': 'A2L', 'family': 'HFO',  'annex': 'A68'},
    'R1234ze(Z)':  {'name': 'HFO-1234ze(Z)',      'coolprop': 'R1234ze(Z)',   'GWP100': 2,    'safety': 'A2L', 'family': 'HFO',  'annex': 'A58/A68'},
    'R1336mzz(Z)': {'name': 'HFO-1336mzz(Z)',    'coolprop': 'R1336mzz(Z)', 'GWP100': 2,    'safety': 'A1',  'family': 'HFO',  'annex': 'A58/A68'},
    # ── HCFOs ────────────────────────────────────────────────────────────────
    'R1233zd(E)':  {'name': 'HCFO-1233zd(E)',    'coolprop': 'R1233zd(E)',  'GWP100': 1,    'safety': 'A1',  'family': 'HCFO', 'annex': 'A58/A68'},
    'R1224yd(Z)':  {'name': 'HCFO-1224yd(Z)',    'coolprop': 'R1224yd(Z)',  'GWP100': 1,    'safety': 'A1',  'family': 'HCFO', 'annex': 'A58'},
    # ── HFCs (no phase-out ban, limited use) ─────────────────────────────────
    'R245fa':      {'name': 'HFC-245fa',          'coolprop': 'R245fa',       'GWP100': 858,  'safety': 'B1',  'family': 'HFC',  'annex': 'A58'},
    'R152a':       {'name': 'HFC-152a',           'coolprop': 'R152A',        'GWP100': 124,  'safety': 'A2',  'family': 'HFC',  'annex': '—'},
    'R32':         {'name': 'HFC-32',             'coolprop': 'R32',          'GWP100': 675,  'safety': 'A2L', 'family': 'HFC',  'annex': '—'},
}

_FAMILY_COLORS = {
    'HC':      '#2ecc71',   # green
    'Natural': '#3498db',   # blue
    'HFO':     '#f39c12',   # orange
    'HCFO':    '#e67e22',   # dark orange
    'HFC':     '#e74c3c',   # red
}


def _compute_refrig_overview(T_evap_C: float, T_cond_C: float) -> pd.DataFrame:
    """
    Computes critical properties and volumetric heating capacity for all
    refrigerants in the catalogue using CoolProp (ideal isentropic cycle).
    """
    from CoolProp.CoolProp import PropsSI

    rows = []
    for refrig_id, meta in _REFRIG_CATALOGUE.items():
        cp = meta['coolprop']
        row = {
            'ID':         refrig_id,
            'Name':       meta['name'],
            'Family':     meta['family'],
            'Safety':     meta['safety'],
            'GWP₁₀₀ [-]': meta['GWP100'],
            'Annex':      meta.get('annex', '—'),
            '_ok':         False,
        }
        try:
            T_crit_C = PropsSI('Tcrit', cp) - 273.15
            p_crit   = PropsSI('Pcrit', cp) / 1e5
            row['T_crit [°C]']  = round(T_crit_C, 1)
            row['p_crit [bar]'] = round(p_crit, 2)

            # Subcritical requires at least 10 K margin below T_crit
            ok = T_crit_C > T_cond_C + 10
            row['_ok'] = ok
            row['Subcritical'] = '✅' if ok else '❌'

            if ok:
                p_evap = PropsSI('P', 'T', T_evap_C + 273.15, 'Q', 1, cp)
                p_cond = PropsSI('P', 'T', T_cond_C + 273.15, 'Q', 0, cp)
                h_1    = PropsSI('H', 'T', T_evap_C + 273.15, 'Q', 1, cp)
                s_1    = PropsSI('S', 'T', T_evap_C + 273.15, 'Q', 1, cp)
                rho_1  = PropsSI('D', 'T', T_evap_C + 273.15, 'Q', 1, cp)
                h_2s   = PropsSI('H', 'P', p_cond, 'S', s_1, cp)
                h_out  = PropsSI('H', 'T', T_cond_C + 273.15, 'Q', 0, cp)

                dh_cond = (h_2s - h_out) / 1e3          # kJ/kg
                q_v     = dh_cond * rho_1                # kJ/m³  (= dh / v_1)
                PR      = p_cond / p_evap

                row['p_evap [bar]']     = round(p_evap / 1e5, 2)
                row['p_cond [bar]']     = round(p_cond / 1e5, 2)
                row['Δh_cond [kJ/kg]']  = round(dh_cond, 1)
                row['ρ₁ [kg/m³]']       = round(rho_1, 3)
                row['q_v [kJ/m³]']      = round(q_v, 0)
                row['PR [-]']           = round(PR, 2)
            else:
                for k in ('p_evap [bar]', 'p_cond [bar]', 'Δh_cond [kJ/kg]',
                          'ρ₁ [kg/m³]', 'q_v [kJ/m³]', 'PR [-]'):
                    row[k] = None

        except Exception:
            row['T_crit [°C]']  = None
            row['p_crit [bar]'] = None
            row['Subcritical']  = '⚠️'
            for k in ('p_evap [bar]', 'p_cond [bar]', 'Δh_cond [kJ/kg]',
                      'ρ₁ [kg/m³]', 'q_v [kJ/m³]', 'PR [-]'):
                row[k] = None

        rows.append(row)

    col_order = [
        'ID', 'Name', 'Family', 'Safety', 'GWP₁₀₀ [-]', 'Annex',
        'T_crit [°C]', 'p_crit [bar]', 'Subcritical',
        'p_evap [bar]', 'p_cond [bar]', 'PR [-]',
        'Δh_cond [kJ/kg]', 'ρ₁ [kg/m³]', 'q_v [kJ/m³]',
    ]
    df = pd.DataFrame(rows)
    # keep only columns that exist
    col_order = [c for c in col_order if c in df.columns]
    return df[col_order], df['_ok'].tolist()


def _style_refrig_table(df: pd.DataFrame, ok_flags: list) -> object:
    """Returns a styled DataFrame with unsuitable refrigerants greyed out."""
    def _row_style(row):
        idx  = df.index.get_loc(row.name)
        ok   = ok_flags[idx] if idx < len(ok_flags) else True
        base = 'color: #aaa; background-color: #f5f5f5;' if not ok else ''
        return [base] * len(row)

    return df.style.apply(_row_style, axis=1)


def _render_refrigerant_overview(params: dict):
    """Renders the refrigerant overview table and volumetric capacity chart."""
    st.markdown("### 🌿 Refrigerant Overview")
    st.write(
        "All refrigerants are evaluated at the operating temperatures derived from your questionnaire. "
        "Refrigerants that cannot be used in **subcritical** operation at the given condensing temperature "
        "are greyed out (T_crit < T_cond + 10 K)."
    )

    # --- Operating point sliders -------------------------------------------
    T_evap_default = float(_val(params, 'source_temp_out') or _val(params, 'source_temp_in') or 70.0)
    if isinstance(T_evap_default, (int, float)):
        T_evap_default = max(20.0, min(T_evap_default, 160.0))

    if _val(params, 'hw_temp_outlet_required'):
        T_cond_default = float(_val(params, 'hw_temp_outlet_required'))
    elif _val(params, 'steam_pressure_outlet'):
        from CoolProp.CoolProp import PropsSI as _PSI
        try:
            T_cond_default = _PSI('T', 'P', float(_val(params, 'steam_pressure_outlet')) * 1e5, 'Q', 1, 'Water') - 273.15
        except Exception:
            T_cond_default = 120.0
    else:
        T_cond_default = 120.0
    T_cond_default = max(40.0, min(float(T_cond_default), 200.0))

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        T_evap = st.slider("Evaporator temperature T_evap [°C]",
                           min_value=20.0, max_value=160.0, value=T_evap_default,
                           step=1.0, key="refrig_T_evap")
    with col_s2:
        T_cond = st.slider("Condensing temperature T_cond [°C]",
                           min_value=40.0, max_value=200.0, value=T_cond_default,
                           step=1.0, key="refrig_T_cond")

    st.caption(
        f"Volumetric heating capacity **q_v = Δh_cond · ρ₁** [kJ/m³]:  "
        f"higher q_v → smaller compressor volume flow for same heat output.  "
        f"ρ₁ = suction gas density at T_evap; Δh_cond = condenser enthalpy drop (isentropic compression)."
    )

    st.markdown("---")

    # --- Compute table -------------------------------------------------------
    with st.spinner("Computing refrigerant properties via CoolProp…"):
        try:
            df_display, ok_flags = _compute_refrig_overview(T_evap, T_cond)
        except Exception as e:
            st.error(f"❌ CoolProp error: {e}")
            return

    # --- Table ---------------------------------------------------------------
    st.markdown("#### Property Table")
    styled = _style_refrig_table(df_display, ok_flags)
    st.dataframe(styled, use_container_width=True, height=460)

    st.markdown("---")

    # --- Chart: volumetric heating capacity ----------------------------------
    st.markdown("#### Volumetric Heating Capacity  q_v [kJ/m³]")
    st.write(
        "Suitable refrigerants only (subcritical operation possible at selected T_cond). "
        "A higher value means smaller compressor volume flows for the same heat output — "
        "i.e., a more compact and generally less expensive system."
    )

    df_chart = df_display[pd.Series(ok_flags).values].copy()
    df_chart = df_chart[df_chart['q_v [kJ/m³]'].notna()].sort_values('q_v [kJ/m³]', ascending=True)

    if df_chart.empty:
        st.warning("No refrigerants suitable for subcritical operation at the selected temperatures.")
        return

    fig, ax = plt.subplots(figsize=(10, max(3.5, 0.55 * len(df_chart))))

    labels  = df_chart['ID'].tolist()
    values  = df_chart['q_v [kJ/m³]'].tolist()
    families = df_chart['Family'].tolist()
    gwps    = df_chart['GWP₁₀₀ [-]'].tolist()
    colors  = [_FAMILY_COLORS.get(f, '#999') for f in families]

    bars = ax.barh(labels, values, color=colors, edgecolor='white', linewidth=0.6)

    # Annotate bars
    for bar, val, gwp in zip(bars, values, gwps):
        ax.text(val + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:,.0f}  (GWP={gwp})", va='center', ha='left', fontsize=8.5)

    ax.set_xlabel("q_v,heat  [kJ/m³]", fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0, max(values) * 1.22)

    # Legend for families
    from matplotlib.patches import Patch
    seen = {}
    legend_patches = []
    for fam, col in _FAMILY_COLORS.items():
        if fam in families and fam not in seen:
            legend_patches.append(Patch(facecolor=col, label=fam))
            seen[fam] = True
    if legend_patches:
        ax.legend(handles=legend_patches, fontsize=8, loc='lower right')

    plt.tight_layout()
    _pyplot_with_download(fig, 'refrigerant_volumetric_capacity')
    plt.close(fig)

    st.markdown("---")

    # --- Secondary chart: pressure ratio ------------------------------------
    st.markdown("#### Pressure Ratio  PR [-]")
    st.write("Lower PR generally means less demanding compression, lower discharge temperatures, and better compressor efficiency.")

    df_pr = df_chart[df_chart['PR [-]'].notna()].sort_values('PR [-]', ascending=True)
    if not df_pr.empty:
        fig2, ax2 = plt.subplots(figsize=(10, max(3.5, 0.55 * len(df_pr))))
        colors2 = [_FAMILY_COLORS.get(f, '#999') for f in df_pr['Family'].tolist()]
        bars2   = ax2.barh(df_pr['ID'].tolist(), df_pr['PR [-]'].tolist(),
                           color=colors2, edgecolor='white', linewidth=0.6)
        for bar, val in zip(bars2, df_pr['PR [-]'].tolist()):
            ax2.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                     f"{val:.2f}", va='center', ha='left', fontsize=8.5)
        ax2.set_xlabel("Pressure ratio  p_cond / p_evap  [-]", fontsize=10)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        plt.tight_layout()
        _pyplot_with_download(fig2, 'refrigerant_pressure_ratio')
        plt.close(fig2)



# ============================================================================
# HELPER: compressor overview (calls CompressorSelector)
# ============================================================================
def _render_compressor_overview(params: dict):
    """Renders the compressor selector overview – calls CompressorSelector if available."""
    st.markdown("### 🔩 Compressor Overview (DORIN HT Series)")
    st.write(
        "Suitable DORIN HT compressors are looked up for the operating point derived from your "
        "questionnaire. The selector reads the **ExcelDorinExtraction** results folder; "
        "if it is not present, an informational placeholder is shown."
    )

    # --- Derive tc / te / tsh from questionnaire ---------------------------
    try:
        from CoolProp.CoolProp import PropsSI as _PSI
        if _val(params, 'hw_temp_outlet_required'):
            tc_default = float(_val(params, 'hw_temp_outlet_required'))
        elif _val(params, 'steam_pressure_outlet'):
            tc_default = _PSI('T', 'P', float(_val(params, 'steam_pressure_outlet')) * 1e5,
                              'Q', 1, 'Water') - 273.15
        else:
            tc_default = 90.0
        te_default = float(_val(params, 'source_temp_out') or _val(params, 'source_temp_in') or 60.0)
    except Exception:
        tc_default, te_default = 90.0, 60.0

    tc_default = max(40.0, min(float(tc_default), 160.0))
    te_default = max(20.0, min(float(te_default), 140.0))

    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        tc  = st.slider("Condensing temp. tc [°C]",  40.0, 160.0, tc_default,  1.0, key="comp_tc")
    with col_c2:
        te  = st.slider("Evaporating temp. te [°C]", 20.0, 140.0, te_default,  1.0, key="comp_te")
    with col_c3:
        tsh = st.slider("Superheat tsh [K]",          0.0,  30.0,  10.0,        1.0, key="comp_tsh")

    if tc <= te + 5:
        st.warning("⚠️ tc must be at least 5 K above te.")
        return

    Q_required = st.number_input(
        "Required heat output Q_cond [kW] (for unit count estimate)",
        min_value=10.0, max_value=50000.0, value=500.0, step=10.0, key="comp_Q"
    )

    st.markdown("---")

    if st.button("🔍 Search compressors", key="comp_search_btn"):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        results_dir = os.path.join(base_dir, 'ExcelDorinExtraction', 'results', 'compressor_data')
        excel_file  = os.path.join(base_dir, 'ExcelDorinExtraction', 'Compressori_DORIN_HT_rev8.xlsm')

        if not os.path.isdir(results_dir):
            st.info(
                "📂 **ExcelDorinExtraction/results** directory not found. "
                "Place the DORIN extraction results there to enable automatic lookup."
            )
            # example = pd.DataFrame([
            #     {'Compressor': 'HEX-EM 300-76H', 'Refrigerant': 'R600',
            #      'P_ass [kW]': 12.5, 'ṁ [kg/h]': 145, 'η_is [-]': 0.78,
            #      'λ_v [-]': 0.85, 'Q_cond/unit [kW]': '~38', '# Units needed': '~13'},
            #     {'Compressor': 'HEX-EM 400-76H', 'Refrigerant': 'R600a',
            #      'P_ass [kW]': 18.2, 'ṁ [kg/h]': 210, 'η_is [-]': 0.76,
            #      'λ_v [-]': 0.83, 'Q_cond/unit [kW]': '~52', '# Units needed': '~10'},
            # ])
            # st.caption("Example table (static placeholder – real data requires ExcelDorinExtraction folder):")
            # st.dataframe(example, use_container_width=True)
            return

        with st.spinner(f"Searching DORIN compressors (tc={tc}°C, te={te}°C, tsh={tsh}K) …"):
            try:
                #sys.path.insert(0, os.path.dirname(__file__))
                #import sys S
                #sys.path.append(r"C:\Users\josi_\OneDrive - mailbox.tu-dresden.de\Uni\11. Semester Norwegen\Alter Code und Hilfsskripte\ExcelDorinExtraction")
                from ExcelDorinExtraction.compressor_selector import CompressorSelector
                selector = CompressorSelector(results_dir=results_dir, excel_file=excel_file, mode='A')
                raw_results = selector.select_compressor(tc=tc, te=te, tsh=tsh)
                selector.close()
            except Exception as exc:
                st.error(f"❌ CompressorSelector error: {exc}")
                return

        if not raw_results:
            st.warning("No suitable compressors found for this operating point.")
            return

        st.success(f"✅ Found **{len(raw_results)}** compressor / refrigerant combination(s).")

        rows = []
        for r in raw_results:
            eta_is   = r.get('eta_is',   0.75)
            lambda_v = r.get('lambda_v', 0.80)
            P_ass    = r.get('P_ass')
            m_flow   = r.get('m')

            try:
                tc_K       = tc + 273.15
                te_K       = te + 273.15
                COP_carnot = tc_K / (tc_K - te_K)
                COP_real   = eta_is * COP_carnot * 0.6
                Q_unit     = P_ass * (COP_real + 1) if P_ass else None
                n_units    = max(1, int(Q_required / Q_unit + 0.999)) if Q_unit else '—'
                Q_unit_str = f"{Q_unit:.1f}" if Q_unit else '—'
            except Exception:
                Q_unit_str, n_units = '—', '—'

            cp = r.get('compressor_params', {}) or {}
            rows.append({
                'Compressor':        r.get('compressor', '—'),
                'Refrigerant':       r.get('refrigerant', '—'),
                'P_ass [kW]':        f"{P_ass:.2f}" if P_ass else '—',
                'ṁ [kg/h]':          f"{m_flow:.1f}" if m_flow else '—',
                'η_is [-]':          f"{eta_is:.3f}",
                'λ_v [-]':           f"{lambda_v:.3f}",
                'Q_cond/unit [kW]':  Q_unit_str,
                '# Units needed':    n_units,
                'Cylinders':         cp.get('Cylinders', '—'),
                'Swept Vol. [m³/h]': cp.get('SweptVolume50Hz', '—'),
            })

        df_comp = pd.DataFrame(rows)
        st.dataframe(df_comp, use_container_width=True)

        try:
            best_idx = max(range(len(raw_results)), key=lambda i: raw_results[i].get('eta_is', 0))
            best = raw_results[best_idx]
            st.info(
                f"🏆 **Best isentropic efficiency:** {best.get('compressor')} / "
                f"{best.get('refrigerant')} — η_is = {best.get('eta_is', 0):.3f} "
                f"({best.get('eta_is', 0) * 100:.1f} %)"
            )
        except Exception:
            pass


# ============================================================================
# HELPER: MVR state diagrams – ruft mvr.generate_state_diagram() korrekt auf
# ============================================================================
def _mvr_calc_limits(mvr, prop: str, padding_rel: float, scale: str = 'linear'):
    """
    Berechnet Achsenlimits fuer MVR-Diagramme aus den tatsaechlich geloesten
    TESPy-Netzwerkergebnissen (analog zu hp_dashboard.py / StateDiagramGenerator).
    MVR-Netzwerk verwendet T[°C], p[bar], h[kJ/kg] als Einheiten.
    """
    import numpy as np
    wf = mvr.wf  # z.B. 'Water'
    conn = mvr.nw.results['Connection']
    mask = conn[wf] == 1.0
    min_val = conn.loc[mask, prop].min()
    max_val = conn.loc[mask, prop].max()
    if scale == 'linear':
        delta = max_val - min_val
        return min_val - padding_rel * delta, max_val + padding_rel * delta
    else:  # log
        import math
        log_min = math.log10(min_val)
        log_max = math.log10(max_val)
        delta_log = log_max - log_min
        return 10 ** (log_min - padding_rel * delta_log), 10 ** (log_max + padding_rel * delta_log)


def _draw_feedwater_path_on_mvr_diagram(diagram, diagram_type: str, feedwater: dict):
    """
    Draws the HTHP-side heating and evaporation path
    (feed water inlet → compressor stage 1 inlet) onto an existing
    FluidPropertyDiagram (Water / MVR circuit).

    Segment 1: Subcooled heating  T_fw_C → T_sat  (isobaric, if T_fw < T_sat)
    Segment 2: Evaporation at T_sat               (isothermal/isobaric, Q: 0 → 1)
    Segment 3: Superheating T_sat → T_end_C        (isobaric, if T_end > T_sat)

    Unit notes
    ----------
    log(p)-h : axes are kJ/kg (h) and bar (p)  → divide CoolProp output by 1000 / 1e5
    T-s      : axes are °C (T) and J/(kg·K) (s) → do NOT divide CoolProp entropy by 1000

    Parameters
    ----------
    diagram      : FluidPropertyDiagram object (.ax or .fig)
    diagram_type : 'logph' or 'Ts'
    feedwater    : dict with 'T_fw_C', 'p_bar', 'T_end_C'
    """
    from CoolProp.CoolProp import PropsSI as _PSI
    import numpy as _np

    T_fw_C  = float(feedwater['T_fw_C'])
    p_bar   = float(feedwater['p_bar'])
    T_end_C = float(feedwater['T_end_C'])
    p_Pa    = p_bar * 1e5

    T_sat_K = _PSI('T', 'P', p_Pa, 'Q', 0, 'Water')
    T_sat_C = T_sat_K - 273.15

    ax = getattr(diagram, 'ax', None) or diagram.fig.axes[0]

    COLOUR = '#ea6a0e'   # heatpumps/fluprodia standard orange (MVRBase default)
    LW     = 1.6
    N      = 40

    if diagram_type == 'logph':
        # Axes: kJ/kg (h), bar (p)
        h_fw      = _PSI('H', 'T', T_fw_C  + 273.15, 'P', p_Pa, 'Water') / 1000
        h_end     = _PSI('H', 'T', T_end_C + 273.15, 'P', p_Pa, 'Water') / 1000
        h_sat_liq = _PSI('H', 'P', p_Pa, 'Q', 0, 'Water') / 1000
        h_sat_vap = _PSI('H', 'P', p_Pa, 'Q', 1, 'Water') / 1000

        h_path = sorted({h_fw, h_sat_liq, h_sat_vap, h_end})
        p_path = [p_bar] * len(h_path)

        ax.plot(h_path, p_path,
                color=COLOUR, lw=LW, ls='--', zorder=4,
                label='HTHP – Feed Water Heating / Evaporation')
        ax.scatter([h_fw], [p_bar], color=COLOUR, s=60, zorder=6,
                   marker='o', edgecolors='white', linewidths=0.5)
        ax.annotate(
            f'Feed water\ninlet ({T_fw_C:.0f} °C)',
            xy=(h_fw, p_bar),
            xytext=(6, 8), textcoords='offset points',
            fontsize=7, color=COLOUR, zorder=7,
        )

    elif diagram_type == 'Ts':
        # Axes: °C (T), J/(kg·K) (s — SI, do NOT divide by 1000)
        s_pts, T_pts = [], []

        # Segment 1: subcooled heating T_fw_C → T_sat_C (isobaric)
        if T_fw_C < T_sat_C - 0.1:
            for T_K in _np.linspace(T_fw_C + 273.15, T_sat_C + 273.15 - 0.05, N):
                try:
                    s_pts.append(_PSI('S', 'T', T_K, 'P', p_Pa, 'Water'))
                    T_pts.append(T_K - 273.15)
                except Exception:
                    pass

        # Segment 2: evaporation Q=0 → Q=1 (isothermal/isobaric)
        h_sl = _PSI('H', 'P', p_Pa, 'Q', 0, 'Water')
        h_sv = _PSI('H', 'P', p_Pa, 'Q', 1, 'Water')
        for q in _np.linspace(0.0, 1.0, N * 2):
            h_q = h_sl + q * (h_sv - h_sl)
            try:
                s_pts.append(_PSI('S', 'P', p_Pa, 'H', h_q, 'Water'))
                T_pts.append(T_sat_C)
            except Exception:
                pass

        # Segment 3: superheating T_sat_C → T_end_C (isobaric)
        # Threshold kept small (0.05 K) so even MIN_SUPERHEAT = 1 K is captured
        if T_end_C > T_sat_C + 0.05:
            for T_K in _np.linspace(T_sat_C + 273.15 + 0.05, T_end_C + 273.15, N):
                try:
                    s_pts.append(_PSI('S', 'T', T_K, 'P', p_Pa, 'Water'))
                    T_pts.append(T_K - 273.15)
                except Exception:
                    pass

        if s_pts:
            ax.plot(s_pts, T_pts,
                    color=COLOUR, lw=LW, ls='--', zorder=4,
                    label='HTHP – Feed Water Heating / Evaporation')
            ax.scatter([s_pts[0]], [T_pts[0]], color=COLOUR, s=60, zorder=6,
                       marker='o', edgecolors='white', linewidths=0.5)
            ax.annotate(
                f'Feed water\ninlet ({T_fw_C:.0f} °C)',
                xy=(s_pts[0], T_pts[0]),
                xytext=(6, 8), textcoords='offset points',
                fontsize=7, color=COLOUR, zorder=7,
            )

    # Redraw legend
    try:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, fontsize=7, loc='upper left')
    except Exception:
        pass


def _generate_mvr_diagrams(mvr, case_id: str, results_dir: str = 'results/plots',
                            feedwater: dict = None) -> dict:
    """
    Erzeugt log(p)-h und T-s Diagramm fuer ein MVR-Objekt (MVRMultiStage / MVRBase)
    durch direkten Aufruf von mvr.generate_state_diagram().
    Gibt dict {'logph': pfad, 'Ts': pfad} zurueck.

    Parameters
    ----------
    feedwater : dict, optional
        Wird als Kwarg an get_plotting_states weitergegeben, um den HTHP-
        Kondensator-Pfad (Speisewasser → Kompressor-1-Eingang) einzuzeichnen.
        Erwartet: {'T_fw_C': float, 'p_bar': float, 'T_end_C': float}
    """
    import matplotlib
    matplotlib.use('Agg')
    os.makedirs(results_dir, exist_ok=True)
    paths = {}

    for diagram_type in ('logph', 'Ts'):
        try:
            # Limits from actual network results – increased padding for breathing room
            if diagram_type == 'logph':
                xlims = _mvr_calc_limits(mvr, 'h', padding_rel=0.50, scale='linear')
                ylims = _mvr_calc_limits(mvr, 'p', padding_rel=0.35, scale='log')
            else:  # Ts
                xlims = _mvr_calc_limits(mvr, 's', padding_rel=0.50, scale='linear')
                ylims = _mvr_calc_limits(mvr, 'T', padding_rel=0.35, scale='linear')

            # Extend axis limits to include feedwater inlet state and comp-1 inlet
            _fw_valid = (
                feedwater is not None
                and feedwater.get('T_fw_C') is not None
                and feedwater.get('p_bar')   is not None
                and feedwater.get('T_end_C') is not None
            )
            if _fw_valid:
                from CoolProp.CoolProp import PropsSI as _PSI2
                try:
                    T_fw_K  = float(feedwater['T_fw_C'])  + 273.15
                    T_end_K = float(feedwater['T_end_C']) + 273.15
                    p_Pa    = float(feedwater['p_bar']) * 1e5
                    # Clamp T_fw away from exact saturation
                    T_sat_fw = _PSI2('T', 'P', p_Pa, 'Q', 1, 'Water') - 273.15
                    if abs(float(feedwater['T_fw_C']) - T_sat_fw) < 0.1:
                        T_fw_K = (T_sat_fw - 0.1) + 273.15
                    if diagram_type == 'logph':
                        h_fw  = _PSI2('H', 'T', T_fw_K,  'P', p_Pa, 'Water') / 1000
                        h_end = _PSI2('H', 'T', T_end_K, 'P', p_Pa, 'Water') / 1000
                        margin = (xlims[1] - xlims[0]) * 0.15
                        xlims = (min(xlims[0], h_fw - margin),
                                 max(xlims[1], h_end + margin))
                    else:  # Ts – units are J/(kg·K), do NOT divide by 1000
                        s_fw  = _PSI2('S', 'T', T_fw_K,  'P', p_Pa, 'Water')
                        s_end = _PSI2('S', 'T', T_end_K, 'P', p_Pa, 'Water')
                        margin = (xlims[1] - xlims[0]) * 0.15
                        xlims = (min(xlims[0], s_fw - margin),
                                 max(xlims[1], s_end + margin))
                        T_fw_C_val = float(feedwater['T_fw_C'])
                        ylims = (min(ylims[0], T_fw_C_val - 10), ylims[1])
                except Exception:
                    pass

            diagram = mvr.generate_state_diagram(
                diagram_type=diagram_type,
                savefig=False,
                return_diagram=True,
                open_file=False,
                xlims=xlims,
                ylims=ylims,
            )

            if diagram is None or not hasattr(diagram, 'fig'):
                continue

            # Post-processing: draw HTHP feed water heating path
            if _fw_valid:
                try:
                    _draw_feedwater_path_on_mvr_diagram(diagram, diagram_type, feedwater)
                except Exception as _e_fw:
                    print(f'  Feed water path overlay failed ({diagram_type}): {_e_fw}')

            filepath_pdf = os.path.join(results_dir, f'{case_id}_mvr_{diagram_type}.pdf')
            filepath_png = os.path.join(results_dir, f'{case_id}_mvr_{diagram_type}.png')
            diagram.fig.savefig(filepath_pdf, dpi=150, bbox_inches='tight')
            diagram.fig.savefig(filepath_png, dpi=150, bbox_inches='tight')
            import matplotlib.pyplot as _plt
            _plt.close(diagram.fig)
            # Verwende PNG für die Anzeige im Streamlit-Interface
            paths[diagram_type] = filepath_png

        except Exception as e:
            print(f'  MVR {diagram_type} diagram failed: {e}')

    return paths



# ============================================================================
# MANUAL ENTRY READER  – mimics QuestionnaireReader for manually entered data
# ============================================================================
class ManualEntryReader:
    """
    Wraps a manually built parameter dict so the rest of the tool works
    identically regardless of whether parameters came from a file or were
    entered manually.
    """

    def __init__(self, params: dict):
        self._params = params

    def get_params(self) -> dict:
        return self._params

    def get_value(self, key: str):
        entry = self._params.get(key, {})
        if isinstance(entry, dict):
            return entry.get('value')
        return entry

    def get_application_case(self) -> str:
        app = self.get_value('application_type')
        if app is None:
            return 'unknown'
        app_l = str(app).lower().strip()
        if any(kw in app_l for kw in ('hot water', 'hotwater', 'hw')):
            return 'hot_water'
        if any(kw in app_l for kw in ('steam', 'dampf')):
            return 'steam'
        return 'unknown'


# ============================================================================
# PAGE 1: UPLOAD & QUESTIONNAIRE
# ============================================================================
if st.session_state.page == 'upload':
    st.markdown("""
## Welcome to the HTHP Modeling Suite

This tool enables the **rapid pre-selection and comparison of industrial heat supply system architectures**.
It covers closed vapour compression cycles, Mechanical Vapour Recompression,
and combined configurations, allowing multiple system designs to be evaluated
side by side under identical boundary conditions.
 
Process parameters can be loaded from a standardised questionnaire file or entered manually.
Results include COP, exergetic efficiency, heat and power balances, state point diagrams,
and a comparative overview across all configured cases.


---
""")

    st.markdown("## Step 1 – Load Process Parameters")

    _tab_file, _tab_manual = st.tabs(
        ["📁 Upload Questionnaire File", "✏️ Enter Parameters Manually"]
    )

    with _tab_file:
        st.info(
            "Provide your **questionnaire file** (.xlsx) containing system specifications "
            "and operating conditions for your simulations."
        )
        uploaded_file = st.file_uploader(
            "Upload questionnaire file", type=["xlsx"],
            help="Select your configuration file"
        )
        if uploaded_file is not None:
            st.success(f"✅ '{uploaded_file.name}' uploaded successfully!")
            col1, col2, col3 = st.columns(3)
            col1.metric("File name", uploaded_file.name)
            col2.metric("File size", f"{uploaded_file.size / 1024:.2f} KB")
            col3.metric("File type", uploaded_file.type)
            try:
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    tmp_path = tmp.name
                reader             = QuestionnaireReader(tmp_path)
                questionnaire_data = reader.get_params()
                st.session_state.questionnaire_reader = reader
                st.session_state.questionnaire_data   = questionnaire_data
            except Exception as e:
                st.error(f"❌ Error reading questionnaire: {e}")

    with _tab_manual:
        st.info(
            "Fill in all process parameters using the interactive form. "
            "Fields marked **★** are required for thermodynamic calculations. "
            "Parameters can be saved to JSON and reloaded at any time."
        )
        if st.button("✏️ Open Manual Entry Form →", use_container_width=True,
                     key="btn_open_manual"):
            st.session_state.page = 'manual_entry'
            st.rerun()
        if st.session_state.questionnaire_data is not None and \
                isinstance(st.session_state.get('questionnaire_reader'), ManualEntryReader):
            st.success("✅ Manual parameters already loaded — see summary below.")

    # ── Parameter summary + navigation (shown once data is loaded either way) ──
    if st.session_state.questionnaire_data is not None:
        questionnaire_data = st.session_state.questionnaire_data
        reader             = st.session_state.questionnaire_reader
        case               = reader.get_application_case() if reader else 'unknown'

        def _get(d, key):
            v    = d.get(key, {})
            val  = v.get('value', 'N/A') if isinstance(v, dict) else v
            unit = v.get('unit',  '')    if isinstance(v, dict) else ''
            return val, unit

        st.markdown("---")
        st.markdown("## ✅ Parameters Loaded")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Source Parameters")
            v, u = _get(questionnaire_data, 'source_temp_in');   st.write(f"**Source inlet temp:** {v} {u}")
            v, u = _get(questionnaire_data, 'source_temp_out');  st.write(f"**Source outlet temp:** {v} {u}")
            v, u = _get(questionnaire_data, 'source_pressure');  st.write(f"**Source pressure:** {v} {u}")

        with col2:
            st.subheader("Application Case")
            if case == 'hot_water':
                st.write("**Type:** Hot Water Production")
                v, u = _get(questionnaire_data, 'hw_temp_inlet');           st.write(f"**Inlet temp:** {v} {u}")
                v, u = _get(questionnaire_data, 'hw_temp_outlet_required'); st.write(f"**Outlet temp:** {v} {u}")
                v, u = _get(questionnaire_data, 'hw_heat_capacity');        st.write(f"**Heat power:** {v} {u}")
            elif case == 'steam':
                st.write("**Type:** Steam Generation")
                v, u = _get(questionnaire_data, 'steam_temp_inlet');      st.write(f"**Inlet temp:** {v} {u}")
                v, u = _get(questionnaire_data, 'steam_mass_flow_inlet'); st.write(f"**Mass flow:** {v} {u}")
                v, u = _get(questionnaire_data, 'steam_pressure_outlet'); st.write(f"**Outlet pressure:** {v} {u}")
                v, u = _get(questionnaire_data, 'steam_superheat');       st.write(f"**Superheat:** {v} {u}")
            else:
                st.write("**Type:** Unknown — verify the *Application Type* field.")

        st.markdown("---")
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("📊 Go to Pinch Analysis", use_container_width=True,
                         key="btn_pinch_upload"):
                st.session_state.page = 'pinch_analysis'
                st.rerun()
        with col_btn2:
            if st.button("📋 Go to Case Generation", use_container_width=True,
                         key="btn_cases_upload"):
                st.session_state.page = 'case_generation'
                st.rerun()
    else:
        st.markdown("---")
        st.warning("⚠️ Please upload a questionnaire file or enter parameters manually to proceed.")


# ============================================================================
# PAGE 1b: MANUAL PARAMETER ENTRY
# ============================================================================
elif st.session_state.page == 'manual_entry':
    import json as _json_me
    from utils.questionnaire_reader import ROW_MAPPING as _ME_RM

    # ── Field metadata ────────────────────────────────────────────────────────
    _ME_UNITS = {
        'source_heat_capacity':      'kW',     'source_temp_in':           '°C',
        'source_temp_out':           '°C',     'source_pressure':          'bar',
        'hw_temp_inlet':             '°C',     'hw_temp_outlet_required':  '°C',
        'hw_temp_outlet_min':        '°C',     'hw_mass_flow':             'kg/h',
        'hw_heat_capacity':          'kW',     'hw_operating_hours':       'h/day',
        'steam_temp_inlet':          '°C',     'steam_pressure_inlet':     'bar',
        'steam_mass_flow_inlet':     'kg/h',   'steam_pressure_outlet':    'bar',
        'steam_temp_saturation':     '°C',     'steam_superheat':          'K',
        'steam_operating_hours':     'h/day',
        'add_hw_temp_inlet':         '°C',     'add_hw_temp_outlet_required': '°C',
        'add_hw_temp_outlet_min':    '°C',     'add_hw_heat_capacity':     'kW',
        'add_hw_operating_hours':    'h/day',
        'electrical_power_max':      'kW',     'cooling_water_temp':       '°C',
        'cooling_water_flow':        'm³/h',   'water_hardness':           '°dH',
        'waste_heat_1_temp_supply':  '°C',     'waste_heat_1_temp_outlet': '°C',
        'waste_heat_1_pressure':     'bar',    'waste_heat_1_mass_flow':   'kg/h',
        'waste_heat_2_temp_supply':  '°C',     'waste_heat_2_temp_outlet': '°C',
        'waste_heat_2_pressure':     'bar',    'waste_heat_2_mass_flow':   'kg/h',
        'waste_heat_3_temp_supply':  '°C',     'waste_heat_3_temp_outlet': '°C',
        'waste_heat_3_pressure':     'bar',    'waste_heat_3_mass_flow':   'kg/h',
        'modulation_range':          '%',      'storage_volume':           'm³',
        'room_temp_winter':          '°C',     'room_temp_summer':         '°C',
        'air_humidity':              '%',      'altitude':                 'm',
        'safety_valve_pressure':     'bar',    'sound_db_distance':        'dB(A)',
        'electricity_price':         '€/kWh', 'gas_price':                '€/MWh',
        'investment_budget':         '€',      'payback_period':           'years',
        'annual_operating_hours':    'h/a',
    }

    # Fields that are required for thermodynamic calculations
    _ME_REQUIRED = {
        'source_temp_in', 'source_temp_out', 'source_heat_capacity',
        'application_type',
        'hw_temp_inlet', 'hw_temp_outlet_required', 'hw_heat_capacity',
        'steam_temp_inlet', 'steam_mass_flow_inlet', 'steam_pressure_outlet',
        'electricity_price', 'electrical_power_max',
    }

    _ME_NUMERIC = {
        'source_heat_capacity', 'source_temp_in', 'source_temp_out', 'source_pressure',
        'hw_temp_inlet', 'hw_temp_outlet_required', 'hw_temp_outlet_min',
        'hw_mass_flow', 'hw_heat_capacity', 'hw_operating_hours',
        'steam_temp_inlet', 'steam_pressure_inlet', 'steam_mass_flow_inlet',
        'steam_pressure_outlet', 'steam_temp_saturation', 'steam_superheat',
        'steam_operating_hours',
        'add_hw_temp_inlet', 'add_hw_temp_outlet_required', 'add_hw_temp_outlet_min',
        'add_hw_heat_capacity', 'add_hw_operating_hours',
        'electrical_power_max', 'cooling_water_temp', 'cooling_water_flow',
        'water_hardness', 'waste_heat_count',
        'waste_heat_1_temp_supply', 'waste_heat_1_temp_outlet',
        'waste_heat_1_pressure',    'waste_heat_1_mass_flow',
        'waste_heat_2_temp_supply', 'waste_heat_2_temp_outlet',
        'waste_heat_2_pressure',    'waste_heat_2_mass_flow',
        'waste_heat_3_temp_supply', 'waste_heat_3_temp_outlet',
        'waste_heat_3_pressure',    'waste_heat_3_mass_flow',
        'modulation_range', 'storage_volume',
        'room_temp_winter', 'room_temp_summer', 'air_humidity', 'altitude',
        'safety_valve_pressure', 'sound_db_distance',
        'electricity_price', 'gas_price', 'investment_budget',
        'payback_period', 'annual_operating_hours',
    }

    _ME_DROPDOWNS = {
        'application_type':         ['', 'Steam', 'Hot Water'],
        'source_type':              ['', 'Process waste heat', 'Groundwater',
                                     'Exhaust gas', 'Geothermal', 'Other'],
        'source_year_round':        ['', 'Yes', 'No'],
        'steam_quality':            ['', 'Saturated', 'Superheated'],
        'steam_flow_config':        ['', 'Open loop', 'Closed loop'],
        'hw_operating_mode':        ['', 'Continuous', 'Intermittent', 'Variable'],
        'steam_operating_mode':     ['', 'Continuous', 'Intermittent', 'Variable'],
        'add_hw_operating_mode':    ['', 'Continuous', 'Intermittent', 'Variable'],
        'electrical_power_supply':  ['', 'Yes', 'No'],
        'cooling_water_available':  ['', 'Yes', 'No'],
        'modulation_required':      ['', 'Yes', 'No'],
        'storage_available':        ['', 'Yes', 'No'],
        'ped_required':             ['', 'Yes', 'No'],
        'ped_category':             ['', 'I', 'II', 'III', 'IV'],
        'atex_required':            ['', 'Yes', 'No'],
        'safety_valves_required':   ['', 'Yes', 'No'],
        'waste_heat_1_variability': ['', 'Constant', 'Variable'],
        'waste_heat_2_variability': ['', 'Constant', 'Variable'],
        'waste_heat_3_variability': ['', 'Constant', 'Variable'],
    }

    # ── Initialise session-state keys for all fields ──────────────────────────
    for _fk in _ME_RM:
        if f'me_{_fk}' not in st.session_state:
            st.session_state[f'me_{_fk}'] = None

    # ── Page header ───────────────────────────────────────────────────────────
    st.markdown("## ✏️ Manual Parameter Entry")
    st.markdown(
        "<p style='color:gray;font-size:0.9em;margin-top:-0.5rem;'>"
        "Fields marked <b>★</b> are required for thermodynamic calculations. "
        "All other fields are optional."
        "</p>",
        unsafe_allow_html=True,
    )

    # ── Top action bar: Back | Load ───────────────────────────────────────────
    _hcol_back, _hcol_load = st.columns([2, 5])
    with _hcol_back:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("← Back to Upload", use_container_width=True, key="me_back_btn"):
            st.session_state.page = 'upload'
            st.rerun()
    with _hcol_load:
        _lf = st.file_uploader(
            "📂 Load saved parameters (.json)",
            type=["json"],
            key="me_load_file",
        )
        if _lf is not None:
            try:
                _loaded = _json_me.loads(_lf.read().decode('utf-8'))
                for _lk, _lv in _loaded.items():
                    st.session_state[f'me_{_lk}'] = _lv
                st.success(f"✅ {len(_loaded)} parameter(s) loaded from file.")
                st.rerun()
            except Exception as _le:
                st.error(f"❌ Error loading file: {_le}")

    st.markdown("---")

    # ── Single field renderer ─────────────────────────────────────────────────
    def _me_field(fkey: str, desc: str, ctx=None):
        """Render one labelled input widget.  ctx = st or a column object."""
        _c   = ctx if ctx is not None else st
        unit = _ME_UNITS.get(fkey, '')
        req  = fkey in _ME_REQUIRED

        # Bold label for required fields
        _unit_part = f'  [{unit}]' if unit else ''
        if req:
            label = f"**{desc} ★**{_unit_part}"
        else:
            label = f"{desc}{_unit_part}"

        ss_key  = f'me_{fkey}'
        current = st.session_state.get(ss_key)

        if fkey in _ME_DROPDOWNS:
            opts    = _ME_DROPDOWNS[fkey]
            cur_str = str(current) if current is not None else ''
            idx     = opts.index(cur_str) if cur_str in opts else 0
            val     = _c.selectbox(label, opts, index=idx, key=ss_key)
            return val if val else None

        elif fkey in _ME_NUMERIC:
            try:
                def_val = float(current) if (current is not None and current != '') else 0.0
            except (ValueError, TypeError):
                def_val = 0.0
            val = _c.number_input(label, value=def_val, format='%.6g', key=ss_key)
            return val

        else:
            def_str = str(current) if current is not None else ''
            val = _c.text_input(label, value=def_str, key=ss_key)
            return val.strip() if val else ''

    # ── SECTION 1 · Project Data ──────────────────────────────────────────────
    with st.expander("📋 1. Project Data", expanded=False):
        _c1, _c2 = st.columns(2)
        _me_field('company_name',         'Company / Name',                _c1)
        _me_field('contact_person',       'Contact Person',                _c2)
        _me_field('contact_phone_email',  'Phone / Email',                 _c1)
        _me_field('commissioning_date',   'Planned Commissioning Date',    _c2)
        _me_field('location_country',     'Plant Location (Country)',       _c1)
        _me_field('location_city_street', 'Plant Location (City, Street)', _c2)

    # ── SECTION 2 · Heat Source ───────────────────────────────────────────────
    with st.expander("🟦 2. Heat Source", expanded=True):
        _c1, _c2 = st.columns(2)
        _me_field('source_type',            'Heat Source Type',                    _c1)
        _me_field('source_medium',          'Heat Transfer Medium',                _c2)
        _me_field('source_temp_in',         'Source Inlet Temperature',            _c1)
        _me_field('source_temp_out',        'Source Outlet Temperature (min.)',    _c2)
        _me_field('source_heat_capacity',   'Available Heat Capacity',             _c1)
        _me_field('source_pressure',        'Source Side Pressure',                _c2)
        _me_field('source_year_round',      'Year-Round Availability?',            _c1)
        _me_field('source_seasonal_period', 'Seasonal Period (From – To)',          _c2)

    # ── SECTION 3 · Heat Consumer ─────────────────────────────────────────────
    with st.expander("🟥 3. Heat Consumer", expanded=True):
        _me_field('application_type', 'Application Type')

        st.markdown("##### 3a. Hot Water")
        _c1, _c2 = st.columns(2)
        _me_field('hw_temp_inlet',           'Feed Water Temperature',              _c1)
        _me_field('hw_temp_outlet_required', 'Required Outlet Temperature',         _c2)
        _me_field('hw_temp_outlet_min',      'Minimum Outlet Temperature',          _c1)
        _me_field('hw_mass_flow',            'Mass Flow',                           _c2)
        _me_field('hw_heat_capacity',        'Required Heat Capacity',              _c1)
        _me_field('hw_operating_mode',       'Operating Mode',                      _c2)
        _me_field('hw_operating_hours',      'Daily Operating Hours',               _c1)

        st.markdown("##### 3b. Steam")
        _c1, _c2 = st.columns(2)
        _me_field('steam_flow_config',       'System Configuration',                _c1)
        _me_field('steam_quality',           'Output Steam Quality',                _c2)
        _me_field('steam_temp_inlet',        'Feed Water / Steam Temperature',      _c1)
        _me_field('steam_pressure_inlet',    'Feed Water / Steam Pressure',         _c2)
        _me_field('steam_mass_flow_inlet',   'Feed Water / Supply Mass Flow',       _c1)
        _me_field('steam_pressure_outlet',   'Steam Target Pressure',               _c2)
        _me_field('steam_temp_saturation',   'Saturation Temp. at Steam Pressure',  _c1)
        _me_field('steam_superheat',         'Required Superheat',                  _c2)
        _me_field('steam_operating_mode',    'Steam Operating Mode',                _c1)
        _me_field('steam_operating_hours',   'Daily Operating Hours',               _c2)

        st.markdown("##### 3c. Additional Heat Consumer (optional)")
        _c1, _c2 = st.columns(2)
        _me_field('add_hw_temp_inlet',           'Additional: Feed Water Temperature',      _c1)
        _me_field('add_hw_temp_outlet_required', 'Additional: Required Outlet Temperature', _c2)
        _me_field('add_hw_temp_outlet_min',      'Additional: Min. Outlet Temperature',     _c1)
        _me_field('add_hw_heat_capacity',        'Additional: Required Heat Capacity',      _c2)
        _me_field('add_hw_operating_mode',       'Additional: Operating Mode',              _c1)
        _me_field('add_hw_operating_hours',      'Additional: Daily Operating Hours',       _c2)

    # ── SECTION 4 · Infrastructure ────────────────────────────────────────────
    with st.expander("⚙️ 4. Infrastructure", expanded=False):
        _c1, _c2 = st.columns(2)
        _me_field('electrical_power_supply', 'Electrical Power Supply Available', _c1)
        _me_field('electrical_power_max',    'Available Power',                   _c2)
        _me_field('cooling_water_available', 'Cooling Water Available',           _c1)
        _me_field('cooling_water_temp',      'Cooling Water Temperature',         _c2)
        _me_field('cooling_water_flow',      'Cooling Water Volume Flow',         _c1)
        _me_field('water_hardness',          'Water Hardness',                    _c2)

    # ── SECTION 5 · Optimization & Waste Heat ────────────────────────────────
    with st.expander("⭐ 5. Optimization & Waste Heat (optional)", expanded=False):
        _me_field('waste_heat_count', 'Number of Additional Waste Heat Flows')
        for _whn in (1, 2, 3):
            st.markdown(f"**Waste Heat Flow {_whn}**")
            _c1, _c2 = st.columns(2)
            _me_field(f'waste_heat_{_whn}_variability', 'Variability',        _c1)
            _me_field(f'waste_heat_{_whn}_medium',      'Medium',             _c2)
            _me_field(f'waste_heat_{_whn}_temp_supply', 'Supply Temperature', _c1)
            _me_field(f'waste_heat_{_whn}_temp_outlet', 'Outlet Temperature', _c2)
            _me_field(f'waste_heat_{_whn}_pressure',    'Supply Pressure',    _c1)
            _me_field(f'waste_heat_{_whn}_mass_flow',   'Mass Flow',          _c2)
        st.markdown("**Storage & Modulation**")
        _c1, _c2 = st.columns(2)
        _me_field('modulation_required', 'Modulation Required?', _c1)
        _me_field('modulation_range',    'Modulation Range',     _c2)
        _me_field('storage_available',   'Storage Available?',   _c1)
        _me_field('storage_volume',      'Storage Volume',       _c2)

    # ── SECTION 6 · Location & Environment ───────────────────────────────────
    with st.expander("🌍 6. Location & Environment (optional)", expanded=False):
        _c1, _c2 = st.columns(2)
        _me_field('installation_space', 'Available Installation Space',  _c1)
        _me_field('altitude',           'Altitude',                      _c2)
        _me_field('room_temp_winter',   'Room Temperature Winter',       _c1)
        _me_field('room_temp_summer',   'Room Temperature Summer',       _c2)
        _me_field('air_humidity',       'Air Humidity',                  _c1)
        _me_field('noise_sensitivity',  'Vibration / Noise Sensitivity', _c2)

    # ── SECTION 7 · Safety & Permissions ─────────────────────────────────────
    with st.expander("🔒 7. Safety & Permissions (optional)", expanded=False):
        _c1, _c2 = st.columns(2)
        _me_field('ped_required',           'Pressure Equipment Directive (PED)?', _c1)
        _me_field('ped_category',           'PED Category',                        _c2)
        _me_field('atex_required',          'ATEX Requirements',                   _c1)
        _me_field('atex_zone',              'ATEX Zone',                           _c2)
        _me_field('safety_valves_required', 'Safety Valves Required?',             _c1)
        _me_field('safety_valve_pressure',  'Max. Pressure (Safety Valve)',        _c2)
        _me_field('sound_protection',       'Sound Protection Requirement',        _c1)
        _me_field('sound_db_distance',      'dB(A) at Distance',                  _c2)
        _me_field('other_permits',          'Other Permits / Standards')

    # ── SECTION 8 · Economic Parameters ──────────────────────────────────────
    with st.expander("💰 8. Economic Parameters", expanded=False):
        _c1, _c2 = st.columns(2)
        _me_field('electricity_price',      'Electricity Price',               _c1)
        _me_field('gas_price',              'Gas Price',                      _c2)
        _me_field('investment_budget',      'Available Investment Budget',    _c1)
        _me_field('payback_period',         'Required Payback Period',        _c2)
        _me_field('annual_operating_hours', 'Planned Annual Operating Hours', _c1)

    st.markdown("---")

    # ── Collect all field values into the standard params dict ────────────────
    def _me_collect() -> dict:
        _out = {}
        for _fk, (_, _desc, _sec) in _ME_RM.items():
            _v = st.session_state.get(f'me_{_fk}')
            if isinstance(_v, str) and _v == '':
                _v = None
            _out[_fk] = {
                'description': _desc,
                'value':       _v,
                'unit':        _ME_UNITS.get(_fk),
                'section':     _sec,
            }
        return _out

    # ── Bottom action bar: Save | Process Parameters ──────────────────────────
    _bcol_save, _bcol_proc, _ = st.columns([2, 2.5, 5.5])

    with _bcol_save:
        _save_dict = {
            _fk: st.session_state.get(f'me_{_fk}')
            for _fk in _ME_RM
            if st.session_state.get(f'me_{_fk}') is not None
            and st.session_state.get(f'me_{_fk}') != ''
        }
        st.download_button(
            label="💾 Save Parameters",
            data=_json_me.dumps(_save_dict, indent=2, ensure_ascii=False),
            file_name="hthp_parameters.json",
            mime="application/json",
            use_container_width=True,
            key="me_save_btn",
        )

    with _bcol_proc:
        if st.button("⚙️ Process Parameters", type="primary",
                     use_container_width=True, key="me_process_btn"):
            _params = _me_collect()

            # Validate base required fields
            _missing = []
            for _rk in ('source_temp_in', 'source_temp_out',
                         'source_heat_capacity', 'application_type'):
                _v = _params.get(_rk, {}).get('value')
                if _v is None or _v == '' or (isinstance(_v, float) and _v == 0.0
                                               and _rk != 'steam_superheat'):
                    _missing.append(_params[_rk]['description'])

            # Validate application-specific fields
            _app_val = _params.get('application_type', {}).get('value')
            _app_str = str(_app_val).lower() if _app_val else ''
            if 'steam' in _app_str:
                for _rk in ('steam_temp_inlet', 'steam_mass_flow_inlet',
                             'steam_pressure_outlet'):
                    _v = _params.get(_rk, {}).get('value')
                    if not _v or (isinstance(_v, float) and _v == 0.0):
                        _missing.append(_params[_rk]['description'])
            elif 'hot water' in _app_str:
                for _rk in ('hw_temp_outlet_required', 'hw_heat_capacity'):
                    _v = _params.get(_rk, {}).get('value')
                    if not _v or (isinstance(_v, float) and _v == 0.0):
                        _missing.append(_params[_rk]['description'])

            if _missing:
                st.error(
                    f"❌ Missing required fields: **{', '.join(_missing)}**. "
                    "Please fill in all ★ fields before proceeding."
                )
            else:
                _me_reader = ManualEntryReader(_params)
                st.session_state.questionnaire_data   = _params
                st.session_state.questionnaire_reader = _me_reader
                st.success("✅ Parameters processed successfully!")
                import time as _time_me; _time_me.sleep(0.6)
                st.session_state.page = 'upload'
                st.rerun()



# ============================================================================
# PAGE 2: PINCH ANALYSIS
# ============================================================================
elif st.session_state.page == 'pinch_analysis':
    st.markdown("## 📊 Pinch Analysis")

    col_back, _ = st.columns([1, 9])
    with col_back:
        if st.button("← Back to Upload"):
            st.session_state.page = 'upload'
            st.rerun()

    st.markdown("---")

    try:
        params = st.session_state.questionnaire_data
        reader = st.session_state.questionnaire_reader

        st.info("📈 Computing and analysing pinch parameters…")

        with st.spinner("Generating pinch parameters…"):
            builder     = PinchParamBuilder(params, delta_T_min=10.0)
            pinch_input = builder.get_pinch_params()

        st.success("✅ Pinch parameters generated")

        st.markdown("---")

        with st.spinner("Running pinch analysis…"):
            analyzer = PinchAnalyzer(
                streams=pinch_input['streams'],
                delta_T_min=pinch_input['delta_T_min'],
                warnings=pinch_input.get('warnings', [])
            )
            analyzer.run()

        st.success("✅ Pinch analysis complete")

        st.session_state.pinch_analysis_data = {
            'analyzer':    analyzer,
            'builder':     builder,
            'pinch_input': pinch_input,
        }

        st.markdown("---")
        st.subheader("📋 Analysis Results")
        res = analyzer.results
        c1, c2, c3, c4 = st.columns(4)
        if res.T_pinch_hot  is not None: c1.metric("Pinch temp. (hot)",   f"{res.T_pinch_hot:.2f} °C")
        if res.T_pinch_cold is not None: c2.metric("Pinch temp. (cold)",  f"{res.T_pinch_cold:.2f} °C")
        if res.Q_H_min      is not None: c3.metric("Min. heating demand", f"{res.Q_H_min:.1f} kW")
        if res.Q_C_min      is not None: c4.metric("Min. cooling demand", f"{res.Q_C_min:.1f} kW")

        st.markdown("---")
        st.subheader("📊 Pinch Diagrams")
        tab1, tab2 = st.tabs(["T-Q Diagram", "Grand Composite Curve"])

        with tab1:
            st.write("**T-Q Diagram (Composite Curves)** – hot and cold composite curves with heating/cooling demand.")
            try:
                fig, ax = plt.subplots(figsize=(11, 7))
                analyzer.plot(ax=ax)
                _pyplot_interactive(fig, 'pinch_tq_diagram')
            except Exception as e:
                st.error(f"❌ Error plotting T-Q diagram: {e}")

        with tab2:
            st.write("**Grand Composite Curve (GCC)** – heat source and sink potential.")
            try:
                fig, ax = plt.subplots(figsize=(7, 7))
                analyzer.plot_gcc(ax=ax)
                _pyplot_interactive(fig, 'pinch_grand_composite_curve')
            except Exception as e:
                st.error(f"❌ Error plotting GCC: {e}")

        st.markdown("---")
        st.subheader("Next Steps")
        st.info("You can now move to Case Generation to define your heat pump configurations.")
        if st.button("📋 Go to Case Generation", use_container_width=True, type="primary"):
            st.session_state.page = 'case_generation'
            st.rerun()

    except Exception as e:
        st.error(f"❌ Pinch analysis error: {e}")
        st.write("Please check the input parameters and try again.")
        if st.button("← Back to Upload"):
            st.session_state.page = 'upload'
            st.rerun()


# ============================================================================
# PAGE 3: CASE GENERATION
# ============================================================================
elif st.session_state.page == 'case_generation':
    st.markdown("## 📋 Case Generation")

    col_back, _ = st.columns([1, 9])
    with col_back:
        if st.button("← Back to Upload"):
            st.session_state.page = 'upload'
            st.rerun()

    st.markdown("---")

    reader = st.session_state.questionnaire_reader
    case   = reader.get_application_case()
    params = st.session_state.questionnaire_data

    st.info(f"**Application case:** {case.upper()}")

    if case == 'hot_water':
        available_systems = ['HTHP']
    elif case == 'steam':
        available_systems = ['HTHP', 'MVR', 'HTHP+MVR']
    else:
        available_systems = ['HTHP', 'MVR', 'HTHP+MVR']

    # =========================================================================
    # TABS
    # =========================================================================
    tab_configure, tab_overview, tab_compressor = st.tabs(
        ["⚙️ Configure Case", "🌿 Refrigerant Overview", "🔩 Compressor Overview"]
    )

    # ── initialise variables that must exist after both tabs ──────────────────
    system_type       = available_systems[0]
    refrigerant       = None
    eta_s_values      = None
    overheat_values   = None
    mvr_stage_count   = None
    mvr_efficiencies  = None
    mvr_dT_per_stage  = 10.0
    hthp_model        = None
    hthp_econ_type    = None
    category          = "Simple Cycle"
    is_transcritical  = False
    # HTHP+MVR MVR-Stufe
    hthp_mvr_p_intermediate  = None
    hthp_mvr_dT_per_stage    = 10.0
    hthp_mvr_n_stages        = None
    hthp_mvr_efficiencies    = []
    hthp_mvr_sh              = 5.0   # Sauggasüberhitzung MVR [K]
    mvr_sh                   = 5.0   # Sauggasüberhitzung MVR [K]

    # =========================================================================
    # TAB 1: CONFIGURE CASE
    # =========================================================================
    with tab_configure:
        st.subheader("Configure New Case")

        col1, col2 = st.columns(2)
        with col1:
            system_type = st.selectbox("System Architecture", available_systems, key="system_type_sel")

            if system_type == 'MVR':
                p_in  = float(_val(params, 'steam_pressure_inlet')  or 1.0)
                p_out = float(_val(params, 'steam_pressure_outlet') or _val(params, 'hw_tank_pressure') or (p_in * 2))

                mvr_dT_per_stage = st.number_input(
                    "ΔT_sat per compressor stage [K]",
                    min_value=3.0, max_value=30.0, value=10.0, step=1.0,
                    key="mvr_dT_stage",
                    help="Saturation temperature difference per stage – determines the automatic stage count."
                )

                mvr_sh = st.number_input(
                    "Suction gas superheat at compressor inlet [K]",
                    min_value=1.0, max_value=30.0, value=5.0, step=0.5,
                    key="mvr_sh_input",
                    help=(
                        "Superheat applied at each compressor inlet after water injection. "
                        "The injected water cools the steam down to T_sat(p_stage) + SH. "
                        "Default: 5 K."
                    )
                )

                try:
                    mvr_stage_count, mvr_info = determine_n_stages(
                        p_in=p_in,
                        p_out=p_out,
                        dT_per_stage=mvr_dT_per_stage,
                        design_philosophy='standard',
                        verbose=False,
                    )
                except Exception:
                    mvr_stage_count, mvr_info = 2, {}

                st.info(f"Recommended MVR stages: **{mvr_stage_count}**")
                if mvr_info:
                    with st.expander("Stage selection details"):
                        _T_sat_out_mvr = mvr_info.get('T_sat_out', float('nan'))
                        st.write(f"T_sat in:  {mvr_info.get('T_sat_in', '—'):.1f} °C  →  T_sat out: {_T_sat_out_mvr:.1f} °C")
                        st.write(f"ΔT total: {mvr_info.get('dT_total', '—'):.1f} K  |  ΔT/stage: {mvr_info.get('dT_per_stage', '—')} K")
                        st.write(f"PR total: {mvr_info.get('PR_total', '—'):.3f}  |  PR/stage: {mvr_info.get('PR_per_stage', '—'):.3f}")
                        st.write(f"Reason: {mvr_info.get('reason', '—')}")

                mvr_efficiencies = []
                for i in range(mvr_stage_count):
                    eta = st.slider(f"Stage {i+1} efficiency", 0.50, 1.00, 0.85, 0.01, key=f"mvr_eta_{i}")
                    mvr_efficiencies.append(eta)

            else:
                if system_type in ('HTHP', 'HTHP+MVR'):

                    # ── Step 1: Base topology ─────────────────────────────────
                    category = st.selectbox(
                        'Cycle Type',
                        ['Simple Cycle', 'Intercooling', 'Economizer', 'Flash Tank', 'Cascade Cycle'],
                        key='category_sel',
                    )

                    # ── Step 2: Cascade sub-topology ──────────────────────────
                    cascade_sub = None
                    if category == 'Cascade Cycle':
                        cascade_sub = st.selectbox(
                            'Cascade enhancement',
                            ['Base', 'Intercooling', 'Economizer', 'Flash Tank'],
                            key='cascade_sub_sel',
                            help=(
                                'Base: two independent sub-circuits (LT + HT) sharing a heat exchanger.\n'
                                'Intercooling: intermediate cooling between LT and HT compressor.\n'
                                'Economizer: partial evaporation at midpressure to reduce compressor work.\n'
                                'Flash Tank: flash vessel separates liquid/vapor at midpressure.'
                            ),
                        )

                    needs_econ = (category == 'Economizer') or (cascade_sub == 'Economizer')

                    # ── Step 3: Economizer options ────────────────────────────
                    econ_type = None
                    comp_var  = None
                    if needs_econ:
                        col_et, col_cv = st.columns(2)
                        with col_et:
                            econ_type = st.radio(
                                'Economizer type',
                                ['closed', 'open'],
                                format_func=lambda x: (
                                    'Closed  (subcooling heat exchanger)' if x == 'closed'
                                    else 'Open  (flash-tank injection)'
                                ),
                                key='econ_type_sel',
                                help=(
                                    'Closed: a heat exchanger subcools the high-pressure liquid using the '
                                    'midpressure stream. No direct mixing.\n'
                                    'Open: midpressure flash vapor is injected directly into the '
                                    'compressor interstage — simpler but causes mixing.'
                                ),
                            )
                        with col_cv:
                            comp_var = st.radio(
                                'Compression topology',
                                ['series', 'parallel'],
                                format_func=lambda x: (
                                    'Series  (Reihenschaltung)' if x == 'series'
                                    else 'Parallel / PC  (Parallelschaltung)'
                                ),
                                key='comp_var_sel',
                                help=(
                                    'Series: LT and HT compressors are in series — all refrigerant '
                                    'passes through both stages sequentially.\n'
                                    'Parallel (PC = Parallel Compression): a separate smaller compressor '
                                    'handles only the midpressure flash vapor, while the main compressor '
                                    'handles the evaporator vapor. Reduces compression work.'
                                ),
                            )

                    # ── Step 4: IHX ──────────────────────────────────────────
                    ihx_variant = None
                    supports_ihx = category in ('Simple Cycle', 'Economizer') or cascade_sub in ('Base', 'Economizer')

                    if supports_ihx:
                        use_ihx = st.checkbox('Internal Heat Exchanger (IHX)', value=False, key='ihx_cb')
                        if use_ihx:
                            if category == 'Simple Cycle':
                                ihx_variant = 'ihx'
                                st.caption(
                                    'IHX: subcools the high-pressure liquid using the low-pressure '
                                    'suction vapor, increasing subcooling and superheating simultaneously.'
                                )
                            elif cascade_sub == 'Base':
                                ihx_variant = '2ihx'
                                st.caption(
                                    '2× IHX: one internal heat exchanger per sub-circuit (LT and HT), '
                                    'each in the standard position between condenser and expansion valve.'
                                )
                            else:
                                # Economizer or Cascade+Economizer — position matters
                                allow_both = (comp_var == 'parallel')
                                pos_opts = ['A', 'B'] + (['both'] if allow_both else [])
                                def _pos_label(x):
                                    if x == 'A':
                                        return 'Variant A  — LT side  (before economizer)'
                                    if x == 'B':
                                        return 'Variant B  — HT side  (after economizer)'
                                    return 'Both sides  (A + B, parallel compression only)'
                                ihx_variant = st.radio(
                                    'IHX position',
                                    pos_opts,
                                    format_func=_pos_label,
                                    key='ihx_pos_sel',
                                    help=(
                                        'Variant A: IHX placed between the LT compressor outlet and the '
                                        'economizer (midpressure level). The cold LT suction vapor from '
                                        'the evaporator subcools the high-pressure liquid — effective on '
                                        'the low-temperature side.\n\n'
                                        'Variant B: IHX placed between the economizer and the HT '
                                        'compressor inlet. The midpressure vapor superheats the HT '
                                        'suction gas before it enters the high-temperature compressor — '
                                        'effective on the high-temperature side.\n\n'
                                        'Both (PC only): one IHX in each position simultaneously.'
                                    ),
                                )

                    # ── Step 5: Transcritical ─────────────────────────────────
                    is_transcritical = st.checkbox('Transcritical process', value=False, key='trans_crit')

                    # ── Resolve model ─────────────────────────────────────────
                    _lookup_key = (
                        category, cascade_sub, econ_type, comp_var,
                        ihx_variant, is_transcritical,
                    )
                    _resolved = HP_MODEL_LOOKUP.get(_lookup_key)

                    if _resolved:
                        hthp_model, hthp_econ_type = _resolved
                        st.info(f'Model: **`{hthp_model}`**' + (f'  ·  econ_type: `{hthp_econ_type}`' if hthp_econ_type else ''))
                    else:
                        hthp_model, hthp_econ_type = None, None
                        st.warning('No model available for this combination.')


        st.markdown("---")

        # ── HTHP+MVR: MVR-Stufen-Konfiguration in col2 ───────────────────────
        if system_type == 'HTHP+MVR':
            st.markdown(
                "<div style='background:#0d2e1a;border-radius:8px;padding:10px 14px 6px 14px;"
                "border-left:4px solid #2ecc71;color:#d0f5dc'>"
                "<b>💨 MVR Stage – Handover & Stage Configuration</b><br>"
                "<small>Steam handover from HTHP condenser → MVR compresses to final pressure</small></div>",
                unsafe_allow_html=True
            )
            st.markdown("")

            _p_final_def = float(_val(params, 'steam_pressure_outlet') or 3.0)
            _p_int_def   = max(1.013, round(_p_final_def * 0.40, 2))

            col_pi, col_dt, col_sh = st.columns(3)
            with col_pi:
                hthp_mvr_p_intermediate = st.number_input(
                    "Handover pressure p_int [bar]  (HTHP → MVR)",
                    min_value=1.013, max_value=float(max(1.014, _p_final_def * 0.95)),
                    value=float(_p_int_def), step=0.1, key="hthp_mvr_p_int",
                    help="HTHP condenses at this pressure. MVR receives steam at T_sat(p_int) + SH and compresses to p_final."
                )
                try:
                    from CoolProp.CoolProp import PropsSI as _PSI_ui
                    _T_sat_int = _PSI_ui('T', 'P', hthp_mvr_p_intermediate * 1e5, 'Q', 1, 'Water') - 273.15
                    st.caption(f"T_sat({hthp_mvr_p_intermediate:.2f} bar) = **{_T_sat_int:.1f} °C**")
                except Exception:
                    _T_sat_int = None

            with col_dt:
                hthp_mvr_dT_per_stage = st.number_input(
                    "ΔT_sat per MVR stage [K]",
                    min_value=3.0, max_value=30.0, value=10.0, step=1.0,
                    key="hthp_mvr_dT_stage",
                    help="Saturation temperature difference per MVR stage – determines the automatic stage count."
                )

            with col_sh:
                hthp_mvr_sh = st.number_input(
                    "MVR suction gas superheat [K]",
                    min_value=1.0, max_value=30.0, value=5.0, step=0.5,
                    key="hthp_mvr_sh_input",
                    help=(
                        "Superheat at each MVR compressor inlet after water injection. "
                        "The HTHP must therefore condense at T_sat(p_int) + SH — i.e. "
                        "the HTHP condensation temperature is raised by this value. "
                        "Water injection cools back down to T_sat(p_stage) + SH. Default: 5 K."
                    )
                )
            try:
                _T_cond_hthp = _T_sat_int + hthp_mvr_sh if _T_sat_int is not None else None
                if _T_cond_hthp is not None:
                    st.caption(
                        f"⚠️ HTHP condenses at **{_T_cond_hthp:.1f} °C** "
                        f"(= T_sat + {hthp_mvr_sh:.1f} K)  ·  "
                        f"Water injection targets T_sat(p_stage) + {hthp_mvr_sh:.1f} K at each stage inlet."
                    )
            except Exception:
                pass

            try:
                hthp_mvr_n_stages, _hm_info = determine_n_stages(
                    p_in=hthp_mvr_p_intermediate,
                    p_out=_p_final_def,
                    dT_per_stage=hthp_mvr_dT_per_stage,
                    design_philosophy='standard',
                    verbose=False,
                )
            except Exception:
                hthp_mvr_n_stages, _hm_info = 1, {}

            st.info(f"Calculated MVR stage count: **{hthp_mvr_n_stages}**")
            if _hm_info:
                with st.expander("Stage design details"):
                    try:
                        from CoolProp.CoolProp import PropsSI as _PSI_st
                        _T_sat_in_hm  = _PSI_st('T', 'P', hthp_mvr_p_intermediate * 1e5, 'Q', 1, 'Water') - 273.15
                        _T_sat_out_hm = _PSI_st('T', 'P', _p_final_def * 1e5, 'Q', 1, 'Water') - 273.15
                        _dT_total_hm  = _T_sat_out_hm - _T_sat_in_hm
                    except Exception:
                        _T_sat_in_hm  = _hm_info.get('T_sat_in', float('nan'))
                        _T_sat_out_hm = _hm_info.get('T_sat_out', float('nan'))
                        _dT_total_hm  = _hm_info.get('dT_total', float('nan'))
                    st.write(f"T_sat: {_T_sat_in_hm:.1f} °C  →  {_T_sat_out_hm:.1f} °C")
                    st.write(f"ΔT total: {_dT_total_hm:.1f} K  |  ΔT/stage: {hthp_mvr_dT_per_stage} K")
                    st.write(f"PR total: {_hm_info.get('PR_total', 0):.3f}  |  PR/stage: {_hm_info.get('PR_per_stage', 0):.3f}")

            st.markdown("**η_s MVR compressor per stage:**")
            _mvr_eta_cols = st.columns(min(hthp_mvr_n_stages, 3))
            hthp_mvr_efficiencies = []
            for _i in range(hthp_mvr_n_stages):
                _col = _mvr_eta_cols[_i % len(_mvr_eta_cols)]
                with _col:
                    _eta_mvr = st.slider(
                        f"η_s MVR Stage {_i + 1}", 0.50, 1.00, 0.85, 0.01,
                        key=f"hthp_mvr_eta_{_i}"
                    )
                hthp_mvr_efficiencies.append(_eta_mvr)

        # ─────────────────────────────────────────────────────────────────────

        # ── helper flags ──────────────────────────────────────────────────────
        _is_cascade   = (category == "Cascade Cycle") and system_type != 'MVR'
        _is_multistage = (category not in ("Simple Cycle",)) and system_type != 'MVR'
        _avail = list(_REFRIG_CATALOGUE.keys())

        # How many IHX the resolved model actually has (0, 1, 2, or 4)
        # Used to decide how many superheat inputs to show
        _NR_IHX_1 = {
            'HeatPumpIHX','HeatPumpIHXTrans',
            'HeatPumpIHXEcon','HeatPumpIHXEconTrans',
            'HeatPumpEconIHX','HeatPumpEconIHXTrans',
            'HeatPumpIHXPC','HeatPumpIHXPCTrans',
            'HeatPumpPCIHX','HeatPumpPCIHXTrans',
        }
        _NR_IHX_2 = {
            'HeatPumpIHXPCIHX','HeatPumpIHXPCIHXTrans',
            'HeatPumpCascade2IHX','HeatPumpCascade2IHXTrans',
            'HeatPumpCascadeIHXEcon','HeatPumpCascadeIHXEconTrans',
            'HeatPumpCascadeEconIHX','HeatPumpCascadeEconIHXTrans',
            'HeatPumpCascadeIHXPC','HeatPumpCascadeIHXPCTrans',
            'HeatPumpCascadePCIHX','HeatPumpCascadePCIHXTrans',
        }
        _NR_IHX_4 = {
            'HeatPumpCascadeIHXPCIHX','HeatPumpCascadeIHXPCIHXTrans',
        }
        def _n_ihx_for(model):
            if model in _NR_IHX_4: return 4
            if model in _NR_IHX_2: return 2
            if model in _NR_IHX_1: return 1
            return 0

        intermediate_t = None
        intermediate_p = None

        # ── MVR ───────────────────────────────────────────────────────────────
        if system_type == 'MVR':
            for _k in [k for k in st.session_state.keys() if k.startswith('ref_')]:
                del st.session_state[_k]
            refrigerant  = "Steam (Water)"
            eta_s_values = {
                'mvr_efficiencies': mvr_efficiencies,
                'n_stages':         mvr_stage_count,
                'dT_per_stage':     mvr_dT_per_stage,
            }

        # ── CASCADE: 2-column block per circuit ───────────────────────────────
        elif _is_cascade:
            st.markdown("#### 🔁 Circuit Configuration")
            col_lp, col_hp = st.columns(2)

            with col_lp:
                st.markdown(
                    "<div style='background:#dbeeff;border-radius:8px;padding:10px 14px 4px 14px;"
                    "border-left:4px solid #1a7acc;color:#0d2a44'>"
                    "<b>🔵 LT Circuit – Low Temperature</b><br>"
                    "<small>Evaporates at heat source · condensate → Cascade HX</small></div>",
                    unsafe_allow_html=True
                )
                st.markdown("")
                _ref_lp = st.selectbox("Refrigerant LT", _avail, key="ref_0")
                _e1     = st.slider("η_s LT compressor", 0.50, 1.00, 0.85, 0.01, key="eta1")
                _o1     = st.number_input("Superheat LT [K]", 0.0, 50.0, 10.0, 1.0, key="oh_1")

            with col_hp:
                st.markdown(
                    "<div style='background:#ffe0e0;border-radius:8px;padding:10px 14px 4px 14px;"
                    "border-left:4px solid #cc2020;color:#3d0a0a'>"
                    "<b>🔴 HT Circuit – High Temperature</b><br>"
                    "<small>Evaporates from Cascade HX · condenses into heat sink</small></div>",
                    unsafe_allow_html=True
                )
                st.markdown("")
                _ref_hp = st.selectbox("Refrigerant HT", _avail, key="ref_1")
                _e2     = st.slider("η_s HT compressor", 0.50, 1.00, 0.85, 0.01, key="eta2")
                _o2     = st.number_input("Superheat HT [K]", 0.0, 50.0, 10.0, 1.0, key="oh_2")

            refrigerant     = f"{_ref_lp}, {_ref_hp}"
            eta_s_values    = [_e1, _e2]
            overheat_values = [_o1, _o2]

            # Cascade HX coupling temperature
            st.markdown("---")
            st.markdown("#### 🔗 Cascade Heat Exchanger")
            _te_def = float(_val(params, 'source_temp_out') or _val(params, 'source_temp_in') or 60.0)
            try:
                from CoolProp.CoolProp import PropsSI as _PSI2
                if _val(params, 'hw_temp_outlet_required'):
                    _tc_def = float(_val(params, 'hw_temp_outlet_required'))
                elif _val(params, 'steam_pressure_outlet'):
                    _tc_def = _PSI2('T', 'P', float(_val(params, 'steam_pressure_outlet')) * 1e5,
                                    'Q', 1, 'Water') - 273.15
                else:
                    _tc_def = 100.0
            except Exception:
                _tc_def = 100.0
            _t_mid_def = max(_te_def + 5.0, min(round((_te_def + _tc_def) / 2.0), _tc_def - 5.0))
            intermediate_t = st.number_input(
                "Cascade HX temperature T_casc [°C]",
                min_value=float(_te_def + 2), max_value=float(_tc_def - 2),
                value=float(_t_mid_def), step=1.0, key="t_casc",
                help="LT condensing = HT evaporating temperature level"
            )

        # ── SINGLE-STAGE HTHP ─────────────────────────────────────────────────
        elif not _is_multistage:
            refrigerant = st.selectbox("Refrigerant", _avail, key="ref_0")
            st.markdown("#### ⚙️ Compressor")
            _eta = st.slider("η_s compressor", 0.50, 1.00, 0.85, 0.01, key="eta_1")
            eta_s_values = [_eta]
            if system_type in ('HTHP', 'HTHP+MVR'):
                _oh = st.number_input("Superheat [K]", 0.0, 50.0, 10.0, 1.0, key="oh_1")
                overheat_values = [_oh]

        # ── MULTI-STAGE HTHP (IC / Econ / Flash / PC): same refrigerant, 2 stages ─
        else:
            refrigerant = st.selectbox("Refrigerant", _avail, key="ref_0")
            st.markdown("#### ⚙️ Compressor Stages")
            col_lp, col_hp = st.columns(2)

            _n_ihx = _n_ihx_for(hthp_model) if hthp_model else 0

            with col_lp:
                st.markdown(
                    "<div style='background:#dbeeff;border-radius:8px;padding:10px 14px 4px 14px;"
                    "border-left:4px solid #0d2a44'>"
                    "<b>LT Stage</b><br>"
                    "<small>Low-temperature compressor</small></div>",
                    unsafe_allow_html=True
                )
                st.markdown("")
                _e1 = st.slider("η_s LT", 0.50, 1.00, 0.85, 0.01, key="eta1")
                # IHX on LP side: Variant A (IHXEcon/IHXPC) or single-IHX EconIHX/PCIHX
                # = any model with exactly 1 IHX (all single-IHX models have it between
                #   LP outlet and economizer, i.e. on the LP suction side)
                _lp_has_ihx = _n_ihx >= 1
                if _lp_has_ihx:
                    _o1 = st.number_input(
                        "Superheat LT [K]",
                        0.0, 50.0, 5.0, 1.0, key="oh_1",
                        help="Superheating by the IHX on the LT side (between evaporator and LT compressor inlet)."
                    )
                else:
                    _o1 = 0.0

            with col_hp:
                st.markdown(
                    "<div style='background:#fff0dd;border-radius:8px;padding:10px 14px 4px 14px;"
                    "border-left:4px solid #c87000;color:#3d2000'>"
                    "<b>HT Stage</b><br>"
                    "<small>High-temperature compressor</small></div>",
                    unsafe_allow_html=True
                )
                st.markdown("")
                _e2 = st.slider("η_s HT", 0.50, 1.00, 0.85, 0.01, key="eta2")
                # IHX on HP side only for: IHXPCIHXTrans and Cascade equivalents (nr_ihx >= 2)
                # For single-IHX EconIHX/PCIHX: IHX is between economizer and HP inlet (Variant B)
                # → show for Variant B models OR for "both" (IHXPCIHXTrans)
                _model_is_varB = hthp_model in {
                    'HeatPumpEconIHX','HeatPumpEconIHXTrans',
                    'HeatPumpPCIHX','HeatPumpPCIHXTrans',
                    'HeatPumpCascadeEconIHX','HeatPumpCascadeEconIHXTrans',
                    'HeatPumpCascadePCIHX','HeatPumpCascadePCIHXTrans',
                }
                _hp_has_ihx = _model_is_varB or _n_ihx >= 2
                if _hp_has_ihx:
                    _o2 = st.number_input(
                        "Superheat HT [K]",
                        0.0, 50.0, 5.0, 1.0, key="oh_2",
                        help="Superheating by the IHX on the HT side (between economizer and HT compressor inlet)."
                    )
                else:
                    _o2 = 0.0

            eta_s_values = [_e1, _e2]
            # Build overheats list preserving position:
            #   index 0 → sh_lp (LP-side IHX, Variant A)
            #   index 1 → sh_hp (HP-side IHX, Variant B)
            # _apply_hthp_superheat reads them by position, so 0-padding matters.
            if _n_ihx == 0:
                overheat_values = None
            elif _model_is_varB and _n_ihx == 1:
                # Single IHX on HP side only: store as [0, sh_hp] so index 1 is correct.
                # _apply_hthp_superheat n=1 uses sh_lp if >0 else sh_hp → pass sh_hp in slot 1,
                # but since n=1 the function already takes whichever is non-zero, so [_o2] suffices.
                overheat_values = [_o2] if _o2 > 0 else None
            else:
                # Variant A, Both, or cascade with 2 IHX: preserve both positions
                overheat_values = [_o1, _o2] if (_o1 > 0 or _o2 > 0) else None

            # Intermediate pressure
            st.markdown("---")
            st.markdown("#### 🔀 Intermediate Pressure")
            st.caption("0 = model selects automatically via √(p_low · p_high)")
            intermediate_p = st.number_input(
                "p_mid [bar]  (0 = auto)", 0.0, 200.0, 0.0, 0.5, key="p_mid"
            ) or None

        # ── Case name + Add / Update button ──────────────────────────────────
        st.markdown("---")
        _edit_idx = st.session_state.editing_case_idx
        _is_editing = _edit_idx is not None and _edit_idx < len(st.session_state.cases)

        if _is_editing:
            st.info(f"✏️ Editing **Case #{_edit_idx + 1}** — adjust settings above, "
                    f"then click **Update Case** to save, or click **Cancel Edit** to discard.")

        _default_name = (st.session_state.cases[_edit_idx].get('name') or '') if _is_editing else ''
        case_name = st.text_input(
            "Case name (optional)",
            value=_default_name,
            placeholder="e.g. 'Cascade R600/R600a – 120 °C sink'",
            key="case_name_input"
        )

        def _build_case_dict(system_type, refrigerant, eta_s_values, case_name,
                             overheat_values, mvr_stage_count, mvr_dT_per_stage,
                             hthp_mvr_n_stages, hthp_mvr_p_intermediate,
                             hthp_mvr_dT_per_stage, hthp_mvr_efficiencies,
                             hthp_model, hthp_econ_type, intermediate_t, intermediate_p,
                             mvr_sh, hthp_mvr_sh):
            d = {
                'system_type':  system_type,
                'refrigerant':  refrigerant,
                'efficiencies': eta_s_values,
                'name':         case_name.strip() or None,
            }
            if overheat_values is not None:
                d['overheats'] = overheat_values
            if system_type == 'MVR':
                d['mvr_stages']    = mvr_stage_count
                d['dT_per_stage']  = float(mvr_dT_per_stage)
                d['mvr_sh']        = float(mvr_sh)
            if system_type == 'HTHP+MVR':
                d['mvr_stages']           = hthp_mvr_n_stages
                d['p_intermediate']        = float(hthp_mvr_p_intermediate) if hthp_mvr_p_intermediate is not None else None
                d['dT_per_stage']          = float(hthp_mvr_dT_per_stage)
                d['hthp_mvr_efficiencies'] = list(hthp_mvr_efficiencies or [])
                d['mvr_sh']                = float(hthp_mvr_sh)
            if hthp_model:
                d['hthp_model'] = hthp_model
            if hthp_econ_type is not None:
                d['econ_type'] = hthp_econ_type
            if intermediate_t is not None:
                d['t_cascade_hx'] = float(intermediate_t)
            if intermediate_p is not None:
                d['p_mid_bar'] = float(intermediate_p)
            return d

        _btn_row = st.columns([2, 2, 1]) if _is_editing else st.columns([3, 1])
        with _btn_row[0]:
            _btn_label = "💾 Update Case" if _is_editing else "➕ Add Case"
            if st.button(_btn_label, use_container_width=True, type="primary"):
                try:
                    case_dict = _build_case_dict(
                        system_type, refrigerant, eta_s_values, case_name,
                        overheat_values, mvr_stage_count, mvr_dT_per_stage,
                        hthp_mvr_n_stages, hthp_mvr_p_intermediate,
                        hthp_mvr_dT_per_stage, hthp_mvr_efficiencies,
                        hthp_model, hthp_econ_type, intermediate_t, intermediate_p,
                        mvr_sh, hthp_mvr_sh
                    )
                    if _is_editing:
                        st.session_state.cases[_edit_idx] = case_dict
                        st.session_state.editing_case_idx = None
                        _lbl = case_dict.get('name') or f"{system_type} – {refrigerant}"
                        st.success(f"✅ Case #{_edit_idx + 1} updated: {_lbl}")
                    else:
                        st.session_state.cases.append(case_dict)
                        _lbl = case_dict.get('name') or f"{system_type} – {refrigerant}"
                        st.success(f"✅ Case added: {_lbl}")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

        if _is_editing:
            with _btn_row[1]:
                if st.button("↩ Cancel Edit", use_container_width=True):
                    st.session_state.editing_case_idx = None
                    st.rerun()

    # =========================================================================
    # TAB 2: REFRIGERANT OVERVIEW
    # =========================================================================
    with tab_overview:
        _render_refrigerant_overview(params)

    with tab_compressor:
        _render_compressor_overview(params)

    # =========================================================================
    # DEFINED CASES (always visible, outside tabs)
    # =========================================================================
    st.markdown("---")
    st.subheader("📑 Defined Cases")

    if st.session_state.cases:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Cases",  len(st.session_state.cases))
        c2.metric("System Types", len(set(c['system_type'] for c in st.session_state.cases)))
        c3.metric("Refrigerants", len(set(c['refrigerant'] for c in st.session_state.cases)))

        st.markdown("---")

        n_cases = len(st.session_state.cases)
        for idx, case_item in enumerate(st.session_state.cases):
            _i          = idx          # 0-based index
            _num        = idx + 1      # 1-based display
            effs        = case_item['efficiencies']
            _is_cascade = ',' in (case_item.get('refrigerant') or '')
            _refrig_raw = case_item.get('refrigerant', '—')
            _cname      = case_item.get('name') or ''

            if _is_cascade:
                _parts = [r.strip() for r in _refrig_raw.split(',')]
                _ref_display = f"🔵 {_parts[0]}  /  🔴 {_parts[1] if len(_parts) > 1 else '?'}"
            else:
                _ref_display = _refrig_raw

            _title_name = f"  ·  \"{_cname}\"" if _cname else ''
            _edit_marker = '  ✏️ *editing*' if st.session_state.editing_case_idx == _i else ''
            title = (f"**Case #{_num}**{_title_name}{_edit_marker}  |  "
                     f"{case_item['system_type']}  |  {_ref_display}")
            if case_item.get('hthp_model'):
                title += f"  |  `{case_item['hthp_model']}`"

            with st.expander(title, expanded=(st.session_state.editing_case_idx == _i)):

                # ── action buttons row ────────────────────────────────────────
                _btn_cols = st.columns([1, 1, 1, 1, 4])
                with _btn_cols[0]:
                    if _i > 0 and st.button("▲", key=f"up_{_i}", help="Move up"):
                        c = st.session_state.cases
                        c[_i - 1], c[_i] = c[_i], c[_i - 1]
                        if st.session_state.editing_case_idx == _i:
                            st.session_state.editing_case_idx = _i - 1
                        elif st.session_state.editing_case_idx == _i - 1:
                            st.session_state.editing_case_idx = _i
                        st.rerun()
                with _btn_cols[1]:
                    if _i < n_cases - 1 and st.button("▼", key=f"dn_{_i}", help="Move down"):
                        c = st.session_state.cases
                        c[_i], c[_i + 1] = c[_i + 1], c[_i]
                        if st.session_state.editing_case_idx == _i:
                            st.session_state.editing_case_idx = _i + 1
                        elif st.session_state.editing_case_idx == _i + 1:
                            st.session_state.editing_case_idx = _i
                        st.rerun()
                with _btn_cols[2]:
                    _edit_label = "✏️ Edit" if st.session_state.editing_case_idx != _i else "↩ Cancel"
                    if st.button(_edit_label, key=f"edit_{_i}", use_container_width=True):
                        if st.session_state.editing_case_idx == _i:
                            st.session_state.editing_case_idx = None
                        else:
                            st.session_state.editing_case_idx = _i
                        st.rerun()
                with _btn_cols[3]:
                    if st.button("🗑️", key=f"del_{_i}", help="Delete this case"):
                        st.session_state.cases.pop(_i)
                        if st.session_state.editing_case_idx == _i:
                            st.session_state.editing_case_idx = None
                        elif (st.session_state.editing_case_idx or 0) > _i:
                            st.session_state.editing_case_idx -= 1
                        st.rerun()

                st.markdown("---")

                # ── summary display ───────────────────────────────────────────
                st.write(f"**System:** {case_item['system_type']}"
                         + (f"  ·  `{case_item['hthp_model']}`" if case_item.get('hthp_model') else ''))

                if _is_cascade:
                    _parts = [r.strip() for r in _refrig_raw.split(',')]
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown(
                            "<div style='background:#1a3a5c;border-radius:6px;"
                            "padding:8px 12px;border-left:3px solid #4da6ff'>"
                            "<b>🔵 LT Circuit</b></div>", unsafe_allow_html=True)
                        st.write(f"Refrigerant: **{_parts[0]}**")
                        if isinstance(effs, list) and len(effs) >= 1:
                            st.write(f"η_s: {effs[0]:.3f}")
                        oh = case_item.get('overheats', [])
                        if oh: st.write(f"SH: {oh[0]:.1f} K")
                    with col_b:
                        st.markdown(
                            "<div style='background:#5c1a1a;border-radius:6px;"
                            "padding:8px 12px;border-left:3px solid #ff6b6b'>"
                            "<b>🔴 HT Circuit</b></div>", unsafe_allow_html=True)
                        st.write(f"Refrigerant: **{_parts[1] if len(_parts) > 1 else '?'}**")
                        if isinstance(effs, list) and len(effs) >= 2:
                            st.write(f"η_s: {effs[1]:.3f}")
                        if oh and len(oh) > 1: st.write(f"SH: {oh[1]:.1f} K")
                elif isinstance(effs, list) and len(effs) >= 2:
                    col_a, col_b = st.columns(2)
                    oh = case_item.get('overheats', [])
                    with col_a:
                        st.markdown(
                            "<div style='background:#1a3a5c;border-radius:6px;"
                            "padding:8px 12px;border-left:3px solid #4da6ff'>"
                            "<b>LT Stage</b></div>", unsafe_allow_html=True)
                        st.write(f"η_s: {effs[0]:.3f}")
                        if oh: st.write(f"SH: {oh[0]:.1f} K")
                    with col_b:
                        st.markdown(
                            "<div style='background:#5c3a1a;border-radius:6px;"
                            "padding:8px 12px;border-left:3px solid #ffb366'>"
                            "<b>HT Stage</b></div>", unsafe_allow_html=True)
                        st.write(f"η_s: {effs[1]:.3f}")
                        if oh and len(oh) > 1: st.write(f"SH: {oh[1]:.1f} K")
                else:
                    if isinstance(effs, list):
                        st.write(f"η_s: {effs[0]:.3f}")
                    elif isinstance(effs, dict) and 'mvr_efficiencies' in effs:
                        st.write("MVR: " + "  /  ".join(f"Stage {i+1}: {v:.3f}"
                                 for i, v in enumerate(effs.get('mvr_efficiencies') or [])))
                    oh = case_item.get('overheats', [])
                    if oh: st.write(f"SH: {oh[0]:.1f} K")

                _tc  = case_item.get('t_cascade_hx')
                _pm  = case_item.get('p_mid_bar')
                _pi  = case_item.get('p_intermediate')
                _dts = case_item.get('dT_per_stage')
                _msh = case_item.get('mvr_sh')
                if any(v is not None for v in (_tc, _pm, _pi, _dts, _msh)):
                    st.markdown("---")
                    if _tc  is not None: st.write(f"🔗 Cascade HX: **{_tc:.1f} °C**")
                    if _pm  is not None: st.write(f"⚙️ p_mid: **{_pm:.2f} bar**")
                    if _pi  is not None: st.write(f"🔀 Handover pressure HTHP→MVR: **{_pi:.2f} bar**")
                    if _dts is not None: st.write(f"📐 ΔT_sat/stage (MVR): **{_dts:.1f} K**")
                    if _msh is not None: st.write(f"🌡️ MVR suction gas superheat: **{_msh:.1f} K**")

                if st.session_state.editing_case_idx == _i:
                    st.info("✏️ This case is loaded in the **Configure Case** tab above for editing. "
                            "Modify the settings there and click **Update Case** to save changes.")
                else:
                    st.caption("✅ Ready for calculation")

        st.markdown("---")
        if st.button("🗑️ Delete All Cases", use_container_width=True):
            st.session_state.cases = []
            st.session_state.editing_case_idx = None
            st.rerun()
    else:
        st.info("ℹ️ No cases defined yet. Add at least one case to proceed.")

    st.markdown("---")

    if st.session_state.cases:
        if st.button("▶️ Start Calculation", use_container_width=True, type="primary", key="calculate_btn"):
            st.session_state.calculation_results = None
            st.session_state.calc_case_ids       = []
            st.session_state.page                = 'calculation'
            st.session_state.calculation_started = True
            st.rerun()
    else:
        st.warning("⚠️ Please add at least one case before starting the calculation.")



# ============================================================================
# PAGE 4: CALCULATION
# ============================================================================
elif st.session_state.page == 'calculation':

    # If results already exist (user navigated back/forward), go straight to results
    if st.session_state.calculation_results is not None:
        st.session_state.page = 'results'
        st.rerun()

    st.markdown("## 🔄 Running Calculations")

    col_back, _ = st.columns([1, 9])
    with col_back:
        if st.button("← Back to Case Generation"):
            st.session_state.page                = 'case_generation'
            st.session_state.calculation_started = False
            st.rerun()

    st.markdown("---")

    total_cases = len(st.session_state.cases)
    st.write(f"**{total_cases} case(s) queued for calculation.**")

    # Progress UI elements
    progress_bar  = st.progress(0)
    status_text   = st.empty()
    log_container = st.empty()

    status_text.info("⚙️ Importing calculation modules…")

    try:
        from case_calculator import calculate_cases, CaseStatus
    except ImportError as exc:
        progress_bar.empty()
        st.error(f"❌ Could not import `case_calculator`: {exc}")
        st.stop()

    # Convert UI cases → calculator format (no math, just restructuring)
    try:
        calc_cases = _ui_cases_to_calc_format(st.session_state.cases)
    except Exception as exc:
        st.error(f"❌ Case conversion error: {exc}")
        st.code(traceback.format_exc())
        st.stop()

    params    = st.session_state.questionnaire_data
    log_lines = []

    def _on_progress(current: int, total: int, case_id: str, status_str: str):
        """Called by case_calculator after each case completes."""
        progress_bar.progress(current / total)
        icon = {"success": "✅", "failed": "❌", "skipped": "⏭️"}.get(status_str, "🔄")
        status_text.info(f"Progress: **{current} / {total}** cases finished")
        log_lines.append(f"{icon}  [{current:>2}/{total}]  {case_id:<55}  {status_str.upper()}")
        log_container.code("\n".join(log_lines[-15:]))

    # Run calculation
    status_text.info(f"🚀 Starting calculation of {total_cases} case(s)…")
    try:
        results = calculate_cases(
            cases=calc_cases,
            params=params,
            verbose=False,
            progress_callback=_on_progress,
        )
    except Exception as exc:
        progress_bar.empty()
        st.error(f"❌ Calculation error: {exc}")
        st.code(traceback.format_exc())
        st.stop()

    # Optional: generate state diagrams for successful cases
    if _DIAGRAMS_AVAILABLE:
        status_text.info("📊 Generating state diagrams…")
        os.makedirs('results/plots', exist_ok=True)
        for case_id, result in results.items():
            if result.status != CaseStatus.SUCCESS or result.model_instance is None:
                continue
            try:
                diagram_paths = {}
                # ── HTHP+MVR: model_instance is a (hp, mvr) tuple ───────────
                if isinstance(result.model_instance, tuple):
                    hp_inst, mvr_inst = result.model_instance
                    # HTHP-Teil: log(p)-h und T-s via StateDiagramGenerator
                    gen = StateDiagramGenerator(
                        hp_inst, results_dir='results/plots', style='light'
                    )
                    hthp_diags = gen.generate_diagrams(scenario_name=f'{case_id}_hthp')
                    for k, v in hthp_diags.items():
                        if not k.startswith('combined_'):
                            diagram_paths[f'hthp_{k}'] = v
                    # MVR-Teil: log(p)-h und T-s via mvr.generate_state_diagram()
                    # Build feedwater dict so the feedwater inlet state appears in the diagram
                    _fw_dict = None
                    try:
                        # Read directly from params (always in original units, never overwritten by TESPy)
                        _T_fw  = mvr_inst.params.get('cooling_water', {}).get('T_in')
                        _p_bar = mvr_inst.params.get('inlet', {}).get('p')
                        _T_end = mvr_inst.params.get('inlet', {}).get('T')
                        # Fallback: read T_end from solved TESPy connection if params not set
                        if _T_end is None:
                            try:
                                _T_end = mvr_inst.comps['comp_1'].inl[0].T.val
                            except Exception:
                                pass
                        if _T_fw is not None and _p_bar is not None and _T_end is not None:
                            _fw_dict = {
                                'T_fw_C':  float(_T_fw),
                                'p_bar':   float(_p_bar),
                                'T_end_C': float(_T_end),
                            }
                    except Exception:
                        pass
                    mvr_diags = _generate_mvr_diagrams(
                        mvr_inst, case_id, results_dir='results/plots',
                        feedwater=_fw_dict,
                    )
                    for k, v in mvr_diags.items():
                        diagram_paths[f'mvr_{k}'] = v
                # ── Pure MVR ─────────────────────────────────────────────────
                elif result.case_type and result.case_type.value == 'mvr':
                    diagram_paths = _generate_mvr_diagrams(
                        result.model_instance,
                        case_id=case_id,
                        results_dir='results/plots',
                    )
                # ── Pure HTHP ────────────────────────────────────────────────
                else:
                    gen = StateDiagramGenerator(
                        result.model_instance,
                        results_dir='results/plots',
                        style='light',
                    )
                    diagram_paths = gen.generate_diagrams(scenario_name=case_id)
                    diagram_paths = {k: v for k, v in diagram_paths.items()
                                     if not k.startswith('combined_')}
                if diagram_paths:
                    result.extra['diagram_paths'] = diagram_paths
            except Exception:
                pass   # diagrams are optional

    # Finish
    progress_bar.progress(1.0)
    n_ok   = sum(1 for r in results.values() if r.status == CaseStatus.SUCCESS)
    n_fail = len(results) - n_ok
    status_text.success(
        f"✅ All calculations finished — {n_ok} successful, {n_fail} failed / skipped."
    )

    st.session_state.calculation_results = results
    st.session_state.calc_case_ids       = [c['id'] for c in calc_cases]

    time.sleep(1.2)
    st.session_state.page = 'results'
    st.rerun()


# ============================================================================
# PAGE 5: RESULTS
# ============================================================================
elif st.session_state.page == 'results':
    st.markdown("## 📈 Simulation Results")

    col_back, col_recalc, _ = st.columns([1.5, 1.5, 7])
    with col_back:
        if st.button("← Case Generation"):
            st.session_state.page                = 'case_generation'
            st.session_state.calculation_results = None
            st.session_state.calculation_started = False
            st.rerun()
    with col_recalc:
        if st.button("🔄 Recalculate"):
            st.session_state.calculation_results = None
            st.session_state.page                = 'calculation'
            st.session_state.calculation_started = True
            st.rerun()

    st.markdown("---")

    results  = st.session_state.calculation_results
    ui_cases = st.session_state.cases

    if results is None or len(results) == 0:
        st.warning("No results available. Please run the calculation first.")
        st.stop()

    # One tab per case + comparative overview
    individual_labels = []
    for i, (case_id, result) in enumerate(results.items()):
        ui_case = ui_cases[i] if i < len(ui_cases) else {}
        label   = f"Case {i+1} · {ui_case.get('system_type', '?')}"
        individual_labels.append(label)

    all_tabs = st.tabs(individual_labels + ["🔍 Comparative Overview"])

    # Individual case tabs
    for i, (tab, (case_id, result)) in enumerate(zip(all_tabs[:-1], results.items())):
        with tab:
            ui_case  = ui_cases[i] if i < len(ui_cases) else {}
            refrig   = ui_case.get('refrigerant', '—')
            model    = ui_case.get('hthp_model',  '—')
            sys_type = ui_case.get('system_type', '—')

            st.markdown(f"#### {sys_type}  ·  {model}  ·  {refrig}")
            st.caption(f"Case ID: `{case_id}`")
            st.markdown("---")
            _render_case_result(result, ui_case)

    # Comparative overview tab
    with all_tabs[-1]:
        st.markdown("#### Comparative Overview – All Cases")
        st.markdown("---")
        _render_comparative_overview(results, ui_cases)


# ============================================================================
# Footer
# ============================================================================
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray; font-size:0.85em;'>"
    "HTHP Modeling Environment · Version 2.1"
    "</div>",
    unsafe_allow_html=True,
)