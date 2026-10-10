"""
theme.py - drop-in visual upgrade for the SignalSage Streamlit app (CSS only, no logic changes).

Use (two lines in app.py, right after st.set_page_config(...)):
    import theme
    theme.apply_theme("light")        # or "dark"  (dark also needs base="dark" in .streamlit/config.toml)
Optional one-line chart polish, at the end of make_chart():   return theme.style_fig(fig)
"""
import streamlit as st

PLOTLY_TEMPLATE = "plotly_white"
_MODE = "light"

WAVE = ("url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='220' height='60' viewBox='0 0 220 60'>"
        "<path d='M0 30 H60 L70 10 L80 50 L90 30 H140 C150 30 154 18 164 18 C174 18 178 30 190 30 H220' "
        "fill='none' stroke='%2399f6e4' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'/></svg>\")")

BASE = """
<style>
:root{--a1:#0f766e;--a2:#2563eb;--a3:#14b8a6;}
@keyframes shift{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
@keyframes scrollwave{to{background-position-x:-220px}}
@keyframes rise{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:none}}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(239,68,68,.65)}70%{box-shadow:0 0 0 8px rgba(239,68,68,0)}100%{box-shadow:0 0 0 0 rgba(239,68,68,0)}}
header[data-testid="stHeader"]{background:transparent}
.block-container{animation:rise .6s ease both}
::selection{background:#14b8a6;color:#fff}

/* hero */
.hero{position:relative;overflow:hidden;background:linear-gradient(120deg,#0f172a,#115e59,#0f766e,#2563eb,#0f172a)!important;
  background-size:320% 320%!important;animation:shift 16s ease infinite;padding:30px 34px 66px!important;
  box-shadow:0 18px 50px rgba(15,118,110,.38);border:1px solid rgba(255,255,255,.12)}
.hero::before{content:"";position:absolute;inset:0;pointer-events:none;
  background-image:linear-gradient(rgba(255,255,255,.07) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.07) 1px,transparent 1px);
  background-size:30px 30px;-webkit-mask-image:radial-gradient(circle at 85% 15%,#000,transparent 70%);mask-image:radial-gradient(circle at 85% 15%,#000,transparent 70%)}
.hero::after{content:"";position:absolute;left:0;right:0;bottom:8px;height:60px;pointer-events:none;opacity:.6;
  background:__WAVE__ repeat-x;background-size:220px 60px;animation:scrollwave 7s linear infinite}
.hero > *{position:relative;z-index:1}
.hero h1{font-size:2.5rem!important;letter-spacing:-.02em;background:linear-gradient(90deg,#fff 20%,#99f6e4 80%);
  -webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent}
.hero p{color:#cffafe!important;font-size:1.05rem!important}

/* tabs as pills + live dot (works for both the old BaseWeb and the new react-aria tab markup) */
.stTabs [data-baseweb="tab-list"],[role="tablist"]{gap:6px!important;padding:6px!important;border-radius:999px!important;width:fit-content!important;
  background:var(--glass)!important;border:1px solid var(--gb)!important;backdrop-filter:blur(10px);box-shadow:var(--shadow)}
.stTabs [data-baseweb="tab"],[role="tab"]{height:auto!important;padding:9px 20px!important;border-radius:999px!important;font-weight:600;
  transition:all .2s ease;border:0!important;margin:0!important}
.stTabs [data-baseweb="tab"]:hover,[role="tab"]:hover{background:rgba(20,184,166,.16)!important}
.stTabs [aria-selected="true"],[role="tab"][aria-selected="true"]{background:linear-gradient(90deg,var(--a1),var(--a2))!important;
  box-shadow:0 6px 16px rgba(37,99,235,.35)}
.stTabs [aria-selected="true"] *,[role="tab"][aria-selected="true"] *{color:#fff!important}
.stTabs [aria-selected="true"],[role="tab"][aria-selected="true"]{color:#fff!important}
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"],[data-testid="stTabs"] [role="tablist"]::after{display:none!important}
.stTabs [data-baseweb="tab"]:nth-child(2)::after,[role="tab"]:nth-of-type(2)::after{content:"";display:inline-block;width:8px;height:8px;margin-left:9px;
  border-radius:50%;background:#ef4444;animation:pulse 1.4s infinite;vertical-align:middle}

[role="tab"]::before{display:none!important}
[role="tab"]:not(:nth-of-type(2))::after{display:none!important}

/* buttons */
button[data-testid="stBaseButton-primary"],.stButton>button[kind="primary"]{background:linear-gradient(90deg,var(--a1),var(--a2))!important;
  border:0!important;color:#fff!important;border-radius:999px!important;padding:.55rem 1.5rem!important;font-weight:600!important;
  box-shadow:0 8px 22px rgba(37,99,235,.38);transition:transform .15s ease,box-shadow .15s ease}
button[data-testid="stBaseButton-primary"]:hover{transform:translateY(-2px);box-shadow:0 12px 28px rgba(37,99,235,.5)}
button[data-testid="stBaseButton-secondary"],.stButton>button[kind="secondary"],.stDownloadButton>button{border-radius:999px!important;
  border:1px solid var(--gb)!important;transition:all .15s ease}
button[data-testid="stBaseButton-secondary"]:hover,.stDownloadButton>button:hover{border-color:var(--a3)!important;transform:translateY(-2px)}

/* KPI + alert cards */
.kpi{position:relative;overflow:hidden;background:var(--glass)!important;backdrop-filter:blur(10px);border:1px solid var(--gb)!important;
  border-left:1px solid var(--gb)!important;border-radius:16px!important;padding:16px 20px!important;box-shadow:var(--shadow)!important;
  animation:rise .5s ease both;transition:transform .2s ease,box-shadow .2s ease}
.kpi::before{content:"";position:absolute;left:0;top:0;bottom:0;width:5px;background:linear-gradient(180deg,var(--a3),var(--a2))}
.kpi:hover{transform:translateY(-4px);box-shadow:0 16px 36px rgba(20,184,166,.28)!important}
.kpi .v{font-size:1.9rem!important;background:linear-gradient(90deg,var(--a1),var(--a2));-webkit-background-clip:text;background-clip:text;
  -webkit-text-fill-color:transparent}
.card{background:var(--glass)!important;backdrop-filter:blur(10px);border:1px solid var(--gb);border-radius:18px!important;
  box-shadow:var(--shadow)!important;animation:rise .55s ease both;transition:transform .2s ease}
.card:hover{transform:translateX(4px)}
.badge{box-shadow:0 4px 12px rgba(0,0,0,.25);letter-spacing:.02em}

/* widgets */
div[data-testid="stPlotlyChart"]{background:var(--glass);border:1px solid var(--gb);border-radius:18px;padding:8px;box-shadow:var(--shadow);animation:rise .6s ease both}
div[data-testid="stDataFrame"]{border-radius:14px;overflow:hidden;box-shadow:var(--shadow);border:1px solid var(--gb)}
section[data-testid="stFileUploaderDropzone"]{border:2px dashed var(--a3)!important;border-radius:18px!important;background:rgba(20,184,166,.08)!important;transition:all .2s ease}
section[data-testid="stFileUploaderDropzone"]:hover{background:rgba(20,184,166,.16)!important;box-shadow:0 0 0 4px rgba(20,184,166,.15)}
div[data-testid="stAlert"]{border-radius:14px}
details{border-radius:14px!important;border:1px solid var(--gb)!important}
div[data-baseweb="select"]>div,div[data-baseweb="input"]>div{border-radius:12px!important}
section[data-testid="stSidebar"]{border-right:1px solid var(--gb)}
section[data-testid="stSidebar"] h3{background:linear-gradient(90deg,var(--a1),var(--a2));-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent}
h4{letter-spacing:-.01em}
::-webkit-scrollbar{width:10px;height:10px}::-webkit-scrollbar-thumb{background:linear-gradient(var(--a3),var(--a2));border-radius:10px}
</style>
"""

LIGHT = """
<style>
:root{--glass:rgba(255,255,255,.80);--gb:rgba(15,118,110,.18);--shadow:0 8px 28px rgba(15,23,42,.10)}
.stApp{background:radial-gradient(1100px 600px at 8% -8%,rgba(20,184,166,.20),transparent 60%),
  radial-gradient(900px 520px at 100% 0%,rgba(99,102,241,.16),transparent 58%),#f4f7fb}
.stTabs [data-baseweb="tab"]{color:#475569}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,rgba(20,184,166,.14),rgba(37,99,235,.08)),#eef4f3}
</style>
"""

DARK = """
<style>
:root{--glass:rgba(17,24,39,.72);--gb:rgba(148,163,184,.22);--shadow:0 10px 34px rgba(0,0,0,.45);--a1:#14b8a6;--a2:#6366f1}
.stApp{background:radial-gradient(1100px 600px at 8% -8%,rgba(20,184,166,.22),transparent 60%),
  radial-gradient(900px 520px at 100% 0%,rgba(99,102,241,.25),transparent 58%),#0b1220}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,rgba(20,184,166,.12),rgba(99,102,241,.10)),#0d1526}
.card,.card h4,.card p{color:#e5e7eb!important}
.card .meta,.card .lab{color:#94a3b8!important}
.chk{color:#cbd5e1!important}
.pill{background:rgba(148,163,184,.22)!important;color:#e5e7eb!important}
.kpi .l{color:#94a3b8!important}
.stTabs [data-baseweb="tab"]{color:#94a3b8}
</style>
"""


def apply_theme(mode="light"):
    global PLOTLY_TEMPLATE, _MODE
    _MODE = mode
    PLOTLY_TEMPLATE = "plotly_dark" if mode == "dark" else "plotly_white"
    st.markdown(BASE.replace("__WAVE__", WAVE) + (DARK if mode == "dark" else LIGHT), unsafe_allow_html=True)


def style_fig(fig):
    """Optional: transparent chart background (the card behind it comes from CSS)."""
    fig.update_layout(template=PLOTLY_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    if _MODE == "dark":
        fig.update_annotations(font_color="#cbd5e1")
    return fig
