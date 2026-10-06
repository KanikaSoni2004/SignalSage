"""
app.py  -  SignalSage web interface (Streamlit)
Run:  streamlit run app.py
"""
import html
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import engine
from engine import W
from llm import MockBackend, OllamaBackend, OpenAICompatBackend
from signalsage_core import FAULTS

st.set_page_config(page_title="SignalSage", page_icon="📡", layout="wide")

GREEN, RED, AMBER, BLUE, GREY = "#16a34a", "#dc2626", "#d97706", "#2563eb", "#94a3b8"
VERDICT = {"normal": ("Looks healthy", GREEN), "data_fault": ("Data fault suspected", RED),
           "unsure": ("Needs a human look", AMBER)}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }
.block-container { padding-top: 1.2rem; max-width: 1250px; }
.hero { background: linear-gradient(120deg,#0f172a 0%,#115e59 60%,#0f766e 100%); color: #fff; border-radius: 18px;
        padding: 26px 32px; margin-bottom: 18px; }
.hero h1 { margin: 0; font-size: 2.0rem; font-weight: 700; color:#fff; }
.hero p { margin: 6px 0 0 0; font-size: 1.02rem; color: #d1fae5; }
.kpi { background:#fff; border-radius:14px; padding:14px 18px; box-shadow:0 1px 3px rgba(15,23,42,.10); border-left:5px solid #0f766e; }
.kpi .v { font-size:1.7rem; font-weight:700; color:#0f172a; line-height:1.1; }
.kpi .l { font-size:.82rem; color:#475569; text-transform:uppercase; letter-spacing:.04em; }
.card { background:#fff; color:#0f172a; border-radius:14px; padding:16px 20px; margin:10px 0;
        box-shadow:0 1px 4px rgba(15,23,42,.12); border-left:6px solid #94a3b8; }
.card h4 { margin:0 0 4px 0; font-size:1.05rem; color:#0f172a; }
.card .meta { color:#64748b; font-size:.83rem; margin-bottom:8px; }
.card .lab { font-size:.75rem; font-weight:700; letter-spacing:.06em; color:#475569; text-transform:uppercase; margin-top:8px; }
.card p { margin:2px 0 4px 0; color:#0f172a; }
.badge { display:inline-block; padding:2px 10px; border-radius:999px; color:#fff; font-size:.78rem; font-weight:600; margin-right:6px; }
.pill { display:inline-block; padding:2px 10px; border-radius:999px; background:#e2e8f0; color:#0f172a; font-size:.78rem; margin-right:6px; }
.chk { font-size:.8rem; color:#334155; margin-top:8px; }
div[data-testid="stSidebar"] { background:#eef4f3; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>📡 SignalSage</h1><p>Can you trust your sensor data? '
            'Upload a sensor series and get a verified verdict, the evidence, and a plain-language message for your engineers.</p></div>',
            unsafe_allow_html=True)


@st.cache_resource
def get_models():
    return engine.load_models()


try:
    MODELS = get_models()
except Exception as e:
    st.error(f"Could not load models.joblib ({e}). Run `python step3_train_models.py` first.")
    st.stop()

# ------------------------------------------------------------------------------------ sidebar
PRESETS = {"Cautious": (0.90, 0.10), "Balanced": (0.75, 0.25), "Sensitive": (0.60, 0.40)}
TRADEOFF = {"Cautious": "answers about 1 in 3 windows, ~98% right on those",
            "Balanced": "answers about 2 in 3 windows, ~97% right on those",
            "Sensitive": "answers about 85% of windows, ~94% right on those"}
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    preset = st.radio("Decision strictness", ["Cautious", "Balanced", "Sensitive", "Custom"], index=1, key="preset",
                      help="How confident the classifier must be before it calls a window faulty or healthy. "
                           "Anything in between goes to a human.")
    if preset == "Custom":
        hi = st.slider("Flag as fault at or above", 0.50, 0.95, 0.75, 0.01, key="hi")
        lo = st.slider("Call healthy at or below", 0.05, 0.50, 0.25, 0.01, key="lo")
    else:
        hi, lo = PRESETS[preset]
        st.caption(f"{preset}: fault ≥ {hi:.2f}, healthy ≤ {lo:.2f}. On our NAB-based development data it {TRADEOFF[preset]} "
                   "(fault vs. not-fault). Your sensors may behave differently.")
    st.markdown("**Window length:** 256 points *(fixed: the model was trained on this)*")
    stride = st.select_slider("Audit every N points", options=[16, 32, 64, 128, 256], value=64, key="stride",
                              help="Smaller = more windows and finer timing, but more compute and more alerts.")
    hist_share = st.slider("History used as baseline (%)", 20, 70, 45, 5, key="hist") / 100
    st.markdown("---")
    ai_mode = st.radio("Engineer messages", ["Screening only", "Template (no AI)", "Demo model (mock)", "Local AI (Ollama)", "Cloud AI (Groq)"],
                   index=4, key="ai_mode",
                   help="Template = rule-written text. Mock = fake model for testing. "
                        "Ollama = local model. Groq = fast cloud LLM using Groq API.")
    ollama_model = "granite4:tiny-h"
    if ai_mode == "Local AI (Ollama)":
        installed = engine.ollama_models()
        if installed is None:
            st.warning("Ollama is not reachable at localhost:11434. Start it, then reload.")
            ollama_model = st.text_input("Model name", "granite4:tiny-h", key="om_txt")
        else:
            default = installed.index("granite4:tiny-h") if "granite4:tiny-h" in installed else 0
            ollama_model = st.selectbox("Installed model", installed, index=default, key="om_sel") if installed else "granite4:tiny-h"
    max_alerts = st.slider("Max AI messages per run", 1, 10, 4, key="max_alerts",
                           help="Each AI message can take 30-60 s on a laptop.")


def make_backend():
    if ai_mode == "Demo model (mock)":
        return MockBackend("good")
    if ai_mode == "Local AI (Ollama)":
        return OllamaBackend(ollama_model)
        if ai_mode == "Cloud AI (Groq)":
        return OpenAICompatBackend(model="openai/gpt-oss-120b", json_mode=True)
    return None


def build_alert(case, hi_, lo_):
    be = make_backend()
    if be is None:
        return engine.template_alert(case, MODELS, hi_, lo_)
    return engine.ai_alert(be, case, MODELS, hi_, lo_)


# ------------------------------------------------------------------------------------ helpers
DEFAULT_SERIES = "realKnownCause / ambient_temperature_system_failure"


def default_index(names):
    return names.index(DEFAULT_SERIES) if DEFAULT_SERIES in names else 0


def xval(ts, t, i):
    return ts.iloc[i].to_pydatetime() if ts is not None else float(i)


def kpi(col, value, label):
    col.markdown(f'<div class="kpi"><div class="v">{value}</div><div class="l">{label}</div></div>', unsafe_allow_html=True)


def render_alert(a, sensor, t_from, t_to):
    rep = a["report"]
    verdict = str(rep.get("verdict", "unsure"))
    label, color = VERDICT.get(verdict, ("Needs a human look", AMBER))
    ft = rep.get("fault_type")
    pill = f'<span class="pill">{html.escape(str(ft).replace("_", " "))}</span>' if ft else ""
    flags = []
    if a.get("numbers_ok"):
        flags.append(f"numbers verified {a['numbers_ok']}")
    if a.get("repaired"):
        flags.append("corrected once by verifier")
    if a.get("overridden"):
        flags.append("verdict set by decision rule")
    for c in a.get("support", []) or []:
        flags.append(f"{c['claim'].replace('_', ' ')}: {c['status']}")
    st.markdown(
        f'<div class="card" style="border-left-color:{color}">'
        f'<h4><span class="badge" style="background:{color}">{label}</span>{pill}{html.escape(str(sensor))}</h4>'
        f'<div class="meta">{html.escape(str(t_from))} → {html.escape(str(t_to))} &nbsp;·&nbsp; {html.escape(a["source"])}'
        f'{" · %.0f s" % a["seconds"] if a.get("seconds") else ""}</div>'
        f'<div class="lab">What we found</div><p>{html.escape(str(rep.get("explanation", "")))}</p>'
        f'<div class="lab">Suggested next step</div><p>{html.escape(str(rep.get("next_step", "")))}</p>'
        f'<div class="chk">✔ {" · ".join(html.escape(f) for f in flags) if flags else "no extra checks recorded"}</div>'
        f'</div>', unsafe_allow_html=True)
    ev = rep.get("evidence") or []
    if ev:
        with st.expander("Evidence behind this message"):
            st.dataframe(pd.DataFrame(ev), hide_index=True, width="stretch")


def make_chart(t, v, ts, hist_end, rows, upto, shade=None, labels=None, height=440):
    n = min(upto, len(v))
    step = max(1, n // 4000)
    idx = np.arange(0, n, step)
    xs = [xval(ts, t, i) for i in idx]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.84, 0.16], vertical_spacing=0.03)
    fig.add_trace(go.Scatter(x=xs, y=v[idx], mode="lines", line=dict(color=BLUE, width=1.4), name="sensor"), row=1, col=1)

    def band(i0, i1, color, text):
        fig.add_shape(type="rect", xref="x", yref="y domain", x0=xval(ts, t, max(0, i0)), x1=xval(ts, t, min(len(v) - 1, i1)),
                      y0=0, y1=1, fillcolor=color, opacity=0.13, line_width=0, layer="below")
        fig.add_annotation(x=xval(ts, t, max(0, i0)), y=1, xref="x", yref="y domain", text=text, showarrow=False,
                           xanchor="left", yanchor="top", font=dict(size=10, color="#334155"))

    if hist_end > 0:
        band(0, min(hist_end, n - 1), GREY, "baseline (learned here)")
    if shade:
        band(shade[0], shade[1], RED, "injected fault (demo ground truth)")
    for a, b in labels or []:
        band(a, b, AMBER, "NAB-labelled event")
    if rows:
        d = pd.DataFrame(rows)
        d = d[d.end <= n]
        if len(d):
            mid = ((d.start + d.end) // 2).clip(upper=len(v) - 1)
            fig.add_trace(go.Scatter(
                x=[xval(ts, t, i) for i in mid], y=[0.5] * len(d), mode="markers",
                marker=dict(size=9, symbol="square", color=[VERDICT[x][1] for x in d.verdict]),
                text=[f"{VERDICT[a][0]} (p={p:.2f}){'  ' + str(f) if f else ''}" for a, p, f in zip(d.verdict, d.p_fault, d.fault_type)],
                hoverinfo="text", name="verdict"), row=2, col=1)
    fig.update_yaxes(visible=False, row=2, col=1, range=[0, 1])
    fig.update_layout(template="plotly_white", height=height, showlegend=False, margin=dict(l=10, r=10, t=10, b=10),
                      font=dict(family="Inter, sans-serif"))
    return fig


def kpis(container, df):
    c = container.columns(4)
    nn = max(len(df), 1)
    kpi(c[0], len(df), "windows audited")
    kpi(c[1], f"{(df.verdict == 'normal').sum() / nn:.0%}" if len(df) else "–", "healthy")
    kpi(c[2], f"{(df.verdict == 'data_fault').sum() / nn:.0%}" if len(df) else "–", "fault suspected")
    kpi(c[3], f"{(df.verdict == 'unsure').sum() / nn:.0%}" if len(df) else "–", "needs a human look")


# ------------------------------------------------------------------------------------ tabs
tab1, tab2, tab3 = st.tabs(["📈 Audit your data", "⏱️ Live replay", "ℹ️ How it works"])

# ======================================================================= TAB 1: batch audit
with tab1:
    src = st.radio("Data source", ["Upload a CSV", "Use a sample series (NAB)"], horizontal=True, key="src")
    t = v = ts = None
    name = ""
    if src == "Upload a CSV":
        up = st.file_uploader("CSV with a value column (and optionally a timestamp column)", type=["csv"], key="upload")
        if up is not None:
            try:
                t, v, ts, tcol, vcol = engine.parse_table(pd.read_csv(up))
                name = up.name
                st.caption(f"Using value column **{vcol}**" + (f" and time column **{tcol}**" if tcol else " (no time column: using sample numbers)")
                           + f" · {len(v):,} rows")
            except Exception as e:
                st.error(f"Could not read this file: {e}")
    else:
        files = engine.nab_files()
        if not files:
            st.info("No sample data found. Run `python step1_build_benchmark.py` once to download the NAB series.")
        else:
            name = st.selectbox("Sample series", list(files), index=default_index(list(files)), key="nab_pick")
            t, v, ts = engine.load_nab(files[name])
            st.caption(f"{len(v):,} rows")

    if v is not None and st.button("Run audit", type="primary", key="run_audit"):
        hist_end = int(hist_share * len(v))
        try:
            prof = engine.build_profile(t, v, hist_end)
            starts = list(range(hist_end, len(v) - W + 1, stride))
            if not starts:
                raise ValueError("Not enough data after the baseline to form even one window.")
            df = engine.audit_windows(t, v, prof, MODELS, hi, lo, starts)
            st.session_state["audit"] = dict(t=t, v=v, ts=ts, prof=prof, df=df, name=name, hist_end=hist_end,
                                             settings=dict(hi=hi, lo=lo, stride=stride, history=hist_share))
            st.session_state.pop("alert_cards", None)
        except Exception as e:
            st.error(str(e))

    A = st.session_state.get("audit")
    if A:
        df = A["df"]
        st.markdown(f"#### Results for **{A['name']}**")
        kpis(st, df)
        labels = []
        if src != "Upload a CSV" and A["ts"] is not None:
            for a, b in engine.nab_label_windows(A["name"]):
                ia = int(np.searchsorted(A["ts"].values, np.datetime64(pd.to_datetime(a))))
                ib = int(np.searchsorted(A["ts"].values, np.datetime64(pd.to_datetime(b))))
                labels.append((ia, ib))
        st.plotly_chart(make_chart(A["t"], A["v"], A["ts"], A["hist_end"], df.to_dict("records"), len(A["v"]), labels=labels),
                        width="stretch", key="batch_chart")
        st.caption("Top: the sensor signal. Bottom strip: one square per audited window (green healthy · red fault suspected · amber needs a human).")

        flagged = df[df.verdict != "normal"].reset_index(drop=True)
        st.markdown("#### Flagged windows")
        if flagged.empty:
            st.success("No window was flagged with these settings.")
        else:
            show = flagged.copy()
            show["from"] = [engine._fmt_time(A["ts"], A["t"], i) for i in show.start]
            show["to"] = [engine._fmt_time(A["ts"], A["t"], i - 1) for i in show.end]
            show["p_fault"] = show.p_fault.round(2)
            st.dataframe(show[["from", "to", "verdict", "fault_type", "p_fault", "support"]], hide_index=True, width="stretch")
            opts = [f"{r['from']}  →  {r['verdict']}" + (f" ({r['fault_type']})" if r["fault_type"] else "") for _, r in show.iterrows()]
            pick = st.selectbox("Write an engineer message for…", range(len(opts)), format_func=lambda i: opts[i], key="alert_pick")
            if st.button("Write engineer message", key="make_alert"):
                row = flagged.iloc[pick]
                with st.spinner("Writing the message… (a local AI model can take up to a minute)"):
                    try:
                        case = engine.window_case(A["t"], A["v"], int(row.start), A["prof"])
                        alert = build_alert(case, A["settings"]["hi"], A["settings"]["lo"])
                        st.session_state.setdefault("alert_cards", []).insert(
                            0, (alert, A["name"], show.iloc[pick]["from"], show.iloc[pick]["to"]))
                    except Exception as e:
                        st.error(f"Could not write the message: {e}")
            for alert, nm, a_from, a_to in st.session_state.get("alert_cards", []):
                render_alert(alert, nm, a_from, a_to)
        st.download_button("Download audit table (CSV)", df.to_csv(index=False).encode(), "signalsage_audit.csv",
                           key="dl_audit", help="Includes every window. The settings used are shown below.")
        st.caption(f"Settings used: fault ≥ {A['settings']['hi']:.2f}, healthy ≤ {A['settings']['lo']:.2f}, "
                   f"every {A['settings']['stride']} points, history {A['settings']['history']:.0%}. Changing them changes the results.")

# ======================================================================= TAB 2: live replay
with tab2:
    st.markdown("Replay a stored series **point by point**, as if it were arriving from a live sensor. "
                "Each new window is screened in milliseconds; flagged windows are handed to the message writer in the background.")
    files = engine.nab_files()
    if not files:
        st.info("No sample data found. Run `python step1_build_benchmark.py` once to download the NAB series.")
    else:
        c1, c2, c3 = st.columns(3)
        series = c1.selectbox("Series to replay", list(files), index=default_index(list(files)), key="rp_series")
        inj_kind = c2.selectbox("Inject a demo fault", ["None"] + FAULTS, index=1 + FAULTS.index("stuck_at"), key="rp_inj",
                                help="NAB data is mostly healthy, so inject a synthetic fault to watch the detector react. "
                                     "Windows fully inside the fault are flagged most reliably; windows only partly covered often "
                                     "come out as 'needs a human look'.")
        inj_pos = c2.slider("Fault position (% of replay)", 5, 90, 50, 5, key="rp_pos")
        inj_len = c3.slider("Fault length (points)", 120, 600, 320, 20, key="rp_len")
        n_windows = c3.slider("Windows to replay", 10, 200, 60, 10, key="rp_n")
        speed = c1.slider("Delay per step (seconds)", 0.0, 1.0, 0.15, 0.05, key="rp_delay")
        cooldown = c1.slider("Alert cooldown (windows)", 1, 12, 4, key="rp_cool",
                             help="Avoid writing a new message for every consecutive flagged window.")
        st.caption("Changing the message mode or the series? Press Reset first so the replay restarts with the new settings.")
        b1, b2, _ = st.columns([1, 1, 4])
        start = b1.button("▶ Start / continue", type="primary", key="rp_start")
        reset = b2.button("↺ Reset", key="rp_reset")

        if reset:
            old = st.session_state.pop("rp", None)
            if old and old.get("worker"):
                old["worker"].stop()
        sig = (series, inj_kind, inj_pos, inj_len, hist_share, stride, n_windows)
        if start and ("rp" not in st.session_state or st.session_state["rp"]["sig"] != sig):
            t0_, v0_, ts_ = engine.load_nab(files[series])
            hist_end = int(hist_share * len(v0_))
            region = len(v0_) - hist_end - W
            n_w = min(n_windows, max(1, region // stride + 1))
            inj = None
            t_, v_ = t0_, v0_
            if inj_kind != "None":
                s0 = hist_end + int(inj_pos / 100 * (n_w * stride))
                t_, v_, inj = engine.inject_into_stream(t0_, v0_, inj_kind, s0, inj_len)
            try:
                prof = engine.build_profile(t_, v_, hist_end)
                st.session_state["rp"] = dict(sig=sig, t=t_, v=v_, ts=ts_, hist_end=hist_end, prof=prof, inj=inj,
                                              ends=[hist_end + W + k * stride for k in range(n_w)], k=0, rows=[], worker=None,
                                              last_alert=-99, labels=[], name=series)
                if ts_ is not None:
                    lab = []
                    for a, b in engine.nab_label_windows(series):
                        lab.append((int(np.searchsorted(ts_.values, np.datetime64(pd.to_datetime(a)))),
                                    int(np.searchsorted(ts_.values, np.datetime64(pd.to_datetime(b))))))
                    st.session_state["rp"]["labels"] = lab
            except Exception as e:
                st.error(str(e))

        RP = st.session_state.get("rp")
        kpi_box, chart_box, feed_title = st.container(), st.empty(), st.container()
        feed_box = st.container()
        if RP:
            def draw(upto):
                df_ = pd.DataFrame(RP["rows"]) if RP["rows"] else pd.DataFrame(columns=["start", "end", "p_fault", "verdict", "fault_type"])
                with kpi_box:
                    kpis(st, df_)
                chart_box.plotly_chart(make_chart(RP["t"], RP["v"], RP["ts"], RP["hist_end"], RP["rows"], upto,
                                                  shade=RP["inj"] if RP["inj"] and RP["inj"][0] < upto else None,
                                                  labels=[x for x in RP["labels"] if x[0] < upto]),
                                       width="stretch", key=f"rp_chart_{len(RP['rows'])}_{time.time_ns()}")

            def draw_feed():
                wk = RP["worker"]
                if wk is None:
                    return
                with feed_box:
                    if wk.pending:
                        st.info(f"✍️ {wk.pending} message(s) being written…")
                    for e in wk.errors[-2:]:
                        st.warning(f"Message writer problem: {e}")
                    for a in sorted(wk.done, key=lambda x: -x["job"]["end"]):
                        m = a["job"]
                        render_alert(a, RP["name"], engine._fmt_time(RP["ts"], RP["t"], m["start"]),
                                     engine._fmt_time(RP["ts"], RP["t"], m["end"] - 1))

            if start:
                if RP["worker"] is None and ai_mode != "Screening only":
                    be_hi, be_lo = hi, lo
                    RP["worker"] = engine.AlertWorker(lambda job: build_alert(job["case"], be_hi, be_lo))
                n_alerts = 0
                while RP["k"] < len(RP["ends"]):
                    end = RP["ends"][RP["k"]]
                    df1 = engine.audit_windows(RP["t"], RP["v"], RP["prof"], MODELS, hi, lo, [end - W])
                    row = df1.iloc[0].to_dict()
                    RP["rows"].append(row)
                    if (RP["worker"] and row["verdict"] != "normal" and RP["k"] - RP["last_alert"] >= cooldown
                            and (len(RP["worker"].done) + RP["worker"].pending) < max_alerts):
                        RP["last_alert"] = RP["k"]
                        RP["worker"].submit({"case": engine.window_case(RP["t"], RP["v"], end - W, RP["prof"]),
                                             "meta": {"start": end - W, "end": end}})
                    RP["k"] += 1
                    draw(end)
                    time.sleep(speed)
                st.success("Replay finished.")
                if RP["worker"]:
                    waited = 0
                    while RP["worker"].pending and waited < 180:
                        time.sleep(1)
                        waited += 1
                    draw_feed()
                    if RP["worker"].pending:
                        st.info("Some messages are still being written. Press Start again to refresh.")
            else:
                if RP["rows"]:
                    draw(RP["ends"][max(RP["k"] - 1, 0)])
                draw_feed()
        else:
            st.caption("Press **Start** to begin the replay.")

# ======================================================================= TAB 3: about
with tab3:
    st.markdown("""
**What happens to your data**
1. **Baseline:** the first part of your series is used to learn what is *normal for this sensor*.
2. **Screening (milliseconds per window):** deterministic checks (gaps, frozen values, spikes, drift, noise, clipping, timestamps) plus a trained classifier give a fault probability.
3. **Decision rule:** high probability → *fault suspected*; low → *healthy*; in between → *needs a human look*. You set where those lines are in the sidebar.
4. **Engineer message:** flagged windows get a plain-language message. Numbers in it are re-computed and checked, the verdict is forced to follow the decision rule, and fault claims are checked against the measured evidence where a reliable rule exists.

**What you control, and how it changes the results**
- *Strictness:* stricter = fewer windows get a verdict but those verdicts are more often right; more windows go to humans.
- *Audit every N points:* smaller = finer timing and more alerts.
- *History share:* a longer baseline gives a steadier idea of "normal", but less data left to audit.
- *Window length is fixed* at 256 points because the classifier was trained on that length.

**Limits to keep in mind**
- The classifier was trained and tested on NAB-based series with *synthetic* faults; accuracy on your sensors is not guaranteed.
- It spots **data faults**. It cannot reliably tell a genuine process event from normal behaviour.
- Free-text explanations are only partly verifiable; three of eight fault types have a reliable evidence rule.
- A history window containing many faults will teach the system a distorted idea of "normal".
""")
