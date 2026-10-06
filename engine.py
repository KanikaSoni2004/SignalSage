"""
engine.py
---------
All non-UI logic used by the web app (app.py): reading a user's table, learning a sensor's
personal baseline, auditing windows, building engineer alerts (AI or template) and a background
worker that writes alerts while a live replay is running.
"""
import glob
import json
import os
import queue
import threading
import time

import joblib
import numpy as np
import pandas as pd
import requests

import agent
from agent import ToolBox, run_tools_pair, verify_report
from signalsage_core import FAULTS, history_profile, inject_fault, window_features, z_features
from support import check_support, load_rules

W = 256                      # window length the classifier was trained on (fixed)
MIN_HISTORY = W + 60         # minimum points needed to learn a baseline

NEXT_STEPS = {
    "stuck_at": "Check the sensor wiring, connector and power; compare with a neighbouring sensor. "
                "If the value stays frozen after a restart, replace the sensor.",
    "spike": "Inspect cabling, shielding and grounding for intermittent contacts or interference; "
             "check whether the spikes line up with equipment switching.",
    "drift": "Schedule a recalibration against a reference and check for temperature or ageing effects.",
    "noise_burst": "Look for nearby electrical noise sources, loose grounding or an unstable supply voltage.",
    "dropout_gap": "Check the data link, logger and sensor power during the gap; backfill the data if possible.",
    "clipping": "The reading is hitting the end of its range: verify the range setting or use a wider-range sensor.",
    "quantization": "Resolution looks too coarse: check ADC settings and any compression applied upstream.",
    "timestamp_jitter": "Check logger clock sync and network delays; resample onto a regular time grid.",
}
FAULT_LABEL = {
    "stuck_at": "frozen (stuck) readings", "spike": "isolated spikes", "drift": "slow drift",
    "noise_burst": "a burst of noise", "dropout_gap": "missing data", "clipping": "clipped (saturated) readings",
    "quantization": "coarse value steps", "timestamp_jitter": "irregular timestamps",
}


# ----------------------------------------------------------------------------- data in
def parse_table(df):
    """Find the time column and the value column in an uploaded table."""
    cols = list(df.columns)
    tcol = next((c for c in cols if str(c).lower() in ("timestamp", "time", "datetime", "date", "ts")), None)
    if tcol is None:
        for c in cols:
            if df[c].dtype == object:
                if pd.to_datetime(df[c], errors="coerce").notna().mean() > 0.9:
                    tcol = c
                    break
    vcol = next((c for c in cols if c != tcol and str(c).lower() in ("value", "reading", "sensor", "y")), None)
    if vcol is None:
        vcol = next((c for c in cols if c != tcol and pd.api.types.is_numeric_dtype(df[c])), None)
    if vcol is None:
        raise ValueError("No numeric column found. The file needs a column of sensor values.")
    v = pd.to_numeric(df[vcol], errors="coerce").to_numpy(float)
    ts = None
    t = np.arange(len(v), dtype=float)
    if tcol is not None:
        ts = pd.to_datetime(df[tcol], errors="coerce")
        if ts.notna().all():
            t = (ts - ts.iloc[0]).dt.total_seconds().to_numpy(float)
        else:
            ts = None
    return t, v, ts, tcol, vcol


def nab_files(cache="nab_data"):
    """Series that step1 already downloaded -> {display name: path}."""
    out = {}
    for f in sorted(glob.glob(os.path.join(cache, "data__*.csv"))):
        out[os.path.basename(f)[6:-4].replace("__", " / ")] = f
    return out


def load_nab(path):
    df = pd.read_csv(path)
    t, v, ts, _, _ = parse_table(df)
    return t, v, ts


def nab_label_windows(display_name, cache="nab_data"):
    """NAB's human-labelled anomaly windows (only used to SHADE the demo chart, never by the model)."""
    f = os.path.join(cache, "labels__combined_windows.json")
    if not os.path.exists(f):
        return []
    key = display_name.replace(" / ", "/") + ".csv"
    return json.load(open(f)).get(key, [])


# ----------------------------------------------------------------------------- baseline + audit
def load_models(path="models.joblib"):
    return joblib.load(path)


def build_profile(t, v, hist_end, stride=15, trim=0.10):
    """Learn what is NORMAL for this sensor from its history (assumed mostly healthy)."""
    starts = list(range(0, hist_end - W + 1, stride))
    if len(starts) < 5:
        raise ValueError(f"Need at least {MIN_HISTORY} history points to learn a baseline "
                         f"(history has {hist_end}). Add more data or raise the history share.")
    rows = [window_features(v[s:s + W], t[s:s + W]) for s in starts]
    if len(rows) >= 10 and trim > 0:                          # drop the most extreme history windows
        keys = list(rows[0].keys())
        M = np.array([[r[k] for k in keys] for r in rows])
        med = np.median(M, 0)
        mad = 1.4826 * np.median(np.abs(M - med), 0) + 1e-9
        score = np.clip(np.abs(M - med) / mad, 0, 20).mean(1)
        keep = np.argsort(score)[: int(len(rows) * (1 - trim))]
        rows = [rows[i] for i in sorted(keep)]
    return history_profile(rows)


def verdict_from_p(p, hi, lo):
    return "data_fault" if p >= hi else "normal" if p <= lo else "unsure"


def window_case(t, v, i0, prof):
    seg_v, seg_t = v[i0:i0 + W], t[i0:i0 + W]
    return {"v": [None if not np.isfinite(x) else float(x) for x in seg_v],
            "t": [float(x) for x in seg_t], "profile": prof}


def audit_windows(t, v, prof, models, hi, lo, starts):
    """Fast, no-LLM screening of many windows at once. Returns a DataFrame."""
    feats = []
    for i0 in starts:
        f = window_features(v[i0:i0 + W], t[i0:i0 + W])
        f.update(z_features(f, prof))
        feats.append(f)
    if not feats:
        return pd.DataFrame(columns=["start", "end", "p_fault", "verdict", "fault_type", "support"])
    X = pd.DataFrame(feats).reindex(columns=models["columns"]).fillna(0.0)
    p = models["detector"].predict_proba(X)[:, 1]
    types = models["typer"].predict(X)
    rules = load_rules()
    rows = []
    for i0, pf, ft, f in zip(starts, p, types, feats):
        verdict = verdict_from_p(pf, hi, lo)
        sup = None
        if verdict == "data_fault":
            c = check_support({"fault_type": ft, "explanation": ""}, f, rules)
            sup = c[0]["status"] if c else None
        rows.append({"start": int(i0), "end": int(i0 + W), "p_fault": float(pf), "verdict": verdict,
                     "fault_type": ft if verdict == "data_fault" else None, "support": sup})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- engineer alerts
def _fmt_time(ts, t, i):
    return str(ts.iloc[i]) if ts is not None else f"sample {i}"


def template_alert(case, models, hi, lo):
    """Deterministic engineer message (NO AI) built only from measured evidence."""
    agent.set_thresholds(hi, lo)
    tb = ToolBox(case, models)
    ml = tb.call("ml_fault_probability")
    p = ml["fault_probability"]
    verdict = verdict_from_p(p, hi, lo)
    ft = ml["predicted_fault_type"] if verdict == "data_fault" else None
    ev = [{"tool": "ml_fault_probability", "field": "fault_probability", "value": p}]
    if ft:
        spec = {"stuck_at": ("check_stuck", "longest_flat_run"), "spike": ("check_spikes", "n_spikes"),
                "drift": ("check_drift", "drift_ratio"), "noise_burst": ("check_noise", "noise_burst_ratio"),
                "dropout_gap": ("check_missing", "longest_missing_run"), "clipping": ("check_clipping", "clip_fraction"),
                "quantization": ("check_quantization", "unique_fraction"),
                "timestamp_jitter": ("check_timestamps", "dt_irregular_fraction")}[ft]
        ev.append({"tool": spec[0], "field": spec[1], "value": tb.call(spec[0])[spec[1]]})
        expl = (f"The classifier is {p:.0%} confident that this window contains {FAULT_LABEL[ft]}. "
                f"The measured {spec[1].replace('_', ' ')} is {ev[1]['value']}, which is unusual for this sensor.")
        nxt = NEXT_STEPS[ft]
    elif verdict == "unsure":
        expl = (f"The classifier gives a {p:.0%} chance of a data fault, which is too uncertain to call. "
                "This could be a mild data problem or a genuine change in the process.")
        nxt = "Have an engineer look at the highlighted window and compare it with related sensors."
    else:
        expl = f"No sign of a data fault: the fault probability is only {p:.0%} and the checks look normal for this sensor."
        nxt = "No action needed."
    report = {"verdict": verdict, "fault_type": ft, "confidence": round(abs(p - 0.5) * 2, 3),
              "evidence": ev, "explanation": expl, "next_step": nxt}
    n, ok, _ = verify_report(report, tb)
    sup = check_support(report, tb._features())
    return {"report": report, "source": "template (no AI)", "seconds": 0.0,
            "numbers_ok": f"{ok}/{n}", "repaired": False, "overridden": False, "llm_agreed": None,
            "support": [{"claim": c["claim"], "source": c["source"], "status": c["status"]} for c in sup]}


def ai_alert(backend, case, models, hi, lo):
    """LLM-written engineer message, with numbers verified and decision rule enforced."""
    agent.set_thresholds(hi, lo)
    t0 = time.time()
    _, res = run_tools_pair(backend, case, models)
    rep = res["report"]
    return {"report": rep, "source": f"AI: {backend.name}", "seconds": time.time() - t0,
            "numbers_ok": f"{res['claims_ok']}/{res['claims_total']}",
            "repaired": res.get("repaired", False), "overridden": res.get("overridden", False),
            "llm_agreed": res.get("llm_agreed"), "support": rep.get("support", [])}


def ollama_models(host="http://localhost:11434", timeout=3):
    """Names of locally installed Ollama models, or None if Ollama is not reachable."""
    try:
        r = requests.get(host.rstrip("/") + "/api/tags", timeout=timeout)
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return None


class AlertWorker:
    """Writes alerts in the background so the live chart never freezes while the LLM thinks."""

    def __init__(self, make_alert):
        self.make_alert = make_alert          # callable(job) -> alert dict
        self.q = queue.Queue()
        self.done = []
        self.errors = []
        self.pending = 0
        self._lock = threading.Lock()
        self._stop = False
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def submit(self, job):
        with self._lock:
            self.pending += 1
        self.q.put(job)

    def _run(self):
        while not self._stop:
            try:
                job = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                res = self.make_alert(job)
                res["job"] = job["meta"]
                self.done.append(res)
            except Exception as e:                                    # never crash the app
                self.errors.append(f"{type(e).__name__}: {e}")
            finally:
                with self._lock:
                    self.pending -= 1

    def stop(self):
        self._stop = True


def inject_into_stream(t, v, kind, start, length, seed=3):
    """Demo helper: corrupt part of a replayed stream with a synthetic fault."""
    v2, t2 = v.copy(), t.copy()
    end = min(len(v), start + length)
    rng = np.random.default_rng(seed)
    vv, tt, meta = inject_fault(v2[start:end], t2[start:end], kind, rng, severity=0.9)
    v2[start:end], t2[start:end] = vv, tt
    return t2, v2, (start, end)
