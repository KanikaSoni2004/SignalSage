"""
agent.py
--------
SignalSage agent = LLM + deterministic tools + numeric verifier.

Protocol (works with any chat model, no native function-calling needed). Every model turn
must be ONE JSON object:
  {"action": "call_tool", "tools": ["<name>", "<name>", ...]}      (several tools per turn = faster)
  {"action": "final", "report": {...}}

Configurations used in the evaluation
  llm_only        the model sees a down-sampled copy of the window, no tools
  tools           model + tools (numbers in its report are measured but NOT corrected)
  tools_verified  model + tools + verifier that re-computes every cited number and asks for
                  one correction if something does not match.
`tools` and `tools_verified` share ONE model trajectory (run_tools_pair): the verified result is
the same conversation plus one optional correction call, which halves the number of LLM calls.
"""
import json
import time

import numpy as np

from llm import extract_json
from signalsage_core import FAULTS, TOOLS, window_features, z_features
from support import check_support

VERDICTS = ["normal", "data_fault", "unsure"]
P_HI, P_LO = 0.75, 0.25          # decision playbook thresholds on the ML fault probability


def playbook_verdict(p):
    """What the playbook says for a given fault probability (used to prompt AND to score)."""
    return "data_fault" if p >= P_HI else "normal" if p <= P_LO else "unsure"

TOOL_DOCS = {
    "check_missing": ("share and longest run of missing samples", ["missing_fraction", "longest_missing_run"]),
    "check_timestamps": ("sampling-time regularity", ["dt_median_s", "dt_jitter", "duplicate_ts", "non_monotonic", "big_gap_count", "dt_irregular_fraction"]),
    "check_stuck": ("flat / frozen readings", ["longest_flat_run", "flat_fraction"]),
    "check_clipping": ("saturation at min or max", ["clip_fraction"]),
    "check_spikes": ("isolated outliers", ["n_spikes", "max_spike_z"]),
    "check_drift": ("slow trend across the window", ["drift_ratio", "drift_r2"]),
    "check_noise": ("noise level and noise bursts", ["noise_est", "noise_burst_ratio"]),
    "check_quantization": ("coarse value steps", ["unique_fraction", "levels_per_sigma"]),
    "check_level_shift": ("sudden change of level", ["level_shift_score", "level_shift_pos"]),
    "check_distribution": ("value distribution shape", ["top_bin_fraction", "near_max_fraction", "near_min_fraction", "kurtosis", "lag1_autocorr", "max_jump_z"]),
    "spectral_summary": ("frequency content", ["fs_hz", "dominant_freq_hz", "spectral_entropy", "peak_power_ratio"]),
    "compare_to_history": ("the 8 features most unusual versus this sensor's own history, as z_<feature> values (standard deviations from its usual)", ["z_<feature> ..."]),
    "ml_fault_probability": ("trained classifier: probability that the DATA is faulty and the most likely fault type", ["fault_probability", "predicted_fault_type"]),
}


def _tool_list_text():
    return "\n".join(f"- {n}: {d}. fields: {', '.join(f)}" for n, (d, f) in TOOL_DOCS.items())


def _system_tools(hi, lo):
    return f"""You are SignalSage. You audit ONE window of sensor data and decide whether the DATA can be trusted.
Verdicts: "normal" (data looks trustworthy), "data_fault" (sensor/data problem), "unsure" (borderline - send to a human; it may be a mild fault or a real event).
Fault types: {', '.join(FAULTS)}.

Tools (no arguments; they act on the current window):
{_tool_list_text()}

Reply with exactly ONE JSON object, nothing else.
Step 1 - call several tools at once, always including ml_fault_probability and compare_to_history:
  {{"action":"call_tool","tools":["ml_fault_probability","compare_to_history","check_stuck","check_spikes"]}}
Step 2 - after reading the results, give your final answer:
  {{"action":"final","report":{{"verdict":"...","fault_type":"... or null","confidence":0.0,"evidence":[{{"tool":"<tool>","field":"<field>","value":<copied from tool result>}}],"explanation":"2-4 plain sentences","next_step":"one short recommendation"}}}}
DECISION PLAYBOOK - follow it. Let p = fault_probability from ml_fault_probability.
  p >= {hi}  -> verdict "data_fault", fault_type = predicted_fault_type
  p <= {lo} -> verdict "normal", fault_type null
  otherwise  -> verdict "unsure", fault_type null, next_step = human review
Use the other tools as supporting evidence and to explain which checks look unusual.
Rules: evidence values must be copied exactly from tool results, never invented. Give 2-4 evidence items, and always include fault_probability as one of them."""


SYSTEM_TOOLS = _system_tools(P_HI, P_LO)


def set_thresholds(hi, lo):
    """Let the user change the decision playbook (also rewrites the prompt the LLM sees)."""
    global P_HI, P_LO, SYSTEM_TOOLS
    P_HI, P_LO = float(hi), float(lo)
    SYSTEM_TOOLS = _system_tools(P_HI, P_LO)


SYSTEM_NO_TOOLS = f"""NO_TOOLS. You are SignalSage, an assistant that audits ONE window of sensor data and decides whether the DATA can be trusted.
Possible verdicts: "normal", "data_fault", "unsure" (borderline - needs a human). Possible fault types: {', '.join(FAULTS)}.
You only see a down-sampled copy of the values. Reply with ONE JSON object and nothing else:
{{"action":"final","report":{{"verdict":"...","fault_type":"... or null","confidence":0.0,"evidence":[],"explanation":"2-4 plain sentences","next_step":"one short recommendation"}}}}"""


def _r(x):
    if isinstance(x, (float, np.floating)):
        return round(float(x), 4)
    if isinstance(x, (int, np.integer)):
        return int(x)
    return x


class ToolBox:
    """Runs tools on one benchmark case. Results are cached and rounded to 4 decimals."""

    def __init__(self, case, models=None):
        self.v = np.array([np.nan if x is None else x for x in case["v"]], dtype=float)
        self.t = np.array(case["t"], dtype=float)
        self.profile = case["profile"]
        self.models = models
        self._cache = {}
        self._feat = None

    def _features(self):
        if self._feat is None:
            f = window_features(self.v, self.t)
            f.update(z_features(f, self.profile))
            self._feat = f
        return self._feat

    def call(self, name):
        if name in self._cache:
            return self._cache[name]
        if name in TOOLS:
            out = {k: v for k, v in TOOLS[name](self.v, self.t).items() if k != "spike_indices"}
        elif name == "compare_to_history":
            z = {k: v for k, v in self._features().items() if k.startswith("z_")}
            top = sorted(z, key=lambda k: -abs(z[k]))[:8]
            out = {k: z[k] for k in top}
        elif name == "ml_fault_probability":
            if self.models is None:
                raise ValueError("no trained models loaded")
            f = self._features()
            import pandas as pd
            X = pd.DataFrame([f]).reindex(columns=self.models["columns"]).fillna(0.0)
            out = {"fault_probability": float(self.models["detector"].predict_proba(X)[0, 1]),
                   "predicted_fault_type": str(self.models["typer"].predict(X)[0])}
        else:
            raise KeyError(name)
        out = {k: _r(v) for k, v in out.items()}
        self._cache[name] = out
        return out


def verify_report(report, tb):
    """Re-compute every cited number. Returns (n_claims, n_ok, mismatches)."""
    ev = report.get("evidence") if isinstance(report, dict) else None
    ev = ev if isinstance(ev, list) else []
    bad, ok = [], 0
    for e in ev:
        try:
            tool, field, val = e["tool"], e["field"], e["value"]
            truth = tb.call(tool)[field]
        except Exception:
            bad.append({"claim": e, "problem": "unknown tool/field or malformed evidence"})
            continue
        try:
            if isinstance(truth, str) or isinstance(val, str):
                good = str(truth) == str(val)
            else:
                good = abs(float(val) - float(truth)) <= max(1e-3, 0.02 * abs(float(truth)))
        except Exception:
            good = False
        if good:
            ok += 1
        else:
            bad.append({"claim": e, "problem": f"tool returned {truth}"})
    return len(ev), ok, bad


def _llm_only_view(case, n=64):
    v = case["v"]
    idx = np.linspace(0, len(v) - 1, min(n, len(v))).astype(int)
    t = np.array(case["t"], dtype=float)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
    vals = [None if v[i] is None else round(float(v[i]), 3) for i in idx]
    return (f"Window of {len(v)} samples, median sampling interval {dt:.1f} s. "
            f"Down-sampled values (null = missing): {json.dumps(vals)}")


def _tool_turn(obj, tb, messages, stats):
    """Execute one call_tool turn (single `tool` or a list `tools`)."""
    names = obj.get("tools") or ([obj.get("tool")] if obj.get("tool") else [])
    names = [n for n in names if isinstance(n, str)][:8]
    results = {}
    for n in names:
        try:
            results[n] = tb.call(n)
        except Exception:
            stats["bad_tool_calls"] += 1
    if results:
        messages.append({"role": "user", "content": "TOOL_RESULTS " + json.dumps(results)})
    else:
        stats["bad_tool_calls"] += 1
        messages.append({"role": "user", "content": f"No valid tool requested. Valid tools: {', '.join(TOOL_DOCS)}"})


def _support_counts(sup):
    return {"support_supported": sum(c["status"] == "supported" for c in sup),
            "support_unsupported": sum(c["status"] == "unsupported" for c in sup),
            "support_unverifiable": sum(c["status"] == "unverifiable" for c in sup)}


def run_llm_only(backend, case):
    t0 = time.time()
    stats = {"steps": 1, "invalid_json": 0, "bad_tool_calls": 0, "repaired": False}
    messages = [{"role": "system", "content": SYSTEM_NO_TOOLS},
                {"role": "user", "content": _llm_only_view(case)}]
    obj = extract_json(backend.chat(messages))
    report = (obj or {}).get("report", obj) if isinstance(obj, dict) else None
    if not isinstance(report, dict) or "verdict" not in report:
        stats["invalid_json"] += 1
        report = {"verdict": "unsure", "fault_type": None, "evidence": []}
    return {"report": report, "claims_total": 0, "claims_ok": 0, "first_pass_ok": None,
            "seconds": time.time() - t0, **stats}


def run_tools_pair(backend, case, models, max_steps=5):
    """One model trajectory -> two results: (tools, tools_verified)."""
    t0 = time.time()
    tb = ToolBox(case, models)
    stats = {"steps": 0, "invalid_json": 0, "bad_tool_calls": 0, "repaired": False}
    messages = [{"role": "system", "content": SYSTEM_TOOLS},
                {"role": "user", "content": f"Audit the current window ({len(case['v'])} samples). "
                                            "Start by calling tools."}]
    first, last_text = None, ""
    for _ in range(max_steps):
        text = backend.chat(messages)
        stats["steps"] += 1
        obj = extract_json(text)
        messages.append({"role": "assistant", "content": text})
        if not isinstance(obj, dict):
            stats["invalid_json"] += 1
            messages.append({"role": "user", "content": "Invalid reply. Respond with ONE JSON object only."})
            continue
        if obj.get("action") == "call_tool" or (obj.get("action") is None and (obj.get("tools") or obj.get("tool"))):
            _tool_turn(obj, tb, messages, stats)
            continue
        if obj.get("action") == "final" or "verdict" in obj or isinstance(obj.get("report"), dict):
            first, last_text = obj.get("report", obj), text
            break
        stats["invalid_json"] += 1
        messages.append({"role": "user", "content": "Reply must have action 'call_tool' or 'final'."})
    if not isinstance(first, dict):
        first = {"verdict": "unsure", "fault_type": None, "evidence": [], "explanation": "no final report"}

    n1, ok1, bad1 = verify_report(first, tb)
    ml = tb.call("ml_fault_probability") if models else None
    pb = playbook_verdict(ml["fault_probability"]) if ml else None
    norm = lambda r: str(r.get("verdict", "")).lower().strip()
    agreed = (norm(first) == pb) if pb else None
    if pb and not agreed:                                        # policy check joins the numeric check
        bad1 = bad1 + [{"claim": {"verdict": first.get("verdict")},
                        "problem": f"decision playbook requires verdict '{pb}' because "
                                   f"fault_probability={ml['fault_probability']}"}]
    feats = tb._features()
    sup1 = check_support(first, feats)                           # explanation-support check
    for c in sup1:
        if c["status"] == "unsupported":
            bad1 = bad1 + [{"claim": {"mentions": c["claim"], "source": c["source"]},
                            "problem": f"measured evidence does not support this (score {c['score']} "
                                       f"is below threshold {c['threshold']}); remove or correct it"}]
    base = {"claims_total": n1, "claims_ok": ok1, "first_pass_ok": (ok1 / n1) if n1 else 1.0,
            **_support_counts(sup1),
            "playbook": pb, "llm_agreed": agreed, "overridden": False,
            "raw": [m["content"][:600] for m in messages if m["role"] == "assistant"]}
    res_tools = {"report": json.loads(json.dumps(first)), **base,
                 "seconds": time.time() - t0, **stats}

    second, stats2 = json.loads(json.dumps(first)), dict(stats)
    if bad1:                                                     # verifier: ask for ONE correction
        stats2["repaired"] = True
        messages.append({"role": "user", "content":
                         "VERIFIER: these claims do not match the tools: "
                         f"{json.dumps(bad1)}. Reply with a corrected final report using only "
                         "values returned by the tools."})
        text = backend.chat(messages)
        stats2["steps"] += 1
        obj = extract_json(text)
        if isinstance(obj, dict) and ("verdict" in obj or "report" in obj):
            second = obj.get("report", obj)
        else:
            stats2["invalid_json"] += 1
        n2, ok2, bad2 = verify_report(second, tb)
        if bad2:                                                 # drop claims that still fail
            drop = [b["claim"] for b in bad2]
            second["evidence"] = [e for e in second.get("evidence", []) if e not in drop]
            second["verification"] = "partially failed"
        n2, ok2, _ = verify_report(second, tb)
    else:
        n2, ok2 = n1, ok1
    overridden = False
    if pb and norm(second) != pb:                                # still disobeys the playbook -> override
        second["verdict"] = pb
        second["fault_type"] = ml["predicted_fault_type"] if pb == "data_fault" else None
        second["verification"] = "verdict overridden by decision playbook"
        overridden = True
    sup2 = check_support(second, feats)
    second["support"] = [{"claim": c["claim"], "source": c["source"], "status": c["status"]} for c in sup2]
    if any(c["status"] == "unsupported" for c in sup2):
        second["unsupported_claims"] = [c["claim"] for c in sup2 if c["status"] == "unsupported"]
        second["verification"] = (second.get("verification", "") + " | unsupported claims flagged").strip(" |")
    res_verified = {"report": second, "claims_total": n2, "claims_ok": ok2, **_support_counts(sup2),
                    "first_pass_ok": base["first_pass_ok"], "playbook": pb, "llm_agreed": agreed,
                    "overridden": overridden, "raw": base["raw"],
                    "seconds": time.time() - t0, **stats2}
    return res_tools, res_verified