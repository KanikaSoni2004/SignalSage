"""
support.py
----------
EXPLANATION SUPPORT CHECK (the symbolic layer).

Idea: a fault claim in a report -- either the `fault_type` field or a condition mentioned in the
free-text explanation ("flat readings", "spikes", ...) -- is only SUPPORTED if an independent
rule on the window's measured features says that condition is really present.

Each fault type has a score (a function of the tool features). A claim is SUPPORTED when
score >= threshold; thresholds are LEARNED from DEV windows only (step5): the 95th percentile of
that score on normal dev windows ("more extreme than 95% of normal windows").

Not every rule is equally trustworthy, so step5 also measures each rule's recall with grouped
cross-validation over DEV series and marks it "reliable" only if it fires on >=80% of real faults
of its type. A failed claim is then
    unsupported  -> the rule is reliable and does NOT fire: the claim is contradicted by the data
    unverifiable -> the rule is weak, so we cannot confirm or refute the claim
The check runs on the data itself, never on ground-truth labels.
"""
import json
import os
import re

SCORES = {
    "stuck_at":         lambda f: max(f["z_longest_flat_run"], f["z_flat_fraction"]),
    "spike":            lambda f: max(f["z_n_spikes"], f["z_max_spike_z"]),
    "drift":            lambda f: f["z_drift_ratio"],
    "noise_burst":      lambda f: f["z_noise_burst_ratio"],
    "dropout_gap":      lambda f: 50.0 if f["missing_fraction"] > 0 else 0.0,   # hard symbolic rule
    "clipping":         lambda f: max(f["z_clip_fraction"], f["z_near_max_fraction"], f["z_near_min_fraction"]),
    "quantization":     lambda f: -f["z_unique_fraction"],
    "timestamp_jitter": lambda f: max(f["z_dt_irregular_fraction"], f["z_dt_jitter"], f["z_duplicate_ts"]),
}
FLOOR = {k: 3.0 for k in SCORES}
FLOOR["dropout_gap"] = 0.5

LEXICON = {
    "stuck_at": ["stuck", "flat", "frozen", "flatline", "constant value", "unchanging"],
    "spike": ["spike", "outlier", "impulse"],
    "drift": ["drift", "trend", "ramp", "gradual"],
    "noise_burst": ["noise", "noisy", "burst"],
    "dropout_gap": ["missing", "gap", "dropout", "drop-out"],
    "clipping": ["clip", "saturat", "ceiling"],
    "quantization": ["quantiz", "coarse", "discrete step", "low resolution", "stair"],
    "timestamp_jitter": ["timestamp", "time stamp", "sampling interval", "irregular sampl", "duplicate time", "jitter"],
}
NEGATIONS = ["no ", "not ", "without", "absence", "free of", "n't", "neither", "nor ",
             "within normal", "within the normal", "within its normal", "typical", "usual", "no evidence"]

RULES_FILE = "support_rules.json"
_rules = None


def load_rules(path=RULES_FILE):
    global _rules
    if _rules is None:
        _rules = json.load(open(path)) if os.path.exists(path) else {}
    return _rules


def score(kind, feats):
    return float(SCORES[kind](feats))


def mentioned_conditions(text):
    """Fault conditions the explanation asserts (simple keyword match, skipping negated mentions)."""
    found = set()
    low = str(text or "").lower()
    for kind, words in LEXICON.items():
        for w in words:
            for m in re.finditer(re.escape(w), low):
                before = low[max(0, m.start() - 28): m.start()]
                if any(n in before for n in NEGATIONS):
                    continue
                found.add(kind)
    return found


def check_support(report, feats, rules=None):
    """
    Returns a list of {"claim", "source": "fault_type"|"explanation", "score", "threshold",
                       "status": "supported"|"unsupported"|"unverifiable"}.
    Empty list if no rules are loaded (check disabled).
    """
    rules = load_rules() if rules is None else rules
    if not rules:
        return []
    thr, reliable = rules["thresholds"], set(rules.get("reliable", []))
    claims = []
    ft = report.get("fault_type") if isinstance(report, dict) else None
    if isinstance(ft, str) and ft and ft.lower() not in ("none", "null"):
        claims.append((ft.lower().strip(), "fault_type"))
    for kind in sorted(mentioned_conditions(report.get("explanation") if isinstance(report, dict) else "")):
        if (kind, "fault_type") not in claims:
            claims.append((kind, "explanation"))
    out = []
    for kind, src in claims:
        if kind not in SCORES or kind not in thr:
            out.append({"claim": kind, "source": src, "score": None, "threshold": None, "status": "unsupported"})
            continue
        sc = score(kind, feats)
        status = "supported" if sc >= thr[kind] else ("unsupported" if kind in reliable else "unverifiable")
        out.append({"claim": kind, "source": src, "score": round(sc, 3),
                    "threshold": round(thr[kind], 3), "status": status})
    return out
