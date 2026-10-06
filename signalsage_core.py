"""
signalsage_core.py
------------------
Deterministic diagnostic TOOLS and FAULT INJECTION for SignalSage.

Conventions
  v : 1-D numpy array of sensor values (NaN = missing)
  t : 1-D numpy array of timestamps in seconds (same length as v)

Every tool returns a small JSON-serialisable dict, so the same functions can be
(a) used by the rule/tree baseline and (b) exposed later to the LLM agent as tools.
"""
import numpy as np
from scipy import signal as sps
from scipy.ndimage import median_filter

# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _fill(v):
    """Linearly interpolate NaNs so filters/FFT work."""
    v = np.asarray(v, dtype=float)
    bad = ~np.isfinite(v)
    if bad.all():
        return np.zeros_like(v)
    if bad.any():
        idx = np.arange(len(v))
        v = v.copy()
        v[bad] = np.interp(idx[bad], idx[~bad], v[~bad])
    return v


def _scale(x):
    """Robust scale (MAD-based sigma); falls back to std, never returns 0."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return 1.0
    s = 1.4826 * np.median(np.abs(x - np.median(x)))
    if s < 1e-12:
        s = float(np.std(x))
    return max(float(s), 1e-9)


def _longest_run(mask):
    best = run = 0
    for b in mask:
        run = run + 1 if b else 0
        if run > best:
            best = run
    return int(best)


# ----------------------------------------------------------------------------
# diagnostic tools
# ----------------------------------------------------------------------------

def check_missing(v, t=None):
    nan = ~np.isfinite(v)
    return {"missing_fraction": float(nan.mean()),
            "longest_missing_run": _longest_run(nan)}


def check_timestamps(v, t):
    dt = np.diff(t)
    if len(dt) == 0:
        return {"dt_median_s": 0.0, "dt_jitter": 0.0, "duplicate_ts": 0,
                "non_monotonic": 0, "big_gap_count": 0, "dt_irregular_fraction": 0.0}
    med = float(np.median(dt))
    med = med if med > 0 else 1.0
    return {"dt_median_s": med,
            "dt_jitter": float(np.std(dt) / med),
            "duplicate_ts": int(np.sum(dt == 0)),
            "non_monotonic": int(np.sum(dt < 0)),
            "big_gap_count": int(np.sum(dt > 3 * med)),
            "dt_irregular_fraction": float(np.mean(np.abs(dt - med) > 0.05 * med))}


def check_stuck(v, t=None):
    d = np.diff(v)
    same = np.isfinite(d) & (d == 0)
    run = _longest_run(same)
    return {"longest_flat_run": run + 1 if run else 0,
            "flat_fraction": float((run + 1 if run else 0) / max(len(v), 1))}


def check_clipping(v, t=None):
    x = v[np.isfinite(v)]
    if len(x) < 3:
        return {"clip_fraction": 0.0}
    n_max = int(np.sum(x == x.max()))
    n_min = int(np.sum(x == x.min()))
    return {"clip_fraction": float(max(n_max, n_min) / len(x))}


def check_spikes(v, t=None, k=6.0, win=11):
    x = _fill(v)
    res = x - median_filter(x, size=win, mode="nearest")
    sigma = _scale(res)
    z = np.abs(res) / sigma
    idx = np.where(z > k)[0]
    return {"n_spikes": int(len(idx)),
            "max_spike_z": float(z.max()) if len(z) else 0.0,
            "spike_indices": [int(i) for i in idx[:10]]}


def check_drift(v, t=None):
    x = _fill(v)
    n = len(x)
    if n < 5:
        return {"drift_ratio": 0.0, "drift_r2": 0.0}
    tt = np.linspace(0, 1, n)
    slope, icpt = np.polyfit(tt, x, 1)
    resid = x - (slope * tt + icpt)
    r2 = 1.0 - np.var(resid) / max(np.var(x), 1e-12)
    return {"drift_ratio": float(abs(slope) / _scale(resid)),
            "drift_r2": float(r2)}


def check_noise(v, t=None, win=20):
    x = _fill(v)
    if len(x) < win + 3:
        return {"noise_est": 0.0, "noise_burst_ratio": 1.0}
    d2 = np.diff(x, 2)
    noise_est = _scale(d2) / np.sqrt(6.0)
    roll = np.array([np.std(x[i:i + win]) for i in range(0, len(x) - win, win // 2)])
    ratio = float(roll.max() / max(np.median(roll), 1e-9))
    return {"noise_est": float(noise_est), "noise_burst_ratio": ratio}


def check_quantization(v, t=None):
    x = v[np.isfinite(v)]
    if len(x) < 3:
        return {"unique_fraction": 1.0, "levels_per_sigma": 0.0}
    u = np.unique(x)
    step = np.min(np.diff(u)) if len(u) > 1 else 0.0
    lps = _scale(x) / step if step > 0 else 0.0
    return {"unique_fraction": float(len(u) / len(x)),
            "levels_per_sigma": float(min(lps, 1e6))}


def check_level_shift(v, t=None):
    """Largest standardised difference between left and right means over all splits."""
    x = _fill(v)
    n = len(x)
    if n < 20:
        return {"level_shift_score": 0.0, "level_shift_pos": 0.5}
    cs = np.cumsum(x)
    tot = cs[-1]
    ks = np.arange(int(0.1 * n), int(0.9 * n))
    left = cs[ks - 1] / ks
    right = (tot - cs[ks - 1]) / (n - ks)
    score = np.abs(left - right) / _scale(np.diff(x))
    j = int(np.argmax(score))
    return {"level_shift_score": float(score[j]),
            "level_shift_pos": float(ks[j] / n)}


def check_distribution(v, t=None):
    """Shape of the value distribution: plateaus (clipping), skew, smoothness."""
    x = _fill(v)
    if len(x) < 10 or np.ptp(x) == 0:
        return {"top_bin_fraction": 1.0, "near_max_fraction": 1.0, "near_min_fraction": 1.0,
                "kurtosis": 0.0, "lag1_autocorr": 1.0, "max_jump_z": 0.0}
    rng_ = np.ptp(x)
    hist, _ = np.histogram(x, bins=20)
    d = np.diff(x)
    from scipy.stats import kurtosis
    return {"top_bin_fraction": float(hist.max() / len(x)),
            "near_max_fraction": float(np.mean(x >= x.max() - 0.01 * rng_)),
            "near_min_fraction": float(np.mean(x <= x.min() + 0.01 * rng_)),
            "kurtosis": float(kurtosis(x)),
            "lag1_autocorr": float(np.corrcoef(x[:-1], x[1:])[0, 1]),
            "max_jump_z": float(np.max(np.abs(d)) / _scale(d))}


def spectral_summary(v, t):
    x = _fill(v)
    x = x - x.mean()
    dt = np.median(np.diff(t)) if len(t) > 1 else 1.0
    dt = dt if dt > 0 else 1.0
    fs = 1.0 / dt
    f, p = sps.welch(x, fs=fs, nperseg=min(128, len(x)))
    p = p[1:] if len(p) > 2 else p
    f = f[1:] if len(f) > 2 else f
    pn = p / max(p.sum(), 1e-18)
    ent = float(-(pn * np.log(pn + 1e-18)).sum() / np.log(len(pn))) if len(pn) > 1 else 0.0
    return {"fs_hz": float(fs),
            "dominant_freq_hz": float(f[int(np.argmax(p))]) if len(f) else 0.0,
            "spectral_entropy": ent,
            "peak_power_ratio": float(pn.max()) if len(pn) else 0.0}


def ref_stats(v_full):
    """Summary of a sensor's own (long) history. In deployment this comes from the
    sensor's past data; it lets tools judge a window relative to what is normal
    for THAT sensor."""
    x = _fill(np.asarray(v_full, dtype=float))
    res = x - median_filter(x, size=11, mode="nearest")
    return {"scale": _scale(x),
            "noise": _scale(np.diff(x, 2)) / np.sqrt(6.0),
            "resid_scale": _scale(res)}


def ref_features(v, ref):
    """Features that compare the window with the sensor's own history."""
    x = _fill(v)
    n = len(x)
    tt = np.linspace(0, 1, n)
    slope = np.polyfit(tt, x, 1)[0] if n > 5 else 0.0
    res = x - median_filter(x, size=11, mode="nearest")
    d2 = np.diff(x, 2)
    return {"ref_drift": float(abs(slope) / ref["scale"]),
            "ref_noise_ratio": float(_scale(d2) / np.sqrt(6.0) / max(ref["noise"], 1e-9)),
            "ref_spike_z": float(np.max(np.abs(res)) / max(ref["resid_scale"], 1e-9)),
            "ref_range_ratio": float((np.max(x) - np.min(x)) / ref["scale"])}


TOOLS = {
    "check_missing": check_missing,
    "check_timestamps": check_timestamps,
    "check_stuck": check_stuck,
    "check_clipping": check_clipping,
    "check_spikes": check_spikes,
    "check_drift": check_drift,
    "check_noise": check_noise,
    "check_quantization": check_quantization,
    "check_level_shift": check_level_shift,
    "check_distribution": check_distribution,
    "spectral_summary": spectral_summary,
}


def run_all_checks(v, t, ref=None):
    """Run every tool and merge results into one flat dict.
    If `ref` (from ref_stats) is given, history-relative features are added."""
    v = np.asarray(v, dtype=float)
    t = np.asarray(t, dtype=float)
    out = {}
    for name, fn in TOOLS.items():
        out.update(fn(v, t))
    if ref is not None:
        out.update(ref_features(v, ref))
    return out


# ----------------------------------------------------------------------------
# fault injection (ground truth for the benchmark)
# ----------------------------------------------------------------------------

FAULTS = ["stuck_at", "spike", "drift", "noise_burst", "dropout_gap",
          "clipping", "quantization", "timestamp_jitter"]


def inject_fault(v, t, kind, rng, severity=None):
    """Return (v_faulty, t_faulty, meta). `severity` in [0.3, 1.0]."""
    v = np.asarray(v, dtype=float).copy()
    t = np.asarray(t, dtype=float).copy()
    n = len(v)
    sev = float(rng.uniform(0.3, 1.0)) if severity is None else float(severity)
    s = _scale(v)

    def seg(frac):
        L = max(5, int(n * frac))
        start = int(rng.integers(0, n - L))
        return start, L

    if kind == "stuck_at":
        a, L = seg(0.12 + 0.3 * sev)
        v[a:a + L] = v[a]
    elif kind == "spike":
        k = int(rng.integers(3, 9))
        idx = rng.choice(n, k, replace=False)
        v[idx] += rng.choice([-1.0, 1.0], k) * s * (6 + 8 * sev)
    elif kind == "drift":
        v += np.linspace(0, 1, n) * s * (1 + 4 * sev) * rng.choice([-1.0, 1.0])
    elif kind == "noise_burst":
        a, L = seg(0.1 + 0.2 * sev)
        v[a:a + L] += rng.normal(0, s * (1 + 3 * sev), L)
    elif kind == "dropout_gap":
        for _ in range(int(rng.integers(1, 4))):
            a, L = seg(0.04 + 0.1 * sev)
            v[a:a + L] = np.nan
    elif kind == "clipping":
        hi = np.nanpercentile(v, 100 - (5 + 15 * sev))
        v = np.minimum(v, hi)
    elif kind == "quantization":
        step = s * (0.5 + 1.5 * sev)
        v = np.round(v / step) * step
    elif kind == "timestamp_jitter":
        dt = float(np.median(np.diff(t))) or 1.0
        t = np.sort(t + rng.normal(0, dt * (0.2 + 0.5 * sev), n))
        dup = rng.choice(np.arange(1, n), max(2, int(0.03 * n)), replace=False)
        t[dup] = t[dup - 1]
    else:
        raise ValueError(f"unknown fault kind: {kind}")
    return v, t, {"fault": kind, "severity": sev}


# ----------------------------------------------------------------------------
# personalised baseline: judge a window against the sensor's OWN history
# ----------------------------------------------------------------------------

def window_features(v, t):
    """Flat numeric feature dict for one window (no history needed)."""
    f = run_all_checks(v, t)
    f.pop("spike_indices", None)
    return {k: float(x) for k, x in f.items()}


def history_profile(feature_rows):
    """feature_rows: list of window_features dicts from CLEAN past windows of one sensor."""
    keys = list(feature_rows[0].keys())
    M = np.array([[r[k] for k in keys] for r in feature_rows], dtype=float)
    return {"keys": keys, "mean": M.mean(0).tolist(), "std": M.std(0).tolist()}


def z_features(f, prof):
    """How unusual is this window compared with this sensor's usual windows?"""
    out = {}
    for k, mu, sd in zip(prof["keys"], prof["mean"], prof["std"]):
        denom = sd + 0.1 * abs(mu) + 1e-3
        out["z_" + k] = float(np.clip((f[k] - mu) / denom, -50, 50))
    return out
