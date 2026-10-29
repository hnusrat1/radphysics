"""Scoring of AI answers against the Monte Carlo reference (STUDY_PLAN Section 6; DECISIONS D028 item 4).

Pure functions; the reference and the answers are assembled in analysis/reference.py and analysis/m5.py.

Verdict for one quantity:
  pass        inside the tolerance
  fail        outside the tolerance and outside the Monte Carlo resolution floor
  unresolved  outside the tolerance but inside the resolution floor (the codes disagree that much themselves)
  missing     no usable value in the answer
"""
import numpy as np
from scipy import stats

DOSE_TOL = 0.02          # Layer 5 dose quantities, relative
SCALAR_TOL = 0.05        # other scalars and fractions, relative ...
SCALAR_ABS_FLOOR = 0.005  # ... or this absolute difference, whichever is larger
DIST_FACTOR = 3.0        # distributions: 3 x the Monte Carlo half-sample distance


def finite(x):
    try:
        return x is not None and np.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def scalar_verdict(ai, mc, tol_class, floor_abs=0.0):
    """ai, mc: numbers; floor_abs: resolution floor as an absolute difference."""
    if not finite(ai):
        return {"verdict": "missing"}
    ai, mc = float(ai), float(mc)
    diff = ai - mc
    rel = diff / mc if mc != 0 else np.inf
    if tol_class == "dose":
        tol = DOSE_TOL * abs(mc)
    else:
        tol = max(SCALAR_TOL * abs(mc), SCALAR_ABS_FLOOR)
    if abs(diff) <= tol:
        v = "pass"
    elif abs(diff) <= floor_abs:
        v = "unresolved"
    else:
        v = "fail"
    return {"verdict": v, "ai": ai, "mc": mc, "diff": diff, "rel_diff": rel, "tolerance_abs": tol,
            "floor_abs": floor_abs}


def normalize(v):
    v = np.clip(np.asarray(v, dtype=float), 0, None)
    s = v.sum()
    return v / s if s > 0 else None


def wasserstein1(p, q, edges):
    """W1 between two binned distributions on the same grid, mass at bin centres (STUDY_PLAN Section 6)."""
    c = 0.5 * (np.asarray(edges[:-1]) + np.asarray(edges[1:]))
    return float(np.sum(np.abs(np.cumsum(p) - np.cumsum(q))[:-1] * np.diff(c)))


def jsd(p, q):
    """Jensen-Shannon divergence, base 2 (0 to 1)."""
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def distribution_verdict(ai, mc, edges, half_w1, half_jsd, floor_w1=0.0, floor_jsd=0.0):
    """Shape comparison: both normalized. Thresholds are DIST_FACTOR x the Monte Carlo half-sample distance."""
    n = len(edges) - 1
    if ai is None or len(ai) != n or not all(finite(x) for x in ai):
        return {"verdict": "missing"}
    p, q = normalize(ai), normalize(mc)
    if p is None:
        return {"verdict": "missing"}
    w, j = wasserstein1(p, q, edges), jsd(p, q)
    tw, tj = DIST_FACTOR * half_w1, DIST_FACTOR * half_jsd
    if w <= tw and j <= tj:
        v = "pass"
    elif w <= max(tw, floor_w1) and j <= max(tj, floor_jsd):
        v = "unresolved"
    else:
        v = "fail"
    return {"verdict": v, "w1": w, "jsd": j, "threshold_w1": tw, "threshold_jsd": tj,
            "floor_w1": floor_w1, "floor_jsd": floor_jsd, "ai_norm": p.tolist(), "mc_norm": q.tolist()}


# Revised shape criterion (DECISIONS D038). Chosen after M4-R0 was scored, so it is exploratory for M4-R0 and
# pre-registered for any later run; the half-sample criterion above stays the pre-specified one.
DIST_W1_REL_MEAN = 0.05  # energy distributions: W1 <= 5% of the Monte Carlo mean energy (the scalar tolerance)
DIST_W1_COS = 0.1        # angular distributions: W1 <= 0.1 in cos(theta), 5% of its range


def w1_tolerance(grid, mc, edges):
    """The D038 W1 tolerance for one Monte Carlo distribution, in the units of its grid."""
    if grid == "cos_theta":
        return DIST_W1_COS
    c = 0.5 * (np.asarray(edges[:-1]) + np.asarray(edges[1:]))
    return DIST_W1_REL_MEAN * float(np.sum(normalize(mc) * c))


def distribution_verdict_w1(ai, mc, edges, tol_w1, floor_w1=0.0):
    """D038 shape comparison: W1 against a physical tolerance; JSD is reported but does not enter the verdict."""
    n = len(edges) - 1
    if ai is None or len(ai) != n or not all(finite(x) for x in ai):
        return {"verdict": "missing"}
    p, q = normalize(ai), normalize(mc)
    if p is None:
        return {"verdict": "missing"}
    w = wasserstein1(p, q, edges)
    v = "pass" if w <= tol_w1 else "unresolved" if w <= floor_w1 else "fail"
    return {"verdict": v, "w1": w, "jsd": jsd(p, q), "tolerance_w1": tol_w1, "floor_w1": floor_w1}


def fractions_verdict(ai, mc, categories, floors=None):
    """Per-category scalar verdicts (tolerance class 'scalar'); the quantity passes when every category passes."""
    if not isinstance(ai, dict):
        return {"verdict": "missing"}
    per = {c: scalar_verdict(ai.get(c), mc[c], "scalar", (floors or {}).get(c, 0.0)) for c in categories}
    vs = [p["verdict"] for p in per.values()]
    v = "missing" if "missing" in vs else "fail" if "fail" in vs else "unresolved" if "unresolved" in vs else "pass"
    return {"verdict": v, "categories": per}


LAYER_ORDER = [1, 2, 3, 4, 5]


def breakpoint(verdicts_by_layer):
    """First layer with a resolved failure (STUDY_PLAN Section 6). verdicts_by_layer: {layer: [verdict, ...]}.
    Returns (layer or None, reason)."""
    for L in LAYER_ORDER:
        vs = verdicts_by_layer.get(L, [])
        if "fail" in vs:
            return L, "fail"
    return None, "no resolved failure"


# ---------------------------------------------------------------- sampled events
EVENT_FEATURES = ["e_in", "e_out", "cos_scatter", "t_e"]


def weighted_ks(x_ai, x_mc, w_mc):
    """Two-sample KS statistic with weights on the Monte Carlo side; p-value with the effective MC sample size."""
    x_ai = np.sort(np.asarray(x_ai, float))
    order = np.argsort(x_mc)
    xm, wm = np.asarray(x_mc, float)[order], np.asarray(w_mc, float)[order]
    cdf_m = np.cumsum(wm) / wm.sum()
    grid = np.concatenate([x_ai, xm])
    f_ai = np.searchsorted(x_ai, grid, side="right") / len(x_ai)
    idx = np.searchsorted(xm, grid, side="right") - 1
    f_mc = np.where(idx >= 0, cdf_m[np.clip(idx, 0, None)], 0.0)
    d = float(np.max(np.abs(f_ai - f_mc)))
    n_eff = wm.sum() ** 2 / np.sum(wm ** 2)
    n, m = len(x_ai), n_eff
    en = np.sqrt(n * m / (n + m))
    p = float(stats.kstwobign.sf(en * d)) if n > 0 else np.nan
    return {"ks": d, "p": p, "n_ai": int(n), "n_eff_mc": float(n_eff)}


def resample(features, weights, n, rng):
    p = np.asarray(weights, float) / np.sum(weights)
    idx = rng.choice(len(p), size=n, replace=True, p=p)
    return features[idx]


def energy_distance(a, b):
    """Energy distance between two samples of feature vectors (rows), Euclidean (Szekely)."""
    def mean_dist(x, y):
        tot, cnt = 0.0, 0
        for i in range(0, len(x), 500):
            d = np.sqrt(((x[i:i + 500, None, :] - y[None, :, :]) ** 2).sum(-1))
            tot += d.sum()
            cnt += d.size
        return tot / cnt
    return float(2 * mean_dist(a, b) - mean_dist(a, a) - mean_dist(b, b))


def event_features(ev, scale):
    """Stack the features used for the joint comparison, scaled to comparable ranges (MeV / cos)."""
    cols = []
    for f in EVENT_FEATURES:
        x = np.asarray(ev[f], float)
        x = np.where(np.isfinite(x), x, -1.0)       # absent outgoing photon -> -1 (outside every physical range)
        cols.append(x / scale[f])
    return np.stack(cols, 1)
