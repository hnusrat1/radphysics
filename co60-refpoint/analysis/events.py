"""Monte Carlo event sample: per-event validity baseline and the Doppler tolerance (DECISIONS D014).

The event sample holds individual photon interactions in the scoring volume
(STUDY_PLAN Section 4.1). The fields an AI model returns are a subset: incident
energy, process, scattered photon energy and angle, electron kinetic energy and
angle. The Monte Carlo events are mapped onto that subset here, so the same
checks run on both.
"""

import csv
from pathlib import Path

import numpy as np

from analysis.validity import BINDING_TOLERANCE, check_events, doppler_tolerance

ROOT = Path(__file__).resolve().parent.parent


def load_events(run_dir):
    rows = []
    for f in sorted(Path(run_dir).glob("batch_*/events.csv")):
        with open(f) as fh:
            rows.extend(csv.DictReader(fh))
    if not rows:
        return None
    col = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    ev = {k: col[k].astype(float) for k in col if k not in ("process",)}
    ev["process"] = col["process"]
    return ev


def ai_view(ev):
    """Map Monte Carlo records onto the fields an AI event carries."""
    pair = ev["process"] == "conv"
    t_pos = np.where(pair, ev["t_positron_MeV"], 0.0)
    # Electron kinetic energy: the largest charged secondary, or for pair production the electron.
    t_e = np.where(pair, ev["t_charged_sum_MeV"] - t_pos, ev["t_max_MeV"])
    return {"process": ev["process"], "e_in": ev["e_in_MeV"], "e_out": ev["e_out_MeV"],
            "cos_scatter": ev["cos_scatter"], "t_e": t_e, "t_pos": t_pos, "weight": ev["weight"]}


def summarize(run_dir):
    ev = load_events(run_dir)
    if ev is None:
        return None
    a = ai_view(ev)
    tol = doppler_tolerance(a["process"], a["e_in"], a["e_out"], a["cos_scatter"], a["t_e"])
    strict = check_events(a["process"], a["e_in"], a["e_out"], a["cos_scatter"], a["t_e"], a["t_pos"])
    bound = check_events(a["process"], a["e_in"], a["e_out"], a["cos_scatter"], a["t_e"], a["t_pos"],
                         delta_c=tol["delta_c"], energy_tol=BINDING_TOLERANCE)
    w = a["weight"]
    procs, counts = np.unique(a["process"], return_counts=True)

    def rate(mask):
        return float(np.sum(w * ~mask) / np.sum(w))

    out = {
        "n_events": int(len(w)),
        "process_counts": {str(p): int(c) for p, c in zip(procs, counts)},
        "weights": {str(k): int(v) for k, v in zip(*np.unique(np.round(w, 6), return_counts=True))},
        "doppler_tolerance": tol,
        "binding_tolerance_MeV": BINDING_TOLERANCE,
        "strict_failure_rate": {k: rate(v) for k, v in strict.items()},
        "bound_failure_rate": {k: rate(v) for k, v in bound.items()},
    }
    return out
