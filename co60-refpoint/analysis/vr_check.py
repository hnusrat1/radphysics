"""Variance-reduction diagnostics: photon quantities of two runs at matched geometry, compared batch by batch.

Used for the photon-only importance-sampling checks (M1-check-primary-is-*): primary
fluence against the analytic value, total and scattered fluence, kerma. Values are
batch means with the standard error over batches; differences are in units of the
combined standard error.

Usage: python analysis/vr_check.py <run_a> <run_b> [...]
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.m1 import TABLES, load_batches  # noqa: E402
from analysis.primary import fluence_cylinder_average, g4_mu_water  # noqa: E402

KEYS = ["primary_line1_over_analytic", "primary_line2_over_analytic", "primary_over_analytic",
        "fluence_total_per_phi0", "scattered_fraction_fluence", "scattered_fraction_energy_fluence",
        "kerma_per_phi0_Gy_cm2"]


def batch_values(b):
    from analysis.m1 import J_PER_MEV, tally
    g, s, n = b["geometry"], b["source"], b["histories"]
    vol, phi0 = g["scoring_volume_cm3"], s["phi0_per_history_cm-2"]
    tl = tally(b, "photon_track_length")
    efl = tally(b, "photon_energy_fluence")
    o = {}
    ratios = []
    for line, cat in (("line1", 0), ("line2", 1)):
        mu, _ = g4_mu_water(TABLES, s[f"{line}_MeV"])
        frac = b["scalars"][f"histories_{line}"] / n
        expected = frac * fluence_cylinder_average(g["scoring_radius"], g["scoring_thickness"], g["scoring_depth"],
                                                   s["ssd"], s["solid_angle_sr"], mu)
        o[f"primary_{line}_over_analytic"] = tl[cat] / (vol * n) / expected
        ratios.append(o[f"primary_{line}_over_analytic"])
    o["primary_over_analytic"] = float(np.mean(ratios))
    o["fluence_total_per_phi0"] = tl.sum() / (vol * n * phi0)
    o["scattered_fraction_fluence"] = tl[2:4].sum() / tl.sum()
    o["scattered_fraction_energy_fluence"] = efl[2:4].sum() / efl.sum()
    o["kerma_per_phi0_Gy_cm2"] = b["scalars"]["kerma_energy_transfer"] * J_PER_MEV * 1e3 / (g["scoring_mass_g"] * n * phi0)
    return o


def summarize(run_id):
    per = [batch_values(b) for b in load_batches(run_id)]
    out = {"batches": len(per)}
    for k in KEYS:
        v = np.array([p[k] for p in per])
        out[k] = (v.mean(), v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan"))
    return out


def main(runs):
    res = {r: summarize(r) for r in runs}
    print(f"{'quantity':36s}" + "".join(f"{r:>34s}" for r in runs))
    for k in KEYS:
        row = f"{k:36s}"
        for r in runs:
            m, s = res[r][k]
            row += f"{m:>22.6g} ± {s:<9.2g}"
        print(row)
    if len(runs) == 2:
        a, b = res[runs[0]], res[runs[1]]
        print("\ndifference (second - first) / combined SEM:")
        for k in KEYS:
            (ma, sa), (mb, sb) = a[k], b[k]
            print(f"  {k:36s} {100 * (mb / ma - 1):+.3f}%  ({(mb - ma) / np.hypot(sa, sb):+.1f} sigma)")
    return res


if __name__ == "__main__":
    main(sys.argv[1:])
