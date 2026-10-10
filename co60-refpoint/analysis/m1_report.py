"""Build the M1 results page (HTML) from analysis/results/M1_results.json and the Fano summary.

Every number on the page is read from those files; the narrative text that
interprets them lives in analysis/m1_report_text.yaml.

Usage:
  python analysis/m1_report.py      # writes analysis/results/co60_r0_m1.html
"""

import json
import math
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RES = ROOT / "analysis" / "results"


def sig(x, n=4):
    if x is None or not math.isfinite(x):
        return "n/a"
    if x == 0:
        return "0"
    return f"{x:.{n}g}"


def sci(x, n=4):
    """1.234 × 10⁻¹² as HTML."""
    if x == 0:
        return "0"
    e = math.floor(math.log10(abs(x)))
    m = x / 10**e
    return f"{m:.{n - 1}f} × 10<sup>{e}</sup>"


def pm(mean, sem, digits=None):
    """Value ± uncertainty with matching decimals (uncertainty to 2 significant figures)."""
    if sem is None or not math.isfinite(sem) or sem == 0:
        return sig(mean, 6)
    d = max(0, -int(math.floor(math.log10(abs(sem)))) + 1) if digits is None else digits
    return f"{mean:.{d}f} ± {sem:.{d}f}"


def rel(mean, sem):
    return f"{100 * sem / abs(mean):.2g}%"


def git_head():
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()


def main():
    r = json.load(open(RES / "M1_results.json"))
    fano = yaml.safe_load(open(ROOT / "analysis" / "M1_fano_summary.yaml"))
    text = yaml.safe_load(open(ROOT / "analysis" / "m1_report_text.yaml"))
    main_run = r["runs"]["M1-R0-option4"]
    meta = main_run["_meta"]

    def v(key, run=main_run):
        return run[key]["mean"], run[key]["sem"]

    # Compact data for the charts
    data = {
        "photon": {k: main_run["L1_photon_fluence_10keV"][k] for k in
                   ("primary", "scattered_source", "secondary_photons", "total")},
        "photon_angular": {"primary": main_run["L1_angular_primary"], "scattered": main_run["L1_angular_scattered"]},
        "initial": {k: main_run["L3_spectrum_10keV"][k] for k in ("compton", "photoelectric", "pair_electron", "all")},
        "initial_cos": {"primary": main_run["L3_cos_parent_primary"], "scattered": main_run["L3_cos_parent_scattered"]},
        "electron": {"all": main_run["L4_electron_fluence_10keV_per_MeV"],
                     "delta": main_run["L4_delta_fluence_10keV_per_MeV"],
                     "positron": main_run["L4_positron_fluence_10keV_per_MeV"]},
        "electron_angular": main_run["L4_angular"],
        "pdd": r.get("pdd"),
        "fano": [{"label": k, **{kk: fano[k][kk] for kk in ("ratio_mean", "ratio_sem_batch_spread", "batches",
                                                             "events_total", "deviation_sigma")}}
                 for k in fano],
        "edges": {"photon_10keV": [i * 0.01 for i in range(141)], "electron_10keV": [i * 0.01 for i in range(141)],
                  "cos_10": [-1 + 0.2 * i for i in range(11)]},
    }
    data_json = json.dumps(data, separators=(",", ":"))

    D, Ds = v("L5_D_over_phi0_Gy_cm2")
    beta, betas = v("L5_beta")
    numbers = {
        "D": f"{sci(D, 5)} Gy cm²", "D_rel": rel(D, Ds),
        "K": f"{sci(v('L5_K_over_phi0_Gy_cm2')[0], 5)} Gy cm²",
        "Kcol": f"{sci(v('L5_Kcol_over_phi0_Gy_cm2')[0], 5)} Gy cm²",
        "beta": pm(beta, betas), "beta_rel": rel(beta, betas),
        "g": pm(*v("L5_g")),
        "hist": f"{meta['histories_total']:.2e}".replace("e+0", " × 10<sup>").replace("e+", " × 10<sup>") + "</sup>",
        "batches": meta["batches"],
        "commit": git_head(),
    }
    page = build_page(r, main_run, numbers, data_json, text, fano, v)
    (RES / "co60_r0_m1.html").write_text(page)
    print(f"wrote {RES / 'co60_r0_m1.html'} ({len(page) / 1024:.0f} kB)")


def build_page(r, m, n, data_json, text, fano, v):
    from analysis.m1_page import render
    return render(r, m, n, data_json, text, fano, v, pm, sci, rel, sig)


if __name__ == "__main__":
    main()
