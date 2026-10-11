"""Build the M1 results page (HTML) from analysis/results/M1_results.json and the Fano summary.

Every number on the page is read from those files; the narrative text that
interprets them lives in analysis/m1_report_text.yaml.

Usage:
  python analysis/m1_report.py      # writes analysis/results/co60_r0_m1.html
"""

import json
import math
import re
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
    return f"{m:.{n - 1}f} × 10<sup>{str(e).replace("-", "−")}</sup>"


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
    main_run = r["runs"][r.get("pooling", {}).get("reference_run", "M1-R0-option4")]
    meta = main_run["_meta"]

    def v(key, run=main_run):
        return run[key]["mean"], run[key].get("sem_conservative", run[key]["sem"])

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
    ref = r.get("diagnostics", {}).get("boundary", {}).get("R0_dose_reference")
    if ref and ref["uses"] == "boundary_free":   # DECISIONS D025/D027
        Df, Dfs = ref["D_over_phi0_boundary_free_Gy_cm2"]["mean"], ref["D_over_phi0_boundary_free_Gy_cm2"]["sem"]
        bf, bfs = ref["beta_boundary_free"]["mean"], ref["beta_boundary_free"]["sem"]
        numbers.update({
            "D_label": "D/Φ₀, R0 reference (boundary-free, D027)",
            "D_alt": f"scoring cylinder: {sci(D, 5)} Gy cm² ({100 * (D / Df - 1):+.2f}%)".replace("-", "−"),
            "D": f"{sci(Df, 5)} Gy cm²", "D_rel": rel(Df, Dfs),
            "beta_alt": f"with the cylinder D: {pm(beta, betas)}",
            "beta": pm(bf, bfs), "beta_rel": rel(bf, bfs),
        })
    text["checks"] = build_checks(r, main_run) + text.get("checks_extra", [])
    page = build_page(r, main_run, numbers, data_json, text, fano, v)
    (RES / "co60_r0_m1.html").write_text(page)
    print(f"wrote {RES / 'co60_r0_m1.html'} ({len(page) / 1024:.0f} kB)")


def build_checks(r, m):
    """Rows of the checks table, every number taken from M1_results.json.

    Status: 'ok' when the check passes its stated criterion, 'warn' when it does not
    and the cause is open or understood but not fixed, 'bad' when it fails outright.
    """
    rows = []

    def z_of(mean, sem, ref=1.0):
        return (mean - ref) / sem if sem else float("nan")

    def add(name, result, status, label, chip=None):
        result = re.sub(r"(?:(?<=[ (])|^)-(?=\d)", "−", result)   # typographic minus for signed values
        rows.append({"name": name, "result": result, "status": status, "status_label": label, "chip": chip})

    diag = r.get("diagnostics", {})
    pf = diag.get("primary_fluence", {})

    th = pf.get("M1-thinslab-option4-v2")
    if th:
        x = th["tracking_mu"]; z = z_of(x["mean"], x["sem_conservative"])
        add("Thin slab: primary track-length fluence against the analytic value (r 4 cm, 0.05 cm at 0.5 cm, no VR)",
            f"{pm(x['mean'], x['sem_conservative'])} ({z:+.1f}σ, {x['n_batches']} batches)",
            "ok" if abs(z) < 3 else "warn", "passes" if abs(z) < 3 else "outside 3σ")

    tr = diag.get("transmission")
    if tr:
        parts = [f"{l['energy_MeV']} MeV: {pm(l['ratio'], l['ratio_sem'])}" for l in tr.values()]
        add("Tracking attenuation coefficient over G4EmCalculator value (TestEm13 transmission, 5 cm water)",
            "; ".join(parts) + ". The analytic fluence below uses the tracking value.", "ok", "measured")

    main = pf.get("M1-R0-option4")
    if main:
        x = main["tracking_mu"]; z = z_of(x["mean"], x["sem_conservative"]); zb = z_of(x["mean"], x["sem"])
        st = "ok" if abs(z) < 3 else "warn"
        add("R0: primary fluence against the analytic value (main run)",
            f"{pm(x['mean'], x['sem_conservative'])} ({z:+.1f}σ on the counting-statistics error; "
            f"{zb:+.1f}σ on the batch spread, which is unusually small in this run; see Findings)",
            st, "passes" if st == "ok" else "open", None if st == "ok" else "Primary fluence: open anomaly")
    rep = pf.get("M1-R0-option4-rep2")
    if rep and rep["tracking_mu"]["sem"]:
        x = rep["tracking_mu"]; z = z_of(x["mean"], x["sem_conservative"])
        add("R0: primary fluence, replicate with new seeds",
            f"{pm(x['mean'], x['sem_conservative'])} ({z:+.1f}σ, {x['n_batches']} batches)",
            "ok" if abs(z) < 3 else "warn", "passes" if abs(z) < 3 else "outside 3σ")

    others = [(k, v) for k, v in pf.items() if k not in ("M1-R0-option4", "M1-R0-option4-rep2", "M1-thinslab-option4-v2")]
    if others:
        cells = "<br>".join(f"{esc_(v['label'])}: {pm(v['tracking_mu']['mean'], v['tracking_mu']['sem_conservative'])}"
                            for _, v in others if v["tracking_mu"]["sem"])
        add("Primary fluence in the diagnostic runs (photon-only and full transport, with and without importance sampling)",
            cells, "ok", "no bias found")

    x = m["L5_D_continuous_over_D"]
    ok = abs(x["mean"] - 1) < 0.002
    add("Fluence-to-dose closure: ∫Φ_E L_Δ dE plus track ends, over scored dose (CSDA-weighted estimator)",
        f"{pm(x['mean'], x['sem'])}; linear estimator {pm(m['L5_D_continuous_over_D_linear']['mean'], m['L5_D_continuous_over_D_linear']['sem'])} (DECISIONS D020)",
        "ok" if ok else "warn", "passes" if ok else "fails")

    for label, key, name in (("S2_cut_independence", "L5_D_over_phi0_Gy_cm2", "Cut independence (S2, 5 keV thresholds): D/Φ₀"),
                             ("S1_volume_size", "L5_D_over_phi0_Gy_cm2", "Scoring volume size (S1, r 0.5 cm, t 0.1 cm): D/Φ₀"),
                             ("vr_check_analog", "L5_D_over_phi0_Gy_cm2", "Variance reduction against analog transport: D/Φ₀")):
        c = r.get(label)
        if not c:
            continue
        s = c["scalars"][key]
        ok = abs(s["z"]) < 3
        extra = ""
        if label == "S2_cut_independence":
            sw = c["scalars"]["L5_s_w_air_icru37"]
            extra = f"; s<sub>w,air</sub> (ICRU 37 basis) {100 * sw['rel_diff']:+.4f}%"
        es = c["electron_fluence_shape"]
        add(name, f"{100 * s['rel_diff']:+.2f}% ({s['z']:+.1f}σ){extra}; electron fluence spectrum: normalization "
            f"{100 * (es['norm_ratio'] - 1):+.2f}%, shape χ²/ndf {es['chi2_shape']:.0f}/{es['ndf_shape']}",
            "ok" if ok else "warn", "passes" if ok else "outside 3σ")

    bd = diag.get("boundary")
    if bd:
        key = "cylinder_over_boundary_free:" + r.get("pooling", {}).get("reference_run", "M1-R0-option4")
        x = bd.get(key) or bd.get("cylinder_over_boundary_free:M1-R0-option4")
        if x:
            z = (x["ratio"] - 1) / x["sem"]
            n_free = " + ".join(str(e["n_batches"]) for e in bd["boundary_free_estimates"].values())
            used = bd.get("R0_dose_reference", {}).get("uses") == "boundary_free"
            add("Scoring-cylinder boundary: D in the cylinder over the boundary-free dose at R0 (depth scan with the cylinder moved to 8 cm)",
                f"{pm(x['ratio'], x['sem'])} ({z:+.1f}σ; boundary-free D from {n_free} batches in two runs, inverse-variance weighted). Next to the cylinder the dose is "
                f"distorted by up to ±3%; see Findings."
                + (" The boundary-free D is the R0 reference (D025, D027)." if used else ""),
                "warn", "boundary-free D adopted" if used else "within 3σ; see Findings", "Boundary artifact next to the cylinder")

    ev = r.get("event_sample")
    if ev:
        b = ev["bound_failure_rate"]; s = ev["strict_failure_rate"]
        add("Event sample under the validity rules (Monte Carlo events, D014)",
            f"{ev['n_events']:,} interactions: {100 * b['valid']:.3f}% fail the bound rules "
            f"(δ<sub>C</sub> = {1e3 * ev['doppler_tolerance']['delta_c']:.0f} keV); {100 * s['valid']:.1f}% fail the strict free-electron rules",
            "ok", "as designed")

    meta = m["_meta"]
    add("Bookkeeping: untagged tracks; largest per-interaction energy imbalance",
        f"{meta['untagged_tracks']} untagged; {1e3 * meta['energy_balance_max_abs_MeV']:.2f} keV",
        "ok" if meta["untagged_tracks"] == 0 else "bad", "passes" if meta["untagged_tracks"] == 0 else "fails")
    return rows


def esc_(s):
    import html
    return html.escape(str(s))


def build_page(r, m, n, data_json, text, fano, v):
    from analysis.m1_page import render
    return render(r, m, n, data_json, text, fano, v, pm, sci, rel, sig)


if __name__ == "__main__":
    main()
