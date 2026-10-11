#!/usr/bin/env python3
"""M2 Monte Carlo validation (STUDY_PLAN Sections 4.1-4.3, 9): results and comparisons.

    python analysis/m2.py      # writes analysis/results/M2_results.json

Contents (each part is skipped while its runs are incomplete):
  runs            M2-R0-option4 / -livermore / -penelope, all layers (same observables as M1)
  dose_bf         boundary-free D/Phi0 at R0 per list (M2-R0-<list>-bf) and its ratio to the cylinder D
  pdd             PDD(5) and d_max from the boundary-free depth scan (M2-R0-option4-pdd)
  step_size       M2 option4 (range factor 0.04) over M1 option4 (0.08) for every scalar
  list_spread     Livermore and Penelope against option4 for every scalar, and distribution distances
  half_sample     W1 and JSD between the two halves of the option4 batches, per distribution
  b0              EGSnrc (DOSRZnrc, FLURZnrc, SPRRZnrc) against Geant4 option4 in geometry B0
"""
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis import m1  # noqa: E402
from analysis.m1 import (BINS, J_PER_MEV, RUNS, TABLES, analyse_run, combine, load_batches, observables,  # noqa: E402
                         strip, tally, to_jsonable)
from analysis.scoring import jsd, normalize, wasserstein1  # noqa: E402
from analysis.stopping import bases  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis" / "results"
LISTS = ("option4", "livermore", "penelope")
M1_REF = "M1-R0-option4+M1-R0-option4-rep2"
MIN_BATCHES = 20


def complete(run_id, n=MIN_BATCHES):
    return len(list((RUNS / run_id).glob("batch_*/output.json"))) >= n


def scalar_keys(res):
    return [k for k, v in res.items() if isinstance(v, dict) and isinstance(v.get("mean"), (float, np.floating))]


def ratio(a, b):
    """b/a with propagated standard errors (independent runs)."""
    r = b["mean"] / a["mean"]
    sa = a.get("sem_conservative", a["sem"])
    sb = b.get("sem_conservative", b["sem"])
    return {"ratio": float(r), "sem": float(abs(r) * math.hypot(sa / a["mean"], sb / b["mean"]))}


def d_bf_per_history(b):
    """Boundary-free dose per history (Gy) in r < 1 cm, 4.9-5.1 cm from a run's depth scan."""
    d = b["bins"]["depth"]
    edges = np.linspace(d["min"], d["max"], d["n"] + 1)
    sel = (edges[:-1] >= 4.9 - 1e-9) & (edges[1:] <= 5.1 + 1e-9)
    width = float(np.sum(np.diff(edges)[sel]))
    assert abs(width - 0.2) < 1e-9
    r = d.get("radius", 1.0)
    return strip(tally(b, "depth_edep"))[sel].sum() / b["histories"] * J_PER_MEV / 1e-3 / (math.pi * r * r * width)


def dose_bf(run_id, kcol_res=None, cyl_res=None):
    bs = load_batches(run_id)
    phi0 = bs[0]["source"]["phi0_per_history_cm-2"]
    v = np.array([d_bf_per_history(b) / phi0 for b in bs])
    out = {"D_over_phi0_Gy_cm2": {"mean": float(v.mean()), "sem": float(v.std(ddof=1) / np.sqrt(len(v))),
                                  "n_batches": len(v)}}
    if kcol_res:
        k = kcol_res["L5_Kcol_over_phi0_Gy_cm2"]
        out["beta"] = ratio(k, out["D_over_phi0_Gy_cm2"])
    if cyl_res:
        out["cylinder_over_bf"] = ratio(out["D_over_phi0_Gy_cm2"], cyl_res["L5_D_over_phi0_Gy_cm2"])
    return out


def half_sample_distances(run_id, sp):
    """W1 and JSD between observables from the first and second half of the batches (shape only)."""
    bs = load_batches(run_id)
    h = len(bs) // 2
    a = combine([observables(b, sp) for b in bs[:h]])
    b = combine([observables(b, sp) for b in bs[h:2 * h]])
    return distribution_distances(a, b)


DISTS = {   # result key (with sub-key) -> grid
    "L1_photon_fluence_comparison/primary": "photon_energy", "L1_photon_fluence_comparison/scattered": "photon_energy",
    "L1_angular_primary": "cos_theta", "L1_angular_scattered": "cos_theta",
    "L3_spectrum_comparison/all": "electron_energy", "L3_spectrum_comparison/parent_primary": "electron_energy",
    "L3_cos_all": "cos_theta", "L4_electron_fluence_comparison": "electron_energy", "L4_angular": "cos_theta",
}


def get(res, key):
    for part in key.split("/"):
        res = res[part]
    return np.asarray(res["mean"], float)


def distribution_distances(a, b):
    out = {}
    for key, grid in DISTS.items():
        edges = BINS["comparison"][grid]["edges"]
        p, q = normalize(get(a, key)), normalize(get(b, key))
        out[key] = {"w1": wasserstein1(p, q, edges), "jsd": jsd(p, q)}
    return out


# ---------------------------------------------------------------- EGSnrc outputs
def egs_jobs(run_id):
    return sorted((RUNS / run_id).glob("job_*/*.egslst"))


def dosrz_dose(lst, region=3):
    """Total dose (Gy cm^2 per incident fluence on axis at the front face) and % uncertainty in a region."""
    txt = Path(lst).read_text()
    m = re.search(rf"^\s*{region}\s+\d+\s+\d+\s+([\d.E+-]+)\s*\+/-\s*([\d.]+)%", txt, re.M)
    return float(m.group(1)), float(m.group(2)) / 100


def sprrz_spr(lst, nz=2, nr=1):
    """SPR with track ends (medium / medium 2) in planar zone nz, cylinder nr, and its absolute uncertainty."""
    txt = Path(lst).read_text()
    i = txt.index("spr with track ends")
    m = re.search(rf"NZ,NR=\s*{nz},\s*{nr}\s+[\d.]+\s*-\s*[\d.]+\s+([\d.]+)\s*\(\+/-\s*([\d.]+)\)", txt[i:])
    return float(m.group(1)), float(m.group(2))


ROW = re.compile(r"^\s*([\d.]+)-*\s*([\d.]+E[+-]\d+)\+-\s*([\d.]+)%-*\s*([\d.]+E[+-]\d+)\+-\s*([\d.]+)%-*\s*"
                 r"([\d.]+E[+-]\d+)\+-\s*([\d.]+)%", re.M)


def flurz_spectra(lst, region=3):
    """FLURZnrc fluence spectra in one region, per MeV per unit incident fluence: {'total'|'primary':
    {'bintop', 'electrons', 'photons', 'positrons'}} (relative uncertainties dropped; jobs give the batch spread)."""
    txt = Path(lst).read_text()
    out = {}
    for label, title in (("total", "Total fluence/(MeV)"), ("primary", "Primary fluence/(MeV)")):
        for m in re.finditer(rf"REGION =\s*{region}\b", txt):
            seg = txt[m.end():m.end() + 20000]
            j = seg.find(title)
            if j < 0 or j > 400:
                continue
            block = seg[j:seg.find("Avg E", j)]
            rows = ROW.findall(block)
            a = np.array([[float(r[0]), float(r[1]), float(r[3]), float(r[5])] for r in rows])
            out[label] = {"bintop": a[:, 0], "electrons": a[:, 1], "photons": a[:, 2], "positrons": a[:, 3]}
            break
    return out


def batch_mean(vals):
    v = np.array(vals, float)
    return {"mean": float(v.mean()), "sem": float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan"),
            "n_batches": len(v)}


def b0(sp):
    out = {}
    sd = 100.0
    d0 = 105.0
    scale = (d0 / sd) ** 2      # dose per surface fluence -> per Phi0 at 105 cm (inverse square, vacuum)
    jobs = egs_jobs("M2-B0-egsnrc-dose")
    spec = yaml.safe_load(open(ROOT / "mc/egsnrc/b0/specs/M2-B0-egsnrc-dose.yaml"))
    if len(jobs) == spec["jobs"]:
        out["egsnrc_D_over_phi0_Gy_cm2"] = batch_mean([dosrz_dose(f)[0] * scale for f in jobs])
    jobs = egs_jobs("M2-B0-egsnrc-spr")
    spec = yaml.safe_load(open(ROOT / "mc/egsnrc/b0/specs/M2-B0-egsnrc-spr.yaml"))
    if len(jobs) == spec["jobs"]:
        out["egsnrc_s_w_air"] = batch_mean([sprrz_spr(f)[0] for f in jobs])
    jobs = egs_jobs("M2-B0-egsnrc-fluence")
    spec = yaml.safe_load(open(ROOT / "mc/egsnrc/b0/specs/M2-B0-egsnrc-fluence.yaml"))
    if len(jobs) == spec["jobs"]:
        sp_ = [flurz_spectra(f) for f in jobs]
        top = sp_[0]["total"]["bintop"]
        width = np.diff(np.concatenate([[0.0], top]))
        per_bin = lambda arr: arr * width * scale     # per MeV per surface fluence -> per bin per Phi0
        out["egsnrc_spectra_10keV"] = {
            "bintop_MeV": top,
            "photon_total": combine([per_bin(x["total"]["photons"]) for x in sp_]),
            "photon_primary": combine([per_bin(x["primary"]["photons"]) for x in sp_]),
            "electron_total": combine([per_bin(x["total"]["electrons"]) for x in sp_]),
        }
        tot = [per_bin(x["total"]["photons"]).sum() for x in sp_]
        pri = [per_bin(x["primary"]["photons"]).sum() for x in sp_]
        el = [per_bin(x["total"]["electrons"])[top > 0.0100001].sum() for x in sp_]
        out["egsnrc_photon_fluence_total_per_phi0"] = batch_mean(tot)
        out["egsnrc_photon_fluence_first_generation_per_phi0"] = batch_mean(pri)
        out["egsnrc_electron_fluence_above_10keV_per_phi0"] = batch_mean(el)
    if complete("M2-B0-option4-bf"):
        out["geant4_D_over_phi0_Gy_cm2"] = dose_bf("M2-B0-option4-bf")["D_over_phi0_Gy_cm2"]
    if complete("M2-B0-option4"):
        g = analyse_run("M2-B0-option4", sp)
        out["geant4_s_w_air_pegs4"] = g["L5_s_w_air_pegs4"]
        out["geant4_layers"] = {k: g[k] for k in ("L1_fluence_primary_per_phi0", "L1_fluence_total_per_phi0",
                                                  "L4_electron_fluence_above_10keV_per_phi0")}
    if "egsnrc_D_over_phi0_Gy_cm2" in out and "geant4_D_over_phi0_Gy_cm2" in out:
        out["D_egsnrc_over_geant4"] = ratio(out["geant4_D_over_phi0_Gy_cm2"], out["egsnrc_D_over_phi0_Gy_cm2"])
    if "egsnrc_s_w_air" in out and "geant4_s_w_air_pegs4" in out:
        out["s_w_air_egsnrc_over_geant4"] = ratio(out["geant4_s_w_air_pegs4"], out["egsnrc_s_w_air"])
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sp = bases(TABLES, m1.PEGS4_DIR)
    res = {"runs": {}, "dose_bf": {}}
    for L in LISTS:
        rid = f"M2-R0-{L}"
        if complete(rid):
            res["runs"][rid] = analyse_run(rid, sp)
    for L in LISTS:
        rid = f"M2-R0-{L}-bf"
        if complete(rid):
            lay = res["runs"].get(f"M2-R0-{L}")
            res["dose_bf"][L] = dose_bf(rid, lay, lay)
    if complete("M2-R0-option4-pdd", 10):
        res["pdd"] = m1.pdd("M2-R0-option4-pdd")
    ref = res["runs"].get("M2-R0-option4")
    if ref:
        m1ref = analyse_run(M1_REF, sp)
        res["step_size"] = {k: ratio(m1ref[k], ref[k]) for k in scalar_keys(ref)
                            if k in m1ref and m1ref[k]["mean"] not in (0, 0.0)}
        res["half_sample"] = half_sample_distances("M2-R0-option4", sp)
        res["list_spread"] = {}
        for L in ("livermore", "penelope"):
            o = res["runs"].get(f"M2-R0-{L}")
            if o:
                res["list_spread"][L] = {
                    "scalars": {k: ratio(ref[k], o[k]) for k in scalar_keys(ref) if k in o and ref[k]["mean"] != 0},
                    "distributions": distribution_distances(ref, o)}
    if complete("M2-R0-option4"):
        from analysis.events import summarize as summarize_events
        res["event_sample"] = summarize_events(RUNS / "M2-R0-option4")
    res["b0"] = b0(sp)
    json.dump(to_jsonable(res), open(OUT / "M2_results.json", "w"), indent=1)
    print(f"wrote {OUT / 'M2_results.json'}: runs {list(res['runs'])}, dose_bf {list(res['dose_bf'])}, "
          f"b0 {list(res['b0'])}")


if __name__ == "__main__":
    main()
