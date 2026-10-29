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
  b0              EGSnrc (DOSRZnrc, FLURZnrc, SPRRZnrc) against Geant4 option4 in geometry B0, with the shape
                  metrics of the B0 spectra and the check of the s_w,air calculation against SPRRZnrc (D039)
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
from analysis.stopping import bases, spencer_attix  # noqa: E402

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


R0_DEPTH_CM = 5.0


def phi0_at_r0(b):
    """Phi0 per history at the point of interest (SSD + 5 cm). The run output's phi0 refers to the scoring
    cylinder's depth, which is 8 cm in the boundary-free runs, so it is recomputed here from the solid angle."""
    s = b["source"]
    return 1.0 / (s["solid_angle_sr"] * (s["ssd"] + R0_DEPTH_CM) ** 2)


def dose_bf(run_id, kcol_res=None, cyl_res=None):
    bs = load_batches(run_id)
    phi0 = phi0_at_r0(bs[0])
    v = np.array([d_bf_per_history(b) / phi0 for b in bs])
    out = {"run": run_id,
           "D_over_phi0_Gy_cm2": {"mean": float(v.mean()), "sem": float(v.std(ddof=1) / np.sqrt(len(v))),
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
            seg = txt[m.end():]
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


def shape_metrics(a, b, edges):
    """Shape comparison of spectrum a against b (per bin, same edges): W1 (MeV), JSD, the ratio of mean energies."""
    p, q = normalize(a), normalize(b)
    c = 0.5 * (edges[:-1] + edges[1:])
    return {"w1_MeV": wasserstein1(p, q, edges), "jsd": jsd(p, q),
            "mean_energy_ratio": float(np.sum(p * c) / np.sum(q * c))}


def bin_ratios(a_batches, b_batches, edges):
    """Per-bin ratio of the shapes (each batch normalized to unit sum), a over b, with standard errors from the batch
    spread, over bins holding at least 0.5% of b's peak: the lowest bin, the largest deviation and the largest |z|."""
    pa = np.array([x / x.sum() for x in a_batches])
    pb = np.array([x / x.sum() for x in b_batches])
    ma, mb = pa.mean(0), pb.mean(0)
    keep = mb >= 0.005 * mb.max()
    r = np.where(keep, ma / np.where(mb > 0, mb, 1), np.nan)
    s = r * np.hypot(pa.std(0, ddof=1) / np.sqrt(len(pa)) / np.where(ma > 0, ma, 1),
                     pb.std(0, ddof=1) / np.sqrt(len(pb)) / np.where(mb > 0, mb, 1))
    z = (r - 1) / s
    c = 0.5 * (edges[:-1] + edges[1:])

    def at(i):
        return {"MeV": [float(edges[i]), float(edges[i + 1])], "ratio": float(r[i]), "sem": float(s[i]), "z": float(z[i])}
    idx = np.flatnonzero(keep)
    return {"lowest_bin": at(idx[0]), "largest_deviation": at(idx[np.nanargmax(np.abs(r[idx] - 1))]),
            "largest_z": at(idx[np.nanargmax(np.abs(z[idx]))]),
            "bins_beyond_3_sigma": int(np.sum(np.abs(z[idx]) > 3)), "bins_compared": int(idx.size),
            "centres_MeV": c[idx].tolist()}


def b0_spectra(sp):
    """EGSnrc (FLURZnrc) against Geant4 option4 spectra in B0 on the common 10 keV grid above 10 keV (both codes'
    cut-offs): shape metrics, and the same metrics between half-samples of each code as its statistical scale."""
    jobs = egs_jobs("M2-B0-egsnrc-fluence")
    egs = [flurz_spectra(f)["total"] for f in jobs]
    top = egs[0]["bintop"]
    n = len(top)
    edges = np.concatenate([[0.0], top])
    assert np.allclose(np.diff(edges), 0.01)
    width = np.diff(edges)
    obs = [observables(b, sp) for b in load_batches("M2-B0-option4")]
    series = {
        "photon": ([x["photons"] * width for x in egs],
                   [np.asarray(o["L1_photon_fluence_10keV"]["total"])[:n] for o in obs]),
        "electron": ([x["electrons"] * width for x in egs],
                     [np.asarray(o["L4_electron_fluence_10keV_per_MeV"])[:n] * 0.010 for o in obs]),
        "electron_linear_estimator": ([x["electrons"] * width for x in egs],
                                      [np.asarray(o["L4lin_electron_fluence_10keV_per_MeV"])[:n] * 0.010
                                       for o in obs]),
    }
    sel, e_edges = slice(1, n), edges[1:]
    centres = 0.5 * (e_edges[:-1] + e_edges[1:])
    out = {"grid": "10 keV bins from 0.01 to 1.34 MeV; ratios are EGSnrc over Geant4"}
    for name, (e_, g_) in series.items():
        m = shape_metrics(np.mean(e_, 0)[sel], np.mean(g_, 0)[sel], e_edges)
        mean_e = [batch_mean([float(np.sum(x[sel] * centres) / np.sum(x[sel])) for x in xs]) for xs in (g_, e_)]
        r = ratio(*mean_e)
        m["mean_energy_ratio_batches"] = dict(r, z=(r["ratio"] - 1) / r["sem"])
        m["bins"] = bin_ratios([x[sel] for x in e_], [x[sel] for x in g_], e_edges)
        m["bins"].pop("centres_MeV")
        for code, x in (("egsnrc", e_), ("geant4", g_)):
            h = len(x) // 2
            m[f"half_sample_{code}"] = shape_metrics(np.mean(x[:h], 0)[sel], np.mean(x[h:2 * h], 0)[sel], e_edges)
        out[name] = m
    return out


def s_w_air_check(sp, spr):
    """STUDY_PLAN Section 8 required test (DECISIONS D039): our Spencer-Attix calculation on the FLURZnrc electron
    fluence (2 keV bins; same transport, geometry and PEGS4 stopping powers as SPRRZnrc) against SPRRZnrc; the
    effect of 2 keV bins, from the Geant4 B0 fluence summed to 2 keV, paired by batch; and Geant4's fluence
    against EGSnrc's under the same calculation."""
    peg = sp["pegs4"]
    out = {}
    run = "M2-B0-egsnrc-fluence-2keV"
    spec = yaml.safe_load(open(ROOT / "mc/egsnrc/b0/specs" / f"{run}.yaml"))
    jobs = egs_jobs(run)
    if len(jobs) == spec["jobs"]:
        w = spec["bin_width_MeV"]
        sa, sa_ep = [], []
        for f in jobs:
            t = flurz_spectra(f)["total"]
            assert np.allclose(np.diff(np.concatenate([[0.0], t["bintop"]])), w) and t["bintop"][-1] >= 1.3325, f
            el, pos = t["electrons"] * w, t["positrons"] * w
            sa.append(spencer_attix(el, 0.010, peg["water"], peg["air"], width=w)["s_w_air"])
            sa_ep.append(spencer_attix(el + pos, 0.010, peg["water"], peg["air"], width=w)["s_w_air"])
        out["calc_on_egsnrc_fluence_2keV"] = batch_mean(sa)
        out["calc_on_egsnrc_fluence_2keV_with_positrons"] = batch_mean(sa_ep)
    if complete("M2-B0-option4"):
        s1, s2 = [], []
        for b in load_batches("M2-B0-option4"):
            el = strip(tally(b, "electron_fluence_csda"))
            el2 = el[: el.size // 2 * 2].reshape(-1, 2).sum(1)
            s1.append(spencer_attix(el, 0.010, peg["water"], peg["air"])["s_w_air"])
            s2.append(spencer_attix(el2, 0.010, peg["water"], peg["air"], width=0.002)["s_w_air"])
        out["calc_on_geant4_fluence_1keV"] = batch_mean(s1)
        out["calc_on_geant4_fluence_2keV"] = batch_mean(s2)
        out["geant4_2keV_minus_1keV_paired"] = batch_mean(np.array(s2) - np.array(s1))
    if spr and "calc_on_egsnrc_fluence_2keV" in out:
        r = ratio(out["calc_on_egsnrc_fluence_2keV"], spr)
        out["sprrznrc_over_calc_on_egsnrc_fluence"] = dict(r, z=(r["ratio"] - 1) / r["sem"])
    if "calc_on_egsnrc_fluence_2keV" in out and "calc_on_geant4_fluence_2keV" in out:
        r = ratio(out["calc_on_egsnrc_fluence_2keV"], out["calc_on_geant4_fluence_2keV"])
        out["geant4_over_egsnrc_fluence_same_calc"] = dict(r, z=(r["ratio"] - 1) / r["sem"])
    return out


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
        # 10 keV spectra for the charts: electrons per MeV -> per bin; photons are per bin already (shape only)
        out["geant4_spectra_10keV"] = {
            "electron": (np.array(g["L4_electron_fluence_10keV_per_MeV"]["mean"]) * 0.010).tolist(),
            "photon": np.array(g["L1_photon_fluence_10keV"]["total"]["mean"]).tolist()}
    if "egsnrc_D_over_phi0_Gy_cm2" in out and "geant4_D_over_phi0_Gy_cm2" in out:
        out["D_egsnrc_over_geant4"] = ratio(out["geant4_D_over_phi0_Gy_cm2"], out["egsnrc_D_over_phi0_Gy_cm2"])
    if "egsnrc_s_w_air" in out and "geant4_s_w_air_pegs4" in out:
        out["s_w_air_egsnrc_over_geant4"] = ratio(out["geant4_s_w_air_pegs4"], out["egsnrc_s_w_air"])
    out["s_w_air_check"] = s_w_air_check(sp, out.get("egsnrc_s_w_air"))
    if "egsnrc_spectra_10keV" in out and complete("M2-B0-option4"):
        out["spectra_metrics"] = b0_spectra(sp)
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
        if L == "option4" and complete("M2-R0-option4-bf-ext", 25):
            rid = "M2-R0-option4-bf+M2-R0-option4-bf-ext"      # pooled (D022, D027)
        if complete(rid.split("+")[0]):
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
