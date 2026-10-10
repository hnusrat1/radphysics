"""M1 analysis: all five layers at R0 from co60app batch outputs, with checks.

Every observable is computed per batch; the reported value is the batch mean
and the uncertainty is the standard error of the mean over batches (STUDY_PLAN
Section 4.1, at least 10 independent batches). Ratios are formed per batch so
that correlations are carried correctly.

Usage:
  python analysis/m1.py            # writes analysis/results/M1_results.json and figures
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.kinematics import compton_edge  # noqa: E402
from analysis.primary import fluence_cylinder_average, g4_mu_water  # noqa: E402
from analysis.refs import ref_value  # noqa: E402
from analysis.events import summarize as summarize_events  # noqa: E402
from analysis.stopping import bases, spencer_attix  # noqa: E402
from analysis.xcom import water_mu_rho  # noqa: E402

RUNS = ROOT / "mc" / "runs"
OUT = ROOT / "analysis" / "results"
J_PER_MEV = ref_value("joule_per_MeV")
BINS = yaml.safe_load(open(ROOT / "config" / "bins.yaml"))
TABLES = RUNS / "M1-tables-option4-v2" / "batch_00" / "output.json"
PEGS4_DIR = ROOT / "mc" / "egsnrc" / "b0"
TRANSMISSION = RUNS / "M1-check-transmission" / "results.yaml"

CAT_PRIMARY, CAT_SCATTERED = [0, 1], [2, 3]
PROCS = ["compt", "phot", "conv", "Rayl", "other"]


# ---------------------------------------------------------------- loading
def load_batches(run_id):
    """Batches of one run, or of several runs joined with '+' (pooling, DECISIONS D022)."""
    files = [f for part in run_id.split("+") for f in sorted((RUNS / part).glob("batch_*/output.json"))]
    return [json.load(open(f)) for f in files]


def tally(b, name):
    t = b["tallies"][name]
    return np.array(t["data"]).reshape(t["dims"])


def strip(a):
    """Drop the underflow and overflow entries of the last axis."""
    return a[..., 1:-1]


def rebin(a, factor):
    """Sum consecutive groups of `factor` bins along the last axis."""
    n = a.shape[-1] // factor
    return a[..., : n * factor].reshape(*a.shape[:-1], n, factor).sum(-1)


def rebin_edges(a, fine_lo, fine_width, edges):
    """Sum fine bins (lower edge fine_lo, width fine_width) into arbitrary coarse edges (exact on the grid)."""
    idx = np.rint((np.asarray(edges) - fine_lo) / fine_width).astype(int)
    return np.stack([a[..., i0:i1].sum(-1) for i0, i1 in zip(idx[:-1], idx[1:])], -1)


def safe_div(a, b):
    return np.divide(a, b, out=np.zeros_like(np.asarray(a, dtype=float)), where=np.asarray(b) != 0)


# ---------------------------------------------------------------- per-batch observables
def observables(b, sp, tables_json=TABLES):
    n = b["histories"]
    g, src = b["geometry"], b["source"]
    vol, mass, phi0 = g["scoring_volume_cm3"], g["scoring_mass_g"], src["phi0_per_history_cm-2"]
    norm = 1.0 / (vol * n * phi0)               # track length (cm) -> fluence per unit Phi0
    gy = J_PER_MEV / (mass * 1e-3) / n / phi0   # MeV deposited -> Gy cm^2 per unit Phi0
    s = b["scalars"]
    o = {}

    # Layer 1: photons arriving at the point
    pf = strip(tally(b, "photon_fluence")) * norm                    # (4, 140) per 10 keV bin
    o["L1_photon_fluence_10keV"] = {"primary": pf[CAT_PRIMARY].sum(0), "scattered": pf[CAT_SCATTERED].sum(0),
                                    "scattered_source": pf[2], "secondary_photons": pf[3], "total": pf.sum(0)}
    ph_edges = BINS["comparison"]["photon_energy"]["edges"]
    o["L1_photon_fluence_comparison"] = {k: rebin_edges(v, 0.0, 0.01, ph_edges)
                                         for k, v in o["L1_photon_fluence_10keV"].items()}
    tl = tally(b, "photon_track_length")
    ef = tally(b, "photon_energy_fluence")
    o["L1_fluence_total_per_phi0"] = tl.sum() * norm
    o["L1_fluence_primary_per_phi0"] = tl[CAT_PRIMARY].sum() * norm
    o["L1_scattered_fraction_fluence"] = tl[CAT_SCATTERED].sum() / tl.sum()
    o["L1_scattered_fraction_energy_fluence"] = ef[CAT_SCATTERED].sum() / ef.sum()
    o["L1_secondary_photon_fraction_fluence"] = tl[3] / tl.sum()
    o["L1_mean_energy_all_MeV"] = ef.sum() / tl.sum()
    o["L1_mean_energy_primary_MeV"] = ef[CAT_PRIMARY].sum() / tl[CAT_PRIMARY].sum()
    o["L1_mean_energy_scattered_MeV"] = ef[CAT_SCATTERED].sum() / tl[CAT_SCATTERED].sum()
    ang = tally(b, "photon_angular")
    o["L1_angular_primary"] = rebin(ang[CAT_PRIMARY].sum(0), 4) / ang[CAT_PRIMARY].sum()
    o["L1_angular_scattered"] = rebin(ang[CAT_SCATTERED].sum(0), 4) / ang[CAT_SCATTERED].sum()
    o["L1_angular_scattered_fine"] = ang[CAT_SCATTERED].sum(0) / ang[CAT_SCATTERED].sum()
    # Primary fluence per line against exp(-mu d) / (Omega R^2), averaged over the volume
    for line, cat in (("line1", 0), ("line2", 1)):
        mu, _ = g4_mu_water(tables_json, src[f"{line}_MeV"])
        frac = s[f"histories_{line}"] / n
        expected = frac * fluence_cylinder_average(g["scoring_radius"], g["scoring_thickness"],
                                                   g["scoring_depth"], src["ssd"], src["solid_angle_sr"], mu)
        o[f"L1_primary_{line}_measured_over_analytic"] = tl[cat] / (vol * n) / expected
        o[f"L1_primary_{line}_per_phi0"] = tl[cat] * norm
        # Same check with mu from the XCOM elemental cross sections (independent of Geant4's models)
        mu_x = float(water_mu_rho(np.array([src[f"{line}_MeV"]]), tables_json)["total"][0]) * g["water_density_g_cm3"]
        expected_x = frac * fluence_cylinder_average(g["scoring_radius"], g["scoring_thickness"],
                                                     g["scoring_depth"], src["ssd"], src["solid_angle_sr"], mu_x)
        o[f"L1_primary_{line}_measured_over_analytic_xcom"] = tl[cat] / (vol * n) / expected_x
        # Same check with the attenuation coefficient Geant4 uses in tracking, measured by transmission
        # (TestEm13, run M1-check-transmission); its uncertainty is not propagated here (about 0.04% in mu).
        if TRANSMISSION.exists():
            mu_t = yaml.safe_load(open(TRANSMISSION))["lines"][line]["mu_measured_cm-1"]
            expected_t = frac * fluence_cylinder_average(g["scoring_radius"], g["scoring_thickness"],
                                                         g["scoring_depth"], src["ssd"], src["solid_angle_sr"], mu_t)
            o[f"L1_primary_{line}_measured_over_analytic_tracking_mu"] = tl[cat] / (vol * n) / expected_t

    # Layer 2: photon interactions in the scoring volume
    inter = tally(b, "interactions")
    etr = tally(b, "interaction_energy_transfer")
    for label, cats in (("all", [0, 1, 2, 3]), ("primary", CAT_PRIMARY), ("scattered", CAT_SCATTERED)):
        o[f"L2_fraction_{label}"] = inter[cats].sum(0) / inter[cats].sum()
        o[f"L2_etr_fraction_{label}"] = etr[cats].sum(0) / etr[cats].sum()
    o["L2_scattered_share_of_interactions"] = inter[CAT_SCATTERED].sum() / inter.sum()
    o["L2_interactions_per_g_per_phi0"] = inter.sum() / (mass * n * phi0)
    # Interactions per unit primary track length = mu(line) by process (consistency with the cross sections)
    for line, cat in (("line1", 0), ("line2", 1)):
        o[f"L2_mu_measured_{line}_cm-1"] = inter[cat] / tl[cat]

    # Layer 3: electrons and positrons set in motion
    isp = strip(tally(b, "initial_spectrum"))                         # (5 proc, 4 cat, 2 part, 1400) per 1 keV
    icount = tally(b, "initial_count")
    ienergy = tally(b, "initial_energy")
    per_mass = 1.0 / (mass * n * phi0)
    o["L3_spectrum_10keV"] = {
        "all": rebin(isp.sum((0, 1)), 10).sum(0) * per_mass,
        "compton": rebin(isp[0].sum(0)[0], 10) * per_mass,
        "photoelectric": rebin(isp[1].sum(0)[0], 10) * per_mass,
        "pair_electron": rebin(isp[2].sum(0)[0], 10) * per_mass,
        "pair_positron": rebin(isp[2].sum(0)[1], 10) * per_mass,
        "parent_primary": rebin(isp[:, CAT_PRIMARY].sum((0, 1)), 10).sum(0) * per_mass,
        "parent_scattered": rebin(isp[:, CAT_SCATTERED].sum((0, 1)), 10).sum(0) * per_mass,
    }
    el_edges = BINS["comparison"]["electron_energy"]["edges"]
    o["L3_spectrum_comparison"] = {k: rebin_edges(v, 0.0, 0.01, el_edges)
                                   for k, v in o["L3_spectrum_10keV"].items()}
    o["L3_mean_T_all_MeV"] = ienergy.sum() / icount.sum()
    o["L3_mean_T_electrons_MeV"] = ienergy[..., 0].sum() / icount[..., 0].sum()
    o["L3_mean_T_compton_MeV"] = ienergy[0].sum() / icount[0].sum()
    o["L3_mean_T_parent_primary_MeV"] = ienergy[:, CAT_PRIMARY].sum() / icount[:, CAT_PRIMARY].sum()
    o["L3_mean_T_parent_scattered_MeV"] = ienergy[:, CAT_SCATTERED].sum() / icount[:, CAT_SCATTERED].sum()
    o["L3_count_fraction_by_process"] = icount.sum((1, 2)) / icount.sum()
    o["L3_positron_fraction"] = icount[..., 1].sum() / icount.sum()
    icos = tally(b, "initial_cos")                                    # (5, 2, 40)
    o["L3_cos_all"] = rebin(icos.sum((0, 1)), 4) / icos.sum()
    o["L3_cos_parent_primary"] = rebin(icos[:, 0].sum(0), 4) / icos[:, 0].sum()
    o["L3_cos_parent_scattered"] = rebin(icos[:, 1].sum(0), 4) / icos[:, 1].sum()
    for line, cat, e in (("line1", 0, src["line1_MeV"]), ("line2", 1, src["line2_MeV"])):
        h = isp[0, cat, 0]
        edge_bin = int(compton_edge(e) / 0.001)
        o[f"L3_compton_{line}_fraction_above_free_edge"] = h[edge_bin + 1:].sum() / h.sum()
        nz = np.nonzero(h)[0]
        o[f"L3_compton_{line}_max_T_MeV"] = (nz.max() + 1) * 0.001 if nz.size else 0.0

    # Layer 4: electrons slowing down. The CSDA-weighted estimator (DECISIONS D020)
    # is the reference; the plan's linear-loss estimator is reported alongside.
    csda = "electron_fluence_csda" in b["tallies"]
    sfx = "_csda" if csda else ""
    el = strip(tally(b, "electron_fluence" + sfx))                     # (1400) per 1 keV
    eld = strip(tally(b, "electron_fluence_delta" + sfx))
    pos = strip(tally(b, "positron_fluence" + sfx))
    elp = strip(tally(b, "electron_fluence_by_parent" + sfx))
    el_lin = strip(tally(b, "electron_fluence"))
    pos_lin = strip(tally(b, "positron_fluence"))
    o["L4_estimator"] = 1.0 if csda else 0.0
    o["L4lin_electron_fluence_10keV_per_MeV"] = rebin(el_lin, 10) * norm / 0.010
    o["L4lin_to_csda_fluence_10_20keV"] = el_lin[10:20].sum() / el[10:20].sum()
    o["L4lin_to_csda_fluence_above_100keV"] = el_lin[100:].sum() / el[100:].sum()
    above = slice(10, None)
    o["L4_electron_fluence_10keV_per_MeV"] = rebin(el, 10) * norm / 0.010
    o["L4_delta_fluence_10keV_per_MeV"] = rebin(eld, 10) * norm / 0.010
    o["L4_positron_fluence_10keV_per_MeV"] = rebin(pos, 10) * norm / 0.010
    o["L4_electron_fluence_comparison"] = rebin_edges(el, 0.0, 0.001, el_edges) * norm
    o["L4_electron_fluence_above_10keV_per_phi0"] = el[above].sum() * norm
    o["L4_delta_fraction_above_10keV"] = eld[above].sum() / el[above].sum()
    o["L4_positron_to_electron_fluence"] = pos[above].sum() / el[above].sum()
    o["L4_parent_scattered_fraction"] = elp[1, above].sum() / elp[:, above].sum()
    o["L4_mean_energy_above_10keV_MeV"] = (np.sum(el[above] * (np.arange(1400)[above] + 0.5) * 0.001)
                                           / el[above].sum())
    eang = tally(b, "electron_angular")
    o["L4_angular"] = rebin(eang, 4) / eang.sum()
    o["L4_angular_fine"] = eang / eang.sum()
    loss = tally(b, "charged_energy_loss")
    o["L4_radiative_to_collision"] = loss[0, 1] / loss[0, 0]

    # Layer 5: energy deposited
    d = s["edep"] * gy
    k = s["kerma_energy_transfer"] * gy
    kcol = (s["kerma_energy_transfer"] - s["kcol_radiative_loss"]) * gy
    o["L5_D_over_phi0_Gy_cm2"] = d
    o["L5_K_over_phi0_Gy_cm2"] = k
    o["L5_Kcol_over_phi0_Gy_cm2"] = kcol
    o["L5_beta"] = d / kcol
    o["L5_g"] = s["kcol_radiative_loss"] / s["kerma_energy_transfer"]
    ed = tally(b, "edep_by_parent")                                    # (5 parent, 3 particle)
    o["L5_dose_fraction_parent_scattered"] = ed[[2, 3]].sum() / ed.sum()
    o["L5_dose_fraction_parent_scattered_source"] = ed[2].sum() / ed.sum()
    o["L5_dose_fraction_parent_secondary_photon"] = ed[3].sum() / ed.sum()
    o["L5_dose_fraction_photon_local"] = ed[:, 0].sum() / ed.sum()
    o["L5_dose_fraction_unknown_parent"] = ed[4].sum() / ed.sum()
    o["L5_D_rel_sem_history_by_history"] = math.sqrt(max(s["edep_sq"] / n - (s["edep"] / n) ** 2, 0) / n) / (
        s["edep"] / n)
    # Fluence-to-dose closure: continuous energy loss reconstructed from the scored
    # electron and positron fluence with Geant4's restricted stopping power at the
    # production threshold, against the scored dose.
    cut = b["cuts_g4_water"]["e-_threshold_MeV"]
    key = "L_10keV" if abs(cut - 0.010) < 1e-6 else "L_5keV"
    rows = json.load(open(tables_json))["materials"]["G4_WATER"]["electron"]
    te = np.array([r["E"] for r in rows])
    lv = np.array([r[key] for r in rows])
    mid = (np.arange(el.size) + 0.5) * 0.001
    l_mid = np.exp(np.interp(np.log(mid), np.log(te), np.log(lv)))
    d_cont = np.sum((el + pos) * l_mid) / (vol * n) * J_PER_MEV * 1e3 / phi0
    o["L5_D_continuous_over_D"] = d_cont / d
    o["L5_D_continuous_over_D_linear"] = (np.sum((el_lin + pos_lin) * l_mid) / (vol * n) * J_PER_MEV * 1e3 / phi0) / d
    for name, basis in sp.items():
        sa = spencer_attix(el, 0.010, basis["water"], basis["air"])
        o[f"L5_s_w_air_{name}"] = sa["s_w_air"]
        o[f"L5_track_end_share_water_{name}"] = sa["water_track_end"] / (sa["water_integral"] + sa["water_track_end"])
        # Spencer-Attix dose to water from the scored fluence, against the scored dose
        d_sa = (sa["water_integral"] + sa["water_track_end"]) / (vol * n) * J_PER_MEV * 1e3 / phi0
        o[f"L5_D_SpencerAttix_over_D_{name}"] = d_sa / d
        sa_ep = spencer_attix(el + pos, 0.010, basis["water"], basis["air"])
        o[f"L5_s_w_air_{name}_with_positrons"] = sa_ep["s_w_air"]
        o[f"L5_s_w_air_{name}_linear_estimator"] = spencer_attix(el_lin, 0.010, basis["water"], basis["air"])["s_w_air"]
    return o


# ---------------------------------------------------------------- combination
def combine(per_batch):
    """Mean and standard error over batches, recursively for dicts and arrays."""
    first = per_batch[0]
    if isinstance(first, dict):
        return {k: combine([p[k] for p in per_batch]) for k in first}
    a = np.array(per_batch, dtype=float)
    n = a.shape[0]
    mean = a.mean(0)
    sem = a.std(0, ddof=1) / np.sqrt(n) if n > 1 else np.full_like(mean, np.nan)
    return {"mean": mean, "sem": sem, "n_batches": n}


# Scalars with history-by-history second moments: (result key, scalar, normalisation per batch)
HBH = [
    ("L5_D_over_phi0_Gy_cm2", "edep", lambda b: J_PER_MEV / (b["geometry"]["scoring_mass_g"] * 1e-3)
                                                / b["histories"] / b["source"]["phi0_per_history_cm-2"]),
    ("L5_K_over_phi0_Gy_cm2", "kerma_energy_transfer", lambda b: J_PER_MEV / (b["geometry"]["scoring_mass_g"] * 1e-3)
                                                               / b["histories"] / b["source"]["phi0_per_history_cm-2"]),
    ("L1_fluence_total_per_phi0", "photon_track_length_total",
     lambda b: 1.0 / (b["geometry"]["scoring_volume_cm3"] * b["histories"] * b["source"]["phi0_per_history_cm-2"])),
]


def add_history_variance(res, batches):
    """Second uncertainty estimate from history-by-history second moments, and a conservative one.

    The batch spread of 10 batches estimates the standard error to about +-24% (1 sigma); it can be
    much smaller by chance (the main run's batches agree better than their history variance allows,
    STATUS M1). For quantities with second moments, `sem_hbh` is computed from the variance of the
    per-history scores and `sem_conservative` is the larger of the two. For the primary fluence
    ratios, `sem_counting` is the Poisson error of the number of primary crossings (track length
    over thickness), a floor for the track-length estimator.
    """
    k = len(batches)
    for key, scalar, norm in HBH:
        if key not in res:
            continue
        var = 0.0
        for b in batches:
            s, n, c = b["scalars"], b["histories"], norm(b)
            mean_h = s[scalar] / n
            # batch value = c * sum over histories, so its variance is c^2 * n * var(per-history score)
            var += c * c * n * max(s[scalar + "_sq"] / n - mean_h * mean_h, 0.0)
        res[key]["sem_hbh"] = float(np.sqrt(var) / k)
        res[key]["sem_conservative"] = float(max(res[key]["sem"], res[key]["sem_hbh"]))
    t = batches[0]["geometry"]["scoring_thickness"]
    for line, cat in (("line1", 0), ("line2", 1)):
        crossings = sum(tally(b, "photon_track_length")[cat] for b in batches) / t
        for suffix in ("", "_xcom", "_tracking_mu"):
            key = f"L1_primary_{line}_measured_over_analytic{suffix}"
            if key in res:
                res[key]["sem_counting"] = float(res[key]["mean"] / np.sqrt(crossings))
                res[key]["sem_conservative"] = float(max(res[key]["sem"], res[key]["sem_counting"]))


def analyse_run(run_id, sp):
    batches = load_batches(run_id)
    if not batches:
        return None
    per = [observables(b, sp) for b in batches]
    res = combine(per)
    add_history_variance(res, batches)
    res["_meta"] = {
        "run_id": run_id,
        "batches": len(batches),
        "histories_total": int(sum(b["histories"] for b in batches)),
        "wall_time_s_total": float(sum(b["wall_time_s"] for b in batches)),
        "physics": batches[0]["physics"],
        "geometry": batches[0]["geometry"],
        "source": batches[0]["source"],
        "cuts_g4_water": batches[0]["cuts_g4_water"],
        "em_parameters": batches[0]["em_parameters"],
        "variance_reduction": batches[0]["variance_reduction"],
        "energy_balance_max_abs_MeV": max(b["energy_balance_max_abs_MeV"] for b in batches),
        "untagged_tracks": sum(b["scalars"]["untagged_tracks"] for b in batches),
        "geant4_version": batches[0]["geant4_version"],
    }
    return res


def compare(a, b, keys):
    """Difference b - a for scalar keys, in units of the combined standard error."""
    out = {}
    for k in keys:
        if k not in a or k not in b:
            continue
        ma, sa = float(a[k]["mean"]), float(a[k].get("sem_conservative", a[k]["sem"]))
        mb, sb = float(b[k]["mean"]), float(b[k].get("sem_conservative", b[k]["sem"]))
        comb = math.hypot(sa, sb)
        out[k] = {"reference": ma, "reference_sem": sa, "variant": mb, "variant_sem": sb,
                  "rel_diff": mb / ma - 1 if ma else None, "z": (mb - ma) / comb if comb else None}
    return out


def compare_spectrum(a, b, path):
    """Chi-square of two binned spectra (same normalization), bins with content in both."""
    def get(r):
        x = r
        for p in path:
            x = x[p]
        return np.asarray(x["mean"]), np.asarray(x["sem"])
    ma, sa = get(a)
    mb, sb = get(b)
    ok = (sa > 0) & (sb > 0)
    chi2 = float(np.sum(((ma - mb)[ok]) ** 2 / (sa[ok] ** 2 + sb[ok] ** 2)))
    return {"chi2": chi2, "ndf": int(ok.sum()), "chi2_per_ndf": chi2 / max(int(ok.sum()), 1)}


def targets(res):
    """Statistical targets of STUDY_PLAN Section 4.1."""
    def rel(k):
        return float(res[k].get("sem_conservative", res[k]["sem"]) / abs(res[k]["mean"]))

    def worst_bin(mean, sem):
        mean, sem = np.asarray(mean), np.asarray(sem)
        big = mean >= 0.01 * mean.sum()
        return float(np.max(sem[big] / mean[big])) if big.any() else None

    spectra = {
        "photon fluence, 10 keV bins": res["L1_photon_fluence_10keV"]["total"],
        "photon fluence, comparison grid": res["L1_photon_fluence_comparison"]["total"],
        "electron fluence, 10 keV bins": res["L4_electron_fluence_10keV_per_MeV"],
        "electron fluence, comparison grid": res["L4_electron_fluence_comparison"],
        "initial electron spectrum, 10 keV bins": res["L3_spectrum_10keV"]["all"],
        "initial electron spectrum, comparison grid": res["L3_spectrum_comparison"]["all"],
    }
    out = {
        "D/Φ₀, relative standard error": {"target": 0.001, "achieved": rel("L5_D_over_phi0_Gy_cm2")},
        "β, relative standard error": {"target": 0.002, "achieved": rel("L5_beta")},
    }
    for name, sp in spectra.items():
        out[f"{name[0].upper() + name[1:]}: worst bin holding ≥ 1% of the total"] = {
            "target": 0.01, "achieved": worst_bin(sp["mean"], sp["sem"])}
    for v in out.values():
        v["met"] = v["achieved"] is not None and v["achieved"] <= v["target"]
    return out


def pdd(run_id):
    """PDD(5) = 100 D(5 cm) / D(d_max) per batch, D(5 cm) over 4.9-5.1 cm, D_max from a parabola near the peak."""
    vals, dmax_depths, profiles = [], [], []
    for b in load_batches(run_id):
        dz = strip(tally(b, "depth_edep"))                             # 100 slabs of 1 mm
        z = (np.arange(dz.size) + 0.5) * 0.1
        i = int(np.argmax(dz))
        lo, hi = max(i - 3, 0), min(i + 4, dz.size)
        c = np.polyfit(z[lo:hi], dz[lo:hi], 2)
        zmax = -c[1] / (2 * c[0])
        dmax = np.polyval(c, zmax)
        d5 = dz[49:51].mean()
        vals.append(100 * d5 / dmax)
        dmax_depths.append(zmax)
        profiles.append(dz / dmax * 100)
    if not vals:
        return None
    v, zz, pr = np.array(vals), np.array(dmax_depths), np.array(profiles)
    n = len(v)
    return {"PDD5": {"mean": v.mean(), "sem": v.std(ddof=1) / np.sqrt(n), "n_batches": n},
            "d_max_cm": {"mean": zz.mean(), "sem": zz.std(ddof=1) / np.sqrt(n), "n_batches": n},
            "profile_percent": {"mean": pr.mean(0), "sem": pr.std(0, ddof=1) / np.sqrt(n), "n_batches": n},
            "depth_cm": (np.arange(pr.shape[1]) + 0.5) * 0.1}


DIAGNOSTIC_RUNS = {   # primary-fluence diagnostics (STATUS M1): transport, variance reduction, geometry
    "M1-check-primary-is-off": "photon only, no VR, r 4 cm t 1 cm",
    "M1-check-primary-is-on": "photon only, parallel importance, r 4 cm t 1 cm",
    "M1-check-primary-full-r4": "full transport, no VR, r 4 cm t 1 cm",
    "M1-check-primary-is-on-prodgeom": "photon only, parallel importance, r 1 cm t 0.2 cm",
    "M1-check-primary-ismass-prodgeom": "photon only, mass-geometry importance, r 1 cm t 0.2 cm",
}


def primary_ratios(run_id):
    """Primary fluence over the analytic value (model mu and tracking mu), both lines averaged, per batch."""
    batches = load_batches(run_id)
    if not batches:
        return None
    tr = yaml.safe_load(open(TRANSMISSION))["lines"] if TRANSMISSION.exists() else None
    out = {"model_mu": [], "tracking_mu": []}
    for b in batches:
        g, src, n = b["geometry"], b["source"], b["histories"]
        tl = tally(b, "photon_track_length")
        for kind in out:
            vals = []
            for line, cat in (("line1", 0), ("line2", 1)):
                mu = g4_mu_water(TABLES, src[f"{line}_MeV"])[0] if kind == "model_mu" else tr[line]["mu_measured_cm-1"]
                frac = b["scalars"][f"histories_{line}"] / n
                expected = frac * fluence_cylinder_average(g["scoring_radius"], g["scoring_thickness"],
                                                           g["scoring_depth"], src["ssd"], src["solid_angle_sr"], mu)
                vals.append(tl[cat] / (g["scoring_volume_cm3"] * n) / expected)
            out[kind].append(np.mean(vals))
    k = len(batches)
    t = batches[0]["geometry"]["scoring_thickness"]
    crossings = sum(tally(b, "photon_track_length")[:2].sum() for b in batches) / t
    res = {}
    for kind, v in out.items():
        mean = float(np.mean(v))
        sem = float(np.std(v, ddof=1) / np.sqrt(k)) if k > 1 else None
        counting = mean / float(np.sqrt(crossings))
        res[kind] = {"mean": mean, "sem": sem, "sem_counting": counting,
                     "sem_conservative": max(sem or 0.0, counting), "n_batches": k}
    return res


def diagnostics():
    d = {}
    if TRANSMISSION.exists():
        tr = yaml.safe_load(open(TRANSMISSION))
        d["transmission"] = {}
        for line, rec in tr["lines"].items():
            mu_model = g4_mu_water(TABLES, rec["energy_MeV"])[0]
            d["transmission"][line] = {"energy_MeV": rec["energy_MeV"], "mu_tracking_cm-1": rec["mu_measured_cm-1"],
                                       "mu_tracking_sem_cm-1": rec["mu_measured_sem_cm-1"], "mu_model_cm-1": mu_model,
                                       "ratio": rec["mu_measured_cm-1"] / mu_model,
                                       "ratio_sem": rec["mu_measured_sem_cm-1"] / mu_model,
                                       "photons": rec["incident"]}
    d["primary_fluence"] = {}
    for run_id, label in [("M1-R0-option4", "main run: full transport, importance + roulette, r 1 cm t 0.2 cm"),
                          ("M1-R0-option4-rep2", "replicate of the main run (new seeds)"),
                          ("M1-R0-option4-S2-5keV", "S2: as main run, 5 keV cuts"),
                          ("M1-R0-option4-S1-half", "S1: as main run, r 0.5 cm t 0.1 cm"),
                          ("M1-vrcheck-analog", "full transport, no VR, r 1 cm t 0.2 cm"),
                          ("M1-thinslab-option4-v2", "thin slab: full transport, no VR, r 4 cm t 0.05 cm at 0.5 cm")
                          ] + list(DIAGNOSTIC_RUNS.items()) + (
                          [(POOLED, "main run and replicate pooled (20 batches)")]
                          if len(list((RUNS / "M1-R0-option4-rep2").glob("batch_*/output.json"))) >= 10 else []):
        pr = primary_ratios(run_id)
        if pr:
            d["primary_fluence"][run_id] = {"label": label, **pr}
    return d


POOLED = "M1-R0-option4+M1-R0-option4-rep2"


def pooling_decision(runs):
    """DECISIONS D022: pool the replicate with the main run if both agree within 3 combined
    standard errors on D/Phi0 and on the primary fluence ratio (tracking mu); otherwise keep them apart."""
    a, b = runs.get("M1-R0-option4"), runs.get("M1-R0-option4-rep2")
    out = {"rule": "DECISIONS D022", "pooled": False, "reference_run": "M1-R0-option4"}
    if not b or b["_meta"]["batches"] < 10:
        out["reason"] = "replicate not complete"
        return out
    zs = {}
    for k in ("L5_D_over_phi0_Gy_cm2", "L1_primary_line1_measured_over_analytic_tracking_mu",
              "L1_primary_line2_measured_over_analytic_tracking_mu"):
        sa, sb = a[k].get("sem_conservative", a[k]["sem"]), b[k].get("sem_conservative", b[k]["sem"])
        zs[k] = float((b[k]["mean"] - a[k]["mean"]) / math.hypot(sa, sb))
    out["z"] = zs
    if all(abs(z) < 3 for z in zs.values()):
        out.update({"pooled": True, "reference_run": POOLED, "reason": "agree within 3 sigma"})
    else:
        out["reason"] = "disagree on " + ", ".join(k for k, z in zs.items() if abs(z) >= 3)
    return out


def to_jsonable(x):
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return [to_jsonable(v) for v in x.tolist()]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


SCALAR_KEYS_S = [
    "L5_D_over_phi0_Gy_cm2", "L5_K_over_phi0_Gy_cm2", "L5_Kcol_over_phi0_Gy_cm2", "L5_beta", "L5_g",
    "L5_s_w_air_icru37", "L5_s_w_air_icru90", "L5_s_w_air_geant4", "L5_s_w_air_pegs4",
    "L5_D_continuous_over_D", "L5_D_SpencerAttix_over_D_geant4", "L5_dose_fraction_parent_scattered",
    "L4_electron_fluence_above_10keV_per_phi0", "L4_delta_fraction_above_10keV", "L4_radiative_to_collision",
    "L4_mean_energy_above_10keV_MeV", "L1_fluence_total_per_phi0", "L1_scattered_fraction_fluence",
    "L1_mean_energy_all_MeV", "L3_mean_T_all_MeV",
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sp = bases(TABLES, PEGS4_DIR)
    runs = {k: analyse_run(k, sp) for k in ("M1-R0-option4", "M1-R0-option4-rep2", "M1-R0-option4-S2-5keV",
                                           "M1-R0-option4-S1-half", "M1-vrcheck-analog", "M1-thinslab-option4-v2")}
    result = {"runs": {k: v for k, v in runs.items() if v is not None}}
    result["pooling"] = pooling_decision(runs)
    if result["pooling"]["pooled"]:
        runs[POOLED] = analyse_run(POOLED, sp)
        result["runs"][POOLED] = runs[POOLED]
    main_res = runs[result["pooling"]["reference_run"]]
    if main_res:
        result["targets"] = targets(main_res)
        for label, key, layers in (("S2_cut_independence", "M1-R0-option4-S2-5keV", None),
                                   ("S1_volume_size", "M1-R0-option4-S1-half", None),
                                   ("vr_check_analog", "M1-vrcheck-analog", None)):
            other = runs.get(key)
            if other:
                result[label] = {
                    "scalars": compare(main_res, other, SCALAR_KEYS_S),
                    "electron_fluence_shape": compare_spectrum(main_res, other, ["L4_electron_fluence_comparison"]),
                    "photon_fluence_shape": compare_spectrum(main_res, other,
                                                             ["L1_photon_fluence_comparison", "total"]),
                }
    result["diagnostics"] = diagnostics()
    ev = summarize_events(RUNS / "M1-R0-option4")   # the main run's sample (100 000 interactions)
    if ev:
        result["event_sample"] = ev
    p = pdd("M1-R0-option4-pdd")
    if p:
        result["pdd"] = p
    result["stopping_power_bases"] = {k: {kk: vv for kk, vv in v.items() if kk in ("I_water_eV", "I_air_eV", "source")}
                                      for k, v in sp.items()}
    json.dump(to_jsonable(result), open(OUT / "M1_results.json", "w"), indent=1)
    print(f"wrote {OUT / 'M1_results.json'}")


if __name__ == "__main__":
    main()
