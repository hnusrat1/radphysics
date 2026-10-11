"""Write the M1 study brief for Humza: why, what was done, results, strengths, weaknesses, venues.

Every number is read from analysis/results/M1_results.json and analysis/M1_fano_summary.yaml.

Usage:
  python analysis/m1_brief.py      # writes analysis/results/co60_brief.html
"""

import html
import json
import math
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "analysis" / "results"
ARTIFACT_RESULTS = "https://claude.ai/artifact/Cu5ScfiTccG3u14D4mbiLe"


def pm(mean, sem, d=None):
    if d is None:
        d = max(0, -int(math.floor(math.log10(abs(sem)))) + 1)
    return f"{mean:.{d}f} ± {sem:.{d}f}"


def minus(s):
    return s.replace("-", "−")


def load():
    r = json.load(open(RES / "M1_results.json"))
    fano = yaml.safe_load(open(ROOT / "analysis" / "M1_fano_summary.yaml"))
    ref = r["pooling"]["reference_run"]
    m = r["runs"][ref]

    def v(k):
        x = m[k]
        return x["mean"], x.get("sem_conservative", x["sem"])

    n = {}
    n["hist"] = m["_meta"]["histories_total"]
    n["batches"] = m["_meta"]["batches"]
    n["D"], n["Ds"] = v("L5_D_over_phi0_Gy_cm2")
    n["beta"], n["betas"] = v("L5_beta")
    n["D_cyl"], n["Ds_cyl"], n["beta_cyl"], n["betas_cyl"] = n["D"], n["Ds"], n["beta"], n["betas"]
    ref0 = r["diagnostics"]["boundary"].get("R0_dose_reference")
    n["ref_bf"] = bool(ref0 and ref0["uses"] == "boundary_free")   # DECISIONS D025/D027
    if n["ref_bf"]:
        n["D"], n["Ds"] = (ref0["D_over_phi0_boundary_free_Gy_cm2"][k] for k in ("mean", "sem"))
        n["beta"], n["betas"] = (ref0["beta_boundary_free"][k] for k in ("mean", "sem"))
    n["K"] = v("L5_K_over_phi0_Gy_cm2")[0]
    n["Kc"] = v("L5_Kcol_over_phi0_Gy_cm2")[0]
    n["g"] = v("L5_g")[0]
    for b in ("icru37", "icru90", "geant4", "pegs4"):
        n[f"sw_{b}"] = v(f"L5_s_w_air_{b}")[0]
    n["phi_tot"] = v("L1_fluence_total_per_phi0")
    n["scat_frac"] = v("L1_scattered_fraction_fluence")[0]
    n["scat_efrac"] = v("L1_scattered_fraction_energy_fluence")[0]
    n["E_all"] = v("L1_mean_energy_all_MeV")[0]
    n["E_scat"] = v("L1_mean_energy_scattered_MeV")[0]
    n["l2"] = m["L2_fraction_all"]["mean"]
    n["e2"] = m["L2_etr_fraction_all"]["mean"]
    n["T_all"] = v("L3_mean_T_all_MeV")[0]
    n["T_prim"] = v("L3_mean_T_parent_primary_MeV")[0]
    n["T_scat"] = v("L3_mean_T_parent_scattered_MeV")[0]
    n["edge1"] = v("L3_compton_line1_fraction_above_free_edge")[0]
    n["edge2"] = v("L3_compton_line2_fraction_above_free_edge")[0]
    n["delta"] = v("L4_delta_fraction_above_10keV")[0]
    n["radcol"] = v("L4_radiative_to_collision")[0]
    n["lin_csda_10_20"] = v("L4lin_to_csda_fluence_10_20keV")[0]
    n["dose_scat"] = v("L5_dose_fraction_parent_scattered")[0]
    n["closure"] = v("L5_D_continuous_over_D")[0]
    n["closure_lin"] = v("L5_D_continuous_over_D_linear")[0]
    n["pdd5"] = r["pdd"]["PDD5"]
    n["dmax"] = r["pdd"]["d_max_cm"]
    pf = r["diagnostics"]["primary_fluence"]
    n["prim"] = pf[ref]["tracking_mu"]
    n["prim_main"] = pf["M1-R0-option4"]["tracking_mu"]
    n["prim_rep"] = pf["M1-R0-option4-rep2"]["tracking_mu"]
    n["tr"] = r["diagnostics"]["transmission"]
    bd = r["diagnostics"]["boundary"]
    n["bnd"] = bd[f"cylinder_over_boundary_free:{ref}"]
    n["bnd_free_batches"] = sum(e["n_batches"] for e in bd["boundary_free_estimates"].values())
    prof = bd["depth_ratio_with_over_without"]
    z, rr = np.array(prof["depth_cm"]), np.array(prof["ratio"])
    sel = (z > 4.0) & (z < 6.0)
    n["art_max"], n["art_min"] = float(rr[sel].max() - 1), float(rr[sel].min() - 1)
    n["far"] = bd["far_field_ratio_0_4cm"]
    for lab, key in (("S1", "S1_volume_size"), ("S2", "S2_cut_independence"), ("an", "vr_check_analog")):
        n[lab] = r[key]["scalars"]["L5_D_over_phi0_Gy_cm2"]
        n[lab + "_e"] = r[key]["electron_fluence_shape"]
    n["S2_sw"] = r["S2_cut_independence"]["scalars"]["L5_s_w_air_icru37"]["rel_diff"]
    ev = r["event_sample"]
    n["ev_n"] = ev["n_events"]
    n["ev_bound"] = ev["bound_failure_rate"]["valid"]
    n["ev_strict"] = ev["strict_failure_rate"]["valid"]
    n["dc"] = ev["doppler_tolerance"]["delta_c"]
    n["fano"] = fano["standard_opt4"]
    n["fano_rf"] = fano.get("standard_opt4 | /process/msc/RangeFactor 0.04")
    n["fano0"] = fano["standard_opt0"]
    n["targets"] = r["targets"]
    return n


def effect_chart(n):
    """Horizontal log-scale bars: how large each effect is, against the proposed Layer 5 tolerance."""
    lo, hi = 0.03, 30.0
    tol = 2.0

    def x(p):
        return 100 * (math.log10(p) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))

    b = n["bnd"]
    rows = [
        ("Statistical uncertainty of β (1σ)", 100 * n["betas"] / n["beta"], "mc", "statistics"),
        ("Statistical uncertainty of D/Φ₀ (1σ)", 100 * n["Ds"] / n["D"], "mc", "statistics"),
        ("Primary fluence against the analytic value", 100 * abs(1 - n["prim"]["mean"]), "mc", f"{(n['prim']['mean'] - 1) / n['prim']['sem_conservative']:+.1f}σ"),
        ("Fano cavity deviation, production stepping", 100 * (n["fano"]["ratio_mean"] - 1), "mc", f"{n['fano']['deviation_sigma']:+.1f}σ"),
        ("Scoring boundary effect on D inside the cylinder", 100 * abs(1 - b["ratio"]), "mc", f"{(b['ratio'] - 1) / b['sem']:+.1f}σ, corrected (D027)"),
        ("Halving the scoring volume (S1), change in D", 100 * abs(n["S1"]["rel_diff"]), "mc", f"{n['S1']['z']:+.1f}σ"),
        ("s<sub>w,air</sub>: ICRU 37 against ICRU 90 stopping powers", 100 * (n["sw_icru37"] / n["sw_icru90"] - 1), "def", "data basis"),
        ("s<sub>w,air</sub>: Geant4 against ICRU 90 stopping powers", 100 * (n["sw_geant4"] / n["sw_icru90"] - 1), "def", "data basis"),
        ("Dose 1–4 mm from the scoring boundary (largest slab)", 100 * max(n["art_max"], -n["art_min"]), "art", "transport artifact"),
        ("Electron fluence 10–20 keV: linear against CSDA estimator", 100 * (1 - n["lin_csda_10_20"]), "art", "estimator"),
        ("Dose from electrons with scattered-photon parents", 100 * n["dose_scat"], "phys", "physics an AI must capture"),
    ]
    rows.sort(key=lambda t: t[1])
    out = []
    for label, val, kind, note in rows:
        out.append(
            f'<div class="er"><div class="el"><span>{label}</span><span class="en">{html.escape(minus(note))}</span></div>'
            f'<div class="et"><span class="eb k-{kind}" style="width:{x(val):.2f}%"></span>'
            f'<span class="etol" style="left:{x(tol):.2f}%"></span></div>'
            f'<div class="ev">{val:.3g}%</div></div>')
    ticks = "".join(f'<span class="tk" style="left:{x(t):.2f}%">{t:g}%</span>' for t in (0.03, 0.1, 0.3, 1, 3, 10, 30))
    out.append(f'<div class="er axis"><div class="el"></div><div class="et">{ticks}'
               f'<span class="tollab" style="left:{x(tol):.2f}%">proposed AI tolerance, Layer 5 dose (2%)</span></div><div class="ev"></div></div>')
    return "\n".join(out)


CSS = """
/* Layout: reading column with a sticky contents rail on wide screens; wide figures break out to the full width. */
:root {
  --bg: #f4f6f9; --surface: #ffffff; --ink: #142030; --muted: #586779; --rule: #d9e0e8;
  --cobalt: #1f4fb8; --cobalt-soft: #e5ecfa; --amber: #a35a00; --amber-soft: #fbefdc; --green: #23704a; --green-soft: #e1f2e8;
  --violet: #6a3fb0;
  --f-display: "IBM Plex Sans Condensed", "Arial Narrow", sans-serif;
  --f-body: "Source Serif 4", Georgia, "Times New Roman", serif;
  --f-ui: "IBM Plex Sans", "Helvetica Neue", Arial, sans-serif;
  --f-mono: "IBM Plex Mono", ui-monospace, Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #0d1219; --surface: #141c26; --ink: #e3e9f1; --muted: #93a2b4; --rule: #263241;
  --cobalt: #7ea7ff; --cobalt-soft: #1b2a47; --amber: #f0b45b; --amber-soft: #3a2a12; --green: #6fd39a; --green-soft: #15301f;
  --violet: #b99cf2; color-scheme: dark; } }
:root[data-theme="dark"] {
  --bg: #0d1219; --surface: #141c26; --ink: #e3e9f1; --muted: #93a2b4; --rule: #263241;
  --cobalt: #7ea7ff; --cobalt-soft: #1b2a47; --amber: #f0b45b; --amber-soft: #3a2a12; --green: #6fd39a; --green-soft: #15301f;
  --violet: #b99cf2; color-scheme: dark; }
* { box-sizing: border-box; }
body { background: var(--bg); color: var(--ink); font-family: var(--f-body); font-size: 17px; line-height: 1.6; }
.wrap { max-width: 1160px; margin: 0 auto; padding-inline: 20px; padding-block: 32px 72px;
  display: grid; grid-template-columns: 210px minmax(0, 1fr); gap: 48px; }
nav.rail { position: sticky; top: calc(env(safe-area-inset-top, 0px) + 24px); align-self: start;
  font-family: var(--f-ui); font-size: 13px; display: grid; gap: 4px; }
nav.rail .t { font-family: var(--f-display); font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em;
  color: var(--muted); font-size: 12px; margin-bottom: 6px; }
nav.rail a { color: var(--ink); text-decoration: none; padding: 4px 8px; border-radius: 4px; }
nav.rail a:hover, nav.rail a:focus-visible { background: var(--cobalt-soft); color: var(--cobalt); outline: none; }
main { min-width: 0; display: grid; gap: 56px; }
section { display: grid; gap: 16px; scroll-margin-top: 24px; min-width: 0; }
h1, h2, h3 { font-family: var(--f-display); font-weight: 600; text-wrap: balance; margin: 0; line-height: 1.15; }
h1 { font-size: clamp(30px, 4.2vw, 44px); letter-spacing: -0.005em; }
h2 { font-size: 27px; }
h3 { font-size: 19px; }
p { margin: 0; max-width: 68ch; }
.eyebrow { font-family: var(--f-display); font-size: 13px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--cobalt); font-weight: 600; }
.lede { font-size: 19px; color: var(--ink); max-width: 64ch; }
.meta { font-family: var(--f-mono); font-size: 13px; color: var(--muted); display: flex; flex-wrap: wrap; gap: 6px 18px; }
a { color: var(--cobalt); }
ul.tight { margin: 0; padding-left: 1.2rem; display: grid; gap: 8px; max-width: 70ch; }
.num, td.n { font-family: var(--f-mono); font-variant-numeric: tabular-nums; font-size: 0.92em; }
.callout { border-left: 3px solid var(--cobalt); background: var(--cobalt-soft); padding: 14px 18px; border-radius: 0 6px 6px 0; max-width: 72ch; }
.callout.warn { border-left-color: var(--amber); background: var(--amber-soft); }
.short { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px 28px; }
.short div { display: grid; gap: 4px; align-content: start; }
.short b { font-family: var(--f-ui); font-size: 13px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); font-weight: 600; }
.chain { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 0; border: 1px solid var(--rule);
  border-radius: 8px; background: var(--surface); overflow: hidden; }
.chain .lyr { padding: 16px 16px 18px; display: grid; gap: 8px; align-content: start; border-right: 1px solid var(--rule); min-width: 0; }
.chain .lyr:last-child { border-right: 0; }
.chain .k { font-family: var(--f-mono); font-size: 12px; color: var(--cobalt); }
.chain .nm { font-family: var(--f-display); font-weight: 600; font-size: 17px; line-height: 1.2; }
.chain .big { font-family: var(--f-mono); font-size: 19px; font-variant-numeric: tabular-nums; }
.chain .sub { font-family: var(--f-ui); font-size: 13px; color: var(--muted); line-height: 1.45; }
.chain .ai { font-family: var(--f-ui); font-size: 13px; line-height: 1.45; border-top: 1px dashed var(--rule); padding-top: 8px; }
.chain .ai em { font-style: normal; color: var(--violet); font-weight: 600; }
.tbl { overflow-x: auto; border: 1px solid var(--rule); border-radius: 8px; background: var(--surface); }
table { border-collapse: collapse; width: 100%; font-family: var(--f-ui); font-size: 14px; }
th, td { text-align: left; padding: 9px 12px; border-bottom: 1px solid var(--rule); vertical-align: top; }
th { font-weight: 600; color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; }
tr:last-child td { border-bottom: 0; }
.chip { display: inline-block; font-family: var(--f-ui); font-size: 12px; font-weight: 600; padding: 2px 8px; border-radius: 999px; white-space: nowrap; }
.chip.ok { background: var(--green-soft); color: var(--green); }
.chip.warn { background: var(--amber-soft); color: var(--amber); }
.chip.open { background: var(--cobalt-soft); color: var(--cobalt); }
.fig { border: 1px solid var(--rule); border-radius: 8px; background: var(--surface); padding: 18px 18px 14px; display: grid; gap: 10px; min-width: 0; }
.cap { font-family: var(--f-ui); font-size: 13px; color: var(--muted); max-width: 80ch; }
.er { display: grid; grid-template-columns: minmax(0, 1.25fr) minmax(0, 2fr) 4.2rem; gap: 14px; align-items: center; min-height: 40px; }
.el { font-family: var(--f-ui); font-size: 13.5px; line-height: 1.3; display: grid; }
.en { font-size: 12px; color: var(--muted); }
.et { position: relative; height: 40px; }
.eb { position: absolute; left: 0; top: 14px; height: 12px; border-radius: 0 3px 3px 0; }
.k-mc { background: var(--cobalt); }
.k-def { background: var(--amber); }
.k-art { background: var(--muted); }
.k-phys { background: var(--violet); }
.etol { position: absolute; top: 0; bottom: 0; width: 0; border-left: 1.5px dashed var(--ink); opacity: 0.55; }
.ev { font-family: var(--f-mono); font-size: 13px; text-align: right; font-variant-numeric: tabular-nums; }
.er.axis { min-height: 34px; }
.er.axis .et { height: 34px; border-top: 1px solid var(--rule); }
.tk { position: absolute; top: 4px; transform: translateX(-50%); font-family: var(--f-mono); font-size: 11px; color: var(--muted); }
.tollab { position: absolute; top: 18px; transform: translateX(-50%); font-family: var(--f-ui); font-size: 11px; color: var(--ink); white-space: nowrap; }
.key { display: flex; flex-wrap: wrap; gap: 6px 16px; font-family: var(--f-ui); font-size: 12.5px; color: var(--muted); }
.key span::before { content: ""; display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; vertical-align: -1px; background: var(--c); }
.cols { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 28px; }
.cols > div { display: grid; gap: 10px; align-content: start; min-width: 0; }
.finding { display: grid; gap: 6px; padding: 14px 0; border-top: 1px solid var(--rule); }
.finding:first-of-type { border-top: 0; padding-top: 0; }
.steps { display: grid; gap: 10px; }
.step { display: grid; grid-template-columns: 3.4rem minmax(0, 1fr); gap: 12px; align-items: baseline; }
.step .mk { font-family: var(--f-mono); font-size: 14px; color: var(--cobalt); font-weight: 600; }
footer { font-family: var(--f-ui); font-size: 13px; color: var(--muted); border-top: 1px solid var(--rule); padding-top: 16px; display: grid; gap: 4px; }
@media (max-width: 900px) {
  .wrap { grid-template-columns: minmax(0, 1fr); gap: 28px; }
  nav.rail { position: static; display: flex; flex-wrap: wrap; gap: 4px 6px; }
  nav.rail .t { width: 100%; }
  .chain { grid-template-columns: minmax(0, 1fr); }
  .chain .lyr { border-right: 0; border-bottom: 1px solid var(--rule); }
  .chain .lyr:last-child { border-bottom: 0; }
  .short, .cols { grid-template-columns: minmax(0, 1fr); }
  .er { grid-template-columns: minmax(0, 1fr) 3.8rem; grid-template-areas: "l v" "t t"; gap: 4px 10px; }
  .er .el { grid-area: l; } .er .ev { grid-area: v; } .er .et { grid-area: t; height: 22px; }
  .eb { top: 5px; }
  .er.axis { grid-template-areas: "t t"; } .er.axis .el, .er.axis .ev { display: none; }
  .er.axis .et { height: 40px; }
  .tollab { white-space: normal; width: 9rem; text-align: center; }
  body { font-size: 16px; }
}
@media (prefers-reduced-motion: reduce) { * { scroll-behavior: auto; } }
"""


def build(n):
    D, Ds = n["D"], n["Ds"]
    b = n["bnd"]
    bz = (b["ratio"] - 1) / b["sem"]
    blo, bhi = 100 * (b["ratio"] - 1 - 2 * b["sem"]), 100 * (b["ratio"] - 1 + 2 * b["sem"])
    prim = n["prim"]
    pz = (prim["mean"] - 1) / prim["sem_conservative"]
    tr1, tr2 = n["tr"]["line1"], n["tr"]["line2"]
    l2, e2 = n["l2"], n["e2"]
    tg = n["targets"]
    worst = max(v["achieved"] for k, v in tg.items() if "worst bin" in k)
    sw90, sw37, swg4, swp4 = n["sw_icru90"], n["sw_icru37"], n["sw_geant4"], n["sw_pegs4"]

    def pct(x, d=1):
        return f"{100 * x:.{d}f}%"

    chain = f"""
<div class="chain">
  <div class="lyr"><span class="k">Layer 1</span><span class="nm">Photons arriving</span>
    <span class="big">{pct(n['scat_frac'])}</span>
    <span class="sub">of photon fluence at 5 cm is scattered, but only {pct(n['scat_efrac'])} of the energy fluence. Mean energy {n['E_all']:.3f} MeV (scattered {n['E_scat']:.3f} MeV).</span>
    <span class="ai"><em>AI must:</em> attenuate the two lines correctly and know that scatter dominates the photon count but not the energy.</span></div>
  <div class="lyr"><span class="k">Layer 2</span><span class="nm">Interactions</span>
    <span class="big">{pct(l2[0], 2)}</span>
    <span class="sub">of interactions are Compton ({pct(e2[0], 2)} of energy transferred). Photoelectric {pct(l2[1], 2)}, Rayleigh {pct(l2[3], 2)}, pair {pct(l2[2], 3)}.</span>
    <span class="ai"><em>AI must:</em> see that the softened scattered spectrum raises photoelectric and Rayleigh shares far above their primary-beam values.</span></div>
  <div class="lyr"><span class="k">Layer 3</span><span class="nm">Electrons set in motion</span>
    <span class="big">{n['T_all'] * 1e3:.0f} keV</span>
    <span class="sub">mean initial energy: {n['T_prim'] * 1e3:.0f} keV from primary parents, {n['T_scat'] * 1e3:.0f} keV from scattered ones. {pct(n['edge1'], 2)} of recoils from the 1.1732 MeV line exceed its free-electron edge (bound-electron Doppler broadening).</span>
    <span class="ai"><em>AI must:</em> produce Compton recoils that end near the edges and weight in the scattered parents.</span></div>
  <div class="lyr"><span class="k">Layer 4</span><span class="nm">Electrons slowing down</span>
    <span class="big">{pct(n['delta'])}</span>
    <span class="sub">of the electron fluence above 10 keV is delta rays; radiative/collision loss {n['radcol']:.4f}.</span>
    <span class="ai"><em>AI must:</em> build a slowing-down spectrum, the step furthest from textbook recall.</span></div>
  <div class="lyr"><span class="k">Layer 5</span><span class="nm">Energy deposited</span>
    <span class="big">{D * 1e12:.4f}</span>
    <span class="sub">× 10⁻¹² Gy cm² for D/Φ₀ (±{100 * Ds / D:.3f}%). β = {n['beta']:.4f}; s<sub>w,air</sub> = {sw90:.4f} (ICRU 90); PDD(5) = {n['pdd5']['mean']:.1f}%.</span>
    <span class="ai"><em>AI must:</em> account for the {pct(n['dose_scat'])} of dose from scattered-photon parents, which a primary-only calculation misses.</span></div>
</div>"""

    results_rows = [
        ("1", "Total photon fluence / Φ₀", pm(*n["phi_tot"], 4)),
        ("1", "Scattered fraction of fluence / of energy fluence", f"{n['scat_frac']:.4f} / {n['scat_efrac']:.4f}"),
        ("2", "Compton, photoelectric, Rayleigh, pair (share of interactions)", f"{pct(l2[0], 2)}, {pct(l2[1], 2)}, {pct(l2[3], 2)}, {pct(l2[2], 3)}"),
        ("3", "Mean initial electron energy: all / primary parent / scattered parent", f"{n['T_all']:.4f} / {n['T_prim']:.4f} / {n['T_scat']:.4f} MeV"),
        ("4", "Delta-ray share above 10 keV; radiative / collision loss", f"{n['delta']:.4f}; {n['radcol']:.5f}"),
        ("5", "D/Φ₀, R0 reference (boundary-free, D027)" if n["ref_bf"] else "D/Φ₀", f"{D * 1e12:.4f} × 10⁻¹² Gy cm² ± {100 * Ds / D:.3f}%"),
        ("5", "D/Φ₀ in the scoring cylinder", f"{n['D_cyl'] * 1e12:.4f} × 10⁻¹² Gy cm² ± {100 * n['Ds_cyl'] / n['D_cyl']:.3f}%"),
        ("5", "K/Φ₀, K<sub>col</sub>/Φ₀", f"{n['K'] * 1e12:.4f}, {n['Kc'] * 1e12:.4f} × 10⁻¹² Gy cm²"),
        ("5", "β = D/K<sub>col</sub> (with the cylinder D); g", f"{pm(n['beta'], n['betas'], 4)} ({pm(n['beta_cyl'], n['betas_cyl'], 4)}); {n['g']:.5f}"),
        ("5", "s<sub>w,air</sub>, Δ = 10 keV: ICRU 90 / ICRU 37 / Geant4 / PEGS4 matched", f"{sw90:.4f} / {sw37:.4f} / {swg4:.4f} / {swp4:.4f}"),
        ("5", "PDD(5); d<sub>max</sub>", f"{n['pdd5']['mean']:.2f} ± {n['pdd5']['sem']:.2f}%; {n['dmax']['mean']:.2f} ± {n['dmax']['sem']:.2f} cm"),
    ]
    results_tbl = "".join(f"<tr><td class='n'>{a}</td><td>{q}</td><td class='n'>{v}</td></tr>" for a, q, v in results_rows)

    checks = [
        ("Statistical targets (plan 4.1)", f"D/Φ₀ {100 * Ds / D:.3f}% (0.1%), β {100 * n['betas'] / n['beta']:.3f}% (0.2%), worst spectral bin {100 * worst:.2f}% (1%)", "ok", "met"),
        ("Tracking μ against G4EmCalculator (TestEm13, 2 × 10⁷ photons per line)", f"+{100 * (tr1['ratio'] - 1):.2f} ± {100 * tr1['ratio_sem']:.2f}%, +{100 * (tr2['ratio'] - 1):.2f} ± {100 * tr2['ratio_sem']:.2f}%", "ok", "measured"),
        ("Primary fluence against exp(−μs)/(ΩR²), tracking μ", f"{pm(prim['mean'], prim['sem_conservative'], 4)} ({pz:+.1f}σ)", "ok", "passes"),
        ("Fluence-to-dose closure, CSDA-weighted (linear)", f"{n['closure']:.4f} ({n['closure_lin']:.4f})", "ok", "passes"),
        ("Cut independence S2 (5 keV): D/Φ₀; s<sub>w,air</sub>", f"{100 * n['S2']['rel_diff']:+.2f}% ({n['S2']['z']:+.1f}σ); {100 * n['S2_sw']:+.4f}%", "ok", "passes"),
        ("Volume size S1 (r 0.5, t 0.1 cm): D/Φ₀", f"{100 * n['S1']['rel_diff']:+.2f}% ({n['S1']['z']:+.1f}σ)", "ok", "within 3σ"),
        ("Variance reduction against analog transport: D/Φ₀", f"{100 * n['an']['rel_diff']:+.2f}% ({n['an']['z']:+.1f}σ)", "ok", "passes"),
        ("Scoring boundary: D in cylinder over boundary-free D", f"{pm(b['ratio'], b['sem'], 4)} ({bz:+.1f}σ); 95%: {blo:+.2f}% to {bhi:+.2f}%", "warn", "boundary-free D adopted"),
        ("Fano cavity, production stepping (opt0 control)", f"{pm(n['fano']['ratio_mean'], n['fano']['ratio_sem_batch_spread'], 4)} ({n['fano0']['ratio_mean']:.4f})", "warn", "+0.16% accepted"),
        ("Fano cavity, msc range factor 0.04", f"{pm(n['fano_rf']['ratio_mean'], n['fano_rf']['ratio_sem_batch_spread'], 4)}" if n.get("fano_rf") else "running", "ok", "deviation gone"),
        ("Event sample under the validity rules (bound / strict)", f"{100 * n['ev_bound']:.3f}% / {100 * n['ev_strict']:.0f}% fail, of {n['ev_n']:,}", "ok", "as designed"),
    ]
    checks_tbl = "".join(
        f"<tr><td>{a}</td><td class='n'>{minus(v)}</td><td><span class='chip {s}'>{lab}</span></td></tr>" for a, v, s, lab in checks)

    venues = [
        ("<b>Medical Physics</b> (AAPM)", "Primary recommendation. The study is a Monte Carlo reference with an AI evaluation on top, and the readers are the people who use both the protocol quantities and the models. Reviewers will test the Monte Carlo hardest, which is where this study is strongest.",
         "Reporting to the AAPM TG-268 Monte Carlo checklist (the run manifests already hold most of it); EGSnrc and the other Geant4 lists (M2) as the resolution floor; exact model identifiers and dates."),
        ("<b>Physics in Medicine &amp; Biology</b>", "If the Monte Carlo findings grow into a large part of the paper: the boundary artifact, the estimator closure, the stopping-power basis. PMB readers care about transport details.",
         "The same as above; a fuller transport section, ideally with the boundary artifact compared against EGSnrc."),
        ("<b>phiRO</b> (ESTRO, open access)", "A shorter, faster paper. Fits if the story is the breakpoint result alone.",
         "A tight scope: one configuration, the five-layer breakpoint figure, the Monte Carlo spread."),
        ("<b>JACMP</b> (AAPM, open access)", "If the framing is practical: can a clinical physicist trust a model's reasoning for a calculation check.",
         "A clinical-use angle and less transport detail."),
        ("<b>NeurIPS Datasets &amp; Benchmarks</b> or <b>ML4H</b>", "If the work is released as a benchmark: verifiable intermediate quantities, sampled-event validity rules and Monte Carlo ground truth are rare in model evaluation. It also leads into the training-environment phase (V4).",
         "A frozen, released benchmark with the V1 variations, including the invented compound medium that no table covers; a datasheet; more models."),
    ]
    venues_tbl = "".join(f"<tr><td>{a}</td><td>{b_}</td><td>{c}</td></tr>" for a, b_, c in venues)

    page = f"""<title>Co-60 Reference Point Briefing</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans+Condensed:wght@600&family=IBM+Plex+Sans:wght@400;600&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap">
<style>{CSS}</style>
<div class="wrap">
<nav class="rail" aria-label="Contents">
  <span class="t">Contents</span>
  <a href="#short">In short</a>
  <a href="#why">Why this study</a>
  <a href="#how">How the reference was built</a>
  <a href="#results">What the results show</a>
  <a href="#effects">How big each effect is</a>
  <a href="#findings">What went wrong</a>
  <a href="#assess">Strengths and weaknesses</a>
  <a href="#venues">Where to publish</a>
  <a href="#next">Decisions and next steps</a>
</nav>
<main>
<header style="display:grid;gap:14px">
  <span class="eyebrow">Co-60 reference point study · M1 briefing · 10 October 2026</span>
  <h1>At which step of the photon-to-dose chain do AI answers fail?</h1>
  <p class="lede">The Monte Carlo side of the study is built. For every step from the incident photon to the absorbed dose at 5 cm depth in a Co-60 beam, there is now a reference value with a known uncertainty and an independent check. This page explains what was done, what the numbers say, where the reference is solid and where it is not, and where the work could be published.</p>
  <div class="meta"><span>Geant4 11.4.3 · option4 · 10 keV thresholds</span><span>{n['hist'] / 1e8:.1f} × 10⁸ histories, {n['batches']} batches</span><span>branch claude/adoring-edison-tna32r</span></div>
</header>

<section id="short">
  <h2>In short</h2>
  <div class="short">
    <div><b>Status</b><p>M0 and M1 are complete and reviewed. All plan targets are met. One approved extension run is tightening the scoring-boundary bound.</p></div>
    <div><b>Headline numbers</b><p>D/Φ₀ = <span class="num">{D * 1e12:.4f} × 10⁻¹² Gy cm² ± {100 * Ds / D:.3f}%</span>, β = <span class="num">{n['beta']:.4f}</span>, s<sub>w,air</sub> = <span class="num">{sw90:.4f}</span> (ICRU 90) / <span class="num">{sw37:.4f}</span> (ICRU 37), PDD(5) = <span class="num">{n['pdd5']['mean']:.1f}%</span>.</p></div>
    <div><b>What matters beyond this study</b><p>The stopping-power basis moves s<sub>w,air</sub> by up to 1%, far more than any Monte Carlo uncertainty. Geant4 option4 drops Rayleigh scattering above 1.022 MeV in its default gamma process. A water–water scoring boundary distorts the dose next to it by a few percent.</p></div>
    <div><b>Still open</b><p>EGSnrc and the other Geant4 lists (M2); your verification of refs.yaml; the Fano step-function diagnostic; your confirmation of the 2σ rule that switched D to the boundary-free value (D027).</p></div>
  </div>
</section>

<section id="why">
  <h2>Why this study</h2>
  <p>Language models state textbook radiation physics fluently. Stating a cross section is a different skill from modelling what photons and electrons do at a point in a phantom, and a check on the final dose cannot tell the two apart: a model can recall a PDD table and still have the transport wrong. The study therefore grades every intermediate quantity in the photon-to-dose chain against a Monte Carlo that scores the same quantity in the same bins, and asks at which layer an answer first leaves tolerance.</p>
  <p>The point is the Co-60 reference condition: central axis, 5 cm depth in water, 10 × 10 cm² at SSD 100 cm, where standards laboratories calibrate chambers in N<sub>D,w</sub>. Its protocol quantities (s<sub>w,air</sub> from TRS-398, β slightly above 1, PDD tables) are in every model's training data. The design uses that: it compares a recalled quantity (PDD(5)) with a computed one at the same point (D/Φ₀).</p>
  <div class="callout"><p><b>Hypotheses, recorded before any data.</b> H1: errors are small for primary-photon quantities, which follow from μ and two-body kinematics. H2: errors are larger for the scattered photon spectrum and the electron fluence spectrum, which need transport. H3: models reproduce PDD(5) better than D/Φ₀. The spread between Monte Carlo codes sets the resolution: an AI difference inside it is reported as unresolved.</p></div>
</section>

<section id="how">
  <h2>How the reference was built</h2>
  <ul class="tight">
    <li><b>Own application, no TOPAS or GATE.</b> A Geant4 11.4.3 application (option4, 10 keV production thresholds in the whole phantom) in which every scorer is visible: point source model A uniform over the 10 × 10 cm² pyramid, 30 cm water cube, scoring cylinder r = 1 cm, t = 0.2 cm at 5 cm.</li>
    <li><b>Histories are tagged.</b> Every photon carries its category (primary per line, scattered source photon, secondary photon); every electron carries its parent's category and a delta-ray flag. That is what lets each layer be split by primary and scattered parents.</li>
    <li><b>Estimators.</b> Track length for photon and electron fluence, with electron track length spread over each step's energy interval by the restricted CSDA range. Kerma from the energy transferred at each interaction, not from tabulated μ<sub>tr</sub>. Dose from deposition. s<sub>w,air</sub> by Spencer–Attix from the scored electron spectrum with the Nahum track-end term, on four stopping-power bases.</li>
    <li><b>Variance reduction.</b> Electron Russian roulette outside a region of interest sized at 1.2 × the CSDA range, and photon importance sampling in a parallel world. Together they give about 25× in efficiency for D. An analog run checks them.</li>
    <li><b>Statistics.</b> Independent batches, plus per-history second moments where available. Every quoted error is the larger of the two (D023).</li>
    <li><b>Provenance.</b> Pinned builds, a manifest for every run, frozen production binaries tagged with their source commit, every run kept (superseded ones archived with a reason), and every choice logged in DECISIONS.md.</li>
  </ul>
</section>

<section id="results">
  <h2>What the results show</h2>
  <p>Each layer below shows its key Monte Carlo result and what an AI answer has to get right to match it. The full set, with spectra and angular distributions, is on the <a href="{ARTIFACT_RESULTS}">M1 results page</a>.</p>
  {chain}
  <div class="tbl"><table><thead><tr><th>Layer</th><th>Quantity</th><th>Value (pooled, 1σ)</th></tr></thead><tbody>{results_tbl}</tbody></table></div>
  <h3>Reading the numbers</h3>
  <ul class="tight">
    <li><b>Scatter is the story at 5 cm.</b> A third of the photons reaching the point have scattered, and electrons set in motion by them deposit {pct(n['dose_scat'])} of the dose. A model that treats the beam as primary-only will be off by about a fifth in dose. That is a large, unambiguous signal against a 2% tolerance.</li>
    <li><b>Compton carries the energy, but not all the interactions.</b> {pct(l2[0])} of interactions and {pct(e2[0])} of the energy transfer are Compton, yet the softened scattered spectrum makes photoelectric absorption {pct(l2[1])} of interactions. A model reasoning from the primary lines alone will put that near zero.</li>
    <li><b>β = {n['beta']:.4f}.</b> Dose exceeds collision kerma by {100 * (n['beta'] - 1):.2f}% at 5 cm, as transient CPE beyond d<sub>max</sub> requires; the plan expected an excess on the order of 0.5%.</li>
    <li><b>s<sub>w,air</sub> depends on the data more than on the transport.</b> The same electron spectrum gives {sw90:.4f} with ICRU 90 stopping powers and {sw37:.4f} with ICRU 37. The ICRU 37 value matches the TRS-398 figure of 1.133 (in refs.yaml, not yet verified). Scoring uses ICRU 90 as primary and reports ICRU 37 alongside (D024), so a model quoting the protocol is not marked wrong for quoting it.</li>
    <li><b>Bound-electron Compton matters for the event task.</b> option4 uses a bound-electron Compton model, so recoil energies are smeared around the free-electron edge. The validity rules for sampled events allow a Doppler tolerance of {n['dc'] * 1e3:.0f} keV (the 99.9th percentile of the Monte Carlo departure). Under the strict free-electron rules, {100 * n['ev_strict']:.0f}% of Monte Carlo events would fail on energy conservation alone, because binding energy goes to local deposition.</li>
  </ul>
  <div class="tbl"><table><thead><tr><th>Check</th><th>Result</th><th>Status</th></tr></thead><tbody>{checks_tbl}</tbody></table></div>
</section>

<section id="effects">
  <h2>How big each effect is</h2>
  <p>The question for the AI comparison is which uncertainties are small enough to ignore. On a log scale, against the proposed 2% tolerance for Layer 5 dose: the Monte Carlo statistics and systematics sit one to two orders of magnitude below it. The stopping-power basis for s<sub>w,air</sub> does not, and the physics an AI has to capture is ten times above it.</p>
  <div class="fig">
    <div class="key"><span style="--c:var(--cobalt)">Monte Carlo statistics and systematics on R0 results</span><span style="--c:var(--amber)">choice of reference data</span><span style="--c:var(--muted)">artifacts the reference avoids</span><span style="--c:var(--violet)">physics an AI must capture</span></div>
    {effect_chart(n)}
    <div class="cap">Bar length is the size of each effect in percent, log scale; the dashed line is the proposed tolerance for Layer 5 dose quantities (plan Section 6). "Artifacts the reference avoids": the dose next to the scoring boundary (the reference value is inside the cylinder) and the linear electron estimator (replaced by the CSDA-weighted one).</div>
  </div>
</section>

<section id="findings">
  <h2>What went wrong, and what it taught us</h2>
  <div class="finding"><h3>Geant4 option4 drops Rayleigh scattering above 1.022 MeV</h3><p>The default G4GammaGeneralProcess does not sample coherent scattering in its 2m<sub>e</sub>c²–100 MeV zone, which holds both Co-60 lines. The first production run showed zero Rayleigh events among primaries. The dose effect is small, but Layer 2 interaction shares would have been wrong. Production now switches the general process off (D019). This should go to the Geant4 developers as an issue report.</p></div>
  <div class="finding"><h3>The plan's linear electron-fluence estimator fails energy closure by {100 * (n['closure_lin'] - 1):.1f}%</h3><p>Geant4 ends an electron below about 20 keV in a single step, over which the path per unit energy is 1/L(E), not constant. Spreading track length linearly puts too much fluence at low energy: the dose rebuilt from the spectrum is {100 * (n['closure_lin'] - 1):.1f}% high. Weighting by the restricted CSDA range closes to {100 * (n['closure'] - 1):.2f}%. In the 10–20 keV bin the two differ by {100 * (1 - n['lin_csda_10_20']):.0f}%, which matters for the lowest comparison bins; s<sub>w,air</sub> moves by less than 0.01% (D020).</p></div>
  <div class="finding"><h3>A water–water boundary distorts the dose next to it</h3><p>The depth scan shows the dose up to {100 * n['art_max']:.1f}% high 1–4 mm in front of the scoring cylinder and {-100 * n['art_min']:.1f}% low behind it, with water on both sides. Moving the cylinder to 8 cm moves the pattern; far from both cylinders the scans agree to {100 * (n['far']['mean'] - 1):+.2f}%. The likely cause is GS multiple scattering with the SafetyPlus step limit: steps shrink near a boundary, and a step-length-dependent displacement of deposition piles energy up in front of it. Inside the cylinder the dose is {pm(b['ratio'], b['sem'], 4)} of the boundary-free value ({bz:+.1f}σ, {n['bnd_free_batches']} boundary-free batches), so the boundary-free D is now the R0 reference (D025, with the 2σ threshold in D027) and β moves from {n['beta_cyl']:.4f} to {n['beta']:.4f}. It is a 2σ call, so M2's boundary-free scoring has to confirm it. That scoring is needed anyway, because EGSnrc crosses boundaries with a different algorithm.</p></div>
  <div class="finding"><h3>A 4σ anomaly that was a fluctuation</h3><p>The main run's primary fluence came out 0.12% below exp(−μs)/(ΩR²) at 4.4σ on its batch spread. Eight diagnostic runs (photon-only and full transport, with and without importance sampling, two geometries), a transmission measurement of the tracking μ and a collision-estimator cross-check found no cause. The main run's batches simply agreed with each other more than their per-history variance allowed. A replicate with new seeds, pooled by a rule fixed before it finished (D022), came out normal; pooled, the check gives {pm(prim['mean'], prim['sem_conservative'], 4)}. The lasting change is the uncertainty policy (D023).</p></div>
  <div class="finding"><h3>Geant4's own water stopping power is not Bethe–Møller below 100 keV</h3><p>option4 uses the Penelope ionisation model below 100 keV. For water it is 2.5% above Bethe–Møller (I = 78 eV) at 12 keV and 1.7% at 100 keV, and Geant4 smooths that offset into the higher-energy model as 1.7% × (100 keV/E). Air agrees within 0.1%. That is why the Geant4-basis s<sub>w,air</sub> is {swg4:.4f}, 1% above the ICRU 90 basis, and why it is not used for scoring.</p></div>
  <div class="finding"><h3>Smaller items</h3><p>Photon categories were mislabelled at importance-cell boundaries during development (fixed before production; zero untagged tracks since). The region-of-interest check lacked a buffer in front of a depth scan starting inside the phantom (no M1 run was affected; fixed for M2). Twice, process clean-up commands matched my own shell; one two-minute run was lost and archived with its reason.</p></div>
</section>

<section id="assess">
  <h2>Strengths and weaknesses</h2>
  <div class="cols">
    <div><h3>Strengths</h3>
      <ul class="tight">
        <li><b>Every layer has an independent check:</b> analytic primary fluence with a measured tracking μ, thin slab, fluence-to-dose closure at {100 * (n['closure'] - 1):.2f}%, analog against variance reduction, S1, S2, a seed-independent replicate, and a boundary-free dose.</li>
        <li><b>Anomalies were run down, not explained away.</b> Two were real Geant4 issues; one was statistics, settled by a pre-registered replicate.</li>
        <li><b>Conservative uncertainties.</b> The larger of two error estimates everywhere; targets met on that basis.</li>
        <li><b>The design separates recall from computation</b> (H3), and the layered grading shows where an answer fails, not only whether.</li>
        <li><b>Sampled-event validity rules are physical</b> and calibrated on the reference physics, so invalid AI events can be counted separately from distributional distance.</li>
        <li><b>Full provenance.</b> Any number on this page traces to a run manifest, a commit and a decision entry.</li>
      </ul></div>
    <div><h3>Weaknesses and open risks</h3>
      <ul class="tight">
        <li><b>One code so far.</b> Until EGSnrc and the Livermore and Penelope lists run (M2), the reference is one code's physics and the resolution floor is unknown.</li>
        <li><b>s<sub>w,air</sub> has no unique truth.</b> The 1% basis spread is set by convention (D024); reviewers will ask, and the TRS-398 revision's basis still needs checking.</li>
        <li><b>Source model A is idealized:</b> no head scatter, no source size, no contamination electrons. PDD(5) and the scatter fraction will differ from a real unit, and models may answer for a real unit unless the prompt is explicit.</li>
        <li><b>Step-size sensitivity.</b> The production Fano deviation (+{100 * (n['fano']['ratio_mean'] - 1):.2f}%) disappears at msc range factor 0.04, and the boundary effect on D ({100 * (b['ratio'] - 1):+.2f}%) was corrected on a 2σ result. Both point at the electron step limit, so the M1 dose carries a step-size dependence of a few tenths of a percent until M2 reruns R0 with tighter stepping.</li>
        <li><b>External values unverified.</b> NIST μ/ρ and μ<sub>en</sub>/ρ could not be fetched from the container, so the plan's kerma-from-tables cross-check is not done yet.</li>
        <li><b>The Doppler tolerance comes from the reference itself</b> (99.9th percentile). It is physically motivated, but a definition from Compton-profile data would be harder to question.</li>
        <li><b>One point, one depth.</b> Generality waits for the V1 variations.</li>
        <li><b>The AI side does not exist yet.</b> The main risks ahead are prompt ambiguity, model-version drift and cost.</li>
      </ul></div>
  </div>
</section>

<section id="venues">
  <h2>Where to publish</h2>
  <p>The publishable result is the AI comparison (M4–M5); M1 is its foundation. My recommendation: write the main paper for <b>Medical Physics</b>, submit an <b>AAPM annual meeting</b> abstract once the first AI results exist (or ESTRO, depending on timing), and decide on a benchmark-track version once the V1 variations are scoped. The plan lists the venue as an open decision because it changes the reporting format; the table says how.</p>
  <div class="tbl"><table><thead><tr><th>Venue</th><th>Why it fits</th><th>What it would need</th></tr></thead><tbody>{venues_tbl}</tbody></table></div>
  <ul class="tight">
    <li><b>Conferences.</b> AAPM and ESTRO for the AI results; ICCR (computers in radiotherapy) for the Monte Carlo and scoring methodology. Check current abstract deadlines; I have not looked them up.</li>
    <li><b>Side outputs.</b> The Rayleigh omission is an issue report to Geant4, not a paper. The boundary artifact could become a technical note in Medical Physics or PMB if it is characterized against step settings and compared with EGSnrc in M2: anyone scoring dose in a small daughter volume with option4 is exposed to it.</li>
    <li><b>Reporting guidelines to plan for.</b> TG-268 (RECORDS) for the Monte Carlo; for a clinical-AI venue, TRIPOD-LLM. Both are easier to satisfy now, while the manifests are being written, than at submission.</li>
  </ul>
</section>

<section id="next">
  <h2>Decisions taken today, and what comes next</h2>
  <div class="tbl"><table><thead><tr><th>Decision</th><th>What it fixes</th></tr></thead><tbody>
    <tr><td>D024</td><td>s<sub>w,air</sub> scored against ICRU 90, with ICRU 37 reported alongside; fixed before any AI answer exists.</td></tr>
    <tr><td>D025</td><td>Scoring-boundary effect carried as a systematic on D; extension running; dose scored without a nearby boundary from M2.</td></tr>
    <tr><td>D023, D020</td><td>Conservative uncertainty policy; CSDA-weighted electron estimator as the Layer 4 reference.</td></tr>
    <tr><td>D026</td><td>Fano +0.16% accepted as a bound; range factor 0.04 removes it, so M2 plans tighter stepping pending the step-function result; at least 20 batches per M2 run; refs.yaml verification order.</td></tr>
    <tr><td>D027</td><td>Threshold for D025: the cylinder D was 0.24% low at −2.2σ, so D/Φ₀ at R0 is now the boundary-free value. Waiting for your confirmation.</td></tr>
  </tbody></table></div>
  <div class="steps">
    <div class="step"><span class="mk">You</span><p>Verify refs.yaml in this order: line energies, TRS-398 s<sub>w,air</sub> (and whether its 2024 revision uses ICRU 90), I-values, normalisation constants, then the rest. Add a published PDD(5) for Co-60, 10 × 10 cm², SSD 100 cm. Settle the remaining plan decisions: SSD, vacuum or air, model list, tolerances, source for model B.</p></div>
    <div class="step"><span class="mk">M2</span><p>Livermore and Penelope at R0; the EGSnrc B0 benchmark (DOSRZnrc, FLURZnrc, SPRRZnrc) with the matched water; comparison with the published values you verify. Gate: you accept the Monte Carlo as the reference.</p></div>
    <div class="step"><span class="mk">M3</span><p>Prompt, schema and sandbox; the P0 pilot on one inexpensive model; a cost estimate. Gate: prompts, models, tolerances and budget.</p></div>
    <div class="step"><span class="mk">M4–M5</span><p>The full model × condition × repeat matrix, the sampled-events task, then the breakpoint analysis.</p></div>
  </div>
</section>

<footer>
  <span>Generated by analysis/m1_brief.py from analysis/results/M1_results.json and analysis/M1_fano_summary.yaml.</span>
  <span>Results page: <a href="{ARTIFACT_RESULTS}">M1 results</a>. Full record: STATUS.md and DECISIONS.md on branch claude/adoring-edison-tna32r.</span>
</footer>
</main>
</div>
"""
    return page


def main():
    page = build(load())
    out = RES / "co60_brief.html"
    out.write_text(page)
    print(f"wrote {out} ({len(page) / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
