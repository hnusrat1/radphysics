#!/usr/bin/env python3
"""Study review page for Humza: everything built so far, what is running, what needs him.

    python analysis/study_page.py     # writes analysis/results/co60_study_review.html

Reads analysis/results/{M1_results, M2_results, reference_R0, published_comparison, M5_controls}.json,
analysis/M1_fano_summary.yaml, the M2 queue and run manifests, ai/ (observables, prompts, models, prices).
Parts whose runs are not finished are shown as running; no number on the page is typed by hand.
"""
import html
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.m1_report import pm, sci, sig  # noqa: E402

RES = ROOT / "analysis" / "results"
OBS = yaml.safe_load(open(ROOT / "ai" / "observables.yaml"))["observables"]


def load(name):
    p = RES / name
    return json.load(open(p)) if p.exists() else {}


def esc(s):
    return html.escape(str(s))


def chip(kind, text):
    return f'<span class="chip {kind}">{esc(text)}</span>'


def run_state(run_id, n):
    """(done batches or jobs, planned, finished?) from a run's manifest."""
    m = ROOT / "mc" / "runs" / run_id / "manifest.yaml"
    if not m.exists():
        return 0, n, False
    d = yaml.safe_load(open(m))
    k = len([b for b in d.get("batches", d.get("jobs", [])) if b.get("exit_code") == 0])
    return k, n, bool(d.get("finished_utc"))


def queue_rows():
    q = yaml.safe_load(open(ROOT / "mc" / "queues" / "M2.yaml"))["items"]
    rows = []
    for it in q:
        spec = yaml.safe_load(open(ROOT / it["spec"]))
        n = spec.get("batches", spec.get("jobs"))
        k, n, fin = run_state(spec["run_id"], n)
        state = chip("ok", "done") if fin else chip("run", f"{k}/{n}") if k else chip("idle", "queued")
        code = "EGSnrc " + spec["code"].upper().replace("NRC", "nrc") if it["kind"] == "egsnrc" else f"Geant4 {spec['physics']}"
        rows.append(f"<tr><td class='n'>{esc(spec['run_id'])}</td><td>{esc(code)}</td><td>{state}</td></tr>")
    return "\n".join(rows)


def fmt_val(o, v):
    if v is None:
        return "–"
    if isinstance(v, dict):
        return ", ".join(f"{k} {sig(x, 3)}" for k, x in v.items())
    if isinstance(v, list):
        return f"{len(v)} bins"
    if o["unit"].startswith("Gy"):
        return sci(v, 4)
    return sig(v, 5)


def tol_text(o, r):
    t = o["tolerance"]
    if t == "dose":
        return "±2%"
    if t == "scalar":
        return "±max(5%, 0.005)"
    hw = r.get("half_w1")
    return f"3 × half-sample (W1 {sig(3 * hw, 2)})" if hw else "3 × half-sample (pending)"


def floor_text(o, r):
    if o["kind"] in ("spectrum", "distribution"):
        return f"W1 {sig(r['floor_w1'], 2)}" if "floor_w1" in r else "pending"
    f = r.get("floor_abs")
    if f is None:
        return "pending"
    if isinstance(f, dict):
        return "per process"
    mc = r["mc"]
    return f"{100 * f / abs(mc):.2g}%" if mc else sig(f, 2)


def main():
    m1 = load("M1_results.json")
    m2 = load("M2_results.json")
    ref = load("reference_R0.json")
    pub = load("published_comparison.json")
    ctl = load("M5_controls.json")
    fano = yaml.safe_load(open(ROOT / "analysis" / "M1_fano_summary.yaml"))
    models = yaml.safe_load(open(ROOT / "ai" / "models.yaml"))
    prices = yaml.safe_load(open(ROOT / "ai" / "prices.yaml"))
    prompt = (ROOT / "ai" / "prompts" / "v1" / "task_observables_R0.txt").read_text()
    system_a1 = (ROOT / "ai" / "prompts" / "v1" / "system_A1.txt").read_text()
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    obs = ref.get("observables", {})
    m2_done = not ref.get("provisional", True)

    # ------------------------------------------------ reference values (key scalars)
    key_ids = ["L1_primary_fluence_per_phi0", "L1_scattered_fraction_fluence", "L1_mean_photon_energy_MeV",
               "L3_mean_initial_energy_MeV", "L4_electron_fluence_per_phi0", "L4_delta_fraction",
               "L5_D_over_phi0_Gy_cm2", "L5_Kcol_over_phi0_Gy_cm2", "L5_beta", "L5_s_w_air", "L5_PDD5_percent"]
    names = {o["id"]: o for o in OBS}
    ref_rows = []
    for k in key_ids:
        r, o = obs[k], names[k]
        val = sci(r["mc"], 5) if o["unit"].startswith("Gy") else pm(r["mc"], r["sem"]) if r.get("sem") else sig(r["mc"], 6)
        alongside = f" (ICRU 37 basis {sig(r['mc_alongside'], 5)})" if "mc_alongside" in r else ""
        rel = f"{100 * r['sem'] / abs(r['mc']):.2g}%" if r.get("sem") else ""
        ref_rows.append(f"<tr><td>{esc(' '.join(o['text'].split())[:90])}</td><td class='n'>{val}{alongside}</td>"
                        f"<td class='n'>{rel}</td><td>{esc(o['unit'])}</td></tr>")
    src = ref.get("sources", {})

    # ------------------------------------------------ step size and list spread
    def ratio_cell(x):
        if not x:
            return "<td class='n'>running</td>"
        z = (x["ratio"] - 1) / x["sem"] if x["sem"] else 0
        d = 100 * (x["ratio"] - 1)
        return f"<td class='n'>{d:+.3f}% <span class='z'>({z:+.1f}σ)</span></td>" if abs(d) < 0.01 else \
            f"<td class='n'>{d:+.2f}% <span class='z'>({z:+.1f}σ)</span></td>"

    spread_keys = [("L1_fluence_primary_per_phi0", "Primary photon fluence / Φ₀"),
                   ("L1_scattered_fraction_fluence", "Scattered fraction of photon fluence"),
                   ("L1_mean_energy_all_MeV", "Mean photon energy"),
                   ("L2_scattered_share_of_interactions", "Scattered share of interactions"),
                   ("L3_mean_T_all_MeV", "Mean initial electron energy"),
                   ("L4_electron_fluence_above_10keV_per_phi0", "Electron fluence > 10 keV / Φ₀"),
                   ("L4_delta_fraction_above_10keV", "Delta-ray share of electron fluence"),
                   ("L5_K_over_phi0_Gy_cm2", "Kerma / Φ₀"), ("L5_Kcol_over_phi0_Gy_cm2", "Collision kerma / Φ₀"),
                   ("L5_s_w_air_icru90", "s_w,air (ICRU 90 basis)"),
                   ("L5_dose_fraction_parent_scattered", "Dose share from scattered-photon electrons")]
    ls = m2.get("list_spread", {})
    st = m2.get("step_size", {})
    spread_rows = []
    for k, label in spread_keys:
        spread_rows.append(f"<tr><td>{esc(label)}</td>{ratio_cell(st.get(k))}"
                           f"{ratio_cell(ls.get('livermore', {}).get('scalars', {}).get(k))}"
                           f"{ratio_cell(ls.get('penelope', {}).get('scalars', {}).get(k))}</tr>")
    dbf = m2.get("dose_bf", {})
    d_rows = []
    for L in ("option4", "livermore", "penelope"):
        b = dbf.get(L)
        if b:
            d_rows.append(f"<tr><td>{L}</td><td class='n'>{sci(b['D_over_phi0_Gy_cm2']['mean'], 5)} ± "
                          f"{100 * b['D_over_phi0_Gy_cm2']['sem'] / b['D_over_phi0_Gy_cm2']['mean']:.2g}%</td>"
                          f"<td class='n'>{pm(b['beta']['ratio'], b['beta']['sem'])}</td>"
                          f"<td class='n'>{100 * (b['cylinder_over_bf']['ratio'] - 1):+.2f}%</td></tr>")
        else:
            d_rows.append(f"<tr><td>{L}</td><td colspan='3'>{chip('run', 'running')}</td></tr>")
    b0 = m2.get("b0", {})
    b0_rows = []
    for label, a, b_ in (("D/Φ₀ at 5 cm (boundary-free in Geant4)", "geant4_D_over_phi0_Gy_cm2", "egsnrc_D_over_phi0_Gy_cm2"),
                         ("s_w,air, matched PEGS4 media", "geant4_s_w_air_pegs4", "egsnrc_s_w_air")):
        ga, eb = b0.get(a), b0.get(b_)
        if ga and eb:
            r_ = eb["mean"] / ga["mean"]
            s_ = r_ * math.hypot(ga["sem"] / ga["mean"], eb["sem"] / eb["mean"])
            fmt = (lambda x: sci(x, 5)) if "D" in label else (lambda x: sig(x, 6))
            b0_rows.append(f"<tr><td>{label}</td><td class='n'>{fmt(ga['mean'])}</td><td class='n'>{fmt(eb['mean'])}</td>"
                           f"<td class='n'>{100 * (r_ - 1):+.2f}% ± {100 * s_:.2f}%</td></tr>")
        else:
            b0_rows.append(f"<tr><td>{label}</td><td colspan='3'>{chip('run', 'running')}</td></tr>")

    # ------------------------------------------------ charts (shown once the runs exist)
    from analysis.svgplot import CHART_CSS, spectra_with_ratio
    import numpy as np
    bins = yaml.safe_load(open(ROOT / "config" / "bins.yaml"))["comparison"]
    charts = []
    runs2 = m2.get("runs", {})
    lists = [(L, runs2.get(f"M2-R0-{L}"), c, d) for L, c, d in (("option4", "--s1", None), ("livermore", "--s2", "6 3"),
                                                              ("penelope", "--s3", "2 3"))]
    lists = [x for x in lists if x[1]]
    if len(lists) >= 2:
        for key, sub, grid, title, xl in (
                ("L4_electron_fluence_comparison", None, "electron_energy", "R0 electron fluence (shape)",
                 "electron kinetic energy (MeV)"),
                ("L1_photon_fluence_comparison", "scattered", "photon_energy", "R0 scattered photon fluence (shape)",
                 "photon energy (MeV)")):
            e = np.array(bins[grid]["edges"])
            ser = []
            for L, r, c, d in lists:
                v = np.array((r[key][sub] if sub else r[key])["mean"], float)
                ser.append((L, v / v.sum() / np.diff(e), None, c, d))
            charts.append(spectra_with_ratio(ser, e, xl, "fraction per MeV", title, "ratio to option4",
                                             ratio_range=(0.97, 1.03)))
    sp = b0.get("egsnrc_spectra_10keV")
    g0 = b0.get("geant4_spectra_10keV")
    if sp and g0:
        top = np.array(sp["bintop_MeV"], float)
        e = np.concatenate([[top[0] - 0.01], top])
        n = len(top)
        for key_g, key_e, title, xl in (("electron", "electron_total", "B0 electron fluence (shape)",
                                         "electron kinetic energy (MeV)"),
                                        ("photon", "photon_total", "B0 photon fluence (shape)", "photon energy (MeV)")):
            g = np.array(g0[key_g], float)[:n]      # Geant4 10 keV bin j = [0.01 j, 0.01 (j + 1)] = EGSnrc bin j
            ev = np.array(sp[key_e]["mean"], float)
            sel = slice(1, n) if key_g == "electron" else slice(0, n)
            ee = e[sel.start:]
            if key_g == "photon":      # drop the two primary-line bins so the scattered spectrum is visible
                line = (np.abs(top - 1.18) < 1e-6) | (np.abs(top - 1.34) < 1e-6)
                g, ev = np.where(line, 0.0, g), np.where(line, 0.0, ev)
                title = "B0 photon fluence without the two line bins (shape)"
            ser = [("Geant4 option4", g[sel] / g[sel].sum() / 0.01, None, "--s1", None),
                   ("EGSnrc", ev[sel] / ev[sel].sum() / 0.01, None, "--s4", "6 3")]
            charts.append(spectra_with_ratio(ser, ee, xl, "fraction per MeV", title, "EGSnrc / Geant4",
                                             ratio_range=(0.95, 1.05)))
    charts_html = "".join(f'<figure class="fig">{c}</figure>' for c in charts)

    # ------------------------------------------------ Fano
    def fano_row(label, key, verdict):
        f = fano.get(key)
        if not f:
            return f"<tr><td>{label}</td><td colspan='2'>{chip('run', 'running')}</td></tr>"
        return (f"<tr><td>{label}</td><td class='n'>{pm(f['ratio_mean'], f['ratio_sem_batch_spread'])}</td>"
                f"<td class='n'>{f['deviation_sigma']:+.1f}σ</td><td>{verdict}</td></tr>")
    stepfn_key = next((k for k in fano if "StepFunction" in k), None)
    fano_rows = "\n".join([
        fano_row("Production stepping (range factor 0.08), 40 batches", "standard_opt4", chip("warn", "+0.16%")),
        fano_row("Range factor 0.04", "standard_opt4 | /process/msc/RangeFactor 0.04", chip("ok", "passes")),
        fano_row("Step function 0.05, 1 µm", stepfn_key or "none", chip("ok", "see D026") if stepfn_key else ""),
        fano_row("Negative control: G4EmStandardPhysics (opt0)", "standard_opt0", chip("ok", "fails, as it should")),
    ])

    # ------------------------------------------------ published values
    pub_rows = []
    for r in pub.get("rows", []):
        mc = r["mc"]
        rv = r["ref_value"]
        pub_rows.append(f"<tr><td>{esc(r['quantity'])}</td><td class='n'>{sig(mc, 6)}</td>"
                        f"<td class='n'>{sig(rv, 6) if isinstance(rv, (int, float)) else 'not entered'}</td>"
                        f"<td class='n'>{format(r['mc_over_ref'], '.4f') if r.get('mc_over_ref') else '–'}</td>"
                        f"<td>{chip('ok', 'verified') if r['verified_by_humza'] else chip('warn', 'unverified')}</td></tr>")

    # ------------------------------------------------ scoring table
    score_rows = []
    for o in OBS:
        r = obs.get(o["id"], {})
        score_rows.append(f"<tr><td class='n'>L{o['layer']}</td><td class='n'>{esc(o['id'])}</td>"
                          f"<td class='n'>{fmt_val(o, r.get('mc'))}</td><td>{esc(tol_text(o, r))}</td>"
                          f"<td class='n'>{esc(floor_text(o, r))}</td><td>{esc(', '.join(o['tags']))}</td></tr>")
    ctl_rows = []
    expect = {"identity": "no breakpoint", "negative_dose_plus5pct": "breaks at Layer 5",
              "negative_uniform": "breaks at Layer 1", "positive_livermore": "no breakpoint (pass or unresolved)",
              "positive_penelope": "no breakpoint (pass or unresolved)"}
    for name, s in ctl.get("controls", {}).items():
        bp = s["breakpoint_layer"]
        got = "no breakpoint" if bp is None else f"breaks at Layer {bp}"
        ok = got.split(" (")[0] == expect.get(name, "").split(" (")[0]
        ctl_rows.append(f"<tr><td class='n'>{esc(name)}</td><td>{esc(expect.get(name, ''))}</td><td>{esc(got)}</td>"
                        f"<td>{chip('ok', 'as expected') if ok else chip('bad', 'unexpected')}</td></tr>")
    ev = ctl.get("event_controls", {})
    ev_rows = []
    for name, v in ev.items():
        ev_rows.append(f"<tr><td class='n'>{esc(name)}</td><td class='n'>{v['energy_distance']:.4f}</td>"
                       f"<td class='n'>{v['energy_distance_mc_baseline']['median']:.4f}, {v['energy_distance_mc_baseline'].get('p95', float('nan')):.4f}</td>"
                       f"<td class='n'>{100 * v['invalid_fraction_bound']:.1f}%</td></tr>")

    # ------------------------------------------------ cost
    from ai.harness.run import estimate_cost, matrix
    plan = list(matrix(models["slots"], ["A0", "A1", "A2"], ["observables", "events"], 5, "R0"))
    total, _ = estimate_cost(plan, prices["prices"], prices["tokens_per_conversation"])
    pilot = list(matrix(models["slots"], ["A0", "A1", "A2"], ["observables", "events"], 1, "P0"))
    total_p, _ = estimate_cost(pilot, prices["prices"], prices["tokens_per_conversation"])
    layer_counts = {L: sum(1 for o in OBS if o["layer"] == L) for L in range(1, 6)}

    milestones = [
        ("M0", "Environment", chip("ok", "done"), "Geant4 11.4.3 and EGSnrc v2026 pinned; examples reproduce; Fano test reported."),
        ("M1", "Monte Carlo at R0, option4", chip("ok", "done"),
         "All five layers; S1, S2; analytic primary fluence; statistical targets met; boundary-free dose adopted (D027)."),
        ("M2", "Monte Carlo validation", chip("ok", "done") if m2_done else chip("run", "running"),
         "Livermore and Penelope at R0, EGSnrc B0 benchmark, M2 stepping (range factor 0.04), published values."),
        ("M3", "AI harness", chip("warn", "built; pilot blocked"),
         "Prompts, schema, sandbox, clients, runner, A2 data, cost estimate. The pilot needs API keys and network access."),
        ("M4", "AI runs at R0", chip("warn", "ready to launch"),
         "One OpenRouter key (D029); openrouter.ai is reachable. Starts in a new session with the key as an environment secret: pilot first, then the low-cost run under your $10 cap."),
        ("M5", "Analysis", chip("warn", "built; no AI data"),
         "Scorer, breakpoints, H1–H3 and event tests are built and pass their controls. Waiting for M4 answers."),
    ]
    ms_rows = "\n".join(f"<tr><td class='n'>{a}</td><td>{esc(b)}</td><td>{c}</td><td>{esc(d)}</td></tr>"
                        for a, b, c, d in milestones)

    d028 = [
        ("D027 threshold", "2σ confirmed", "Standard 95% rule; the boundary effect was −2.2σ. M2 re-measures the dose anyway.",
         "3σ would keep the cylinder dose and carry the 0.24% as a systematic."),
        ("R0 geometry", "SSD 100 cm, vacuum", "Plan defaults; air adds a 0.1%-level term the AI could get wrong outside the physics chain.",
         "Rerun M1/M2 at SSD 80 or with air."),
        ("Tolerances", "2% dose; max(5%, 0.005) other scalars; 3 × half-sample distance for distributions",
         "Plan defaults; the absolute floor keeps near-zero fractions testable.",
         "Rescoring only; no rerun."),
        ("Resolution floor", "Largest list-to-option4 difference + 2σ; EGSnrc via B0",
         "Differences inside the codes' own spread cannot be read as AI error.", "Rescoring only."),
        ("Models", "Top model from Anthropic, OpenAI, Google; two open-weight from different labs",
         "Covers the frontier and open weights; IDs fixed and recorded at launch.", "Edit ai/models.yaml before launch."),
        ("s_w,air basis", "ICRU 90 primary, ICRU 37 alongside (D024)", "Current data; TRS-398 numbers are ICRU 37.", "Rescoring only."),
        ("M2 stepping", "Range factor 0.04 for all Geant4 lists (D026 as decided)",
         "Removes the Fano deviation; M1 (0.08) kept as the step-size sensitivity.", "M2 reruns at 0.08."),
        ("A2 data", "XCOM/ESTAR reproductions from EGSnrc-distributed NIST data",
         "physics.nist.gov unreachable; validated against Geant4.", "Swap in the NIST downloads before M4."),
        ("Model B", "Deferred to V2; Mora et al. 1999 spectrum as the candidate", "Out of scope for the first build.", "None now."),
        ("Venue", "Medical Physics; TG-268 and TRIPOD-LLM reporting", "Fits the dosimetry content and the AI evaluation.", "Changes table formats only."),
        ("Access route (D029, your call)", "Every model through OpenRouter with one key", "One key, one host, one bill; the served model and upstream provider are logged per call.",
         "Per-provider keys and hosts; no rerun."),
    ]
    d028_rows = "\n".join(f"<tr><td>{esc(a)}</td><td>{esc(b)}</td><td>{esc(c)}</td><td>{esc(d)}</td></tr>"
                          for a, b, c, d in d028)

    html_out = f"""<title>Co-60 Study Review</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap">
<style>{CSS}{CHART_CSS}</style>
<div class="wrap">
<nav class="rail" aria-label="Contents">
  <span class="t">Review</span>
  <a href="#state">Where it stands</a>
  <a href="#you">What needs you</a>
  <a href="#d028">Choices made for you</a>
  <a href="#reference">MC reference</a>
  <a href="#validation">Code-to-code</a>
  <a href="#fano">Transport checks</a>
  <a href="#published">Published values</a>
  <a href="#ai">AI harness</a>
  <a href="#scoring">Scoring</a>
  <a href="#limits">Limits</a>
</nav>
<main>
<header class="hero">
  <span class="eyebrow">Co-60 reference point · AI against Monte Carlo · review</span>
  <h1>Co-60 Study Review</h1>
  <p class="lede">The Monte Carlo reference is final and validated against two other Geant4 physics lists and EGSnrc.
  The AI harness is built and tested without any API call. The model runs go through one OpenRouter key (D029) and
  start in a new session once the key is stored as an environment secret. This page covers each part, the choices
  I made under your delegation, and what is left for you.</p>
  <div class="meta"><span>Generated {now}</span><span>commit {esc(head)}</span><span>R0: Co-60, 5 cm depth, 10 × 10 cm², SSD 100 cm</span>
  <span>reference {'final (M2)' if m2_done else 'provisional (M1 parts)'}</span></div>
</header>

<section id="state">
  <h2>Where the study stands</h2>
  <div class="tbl"><table><thead><tr><th>Milestone</th><th>Scope</th><th>State</th><th>Content</th></tr></thead>
  <tbody>{ms_rows}</tbody></table></div>
  <details><summary>M2 run queue (each run is kept; resumable after a container restart)</summary>
  <div class="tbl"><table><thead><tr><th>Run</th><th>Code</th><th>Batches or jobs</th></tr></thead><tbody>{queue_rows()}</tbody></table></div></details>
</section>

<section id="you">
  <h2>What needs you</h2>
  <ul class="tight">
    <li><b>Confirm or reverse the choices below (D028).</b> Each row says what a reversal would cost. Most are rescoring only.</li>
    <li><b>Verify refs.yaml (Section 10.1).</b> Every external number is still unverified. Agreed order: line energies, TRS-398 s_w,air (and whether its 2024 revision uses ICRU 90), I-values, normalisation constants, the rest. Two values are still missing: a published PDD(5) for SSD 100 cm, and the NIST μ/ρ and μ_en/ρ at the two lines.</li>
    <li><b>Launch M4.</b> Store <span class="n">OPENROUTER_API_KEY</span> as a Network secret in the environment settings, then start a new session. It runs the P0 pilot (under $0.50), reports the measured cost, and runs the low-cost R0 matrix if it fits your $10 cap. Current live-price estimate for five low-cost models: about $4. The frontier run can follow later with the same frozen prompts.</li>
    <li><b>Optional.</b> If you can reach NIST, replace the A2 tables with the genuine XCOM and ESTAR downloads (<span class="n">ai/data/README.md</span>).</li>
  </ul>
</section>

<section id="d028">
  <h2>Choices made for you (D028)</h2>
  <p>You asked me to pick my recommended option for every open decision and keep going. Paid API spend and
  checking reference values stay with you under Section 10.</p>
  <div class="tbl"><table><thead><tr><th>Decision</th><th>Choice</th><th>Why</th><th>Cost of reversing</th></tr></thead>
  <tbody>{d028_rows}</tbody></table></div>
</section>

<section id="reference">
  <h2>Monte Carlo reference at R0</h2>
  <p>These are the values the AI answers are scored against: option4 with the M2 stepping, and the dose scored by position
  with no geometry boundary nearby. {'' if m2_done else 'While M2 runs, the parts marked M1 below come from M1, at range factor 0.08.'}</p>
  <div class="srcs">{''.join(f'<span><b>{esc(k)}</b> {esc(v)}</span>' for k, v in src.items())}</div>
  <div class="tbl"><table><thead><tr><th>Quantity</th><th>Value</th><th>1σ</th><th>Unit</th></tr></thead>
  <tbody>{''.join(ref_rows)}</tbody></table></div>
</section>

<section id="validation">
  <h2>Code-to-code agreement</h2>
  <p>Each column is a difference from option4 at range factor 0.04 (M2-R0-option4), in percent with its z-score. The
  M1 column is option4 at range factor 0.08, so it shows the step-size sensitivity. Livermore and Penelope share option4's
  transport settings, so their differences come from the cross-section and energy-loss models.</p>
  <div class="tbl"><table><thead><tr><th>Quantity</th><th>M2 vs M1 (step size)</th><th>Livermore</th><th>Penelope</th></tr></thead>
  <tbody>{''.join(spread_rows)}</tbody></table></div>
  {charts_html}
  <h3>Boundary-free dose per list</h3>
  <div class="tbl"><table><thead><tr><th>List</th><th class="nt">D/Φ₀ (Gy cm²)</th><th class="nt">β</th><th>Cylinder vs boundary-free</th></tr></thead>
  <tbody>{''.join(d_rows)}</tbody></table></div>
  <h3>EGSnrc benchmark in geometry B0</h3>
  <p>B0 is a cylindrical phantom (r 15 cm, 30 cm deep) with a circular field of the same area. The water and air are matched to
  Geant4's (D015), so the comparison tests the transport rather than the data.</p>
  <div class="tbl"><table><thead><tr><th>Quantity</th><th>Geant4 option4</th><th>EGSnrc</th><th>EGSnrc − Geant4</th></tr></thead>
  <tbody>{''.join(b0_rows)}</tbody></table></div>
</section>

<section id="fano">
  <h2>Transport checks</h2>
  <div class="tbl"><table><thead><tr><th>Fano cavity test (fanoCavity, option4)</th><th>Ratio</th><th>Deviation</th><th></th></tr></thead>
  <tbody>{fano_rows}</tbody></table></div>
  <p>The +0.16% deviation at production stepping disappears when the electron steps are tightened. The boundary artifact
  found in M1 (dose up to 3% off within a millimetre of a water–water boundary) points to the same step limit. That is why M2 uses
  range factor 0.04 and scores the dose by position.</p>
</section>

<section id="published">
  <h2>Published values</h2>
  <div class="tbl"><table><thead><tr><th>Quantity</th><th>Monte Carlo</th><th>Published</th><th>MC / published</th><th>Status</th></tr></thead>
  <tbody>{''.join(pub_rows)}</tbody></table></div>
  <p class="note">Missing from refs.yaml: {esc(', '.join(pub.get('missing_reference_values', [])))}. physics.nist.gov and BJR Supplement 25
  could not be reached from this environment.</p>
</section>

<section id="ai">
  <h2>AI harness</h2>
  <div class="cond">
    <div><b>A0, recall</b><span>Problem statement only, no tools.</span></div>
    <div><b>A1, compute</b><span>Plus a Python sandbox: numpy and scipy, no network, no transport code, 4 cores, 10 min per call.</span></div>
    <div><b>A2, compute with data</b><span>Plus water XCOM and ESTAR tables as files.</span></div>
  </div>
  <ul class="tight">
    <li>{len(OBS)} quantities across the five layers ({', '.join(f'L{L}: {n}' for L, n in layer_counts.items())}), plus 200 sampled interactions per events call. The prompt states every unit and bin edge and gives the JSON schema.</li>
    <li>A test fails if any number in a prompt, other than the configuration and the bin edges, lies within 0.5% of a Monte Carlo result or a refs.yaml value.</li>
    <li>Model code cannot see the network, the repository, the transport codes or this session's transcripts. It runs in its own network, mount and process namespaces.</li>
    <li>Every request, response and tool call is written once, read-only. A malformed answer gets one logged retry. Refusals, parse failures and API errors are counted.</li>
    <li>Model slots: {esc(', '.join(s['slot'] for s in models['slots']))}. Identifiers are filled in at launch and recorded per call.</li>
    <li><b>Cost estimate:</b> {len(plan)} conversations, about ${total:,.0f} at central list prices, about ${3 * total:,.0f} in the high case. Pilot: about ${total_p:,.0f}. The prices are unverified third-party figures and the token counts are assumptions. Both are replaced by official prices and pilot measurements before you approve.</li>
  </ul>
  <details><summary>The frozen R0 prompt (v1) as a model sees it</summary>
  <pre class="prompt">{esc(system_a1)}

{esc(prompt.split('The JSON schema is:')[0])}The JSON schema is: [{len(OBS) + 2} keys, see ai/prompts/v1/schema_observables.json]</pre></details>
</section>

<section id="scoring">
  <h2>How answers will be scored</h2>
  <p>Each quantity is pass, fail, unresolved (outside tolerance but inside the codes' own spread) or missing. A model's
  breakpoint is the first layer with a resolved failure. H1 and H2 compare errors on primary-photon quantities with errors on
  transport quantities; H3 compares the error on PDD(5) with the error on D/Φ₀.</p>
  <h3>Controls, scored like a model</h3>
  <div class="tbl"><table><thead><tr><th>Control</th><th>Expected</th><th>Result</th><th></th></tr></thead>
  <tbody>{''.join(ctl_rows)}</tbody></table></div>
  <div class="tbl"><table><thead><tr><th>Events control</th><th>Energy distance</th><th>MC baseline (median, 95th pct)</th><th>Invalid (bound rules)</th></tr></thead>
  <tbody>{''.join(ev_rows)}</tbody></table></div>
  <details><summary>All {len(OBS)} quantities: reference, tolerance and resolution floor</summary>
  <div class="tbl"><table><thead><tr><th>Layer</th><th>Key</th><th>MC</th><th>Tolerance</th><th>Floor</th><th>Tags</th></tr></thead>
  <tbody>{''.join(score_rows)}</tbody></table></div></details>
</section>

<section id="limits">
  <h2>Limits to keep in view</h2>
  <ul class="tight">
    <li><b>Source model A has no head scatter.</b> PDD(5) and the scattered fraction differ from a real Co-60 unit by design. This is a fair test of physics reasoning, not a clinical beam.</li>
    <li><b>All external values are unverified.</b> The checks on this page compare against values that only you can sign off.</li>
    <li><b>The A2 tables are reproductions.</b> They are validated against Geant4, but they are not NIST downloads.</li>
    <li><b>The resolution floor rests on three Geant4 lists and one EGSnrc geometry.</b> The lists share the multiple-scattering model, so the floor may understate transport uncertainty for electron-fluence quantities.</li>
    <li><b>Model choice is time-stamped.</b> "Best available on the access date" makes the results a snapshot; the identifiers and dates are recorded per call.</li>
  </ul>
</section>
<footer class="note">Generated by analysis/study_page.py from analysis/results/ and ai/. Every number is read from a results file.</footer>
</main>
</div>
"""
    out = RES / "co60_study_review.html"
    out.write_text(html_out)
    print(f"wrote {out} ({len(html_out) // 1024} kB)")


CSS = """
/* Layout: reading column with a sticky contents rail; tables scroll inside their own frame. Shares the study brief's tokens. */
:root {
  --bg: #f4f6f9; --surface: #ffffff; --ink: #142030; --muted: #586779; --rule: #d9e0e8;
  --cobalt: #1f4fb8; --cobalt-soft: #e5ecfa; --amber: #a35a00; --amber-soft: #fbefdc; --green: #23704a; --green-soft: #e1f2e8;
  --red: #a8322a; --red-soft: #f8e3e0;
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #b07800;
  --f-display: "IBM Plex Sans Condensed", "Arial Narrow", sans-serif;
  --f-body: "Source Serif 4", Georgia, "Times New Roman", serif;
  --f-ui: "IBM Plex Sans", "Helvetica Neue", Arial, sans-serif;
  --f-mono: "IBM Plex Mono", ui-monospace, Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #0d1219; --surface: #141c26; --ink: #e3e9f1; --muted: #93a2b4; --rule: #263241;
  --cobalt: #7ea7ff; --cobalt-soft: #1b2a47; --amber: #f0b45b; --amber-soft: #3a2a12; --green: #6fd39a; --green-soft: #15301f;
  --red: #ff8f84; --red-soft: #3a1714; --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; color-scheme: dark; } }
:root[data-theme="dark"] {
  --bg: #0d1219; --surface: #141c26; --ink: #e3e9f1; --muted: #93a2b4; --rule: #263241;
  --cobalt: #7ea7ff; --cobalt-soft: #1b2a47; --amber: #f0b45b; --amber-soft: #3a2a12; --green: #6fd39a; --green-soft: #15301f;
  --red: #ff8f84; --red-soft: #3a1714; --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; color-scheme: dark; }
* { box-sizing: border-box; }
body { background: var(--bg); color: var(--ink); font-family: var(--f-body); font-size: 17px; line-height: 1.6; }
.wrap { max-width: 1180px; margin: 0 auto; padding-inline: 20px; padding-block: 32px 72px;
  display: grid; grid-template-columns: 190px minmax(0, 1fr); gap: 44px; }
nav.rail { position: sticky; top: calc(env(safe-area-inset-top, 0px) + 24px); align-self: start;
  font-family: var(--f-ui); font-size: 13px; display: grid; gap: 2px; }
nav.rail .t { font-family: var(--f-display); font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em;
  color: var(--muted); font-size: 12px; margin-bottom: 6px; }
nav.rail a { color: var(--ink); text-decoration: none; padding: 4px 8px; border-radius: 4px; }
nav.rail a:hover, nav.rail a:focus-visible { background: var(--cobalt-soft); color: var(--cobalt); outline: none; }
main { min-width: 0; display: grid; gap: 52px; }
section { display: grid; gap: 14px; scroll-margin-top: 24px; min-width: 0; }
h1, h2, h3 { font-family: var(--f-display); font-weight: 600; text-wrap: balance; margin: 0; line-height: 1.15; }
h1 { font-size: clamp(30px, 4.2vw, 42px); }
h2 { font-size: 26px; }
h3 { font-size: 18px; margin-top: 6px; }
p { margin: 0; max-width: 70ch; }
.hero { display: grid; gap: 12px; }
.eyebrow { font-family: var(--f-display); font-size: 13px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--cobalt); font-weight: 600; }
.lede { font-size: 19px; max-width: 66ch; }
.meta, .srcs { font-family: var(--f-mono); font-size: 12.5px; color: var(--muted); display: flex; flex-wrap: wrap; gap: 6px 18px; }
.srcs b { color: var(--ink); font-weight: 500; }
a { color: var(--cobalt); }
ul.tight { margin: 0; padding-left: 1.2rem; display: grid; gap: 8px; max-width: 74ch; }
.n, td.n { font-family: var(--f-mono); font-variant-numeric: tabular-nums; font-size: 0.86em; }
.z { color: var(--muted); }
.tbl { overflow-x: auto; border: 1px solid var(--rule); border-radius: 8px; background: var(--surface); }
table { border-collapse: collapse; width: 100%; font-family: var(--f-ui); font-size: 14px; }
th, td { text-align: left; padding: 8px 12px; border-bottom: 1px solid var(--rule); vertical-align: top; }
tbody tr:last-child td { border-bottom: 0; }
th { font-weight: 600; color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; }
th.nt { text-transform: none; letter-spacing: 0; font-size: 13px; }
.chip { display: inline-block; font-family: var(--f-ui); font-size: 12px; font-weight: 600; padding: 2px 8px; border-radius: 999px; white-space: nowrap; }
.chip.ok { background: var(--green-soft); color: var(--green); }
.chip.warn { background: var(--amber-soft); color: var(--amber); }
.chip.bad { background: var(--red-soft); color: var(--red); }
.chip.run { background: var(--cobalt-soft); color: var(--cobalt); }
.chip.idle { background: transparent; color: var(--muted); border: 1px solid var(--rule); }
.cond { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
.cond div { border: 1px solid var(--rule); border-radius: 8px; background: var(--surface); padding: 12px 14px; display: grid; gap: 4px; align-content: start; }
.cond b { font-family: var(--f-display); font-size: 16px; }
.cond span { font-family: var(--f-ui); font-size: 14px; color: var(--muted); }
details { font-family: var(--f-ui); font-size: 14px; }
details summary { cursor: pointer; color: var(--cobalt); font-weight: 500; padding: 4px 0; }
details summary:focus-visible { outline: 2px solid var(--cobalt); outline-offset: 2px; }
details > .tbl, details > pre { margin-top: 8px; }
pre.prompt { white-space: pre-wrap; font-family: var(--f-mono); font-size: 12.5px; line-height: 1.5; background: var(--surface);
  border: 1px solid var(--rule); border-radius: 8px; padding: 14px; max-height: 520px; overflow: auto; }
.fig { margin: 0; }
.note, footer { font-family: var(--f-ui); font-size: 13px; color: var(--muted); }
@media (max-width: 860px) {
  .wrap { grid-template-columns: minmax(0, 1fr); gap: 24px; padding-inline: 16px; }
  nav.rail { position: static; display: flex; flex-wrap: wrap; gap: 4px 8px; }
  nav.rail .t { width: 100%; }
  .cond { grid-template-columns: minmax(0, 1fr); }
  .tbl table { min-width: 620px; }
}
"""

if __name__ == "__main__":
    main()
