#!/usr/bin/env python3
"""M4 results page: the R0 AI answers scored against the Monte Carlo reference.

    python analysis/m4_page.py            # writes analysis/results/co60_m4_results.html

Reads analysis/results/M5_M4-R0.json (analysis/m5.py), analysis/results/reference_R0.json, ai/models.yaml and the
charges recorded in ai/raw/. Uses the study pages' design tokens (analysis/study_page.py).
"""
import html
import json
import math
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ai.harness.run import spent  # noqa: E402

RES = ROOT / "analysis" / "results"
TAG = "M4-R0"
NAMES = {"frontier_anthropic": "Claude Haiku 5.5", "frontier_openai": "GPT-6 Luna",
         "frontier_google": "Gemini 2.5 Flash-Lite", "open_weight_1": "DeepSeek V4 Flash",
         "open_weight_2": "Qwen3.7 Flash"}
ORDER = ["frontier_anthropic", "open_weight_1", "frontier_openai", "open_weight_2", "frontier_google"]
CONDS = ["A0", "A1", "A2"]
COND_TEXT = {"A0": "A0 recall, no tools", "A1": "A1 Python sandbox", "A2": "A2 Python + XCOM/ESTAR tables"}
COND_SHAPE = {"A0": "circle", "A1": "square", "A2": "triangle"}
COND_VAR = {"A0": "--s1", "A1": "--s2", "A2": "--s3"}
LAYERS = {1: "Photons arriving", 2: "Photon interactions", 3: "Electrons set in motion",
          4: "Electrons slowing down", 5: "Energy deposited"}
L5_KEYS = [("L5_D_over_phi0_Gy_cm2", "D/Φ₀"), ("L5_K_over_phi0_Gy_cm2", "K/Φ₀"),
           ("L5_Kcol_over_phi0_Gy_cm2", "K<sub>col</sub>/Φ₀"), ("L5_beta", "β"), ("L5_s_w_air", "s<sub>w,air</sub>"),
           ("L5_PDD5_percent", "PDD(5)")]
CLIP = 0.35          # dot-plot axis half-range (relative difference)


def e(s):
    return html.escape(str(s))


def pct(x, digits=1):
    if x is None or not math.isfinite(x):
        return "–"
    if abs(x) >= 10:
        return f"×{1 + x:,.0f}" if x > 0 else "×0"
    return f"{x * 100:+.{digits}f}%"


def chip(verdict):
    label = {"pass": "pass", "fail": "fail", "unresolved": "unresolved", "missing": "missing"}.get(verdict, verdict)
    return f'<span class="chip {e(verdict)}">{e(label)}</span>'


def conv_costs(tag):
    out = {}
    for d in (ROOT / "ai" / "raw" / tag).glob("*/*_r*"):
        c = 0.0
        for f in d.glob("turns/*_response.json"):
            c += (json.loads(f.read_text()).get("usage") or {}).get("cost") or 0.0
        out[d.parent.name] = out.get(d.parent.name, 0.0) + c
    return out


def marker(shape, x, y, r, var, tip):
    t = f' data-tip="{e(tip)}" tabindex="0"'
    if shape == "circle":
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="var({var})" stroke="var(--surface)" stroke-width="2"{t}/>'
    if shape == "square":
        s = r * 1.8
        return (f'<rect x="{x - s / 2:.1f}" y="{y - s / 2:.1f}" width="{s:.1f}" height="{s:.1f}" rx="1.5" '
                f'fill="var({var})" stroke="var(--surface)" stroke-width="2"{t}/>')
    h = r * 2.2
    return (f'<path d="M{x:.1f},{y - h * 0.6:.1f} L{x + h * 0.55:.1f},{y + h * 0.45:.1f} L{x - h * 0.55:.1f},'
            f'{y + h * 0.45:.1f} Z" fill="var({var})" stroke="var(--surface)" stroke-width="2"{t}/>')


def dose_plot(models, ref_mc):
    """Dot plot of the median relative difference of D/Phi0 per model and condition, with the repeat range."""
    W, ml, mr, mt, row, sub = 760, 178, 92, 44, 66, 17
    H = mt + row * len(ORDER) + 40
    pw = W - ml - mr
    X = lambda v: ml + (max(-CLIP, min(CLIP, v)) + CLIP) / (2 * CLIP) * pw  # noqa: E731
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-labelledby="dose-title dose-desc" class="chart">',
           '<title id="dose-title">D/Φ₀ relative to Monte Carlo, per model and condition</title>',
           '<desc id="dose-desc">Median of five repeats with the range over repeats. The shaded band is the ±2% '
           'pass tolerance. Values beyond ±35% are drawn at the edge with their value written next to them.</desc>']
    x0, x1 = X(-0.02), X(0.02)
    out.append(f'<rect x="{x0:.1f}" y="{mt - 8}" width="{x1 - x0:.1f}" height="{row * len(ORDER) + 8}" '
               f'fill="var(--green-soft)"/>')
    for t in (-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3):
        x = X(t)
        out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{mt - 8}" y2="{mt + row * len(ORDER)}" '
                   f'stroke="var(--rule)" stroke-width="{1.5 if t == 0 else 1}"/>')
        out.append(f'<text x="{x:.1f}" y="{mt + row * len(ORDER) + 18}" class="tick" text-anchor="middle">'
                   f'{t * 100:+.0f}%</text>' if t else
                   f'<text x="{x:.1f}" y="{mt + row * len(ORDER) + 18}" class="tick" text-anchor="middle">0</text>')
    out.append(f'<text x="{(x0 + x1) / 2:.1f}" y="{mt - 14}" class="tick band" text-anchor="middle">±2% pass</text>')
    out.append(f'<text x="{ml + pw / 2:.1f}" y="{H - 4}" class="axis" text-anchor="middle">'
               f'(AI − Monte Carlo) / Monte Carlo</text>')
    for i, slot in enumerate(ORDER):
        yc = mt + row * i + row / 2 - 4
        out.append(f'<text x="{ml - 14}" y="{yc + 4:.1f}" class="rowlab" text-anchor="end">{e(NAMES[slot])}</text>')
        if i:
            out.append(f'<line x1="{ml - 170}" x2="{W - 8}" y1="{mt + row * i - 4}" y2="{mt + row * i - 4}" '
                       f'stroke="var(--rule)" stroke-width="1" stroke-dasharray="2 3"/>')
        for j, c in enumerate(CONDS):
            y = yc + (j - 1) * sub
            s = models.get(slot, {}).get(c)
            o = (s or {}).get("observables", {}).get("L5_D_over_phi0_Gy_cm2", {})
            med = o.get("rel_diff_median")
            if med is None:
                out.append(f'<text x="{ml + 4}" y="{y + 4:.1f}" class="note">{c}: no valid answer</text>')
                continue
            lo, hi = o.get("rel_diff_range", [med, med])
            out.append(f'<line x1="{X(lo):.1f}" x2="{X(hi):.1f}" y1="{y:.1f}" y2="{y:.1f}" '
                       f'stroke="var({COND_VAR[c]})" stroke-width="2" stroke-linecap="round" opacity="0.55"/>')
            tip = (f"{NAMES[slot]} · {COND_TEXT[c]} · median {pct(med)} (range {pct(lo)} to {pct(hi)}, "
                   f"n = {s['n']}) · {o.get('verdict')}")
            out.append(marker(COND_SHAPE[c], X(med), y, 5.5, COND_VAR[c], tip))
            if abs(med) > CLIP:
                anchor, dx = ("end", -10) if med > 0 else ("start", 10)
                val = pct(med) if med > -1 else "0 (zeros)"
                out.append(f'<text x="{X(med) + dx:.1f}" y="{y + 4:.1f}" class="offscale" text-anchor="{anchor}">'
                           f'{e(val) + " →" if med > 0 else "← " + e(val)}</text>')
    out.append("</svg>")
    return "\n".join(out)


def layer_table(models):
    rows = []
    for slot in ORDER:
        for c in CONDS:
            s = models.get(slot, {}).get(c)
            if not s:
                rows.append(f'<tr><th scope="row">{e(NAMES[slot])}</th><td>{c}</td>'
                            f'<td colspan="6" class="muted">no valid answer</td></tr>')
                continue
            cells = []
            for L in range(1, 6):
                v = s["verdict_counts_by_layer"].get(str(L), {})
                n, p = sum(v.values()), v.get("pass", 0)
                share = p / n if n else 0
                cells.append(f'<td class="heat" style="--h:{share:.2f}" title="Layer {L}: {p} of {n} pass">'
                             f'<span class="n">{p}/{n}</span></td>')
            rows.append(f'<tr><th scope="row">{e(NAMES[slot])}</th><td>{c}</td><td class="n">{s["n"]}</td>'
                        + "".join(cells) + "</tr>")
    head = "".join(f'<th scope="col"><span class="L">L{L}</span> {e(LAYERS[L])}</th>' for L in range(1, 6))
    return (f'<table class="layers"><thead><tr><th scope="col">Model</th><th scope="col">Cond.</th>'
            f'<th scope="col">n</th>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>')


def l5_table(models, ref):
    head = "".join(f'<th scope="col">{lab}</th>' for _, lab in L5_KEYS)
    mcrow = "".join(f'<td class="n mc">{ref["observables"][k]["mc"]:.4g}</td>' for k, _ in L5_KEYS)
    rows = [f'<tr class="mcrow"><th scope="row">Monte Carlo</th><td></td>{mcrow}</tr>']
    for slot in ORDER:
        for c in CONDS:
            s = models.get(slot, {}).get(c)
            if not s:
                continue
            cells = []
            for k, _ in L5_KEYS:
                o = s["observables"][k]
                cells.append(f'<td><span class="n">{pct(o.get("rel_diff_median"))}</span> {chip(o["verdict"])}</td>')
            rows.append(f'<tr><th scope="row">{e(NAMES[slot])}</th><td>{c}</td>{"".join(cells)}</tr>')
    return (f'<table class="l5"><thead><tr><th scope="col">Model</th><th scope="col">Cond.</th>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


def hyp_table(models):
    rows = []
    for slot in ORDER:
        for c in CONDS:
            s = models.get(slot, {}).get(c)
            if not s:
                continue
            h1, h2, h3 = s.get("H1_median_abs_rel_diff"), s.get("H2_median_abs_rel_diff"), s["H3"]
            rows.append(f'<tr><th scope="row">{e(NAMES[slot])}</th><td>{c}</td>'
                        f'<td class="n">{pct(h1).lstrip("+")}</td><td class="n">{pct(h2).lstrip("+")}</td>'
                        f'<td class="n">{pct(h3["abs_rel_diff_PDD5"]).lstrip("+")}</td>'
                        f'<td class="n">{pct(h3["abs_rel_diff_D_over_phi0"]).lstrip("+")}</td></tr>')
    return ('<table class="hyp"><thead><tr><th scope="col">Model</th><th scope="col">Cond.</th>'
            '<th scope="col">H1 primary photon</th><th scope="col">H2 transport</th>'
            '<th scope="col">H3 PDD(5), recall</th><th scope="col">H3 D/Φ₀, computed</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


def events_table(events, counts, mc_ctl):
    rows = [f'<tr class="mcrow"><th scope="row">Monte Carlo resample</th><td></td><td class="n">1000</td><td></td>'
            f'<td class="n">{mc_ctl["invalid_fraction_bound"] * 100:.1f}%</td>'
            f'<td class="n">{mc_ctl["energy_distance"]:.4f}</td><td class="n">–</td>'
            f'<td class="n">{mc_ctl["process_fraction_ai"]["compt"] * 100:.0f}%</td></tr>']
    for slot in ORDER:
        for c in ("A0", "A1"):
            cnt = counts.get(slot, {}).get("events", {}).get(c, {})
            ev = events.get(slot, {}).get(c, {})
            calls = f'{cnt.get("ok", 0)} ok · {cnt.get("refusal", 0)} refused · {cnt.get("parse_failure", 0)} unreadable'
            if not ev.get("n_events"):
                rows.append(f'<tr><th scope="row">{e(NAMES[slot])}</th><td>{c}</td><td class="n">0</td>'
                            f'<td>{calls}</td><td colspan="4" class="muted">no events to score</td></tr>')
                continue
            b = ev["energy_distance_mc_baseline"]["p95"]
            inv = ev["invalid_fraction_bound"]
            flag = "pass" if inv <= 0.01 else "fail"
            ed_flag = "pass" if ev["energy_distance"] <= b else "fail"
            rows.append(f'<tr><th scope="row">{e(NAMES[slot])}</th><td>{c}</td><td class="n">{ev["n_events"]}</td>'
                        f'<td>{calls}</td>'
                        f'<td><span class="n">{inv * 100:.1f}%</span> {chip(flag)}</td>'
                        f'<td><span class="n">{ev["energy_distance"]:.4f}</span> {chip(ed_flag)}</td>'
                        f'<td class="n">{b:.4f}</td>'
                        f'<td class="n">{ev["process_fraction_ai"]["compt"] * 100:.0f}%</td></tr>')
    return ('<table class="ev"><thead><tr><th scope="col">Model</th><th scope="col">Cond.</th>'
            '<th scope="col">Events</th><th scope="col">Calls (of 5)</th><th scope="col">Invalid, bound rule</th>'
            '<th scope="col">Energy distance</th><th scope="col">MC 95th pct</th><th scope="col">Compton share</th>'
            f'</tr></thead><tbody>{"".join(rows)}</tbody></table>')


def outcome_table(counts, costs):
    rows = []
    for slot in ORDER:
        for task in ("observables", "events"):
            cs = counts.get(slot, {}).get(task, {})
            tot = {k: sum(v.get(k, 0) for v in cs.values()) for k in ("ok", "parse_failure", "refusal", "api_error",
                                                                       "tool_calls")}
            n = sum(tot[k] for k in ("ok", "parse_failure", "refusal", "api_error"))
            first = task == "observables"
            cost = (f'<td class="n" rowspan="2">${costs.get(slot, 0):.2f}</td>' if first else "")
            name = f'<th scope="row" rowspan="2">{e(NAMES[slot])}</th>' if first else ""
            rows.append(f'<tr>{name}<td>{task}</td><td class="n">{n}</td><td class="n">{tot["ok"]}</td>'
                        f'<td class="n">{tot["parse_failure"]}</td><td class="n">{tot["refusal"]}</td>'
                        f'<td class="n">{tot["api_error"]}</td><td class="n">{tot["tool_calls"]}</td>{cost}</tr>')
    return ('<table class="out"><thead><tr><th scope="col">Model</th><th scope="col">Task</th>'
            '<th scope="col">Conversations</th><th scope="col">Valid</th><th scope="col">Unreadable</th>'
            '<th scope="col">Refused</th><th scope="col">API error</th><th scope="col">Python calls</th>'
            f'<th scope="col">R0 charge</th></tr></thead><tbody>{"".join(rows)}</tbody></table>')


def best(models, cond):
    out = []
    for slot in ORDER:
        o = models.get(slot, {}).get(cond, {}).get("observables", {}).get("L5_D_over_phi0_Gy_cm2", {})
        if o.get("rel_diff_median") is not None:
            out.append((abs(o["rel_diff_median"]), slot))
    return sorted(out)


CSS = """
:root {
  /* Layout: one reading column, figures and tables full width inside it; tokens shared with the study pages. */
  --bg: #f4f6f9; --surface: #ffffff; --ink: #142030; --muted: #586779; --rule: #d9e0e8;
  --cobalt: #1f4fb8; --cobalt-soft: #e5ecfa; --amber: #a35a00; --amber-soft: #fbefdc; --green: #23704a;
  --green-soft: #e1f2e8; --red: #a8322a; --red-soft: #f8e3e0;
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a;
  --f-display: "IBM Plex Sans Condensed", "Arial Narrow", sans-serif;
  --f-body: "Source Serif 4", Georgia, "Times New Roman", serif;
  --f-ui: "IBM Plex Sans", "Helvetica Neue", Arial, sans-serif;
  --f-mono: "IBM Plex Mono", ui-monospace, Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #0d1219; --surface: #141c26; --ink: #e3e9f1; --muted: #93a2b4; --rule: #263241;
  --cobalt: #7ea7ff; --cobalt-soft: #1b2a47; --amber: #f0b45b; --amber-soft: #3a2a12; --green: #6fd39a;
  --green-soft: #15301f; --red: #ff8f84; --red-soft: #3a1714; --s1: #3987e5; --s2: #d95926; --s3: #199e70;
  color-scheme: dark; } }
:root[data-theme="dark"] {
  --bg: #0d1219; --surface: #141c26; --ink: #e3e9f1; --muted: #93a2b4; --rule: #263241;
  --cobalt: #7ea7ff; --cobalt-soft: #1b2a47; --amber: #f0b45b; --amber-soft: #3a2a12; --green: #6fd39a;
  --green-soft: #15301f; --red: #ff8f84; --red-soft: #3a1714; --s1: #3987e5; --s2: #d95926; --s3: #199e70;
  color-scheme: dark; }
body { background: var(--bg); color: var(--ink); font-family: var(--f-body); font-size: 17px; line-height: 1.6; }
main { max-width: 1000px; margin: 0 auto; padding-inline: 20px; padding-block: 36px 64px; display: grid; gap: 40px; }
section { display: grid; gap: 14px; min-width: 0; }
h1, h2, h3 { font-family: var(--f-display); font-weight: 600; text-wrap: balance; margin: 0; line-height: 1.15; }
h1 { font-size: clamp(30px, 5vw, 44px); letter-spacing: -0.01em; }
h2 { font-size: 26px; }
p { margin: 0; max-width: 68ch; }
.eyebrow { font-family: var(--f-display); font-size: 13px; letter-spacing: 0.1em; text-transform: uppercase;
  color: var(--cobalt); font-weight: 600; }
.meta { font-family: var(--f-mono); font-size: 12.5px; color: var(--muted); display: flex; flex-wrap: wrap;
  gap: 6px 18px; }
header { display: grid; gap: 12px; }
.lede { font-size: 19px; }
.findings { display: grid; gap: 12px; padding: 0; margin: 0; list-style: none; max-width: 76ch; }
.findings li { padding-left: 18px; border-left: 3px solid var(--rule); }
.findings li b { font-family: var(--f-ui); font-weight: 600; font-size: 0.95em; }
.panel { background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 18px; min-width: 0; }
.scroll { overflow-x: auto; }
.chart { width: 100%; min-width: 640px; height: auto; display: block; font-family: var(--f-ui); }
.chart .tick { font-size: 12px; fill: var(--muted); font-variant-numeric: tabular-nums; }
.chart .tick.band { fill: var(--green); font-weight: 600; }
.chart .axis { font-size: 12.5px; fill: var(--muted); }
.chart .rowlab { font-size: 14px; fill: var(--ink); font-weight: 500; }
.chart .offscale { font-size: 12px; fill: var(--ink); font-family: var(--f-mono); }
.chart .note { font-size: 12px; fill: var(--muted); font-style: italic; }
.chart [data-tip] { cursor: default; }
.chart [data-tip]:focus { outline: none; stroke: var(--ink); }
.legend { display: flex; flex-wrap: wrap; gap: 8px 22px; font-family: var(--f-ui); font-size: 13.5px; color: var(--ink); }
.legend span { display: inline-flex; align-items: center; gap: 7px; }
.legend svg { width: 14px; height: 14px; }
.caption { font-family: var(--f-ui); font-size: 13.5px; color: var(--muted); max-width: 80ch; }
table { border-collapse: collapse; width: 100%; font-family: var(--f-ui); font-size: 14px; }
th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--rule); vertical-align: middle; }
thead th { font-weight: 600; font-size: 12.5px; color: var(--muted); border-bottom: 1.5px solid var(--ink);
  vertical-align: bottom; }
tbody th { font-weight: 500; white-space: nowrap; }
.n { font-family: var(--f-mono); font-variant-numeric: tabular-nums; font-size: 0.92em; }
td.n { white-space: nowrap; }
td { white-space: nowrap; }
.muted { color: var(--muted); }
.mcrow th, .mcrow td { color: var(--muted); font-style: italic; }
.chip { display: inline-block; font-family: var(--f-ui); font-size: 11px; font-weight: 600; letter-spacing: 0.03em;
  padding: 1px 7px; border-radius: 10px; text-transform: uppercase; margin-left: 4px; }
.chip.pass { background: var(--green-soft); color: var(--green); }
.chip.fail { background: var(--red-soft); color: var(--red); }
.chip.unresolved { background: var(--amber-soft); color: var(--amber); }
.chip.missing { background: var(--cobalt-soft); color: var(--muted); }
.layers .L { font-family: var(--f-mono); color: var(--cobalt); margin-right: 4px; }
.layers thead th { white-space: normal; min-width: 92px; }
.heat { background: color-mix(in srgb, var(--cobalt) calc(var(--h) * 55%), var(--surface)); text-align: center; }
.costs { display: flex; flex-wrap: wrap; gap: 12px 40px; font-family: var(--f-ui); }
.costs div { display: grid; gap: 2px; }
.costs .v { font-family: var(--f-mono); font-size: 24px; font-variant-numeric: tabular-nums; }
.costs .k { font-size: 13px; color: var(--muted); }
.notes { display: grid; gap: 10px; padding-left: 20px; margin: 0; max-width: 78ch; font-size: 16px; }
code { font-family: var(--f-mono); font-size: 0.86em; }
#tip { position: fixed; pointer-events: none; z-index: 10; max-width: 320px; background: var(--ink); color: var(--bg);
  font-family: var(--f-ui); font-size: 12.5px; line-height: 1.4; padding: 6px 9px; border-radius: 4px; }
@media (max-width: 560px) { body { font-size: 16px; } main { padding-inline: 16px; } h2 { font-size: 22px; } }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
"""

JS = """
(function () {
  var tip = document.getElementById('tip');
  function show(ev) {
    var t = ev.target.closest('[data-tip]'); if (!t) return;
    tip.textContent = t.getAttribute('data-tip'); tip.hidden = false;
    var r = t.getBoundingClientRect(), x = r.left + r.width / 2, y = r.top;
    var w = tip.offsetWidth;
    tip.style.left = Math.max(8, Math.min(window.innerWidth - w - 8, x - w / 2)) + 'px';
    tip.style.top = Math.max(8, y - tip.offsetHeight - 10) + 'px';
  }
  function hide() { tip.hidden = true; }
  document.querySelectorAll('.chart').forEach(function (c) {
    c.addEventListener('mouseover', show); c.addEventListener('focusin', show);
    c.addEventListener('mouseout', hide); c.addEventListener('focusout', hide);
  });
})();
"""


def legend():
    items = []
    for c in CONDS:
        shape = marker(COND_SHAPE[c], 7, 7, 5, COND_VAR[c], "").replace(' data-tip="" tabindex="0"', "")
        items.append(f'<span><svg viewBox="0 0 14 14" aria-hidden="true">{shape}</svg>{e(COND_TEXT[c])}</span>')
    return f'<div class="legend">{"".join(items)}</div>'


def build():
    r = json.load(open(RES / f"M5_{TAG}.json"))
    ref = json.load(open(RES / "reference_R0.json"))
    ctl = json.load(open(RES / "M5_controls.json"))["event_controls"]["positive_mc_resample"]
    models, counts, events = r["models"], r["counts"], r["events"]
    costs = conv_costs(TAG)
    r0_total, void = spent(ROOT / "ai" / "raw" / TAG), spent(ROOT / "ai" / "raw" / f"{TAG}-void")
    p0_total = sum(spent(ROOT / "ai" / "raw" / t) for t in ("P0-pilot", "P0-pilot-cache", "P0-pilot-killed"))
    n_conv = sum(1 for _ in (ROOT / "ai" / "raw" / TAG).glob("*/*_r*/meta.yaml"))
    mod = yaml.safe_load(open(ROOT / "ai" / "models.yaml"))
    ids = " · ".join(f'{NAMES[s["slot"]]} <code>{e(s["model_id"])}</code>' for s in mod["slots"])
    ref_d = ref["observables"]["L5_D_over_phi0_Gy_cm2"]

    tools_pass = [NAMES[s] for s in ORDER if all(
        models.get(s, {}).get(c, {}).get("observables", {}).get("L5_D_over_phi0_Gy_cm2", {}).get("verdict") == "pass"
        for c in ("A1", "A2"))]
    a0 = best(models, "A0")
    a0_best = f"{NAMES[a0[0][1]]} at {pct(models[a0[0][1]]['A0']['observables']['L5_D_over_phi0_Gy_cm2']['rel_diff_median'])}" if a0 else "–"
    breaks = {s.get("breakpoint_layer") for conds in models.values() for s in conds.values()}

    body = f"""
<title>Co-60 M4 Results</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap">
<style>{CSS}</style>
<main>
<header>
  <div class="eyebrow">Co-60 reference point · M4 AI runs · scored against Monte Carlo</div>
  <h1>Five fast models at the Co-60 reference point</h1>
  <div class="meta"><span>R0: central axis, 5 cm depth, water, 10 × 10 cm², SSD 100 cm</span>
    <span>{n_conv} conversations</span><span>5 models × 3 conditions</span><span>run 2026-10-12</span>
    <span>branch m4-ai-runs</span></div>
  <p class="lede">Each model was asked for 33 quantities that follow the energy from the incident photon to the
  absorbed dose, five times per condition, and for 1,000 sampled photon interactions. Answers are scored against the
  study's Monte Carlo reference (Geant4 option4, boundary-free dose; D/Φ₀ = {ref_d["mc"]:.4e} Gy cm²).</p>
</header>

<section aria-labelledby="h-find">
  <h2 id="h-find">What the runs show</h2>
  <ul class="findings">
    <li><b>Python makes the dose right for two models.</b> With a sandbox, {e(" and ".join(tools_pass) or "no model")}
      put D/Φ₀, kerma, β, s<sub>w,air</sub> and PDD(5) within the 2% tolerance in both tool conditions.</li>
    <li><b>Without tools, every model misses the dose.</b> The closest A0 answer is {e(a0_best)}; the others are off by 5%
      to 18%, and Gemini by a factor of about 3,000.</li>
    <li><b>Tools do not help the other three.</b> GPT-6 Luna stays 7–11% low on D/Φ₀ and 20% high on the mean photon
      energy. Qwen3.7 Flash moves further from the reference with tools. Gemini 2.5 Flash-Lite made no Python call on the
      observables task, answered with zeros in two A1 runs and reported dose in the wrong unit in A2.</li>
    <li><b>Every model breaks at Layer {", ".join(str(b) for b in sorted(x for x in breaks if x)) or "–"}.</b> The
      breakpoint is the first layer with a resolved failure. Spectral and angular shapes must match the Monte Carlo
      within three times its own half-sample distance, and no answer does, even where the scalar quantities pass.</li>
    <li><b>Sampled events.</b> Under the bound-electron rule, Haiku's events (A0 and A1) and DeepSeek's A0 events are
      almost all physically valid; DeepSeek's A1 events are 13% invalid, Qwen's 51% and Gemini's 28–66%. Two
      samples match the Monte Carlo distribution: Haiku's A1 events and the one A1 call GPT-6 Luna answered. It
      declined the other 9 event calls, saying it cannot sample without a transport code.</li>
  </ul>
</section>

<section aria-labelledby="h-dose">
  <h2 id="h-dose">Dose per unit fluence, D/Φ₀</h2>
  <p>Median relative difference over the valid repeats, with the range of the repeats as a line. Hover or focus a
  marker for the numbers.</p>
  <div class="panel">{legend()}<div class="scroll">{dose_plot(models, ref_d["mc"])}</div></div>
  <p class="caption">Shaded: the ±2% pass band for Layer 5 dose quantities (DECISIONS D028). Values outside ±35% are
  drawn at the axis edge with their value. Gemini A2 reports about 1.9 × 10⁶ times the reference, and A0 about
  3,000 times: dose in a different unit. Two Gemini A1 answers were all zeros.</p>
</section>

<section aria-labelledby="h-l5">
  <h2 id="h-l5">Layer 5 quantities</h2>
  <p>Median relative difference and the verdict of the median answer. Monte Carlo values in the first row
  (D, K and K<sub>col</sub> per Φ₀ in Gy cm², PDD in percent).</p>
  <div class="panel scroll">{l5_table(models, ref)}</div>
</section>

<section aria-labelledby="h-layers">
  <h2 id="h-layers">Passes per layer</h2>
  <p>Quantities that pass, out of those scored in each layer, for the median answer. Darker cells pass more.</p>
  <div class="panel scroll">{layer_table(models)}</div>
  <p class="caption">Scalars pass within max(5%, 0.005 absolute) in Layers 1–4 and 2% for Layer 5 dose quantities.
  Distributions pass when W1 and JSD are within 3 × the Monte Carlo half-sample distance. A failure inside the
  spread of the three Geant4 physics lists counts as unresolved (D028 item 4).</p>
</section>

<section aria-labelledby="h-hyp">
  <h2 id="h-hyp">Hypotheses</h2>
  <p>Median absolute relative difference of the scalar quantities tagged H1 (primary photons) and H2 (transport),
  and the two H3 quantities. H1 and H2 hold where the H1 column is smaller than H2; H3 holds where the PDD(5) error
  is smaller than the D/Φ₀ error.</p>
  <div class="panel scroll">{hyp_table(models)}</div>
</section>

<section aria-labelledby="h-ev">
  <h2 id="h-ev">Sampled interaction events</h2>
  <p>Five calls of 200 events per model in A0 and A1, pooled. Validity uses the bound-electron rule of DECISIONS
  D014 (energy conserved within the oxygen K-shell binding energy, Compton kinematics within δ<sub>C</sub>). The
  energy distance compares the pooled events with the Monte Carlo sample; it passes below the 95th percentile of
  Monte Carlo samples of the same size.</p>
  <div class="panel scroll">{events_table(events, counts, ctl)}</div>
  <p class="caption">Under the strict free-electron rule the Monte Carlo's own events fail
  {ctl["invalid_fraction_strict"] * 100:.0f}% of the time (bound Compton scattering with Doppler broadening), so that
  rule is reported in <code>M5_M4-R0.json</code> for context only.</p>
</section>

<section aria-labelledby="h-runs">
  <h2 id="h-runs">Runs and cost</h2>
  <div class="costs">
    <div><span class="v">${r0_total:.2f}</span><span class="k">R0, measured OpenRouter charge</span></div>
    <div><span class="v">${p0_total:.2f}</span><span class="k">P0 pilot and the cached-Haiku test</span></div>
    <div><span class="v">${void:.3f}</span><span class="k">void attempts (API errors, cut runs)</span></div>
  </div>
  <div class="panel scroll">{outcome_table(counts, costs)}</div>
  <p class="caption">Unreadable: no answer matched the required JSON format after one retry. Refused: valid JSON with
  no content. API error: none remain; Gemini's four were rerun (D035).</p>
</section>

<section aria-labelledby="h-notes">
  <h2 id="h-notes">Read these results with</h2>
  <ul class="notes">
    <li>The models are each provider's fast, low-cost tier, not the most capable model (D030). Qwen3.7 Flash fills an
      open-weight slot but has no published weights.</li>
    <li>Haiku runs with prompt caching to stay inside the $10 budget; the model sees the same prompts (D032).</li>
    <li>Event answers count with 150 to 250 events; the prompt asks for 200 (D031).</li>
    <li>Gemini was rate-limited by its upstream throughout; its four API-error conversations were rerun (D033,
      D035).</li>
    <li>A malformed fenced JSON block gets a misleading retry message in this harness version (D034). It affects
      which answers are readable, not how readable answers score.</li>
    <li>The Monte Carlo reference is final (M2, not provisional). Raw conversations are in <code>ai/raw/M4-R0/</code>;
      scores in <code>analysis/results/M5_M4-R0.json</code>.</li>
  </ul>
  <p class="meta">{ids}</p>
</section>
</main>
<div id="tip" hidden></div>
<script>{JS}</script>
"""
    out = RES / "co60_m4_results.html"
    out.write_text(body)
    print(f"wrote {out}")


if __name__ == "__main__":
    build()
