"""HTML template for the M1 results page. Called by analysis/m1_report.py."""

import html
import math


def g4_label(raw):
    """'Geant4 version Name: geant4-11-04-patch-03 [MT] (01-October-2026)' -> 'Geant4 11.4.3 (MT)'."""
    import re
    m = re.search(r"geant4-(\d+)-(\d+)(?:-patch-(\d+))?", raw)
    if not m:
        return raw.strip()
    ver = f"{int(m.group(1))}.{int(m.group(2))}.{int(m.group(3) or 0)}"
    return f"Geant4 {ver}" + (" (MT)" if "[MT]" in raw else "")


def esc(s):
    return html.escape(str(s))


CSS = r"""
/* Layout: a narrow chain rail beside one reading column; the chain is the page's spine. */
:root {
  --bg: #f4f6f9; --panel: #ffffff; --ink: #101828; --ink-2: #475467; --muted: #7b8698;
  --rule: #dfe4ec; --axis: #c3cad6; --accent: #2346c8; --accent-ink: #ffffff;
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100;
  --ok: #0a7d32; --ok-bg: #e5f4ea; --warn: #8a5a00; --warn-bg: #fdf2d8; --bad: #b42318; --bad-bg: #fde8e7;
  --font-display: "IBM Plex Sans Condensed", "Arial Narrow", system-ui, sans-serif;
  --font-body: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --font-data: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #0b1020; --panel: #121a2c; --ink: #e8ecf4; --ink-2: #b4bdcc; --muted: #8590a5;
  --rule: #243049; --axis: #39445d; --accent: #8aa2ff; --accent-ink: #0b1020;
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500;
  --ok: #5bd78a; --ok-bg: #12301f; --warn: #f5c451; --warn-bg: #33270b; --bad: #ff8a80; --bad-bg: #3a1513;
  color-scheme: dark; } }
:root[data-theme="dark"] {
  --bg: #0b1020; --panel: #121a2c; --ink: #e8ecf4; --ink-2: #b4bdcc; --muted: #8590a5;
  --rule: #243049; --axis: #39445d; --accent: #8aa2ff; --accent-ink: #0b1020;
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500;
  --ok: #5bd78a; --ok-bg: #12301f; --warn: #f5c451; --warn-bg: #33270b; --bad: #ff8a80; --bad-bg: #3a1513;
  color-scheme: dark; }
body { background: var(--bg); color: var(--ink); font: 15px/1.55 var(--font-body); }
.wrap { max-width: 1180px; margin: 0 auto; padding-inline: 20px; padding-block: 28px 64px;
  display: grid; grid-template-columns: 220px minmax(0, 1fr); gap: 40px; }
@media (max-width: 860px) { .wrap { grid-template-columns: minmax(0, 1fr); gap: 20px; padding-inline: 16px; } }
h1, h2, h3 { font-family: var(--font-display); text-wrap: balance; margin: 0; letter-spacing: 0.005em; }
h1 { font-size: 2.15rem; line-height: 1.12; font-weight: 600; }
h2 { font-size: 1.45rem; font-weight: 600; }
h3 { font-size: 1.05rem; font-weight: 600; }
p { margin: 0; max-width: 68ch; }
a { color: var(--accent); }
sup { font-size: 0.7em; }
.mono, .num { font-family: var(--font-data); font-variant-numeric: tabular-nums; }
.eyebrow { font: 600 0.72rem/1 var(--font-body); letter-spacing: 0.09em; text-transform: uppercase; color: var(--muted); }
.rail { position: sticky; top: calc(env(safe-area-inset-top, 0px) + 20px); align-self: start; display: grid; gap: 14px; }
@media (max-width: 860px) { .rail { position: static; } }
.rail nav { display: grid; gap: 2px; }
.rail a { text-decoration: none; color: var(--ink-2); padding: 6px 10px; border-radius: 6px; display: grid;
  grid-template-columns: 22px 1fr; gap: 6px; font-size: 0.9rem; }
.rail a:hover, .rail a:focus-visible { background: var(--panel); color: var(--ink); outline: none; }
.rail a .k { font-family: var(--font-data); color: var(--muted); }
.rail .sep { height: 1px; background: var(--rule); margin: 6px 10px; }
main { display: grid; gap: 44px; min-width: 0; }
header.top { display: grid; gap: 14px; }
.meta { display: flex; flex-wrap: wrap; gap: 6px 18px; color: var(--ink-2); font-size: 0.88rem; }
.chips { display: flex; flex-wrap: wrap; gap: 8px; }
.chip { display: inline-flex; align-items: center; gap: 6px; padding: 3px 10px; border-radius: 999px;
  font-size: 0.8rem; font-weight: 600; border: 1px solid transparent; }
.chip.ok { color: var(--ok); background: var(--ok-bg); }
.chip.warn { color: var(--warn); background: var(--warn-bg); }
.chip.bad { color: var(--bad); background: var(--bad-bg); }
.chip svg { width: 12px; height: 12px; }
/* The chain: five links from photon to dose */
.chain { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 0; background: var(--panel);
  border: 1px solid var(--rule); border-radius: 10px; overflow: hidden; }
@media (max-width: 1000px) { .chain { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media (max-width: 520px) { .chain { grid-template-columns: minmax(0, 1fr); } }
.link { padding: 16px 16px 18px; display: grid; gap: 8px; align-content: start; border-right: 1px solid var(--rule);
  position: relative; text-decoration: none; color: inherit; }
.link:last-child { border-right: 0; }
.link:hover { background: color-mix(in srgb, var(--accent) 5%, var(--panel)); }
.link .k { font: 600 0.72rem/1 var(--font-data); color: var(--accent); }
.link .t { font: 600 0.98rem/1.25 var(--font-display); }
.link .big { font: 500 1.3rem/1.15 var(--font-data); letter-spacing: -0.01em; }
.link .sub { font-size: 0.8rem; color: var(--ink-2); }
section { display: grid; gap: 16px; scroll-margin-top: 16px; }
.sec-head { display: grid; gap: 6px; border-top: 2px solid var(--ink); padding-top: 12px; }
.sec-head .k { font: 600 0.78rem/1 var(--font-data); color: var(--accent); }
.lede { color: var(--ink-2); }
.grid2 { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }
@media (max-width: 760px) { .grid2 { grid-template-columns: minmax(0, 1fr); } }
.fig { background: var(--panel); border: 1px solid var(--rule); border-radius: 10px; padding: 14px 14px 10px; display: grid;
  gap: 8px; min-width: 0; }
.fig .cap { font-size: 0.82rem; color: var(--ink-2); }
.legend { display: flex; flex-wrap: wrap; gap: 4px 14px; font-size: 0.8rem; color: var(--ink-2); }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.legend i { width: 14px; height: 3px; border-radius: 2px; display: inline-block; }
.chart { position: relative; width: 100%; min-height: 220px; }
.chart svg { display: block; width: 100%; height: auto; overflow: visible; }
.chart text { fill: var(--muted); font: 11px var(--font-data); }
.chart .lbl { fill: var(--ink-2); font: 11px var(--font-body); }
.tip { position: absolute; pointer-events: none; background: var(--panel); border: 1px solid var(--rule); color: var(--ink);
  border-radius: 6px; padding: 6px 8px; font: 12px/1.4 var(--font-data); box-shadow: 0 4px 14px rgba(0,0,0,0.12);
  white-space: nowrap; z-index: 2; }
.tip b { font-weight: 600; }
.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 10px; }
.stat { background: var(--panel); border: 1px solid var(--rule); border-radius: 10px; padding: 12px 14px; display: grid; gap: 4px; }
.stat .v { font: 500 1.12rem/1.2 var(--font-data); }
.stat .l { font-size: 0.8rem; color: var(--ink-2); }
.tbl { overflow-x: auto; background: var(--panel); border: 1px solid var(--rule); border-radius: 10px; }
table { border-collapse: collapse; width: 100%; font-size: 0.86rem; }
th, td { text-align: left; padding: 7px 12px; border-bottom: 1px solid var(--rule); vertical-align: top; }
th { font-weight: 600; color: var(--ink-2); font-size: 0.78rem; letter-spacing: 0.02em; }
tr:last-child td { border-bottom: 0; }
td.num, th.num { text-align: right; font-family: var(--font-data); font-variant-numeric: tabular-nums; white-space: nowrap; }
.notes { display: grid; gap: 12px; }
.note { background: var(--panel); border: 1px solid var(--rule); border-left: 3px solid var(--accent); border-radius: 8px;
  padding: 12px 14px; display: grid; gap: 6px; }
.note.warn { border-left-color: var(--warn); }
.note h3 { font-size: 1rem; }
.note p { color: var(--ink-2); font-size: 0.92rem; }
ol.decide { margin: 0; padding-left: 1.3rem; display: grid; gap: 10px; max-width: 72ch; }
ol.decide li { padding-left: 4px; }
ol.decide b { font-weight: 600; }
footer { color: var(--muted); font-size: 0.82rem; display: grid; gap: 6px; border-top: 1px solid var(--rule); padding-top: 14px; }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
"""

JS = r"""
(function () {
  const D = JSON.parse(document.getElementById('m1-data').textContent);
  const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const NS = 'http://www.w3.org/2000/svg';
  const el = (tag, attrs, parent) => { const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; };
  const fmt = (x) => { if (x === 0) return '0'; const a = Math.abs(x);
    return (a >= 1e4 || a < 1e-3) ? x.toExponential(2) : x.toPrecision(3); };
  const SUP = {'-': '⁻', '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹'};
  const fmtLog = (x) => { const e = Math.round(Math.log10(x));
    return (e >= -2 && e <= 2) ? String(+x.toPrecision(1)) : '10' + String(e).split('').map(c => SUP[c]).join(''); };

  function ticksLog(lo, hi) { const t = []; for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++) {
    const v = Math.pow(10, e); if (v >= lo * 0.999 && v <= hi * 1.001) t.push(v); } return t; }
  function ticksLin(lo, hi, n) { const span = hi - lo, raw = span / n, p = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map(m => m * p).find(s => span / s <= n) || raw; const t = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) t.push(+v.toFixed(10)); return t; }

  // Step chart of binned data: series [{name, color, edges, y, err}]
  function stepChart(host, cfg) {
    const draw = () => {
      host.innerHTML = '';
      const W = Math.max(host.clientWidth, 280), H = cfg.height || Math.round(Math.min(320, Math.max(220, W * 0.5)));
      const m = {l: 66, r: 14, t: 10, b: 38};
      const svg = el('svg', {viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': cfg.title || ''}, host);
      const xs = cfg.series.flatMap(s => s.edges), ys = cfg.series.flatMap(s => s.y.filter(v => v > 0));
      const x0 = cfg.xdomain ? cfg.xdomain[0] : Math.min(...xs), x1 = cfg.xdomain ? cfg.xdomain[1] : Math.max(...xs);
      let y0, y1;
      if (cfg.ylog) { y1 = Math.max(...ys) * 2; y0 = cfg.yfloor ? y1 * cfg.yfloor : Math.min(...ys) / 2; }
      else { y0 = cfg.ymin ?? 0; y1 = (cfg.ymax ?? Math.max(...cfg.series.flatMap(s => s.y))) * 1.08; }
      const X = (x) => m.l + (x - x0) / (x1 - x0) * (W - m.l - m.r);
      const Y = (y) => cfg.ylog ? (H - m.b) - (Math.log10(Math.max(y, y0)) - Math.log10(y0)) / (Math.log10(y1) - Math.log10(y0)) * (H - m.t - m.b)
                                : (H - m.b) - (y - y0) / (y1 - y0) * (H - m.t - m.b);
      const yt = cfg.ylog ? ticksLog(y0, y1) : ticksLin(y0, y1, 5);
      yt.forEach(t => { el('line', {x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t), stroke: css('--rule'), 'stroke-width': 1}, svg);
        el('text', {x: m.l - 6, y: Y(t) + 4, 'text-anchor': 'end'}, svg).textContent = cfg.ylog ? fmtLog(t) : fmt(t); });
      ticksLin(x0, x1, cfg.xticks || 7).forEach(t => { el('text', {x: X(t), y: H - m.b + 16, 'text-anchor': 'middle'}, svg).textContent = t; });
      el('line', {x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, stroke: css('--axis')}, svg);
      el('text', {x: (m.l + W - m.r) / 2, y: H - 4, 'text-anchor': 'middle', class: 'lbl'}, svg).textContent = cfg.xlabel;
      const yl = el('text', {x: 12, y: (m.t + H - m.b) / 2, 'text-anchor': 'middle', class: 'lbl',
        transform: `rotate(-90 12 ${(m.t + H - m.b) / 2})`}, svg); yl.textContent = cfg.ylabel;
      (cfg.marks || []).forEach(mk => { const x = X(mk.x);
        el('line', {x1: x, x2: x, y1: m.t, y2: H - m.b, stroke: css('--muted'), 'stroke-dasharray': '3 3'}, svg);
        el('text', {x: x + 4, y: m.t + 12 + (mk.dy || 0), class: 'lbl'}, svg).textContent = mk.label; });
      cfg.series.forEach(s => {
        let d = ''; s.y.forEach((y, i) => { const a = X(s.edges[i]), b = X(s.edges[i + 1]);
          if (s.edges[i + 1] <= x0 || s.edges[i] >= x1) return; const yy = Y(y);
          d += (d ? 'L' : 'M') + a.toFixed(1) + ',' + yy.toFixed(1) + 'L' + b.toFixed(1) + ',' + yy.toFixed(1); });
        el('path', {d, fill: 'none', stroke: css(s.color), 'stroke-width': 2, 'stroke-linejoin': 'round'}, svg); });
      // hover layer: crosshair and per-bin readout
      const tip = document.createElement('div'); tip.className = 'tip'; tip.hidden = true; host.appendChild(tip);
      const hair = el('line', {y1: m.t, y2: H - m.b, stroke: css('--axis'), 'stroke-width': 1, visibility: 'hidden'}, svg);
      const hit = el('rect', {x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: 'transparent'}, svg);
      const move = (ev) => { const r = svg.getBoundingClientRect(), px = (ev.clientX - r.left) * W / r.width;
        const xv = x0 + (px - m.l) / (W - m.l - m.r) * (x1 - x0); const e = cfg.series[0].edges;
        let i = e.findIndex((v, k) => k < e.length - 1 && xv >= v && xv < e[k + 1]); if (i < 0) { tip.hidden = true; return; }
        hair.setAttribute('x1', px); hair.setAttribute('x2', px); hair.setAttribute('visibility', 'visible');
        tip.innerHTML = `<b>${e[i].toFixed(3)}–${e[i + 1].toFixed(3)} ${cfg.xunit || ''}</b><br>` +
          cfg.series.map(s => `<span style="color:${css(s.color)}">■</span> ${s.name}: ${fmt(s.y[i])}` +
            (s.err ? ` ± ${fmt(s.err[i])}` : '')).join('<br>');
        tip.hidden = false; const tw = tip.offsetWidth; let left = (ev.clientX - host.getBoundingClientRect().left) + 12;
        if (left + tw > host.clientWidth) left -= tw + 24; tip.style.left = Math.max(0, left) + 'px'; tip.style.top = '8px'; };
      hit.addEventListener('pointermove', move); hit.addEventListener('pointerleave', () => { tip.hidden = true; hair.setAttribute('visibility', 'hidden'); });
    };
    draw(); new ResizeObserver(draw).observe(host);
    matchMedia('(prefers-color-scheme: dark)').addEventListener('change', draw);
    new MutationObserver(draw).observe(document.documentElement, {attributes: true, attributeFilter: ['data-theme']});
  }

  // Points with 2-sigma bars against a reference value
  function forest(host, cfg) {
    const draw = () => {
      host.innerHTML = '';
      const W = Math.max(host.clientWidth, 280), row = 34, m = {l: Math.min(250, W * 0.42), r: 16, t: 8, b: 34};
      const H = m.t + m.b + row * cfg.rows.length;
      const svg = el('svg', {viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': cfg.title || ''}, host);
      const lo = cfg.domain[0], hi = cfg.domain[1];
      const X = (x) => m.l + (x - lo) / (hi - lo) * (W - m.l - m.r);
      ticksLin(lo, hi, 5).forEach(t => { el('line', {x1: X(t), x2: X(t), y1: m.t, y2: H - m.b, stroke: css('--rule')}, svg);
        el('text', {x: X(t), y: H - m.b + 16, 'text-anchor': 'middle'}, svg).textContent = t.toFixed(cfg.dp ?? 3); });
      el('line', {x1: X(cfg.ref), x2: X(cfg.ref), y1: m.t, y2: H - m.b, stroke: css('--ink-2'), 'stroke-dasharray': '4 3'}, svg);
      el('text', {x: (m.l + W - m.r) / 2, y: H - 4, 'text-anchor': 'middle', class: 'lbl'}, svg).textContent = cfg.xlabel;
      cfg.rows.forEach((rw, i) => { const y = m.t + row * i + row / 2;
        el('text', {x: m.l - 10, y: y + 4, 'text-anchor': 'end', class: 'lbl'}, svg).textContent = rw.label;
        const c = css(rw.color || '--s1');
        el('line', {x1: X(Math.max(lo, rw.v - 2 * rw.e)), x2: X(Math.min(hi, rw.v + 2 * rw.e)), y1: y, y2: y, stroke: c, 'stroke-width': 2}, svg);
        el('circle', {cx: X(rw.v), cy: y, r: 4.5, fill: c, stroke: css('--panel'), 'stroke-width': 2}, svg);
        const t = el('title', {}, el('rect', {x: m.l, y: y - row / 2, width: W - m.l - m.r, height: row, fill: 'transparent'}, svg));
        t.textContent = `${rw.label}: ${rw.v.toFixed(5)} ± ${rw.e.toFixed(5)} (1σ)`; });
    };
    draw(); new ResizeObserver(draw).observe(host);
  }

  const E = D.edges;
  const q = (id) => document.getElementById(id);
  if (q('ch-photon')) stepChart(q('ch-photon'), {title: 'Photon fluence spectrum', ylog: true, yfloor: 1e-6, xunit: 'MeV',
    xlabel: 'photon energy (MeV)', ylabel: 'fluence per 10 keV / Φ₀', series: [
      {name: 'primary', color: '--s1', edges: E.photon_10keV, y: D.photon.primary.mean, err: D.photon.primary.sem},
      {name: 'scattered', color: '--s2', edges: E.photon_10keV, y: D.photon.scattered_source.mean, err: D.photon.scattered_source.sem},
      {name: 'secondary photons', color: '--s3', edges: E.photon_10keV, y: D.photon.secondary_photons.mean, err: D.photon.secondary_photons.sem}]});
  if (q('ch-photon-ang')) stepChart(q('ch-photon-ang'), {title: 'Photon angular distribution', ylog: true, yfloor: 1e-5,
    xlabel: 'cos θ to beam axis', ylabel: 'fraction of fluence', xticks: 5, series: [
      {name: 'primary', color: '--s1', edges: E.cos_10, y: D.photon_angular.primary.mean},
      {name: 'scattered', color: '--s2', edges: E.cos_10, y: D.photon_angular.scattered.mean}]});
  if (q('ch-initial')) stepChart(q('ch-initial'), {title: 'Initial electron spectrum', ylog: true, yfloor: 1e-5, xunit: 'MeV',
    xlabel: 'initial kinetic energy (MeV)', ylabel: 'electrons per 10 keV per g / Φ₀',
    marks: [{x: 0.9634, label: 'edge 1.1732'}, {x: 1.1181, label: 'edge 1.3325', dy: 14}], series: [
      {name: 'Compton', color: '--s1', edges: E.electron_10keV, y: D.initial.compton.mean, err: D.initial.compton.sem},
      {name: 'photoelectric', color: '--s2', edges: E.electron_10keV, y: D.initial.photoelectric.mean},
      {name: 'pair (e−)', color: '--s3', edges: E.electron_10keV, y: D.initial.pair_electron.mean}]});
  if (q('ch-initial-cos')) stepChart(q('ch-initial-cos'), {title: 'Initial electron direction', xlabel: 'cos θ to beam axis',
    ylabel: 'fraction of electrons', xticks: 5, series: [
      {name: 'parent primary', color: '--s1', edges: E.cos_10, y: D.initial_cos.primary.mean},
      {name: 'parent scattered', color: '--s2', edges: E.cos_10, y: D.initial_cos.scattered.mean}]});
  if (q('ch-electron')) stepChart(q('ch-electron'), {title: 'Electron fluence spectrum', ylog: true, yfloor: 3e-4, xunit: 'MeV',
    xdomain: [0, 1.4], xlabel: 'electron kinetic energy (MeV)', ylabel: 'Φ_E / Φ₀ (MeV⁻¹)',
    marks: [{x: 0.01, label: 'Δ = 10 keV'}], series: [
      {name: 'all electrons', color: '--s1', edges: E.electron_10keV, y: D.electron.all.mean, err: D.electron.all.sem},
      {name: 'delta rays', color: '--s2', edges: E.electron_10keV, y: D.electron.delta.mean},
      {name: 'positrons', color: '--s3', edges: E.electron_10keV, y: D.electron.positron.mean}]});
  if (q('ch-electron-ang')) stepChart(q('ch-electron-ang'), {title: 'Electron angular distribution', xlabel: 'cos θ to beam axis',
    ylabel: 'fraction of fluence', xticks: 5, series: [{name: 'electrons', color: '--s1', edges: E.cos_10, y: D.electron_angular.mean}]});
  if (q('ch-pdd') && D.pdd) { const z = D.pdd.depth_cm, ed = z.map((v, i) => +(i * 0.1).toFixed(2)).concat([10]);
    stepChart(q('ch-pdd'), {title: 'Central-axis depth dose', xunit: 'cm', xlabel: 'depth (cm)', ylabel: 'percent of D_max', ymin: 0, ymax: 100,
      marks: [{x: 5.0, label: '5 cm'}], series: [{name: 'PDD', color: '--s1', edges: ed, y: D.pdd.profile_percent.mean, err: D.pdd.profile_percent.sem}]}); }
  if (q('ch-fano')) { const order = ['--s1', '--s3', '--s4', '--s2'];
    forest(q('ch-fano'), {title: 'Fano cavity test', ref: 1, domain: [0.994, 1.022], dp: 3, xlabel: '(D/Ψ)/(μ_tr/ρ), mean ± 2σ',
      rows: D.fano.map((f, i) => ({label: f.label_short, v: f.ratio_mean, e: f.ratio_sem_batch_spread, color: f.color}))}); }
})();
"""


def chip(kind, label):
    icon = {"ok": '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M2 6.5l2.5 2.5L10 3.5" fill="none" stroke="currentColor" stroke-width="2"/></svg>',
            "warn": '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M6 2v5M6 9.5v.5" fill="none" stroke="currentColor" stroke-width="2"/></svg>',
            "bad": '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M3 3l6 6M9 3l-6 6" fill="none" stroke="currentColor" stroke-width="2"/></svg>'}[kind]
    return f'<span class="chip {kind}">{icon}{esc(label)}</span>'


def render(r, m, n, data_json, text, fano, v, pm, sci, rel, sig):
    meta = m["_meta"]
    T = text

    def val(key, run=m):
        return run[key]["mean"]

    def row(label, key, unit="", run=m, fmt=None):
        mean, sem = run[key]["mean"], run[key]["sem"]
        shown = fmt(mean, sem) if fmt else pm(mean, sem)
        return f"<tr><td>{label}</td><td class='num'>{shown}</td><td>{unit}</td></tr>"

    # Fano rows with short labels and fixed colors
    fano_rows = []
    short = {"standard_opt4": ("Production (option4 transport)", "--s1"),
             "standard_opt4 | /process/msc/RangeFactor 0.04": ("msc range factor 0.04", "--s3"),
             "standard_opt4 | /process/eLoss/StepFunction 0.05 1 um": ("step function 0.05, 1 µm", "--s4"),
             "standard_opt0": ("Negative control (opt0)", "--s2")}
    import json as _json
    data = _json.loads(data_json)
    for f in data["fano"]:
        lab, col = short.get(f["label"], (f["label"], "--s1"))
        f["label_short"], f["color"] = lab, col
    order = list(short)
    data["fano"].sort(key=lambda f: order.index(f["label"]) if f["label"] in order else 99)
    data_json = _json.dumps(data, separators=(",", ":"))

    # Targets
    tg = r["targets"]
    def achieved(x):
        return "n/a" if x["achieved"] is None else f"{100 * x['achieved']:.3f}%"

    t_rows = "".join(
        f"<tr><td>{esc(k)}</td><td class='num'>{100 * x['target']:.1f}%</td><td class='num'>{achieved(x)}</td>"
        f"<td>{chip('ok', 'met') if x['met'] else chip('bad', 'not met')}</td></tr>" for k, x in tg.items())
    all_met = all(x["met"] for x in tg.values())

    # Checks table from text yaml (status computed there by the caller's numbers)
    checks = "".join(
        f"<tr><td>{esc(c['name'])}</td><td>{c['result']}</td><td>{chip(c['status'], c['status_label'])}</td></tr>"
        for c in T["checks"])

    fano_tbl = "".join(
        f"<tr><td>{esc(f['label_short'])}</td><td class='num'>{f['batches']} × {f['events_total'] // f['batches']:,}</td>"
        f"<td class='num'>{f['ratio_mean']:.5f} ± {f['ratio_sem_batch_spread']:.5f}</td>"
        f"<td class='num'>{100 * (f['ratio_mean'] - 1):+.3f}%</td><td class='num'>{f['deviation_sigma']:+.1f}σ</td></tr>"
        for f in data["fano"])

    sw = "".join(
        f"<tr><td>{esc(T['bases'][b])}</td><td class='num'>{pm(m[f'L5_s_w_air_{b}']['mean'], m[f'L5_s_w_air_{b}']['sem'])}</td></tr>"
        for b in ("icru37", "icru90", "geant4", "pegs4"))

    l2 = m["L2_fraction_all"]["mean"], m["L2_fraction_primary"]["mean"], m["L2_fraction_scattered"]["mean"]
    e2 = m["L2_etr_fraction_all"]["mean"], m["L2_etr_fraction_primary"]["mean"], m["L2_etr_fraction_scattered"]["mean"]
    procs = ["Compton (incoherent)", "Photoelectric", "Pair production", "Rayleigh (coherent)"]
    l2_rows = "".join(
        f"<tr><td>{p}</td>" + "".join(f"<td class='num'>{100 * a[i]:.3f}%</td>" for a in l2) +
        "".join(f"<td class='num'>{100 * a[i]:.3f}%</td>" for a in e2) + "</tr>" for i, p in enumerate(procs))

    notes = "".join(
        f"<div class='note{' warn' if nt.get('warn') else ''}'><h3>{esc(nt['title'])}</h3><p>{nt['body']}</p></div>"
        for nt in T["findings"])
    decide = "".join(f"<li><b>{esc(d['title'])}.</b> {d['body']}</li>" for d in T["decisions"])

    pdd = r.get("pdd")
    pdd_txt = (f"PDD(5) = <span class='num'>{pdd['PDD5']['mean']:.2f} ± {pdd['PDD5']['sem']:.2f}</span>%, "
               f"d<sub>max</sub> = <span class='num'>{pdd['d_max_cm']['mean']:.3f} ± {pdd['d_max_cm']['sem']:.3f}</span> cm"
               ) if pdd else "Depth scan not yet run"

    return f"""<title>Co-60 Reference Point M1</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>{CSS}</style>
<div class="wrap">
<aside class="rail">
  <div class="eyebrow">Photon-to-dose chain</div>
  <nav aria-label="Sections">
    <a href="#chain"><span class="k">R0</span><span>Overview</span></a>
    <a href="#layer1"><span class="k">L1</span><span>Photons arriving</span></a>
    <a href="#layer2"><span class="k">L2</span><span>Photon interactions</span></a>
    <a href="#layer3"><span class="k">L3</span><span>Electrons set in motion</span></a>
    <a href="#layer4"><span class="k">L4</span><span>Electrons slowing down</span></a>
    <a href="#layer5"><span class="k">L5</span><span>Energy deposited</span></a>
    <div class="sep"></div>
    <a href="#checks"><span class="k">✓</span><span>Checks and targets</span></a>
    <a href="#fano"><span class="k">F</span><span>Fano diagnostics</span></a>
    <a href="#findings"><span class="k">!</span><span>Findings</span></a>
    <a href="#decide"><span class="k">?</span><span>Decisions for the gate</span></a>
  </nav>
</aside>
<main>
<header class="top" id="chain">
  <div class="eyebrow">Milestone M1 · Monte Carlo at R0 · {esc(T['date'])}</div>
  <h1>Co-60 reference point: the Monte Carlo reference, layer by layer</h1>
  <p class="lede">{T['intro']}</p>
  <div class="meta">
    <span>{esc(g4_label(meta['geant4_version']))}</span>
    <span>G4EmStandardPhysics_option4</span>
    <span>water, 5 cm depth, 10 × 10 cm², SSD 100 cm</span>
    <span class="num">{n['hist']} histories · {n['batches']} batches</span>
    <span class="num">commit {esc(n['commit'])}</span>
  </div>
  <div class="chips">{chip('ok' if all_met else 'bad', 'All statistical targets met' if all_met else 'Statistical targets not all met')}{''.join(chip(c['status'], c['chip']) for c in T['checks'] if c.get('chip'))}</div>
  <div class="chain" aria-label="The five layers">
    <a class="link" href="#layer1"><span class="k">L1</span><span class="t">Photons arriving</span>
      <span class="big">{100 * val('L1_scattered_fraction_fluence'):.1f}%</span><span class="sub">of photon fluence is scattered; mean energy {val('L1_mean_energy_all_MeV'):.3f} MeV</span></a>
    <a class="link" href="#layer2"><span class="k">L2</span><span class="t">Interactions</span>
      <span class="big">{100 * m['L2_fraction_all']['mean'][0]:.2f}%</span><span class="sub">of interactions are Compton; {100 * m['L2_etr_fraction_all']['mean'][0]:.2f}% of energy transferred</span></a>
    <a class="link" href="#layer3"><span class="k">L3</span><span class="t">Electrons set in motion</span>
      <span class="big">{1000 * val('L3_mean_T_all_MeV'):.0f} keV</span><span class="sub">mean initial kinetic energy</span></a>
    <a class="link" href="#layer4"><span class="k">L4</span><span class="t">Electrons slowing down</span>
      <span class="big">{100 * val('L4_delta_fraction_above_10keV'):.1f}%</span><span class="sub">of electron fluence above 10 keV is delta rays</span></a>
    <a class="link" href="#layer5"><span class="k">L5</span><span class="t">Energy deposited</span>
      <span class="big">{n['D']}</span><span class="sub">D/Φ₀ ± {n['D_rel']} · β = {n['beta']}</span></a>
  </div>
</header>

<section id="layer1">
  <div class="sec-head"><span class="k">Layer 1</span><h2>Photons arriving at the point</h2></div>
  <p class="lede">{T['layer1']}</p>
  <div class="fig"><div class="legend"><span><i style="background:var(--s1)"></i>primary (no interaction since the source)</span><span><i style="background:var(--s2)"></i>scattered source photons</span><span><i style="background:var(--s3)"></i>bremsstrahlung, fluorescence, annihilation</span></div>
    <div class="chart" id="ch-photon"></div><div class="cap">Track-length estimate in the scoring cylinder, 10 keV bins, per unit Φ₀. Hover for values.</div></div>
  <div class="grid2">
    <div class="fig"><div class="legend"><span><i style="background:var(--s1)"></i>primary</span><span><i style="background:var(--s2)"></i>scattered</span></div>
      <div class="chart" id="ch-photon-ang"></div><div class="cap">Fraction of photon fluence per cos θ bin (comparison grid).</div></div>
    <div class="tbl"><table><thead><tr><th>Quantity</th><th class="num">Value</th><th>Unit</th></tr></thead><tbody>
      {row('Total photon fluence / Φ₀', 'L1_fluence_total_per_phi0')}
      {row('Primary fluence / Φ₀', 'L1_fluence_primary_per_phi0')}
      {row('Scattered fraction of fluence', 'L1_scattered_fraction_fluence')}
      {row('Scattered fraction of energy fluence', 'L1_scattered_fraction_energy_fluence')}
      {row('Mean energy, all photons', 'L1_mean_energy_all_MeV', 'MeV')}
      {row('Mean energy, scattered photons', 'L1_mean_energy_scattered_MeV', 'MeV')}
      {row('Primary 1.1732 MeV: measured / analytic', 'L1_primary_line1_measured_over_analytic')}
      {row('Primary 1.3325 MeV: measured / analytic', 'L1_primary_line2_measured_over_analytic')}
    </tbody></table></div>
  </div>
</section>

<section id="layer2">
  <div class="sec-head"><span class="k">Layer 2</span><h2>Photon interactions at the point</h2></div>
  <p class="lede">{T['layer2']}</p>
  <div class="tbl"><table><thead><tr><th>Process</th><th class="num">All</th><th class="num">Primary</th><th class="num">Scattered</th>
    <th class="num">E<sub>tr</sub>: all</th><th class="num">E<sub>tr</sub>: primary</th><th class="num">E<sub>tr</sub>: scattered</th></tr></thead>
    <tbody>{l2_rows}</tbody></table></div>
</section>

<section id="layer3">
  <div class="sec-head"><span class="k">Layer 3</span><h2>Electrons set in motion</h2></div>
  <p class="lede">{T['layer3']}</p>
  <div class="fig"><div class="legend"><span><i style="background:var(--s1)"></i>Compton recoil</span><span><i style="background:var(--s2)"></i>photoelectrons</span><span><i style="background:var(--s3)"></i>pair-production electrons</span></div>
    <div class="chart" id="ch-initial"></div><div class="cap">Initial kinetic energy of electrons created in the scoring volume by photon interactions, all parents. Dashed lines: free-electron Compton edges.</div></div>
  <div class="grid2">
    <div class="fig"><div class="legend"><span><i style="background:var(--s1)"></i>parent photon primary</span><span><i style="background:var(--s2)"></i>parent photon scattered</span></div>
      <div class="chart" id="ch-initial-cos"></div><div class="cap">Initial direction of the electrons.</div></div>
    <div class="tbl"><table><thead><tr><th>Quantity</th><th class="num">Value</th><th>Unit</th></tr></thead><tbody>
      {row('Mean initial T, all', 'L3_mean_T_all_MeV', 'MeV')}
      {row('Mean initial T, parent primary', 'L3_mean_T_parent_primary_MeV', 'MeV')}
      {row('Mean initial T, parent scattered', 'L3_mean_T_parent_scattered_MeV', 'MeV')}
      {row('Positrons among particles set in motion', 'L3_positron_fraction')}
      {row('Compton recoils above the free edge, 1.1732 MeV', 'L3_compton_line1_fraction_above_free_edge')}
      {row('Compton recoils above the free edge, 1.3325 MeV', 'L3_compton_line2_fraction_above_free_edge')}
      {row('Highest Compton recoil, 1.3325 MeV line', 'L3_compton_line2_max_T_MeV', 'MeV')}
    </tbody></table></div>
  </div>
</section>

<section id="layer4">
  <div class="sec-head"><span class="k">Layer 4</span><h2>Electrons slowing down</h2></div>
  <p class="lede">{T['layer4']}</p>
  <div class="fig"><div class="legend"><span><i style="background:var(--s1)"></i>all electrons</span><span><i style="background:var(--s2)"></i>delta rays and descendants</span><span><i style="background:var(--s3)"></i>positrons</span></div>
    <div class="chart" id="ch-electron"></div><div class="cap">Differential fluence per unit Φ₀ from the CSDA-weighted track-length estimator (DECISIONS D020); 10 keV bins.</div></div>
  <div class="grid2">
    <div class="fig"><div class="chart" id="ch-electron-ang"></div><div class="cap">Electron fluence by direction (pre-step direction, track-length weighted).</div></div>
    <div class="tbl"><table><thead><tr><th>Quantity</th><th class="num">Value</th><th>Unit</th></tr></thead><tbody>
      {row('Electron fluence above 10 keV / Φ₀', 'L4_electron_fluence_above_10keV_per_phi0')}
      {row('Delta-ray share above 10 keV', 'L4_delta_fraction_above_10keV')}
      {row('Positron / electron fluence', 'L4_positron_to_electron_fluence')}
      {row('Fluence from scattered-photon parents', 'L4_parent_scattered_fraction')}
      {row('Fluence-weighted mean energy above 10 keV', 'L4_mean_energy_above_10keV_MeV', 'MeV')}
      {row('Radiative / collision energy loss', 'L4_radiative_to_collision')}
    </tbody></table></div>
  </div>
</section>

<section id="layer5">
  <div class="sec-head"><span class="k">Layer 5</span><h2>Energy deposited</h2></div>
  <p class="lede">{T['layer5']}</p>
  <div class="stats">
    <div class="stat"><span class="l">D/Φ₀</span><span class="v">{n['D']}</span><span class="l">± {n['D_rel']} (1σ)</span></div>
    <div class="stat"><span class="l">K/Φ₀ (scored)</span><span class="v">{n['K']}</span></div>
    <div class="stat"><span class="l">K<sub>col</sub>/Φ₀ (scored)</span><span class="v">{n['Kcol']}</span></div>
    <div class="stat"><span class="l">β = D/K<sub>col</sub></span><span class="v">{n['beta']}</span><span class="l">± {n['beta_rel']}</span></div>
    <div class="stat"><span class="l">g = 1 − K<sub>col</sub>/K</span><span class="v">{n['g']}</span></div>
  </div>
  <div class="grid2">
    <div class="tbl"><table><thead><tr><th>s<sub>w,air</sub>, Spencer–Attix, Δ = 10 keV, stopping-power basis</th><th class="num">Value</th></tr></thead><tbody>{sw}</tbody></table></div>
    <div class="tbl"><table><thead><tr><th>Quantity</th><th class="num">Value</th><th>Unit</th></tr></thead><tbody>
      {row('Dose from scattered-photon parents', 'L5_dose_fraction_parent_scattered')}
      {row('… of which scattered source photons', 'L5_dose_fraction_parent_scattered_source')}
      {row('… of which secondary photons', 'L5_dose_fraction_parent_secondary_photon')}
      {row('Spencer–Attix dose / scored dose (Geant4 basis)', 'L5_D_SpencerAttix_over_D_geant4')}
      {row('Continuous-loss dose / scored dose', 'L5_D_continuous_over_D')}
      {row('Track-end share of water term', 'L5_track_end_share_water_geant4')}
    </tbody></table></div>
  </div>
  <div class="fig"><div class="chart" id="ch-pdd"></div><div class="cap">{pdd_txt}. 1 mm slabs within r &lt; 1 cm; D<sub>max</sub> from a parabola through the slabs around the peak. {T['pdd_note']}</div></div>
</section>

<section id="checks">
  <div class="sec-head"><span class="k">Checks</span><h2>Is the reference trustworthy?</h2></div>
  <p class="lede">{T['checks_intro']}</p>
  <div class="tbl"><table><thead><tr><th>Statistical target (STUDY_PLAN 4.1)</th><th class="num">Target</th><th class="num">Achieved</th><th></th></tr></thead><tbody>{t_rows}</tbody></table></div>
  <div class="tbl"><table><thead><tr><th>Check</th><th>Result</th><th></th></tr></thead><tbody>{checks}</tbody></table></div>
</section>

<section id="fano">
  <div class="sec-head"><span class="k">Fano</span><h2>Fano cavity diagnostics</h2></div>
  <p class="lede">{T['fano']}</p>
  <div class="fig"><div class="chart" id="ch-fano"></div><div class="cap">Ratio of cavity dose to energy fluence over μ<sub>tr</sub>/ρ; 1 is the Fano expectation. Bars span ±2σ from the spread of independent batches.</div></div>
  <div class="tbl"><table><thead><tr><th>Configuration</th><th class="num">Batches × events</th><th class="num">Ratio (1σ)</th><th class="num">Deviation</th><th class="num">z</th></tr></thead><tbody>{fano_tbl}</tbody></table></div>
</section>

<section id="findings">
  <div class="sec-head"><span class="k">Findings</span><h2>What to look at</h2></div>
  <div class="notes">{notes}</div>
</section>

<section id="decide">
  <div class="sec-head"><span class="k">Gate</span><h2>Decisions for the M1 gate</h2></div>
  <ol class="decide">{decide}</ol>
</section>

<footer>
  <span>{T['footer']}</span>
  <span class="mono">Branch claude/adoring-edison-tna32r · commit {esc(n['commit'])} · results analysis/results/M1_results.json</span>
</footer>
</main>
</div>
<script type="application/json" id="m1-data">{data_json}</script>
<script>{JS}</script>
"""
