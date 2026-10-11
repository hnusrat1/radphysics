"""Small server-side SVG charts for the study pages: binned spectra from several codes, with a ratio panel.

Colors are CSS variables (--s1 ... --s4, --ink, --muted, --rule) so the chart follows the page theme.
Series identity is carried by a legend and by line style as well as color.
"""
import math

import numpy as np


def _nice_ticks(lo, hi, n=5):
    span = hi - lo
    if span <= 0:
        return [lo]
    raw = span / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min((s * mag for s in (1, 2, 2.5, 5, 10) if s * mag >= raw), default=raw)
    start = math.ceil(lo / step) * step
    return [round(start + i * step, 10) for i in range(int((hi - start) / step + 1e-9) + 1)]


def spectra_with_ratio(series, edges, xlabel, ylabel, title, ratio_label="ratio to first", width=720, height=380,
                       ratio_range=(0.9, 1.1)):
    """series: [(label, values per bin (normalized by caller), sem per bin or None, css color var, dash)]."""
    edges = np.asarray(edges, float)
    ml, mr, mt = 64, 16, 28
    h_top, h_gap, h_rat, mb = 0.60 * (height - 70), 26, 0.40 * (height - 70) - 26, 42
    pw = width - ml - mr
    x0, x1 = edges[0], edges[-1]
    ymax = max(float(np.nanmax(s[1])) for s in series) * 1.08
    X = lambda x: ml + (x - x0) / (x1 - x0) * pw                         # noqa: E731
    Y = lambda y: mt + h_top - y / ymax * h_top                          # noqa: E731
    r0 = mt + h_top + h_gap
    rlo, rhi = ratio_range
    R = lambda r: r0 + h_rat - (min(max(r, rlo), rhi) - rlo) / (rhi - rlo) * h_rat   # noqa: E731
    out = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{title}" class="chart">',
           f'<text x="{ml}" y="16" class="ct">{title}</text>']
    for t in _nice_ticks(0, ymax, 4):
        out.append(f'<line x1="{ml}" x2="{ml + pw}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" class="grid"/>'
                   f'<text x="{ml - 6}" y="{Y(t) + 4:.1f}" class="tick" text-anchor="end">{t:g}</text>')
    for t in _nice_ticks(rlo, rhi, 4):
        out.append(f'<line x1="{ml}" x2="{ml + pw}" y1="{R(t):.1f}" y2="{R(t):.1f}" class="grid"/>'
                   f'<text x="{ml - 6}" y="{R(t) + 4:.1f}" class="tick" text-anchor="end">{t:g}</text>')
    for t in _nice_ticks(x0, x1, 7):
        out.append(f'<text x="{X(t):.1f}" y="{height - mb + 16}" class="tick" text-anchor="middle">{t:g}</text>')
    out.append(f'<line x1="{ml}" x2="{ml + pw}" y1="{R(1.0):.1f}" y2="{R(1.0):.1f}" class="axis"/>')
    base = np.asarray(series[0][1], float)
    for label, vals, sem, color, dash in series:
        v = np.asarray(vals, float)
        pts = []
        for i in range(len(v)):
            pts += [f"{X(edges[i]):.1f},{Y(v[i]):.1f}", f"{X(edges[i + 1]):.1f},{Y(v[i]):.1f}"]
        out.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="var({color})" stroke-width="2"'
                   f'{f" stroke-dasharray={chr(34)}{dash}{chr(34)}" if dash else ""}/>')
        if v is not base:
            # ratio only where the reference bin holds at least 0.5% of its peak; gaps break the line
            keep = (base > 0.005 * np.nanmax(base)) & (v > 0)
            seg = []
            for i in list(range(len(v))) + [None]:
                if i is not None and keep[i]:
                    r = v[i] / base[i]
                    seg += [f"{X(edges[i]):.1f},{R(r):.1f}", f"{X(edges[i + 1]):.1f},{R(r):.1f}"]
                elif seg:
                    out.append(f'<polyline points="{" ".join(seg)}" fill="none" stroke="var({color})" '
                               f'stroke-width="1.6"{f" stroke-dasharray={chr(34)}{dash}{chr(34)}" if dash else ""}/>')
                    seg = []
    # legend
    lx = ml + pw - 8
    for k, (label, _, _, color, dash) in enumerate(series):
        y = mt + 10 + 18 * k
        out.append(f'<line x1="{lx - 150}" x2="{lx - 126}" y1="{y}" y2="{y}" stroke="var({color})" stroke-width="2.5"'
                   f'{f" stroke-dasharray={chr(34)}{dash}{chr(34)}" if dash else ""}/>'
                   f'<text x="{lx - 120}" y="{y + 4}" class="lg">{label}</text>')
    out.append(f'<text x="{ml + pw / 2:.0f}" y="{height - 8}" class="lab" text-anchor="middle">{xlabel}</text>'
               f'<text x="14" y="{mt + h_top / 2:.0f}" class="lab" transform="rotate(-90 14 {mt + h_top / 2:.0f})"'
               f' text-anchor="middle">{ylabel}</text>'
               f'<text x="14" y="{r0 + h_rat / 2:.0f}" class="lab" transform="rotate(-90 14 {r0 + h_rat / 2:.0f})"'
               f' text-anchor="middle">{ratio_label}</text></svg>')
    return "\n".join(out)


CHART_CSS = """
.chart { width: 100%; height: auto; background: var(--surface); border: 1px solid var(--rule); border-radius: 8px; }
.chart .grid { stroke: var(--rule); stroke-width: 1; }
.chart .axis { stroke: var(--muted); stroke-width: 1; stroke-dasharray: 3 3; }
.chart text { fill: var(--muted); font-family: var(--f-ui); }
.chart .tick { font-size: 11px; font-variant-numeric: tabular-nums; }
.chart .lab { font-size: 12px; }
.chart .lg { font-size: 12px; fill: var(--ink); }
.chart .ct { font-size: 13px; fill: var(--ink); font-weight: 600; }
"""
