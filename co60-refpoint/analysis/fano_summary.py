"""Pool Fano cavity batches across runs and plot them.

Batches from runs with the same physics constructor are pooled. The pooled
uncertainty is the standard error of the batch mean (all batches have the same
number of events), shown next to the example's own event-by-event estimate.

Usage:
  python analysis/fano_summary.py M0-fano-opt4 M0-fano-opt4-ext M0-fano-opt0-control
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "mc" / "runs"
OUT = ROOT / "analysis"

# Categorical slots 1 and 2 of the reference palette (validated, light mode).
COLORS = {"standard_opt4": "#2a78d6", "standard_opt0": "#eb6834"}
LABELS = {"standard_opt4": "standard_opt4 (production e- transport)",
          "standard_opt0": "standard_opt0 (negative control)"}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def load(run_ids):
    pooled = {}
    for run_id in run_ids:
        m = yaml.safe_load(open(RUNS / run_id / "manifest.yaml"))
        phys = m["application"]["physics_constructor"]
        entry = pooled.setdefault(phys, {"runs": [], "events_per_batch": set(), "ratio": [], "err": []})
        entry["runs"].append(run_id)
        entry["events_per_batch"].add(m["statistics"]["events_per_batch"])
        for b in m["batches"]:
            if b["exit_code"] == 0 and "ratio" in b:
                entry["ratio"].append(b["ratio"])
                entry["err"].append(b["ratio_err"])
    return pooled


def summarize(pooled):
    out = {}
    for phys, e in pooled.items():
        if len(e["events_per_batch"]) != 1:
            raise ValueError(f"{phys}: batches of different size cannot be pooled by simple mean")
        r, s = np.array(e["ratio"]), np.array(e["err"])
        n = len(r)
        mean = r.mean()
        sem = r.std(ddof=1) / np.sqrt(n)
        out[phys] = {
            "runs": e["runs"],
            "batches": n,
            "events_total": n * e["events_per_batch"].pop(),
            "ratio_mean": round(float(mean), 6),
            "ratio_sem_batch_spread": round(float(sem), 6),
            "ratio_sem_example_estimate": round(float(np.sqrt(np.sum(s**2)) / n), 6),
            "deviation_percent": round(float(100 * (mean - 1)), 4),
            "deviation_sigma": round(float((mean - 1) / sem), 2),
            "interval_95_percent": [round(float(mean - 1.96 * sem), 5), round(float(mean + 1.96 * sem), 5)],
        }
    return out


def plot(pooled, summary, path):
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.axhline(1.0, color=INK_2, lw=1, ls="--", zorder=1)
    for phys, e in pooled.items():
        r, s = np.array(e["ratio"]), np.array(e["err"])
        x = np.arange(1, len(r) + 1)
        c = COLORS.get(phys, INK)
        ax.errorbar(x, r, yerr=s, fmt="o", ms=4, color=c, ecolor=c, elinewidth=1, capsize=0, zorder=3,
                    label=LABELS.get(phys, phys))
        sm = summary[phys]
        m, se = sm["ratio_mean"], sm["ratio_sem_batch_spread"]
        ax.axhspan(m - se, m + se, color=c, alpha=0.15, lw=0, zorder=2)
        ax.annotate(f"mean {m:.4f} ± {se:.4f}", xy=(len(r) + 0.6, m), va="center", ha="left",
                    fontsize=8, color=INK)
    n_max = max(len(e["ratio"]) for e in pooled.values())
    ax.set_xlim(0, n_max * 1.3 + 2)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.annotate("Fano expectation", xy=(0.3, 1.0), va="bottom", ha="left", fontsize=7, color=INK_2)
    ax.set_xlabel("batch", color=INK_2, fontsize=9)
    ax.set_ylabel("(D / Ψ) / (μ_tr / ρ)", color=INK_2, fontsize=9)
    ax.set_title("Fano cavity test, 1.25 MeV, G4_WATER wall and 1 mg/cm³ cavity (Geant4 11.4.3)",
                 color=INK, fontsize=9, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2,
              labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)


def main():
    run_ids = sys.argv[1:]
    pooled = load(run_ids)
    summary = summarize(pooled)
    (OUT / "figures").mkdir(exist_ok=True)
    with open(OUT / "M0_fano_summary.yaml", "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)
    plot(pooled, summary, OUT / "figures" / "M0_fano.png")
    print(yaml.safe_dump(summary, sort_keys=False))


if __name__ == "__main__":
    main()
