#!/usr/bin/env python3
"""Measured cost of a run from ai/raw/<tag>/, and the projection to the R0 matrix (STUDY_PLAN Section 5.4).

    python -m ai.harness.cost --tag P0-pilot             # measured charge, tokens, upstream providers
    python -m ai.harness.cost --tag P0-pilot --k 5       # plus the R0 projection at k repeats

The charge is OpenRouter's usage.cost in each response (requested with usage: {include: true}); a response
without it is priced from ai/prices.yaml and counted as such. The projection takes the mean charge per
conversation for each slot, condition and task and multiplies it by the R0 conversation count from run.matrix.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from ai.harness.run import RAW, ROOT, matrix


def conversation(d, prices):
    meta = yaml.safe_load(open(d / "meta.yaml"))
    c = {"cost": 0.0, "priced_turns": 0, "input": 0, "output": 0, "reasoning": 0, "providers": set(),
         "outcome": meta.get("outcome") or "unknown"}
    for f in sorted((d / "turns").glob("*_response.json")):
        r = json.load(open(f))
        u = r.get("usage") or {}
        i, o = u.get("prompt_tokens", 0) or 0, u.get("completion_tokens", 0) or 0
        c["input"] += i
        c["output"] += o
        c["reasoning"] += (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0
        if r.get("provider"):
            c["providers"].add(r["provider"])
        if u.get("cost") is not None:
            c["cost"] += float(u["cost"])
        elif prices:
            c["cost"] += (i * prices["input_per_mtok"] + o * prices["output_per_mtok"]) / 1e6
            c["priced_turns"] += 1
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--raw-root", default=str(RAW))
    ap.add_argument("--k", type=int, help="project the measured cost to R0 with k repeats")
    a = ap.parse_args()
    prices = yaml.safe_load(open(ROOT / "ai" / "prices.yaml"))["prices"]
    groups = {}
    for d in sorted(Path(a.raw_root, a.tag).glob("*/*_r*")):
        if not (d / "meta.yaml").exists():          # conversation still running
            continue
        _, cond, task, _ = d.name.split("_")
        slot = d.parent.name
        groups.setdefault((slot, cond, task), []).append(conversation(d, prices.get(slot)))
    total = sum(c["cost"] for g in groups.values() for c in g)
    priced = sum(c["priced_turns"] for g in groups.values() for c in g)
    print(f"{a.tag}: {sum(len(g) for g in groups.values())} conversations, measured charge ${total:.4f}"
          + (f" ({priced} turns without usage.cost, priced from ai/prices.yaml)" if priced else ""))
    print(f"{'slot':20s} {'cond':4s} {'task':11s} {'n':>2s} {'ok':>2s} {'$ total':>9s} {'$ mean':>8s} "
          f"{'in mean':>9s} {'out mean':>9s} {'reason':>8s}  upstream")
    for (slot, cond, task), g in sorted(groups.items()):
        ok = sum(c["outcome"].endswith("ok") for c in g)
        print(f"{slot:20s} {cond:4s} {task:11s} {len(g):2d} {ok:2d} {sum(c['cost'] for c in g):9.4f} "
              f"{np.mean([c['cost'] for c in g]):8.4f} {np.mean([c['input'] for c in g]):9.0f} "
              f"{np.mean([c['output'] for c in g]):9.0f} {np.mean([c['reasoning'] for c in g]):8.0f}  "
              f"{', '.join(sorted(set().union(*(c['providers'] for c in g))))}")
    if a.k:
        models = yaml.safe_load(open(ROOT / "ai" / "models.yaml"))["slots"]
        proj, missing = {}, set()
        for s, c, t, _ in matrix(models, ["A0", "A1", "A2"], ["observables", "events"], a.k, "R0"):
            g = groups.get((s["slot"], c, t))
            if g:
                proj[s["slot"]] = proj.get(s["slot"], 0.0) + float(np.mean([x["cost"] for x in g]))
            else:
                missing.add((s["slot"], c, t))
        for slot, v in proj.items():
            print(f"R0 projection {slot:20s} ${v:8.2f}")
        print(f"R0 projection total (k={a.k}): ${sum(proj.values()):.2f}"
              + (f"; no pilot data for {len(missing)} slot/condition/task groups" if missing else ""))


if __name__ == "__main__":
    main()
