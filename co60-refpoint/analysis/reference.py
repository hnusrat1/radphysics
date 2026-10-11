#!/usr/bin/env python3
"""The Monte Carlo reference the AI answers are scored against (STUDY_PLAN Section 6; DECISIONS D025-D028).

    python analysis/reference.py            # writes analysis/results/reference_R0.json

For every quantity in ai/observables.yaml: the reference value and its standard error, the tolerance class, the
resolution floor (the spread of the independent Monte Carlo codes), and for distributions the half-sample
distances that set the threshold. Built from M2 (option4 at range factor 0.04, boundary-free dose; Livermore,
Penelope; EGSnrc in B0). Until M2 is complete it falls back to M1 for the missing parts and says so in
`sources`; a reference with any M1 part is marked provisional and is not used for M4 scoring.

This file is never read by the AI harness (STUDY_PLAN Section 10.2).
"""
import json
import math
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "analysis" / "results"
OBS = yaml.safe_load(open(ROOT / "ai" / "observables.yaml"))["observables"]
BINS = yaml.safe_load(open(ROOT / "config" / "bins.yaml"))["comparison"]
CATS = {"compton": 0, "photoelectric": 1, "pair": 2, "rayleigh": 3}   # analysis/m1.py PROCS order


def get(res, key):
    for part in key.split("/"):
        res = res[part]
    return res


def mean_sem(x):
    return x["mean"], x.get("sem_conservative", x.get("sem"))


def build():
    m2 = json.load(open(RES / "M2_results.json")) if (RES / "M2_results.json").exists() else {}
    m1 = json.load(open(RES / "M1_results.json"))
    sources = {}
    if "M2-R0-option4" in m2.get("runs", {}):
        lay = m2["runs"]["M2-R0-option4"]
        sources["layers"] = "M2-R0-option4"
    else:
        lay = m1["runs"][m1["pooling"]["reference_run"]]
        sources["layers"] = "M1 (provisional): " + m1["pooling"]["reference_run"]
    lists = {L: m2["runs"][f"M2-R0-{L}"] for L in ("livermore", "penelope") if f"M2-R0-{L}" in m2.get("runs", {})}
    sources["lists"] = sorted(lists) or "none yet"
    half = m2.get("half_sample")
    sources["half_sample"] = "M2-R0-option4" if half else "none yet"
    ref = {"sources": sources, "provisional": any("M1" in str(v) for v in sources.values()), "observables": {}}

    # Dose and beta: boundary-free (D025/D027/D028)
    dbf = m2.get("dose_bf", {}).get("option4")
    if dbf:
        d_ref = dbf["D_over_phi0_Gy_cm2"]
        beta_ref = {"mean": dbf["beta"]["ratio"], "sem": dbf["beta"]["sem"]}
        sources["dose"] = "M2-R0-option4-bf"
    else:
        r0 = m1["diagnostics"]["boundary"]["R0_dose_reference"]
        d_ref, beta_ref = r0["D_over_phi0_boundary_free_Gy_cm2"], r0["beta_boundary_free"]
        sources["dose"] = "M1 (provisional): boundary-free R0 reference (D027)"
    if m2.get("pdd"):
        pdd = m2["pdd"]["PDD5"]
        sources["pdd"] = "M2-R0-option4-pdd"
    else:
        pdd = m1["pdd"]["PDD5"]
        sources["pdd"] = "M1 (provisional): M1-R0-option4-pdd"
    ref["provisional"] = any("M1" in str(v) for v in sources.values())

    for o in OBS:
        e = {"layer": o["layer"], "kind": o["kind"], "tolerance": o["tolerance"], "tags": o["tags"]}
        key = o["mc"]
        if o["id"] == "L5_D_over_phi0_Gy_cm2":
            e["mc"], e["sem"] = d_ref["mean"], d_ref["sem"]
        elif o["id"] == "L5_beta":
            e["mc"], e["sem"] = beta_ref["mean"], beta_ref["sem"]
        elif o["id"] == "L5_PDD5_percent":
            e["mc"], e["sem"] = pdd["mean"], pdd["sem"]
        elif o["kind"] == "scalar":
            e["mc"], e["sem"] = mean_sem(get(lay, key))
            if "mc_alongside" in o:
                e["mc_alongside"] = get(lay, o["mc_alongside"])["mean"]
        elif o["kind"] == "fractions":
            v = get(lay, key)
            e["mc"] = {c: float(v["mean"][i]) for c, i in CATS.items()}
            e["sem"] = {c: float(v["sem"][i]) for c, i in CATS.items()}
        else:
            e["grid"] = o["grid"]
            e["mc"] = list(map(float, get(lay, key)["mean"]))
        # resolution floor from the Geant4 lists (EGSnrc enters via B0 for D and s_w,air)
        if lists:
            if o["kind"] == "scalar" and o["id"] not in ("L5_D_over_phi0_Gy_cm2", "L5_beta", "L5_PDD5_percent"):
                diffs = []
                for L, r in lists.items():
                    m, s = mean_sem(get(r, key))
                    diffs.append(abs(m - e["mc"]) + 2 * math.hypot(s, e["sem"]))
                e["floor_abs"] = max(diffs)
            elif o["kind"] == "fractions":
                e["floor_abs"] = {c: max(abs(get(r, key)["mean"][i] - e["mc"][c]) + 2 * math.hypot(
                    get(r, key)["sem"][i], e["sem"][c]) for r in lists.values()) for c, i in CATS.items()}
            elif o["kind"] in ("spectrum", "distribution"):
                ls = m2.get("list_spread", {})
                e["floor_w1"] = max(ls[L]["distributions"][key]["w1"] for L in ls)
                e["floor_jsd"] = max(ls[L]["distributions"][key]["jsd"] for L in ls)
        if o["kind"] in ("spectrum", "distribution") and half:
            e["half_w1"], e["half_jsd"] = half[key]["w1"], half[key]["jsd"]
        ref["observables"][o["id"]] = e

    # Dose-type floors from the boundary-free dose of each list and from EGSnrc in B0
    floors = {}
    for o_id, f in (("L5_D_over_phi0_Gy_cm2", "D_over_phi0_Gy_cm2"),):
        diffs = []
        for L in ("livermore", "penelope"):
            b = m2.get("dose_bf", {}).get(L)
            if b and dbf:
                diffs.append(abs(b[f]["mean"] - d_ref["mean"]) + 2 * math.hypot(b[f]["sem"], d_ref["sem"]))
        b0 = m2.get("b0", {}).get("D_egsnrc_over_geant4")
        if b0:
            diffs.append((abs(b0["ratio"] - 1) + 2 * b0["sem"]) * d_ref["mean"])
        if diffs:
            floors[o_id] = max(diffs)
    b0s = m2.get("b0", {}).get("s_w_air_egsnrc_over_geant4")
    if b0s:
        e = ref["observables"]["L5_s_w_air"]
        e["floor_abs"] = max(e.get("floor_abs", 0.0), (abs(b0s["ratio"] - 1) + 2 * b0s["sem"]) * e["mc"])
    for k, v in floors.items():
        ref["observables"][k]["floor_abs"] = v
    # Event-sample reference (DECISIONS D014): delta_c frozen here, before any AI event is scored
    ev = m2.get("event_sample") or m1.get("event_sample")
    if ev:
        ref["events"] = {"delta_c": ev["doppler_tolerance"]["delta_c"], "source": "M2" if m2.get("event_sample") else
                         "M1 (provisional)", "mc_strict_failure_rate": ev["strict_failure_rate"]["valid"],
                         "mc_bound_failure_rate": ev["bound_failure_rate"]["valid"]}
    return ref


def main():
    ref = build()
    json.dump(ref, open(RES / "reference_R0.json", "w"), indent=1)
    print(f"wrote analysis/results/reference_R0.json (provisional: {ref['provisional']}; sources {ref['sources']})")


if __name__ == "__main__":
    main()
