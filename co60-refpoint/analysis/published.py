#!/usr/bin/env python3
"""Comparison of the R0 Monte Carlo with published values (STUDY_PLAN Section 4.3).

    python analysis/published.py     # writes analysis/results/published_comparison.json

Every external value comes from refs/refs.yaml and carries its verified_by_humza flag into the output; an
unverified value is reported as such and never as a confirmed agreement (STUDY_PLAN Section 10.1). Values that
refs.yaml does not have yet (NIST tables, a published PDD(5)) are listed as missing.
"""
import json
import math
from pathlib import Path

import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RES = ROOT / "analysis" / "results"


def refs():
    return {e["key"]: e for e in yaml.safe_load(open(ROOT / "refs" / "refs.yaml"))}


def main():
    R = refs()
    ref = json.load(open(RES / "reference_R0.json"))
    m1 = json.load(open(RES / "M1_results.json"))
    m2 = json.load(open(RES / "M2_results.json")) if (RES / "M2_results.json").exists() else {}
    lay = m2.get("runs", {}).get("M2-R0-option4") or m1["runs"][m1["pooling"]["reference_run"]]
    rows = []

    def row(quantity, mc, mc_sem, key, note=""):
        e = R.get(key, {})
        v = e.get("value")
        r = {"quantity": quantity, "mc": mc, "mc_sem": mc_sem, "ref_key": key, "ref_value": v,
             "ref_source": e.get("source"), "verified_by_humza": bool(e.get("verified_by_humza")), "note": note}
        if isinstance(v, (int, float)) and mc is not None:
            r["mc_over_ref"] = mc / v
        rows.append(r)

    o = ref["observables"]
    row("s_w,air (Delta = 10 keV), ICRU 37 basis", lay["L5_s_w_air_icru37"]["mean"], lay["L5_s_w_air_icru37"]["sem"],
        "s_w_air_co60_trs398", "TRS-398 tabulates s_w,air for Co-60 on the ICRU 37 basis; model A has no head scatter")
    row("beta = D / K_col", o["L5_beta"]["mc"], o["L5_beta"]["sem"], "beta_co60_beyond_dmax",
        "plan: slightly above 1, of order 0.5% beyond d_max; no numeric value in refs.yaml")
    row("PDD(5), SSD 100 cm, 10 x 10 cm2", o["L5_PDD5_percent"]["mc"], o["L5_PDD5_percent"]["sem"],
        "pdd5_co60_ssd100_10x10", "model A differs from unit data by design (no head scatter); value to be entered")
    # Attenuation: Geant4's own tracking mu (TestEm13) against the XCOM elemental data (EGSnrc-distributed)
    tr = ROOT / "mc" / "runs" / "M1-check-transmission" / "results.yaml"
    if tr.exists():
        t = yaml.safe_load(open(tr))
        from analysis.xcom import water_mu_rho
        from analysis.m1 import TABLES
        for line, e_key in (("line1", "co60_line_1_energy"), ("line2", "co60_line_2_energy")):
            mu = t["lines"][line]["mu_measured_cm-1"]
            mu_x = float(water_mu_rho(np.array([R[e_key]["value"]]), TABLES)["total"][0])
            rows.append({"quantity": f"mu/rho of water at {R[e_key]['value']} MeV (Geant4 tracking, TestEm13)",
                         "mc": mu, "mc_sem": t["lines"][line].get("mu_measured_sem_cm-1"),
                         "ref_key": "xcom_elemental_cross_sections_egsnrc", "ref_value": mu_x,
                         "ref_source": R["xcom_elemental_cross_sections_egsnrc"]["source"],
                         "verified_by_humza": False, "mc_over_ref": mu / mu_x,
                         "note": "NIST table value (nist_mu_rho_water) not yet entered; XCOM elemental data used"})
    missing = [k for k in ("nist_mu_rho_water", "nist_mu_en_rho_water", "pdd5_co60_ssd100_10x10",
                           "beta_co60_beyond_dmax") if R.get(k, {}).get("value") is None]
    out = {"rows": rows, "missing_reference_values": missing,
           "note": "All external values are unverified until Humza checks them (STUDY_PLAN 10.1)."}
    json.dump(out, open(RES / "published_comparison.json", "w"), indent=1, default=float)
    for r in rows:
        mo = r.get("mc_over_ref")
        print(f"{r['quantity'][:60]:60s} MC {r['mc']:.6g}  ref {r['ref_value']}  "
              f"{'ratio ' + format(mo, '.4f') if mo else ''}  {'verified' if r['verified_by_humza'] else 'UNVERIFIED'}")
    print("missing:", missing)


if __name__ == "__main__":
    main()
