#!/usr/bin/env python3
"""M4 results by the method each answer used (A1 and A2 observables).

    python analysis/m4_methods.py        # writes analysis/results/M4_methods.json

Reads the code audit analysis/results/M4_method_audit.json (one entry per A1/A2 observables conversation: method
class, photon field and physics checklist, with quoted evidence from the transcripts) and scores every parsed answer
with analysis/m5.py. Groups the scores by method class, by photon field (simulated, simulated with a major bug,
assumed, no code) and by model.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.m5 import OBS, score_answer  # noqa: E402

RES = ROOT / "analysis" / "results"
SCALARS = ["L1_scattered_fraction_fluence", "L1_mean_photon_energy_MeV", "L2_scattered_share_of_interactions",
           "L3_mean_initial_energy_MeV", "L4_electron_fluence_per_phi0", "L4_delta_fraction",
           "L5_Kcol_over_phi0_Gy_cm2", "L5_D_over_phi0_Gy_cm2", "L5_PDD5_percent"]
DISTS = ["L1_photon_fluence_spectrum_scattered", "L3_initial_spectrum", "L3_initial_direction",
         "L4_electron_fluence_spectrum", "L4_angular"]
PHOTO = ("L2_interaction_fractions", "photoelectric")


def per_conversation(entry, ref):
    d = ROOT / entry["conversation"]
    if not (d / "parsed.json").exists():
        return None
    ans = json.load(open(d / "parsed.json"))
    s = score_answer(ans, ref)
    out = {"scalars": {k: s[k].get("rel_diff") for k in SCALARS},
           "w1": {k: s[k].get("w1") for k in DISTS},
           "photoelectric_fraction": (ans.get(PHOTO[0]) or {}).get(PHOTO[1]) if isinstance(ans.get(PHOTO[0]), dict) else None,
           "passes_by_layer": {}}
    for oid, v in s.items():
        L = OBS[oid]["layer"]
        p = out["passes_by_layer"].setdefault(L, [0, 0])
        p[0] += v["verdict"] == "pass"
        p[1] += 1
    return out


def summarize(rows, ref):
    if not rows:
        return {"n": 0}
    out = {"n": len(rows), "median_abs_rel_diff": {}, "median_w1": {}, "pass_share_by_layer": {}}
    for k in SCALARS:
        v = [abs(r["scalars"][k]) for r in rows if r["scalars"][k] is not None and np.isfinite(r["scalars"][k])]
        out["median_abs_rel_diff"][k] = float(np.median(v)) if v else None
    for k in DISTS:
        v = [r["w1"][k] for r in rows if r["w1"][k] is not None]
        out["median_w1"][k] = float(np.median(v)) if v else None
    pe = [r["photoelectric_fraction"] for r in rows if isinstance(r["photoelectric_fraction"], (int, float))]
    out["median_photoelectric_fraction"] = float(np.median(pe)) if pe else None
    for L in range(1, 6):
        p = sum(r["passes_by_layer"].get(L, [0, 0])[0] for r in rows)
        n = sum(r["passes_by_layer"].get(L, [0, 0])[1] for r in rows)
        out["pass_share_by_layer"][L] = p / n if n else None
    return out


def main():
    ref = json.load(open(RES / "reference_R0.json"))
    audit = json.load(open(RES / "M4_method_audit.json"))
    rows, by_class, by_model_class, by_field = [], {}, {}, {}
    for e in audit:
        sc = per_conversation(e, ref)
        slot = e["conversation"].split("/")[3]
        cond = e["conversation"].rsplit("/", 1)[1].split("_")[1]
        rows.append({"conversation": e["conversation"], "slot": slot, "condition": cond,
                     "method_class": e["method_class"], "photon_field": e["photon_field"],
                     "answer_from_code": e.get("answer_from_code"), "valid_answer": sc is not None,
                     "scores": sc})
        if sc is not None:
            by_class.setdefault(e["method_class"], []).append(sc)
            by_model_class.setdefault(f"{slot}|{e['photon_field']}", []).append(sc)
            by_field.setdefault(e["photon_field"], []).append(sc)
    counts = {}
    for r in rows:
        c = counts.setdefault(r["slot"], {})
        c[r["photon_field"]] = c.get(r["photon_field"], 0) + 1
    mc_photo = ref["observables"][PHOTO[0]]["mc"][PHOTO[1]]
    result = {
        "reference": {"photoelectric_fraction": mc_photo,
                      "half_w1": {k: ref["observables"][k].get("half_w1") for k in DISTS}},
        "method_counts_by_model": counts,
        "by_method_class": {k: summarize(v, ref) for k, v in by_class.items()},
        "by_photon_field": {k: summarize(v, ref) for k, v in by_field.items()},
        "by_model_and_photon_field": {k: summarize(v, ref) for k, v in by_model_class.items()},
        "conversations": rows,
    }
    out = RES / "M4_methods.json"
    json.dump(result, open(out, "w"), indent=1, default=float)
    print(f"wrote {out}")
    for k, v in result["by_photon_field"].items():
        m = v["median_abs_rel_diff"]
        print(f"{k:12s} n={v['n']:2d}  |scatter frac| {m['L1_scattered_fraction_fluence']:.3f}  "
              f"|D/phi0| {m['L5_D_over_phi0_Gy_cm2']:.3f}  photoelectric {v['median_photoelectric_fraction']}  "
              f"W1 scattered spectrum {v['median_w1']['L1_photon_fluence_spectrum_scattered']}")


if __name__ == "__main__":
    main()
