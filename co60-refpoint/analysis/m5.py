#!/usr/bin/env python3
"""M5 analysis: AI answers against the Monte Carlo reference (STUDY_PLAN Section 6).

    python analysis/m5.py --tag M4-R0                 # score ai/raw/M4-R0/
    python analysis/m5.py --controls                  # score the controls only (no AI data needed)

Per conversation: a verdict per quantity (pass / fail / unresolved / missing; analysis/scoring.py).
Per model and condition: the median and range over the k repeats of each scalar's relative difference and of
each distribution's W1 and JSD; the verdict of the median answer; the breakpoint layer (first layer with a
resolved failure); H1/H2 (primary-photon against transport quantities) and H3 (PDD(5) against D/Phi0).
Sampled events: weighted KS per feature and energy distance against the Monte Carlo event sample, and the
fraction of physically invalid events under the strict and bound rules (DECISIONS D014), with the Monte Carlo's
own rate alongside. Counts of refusals, parse failures, API errors and tool calls.

Controls, scored like a model (they validate the scorer before any AI answer exists):
  positive_livermore, positive_penelope   the other Geant4 lists' values (expected: pass or unresolved)
  negative_dose_plus5pct                  option4 values with D/Phi0, K, K_col x 1.05 (expected break at Layer 5)
  negative_uniform                        the harness mock answer (expected break at Layer 1)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis import scoring as S  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "analysis" / "results"
OBS = {o["id"]: o for o in yaml.safe_load(open(ROOT / "ai" / "observables.yaml"))["observables"]}
BINS = yaml.safe_load(open(ROOT / "config" / "bins.yaml"))["comparison"]


def score_answer(ans, ref):
    out = {}
    for oid, r in ref["observables"].items():
        o = OBS[oid]
        a = ans.get(oid)
        if o["kind"] == "scalar":
            out[oid] = S.scalar_verdict(a, r["mc"], r["tolerance"], r.get("floor_abs", 0.0))
        elif o["kind"] == "fractions":
            out[oid] = S.fractions_verdict(a, r["mc"], o["categories"], r.get("floor_abs"))
        else:
            edges = BINS[o["grid"]]["edges"]
            out[oid] = S.distribution_verdict(a, r["mc"], edges, r.get("half_w1", 0.0), r.get("half_jsd", 0.0),
                                              r.get("floor_w1", 0.0), r.get("floor_jsd", 0.0))
    return out


def summarize(answers, ref):
    """answers: list of parsed answers (k repeats) for one model x condition."""
    scored = [score_answer(a, ref) for a in answers]
    summary = {"n": len(answers), "observables": {}}
    by_layer = {}
    for oid, r in ref["observables"].items():
        o = OBS[oid]
        vs = [s[oid]["verdict"] for s in scored]
        e = {"verdicts": vs, "pass_rate": vs.count("pass") / len(vs) if vs else None}
        if o["kind"] == "scalar":
            vals = [a.get(oid) for a in answers if S.finite(a.get(oid))]
            if vals:
                med = float(np.median(vals))
                e["median"] = med
                e["rel_diff_median"] = (med - r["mc"]) / r["mc"]
                e["rel_diff_range"] = [(min(vals) - r["mc"]) / r["mc"], (max(vals) - r["mc"]) / r["mc"]]
                e["verdict"] = S.scalar_verdict(med, r["mc"], r["tolerance"], r.get("floor_abs", 0.0))["verdict"]
            else:
                e["verdict"] = "missing"
        elif o["kind"] == "fractions":
            ok = [a.get(oid) for a in answers if isinstance(a.get(oid), dict)]
            if ok:
                med = {c: float(np.median([x.get(c, np.nan) for x in ok])) for c in o["categories"]}
                e["median"] = med
                e["verdict"] = S.fractions_verdict(med, r["mc"], o["categories"], r.get("floor_abs"))["verdict"]
            else:
                e["verdict"] = "missing"
        else:
            w = [s[oid].get("w1") for s in scored if "w1" in s[oid]]
            j = [s[oid].get("jsd") for s in scored if "jsd" in s[oid]]
            if w:
                e["w1_median"], e["w1_range"] = float(np.median(w)), [float(min(w)), float(max(w))]
                e["jsd_median"], e["jsd_range"] = float(np.median(j)), [float(min(j)), float(max(j))]
                # verdict of the repeat with the median W1
                i = int(np.argsort(w)[len(w) // 2])
                e["verdict"] = [s for s in scored if "w1" in s[oid]][i][oid]["verdict"]
            else:
                e["verdict"] = "missing"
        summary["observables"][oid] = e
        by_layer.setdefault(o["layer"], []).append(e["verdict"])
    summary["breakpoint_layer"], summary["breakpoint_reason"] = S.breakpoint(by_layer)
    summary["verdict_counts_by_layer"] = {L: {v: vs.count(v) for v in set(vs)} for L, vs in by_layer.items()}
    # H1 / H2: median |relative difference| of scalar quantities by tag
    for tag in ("H1", "H2"):
        rd = [abs(summary["observables"][k]["rel_diff_median"]) for k, o in OBS.items()
              if tag in o["tags"] and "rel_diff_median" in summary["observables"].get(k, {})]
        summary[f"{tag}_median_abs_rel_diff"] = float(np.median(rd)) if rd else None
    pd, dd = (summary["observables"].get(k, {}).get("rel_diff_median") for k in ("L5_PDD5_percent",
                                                                                  "L5_D_over_phi0_Gy_cm2"))
    summary["H3"] = {"abs_rel_diff_PDD5": abs(pd) if pd is not None else None,
                     "abs_rel_diff_D_over_phi0": abs(dd) if dd is not None else None}
    return summary


PROC_AI_TO_MC = {"compton": "compt", "photoelectric": "phot", "pair": "conv", "rayleigh": "Rayl"}


def mc_event_view():
    """The Monte Carlo event sample in the AI fields (M2 option4 when available, else M1), with weights."""
    from analysis.events import ai_view, load_events
    for run in ("M2-R0-option4", "M1-R0-option4"):
        ev = load_events(ROOT / "mc" / "runs" / run)
        if ev is not None:
            v = ai_view(ev)
            v["source"] = run
            return v
    return None


def ai_event_view(events):
    def num(x):
        return float(x) if S.finite(x) else np.nan
    return {"process": np.array([PROC_AI_TO_MC.get(e.get("process"), "other") for e in events]),
            "e_in": np.array([num(e.get("e_in_MeV")) for e in events]),
            "e_out": np.array([num(e.get("e_out_MeV")) for e in events]),
            "cos_scatter": np.array([num(e.get("cos_scatter")) for e in events]),
            "t_e": np.array([num(e.get("t_e_MeV")) for e in events]),
            "t_pos": np.array([num(e.get("t_positron_MeV")) if e.get("t_positron_MeV") is not None else 0.0
                               for e in events])}


def score_events(ai_events, mc, delta_c, seed=1):
    """Distributional distances and validity of a pooled AI event sample against the Monte Carlo sample."""
    from analysis.validity import BINDING_TOLERANCE, check_events
    a = ai_event_view(ai_events)
    n = len(a["e_in"])
    out = {"n_events": n}
    if n == 0:
        return out
    w = mc["weight"]
    procs = ["compt", "phot", "conv", "Rayl"]
    out["process_fraction_ai"] = {p: float(np.mean(a["process"] == p)) for p in procs}
    out["process_fraction_mc"] = {p: float(w[mc["process"] == p].sum() / w.sum()) for p in procs}
    out["ks"] = {}
    for f in ("e_in", "e_out", "cos_scatter", "t_e"):
        xa = a[f][np.isfinite(a[f]) & (a["process"] == "compt")]
        sel = (mc["process"] == "compt") & np.isfinite(mc[f]) & ((mc[f] >= 0) if f == "e_out" else True)
        if len(xa) >= 5:
            out["ks"][f + "_compton"] = S.weighted_ks(xa, mc[f][sel], w[sel])
    scale = {"e_in": 1.0, "e_out": 1.0, "cos_scatter": 1.0, "t_e": 1.0}
    rng = np.random.default_rng(seed)
    mc_e = {k: np.where(np.isfinite(mc[k]) & ((mc[k] >= 0) if k == "e_out" else True), mc[k], np.nan)
            for k in ("e_in", "e_out", "cos_scatter", "t_e")}
    fm = S.event_features(mc_e, scale)
    fa = S.event_features(a, scale)
    ref = S.resample(fm, w, 4000, rng)
    out["energy_distance"] = S.energy_distance(fa, ref)
    base = [S.energy_distance(S.resample(fm, w, n, rng), ref) for _ in range(20)]
    out["energy_distance_mc_baseline"] = {"median": float(np.median(base)), "p95": float(np.percentile(base, 95)),
                                          "max": float(np.max(base)), "n_draws": 20}
    for rules, kw in (("strict", {}), ("bound", {"delta_c": delta_c, "energy_tol": BINDING_TOLERANCE})):
        chk = check_events(a["process"], a["e_in"], np.where(np.isfinite(a["e_out"]), a["e_out"], -1.0),
                           a["cos_scatter"], a["t_e"], a["t_pos"], **kw)
        out[f"invalid_fraction_{rules}"] = float(np.mean(~chk["valid"]))
        out[f"invalid_by_check_{rules}"] = {k: float(np.mean(~v)) for k, v in chk.items() if k != "valid"}
    return out


def outcome(d, meta, task):
    """The conversation's outcome. The harness records ok, parse_failure, api_error or tool_budget; a parse
    failure whose last reply is a JSON object with nothing in it (an empty events list, or no observable with a
    value) is counted as a refusal (STUDY_PLAN Section 5.2). The rule reads final.txt and never changes ai/raw/."""
    out = meta["outcome"] or "unknown"
    if out.endswith("ok"):
        return "ok"
    if out == "parse_failure" and (d / "final.txt").exists():
        from ai.harness.parse import extract_json
        obj, _ = extract_json((d / "final.txt").read_text())
        # Only a top-level answer counts (it carries "method"); the parser's fallback can return a nested object
        # from a reply whose JSON is malformed, and that is a parse failure, not a refusal.
        if isinstance(obj, dict) and "method" in obj:
            if task == "events" and obj.get("events") == []:
                return "refusal"
            if task == "observables" and all(obj.get(k) in (None, [], {}) for k in OBS):
                return "refusal"
    return out


def count_outcomes(tag, config="R0"):
    """{slot: {task: {condition: counts}}} for every conversation in ai/raw/<tag>/."""
    counts = {}
    for d in sorted((ROOT / "ai" / "raw" / tag).glob(f"*/{config}_*_r*")):
        if not (d / "meta.yaml").exists():
            continue
        slot, (_, cond, task, _) = d.parent.name, d.name.split("_")
        meta = yaml.safe_load(open(d / "meta.yaml"))
        c = counts.setdefault(slot, {}).setdefault(task, {}).setdefault(
            cond, {"ok": 0, "parse_failure": 0, "refusal": 0, "api_error": 0, "tool_calls": 0, "retries": 0})
        key = outcome(d, meta, task)
        c[key] = c.get(key, 0) + 1
        c["tool_calls"] += meta.get("tool_calls", 0)
        c["retries"] += meta.get("retries", 0)
    return counts


def load_events_answers(tag, config="R0"):
    base = ROOT / "ai" / "raw" / tag
    out = {}
    for d in sorted(base.glob(f"*/{config}_*_events_r*")):
        if (d / "parsed.json").exists():
            slot, cond = d.parent.name, d.name.split("_")[1]
            out.setdefault(slot, {}).setdefault(cond, []).extend(json.load(open(d / "parsed.json"))["events"])
    return out


def load_answers(tag, config="R0"):
    """{slot: {condition: [parsed answers]}} and outcome counts, from ai/raw/<tag>/."""
    base = ROOT / "ai" / "raw" / tag
    answers = {}
    for d in sorted(base.glob(f"*/{config}_*_observables_r*")):
        slot, cond = d.parent.name, d.name.split("_")[1]
        if (d / "parsed.json").exists():
            answers.setdefault(slot, {}).setdefault(cond, []).append(json.load(open(d / "parsed.json")))
    return answers, count_outcomes(tag, config)


def control_answers(ref):
    """Pseudo-answers from Monte Carlo results, to validate the scorer."""
    from ai.harness.mock import placeholder_observables
    out = {}
    m2 = json.load(open(RES / "M2_results.json")) if (RES / "M2_results.json").exists() else {}
    base = {}
    for oid, r in ref["observables"].items():
        base[oid] = r["mc"]
    for L in ("livermore", "penelope"):
        run = m2.get("runs", {}).get(f"M2-R0-{L}")
        if not run:
            continue
        a = {}
        for oid, o in OBS.items():
            if oid == "L5_D_over_phi0_Gy_cm2" and m2.get("dose_bf", {}).get(L):
                a[oid] = m2["dose_bf"][L]["D_over_phi0_Gy_cm2"]["mean"]
            elif oid == "L5_beta" and m2.get("dose_bf", {}).get(L):
                a[oid] = m2["dose_bf"][L]["beta"]["ratio"]
            elif oid == "L5_PDD5_percent":
                a[oid] = base[oid]                      # no PDD run per list; taken as the reference
            else:
                v = run
                for part in o["mc"].split("/"):
                    v = v[part]
                v = v["mean"]
                if o["kind"] == "fractions":
                    a[oid] = {c: v[i] for c, i in (("compton", 0), ("photoelectric", 1), ("pair", 2), ("rayleigh", 3))}
                else:
                    a[oid] = v
        out[f"positive_{L}"] = {"control": [a]}
    neg = dict(base)
    for k in ("L5_D_over_phi0_Gy_cm2", "L5_K_over_phi0_Gy_cm2", "L5_Kcol_over_phi0_Gy_cm2"):
        neg[k] = base[k] * 1.05
    out["negative_dose_plus5pct"] = {"control": [neg]}
    out["negative_uniform"] = {"control": [placeholder_observables()]}
    out["identity"] = {"control": [dict(base)]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag")
    ap.add_argument("--controls", action="store_true")
    a = ap.parse_args()
    ref = json.load(open(RES / "reference_R0.json"))
    result = {"reference_sources": ref["sources"], "reference_provisional": ref["provisional"]}
    if a.controls or not a.tag:
        ctl = control_answers(ref)
        result["controls"] = {name: summarize(v["control"], ref) for name, v in ctl.items()}
        for name, s in result["controls"].items():
            print(f"{name:28s} breakpoint layer {s['breakpoint_layer']}  counts {s['verdict_counts_by_layer']}")
        mc = mc_event_view()
        if mc is not None:
            dc = ref.get("events", {}).get("delta_c", 0.01)
            rng = np.random.default_rng(7)
            idx = rng.choice(len(mc["weight"]), 1000, p=mc["weight"] / mc["weight"].sum())
            names = {"compt": "compton", "phot": "photoelectric", "conv": "pair", "Rayl": "rayleigh"}
            pseudo = [{"process": names[mc["process"][i]], "e_in_MeV": mc["e_in"][i],
                       "e_out_MeV": mc["e_out"][i] if mc["e_out"][i] >= 0 else None,
                       "cos_scatter": mc["cos_scatter"][i] if np.isfinite(mc["cos_scatter"][i]) else None,
                       "t_e_MeV": mc["t_e"][i], "cos_electron": None,
                       "t_positron_MeV": mc["t_pos"][i] if mc["process"][i] == "conv" else None} for i in idx]
            from ai.harness.mock import placeholder_events
            result["event_controls"] = {
                "positive_mc_resample": score_events(pseudo, mc, dc),
                "negative_mock": score_events(placeholder_events(1000)["events"], mc, dc)}
            for k, v in result["event_controls"].items():
                print(f"events {k:22s} energy distance {v['energy_distance']:.4f} (MC baseline "
                      f"{v['energy_distance_mc_baseline']['median']:.4f}, 95th pct {v['energy_distance_mc_baseline']['p95']:.4f}); invalid strict {v['invalid_fraction_strict']:.3f}"
                      f" bound {v['invalid_fraction_bound']:.3f}")
        out = RES / "M5_controls.json"
    else:
        answers, counts = load_answers(a.tag)
        result["counts"] = counts
        result["models"] = {slot: {cond: summarize(v, ref) for cond, v in conds.items()}
                            for slot, conds in answers.items()}
        mc = mc_event_view()
        dc = ref.get("events", {}).get("delta_c")
        result["events"] = {slot: {cond: score_events(ev, mc, dc) for cond, ev in conds.items()}
                            for slot, conds in load_events_answers(a.tag).items()} if mc is not None else {}
        out = RES / f"M5_{a.tag}.json"
    json.dump(result, open(out, "w"), indent=1, default=float)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
