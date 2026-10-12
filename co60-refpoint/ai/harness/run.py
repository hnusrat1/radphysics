#!/usr/bin/env python3
"""Run the AI matrix (STUDY_PLAN Sections 5.2-5.4): models x conditions x tasks x repeats.

    python -m ai.harness.run --tag P0-pilot --config P0 --slots <slot> --conditions A0 A1 A2 --tasks observables events --k 1
    python -m ai.harness.run --tag M4-R0 --config R0 --conditions A0 A1 A2 --tasks observables events --k 5
    python -m ai.harness.run ... --dry-run      # count calls and estimate cost; no API call

Every conversation is written once to ai/raw/<tag>/<slot>/<config>_<condition>_<task>_r<k>/ and made read-only:
  meta.yaml          slot, model as reported by the API, settings, prompt version and hashes, timings, usage,
                     tool calls, retries, outcome (ok | parse_failure | refusal | api_error | tool_budget)
  turns/NNN_request.json, turns/NNN_response.json   verbatim API traffic (no credentials)
  tool_calls/NNN.py, tool_calls/NNN.out.json         code the model ran in the sandbox and what it got back
  final.txt          the last reply;  parsed.json  the validated JSON answer (absent on failure)
A directory that exists is never overwritten (STUDY_PLAN Section 10.4); rerunning skips it.
"""
import argparse
import datetime
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

import yaml

from ai.harness import providers
from ai.harness.parse import parse_reply
from ai.harness.sandbox import Sandbox

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "ai" / "raw"
MAX_TOOL_CALLS = 30
EVENT_CALLS_PER_CONDITION = 5          # STUDY_PLAN 5.2: 5 calls of 200 events per model and condition
EVENT_CONDITIONS = ("A0", "A1")        # STUDY_PLAN 5.2: sampled events in A0 and A1
RETRY_TEXT = ("Your reply could not be read against the required format: {err}. Reply again with only the JSON "
              "object, in a ```json fenced code block, with every key and unit as specified.")
BUDGET_TEXT = "The tool-call budget for this question is used up. Give your final answer now, in the required format."


def load_secrets(path=ROOT / ".secrets"):
    """KEY=VALUE lines from the gitignored .secrets file (STUDY_PLAN Section 10.8) into the environment.
    Variables already set in the environment (for example cloud environment secrets) take precedence."""
    if not path.exists():
        return []
    loaded = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            if k.strip() and not os.environ.get(k.strip()):
                os.environ[k.strip()] = v.strip()
                loaded.append(k.strip())
    return loaded


def utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_prompts(version):
    d = ROOT / "ai" / "prompts" / version
    man = yaml.safe_load(open(d / "MANIFEST.yaml"))
    files = {p.name: p.read_text() for p in d.iterdir() if p.is_file() and p.name != "MANIFEST.yaml"}
    for k, h in man["sha256"].items():
        if hashlib.sha256(files[k].encode()).hexdigest() != h:
            sys.exit(f"prompt file {k} does not match ai/prompts/{version}/MANIFEST.yaml")
    return files, man


def a2_files():
    d = ROOT / "ai" / "data"
    return [d / "xcom_water.txt", d / "estar_water.txt"]


def freeze(path):
    for p in Path(path).rglob("*"):
        if p.is_file():
            p.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def run_conversation(slot, condition, task, config, prompts, version, outdir):
    system = prompts[f"system_{condition}.txt"]
    user = prompts[f"task_{task}_{config}.txt"]
    schema = json.loads(prompts[f"schema_{task}.json"])
    tools = [] if condition == "A0" else [providers.PYTHON_TOOL]
    outdir.mkdir(parents=True)
    (outdir / "turns").mkdir()
    meta = {"slot": slot["slot"], "provider": slot["provider"], "api": slot["api"], "model_requested": slot["model_id"],
            "settings": slot.get("settings", {}), "condition": condition, "task": task, "config": config,
            "prompt_version": version, "prompt_sha256": {k: hashlib.sha256(prompts[k].encode()).hexdigest()
                                                         for k in (f"system_{condition}.txt",
                                                                   f"task_{task}_{config}.txt")},
            "started_utc": utc(), "turns": [], "tool_calls": 0, "retries": 0, "outcome": None}
    sandbox = Sandbox(a2_files() if condition == "A2" else ())
    try:
        conv = providers.start(slot, system, user, tools)
        n_turn, retried, final = 0, False, ""
        while True:
            n_turn += 1
            turn = conv.step()
            (outdir / "turns" / f"{n_turn:03d}_request.json").write_text(json.dumps(turn.raw_request, indent=1))
            (outdir / "turns" / f"{n_turn:03d}_response.json").write_text(json.dumps(turn.raw_response, indent=1))
            meta["turns"].append({"n": n_turn, "usage": turn.usage, "stop": turn.stop, "wall_s": turn.wall_s,
                                  "http_attempts": turn.http_attempts, "model_reported": turn.model_reported,
                                  "tool_calls": len(turn.tool_calls)})
            meta["model_reported"] = turn.model_reported or meta.get("model_reported")
            if turn.tool_calls:
                results = []
                for cid, name, args in turn.tool_calls:
                    meta["tool_calls"] += 1
                    k = meta["tool_calls"]
                    (outdir / "tool_calls").mkdir(exist_ok=True)
                    if name != "python" or "code" not in args or not tools:
                        out = {"error": f"unknown tool or missing code: {name}"}
                    elif k > MAX_TOOL_CALLS:
                        out = {"error": "tool-call budget exhausted"}
                    else:
                        (outdir / "tool_calls" / f"{k:03d}.py").write_text(args["code"])
                        out = sandbox.run(args["code"])
                    (outdir / "tool_calls" / f"{k:03d}.out.json").write_text(json.dumps(out, indent=1))
                    results.append((cid, name, json.dumps(out)))
                conv.add_tool_results(results)
                if meta["tool_calls"] >= MAX_TOOL_CALLS:
                    conv.add_user(BUDGET_TEXT)
                    meta["outcome"] = "tool_budget"
                continue
            final = turn.text
            obj, errs = parse_reply(final, schema)
            if not errs:
                (outdir / "parsed.json").write_text(json.dumps(obj, indent=1))
                meta["outcome"] = "ok" if meta["outcome"] is None else meta["outcome"] + "+ok"
                break
            if retried:
                meta["outcome"] = "parse_failure"
                meta["parse_errors"] = errs
                break
            retried = True
            meta["retries"] += 1
            meta["first_parse_errors"] = errs
            conv.add_user(RETRY_TEXT.format(err="; ".join(errs[:5])))
        (outdir / "final.txt").write_text(final)
    except Exception as e:                       # recorded, never hidden (STUDY_PLAN Section 10.6)
        meta["outcome"] = "api_error"
        meta["error"] = f"{type(e).__name__}: {e}"
    finally:
        sandbox.close()
        meta["finished_utc"] = utc()
        meta["usage_total"] = {k: sum(t["usage"].get(k, 0) or 0 for t in meta["turns"])
                               for k in ("input_tokens", "output_tokens", "reasoning_tokens")}
        (outdir / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
        freeze(outdir)
    return meta


def matrix(slots, conditions, tasks, k, config):
    for s in slots:
        for c in conditions:
            for t in tasks:
                if t == "events" and c not in EVENT_CONDITIONS:
                    continue
                reps = EVENT_CALLS_PER_CONDITION if t == "events" else k
                for r in range(1, reps + 1):
                    yield s, c, t, r


def estimate_cost(plan, prices, tokens):
    total, rows = 0.0, []
    for s, c, t, r in plan:
        p = prices.get(s["slot"])
        tk = tokens[t][c]
        cost = (tk["input"] * p["input_per_mtok"] + tk["output"] * p["output_per_mtok"]) / 1e6 if p else None
        rows.append((s["slot"], c, t, cost))
        total += cost or 0.0
    return total, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--config", default="R0", choices=["R0", "P0"])
    ap.add_argument("--models", default=str(ROOT / "ai" / "models.yaml"))
    ap.add_argument("--slots", nargs="*")
    ap.add_argument("--conditions", nargs="*", default=["A0", "A1", "A2"])
    ap.add_argument("--tasks", nargs="*", default=["observables", "events"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--prompt-version", default="v1")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--raw-root", default=str(RAW))
    a = ap.parse_args()
    loaded = load_secrets()
    if loaded:
        print(f"keys from .secrets: {', '.join(loaded)}")
    models = yaml.safe_load(open(a.models))
    slots = [s for s in models["slots"] if not a.slots or s["slot"] in a.slots]
    plan = list(matrix(slots, a.conditions, a.tasks, a.k, a.config))
    prices = yaml.safe_load(open(ROOT / "ai" / "prices.yaml"))
    total, _ = estimate_cost(plan, prices["prices"], prices["tokens_per_conversation"])
    print(f"{len(plan)} conversations; estimated cost ${total:,.0f} (ai/prices.yaml, unverified list prices)")
    if a.dry_run:
        return
    if a.config == "R0" and not models.get("frozen"):
        sys.exit("ai/models.yaml is not frozen: R0 runs start only after Humza approves the model list and budget.")
    missing = [s["slot"] for s in slots if s["api"] != "mock" and (not s.get("model_id")
                                                                   or not os.environ.get(s["api_key_env"]))]
    if missing:
        sys.exit(f"model_id or API key missing for: {missing}")
    prompts, man = load_prompts(a.prompt_version)
    if a.config == "R0" and not man.get("frozen"):
        sys.exit(f"ai/prompts/{a.prompt_version} is not frozen; freeze it when M4 starts (STUDY_PLAN 10.3).")
    for s, c, t, r in plan:
        out = Path(a.raw_root) / a.tag / s["slot"] / f"{a.config}_{c}_{t}_r{r}"
        if out.exists():
            continue
        m = run_conversation(s, c, t, a.config, prompts, a.prompt_version, out)
        print(f"{s['slot']} {c} {t} r{r}: {m['outcome']} ({m['tool_calls']} tool calls, {m['retries']} retries)",
              flush=True)


if __name__ == "__main__":
    main()
