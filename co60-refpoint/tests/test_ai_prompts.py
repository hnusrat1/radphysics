"""The AI prompts carry inputs only (STUDY_PLAN Section 10.2) and match their frozen hashes."""
import json
import re
from pathlib import Path

import yaml

from ai.harness.build_prompts import build

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "ai" / "prompts" / "v1"
NUM = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def numbers(text):
    return {float(m) for m in NUM.findall(text)}


def input_numbers():
    """Numbers that are legitimately in a prompt: the configuration, the bin edges and structural counts."""
    allowed = numbers((ROOT / "ai" / "configs.yaml").read_text())
    allowed |= numbers((ROOT / "config" / "bins.yaml").read_text())
    allowed |= set(range(0, 31)) | {100, 105, 110, 150, 200, 2020, 12, 1.0, 0.0}
    return allowed


def reference_values():
    vals = []
    res = ROOT / "analysis" / "results" / "M1_results.json"
    if res.exists():
        r = json.load(open(res))
        run = r["runs"][r["pooling"]["reference_run"]]
        for v in run.values():
            if isinstance(v, dict) and isinstance(v.get("mean"), float):
                vals.append(v["mean"])
        pdd = r.get("pdd", {}).get("PDD5", {}).get("mean")
        if pdd:
            vals.append(pdd)
    for e in yaml.safe_load(open(ROOT / "refs" / "refs.yaml")):
        if isinstance(e.get("value"), (int, float)):
            vals.append(float(e["value"]))
    return [v for v in vals if v != 0.0]


def test_manifest_matches():
    assert build("v1", check=True) == []


def test_no_reference_values_in_prompts():
    allowed = input_numbers()
    refs = reference_values()
    assert len(refs) > 50
    for f in PROMPTS.glob("*.txt"):
        for x in numbers(f.read_text()) - allowed:
            close = [v for v in refs if abs(x - v) <= 0.005 * abs(v)]
            assert not close, f"{f.name}: {x} is within 0.5% of reference value(s) {close}"


def test_schema_covers_every_observable():
    schema = json.load(open(PROMPTS / "schema_observables.json"))
    ids = [o["id"] for o in yaml.safe_load(open(ROOT / "ai" / "observables.yaml"))["observables"]]
    assert set(ids) <= set(schema["required"])
