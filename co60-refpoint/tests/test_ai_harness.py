"""Offline tests of the AI harness with the mock provider (no API call is made)."""
import json
import os
import shutil
from pathlib import Path

import pytest
import yaml

from ai.harness.parse import extract_json, parse_reply
from ai.harness.run import load_prompts, matrix, run_conversation

ROOT = Path(__file__).resolve().parents[1]
MOCK = {"slot": "mock", "provider": "mock", "api": "mock", "api_key_env": "NONE", "model_id": "mock"}
needs_root = pytest.mark.skipif(os.geteuid() != 0 or not shutil.which("unshare"),
                                reason="sandbox needs root and unshare")


def test_extract_last_fenced_block():
    obj, err = extract_json("x\n```json\n{\"a\": 1}\n```\nmore\n```json\n{\"a\": 2}\n```")
    assert obj == {"a": 2} and err is None


def test_extract_bare_object_fallback():
    obj, err = extract_json("The answer is {\"a\": {\"b\": 3}} as shown.")
    assert obj == {"a": {"b": 3}}


def test_schema_rejects_wrong_lengths_and_types():
    schema = json.loads((ROOT / "ai/prompts/v1/schema_observables.json").read_text())
    from ai.harness.mock import placeholder_observables
    good = placeholder_observables()
    assert parse_reply("```json\n" + json.dumps(good) + "\n```", schema)[1] == []
    bad = dict(good, L1_angular_primary=[0.5, 0.5], L5_beta="1.0")
    errs = parse_reply("```json\n" + json.dumps(bad) + "\n```", schema)[1]
    assert any("L1_angular_primary" in e for e in errs) and any("L5_beta" in e for e in errs)


def test_matrix_counts():
    plan = list(matrix([MOCK], ["A0", "A1", "A2"], ["observables", "events"], 5, "R0"))
    assert len(plan) == 3 * 5 + 2 * 5        # events only in A0 and A1, 5 calls each


@needs_root
@pytest.mark.parametrize("condition,task", [("A0", "observables"), ("A1", "observables"), ("A2", "observables"),
                                            ("A1", "events")])
def test_conversation_end_to_end(tmp_path, condition, task):
    prompts, _ = load_prompts("v1")
    meta = run_conversation(MOCK, condition, task, "P0", prompts, "v1", tmp_path / "c")
    assert meta["outcome"] == "ok", meta
    assert (tmp_path / "c" / "parsed.json").exists()
    if condition != "A0":
        out = json.loads((tmp_path / "c" / "tool_calls" / "001.out.json").read_text())
        assert out["stdout"].strip() == "4" and out["exit_code"] == 0
    # written once, read-only
    assert not os.access(tmp_path / "c" / "meta.yaml", os.W_OK) or os.geteuid() == 0
    assert oct((tmp_path / "c" / "meta.yaml").stat().st_mode)[-3:] == "444"


@needs_root
def test_parse_failure_retried_once(tmp_path):
    prompts, _ = load_prompts("v1")
    slot = dict(MOCK, settings={"malformed_first": True})
    meta = run_conversation(slot, "A0", "observables", "P0", prompts, "v1", tmp_path / "c")
    assert meta["retries"] == 1 and meta["outcome"] == "ok"
    assert yaml.safe_load((tmp_path / "c" / "meta.yaml").read_text())["first_parse_errors"]


@needs_root
def test_a2_sandbox_sees_data_and_not_the_repository(tmp_path):
    from ai.harness.run import a2_files
    from ai.harness.sandbox import Sandbox
    sb = Sandbox(a2_files())
    try:
        r = sb.run("import os\nprint(sorted(os.listdir('/tmp/data')))\nprint(os.path.exists('/home/user/radphysics'))"
                   "\nprint(os.listdir('/opt/geant4') if os.path.exists('/opt/geant4') else [])")
        assert r["stdout"].split("\n")[:3] == ["['estar_water.txt', 'xcom_water.txt']", "False", "[]"]
    finally:
        sb.close()


@needs_root
def test_sandbox_runs_python_but_no_compiler():
    """A1/A2 are Python conditions (D037): no C, C++ or Fortran build, no assembler or linker, no other runtime."""
    from ai.harness.sandbox import Sandbox
    sb = Sandbox()
    try:
        r = sb.run(
            "import os, subprocess, numpy, scipy.special\n"
            "open('t.c', 'w').write('int main(void){return 0;}')\n"
            "built = []\n"
            "for cmd in (['gcc', 't.c'], ['g++', '-x', 'c++', 't.c'], ['clang', 't.c'], ['gfortran', 't.c'],\n"
            "            ['as', '--version'], ['ld', '--version'], ['node', '-v'], ['java', '-version'], ['go', 'version']):\n"
            "    try:\n"
            "        ok = subprocess.run(cmd, capture_output=True).returncode == 0\n"
            "    except OSError:\n"
            "        ok = False\n"
            "    built += [cmd[0]] if ok else []\n"
            "print(built, os.path.exists('a.out'), round(float(scipy.special.expi(1.0)), 6))")
        assert r["exit_code"] == 0, r["stderr"]
        assert r["stdout"].strip() == "[] False 1.895118"
    finally:
        sb.close()
