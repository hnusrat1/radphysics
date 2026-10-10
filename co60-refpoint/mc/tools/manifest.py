"""Run manifest: what was run, with which code and data, where, and for how long.

STUDY_PLAN Section 8 lists the required fields. Geant4 version and data library
locations come from geant4-config so the manifest describes the build that ran,
not the build that was intended.
"""

import datetime
import os
import platform
import re
import socket
import subprocess
from pathlib import Path

import yaml

STUDY_ROOT = Path(__file__).resolve().parents[2]


def _run(cmd, cwd=None):
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_state():
    commit = _run(["git", "rev-parse", "HEAD"], cwd=STUDY_ROOT)
    status = _run(["git", "status", "--porcelain", "--", "."], cwd=STUDY_ROOT)
    return {"commit": commit, "dirty": bool(status)}


def geant4_state():
    datasets = {}
    for line in (_run(["geant4-config", "--datasets"]) or "").splitlines():
        parts = line.split()
        if len(parts) == 3:
            name, envvar, path = parts
            datasets[name] = {"envvar": envvar, "path": path, "env_value": os.environ.get(envvar)}
    return {
        "version": _run(["geant4-config", "--version"]),
        "prefix": _run(["geant4-config", "--prefix"]),
        "features": _run(["geant4-config", "--features"]),
        "datasets": datasets,
    }


def egsnrc_state():
    hen_house = os.environ.get("HEN_HOUSE")
    if not hen_house:
        return None
    root = Path(hen_house).resolve().parent
    return {
        "hen_house": hen_house,
        "egs_home": os.environ.get("EGS_HOME"),
        "egs_config": os.environ.get("EGS_CONFIG"),
        "commit": _run(["git", "rev-parse", "HEAD"], cwd=root),
        "describe": _run(["git", "describe", "--tags"], cwd=root),
    }


def machine_state():
    cpu = None
    try:
        with open("/proc/cpuinfo") as f:
            m = re.search(r"model name\s*:\s*(.+)", f.read())
            cpu = m.group(1) if m else None
    except OSError:
        cpu = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
    compiler = _run(["c++", "--version"])
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "compiler": compiler.splitlines()[0] if compiler else None,
    }


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def base_manifest(run_id, purpose, code="geant4"):
    manifest = {
        "run_id": run_id,
        "purpose": purpose,
        "created_utc": utc_now(),
        "git": git_state(),
        "machine": machine_state(),
    }
    if code == "geant4":
        manifest["geant4"] = geant4_state()
    elif code == "egsnrc":
        manifest["egsnrc"] = egsnrc_state()
    return manifest


def write_yaml(path, data):
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False, width=100)
