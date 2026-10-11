#!/usr/bin/env python3
"""Run a list of Monte Carlo jobs in order, within a wall-time budget, resumably.

    python mc/tools/queue.py mc/queues/M2.yaml --deadline-min 105

Each item is a Geant4 run spec (run_co60.py) or an EGSnrc spec (run_egsnrc.py). Finished runs are skipped,
unfinished ones are resumed; no batch starts if it would end after the deadline. The cloud session stops
background tasks after 2 h, so a long queue runs as a sequence of invocations of this script (STATUS M1,
second container restart).
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manifest import STUDY_ROOT  # noqa: E402


def finished(run_id):
    m = STUDY_ROOT / "mc" / "runs" / run_id / "manifest.yaml"
    return m.exists() and bool((yaml.safe_load(open(m)) or {}).get("finished_utc"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("queue")
    ap.add_argument("--deadline-min", type=float, required=True)
    args = ap.parse_args()
    t0 = time.time()
    items = yaml.safe_load(open(args.queue))["items"]
    for it in items:
        spec = STUDY_ROOT / it["spec"]
        run_id = yaml.safe_load(open(spec))["run_id"]
        if finished(run_id):
            continue
        left = args.deadline_min - (time.time() - t0) / 60
        if left < 20:   # room for at least one batch
            break
        if it["kind"] == "geant4":
            cmd = [sys.executable, str(STUDY_ROOT / "mc/tools/run_co60.py"), str(spec), "--exe", it["exe"]]
        else:
            cmd = [sys.executable, str(STUDY_ROOT / "mc/tools/run_egsnrc.py"), str(spec)]
        cmd += ["--resume", "--deadline-min", f"{left:.1f}"]
        print(f"START {run_id} ({left:.0f} min left) {time.strftime('%H:%M:%S', time.gmtime())}Z", flush=True)
        subprocess.run(cmd, cwd=STUDY_ROOT, check=False)
        if not finished(run_id):
            print(f"PAUSED in {run_id}", flush=True)
            print("QUEUE_PAUSED", flush=True)
            return
        print(f"END {run_id} {time.strftime('%H:%M:%S', time.gmtime())}Z", flush=True)
    done = all(finished(yaml.safe_load(open(STUDY_ROOT / it["spec"]))["run_id"]) for it in items)
    print("QUEUE_DONE" if done else "QUEUE_PAUSED", flush=True)


if __name__ == "__main__":
    main()
