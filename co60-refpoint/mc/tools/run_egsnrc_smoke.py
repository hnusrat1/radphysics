"""EGSnrc installation check: tutor1 and the DOSRZnrc template input.

Needs the EGSnrc environment (source env/setup_env.sh).

Example:
  python mc/tools/run_egsnrc_smoke.py --run-id M0-egsnrc-smoke
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manifest import STUDY_ROOT, base_manifest, utc_now, write_yaml  # noqa: E402

CASES = [
    # name, user code, input (None for codes without one), pegs4 data
    ("tutor1", "tutor1", None, "tutor_data"),
    ("dosrznrc_template", "dosrznrc", "dosrznrc_template", "521icru"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()

    egs_home = Path(os.environ["EGS_HOME"])
    run_dir = STUDY_ROOT / "mc" / "runs" / args.run_id
    if run_dir.exists():
        sys.exit(f"{run_dir} exists; runs are never overwritten (STUDY_PLAN Section 10.4)")
    run_dir.mkdir(parents=True)

    manifest = base_manifest(args.run_id, "EGSnrc installation check", code="egsnrc")
    manifest["cases"] = []
    manifest["started_utc"] = utc_now()
    for name, code, inp, pegs in CASES:
        cdir = run_dir / name
        cdir.mkdir()
        cmd = [code] + (["-i", inp] if inp else []) + ["-p", pegs]
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=egs_home / code, capture_output=True, text=True)
        wall = time.time() - t0
        (cdir / "stdout.log").write_text(proc.stdout + proc.stderr)
        if inp:
            shutil.copy(egs_home / code / f"{inp}.egsinp", cdir)
            for ext in ("egslst", "errors"):
                f = egs_home / code / f"{inp}.{ext}"
                if f.exists():
                    shutil.copy(f, cdir)
        finished = "Finished simulation" in proc.stdout
        m = re.search(r"NUMBER OF HISTORIES=\s*(\d+)", (egs_home / code / f"{inp}.egsinp").read_text()) if inp else None
        manifest["cases"].append({
            "name": name,
            "command": " ".join(cmd),
            "pegs4_data": pegs,
            "histories": int(m.group(1)) if m else None,
            "exit_code": proc.returncode,
            "finished_simulation": finished,
            "wall_s": round(wall, 2),
        })
        print(f"{name}: exit {proc.returncode}, finished={finished}")
    manifest["finished_utc"] = utc_now()
    write_yaml(run_dir / "manifest.yaml", manifest)


if __name__ == "__main__":
    main()
