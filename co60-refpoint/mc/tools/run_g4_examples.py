"""Build and run Geant4 examples with their shipped macros; compare with shipped outputs.

Each example in the Geant4 distribution ships a macro (*.in) and the output the
Geant4 team obtained with it (*.out). Running the same macro on this build and
comparing the physics results checks the build and the data libraries.

Example:
  python mc/tools/run_g4_examples.py --run-id M0-g4-examples --build-root /opt/build/examples
"""

import argparse
import difflib
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manifest import STUDY_ROOT, base_manifest, utc_now, write_yaml  # noqa: E402

EXAMPLES = [
    # name, path under examples/, executable, macro, reference output
    ("B1", "basic/B1", "exampleB1", "exampleB1.in", "exampleB1.out"),
    ("TestEm0", "extended/electromagnetic/TestEm0", "TestEm0", "TestEm0.in", "TestEm0.out"),
    ("fanoCavity", "extended/medical/fanoCavity", "fanoCavity", "fanoCavity.in", "fanoCavity.out"),
]

# Lines that legitimately differ between machines and runs.
VOLATILE = re.compile(
    r"User=|Real=|Sys=|Elapsed|Date|date|seed|Seed|host|Host|/opt/|/home/|/build|"
    r"Geant4 version|Copyright|www\.|Thread|thread|G4WT|Pool|pool|Tasking|task|"
    r"Available|memory|Memory|---> Begin of event|--> Event"
)

RESULT_LINES = re.compile(
    r"dose|Dose|cross section|CrossSection|mean free path|g/cm2|dE/dx|range|Range|"
    r"Mass_energy_transfer|Mean energy|edep|Imean|Process calls"
)


def build(src, build_dir):
    build_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cmake", "-S", str(src), "-B", str(build_dir), "-DCMAKE_BUILD_TYPE=Release"],
                   check=True, capture_output=True, text=True)
    subprocess.run(["cmake", "--build", str(build_dir), "-j", "4"], check=True, capture_output=True, text=True)


def result_lines(text):
    return [ln.rstrip() for ln in text.splitlines()
            if RESULT_LINES.search(ln) and not VOLATILE.search(ln)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--build-root", required=True)
    ap.add_argument("--examples-dir", default=None,
                    help="installed examples directory (default: <geant4 prefix>/share/Geant4/examples)")
    args = ap.parse_args()

    run_dir = STUDY_ROOT / "mc" / "runs" / args.run_id
    if run_dir.exists():
        sys.exit(f"{run_dir} exists; runs are never overwritten (STUDY_PLAN Section 10.4)")
    run_dir.mkdir(parents=True)

    manifest = base_manifest(args.run_id, "Geant4 build check: shipped example macros against shipped outputs")
    examples_dir = Path(args.examples_dir or Path(manifest["geant4"]["prefix"]) / "share" / "Geant4" / "examples")
    manifest["examples_dir"] = str(examples_dir)
    manifest["examples"] = []
    manifest["started_utc"] = utc_now()

    for name, rel, exe, macro, ref in EXAMPLES:
        src = examples_dir / rel
        bdir = Path(args.build_root) / name
        edir = run_dir / name
        edir.mkdir()
        build(src, bdir)
        shutil.copy(src / macro, edir / macro)
        shutil.copy(src / ref, edir / f"reference_{ref}")
        t0 = time.time()
        proc = subprocess.run([str(bdir / exe), macro], cwd=edir, capture_output=True, text=True)
        wall = time.time() - t0
        (edir / "stdout.log").write_text(proc.stdout)
        if proc.stderr:
            (edir / "stderr.log").write_text(proc.stderr)

        ours = result_lines(proc.stdout)
        theirs = result_lines((src / ref).read_text())
        diff = list(difflib.unified_diff(theirs, ours, "reference", "this_build", lineterm="", n=0))
        (edir / "result_lines.diff").write_text("\n".join(diff) + "\n")
        n_changed = sum(1 for d in diff if d[:1] in "+-" and not d.startswith(("+++", "---")))
        manifest["examples"].append({
            "name": name,
            "macro": macro,
            "exit_code": proc.returncode,
            "wall_s": round(wall, 1),
            "result_lines_compared": len(theirs),
            "result_lines_differing": n_changed,
        })
        print(f"{name}: exit {proc.returncode}, {len(theirs)} result lines, {n_changed} differing lines, {wall:.0f} s",
              flush=True)

    manifest["finished_utc"] = utc_now()
    write_yaml(run_dir / "manifest.yaml", manifest)


if __name__ == "__main__":
    main()
