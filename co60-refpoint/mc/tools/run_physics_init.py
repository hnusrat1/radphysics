"""Check that each production EM physics list initializes with the installed data.

Runs the Geant4 TestEm0 example (no transport; it prints cross sections and
stopping powers from the physics tables) once per physics list, for G4_WATER,
photons at the two Co-60 line energies and electrons at 1 MeV. A missing or
unreadable data file stops initialization, so a clean run of all three lists is
the check. The printed values are archived for later use; they are not
interpreted here.

Example:
  python mc/tools/run_physics_init.py --exe /opt/build/examples/TestEm0/TestEm0 --run-id M0-physics-init
"""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manifest import STUDY_ROOT, base_manifest, utc_now, write_yaml  # noqa: E402

sys.path.insert(0, str(STUDY_ROOT))
from analysis.refs import ref_value  # noqa: E402

# Study name -> TestEm0 physics option
LISTS = {
    "G4EmStandardPhysics_option4": "emstandard_opt4",
    "G4EmLivermorePhysics": "emlivermore",
    "G4EmPenelopePhysics": "empenelope",
}


def macro(option, energies):
    lines = [
        "/control/verbose 1",
        "/run/verbose 1",
        f"/testem/phys/addPhysics {option}",
        "/process/em/printParameters",
        "/run/initialize",
        "/testem/det/setMat G4_WATER",
        "/gun/particle gamma",
    ]
    for e in energies:
        lines += [f"/gun/energy {e} MeV", "/run/beamOn"]
    lines += ["/gun/particle e-", "/gun/energy 1 MeV", "/run/beamOn"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()

    run_dir = STUDY_ROOT / "mc" / "runs" / args.run_id
    if run_dir.exists():
        sys.exit(f"{run_dir} exists; runs are never overwritten (STUDY_PLAN Section 10.4)")
    run_dir.mkdir(parents=True)

    energies = [ref_value("co60_line_1_energy"), ref_value("co60_line_2_energy")]
    manifest = base_manifest(args.run_id, "Initialization of the three production EM lists with the installed data")
    manifest["application"] = {"name": "TestEm0 (Geant4 extended/electromagnetic example, unmodified)",
                               "executable": args.exe, "material": "G4_WATER",
                               "gamma_energies_MeV": energies, "electron_energy_MeV": 1.0}
    manifest["lists"] = []
    manifest["started_utc"] = utc_now()
    for name, option in LISTS.items():
        ldir = run_dir / name
        ldir.mkdir()
        (ldir / "run.mac").write_text(macro(option, energies))
        t0 = time.time()
        proc = subprocess.run([args.exe, "run.mac"], cwd=ldir, capture_output=True, text=True)
        (ldir / "stdout.log").write_text(proc.stdout)
        if proc.stderr:
            (ldir / "stderr.log").write_text(proc.stderr)
        log = proc.stdout
        imean = re.search(r"Material:\s+G4_WATER\b[^\n]*\n[^\n]*?Imean:\s+([\d.]+) eV", log)
        n_runs = log.count("Run terminated")
        exceptions = re.findall(r"G4Exception-START.*?issued by\s*:\s*(\S+).*?\n(.*?)\n", log, re.S)
        manifest["lists"].append({
            "list": name,
            "testem0_option": option,
            "exit_code": proc.returncode,
            "runs_completed": n_runs,
            "runs_expected": len(energies) + 1,
            "g4_water_imean_eV": float(imean.group(1)) if imean else None,
            "g4exceptions": [f"{a}: {b.strip()}" for a, b in exceptions],
            "wall_s": round(time.time() - t0, 1),
        })
        print(f"{name}: exit {proc.returncode}, {n_runs} runs, exceptions={len(exceptions)}", flush=True)
    manifest["finished_utc"] = utc_now()
    write_yaml(run_dir / "manifest.yaml", manifest)


if __name__ == "__main__":
    main()
