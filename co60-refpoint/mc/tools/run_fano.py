"""Run the Geant4 fanoCavity example in independent seeded batches.

Each batch is a separate process with its own seeds, so the spread of the
batch results gives an uncertainty estimate independent of the example's own
event-by-event estimate. Results, logs, macros, and a manifest go to
mc/runs/<run_id>/.

Example:
  python mc/tools/run_fano.py --exe /opt/build/examples/fanoCavity/fanoCavity \
      --physics standard_opt4 --batches 10 --events 300000 --threads 4 \
      --run-id M0-fano-opt4 --seed-base 4101
"""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manifest import STUDY_ROOT, base_manifest, utc_now, write_yaml  # noqa: E402

# name -> (regex, field names for the captured groups)
PATTERNS = {
    "ratio": (r"\(Dose/EnergyFluence\)/Mass_energy_transfer = ([\d.eE+-]+) \+- ([\d.eE+-]+)",
              ("ratio", "ratio_err")),
    "mtr": (r"Mass_energy_transfer coef: ([\d.eE+-]+) cm2/g", ("mass_energy_transfer_cm2_g",)),
    "dose": (r"Dose/EnergyFluence\s+= ([\d.eE+-]+) cm2/g", ("dose_over_energy_fluence_cm2_g",)),
    "esec": (r"Mean energy of secondary e- = ([\d.eE+-]+) keV \+- ([\d.eE+-]+) %",
             ("mean_secondary_energy_keV", "mean_secondary_energy_rel_err_pct")),
    "edep": (r"Total edep in cavity = ([\d.eE+-]+) MeV \+- ([\d.eE+-]+) %",
             ("edep_cavity_MeV", "edep_cavity_rel_err_pct")),
    "imean": (r"Material:\s+G4_WATER\b[^\n]*\n[^\n]*?Imean:\s+([\d.]+) eV", ("g4_water_imean_eV",)),
    "time": (r"Run terminated\.(?:[^\n]*\n){1,3}?[^\n]*?User=([\d.]+)s Real=([\d.]+)s", ("g4_user_s", "g4_real_s")),
}


def make_macro(physics, seeds, events, energy_mev, extra):
    lines = [
        "/control/verbose 1",
        "/run/verbose 1",
        "/control/cout/ignoreThreadsExcept 0",
        f"/random/setSeeds {seeds[0]} {seeds[1]}",
        f"/testem/phys/addPhysics {physics}",
        "/run/initialize",
        "/process/em/printParameters",
        f"/gun/energy {energy_mev} MeV",
        *extra,
        f"/run/printProgress {max(events // 10, 1)}",
        f"/run/beamOn {events}",
    ]
    return "\n".join(lines) + "\n"


def parse(log):
    out = {}
    for regex, fields in PATTERNS.values():
        m = re.search(regex, log)
        if m:
            out.update({f: float(v) for f, v in zip(fields, m.groups())})
    m = re.search(r"Process calls frequency --->(.*)", log)
    if m:
        out["process_calls"] = {k: int(v) for k, v in re.findall(r"(\w+)= (\d+)", m.group(1))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--physics", required=True)
    ap.add_argument("--batches", type=int, default=10)
    ap.add_argument("--events", type=int, required=True, help="events per batch")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--energy-mev", type=float, default=1.25)
    ap.add_argument("--seed-base", type=int, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--purpose", default="Fano cavity test")
    ap.add_argument("--extra", action="append", default=[], help="extra macro command after /run/initialize")
    args = ap.parse_args()

    run_dir = STUDY_ROOT / "mc" / "runs" / args.run_id
    if run_dir.exists():
        sys.exit(f"{run_dir} exists; runs are never overwritten (STUDY_PLAN Section 10.4)")
    run_dir.mkdir(parents=True)

    manifest = base_manifest(args.run_id, args.purpose)
    manifest["application"] = {
        "name": "fanoCavity (Geant4 extended/medical example, unmodified)",
        "executable": args.exe,
        "physics_constructor": args.physics,
        "production_cut": "10 km for gamma, e-, e+ (example default, CSDA)",
        "compton_cross_section_factor": 1000,
        "beam": f"gamma, {args.energy_mev} MeV, uniform over the wall radius, normal incidence",
        "geometry": "wall G4_WATER 5 mm each side, cavity G4_WATER at 1 mg/cm3, 2 mm thick, radius 1 cm",
        "extra_macro_commands": args.extra,
    }
    manifest["statistics"] = {"batches": args.batches, "events_per_batch": args.events,
                              "threads": args.threads, "total_events": args.batches * args.events}
    manifest["batches"] = []
    manifest["started_utc"] = utc_now()

    t_start = time.time()
    results = []
    for i in range(args.batches):
        seeds = (args.seed_base * 1000 + 2 * i + 1, args.seed_base * 1000 + 2 * i + 2)
        bdir = run_dir / f"batch_{i:02d}"
        bdir.mkdir()
        macro = make_macro(args.physics, seeds, args.events, args.energy_mev, args.extra)
        (bdir / "run.mac").write_text(macro)
        t0 = time.time()
        proc = subprocess.run([args.exe, "run.mac", str(args.threads)], cwd=bdir,
                              capture_output=True, text=True)
        wall = time.time() - t0
        (bdir / "stdout.log").write_text(proc.stdout)
        if proc.stderr:
            (bdir / "stderr.log").write_text(proc.stderr)
        res = parse(proc.stdout)
        res.update({"batch": i, "seeds": list(seeds), "exit_code": proc.returncode, "wall_s": round(wall, 1)})
        results.append(res)
        manifest["batches"].append(res)
        write_yaml(run_dir / "manifest.yaml", manifest)
        print(f"batch {i}: ratio={res.get('ratio')} +- {res.get('ratio_err')} wall={wall:.0f}s", flush=True)

    manifest["finished_utc"] = utc_now()
    manifest["wall_time_s"] = round(time.time() - t_start, 1)

    ok = [r for r in results if r["exit_code"] == 0 and "ratio" in r]
    if ok:
        r = np.array([x["ratio"] for x in ok])
        e = np.array([x["ratio_err"] for x in ok])
        n = len(r)
        mean = float(r.mean())
        sem_batch = float(r.std(ddof=1) / np.sqrt(n)) if n > 1 else None
        sem_internal = float(np.sqrt(np.sum(e**2)) / n)
        manifest["summary"] = {
            "n_batches_ok": n,
            "ratio_mean": round(mean, 6),
            "ratio_sem_from_batch_spread": round(sem_batch, 6) if sem_batch else None,
            "ratio_sem_from_example_estimate": round(sem_internal, 6),
            "deviation_from_1_percent": round(100 * (mean - 1), 4),
            "deviation_in_sigma_batch": round((mean - 1) / sem_batch, 2) if sem_batch else None,
            "mean_secondary_energy_keV_mean": round(float(np.mean([x["mean_secondary_energy_keV"] for x in ok])), 4),
            "mass_energy_transfer_cm2_g_mean": round(float(np.mean([x["mass_energy_transfer_cm2_g"] for x in ok])), 5),
            "g4_water_imean_eV": ok[0].get("g4_water_imean_eV"),
        }
    write_yaml(run_dir / "manifest.yaml", manifest)
    print(manifest.get("summary"))


if __name__ == "__main__":
    main()
