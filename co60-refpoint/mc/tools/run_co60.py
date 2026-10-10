"""Run co60app in independent seeded batches from a run specification.

A run spec (YAML, kept in mc/geant4/runspecs/) fixes everything that defines a
run: physics list, geometry, production thresholds, variance reduction, bins,
batches and histories. Each batch is a separate process with its own seeds.
Outputs go to mc/runs/<run_id>/batch_NN/ with a manifest at the run level.

Example:
  python mc/tools/run_co60.py mc/geant4/runspecs/M1-R0-option4.yaml --exe /opt/build/co60app/co60app
  python mc/tools/run_co60.py <spec> --exe ... --resume     # continue after an interruption
"""

import argparse
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manifest import STUDY_ROOT, base_manifest, utc_now, write_yaml  # noqa: E402

BINS = STUDY_ROOT / "config" / "bins.yaml"


def bin_commands():
    b = yaml.safe_load(open(BINS))["mc_scoring"]
    pe, ee, d = b["photon_energy"], b["electron_energy"], b["depth"]
    return [
        f"/co60/bins/photonE {pe['min']} {pe['max']} {pe['n']} MeV",
        f"/co60/bins/electronE {ee['min']} {ee['max']} {ee['n']} MeV",
        f"/co60/bins/cosTheta {b['cos_theta']['n']}",
        f"/co60/bins/depth {d['min']} {d['max']} {d['n']} {d['radius']} cm",
    ]


def make_macro(spec, seeds, out_json, events_csv, verbose):
    if spec.get("table_mode", False):
        # Stopping powers and cross sections from the production physics (no transport).
        return "\n".join([
            "/control/verbose 1", "/run/verbose 1", f"/random/setSeeds {seeds[0]} {seeds[1]}",
            "/co60/geom/tableMode true", f"/co60/out/file {out_json}", "/run/initialize", "/run/beamOn 1",
        ]) + "\n"
    g = spec.get("geometry", {})
    cuts = spec.get("cuts", {})
    rr = spec.get("russian_roulette", {})
    src = spec.get("source", {})
    lines = [
        "/control/verbose 1",
        f"/run/verbose {1 if verbose else 0}",
        "/control/cout/ignoreThreadsExcept 0",
        f"/process/em/verbose {1 if verbose else 0}",
        f"/random/setSeeds {seeds[0]} {seeds[1]}",
        f"/co60/geom/scoringRadius {g.get('scoring_radius_cm', 1.0)} cm",
        f"/co60/geom/scoringThickness {g.get('scoring_thickness_cm', 0.2)} cm",
        f"/co60/geom/scoringDepth {g.get('scoring_depth_cm', 5.0)} cm",
        f"/co60/source/ssd {src.get('ssd_cm', 100.0)} cm",
        f"/co60/source/field {src.get('field_cm', [10.0, 10.0])[0]} {src.get('field_cm', [10.0, 10.0])[1]} cm",
        f"/co60/phys/electronThreshold {cuts.get('electron_threshold_keV', 10)} keV",
        f"/co60/phys/gammaThreshold {cuts.get('gamma_threshold_keV', 10)} keV",
        f"/co60/vr/russianRoulette {'true' if rr.get('enabled', False) else 'false'}",
        f"/co60/vr/survival {rr.get('survival', 0.05)}",
    ]
    if "roi_cm" in rr:
        r, z0, z1 = rr["roi_cm"]
        lines.append(f"/co60/vr/roi {r} {z0} {z1} cm")
    for cell in spec.get("importance_cells_cm", []):
        lines.append(f"/co60/vr/importanceCell {cell[0]} {cell[1]} {cell[2]} cm {cell[3]}")
    lines += bin_commands()
    lines.append(f"/co60/score/depthScan {'true' if spec.get('depth_scan', False) else 'false'}")
    lines.append(f"/co60/out/file {out_json}")
    if spec.get("event_sample_per_batch", 0) > 0:
        lines.append(f"/co60/out/eventSample {events_csv} {spec['event_sample_per_batch']}")
    lines.append("/run/initialize")
    if verbose:
        lines.append("/process/em/printParameters")
    if spec.get("importance_cells_cm"):
        lines.append("/co60/vr/applyImportance")
    lines.append(f"/run/printProgress {max(spec['histories_per_batch'] // 10, 1)}")
    lines.append(f"/run/beamOn {spec['histories_per_batch']}")
    return "\n".join(lines) + "\n"


def parse_log(log):
    """Physics configuration as printed by Geant4: EM models by process and the cut table."""
    out = {}
    models = {}
    for block in re.split(r"\n\s*\n", log):
        head = re.match(r"\s*(\w+):\s+for (\S+)", block)
        if not head:
            continue
        proc, particle = head.groups()
        if particle not in ("gamma", "e-", "e+"):
            continue
        found = re.findall(r"^\s*(\S+) : Emin=\s*([\d.]+ \S+)\s+Emax=\s*([\d.]+ \S+)", block, re.M)
        if found:
            models[f"{particle} {proc}"] = [f"{name} ({lo} - {hi})" for name, lo, hi in found]
    if models:
        out["em_models"] = models
    m = re.search(r"Material : G4_WATER\s*\n\s*Range cuts\s*:(.*)\n\s*Energy thresholds\s*:(.*)\n", log)
    if m:
        out["g4_water_range_cuts"] = " ".join(m.group(1).split())
        out["g4_water_energy_thresholds"] = " ".join(m.group(2).split())
    m = re.search(r"ROI check passed: (.*)", log)
    if m:
        out["roi_check"] = m.group(1).strip()
    m = re.search(r"Material:\s+G4_WATER\b[^\n]*\n[^\n]*?Imean:\s+([\d.]+) eV", log)
    if m:
        out["g4_water_imean_eV"] = float(m.group(1))
    return out


def quick_look(out_json):
    d = json.load(open(out_json))
    s, n = d["scalars"], d["histories"]
    m = s["edep"] / n
    var = s["edep_sq"] / n - m * m
    dose_gy = m * 1.602176634e-13 / (d["geometry"]["scoring_mass_g"] * 1e-3)
    return {
        "histories": n,
        "wall_s": round(d["wall_time_s"], 1),
        "D_over_phi0_Gy_cm2": dose_gy / d["source"]["phi0_per_history_cm-2"],
        "D_rel_sem_history_by_history": math.sqrt(var / n) / m if m > 0 else None,
        "energy_balance_max_abs_MeV": d["energy_balance_max_abs_MeV"],
        "untagged_tracks": s["untagged_tracks"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--exe", required=True)
    ap.add_argument("--resume", action="store_true", help="run only batches without an output file")
    args = ap.parse_args()

    spec_path = Path(args.spec).resolve()
    spec = yaml.safe_load(open(spec_path))
    run_dir = STUDY_ROOT / "mc" / "runs" / spec["run_id"]
    if run_dir.exists() and not args.resume:
        sys.exit(f"{run_dir} exists; runs are never overwritten (STUDY_PLAN Section 10.4). Use --resume.")
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = run_dir / "manifest.yaml"
    if manifest_path.exists():
        manifest = yaml.safe_load(open(manifest_path))
        manifest.setdefault("resumed_utc", []).append(utc_now())
    else:
        manifest = base_manifest(spec["run_id"], spec["purpose"])
        manifest["spec_file"] = str(spec_path.relative_to(STUDY_ROOT))
        manifest["spec"] = spec
        manifest["bins_file"] = str(BINS.relative_to(STUDY_ROOT))
        manifest["executable"] = args.exe
        sidecar = Path(args.exe + ".commit")
        if sidecar.exists():
            # Frozen production binary: the source commit it was built from, which
            # can differ from HEAD at run time.
            manifest["executable_source_commit"] = sidecar.read_text().strip()
        manifest["batches"] = []
        manifest["started_utc"] = utc_now()
    done = {b["batch"] for b in manifest["batches"]}

    for i in range(spec["batches"]):
        bdir = run_dir / f"batch_{i:02d}"
        if i in done and (bdir / "output.json").exists():
            continue
        bdir.mkdir(exist_ok=True)
        seeds = (spec["seed_base"] * 1000 + 2 * i + 1, spec["seed_base"] * 1000 + 2 * i + 2)
        macro = make_macro(spec, seeds, "output.json", "events.csv", verbose=(i == 0))
        (bdir / "run.mac").write_text(macro)
        cmd = [args.exe, "run.mac", spec["physics"], str(spec.get("threads", 4))]
        if spec.get("importance_cells_cm"):
            cmd.append("--importance")
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=bdir, capture_output=True, text=True)
        wall = time.time() - t0
        (bdir / "stdout.log").write_text(proc.stdout)
        if proc.stderr:
            (bdir / "stderr.log").write_text(proc.stderr)
        rec = {"batch": i, "seeds": list(seeds), "exit_code": proc.returncode, "wall_s": round(wall, 1),
               "command": " ".join(cmd)}
        if i == 0:
            manifest["geant4_physics_printout"] = parse_log(proc.stdout)
        if proc.returncode == 0 and (bdir / "output.json").exists() and not spec.get("table_mode", False):
            rec.update(quick_look(bdir / "output.json"))
        manifest["batches"] = [b for b in manifest["batches"] if b["batch"] != i] + [rec]
        manifest["batches"].sort(key=lambda b: b["batch"])
        write_yaml(manifest_path, manifest)
        print(f"batch {i}: exit {proc.returncode}, {wall:.0f} s, D/phi0 {rec.get('D_over_phi0_Gy_cm2')}", flush=True)

    ok = [b for b in manifest["batches"]
          if b.get("exit_code") == 0 and ("D_over_phi0_Gy_cm2" in b or spec.get("table_mode", False))]
    if len(ok) == spec["batches"]:
        manifest["finished_utc"] = utc_now()
        manifest["wall_time_s"] = round(sum(b["wall_s"] for b in ok), 1)
    write_yaml(manifest_path, manifest)


if __name__ == "__main__":
    main()
