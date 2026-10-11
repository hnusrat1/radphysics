#!/usr/bin/env python3
"""Run an EGSnrc RZ user code (DOSRZnrc, FLURZnrc, SPRRZnrc) for geometry B0 as independent parallel jobs.

    python mc/tools/run_egsnrc.py mc/egsnrc/b0/specs/M2-B0-egsnrc-dose.yaml [--resume] [--deadline-min 105]

Each job is one statistical batch with its own seeds. Inputs are generated from the spec (geometry B0,
STUDY_PLAN Section 4.2; transport block mc/egsnrc/b0/inputs/b0_transport.inc), written into
$EGS_HOME/<code>/ (EGSnrc reads inputs there) and copied with all outputs into mc/runs/<run_id>/job_XX/.
Runs are never overwritten (STUDY_PLAN Section 10.4).
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manifest import STUDY_ROOT, base_manifest, utc_now, write_yaml  # noqa: E402

B0 = STUDY_ROOT / "mc" / "egsnrc" / "b0"
TRANSPORT = (B0 / "inputs" / "b0_transport.inc").read_text()


def geometry_block(spec):
    g = spec["geometry"]
    return f"""
 :start geometrical inputs:
 METHOD OF INPUT= individual
 Z OF FRONT FACE= 0.
 DEPTH BOUNDARIES= {", ".join(str(z) for z in g["planes_cm"][1:])}
 RADII= {", ".join(str(r) for r in g["radii_cm"])}
 MEDIA= {", ".join(spec["media"])}
 DESCRIPTION BY= regions
 MEDNUM= 1
 START REGION= 2
 STOP REGION= {1 + (len(g["planes_cm"]) - 1) * len(g["radii_cm"])}
 :stop geometrical inputs:
"""


def source_block(spec):
    s = spec["source"]
    spectrum = B0 / "inputs" / s["spectrum_file"]
    return f"""
 :start source inputs:
 INCIDENT PARTICLE= photon
 SOURCE NUMBER= 1
 SOURCE OPTIONS= {s["ssd_cm"]}, {s["field_radius_cm"]}, 0, 0
 INCIDENT ENERGY= spectrum
 SPEC FILENAME= {spectrum}
 SPEC IOUTSP= include
 :stop source inputs:
"""


def vr_block(spec):
    v = spec.get("variance_reduction", {})
    rr = v.get("electron_range_rejection", False)
    lines = [" :Start Variance Reduction:",
             f" ELECTRON RANGE REJECTION= {'On' if rr else 'Off'}",
             f" ESAVEIN= {v.get('esave_MeV', 0.0)}",
             " RUSSIAN ROULETTE DEPTH= 0.0",
             " RUSSIAN ROULETTE FRACTION= 0.0",
             " EXPONENTIAL TRANSFORM C= 0.0",
             " PHOTON FORCING= Off",
             " START FORCING= 1",
             " STOP FORCING AFTER= 1"]
    if spec["code"] == "dosrznrc":
        lines += [" BREM SPLITTING= Off", " NUMBER OF BREMS PER EVENT= 1",
                  " CHARGED PARTICLE RUSSIAN ROULETTE= Off",
                  " CS ENHANCEMENT FACTOR= 1.0", " CS ENHANCEMENT START REGION= 1",
                  " CS ENHANCEMENT STOP REGION= 1"]
    if spec["code"] == "flurznrc":
        lines += [" BREM SPLITTING= Off", " NUMBER OF BREMS PER EVENT= 1",
                  " CHARGED PARTICLE RUSSIAN ROULETTE= Off"]
    lines.append(" :Stop Variance Reduction:")
    return "\n" + "\n".join(lines) + "\n"


def io_block(spec):
    code = spec["code"]
    nz, nr = len(spec["geometry"]["planes_cm"]) - 1, len(spec["geometry"]["radii_cm"])
    if code == "dosrznrc":
        body = f""" DOSE ZBOUND MIN= 1
 DOSE ZBOUND MAX= {nz + 1}
 DOSE RBOUND MIN= 0
 DOSE RBOUND MAX= {nr}
 IWATCH= off
 STORE INITIAL RANDOM NUMBERS= no
 IRESUME= first
 OUTPUT OPTIONS= material and dose summary
 STORE DATA ARRAYS= yes
 ELECTRON TRANSPORT= normal"""
    elif code == "flurznrc":
        reg = spec["scoring_region"]
        body = f""" IWATCH= off
 STORE INITIAL RANDOM NUMBERS= no
 IRESUME= first
 STORE DATA ARRAYS= yes
 PRINT FLUENCE SPECTRA= specified
 LIST FLUENCE START REGION= {reg}
 LIST FLUENCE STOP REGION= {reg}
 IPRIMARY= photon primaries
 SLOTE= {spec["bin_width_MeV"]}"""
    else:  # sprrznrc
        body = """ IWATCH= off
 STORE INITIAL RANDOM NUMBERS= no
 IRESUME= first
 STORE DATA ARRAYS= yes
 SPR OUTPUT= regions"""
    return f"\n :start I/O control:\n{body}\n :stop I/O control:\n"


def mc_block(spec, seeds):
    extra = ""
    if spec["code"] == "dosrznrc":
        extra = """ IFULL= dose and stoppers
 STATISTICAL ACCURACY SOUGHT= 0.0
 SCORE KERMA= yes
"""
    if spec["code"] == "sprrznrc":
        extra = " PHOTON REGENERATION= no\n"
    return f"""
 :start Monte Carlo inputs:
 NUMBER OF HISTORIES= {spec["histories_per_job"]}
 INITIAL RANDOM NO. SEEDS= {seeds[0]}, {seeds[1]}
 MAX CPU HOURS ALLOWED= 100.0
{extra} :stop Monte Carlo inputs:
"""


def plot_block(spec):
    if spec["code"] == "flurznrc":
        return "\n :start plot control:\n PLOTTING= Off\n :stop plot control:\n"
    if spec["code"] == "dosrznrc":
        return "\n :start plot control:\n PLOTTING= Off\n :stop plot control:\n"
    return "\n :start plot control:\n PLOTTING= Off\n :stop plot control:\n"


def make_input(spec, seeds):
    return (f" TITLE= {spec['run_id']}: {spec['code']} in geometry B0 (co60-refpoint)\n"
            + io_block(spec) + mc_block(spec, seeds) + geometry_block(spec) + source_block(spec)
            + "\n" + TRANSPORT + vr_block(spec) + plot_block(spec))


def run_job(spec, run_dir, j, pegs):
    code = spec["code"]
    egs_home = Path(os.environ["EGS_HOME"])
    name = f"{spec['run_id']}_j{j:02d}"
    seeds = (spec["seed_base"] + j, 97 + j)   # ranmar: both in [1, 30081]
    jdir = run_dir / f"job_{j:02d}"
    if jdir.exists() and any(jdir.glob("*.egslst")):
        return None
    if jdir.exists():
        stamp = utc_now().replace(":", "").replace("-", "")
        jdir.rename(run_dir / f"job_{j:02d}.interrupted-{stamp}")
    jdir.mkdir()
    inp = make_input(spec, seeds)
    (jdir / f"{name}.egsinp").write_text(inp)
    (egs_home / code / f"{name}.egsinp").write_text(inp)
    exe = egs_home / "bin" / os.environ.get("my_machine", "linux") / code
    t0 = time.time()
    proc = subprocess.run([str(exe), "-i", name, "-p", pegs], cwd=egs_home / code,
                          capture_output=True, text=True)
    wall = time.time() - t0
    (jdir / "stdout.log").write_text(proc.stdout)
    if proc.stderr:
        (jdir / "stderr.log").write_text(proc.stderr)
    for ext in (".egslst", ".egsdat", ".egslog", ".plotdat", ".errors", ".egsrns"):
        f = egs_home / code / f"{name}{ext}"
        if f.exists():
            shutil.move(str(f), jdir / f.name)
    (egs_home / code / f"{name}.egsinp").unlink(missing_ok=True)
    return {"job": j, "seeds": list(seeds), "exit_code": proc.returncode, "wall_s": round(wall, 1),
            "command": f"{code} -i {name} -p {pegs}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--deadline-min", type=float, default=None,
                    help="do not start a round of jobs that would end after this many minutes")
    args = ap.parse_args()
    spec_path = Path(args.spec).resolve()
    spec = yaml.safe_load(open(spec_path))
    run_dir = STUDY_ROOT / "mc" / "runs" / spec["run_id"]
    if run_dir.exists() and not args.resume:
        sys.exit(f"{run_dir} exists; runs are never overwritten. Use --resume.")
    run_dir.mkdir(parents=True, exist_ok=True)
    mpath = run_dir / "manifest.yaml"
    if mpath.exists():
        manifest = yaml.safe_load(open(mpath))
        manifest.setdefault("resumed_utc", []).append(utc_now())
    else:
        manifest = base_manifest(spec["run_id"], spec["purpose"], code="egsnrc")
        manifest["spec_file"] = str(spec_path.relative_to(STUDY_ROOT))
        manifest["spec"] = spec
        manifest["egsnrc_run"] = {"hen_house": os.environ.get("HEN_HOUSE"), "egs_home": os.environ.get("EGS_HOME"),
                              "config": os.environ.get("EGS_CONFIG"), "pegs4_file": spec["pegs4_file"],
                              "transport_block": "mc/egsnrc/b0/inputs/b0_transport.inc"}
        manifest["jobs"] = []
        manifest["started_utc"] = utc_now()
    t_start = time.time()
    todo = [j for j in range(spec["jobs"]) if not (run_dir / f"job_{j:02d}").exists()
            or not any((run_dir / f"job_{j:02d}").glob("*.egslst"))]
    par = spec.get("parallel", 4)
    while todo:
        walls = [b["wall_s"] for b in manifest["jobs"] if b.get("exit_code") == 0]
        if args.deadline_min is not None and walls:
            if time.time() - t_start + 1.15 * max(walls) > 60 * args.deadline_min:
                print(f"PAUSED with {len(todo)} jobs left; resume with --resume", flush=True)
                break
        round_jobs, todo = todo[:par], todo[par:]
        with ThreadPoolExecutor(max_workers=par) as ex:
            recs = list(ex.map(lambda j: run_job(spec, run_dir, j, spec["pegs4_file"]), round_jobs))
        for rec in recs:
            if rec is None:
                continue
            manifest["jobs"] = [b for b in manifest["jobs"] if b["job"] != rec["job"]] + [rec]
            print(f"job {rec['job']}: exit {rec['exit_code']}, {rec['wall_s']:.0f} s", flush=True)
        manifest["jobs"].sort(key=lambda b: b["job"])
        write_yaml(mpath, manifest)
    ok = [b for b in manifest["jobs"] if b.get("exit_code") == 0]
    if len(ok) == spec["jobs"]:
        manifest["finished_utc"] = utc_now()
        manifest["cpu_time_s"] = round(sum(b["wall_s"] for b in ok), 1)
    write_yaml(mpath, manifest)


if __name__ == "__main__":
    main()
