"""Build EGSnrc media matched to Geant4 G4_WATER and G4_AIR for the B0 benchmark (DECISIONS D015).

For each material this writes an EGSnrc density-correction file with Geant4's
mean excitation energy, density, composition by mass and density-effect
correction delta(T) (from co60app table mode, M1-tables-option4-v2), and a
PEGS4 input with AE = 0.521 MeV and AP = 0.010 MeV (10 keV kinetic thresholds).
PEGS4 then reads the density file (EPSTFL=1), so its Bethe stopping powers use
the same I and delta as Geant4's Moller-Bhabha model.

Usage (EGSnrc environment loaded):
  python mc/egsnrc/b0/make_matched_media.py
"""

import json
import os
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TABLES = ROOT / "mc" / "runs" / "M1-tables-option4-v2" / "batch_00" / "output.json"
EGS_GRID = Path(os.environ.get("HEN_HOUSE", "/opt/EGSnrc/HEN_HOUSE/")) / "pegs4/density_corrections/compounds/water_liquid.density"

MEDIA = {
    # Geant4 material -> (PEGS4 medium name, file stem, molecule label)
    "G4_WATER": ("H2O_G4", "water_g4_option4", "H2O"),
    "G4_AIR": ("AIR_G4", "air_g4_option4", "AIR"),
}
SYMBOL = {1: "H", 6: "C", 7: "N", 8: "O", 18: "AR"}


def egs_energy_grid():
    lines = EGS_GRID.read_text().splitlines()
    n = int(lines[1].split()[0])
    pairs = " ".join(lines[3:]).split()
    return np.array([float(p.split(",")[0]) for p in pairs[:n]])


def write_density(mat, stem, energies):
    d = mat["density_effect"]
    e = np.array([r["E"] for r in d])
    delta = np.array([r["delta"] for r in d])
    dg = np.interp(np.log(energies), np.log(e), delta)
    els = mat["elements"]
    lines = [f"  {stem}: Geant4 11.4.3 {mat['name']} (I, rho, composition, delta from co60app table mode)",
             f"  {len(energies):10d}  {mat['imean_eV']:10.6f}  {mat['density_g_cm3']:14.8E}  {len(els):10d}"]
    comp = [f"{int(round(x['Z'])):3d} {x['mass_fraction']:.6f}" for x in els]
    for k in range(0, len(comp), 4):
        lines.append(" ".join(comp[k:k + 4]))
    pairs = [f"{ei:.2E},{di:.3E}" for ei, di in zip(energies, dg)]
    for k in range(0, len(pairs), 4):
        lines.append(" " + "  ".join(pairs[k:k + 4]))
    path = HERE / f"{stem}.density"
    path.write_text("\n".join(lines) + "\n")
    return path


def write_input(mat, name, stem, label):
    els = mat["elements"]
    rhoz = ",".join(f"{x['mass_fraction']:.6f}" for x in els)
    syms = "".join(f"{SYMBOL[int(round(x['Z']))]:<3s}" for x in els)
    text = (
        "ENER\n"
        " &INP AE=0.521,UE=2.511,AP=0.010,UP=2.0 &END\n"
        "MIXT\n"
        f" &INP NE={len(els)},RHO={mat['density_g_cm3']:.8g},RHOZ={rhoz},EPSTFL=1,IAPRIM=1,IRAYL=1 &END\n"
        f"{name:<30s}{label}\n"
        f"{syms}\n"
        "TEST\n &INP &END\n"
        "PWLF\n &INP &END\n"
        "DECK\n &INP &END\n"
    )
    path = HERE / f"{stem}.pegs4inp"
    path.write_text(text)
    return path


def main():
    tables = json.load(open(TABLES))["materials"]
    energies = egs_energy_grid()
    pegs = Path(os.environ["HEN_HOUSE"]) / "bin" / "linux" / "pegs4.exe"
    for g4name, (name, stem, label) in MEDIA.items():
        mat = dict(tables[g4name], name=g4name)
        dens = write_density(mat, stem, energies)
        inp = write_input(mat, name, stem, label)
        egs_home = Path(os.environ["EGS_HOME"])
        for old in (egs_home / "pegs4" / "data").glob(f"{stem}.*"):
            old.unlink()   # PEGS4 does not overwrite an existing data file
        proc = subprocess.run([str(pegs), "-e", "-i", str(inp.with_suffix("")), "-o", stem, "-d", str(dens.with_suffix(""))],
                              cwd=HERE, capture_output=True, text=True)
        (HERE / f"{stem}.pegs4.stdout").write_text(proc.stdout + proc.stderr)
        # PEGS4 writes the data file to $EGS_HOME/pegs4/data and the listing next to its input copy;
        # keep both with the inputs.
        for src in (egs_home / "pegs4" / "data" / f"{stem}.pegs4dat",
                    egs_home / "pegs4" / "inputs" / f"{stem}.pegs4lst"):
            if src.exists():
                (HERE / src.name).write_bytes(src.read_bytes())
        print(g4name, "->", stem, "exit", proc.returncode)


if __name__ == "__main__":
    main()
