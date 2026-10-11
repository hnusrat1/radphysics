#!/usr/bin/env python3
"""Check the condition-A2 tables (the reproductions built here, or genuine NIST downloads that replace them).

    python -m ai.data.validate_tables

Checks, none of which uses a remembered value:
  internal   total = sum of parts; CSDA range increasing; radiation yield between 0 and 1
  Geant4     against the option4 tables of the production physics (M1-tables-option4-v2), which end at
             1.4 MeV: photon mu/rho (incoherent and total) and collision and radiative stopping powers.
             Tolerances: photons 1% (2% for the total below 30 keV, where the photoelectric data differ); collision 1% above 0.5 MeV (I = 75 eV here, 78 eV in G4_WATER) and 4%
             below (option4 uses the Penelope ionisation model there, STATUS M1); radiative 3% (both from
             Seltzer-Berger tables).
Prints a table and exits non-zero if a check fails.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "ai" / "data"
G4 = ROOT / "mc" / "runs" / "M1-tables-option4-v2" / "batch_00" / "output.json"


def load(name):
    rows = [list(map(float, ln.split())) for ln in (HERE / name).read_text().splitlines()
            if ln.strip() and not ln.startswith("#")]
    return np.array(rows)


def g4_tables():
    m = json.load(open(G4))["materials"]["G4_WATER"]
    el = {k: np.array([r[k] for r in m["electron"]]) for k in ("E", "S_col", "S_rad")}
    ph = {k: np.array([r[k] for r in m["photon"]]) for k in ("E", "compt", "phot", "conv", "Rayl")}
    return el, ph, m["density_g_cm3"]


def interp(x, xs, ys):
    return np.exp(np.interp(np.log(x), np.log(xs), np.log(np.maximum(ys, 1e-300))))


def main():
    fails = []
    x = load("xcom_water.txt")
    s = load("estar_water.txt")
    # internal consistency
    if np.max(np.abs(x[:, 1:6].sum(1) - x[:, 6]) / x[:, 6]) > 2e-4:
        fails.append("xcom: total_with_coherent != sum of parts")
    if np.max(np.abs(s[:, 1] + s[:, 2] - s[:, 3]) / s[:, 3]) > 2e-4:
        fails.append("estar: total != collision + radiative")
    if np.any(np.diff(s[:, 4]) <= 0):
        fails.append("estar: CSDA range not increasing")
    if np.any((s[:, 5] < 0) | (s[:, 5] > 1)):
        fails.append("estar: radiation yield outside [0, 1]")
    el, ph, rho = g4_tables()
    print("photon mu/rho, table / Geant4 option4 (G4EmCalculator, cm2/g):")
    for e in (0.01, 0.03, 0.1, 0.5, 1.0, 1.25):
        row = x[np.argmin(np.abs(x[:, 0] - e))]
        if abs(row[0] - e) > 1e-9 or e > ph["E"].max():   # the Geant4 tables end at 1.4 MeV
            continue
        g_inc = interp(e, ph["E"], ph["compt"]) / rho
        g_tot = sum(interp(e, ph["E"], ph[k]) for k in ("compt", "phot", "conv", "Rayl")) / rho
        r_inc, r_tot = row[2] / g_inc, row[6] / g_tot
        print(f"  {e:6.3f} MeV  incoherent {r_inc:.4f}  total {r_tot:.4f}")
        if abs(r_inc - 1) > 0.01 or abs(r_tot - 1) > (0.01 if e >= 0.03 else 0.02):
            fails.append(f"xcom vs Geant4 at {e} MeV: {r_inc:.4f}, {r_tot:.4f}")
    print("electron stopping powers, table / Geant4 option4 (MeV cm2/g):")
    for e in (0.01, 0.05, 0.1, 0.5, 1.0, 1.25):
        row = s[np.argmin(np.abs(s[:, 0] - e))]
        if abs(row[0] - e) > 1e-9 or e > el["E"].max():
            continue
        rc = row[1] / (interp(e, el["E"], el["S_col"]) / rho)
        rr = row[2] / (interp(e, el["E"], el["S_rad"]) / rho)
        print(f"  {e:6.2f} MeV  collision {rc:.4f}  radiative {rr:.4f}")
        if abs(rc - 1) > (0.01 if e >= 0.5 else 0.04) or abs(rr - 1) > 0.03:
            fails.append(f"estar vs Geant4 at {e} MeV: {rc:.4f}, {rr:.4f}")
    print("PASS" if not fails else "FAIL:\n  " + "\n  ".join(fails))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
