#!/usr/bin/env python3
"""Build the condition-A2 data files (STUDY_PLAN Section 5.2; DECISIONS D028 item 10).

physics.nist.gov is unreachable from the build environment, so the two files are reproductions computed from
the NIST data that EGSnrc v2026 distributes, by the methods XCOM and ESTAR use. They are not NIST downloads.
If Humza downloads the genuine XCOM and ESTAR tables, they replace these files (same names) and
ai/data/validate_tables.py checks them.

    python -m ai.data.build_tables

xcom_water.txt   XCOM elemental cross sections (HEN_HOUSE/data/xcom_*.data), mixed by mass fraction as XCOM
                 does for compounds; liquid water composition H 0.111894, O 0.888106 (ESTAR/ICRU 37 water).
estar_water.txt  collision stopping power: Bethe formula with I = 75 eV and ESTAR's density-effect correction
                 (HEN_HOUSE/pegs4/density_corrections/compounds/water_liquid.density, computed with ESTAR);
                 radiative stopping power: Seltzer-Berger scaled bremsstrahlung cross sections
                 (HEN_HOUSE/data/nist_brems.data, the BREMX tables ESTAR uses); CSDA range and radiation yield
                 by integration from 1 keV.
"""
from pathlib import Path

import numpy as np

from analysis.refs import ref_value
from analysis.stopping import ME, Medium
from analysis.xcom import XCOM_DIR, read_xcom, sigma

HERE = Path(__file__).resolve().parent
NA = ref_value("avogadro_constant")
WATER = {1: (0.111894, 1.00794), 8: (0.888106, 15.9994)}      # mass fraction, atomic weight (g/mol)
Z_OVER_A = sum(w * z / a for z, (w, a) in WATER.items())

XCOM_E = [1e-3, 1.5e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 8e-3, 1e-2, 1.5e-2, 2e-2, 3e-2, 4e-2, 5e-2, 6e-2, 8e-2,
          0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.022, 1.25, 1.5, 2.0, 2.044, 3.0, 4.0, 5.0, 6.0, 7.0,
          8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 18.0, 20.0]
ESTAR_E = [0.01, 0.0125, 0.015, 0.0175, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.07, 0.08,
           0.09, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0,
           1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 9.0, 10.0, 12.5, 15.0,
           17.5, 20.0]


def xcom_water(energies):
    e = np.asarray(energies, dtype=float)
    out = {}
    for key, fname in (("coherent", "xcom_rayleigh.data"), ("incoherent", "xcom_compton.data"),
                       ("photoelectric", "xcom_photo.data"), ("pair_nuclear", "xcom_pair.data"),
                       ("pair_electron", "xcom_triplet.data")):
        tab = read_xcom(XCOM_DIR / fname)
        mu = np.zeros_like(e)
        for z, (w, a) in WATER.items():
            s = sigma(tab, z, e)
            if key.startswith("pair"):
                thr = 2 * ME if key == "pair_nuclear" else 4 * ME
                s = np.where(e > thr, s, 0.0)
            mu += w * NA / a * s * 1e-24
        out[key] = mu
    out["total_with_coherent"] = sum(out[k] for k in ("coherent", "incoherent", "photoelectric", "pair_nuclear",
                                                      "pair_electron"))
    out["total_without_coherent"] = out["total_with_coherent"] - out["coherent"]
    return out


def read_breme(path="/opt/EGSnrc/HEN_HOUSE/data/nist_brems.data"):
    """Seltzer-Berger tables: 57 electron kinetic energies (MeV), 30 kappa = k/T, then per Z (1-100) the
    scaled cross section chi = (beta^2 / Z^2) k dsigma/dk in mb, stored kappa-major (for each kappa, the 57
    energies); returned as an array [energy, kappa]."""
    tok = Path(path).read_text().split()
    i = tok.index("BREME.DAT:") if "BREME.DAT:" in tok else 0
    # header line ends with a date; the counts follow
    j = next(k for k in range(len(tok)) if tok[k] == "57" and tok[k + 1] == "30")
    t = np.array(tok[j + 2:j + 2 + 57], float)
    kap = np.array(tok[j + 2 + 57:j + 2 + 57 + 30], float)
    pos = j + 2 + 57 + 30
    chi = {}
    for z in range(1, 101):
        while not _isnum(tok[pos]):
            pos += 1
        chi[z] = np.array(tok[pos:pos + 57 * 30], float).reshape(30, 57).T
        pos += 57 * 30
    del i
    return t, kap, chi


def _isnum(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def radiative_stopping(energies):
    t_tab, kap, chi = read_breme()
    e = np.asarray(energies, dtype=float)
    out = np.zeros_like(e)
    for z, (w, a) in WATER.items():
        phi = np.trapezoid(chi[z], kap, axis=1)                 # integral of chi over kappa, mb
        phi_e = np.exp(np.interp(np.log(e), np.log(t_tab), np.log(phi)))
        tau = e / ME
        beta2 = tau * (tau + 2) / (tau + 1) ** 2
        out += w * NA / a * z * z / beta2 * e * phi_e * 1e-27     # MeV cm2/g
    return out


def estar_water(energies):
    water = Medium("water", "water_liquid.density", Z_OVER_A)
    e = np.asarray(energies, dtype=float)
    col = water.collision(e)
    rad = radiative_stopping(e)
    # CSDA range and radiation yield on a fine log grid from 1 keV
    fine = np.geomspace(1e-3, e.max(), 4000)
    s_fine = water.collision(fine) + radiative_stopping(fine)
    r0 = 1e-3 / s_fine[0] / 2            # range below 1 keV, rough (S rises as T falls); < 1e-6 g/cm2
    rng = r0 + np.concatenate([[0], np.cumsum(np.diff(fine) * 0.5 * (1 / s_fine[1:] + 1 / s_fine[:-1]))])
    yfrac = radiative_stopping(fine) / s_fine
    yint = np.concatenate([[0], np.cumsum(np.diff(fine) * 0.5 * (yfrac[1:] + yfrac[:-1]))])
    return {"collision": col, "radiative": rad, "total": col + rad,
            "csda_range": np.interp(np.log(e), np.log(fine), rng),
            "radiation_yield": np.interp(np.log(e), np.log(fine), yint) / e,
            "density_effect_delta": water.delta(e), "I_eV": water.I * 1e6}


def write():
    x = xcom_water(XCOM_E)
    cols = ["coherent", "incoherent", "photoelectric", "pair_nuclear", "pair_electron", "total_with_coherent",
            "total_without_coherent"]
    lines = [
        "# Photon interaction coefficients of liquid water (mass fractions H 0.111894, O 0.888106), cm^2/g.",
        "# Reproduction of an XCOM table: XCOM elemental cross sections as distributed with EGSnrc v2026",
        "# (HEN_HOUSE/data/xcom_*.data), mixed by mass fraction as XCOM does for compounds, log-log interpolation.",
        "# Not downloaded from NIST (physics.nist.gov was unreachable); see ai/data/README.md.",
        "# Columns: photon energy (MeV), " + ", ".join(cols),
    ]
    for i, e in enumerate(XCOM_E):
        lines.append(f"{e:.4E} " + " ".join(f"{x[c][i]:.4E}" for c in cols))
    (HERE / "xcom_water.txt").write_text("\n".join(lines) + "\n")

    s = estar_water(ESTAR_E)
    cols = ["collision", "radiative", "total", "csda_range", "radiation_yield", "density_effect_delta"]
    lines = [
        f"# Electron stopping powers of liquid water (I = {s['I_eV']:.1f} eV, density 1.000 g/cm^3).",
        "# Reproduction of an ESTAR table: collision stopping power from the Bethe formula with ESTAR's",
        "# density-effect correction (as distributed with EGSnrc v2026); radiative stopping power from the",
        "# Seltzer-Berger bremsstrahlung cross sections (EGSnrc HEN_HOUSE/data/nist_brems.data).",
        "# Not downloaded from NIST (physics.nist.gov was unreachable); see ai/data/README.md.",
        "# Columns: kinetic energy (MeV), collision stopping power (MeV cm^2/g), radiative stopping power",
        "# (MeV cm^2/g), total stopping power (MeV cm^2/g), CSDA range (g/cm^2), radiation yield, density-effect",
        "# parameter delta",
    ]
    for i, e in enumerate(ESTAR_E):
        lines.append(f"{e:.4E} " + " ".join(f"{s[c][i]:.4E}" for c in cols))
    (HERE / "estar_water.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    write()
    print("wrote ai/data/xcom_water.txt and ai/data/estar_water.txt")
