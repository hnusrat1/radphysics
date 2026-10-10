"""Photon mass attenuation coefficients of water from the XCOM elemental cross sections
distributed with EGSnrc (HEN_HOUSE/data/xcom_*.data).

Each file lists, for Z = 1 to 100, the number of points followed by pairs
(ln E [MeV], ln sigma [barn/atom]). Water is the mass-weighted sum of H and O
with the G4_WATER composition and atomic weights (co60app table mode), which
is how XCOM itself treats mixtures. Interpolation is linear in log-log.
Provenance of these files (XCOM, Berger and Hubbell) is per the EGSnrc manual
and is flagged unverified in refs/refs.yaml.
"""

import json
from pathlib import Path

import numpy as np

from analysis.refs import ref_value

XCOM_DIR = Path("/opt/EGSnrc/HEN_HOUSE/data")
PROCESSES = {"compt": "xcom_compton.data", "phot": "xcom_photo.data", "pair": "xcom_pair.data",
             "triplet": "xcom_triplet.data", "Rayl": "xcom_rayleigh.data"}


def read_xcom(path):
    tok = Path(path).read_text().split()
    out, i = {}, 0
    for z in range(1, 101):
        n = int(tok[i])
        vals = np.array(tok[i + 1:i + 1 + 2 * n], dtype=float).reshape(n, 2)
        out[z] = vals
        i += 1 + 2 * n
    return out


def sigma(table, z, e):
    lne, lns = table[z][:, 0], table[z][:, 1]
    return np.exp(np.interp(np.log(e), lne, lns))


def water_mu_rho(energies, tables_json):
    """mu/rho (cm2/g) of water by process at the given energies (MeV)."""
    mat = json.load(open(tables_json))["materials"]["G4_WATER"]
    na = ref_value("avogadro_constant")
    out = {}
    for proc, fname in PROCESSES.items():
        tab = read_xcom(XCOM_DIR / fname)
        mu = np.zeros_like(np.asarray(energies, dtype=float))
        for el in mat["elements"]:
            z = int(round(el["Z"]))
            mu = mu + el["mass_fraction"] * na / el["A_g_mol"] * sigma(tab, z, energies) * 1e-24
        out[proc] = mu
    out["total"] = sum(out[p] for p in PROCESSES)
    return out
