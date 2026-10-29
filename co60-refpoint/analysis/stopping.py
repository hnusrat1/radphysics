"""Electron collision stopping powers and the Spencer-Attix stopping-power ratio.

Collision stopping powers follow the Bethe theory with the Moller cross
section for energy transfers above the cut (the form used by ICRU Report 37 and
by EGSnrc/PEGS4):

  L_D/rho = k / beta^2 [ ln(2 (tau + 2) / (I/mc^2)^2) + F(tau, D) - delta ]
  F(tau, D) = -1 - beta^2 + ln((tau - D) D) + tau / (tau - D)
              + [D^2 / 2 + (2 tau + 1) ln(1 - D / tau)] / (tau + 1)^2

with tau = T/mc^2, D = cut/mc^2 (restricted) or tau/2 (unrestricted), and
k = 2 pi r_e^2 mc^2 N_A (Z/A). The density-effect correction delta(T) is read
from ESTAR-computed tables shipped with EGSnrc
(HEN_HOUSE/pegs4/density_corrections), which also give I and rho.

Four stopping-power bases are used (DECISIONS D018):
  icru37   water I = 75 eV (water_liquid.density), air I = 85.7 eV
  icru90   water I = 78 eV (water_icru90.density), air as icru37
  geant4   restricted and unrestricted dE/dx from G4EmCalculator with the
           production physics (co60app table mode)
  pegs4    EGSnrc PEGS4 restricted stopping powers (EDEDX, AE = 0.521 MeV) for
           water and air matched to Geant4 in I, density, composition and
           density effect (mc/egsnrc/b0, DECISIONS D015): what EGSnrc will use in B0
"""

import re

import json
from pathlib import Path

import numpy as np

from analysis.refs import ref_value

ME = ref_value("electron_rest_energy")             # MeV
RE = ref_value("classical_electron_radius_cm")     # cm
NA = ref_value("avogadro_constant")                # 1/mol
K0 = 2 * np.pi * RE**2 * ME * NA                   # MeV cm2 / mol

DENSITY_DIR = Path("/opt/EGSnrc/HEN_HOUSE/pegs4/density_corrections/compounds")


def read_density_file(path):
    """EGSnrc density-correction file: I (eV), rho (g/cm3), composition by mass, delta(T) table."""
    lines = Path(path).read_text().splitlines()
    head = lines[1].split()
    n, i_ev, rho, nel = int(head[0]), float(head[1]), float(head[2]), int(head[3])
    comp_tokens = " ".join(lines[2:2 + (nel + 3) // 4]).split()
    comp = {int(comp_tokens[2 * k]): float(comp_tokens[2 * k + 1]) for k in range(nel)}
    pairs = []
    for line in lines[2 + (nel + 3) // 4:]:
        for tok in line.split():
            e, d = tok.split(",")
            pairs.append((float(e), float(d)))
    pairs = np.array(pairs[:n])
    return {"I_eV": i_ev, "rho": rho, "composition": comp, "T": pairs[:, 0], "delta": pairs[:, 1]}


class Medium:
    """I and delta(T) from an EGSnrc density file; Z/A from the Geant4 material of the same composition."""

    def __init__(self, name, density_file, z_over_a):
        d = read_density_file(DENSITY_DIR / density_file)
        self.name = name
        self.source = density_file
        self.I = d["I_eV"] * 1e-6     # MeV
        self.rho = d["rho"]
        self.composition = d["composition"]
        self.z_over_a = z_over_a
        self._t, self._delta = d["T"], d["delta"]

    def delta(self, t):
        # delta is tabulated against kinetic energy; interpolate linearly in log T.
        return np.interp(np.log(t), np.log(self._t), self._delta)

    def collision(self, t, cut=None):
        """Restricted (cut in MeV) or unrestricted collision stopping power, MeV cm2/g."""
        t = np.asarray(t, dtype=float)
        tau = t / ME
        beta2 = tau * (tau + 2) / (tau + 1) ** 2
        dmax = tau / 2
        d = dmax if cut is None else np.minimum(cut / ME, dmax)
        f = (-1 - beta2 + np.log((tau - d) * d) + tau / (tau - d)
             + (d * d / 2 + (2 * tau + 1) * np.log(1 - d / tau)) / (tau + 1) ** 2)
        bracket = np.log(2 * (tau + 2) / (self.I / ME) ** 2) + f - self.delta(t)
        return K0 * self.z_over_a / beta2 * bracket


def pegs4_ededx(listing):
    """EDEDX (restricted total stopping power, MeV cm2/g) against kinetic energy from a PEGS4 TEST listing."""
    lines = Path(listing).read_text().splitlines()
    i = next(k for k, line in enumerate(lines) if "PLOT OF FUNCTION EDEDX" in line)
    pts = []
    for line in lines[i + 1:]:
        m = re.match(r"\s*\d+\s+([\d.E+-]+)\s+([\d.E+-]+)\s+I", line)
        if m:
            pts.append((float(m.group(1)), float(m.group(2))))
        elif pts and "PLOT OF FUNCTION" in line:
            break
    pts = np.array(pts)
    return lambda t: np.exp(np.interp(np.log(t), np.log(pts[:, 0]), np.log(pts[:, 1])))


def bases(tables_json, pegs4_dir=None):
    """Water and air stopping-power functions for each basis: name -> (water, air), each (L(T, cut), S(T))."""
    tab = json.load(open(tables_json))["materials"]
    out = {}
    z_a = {m: tab[m]["electron_density_per_cm3"] / tab[m]["density_g_cm3"] / NA for m in ("G4_WATER", "G4_AIR")}
    air = Medium("air", "air_dry_nearsealevel.density", z_a["G4_AIR"])
    for name, wfile in (("icru37", "water_liquid.density"), ("icru90", "water_icru90.density")):
        water = Medium("water", wfile, z_a["G4_WATER"])
        out[name] = {
            "water": (lambda t, cut, m=water: m.collision(t, cut), lambda t, m=water: m.collision(t)),
            "air": (lambda t, cut, m=air: m.collision(t, cut), lambda t, m=air: m.collision(t)),
            "I_water_eV": water.I * 1e6, "I_air_eV": air.I * 1e6,
            "source": f"Bethe-Moller formula; {wfile}, air_dry_nearsealevel.density (EGSnrc v2026)",
        }

    def g4(mat, key):
        rows = tab[mat]["electron"]
        e = np.array([r["E"] for r in rows])
        v = np.array([r[key] for r in rows])
        return lambda t: np.exp(np.interp(np.log(t), np.log(e), np.log(v)))

    w10, a10 = g4("G4_WATER", "L_10keV"), g4("G4_AIR", "L_10keV")
    w5, a5 = g4("G4_WATER", "L_5keV"), g4("G4_AIR", "L_5keV")
    ws, as_ = g4("G4_WATER", "S_col"), g4("G4_AIR", "S_col")

    def pick(f10, f5, fs):
        def restricted(t, cut):
            if cut is None:
                return fs(t)
            if abs(cut - 0.010) < 1e-12:
                return f10(t)
            if abs(cut - 0.005) < 1e-12:
                return f5(t)
            raise ValueError("Geant4 tables hold restricted stopping powers for 5 and 10 keV only")
        return restricted, fs

    out["geant4"] = {
        "water": pick(w10, w5, ws), "air": pick(a10, a5, as_),
        "I_water_eV": tab["G4_WATER"]["imean_eV"], "I_air_eV": tab["G4_AIR"]["imean_eV"],
        "source": "G4EmCalculator, " + json.load(open(tables_json))["physics"] + " (co60app table mode)",
    }
    if pegs4_dir is not None:
        w = pegs4_ededx(Path(pegs4_dir) / "water_g4_option4.pegs4lst")
        a = pegs4_ededx(Path(pegs4_dir) / "air_g4_option4.pegs4lst")

        def only10(f):
            def restricted(t, cut):
                if cut is not None and abs(cut - 0.010) > 1e-12:
                    raise ValueError("PEGS4 data were generated for a 10 keV cut only")
                return f(t)
            # Below 2 * Delta the restricted and unrestricted stopping powers coincide,
            # so EDEDX at Delta is the unrestricted value the track-end term needs.
            return restricted, f
        out["pegs4"] = {"water": only10(w), "air": only10(a), "I_water_eV": tab["G4_WATER"]["imean_eV"],
                        "I_air_eV": tab["G4_AIR"]["imean_eV"],
                        "source": "EGSnrc v2026 PEGS4 EDEDX, media matched to Geant4 (mc/egsnrc/b0)"}
    return out


def spencer_attix(fluence_1kev, delta_mev, water, air, width=0.001, phi_e_at_delta=None):
    """Spencer-Attix water-to-air stopping-power ratio with the Nahum track-end term.

    fluence_1kev: electron track length per 1 keV bin, bins [0,1), [1,2), ... keV
    (no under/overflow); `width` (MeV) allows another uniform bin width with Delta on a bin edge.
    water, air: (L(T, cut), S(T)) pairs in MeV cm2/g.
    phi_e_at_delta: the differential fluence at Delta (per MeV) when it cannot be read from the two bins around
    Delta, as for a code that stops electrons at Delta (analysis/m2.py, D039).
    Returns the ratio and the water and air terms (in fluence units times MeV cm2/g).
    """
    phi = np.asarray(fluence_1kev, dtype=float)
    k = int(round(delta_mev / width))
    assert abs(k * width - delta_mev) < 1e-9, "Delta must fall on a bin edge"
    e_mid = (np.arange(phi.size) + 0.5) * width
    above = slice(k, phi.size)
    if phi_e_at_delta is None:
        phi_e_at_delta = 0.5 * (phi[k - 1] + phi[k]) / width   # differential fluence at Delta, per MeV

    def terms(medium):
        restricted, unrestricted = medium
        integral = np.sum(phi[above] * restricted(e_mid[above], delta_mev))
        track_end = phi_e_at_delta * unrestricted(np.array([delta_mev]))[0] * delta_mev
        return integral, track_end

    wi, wt = terms(water)
    ai, at = terms(air)
    return {"s_w_air": (wi + wt) / (ai + at), "water_integral": wi, "water_track_end": wt,
            "air_integral": ai, "air_track_end": at}
