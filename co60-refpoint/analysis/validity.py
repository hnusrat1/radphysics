"""Per-event physical validity checks for photon interaction events (STUDY_PLAN Section 6, DECISIONS D014).

The same checks apply to Monte Carlo and AI events. An event has: process,
incident energy e_in, scattered photon energy e_out (Compton, Rayleigh) and
scattering angle cosine cos_scatter, electron kinetic energy t_e, and for pair
production the positron kinetic energy t_pos. Energies in MeV.

Two rule sets:
  strict  free-electron kinematics, as written in the plan
  bound   allows the electron binding energy in the energy balance and a
          Doppler tolerance delta_c on the Compton kinematics, with delta_c
          taken from the Monte Carlo event sample and frozen before AI scoring
"""

import numpy as np

from analysis.kinematics import ELECTRON_REST_ENERGY as ME
from analysis.kinematics import PAIR_THRESHOLD, compton_edge, compton_scattered_energy
from analysis.refs import ref_value

# Energy balance tolerance for the bound rules: the oxygen K-shell binding energy
# (largest binding energy in water) plus a margin for nuclear recoil in pair
# production, rounded up to 1 keV.
BINDING_TOLERANCE = 1e-3
assert BINDING_TOLERANCE > ref_value("oxygen_k_binding_geant4_eV") * 1e-6

# Numerical floor for every energy comparison (1 eV): covers 10-digit CSV output
# and the difference between the CLHEP and CODATA 2018 electron masses (8e-8 MeV).
NUMERIC_TOL = 1e-6


def _arr(x):
    return np.asarray(x, dtype=float)


def check_events(process, e_in, e_out, cos_scatter, t_e, t_pos=None, delta_c=0.0, energy_tol=0.0):
    """Return a dict of boolean arrays: one per check, plus 'valid' (all applicable checks pass).

    e_out < 0 or NaN means no outgoing photon. delta_c and energy_tol are 0 for
    the strict rules.
    """
    process = np.asarray(process)
    e_in, e_out, cos_s, t_e = _arr(e_in), _arr(e_out), _arr(cos_scatter), _arr(t_e)
    t_pos = np.zeros_like(e_in) if t_pos is None else np.where(_arr(t_pos) > 0, _arr(t_pos), 0.0)
    e_out_c = np.where(np.isfinite(e_out) & (e_out > 0), e_out, 0.0)
    compt, pair, phot, rayl = (process == "compt"), (process == "conv"), (process == "phot"), (process == "Rayl")

    # 1. Energy conservation (rest mass of the pair included)
    balance = e_in - e_out_c - t_e - t_pos - np.where(pair, 2 * ME, 0.0)
    tol = energy_tol + NUMERIC_TOL
    energy_ok = np.where(rayl, np.abs(e_in - e_out_c) <= tol, np.abs(balance) <= tol)

    # 2. Compton: scattered energy consistent with the scattering angle
    expected_out = compton_scattered_energy(e_in, np.arccos(np.clip(cos_s, -1, 1)))
    angle_ok = np.where(compt, np.abs(e_out_c - expected_out) <= delta_c + NUMERIC_TOL, True)

    # 3. Compton: recoil electron below the Compton edge
    edge_ok = np.where(compt, t_e <= compton_edge(e_in) + delta_c + NUMERIC_TOL, True)

    # 4. Pair production only above threshold
    threshold_ok = np.where(pair, e_in >= PAIR_THRESHOLD, True)

    # 5. Photoelectric: no outgoing photon of the incident energy
    phot_ok = np.where(phot, e_out_c == 0.0, True)

    valid = energy_ok & angle_ok & edge_ok & threshold_ok & phot_ok
    return {"energy": energy_ok, "compton_angle": angle_ok, "compton_edge": edge_ok,
            "pair_threshold": threshold_ok, "photoelectric": phot_ok, "valid": valid}


def doppler_tolerance(process, e_in, e_out, cos_scatter, t_e, step=0.010, percentile=99.9):
    """delta_c from a Monte Carlo sample (DECISIONS D014).

    For each Compton event the departure from free-electron kinematics is the larger of
    |E_out - E_KN(theta)| and the recoil energy above the Compton edge. delta_c is the
    given percentile of that departure, rounded up to the next `step`.
    """
    process = np.asarray(process)
    c = process == "compt"
    e_in, e_out, cos_s, t_e = _arr(e_in)[c], _arr(e_out)[c], _arr(cos_scatter)[c], _arr(t_e)[c]
    alive = np.isfinite(e_out) & (e_out > 0)
    angle_dev = np.zeros_like(e_in)
    angle_dev[alive] = np.abs(e_out[alive] - compton_scattered_energy(e_in[alive],
                                                                       np.arccos(np.clip(cos_s[alive], -1, 1))))
    edge_excess = t_e - compton_edge(e_in)
    departure = np.maximum(angle_dev, edge_excess)
    q = float(np.percentile(departure, percentile)) if departure.size else 0.0
    return {
        "delta_c": float(np.ceil(q / step) * step) if q > 0 else step,
        "percentile": percentile,
        "departure_percentiles_keV": {str(p): float(1000 * np.percentile(departure, p)) for p in (50, 90, 99, 99.9, 100)}
        if departure.size else {},
        "max_edge_excess_keV": float(1000 * edge_excess.max(initial=0.0)),
        "fraction_above_edge": float(np.mean(edge_excess > 0)) if edge_excess.size else 0.0,
        "n_compton": int(c.sum()),
    }
