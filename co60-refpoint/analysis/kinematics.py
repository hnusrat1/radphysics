"""Two-body photon interaction kinematics. Energies in MeV, angles in radians."""

import numpy as np

from analysis.refs import ref_value

ELECTRON_REST_ENERGY = ref_value("electron_rest_energy")
PAIR_THRESHOLD = 2.0 * ELECTRON_REST_ENERGY
TRIPLET_THRESHOLD = 4.0 * ELECTRON_REST_ENERGY


def compton_scattered_energy(e_gamma, theta):
    """Scattered photon energy for a free electron at rest."""
    k = e_gamma / ELECTRON_REST_ENERGY
    return e_gamma / (1.0 + k * (1.0 - np.cos(theta)))


def compton_electron_energy(e_gamma, theta):
    """Recoil electron kinetic energy for photon scattering angle theta."""
    return e_gamma - compton_scattered_energy(e_gamma, theta)


def compton_edge(e_gamma):
    """Maximum recoil electron kinetic energy (photon backscattered)."""
    k = e_gamma / ELECTRON_REST_ENERGY
    return e_gamma * 2.0 * k / (1.0 + 2.0 * k)
