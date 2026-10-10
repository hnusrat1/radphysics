"""Log-log interpolation of tabulated photon attenuation coefficients (STUDY_PLAN Section 8 test)."""

import numpy as np


def loglog_interp(e, e_table, mu_table):
    """Interpolate mu(e) linearly in log(mu) against log(e) between table points."""
    e_table, mu_table = np.asarray(e_table, float), np.asarray(mu_table, float)
    if np.any(np.diff(e_table) <= 0):
        raise ValueError("table energies must increase strictly")
    e = np.asarray(e, float)
    if np.any((e < e_table[0]) | (e > e_table[-1])):
        raise ValueError("energy outside the table; no extrapolation")
    return np.exp(np.interp(np.log(e), np.log(e_table), np.log(mu_table)))
