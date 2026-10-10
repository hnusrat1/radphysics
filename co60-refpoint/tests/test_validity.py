import numpy as np

from analysis.kinematics import ELECTRON_REST_ENERGY as ME
from analysis.kinematics import compton_edge, compton_scattered_energy
from analysis.validity import BINDING_TOLERANCE, check_events, doppler_tolerance


def free_compton(e, theta):
    e_out = compton_scattered_energy(e, theta)
    return e_out, e - e_out


def test_free_electron_compton_is_valid_under_both_rule_sets():
    e = np.array([1.1732, 1.3325, 0.4])
    th = np.array([0.3, 2.0, np.pi])
    e_out, t = free_compton(e, th)
    for kw in ({}, {"delta_c": 0.05, "energy_tol": BINDING_TOLERANCE}):
        r = check_events(["compt"] * 3, e, e_out, np.cos(th), t, **kw)
        assert r["valid"].all()


def test_recoil_above_edge_fails_strict_but_passes_bound_within_tolerance():
    e = 1.1732
    t = compton_edge(e) + 0.02
    e_out = e - t
    strict = check_events(["compt"], [e], [e_out], [-1.0], [t])
    bound = check_events(["compt"], [e], [e_out], [-1.0], [t], delta_c=0.05, energy_tol=BINDING_TOLERANCE)
    assert not strict["compton_edge"][0]
    assert bound["compton_edge"][0]


def test_energy_not_conserved_fails():
    r = check_events(["compt"], [1.25], [0.5], [0.0], [0.5], delta_c=0.05, energy_tol=BINDING_TOLERANCE)
    assert not r["energy"][0] and not r["valid"][0]


def test_pair_below_threshold_fails_and_rest_mass_is_counted():
    r = check_events(["conv", "conv"], [0.9, 1.3325], [-1, -1], [np.nan, np.nan], [0.0, 0.2],
                     t_pos=[0.0, 1.3325 - 2 * ME - 0.2])
    assert not r["pair_threshold"][0]
    assert r["energy"][1] and r["valid"][1]


def test_doppler_tolerance_rounds_up():
    e = np.array([1.25, 1.25])
    th = np.array([1.0, np.pi])
    e_out, t = free_compton(e, th)
    t = t + np.array([0.0, 0.013])
    tol = doppler_tolerance(["compt", "compt"], e, e_out - np.array([0, 0.013]), np.cos(th), t)
    assert abs(tol["delta_c"] - 0.02) < 1e-12
    assert tol["fraction_above_edge"] == 0.5
