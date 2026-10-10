import numpy as np
import pytest

from analysis.kinematics import (
    ELECTRON_REST_ENERGY,
    PAIR_THRESHOLD,
    compton_edge,
    compton_electron_energy,
    compton_scattered_energy,
)
from analysis.refs import ref_value

E1 = ref_value("co60_line_1_energy")
E2 = ref_value("co60_line_2_energy")


@pytest.mark.parametrize("e_gamma, edge_plan", [(E1, 0.963), (E2, 1.118)])
def test_compton_edges_match_plan(e_gamma, edge_plan):
    # STUDY_PLAN Layer 3 quotes the edges to 1 keV.
    assert compton_edge(e_gamma) == pytest.approx(edge_plan, abs=0.5e-3)


@pytest.mark.parametrize("e_gamma", [E1, E2])
def test_edge_is_backscatter_limit(e_gamma):
    assert compton_electron_energy(e_gamma, np.pi) == pytest.approx(compton_edge(e_gamma), rel=1e-12)


@pytest.mark.parametrize("e_gamma", [0.1, E1, E2, 5.0])
def test_energy_conservation_and_bounds(e_gamma):
    theta = np.linspace(0.0, np.pi, 1001)
    e_scat = compton_scattered_energy(e_gamma, theta)
    t_elec = compton_electron_energy(e_gamma, theta)
    np.testing.assert_allclose(e_scat + t_elec, e_gamma, rtol=1e-12)
    assert t_elec[0] == pytest.approx(0.0, abs=1e-15)
    assert np.all(np.diff(t_elec) >= 0.0)
    assert np.all(t_elec <= compton_edge(e_gamma) * (1 + 1e-12))


def test_compton_formula_against_closed_form_at_90_degrees():
    # At theta = 90 deg, E' = E / (1 + E / m c^2).
    e = 1.25
    assert compton_scattered_energy(e, np.pi / 2) == pytest.approx(e / (1 + e / ELECTRON_REST_ENERGY), rel=1e-14)


def test_pair_threshold_between_lines():
    # Both Co-60 lines are above the pair threshold (1.022 MeV in the plan).
    assert PAIR_THRESHOLD == pytest.approx(1.022, abs=1e-3)
    assert E1 > PAIR_THRESHOLD and E2 > PAIR_THRESHOLD
