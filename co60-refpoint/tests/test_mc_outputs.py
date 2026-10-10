"""Tests on archived co60app runs: track-length estimator, units, attenuation interpolation, stopping powers."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from analysis.attenuation import loglog_interp
from analysis.primary import primary_check, pyramid_solid_angle
from analysis.stopping import bases

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "mc" / "runs"
TABLES = RUNS / "M1-tables-option4-v2" / "batch_00" / "output.json"
THIN = RUNS / "M1-thinslab-option4"
PEGS4 = ROOT / "mc" / "egsnrc" / "b0"

needs_thin = pytest.mark.skipif(not (THIN / "batch_00" / "output.json").exists(), reason="thin-slab run not present")
needs_tables = pytest.mark.skipif(not TABLES.exists(), reason="tables run not present")


@needs_thin
@needs_tables
def test_track_length_estimator_thin_slab_matches_uncollided_fluence():
    # STUDY_PLAN Section 8: track-length estimator on a thin slab against the analytic uncollided fluence.
    res = primary_check(THIN, TABLES)
    for line in ("line1", "line2"):
        assert abs(res[line]["z"]) < 3.0, res[line]
        assert abs(res[line]["rel_diff"]) < 0.003, res[line]


@needs_thin
def test_output_units_are_cm_g_mev():
    # Geometry-derived numbers in the output must agree with an independent calculation in cm and g.
    d = json.load(open(THIN / "batch_00" / "output.json"))
    g, s = d["geometry"], d["source"]
    assert d["units"] == {"length": "cm", "energy": "MeV", "mass": "g", "fluence": "cm^-2"}
    vol = math.pi * g["scoring_radius"] ** 2 * g["scoring_thickness"]
    assert g["scoring_volume_cm3"] == pytest.approx(vol, rel=1e-12)
    assert g["scoring_mass_g"] == pytest.approx(vol * g["water_density_g_cm3"], rel=1e-9)
    omega = pyramid_solid_angle(s["ssd"], s["field_x"], s["field_y"])
    assert s["solid_angle_sr"] == pytest.approx(omega, rel=1e-12)
    assert s["phi0_per_history_cm-2"] == pytest.approx(1 / (omega * (s["ssd"] + g["scoring_depth"]) ** 2), rel=1e-12)
    # Track lengths in cm: primary fluence is about Phi0 (a length in mm would be 10x larger).
    tl = d["tallies"]["photon_track_length"]["data"]
    phi_primary = (tl[0] + tl[1]) / vol / d["histories"]
    assert 0.9 < phi_primary / s["phi0_per_history_cm-2"] < 1.0
    assert d["cuts_g4_water"]["e-_threshold_MeV"] == pytest.approx(0.010, rel=1e-6)


@needs_tables
def test_attenuation_interpolation_against_directly_computed_values():
    # The tables hold mu/rho on a log grid and, separately, computed exactly at the two lines.
    # Log-log interpolation of the grid must reproduce the exact values.
    rows = json.load(open(TABLES))["materials"]["G4_WATER"]["photon"]
    grid = rows[:-2]
    e = np.array([r["E"] for r in grid])
    mu = np.array([r["compt"] + r["phot"] + r["conv"] + r["Rayl"] for r in grid])
    for exact in rows[-2:]:
        total = exact["compt"] + exact["phot"] + exact["conv"] + exact["Rayl"]
        assert loglog_interp(exact["E"], e, mu) == pytest.approx(total, rel=2e-4)
    assert loglog_interp(e[10], e, mu) == pytest.approx(mu[10], rel=1e-12)
    with pytest.raises(ValueError):
        loglog_interp(5.0, e, mu)


@needs_tables
def test_bethe_stopping_power_matches_geant4_and_pegs4_for_air():
    # Air uses the same I and density effect in all three codes paths above 100 keV,
    # so the Bethe-Moller implementation must agree with Geant4 and PEGS4.
    b = bases(TABLES, PEGS4)
    t = np.array([0.15, 0.3, 0.5, 1.0, 1.25])
    for cut in (None, 0.010):
        ours = b["icru90"]["air"][0](t, cut) if cut else b["icru90"]["air"][1](t)
        g4 = b["geant4"]["air"][0](t, cut) if cut else b["geant4"]["air"][1](t)
        np.testing.assert_allclose(ours, g4, rtol=5e-4)
    np.testing.assert_allclose(b["icru90"]["air"][0](t, 0.010), b["pegs4"]["air"][0](t, 0.010), rtol=5e-4)
