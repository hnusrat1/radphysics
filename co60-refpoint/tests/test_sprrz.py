"""STUDY_PLAN Section 8 required test: the s_w,air calculation reproduces SPRRZnrc on B0 within the combined
uncertainty (DECISIONS D039).

The check holds the transport code fixed. Our Spencer-Attix function (analysis/stopping.py) is applied to the FLURZnrc
electron fluence of the same EGSnrc set-up and compared with SPRRZnrc, which computes the ratio during transport; both
use the same PEGS4 stopping powers. FLURZnrc gives at best 2 keV bins over the whole spectrum. The effect of 2 keV
against 1 keV bins is measured on the Geant4 B0 fluence, batch by batch, and removed.
Reads analysis/results/M2_results.json (analysis/m2.py).
"""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
M2 = ROOT / "analysis" / "results" / "M2_results.json"
B0 = json.load(open(M2)).get("b0", {}) if M2.exists() else {}
CHK = B0.get("s_w_air_check", {})
needs = pytest.mark.skipif("sprrznrc_over_calc_on_egsnrc_fluence" not in CHK,
                           reason="B0 s_w,air check not in the M2 results")


def corrected_ratio():
    """SPRRZnrc over our calculation on the EGSnrc fluence, with the 2 keV binning effect removed."""
    r = CHK["sprrznrc_over_calc_on_egsnrc_fluence"]
    s2 = CHK["calc_on_egsnrc_fluence_2keV"]["mean"]
    d = CHK["geant4_2keV_minus_1keV_paired"]["mean"]        # 2 keV minus 1 keV bins, same fluence
    return r["ratio"] * s2 / (s2 - d), r["sem"]


@needs
def test_binning_effect_is_small_and_measured():
    d = CHK["geant4_2keV_minus_1keV_paired"]
    assert abs(d["mean"]) < 5e-5 and d["sem"] < 1e-6, d


@needs
def test_calculation_reproduces_sprrznrc_within_combined_uncertainty():
    ratio, sem = corrected_ratio()
    assert abs(ratio - 1) <= 2 * sem, (ratio, sem)          # k = 2


@needs
def test_reference_s_w_air_agrees_with_sprrznrc_to_0p1_percent():
    # The reference value: Geant4's fluence with our calculation. Against SPRRZnrc the transport codes differ, so the
    # statistical uncertainty is not the bar; 0.1% is twenty times below the 2% tolerance on s_w,air.
    r = B0["s_w_air_egsnrc_over_geant4"]
    assert abs(r["ratio"] - 1) < 1e-3, r
