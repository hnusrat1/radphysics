"""STUDY_PLAN Section 8 required test: the s_w,air calculation reproduces SPRRZnrc on B0 within the combined
uncertainty (DECISIONS D039).

The check holds the transport code fixed. Our Spencer-Attix function (analysis/stopping.py) is applied to the FLURZnrc
electron fluence of the same EGSnrc set-up and compared with SPRRZnrc, which computes the ratio during transport; both
use the same PEGS4 stopping powers. FLURZnrc gives at best 2 keV bins over the whole spectrum, and EGSnrc stops
electrons at Delta, so the track-end fluence at Delta is read from above it: extrapolated from the 12-30 keV bins
(the first bin above Delta is inflated by FLURZnrc's even spreading of a step's path length down to the cut-off), or
read from that first bin. The effect of this procedure is measured on the Geant4 B0 fluence, which continues below
Delta, and removed. Reads analysis/results/M2_results.json (analysis/m2.py).
"""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
M2 = ROOT / "analysis" / "results" / "M2_results.json"
B0 = json.load(open(M2)).get("b0", {}) if M2.exists() else {}
CHK = B0.get("s_w_air_check", {})
PRACTICAL = 5e-4      # 0.05%: one fortieth of the 2% tolerance on s_w,air
needs = pytest.mark.skipif("sprrznrc_over_calc" not in CHK, reason="B0 s_w,air check not in the M2 results")


@needs
def test_procedure_is_unbiased_on_a_spectrum_without_a_cut_off():
    # 2 keV bins and the extrapolated track-end fluence, against the 1 keV reference calculation, on Geant4's spectrum
    d = CHK["procedure_effect_on_geant4"]["extrapolated"]
    assert abs(d["mean"]) < CHK["sprrznrc"]["sem"], d


@needs
def test_calculation_reproduces_sprrznrc_within_combined_uncertainty():
    r = CHK["sprrznrc_over_calc"]["extrapolated"]
    assert abs(r["ratio"] - 1) <= 2 * r["sem"], r                     # k = 2


@needs
def test_either_track_end_reading_agrees_with_sprrznrc_to_practical_precision():
    # Read from the inflated first bin, the result is 3 sigma off (D039); both readings stay far inside 0.05%.
    for how, r in CHK["sprrznrc_over_calc"].items():
        assert abs(r["ratio"] - 1) < PRACTICAL, (how, r)


@needs
def test_reference_s_w_air_agrees_with_sprrznrc_to_practical_precision():
    # The reference value uses Geant4's fluence, so the transport codes differ and the statistical uncertainty is not
    # the bar.
    r = B0["s_w_air_egsnrc_over_geant4"]
    assert abs(r["ratio"] - 1) < PRACTICAL, r
