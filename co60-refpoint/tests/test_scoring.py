"""Scoring functions (analysis/scoring.py) against hand-checkable cases."""
import numpy as np

from analysis import scoring as S


def test_scalar_tolerances():
    assert S.scalar_verdict(1.019, 1.0, "dose")["verdict"] == "pass"
    assert S.scalar_verdict(1.021, 1.0, "dose")["verdict"] == "fail"
    assert S.scalar_verdict(1.021, 1.0, "dose", floor_abs=0.03)["verdict"] == "unresolved"
    # scalar class: 5% relative or 0.005 absolute, whichever is larger
    assert S.scalar_verdict(0.0049, 0.0, "scalar")["verdict"] == "pass"
    assert S.scalar_verdict(0.104, 0.1, "scalar")["verdict"] == "pass"
    assert S.scalar_verdict(0.11, 0.1, "scalar")["verdict"] == "fail"
    assert S.scalar_verdict(None, 0.1, "scalar")["verdict"] == "missing"
    assert S.scalar_verdict(float("nan"), 0.1, "scalar")["verdict"] == "missing"


def test_wasserstein_on_bin_centres():
    edges = [0, 1, 2, 3]
    p, q = np.array([1.0, 0, 0]), np.array([0, 0, 1.0])
    assert abs(S.wasserstein1(p, q, edges) - 2.0) < 1e-12      # centres 0.5 and 2.5
    assert S.wasserstein1(p, p, edges) == 0.0


def test_jsd_bounds():
    p, q = np.array([1.0, 0]), np.array([0, 1.0])
    assert abs(S.jsd(p, q) - 1.0) < 1e-12
    assert S.jsd(p, p) == 0.0


def test_distribution_verdict_normalizes_and_thresholds():
    edges = [0, 1, 2]
    v = S.distribution_verdict([2, 2], [0.5, 0.5], edges, half_w1=0.0, half_jsd=0.0)
    assert v["verdict"] == "pass"
    v = S.distribution_verdict([0.9, 0.1], [0.5, 0.5], edges, half_w1=0.01, half_jsd=0.001)
    assert v["verdict"] == "fail"
    v = S.distribution_verdict([0.9, 0.1], [0.5, 0.5], edges, 0.01, 0.001, floor_w1=1.0, floor_jsd=1.0)
    assert v["verdict"] == "unresolved"
    assert S.distribution_verdict([1.0], [0.5, 0.5], edges, 0, 0)["verdict"] == "missing"


def test_revised_shape_criterion_d038():
    # A pure shift by one bin moves the mean by the bin width, and W1 equals that shift.
    edges = np.linspace(0.0, 2.0, 201)                          # 10 keV bins
    mc = np.zeros(200)
    mc[99] = 1.0                                                 # all at 0.995 MeV
    tol = S.w1_tolerance("photon_energy", mc, edges)
    assert abs(tol - 0.05 * 0.995) < 1e-12
    near, far = np.roll(mc, 4), np.roll(mc, 5)                   # shifted by 40 and 50 keV
    assert S.distribution_verdict_w1(near, mc, edges, tol)["verdict"] == "pass"
    assert S.distribution_verdict_w1(far, mc, edges, tol)["verdict"] == "fail"
    assert S.distribution_verdict_w1(far, mc, edges, tol, floor_w1=0.06)["verdict"] == "unresolved"
    assert S.w1_tolerance("cos_theta", mc, edges) == 0.1
    assert S.distribution_verdict_w1([1.0], mc, edges, tol)["verdict"] == "missing"


def test_breakpoint_is_first_resolved_failure():
    assert S.breakpoint({1: ["pass"], 2: ["unresolved"], 3: ["fail"], 5: ["fail"]})[0] == 3
    assert S.breakpoint({1: ["pass", "missing"], 5: ["pass"]})[0] is None


def test_weighted_ks_matches_unweighted_case():
    from scipy import stats
    rng = np.random.default_rng(3)
    a, b = rng.normal(0, 1, 300), rng.normal(0.3, 1, 2000)
    ours = S.weighted_ks(a, b, np.ones_like(b))["ks"]
    assert abs(ours - stats.ks_2samp(a, b).statistic) < 1e-12


def test_energy_distance_zero_for_identical_samples():
    x = np.random.default_rng(1).normal(size=(200, 3))
    assert abs(S.energy_distance(x, x)) < 1e-12
