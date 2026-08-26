"""Tests for the exp-129 label-free pool-cut rule.

The rule must be OPEN-WORLD LEGAL: it may see the corpus and the seen-labelled
reference, never a holdout label.  These tests pin the estimator algebra, the
dominance of the detectability-limited rule over the purity-target rule, and
the one-sided (safe) direction of the estimator bias.
"""
import importlib.util
import os

import numpy as np
import pytest

_D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name, fn):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_D, "experiments", fn))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


exp129 = _load("exp129", "129_legal_pool_cut.py")

QS = np.unique(np.concatenate([np.geomspace(0.002, 0.02, 10),
                               np.linspace(0.02, 0.20, 19)]))


def _case(b, overlap=0.0, n=20000, sep=8.0):
    z, is_novel = exp129._make(b, n=n, sep=sep, overlap=overlap)
    mu = np.zeros(z.shape[1])
    mu[0] = sep * (1 - overlap)
    return z, is_novel, exp129.analytic_f(z, mu, b)


def test_selftest_runs():
    exp129._selftest()


@pytest.mark.parametrize("est", ["tv", "mass", "excess"])
@pytest.mark.parametrize("b", [0.01, 0.05, 0.10])
def test_estimators_recover_known_base_rate(est, b):
    _, is_novel, f = _case(b)
    assert exp129.estimate_b(f, ~is_novel, est) == pytest.approx(b, rel=0.1)


def test_estimator_bias_is_one_sided_downward():
    """Under-stating b tightens q, which is safe.  Over-stating would inflate
    the pool and silently dilute purity."""
    vals = [exp129.estimate_b(_case(0.05, overlap=o)[2],
                              ~_case(0.05, overlap=o)[1], "tv")
            for o in (0.0, 0.6, 0.9)]
    assert max(vals) <= 0.05 * 1.05
    assert vals[-1] < vals[0]


def test_detectability_rule_dominates_purity_target_rule():
    """Purity rises monotonically as q shrinks, so the right objective is a
    detectability floor, not a purity level."""
    for b in (0.01, 0.02, 0.05, 0.10):
        _, is_novel, f = _case(b)
        r = exp129.evaluate(f, is_novel, ~is_novel, QS)
        assert r["purity_detect"] >= r["purity_rule"] - 1e-9, (b, r)


def test_detectability_rule_keeps_a_usable_cluster():
    for b in (0.01, 0.05):
        _, is_novel, f = _case(b)
        r = exp129.evaluate(f, is_novel, ~is_novel, QS)
        assert r["n_novel_detect"] >= 0.5 * exp129.N_MIN


def test_rule_aborts_when_too_little_novelty():
    q, ok, why = exp129.rule_q(0.001, 20000)
    assert not ok and "n_min" in why


def test_novelty_weights_estimate_the_count_without_labels():
    """sum of per-point novel-ness must approximate the true novel count."""
    for b in (0.02, 0.05):
        _, is_novel, f = _case(b)
        w = exp129.novelty_weights(f, f[~is_novel])
        assert w.sum() == pytest.approx(is_novel.sum(), rel=0.15)


def test_renormalisation_makes_estimates_offset_invariant():
    """A miscalibrated critic (constant offset in f) must not change b_hat --
    the fitted critic is often far from E_ref[e^f]=1."""
    _, is_novel, f = _case(0.05)
    a = exp129.estimate_b(f, ~is_novel, "tv")
    b = exp129.estimate_b(f + 3.0, ~is_novel, "tv")
    assert a == pytest.approx(b, abs=1e-6)


def test_calibration_error_detects_miscalibration():
    _, is_novel, f = _case(0.05)
    assert exp129.calibration_error(f[~is_novel]) == pytest.approx(1.0, abs=0.15)
    assert exp129.calibration_error(f[~is_novel] + 3.0) > 5.0


def test_reference_side_tv_would_have_been_useless():
    """Guards the bug that was actually hit: integrating the excess against the
    REFERENCE sample collapses to ~0 when novelty is disjoint, because no
    reference point lands where the excess lives."""
    _, is_novel, f = _case(0.01)
    r_ref = np.exp(np.clip(f[~is_novel], -20, 20))
    naive = float(np.mean(np.maximum(0.0, r_ref - 1.0)))
    good = exp129.estimate_b(f, ~is_novel, "tv")
    assert naive < 0.1 * 0.01
    assert good == pytest.approx(0.01, rel=0.1)
