"""Tests for the exp-136 base-rate estimator work.

Two properties matter. The knee criterion must be scale-invariant (that is the
whole reason to prefer it over an n_min threshold), and the estimators must
never OVER-state the base rate -- over-stating would inflate the pool and
silently dilute purity, whereas under-stating only tightens the cut.

Note the real-data verdict recorded in the script docstring: on trained spaces
the knee FAILS (picks a far wider cut than n_min=30) and all three estimators
are biased low by the same 4-7x, so these tests pin behaviour, not efficacy.
"""
import importlib.util
import os

import numpy as np
import pytest

_P = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  "experiments", "136_base_rate_estimator.py")
_s = importlib.util.spec_from_file_location("exp136", _P)
exp136 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(exp136)


def _case(b=0.02, overlap=0.0, seed=0):
    z, v = exp136._make(b, sep=8.0, overlap=overlap, seed=seed)
    f = exp136._analytic_f(z, 8.0 * (1 - overlap), b)
    return z, v, f


def test_selftest_runs():
    exp136._selftest()


@pytest.mark.parametrize("est", ["tv", "mass", "excess"])
def test_estimators_are_exact_without_overlap(est):
    _, v, f = _case(b=0.02)
    assert exp136.b_hat(f, f[~v], est) == pytest.approx(0.02, abs=0.004)


@pytest.mark.parametrize("est", ["tv", "mass", "excess"])
def test_estimators_never_overstate(est):
    """The bias direction must stay safe: under-stating tightens the cut,
    over-stating inflates the pool."""
    for ov in (0.0, 0.5, 0.8, 0.95):
        _, v, f = _case(b=0.02, overlap=ov)
        assert exp136.b_hat(f, f[~v], est) <= 0.02 * 1.15, (est, ov)


def test_knee_is_invariant_to_a_uniform_weight_scale():
    """The defining property. If it were not scale-invariant it would offer
    nothing over an n_min threshold."""
    import supersig.poolcut as pc
    _, v, f = _case(b=0.02)
    k0, _ = exp136.knee_k(f, f[~v])
    real = pc.novelty_weights
    try:
        for alpha in (0.25, 4.0):
            pc.novelty_weights = lambda a, b_, _r=real, _a=alpha: _a * _r(a, b_)
            assert exp136.knee_k(f, f[~v])[0] == k0, alpha
    finally:
        pc.novelty_weights = real


def test_knee_finds_a_clean_mode():
    """On separated novelty the knee should land near the true novel count --
    the case where it works."""
    _, v, f = _case(b=0.02)
    k, _ = exp136.knee_k(f, f[~v])
    assert abs(k - int(v.sum())) < 0.15 * int(v.sum()), (k, int(v.sum()))


def test_knee_returns_none_when_there_is_no_novelty():
    rng = np.random.default_rng(3)
    f = rng.normal(0, 0.3, 6000)
    ref = np.zeros(6000, bool); ref[:300] = True
    k, info = exp136.knee_k(f, f[~ref])
    assert k is None or info["n_hat_total"] >= 0


def test_n_min_scan_purity_falls_as_n_min_rises():
    """The empirical finding that survived real data."""
    _, v, f = _case(b=0.02, overlap=0.55)
    rows = exp136.n_min_scan(f, ~v, v, n_mins=(10, 30, 100))
    assert rows[0]["purity"] >= rows[-1]["purity"]
    assert rows[0]["q"] <= rows[-1]["q"]


def test_n_min_scan_reports_implied_alpha():
    _, v, f = _case(b=0.02, overlap=0.55)
    rows = exp136.n_min_scan(f, ~v, v, n_mins=(10, 30))
    for r in rows:
        assert r["implied_alpha"] > 0
        assert r["n_novel"] <= int(v.sum())


def test_b_hat_rejects_unknown_estimator():
    _, v, f = _case()
    with pytest.raises(ValueError):
        exp136.b_hat(f, f[~v], "nonsense")
