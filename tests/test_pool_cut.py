"""Tests for the exp-128 pool-cut algebra.

The claims these pin, all of which the paper now leans on:

    q      = e_s*b + e_b*(1-b)
    purity = e_s*b / q
    E      = purity/b = e_s/q
    purity <= min(1, b/q)                (the ceiling)

The ceiling is the load-bearing one: it says the pool cut caps achievable purity
independently of how good the scorer is, which is why `tau_quantile=0.95` --
inherited from exp 23 and never derived -- is worth re-deriving.
"""
import importlib.util
import os

import numpy as np
import pytest

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "experiments", "128_pool_cut_optimization.py")
_spec = importlib.util.spec_from_file_location("exp128", _PATH)
exp128 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp128)


@pytest.fixture
def novel():
    rng = np.random.default_rng(0)
    N = 20000
    v = np.zeros(N, dtype=bool)
    v[rng.choice(N, int(N * 0.01), replace=False)] = True
    return v


QS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]


def test_selftest_runs():
    """The script's own --selftest must pass (8 checks)."""
    exp128._selftest()


def test_enrichment_identity(novel):
    rng = np.random.default_rng(1)
    s = novel * rng.normal(1.5, 1.0, len(novel)) + rng.normal(0, 1, len(novel))
    for r in exp128.curve(s, novel, QS):
        assert r["enrichment"] == pytest.approx(r["eps_s"] / r["q"], rel=1e-9)


def test_ceiling_is_never_exceeded(novel):
    rng = np.random.default_rng(2)
    for _ in range(5):
        s = novel * rng.normal(2.0, 1.0, len(novel)) + rng.normal(0, 1, len(novel))
        for r in exp128.curve(s, novel, QS):
            assert r["purity"] <= r["ceiling"] + 1e-12


def test_perfect_scorer_attains_the_ceiling(novel):
    rng = np.random.default_rng(3)
    s = novel.astype(float) + 1e-6 * rng.random(len(novel))
    for r in exp128.curve(s, novel, QS):
        assert r["purity"] == pytest.approx(r["ceiling"], abs=1e-6)


def test_the_headline_ceiling_number(novel):
    """b=1%, q=5% -> ceiling 0.20.  This is the '20x available' claim, and the
    reason the h1 failure is a signal-efficiency floor and not a rate floor."""
    rng = np.random.default_rng(4)
    s = novel.astype(float) + 1e-6 * rng.random(len(novel))
    r = exp128.operating_point(s, novel, 0.05)
    assert r["ceiling"] == pytest.approx(0.20, abs=1e-9)
    assert r["purity"] == pytest.approx(0.20, abs=1e-6)


def test_gate_is_unreachable_at_loose_cuts_even_when_perfect(novel):
    """At b=1%, no scorer however good clears the 0.15 gate with q >= 0.07 --
    a property of the CUT, not of the space."""
    rng = np.random.default_rng(5)
    s = novel.astype(float) + 1e-6 * rng.random(len(novel))
    for q in (0.07, 0.10, 0.20):
        r = exp128.operating_point(s, novel, q)
        assert not r["gate"], (q, r["purity"])


def test_best_q_respects_detectability(novel):
    rng = np.random.default_rng(6)
    s = novel.astype(float) + 1e-6 * rng.random(len(novel))
    rows = exp128.curve(s, novel, QS)
    b = exp128.best_q(rows, n_min=100)
    assert b is not None and b["n_novel"] >= 100
    # and with an impossible floor there is simply no usable cut
    assert exp128.best_q(rows, n_min=10 ** 6) is None


def test_anti_selective_scorer_reports_enrichment_below_one(novel):
    """The C100-distance case: the pool is LESS novel than a random sample.
    Must be visible rather than merely looking small."""
    rng = np.random.default_rng(7)
    s = -(novel * rng.normal(2.0, 1.0, len(novel)) + rng.normal(0, 1, len(novel)))
    r = exp128.operating_point(s, novel, 0.05)
    assert r["enrichment"] < 1.0


def test_high_auc_can_still_pool_badly():
    """AUC is cut-free but blind to TAIL STRUCTURE.  A scorer can have high
    novel-vs-seen AUC and still put zero novel points in the extreme tail --
    which is exactly the CIFAR-100 distance situation (heavy-tailed background
    owns the tail).  So AUC does not replace the operating curve.
    """
    rng = np.random.default_rng(8)
    N = 20000
    v = np.zeros(N, dtype=bool)
    v[rng.choice(N, 200, replace=False)] = True
    # novel: moderately elevated.  background: mostly low, but a heavy tail
    # that reaches far above the novel band.
    s = rng.normal(0.0, 1.0, N)
    s[v] = rng.normal(3.0, 0.3, int(v.sum()))
    outl = rng.choice(np.where(~v)[0], 300, replace=False)
    s[outl] = rng.normal(8.0, 1.0, 300)
    a = exp128.auc(s, v)
    tight = exp128.operating_point(s, v, 0.005)
    assert a > 0.9, a
    assert tight["purity"] < 0.10, tight
