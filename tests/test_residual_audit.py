"""Tests for the exp-134 residual audit.

The property under test is that the CONCATENATION RULE, not just the
construction, determines what the residual appears to buy: a linear probe is
scale-robust and Euclidean distance is not, so a raw concat of two
differently-scaled halves can win the probe while losing the geometry --
exactly the 30/30-win / 14-of-15-loss pattern in the archive.
"""
import importlib.util
import os

import numpy as np
import pytest

_P = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  "experiments", "134_residual_audit.py")
_s = importlib.util.spec_from_file_location("exp134", _P)
exp134 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(exp134)


@pytest.fixture
def halves():
    """Parent hides the novel class; child separates it."""
    rng = np.random.default_rng(0)
    C, d, n = 10, 16, 300
    y = np.repeat(np.arange(C), n)
    pm = rng.normal(0, 3.0, (C, d)); pm[9] = pm[0]
    P = np.concatenate([pm[c] + rng.normal(0, 1, (n, d)) for c in range(C)])
    cm = rng.normal(0, 3.0, (C, d))
    Ch = np.concatenate([cm[c] + rng.normal(0, 1, (n, d)) for c in range(C)])
    return P, Ch, y, {9}, [c for c in range(C) if c != 9]


def test_selftest_runs():
    exp134._selftest()


def test_raw_concat_loses_geometry_when_scales_mismatch(halves):
    P, Ch, y, hold, seen = halves
    good = exp134.eucl_auc(*[np.concatenate([P, Ch], 1)] * 1, y,
                           np.concatenate([P, Ch], 1), y, seen, hold)
    bad = exp134.eucl_auc(np.concatenate([P, Ch * 0.02], 1), y,
                          np.concatenate([P, Ch * 0.02], 1), y, seen, hold)
    assert good > bad + 0.2, (good, bad)


def test_probe_is_robust_to_the_same_mismatch(halves):
    """The asymmetry that makes the artefact hard to notice."""
    P, Ch, y, hold, _ = halves
    a = exp134.probe_auc(np.concatenate([P, Ch], 1), y,
                         np.concatenate([P, Ch], 1), y, hold)
    b = exp134.probe_auc(np.concatenate([P, Ch * 0.02], 1), y,
                         np.concatenate([P, Ch * 0.02], 1), y, hold)
    assert abs(a - b) < 0.1, (a, b)


@pytest.mark.parametrize("mode", ["standardize", "unitnorm", "whiten"])
def test_combiners_repair_the_geometry(halves, mode):
    P, Ch, y, hold, seen = halves
    tr, st = exp134.combine(P, Ch * 0.02, mode)
    raw, _ = exp134.combine(P, Ch * 0.02, "raw")
    a = exp134.eucl_auc(tr, y, tr, y, seen, hold)
    b = exp134.eucl_auc(raw, y, raw, y, seen, hold)
    assert a > b + 0.15, (mode, a, b)


def test_combiner_stats_are_fit_on_train_only(halves):
    P, Ch, y, _, _ = halves
    _, st = exp134.combine(P, Ch, "standardize")
    te, _ = exp134.combine(P + 5.0, Ch, "standardize", ref=st)
    assert abs(float(te[:, :P.shape[1]].mean())) > 1.0


def test_half_scale_ratio_flags_one_sidedness(halves):
    P, Ch, *_ = halves
    assert exp134.half_scale_ratio(P, Ch * 0.01) < 0.05
    assert 0.5 < exp134.half_scale_ratio(P, Ch) < 2.0
    assert exp134.half_scale_ratio(P, Ch * 100) > 20


def test_raw_combiner_is_the_archived_behaviour(halves):
    """`raw` must be byte-identical to np.concatenate, or the audit is not
    comparing against what the campaign actually ran."""
    P, Ch, *_ = halves
    got, st = exp134.combine(P, Ch, "raw")
    assert st == {}
    assert np.allclose(got, np.concatenate([P, Ch], 1))


def test_combine_rejects_unknown_mode(halves):
    P, Ch, *_ = halves
    with pytest.raises(ValueError):
        exp134.combine(P, Ch, "nonsense")
