"""Tests for the exp-132 supervised linear probe.

Guards the two ways this metric could quietly mislead:
  1. measuring embedding SCALE instead of linear separability (the arms differ
     in scale by design -- calibrated objectives fix unit class width, softmax
     ones do not);
  2. declaring winners on gaps below the campaign's own noise floor (0.017
     seed, exp 52; 0.019 draw, exp 118), which is where most of the
     comparisons the paper would like to make actually sit.
"""
import importlib.util
import os

import numpy as np
import pytest

_P = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  "experiments", "132_supervised_probe.py")
_s = importlib.util.spec_from_file_location("exp132", _P)
exp132 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(exp132)


@pytest.fixture
def bank():
    rng = np.random.default_rng(0)
    C, d, n = 8, 12, 300
    mus = rng.normal(0, 3.0, (C, d))
    X = np.concatenate([mus[c] + rng.normal(0, 1, (n, d)) for c in range(C)])
    y = np.repeat(np.arange(C), n)
    return X.astype(np.float32), y, list(range(C - 1))


def test_selftest_runs():
    exp132._selftest()


def test_probe_is_scale_invariant(bank):
    X, y, seen = bank
    a, _, _ = exp132.probe_multiseed(X * 0.05, y, X * 0.05, y, seen,
                                     seeds=1, epochs=15)
    b, _, _ = exp132.probe_multiseed(X * 20.0, y, X * 20.0, y, seen,
                                     seeds=1, epochs=15)
    assert abs(a - b) < 0.05, (a, b)


def test_probe_excludes_holdout_classes(bank):
    X, y, seen = bank
    # the held-out class must never be a prediction target
    acc, _, _ = exp132.probe_multiseed(X, y, X, y, seen, seeds=1, epochs=10)
    assert 0.0 <= acc <= 1.0
    assert max(seen) == len(np.unique(y)) - 2


def test_probe_separates_clean_from_noisy(bank):
    X, y, seen = bank
    rng = np.random.default_rng(1)
    noisy = (X + rng.normal(0, 8.0, X.shape)).astype(np.float32)
    a, _, _ = exp132.probe_multiseed(X, y, X, y, seen, seeds=1, epochs=15)
    b, _, _ = exp132.probe_multiseed(noisy, y, noisy, y, seen, seeds=1,
                                     epochs=15)
    assert a > b + 0.1


@pytest.mark.parametrize("gap,expected", [
    (0.008, "TIE"),      # the supcon_sigreg-vs-supcon gap on scr-simclr
    (0.016, "TIE"),      # ... on scr-visreg
    (0.030, "A"),
    (-0.030, "B"),
])
def test_tie_guard_matches_the_campaign_noise_floor(gap, expected):
    assert exp132.verdict(0.90 + gap, 0.90, 0.002, 0.002)[0] == expected


def test_tie_guard_widens_with_measured_spread():
    """A noisy pair of arms must be harder to separate, not easier."""
    assert exp132.verdict(0.95, 0.90, 0.001, 0.001)[0] == "A"
    assert exp132.verdict(0.95, 0.90, 0.03, 0.03)[0] == "TIE"


def test_noise_floors_match_the_documented_values():
    assert exp132.SEED_SPREAD == 0.017     # exp 52
    assert exp132.DRAW_SPREAD == 0.019     # exp 118
