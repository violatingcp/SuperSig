"""Unit tests for the statistical-power machinery — supersig/sparker.py and
the exp-31/32 toy-calibrated batteries, on synthetic two-sample problems with
known answers (blatant signal -> high power, no signal -> ~alpha)."""
import importlib

import numpy as np
import pytest
import torch

from supersig.config import DEVICE
from supersig.sparker import (aggregate_pvalues, clopper_pearson, krr_term,
                              median_pairwise, mmd2_multi_stats, np_test_stats)

exp31 = importlib.import_module("31_sparker_power")
exp32 = importlib.import_module("32_maha_mmd_power")

def _pools(shift=3.0, n_bg=2000, n_sig=500, dim=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    bg = torch.randn(n_bg, dim, generator=g).to(DEVICE)
    sig = (torch.randn(n_sig, dim, generator=g) + shift).to(DEVICE)
    ref = torch.randn(1000, dim, generator=g).to(DEVICE)
    return bg, sig, ref


def test_median_pairwise_exact_for_two_points():
    X = torch.tensor([[0.0, 0.0], [3.0, 4.0]], device=DEVICE)
    assert median_pairwise(X) == pytest.approx(5.0)


def test_aggregate_pvalues_hand_computed():
    null = [[1.0], [2.0], [3.0], [4.0]]
    # above every null: p = (1+0)/5, agg = -log p (min = mean, one checkpoint)
    assert aggregate_pvalues([10.0], null) == pytest.approx(-np.log(0.2))
    # below every null: p = 1, agg = 0
    assert aggregate_pvalues([0.0], null) == pytest.approx(0.0)


def test_clopper_pearson_bounds():
    lo, hi = clopper_pearson(0, 50)
    assert lo == 0.0 and 0.0 < hi < 0.1
    lo, hi = clopper_pearson(50, 50)
    assert 0.9 < lo < 1.0 and hi == 1.0
    lo, hi = clopper_pearson(25, 50)
    assert lo < 0.5 < hi


def test_np_test_stats_detects_shifted_sample():
    """A moderate (3-sigma) bump is detected at the wide-sigma checkpoint.
    NOTE: signal far outside kernel reach is INVISIBLE to this statistic —
    the sigma-matching lesson of exp 57; that is why the shift here is 3,
    not 8, and why the battery aggregates over the annealing checkpoints."""
    bg, sig, ref = _pools()
    kw = dict(M=8, steps=100, n_checkpoints=2)
    ts_null = np_test_stats(bg[:400], ref, seed=0, **kw)
    D = torch.cat([bg[:360], sig[:40]])            # 10% signal
    ts_sig = np_test_stats(D, ref, seed=0, **kw)
    assert len(ts_null) == len(ts_sig) == kw["n_checkpoints"]
    assert all(np.isfinite(ts_null)) and all(np.isfinite(ts_sig))
    assert ts_sig[0] > 2.0 * ts_null[0]            # wide-sigma checkpoint


def test_mmd2_near_zero_for_same_distribution_positive_for_shift():
    bg, sig, ref = _pools()
    med = median_pairwise(bg)
    sigmas = [0.5 * med, med, 2.0 * med]
    krr = krr_term(ref, sigmas)
    same = mmd2_multi_stats(bg[:500], ref, sigmas, krr)
    shift = mmd2_multi_stats(torch.cat([bg[:450], sig[:50]]), ref, sigmas, krr)
    assert max(abs(v) for v in same) < 0.05        # unbiased, ~0 under H0
    assert all(sh > sa for sh, sa in zip(shift, same))
    assert max(shift) > 0.01


def test_toy_indices_sizes_and_bootstrap_fallback():
    rng = np.random.default_rng(0)
    bg, sg = exp31.toy_indices(rng, 1000, 500, N_D=200, n_sig=20)
    assert len(bg) == 180 and len(sg) == 20
    assert bg.max() < 1000 and sg.max() < 500
    bg, sg = exp31.toy_indices(rng, 1000, 500, N_D=200, n_sig=0)
    assert len(bg) == 200 and len(sg) == 0
    bg, _ = exp31.toy_indices(rng, 50, 500, N_D=200, n_sig=0)
    assert len(bg) == 200                          # bootstrap when pool small


def test_sparker_battery_power_high_for_moderate_signal():
    bg, sig, ref = _pools()
    powers, bands = exp31.run_test_battery(
        bg, sig, ref, fractions=[0.0, 0.1], N_D=200, n_null=8, n_sig_toys=8,
        alpha=0.2, seed=0, sparker_kw=dict(M=8, steps=100, n_checkpoints=2),
        tag="test")
    assert len(powers) == len(bands) == 2
    assert powers[0] <= 0.5                        # no signal ~ alpha
    assert powers[1] >= 0.75                       # 3-sigma bump at f=0.1
    lo, hi = bands[1]
    assert lo <= powers[1] <= hi


def test_maha_and_mmd_batteries_on_synthetic_space():
    g = np.random.default_rng(0)
    dim, shift = 6, 12.0
    seen, holdout = [0, 1, 2], 3
    ytr = np.repeat(seen, 800)
    Xtr = (shift * np.eye(4, dim)[ytr]
           + g.standard_normal((2400, dim))).astype(np.float32)
    yte = np.repeat([0, 1, 2, 3], 300)
    Xte = (shift * np.eye(4, dim)[yte]
           + g.standard_normal((1200, dim))).astype(np.float32)
    maha_fn, mmd_fn, n_bg, n_sig = exp32.make_stats_fns(
        Xtr, ytr, Xte, yte, seen, holdout, seed=0)
    assert n_bg == 900 and n_sig == 300
    for fn, tag in ((maha_fn, "maha"), (mmd_fn, "mmd")):
        powers, bands = exp32.battery(fn, n_bg, n_sig, fractions=[0.1],
                                      N_D=200, n_null=16, n_sig_toys=8,
                                      alpha=0.2, seed=0, tag=tag)
        assert powers[0] >= 0.75, tag              # far-off holdout: easy
