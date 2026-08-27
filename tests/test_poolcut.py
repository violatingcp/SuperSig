"""Tests for supersig.poolcut -- the label-free pool cut used by run_discovery.

The load-bearing properties:
  1. it REFUSES when there is no novelty (or it will pool noise and the
     discovery loop will fine-tune on background structure);
  2. it never uses a holdout label;
  3. run_discovery's default is unchanged, so archived results reproduce.
"""
import numpy as np
import pytest

from supersig import poolcut


def _scores(n=8000, n_novel=200, sep=3.0, noise=0.3, seed=0):
    rng = np.random.default_rng(seed)
    v = np.zeros(n, dtype=bool)
    v[rng.choice(n, n_novel, replace=False)] = True
    f = rng.normal(0.0, noise, n)
    f[v] = rng.normal(sep, noise, int(v.sum()))
    return f, v


def test_finds_real_novelty():
    f, v = _scores()
    mask, info = poolcut.legal_pool(f, ~v)
    assert info["ok"]
    assert v[mask].mean() > 0.9


def test_refuses_pure_noise():
    """THE regression test.  Without a rank-matched null, w = [1-1/r]_+ is a
    positive part and ANY spread in f manufactures novelty: pure noise gave
    sum(w) = 665 of 8000 points before the fix, enough for the rule to engage
    on nothing."""
    rng = np.random.default_rng(1)
    f = rng.normal(0.0, 0.3, 8000)
    fake = np.zeros(8000, dtype=bool)
    fake[rng.choice(8000, 200, replace=False)] = True   # arbitrary "reference"
    _, info = poolcut.legal_pool(f, ~fake)
    assert not info["ok"], info
    assert info["n_hat_total"] < poolcut.N_MIN


@pytest.mark.parametrize("noise", [0.1, 0.3, 1.0])
def test_null_estimate_is_near_zero_under_h0(noise):
    rng = np.random.default_rng(2)
    f = rng.normal(0.0, noise, 6000)
    ref = np.zeros(6000, dtype=bool)
    ref[rng.choice(6000, 300, replace=False)] = True
    n_hat = poolcut.estimated_novel_count(f, f[~ref])
    assert n_hat < 0.02 * len(f), n_hat


def test_null_subtraction_preserves_real_signal():
    f, v = _scores(n_novel=400)
    raw = poolcut.estimated_novel_count(f, f[~v], subtract_null=False)
    sub = poolcut.estimated_novel_count(f, f[~v], subtract_null=True)
    assert sub <= raw
    assert sub > 0.5 * 400, sub


def test_refusal_still_returns_a_usable_pool():
    """`ok=False` must still hand back a mask (the wide-pool fallback), not
    an empty one -- callers should degrade, not crash."""
    rng = np.random.default_rng(3)
    f = rng.normal(0.0, 0.3, 4000)
    ref = np.zeros(4000, dtype=bool); ref[:200] = True
    mask, info = poolcut.legal_pool(f, ~ref)
    assert not info["ok"]
    assert mask.sum() > 0
    assert info["q"] == pytest.approx(poolcut.Q_MAX, abs=1e-3)


def test_tighter_than_the_inherited_cut_when_novelty_is_clear():
    f, v = _scores(n_novel=500, sep=4.0)
    mask, info = poolcut.legal_pool(f, ~v)
    assert info["q"] < 0.05                      # the inherited tau_quantile
    assert v[mask].mean() > (v[f > np.quantile(f[~v], 0.95)]).mean()


def test_kmax_is_label_free_and_bounded():
    f, v = _scores()
    _, info = poolcut.legal_pool(f, ~v)
    assert poolcut.K_FLOOR <= info["kmax"] <= poolcut.K_CAP


def test_run_discovery_default_is_unchanged():
    import inspect
    from supersig.discovery import run_discovery
    p = inspect.signature(run_discovery).parameters
    assert p["cut_rule"].default == "quantile"
    assert p["tau_quantile"].default == 0.95


def test_legal_rule_requires_a_density_ratio_scorer():
    from supersig.discovery import run_discovery
    with pytest.raises(ValueError, match="needs pool_score='np'"):
        run_discovery(None, None, base_ds=None, train_eval_loader=None,
                      test_loader=None, seen=[], holdouts={0},
                      dataset_name="x", rep_weight=0, sigreg_weight=0,
                      n_slices=8, cut_rule="legal", pool_score="dist")
