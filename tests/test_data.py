"""Unit tests for supersig/data.py — the balanced batch sampler and holdout
normalization (pure logic only; no dataset downloads)."""
import torch

from supersig.data import BalancedBatchSampler, _holdout_set


def test_holdout_set_normalization():
    assert _holdout_set(None) == set()
    assert _holdout_set(4) == {4}
    assert _holdout_set([4, 7]) == {4, 7}
    assert _holdout_set(range(90, 93)) == {90, 91, 92}


def test_balanced_batch_sampler_shape():
    targets = [i % 10 for i in range(1000)]
    sampler = BalancedBatchSampler(targets, n_classes=5, n_per_class=8)
    assert len(sampler) == 1000 // (5 * 8)
    t = torch.as_tensor(targets)
    for batch in sampler:
        assert len(batch) == 5 * 8
        labs = t[torch.as_tensor(batch)]
        classes, counts = torch.unique(labs, return_counts=True)
        assert len(classes) == 5
        assert (counts == 8).all()             # SIGReg's MIN_PER_CLASS guarantee


def test_balanced_batch_sampler_clamps_class_count():
    targets = [i % 3 for i in range(300)]
    sampler = BalancedBatchSampler(targets, n_classes=25, n_per_class=8)
    assert sampler.n_classes == 3              # only 3 classes exist
    batch = next(iter(sampler))
    assert len(batch) == 3 * 8
