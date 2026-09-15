"""Unit tests for supersig/metrics.py — gaussianity diagnostics and Mahalanobis novelty."""
import numpy as np

from conftest import make_clusters
from supersig.metrics import (classwise_gaussianity, gaussianity_summary,
                              mahalanobis_novelty, sliced_gaussianity)


def test_sliced_gaussianity_calibrated_near_one_for_gaussian():
    import torch
    z = torch.randn(1500, 8)
    r = sliced_gaussianity(z, n_slices=128, n_null=16)
    assert 0.5 < r["ratio"] < 2.0
    assert r["skew"] < 0.2 and r["ex_kurtosis"] < 0.5


def test_sliced_gaussianity_flags_bimodal():
    import torch
    z = torch.cat([torch.randn(750, 8) - 5.0, torch.randn(750, 8) + 5.0])
    r = sliced_gaussianity(z, n_slices=128, n_null=16)
    assert r["ratio"] > 5.0


def test_classwise_gaussianity_keys():
    embs, labs = make_clusters([[0.0], [8.0]], n_per=400, dim=4)
    out = classwise_gaussianity(embs, labs, n_slices=64, n_null=8)
    assert set(out) == {0, 1}
    assert all("ratio" in v for v in out.values())


def test_gaussianity_summary_unit_gaussians():
    embs, labs = make_clusters([[0.0], [10.0], [0.0, 10.0]], n_per=500, dim=6)
    g = gaussianity_summary(embs, labs, classes=[0, 1, 2], n_slices=64, n_null=8)
    assert 0.85 < g["rms_mean"] < 1.15          # unit blobs
    assert g["cdist_min"] > 8.0                 # centers 10 apart
    assert g["separation"] > 7.0                # sigma-scaled gap
    assert g["sw_ratio_mean"] < 2.0
    for k in ("eig_min", "eig_max", "corr_max", "skew_mean", "kurt_mean"):
        assert k in g


def test_mahalanobis_novelty_separates_and_calibrates():
    tr, tr_lab = make_clusters([[0.0], [10.0]], n_per=1000, dim=6)
    te_seen, _ = make_clusters([[0.0], [10.0]], n_per=100, dim=6, seed=1)
    te_novel, _ = make_clusters([[30.0]], n_per=100, dim=6, seed=2)
    te = np.concatenate([te_seen, te_novel])
    tied, perclass, eigs = mahalanobis_novelty(tr, tr_lab, te, seen=[0, 1])
    assert tied.shape == perclass.shape == (300,)
    assert tied[200:].min() > tied[:200].max()      # novel strictly farther
    assert perclass[200:].mean() > perclass[:200].mean()
    lo, med, hi = eigs
    assert 0.5 < med < 2.0                          # unit within-class covariance
