"""Unit tests for the shared metric battery (exps 28/29/30 helpers) — the
functions every suite script calls to produce probe/acc/eucl/mahaT/lid rows."""
import importlib

import numpy as np
import pytest
import torch

from conftest import make_clusters
from supersig.config import DEVICE

exp28 = importlib.import_module("28_concat_residual")
exp29 = importlib.import_module("29_residual_finetune")
exp30 = importlib.import_module("30_power_curves")

SEEN = [0, 1, 2, 3]
HOLD = {4}


def _split(seed_tr=0, seed_te=1, n_tr=400, n_te=100, dim=8, gap=12.0):
    """5 well-separated blobs; class 4 is the holdout."""
    centers = (gap * np.eye(5, dim)).tolist()
    tr, tr_lab = make_clusters(centers, n_per=n_tr, dim=dim, seed=seed_tr)
    te, te_lab = make_clusters(centers, n_per=n_te, dim=dim, seed=seed_te)
    return tr, tr_lab, te, te_lab


def test_class_centroids_and_fill_means():
    tr, tr_lab, _, _ = _split()
    m = np.isin(tr_lab, SEEN)
    cents = exp28.class_centroids(tr[m], tr_lab[m], SEEN)
    assert cents.shape == (4, 8)
    for i, c in enumerate(SEEN):
        mu = torch.as_tensor(tr[tr_lab == c].mean(0), device=cents.device)
        assert torch.allclose(cents[i].float(), mu.float(), atol=1e-4)
    means = exp28.fill_means(cents, SEEN, dict(pair_dist=5.0, n_classes=5))
    assert means.shape == (5, 8)
    assert torch.allclose(means[:4].float(), cents.float(), atol=1e-4)
    assert means[4].abs().sum() > 0            # holdout row = anchor fill


def test_evaluate_space_full_battery_on_separable_blobs():
    tr, tr_lab, te, te_lab = _split()
    m = np.isin(tr_lab, SEEN)
    anch = exp28.class_centroids(tr[m], tr_lab[m], SEEN).float().to(DEVICE)
    r = exp29.evaluate_space(tr, tr_lab, te, te_lab, anch, SEEN, HOLD)
    assert set(r) >= {"acc", "sup_auc", "eucl", "maha_tied", "maha_pc",
                      "lid", "eigs", "scores"}
    assert r["acc"] > 0.99 and r["sup_auc"] > 0.99
    assert r["eucl"] > 0.99 and r["maha_tied"] > 0.99
    assert 0.0 <= r["lid"] <= 1.0
    s = r["scores"]
    assert set(s) >= {"eucl", "maha_pc", "lid", "is_unseen"}
    assert all(len(v) == len(te) for v in s.values())


def test_lid_novelty_flags_onmanifold_structure_change():
    """The exp-78 regime: novelty at small distance but different local
    dimension — LID must separate it; scale invariance must hold."""
    g = np.random.default_rng(0)
    dim = 8
    tr = np.zeros((2000, dim), dtype=np.float32)
    tr[:, :2] = g.standard_normal((2000, 2))          # seen: 2-D manifold
    tr_lab = np.zeros(2000, dtype=np.int64)
    te_seen = np.zeros((300, dim), dtype=np.float32)
    te_seen[:, :2] = g.standard_normal((300, 2))
    te_novel = g.standard_normal((300, dim)).astype(np.float32)  # full-rank
    te = np.concatenate([te_seen, te_novel])
    s = exp29.lid_novelty(tr, tr_lab, te, seen=[0], k=20)
    assert np.isfinite(s).all()
    assert s[300:].mean() > s[:300].mean()
    from sklearn.metrics import roc_auc_score
    is_novel = np.r_[np.zeros(300), np.ones(300)]
    auc = roc_auc_score(is_novel, s)
    assert auc > 0.9
    # scale-free: multiplying every embedding by 7 changes nothing
    s_scaled = exp29.lid_novelty(7.0 * tr, tr_lab, 7.0 * te, seen=[0], k=20)
    assert np.allclose(s, s_scaled, rtol=1e-4)


def test_lid_novelty_exact_ties_capped():
    tr = np.zeros((100, 4), dtype=np.float32)
    tr_lab = np.zeros(100, dtype=np.int64)
    te = np.zeros((10, 4), dtype=np.float32)
    s = exp29.lid_novelty(tr, tr_lab, te, seen=[0], k=20)
    assert np.isfinite(s).all()


def test_linear_probe_novelty_separable():
    tr, tr_lab, te, te_lab = _split()
    torch.manual_seed(1000)
    auc, scores, is_unseen = exp29.linear_probe_novelty(tr, tr_lab, te,
                                                        te_lab, HOLD)
    assert auc > 0.95
    assert len(scores) == len(te) and len(is_unseen) == len(te)


def test_power_at_alpha_extremes():
    g = np.random.default_rng(0)
    bg = g.standard_normal(5000)
    assert exp30.power_at_alpha(bg, bg + 10.0, alpha=0.05) == 1.0
    null = exp30.power_at_alpha(bg, g.standard_normal(5000), alpha=0.05)
    assert null == pytest.approx(0.05, abs=0.02)     # power = alpha under H0


def test_collect_embeddings_matches_forward():
    from torch.utils.data import DataLoader, TensorDataset
    from supersig.train import collect_embeddings
    X = torch.randn(100, 8)
    y = torch.arange(100) % 5
    net = torch.nn.Linear(8, 4).to(DEVICE)
    loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=False)
    embs, labs = collect_embeddings(net, loader)
    assert embs.shape == (100, 4) and (labs == y.numpy()).all()
    with torch.no_grad():
        ref = net(X.to(DEVICE)).cpu().numpy()
    assert np.allclose(embs, ref, atol=1e-5)
