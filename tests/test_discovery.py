"""Unit tests for supersig/discovery.py — kmeans/BIC, anchor merging, LID pool
scores, and a one-round smoke test of the full run_discovery loop."""
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from conftest import make_clusters
from supersig.config import DEVICE
from supersig.discovery import (PseudoDataset, bic_select, kmeans,
                                lid_pool_scores, merge_anchors, run_discovery)


def _blobs(centers, n_per=200, dim=4, noise=0.5, seed=0):
    embs, _ = make_clusters(centers, n_per=n_per, dim=dim, noise=noise, seed=seed)
    return torch.as_tensor(embs, device=DEVICE)


def test_kmeans_recovers_separated_clusters():
    true = [[0.0, 0.0], [20.0, 0.0], [0.0, 20.0]]
    X = _blobs(true)
    C, assign = kmeans(X, k=3, seed=0)
    T = torch.zeros(3, X.size(1), device=DEVICE)
    for i, mu in enumerate(true):
        T[i, : len(mu)] = torch.tensor(mu, device=DEVICE)
    assert torch.cdist(T, C).min(1).values.max() < 1.0   # every center found
    assert assign.shape == (len(X),)


def test_bic_select_finds_k_for_tight_clusters():
    """The unit-variance BIC selects the true k when clusters are tight
    relative to the model's unit covariance (noise 0.1)."""
    k3, C, _ = bic_select(_blobs([[0.0], [20.0], [0.0, 20.0]], noise=0.1),
                          kmax=5, seed=0)
    assert k3 == 3 and C.shape[0] == 3
    k1, _, _ = bic_select(_blobs([[0.0]], noise=0.1), kmax=4, seed=0)
    assert k1 == 1


def test_bic_select_fragments_unit_clusters():
    """Documented behavior, not a bug: at unit cluster scale (the calibrated-
    latent regime) the d*log(n) penalty is smaller than the SSE gain of
    splitting a unit Gaussian, so BIC saturates at kmax — merge_anchors and
    the repulsion exemption absorb the fragmentation downstream (exp 24)."""
    k, _, _ = bic_select(_blobs([[0.0]], noise=1.0), kmax=4, seed=0)
    assert k == 4


def test_merge_anchors_merges_close_keeps_far():
    disc = torch.tensor([[0.0, 0.0], [0.5, 0.0], [10.0, 10.0]], device=DEVICE)
    memb = torch.tensor([0, 0, 2], device=DEVICE)
    merged = merge_anchors(disc, memb, merge_dist=3.0)
    assert merged.shape[0] == 2
    assert merge_anchors(disc, memb, merge_dist=0.0).shape[0] == 3


def test_lid_pool_scores_flags_offmanifold_novelty():
    g = np.random.default_rng(0)
    # seen data on a 2-D subspace of an 8-D space; novelty is full-rank 8-D
    seen = np.zeros((1500, 8), dtype=np.float32)
    seen[:, :2] = g.standard_normal((1500, 2))
    novel = g.standard_normal((300, 8)).astype(np.float32)
    z = torch.as_tensor(np.concatenate([seen, novel]), device=DEVICE)
    is_seen = np.concatenate([np.ones(1500, bool), np.zeros(300, bool)])
    s = lid_pool_scores(z, is_seen, k=20)
    assert s.shape == (1800,)
    assert torch.isfinite(s).all()
    assert s[1500:].mean() > s[:1500].mean()   # novel points locally higher-dim


def test_lid_pool_scores_caps_exact_ties():
    # 40 identical points: all neighbor ratios are exact ties -> the -1e-3
    # cap must keep the score finite (1000), not inf
    z = torch.zeros(40, 4, device=DEVICE)
    s = lid_pool_scores(z, np.ones(40, bool), k=10)
    assert torch.isfinite(s).all()
    assert torch.allclose(s, torch.full_like(s, 1000.0))


def test_pseudo_dataset_relabels():
    base = TensorDataset(torch.arange(10, dtype=torch.float32).view(-1, 1),
                         torch.zeros(10, dtype=torch.long))
    ds = PseudoDataset(base, indices=[3, 7], labels=[11, 12])
    assert len(ds) == 2
    x, y = ds[1]
    assert float(x) == 7.0 and y == 12 and isinstance(y, int)


def test_run_discovery_one_round_smoke():
    """Full loop on synthetic features: seen classes 0-3, holdout 4."""
    dim, n_cls = 8, 5
    centers = (6.0 * np.eye(n_cls, dim)).tolist()
    Xtr, ytr = make_clusters(centers, n_per=300, dim=dim, noise=0.7, seed=0)
    Xte, yte = make_clusters(centers, n_per=60, dim=dim, noise=0.7, seed=1)
    seen, holdouts = [0, 1, 2, 3], {4}
    # strict open world: the backbone never trains on holdout-labeled points,
    # but they sit in the unlabeled pool exactly like the real pipeline
    Xtr_t, ytr_t = torch.as_tensor(Xtr), torch.as_tensor(ytr)
    base = TensorDataset(Xtr_t, ytr_t)
    tr_loader = DataLoader(base, batch_size=256, shuffle=False)
    te_loader = DataLoader(TensorDataset(torch.as_tensor(Xte),
                                         torch.as_tensor(yte)),
                           batch_size=256, shuffle=False)
    backbone = torch.nn.Linear(dim, dim).to(DEVICE)
    means = torch.zeros(n_cls, dim, device=DEVICE)
    with torch.no_grad():
        z = backbone(Xtr_t.to(DEVICE))
        for c in range(n_cls):
            means[c] = z[ytr_t == c].mean(0)
    cur_means, hist = run_discovery(
        backbone, means, base_ds=base, train_eval_loader=tr_loader,
        test_loader=te_loader, seen=seen, holdouts=holdouts,
        dataset_name="toy", rep_weight=0.01, sigreg_weight=1.0, n_slices=16,
        rounds=1, ft_epochs=1, seed=0)
    assert cur_means.shape[0] > n_cls           # anchors were appended
    assert len(hist) == 1
    h = hist[0]
    assert set(h) >= {"round", "pool", "purity", "khat", "n_anchors",
                      "margin", "per_class", "mean_pc"}
    assert h["pool"] > 0 and 0.0 <= h["purity"] <= 1.0
    assert 0.0 <= h["margin"] <= 1.0


def test_run_discovery_lid_scorer_smoke():
    """pool_score='lid' takes the exp-79 branch and completes a round."""
    dim, n_cls = 8, 5
    centers = (6.0 * np.eye(n_cls, dim)).tolist()
    Xtr, ytr = make_clusters(centers, n_per=300, dim=dim, noise=0.7, seed=2)
    Xte, yte = make_clusters(centers, n_per=60, dim=dim, noise=0.7, seed=3)
    seen, holdouts = [0, 1, 2, 3], {4}
    base = TensorDataset(torch.as_tensor(Xtr), torch.as_tensor(ytr))
    tr_loader = DataLoader(base, batch_size=256, shuffle=False)
    te_loader = DataLoader(TensorDataset(torch.as_tensor(Xte),
                                         torch.as_tensor(yte)),
                           batch_size=256, shuffle=False)
    backbone = torch.nn.Linear(dim, dim).to(DEVICE)
    means = torch.randn(n_cls, dim, device=DEVICE)
    _, hist = run_discovery(
        backbone, means, base_ds=base, train_eval_loader=tr_loader,
        test_loader=te_loader, seen=seen, holdouts=holdouts,
        dataset_name="toy", rep_weight=0.01, sigreg_weight=1.0, n_slices=16,
        rounds=1, ft_epochs=1, seed=0, pool_score="lid")
    assert len(hist) == 1 and hist[0]["pool"] > 0
