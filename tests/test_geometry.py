"""Unit tests for the exp-76/77 geometry metrics — the functions behind every
verdict in docs/SPACE_GEOMETRY.md, checked against exact ground truths
(rotations, linear maps, known intrinsic dimensions, perfect hierarchies)."""
import importlib

import numpy as np
import pytest
from scipy.cluster import hierarchy
from scipy.stats import ortho_group

exp76 = importlib.import_module("76_interpretability")
exp77 = importlib.import_module("77_space_similarity")


def _X(n=600, d=8, seed=0):
    return np.random.default_rng(seed).standard_normal((n, d))


def _rot(d, seed=0):
    return ortho_group.rvs(d, random_state=seed)


# ------------------------------------------------------------------- exp 77

def test_sqdist_matches_direct_computation():
    A, B = _X(50, 6), _X(40, 6, seed=1)
    D = exp77.sqdist(A, B)
    ref = ((A[:, None] - B[None]) ** 2).sum(-1)
    assert np.allclose(D, ref, atol=1e-8)
    assert (D >= 0).all()
    assert np.allclose(np.diag(exp77.sqdist(A, A)), 0.0, atol=1e-8)


def test_cka_identity_rotation_and_independence():
    X = _X()
    assert exp77.cka(X, X) == pytest.approx(1.0, abs=1e-6)
    assert exp77.cka(X, 3.0 * X @ _rot(8)) == pytest.approx(1.0, abs=1e-6)
    assert exp77.cka(X, _X(seed=1)) < 0.2          # independent spaces


def test_mutual_knn_identity_and_rotation_invariance():
    X = _X(300, 6)
    iA = exp77.knn_idx(X, k=10)
    assert exp77.mutual_knn(iA, iA) == pytest.approx(1.0)
    # knn_idx uses cosine similarity, which a rotation preserves exactly
    iB = exp77.knn_idx(X @ _rot(6), k=10)
    assert exp77.mutual_knn(iA, iB) == pytest.approx(1.0)
    assert exp77.mutual_knn([[1, 2]], [[3, 4]]) == 0.0


def test_lle_transfer_valid_under_rotation_useless_when_independent():
    X = _X(300, 6)
    iA = exp77.knn_idx(X, k=10)
    same = exp77.lle_transfer(X, X @ _rot(6), iA)
    indep = exp77.lle_transfer(X, _X(300, 6, seed=1), iA)
    assert same < 1.0 < indep + 0.15               # ratio 1 = uniform baseline
    assert same < indep


def test_procrustes_residual_zero_under_scaled_rotation():
    X = _X()
    assert exp77.procrustes_resid(X, 5.0 * X @ _rot(8)) == pytest.approx(
        0.0, abs=1e-8)
    assert exp77.procrustes_resid(X, _X(seed=1)) > 0.5
    assert np.isnan(exp77.procrustes_resid(X, _X(d=4, seed=1)))


def test_ridge_r2_recovers_linear_maps_and_is_directional():
    X = _X()
    W_true = np.random.default_rng(2).standard_normal((8, 5))
    Y = X @ W_true + 3.0
    assert exp77.ridge_r2(X, Y) > 0.99
    assert exp77.ridge_r2(X, _X(seed=1)) < 0.3     # nothing to explain
    # directionality: a projection is decodable from the full space, not back
    P = X[:, :3]
    assert exp77.ridge_r2(X, P) > 0.99
    assert exp77.ridge_r2(P, X) < 0.7


def test_ridge_map_reusable_on_held_out_points():
    X = _X()
    W_true = np.random.default_rng(3).standard_normal((8, 8))
    Y = X @ W_true
    W, mx, my = exp77.ridge_map(X[:400], Y[:400])
    assert exp77.ridge_r2(X[400:], Y[400:], W, mx, my) > 0.99


def test_twonn_id_recovers_known_dimension():
    g = np.random.default_rng(0)
    rng = np.random.default_rng(1)
    for d_true, lo, hi in ((2, 1.5, 3.0), (5, 3.8, 6.5)):
        X = np.zeros((2000, 8))
        X[:, :d_true] = g.uniform(size=(2000, d_true))
        assert lo < exp77.twonn_id(X, rng) < hi


def test_lid_scores_scale_free_and_tie_capped():
    g = np.random.default_rng(0)
    ref = np.zeros((1500, 8))
    ref[:, :2] = g.standard_normal((1500, 2))      # 2-D reference manifold
    q_on = np.zeros((200, 8))
    q_on[:, :2] = g.standard_normal((200, 2))
    q_off = g.standard_normal((200, 8))
    s = exp77.lid_scores(ref, np.concatenate([q_on, q_off]), k=20)
    assert np.isfinite(s).all()
    assert s[200:].mean() > s[:200].mean()         # off-manifold = higher LID
    s7 = exp77.lid_scores(7.0 * ref, 7.0 * np.concatenate([q_on, q_off]), k=20)
    assert np.allclose(s, s7, rtol=1e-6)           # ratios only
    ties = exp77.lid_scores(np.zeros((100, 4)), np.zeros((10, 4)), k=20)
    assert np.allclose(ties, 1000.0)               # -1e-3 cap, not inf


def test_eucl_auc_separable():
    g = np.random.default_rng(0)
    seen, hold = [0, 1], {2}
    ytr = np.repeat([0, 1], 300)
    Xtr = g.standard_normal((600, 6)) + 10.0 * np.eye(3, 6)[ytr]
    yte = np.repeat([0, 1, 2], 100)
    Xte = g.standard_normal((300, 6)) + 10.0 * np.eye(3, 6)[yte]
    assert exp77.eucl_auc(Xtr, ytr, Xte, yte, seen, hold) > 0.99


# ------------------------------------------------------------------- exp 76

def test_centroid_dist_orthogonal_vs_aligned():
    g = np.random.default_rng(0)
    y = np.repeat([0, 1, 2], 200)
    X = 20.0 * np.eye(3, 6)[y] + 0.1 * g.standard_normal((600, 6))
    Cn, D, ok = exp76.centroid_dist(X, y, n_cls=4)
    assert ok.tolist() == [True, True, True, False]  # class 3 empty
    assert np.allclose(np.diag(D), 0.0)
    assert np.allclose(np.linalg.norm(Cn[:3], axis=1), 1.0, atol=1e-6)
    # orthogonal centroids -> cosine distance 1
    assert D[0, 1] == pytest.approx(1.0, abs=0.01)
    y2 = np.repeat([0, 1], 200)
    X2 = np.tile([5.0, 0.0], (400, 1)) + 0.01 * g.standard_normal((400, 2))
    _, D2, _ = exp76.centroid_dist(X2, y2, n_cls=2)
    assert D2[0, 1] == pytest.approx(0.0, abs=0.01)  # same direction -> 0


def _linkage_of(points):
    return hierarchy.linkage(points, method="average", metric="cosine")


def test_dendrogram_purity_perfect_and_mixed():
    # two superclasses whose members merge among themselves first
    pts = np.array([[1.0, 0.0], [0.99, 0.01], [0.98, 0.02],
                    [0.0, 1.0], [0.01, 0.99], [0.02, 0.98]])
    assert exp76.dendrogram_purity(_linkage_of(pts),
                                   [0, 0, 0, 1, 1, 1]) == pytest.approx(1.0)
    # same tree, superclass labels interleaved -> impure merges
    mixed = exp76.dendrogram_purity(_linkage_of(pts), [0, 1, 0, 1, 0, 1])
    assert mixed < 0.7


def test_sup_metrics_perfect_superclass_geometry():
    pts = np.array([[1.0, 0.0], [0.99, 0.01], [0.98, 0.02],
                    [0.0, 1.0], [0.01, 0.99], [0.02, 0.98]])
    Cn = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    D = 1.0 - Cn @ Cn.T
    np.fill_diagonal(D, 0.0)
    m = exp76.sup_metrics(D, Cn, [0, 0, 0, 1, 1, 1])
    assert m["agree1"] == pytest.approx(1.0)       # NN always same superclass
    assert m["chance"] == pytest.approx(12.0 / 30.0)
    assert m["dendro_purity"] == pytest.approx(1.0)
    assert m["wb_ratio"] < 0.1                     # tight within, far between
    assert m["silhouette"] > 0.8
