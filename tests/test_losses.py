"""Unit tests for supersig/losses.py — SIGReg, anchors, SupCon, the hybrid cube."""
import itertools
import math

import pytest
import torch

from supersig.config import DEVICE
from supersig.losses import (
    ANCHOR_SCALE, MIN_PER_CLASS, HybridContrastiveLoss, _nplm_interaction,
    classwise_sigreg_loss, make_anchors, mean_geometry, repulsion_loss,
    separation_loss, shrink_loss, sigreg_loss, standard_normal_quantiles,
    supcon_loss,
)


def test_standard_normal_quantiles_symmetric():
    q = standard_normal_quantiles(1001, "cpu")
    assert torch.allclose(q, -q.flip(0), atol=1e-5)
    assert abs(q[500].item()) < 1e-5          # median = 0


def test_sigreg_low_for_gaussian_high_for_shifted():
    z = torch.randn(2048, 8, device=DEVICE)
    l_gauss = sigreg_loss(z, n_slices=256).item()
    l_shift = sigreg_loss(z + 5.0, n_slices=256).item()
    l_scale = sigreg_loss(3.0 * z, n_slices=256).item()
    assert l_gauss < 0.1
    assert l_shift > 10 * l_gauss
    assert l_scale > 10 * l_gauss


def test_sigreg_differentiable():
    z = torch.randn(256, 8, device=DEVICE, requires_grad=True)
    sigreg_loss(z).backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()


def test_classwise_sigreg_centered_on_means():
    means = torch.stack([torch.full((8,), 0.0), torch.full((8,), 6.0)]).to(DEVICE)
    y = torch.cat([torch.zeros(512), torch.ones(512)]).long().to(DEVICE)
    z = means[y] + torch.randn(1024, 8, device=DEVICE)
    on = classwise_sigreg_loss(z, y, means, n_slices=256).item()
    off = classwise_sigreg_loss(z + 4.0, y, means, n_slices=256).item()
    assert on < 0.1 < off


def test_classwise_sigreg_skips_small_classes():
    means = torch.zeros(2, 8, device=DEVICE)
    z = torch.randn(MIN_PER_CLASS - 1, 8, device=DEVICE)
    y = torch.zeros(MIN_PER_CLASS - 1, dtype=torch.long, device=DEVICE)
    assert classwise_sigreg_loss(z, y, means).item() == 0.0


def test_make_anchors_orthogonal_when_dim_allows():
    a = make_anchors(scale=6.0, emb_dim=16, n_classes=10)
    assert a.shape == (10, 16)
    assert torch.allclose(a.norm(dim=1), torch.full((10,), 6.0, device=a.device))
    gram = a @ a.t()
    assert torch.allclose(gram, torch.diag(torch.diagonal(gram)), atol=1e-5)


def test_make_anchors_random_fallback_below_dim():
    a = make_anchors(scale=6.0, emb_dim=8, n_classes=10)
    b = make_anchors(scale=6.0, emb_dim=8, n_classes=10)
    assert a.shape == (10, 8)
    assert torch.allclose(a.norm(dim=1), torch.full((10,), 6.0, device=a.device),
                          atol=1e-5)
    assert torch.allclose(a, b)               # deterministic
    assert torch.cdist(a, a)[torch.eye(10) == 0].min() > 0.1


def test_mean_geometry_and_separation_and_shrink():
    far = make_anchors(scale=10.0, emb_dim=4, n_classes=3)
    mn, mean = mean_geometry(far)
    assert mn == pytest.approx(10.0 * math.sqrt(2.0), rel=1e-4)
    assert separation_loss(far, margin=3.0).item() == 0.0
    assert separation_loss(0.01 * far, margin=3.0).item() > 0.0
    assert shrink_loss(torch.zeros(3, 4, device=DEVICE)).item() == 0.0
    assert shrink_loss(far).item() == pytest.approx(100.0, rel=1e-4)


def test_repulsion_decays_and_respects_exemption():
    near = make_anchors(scale=1.0, emb_dim=4, n_classes=3)
    far = make_anchors(scale=10.0, emb_dim=4, n_classes=3)
    assert repulsion_loss(near).item() > repulsion_loss(far).item()
    # means 1 and 2 coincide: full repulsion diverges, exempting the
    # discovered-discovered pair (both indices >= 1) removes the blow-up
    m = torch.zeros(3, 4, device=DEVICE)
    m[0, 0] = 5.0
    full = repulsion_loss(m).item()
    exempt = repulsion_loss(m, exempt_from=1).item()
    assert full > 1e4 and exempt < 1.0


def test_supcon_prefers_true_class_clusters():
    centers = 4.0 * torch.eye(4, device=DEVICE)
    y = torch.arange(4, device=DEVICE).repeat_interleave(16)
    feats = torch.nn.functional.normalize(
        centers[y] + 0.1 * torch.randn(64, 4, device=DEVICE), dim=1)
    good = supcon_loss(feats, y).item()
    bad = supcon_loss(feats, y[torch.randperm(64)]).item()
    assert good < bad


def test_nplm_interaction_hand_computed():
    g = torch.tensor([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]])
    self_mask = torch.eye(3, dtype=torch.bool)
    pos = torch.tensor([[False, True, False],
                        [True, False, False],
                        [False, False, False]])
    ref_mean = ((math.e ** 2 - 1) * 2 + (math.e ** 3 - 1) * 2) / 4.0
    expect = ref_mean - 1.0                   # E_ref[e^g - 1] - E_pos[g]
    assert _nplm_interaction(g, pos, self_mask).item() == pytest.approx(expect,
                                                                        rel=1e-5)


def test_nplm_clamp_prevents_overflow():
    g = torch.full((4, 4), 1e4)
    self_mask = torch.eye(4, dtype=torch.bool)
    pos = torch.zeros(4, 4, dtype=torch.bool)
    assert torch.isfinite(_nplm_interaction(g, pos, self_mask))


@pytest.mark.parametrize("positives,critic,estimator,marginal",
                         list(itertools.product(
                             ["instance", "supervised"],
                             ["cosine", "bilinear", "distance"],
                             ["softmax", "nplm"],
                             ["none", "sigreg", "classwise_sigreg"])))
def test_hybrid_cube_every_corner_finite_and_differentiable(
        positives, critic, estimator, marginal):
    n, dim, n_cls = 32, 8, 2
    z = torch.randn(2 * n, dim, device=DEVICE, requires_grad=True)
    if positives == "instance":
        labels = torch.arange(n, device=DEVICE).repeat(2)
    else:
        labels = (torch.arange(2 * n, device=DEVICE) % n_cls)
    means = make_anchors(scale=3.0, emb_dim=dim, n_classes=n_cls)
    loss_fn = HybridContrastiveLoss(positives=positives, critic=critic,
                                    estimator=estimator, marginal=marginal,
                                    tau=1.0, lam=1.0, n_slices=32)
    total, parts = loss_fn(z, labels, means=means)
    assert torch.isfinite(total)
    assert set(parts) == {"interaction", "marginal"}
    total.backward()
    assert torch.isfinite(z.grad).all()


def test_hybrid_rejects_unknown_knobs():
    with pytest.raises(ValueError):
        HybridContrastiveLoss(positives="nope")
    with pytest.raises(ValueError):
        HybridContrastiveLoss(estimator="nope")
    with pytest.raises(ValueError):
        HybridContrastiveLoss(marginal="nope")
    with pytest.raises(ValueError):
        HybridContrastiveLoss(critic="nope").interaction(
            torch.randn(4, 2, device=DEVICE), torch.zeros(4, device=DEVICE))
    with pytest.raises(ValueError):
        z = torch.randn(32, 4, device=DEVICE)
        HybridContrastiveLoss(marginal="classwise_sigreg")(
            z, torch.zeros(32, dtype=torch.long, device=DEVICE))
