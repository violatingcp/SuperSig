"""
Loss functions for the SuperSig study.

Includes SIGReg (Sketched Isotropic Gaussian Regularization) and its supervised
class-conditional variants, the mean-geometry regularizers (hinge separation and
inverse-square repulsion), the supervised contrastive (SupCon) loss, and the
dual (local + global) supervised visual regularizer (DualSuperVisReg).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import EMB_DIM, N_CLASSES, DEVICE

MIN_PER_CLASS = 8       # skip a class in a batch if it has too few samples for the test
ANCHOR_SCALE = 6.0      # spacing of the per-class Gaussian centres
MEANS_MARGIN = 6.0      # hinge-separation margin between learnable means
REPULSE_EPS = 1e-6


# --------------------------------------------------------------------------- #
# SIGReg                                                                       #
# --------------------------------------------------------------------------- #
def standard_normal_quantiles(n, device):
    """Quantiles of N(0,1) at plotting positions (k-0.5)/n, k=1..n."""
    p = (torch.arange(n, device=device, dtype=torch.float32) + 0.5) / n
    return torch.sqrt(torch.tensor(2.0, device=device)) * torch.erfinv(2.0 * p - 1.0)


def sigreg_loss(z, n_slices=64):
    """
    Sketched Isotropic Gaussian Regularization.

    Project the batch `z` (B x D) onto `n_slices` random unit directions and, for
    each slice, measure the squared 1-D Wasserstein-2 distance between the sorted
    projections and the standard-normal quantiles.  Averaging over random slices
    approximates the deviation of the joint distribution from N(0, I).
    """
    b, d = z.shape
    directions = F.normalize(torch.randn(d, n_slices, device=z.device), dim=0)
    proj = z @ directions
    proj_sorted, _ = torch.sort(proj, dim=0)
    q = standard_normal_quantiles(b, z.device).unsqueeze(1)
    return ((proj_sorted - q) ** 2).mean()


def classwise_sigreg_loss(z, y, means, n_slices=64, class_sigma=None):
    """SIGReg applied independently to each supervised category in the batch.

    class_sigma: optional per-class target std (len = means rows).  Residuals
    are divided by sigma_c, so class c is pulled toward N(mean_c, sigma_c^2 I)
    -- the asymmetric variance-annealing hook of exp 35 (JEPAMatch-inspired).
    """
    losses = []
    for c in torch.unique(y):
        mask = y == c
        if mask.sum() < MIN_PER_CLASS:
            continue
        zc = z[mask] - means[c]
        if class_sigma is not None:
            zc = zc / class_sigma[c]
        losses.append(sigreg_loss(zc, n_slices=n_slices))
    if not losses:
        return z.new_zeros(())
    return torch.stack(losses).mean()


# --------------------------------------------------------------------------- #
# Class-mean geometry                                                          #
# --------------------------------------------------------------------------- #
def make_anchors(scale=ANCHOR_SCALE, emb_dim=EMB_DIM, n_classes=N_CLASSES):
    """
    One fixed, well-separated anchor per class, each with norm `scale`.

    If `emb_dim >= n_classes` the anchors are scaled standard basis vectors
    (mutually orthogonal).  Otherwise (e.g. 8 dims for 10 classes) orthogonal
    axes do not exist, so we fall back to deterministic unit-norm random
    directions scaled to `scale` -- still distinct and reasonably spread out.
    """
    if emb_dim >= n_classes:
        anchors = torch.zeros(n_classes, emb_dim, device=DEVICE)
        for c in range(n_classes):
            anchors[c, c] = scale
        return anchors
    g = torch.Generator().manual_seed(0)                    # deterministic
    v = torch.randn(n_classes, emb_dim, generator=g)
    v = torch.nn.functional.normalize(v, dim=1) * scale
    return v.to(DEVICE)


def _pairwise(means):
    d = torch.cdist(means, means)
    iu = torch.triu_indices(means.size(0), means.size(0), offset=1)
    return d[iu[0], iu[1]]


def separation_loss(means, margin=MEANS_MARGIN):
    """Hinge penalty on pairs of class means closer than `margin` (prevents collapse)."""
    pair_d = _pairwise(means)
    return torch.clamp(margin - pair_d, min=0.0).pow(2).mean()


def repulsion_loss(means, exempt_from=None):
    """
    Inverse-square repulsion between every pair of class means (Coulomb-like).

    exempt_from: if set, pairs where BOTH means have index >= exempt_from are
    excluded (e.g. discovered anchors may collapse onto each other while still
    being repelled from the original class means).
    """
    d = torch.cdist(means, means)
    iu = torch.triu_indices(means.size(0), means.size(0), offset=1)
    vals = 1.0 / (d[iu[0], iu[1]].pow(2) + REPULSE_EPS)
    if exempt_from is not None:
        keep = ~((iu[0] >= exempt_from) & (iu[1] >= exempt_from))
        vals = vals[keep]
    return vals.sum()


def shrink_loss(means):
    """Mild pull toward the origin so repulsed means stay finite."""
    return means.pow(2).sum(dim=1).mean()


def mean_geometry(means):
    """Diagnostics: (min, mean) pairwise distance between class means."""
    pair_d = _pairwise(means)
    return pair_d.min().item(), pair_d.mean().item()


# --------------------------------------------------------------------------- #
# Dual supervised visual regularizer (local + global Gaussians)               #
# --------------------------------------------------------------------------- #
class DualSuperVisReg(nn.Module):
    """
    Every class is an isotropic Gaussian; every class centroid lives in a
    global Gaussian.  Anchor-free: class means emerge from the batch instead
    of being learnable parameters.

    global loss = L_center + L_scale + L_shape   (on the batch class centroids)
    local loss  = L_local_center + L_local_cov + L_local_shape  (per class)
    total loss  = w_global * global loss + local loss

    Unlike sigreg_loss, the slicing directions are a fixed buffer (W_fixed) so
    the loss curve is comparable across steps.
    """

    def __init__(
        self,
        num_classes=9,
        embed_dim=4,
        num_projections=4096,
        global_scale=20.0,
        local_target_scale=1.0,
        w_center=1.0,
        w_scale=1.0,
        w_shape=1.0,
        w_global=0.3,
        w_local_center=0.5,
        w_local_shape=10.0,
        w_local_cov=10.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_projections = num_projections
        self.global_scale = global_scale
        self.local_target_scale = local_target_scale

        self.w_center = w_center
        self.w_scale = w_scale
        self.w_shape = w_shape
        self.w_global = w_global
        self.w_local_center = w_local_center
        self.w_local_shape = w_local_shape
        self.w_local_cov = w_local_cov

        # Fixed projection matrix so the loss curve is monitorable
        W_init = torch.randn(self.embed_dim, self.num_projections)
        W_fixed = F.normalize(W_init, p=2, dim=0)
        self.register_buffer("W_fixed", W_fixed)

    def _global_macro_visreg(self, points):
        N, D = points.shape
        device = points.device
        if N < 4:
            zero = torch.zeros((), device=device)
            return zero, {"global_center": zero, "global_scale": zero, "global_shape": zero}

        mu = points.mean(dim=0)
        L_center = mu.pow(2).mean()

        p_cent = points - mu
        std = p_cent.std(dim=0, unbiased=False) + 1e-6
        L_scale = (self.global_scale - std).pow(2).mean()

        p_norm = p_cent / std.detach()

        proj = p_norm @ self.W_fixed
        proj_sorted = torch.sort(proj, dim=0).values

        u = torch.arange(1, N + 1, device=device).float() / (N + 1)
        target = (self.global_scale * torch.distributions.Normal(0, 1).icdf(u)).unsqueeze(-1)
        L_shape = (proj_sorted - target).pow(2).mean()

        weighted = self.w_center * L_center + self.w_scale * L_scale + self.w_shape * L_shape
        raw = {
            "global_center": L_center.detach(),
            "global_scale": L_scale.detach(),
            "global_shape": L_shape.detach(),
        }
        return weighted, raw

    def forward(self, z, y):
        device = z.device
        unique_classes = torch.unique(y)

        loss_local = torch.zeros((), device=device)
        raw_local_center_sum = torch.zeros((), device=device)
        raw_local_cov_sum = torch.zeros((), device=device)
        raw_local_shape_sum = torch.zeros((), device=device)
        centroids = []
        count = 0

        for c in unique_classes:
            if c.item() < 0:
                continue

            mask = y == c
            zc = z[mask]
            if zc.shape[0] < 4:
                continue

            N, D = zc.shape

            mu_c = zc.mean(dim=0, keepdim=True)
            centroids.append(mu_c.squeeze(0))

            L_local_center = mu_c.pow(2).mean()

            zc_cent = zc - mu_c
            std_c = zc_cent.std(dim=0, unbiased=False) + 1e-6
            zc_norm = zc_cent / std_c.detach()

            if N > 1:
                cov_c = (zc_cent.t() @ zc_cent) / (N - 1)
                target_cov = (self.local_target_scale ** 2) * torch.eye(D, device=device)
                L_cov = F.mse_loss(cov_c, target_cov)
            else:
                L_cov = torch.zeros((), device=device)

            # Fixed-projection slice step
            p = zc_norm @ self.W_fixed
            p_sorted = torch.sort(p, dim=0).values

            u = torch.arange(1, N + 1, device=device).float() / (N + 1)
            target = torch.distributions.Normal(0, 1).icdf(u).unsqueeze(-1)
            L_local_shape = (p_sorted - target).pow(2).mean()

            loss_local = loss_local + (
                self.w_local_center * L_local_center
                + self.w_local_shape * L_local_shape
                + self.w_local_cov * L_cov
            )

            raw_local_center_sum = raw_local_center_sum + L_local_center.detach()
            raw_local_cov_sum = raw_local_cov_sum + L_cov.detach()
            raw_local_shape_sum = raw_local_shape_sum + L_local_shape.detach()

            count += 1

        if count >= 4:
            centroid_points = torch.stack(centroids, dim=0)
            loss_global, raw_global = self._global_macro_visreg(centroid_points)
        else:
            zero = torch.zeros((), device=device)
            loss_global = zero
            raw_global = {"global_center": zero, "global_scale": zero, "global_shape": zero}

        if count == 0:
            return self.w_global * loss_global, {
                "global_loss": loss_global.detach(),
                "local_loss": torch.tensor(0.0, device=device),
                **raw_global,
                "raw_local_center": torch.tensor(0.0, device=device),
                "raw_local_cov": torch.tensor(0.0, device=device),
                "raw_local_shape": torch.tensor(0.0, device=device),
            }

        loss_local = loss_local / count
        total_loss = self.w_global * loss_global + loss_local

        return total_loss, {
            "global_loss": loss_global.detach(),
            "local_loss": loss_local.detach(),
            **raw_global,
            "raw_local_center": (raw_local_center_sum / count).detach(),
            "raw_local_cov": (raw_local_cov_sum / count).detach(),
            "raw_local_shape": (raw_local_shape_sum / count).detach(),
        }


# --------------------------------------------------------------------------- #
# Supervised contrastive (supervised SimCLR)                                  #
# --------------------------------------------------------------------------- #
def supcon_loss(feats, labels, temp=0.1):
    """feats: (N, D) L2-normalised; labels: (N,).  Standard SupCon (Khosla 2020)."""
    n = feats.size(0)
    sim = feats @ feats.t() / temp
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()
    self_mask = torch.eye(n, dtype=torch.bool, device=feats.device)
    exp_sim = torch.exp(sim).masked_fill(self_mask, 0.0)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

    pos = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~self_mask
    pos_count = pos.sum(dim=1).clamp(min=1)
    mean_log_prob_pos = (pos * log_prob).sum(dim=1) / pos_count
    return -mean_log_prob_pos.mean()


# --------------------------------------------------------------------------- #
# Configurable hybrid contrastive loss (the exp-34e "one class" of the note)   #
# --------------------------------------------------------------------------- #
def _critic_matrix(z, kind, tau):
    """Pairwise critic g_ij on the (2N x D) batch of RAW embeddings.

    cosine   : <z_i/|z_i|, z_j/|z_j|> / tau         (unit sphere, scale-free)
    bilinear : <z_i, z_j> / tau                     (raw Euclidean inner product)
    distance : (-1/2 |z_i - z_j|^2) / tau           (raw squared distance)

    The `bilinear` and `distance` critics live in the SAME raw space that SIGReg
    regularises, so the interaction and marginal terms share one geometry; the
    `cosine` critic reproduces the sphere geometry of NT-Xent / SupCon.
    """
    if kind == "cosine":
        zc = F.normalize(z, dim=1)
        return (zc @ zc.t()) / tau
    if kind == "bilinear":
        return (z @ z.t()) / tau
    if kind == "distance":
        return (-0.5 * torch.cdist(z, z).pow(2)) / tau
    raise ValueError(f"unknown critic {kind!r}")


def _softmax_interaction(g, pos, self_mask):
    """Self-normalised (NT-Xent / SupCon) estimator of PMI from a critic matrix."""
    g = g - g.max(dim=1, keepdim=True).values.detach()
    exp_g = torch.exp(g).masked_fill(self_mask, 0.0)
    log_prob = g - torch.log(exp_g.sum(dim=1, keepdim=True) + 1e-12)
    pos_count = pos.sum(dim=1).clamp(min=1)
    return -((pos * log_prob).sum(dim=1) / pos_count).mean()


def _nplm_interaction(g, pos, self_mask, clamp=30.0):
    """Un-normalised maximum-likelihood (NPLM) estimator of PMI.

    L = E_ref[e^g - 1] - E_pos[g], with the reference = independent (cross-group)
    pairs and the positives = same-group pairs.  Minimiser g* = log p(x,x')/p(x)p(x')
    with E_ref[e^{g*}] = 1, so the ABSOLUTE (calibrated) log-ratio is recovered.

    The reference explicitly EXCLUDES positives (and the diagonal): with supervised
    positives those same-group pairs are not independent draws, so counting them
    would bias E_ref[e^g] -- the NPLM analogue of SupCon's positive masking.

    Stabilisation is by CLAMPING the exponent, not by subtracting a per-row max:
    a subtracted constant does not cancel here (there is no enclosing log-ratio as
    in softmax), so it would shift the recovered critic by that constant and break
    the absolute calibration.  Clamping only affects pathologically large entries
    and leaves g* unchanged in the normal range.
    """
    ref = (~pos) & (~self_mask)
    exp_term = torch.exp(g.clamp(max=clamp)) - 1.0
    ref_mean = exp_term[ref].mean() if ref.any() else g.new_zeros(())
    pos_mean = g[pos].mean() if pos.any() else g.new_zeros(())
    return ref_mean - pos_mean


class HybridContrastiveLoss(nn.Module):
    """One configurable contrastive loss spanning the design cube of the note.

    L = L_interaction(critic g ; positives P) + lam * L_marginal(z)

    Four orthogonal knobs (every existing exp-34 arm is a corner):
      positives  : 'instance'   two augmented views of the same sample
                   'supervised' same class label
      critic     : 'cosine' | 'bilinear' | 'distance'   (see _critic_matrix)
      estimator  : 'softmax' (NT-Xent / SupCon)  |  'nplm' (calibrated log-ratio)
      marginal   : 'none' | 'sigreg'  |  'classwise_sigreg'

    Recipes:
      SimCLR              instance / cosine   / softmax / none
      SupCon              supervised / cosine / softmax / none
      exp-34e hybrid      instance / cosine   / softmax / sigreg
      34g supcon+sigreg   supervised / cosine / softmax / sigreg
      NPLM geometry       */ bilinear|distance/ nplm    / sigreg   (log-likelihood space)
      class-wise NPLM     supervised / distance / nplm  / classwise_sigreg

    forward(z, labels, means=None, class_sigma=None):
      z       : (2N, D) RAW embeddings, the two views stacked as in train.py.
      labels  : (2N,) positive-group ids -- instance ids for 'instance'
                (e.g. cat([arange(N), arange(N)])), class labels for 'supervised'.
                For 'classwise_sigreg' pass the CLASS labels here regardless.
      means   : (C, D) class means, required for 'classwise_sigreg'.
    """

    def __init__(self, positives="instance", critic="bilinear",
                 estimator="nplm", marginal="sigreg", tau=1.0, lam=1.0,
                 n_slices=64):
        super().__init__()
        if positives not in ("instance", "supervised"):
            raise ValueError(f"unknown positives {positives!r}")
        if estimator not in ("softmax", "nplm"):
            raise ValueError(f"unknown estimator {estimator!r}")
        if marginal not in ("none", "sigreg", "classwise_sigreg"):
            raise ValueError(f"unknown marginal {marginal!r}")
        self.positives = positives
        self.critic = critic
        self.estimator = estimator
        self.marginal = marginal
        self.tau = tau
        self.lam = lam
        self.n_slices = n_slices

    def interaction(self, z, labels):
        n = z.size(0)
        self_mask = torch.eye(n, dtype=torch.bool, device=z.device)
        pos = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~self_mask
        g = _critic_matrix(z, self.critic, self.tau)
        if self.estimator == "softmax":
            return _softmax_interaction(g, pos, self_mask)
        return _nplm_interaction(g, pos, self_mask)

    def marginal_term(self, z, labels, means, class_sigma):
        if self.marginal == "none":
            return z.new_zeros(())
        if self.marginal == "sigreg":
            return sigreg_loss(z, n_slices=self.n_slices)
        if means is None:
            raise ValueError("classwise_sigreg requires `means`")
        return classwise_sigreg_loss(z, labels, means, n_slices=self.n_slices,
                                     class_sigma=class_sigma)

    def forward(self, z, labels, means=None, class_sigma=None):
        inter = self.interaction(z, labels)
        marg = self.marginal_term(z, labels, means, class_sigma)
        total = inter + self.lam * marg
        return total, {"interaction": inter.detach(), "marginal": marg.detach()}
