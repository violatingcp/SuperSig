"""Distributional diagnostics for the learned latent spaces."""
import torch
import torch.nn.functional as F

from .config import DEVICE
from .losses import standard_normal_quantiles


@torch.no_grad()
def sliced_gaussianity(z, n_slices=256, n_null=32, seed=0):
    """
    Calibrated sliced-Wasserstein Gaussianity index of a point cloud.

    Projects `z` (n, d) onto `n_slices` random unit directions, standardizes each
    1-D projection to mean 0 / std 1, and averages the squared Wasserstein-2
    distance between the sorted projections and the standard-normal quantiles.
    Per-slice standardization removes location and scale direction by direction,
    so the statistic responds to *shape* only (skew, multimodality, heavy tails).
    By Cramer-Wold, all 1-D projections Gaussian <=> the joint is Gaussian.

    A finite sample of a true Gaussian never scores 0, so the raw statistic is
    calibrated against `n_null` synthetic N(0, I) clouds of the same (n, d):

        ratio = stat / null_mean      ~1 = as Gaussian as chance allows

    Returns dict(stat, null_mean, null_std, ratio, z_score, skew, ex_kurtosis),
    where skew / ex_kurtosis are means of |slice skewness| and |slice excess
    kurtosis| over the same projections.

    Limitation: distributions whose random projections Gaussianize by the CLT
    (e.g. a product of independent uniforms) score ~1 despite being non-Gaussian.
    The failure modes that matter for embeddings -- sub-clusters / multimodality,
    heavy tails, outliers, skew -- are all detected (validated: bimodal ~50x,
    t_3 tails ~9x, lognormal ~47x, true Gaussian ~1x).
    """
    z = torch.as_tensor(z, dtype=torch.float32).to(DEVICE)
    n, d = z.shape
    gen = torch.Generator().manual_seed(seed)
    dirs = F.normalize(torch.randn(d, n_slices, generator=gen), dim=0).to(DEVICE)
    q = standard_normal_quantiles(n, DEVICE).unsqueeze(1)

    def _standardized(cloud):
        p = cloud @ dirs
        return (p - p.mean(0)) / (p.std(0) + 1e-12)

    def _stat(cloud):
        p, _ = torch.sort(_standardized(cloud), dim=0)
        return ((p - q) ** 2).mean().item()

    stat = _stat(z)
    null = torch.tensor([_stat(torch.randn(n, d, generator=gen).to(DEVICE))
                         for _ in range(n_null)])
    null_mean, null_std = null.mean().item(), null.std().item()

    p = _standardized(z)
    skew = (p ** 3).mean(0).abs().mean().item()
    ex_kurt = ((p ** 4).mean(0) - 3.0).abs().mean().item()

    return {
        "stat": stat, "null_mean": null_mean, "null_std": null_std,
        "ratio": stat / null_mean,
        "z_score": (stat - null_mean) / (null_std + 1e-12),
        "skew": skew, "ex_kurtosis": ex_kurt,
    }


def classwise_gaussianity(embs, labels, **kw):
    """{class label: sliced_gaussianity(...) of that class's embeddings}."""
    import numpy as np
    return {int(c): sliced_gaussianity(embs[labels == c], **kw)
            for c in np.unique(labels)}


@torch.no_grad()
def gaussianity_summary(embs, labels, classes, n_slices=256, n_null=16, seed=0):
    """
    Aggregate per-class Gaussianity / geometry metrics of an embedding space.

    For each class in `classes` (rows of `embs` selected by `labels`) compute
    the empirical covariance eigenspectrum, the per-dimension RMS (sqrt of the
    mean variance -- ~1 in a calibrated latent), the largest off-diagonal
    correlation, and the calibrated sliced-Wasserstein shape statistics, then
    aggregate across classes.  Centroid distances are between the empirical
    class means; `separation` divides the closest centroid pair by the mean
    class RMS (how many sigma apart the two nearest classes sit).
    """
    import numpy as np
    z = torch.as_tensor(embs, dtype=torch.float32).to(DEVICE)
    labels = np.asarray(labels)
    eig_all, rms, corr_max, cond = [], [], [], []
    ratios, skews, kurts, mus = [], [], [], []
    for c in classes:
        zc = z[torch.as_tensor(labels == c).to(DEVICE)]
        mu = zc.mean(0)
        mus.append(mu)
        X = zc - mu
        S = X.t() @ X / max(len(zc) - 1, 1)
        ev = torch.linalg.eigvalsh(S)
        eig_all += ev.tolist()
        cond.append((ev.max() / ev.min().clamp_min(1e-12)).item())
        rms.append(X.pow(2).mean().sqrt().item())
        Xn = X / (X.std(0) + 1e-12)
        C = Xn.t() @ Xn / max(len(zc) - 1, 1)
        off = C - torch.diag(torch.diag(C))
        corr_max.append(off.abs().max().item())
        g = sliced_gaussianity(zc, n_slices=n_slices, n_null=n_null, seed=seed)
        ratios.append(g["ratio"]); skews.append(g["skew"]); kurts.append(g["ex_kurtosis"])
    M = torch.stack(mus)
    iu = torch.triu_indices(len(classes), len(classes), offset=1)
    pdist = torch.cdist(M, M)[iu[0], iu[1]]
    return {
        "eig_min": min(eig_all), "eig_max": max(eig_all),
        "eig_cond_max": max(cond),
        "rms_min": min(rms), "rms_mean": float(np.mean(rms)), "rms_max": max(rms),
        "corr_max": max(corr_max),
        "sw_ratio_mean": float(np.mean(ratios)), "sw_ratio_max": max(ratios),
        "skew_mean": float(np.mean(skews)), "kurt_mean": float(np.mean(kurts)),
        "cdist_min": pdist.min().item(), "cdist_mean": pdist.mean().item(),
        "separation": pdist.min().item() / (float(np.mean(rms)) + 1e-12),
    }


@torch.no_grad()
def mahalanobis_novelty(tr_embs, tr_lab, te_embs, seen, shrink=0.1, device=None):
    """
    Lee-et-al-style novelty scores from empirically fitted seen-class Gaussians.

    Returns (tied_scores, perclass_scores, (eig_min, eig_med, eig_max)) where the
    eigenvalues are of the pooled within-class covariance -- all ~1 iff the
    latent is a true unit-Mahalanobis space.
    """
    import numpy as np
    dev = device or DEVICE
    zt = torch.as_tensor(tr_embs, dtype=torch.float32).to(dev)
    z = torch.as_tensor(te_embs, dtype=torch.float32).to(dev)
    d = zt.size(1)
    mus, diffs, chols = [], [], []
    for c in seen:
        zc = zt[torch.as_tensor(np.asarray(tr_lab) == c).to(dev)]
        mu = zc.mean(0)
        mus.append(mu)
        diffs.append(zc - mu)
        S = (zc - mu).t() @ (zc - mu) / max(len(zc) - 1, 1)
        S = (1 - shrink) * S + shrink * (torch.trace(S) / d) * torch.eye(d, device=dev)
        chols.append(torch.linalg.cholesky(S))
    D = torch.cat(diffs)
    S_tied = D.t() @ D / max(len(D) - 1, 1) + 1e-3 * torch.eye(d, device=dev)
    L_tied = torch.linalg.cholesky(S_tied)

    def min_m2(Ls):
        m2 = []
        for mu, L in zip(mus, Ls):
            w = torch.linalg.solve_triangular(L, (z - mu).t(), upper=False)
            m2.append((w ** 2).sum(0))
        return torch.stack(m2).min(0).values.sqrt().cpu().numpy()

    evals = torch.linalg.eigvalsh(S_tied)
    eigs = (evals.min().item(), evals.median().item(), evals.max().item())
    return min_m2([L_tied] * len(mus)), min_m2(chols), eigs
