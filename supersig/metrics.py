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
