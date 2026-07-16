"""
SparKer-lite: sparse, self-organizing Gaussian-kernel Neyman-Pearson test
(Grosso et al., arXiv:2511.03095), for dataset-level anomaly detection in a
frozen latent space.

Model (paper Eq. 2):   f(x) = a^T [ p[k](x) . k(x) ],
                       k_i(x) = exp(-||x - mu_i||^2 / (2 sigma^2)),
                       p_i(x) = k_i(x) / sum_j k_j(x)          (kernel gating)
NP loss (paper Eq. 3): L_NP[f] = sum_R w (e^f - 1) - sum_D f,  w = N_D / N_R
Statistic:             t_NP = -2 L_NP, recorded at several sigma checkpoints
                       of a linear annealing schedule.

Power protocol: calibrate per-sigma p-values on anomaly-free toy datasets,
aggregate p(D) = -1/2 min_sigma log p_sigma - 1/2 mean_sigma log p_sigma,
detect when the aggregate exceeds the (1-alpha) quantile of its own null.
"""
import numpy as np
import torch

from .config import DEVICE

F_CLAMP = 20.0          # keeps exp(f) finite early in training


def median_pairwise(X, n=1024, seed=0):
    """Median pairwise distance of a subsample (kernel width heuristic)."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(X), generator=g)[:n].to(X.device)
    sub = X[idx]
    d = torch.cdist(sub, sub)
    iu = torch.triu_indices(len(sub), len(sub), offset=1)
    return d[iu[0], iu[1]].median().item()


def np_test_stats(D, R, M=16, steps=300, sigma0=None, sigma_ratio=10.0,
                  n_checkpoints=3, lr=0.05, seed=0):
    """
    Train the kernel ensemble on the NP loss for one data-vs-reference pair.

    D: (N_D, dim) data sample (may contain anomalies), torch tensor on DEVICE.
    R: (N_R, dim) anomaly-free reference, torch tensor on DEVICE.
    Returns the list of t_NP values at `n_checkpoints` sigma checkpoints.
    """
    torch.manual_seed(seed)
    w = len(D) / len(R)
    if sigma0 is None:
        sigma0 = median_pairwise(D, seed=seed)
    sigmaT = sigma0 / sigma_ratio
    g = torch.Generator().manual_seed(seed)
    mu = D[torch.randperm(len(D), generator=g)[:M].to(D.device)] \
        .clone().requires_grad_(True)
    a = torch.zeros(M, device=D.device, requires_grad=True)
    opt = torch.optim.Adam([mu, a], lr=lr)

    def f(X, sigma):
        k = torch.exp(-0.5 * torch.cdist(X, mu).pow(2) / sigma ** 2)
        p = k / (k.sum(dim=1, keepdim=True) + 1e-12)
        return ((p * k) @ a).clamp(-F_CLAMP, F_CLAMP)

    ck = {int(round(steps * (i + 1) / n_checkpoints))
          for i in range(n_checkpoints)}
    ts = []
    for t in range(1, steps + 1):
        sigma = sigma0 + (sigmaT - sigma0) * t / steps
        loss = w * (torch.exp(f(R, sigma)) - 1).sum() - f(D, sigma).sum()
        opt.zero_grad(); loss.backward(); opt.step()
        if t in ck:
            with torch.no_grad():
                tnp = -2.0 * (w * (torch.exp(f(R, sigma)) - 1).sum()
                              - f(D, sigma).sum())
            ts.append(float(tnp))
    return ts


def krr_term(R, sigmas):
    """Unbiased within-reference kernel means, one per bandwidth."""
    with torch.no_grad():
        d2 = torch.cdist(R, R).pow(2)
        n = len(R)
        out = []
        for s in sigmas:
            k = torch.exp(-d2 / (2 * s * s))
            out.append(float((k.sum() - k.diagonal().sum()) / (n * (n - 1))))
    return out


def mmd2_multi_stats(D, R, sigmas, krr):
    """
    Unbiased multi-bandwidth MMD^2 between data D and reference R (Gaussian
    kernels).  Same role as np_test_stats: one statistic per scale, to be
    aggregated with aggregate_pvalues against toy-calibrated nulls.
    """
    with torch.no_grad():
        dDD2 = torch.cdist(D, D).pow(2)
        dDR2 = torch.cdist(D, R).pow(2)
        n = len(D)
        out = []
        for s, krr_s in zip(sigmas, krr):
            kDD = torch.exp(-dDD2 / (2 * s * s))
            udd = (kDD.sum() - kDD.diagonal().sum()) / (n * (n - 1))
            kdr = torch.exp(-dDR2 / (2 * s * s)).mean()
            out.append(float(udd - 2 * kdr + krr_s))
    return out


def aggregate_pvalues(ts, null_ts):
    """
    Multi-scale aggregate score of one realization against per-sigma nulls.

    ts: (n_ck,) statistics of the realization.
    null_ts: (S0, n_ck) statistics of the anomaly-free calibration toys.
    p_sigma via add-one empirical tail; aggregate =
    -1/2 min log p - 1/2 mean log p (larger = more anomalous).
    """
    ts = np.asarray(ts)
    null_ts = np.asarray(null_ts)
    p = (1.0 + (null_ts >= ts[None, :]).sum(axis=0)) / (len(null_ts) + 1.0)
    logp = np.log(p)
    return -0.5 * logp.min() - 0.5 * logp.mean()


def clopper_pearson(k, n, cl=0.68):
    """Central Clopper-Pearson interval for k successes of n."""
    from scipy.stats import beta
    lo = beta.ppf((1 - cl) / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - (1 - cl) / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)
