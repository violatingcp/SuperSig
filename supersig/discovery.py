"""
Open-world anchor discovery on a SIGReg latent — the settled pipeline from
experiments 23/24.

Loop (per round): embed the full train set -> outlier pool (points beyond the
tau-quantile shell of every current anchor) -> BIC-selected k-means -> append
cluster centers as anchors -> optional merge -> pseudo-label pooled points by
nearest discovered anchor -> fine-tune with classwise SIGReg + proto, with
discovered-discovered anchor pairs EXEMPT from repulsion (they huddle around
the novel class instead of partitioning it; exps 24).
"""
import math
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score

from .config import DEVICE
from .data import BalancedBatchSampler
from .train import train_sigreg_hybrid, collect_embeddings


class PseudoDataset(Dataset):
    """Subset of a base dataset with externally supplied (pseudo-)labels."""

    def __init__(self, base, indices, labels):
        self.base, self.indices, self.labels = base, indices, labels

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, _ = self.base[self.indices[i]]
        return img, int(self.labels[i])


def kmeans(X, k, iters=30, seed=0):
    """k-means with k-means++ init (Euclidean = Mahalanobis in a calibrated latent)."""
    g = torch.Generator().manual_seed(seed)
    idx = [int(torch.randint(len(X), (1,), generator=g))]
    for _ in range(k - 1):
        d2 = torch.cdist(X, X[idx]).min(1).values ** 2
        idx.append(int(torch.multinomial((d2 / d2.sum()).cpu(), 1, generator=g)))
    C = X[idx].clone()
    for _ in range(iters):
        a = torch.cdist(X, C).argmin(1)
        for j in range(k):
            if (a == j).any():
                C[j] = X[a == j].mean(0)
    return C, torch.cdist(X, C).argmin(1)


def bic_select(X, kmax=4, seed=0):
    """Pick k by BIC under unit-variance Gaussians (the model's own likelihood)."""
    n, d = X.shape
    best = None
    for k in range(1, kmax + 1):
        C, a = kmeans(X, k, seed=seed + k)
        ll = -0.5 * ((X - C[a]) ** 2).sum().item() - 0.5 * n * d * math.log(2 * math.pi)
        bic = -2 * ll + k * d * math.log(n)
        if best is None or bic < best[1]:
            best = (k, bic, C, a)
    return best[0], best[2], best[3]


def merge_anchors(disc, members_assign, merge_dist):
    """Union-find merge of discovered anchors closer than merge_dist (weighted)."""
    if merge_dist <= 0 or disc.size(0) < 2:
        return disc
    counts = torch.bincount(members_assign, minlength=disc.size(0)).float() + 1e-6
    parent = list(range(disc.size(0)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    dmat = torch.cdist(disc, disc)
    for i in range(disc.size(0)):
        for j in range(i + 1, disc.size(0)):
            if dmat[i, j] < merge_dist:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(disc.size(0)):
        groups.setdefault(find(i), []).append(i)
    merged = torch.stack([
        (disc[g] * counts[g, None]).sum(0) / counts[g].sum()
        for g in [torch.as_tensor(v, device=disc.device) for v in groups.values()]])
    if merged.size(0) < disc.size(0):
        print(f"    merged {disc.size(0)} -> {merged.size(0)} anchors")
    return merged


def lid_pool_scores(z, is_seen_lab, k=20, max_ref=4000, seed=0):
    """Levina-Bickel LID of every train point vs the seen-labeled points
    (exp-78/79 pool scorer).  Neighbor ranks 1..k for ALL queries -- rank
    0 is skipped because seen-labeled queries are themselves reference
    points (self at distance 0).  Higher = more novel."""
    ref = z[torch.as_tensor(is_seen_lab, device=z.device)]
    if len(ref) > max_ref:
        g = torch.Generator().manual_seed(seed)
        keep = torch.randperm(len(ref), generator=g)[:max_ref]
        ref = ref[keep.to(z.device)]
    P = torch.cdist(z, ref).topk(k + 1, largest=False).values[:, 1:]
    rk = P[:, -1:].clamp_min(1e-12)
    m = (P[:, :-1].clamp_min(1e-12) / rk).log().mean(1)
    return -1.0 / m.clamp_max(-1e-3)


def np_pool_scores(z, is_seen_lab, M=16, steps=300, sigma_ratio=10.0,
                   lr=0.05, max_ref=4000, seed=0):
    """Neyman-Pearson pool scores (exp-92/93): fit the SparKer NP critic f
    on (full train corpus z vs seen-labeled reference) and score every
    point by its estimated log density ratio.  Higher = more novel."""
    from .sparker import median_pairwise
    ref = z[torch.as_tensor(is_seen_lab, device=z.device)]
    if len(ref) > max_ref:
        g = torch.Generator().manual_seed(seed)
        ref = ref[torch.randperm(len(ref), generator=g)[:max_ref]
                  .to(z.device)]
    torch.manual_seed(seed)
    w = len(z) / len(ref)
    sigma0 = median_pairwise(z, seed=seed)
    sigmaT = sigma0 / sigma_ratio
    g = torch.Generator().manual_seed(seed)
    mu = z[torch.randperm(len(z), generator=g)[:M].to(z.device)] \
        .clone().requires_grad_(True)
    a = torch.zeros(M, device=z.device, requires_grad=True)
    opt = torch.optim.Adam([mu, a], lr=lr)

    def f(X, sigma):
        k = torch.exp(-0.5 * torch.cdist(X, mu).pow(2) / sigma ** 2)
        p = k / (k.sum(dim=1, keepdim=True) + 1e-12)
        return ((p * k) @ a).clamp(-20.0, 20.0)

    sigma = sigma0
    for t in range(1, steps + 1):
        sigma = sigma0 + (sigmaT - sigma0) * t / steps
        loss = w * (torch.exp(f(ref, sigma)) - 1).sum() - f(z, sigma).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return f(z, sigma).detach()


def run_discovery(backbone, means, *, base_ds, train_eval_loader, test_loader,
                  seen, holdouts, dataset_name, rep_weight, sigreg_weight,
                  n_slices, rounds=2, ft_epochs=5, tau_quantile=0.95,
                  kmax=None, merge_dist=3.0, exempt_repulsion=True,
                  names=None, seed=0, pool_score="dist"):
    """
    Iterated anchor discovery.  `means` holds the trained class anchors
    (n_classes rows); returns (extended_means, history) where history is one
    dict per round: pool, purity, khat, margin AUC, mean per-class anchor AUC.
    """
    n_classes = means.size(0)
    cur_means = means.detach().clone()
    pooled = np.zeros(len(train_eval_loader.dataset), dtype=bool)
    history = []
    for r in range(1, rounds + 1):
        tr_embs, tr_lab = collect_embeddings(backbone, train_eval_loader)
        z = torch.as_tensor(tr_embs, device=DEVICE)
        anchor_mat = torch.cat([cur_means[seen], cur_means[n_classes:]]) \
            if cur_means.size(0) > n_classes else cur_means[seen]
        is_seen_lab = np.isin(tr_lab, seen)
        if pool_score == "lid":
            s = lid_pool_scores(z, is_seen_lab, seed=seed + r)
        elif pool_score == "np":
            s = np_pool_scores(z, is_seen_lab, seed=seed + r)
        else:
            s = torch.cdist(z, anchor_mat).min(1).values
        tau = torch.quantile(s[torch.as_tensor(is_seen_lab, device=DEVICE)],
                             tau_quantile)
        pool = (s > tau).cpu().numpy()
        purity = (~is_seen_lab[pool]).mean() if pool.any() else float("nan")
        km = kmax or max(4, len(holdouts) + 2)
        khat, centers, _ = bic_select(z[torch.as_tensor(pool, device=DEVICE)],
                                      kmax=km, seed=seed + r)
        cur_means = torch.cat([cur_means, centers.detach()], dim=0)
        pooled |= pool
        disc = cur_means[n_classes:]
        memb = torch.cdist(z[torch.as_tensor(pooled, device=DEVICE)], disc).argmin(1)
        disc = merge_anchors(disc, memb, merge_dist)
        cur_means = torch.cat([cur_means[:n_classes], disc], dim=0)
        p_idx = np.where(pooled)[0]
        p_lab = n_classes + torch.cdist(
            z[torch.as_tensor(pooled, device=DEVICE)], disc).argmin(1).cpu().numpy()
        lab_idx = np.where(is_seen_lab)[0]
        ft_idx = np.concatenate([lab_idx, p_idx])
        ft_lab = np.concatenate([tr_lab[lab_idx], p_lab])
        n_pb = len(seen) + disc.size(0) if n_classes <= 10 else 25
        sampler = BalancedBatchSampler(list(ft_lab), n_classes=n_pb, n_per_class=24)
        ft_loader = DataLoader(PseudoDataset(base_ds, ft_idx, ft_lab),
                               batch_sampler=sampler, num_workers=2)
        train_sigreg_hybrid(backbone, ft_loader, ft_epochs, cur_means,
                            mode="repulse", disc="proto", alpha=1.0,
                            rep_weight=rep_weight, sigreg_weight=sigreg_weight,
                            n_slices=n_slices,
                            rep_exempt_from=n_classes if exempt_repulsion else None)
        cur_means = cur_means.detach()

        te_embs, te_lab = collect_embeddings(backbone, test_loader)
        zt = torch.as_tensor(te_embs, device=DEVICE)
        d_seen = torch.cdist(zt, cur_means[seen]).min(1).values
        d_each = torch.cdist(zt, cur_means[n_classes:])
        is_unseen = np.isin(te_lab, list(holdouts)).astype(int)
        margin = roc_auc_score(is_unseen, (d_seen - d_each.min(1).values).cpu().numpy())
        per_class = {}
        for c in sorted(holdouts):
            counts = [int(((te_lab == c) & (d_each.argmin(1).cpu().numpy() == j)).sum())
                      for j in range(d_each.size(1))]
            j = int(np.argmax(counts))
            per_class[c] = roc_auc_score((te_lab == c).astype(int),
                                         (-d_each[:, j]).cpu().numpy())
        history.append(dict(round=r, pool=int(pool.sum()), purity=float(purity),
                            khat=khat, n_anchors=int(disc.size(0)),
                            margin=float(margin),
                            per_class={int(c): float(a) for c, a in per_class.items()},
                            mean_pc=float(np.mean(list(per_class.values())))))
        h = history[-1]
        pc = "  ".join(f"{(names[c] if names else c)}={a:.3f}"
                       for c, a in per_class.items()) if len(per_class) <= 5 else ""
        print(f"  round {r}: pool={h['pool']} purity={h['purity']:.3f} "
              f"k-hat={h['khat']} anchors={h['n_anchors']}  "
              f"margin={h['margin']:.4f}  mean-anchor={h['mean_pc']:.4f}  {pc}")
    return cur_means, history
