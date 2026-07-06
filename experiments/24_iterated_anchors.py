"""
Iterated anchor discovery: repeat the discover -> fine-tune loop.

Round r (SIGReg space):
  1. embed the full train set; anchors = seen means + all anchors discovered
     in earlier rounds.
  2. pool = points whose distance to EVERY current anchor exceeds the
     tau-quantile of labeled seen-class distances (recalibrated each round).
  3. BIC-select k, k-means the pool, append the new centers as anchors.
  4. pseudo-labels: every point ever pooled is assigned to its nearest
     discovered anchor; fine-tune classwise SIGReg + proto on labeled seen
     (true labels) + pooled (pseudo-labels).
  5. evaluate: unseen-vs-rest margin AUC and per-held-out-class matched-anchor
     AUC, printed per round.

Hypothesis: round 1 pulls the discoverable part of each novel class onto its
anchor; the round-2 pool should be purer and catch what was hiding inside the
seen shells.

Outputs:
    plots/iterated_anchors_<dataset>.png
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from supersig.config import plot_path, DATA_DIR, DEVICE
from supersig.data import (
    get_cifar_loaders, cifar_balanced_loader, BalancedBatchSampler, _cifar_spec,
)
from supersig.models import CIFARResNetBackbone
from supersig.losses import make_anchors
from supersig.train import train_sigreg_hybrid, collect_embeddings, REP_WEIGHT

PAIR_DIST = 5.0
HOLDOUT_SETS_ALL = {
    "cifar10": {1: [4], 2: [4, 9], 3: [0, 4, 9]},
    "cifar100": {1: [4], 3: [4, 30, 70],
                 10: [4, 14, 24, 34, 44, 54, 64, 74, 84, 94],
                 20: [4 + 5 * i for i in range(20)]},
}


class PseudoDataset(Dataset):
    def __init__(self, base, indices, labels):
        self.base, self.indices, self.labels = base, indices, labels

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, _ = self.base[self.indices[i]]
        return img, int(self.labels[i])


def kmeans(X, k, iters=30, seed=0):
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
    n, d = X.shape
    best = None
    for k in range(1, kmax + 1):
        C, a = kmeans(X, k, seed=seed + k)
        ll = -0.5 * ((X - C[a]) ** 2).sum().item() - 0.5 * n * d * math.log(2 * math.pi)
        bic = -2 * ll + k * d * math.log(n)
        if best is None or bic < best[1]:
            best = (k, bic, C, a)
    return best[0], best[2], best[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ssl-epochs", type=int, default=None)
    ap.add_argument("--ft-epochs", type=int, default=None)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arch", default="resnet20")
    ap.add_argument("--ks", default=None)
    ap.add_argument("--tau-quantile", type=float, default=0.95)
    ap.add_argument("--kmax", type=int, default=4)
    ap.add_argument("--emb-dim", type=int, default=None)
    ap.add_argument("--merge-dist", type=float, default=0.0,
                    help="merge discovered anchors closer than this (0 = off)")
    ap.add_argument("--sigreg-weight", type=float, default=None)
    ap.add_argument("--out-tag", default="")
    args = ap.parse_args()
    ds_name = args.dataset
    n_classes = 100 if ds_name == "cifar100" else 10
    emb_dim = args.emb_dim or (100 if ds_name == "cifar100" else 16)
    w, slices = (1.0, 64) if ds_name == "cifar100" else (20.0, 256)
    if args.sigreg_weight is not None:
        w = args.sigreg_weight
    pretrain = ds_name
    holdout_sets = HOLDOUT_SETS_ALL[ds_name]
    ks = [int(x) for x in (args.ks or ("1,2,3" if ds_name == "cifar10"
                                       else "10,20")).split(",")]
    ssl_ep = args.ssl_epochs or (2 if args.quick else 10)
    ft_ep = args.ft_epochs or (1 if args.quick else 5)
    rep_w = REP_WEIGHT * 45.0 / (n_classes * (n_classes - 1) / 2)

    if ds_name == "cifar100":
        from torchvision import datasets as tvd
        names = tvd.CIFAR100(DATA_DIR, train=False, download=True).classes
    else:
        names = ["airplane", "automobile", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck"]

    train_loader, test_loader = get_cifar_loaders(quick=args.quick, limit=args.limit,
                                                  dataset=ds_name)
    train_eval_loader = DataLoader(train_loader.dataset, batch_size=256,
                                   shuffle=False, num_workers=2)
    cls, plain, _ = _cifar_spec(ds_name)
    base = cls(DATA_DIR, train=True, download=True, transform=plain)

    history = {}
    for k in ks:
        holdouts = set(holdout_sets[k])
        seen = [c for c in range(n_classes) if c not in holdouts]
        print(f"\n===== k={k} ({', '.join(names[c] for c in sorted(holdouts))}) =====")
        torch.manual_seed(args.seed); np.random.seed(args.seed)
        backbone = CIFARResNetBackbone(emb_dim, arch=args.arch,
                                       pretrain=pretrain).to(DEVICE)
        means = make_anchors(PAIR_DIST / math.sqrt(2.0), emb_dim=emb_dim,
                             n_classes=n_classes).clone()
        train_sigreg_hybrid(backbone, cifar_balanced_loader(
            ds_name, holdout=holdouts, quick=args.quick, limit=args.limit),
            ssl_ep, means, mode="repulse", disc="proto", alpha=1.0,
            rep_weight=rep_w, sigreg_weight=w, n_slices=slices)

        cur_means = means.detach().clone()
        pooled = np.zeros(len(train_loader.dataset), dtype=bool)
        history[k] = []
        for r in range(1, args.rounds + 1):
            tr_embs, tr_lab = collect_embeddings(backbone, train_eval_loader)
            z = torch.as_tensor(tr_embs, device=DEVICE)
            anchor_mat = torch.cat([cur_means[seen], cur_means[n_classes:]]) \
                if cur_means.size(0) > n_classes else cur_means[seen]
            dmin = torch.cdist(z, anchor_mat).min(1).values
            is_seen_lab = np.isin(tr_lab, seen)
            tau = torch.quantile(dmin[torch.as_tensor(is_seen_lab, device=DEVICE)],
                                 args.tau_quantile)
            pool = (dmin > tau).cpu().numpy()
            purity = (~is_seen_lab[pool]).mean() if pool.any() else float("nan")
            kmax = max(args.kmax, k + 2)
            khat, centers, _ = bic_select(z[torch.as_tensor(pool, device=DEVICE)],
                                          kmax=kmax, seed=args.seed + r)
            cur_means = torch.cat([cur_means, centers.detach()], dim=0)
            pooled |= pool
            disc = cur_means[n_classes:]
            # merge discovered anchors closer than --merge-dist (weighted by members)
            if args.merge_dist > 0 and disc.size(0) > 1:
                memb = torch.cdist(z[torch.as_tensor(pooled, device=DEVICE)],
                                   disc).argmin(1)
                counts = torch.bincount(memb, minlength=disc.size(0)).float() + 1e-6
                parent = list(range(disc.size(0)))

                def find(a):
                    while parent[a] != a:
                        parent[a] = parent[parent[a]]
                        a = parent[a]
                    return a

                dmat = torch.cdist(disc, disc)
                for i in range(disc.size(0)):
                    for j in range(i + 1, disc.size(0)):
                        if dmat[i, j] < args.merge_dist:
                            parent[find(i)] = find(j)
                groups = {}
                for i in range(disc.size(0)):
                    groups.setdefault(find(i), []).append(i)
                merged = torch.stack([
                    (disc[g] * counts[g, None]).sum(0) / counts[g].sum()
                    for g in [torch.as_tensor(v, device=DEVICE)
                              for v in groups.values()]])
                if merged.size(0) < disc.size(0):
                    print(f"    merged {disc.size(0)} -> {merged.size(0)} anchors")
                cur_means = torch.cat([cur_means[:n_classes], merged], dim=0)
                disc = cur_means[n_classes:]
            # refresh pseudo-labels: nearest discovered anchor for every pooled pt
            p_idx = np.where(pooled)[0]
            p_lab = n_classes + torch.cdist(
                z[torch.as_tensor(pooled, device=DEVICE)], disc).argmin(1).cpu().numpy()
            lab_idx = np.where(is_seen_lab)[0]
            ft_idx = np.concatenate([lab_idx, p_idx])
            ft_lab = np.concatenate([tr_lab[lab_idx], p_lab])
            n_pb = len(seen) + len(disc) if n_classes == 10 else 25
            sampler = BalancedBatchSampler(list(ft_lab), n_classes=n_pb,
                                           n_per_class=24)
            ft_loader = DataLoader(PseudoDataset(base, ft_idx, ft_lab),
                                   batch_sampler=sampler, num_workers=2)
            train_sigreg_hybrid(backbone, ft_loader, ft_ep, cur_means,
                                mode="repulse", disc="proto", alpha=1.0,
                                rep_weight=rep_w, sigreg_weight=w, n_slices=slices)
            cur_means = cur_means.detach()

            te_embs, te_lab = collect_embeddings(backbone, test_loader)
            zt = torch.as_tensor(te_embs, device=DEVICE)
            d_seen = torch.cdist(zt, cur_means[seen]).min(1).values
            d_new = torch.cdist(zt, cur_means[n_classes:]).min(1).values
            is_unseen = np.isin(te_lab, list(holdouts)).astype(int)
            after = roc_auc_score(is_unseen, (d_seen - d_new).cpu().numpy())
            d_each = torch.cdist(zt, cur_means[n_classes:])
            per_class = {}
            for c in sorted(holdouts):
                counts = [int(((te_lab == c) &
                               (d_each.argmin(1).cpu().numpy() == j)).sum())
                          for j in range(len(disc))]
                j = int(np.argmax(counts))
                per_class[c] = roc_auc_score((te_lab == c).astype(int),
                                             (-d_each[:, j]).cpu().numpy())
            mean_pc = float(np.mean(list(per_class.values())))
            history[k].append(dict(r=r, pool=int(pool.sum()), purity=purity,
                                   khat=khat, after=after, mean_pc=mean_pc))
            pc = "  ".join(f"{names[c]}={a:.3f}" for c, a in per_class.items())
            print(f"  round {r}: pool={pool.sum()} purity={purity:.3f} k-hat={khat}"
                  f"  margin AUC={after:.4f}  mean anchor AUC={mean_pc:.4f}")
            if len(per_class) <= 5:
                print(f"           {pc}")

    print("\n===== ITERATED-ANCHOR SUMMARY =====")
    print(f"{'k':>3}{'round':>6}{'pool':>7}{'purity':>8}{'k-hat':>6}"
          f"{'margin':>9}{'mean-pc':>9}")
    for k in ks:
        for h in history[k]:
            print(f"{k:>3}{h['r']:>6}{h['pool']:>7}{h['purity']:>8.3f}"
                  f"{h['khat']:>6}{h['after']:>9.4f}{h['mean_pc']:>9.4f}")

    plt.figure(figsize=(7, 5))
    for i, k in enumerate(ks):
        rs = [h["r"] for h in history[k]]
        plt.plot(rs, [h["after"] for h in history[k]], f"C{i}-o", lw=2,
                 label=f"k={k} margin AUC")
        plt.plot(rs, [h["mean_pc"] for h in history[k]], f"C{i}--^", lw=1.5,
                 alpha=0.8, label=f"k={k} mean anchor AUC")
    plt.xticks(range(1, args.rounds + 1))
    plt.xlabel("discovery round"); plt.ylabel("AUC")
    plt.title(f"{ds_name}: iterated anchor discovery")
    plt.legend(fontsize=9); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(plot_path(f"iterated_anchors_{ds_name}{args.out_tag}.png"), dpi=150); plt.close()
    print(f"\n  saved {plot_path(f'iterated_anchors_{ds_name}{args.out_tag}.png')}")
    print("Done.")


if __name__ == "__main__":
    main()
