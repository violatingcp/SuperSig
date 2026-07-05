"""
Discovered anchors: Mahalanobis-style clustering of unlabeled data adds new
class anchors after supervised training, followed by pseudo-labeled fine-tuning.

Pipeline (CIFAR-10, 16-dim tuned recipe, sigreg+proto):
  1. supervised : train the embedding without the held-out classes (labeled
                  seen data only), as in exp 20.
  2. discover   : embed the FULL train set as unlabeled data.  In the
                  self-calibrated space Mahalanobis distance = Euclidean, so
                  the model's own geometry defines everything:
                    - outlier pool = points whose distance to every seen mean
                      exceeds the tau-quantile (default 0.95) of seen-class
                      distances;
                    - cluster count = BIC under unit-variance Gaussians
                      (the model's own likelihood), k in 1..kmax;
                    - new anchors = the k-means centers.
  3. fine-tune  : append the discovered anchors to the means and continue
                  training with classwise SIGReg + proto, where pool points
                  carry pseudo-labels (their cluster) and labeled data keep
                  true labels.

Evaluation (test set):
  - k-hat vs true number of held-out classes; pool purity (fraction of pool
    that is genuinely unseen classes);
  - per held-out class: AUC of -distance to its matched discovered anchor
    (a probe-free, label-free detector for a class nobody ever labeled);
  - unseen-vs-rest AUC before (min dist to seen means) and after
    (dist_seen - dist_new margin).

Outputs:
    plots/discovered_anchors_cifar10.png
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from supersig.config import plot_path, DEVICE
from supersig.data import (
    get_cifar_loaders, cifar_balanced_loader, BalancedBatchSampler, _cifar_spec,
)
from supersig.models import CIFARResNetBackbone
from supersig.losses import make_anchors
from supersig.train import train_sigreg_hybrid, collect_embeddings, REP_WEIGHT
from torch.utils.data import DataLoader, Dataset

EMB_DIM = 16
N_CLASSES = 10
PAIR_DIST = 5.0
DATASET = "cifar10"
HOLDOUT_SETS = {1: [4], 2: [4, 9], 3: [0, 4, 9]}
CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck"]


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
    g = torch.Generator().manual_seed(seed)
    idx = [int(torch.randint(len(X), (1,), generator=g))]
    for _ in range(k - 1):                       # k-means++ init
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ssl-epochs", type=int, default=None)
    ap.add_argument("--ft-epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arch", default="resnet20")
    ap.add_argument("--pretrain", default="cifar10")
    ap.add_argument("--ks", default="1,2,3")
    ap.add_argument("--sigreg-weight", type=float, default=20.0)
    ap.add_argument("--n-slices", type=int, default=256)
    ap.add_argument("--tau-quantile", type=float, default=0.95)
    ap.add_argument("--kmax", type=int, default=4)
    args = ap.parse_args()
    ssl_ep = args.ssl_epochs or (2 if args.quick else 10)
    ft_ep = args.ft_epochs or (1 if args.quick else 5)
    ks = [int(x) for x in args.ks.split(",")]

    train_loader, test_loader = get_cifar_loaders(quick=args.quick, limit=args.limit,
                                                  dataset=DATASET)
    train_eval_loader = DataLoader(train_loader.dataset, batch_size=256,
                                   shuffle=False, num_workers=2)
    cls, plain, _ = _cifar_spec(DATASET)
    from supersig.config import DATA_DIR
    base = cls(DATA_DIR, train=True, download=True, transform=plain)

    results = {}
    for k in ks:
        holdouts = set(HOLDOUT_SETS[k])
        seen = [c for c in range(N_CLASSES) if c not in holdouts]
        hnames = [CIFAR_CLASSES[c] for c in sorted(holdouts)]
        print(f"\n===== k={k} (holdout: {', '.join(hnames)}) =====")

        # 1. supervised training on labeled seen data
        torch.manual_seed(args.seed); np.random.seed(args.seed)
        backbone = CIFARResNetBackbone(EMB_DIM, arch=args.arch,
                                       pretrain=args.pretrain).to(DEVICE)
        means = make_anchors(PAIR_DIST / math.sqrt(2.0), emb_dim=EMB_DIM,
                             n_classes=N_CLASSES).clone()
        rep_w = REP_WEIGHT
        train_sigreg_hybrid(backbone, cifar_balanced_loader(
            DATASET, holdout=holdouts, quick=args.quick, limit=args.limit),
            ssl_ep, means, mode="repulse", disc="proto", alpha=1.0,
            rep_weight=rep_w, sigreg_weight=args.sigreg_weight,
            n_slices=args.n_slices)

        # 2. discovery on the full train set treated as unlabeled
        tr_embs, tr_lab = collect_embeddings(backbone, train_eval_loader)
        z = torch.as_tensor(tr_embs, device=DEVICE)
        seen_means = means.detach()[seen]
        dmin = torch.cdist(z, seen_means).min(1).values
        is_seen_lab = np.isin(tr_lab, seen)
        tau = torch.quantile(dmin[torch.as_tensor(is_seen_lab, device=DEVICE)],
                             args.tau_quantile)
        pool = (dmin > tau).cpu().numpy()
        pool_purity = (~is_seen_lab[pool]).mean() if pool.any() else float("nan")
        khat, centers, assign = bic_select(z[torch.as_tensor(pool, device=DEVICE)],
                                           kmax=args.kmax, seed=args.seed)
        print(f"  pool: {pool.sum()} pts (tau={tau:.2f}, purity={pool_purity:.3f})  "
              f"BIC k-hat={khat}  (true k={k})")

        # 3. fine-tune with discovered anchors + pseudo-labels
        new_means = torch.cat([means.detach(), centers.detach()], dim=0)
        pool_idx = np.where(pool)[0]
        pseudo = N_CLASSES + assign.cpu().numpy()
        lab_idx = np.where(is_seen_lab)[0]
        ft_idx = np.concatenate([lab_idx, pool_idx])
        ft_lab = np.concatenate([tr_lab[lab_idx], pseudo])
        ds = PseudoDataset(base, ft_idx, ft_lab)
        sampler = BalancedBatchSampler(list(ft_lab), n_classes=len(seen) + khat,
                                       n_per_class=24)
        ft_loader = DataLoader(ds, batch_sampler=sampler, num_workers=2)
        train_sigreg_hybrid(backbone, ft_loader, ft_ep, new_means, mode="repulse",
                            disc="proto", alpha=1.0, rep_weight=rep_w,
                            sigreg_weight=args.sigreg_weight, n_slices=args.n_slices)

        # 4. evaluate on test
        te_embs, te_lab = collect_embeddings(backbone, test_loader)
        zt = torch.as_tensor(te_embs, device=DEVICE)
        d_seen = torch.cdist(zt, new_means.detach()[seen]).min(1).values
        d_new = torch.cdist(zt, new_means.detach()[N_CLASSES:]).min(1).values
        is_unseen = np.isin(te_lab, list(holdouts)).astype(int)
        before = roc_auc_score(is_unseen, dmin_test := torch.cdist(
            zt, seen_means).min(1).values.cpu().numpy())
        after = roc_auc_score(is_unseen, (d_seen - d_new).cpu().numpy())
        per_class = {}
        d_each = torch.cdist(zt, new_means.detach()[N_CLASSES:])
        for c in sorted(holdouts):
            counts = [int(((te_lab == c) & (d_each.argmin(1).cpu().numpy() == j)
                           & (d_new < d_seen).cpu().numpy()).sum())
                      for j in range(khat)]
            j = int(np.argmax(counts))
            per_class[c] = roc_auc_score((te_lab == c).astype(int),
                                         (-d_each[:, j]).cpu().numpy())
        results[k] = dict(khat=khat, purity=pool_purity, before=before,
                          after=after, per_class=per_class)
        pc = "  ".join(f"{CIFAR_CLASSES[c]}={a:.4f}" for c, a in per_class.items())
        print(f"  novelty AUC before={before:.4f}  after (margin)={after:.4f}")
        print(f"  discovered-anchor AUC per held-out class: {pc}")

    print("\n===== DISCOVERED-ANCHOR SUMMARY =====")
    print(f"{'k':>3}{'k-hat':>6}{'purity':>8}{'before':>9}{'after':>9}  per-class anchors")
    for k in ks:
        r = results[k]
        pc = "  ".join(f"{CIFAR_CLASSES[c]}={a:.3f}" for c, a in r["per_class"].items())
        print(f"{k:>3}{r['khat']:>6}{r['purity']:>8.3f}{r['before']:>9.4f}"
              f"{r['after']:>9.4f}  {pc}")

    plt.figure(figsize=(7, 5))
    plt.plot(ks, [results[k]["before"] for k in ks], "C0--s", lw=1.5,
             label="unseen-vs-rest, before (dist to seen means)")
    plt.plot(ks, [results[k]["after"] for k in ks], "C0-o", lw=2,
             label="unseen-vs-rest, after (seen-new margin)")
    avg_pc = [np.mean(list(results[k]["per_class"].values())) for k in ks]
    plt.plot(ks, avg_pc, "C1-.^", lw=2, label="mean per-class discovered-anchor AUC")
    plt.xticks(ks, [str(k) for k in ks])
    plt.xlabel("classes held out (k)"); plt.ylabel("AUC")
    plt.title("CIFAR-10 16d: anchors discovered from unlabeled data")
    plt.legend(fontsize=9); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(plot_path("discovered_anchors_cifar10.png"), dpi=150); plt.close()
    print(f"\n  saved {plot_path('discovered_anchors_cifar10.png')}")
    print("Done.")


if __name__ == "__main__":
    main()
