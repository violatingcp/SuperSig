"""
Two-stage factorization: augmentation-invariant SSL trunk, then SIGReg / SupCon
heads on the frozen features.

Hypothesis (see README, exp 20): the class labels are too coarse -- the SIGReg
constraint crushes within-class nuisance diversity (background, pose, color)
into the class Gaussian, which both distorts the geometry and destroys feature
richness.  Factorize instead:

    stage 1 : unsupervised SimCLR (NT-Xent, no labels) on two augmented views
              -> 64-dim trunk features.  The augmentations define and remove
              the nuisance directions.  Holdout classes excluded for protocol
              purity.
    stage 2 : trunk FROZEN; a small MLP head (64 -> 64 -> 16) is trained with
              the tuned classwise SIGReg (proto, w=20, 256 slices) or SupCon.
              The Gaussianisation shapes factorized content and cannot damage
              the trunk.

Full metric suite per (method, k): probed binary AUC, tied/per-class
Mahalanobis, unit-covariance (learned means / cosine centroids), eigenspectrum.
CIFAR-10, d=16, 5-sigma orthogonal seed, seed 0.

Outputs:
    plots/novelty_factorized_cifar10_16d.png
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from supersig.config import plot_path, DEVICE
from supersig.data import (
    get_cifar_loaders, build_cifar_holdout_loaders, cifar_balanced_loader,
    cifar_two_view_loader,
)
from supersig.models import CIFARResNetBackbone
from supersig.losses import make_anchors
from supersig.train import (
    train_simclr, train_sigreg_ssl, train_supcon, train_sigreg_hybrid, train_binary_probe,
    collect_binary_scores, collect_embeddings, REP_WEIGHT,
)
from supersig.metrics import mahalanobis_novelty

EMB_DIM = 16
TRUNK_DIM = 64
N_CLASSES = 10
PAIR_DIST = 5.0
DATASET = "cifar10"
HOLDOUT_SETS = {1: [4], 2: [4, 9], 3: [0, 4, 9]}

# end-to-end references at the same width (exp 20 runs)
REF = {
    "sigreg+proto": {"probed": {1: 0.8804, 2: 0.7659, 3: 0.7339},
                     "free":   {1: 0.8002, 2: 0.7386, 3: 0.7813}},
    "supcon":       {"probed": {1: 0.9210, 2: 0.9040, 3: 0.8982},
                     "free":   {1: 0.8127, 2: 0.7239, 3: 0.7060}},
}


class Stack(nn.Module):
    """Trunk + head, everything trainable (early layers float during stage 2)."""

    def __init__(self, trunk, head):
        super().__init__()
        self.trunk, self.head = trunk, head

    def forward(self, x):
        return self.head(self.trunk(x))


class FrozenStack(nn.Module):
    """Frozen trunk + trainable head; exposes only head params to optimizers."""

    def __init__(self, trunk, head):
        super().__init__()
        self.trunk, self.head = trunk, head
        for p in self.trunk.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        with torch.no_grad():
            f = self.trunk(x)
        return self.head(f)

    def parameters(self, recurse=True):
        return self.head.parameters(recurse)

    def train(self, mode=True):
        super().train(mode)
        self.trunk.eval()               # keep BN statistics frozen
        return self


def make_head():
    return nn.Sequential(nn.Linear(TRUNK_DIM, 64), nn.ReLU(),
                         nn.Linear(64, EMB_DIM)).to(DEVICE)


def stage1(holdouts, ssl_ep, args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    trunk = CIFARResNetBackbone(TRUNK_DIM, arch=args.arch, pretrain=args.pretrain).to(DEVICE)
    loader = cifar_two_view_loader(quick=args.quick, labeled=False,
                                   holdout=holdouts or None, limit=args.limit,
                                   dataset=DATASET)
    if args.stage1 == "sigreg":         # LeJEPA-style: invariance + global SIGReg
        train_sigreg_ssl(trunk, loader, ssl_ep)
    else:
        train_simclr(trunk, loader, ssl_ep)
    return trunk


def stage2(method, trunk, holdouts, head_ep, args):
    torch.manual_seed(args.seed + 1); np.random.seed(args.seed + 1)
    if args.finetune:                   # own copy: fine-tuning mutates the trunk
        model = Stack(copy.deepcopy(trunk), make_head())
    else:
        model = FrozenStack(trunk, make_head())
    if method == "sigreg+proto":
        means = make_anchors(PAIR_DIST / math.sqrt(2.0), emb_dim=EMB_DIM,
                             n_classes=N_CLASSES).clone()
        loader = cifar_balanced_loader(DATASET, holdout=holdouts or None,
                                       quick=args.quick, limit=args.limit)
        train_sigreg_hybrid(model, loader, head_ep, means, mode="repulse",
                            disc="proto", alpha=1.0, rep_weight=REP_WEIGHT,
                            sigreg_weight=args.sigreg_weight, n_slices=args.n_slices)
        return model, means
    train_supcon(model, cifar_two_view_loader(quick=args.quick, labeled=True,
                                              holdout=holdouts or None,
                                              limit=args.limit, dataset=DATASET), head_ep)
    return model, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ssl-epochs", type=int, default=None, help="stage-1 SimCLR epochs")
    ap.add_argument("--head-epochs", type=int, default=None, help="stage-2 head epochs")
    ap.add_argument("--probe-epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arch", default="resnet20")
    ap.add_argument("--pretrain", default="cifar10")
    ap.add_argument("--ks", default="1,2,3")
    ap.add_argument("--sigreg-weight", type=float, default=20.0)
    ap.add_argument("--n-slices", type=int, default=256)
    ap.add_argument("--stage1", choices=["simclr", "sigreg"], default="simclr",
                    help="stage-1 SSL objective (sigreg = LeJEPA-style inv+SIGReg)")
    ap.add_argument("--finetune", action="store_true",
                    help="stage 2 fine-tunes the whole stack (trunk floats)")
    ap.add_argument("--out-tag", default="")
    args = ap.parse_args()
    ssl_ep = args.ssl_epochs or (2 if args.quick else 20)
    head_ep = args.head_epochs or (1 if args.quick else 10)
    probe_ep = args.probe_epochs or (1 if args.quick else 5)
    ks = [int(x) for x in args.ks.split(",")]
    methods = ["sigreg+proto", "supcon"]

    train_loader, test_loader = get_cifar_loaders(quick=args.quick, limit=args.limit,
                                                  dataset=DATASET)
    res = {m: {} for m in methods}
    for k in ks:
        holdouts = set(HOLDOUT_SETS[k])
        seen = [c for c in range(N_CLASSES) if c not in holdouts]
        print(f"\n===== k={k}: stage-1 {args.stage1} trunk (no {sorted(holdouts)}) =====")
        trunk = stage1(holdouts, ssl_ep, args)
        for m in methods:
            print(f"\n=== k={k}: stage-2 {m} head ===")
            model, means = stage2(m, trunk, holdouts, head_ep, args)
            _, probe_loader, _ = build_cifar_holdout_loaders(
                quick=args.quick, holdout=holdouts, limit=args.limit, dataset=DATASET)
            head = nn.Linear(EMB_DIM, 2).to(DEVICE)
            train_binary_probe(model, head, probe_loader, probe_ep, positive=holdouts)
            scores, ybin = collect_binary_scores(model, head, test_loader,
                                                 positive=holdouts)
            probed = roc_auc_score(ybin, scores)
            tr_embs, tr_lab = collect_embeddings(model, train_loader)
            keep = np.isin(tr_lab, seen)
            te_embs, te_lab = collect_embeddings(model, test_loader)
            tied, percls, eigs = mahalanobis_novelty(tr_embs[keep], tr_lab[keep],
                                                     te_embs, seen)
            z = torch.as_tensor(te_embs, device=DEVICE)
            if means is not None:
                free = torch.cdist(z, means.detach()[seen]).min(1).values.cpu().numpy()
            else:
                zt = F.normalize(torch.as_tensor(tr_embs[keep], device=DEVICE), dim=1)
                lab = tr_lab[keep]
                cents = F.normalize(torch.stack(
                    [zt[torch.as_tensor(lab == c, device=DEVICE)].mean(0)
                     for c in seen]), dim=1)
                free = (1.0 - (F.normalize(z, dim=1) @ cents.t()).max(1).values).cpu().numpy()
            is_unseen = np.isin(te_lab, list(holdouts)).astype(int)
            res[m][k] = {
                "probed": probed,
                "tied": roc_auc_score(is_unseen, tied),
                "percls": roc_auc_score(is_unseen, percls),
                "free": roc_auc_score(is_unseen, free),
                "eigs": eigs,
            }
            r = res[m][k]
            print(f"  probed={r['probed']:.4f}  mahal tied/pc={r['tied']:.4f}/{r['percls']:.4f}"
                  f"  probe-free={r['free']:.4f}  "
                  f"eig min/med/max={eigs[0]:.3f}/{eigs[1]:.3f}/{eigs[2]:.3f}")

    mode = "finetuned" if args.finetune else "frozen"
    print(f"\n===== FACTORIZED SUMMARY (stage1={args.stage1}, trunk {mode}, "
          f"w={args.sigreg_weight}) =====")
    print(f"{'k':>4}{'method':>14}{'probed':>9}{'(e2e)':>8}{'free':>7}{'(e2e)':>8}{'eig med':>9}")
    for k in ks:
        for m in methods:
            r = res[m][k]
            print(f"{k:>4}{m:>14}{r['probed']:>9.4f}{REF[m]['probed'][k]:>8.4f}"
                  f"{r['free']:>7.4f}{REF[m]['free'][k]:>8.4f}{r['eigs'][1]:>9.3f}")

    plt.figure(figsize=(8, 5.5))
    for i, m in enumerate(methods):
        plt.plot(ks, [res[m][k]["probed"] for k in ks], f"C{i}-o", lw=2,
                 label=f"{m} probed (factorized)")
        plt.plot(ks, [res[m][k]["free"] for k in ks], f"C{i}-.^", lw=1.5,
                 label=f"{m} probe-free (factorized)")
        plt.plot(ks, [REF[m]["probed"][k] for k in ks], f"C{i}--s", lw=1, alpha=0.5,
                 label=f"{m} probed (end-to-end)")
        plt.plot(ks, [REF[m]["free"][k] for k in ks], f"C{i}:v", lw=1, alpha=0.5,
                 label=f"{m} probe-free (end-to-end)")
    plt.xticks(ks, [str(k) for k in ks])
    plt.xlabel("classes held out (k)"); plt.ylabel("unseen-vs-rest AUC")
    plt.title("CIFAR-10 16d: SSL-factorized trunk vs end-to-end")
    plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(plot_path(f"novelty_factorized_cifar10_16d{args.out_tag}.png"), dpi=150); plt.close()
    print(f"\n  saved {plot_path(f'novelty_factorized_cifar10_16d{args.out_tag}.png')}")
    print("Done.")


if __name__ == "__main__":
    main()
