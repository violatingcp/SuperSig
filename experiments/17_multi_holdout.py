"""
Multi-class holdout on CIFAR-100 (100-dim latent, 5-sigma seed): remove a SET
of classes from embedding training, then test whether the group of unseen
classes is separable from the rest.

    embedding : trained without all classes in --holdouts
    probe     : binary "in holdout set" vs rest (full train set, frozen backbone)
    ROC       : combined (any held-out class vs rest) + one restricted ROC per
                held-out class (that class's positives vs all negatives, same score)

Methods: sigreg+ce (best CIFAR-100 holdout config) and supcon (aug reference).

Outputs (plots/):
    roc_cifar100_100d_hold<k>.png   combined (solid) and per-class (dashed) ROCs
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve, auc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import datasets

from supersig.config import plot_path, DATA_DIR, DEVICE
from supersig.data import (
    build_cifar_holdout_loaders, cifar_two_view_loader, cifar_balanced_loader,
)
from supersig.models import CIFARResNetBackbone
from supersig.losses import make_anchors
from supersig.train import (
    train_sigreg_hybrid, train_supcon, train_binary_probe,
    collect_binary_scores, REP_WEIGHT,
)

EMB_DIM = 100
N_CLASSES = 100
PAIR_DIST = 5.0
DATASET = "cifar100"


def build_backbone(method, holdouts, ssl_ep, args):
    backbone = CIFARResNetBackbone(EMB_DIM, arch=args.arch, pretrain=args.pretrain).to(DEVICE)
    if method == "sigreg+ce":
        means = make_anchors(PAIR_DIST / math.sqrt(2.0), emb_dim=EMB_DIM,
                             n_classes=N_CLASSES).clone()
        rep_w = REP_WEIGHT * 45.0 / (N_CLASSES * (N_CLASSES - 1) / 2)
        loader = cifar_balanced_loader(DATASET, holdout=holdouts, quick=args.quick,
                                       limit=args.limit)
        train_sigreg_hybrid(backbone, loader, ssl_ep, means, mode="repulse",
                            disc="ce", alpha=1.0, rep_weight=rep_w)
    else:
        train_supcon(backbone, cifar_two_view_loader(
            quick=args.quick, labeled=True, holdout=holdouts,
            limit=args.limit, dataset=DATASET), ssl_ep)
    return backbone


def run_method(method, holdouts, ssl_ep, probe_ep, args):
    print(f"\n=== holdout {sorted(holdouts)}: {method} ===")
    _, probe_loader, test_loader = build_cifar_holdout_loaders(
        quick=args.quick, holdout=holdouts, limit=args.limit, dataset=DATASET)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    backbone = build_backbone(method, holdouts, ssl_ep, args)
    head = nn.Linear(EMB_DIM, 2).to(DEVICE)
    train_binary_probe(backbone, head, probe_loader, probe_ep, positive=holdouts)
    scores, ybin = collect_binary_scores(backbone, head, test_loader, positive=holdouts)
    # per-class labels for restricted ROCs
    ylab = np.concatenate([y.numpy() for _, y in test_loader])
    out = {}
    fpr, tpr, _ = roc_curve(ybin, scores)
    out["combined"] = (fpr, tpr, auc(fpr, tpr))
    for c in sorted(holdouts):
        keep = (ylab == c) | (ybin == 0)
        fpr, tpr, _ = roc_curve((ylab[keep] == c).astype(int), scores[keep])
        out[c] = (fpr, tpr, auc(fpr, tpr))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdouts", default="4,70",
                    help="comma-separated CIFAR-100 class indices to hold out")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ssl-epochs", type=int, default=None)
    ap.add_argument("--probe-epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arch", default="resnet20")
    ap.add_argument("--pretrain", default="cifar100")
    args = ap.parse_args()
    ssl_ep = args.ssl_epochs or (2 if args.quick else 10)
    probe_ep = args.probe_epochs or (1 if args.quick else 5)
    holdouts = sorted({int(x) for x in args.holdouts.split(",")})
    names = datasets.CIFAR100(DATA_DIR, train=False, download=True).classes
    hnames = [names[c] for c in holdouts]
    print(f"device={DEVICE}  emb_dim={EMB_DIM}  holdouts={holdouts} ({', '.join(hnames)})")

    results = {m: run_method(m, set(holdouts), ssl_ep, probe_ep, args)
               for m in ["sigreg+ce", "supcon"]}

    k = len(holdouts)
    per_class_legend = k <= 4          # beyond that the legend drowns the plot
    plt.figure(figsize=(7, 7))
    for i, (m, res) in enumerate(results.items()):
        fpr, tpr, a = res["combined"]
        plt.plot(fpr, tpr, color=f"C{i}", lw=2.5, label=f"{m} combined (AUC={a:.4f})")
        for j, c in enumerate(holdouts):
            fpr, tpr, a = res[c]
            lbl = f"{m} {names[c]} (AUC={a:.4f})" if per_class_legend else \
                  (f"{m} per-class" if j == 0 else None)
            plt.plot(fpr, tpr, color=f"C{i}", lw=0.8,
                     ls=["--", ":", "-."][j % 3] if per_class_legend else "-",
                     alpha=0.8 if per_class_legend else 0.25, label=lbl)
    plt.plot([0, 1], [0, 1], color="grey", lw=1, ls=":")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    title_names = ", ".join(hnames) if per_class_legend else f"{k} classes"
    plt.title(f"CIFAR-100 hold-out {title_names} vs rest")
    plt.legend(loc="lower right", fontsize=9); plt.tight_layout()
    out = f"roc_cifar100_{EMB_DIM}d_hold{k}.png"
    plt.savefig(plot_path(out), dpi=150); plt.close()
    print(f"\n  saved {plot_path(out)}")

    print(f"\n===== HOLDOUT-{k} SUMMARY ({', '.join(hnames)}) =====")
    for m, res in results.items():
        percls = "  ".join(f"{names[c]}={res[c][2]:.4f}" for c in holdouts)
        print(f"  {m:<12} combined AUC={res['combined'][2]:.4f}   {percls}")
    print("\nDone.")


if __name__ == "__main__":
    main()
