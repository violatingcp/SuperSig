"""
Can the linear-head cross-entropy in the hybrid (12_cifar_hybrid.py) be dropped?

Two head-free / CE-free discriminative terms, both derived from the Gaussian
latent model itself (classwise SIGReg + repulsive floating means, 3-sigma seed,
no augmentation -- otherwise identical to run 12):

    sigreg+proto : cross-entropy of the model's own posterior,
                   logits = -||z - mean_c||^2 / 2.  No extra parameters; the
                   Gaussian model classifies itself.
    sigreg+hinge : CE-free, purely geometric: relu(margin - ||z - mean_wrong||)^2
                   forbids samples from sitting within `--margin` sigma of a
                   wrong class mean.

Reference numbers (10 ssl epochs, 5 probe epochs, seed 0):
    SIGReg repulse (11)          inclusive 0.9822   holdout 0.8509
    SIGReg repulse + CE (12)     inclusive 0.9917   holdout 0.9235
    SupCon + aug (09)            inclusive 0.9920   holdout 0.9290
    SupCon no aug (10)           inclusive 0.9904   holdout 0.9143

Outputs (plots/):
    roc_cifar_noce_inclusive.png / roc_cifar_noce_holdout.png
    corner_cifar_noce_<method>_<case>.png (first 10 latent dims)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from supersig.config import plot_path, N_CLASSES, HOLDOUT, DEVICE
from supersig.data import get_cifar_loaders, build_cifar_holdout_loaders
from supersig.models import CIFARResNetBackbone
from supersig.losses import make_anchors, mean_geometry
from supersig.train import (
    train_sigreg_hybrid, train_linear_probe, train_binary_probe,
    collect_probs, collect_binary_scores, collect_embeddings,
)
from supersig.plotting import plot_corner

EMB_DIM = 32
CORNER_DIMS = 10
CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck"]
DISCS = {"sigreg+proto": "proto", "sigreg+hinge": "hinge"}


def micro_roc(probs, labels):
    y_bin = label_binarize(labels, classes=list(range(N_CLASSES)))
    fpr, tpr, _ = roc_curve(y_bin.ravel(), probs.ravel())
    return fpr, tpr, auc(fpr, tpr)


def make_backbone(loader, method, ssl_ep, args):
    scale = args.pair_dist / math.sqrt(2.0)
    means = make_anchors(scale, emb_dim=EMB_DIM).clone()
    d0, _ = mean_geometry(means)
    print(f"  seed anchors: norm={scale:.3f}  pairwise distance={d0:.3f}")
    backbone = CIFARResNetBackbone(EMB_DIM, arch=args.arch, pretrain=args.pretrain).to(DEVICE)
    train_sigreg_hybrid(backbone, loader, ssl_ep, means, mode="repulse",
                        disc=DISCS[method], alpha=args.alpha, margin=args.margin)
    return backbone


def inclusive(method, ssl_ep, probe_ep, args):
    print(f"\n=== CIFAR no-CE inclusive: {method} ===")
    train_loader, test_loader = get_cifar_loaders(quick=args.quick, limit=args.limit)
    backbone = make_backbone(train_loader, method, ssl_ep, args)
    head = nn.Linear(EMB_DIM, N_CLASSES).to(DEVICE)
    train_linear_probe(backbone, head, train_loader, probe_ep)
    probs, labels = collect_probs(lambda x: head(backbone(x)), test_loader)
    embs, elab = collect_embeddings(backbone, test_loader)
    tag = method.replace("+", "_")
    plot_corner(embs[:, :CORNER_DIMS], elab, plot_path(f"corner_cifar_noce_{tag}_inclusive.png"),
                title=f"CIFAR {method} 32-dim latent, dims 0-{CORNER_DIMS-1} (inclusive)")
    return micro_roc(probs, labels)


def holdout(method, ssl_ep, probe_ep, args):
    print(f"\n=== CIFAR no-CE holdout ({CIFAR_CLASSES[HOLDOUT]}): {method} ===")
    emb_loader, probe_loader, test_loader = build_cifar_holdout_loaders(
        quick=args.quick, limit=args.limit)
    backbone = make_backbone(emb_loader, method, ssl_ep, args)
    head = nn.Linear(EMB_DIM, 2).to(DEVICE)
    train_binary_probe(backbone, head, probe_loader, probe_ep)
    scores, ytrue = collect_binary_scores(backbone, head, test_loader)
    fpr, tpr, _ = roc_curve(ytrue, scores)
    a = auc(fpr, tpr)
    embs, elab = collect_embeddings(backbone, test_loader)
    ish = (elab == HOLDOUT).astype(int)
    tag = method.replace("+", "_")
    plot_corner(embs[:, :CORNER_DIMS], ish, plot_path(f"corner_cifar_noce_{tag}_holdout.png"),
                title=f"CIFAR {method} latent dims 0-{CORNER_DIMS-1} (holdout): "
                      f"1={CIFAR_CLASSES[HOLDOUT]} (unseen), 0=rest")
    return fpr, tpr, a


def overlay(results, title, out):
    plt.figure(figsize=(6, 6))
    for name, (fpr, tpr, a) in results.items():
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={a:.4f})")
    plt.plot([0, 1], [0, 1], "k:", lw=1)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title(title); plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(plot_path(out), dpi=150); plt.close()
    print(f"  saved {plot_path(out)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ssl-epochs", type=int, default=None)
    ap.add_argument("--probe-epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arch", default="resnet20",
                    choices=["resnet20", "resnet32", "resnet44", "resnet56"])
    ap.add_argument("--pretrain", default="cifar10", choices=["cifar10", "cifar100"])
    ap.add_argument("--pair-dist", type=float, default=3.0,
                    help="pairwise seed distance between means, in sigma units")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="weight of the discriminative term")
    ap.add_argument("--margin", type=float, default=3.0,
                    help="hinge margin (sigma) from wrong class means")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    ssl_ep = args.ssl_epochs or (2 if args.quick else 10)
    probe_ep = args.probe_epochs or (1 if args.quick else 5)
    print(f"device={DEVICE}  arch={args.arch}  pretrain={args.pretrain}  emb_dim={EMB_DIM}  "
          f"seed pair-dist={args.pair_dist} sigma  alpha={args.alpha}  margin={args.margin}")

    methods = list(DISCS)
    inc = {m: inclusive(m, ssl_ep, probe_ep, args) for m in methods}
    hold = {m: holdout(m, ssl_ep, probe_ep, args) for m in methods}

    overlay(inc, "CIFAR-10 inclusive: head-free / CE-free discriminative terms",
            "roc_cifar_noce_inclusive.png")
    overlay(hold, f"CIFAR-10 hold-out '{CIFAR_CLASSES[HOLDOUT]}' vs rest: no-CE variants",
            "roc_cifar_noce_holdout.png")

    print("\n===== NO-CE SUMMARY (vs runs 09-12) =====")
    print("  SIGReg repulse (11)            inclusive micro-AUC=0.9822   holdout AUC=0.8509")
    print("  SIGReg repulse + CE (12)       inclusive micro-AUC=0.9917   holdout AUC=0.9235")
    print("  SupCon + aug (09)              inclusive micro-AUC=0.9920   holdout AUC=0.9290")
    print("  SupCon no aug (10)             inclusive micro-AUC=0.9904   holdout AUC=0.9143")
    for m in methods:
        print(f"  {m:<30} inclusive micro-AUC={inc[m][2]:.4f}   holdout AUC={hold[m][2]:.4f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
