"""
CIFAR-100 upgrade of the 32-dim latent study: SIGReg+proto vs SupCon.

The two champions from the CIFAR-10 series (09-13) rematched on CIFAR-100:

    sigreg+proto : classwise SIGReg, repulsive floating means seeded ~3 sigma
                   apart, Gaussian-posterior discriminative term (no head).
    supcon       : supervised contrastive with two-view augmentation.

CIFAR-100 changes two things structurally:
  * 100 classes > 32 dims -- anchors cannot be orthogonal; make_anchors falls
    back to deterministic random directions (typical pairwise distance
    ~ pair_dist, no longer exact).
  * random 256-batches give ~2.5 samples/class, below MIN_PER_CLASS=8, so the
    SIGReg run uses a class-balanced sampler (`--classes-per-batch` x
    `--per-class`).  The repulsion weight is rescaled by the pair count
    (45 / C*(C-1)/2) to keep the mean-geometry equilibrium comparable.

Backbone is pretrained on CIFAR-100 by default (see 09 caveat: the weights have
seen the held-out class; --pretrain cifar10 gives a label-disjoint init).

Protocols as before: inclusive (100-way probe -> micro-AUC) and holdout
(class `--holdout`, default 4 = "beaver", removed from embedding training ->
frozen -> binary vs-rest probe -> ROC).

Outputs (plots/):
    roc_cifar100_<D>d_inclusive.png / roc_cifar100_<D>d_holdout.png
    corner_cifar100_<D>d_<method>_<case>.png (first 10 dims; inclusive shows classes 0-9)
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
from torchvision import datasets

from supersig.config import plot_path, DATA_DIR, DEVICE
from supersig.data import (
    get_cifar_loaders, build_cifar_holdout_loaders, cifar_two_view_loader,
    cifar_balanced_loader,
)
from supersig.models import CIFARResNetBackbone
from supersig.losses import make_anchors, mean_geometry
from supersig.train import (
    train_sigreg_hybrid, train_supcon, train_linear_probe, train_binary_probe,
    collect_probs, collect_binary_scores, collect_embeddings, REP_WEIGHT,
)
from supersig.plotting import plot_corner

EMB_DIM = 32          # overridden by --emb-dim
SEED_TAG = ""         # set to "_s<pair-dist>" for non-default seeding
N_CLASSES = 100
CORNER_DIMS = 10
DATASET = "cifar100"


def micro_roc(probs, labels):
    y_bin = label_binarize(labels, classes=list(range(N_CLASSES)))
    fpr, tpr, _ = roc_curve(y_bin.ravel(), probs.ravel())
    return fpr, tpr, auc(fpr, tpr)


def sigreg_backbone(loader, ssl_ep, args, disc="proto"):
    scale = args.pair_dist / math.sqrt(2.0)
    means = make_anchors(scale, emb_dim=EMB_DIM, n_classes=N_CLASSES).clone()
    dmin, dmean = mean_geometry(means)
    rep_w = REP_WEIGHT * 45.0 / (N_CLASSES * (N_CLASSES - 1) / 2) * args.rep_scale
    print(f"  seed means: norm={scale:.3f}  pairwise min={dmin:.3f} mean={dmean:.3f}  "
          f"rep_weight={rep_w:.4f}  disc={disc}")
    backbone = CIFARResNetBackbone(EMB_DIM, arch=args.arch, pretrain=args.pretrain).to(DEVICE)
    train_sigreg_hybrid(backbone, loader, ssl_ep, means, mode="repulse",
                        disc=disc, alpha=args.alpha, rep_weight=rep_w)
    return backbone


def supcon_backbone(loader, ssl_ep, args):
    backbone = CIFARResNetBackbone(EMB_DIM, arch=args.arch, pretrain=args.pretrain).to(DEVICE)
    train_supcon(backbone, loader, ssl_ep)
    return backbone


def inclusive(method, ssl_ep, probe_ep, args):
    print(f"\n=== CIFAR-100 inclusive: {method} ===")
    train_loader, test_loader = get_cifar_loaders(quick=args.quick, limit=args.limit,
                                                  dataset=DATASET)
    if method.startswith("sigreg+"):
        emb_loader = cifar_balanced_loader(DATASET, quick=args.quick, limit=args.limit,
                                           classes_per_batch=args.classes_per_batch,
                                           per_class=args.per_class)
        backbone = sigreg_backbone(emb_loader, ssl_ep, args, disc=method.split("+", 1)[1])
    else:
        backbone = supcon_backbone(
            cifar_two_view_loader(quick=args.quick, labeled=True, limit=args.limit,
                                  dataset=DATASET), ssl_ep, args)
    head = nn.Linear(EMB_DIM, N_CLASSES).to(DEVICE)
    train_linear_probe(backbone, head, train_loader, probe_ep)
    probs, labels = collect_probs(lambda x: head(backbone(x)), test_loader)
    embs, elab = collect_embeddings(backbone, test_loader)
    sel = elab < 10                       # corner plot readable with 10 classes only
    tag = method.replace("+", "_")
    plot_corner(embs[sel][:, :CORNER_DIMS], elab[sel],
                plot_path(f"corner_cifar100_{EMB_DIM}d{SEED_TAG}_{tag}_inclusive.png"),
                title=f"CIFAR-100 {method} latent dims 0-{CORNER_DIMS-1} (classes 0-9 shown)")
    return micro_roc(probs, labels)


def holdout(method, ssl_ep, probe_ep, args, holdout_name):
    print(f"\n=== CIFAR-100 holdout ({holdout_name}): {method} ===")
    _, probe_loader, test_loader = build_cifar_holdout_loaders(
        quick=args.quick, holdout=args.holdout, limit=args.limit, dataset=DATASET)
    if method.startswith("sigreg+"):
        emb_loader = cifar_balanced_loader(DATASET, holdout=args.holdout, quick=args.quick,
                                           limit=args.limit,
                                           classes_per_batch=args.classes_per_batch,
                                           per_class=args.per_class)
        backbone = sigreg_backbone(emb_loader, ssl_ep, args, disc=method.split("+", 1)[1])
    else:
        backbone = supcon_backbone(
            cifar_two_view_loader(quick=args.quick, labeled=True, holdout=args.holdout,
                                  limit=args.limit, dataset=DATASET), ssl_ep, args)
    head = nn.Linear(EMB_DIM, 2).to(DEVICE)
    train_binary_probe(backbone, head, probe_loader, probe_ep, positive=args.holdout)
    scores, ytrue = collect_binary_scores(backbone, head, test_loader, positive=args.holdout)
    fpr, tpr, _ = roc_curve(ytrue, scores)
    a = auc(fpr, tpr)
    embs, elab = collect_embeddings(backbone, test_loader)
    ish = (elab == args.holdout).astype(int)
    tag = method.replace("+", "_")
    plot_corner(embs[:, :CORNER_DIMS], ish, plot_path(f"corner_cifar100_{EMB_DIM}d{SEED_TAG}_{tag}_holdout.png"),
                title=f"CIFAR-100 {method} latent dims 0-{CORNER_DIMS-1} (holdout): "
                      f"1={holdout_name} (unseen), 0=rest")
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
    ap.add_argument("--pretrain", default="cifar100", choices=["cifar10", "cifar100"])
    ap.add_argument("--holdout", type=int, default=4,
                    help="CIFAR-100 class index held out of embedding training")
    ap.add_argument("--emb-dim", type=int, default=32)
    ap.add_argument("--pair-dist", type=float, default=3.0)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--classes-per-batch", type=int, default=25)
    ap.add_argument("--per-class", type=int, default=24)
    ap.add_argument("--methods", default="sigreg+proto,supcon",
                    help="comma-separated: sigreg+proto, sigreg+ce, supcon")
    ap.add_argument("--rep-scale", type=float, default=1.0,
                    help="multiplier on the (pair-count-rescaled) repulsion weight")
    ap.add_argument("--out-tag", default="",
                    help="extra tag appended to output plot filenames")
    args = ap.parse_args()
    global EMB_DIM, SEED_TAG
    EMB_DIM = args.emb_dim
    if args.pair_dist != 3.0:
        SEED_TAG = f"_s{args.pair_dist:g}"
    SEED_TAG += args.out_tag
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    ssl_ep = args.ssl_epochs or (2 if args.quick else 10)
    probe_ep = args.probe_epochs or (1 if args.quick else 5)

    holdout_name = datasets.CIFAR100(DATA_DIR, train=False, download=True).classes[args.holdout]
    print(f"device={DEVICE}  arch={args.arch}  pretrain={args.pretrain}  emb_dim={EMB_DIM}  "
          f"n_classes={N_CLASSES}  holdout={args.holdout} ({holdout_name})  "
          f"pair-dist={args.pair_dist}  alpha={args.alpha}")

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    inc = {m: inclusive(m, ssl_ep, probe_ep, args) for m in methods}
    hold = {m: holdout(m, ssl_ep, probe_ep, args, holdout_name) for m in methods}

    overlay(inc, "CIFAR-100 inclusive (100-way, micro-AUC): SIGReg+proto vs SupCon",
            f"roc_cifar100_{EMB_DIM}d{SEED_TAG}_inclusive.png")
    overlay(hold, f"CIFAR-100 hold-out '{holdout_name}' vs rest: SIGReg+proto vs SupCon",
            f"roc_cifar100_{EMB_DIM}d{SEED_TAG}_holdout.png")

    print("\n===== CIFAR-100 SUMMARY =====")
    for m in methods:
        print(f"  {m:<28} inclusive micro-AUC={inc[m][2]:.4f}   holdout AUC={hold[m][2]:.4f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
