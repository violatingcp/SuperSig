"""
CIFAR-10 -> 32-dim latent with a CIFAR-pretrained ResNet backbone:
floating-anchor SIGReg (anchors seeded at scale 5, means learnable) vs SupCon.

Same protocol as 08_cifar_compare.py, but the backbone is a ResNet pretrained on
CIFAR (torch.hub chenyaofo/pytorch-cifar-models) and the latent space is 32-dim.
For each method we fine-tune the embedding, freeze it, and train linear probes:

    inclusive : embedding on all 10 classes -> 10-way probe -> micro-AUC ROC
    holdout   : embedding with class `HOLDOUT` (default 4 = "deer") removed ->
                frozen -> binary "held-out class vs rest" probe -> ROC

Anchors: one per class, mutually orthogonal, norm `--anchor-scale` (default 5).
Means are learnable ("float") with the hinge-separation regularizer
(mode="learnmeans"), so they drift from the seed during training.

Caveat: with --pretrain cifar10 (the default) the pretrained weights have already
seen the held-out class during supervised pretraining, so the holdout is only
clean w.r.t. the embedding fine-tuning.  Use --pretrain cifar100 for an
initialization that never saw CIFAR-10 labels.

Outputs (plots/):
    roc_cifar_resnet32_inclusive.png   micro-ROC, SIGReg vs SupCon (all classes)
    roc_cifar_resnet32_holdout.png     held-out-class-vs-rest ROC, SIGReg vs SupCon
    corner_cifar_resnet32_<method>_<case>.png   latent corner plots (first 10 dims)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from supersig.config import plot_path, N_CLASSES, HOLDOUT, DEVICE
from supersig.data import (
    get_cifar_loaders, build_cifar_holdout_loaders, cifar_two_view_loader,
)
from supersig.models import CIFARResNetBackbone
from supersig.losses import make_anchors
from supersig.train import (
    train_sigreg_classwise, train_supcon, train_linear_probe, train_binary_probe,
    collect_probs, collect_binary_scores, collect_embeddings,
)
from supersig.plotting import plot_corner

EMB_DIM = 32
CORNER_DIMS = 10        # corner-plot only the first (anchor-seeded) latent dims
CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck"]


def micro_roc(probs, labels):
    y_bin = label_binarize(labels, classes=list(range(N_CLASSES)))
    fpr, tpr, _ = roc_curve(y_bin.ravel(), probs.ravel())
    return fpr, tpr, auc(fpr, tpr)


def sigreg_backbone(loader, ssl_ep, args):
    means = make_anchors(args.anchor_scale, emb_dim=EMB_DIM).clone()
    backbone = CIFARResNetBackbone(EMB_DIM, arch=args.arch, pretrain=args.pretrain).to(DEVICE)
    train_sigreg_classwise(backbone, loader, ssl_ep, means, learn_means=True, mode="learnmeans")
    return backbone


def supcon_backbone(loader, ssl_ep, args):
    backbone = CIFARResNetBackbone(EMB_DIM, arch=args.arch, pretrain=args.pretrain).to(DEVICE)
    train_supcon(backbone, loader, ssl_ep)
    return backbone


def inclusive(method, ssl_ep, probe_ep, args):
    print(f"\n=== CIFAR/ResNet32 inclusive: {method} ===")
    train_loader, test_loader = get_cifar_loaders(quick=args.quick, limit=args.limit)
    if method == "sigreg":
        backbone = sigreg_backbone(train_loader, ssl_ep, args)
    else:
        backbone = supcon_backbone(
            cifar_two_view_loader(quick=args.quick, labeled=True, limit=args.limit), ssl_ep, args)
    head = nn.Linear(EMB_DIM, N_CLASSES).to(DEVICE)
    train_linear_probe(backbone, head, train_loader, probe_ep)
    probs, labels = collect_probs(lambda x: head(backbone(x)), test_loader)
    embs, elab = collect_embeddings(backbone, test_loader)
    plot_corner(embs[:, :CORNER_DIMS], elab, plot_path(f"corner_cifar_resnet32_{method}_inclusive.png"),
                title=f"CIFAR {method} ResNet 32-dim latent, dims 0-{CORNER_DIMS-1} (inclusive)")
    return micro_roc(probs, labels)


def holdout(method, ssl_ep, probe_ep, args):
    print(f"\n=== CIFAR/ResNet32 holdout ({CIFAR_CLASSES[HOLDOUT]}): {method} ===")
    emb_loader, probe_loader, test_loader = build_cifar_holdout_loaders(
        quick=args.quick, limit=args.limit)
    if method == "sigreg":
        backbone = sigreg_backbone(emb_loader, ssl_ep, args)
    else:
        backbone = supcon_backbone(
            cifar_two_view_loader(quick=args.quick, labeled=True, holdout=HOLDOUT,
                                  limit=args.limit), ssl_ep, args)
    head = nn.Linear(EMB_DIM, 2).to(DEVICE)
    train_binary_probe(backbone, head, probe_loader, probe_ep)
    scores, ytrue = collect_binary_scores(backbone, head, test_loader)
    fpr, tpr, _ = roc_curve(ytrue, scores)
    a = auc(fpr, tpr)
    embs, elab = collect_embeddings(backbone, test_loader)
    ish = (elab == HOLDOUT).astype(int)
    plot_corner(embs[:, :CORNER_DIMS], ish, plot_path(f"corner_cifar_resnet32_{method}_holdout.png"),
                title=f"CIFAR {method} ResNet latent dims 0-{CORNER_DIMS-1} (holdout): "
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
    ap.add_argument("--limit", type=int, default=None,
                    help="cap number of training images (test set stays full)")
    ap.add_argument("--ssl-epochs", type=int, default=None)
    ap.add_argument("--probe-epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arch", default="resnet20",
                    choices=["resnet20", "resnet32", "resnet44", "resnet56"],
                    help="CIFAR ResNet architecture from pytorch-cifar-models")
    ap.add_argument("--pretrain", default="cifar10", choices=["cifar10", "cifar100"],
                    help="pretraining dataset of the hub weights")
    ap.add_argument("--anchor-scale", type=float, default=5.0,
                    help="norm of the seed anchors (means float from there)")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    ssl_ep = args.ssl_epochs or (2 if args.quick else 10)
    probe_ep = args.probe_epochs or (1 if args.quick else 5)
    print(f"device={DEVICE}  arch={args.arch}  pretrain={args.pretrain}  "
          f"emb_dim={EMB_DIM}  anchor_scale={args.anchor_scale}")

    inc = {
        "SIGReg (floating anchors)": inclusive("sigreg", ssl_ep, probe_ep, args),
        "SupCon": inclusive("supcon", ssl_ep, probe_ep, args),
    }
    hold = {
        "SIGReg (floating anchors)": holdout("sigreg", ssl_ep, probe_ep, args),
        "SupCon": holdout("supcon", ssl_ep, probe_ep, args),
    }

    overlay(inc, f"CIFAR-10 inclusive (10-way, micro-AUC), ResNet {EMB_DIM}-dim",
            "roc_cifar_resnet32_inclusive.png")
    overlay(hold, f"CIFAR-10 hold-out '{CIFAR_CLASSES[HOLDOUT]}' vs rest, ResNet {EMB_DIM}-dim",
            "roc_cifar_resnet32_holdout.png")

    print("\n===== CIFAR ResNet-32d SUMMARY =====")
    for name in inc:
        print(f"  {name:<28} inclusive micro-AUC={inc[name][2]:.4f}   "
              f"holdout AUC={hold[name][2]:.4f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
