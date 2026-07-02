"""
CIFAR-10 study: floating-anchor SIGReg (learnable means) vs SupCon.

Mirrors the MNIST comparison on CIFAR-10 with a 3-channel CNN backbone.  For each
method we train a frozen embedding and a linear probe under two protocols:

    inclusive : embedding on all 10 classes -> 10-way probe -> micro-AUC ROC
    holdout   : embedding with class `HOLDOUT` (default 4 = "deer") removed -> frozen
                -> binary "held-out class vs rest" probe -> ROC

"Floating anchors" = class means initialized at spread-out anchors (scale 5) and then
learned (mode="learnmeans"), so the anchors drift/float during training.

Outputs (plots/):
    roc_cifar_inclusive.png    micro-ROC, SIGReg vs SupCon (all classes)
    roc_cifar_holdout.png      held-out-class-vs-rest ROC, SIGReg vs SupCon
    corner_cifar_<method>_<case>.png    latent corner plots
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

from supersig.config import plot_path, N_CLASSES, HOLDOUT
from supersig.data import (
    get_cifar_loaders, build_cifar_holdout_loaders, cifar_two_view_loader,
)
from supersig.models import CIFARBackbone
from supersig.losses import make_anchors
from supersig.train import (
    train_sigreg_classwise, train_supcon, train_linear_probe, train_binary_probe,
    collect_probs, collect_binary_scores, collect_embeddings,
)
from supersig.plotting import plot_corner

EMB_DIM = 16
SCALE = 5.0
CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck"]


def micro_roc(probs, labels):
    y_bin = label_binarize(labels, classes=list(range(N_CLASSES)))
    fpr, tpr, _ = roc_curve(y_bin.ravel(), probs.ravel())
    return fpr, tpr, auc(fpr, tpr)


def sigreg_backbone(loader, ssl_ep):
    means = make_anchors(SCALE, emb_dim=EMB_DIM).clone()
    backbone = CIFARBackbone(EMB_DIM)
    train_sigreg_classwise(backbone, loader, ssl_ep, means, learn_means=True, mode="learnmeans")
    return backbone


def supcon_backbone(loader, ssl_ep):
    backbone = CIFARBackbone(EMB_DIM)
    train_supcon(backbone, loader, ssl_ep)
    return backbone


def inclusive(method, ssl_ep, probe_ep, quick, limit):
    print(f"\n=== CIFAR inclusive: {method} ===")
    train_loader, test_loader = get_cifar_loaders(quick=quick, limit=limit)
    if method == "sigreg":
        backbone = sigreg_backbone(train_loader, ssl_ep)
    else:
        backbone = supcon_backbone(cifar_two_view_loader(quick=quick, labeled=True, limit=limit), ssl_ep)
    head = nn.Linear(EMB_DIM, N_CLASSES)
    train_linear_probe(backbone, head, train_loader, probe_ep)
    probs, labels = collect_probs(lambda x: head(backbone(x)), test_loader)
    embs, elab = collect_embeddings(backbone, test_loader)
    plot_corner(embs, elab, plot_path(f"corner_cifar_{method}_inclusive.png"),
                title=f"CIFAR {method} {EMB_DIM}-dim latent (inclusive, colored by class)")
    return micro_roc(probs, labels)


def holdout(method, ssl_ep, probe_ep, quick, limit):
    print(f"\n=== CIFAR holdout ({CIFAR_CLASSES[HOLDOUT]}): {method} ===")
    emb_loader, probe_loader, test_loader = build_cifar_holdout_loaders(quick=quick, limit=limit)
    if method == "sigreg":
        backbone = sigreg_backbone(emb_loader, ssl_ep)
    else:
        backbone = supcon_backbone(
            cifar_two_view_loader(quick=quick, labeled=True, holdout=HOLDOUT, limit=limit), ssl_ep)
    head = nn.Linear(EMB_DIM, 2)
    train_binary_probe(backbone, head, probe_loader, probe_ep)
    scores, ytrue = collect_binary_scores(backbone, head, test_loader)
    fpr, tpr, _ = roc_curve(ytrue, scores)
    a = auc(fpr, tpr)
    embs, elab = collect_embeddings(backbone, test_loader)
    ish = (elab == HOLDOUT).astype(int)
    plot_corner(embs, ish, plot_path(f"corner_cifar_{method}_holdout.png"),
                title=f"CIFAR {method} latent (holdout): 1={CIFAR_CLASSES[HOLDOUT]} (unseen), 0=rest")
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
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    ssl_ep = args.ssl_epochs or (2 if args.quick else 15)
    probe_ep = args.probe_epochs or (1 if args.quick else 6)

    inc = {
        "SIGReg (floating anchors)": inclusive("sigreg", ssl_ep, probe_ep, args.quick, args.limit),
        "SupCon": inclusive("supcon", ssl_ep, probe_ep, args.quick, args.limit),
    }
    hold = {
        "SIGReg (floating anchors)": holdout("sigreg", ssl_ep, probe_ep, args.quick, args.limit),
        "SupCon": holdout("supcon", ssl_ep, probe_ep, args.quick, args.limit),
    }

    overlay(inc, "CIFAR-10 inclusive (10-way, micro-AUC): SIGReg vs SupCon",
            "roc_cifar_inclusive.png")
    overlay(hold, f"CIFAR-10 hold-out '{CIFAR_CLASSES[HOLDOUT]}' vs rest: SIGReg vs SupCon",
            "roc_cifar_holdout.png")

    print("\n===== CIFAR SUMMARY =====")
    for name in inc:
        print(f"  {name:<28} inclusive micro-AUC={inc[name][2]:.4f}   "
              f"holdout AUC={hold[name][2]:.4f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
