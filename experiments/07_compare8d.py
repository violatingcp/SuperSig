"""
8-dimensional comparison: SIGReg (learnable means, anchor scale 5) vs SupCon.

Repeats the anchor-init=5 learnable-means SIGReg setup but with an 8-dim embedding,
and compares it head-to-head with supervised contrastive learning (SupCon /
supervised SimCLR), also 8-dim, on both protocols:

    inclusive : embedding on all digits -> 10-way linear probe -> micro-AUC ROC
    holdout-4 : embedding without digit 4 -> 4-vs-rest linear probe -> ROC

(With 8 dims and 10 classes the SIGReg anchors fall back to spread-out random
directions -- see supersig.losses.make_anchors.)

Outputs (plots/):
    roc_compare8d_inclusive.png     micro-ROC, SIGReg vs SupCon
    roc_compare8d_holdout4.png      4-vs-rest ROC, SIGReg vs SupCon
    corner_compare8d_<method>_<case>.png     8-dim latent corner plots
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
from supersig.data import get_loaders, build_holdout_loaders, two_view_loader
from supersig.models import ConvBackbone
from supersig.losses import make_anchors
from supersig.train import (
    train_sigreg_classwise, train_supcon, train_linear_probe, train_binary_probe,
    collect_probs, collect_binary_scores, collect_embeddings,
)
from supersig.plotting import plot_corner

EMB_DIM = 8
SCALE = 5.0


def micro_roc(probs, labels):
    y_bin = label_binarize(labels, classes=list(range(N_CLASSES)))
    fpr, tpr, _ = roc_curve(y_bin.ravel(), probs.ravel())
    return fpr, tpr, auc(fpr, tpr)


# --------------------------------------------------------------------------- #
# Embedding trainers (return a frozen backbone)                               #
# --------------------------------------------------------------------------- #
def sigreg_backbone(loader, ssl_ep):
    means = make_anchors(SCALE, emb_dim=EMB_DIM).clone()
    backbone = ConvBackbone(EMB_DIM)
    train_sigreg_classwise(backbone, loader, ssl_ep, means, learn_means=True, mode="learnmeans")
    return backbone


def supcon_backbone(loader, ssl_ep):
    backbone = ConvBackbone(EMB_DIM)
    train_supcon(backbone, loader, ssl_ep)
    return backbone


# --------------------------------------------------------------------------- #
# Protocols                                                                    #
# --------------------------------------------------------------------------- #
def inclusive(method, ssl_ep, probe_ep, quick):
    print(f"\n=== inclusive 8d: {method} ===")
    train_loader, test_loader = get_loaders(batch_size=256, quick=quick)
    if method == "sigreg":
        backbone = sigreg_backbone(train_loader, ssl_ep)
    else:
        backbone = supcon_backbone(two_view_loader(quick=quick, labeled=True), ssl_ep)
    head = nn.Linear(EMB_DIM, N_CLASSES)
    train_linear_probe(backbone, head, train_loader, probe_ep)
    probs, labels = collect_probs(lambda x: head(backbone(x)), test_loader)
    embs, elab = collect_embeddings(backbone, test_loader)
    plot_corner(embs, elab, plot_path(f"corner_compare8d_{method}_inclusive.png"),
                title=f"{method} 8-dim latent (inclusive, colored by digit)")
    return micro_roc(probs, labels)


def holdout(method, ssl_ep, probe_ep, quick):
    print(f"\n=== holdout-4 8d: {method} ===")
    emb_loader, probe_loader, test_loader = build_holdout_loaders(quick=quick)
    if method == "sigreg":
        backbone = sigreg_backbone(emb_loader, ssl_ep)
    else:
        backbone = supcon_backbone(
            two_view_loader(quick=quick, labeled=True, holdout=HOLDOUT), ssl_ep)
    head = nn.Linear(EMB_DIM, 2)
    train_binary_probe(backbone, head, probe_loader, probe_ep)
    scores, ytrue = collect_binary_scores(backbone, head, test_loader)
    fpr, tpr, _ = roc_curve(ytrue, scores)
    a = auc(fpr, tpr)
    embs, elab = collect_embeddings(backbone, test_loader)
    is4 = (elab == HOLDOUT).astype(int)
    plot_corner(embs, is4, plot_path(f"corner_compare8d_{method}_holdout4.png"),
                title=f"{method} 8-dim latent (holdout-4): 1=digit {HOLDOUT} (unseen), 0=rest")
    return fpr, tpr, a


def overlay(results, title, out, xlabel="False positive rate"):
    plt.figure(figsize=(6, 6))
    for name, (fpr, tpr, a) in results.items():
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={a:.4f})")
    plt.plot([0, 1], [0, 1], "k:", lw=1)
    plt.xlabel(xlabel); plt.ylabel("True positive rate")
    plt.title(title); plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(plot_path(out), dpi=150); plt.close()
    print(f"  saved {plot_path(out)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--ssl-epochs", type=int, default=None)
    ap.add_argument("--probe-epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    ssl_ep = args.ssl_epochs or (2 if args.quick else 8)
    probe_ep = args.probe_epochs or (1 if args.quick else 4)

    inc = {
        "SIGReg (learnmeans, s5)": inclusive("sigreg", ssl_ep, probe_ep, args.quick),
        "SupCon": inclusive("supcon", ssl_ep, probe_ep, args.quick),
    }
    hold = {
        "SIGReg (learnmeans, s5)": holdout("sigreg", ssl_ep, probe_ep, args.quick),
        "SupCon": holdout("supcon", ssl_ep, probe_ep, args.quick),
    }

    overlay(inc, "8-dim inclusive (10-way, micro-AUC): SIGReg vs SupCon",
            "roc_compare8d_inclusive.png")
    overlay(hold, f"8-dim hold-out-{HOLDOUT} (4-vs-rest): SIGReg vs SupCon",
            "roc_compare8d_holdout4.png")

    print("\n===== 8-dim SUMMARY =====")
    for name in inc:
        print(f"  {name:<28} inclusive micro-AUC={inc[name][2]:.4f}   "
              f"holdout4 AUC={hold[name][2]:.4f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
