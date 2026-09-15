"""
Scan the anchor-initialization scale for the learnable-means SIGReg embedding.

`make_anchors(scale)` places the initial class means at `scale * e_c` (orthogonal
basis vectors), so `scale` sets how far apart the means START before training moves
them.  This script sweeps `scale` and, for each value, measures BOTH protocols:

    inclusive : embedding on all digits -> 10-way linear probe -> micro-AUC / accuracy
    holdout-4 : embedding without digit 4 -> 4-vs-rest linear probe -> AUC

Motivation: larger initial spacing spreads the seen classes more (repulsion-like),
which we expect to help closed-set discrimination but hurt detection of the unseen 4.

Output: plots/anchor_scan.png  (micro-AUC and 4-vs-rest AUC vs anchor scale).
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

from supersig.config import plot_path, EMB_DIM, N_CLASSES, HOLDOUT
from supersig.data import get_loaders, build_holdout_loaders
from supersig.models import ConvBackbone
from supersig.losses import make_anchors
from supersig.train import (
    train_sigreg_classwise, train_linear_probe, train_binary_probe,
    collect_probs, collect_binary_scores,
)


def micro_auc(probs, labels):
    y_bin = label_binarize(labels, classes=list(range(N_CLASSES)))
    fpr, tpr, _ = roc_curve(y_bin.ravel(), probs.ravel())
    return auc(fpr, tpr)


def run_inclusive(scale, train_loader, test_loader, ssl_ep, probe_ep):
    means = make_anchors(scale).clone()
    backbone = ConvBackbone()
    train_sigreg_classwise(backbone, train_loader, ssl_ep, means,
                           learn_means=True, mode="learnmeans")
    head = nn.Linear(EMB_DIM, N_CLASSES)
    train_linear_probe(backbone, head, train_loader, probe_ep)
    probs, labels = collect_probs(lambda x: head(backbone(x)), test_loader)
    acc = (probs.argmax(1) == labels).mean()
    return micro_auc(probs, labels), acc


def run_holdout(scale, emb_loader, probe_loader, test_loader, ssl_ep, probe_ep):
    means = make_anchors(scale).clone()
    backbone = ConvBackbone()
    train_sigreg_classwise(backbone, emb_loader, ssl_ep, means,
                           learn_means=True, mode="learnmeans")
    head = nn.Linear(EMB_DIM, 2)
    train_binary_probe(backbone, head, probe_loader, probe_ep)
    scores, ytrue = collect_binary_scores(backbone, head, test_loader)
    fpr, tpr, _ = roc_curve(ytrue, scores)
    return auc(fpr, tpr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", type=float, nargs="+",
                    default=[2, 4, 6, 8, 12, 16, 24])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--ssl-epochs", type=int, default=None)
    ap.add_argument("--probe-epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    ssl_ep = args.ssl_epochs or (2 if args.quick else 6)
    probe_ep = args.probe_epochs or (1 if args.quick else 3)

    train_loader, test_loader = get_loaders(batch_size=256, quick=args.quick)
    emb_loader, probe_loader, test_loader_h = build_holdout_loaders(quick=args.quick)

    inc_auc, inc_acc, hold_auc = [], [], []
    for s in args.scales:
        print(f"\n########## anchor scale = {s} ##########")
        print("--- inclusive (10-way) ---")
        a, acc = run_inclusive(s, train_loader, test_loader, ssl_ep, probe_ep)
        print("--- holdout-4 (4-vs-rest) ---")
        h = run_holdout(s, emb_loader, probe_loader, test_loader_h, ssl_ep, probe_ep)
        inc_auc.append(a); inc_acc.append(acc); hold_auc.append(h)
        print(f"  >> scale={s}: inclusive micro-AUC={a:.4f} acc={acc:.4f} | holdout4 AUC={h:.4f}")

    print("\n===== SUMMARY =====")
    print(f"{'scale':>8}{'inc_microAUC':>14}{'inc_acc':>10}{'holdout4_AUC':>14}")
    for s, a, acc, h in zip(args.scales, inc_auc, inc_acc, hold_auc):
        print(f"{s:>8.1f}{a:>14.4f}{acc:>10.4f}{h:>14.4f}")

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(args.scales, inc_auc, "o-", color="C0", label="inclusive micro-AUC (10-way)")
    ax1.plot(args.scales, hold_auc, "s-", color="C3", label="hold-out-4 AUC (4-vs-rest)")
    ax1.set_xlabel("anchor initialization scale")
    ax1.set_ylabel("ROC AUC")
    ax1.set_title("Learnable-means SIGReg: effect of anchor init scale")
    ax1.grid(alpha=0.3); ax1.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(plot_path("anchor_scan.png"), dpi=150)
    plt.close(fig)
    print(f"\n  saved {plot_path('anchor_scan.png')}")


if __name__ == "__main__":
    main()
