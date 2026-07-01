"""
Hold-out-4 study: SIGReg embedding trained without digit 4, frozen, then a binary
"4 vs rest" linear probe -- does the embedding place the unseen digit in its own region?

--mode learnmeans | repulse | both
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import numpy as np
import torch
import torch.nn as nn

from nllreg.config import plot_path, EMB_DIM, HOLDOUT
from nllreg.data import build_holdout_loaders
from nllreg.models import ConvBackbone
from nllreg.losses import make_anchors
from nllreg.train import (
    train_sigreg_classwise, train_binary_probe, collect_binary_scores, collect_embeddings,
)
from nllreg.plotting import plot_binary_roc, plot_corner
import matplotlib.pyplot as plt


def run_mode(mode, emb_loader, probe_loader, test_loader, ssl_ep, probe_ep):
    print(f"\n===== MODE: {mode} (embedding trained WITHOUT digit {HOLDOUT}) =====")
    means = make_anchors().clone()
    backbone = ConvBackbone()
    train_sigreg_classwise(backbone, emb_loader, ssl_ep, means,
                           learn_means=True, mode=mode)

    print("  --- freeze, train 4-vs-rest linear head ---")
    head = nn.Linear(EMB_DIM, 2)
    train_binary_probe(backbone, head, probe_loader, probe_ep)
    scores, ytrue = collect_binary_scores(backbone, head, test_loader)
    fpr, tpr, roc_auc = plot_binary_roc(
        scores, ytrue,
        f"Hold-out-{HOLDOUT} detection ROC ({mode})",
        plot_path(f"roc_holdout4_{mode}.png"), label=mode)

    embs, ylab = collect_embeddings(backbone, test_loader)
    is4 = (ylab == HOLDOUT).astype(int)
    plot_corner(embs, is4, plot_path(f"corner_holdout4_{mode}.png"),
                title=f"Hold-out-{HOLDOUT} latent ({mode}): 1=digit {HOLDOUT} (unseen), 0=rest")
    return fpr, tpr, roc_auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["learnmeans", "repulse", "both"], default="both")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--ssl-epochs", type=int, default=None)
    ap.add_argument("--probe-epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    ssl_ep = args.ssl_epochs or (2 if args.quick else 8)
    probe_ep = args.probe_epochs or (1 if args.quick else 4)

    emb_loader, probe_loader, test_loader = build_holdout_loaders(quick=args.quick)
    modes = ["repulse", "learnmeans"] if args.mode == "both" else [args.mode]
    results = {m: run_mode(m, emb_loader, probe_loader, test_loader, ssl_ep, probe_ep)
               for m in modes}

    if len(results) > 1:
        plt.figure(figsize=(6, 6))
        for m, (fpr, tpr, a) in results.items():
            plt.plot(fpr, tpr, lw=2, label=f"{m} (AUC={a:.4f})")
        plt.plot([0, 1], [0, 1], "k:", lw=1)
        plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
        plt.title(f"Hold-out-{HOLDOUT} detection: repulsive vs learnable means")
        plt.legend(loc="lower right"); plt.tight_layout()
        plt.savefig(plot_path("roc_holdout4_compare.png"), dpi=150); plt.close()
        print("  saved roc_holdout4_compare.png")


if __name__ == "__main__":
    main()
