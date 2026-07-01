"""
Supervised SimCLR (SupCon) embedding: closed-set (10-way) and hold-out-4 studies.

Runs two experiments:
    (1) no holdout -> 10-way linear probe, ROC + corner
    (2) holdout 4  -> 4-vs-rest binary probe, ROC + corner
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import numpy as np
import torch
import torch.nn as nn

from nllreg.config import plot_path, EMB_DIM, N_CLASSES, HOLDOUT
from nllreg.data import build_holdout_loaders, two_view_loader
from nllreg.models import ConvBackbone
from nllreg.train import (
    train_supcon, train_linear_probe, train_binary_probe,
    collect_probs, collect_binary_scores, collect_embeddings,
)
from nllreg.plotting import plot_roc, plot_binary_roc, plot_corner


def run_no_holdout(quick, ssl_ep, probe_ep):
    print("\n===== SupCon, NO holdout (10-way) =====")
    tv = two_view_loader(quick=quick, labeled=True)
    _, probe_loader, test_loader = build_holdout_loaders(quick=quick)

    backbone = ConvBackbone()
    train_supcon(backbone, tv, ssl_ep)

    head = nn.Linear(EMB_DIM, N_CLASSES)
    train_linear_probe(backbone, head, probe_loader, probe_ep)
    probs, labels = collect_probs(lambda x: head(backbone(x)), test_loader)
    plot_roc(probs, labels, "Supervised SimCLR (SupCon) + linear head ROC",
             plot_path("roc_supcon_linear.png"))
    embs, elab = collect_embeddings(backbone, test_loader)
    plot_corner(embs, elab, plot_path("corner_supcon_16d.png"),
                title="SupCon 16-dim latent space (colored by digit)")


def run_holdout(quick, ssl_ep, probe_ep):
    print("\n===== SupCon, HOLDOUT 4 (4-vs-rest) =====")
    tv = two_view_loader(quick=quick, labeled=True, holdout=HOLDOUT)
    _, probe_loader, test_loader = build_holdout_loaders(quick=quick)

    backbone = ConvBackbone()
    train_supcon(backbone, tv, ssl_ep)

    head = nn.Linear(EMB_DIM, 2)
    train_binary_probe(backbone, head, probe_loader, probe_ep)
    scores, ytrue = collect_binary_scores(backbone, head, test_loader)
    plot_binary_roc(scores, ytrue, f"SupCon hold-out-{HOLDOUT} detection ROC",
                    plot_path("roc_supcon_holdout4.png"), label="SupCon")
    embs, elab = collect_embeddings(backbone, test_loader)
    is4 = (elab == HOLDOUT).astype(int)
    plot_corner(embs, is4, plot_path("corner_supcon_holdout4.png"),
                title=f"SupCon hold-out-{HOLDOUT} latent: 1=digit {HOLDOUT} (unseen), 0=rest")


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

    run_no_holdout(args.quick, ssl_ep, probe_ep)
    run_holdout(args.quick, ssl_ep, probe_ep)
    print("\nDone.")


if __name__ == "__main__":
    main()
