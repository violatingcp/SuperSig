"""
Class-conditional SIGReg with a chosen mean-geometry strategy, frozen linear probe.

--mode fixed       fixed orthogonal anchors (means not trained)
--mode learnmeans  learnable means + hinge separation term
--mode repulse     learnable means + inverse-square repulsion + shrinkage
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import numpy as np
import torch
import torch.nn as nn

from nllreg.config import plot_path, EMB_DIM, N_CLASSES
from nllreg.data import get_loaders
from nllreg.models import ConvBackbone
from nllreg.losses import make_anchors
from nllreg.train import (
    train_sigreg_classwise, train_linear_probe, collect_probs, collect_embeddings,
)
from nllreg.plotting import plot_roc, plot_corner

TITLES = {
    "fixed": "fixed anchors",
    "learnmeans": "learnable means (hinge)",
    "repulse": "repulsive means",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=list(TITLES), default="fixed")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--ssl-epochs", type=int, default=None)
    ap.add_argument("--probe-epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    ssl_ep = args.ssl_epochs or (2 if args.quick else 8)
    probe_ep = args.probe_epochs or (1 if args.quick else 4)

    train_loader, test_loader = get_loaders(batch_size=256, quick=args.quick)

    means = make_anchors().clone()
    backbone = ConvBackbone()
    train_sigreg_classwise(backbone, train_loader, ssl_ep, means,
                           learn_means=(args.mode != "fixed"), mode=args.mode)

    head = nn.Linear(EMB_DIM, N_CLASSES)
    train_linear_probe(backbone, head, train_loader, probe_ep)
    probs, labels = collect_probs(lambda x: head(backbone(x)), test_loader)
    plot_roc(probs, labels,
             f"Class-conditional SIGReg ({TITLES[args.mode]}) + linear head ROC",
             plot_path(f"roc_sigreg_{args.mode}_linear.png"))

    embs, elab = collect_embeddings(backbone, test_loader)
    plot_corner(embs, elab, plot_path(f"corner_sigreg_{args.mode}_16d.png"),
                title=f"Class-conditional SIGReg ({TITLES[args.mode]}) 16-dim latent")


if __name__ == "__main__":
    main()
