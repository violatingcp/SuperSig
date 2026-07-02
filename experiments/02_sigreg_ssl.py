"""Self-supervised SIGReg (invariance + global isotropic Gaussian), frozen linear probe."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import numpy as np
import torch
import torch.nn as nn

from supersig.config import plot_path, EMB_DIM, N_CLASSES
from supersig.data import get_loaders, two_view_loader
from supersig.models import ConvBackbone
from supersig.train import train_sigreg_ssl, train_linear_probe, collect_probs, collect_embeddings
from supersig.plotting import plot_roc, plot_corner


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

    train_loader, test_loader = get_loaders(quick=args.quick)
    tv = two_view_loader(quick=args.quick)

    backbone = ConvBackbone()
    train_sigreg_ssl(backbone, tv, ssl_ep)

    head = nn.Linear(EMB_DIM, N_CLASSES)
    train_linear_probe(backbone, head, train_loader, probe_ep)
    probs, labels = collect_probs(lambda x: head(backbone(x)), test_loader)
    plot_roc(probs, labels, "SIGReg (SSL) embedding + frozen linear head ROC",
             plot_path("roc_sigreg_linear.png"))

    embs, elab = collect_embeddings(backbone, test_loader)
    plot_corner(embs, elab, plot_path("corner_sigreg_16d.png"),
                title="SIGReg 16-dim latent space (colored by digit)")


if __name__ == "__main__":
    main()
