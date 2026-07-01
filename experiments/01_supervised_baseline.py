"""Baseline: CNN trained end-to-end with categorical cross-entropy + ROC."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import numpy as np
import torch

from nllreg.config import plot_path
from nllreg.data import get_loaders
from nllreg.models import SupervisedCNN
from nllreg.train import train_supervised, collect_probs
from nllreg.plotting import plot_roc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    epochs = args.epochs or (1 if args.quick else 3)

    train_loader, test_loader = get_loaders(quick=args.quick)
    model = SupervisedCNN()
    train_supervised(model, train_loader, epochs)
    probs, labels = collect_probs(lambda x: model(x), test_loader)
    plot_roc(probs, labels, "MNIST supervised CNN ROC", plot_path("roc_supervised.png"))


if __name__ == "__main__":
    main()
