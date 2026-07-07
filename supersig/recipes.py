"""
Settled default recipes from the CIFAR series (experiments 09-25).

RECIPES holds the converged per-dataset configuration:
  - cifar10  : 16-dim self-calibrated latent (w=20, 256 slices) -- eigenspectrum
               ~1, unit-covariance novelty works probe-free, discovery strong.
  - cifar100 : 32-dim, w=1 -- best discovery quality per compute; calibration
               is unattainable at 100 classes (see exp 20), so the empirical
               scores matter more than the unit ones here.

Both use 5-sigma-seeded floating means with inverse-square repulsion and the
Gaussian-posterior ("proto") discriminative term.

Entry points:
  supervised_embedding(dataset, holdouts=...)  -> backbone, means, cfg
  (then supersig.discovery.run_discovery for the open-world loop)
"""
import math
import numpy as np
import torch

from .config import DEVICE
from .data import cifar_balanced_loader
from .models import CIFARResNetBackbone
from .losses import make_anchors
from .train import train_sigreg_hybrid, REP_WEIGHT

RECIPES = {
    "cifar10": dict(n_classes=10, emb_dim=16, sigreg_weight=20.0, n_slices=256,
                    pair_dist=5.0, ssl_epochs=10, ft_epochs=5, arch="resnet20"),
    "cifar100": dict(n_classes=100, emb_dim=32, sigreg_weight=1.0, n_slices=64,
                     pair_dist=5.0, ssl_epochs=10, ft_epochs=5, arch="resnet20"),
}


def recipe(dataset, **overrides):
    cfg = dict(RECIPES[dataset])
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    cfg["rep_weight"] = REP_WEIGHT * 45.0 / (cfg["n_classes"] *
                                             (cfg["n_classes"] - 1) / 2)
    return cfg


def supervised_embedding(dataset, holdouts=None, quick=False, limit=None,
                         seed=0, pretrain=None, **overrides):
    """
    Train the settled supervised SIGReg+proto embedding.

    holdouts: iterable of class indices excluded from embedding training.
    Returns (backbone, means, cfg); `means` are the learned class anchors.
    """
    cfg = recipe(dataset, **overrides)
    torch.manual_seed(seed); np.random.seed(seed)
    backbone = CIFARResNetBackbone(cfg["emb_dim"], arch=cfg["arch"],
                                   pretrain=pretrain or dataset).to(DEVICE)
    means = make_anchors(cfg["pair_dist"] / math.sqrt(2.0),
                         emb_dim=cfg["emb_dim"],
                         n_classes=cfg["n_classes"]).clone()
    loader = cifar_balanced_loader(dataset, holdout=holdouts, quick=quick,
                                   limit=limit)
    train_sigreg_hybrid(backbone, loader, cfg["ssl_epochs"], means,
                        mode="repulse", disc="proto", alpha=1.0,
                        rep_weight=cfg["rep_weight"],
                        sigreg_weight=cfg["sigreg_weight"],
                        n_slices=cfg["n_slices"])
    return backbone, means, cfg
