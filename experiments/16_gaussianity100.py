"""
Per-class Gaussianity of the CIFAR-100 100-dim latents, one row per config of
the comparison table (all 5-sigma orthogonal seed, 10 SSL epochs, seed 0):

    proto x1 / x3 / x10 : Gaussian-posterior term, repulsion weight scaled
    CE (linear head)    : jointly trained linear-head cross-entropy
    SupCon (aug)        : supervised contrastive reference

Each embedding is retrained exactly as in the AUC runs (inclusive protocol,
no probes), its test-set per-class sliced-Wasserstein Gaussianity ratio is
computed (supersig/metrics.py; 1 = as Gaussian as a finite sample can look),
and the backbone is checkpointed to checkpoints/ for future reuse.

Outputs:
    plots/gaussianity_cifar100.png    mean per-class ratio per config
    checkpoints/cifar100_100d_<tag>.pt
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import datasets

from supersig.config import plot_path, DATA_DIR, REPO_DIR, DEVICE
from supersig.data import get_cifar_loaders, cifar_two_view_loader, cifar_balanced_loader
from supersig.models import CIFARResNetBackbone
from supersig.losses import make_anchors
from supersig.train import train_sigreg_hybrid, train_supcon, collect_embeddings, REP_WEIGHT
from supersig.metrics import classwise_gaussianity

EMB_DIM = 100
N_CLASSES = 100
PAIR_DIST = 5.0
DATASET = "cifar100"
CKPT_DIR = os.path.join(REPO_DIR, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

CONFIGS = [
    ("proto x1", "proto_x1", "proto", 1.0),
    ("proto x3", "proto_x3", "proto", 3.0),
    ("proto x10", "proto_x10", "proto", 10.0),
    ("CE (linear head)", "ce", "ce", 1.0),
    ("SupCon (aug)", "supcon", None, None),
]


def train_config(disc, rep_scale, ssl_ep, args):
    if disc is None:
        backbone = CIFARResNetBackbone(EMB_DIM, arch=args.arch, pretrain=args.pretrain).to(DEVICE)
        train_supcon(backbone, cifar_two_view_loader(quick=args.quick, labeled=True,
                                                     limit=args.limit, dataset=DATASET), ssl_ep)
        return backbone
    means = make_anchors(PAIR_DIST / math.sqrt(2.0), emb_dim=EMB_DIM,
                         n_classes=N_CLASSES).clone()
    rep_w = REP_WEIGHT * 45.0 / (N_CLASSES * (N_CLASSES - 1) / 2) * rep_scale
    backbone = CIFARResNetBackbone(EMB_DIM, arch=args.arch, pretrain=args.pretrain).to(DEVICE)
    loader = cifar_balanced_loader(DATASET, quick=args.quick, limit=args.limit)
    train_sigreg_hybrid(backbone, loader, ssl_ep, means, mode="repulse",
                        disc=disc, alpha=1.0, rep_weight=rep_w)
    return backbone


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ssl-epochs", type=int, default=None)
    ap.add_argument("--arch", default="resnet20")
    ap.add_argument("--pretrain", default="cifar100")
    args = ap.parse_args()
    ssl_ep = args.ssl_epochs or (2 if args.quick else 10)
    class_names = datasets.CIFAR100(DATA_DIR, train=False, download=True).classes

    _, test_loader = get_cifar_loaders(quick=args.quick, dataset=DATASET)
    rows = {}
    for label, tag, disc, rep in CONFIGS:
        print(f"\n=== config: {label} ===")
        torch.manual_seed(0); np.random.seed(0)     # replicate the AUC runs
        backbone = train_config(disc, rep, ssl_ep, args)
        torch.save(backbone.state_dict(),
                   os.path.join(CKPT_DIR, f"cifar100_{EMB_DIM}d_{tag}.pt"))
        embs, labels = collect_embeddings(backbone, test_loader)
        g = classwise_gaussianity(embs, labels)
        ratios = np.array([g[c]["ratio"] for c in sorted(g)])
        worst = int(np.argmax(ratios))
        rows[label] = (ratios.mean(), np.median(ratios), ratios.max(),
                       class_names[sorted(g)[worst]])
        print(f"  mean ratio={ratios.mean():.2f}  median={np.median(ratios):.2f}  "
              f"worst={ratios.max():.2f} ({rows[label][3]})")

    print(f"\n{'config':<20}{'mean':>8}{'median':>8}{'worst':>8}  worst class")
    for label, (m, md, w, wc) in rows.items():
        print(f"{label:<20}{m:>8.2f}{md:>8.2f}{w:>8.2f}  {wc}")

    labels_ = list(rows)
    plt.figure(figsize=(8, 4.5))
    plt.bar(range(len(rows)), [rows[l][0] for l in labels_], color="C0")
    plt.axhline(1.0, color="k", ls=":", lw=1, label="true-Gaussian level")
    plt.yscale("log")
    plt.xticks(range(len(rows)), labels_, rotation=20, ha="right")
    plt.ylabel("mean per-class Gaussianity ratio (log)")
    plt.title(f"CIFAR-100 {EMB_DIM}-dim latents: class Gaussianity (1 = Gaussian)")
    plt.legend(); plt.tight_layout()
    plt.savefig(plot_path("gaussianity_cifar100.png"), dpi=150); plt.close()
    print(f"\n  saved {plot_path('gaussianity_cifar100.png')}")
    print("Done.")


if __name__ == "__main__":
    main()
