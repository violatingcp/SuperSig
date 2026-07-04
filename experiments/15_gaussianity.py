"""
Per-class Gaussianity of the learned latent spaces: SIGReg+proto vs SupCon.

Metric (supersig/metrics.py): calibrated sliced-Wasserstein Gaussianity ratio.
Each class's test embeddings are projected onto random unit directions, each
1-D projection is standardized (shape only -- location/scale removed), and the
squared W2 distance to the standard-normal quantiles is averaged over slices,
then divided by the same statistic for true N(0, I) samples of identical (n, d):

    ratio ~ 1   as Gaussian as a finite sample can look
    ratio >> 1  structured deviation (sub-clusters, heavy tails, skew)

Trains the two champion CIFAR-10 embeddings from the series (no probes needed):
    sigreg+proto : classwise SIGReg, repulsive means, 3-sigma seed, proto term
    supcon       : supervised contrastive, two-view augmentation

Outputs (plots/):
    gaussianity_cifar10.png   per-class ratio, both methods (log scale)
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

from supersig.config import plot_path, N_CLASSES, DEVICE
from supersig.data import get_cifar_loaders, cifar_two_view_loader
from supersig.models import CIFARResNetBackbone
from supersig.losses import make_anchors
from supersig.train import train_sigreg_hybrid, train_supcon, collect_embeddings
from supersig.metrics import classwise_gaussianity

EMB_DIM = 32
CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ssl-epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arch", default="resnet20")
    ap.add_argument("--pretrain", default="cifar10")
    ap.add_argument("--pair-dist", type=float, default=3.0)
    ap.add_argument("--alpha", type=float, default=1.0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    ssl_ep = args.ssl_epochs or (2 if args.quick else 10)

    train_loader, test_loader = get_cifar_loaders(quick=args.quick, limit=args.limit)

    print("\n=== embedding: sigreg+proto ===")
    means = make_anchors(args.pair_dist / math.sqrt(2.0), emb_dim=EMB_DIM).clone()
    sig = CIFARResNetBackbone(EMB_DIM, arch=args.arch, pretrain=args.pretrain).to(DEVICE)
    train_sigreg_hybrid(sig, train_loader, ssl_ep, means, mode="repulse",
                        disc="proto", alpha=args.alpha)

    print("\n=== embedding: supcon ===")
    sup = CIFARResNetBackbone(EMB_DIM, arch=args.arch, pretrain=args.pretrain).to(DEVICE)
    train_supcon(sup, cifar_two_view_loader(quick=args.quick, labeled=True,
                                            limit=args.limit), ssl_ep)

    results = {}
    for name, bb in [("SIGReg+proto", sig), ("SupCon", sup)]:
        embs, labels = collect_embeddings(bb, test_loader)
        results[name] = classwise_gaussianity(embs, labels)

    print(f"\n{'class':<12}" + "".join(f"{n:>16}" for n in results)
          + f"{'SIGReg skew':>13}{'SIGReg kurt':>13}")
    for c in range(N_CLASSES):
        row = f"{CIFAR_CLASSES[c]:<12}"
        for n in results:
            row += f"{results[n][c]['ratio']:>16.2f}"
        row += f"{results['SIGReg+proto'][c]['skew']:>13.3f}"
        row += f"{results['SIGReg+proto'][c]['ex_kurtosis']:>13.3f}"
        print(row)
    for n in results:
        mean_r = np.mean([results[n][c]["ratio"] for c in range(N_CLASSES)])
        print(f"  {n:<14} mean ratio = {mean_r:.2f}")

    x = np.arange(N_CLASSES)
    w = 0.38
    plt.figure(figsize=(9, 5))
    for i, (n, res) in enumerate(results.items()):
        plt.bar(x + (i - 0.5) * w, [res[c]["ratio"] for c in range(N_CLASSES)],
                width=w, label=n)
    plt.axhline(1.0, color="k", ls=":", lw=1, label="true-Gaussian level")
    plt.yscale("log")
    plt.xticks(x, CIFAR_CLASSES, rotation=45, ha="right")
    plt.ylabel("sliced-Wasserstein Gaussianity ratio (log)")
    plt.title("Per-class Gaussianity of the 32-dim CIFAR-10 latent (1 = Gaussian)")
    plt.legend(); plt.tight_layout()
    plt.savefig(plot_path("gaussianity_cifar10.png"), dpi=150); plt.close()
    print(f"\n  saved {plot_path('gaussianity_cifar10.png')}")
    print("Done.")


if __name__ == "__main__":
    main()
