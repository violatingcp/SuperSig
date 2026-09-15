"""
Probe-free novelty detection from the Gaussian latent: grade SIGReg on its own
exam.

The multi-holdout studies (exp 17) scored "unseen vs rest" with a trained
binary probe -- a discriminative game that never uses SIGReg's structure.  Here
the novelty score is the model's own likelihood:

    SIGReg (proto / ce) : score(z) = min over SEEN class means of ||z - mu_c||
                          (learned means; low max-likelihood = novel).
    SupCon              : nearest-centroid analogue, 1 - max cosine similarity
                          to seen-class centroids estimated from train data.

No probe is trained; nothing ever sees the held-out classes.  Same holdout
sets, recipe, and seeds as the probed runs (CIFAR-100, 100-dim, 5-sigma seed,
repulsion x1, plain images, 10 SSL epochs).

Outputs:
    plots/novelty_likelihood_cifar100.png   AUC vs k, probe-free vs probed
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from supersig.config import plot_path, DEVICE
from supersig.data import get_cifar_loaders, cifar_two_view_loader, cifar_balanced_loader
from supersig.models import CIFARResNetBackbone
from supersig.losses import make_anchors
from supersig.train import train_sigreg_hybrid, train_supcon, collect_embeddings, REP_WEIGHT

EMB_DIM = 100
N_CLASSES = 100
PAIR_DIST = 5.0
DATASET = "cifar100"

HOLDOUT_SETS = {
    1: [4],
    2: [4, 70],
    3: [4, 30, 70],
    10: [4, 14, 24, 34, 44, 54, 64, 74, 84, 94],
    20: [4 + 5 * i for i in range(20)],
}

# combined AUCs from the probed runs (17), for comparison
PROBED = {
    "sigreg+proto": {1: 0.9198, 2: 0.8912, 3: 0.8715, 10: 0.7940, 20: 0.6938},
    "sigreg+ce":    {1: 0.9488, 2: 0.9078, 3: 0.8828, 10: 0.8152, 20: 0.7023},
    "supcon":       {1: 0.9245, 2: 0.8872, 3: 0.8984, 10: 0.8224, 20: 0.7423},
}


def train_and_score(method, holdouts, ssl_ep, args, test_loader, train_plain_loader):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    backbone = CIFARResNetBackbone(EMB_DIM, arch=args.arch, pretrain=args.pretrain).to(DEVICE)
    seen = [c for c in range(N_CLASSES) if c not in holdouts]
    if method.startswith("sigreg+"):
        means = make_anchors(PAIR_DIST / math.sqrt(2.0), emb_dim=EMB_DIM,
                             n_classes=N_CLASSES).clone()
        rep_w = REP_WEIGHT * 45.0 / (N_CLASSES * (N_CLASSES - 1) / 2)
        loader = cifar_balanced_loader(DATASET, holdout=holdouts, quick=args.quick,
                                       limit=args.limit)
        train_sigreg_hybrid(backbone, loader, ssl_ep, means, mode="repulse",
                            disc=method.split("+", 1)[1], alpha=1.0, rep_weight=rep_w)
        centers = means.detach()[seen]                       # learned seen means
        embs, ylab = collect_embeddings(backbone, test_loader)
        z = torch.as_tensor(embs, device=DEVICE)
        dmin = torch.cdist(z, centers).min(dim=1).values
        scores = {
            # naive likelihood: closer to a seen mean = more likely = less novel
            "likelihood": dmin.cpu().numpy(),
            # typicality: a class sample lives on the sqrt(d) shell of its own
            # Gaussian; novelty = distance from the nearest seen shell
            "typicality": (dmin - math.sqrt(EMB_DIM)).abs().cpu().numpy(),
        }
    else:
        train_supcon(backbone, cifar_two_view_loader(
            quick=args.quick, labeled=True, holdout=holdouts,
            limit=args.limit, dataset=DATASET), ssl_ep)
        tr_embs, tr_lab = collect_embeddings(backbone, train_plain_loader)
        zt = F.normalize(torch.as_tensor(tr_embs, device=DEVICE), dim=1)
        cents = torch.stack([zt[torch.as_tensor(tr_lab == c, device=DEVICE)].mean(0)
                             for c in seen])
        cents = F.normalize(cents, dim=1)                    # empirical seen centroids
        embs, ylab = collect_embeddings(backbone, test_loader)
        z = F.normalize(torch.as_tensor(embs, device=DEVICE), dim=1)
        cos_d = 1.0 - (z @ cents.t()).max(dim=1).values
        med = cos_d.median()                # typical seen distance to own centroid
        scores = {
            "likelihood": cos_d.cpu().numpy(),
            "typicality": (cos_d - med).abs().cpu().numpy(),
        }
    is_unseen = np.isin(ylab, list(holdouts)).astype(int)
    return {name: roc_auc_score(is_unseen, sc) for name, sc in scores.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ssl-epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arch", default="resnet20")
    ap.add_argument("--pretrain", default="cifar100")
    ap.add_argument("--ks", default="1,2,3,10,20")
    args = ap.parse_args()
    ssl_ep = args.ssl_epochs or (2 if args.quick else 10)
    ks = [int(x) for x in args.ks.split(",")]

    train_plain_loader, test_loader = get_cifar_loaders(quick=args.quick,
                                                        limit=args.limit, dataset=DATASET)
    methods = ["sigreg+proto", "sigreg+ce", "supcon"]
    free = {m: {} for m in methods}
    for k in ks:
        holdouts = set(HOLDOUT_SETS[k])
        for m in methods:
            print(f"\n=== k={k}: {m} (probe-free) ===")
            a = train_and_score(m, holdouts, ssl_ep, args, test_loader, train_plain_loader)
            free[m][k] = a
            print(f"  likelihood AUC={a['likelihood']:.4f}   typicality AUC={a['typicality']:.4f}"
                  f"   (probed reference: {PROBED[m][k]:.4f})")

    print(f"\n{'k':>4}" + "".join(f"{m:>16} lik/typ/probed" for m in methods))
    for k in ks:
        row = f"{k:>4}"
        for m in methods:
            row += (f"{free[m][k]['likelihood']:>12.4f}/{free[m][k]['typicality']:.4f}"
                    f"/{PROBED[m][k]:.4f}")
        print(row)

    plt.figure(figsize=(8, 5.5))
    for i, m in enumerate(methods):
        plt.plot(ks, [free[m][k]["typicality"] for k in ks], f"C{i}-o", lw=2,
                 label=f"{m} typicality (probe-free)")
        plt.plot(ks, [free[m][k]["likelihood"] for k in ks], f"C{i}:^", lw=1.2,
                 alpha=0.7, label=f"{m} naive likelihood")
        plt.plot(ks, [PROBED[m][k] for k in ks], f"C{i}--s", lw=1.2, alpha=0.7,
                 label=f"{m} probed")
    plt.xscale("log"); plt.xticks(ks, [str(k) for k in ks])
    plt.xlabel("classes held out (k)"); plt.ylabel("unseen-vs-rest AUC")
    plt.title("CIFAR-100 novelty detection: model likelihood vs trained probe")
    plt.legend(fontsize=9); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(plot_path("novelty_likelihood_cifar100.png"), dpi=150); plt.close()
    print(f"\n  saved {plot_path('novelty_likelihood_cifar100.png')}")
    print("Done.")


if __name__ == "__main__":
    main()
