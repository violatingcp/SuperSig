"""
Empirical per-class Mahalanobis novelty vs the unit-covariance assumption.

SIGReg's design goal is a latent that IS a Mahalanobis space: every class
N(mu_c, I), so no empirical covariance estimation should be needed.  Exp 18
showed the unit-covariance density fails (novel classes embed onto related
seen shells, displaced directionally).  Here we fit the seen classes
empirically (train-set embeddings) and score novelty the Lee et al. (2018)
way, plus diagnose how far each latent is from the unit-covariance ideal:

    mahal-tied : min_c Mahalanobis(z; mu_c, Sigma_shared), Sigma_shared =
                 pooled within-class covariance (+1e-3 I)
    mahal-pc   : per-class Sigma_c with shrinkage (0.9 S_c + 0.1 tr/d I)
    anisotropy : eigenvalue spread of Sigma_shared -- ~1 everywhere iff the
                 space is truly unit-Mahalanobis as designed

Same recipe/holdout sets/seeds as exps 17/18 (CIFAR-100, 100-dim, 5-sigma).

Outputs:
    plots/novelty_mahalanobis_cifar100.png   AUC vs k
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
    1: [4], 2: [4, 70], 3: [4, 30, 70],
    10: [4, 14, 24, 34, 44, 54, 64, 74, 84, 94],
    20: [4 + 5 * i for i in range(20)],
}
PROBED = {
    "sigreg+proto": {1: 0.9198, 2: 0.8912, 3: 0.8715, 10: 0.7940, 20: 0.6938},
    "sigreg+ce":    {1: 0.9488, 2: 0.9078, 3: 0.8828, 10: 0.8152, 20: 0.7023},
    "supcon":       {1: 0.9245, 2: 0.8872, 3: 0.8984, 10: 0.8224, 20: 0.7423},
}
UNIT18 = {   # exp-18 unit-covariance scores (naive likelihood), for reference
    "sigreg+proto": {1: 0.3804, 2: 0.5462, 3: 0.4583, 10: 0.5550, 20: 0.5860},
    "sigreg+ce":    {1: 0.5436, 2: 0.4940, 3: 0.4839, 10: 0.4963, 20: 0.5100},
    "supcon":       {1: 0.6891, 2: 0.6786, 3: 0.6456, 10: 0.6572, 20: 0.7001},
}


def train_backbone(method, holdouts, ssl_ep, args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    backbone = CIFARResNetBackbone(EMB_DIM, arch=args.arch, pretrain=args.pretrain).to(DEVICE)
    if method.startswith("sigreg+"):
        means = make_anchors(PAIR_DIST / math.sqrt(2.0), emb_dim=EMB_DIM,
                             n_classes=N_CLASSES).clone()
        rep_w = REP_WEIGHT * 45.0 / (N_CLASSES * (N_CLASSES - 1) / 2)
        loader = cifar_balanced_loader(DATASET, holdout=holdouts, quick=args.quick,
                                       limit=args.limit)
        train_sigreg_hybrid(backbone, loader, ssl_ep, means, mode="repulse",
                            disc=method.split("+", 1)[1], alpha=1.0, rep_weight=rep_w)
    else:
        train_supcon(backbone, cifar_two_view_loader(
            quick=args.quick, labeled=True, holdout=holdouts,
            limit=args.limit, dataset=DATASET), ssl_ep)
    return backbone


@torch.no_grad()
def mahalanobis_scores(tr_embs, tr_lab, te_embs, seen):
    zt = torch.as_tensor(tr_embs, device=DEVICE)
    z = torch.as_tensor(te_embs, device=DEVICE)
    d = zt.size(1)
    mus, diffs, covs = [], [], []
    for c in seen:
        zc = zt[torch.as_tensor(tr_lab == c, device=DEVICE)]
        mu = zc.mean(0)
        mus.append(mu)
        diffs.append(zc - mu)
        S = (zc - mu).t() @ (zc - mu) / max(len(zc) - 1, 1)
        covs.append(0.9 * S + 0.1 * (torch.trace(S) / d) * torch.eye(d, device=DEVICE))
    D = torch.cat(diffs)
    S_tied = D.t() @ D / max(len(D) - 1, 1) + 1e-3 * torch.eye(d, device=DEVICE)
    L_tied = torch.linalg.cholesky(S_tied)

    def min_m2(L_per_class):
        m2 = []
        for mu, L in zip(mus, L_per_class):
            w = torch.linalg.solve_triangular(L, (z - mu).t(), upper=False)
            m2.append((w ** 2).sum(0))
        return torch.stack(m2).min(0).values

    tied = min_m2([L_tied] * len(mus)).sqrt().cpu().numpy()
    percls = min_m2([torch.linalg.cholesky(S) for S in covs]).sqrt().cpu().numpy()
    evals = torch.linalg.eigvalsh(S_tied)
    aniso = (evals.min().item(), evals.median().item(), evals.max().item())
    return tied, percls, aniso


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
    res = {m: {} for m in methods}
    for k in ks:
        holdouts = set(HOLDOUT_SETS[k])
        seen = [c for c in range(N_CLASSES) if c not in holdouts]
        for m in methods:
            print(f"\n=== k={k}: {m} (Mahalanobis) ===")
            backbone = train_backbone(m, holdouts, ssl_ep, args)
            tr_embs, tr_lab = collect_embeddings(backbone, train_plain_loader)
            tr_keep = np.isin(tr_lab, seen)
            te_embs, te_lab = collect_embeddings(backbone, test_loader)
            tied, percls, aniso = mahalanobis_scores(
                tr_embs[tr_keep], tr_lab[tr_keep], te_embs, seen)
            is_unseen = np.isin(te_lab, list(holdouts)).astype(int)
            res[m][k] = {
                "tied": roc_auc_score(is_unseen, tied),
                "percls": roc_auc_score(is_unseen, percls),
                "aniso": aniso,
            }
            r = res[m][k]
            print(f"  mahal-tied AUC={r['tied']:.4f}  mahal-perclass AUC={r['percls']:.4f}")
            print(f"  within-class cov eigenvalues (min/med/max): "
                  f"{aniso[0]:.3f} / {aniso[1]:.3f} / {aniso[2]:.3f}   "
                  f"(unit-Mahalanobis ideal: 1/1/1)")

    print(f"\n{'k':>4}" + "".join(f"{m:>14} tied/pc/unit/probed" for m in methods))
    for k in ks:
        row = f"{k:>4}"
        for m in methods:
            r = res[m][k]
            row += (f"{r['tied']:>12.4f}/{r['percls']:.4f}"
                    f"/{UNIT18[m][k]:.4f}/{PROBED[m][k]:.4f}")
        print(row)

    plt.figure(figsize=(8, 5.5))
    for i, m in enumerate(methods):
        plt.plot(ks, [res[m][k]["tied"] for k in ks], f"C{i}-o", lw=2,
                 label=f"{m} Mahalanobis tied")
        plt.plot(ks, [res[m][k]["percls"] for k in ks], f"C{i}-.^", lw=1.2,
                 alpha=0.8, label=f"{m} Mahalanobis per-class")
        plt.plot(ks, [PROBED[m][k] for k in ks], f"C{i}--s", lw=1.2, alpha=0.6,
                 label=f"{m} probed")
    plt.xscale("log"); plt.xticks(ks, [str(k) for k in ks])
    plt.xlabel("classes held out (k)"); plt.ylabel("unseen-vs-rest AUC")
    plt.title("CIFAR-100 novelty: empirical Mahalanobis (probe-free) vs probes")
    plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(plot_path("novelty_mahalanobis_cifar100.png"), dpi=150); plt.close()
    print(f"\n  saved {plot_path('novelty_mahalanobis_cifar100.png')}")
    print("Done.")


if __name__ == "__main__":
    main()
