"""
Dual-space embedding: SSL space (augmentations) + supervised space, same input,
concatenated at the end.

    SSL space (64d)   : trunk trained with SIGReg-SSL (two-view invariance +
                        global N(0,I); no labels, holdouts excluded).
    sup space (16d)   : a copy of that trunk fine-tuned (floating) with the
                        tuned classwise SIGReg (proto, w=20) or SupCon.
    concat (80d)      : [sup ; ssl] evaluated jointly.

For each space: probed binary AUC (linear head on frozen embeddings, trained
with labeled holdout samples as usual) and a probe-free center-distance score
(learned means for the proto sup space, empirical class centers elsewhere;
cosine for the SupCon sup space).  Concat also gets per-class Mahalanobis and
the within-class eigenspectrum.  Leakage-free: nothing sees the held-out
classes before the probe stage.  CIFAR-10, seed 0.

Outputs:
    plots/novelty_concat_cifar10.png
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from supersig.config import plot_path, DEVICE
from supersig.data import (
    get_cifar_loaders, cifar_balanced_loader, cifar_two_view_loader,
)
from supersig.models import CIFARResNetBackbone
from supersig.losses import make_anchors
from supersig.train import (
    train_sigreg_ssl, train_simclr, train_supcon, train_sigreg_hybrid,
    collect_embeddings, REP_WEIGHT,
)
from supersig.metrics import mahalanobis_novelty

SUP_DIM = 16
SSL_DIM = 64          # overridden by --ssl-dim
N_CLASSES = 10
PAIR_DIST = 5.0
DATASET = "cifar10"
HOLDOUT_SETS = {1: [4], 2: [4, 9], 3: [0, 4, 9]}


class Stack(nn.Module):
    def __init__(self, trunk, head):
        super().__init__()
        self.trunk, self.head = trunk, head

    def forward(self, x):
        return self.head(self.trunk(x))


def binary_probe_auc(tr_X, tr_pos, te_X, te_pos, epochs=5, seed=0):
    """Linear binary probe trained on precomputed embeddings."""
    g = torch.Generator().manual_seed(seed)
    Xtr = torch.as_tensor(tr_X, dtype=torch.float32, device=DEVICE)
    ytr = torch.as_tensor(tr_pos, dtype=torch.long, device=DEVICE)
    Xte = torch.as_tensor(te_X, dtype=torch.float32, device=DEVICE)
    head = nn.Linear(Xtr.size(1), 2).to(DEVICE)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    for _ in range(epochs):
        for idx in torch.randperm(len(Xtr), generator=g).split(256):
            idx = idx.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(head(Xtr[idx]), ytr[idx]).backward()
            opt.step()
    with torch.no_grad():
        s = F.softmax(head(Xte), dim=1)[:, 1].cpu().numpy()
    return roc_auc_score(te_pos, s)


def center_score(te_X, centers):
    z = torch.as_tensor(te_X, dtype=torch.float32, device=DEVICE)
    return torch.cdist(z, centers).min(1).values.cpu().numpy()


def emp_centers(tr_X, tr_lab, seen):
    z = torch.as_tensor(tr_X, dtype=torch.float32, device=DEVICE)
    return torch.stack([z[torch.as_tensor(tr_lab == c, device=DEVICE)].mean(0)
                        for c in seen])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ssl-epochs", type=int, default=None)
    ap.add_argument("--head-epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arch", default="resnet20")
    ap.add_argument("--pretrain", default="cifar10")
    ap.add_argument("--ks", default="1,2,3")
    ap.add_argument("--sigreg-weight", type=float, default=20.0)
    ap.add_argument("--n-slices", type=int, default=256)
    ap.add_argument("--ssl-obj", choices=["sigreg", "simclr"], default="sigreg")
    ap.add_argument("--ssl-dim", type=int, default=64)
    ap.add_argument("--out-tag", default="")
    args = ap.parse_args()
    global SSL_DIM
    SSL_DIM = args.ssl_dim
    ssl_ep = args.ssl_epochs or (2 if args.quick else 20)
    head_ep = args.head_epochs or (1 if args.quick else 10)
    ks = [int(x) for x in args.ks.split(",")]
    methods = ["sigreg+proto", "supcon"]

    train_loader, test_loader = get_cifar_loaders(quick=args.quick, limit=args.limit,
                                                  dataset=DATASET)
    # fixed-order train loader so multi-pass embedding collections align row-wise
    train_eval_loader = torch.utils.data.DataLoader(
        train_loader.dataset, batch_size=256, shuffle=False, num_workers=2)
    res = {m: {} for m in methods}
    for k in ks:
        holdouts = set(HOLDOUT_SETS[k])
        seen = [c for c in range(N_CLASSES) if c not in holdouts]
        print(f"\n===== k={k}: SSL trunk ({args.ssl_obj}, {SSL_DIM}d, no {sorted(holdouts)}) =====")
        torch.manual_seed(args.seed); np.random.seed(args.seed)
        trunk = CIFARResNetBackbone(SSL_DIM, arch=args.arch,
                                    pretrain=args.pretrain).to(DEVICE)
        ssl_loader = cifar_two_view_loader(quick=args.quick, labeled=False,
                                           holdout=holdouts, limit=args.limit,
                                           dataset=DATASET)
        if args.ssl_obj == "simclr":
            train_simclr(trunk, ssl_loader, ssl_ep)
        else:
            train_sigreg_ssl(trunk, ssl_loader, ssl_ep)
        tr_ssl, tr_lab = collect_embeddings(trunk, train_eval_loader)
        te_ssl, te_lab = collect_embeddings(trunk, test_loader)
        tr_pos = np.isin(tr_lab, list(holdouts)).astype(int)
        te_pos = np.isin(te_lab, list(holdouts)).astype(int)
        ssl_probe = binary_probe_auc(tr_ssl, tr_pos, te_ssl, te_pos)
        ssl_free = roc_auc_score(te_pos, center_score(
            te_ssl, emp_centers(tr_ssl[tr_pos == 0], tr_lab[tr_pos == 0], seen)))

        for m in methods:
            print(f"\n=== k={k}: supervised branch {m} ===")
            torch.manual_seed(args.seed + 1); np.random.seed(args.seed + 1)
            model = Stack(copy.deepcopy(trunk),
                          nn.Sequential(nn.Linear(SSL_DIM, 64), nn.ReLU(),
                                        nn.Linear(64, SUP_DIM)).to(DEVICE))
            means = None
            if m == "sigreg+proto":
                means = make_anchors(PAIR_DIST / math.sqrt(2.0), emb_dim=SUP_DIM,
                                     n_classes=N_CLASSES).clone()
                train_sigreg_hybrid(model, cifar_balanced_loader(
                    DATASET, holdout=holdouts, quick=args.quick, limit=args.limit),
                    head_ep, means, mode="repulse", disc="proto", alpha=1.0,
                    rep_weight=REP_WEIGHT, sigreg_weight=args.sigreg_weight,
                    n_slices=args.n_slices)
            else:
                train_supcon(model, cifar_two_view_loader(
                    quick=args.quick, labeled=True, holdout=holdouts,
                    limit=args.limit, dataset=DATASET), head_ep)
            tr_sup, _ = collect_embeddings(model, train_eval_loader)
            te_sup, _ = collect_embeddings(model, test_loader)
            tr_cat = np.concatenate([tr_sup, tr_ssl], axis=1)
            te_cat = np.concatenate([te_sup, te_ssl], axis=1)

            sup_probe = binary_probe_auc(tr_sup, tr_pos, te_sup, te_pos)
            cat_probe = binary_probe_auc(tr_cat, tr_pos, te_cat, te_pos)
            if means is not None:
                sup_centers = means.detach()[seen]
                sup_free = roc_auc_score(te_pos, center_score(te_sup, sup_centers))
            else:
                zt = F.normalize(torch.as_tensor(
                    tr_sup[tr_pos == 0], dtype=torch.float32, device=DEVICE), dim=1)
                cents = F.normalize(emp_centers(
                    F.normalize(torch.as_tensor(tr_sup[tr_pos == 0],
                                                dtype=torch.float32,
                                                device=DEVICE), dim=1).cpu().numpy(),
                    tr_lab[tr_pos == 0], seen), dim=1)
                zn = F.normalize(torch.as_tensor(te_sup, dtype=torch.float32,
                                                 device=DEVICE), dim=1)
                sup_free = roc_auc_score(
                    te_pos, (1.0 - (zn @ cents.t()).max(1).values).cpu().numpy())
                sup_centers = emp_centers(tr_sup[tr_pos == 0],
                                          tr_lab[tr_pos == 0], seen)
            ssl_centers = emp_centers(tr_ssl[tr_pos == 0], tr_lab[tr_pos == 0], seen)
            cat_centers = torch.cat([sup_centers, ssl_centers], dim=1)
            cat_free = roc_auc_score(te_pos, center_score(te_cat, cat_centers))
            _, percls, eigs = mahalanobis_novelty(
                tr_cat[tr_pos == 0], tr_lab[tr_pos == 0], te_cat, seen)
            cat_mahal = roc_auc_score(te_pos, percls)

            res[m][k] = dict(sup_probe=sup_probe, ssl_probe=ssl_probe,
                             cat_probe=cat_probe, sup_free=sup_free,
                             ssl_free=ssl_free, cat_free=cat_free,
                             cat_mahal=cat_mahal, eigs=eigs)
            r = res[m][k]
            print(f"  probed  sup={r['sup_probe']:.4f}  ssl={r['ssl_probe']:.4f}  "
                  f"concat={r['cat_probe']:.4f}")
            print(f"  free    sup={r['sup_free']:.4f}  ssl={r['ssl_free']:.4f}  "
                  f"concat={r['cat_free']:.4f}  (concat mahal-pc={r['cat_mahal']:.4f})")
            print(f"  concat eig min/med/max={eigs[0]:.3f}/{eigs[1]:.3f}/{eigs[2]:.3f}")

    print(f"\n===== CONCAT SUMMARY (ssl={args.ssl_obj}, {SSL_DIM}d) =====")
    print(f"{'k':>3}{'method':>14}{'probed s/ssl/cat':>26}{'free s/ssl/cat':>26}")
    for k in ks:
        for m in methods:
            r = res[m][k]
            print(f"{k:>3}{m:>14}"
                  f"{r['sup_probe']:>10.4f}/{r['ssl_probe']:.4f}/{r['cat_probe']:.4f}"
                  f"{r['sup_free']:>10.4f}/{r['ssl_free']:.4f}/{r['cat_free']:.4f}")

    plt.figure(figsize=(8, 5.5))
    for i, m in enumerate(methods):
        for style, key, lbl in [("-o", "cat_probe", "concat probed"),
                                ("-.^", "cat_free", "concat probe-free"),
                                ("--s", "sup_probe", "sup-only probed"),
                                (":v", "sup_free", "sup-only probe-free")]:
            plt.plot(ks, [res[m][k][key] for k in ks], f"C{i}{style}",
                     lw=1.8 if "cat" in key else 1.0,
                     alpha=1.0 if "cat" in key else 0.5, label=f"{m} {lbl}")
    plt.xticks(ks, [str(k) for k in ks])
    plt.xlabel("classes held out (k)"); plt.ylabel("unseen-vs-rest AUC")
    plt.title("CIFAR-10: concatenated SSL + supervised spaces")
    plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(plot_path(f"novelty_concat_cifar10{args.out_tag}.png"), dpi=150); plt.close()
    print(f"\n  saved {plot_path(f'novelty_concat_cifar10{args.out_tag}.png')}")
    print("Done.")


if __name__ == "__main__":
    main()
