"""
Experiment 34 round 4 (final): plain concat [sup16 ; ssl16] with the tuned
halves -- supervised half at the round-1 winner recipe (w=1, batch 99x24,
10 epochs), SSL half at the round-2/3 sweet spot (40 two-view epochs).

Unlike ssl->supres (supervised training warm-started FROM the trunk, probe
0.9213), here the two halves never interact, so the supervised compression
cannot erode the label-free features. Targets: supcon 0.9268,
supcon+simclr 0.9394.

    python experiments/34d_cifar100_probe_round4.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import importlib
import math
import numpy as np
import torch
from torch.utils.data import DataLoader

from supersig.config import DEVICE
from supersig.data import (get_cifar_loaders, cifar_balanced_loader,
                           cifar_two_view_loader)
from supersig.losses import make_anchors
from supersig.models import CIFARResNetBackbone
from supersig.recipes import recipe
from supersig.train import train_sigreg_ssl, train_sigreg_hybrid, \
    collect_embeddings

exp28 = importlib.import_module("28_concat_residual")
exp29 = importlib.import_module("29_residual_finetune")

REF = {"ssl->supres [40ep] (r2)": 0.9213, "supcon (r1)": 0.9268,
       "supcon+simclr (r1)": 0.9394}
CPB, PC = 99, 24


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", type=int, default=4)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dim-half", type=int, default=16)
    args = ap.parse_args()
    ds = "cifar100"
    cfgH = recipe(ds, emb_dim=args.dim_half)
    n_cls = cfgH["n_classes"]
    holdouts = {args.holdout}
    seen = [c for c in range(n_cls) if c not in holdouts]
    print(f"exp34d [{ds}] round 4 (plain tuned concat), "
          f"holdout={sorted(holdouts)}")
    print("  refs: " + ", ".join(f"{k}={v:.4f}" for k, v in REF.items()))

    train_loader, test_loader = get_cifar_loaders(quick=args.quick, dataset=ds)
    train_eval_loader = DataLoader(train_loader.dataset, batch_size=256,
                                   shuffle=False, num_workers=2)

    print("\n----- sup16 (winner recipe) -----")
    torch.manual_seed(args.seed + 10); np.random.seed(args.seed + 10)
    sup = CIFARResNetBackbone(args.dim_half, arch=cfgH["arch"],
                              pretrain=ds).to(DEVICE)
    means = make_anchors(cfgH["pair_dist"] / math.sqrt(2.0),
                         emb_dim=args.dim_half, n_classes=n_cls).clone()
    train_sigreg_hybrid(sup, cifar_balanced_loader(ds, holdout=holdouts,
                                                   quick=args.quick,
                                                   classes_per_batch=CPB,
                                                   per_class=PC),
                        2 if args.quick else 10, means, mode="repulse",
                        disc="proto", alpha=1.0, rep_weight=cfgH["rep_weight"],
                        sigreg_weight=1.0, n_slices=cfgH["n_slices"])
    means = means.detach()

    print("\n----- ssl16 (40 two-view epochs) -----")
    torch.manual_seed(args.seed + 12); np.random.seed(args.seed + 12)
    ssl = CIFARResNetBackbone(args.dim_half, arch=cfgH["arch"],
                              pretrain=ds).to(DEVICE)
    train_sigreg_ssl(ssl, cifar_two_view_loader(quick=args.quick,
                                                labeled=False,
                                                holdout=holdouts, dataset=ds),
                     2 if args.quick else 40)
    e, l = collect_embeddings(ssl, train_eval_loader)
    m = np.isin(l, seen)
    ssl_cents = exp28.class_centroids(e[m], l[m], seen)

    tr, tr_lab = collect_embeddings(sup, train_eval_loader)
    te, te_lab = collect_embeddings(sup, test_loader)
    tra, _ = collect_embeddings(ssl, train_eval_loader)
    tea, _ = collect_embeddings(ssl, test_loader)
    tr = np.concatenate([tr, tra], axis=1)
    te = np.concatenate([te, tea], axis=1)
    anchors = torch.cat([means[seen], ssl_cents], dim=1)
    r = exp29.evaluate_space(tr, tr_lab, te, te_lab, anchors, seen, holdouts)

    def probe_stat(n_rep=3):
        aucs = []
        for s in range(n_rep):
            torch.manual_seed(1000 + s)
            a, _, _ = exp29.linear_probe_novelty(tr, tr_lab, te, te_lab,
                                                 holdouts)
            aucs.append(a)
        return float(np.mean(aucs)), float(np.std(aucs))

    pm, psd = probe_stat()
    print("\n===== round-4 summary =====")
    for k, v in REF.items():
        print(f"  {k:<30} probe={v:.4f}  (reference)")
    print(f"  {'concat [sup16;ssl16-40ep]':<30} probe={pm:.4f}+-{psd:.4f}  "
          f"acc={r['acc']:.4f}  supAUC={r['sup_auc']:.4f}  "
          f"eucl={r['eucl']:.4f}")

    outdir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "logs", "exp34")
    os.makedirs(outdir, exist_ok=True)
    np.savez(os.path.join(outdir, "probe_round4.npz"),
             probe=pm, probe_sd=psd, acc=r["acc"])
    print(f"saved {outdir}/probe_round4.npz")


if __name__ == "__main__":
    main()
