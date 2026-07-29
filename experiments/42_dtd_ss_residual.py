"""
DTD: residual training on the ss[lam5] half (exp 34h's cls->resfeat family,
pretrained suite).

Exp 41 made SupCon+SIGReg (ss[lam5]) the single-space champion on the
frozen-ViT bases.  Here its residual couplings: train ss[lam5], freeze its
class centroids, then train a warm-started copy on the residuals
z - centroid_y with augmentations (two-view feature banks), two ways:

  ss->res-sigreg : aug-invariance (MSE between view embeddings) + classwise
                   SIGReg on the residual (train_sigreg_residual_ssl,
                   classwise=True; the exp 36 classic residual objective)
  ss->res-hybrid : NT-Xent on the normalised residuals + SIGReg (lam=5) on
                   the raw residuals (train_simclr_residual; exp 34h's
                   cls->resfeat coupling)

Both evaluated as [ss ; res] concats next to the ss half alone.  Bases:
DINO / LeJEPA repro / VISReg released weights; 100-d halves by default,
same probes as exps 40/41.

    python experiments/42_dtd_ss_residual.py
    python experiments/42_dtd_ss_residual.py --emb-dim 16 --bases visreg
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import copy
import importlib
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from supersig.config import DEVICE, plot_path
from supersig.data import BalancedBatchSampler
from supersig.train import (train_supcon_sigreg, train_simclr_residual,
                            train_sigreg_residual_ssl)

exp37 = importlib.import_module("37_dtd_vit")
exp40 = importlib.import_module("40_dtd_bases")
exp41 = importlib.import_module("41_dtd_calibrated")

# Exp 41 100-d rows (same features/protocol/seed) for the summary table.
EXP41_D100 = {
    "dino":   {"ss[lam5] [41]": 77.1, "ss[lam5]+hybrid (parallel) [41]": 78.0},
    "lejepa": {"ss[lam5] [41]": 75.8, "ss[lam5]+hybrid (parallel) [41]": 76.6},
    "visreg": {"ss[lam5] [41]": 78.3, "ss[lam5]+hybrid (parallel) [41]": 78.5},
}


def run_base(base, args):
    print(f"\n################ base: {exp40.BASE_LABELS[base]} ################")
    plain, bank = exp40.build_features(base, args)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = (plain["train"], plain["val"],
                                          plain["test"])
    Xtv, ytv = torch.cat([Xtr, Xva]), torch.cat([ytr, yva])
    two_view_lab = DataLoader(exp37.TwoViewFeatures(bank),
                              batch_size=args.batch_size, shuffle=True,
                              drop_last=True)
    two_view_bal = DataLoader(
        exp37.TwoViewFeatures(bank),
        batch_sampler=BalancedBatchSampler(bank["labels"].tolist(),
                                           n_classes=24, n_per_class=24))
    R = {}

    def probe(Z, Zt):
        return exp37.probe_accuracy(Z, ytv, Zt, yte, args.probe_epochs)

    print(f"--- [{base}] ss[lam{args.lam:g}] half ---")
    torch.manual_seed(args.seed + 30); np.random.seed(args.seed + 30)
    ss = exp37.FeatureHead(args.emb_dim).to(DEVICE)
    train_supcon_sigreg(ss, two_view_lab, args.embed_epochs, temp=0.1,
                        lam=args.lam, n_slices=64)
    Zss, Zss_t = exp37.embed(ss, Xtv), exp37.embed(ss, Xte)
    R["ss[lam5] (half), rerun"] = probe(Zss, Zss_t)
    cents = exp41.class_centroids(Zss, ytv)

    print(f"--- [{base}] ss->res-sigreg (classwise SIGReg residual) ---")
    torch.manual_seed(args.seed + 40); np.random.seed(args.seed + 40)
    res_s = copy.deepcopy(ss)
    train_sigreg_residual_ssl(res_s, two_view_bal, args.res_epochs, cents,
                              n_slices=64, classwise=True)
    R["ss->res-sigreg (concat)"] = probe(
        torch.cat([Zss, exp37.embed(res_s, Xtv)], dim=1),
        torch.cat([Zss_t, exp37.embed(res_s, Xte)], dim=1))

    print(f"--- [{base}] ss->res-hybrid (NT-Xent + SIGReg lam={args.lam:g} "
          f"residual) ---")
    torch.manual_seed(args.seed + 41); np.random.seed(args.seed + 41)
    res_h = copy.deepcopy(ss)
    train_simclr_residual(res_h, two_view_lab, args.embed_epochs, cents,
                          lam=args.lam, n_slices=64)
    R["ss->res-hybrid (concat)"] = probe(
        torch.cat([Zss, exp37.embed(res_h, Xtv)], dim=1),
        torch.cat([Zss_t, exp37.embed(res_h, Xte)], dim=1))

    for k, v in R.items():
        print(f"  [{base}] {k:<32} {100 * v:.1f}")
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bases", default="dino,lejepa,visreg")
    ap.add_argument("--embed-epochs", type=int, default=None)
    ap.add_argument("--res-epochs", type=int, default=None,
                    help="classic residual epochs (exp 36/38: half the "
                         "contrastive budget)")
    ap.add_argument("--probe-epochs", type=int, default=None)
    ap.add_argument("--aug-reps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--pair-dist", type=float, default=5.0)
    ap.add_argument("--emb-dim", type=int, default=100)
    ap.add_argument("--lam", type=float, default=5.0)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    args.embed_epochs = args.embed_epochs or (5 if args.quick else 120)
    args.res_epochs = args.res_epochs or (3 if args.quick else 60)
    args.probe_epochs = args.probe_epochs or (5 if args.quick else 50)
    args.aug_reps = args.aug_reps or (2 if args.quick else 24)
    bases = args.bases.split(",")
    print(f"device={DEVICE}  bases={bases}  emb_dim={args.emb_dim}  "
          f"lam={args.lam:g}")

    all_r = {b: run_base(b, args) for b in bases}

    print(f"\n===== DTD TOP-1, ss-residual arms ({args.emb_dim}-d halves; "
          f"[41] rows are 100-d references) =====")
    methods = list(next(iter(all_r.values())))
    print(f"  {'method':<36}" + "".join(f"{b:>10}" for b in bases))
    for m in ("ss[lam5] [41]", "ss[lam5]+hybrid (parallel) [41]"):
        print(f"  {m:<36}"
              + "".join(f"{EXP41_D100[b][m]:>10.1f}" for b in bases))
    for m in methods:
        print(f"  {m:<36}"
              + "".join(f"{100 * all_r[b][m]:>10.1f}" for b in bases))

    x = np.arange(len(methods))
    w = 0.8 / len(bases)
    plt.figure(figsize=(10, 5))
    for i, b in enumerate(bases):
        plt.bar(x + i * w, [100 * all_r[b][m] for m in methods], w,
                label=exp40.BASE_LABELS[b])
    plt.xticks(x + 0.4 - w / 2, methods, rotation=15, ha="right", fontsize=8)
    plt.ylabel("DTD test top-1 (%)")
    plt.ylim(60, 82)
    plt.title(f"DTD ss[lam5] residual couplings by base "
              f"({args.emb_dim}-d halves)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    out = plot_path(f"dtd_ss_residual_d{args.emb_dim}.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\n  saved {out}")
    print("Done.")


if __name__ == "__main__":
    main()
