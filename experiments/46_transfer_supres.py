"""
Exp 36 on Cars and Aircraft over the frozen ViT bases (sup->res A/B).

The exp 36 CIFAR-10 study compared the classic sup->res coupling (classwise
SIGReg residual) against the hybrid residual (NT-Xent + SIGReg on
z - mean_y) and crowned the hybrid.  On DTD (exps 38/42) the hybrid also
beat the classic residual but both lost to parallel concats.  Here the
same A/B runs in the hard fine-grained regime: aircraft (100 classes =
emb_dim, orthogonal-tight anchors) and cars (196 classes > emb_dim,
random-direction anchors), on the three frozen ViT-B/16 bases.

Arms per cell (100-d halves, features cached from exp 44):

  sup-proto (half)  : exp 45's supervised SIGReg recipe -- also supplies
                      the frozen learned means for the residuals
  sup->res          : + classwise-SIGReg residual half (aug invariance +
                      per-class N(0,I) on z - mean_y; balanced two-view
                      batches), warm-started from sup    [exp 36 control]
  sup->res-hybrid   : + NT-Xent + SIGReg lam=5 on the residual
                      [exp 36 champion]

Both residual spaces probed as [sup ; res] 200-d concats.

    python experiments/46_transfer_supres.py
    python experiments/46_transfer_supres.py --datasets aircraft --bases dino
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import copy
import importlib
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from supersig.config import DEVICE, plot_path
from supersig.data import BalancedBatchSampler
from supersig.train import train_sigreg_residual_ssl, train_simclr_residual

exp37 = importlib.import_module("37_dtd_vit")
exp40 = importlib.import_module("40_dtd_bases")
exp44 = importlib.import_module("44_transfer_32d")
exp45 = importlib.import_module("45_transfer_supsigreg")

# Exp 44/45 100-d rows (same features/protocol/seed) for context.
REFS = {
    "aircraft": {"raw probe [44]": {"dino": 64.9, "lejepa": 34.3,
                                    "visreg": 37.9},
                 "SupCon (aug) [44]": {"dino": 60.8, "lejepa": 48.0,
                                       "visreg": 52.4},
                 "sup-proto [45]": {"dino": 56.7, "lejepa": 40.8,
                                    "visreg": 36.8}},
    "cars": {"raw probe [44]": {"dino": 70.0, "lejepa": 40.8,
                                "visreg": 48.1},
             "SupCon (aug) [44]": {"dino": 65.9, "lejepa": 50.2,
                                   "visreg": 56.0},
             "sup-proto [45]": {"dino": 53.4, "lejepa": 38.8,
                                "visreg": 34.8}},
}


def run_cell(ds_name, base, args):
    print(f"\n######## {ds_name} on {exp40.BASE_LABELS[base]} ########")
    plain, bank = exp44.build_features(ds_name, base, args)
    (Xtv, ytv), (Xte, yte) = plain["train"], plain["test"]
    ncls = exp44.N_CLASSES[ds_name]
    two_view_lab = DataLoader(exp37.TwoViewFeatures(bank),
                              batch_size=args.batch_size, shuffle=True,
                              drop_last=True)
    two_view_bal = DataLoader(
        exp37.TwoViewFeatures(bank),
        batch_sampler=BalancedBatchSampler(bank["labels"].tolist(),
                                           n_classes=24, n_per_class=24))
    R = {}

    def probe(Z, Zt):
        import torch.nn as nn
        import torch.nn.functional as F
        head = nn.Linear(Z.size(1), ncls).to(DEVICE)
        loader = DataLoader(TensorDataset(Z, ytv), batch_size=256,
                            shuffle=True)
        opt = torch.optim.Adam(head.parameters(), lr=1e-3)
        for _ in range(args.probe_epochs):
            for z, y in loader:
                z, y = z.to(DEVICE), y.to(DEVICE)
                opt.zero_grad()
                F.cross_entropy(head(z), y).backward()
                opt.step()
        with torch.no_grad():
            pred = head(Zt.to(DEVICE)).argmax(1).cpu()
        return (pred == yte).float().mean().item()

    print(f"--- [{ds_name}/{base}] sup-proto half ---")
    torch.manual_seed(args.seed + 50); np.random.seed(args.seed + 50)
    sup, means = exp45.train_sup_sigreg(Xtv, ytv, ncls, "proto", args)
    means = means.detach()
    Zs, Zs_t = exp37.embed(sup, Xtv), exp37.embed(sup, Xte)
    R["sup-proto (half)"] = probe(Zs, Zs_t)

    print(f"--- [{ds_name}/{base}] sup->res (classwise SIGReg residual) ---")
    torch.manual_seed(args.seed + 60); np.random.seed(args.seed + 60)
    res = copy.deepcopy(sup)
    train_sigreg_residual_ssl(res, two_view_bal, args.res_epochs, means,
                              n_slices=64, classwise=True)
    R["sup->res (concat)"] = probe(
        torch.cat([Zs, exp37.embed(res, Xtv)], 1),
        torch.cat([Zs_t, exp37.embed(res, Xte)], 1))

    print(f"--- [{ds_name}/{base}] sup->res-hybrid (NT-Xent + SIGReg "
          f"lam={args.lam:g}) ---")
    torch.manual_seed(args.seed + 61); np.random.seed(args.seed + 61)
    resh = copy.deepcopy(sup)
    train_simclr_residual(resh, two_view_lab, args.embed_epochs, means,
                          lam=args.lam, n_slices=64)
    R["sup->res-hybrid (concat)"] = probe(
        torch.cat([Zs, exp37.embed(resh, Xtv)], 1),
        torch.cat([Zs_t, exp37.embed(resh, Xte)], 1))

    for k, v in R.items():
        print(f"  [{ds_name}/{base}] {k:<26} {100 * v:.1f}")
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bases", default="dino,lejepa,visreg")
    ap.add_argument("--datasets", default="aircraft,cars")
    ap.add_argument("--embed-epochs", type=int, default=None)
    ap.add_argument("--res-epochs", type=int, default=None)
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
    args.aug_reps = args.aug_reps or (2 if args.quick else 8)
    bases = args.bases.split(",")
    ds_names = args.datasets.split(",")
    print(f"device={DEVICE}  emb_dim={args.emb_dim}  datasets={ds_names}  "
          f"bases={bases}")

    all_r = {d: {b: run_cell(d, b, args) for b in bases} for d in ds_names}

    print(f"\n===== SUP->RES TRANSFER (exp 36 A/B), {args.emb_dim}-d =====")
    for d in ds_names:
        print(f"\n  --- {d} ---")
        methods = list(next(iter(all_r[d].values())))
        print(f"  {'method':<28}" + "".join(f"{b:>10}" for b in bases))
        for m, row in REFS.get(d, {}).items():
            print(f"  {m:<28}"
                  + "".join(f"{row[b]:>10.1f}" for b in bases))
        for m in methods:
            print(f"  {m:<28}"
                  + "".join(f"{100 * all_r[d][b][m]:>10.1f}" for b in bases))

    fig, axes = plt.subplots(1, len(ds_names),
                             figsize=(4.5 * len(ds_names), 4.5),
                             squeeze=False)
    for ax, d in zip(axes[0], ds_names):
        methods = list(next(iter(all_r[d].values())))
        x = np.arange(len(methods))
        w = 0.8 / len(bases)
        for i, b in enumerate(bases):
            ax.bar(x + i * w, [100 * all_r[d][b][m] for m in methods], w,
                   label=exp40.BASE_LABELS[b])
        ax.set_xticks(x + 0.4 - w / 2)
        ax.set_xticklabels(methods, rotation=25, ha="right", fontsize=7)
        ax.set_title(d, fontsize=9)
    axes[0][0].set_ylabel("test top-1 (%)")
    axes[0][0].legend(fontsize=7)
    plt.tight_layout()
    out = plot_path(f"transfer_supres_d{args.emb_dim}.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\n  saved {out}")
    print("Done.")


if __name__ == "__main__":
    main()
