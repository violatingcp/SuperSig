"""
Supervised SIGReg on the transfer suite (aircraft/cars/flowers102/galaxy10).

Exp 44 omitted the anchor-based supervised SIGReg arms (32-d < class
count).  At 100-d they become runnable, in three anchor regimes:

  galaxy10 (10 cls)  : orthogonal 5-sigma anchors, wide margin
  aircraft (100 cls) : orthogonal anchors at exactly emb_dim = n_classes
  flowers (102 cls), cars (196 cls) : emb_dim < n_classes -- deterministic
      random-direction anchors (the crowded regime of exp 14)

Arms per dataset x base (frozen ViT features, cached from exp 44):

  sup-proto        : classwise SIGReg + repulsive floating means + proto CE
                     (the settled supervised recipe; posterior acc reported)
  sup-CE           : same with a jointly-trained linear head (exp 12)
  sup-proto+hybrid : concat with a SAME-DIMENSION (100-d) unsupervised
  sup-CE+hybrid      SimCLR+SIGReg feature half, parallel (exp 22 design)

The class-count-scaled repulsion weight follows recipes.py.  The raw-
feature probe is rerun per cell as the in-run anchor.

    python experiments/45_transfer_supsigreg.py
    python experiments/45_transfer_supsigreg.py --datasets galaxy10 --bases dino
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import importlib
import math
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from supersig.config import DEVICE, plot_path
from supersig.data import BalancedBatchSampler
from supersig.losses import make_anchors, mean_geometry
from supersig.train import train_sigreg_hybrid, train_simclr_sigreg, REP_WEIGHT

exp37 = importlib.import_module("37_dtd_vit")
exp38 = importlib.import_module("38_dtd_arc")
exp40 = importlib.import_module("40_dtd_bases")
exp44 = importlib.import_module("44_transfer_32d")


def train_sup_sigreg(X, y, n_classes, disc, args):
    """Dataset-general version of exp37.train_sigreg (n_classes-aware)."""
    head = exp37.FeatureHead(args.emb_dim).to(DEVICE)
    means = make_anchors(args.pair_dist / math.sqrt(2.0),
                         emb_dim=args.emb_dim, n_classes=n_classes).clone()
    d0, _ = mean_geometry(means)
    kind = "orthogonal" if args.emb_dim >= n_classes else "random-direction"
    print(f"  seed anchors ({kind}): min pairwise distance={d0:.2f} sigma")
    rep_w = REP_WEIGHT * 45.0 / (n_classes * (n_classes - 1) / 2)
    sampler = BalancedBatchSampler(y.tolist(), n_classes=24, n_per_class=24)
    loader = DataLoader(TensorDataset(X, y), batch_sampler=sampler)
    print(f"  balanced loader: {X.size(0)} samples, {len(sampler)} "
          f"batches of {sampler.n_classes}x24")
    train_sigreg_hybrid(head, loader, args.embed_epochs, means,
                        mode="repulse", disc=disc, alpha=1.0,
                        rep_weight=rep_w, sigreg_weight=1.0, n_slices=64)
    return head, means


def run_cell(ds_name, base, args):
    print(f"\n######## {ds_name} on {exp40.BASE_LABELS[base]} ########")
    plain, bank = exp44.build_features(ds_name, base, args)
    (Xtv, ytv), (Xte, yte) = plain["train"], plain["test"]
    ncls = exp44.N_CLASSES[ds_name]
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

    R["raw probe (768d)"] = probe(Xtv, Xte)

    torch.manual_seed(args.seed + 21); np.random.seed(args.seed + 21)
    hyb = exp37.FeatureHead(args.emb_dim).to(DEVICE)
    train_simclr_sigreg(hyb, DataLoader(exp38.TwoViewUnlabeled(bank),
                                        batch_size=args.batch_size,
                                        shuffle=True, drop_last=True),
                        args.embed_epochs, lam=args.lam, n_slices=64)
    Zh, Zh_t = exp37.embed(hyb, Xtv), exp37.embed(hyb, Xte)

    for disc in ("proto", "ce"):
        print(f"--- [{ds_name}/{base}] sup-SIGReg ({disc}) ---")
        torch.manual_seed(args.seed + 50); np.random.seed(args.seed + 50)
        head, means = train_sup_sigreg(Xtv, ytv, ncls, disc, args)
        Z, Zt = exp37.embed(head, Xtv), exp37.embed(head, Xte)
        R[f"sup-{disc}"] = probe(Z, Zt)
        if disc == "proto":
            with torch.no_grad():
                pred = torch.cdist(Zt.to(DEVICE), means).argmin(1).cpu()
            R["sup-proto (posterior)"] = (pred == yte).float().mean().item()
        R[f"sup-{disc}+hybrid (concat)"] = probe(
            torch.cat([Z, Zh], 1), torch.cat([Zt, Zh_t], 1))

    for k, v in R.items():
        print(f"  [{ds_name}/{base}] {k:<28} {100 * v:.1f}")
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bases", default="dino,lejepa,visreg")
    ap.add_argument("--datasets", default="aircraft,cars,flowers,galaxy10")
    ap.add_argument("--embed-epochs", type=int, default=None)
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
    args.probe_epochs = args.probe_epochs or (5 if args.quick else 50)
    args.aug_reps = args.aug_reps or (2 if args.quick else 8)
    bases = args.bases.split(",")
    ds_names = args.datasets.split(",")
    print(f"device={DEVICE}  emb_dim={args.emb_dim}  datasets={ds_names}  "
          f"bases={bases}")

    all_r = {d: {b: run_cell(d, b, args) for b in bases} for d in ds_names}

    print(f"\n===== SUPERVISED SIGREG TRANSFER, {args.emb_dim}-d =====")
    for d in ds_names:
        print(f"\n  --- {d} (paper: "
              + "  ".join(f"{k} {v}" for k, v in exp44.PAPER[d].items())
              + ") ---")
        methods = list(next(iter(all_r[d].values())))
        print(f"  {'method':<30}" + "".join(f"{b:>10}" for b in bases))
        for m in methods:
            print(f"  {m:<30}"
                  + "".join(f"{100 * all_r[d][b][m]:>10.1f}" for b in bases))

    fig, axes = plt.subplots(1, len(ds_names),
                             figsize=(4.2 * len(ds_names), 4.5),
                             squeeze=False)
    for ax, d in zip(axes[0], ds_names):
        methods = list(next(iter(all_r[d].values())))
        x = np.arange(len(methods))
        w = 0.8 / len(bases)
        for i, b in enumerate(bases):
            ax.bar(x + i * w, [100 * all_r[d][b][m] for m in methods], w,
                   label=exp40.BASE_LABELS[b])
        ax.set_xticks(x + 0.4 - w / 2)
        ax.set_xticklabels(methods, rotation=35, ha="right", fontsize=6)
        ax.set_title(d, fontsize=9)
    axes[0][0].set_ylabel("test top-1 (%)")
    axes[0][0].legend(fontsize=6)
    plt.tight_layout()
    out = plot_path(f"transfer_supsigreg_d{args.emb_dim}.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\n  saved {out}")
    print("Done.")


if __name__ == "__main__":
    main()
