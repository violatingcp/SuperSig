"""
Does the residual (or any featurization) add anything to a SupCon head?
Aircraft + Cars, frozen ViT bases, 100-d halves.

Exp 46 showed residual coupling can't rescue a weak supervised-SIGReg
half on fine-grained data.  This experiment asks the sharper question on
the STRONG supervised half (SupCon, the best head family there):

  supcon (half)          : the baseline head
  simclr (half)          : unsupervised featurization alone -- does it
                           carry any fine-grained signal by itself?
  supcon+res-simclr      : SimCLR (NT-Xent, lam=0 -- plain, not hybrid)
                           trained on the residual z - centroid_y against
                           the SupCon half's frozen centroids, warm-started
                           from it (exp 34e coupling); [supcon ; res] concat
  supcon+simclr          : the parallel concat, no residual

If supcon+simclr ~= supcon alone, featurization adds nothing here; if
supcon+res-simclr ~= supcon+simclr, the residualization specifically adds
nothing over plain featurization.

    python experiments/47_transfer_supcon_res.py
    python experiments/47_transfer_supcon_res.py --datasets cars --bases dino
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
from supersig.train import train_simclr, train_simclr_residual

exp37 = importlib.import_module("37_dtd_vit")
exp38 = importlib.import_module("38_dtd_arc")
exp40 = importlib.import_module("40_dtd_bases")
exp41 = importlib.import_module("41_dtd_calibrated")
exp44 = importlib.import_module("44_transfer_32d")

REFS = {
    "aircraft": {"raw probe [44]": {"dino": 64.9, "lejepa": 34.3,
                                    "visreg": 37.9}},
    "cars": {"raw probe [44]": {"dino": 70.0, "lejepa": 40.8,
                                "visreg": 48.1}},
}


def run_cell(ds_name, base, args):
    print(f"\n######## {ds_name} on {exp40.BASE_LABELS[base]} ########")
    plain, bank = exp44.build_features(ds_name, base, args)
    (Xtv, ytv), (Xte, yte) = plain["train"], plain["test"]
    ncls = exp44.N_CLASSES[ds_name]
    two_view_lab = DataLoader(exp37.TwoViewFeatures(bank),
                              batch_size=args.batch_size, shuffle=True,
                              drop_last=True)
    two_view_unlab = DataLoader(exp38.TwoViewUnlabeled(bank),
                                batch_size=args.batch_size, shuffle=True,
                                drop_last=True)
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

    print(f"--- [{ds_name}/{base}] supcon half ---")
    torch.manual_seed(args.seed + 20); np.random.seed(args.seed + 20)
    supcon = exp37.train_supcon_aug(bank, args)
    Zsc, Zsc_t = exp37.embed(supcon, Xtv), exp37.embed(supcon, Xte)
    R["supcon (half)"] = probe(Zsc, Zsc_t)

    print(f"--- [{ds_name}/{base}] simclr half ---")
    torch.manual_seed(args.seed + 22); np.random.seed(args.seed + 22)
    sim = exp37.FeatureHead(args.emb_dim).to(DEVICE)
    train_simclr(sim, two_view_unlab, args.embed_epochs)
    Zsim, Zsim_t = exp37.embed(sim, Xtv), exp37.embed(sim, Xte)
    R["simclr (half)"] = probe(Zsim, Zsim_t)

    print(f"--- [{ds_name}/{base}] supcon+res-simclr (residual, lam=0) ---")
    cents = exp41.class_centroids(Zsc, ytv, n_classes=ncls)
    torch.manual_seed(args.seed + 23); np.random.seed(args.seed + 23)
    res = copy.deepcopy(supcon)
    train_simclr_residual(res, two_view_lab, args.embed_epochs, cents,
                          lam=0.0, n_slices=64)
    R["supcon+res-simclr (concat)"] = probe(
        torch.cat([Zsc, exp37.embed(res, Xtv)], 1),
        torch.cat([Zsc_t, exp37.embed(res, Xte)], 1))

    R["supcon+simclr (concat)"] = probe(torch.cat([Zsc, Zsim], 1),
                                        torch.cat([Zsc_t, Zsim_t], 1))

    for k, v in R.items():
        print(f"  [{ds_name}/{base}] {k:<28} {100 * v:.1f}")
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bases", default="dino,lejepa,visreg")
    ap.add_argument("--datasets", default="aircraft,cars")
    ap.add_argument("--embed-epochs", type=int, default=None)
    ap.add_argument("--probe-epochs", type=int, default=None)
    ap.add_argument("--aug-reps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--pair-dist", type=float, default=5.0)
    ap.add_argument("--emb-dim", type=int, default=100)
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

    print(f"\n===== SUPCON RESIDUAL/FEATURIZATION A/B, {args.emb_dim}-d =====")
    for d in ds_names:
        print(f"\n  --- {d} ---")
        methods = list(next(iter(all_r[d].values())))
        print(f"  {'method':<30}" + "".join(f"{b:>10}" for b in bases))
        for m, row in REFS.get(d, {}).items():
            print(f"  {m:<30}"
                  + "".join(f"{row[b]:>10.1f}" for b in bases))
        for m in methods:
            print(f"  {m:<30}"
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
    out = plot_path(f"transfer_supcon_res_d{args.emb_dim}.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\n  saved {out}")
    print("Done.")


if __name__ == "__main__":
    main()
