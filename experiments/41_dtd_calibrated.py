"""
DTD calibrated-contrastive arms (exp 34g/34h family) on the pretrained suite.

Extends exp 40's three frozen ViT-B/16 bases (DINO / LeJEPA community repro /
VISReg's released weights) with the calibrated arms the user asked after:

  ss[lam5]            : SupCon + SIGReg-to-N(0,I) on the raw embeddings
                        (train_supcon_sigreg, exp 34g) -- "SupCon with the
                        Gaussian loss", single 100-d head
  ss[lam5]+hybrid     : concat with the parallel SimCLR+SIGReg feature half
                        (exp 34g's fully calibrated space, 200-d)
  supcon->res-hybrid  : the RESIDUAL counterpart of supcon+hybrid -- the
                        hybrid objective (NT-Xent + SIGReg lam=5, two-view
                        augmented) trained on z - centroid_y against the
                        SupCon half's frozen class centroids, warm-started
                        from the SupCon net (exp 34h cls->resfeat / exp 36
                        pattern), then [supcon ; res-hybrid] (200-d)

SupCon (aug) and the parallel supcon+hybrid are retrained here as in-run
references (exp 40 did not checkpoint heads).  Same protocol throughout:
cached features, 100-d halves, train+val supervision, Adam probes, DTD
test top-1.

    python experiments/41_dtd_calibrated.py
    python experiments/41_dtd_calibrated.py --bases visreg --quick
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
from supersig.train import (train_simclr_sigreg, train_supcon_sigreg,
                            train_simclr_residual)

exp37 = importlib.import_module("37_dtd_vit")
exp38 = importlib.import_module("38_dtd_arc")
exp40 = importlib.import_module("40_dtd_bases")

# Exp 40 rows (same features/protocol, seed 0) for the summary table.
EXP40_ROWS = {
    "dino":   {"raw probe (train+val)": 76.5, "SupCon (aug)": 75.5,
               "supcon+hybrid (parallel)": 77.9},
    "lejepa": {"raw probe (train+val)": 76.2, "SupCon (aug)": 77.0,
               "supcon+hybrid (parallel)": 76.0},
    "visreg": {"raw probe (train+val)": 77.9, "SupCon (aug)": 76.5,
               "supcon+hybrid (parallel)": 78.1},
}


def class_centroids(Z, y, n_classes=exp37.N_CLASSES):
    return torch.stack([Z[y == c].mean(0) for c in range(n_classes)]).to(DEVICE)


def run_base(base, args):
    print(f"\n################ base: {exp40.BASE_LABELS[base]} ################")
    plain, bank = exp40.build_features(base, args)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = (plain["train"], plain["val"],
                                          plain["test"])
    Xtv, ytv = torch.cat([Xtr, Xva]), torch.cat([ytr, yva])
    two_view_lab = DataLoader(exp37.TwoViewFeatures(bank),
                              batch_size=args.batch_size, shuffle=True,
                              drop_last=True)
    two_view_unlab = DataLoader(exp38.TwoViewUnlabeled(bank),
                                batch_size=args.batch_size, shuffle=True,
                                drop_last=True)
    R = {}

    def probe(Z, Zt):
        return exp37.probe_accuracy(Z, ytv, Zt, yte, args.probe_epochs)

    print(f"--- [{base}] ss[lam{args.lam:g}]: SupCon + SIGReg on raw z ---")
    torch.manual_seed(args.seed + 30); np.random.seed(args.seed + 30)
    ss = exp37.FeatureHead(args.emb_dim).to(DEVICE)
    train_supcon_sigreg(ss, two_view_lab, args.embed_epochs, temp=0.1,
                        lam=args.lam, n_slices=64)
    Zss, Zss_t = exp37.embed(ss, Xtv), exp37.embed(ss, Xte)
    R["SupCon+SIGReg (ss[lam5])"] = probe(Zss, Zss_t)

    print(f"--- [{base}] hybrid half (NT-Xent + SIGReg lam={args.lam:g}) ---")
    torch.manual_seed(args.seed + 21); np.random.seed(args.seed + 21)
    hyb = exp37.FeatureHead(args.emb_dim).to(DEVICE)
    train_simclr_sigreg(hyb, two_view_unlab, args.embed_epochs, lam=args.lam,
                        n_slices=64)
    Zh, Zh_t = exp37.embed(hyb, Xtv), exp37.embed(hyb, Xte)
    R["ss[lam5]+hybrid (concat)"] = probe(torch.cat([Zss, Zh], dim=1),
                                          torch.cat([Zss_t, Zh_t], dim=1))

    print(f"--- [{base}] SupCon (aug) reference ---")
    torch.manual_seed(args.seed + 20); np.random.seed(args.seed + 20)
    supcon = exp37.train_supcon_aug(bank, args)
    Zsc, Zsc_t = exp37.embed(supcon, Xtv), exp37.embed(supcon, Xte)
    R["SupCon (aug), rerun"] = probe(Zsc, Zsc_t)
    R["supcon+hybrid (parallel), rerun"] = probe(
        torch.cat([Zsc, Zh], dim=1), torch.cat([Zsc_t, Zh_t], dim=1))

    print(f"--- [{base}] supcon->res-hybrid (residual, augmented) ---")
    cents = class_centroids(Zsc, ytv)
    torch.manual_seed(args.seed + 31); np.random.seed(args.seed + 31)
    resh = copy.deepcopy(supcon)
    train_simclr_residual(resh, two_view_lab, args.embed_epochs, cents,
                          lam=args.lam, n_slices=64)
    Zr, Zr_t = exp37.embed(resh, Xtv), exp37.embed(resh, Xte)
    R["supcon->res-hybrid (concat)"] = probe(torch.cat([Zsc, Zr], dim=1),
                                             torch.cat([Zsc_t, Zr_t], dim=1))

    for k, v in R.items():
        print(f"  [{base}] {k:<34} {100 * v:.1f}")
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bases", default="dino,lejepa,visreg")
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
    args.aug_reps = args.aug_reps or (2 if args.quick else 24)
    bases = args.bases.split(",")
    print(f"device={DEVICE}  bases={bases}  emb_dim={args.emb_dim}  "
          f"lam={args.lam:g}")

    all_r = {b: run_base(b, args) for b in bases}

    print(f"\n===== DTD TOP-1, calibrated arms ({args.emb_dim}-d heads; "
          f"[40] rows are 100-d references) =====")
    methods = list(next(iter(all_r.values())))
    print(f"  {'method':<36}" + "".join(f"{b:>10}" for b in bases))
    for m in ("raw probe (train+val)", "SupCon (aug)",
              "supcon+hybrid (parallel)"):
        print(f"  {m + ' [40]':<36}"
              + "".join(f"{EXP40_ROWS[b][m]:>10.1f}" for b in bases))
    for m in methods:
        print(f"  {m:<36}"
              + "".join(f"{100 * all_r[b][m]:>10.1f}" for b in bases))

    x = np.arange(len(methods))
    w = 0.8 / len(bases)
    plt.figure(figsize=(11, 5))
    for i, b in enumerate(bases):
        plt.bar(x + i * w, [100 * all_r[b][m] for m in methods], w,
                label=exp40.BASE_LABELS[b])
    plt.xticks(x + 0.4 - w / 2, methods, rotation=20, ha="right", fontsize=8)
    plt.ylabel("DTD test top-1 (%)")
    plt.ylim(60, 82)
    plt.title("DTD calibrated-contrastive arms by base (frozen ViT-B/16, "
              "100-d heads)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    out = plot_path(f"dtd_calibrated_accuracy_d{args.emb_dim}.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\n  saved {out}")
    print("Done.")


if __name__ == "__main__":
    main()
