"""
DTD transfer of the exp 34-36 leaderboard arms (follow-up to 37_dtd_vit.py).

Exp 37 benchmarked the exp 09-13 CIFAR-10 leaders on DTD over frozen DINO
ViT-B/16 features against the VISReg paper (arXiv 2606.02572).  Here the
TOP arms of the exp 34-36 program (logs/SUMMARY_TABLES.md) run on the same
base, same cached features, same probe protocol:

  sup->res-hybrid : [sup ; NT-Xent+SIGReg(lam=5) on the residual z - mean_y]
                    -- exp 36 CIFAR-10 champion (Pareto-dominant)
  sup->res        : [sup ; classwise-SIGReg residual]  -- exp 33 classic
  supcon+hybrid   : [supcon ; NT-Xent+SIGReg(lam=5) trunk]
                    -- exp 34e, CIFAR-100 probe champion (0.9423)
  supcon+simclr   : [supcon ; plain SimCLR trunk]  -- exp 33/34i probe leader

Each half is a 64-dim FeatureHead on the frozen 768-dim CLS features
(64 >= 47 classes, so the sup half keeps orthogonal 5-sigma anchors); the
concat spaces are 128-dim.  DTD sits in the crowded-many-class regime, so
the program's matching rule predicts the contrastive+calibration family
(supcon+hybrid) should lead, as on CIFAR-100.

Halves and concats are all probed (train+val supervision, Adam probe as in
exp 37) -> DTD test top-1.  Note the concat probes see 128 dims vs 64 for
the halves and 768 for the raw-feature baseline.

Outputs (plots/): dtd_vitb16_arc_accuracy.png
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import copy
import importlib
import math
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from supersig.config import DEVICE, plot_path
from supersig.data import BalancedBatchSampler
from supersig.train import (
    train_sigreg_residual_ssl, train_simclr_residual, train_simclr,
    train_simclr_sigreg,
)

exp37 = importlib.import_module("37_dtd_vit")

# Exp 37 results on the identical features/protocol (seed 0), for the table.
EXP37_ROWS = [
    ("Linear probe, raw DINO (train+val) [37]", 76.4),
    ("SupCon (aug) [37, 64d]", 76.1),
    ("SIGReg repulse + CE [37, 64d]", 74.3),
    ("SupCon (no aug) [37, 64d]", 74.1),
    ("SIGReg repulse + proto [37, 64d]", 70.9),
]


class TwoViewUnlabeled(Dataset):
    """Two-view feature bank without labels (SimCLR / hybrid trunks)."""

    def __init__(self, bank):
        self.tv = exp37.TwoViewFeatures(bank)

    def __len__(self):
        return len(self.tv)

    def __getitem__(self, idx):
        f1, f2, _ = self.tv[idx]
        return f1, f2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--embed-epochs", type=int, default=None)
    ap.add_argument("--res-epochs", type=int, default=None,
                    help="classic classwise-SIGReg residual epochs "
                         "(exp 36 used half the contrastive budget)")
    ap.add_argument("--probe-epochs", type=int, default=None)
    ap.add_argument("--aug-reps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--pair-dist", type=float, default=5.0)
    ap.add_argument("--res-lam", type=float, default=5.0)
    ap.add_argument("--emb-dim", type=int, default=exp37.EMB_DIM)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    args.embed_epochs = args.embed_epochs or (5 if args.quick else 120)
    args.res_epochs = args.res_epochs or (3 if args.quick else 60)
    args.probe_epochs = args.probe_epochs or (5 if args.quick else 50)
    args.aug_reps = args.aug_reps or (2 if args.quick else 24)
    print(f"device={DEVICE}  base=DINO ViT-B/16 (frozen)  "
          f"halves={args.emb_dim}d  res_lam={args.res_lam}  "
          f"embed_epochs={args.embed_epochs}")

    print("\n--- features ---")
    plain = exp37.build_plain_features(args)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = (plain["train"], plain["val"],
                                          plain["test"])
    Xtv, ytv = torch.cat([Xtr, Xva]), torch.cat([ytr, yva])
    bank = exp37.build_aug_bank(args)
    two_view = DataLoader(exp37.TwoViewFeatures(bank),
                          batch_size=args.batch_size, shuffle=True,
                          drop_last=True)
    two_view_unlab = DataLoader(TwoViewUnlabeled(bank),
                                batch_size=args.batch_size, shuffle=True,
                                drop_last=True)
    tv_ds = exp37.TwoViewFeatures(bank)
    two_view_bal = DataLoader(
        tv_ds, batch_sampler=BalancedBatchSampler(bank["labels"].tolist(),
                                                  n_classes=24, n_per_class=24))

    def probe(Z, Zt):
        return exp37.probe_accuracy(Z, ytv, Zt, yte, args.probe_epochs)

    def concat(a, b):
        return torch.cat([a, b], dim=1)

    results = {}

    # ----- sup half + residual halves (exp 36 construction) ----------------
    print("\n----- sup64 (exp 37 SIGReg repulse + proto) -----")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    sup, means_sup = exp37.train_sigreg(Xtv, ytv, "proto", args)
    means_sup = means_sup.detach()
    Zsup, Zsup_t = exp37.embed(sup, Xtv), exp37.embed(sup, Xte)
    results["sup (half)"] = probe(Zsup, Zsup_t)
    print(f"  sup half probe TEST={results['sup (half)']:.4f}")

    print("\n----- res64 classic (classwise SIGReg residual) -----")
    torch.manual_seed(args.seed + 11); np.random.seed(args.seed + 11)
    res = copy.deepcopy(sup)
    train_sigreg_residual_ssl(res, two_view_bal, args.res_epochs, means_sup,
                              n_slices=64, classwise=True)
    Zres, Zres_t = exp37.embed(res, Xtv), exp37.embed(res, Xte)
    results["sup->res (concat)"] = probe(concat(Zsup, Zres),
                                         concat(Zsup_t, Zres_t))
    print(f"  sup->res concat probe TEST={results['sup->res (concat)']:.4f}")

    print(f"\n----- res-hybrid64 (NT-Xent + SIGReg lam={args.res_lam} "
          f"on sup residual) -----")
    torch.manual_seed(args.seed + 11); np.random.seed(args.seed + 11)
    resh = copy.deepcopy(sup)
    train_simclr_residual(resh, two_view, args.embed_epochs, means_sup,
                          lam=args.res_lam, n_slices=64)
    Zresh, Zresh_t = exp37.embed(resh, Xtv), exp37.embed(resh, Xte)
    results["sup->res-hybrid (concat)"] = probe(concat(Zsup, Zresh),
                                                concat(Zsup_t, Zresh_t))
    print(f"  sup->res-hybrid concat probe "
          f"TEST={results['sup->res-hybrid (concat)']:.4f}")

    # ----- contrastive family (exp 34e construction) -----------------------
    print("\n----- supcon64 (two-view aug) -----")
    torch.manual_seed(args.seed + 20); np.random.seed(args.seed + 20)
    supcon = exp37.train_supcon_aug(bank, args)
    Zsc, Zsc_t = exp37.embed(supcon, Xtv), exp37.embed(supcon, Xte)
    results["supcon (half)"] = probe(Zsc, Zsc_t)
    print(f"  supcon half probe TEST={results['supcon (half)']:.4f}")

    print(f"\n----- hybrid64 (SimCLR NT-Xent + SIGReg lam={args.res_lam}) -----")
    torch.manual_seed(args.seed + 21); np.random.seed(args.seed + 21)
    hyb = exp37.FeatureHead(args.emb_dim).to(DEVICE)
    train_simclr_sigreg(hyb, two_view_unlab, args.embed_epochs,
                        lam=args.res_lam, n_slices=64)
    Zh, Zh_t = exp37.embed(hyb, Xtv), exp37.embed(hyb, Xte)
    results["supcon+hybrid (concat)"] = probe(concat(Zsc, Zh),
                                              concat(Zsc_t, Zh_t))
    print(f"  supcon+hybrid concat probe "
          f"TEST={results['supcon+hybrid (concat)']:.4f}")

    print("\n----- simclr64 (plain NT-Xent) -----")
    torch.manual_seed(args.seed + 22); np.random.seed(args.seed + 22)
    sim = exp37.FeatureHead(args.emb_dim).to(DEVICE)
    train_simclr(sim, two_view_unlab, args.embed_epochs)
    Zsim, Zsim_t = exp37.embed(sim, Xtv), exp37.embed(sim, Xte)
    results["supcon+simclr (concat)"] = probe(concat(Zsc, Zsim),
                                              concat(Zsc_t, Zsim_t))
    print(f"  supcon+simclr concat probe "
          f"TEST={results['supcon+simclr (concat)']:.4f}")

    # ----- summary ----------------------------------------------------------
    print("\n===== DTD TOP-1 SUMMARY, exp 34-36 arms "
          "(base: frozen DINO ViT-B/16) =====")
    for name, acc in exp37.PAPER_ROWS:
        print(f"  {name:<44} {acc:.1f}")
    for name, acc in EXP37_ROWS:
        print(f"  {name:<44} {acc:.1f}")
    for name, acc in results.items():
        print(f"  {name:<44} {100 * acc:.1f}")

    names = ([n for n, _ in exp37.PAPER_ROWS] + [n for n, _ in EXP37_ROWS]
             + list(results))
    accs = ([a for _, a in exp37.PAPER_ROWS] + [a for _, a in EXP37_ROWS]
            + [100 * a for a in results.values()])
    colors = (["#999999"] * len(exp37.PAPER_ROWS)
              + ["#7fb3d9"] * len(EXP37_ROWS)
              + ["#d62728"] * len(results))
    plt.figure(figsize=(9, 6.5))
    ypos = np.arange(len(names))
    plt.barh(ypos, accs, color=colors)
    plt.yticks(ypos, names, fontsize=8)
    plt.xlabel("DTD test top-1 (%)")
    plt.xlim(min(accs) - 3, max(accs) + 1)
    plt.title("DTD: paper vs exp-37 heads vs exp-34-36 arc "
              "on frozen DINO ViT-B/16")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    tag = "" if args.emb_dim == exp37.EMB_DIM else f"_d{args.emb_dim}"
    plt.savefig(plot_path(f"dtd_vitb16_arc_accuracy{tag}.png"), dpi=150)
    plt.close()
    print(f"\n  saved {plot_path(f'dtd_vitb16_arc_accuracy{tag}.png')}")
    print("Done.")


if __name__ == "__main__":
    main()
