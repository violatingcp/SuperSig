"""
Tune SIGReg by the eigenspectrum diagnostic, then re-run the holdout studies.

Exp 19 showed the latent is far from the designed unit-Mahalanobis space
(within-class covariance eigenvalues ~0.001/0.02/1-5 vs the 1/1/1 ideal).
Here the enforcement knobs are tuned against that diagnostic, then the SIGReg
cases are re-run producing BOTH metric families:

    --mode tune : short inclusive trainings over a grid of
                  (sigreg_weight, n_slices, per_class) configs; prints the
                  eigenspectrum of the pooled within-class covariance.
                  Target: min/median/max ~ 1/1/1.
    --mode full : proto and CE across k = 1,2,3,10,20 with the chosen knobs
                  (--sigreg-weight/--n-slices/--per-class), reporting
                  probed binary AUC, tied/per-class Mahalanobis AUC, the
                  unit-covariance (learned-mean distance) AUC, and the
                  eigenspectrum.

CIFAR-100, 100-dim, 5-sigma seed, repulsion x1, plain images, seed 0.

Outputs (full mode):
    plots/novelty_tuned_cifar100.png
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from supersig.config import plot_path, DEVICE
from supersig.data import (
    get_cifar_loaders, build_cifar_holdout_loaders, cifar_balanced_loader,
)
from supersig.models import CIFARResNetBackbone
from supersig.losses import make_anchors
from supersig.train import (
    train_sigreg_hybrid, train_binary_probe, collect_binary_scores,
    collect_embeddings, REP_WEIGHT,
)
from supersig.metrics import mahalanobis_novelty

EMB_DIM = 100         # overridden by --emb-dim
N_CLASSES = 100
PAIR_DIST = 5.0
DATASET = "cifar100"

HOLDOUT_SETS = {
    1: [4], 2: [4, 70], 3: [4, 30, 70],
    10: [4, 14, 24, 34, 44, 54, 64, 74, 84, 94],
    20: [4 + 5 * i for i in range(20)],
}
# untuned references (exps 17/19), sigreg_weight=1, n_slices=64, per_class=24
REF = {
    "sigreg+proto": {"probed": {1: 0.9198, 2: 0.8912, 3: 0.8715, 10: 0.7940, 20: 0.6938},
                     "mahal":  {1: 0.4957, 2: 0.5315, 3: 0.5266, 10: 0.6366, 20: 0.6725}},
    "sigreg+ce":    {"probed": {1: 0.9488, 2: 0.9078, 3: 0.8828, 10: 0.8152, 20: 0.7023},
                     "mahal":  {1: 0.5828, 2: 0.5508, 3: 0.5507, 10: 0.6307, 20: 0.6778}},
}

TUNE_GRID = [
    # (label, sigreg_weight, n_slices, per_class)
    ("w1_s64_n24 (baseline)", 1.0, 64, 24),
    ("w5_s64_n24", 5.0, 64, 24),
    ("w20_s64_n24", 20.0, 64, 24),
    ("w20_s256_n24", 20.0, 256, 24),
    ("w100_s256_n24", 100.0, 256, 24),
    ("w20_s256_n48", 20.0, 256, 48),
]


def train_sigreg(disc, holdouts, ssl_ep, args, w, slices, per_class):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    backbone = CIFARResNetBackbone(EMB_DIM, arch=args.arch, pretrain=args.pretrain).to(DEVICE)
    means = make_anchors(PAIR_DIST / math.sqrt(2.0), emb_dim=EMB_DIM,
                         n_classes=N_CLASSES).clone()
    rep_w = REP_WEIGHT * 45.0 / (N_CLASSES * (N_CLASSES - 1) / 2)
    loader = cifar_balanced_loader(DATASET, holdout=holdouts or None, quick=args.quick,
                                   limit=args.limit, per_class=per_class)
    train_sigreg_hybrid(backbone, loader, ssl_ep, means, mode="repulse", disc=disc,
                        alpha=1.0, rep_weight=rep_w, sigreg_weight=w, n_slices=slices)
    return backbone, means


def eig_diag(backbone, train_loader, seen):
    tr_embs, tr_lab = collect_embeddings(backbone, train_loader)
    keep = np.isin(tr_lab, seen)
    _, _, eigs = mahalanobis_novelty(tr_embs[keep], tr_lab[keep],
                                     tr_embs[keep][:10], seen)
    return eigs


def mode_tune(args):
    train_loader, _ = get_cifar_loaders(quick=args.quick, limit=args.limit, dataset=DATASET)
    seen = list(range(N_CLASSES))
    ep = args.ssl_epochs or (1 if args.quick else 5)
    print(f"tuning grid, {ep} epochs each, disc=proto")
    rows = []
    for label, w, slices, pc in TUNE_GRID:
        print(f"\n=== tune: {label} ===")
        backbone, _ = train_sigreg("proto", set(), ep, args, w, slices, pc)
        eigs = eig_diag(backbone, train_loader, seen)
        rows.append((label, eigs))
        print(f"  eigenspectrum min/med/max: {eigs[0]:.3f} / {eigs[1]:.3f} / {eigs[2]:.3f}"
              f"   (ideal 1/1/1)")
    print(f"\n{'config':<26}{'eig min':>9}{'eig med':>9}{'eig max':>9}")
    for label, e in rows:
        print(f"{label:<26}{e[0]:>9.3f}{e[1]:>9.3f}{e[2]:>9.3f}")


def mode_full(args):
    train_loader, test_loader = get_cifar_loaders(quick=args.quick, limit=args.limit,
                                                  dataset=DATASET)
    ssl_ep = args.ssl_epochs or (2 if args.quick else 10)
    probe_ep = args.probe_epochs or (1 if args.quick else 5)
    ks = [int(x) for x in args.ks.split(",")]
    methods = ["sigreg+proto", "sigreg+ce"]
    res = {m: {} for m in methods}
    w, slices, pc = args.sigreg_weight, args.n_slices, args.per_class
    print(f"full mode: sigreg_weight={w}  n_slices={slices}  per_class={pc}")
    for k in ks:
        holdouts = set(HOLDOUT_SETS[k])
        seen = [c for c in range(N_CLASSES) if c not in holdouts]
        for m in methods:
            print(f"\n=== k={k}: {m} (tuned) ===")
            backbone, means = train_sigreg(m.split("+", 1)[1], holdouts, ssl_ep,
                                           args, w, slices, pc)
            # probed metric
            _, probe_loader, _ = build_cifar_holdout_loaders(
                quick=args.quick, holdout=holdouts, limit=args.limit, dataset=DATASET)
            head = nn.Linear(EMB_DIM, 2).to(DEVICE)
            train_binary_probe(backbone, head, probe_loader, probe_ep, positive=holdouts)
            scores, ybin = collect_binary_scores(backbone, head, test_loader,
                                                 positive=holdouts)
            probed = roc_auc_score(ybin, scores)
            # probe-free metrics
            tr_embs, tr_lab = collect_embeddings(backbone, train_loader)
            keep = np.isin(tr_lab, seen)
            te_embs, te_lab = collect_embeddings(backbone, test_loader)
            tied, percls, eigs = mahalanobis_novelty(tr_embs[keep], tr_lab[keep],
                                                     te_embs, seen)
            z = torch.as_tensor(te_embs, device=DEVICE)
            unit = torch.cdist(z, means.detach()[seen]).min(1).values.cpu().numpy()
            is_unseen = np.isin(te_lab, list(holdouts)).astype(int)
            res[m][k] = {
                "probed": probed,
                "tied": roc_auc_score(is_unseen, tied),
                "percls": roc_auc_score(is_unseen, percls),
                "unit": roc_auc_score(is_unseen, unit),
                "eigs": eigs,
            }
            r = res[m][k]
            print(f"  probed={r['probed']:.4f}  mahal tied/pc={r['tied']:.4f}/{r['percls']:.4f}"
                  f"  unit={r['unit']:.4f}  eig min/med/max={eigs[0]:.3f}/{eigs[1]:.3f}/{eigs[2]:.3f}")

    print(f"\n===== TUNED SUMMARY (w={w}, slices={slices}, per_class={pc}) =====")
    print(f"{'k':>4}{'method':>14}{'probed':>9}{'(ref)':>8}{'mahal-pc':>10}{'(ref)':>8}"
          f"{'unit':>7}{'eig med':>9}")
    for k in ks:
        for m in methods:
            r = res[m][k]
            ref_p = REF[m]["probed"][k] if EMB_DIM == 100 else float("nan")
            ref_m = REF[m]["mahal"][k] if EMB_DIM == 100 else float("nan")
            print(f"{k:>4}{m:>14}{r['probed']:>9.4f}{ref_p:>8.4f}"
                  f"{r['percls']:>10.4f}{ref_m:>8.4f}"
                  f"{r['unit']:>7.4f}{r['eigs'][1]:>9.3f}")

    plt.figure(figsize=(8, 5.5))
    for i, m in enumerate(methods):
        plt.plot(ks, [res[m][k]["probed"] for k in ks], f"C{i}-o", lw=2,
                 label=f"{m} probed (tuned)")
        plt.plot(ks, [res[m][k]["percls"] for k in ks], f"C{i}-.^", lw=1.5,
                 label=f"{m} Mahalanobis pc (tuned)")
        plt.plot(ks, [res[m][k]["unit"] for k in ks], f"C{i}:v", lw=1.2, alpha=0.8,
                 label=f"{m} unit-cov (tuned)")
        if EMB_DIM == 100:
            plt.plot(ks, [REF[m]["probed"][k] for k in ks], f"C{i}--s", lw=1, alpha=0.5,
                     label=f"{m} probed (untuned)")
    plt.xscale("log"); plt.xticks(ks, [str(k) for k in ks])
    plt.xlabel("classes held out (k)"); plt.ylabel("unseen-vs-rest AUC")
    plt.title("CIFAR-100 novelty with eigenspectrum-tuned SIGReg")
    plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(plot_path(f"novelty_tuned_cifar100_{EMB_DIM}d.png"), dpi=150); plt.close()
    print(f"\n  saved {plot_path(f'novelty_tuned_cifar100_{EMB_DIM}d.png')}")
    print("Done.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["tune", "full"], default="tune")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ssl-epochs", type=int, default=None)
    ap.add_argument("--probe-epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arch", default="resnet20")
    ap.add_argument("--pretrain", default="cifar100")
    ap.add_argument("--ks", default="1,2,3,10,20")
    ap.add_argument("--emb-dim", type=int, default=100)
    ap.add_argument("--sigreg-weight", type=float, default=20.0)
    ap.add_argument("--n-slices", type=int, default=256)
    ap.add_argument("--per-class", type=int, default=24)
    args = ap.parse_args()
    global EMB_DIM
    EMB_DIM = args.emb_dim
    if args.mode == "tune":
        mode_tune(args)
    else:
        mode_full(args)


if __name__ == "__main__":
    main()
