"""
Experiment 34 round 3: scale the SSL-SIGReg trunk feeding ssl->supres, the
only knob that gained probe AUC in round 2 (20ep 0.9123 -> 40ep 0.9213).
Round-1/2 settled the supervised half (w=1, batch 99x24, 10 epochs); here the
trunk gets 80 epochs, and separately 40 epochs at a doubled two-view batch.

Targets (same holdout 4): supcon 0.9268, supcon+simclr 0.9394.

    python experiments/34c_cifar100_probe_round3.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import copy
import importlib
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from supersig.config import DEVICE, plot_path
from supersig.data import (get_cifar_loaders, cifar_balanced_loader,
                           cifar_two_view_loader)
from supersig.models import CIFARResNetBackbone
from supersig.recipes import recipe
from supersig.train import train_sigreg_ssl, train_sigreg_hybrid, \
    collect_embeddings

exp28 = importlib.import_module("28_concat_residual")
exp29 = importlib.import_module("29_residual_finetune")

REF = {"ssl->supres [20ep] (r1)": 0.9123, "ssl->supres [40ep] (r2)": 0.9213,
       "supcon (r1)": 0.9268, "supcon+simclr (r1)": 0.9394}
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
    sup_ep = 2 if args.quick else 10
    print(f"exp34c [{ds}] round 3 (SSL trunk scaling), "
          f"holdout={sorted(holdouts)}")
    print("  refs: " + ", ".join(f"{k}={v:.4f}" for k, v in REF.items()))

    train_loader, test_loader = get_cifar_loaders(quick=args.quick, dataset=ds)
    train_eval_loader = DataLoader(train_loader.dataset, batch_size=256,
                                   shuffle=False, num_workers=2)

    def probe_stat(tr, tr_lab, te, te_lab, n_rep=3):
        aucs = []
        for s in range(n_rep):
            torch.manual_seed(1000 + s)
            a, _, _ = exp29.linear_probe_novelty(tr, tr_lab, te, te_lab,
                                                 holdouts)
            aucs.append(a)
        return float(np.mean(aucs)), float(np.std(aucs))

    def cents_of(net):
        e, l = collect_embeddings(net, train_eval_loader)
        m = np.isin(l, seen)
        return exp28.class_centroids(e[m], l[m], seen)

    TRUNKS = [("ssl-80ep", dict(epochs=80, bs=256)),
              ("ssl-40ep-b512", dict(epochs=40, bs=512))]
    results = {}
    for tname, tkw in TRUNKS:
        print(f"\n----- trunk {tname} -----")
        torch.manual_seed(args.seed + 12); np.random.seed(args.seed + 12)
        ssl = CIFARResNetBackbone(args.dim_half, arch=cfgH["arch"],
                                  pretrain=ds).to(DEVICE)
        train_sigreg_ssl(ssl, cifar_two_view_loader(batch_size=tkw["bs"],
                                                    quick=args.quick,
                                                    labeled=False,
                                                    holdout=holdouts,
                                                    dataset=ds),
                         2 if args.quick else tkw["epochs"])
        ssl_cents = cents_of(ssl)
        print(f"----- supres on {tname} (w=1, batch {CPB}x{PC}) -----")
        torch.manual_seed(args.seed + 13); np.random.seed(args.seed + 13)
        supres = copy.deepcopy(ssl)
        means = exp28.fill_means(ssl_cents, seen, cfgH).clone()
        train_sigreg_hybrid(supres,
                            cifar_balanced_loader(ds, holdout=holdouts,
                                                  quick=args.quick,
                                                  classes_per_batch=CPB,
                                                  per_class=PC),
                            sup_ep, means, mode="repulse", disc="proto",
                            alpha=1.0, rep_weight=cfgH["rep_weight"],
                            sigreg_weight=1.0, n_slices=cfgH["n_slices"])
        means = means.detach()
        tr, tr_lab = collect_embeddings(supres, train_eval_loader)
        te, te_lab = collect_embeddings(supres, test_loader)
        tra, _ = collect_embeddings(ssl, train_eval_loader)
        tea, _ = collect_embeddings(ssl, test_loader)
        tr = np.concatenate([tr, tra], axis=1)
        te = np.concatenate([te, tea], axis=1)
        anchors = torch.cat([means[seen], ssl_cents], dim=1)
        r = exp29.evaluate_space(tr, tr_lab, te, te_lab, anchors, seen,
                                 holdouts)
        pm, psd = probe_stat(tr, tr_lab, te, te_lab)
        name = f"ssl->supres [{tname}]"
        print(f"  [{name:<26}] probe={pm:.4f}+-{psd:.4f} acc={r['acc']:.4f} "
              f"supAUC={r['sup_auc']:.4f} eucl={r['eucl']:.4f}")
        results[name] = dict(probe=pm, probe_sd=psd, acc=r["acc"],
                             sup_auc=r["sup_auc"], eucl=r["eucl"])
        del ssl, supres
        torch.cuda.empty_cache()

    print("\n===== round-3 summary =====")
    for k, v in REF.items():
        print(f"  {k:<30} probe={v:.4f}  (reference)")
    for name, r in results.items():
        print(f"  {name:<30} probe={r['probe']:.4f}+-{r['probe_sd']:.4f}  "
              f"acc={r['acc']:.4f}")

    order = list(results)
    plt.figure(figsize=(8, 5.5))
    xs = np.arange(len(order))
    plt.bar(xs, [results[n]["probe"] for n in order],
            yerr=[results[n]["probe_sd"] for n in order], color="#2a78d6",
            capsize=3)
    for (label, v), c in zip(REF.items(),
                             ["#9ecae1", "#1baf7a", "#4a3aa7", "#e34948"]):
        plt.axhline(v, color=c, ls="--", lw=1.2, label=label)
    plt.xticks(xs, order, rotation=15, ha="right")
    plt.ylim(0.85, 0.97)
    plt.ylabel("holdout probe ROC AUC (pre-discovery)")
    plt.title("exp34 round 3: SSL trunk scaling, CIFAR-100")
    plt.legend(fontsize=8); plt.grid(alpha=0.25, axis="y")
    plt.tight_layout()
    plt.savefig(plot_path("exp34_probe_round3_cifar100.png"), dpi=150)
    plt.close()
    print("saved", plot_path("exp34_probe_round3_cifar100.png"))

    outdir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "logs", "exp34")
    os.makedirs(outdir, exist_ok=True)
    np.savez(os.path.join(outdir, "probe_round3.npz"),
             names=np.array(order, dtype=object),
             probes=np.array([results[n]["probe"] for n in order]),
             probe_sds=np.array([results[n]["probe_sd"] for n in order]),
             allow_pickle=True)
    print(f"saved {outdir}/probe_round3.npz")


if __name__ == "__main__":
    main()
