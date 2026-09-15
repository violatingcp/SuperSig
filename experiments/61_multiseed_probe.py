"""
Experiment 61: multi-seed nplm_bilinear vs supcon on CIFAR-100.

Exp 50 (seed 0) put label-free nplm_bilinear within 0.007 probe AUC of
SupCon on CIFAR-100; exp 52 measured single-seed spread ~0.017, leaving the
gap unresolved.  This trains both arms with N paired seeds (same init/
loader seed per pair) under the exp-50 protocol and reports per-seed Part A
metrics, per-arm mean +- sd, and the paired probe difference.

    python experiments/61_multiseed_probe.py
    python experiments/61_multiseed_probe.py --quick --seeds 2
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import importlib
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from supersig.config import DEVICE, plot_path
from supersig.data import get_cifar_loaders, cifar_two_view_loader
from supersig.models import CIFARResNetBackbone
from supersig.recipes import recipe
from supersig.train import collect_embeddings

exp28 = importlib.import_module("28_concat_residual")
exp29 = importlib.import_module("29_residual_finetune")
exp34h = importlib.import_module("34h_hybrid_nplm_cifar")
exp50 = importlib.import_module("50_nplm_cifar10_suite")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--holdout", type=int, default=4)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--arms", nargs="+", default=["nplm_bilinear", "supcon"],
                    choices=list(exp50.ARMS))
    args = ap.parse_args()
    ds = args.dataset

    cfg = recipe(ds, emb_dim=args.dim)
    n_cls = cfg["n_classes"]
    holdouts = {args.holdout}
    seen = [c for c in range(n_cls) if c not in holdouts]
    con_ep = args.epochs or (2 if args.quick else 20)
    print(f"exp61 [{ds}] multi-seed {args.arms}, seeds={args.seeds}, "
          f"dim={args.dim}, epochs={con_ep}")

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
        return float(np.mean(aucs))

    results = {a: {k: [] for k in ("probe", "acc", "eucl", "mahaT")}
               for a in args.arms}
    for s in range(args.seeds):
        for name in args.arms:
            kind, spec, labeled = exp50.ARMS[name]
            print(f"\n----- seed {s}: {name} -----")
            torch.manual_seed(100 * s + 7)   # paired: same seed per pair
            np.random.seed(100 * s + 7)
            net = CIFARResNetBackbone(args.dim, arch=cfg["arch"],
                                      pretrain=ds).to(DEVICE)
            loader = cifar_two_view_loader(quick=args.quick, labeled=labeled,
                                           holdout=holdouts, dataset=ds)
            if kind == "hybrid":
                exp34h.train_hybrid(net, loader, con_ep, spec, labeled,
                                    lam=args.lam, n_slices=cfg["n_slices"])
            else:
                spec(net, loader, con_ep)
            tr, tr_lab = collect_embeddings(net, train_eval_loader)
            te, te_lab = collect_embeddings(net, test_loader)
            m = np.isin(tr_lab, seen)
            cents = exp28.class_centroids(tr[m], tr_lab[m], seen)
            anchors = torch.as_tensor(cents, dtype=torch.float32,
                                      device=DEVICE)
            r = exp29.evaluate_space(tr, tr_lab, te, te_lab, anchors, seen,
                                     holdouts)
            pm = probe_stat(tr, tr_lab, te, te_lab)
            results[name]["probe"].append(pm)
            results[name]["acc"].append(r["acc"])
            results[name]["eucl"].append(r["eucl"])
            results[name]["mahaT"].append(r["maha_tied"])
            print(f"  [s{s} {name:<14}] probe={pm:.4f} acc={r['acc']:.4f} "
                  f"eucl={r['eucl']:.4f} mahaT={r['maha_tied']:.4f}")
            del net
            torch.cuda.empty_cache()

    print(f"\n===== EXP61 SUMMARY [{ds}, {args.seeds} seeds] =====")
    for name in args.arms:
        for k in ("probe", "acc", "eucl", "mahaT"):
            v = np.array(results[name][k])
            print(f"  {name:<16} {k:<6} {v.mean():.4f} +- {v.std(ddof=1):.4f}"
                  f"   per-seed: {np.round(v, 4)}")
    if len(args.arms) == 2:
        a, b = args.arms
        d = np.array(results[a]["probe"]) - np.array(results[b]["probe"])
        sem = d.std(ddof=1) / np.sqrt(len(d))
        print(f"\n  paired probe diff ({a} - {b}): "
              f"{d.mean():.4f} +- {d.std(ddof=1):.4f} (sem {sem:.4f}, "
              f"t={d.mean()/sem:.2f}, n={len(d)})")
        print(f"  per-seed diffs: {np.round(d, 4)}")

    plt.figure(figsize=(7, 5))
    for i, name in enumerate(args.arms):
        v = results[name]["probe"]
        plt.scatter([i] * len(v), v, s=40, alpha=0.8)
        plt.errorbar([i], [np.mean(v)], yerr=[np.std(v, ddof=1)], fmt="_",
                     color="k", ms=25, capsize=6)
    plt.xticks(range(len(args.arms)), args.arms)
    plt.ylabel("holdout probe ROC AUC")
    plt.title(f"exp61 multi-seed probe ({ds}, {args.seeds} seeds)")
    plt.grid(alpha=0.25, axis="y")
    plt.tight_layout()
    out = plot_path(f"exp61_multiseed_probe_{ds}.png")
    plt.savefig(out, dpi=150); plt.close()
    print("saved", out)

    os.makedirs(os.path.join("logs", "exp61"), exist_ok=True)
    np.savez(os.path.join("logs", "exp61", f"multiseed_{ds}.npz"),
             arms=np.array(args.arms), seeds=args.seeds,
             **{f"{k}_{n}": np.array(results[n][k]) for n in args.arms
                for k in ("probe", "acc", "eucl", "mahaT")})
    print("Done.")


if __name__ == "__main__":
    main()
