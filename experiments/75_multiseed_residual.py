"""
Experiment 75: multi-seed validation of the exp-73 CIFAR residual concats
(the exp-61 protocol applied to the new record spaces).

Per seed s in {0..n_seeds-1}, fully paired: train the supcon parent
(exp-50 arm at seed s), then the two exp-73 children (deepcopy + e2e
residual ft, seed s+7): supcon->res and supcon->res-nplm.  Report the
3-probe-seed holdout probe (and acc/eucl/mahaT) for parent and both
concats, then mean +- sd across seeds and the PAIRED concat-minus-parent
deltas -- the honest residual effect, free of parent seed luck.

    python experiments/75_multiseed_residual.py --dataset cifar10 --dim 32
    python experiments/75_multiseed_residual.py --dataset cifar100 --dim 100 --n-seeds 5
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
from supersig.data import get_cifar_loaders, cifar_two_view_loader
from supersig.recipes import recipe
from supersig.train import collect_embeddings

exp28 = importlib.import_module("28_concat_residual")
exp29 = importlib.import_module("29_residual_finetune")
exp55 = importlib.import_module("55_nplm_discovery")
exp73 = importlib.import_module("73_cifar_residual_ft")

SPACES = ["parent", "res concat", "res-nplm concat"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10",
                    choices=["cifar10", "cifar100"])
    ap.add_argument("--holdout", type=int, default=4)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=1.0)
    args = ap.parse_args()
    ds = args.dataset
    cfg = recipe(ds, emb_dim=args.dim)
    n_cls = cfg["n_classes"]
    holdouts = {args.holdout}
    seen = [c for c in range(n_cls) if c not in holdouts]
    con_ep = args.epochs or (2 if args.quick else 20)
    tag = f"{ds}_{args.dim}d"
    print(f"exp75 [{tag}] multi-seed residual concats, "
          f"{args.n_seeds} paired seeds")

    train_loader, test_loader = get_cifar_loaders(quick=args.quick,
                                                  dataset=ds)
    train_eval_loader = DataLoader(train_loader.dataset, batch_size=256,
                                   shuffle=False, num_workers=2)

    def score(tr, te, tr_lab, te_lab):
        m = np.isin(tr_lab, seen)
        cents = exp28.class_centroids(tr[m], tr_lab[m], seen)
        anch = cents.detach().float().to(DEVICE)
        r = exp29.evaluate_space(tr, tr_lab, te, te_lab, anch, seen,
                                 holdouts)
        aucs = []
        for s in range(3):
            torch.manual_seed(1000 + s)
            a, _, _ = exp29.linear_probe_novelty(tr, tr_lab, te, te_lab,
                                                 holdouts)
            aucs.append(a)
        return dict(probe=float(np.mean(aucs)), acc=r["acc"],
                    eucl=r["eucl"], mahaT=r["maha_tied"])

    R = {sp: [] for sp in SPACES}
    for s in range(args.n_seeds):
        print(f"\n########## seed {s} ##########")
        sargs = argparse.Namespace(**vars(args), seed=s)
        parent = exp55.train_arm("supcon", ds, cfg, sargs, con_ep,
                                 holdouts)
        ptr, tr_lab = collect_embeddings(parent, train_eval_loader)
        pte, te_lab = collect_embeddings(parent, test_loader)
        m = np.isin(tr_lab, seen)
        cents_full = torch.zeros(n_cls, args.dim, device=DEVICE)
        cents_full[torch.as_tensor(seen, device=DEVICE)] = \
            exp28.class_centroids(ptr[m], tr_lab[m],
                                  seen).detach().float().to(DEVICE)
        R["parent"].append(score(ptr, pte, tr_lab, te_lab))
        print(f"  [seed {s}] parent probe={R['parent'][-1]['probe']:.4f}")

        for obj in ("res", "res-nplm"):
            torch.manual_seed(s + 7); np.random.seed(s + 7)
            child = copy.deepcopy(parent)
            loader = cifar_two_view_loader(quick=args.quick, labeled=True,
                                           holdout=holdouts, dataset=ds)
            step = (exp73.make_res_step(cents_full, 5.0) if obj == "res"
                    else exp73.make_res_nplm_step(cents_full, args.lam,
                                                  cfg["n_slices"]))
            exp73.residual_ft(child, loader, con_ep, step,
                              f"s{s}-{obj}")
            rtr, _ = collect_embeddings(child, train_eval_loader)
            rte, _ = collect_embeddings(child, test_loader)
            del child
            torch.cuda.empty_cache()
            R[f"{obj} concat"].append(score(
                np.concatenate([ptr, rtr], 1),
                np.concatenate([pte, rte], 1), tr_lab, te_lab))
            print(f"  [seed {s}] {obj} concat "
                  f"probe={R[f'{obj} concat'][-1]['probe']:.4f}")
        del parent
        torch.cuda.empty_cache()

    print(f"\n===== EXP75 SUMMARY [{tag}] ({args.n_seeds} seeds) =====")
    print(f"  {'space':<18}{'probe':>18}{'acc':>16}{'eucl':>16}"
          f"{'mahaT':>16}")
    for sp in SPACES:
        row = ""
        for f in ("probe", "acc", "eucl", "mahaT"):
            v = np.array([r[f] for r in R[sp]])
            row += f"{v.mean():>9.4f}+-{v.std():.4f}"
        print(f"  {sp:<18}{row}")
    for sp in SPACES[1:]:
        d = np.array([R[sp][i]["probe"] - R["parent"][i]["probe"]
                      for i in range(args.n_seeds)])
        wins = int((d > 0).sum())
        print(f"  paired delta {sp:<16} probe: {d.mean():+.4f}+-{d.std():.4f}"
              f"  ({wins}/{args.n_seeds} seeds positive)")
    print("  per-seed probes:")
    for sp in SPACES:
        print(f"    {sp:<18}" + " ".join(f"{r['probe']:.4f}"
                                         for r in R[sp]))

    xs = np.arange(args.n_seeds)
    plt.figure(figsize=(7, 5))
    colors = {"parent": "#eda100", "res concat": "#008300",
              "res-nplm concat": "#8c2d9e"}
    for sp in SPACES:
        plt.plot(xs, [r["probe"] for r in R[sp]], "-o", label=sp,
                 color=colors[sp])
    plt.xlabel("seed")
    plt.ylabel("holdout probe ROC AUC")
    plt.title(f"exp75: multi-seed residual concats ({tag})")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    out = plot_path(f"exp75_probe_{tag}.png")
    plt.savefig(out, dpi=150); plt.close()
    print("saved", out)

    os.makedirs(os.path.join("logs", "exp75"), exist_ok=True)
    np.savez(os.path.join("logs", "exp75", f"results_{tag}.npz"),
             spaces=np.array(SPACES), n_seeds=args.n_seeds,
             **{f"{sp.replace(' ', '_')}__{f}":
                np.array([r[f] for r in R[sp]])
                for sp in SPACES for f in ("probe", "acc", "eucl",
                                           "mahaT")})
    print("Done.")


if __name__ == "__main__":
    main()
