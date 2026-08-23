"""
Experiment 109 (IMPROVEMENT_TESTS.md #109): density-ratio pooling on
CIFAR-100 -- the remaining lever on the one fully-blocked dataset.

Exp 89: C100 discovery is geometry-blocked (purity <= 0.121 at every
holdout size x quantile; the distance tail is owned by background
outliers).  Exp 92/92b: the SparKer density-ratio pool is immune to
exactly that failure mode, and in a frozen space it costs nothing.  Never
combined on C100.

Recipe: exp-92b sparker-frozen (freeze the whole exp-89 concat space,
anchors only) on the exp-89 cached spaces at holdout sizes {1, 5, 10},
tau_quantile {0.95, 0.99} -- six discovery runs, no retraining.  Compare
r1/r2 purity against the exp-89 distance grid at matched sizes.

Prediction: round-1 purity clears 0.15 for the first time on C100, with
round 2 holding.  Falsifier: purity stays below 0.15 -> C100 novelty is
invisible to the density ratio as well as to distance; tail-pooling of any
kind is out, a clean citable stopping point for the C100 thread.

    python experiments/109_c100_density_pool.py
    python experiments/109_c100_density_pool.py --sizes 1 --quantiles 0.95 --quick
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import copy
import importlib
import numpy as np
import torch
from torch.utils.data import DataLoader

from supersig.data import get_cifar_loaders
from supersig.recipes import recipe
from supersig.train import collect_embeddings

exp28 = importlib.import_module("28_concat_residual")
exp29 = importlib.import_module("29_residual_finetune")
exp89 = importlib.import_module("89_c100_rate_grid")
exp92 = importlib.import_module("92_sparker_discovery")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", type=int, default=[1, 5, 10])
    ap.add_argument("--quantiles", nargs="+", type=float,
                    default=[0.95, 0.99])
    ap.add_argument("--dim", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kernels", type=int, default=16)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    ds = "cifar100"
    cfg = recipe(ds, emb_dim=args.dim)
    n_cls = cfg["n_classes"]
    con_ep = args.epochs or (2 if args.quick else 20)
    ft_ep = 1 if args.quick else cfg["ft_epochs"]
    steps = 50 if args.quick else args.steps

    train_loader, test_loader = get_cifar_loaders(quick=args.quick,
                                                  dataset=ds)
    tel = DataLoader(train_loader.dataset, batch_size=256, shuffle=False,
                     num_workers=2)

    # exp-89 distance-grid archive for the comparison column
    arch = {}
    try:
        d = np.load("logs/exp89/results.npz", allow_pickle=True)
        arch = eval(d["summary"][0], {"nan": float("nan")})
    except Exception as e:
        print(f"  (exp-89 archive unavailable: {e})")

    results = {}
    for size in args.sizes:
        holdouts = {4} if size == 1 else set(range(n_cls - size, n_cls))
        seen = [c for c in range(n_cls) if c not in holdouts]
        print(f"\n######## holdout size {size}: sparker-frozen on the "
              f"exp-89 space ########", flush=True)
        bb0 = exp89.build_space(holdouts, cfg, args, con_ep)
        tr0, tr_lab = collect_embeddings(bb0, tel)
        te0, te_lab = collect_embeddings(bb0, test_loader)
        torch.manual_seed(1000)
        a_pre, _, _ = exp29.linear_probe_novelty(tr0, tr_lab, te0, te_lab,
                                                 holdouts)
        m = np.isin(tr_lab, seen)
        means0 = exp28.fill_means(
            exp28.class_centroids(tr0[m], tr_lab[m], seen), seen,
            dict(pair_dist=cfg["pair_dist"], n_classes=n_cls)).detach()
        print(f"  pre probe={a_pre:.4f}", flush=True)

        for q in args.quantiles:
            bb = copy.deepcopy(bb0)
            for p in bb.parameters():
                p.requires_grad_(False)
            _, hist = exp92.sparker_discovery(
                bb, means0.clone(), base_ds=train_loader.dataset,
                train_eval_loader=tel, test_loader=test_loader, seen=seen,
                holdouts=holdouts, rep_weight=cfg["rep_weight"],
                sigreg_weight=cfg["sigreg_weight"],
                n_slices=cfg["n_slices"], rounds=2, ft_epochs=ft_ep,
                tau_quantile=q, M=args.kernels, steps=steps,
                seed=args.seed)
            a89 = arch.get(f"h{size}:q{q}", {})
            results[f"h{size}:q{q}"] = dict(
                probe_pre=float(a_pre),
                pool=[h["pool"] for h in hist],
                purity=[h["purity"] for h in hist],
                margin=[h.get("margin", float("nan")) for h in hist],
                dist_purity=a89.get("purity", []))
            pur = [h["purity"] for h in hist]
            dref = a89.get("purity", [float("nan")])
            print(f"  [h{size} q={q}] sparker-frozen purity " +
                  " ".join(f"r{h['round']}={h['purity']:.3f}"
                           f"(n={h['pool']})" for h in hist) +
                  f"   exp-89 distance r1={dref[0]:.3f}", flush=True)
            del bb
            torch.cuda.empty_cache()
        del bb0
        torch.cuda.empty_cache()

    print("\n===== EXP109 SUMMARY (C100 density-ratio pooling; gate 0.15) "
          "=====")
    print(f"  {'cfg':<12}{'spk r1':>8}{'spk r2':>8}{'dist r1 (89)':>14}")
    for k, r in results.items():
        pur = r["purity"] + [float("nan")] * (2 - len(r["purity"]))
        dr = (r["dist_purity"] or [float("nan")])[0]
        print(f"  {k:<12}{pur[0]:>8.3f}{pur[1]:>8.3f}{dr:>14.3f}")
    best = max((r["purity"][0] for r in results.values()
                if r["purity"]), default=float("nan"))
    print(f"  best r1 purity {best:.3f} -> "
          f"{'PREDICTION HOLDS (>0.15)' if best > 0.15 else 'FALSIFIER: C100 undiscoverable by tail-pooling'}")

    os.makedirs(os.path.join("logs", "exp109"), exist_ok=True)
    np.savez(os.path.join("logs", "exp109", "results.npz"),
             summary=np.array([repr(results)], dtype=object))
    print("EXP109 DONE.")


if __name__ == "__main__":
    main()
