"""
Experiment 108 (IMPROVEMENT_TESTS.md #108): the on-manifold diagnostic,
second attempt -- now with mechanism-derived features.

Exp 90 tried generic tail diagnostics and failed (12/17 vs 11/17 base
rate).  Exp 103 supplied the mechanism: on-manifold novelty sits ON a seen
class by distance but OFF its local sheet, with a more MIXED neighbourhood
than a true member's.  Label-free features per champion cell (novelty
labels used only for the response):

  comp_gap   mean frac-mixed composition of the label-free pool tail
             (test points beyond the 0.95 seen-train distance quantile)
             minus the same mean over the non-tail test population
  rr_disp    dispersion (std over all test queries) of the within-modal-
             class mean log radial ratio  mean_j log(d_j / d_k)  computed
             on the same-class neighbour subset -- exp 103's same-class
             LB statistic before inversion
  ratio      comp_gap / rr_disp

Response: gap = LID20 AUC - eucl AUC on the true holdouts, over the 15
transfer champions + 2 CIFAR concats.  Report Spearman and best-threshold
sign accuracy vs the base rate.

Prediction: sign accuracy >= 15/17.  Falsifier: at or near base rate ->
STOP; the regime is declared empirically identifiable but not predictable
(pre-committed: no third attempt).

    python experiments/108_regime_predictor2.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from supersig.holdouts import n_holdout, run_tag
import argparse
import importlib
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

exp44 = importlib.import_module("44_transfer_32d")
exp77 = importlib.import_module("77_space_similarity")
exp100 = importlib.import_module("100_dense_reach")
exp107 = importlib.import_module("107_composition_vs_lid")

CELLS = [f"{d}:{b}" for d in ("aircraft", "cars", "flowers", "dtd",
                              "galaxy10") for b in ("dino", "lejepa",
                                                    "visreg")] \
    + ["cifar10", "cifar100"]
K = 20


def same_class_logratio(Xref, yref, Xq, k=K):
    """Per-query mean log(d_j/d_k) over the same-(modal)-class subset."""
    D = np.sqrt(exp77.sqdist(np.asarray(Xq, np.float64),
                             np.asarray(Xref, np.float64)))
    idx = np.argsort(D, axis=1)[:, :k]
    out = np.full(len(Xq), np.nan)
    for i in range(len(Xq)):
        d = D[i, idx[i]]
        labs = yref[idx[i]]
        vals, counts = np.unique(labs, return_counts=True)
        same = np.sort(d[labs == vals[np.argmax(counts)]])
        if len(same) >= 5:
            rk = max(same[-1], 1e-12)
            out[i] = np.mean(np.log(np.maximum(same[:-1], 1e-12) / rk))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=",".join(CELLS))
    ap.add_argument("--emb-dim", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-ref", type=int, default=4000)
    ap.add_argument("--q", type=float, default=0.95)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    rows = {}
    print(f"  {'cell':<18}{'gap':>8}{'comp_gap':>9}{'rr_disp':>9}"
          f"{'ratio':>8}{'pool':>6}")
    for cell in args.cells.split(","):
        if cell.startswith("cifar"):
            sp, _ = exp100.cifar_space(cell)
            n_cls = 10 if cell == "cifar10" else 100
            holdouts = {4}
        else:
            ds, base = cell.split(":")
            sp = exp100.champion_space(ds, base)
            n_cls = 47 if ds == "dtd" else exp44.N_CLASSES[ds]
            nh = n_holdout(ds)
            holdouts = set(range(n_cls - nh, n_cls))
        if sp is None:
            print(f"  !! [{cell}] missing banks, skipping")
            continue
        Xtr, ytr, Xte, yte = sp
        Xtr = np.asarray(Xtr, np.float64); Xte = np.asarray(Xte, np.float64)
        ytr = np.asarray(ytr); yte = np.asarray(yte)
        seen = [c for c in range(n_cls) if c not in holdouts]
        m = np.isin(ytr, seen)
        Xr, yr = Xtr[m], ytr[m]
        if len(Xr) > args.max_ref:
            sub = rng.choice(len(Xr), args.max_ref, replace=False)
            Xr, yr = Xr[sub], yr[sub]

        # label-free pool tail (exp-90 construction)
        cents = np.stack([Xtr[ytr == c].mean(0) for c in seen])
        d_tr = np.sqrt(exp77.sqdist(Xtr[m], cents)).min(1)
        d_te = np.sqrt(exp77.sqdist(Xte, cents)).min(1)
        pool = d_te > np.quantile(d_tr, args.q)
        if pool.sum() < 30 or (~pool).sum() < 30:
            print(f"  !! [{cell}] pool too small, skipping")
            continue

        comp = exp107.comp_scores(Xr, yr, Xte, k=K)
        comp_gap = float(comp[pool].mean() - comp[~pool].mean())
        rr = same_class_logratio(Xr, yr, Xte, k=K)
        rr_disp = float(np.nanstd(rr))
        ratio = comp_gap / max(rr_disp, 1e-12)

        # response (labels used ONLY here)
        hm = np.isin(yte, list(holdouts))
        lid_auc = float(roc_auc_score(hm, exp77.lid_scores(Xr, Xte, k=K)))
        eucl_auc = float(roc_auc_score(hm, d_te))
        rows[cell] = dict(gap=lid_auc - eucl_auc, comp_gap=comp_gap,
                          rr_disp=rr_disp, ratio=ratio,
                          pool_n=int(pool.sum()))
        r = rows[cell]
        print(f"  {cell:<18}{r['gap']:>+8.4f}{r['comp_gap']:>9.4f}"
              f"{r['rr_disp']:>9.4f}{r['ratio']:>8.2f}{r['pool_n']:>6}",
              flush=True)

    names = list(rows)
    gap = np.array([rows[c]["gap"] for c in names])
    base = max((gap > 0).sum(), (gap <= 0).sum())
    print(f"\n===== EXP108 SUMMARY ({len(names)} cells; base rate "
          f"{base}/{len(names)}; exp-90 best was 12/17) =====")
    for pred in ("comp_gap", "rr_disp", "ratio"):
        x = np.array([rows[c][pred] for c in names])
        rho = spearmanr(x, gap).correlation
        best = 0
        for t in x:
            for sgn in (1, -1):
                acc = ((sgn * x > sgn * t) == (gap > 0)).mean()
                acc = max(acc, ((sgn * x >= sgn * t) == (gap > 0)).mean())
                best = max(best, acc)
        print(f"  {pred:<9} Spearman(gap)={rho:+.2f}  best sign acc "
              f"{best * len(names):.0f}/{len(names)}")

    os.makedirs(os.path.join("logs", "exp108"), exist_ok=True)
    np.savez(os.path.join("logs", "exp108", f"results{run_tag()}.npz"),
             summary=np.array([repr(rows)], dtype=object))
    print("EXP108 DONE.")


if __name__ == "__main__":
    main()
