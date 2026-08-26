"""
Experiment 107 (IMPROVEMENT_TESTS.md #107): is neighbourhood composition
enough?  (LID demotion test)

Exp 103's coda: the bare fraction of a query's k=20 neighbours that are not
of the modal class scores 0.83-0.94 against LID's 0.87-0.96.  Head-to-head
across the 15 transfer champion cells + the two CIFAR record concats:

  lid20   Levina-Bickel LID (the battery's current statistic)
  comp    frac-mixed neighbourhood composition (parameter-free counting)
  knnd    mean distance to the 10 nearest seen refs (exp-78 control)
  eucl    min-centroid distance
  ens     rank(comp) + rank(knnd)  -- scale-free + scale-full ensemble
  ens-lid rank(lid20) + rank(eucl) -- the exp-78 ensemble for reference

plus the exp-78 holdout-rotation control (10 random holdout draws, frozen
ViT banks where every class is legitimately rotatable) for comp vs lid on
flowers and cars.

Prediction: comp ties lid within 0.02 on the on-manifold cells and is no
worse elsewhere; ens beats both.  Falsifier: comp trails by >0.05 on
flowers -> LID's radial-ratio information does real work beyond counting.

    python experiments/107_composition_vs_lid.py
    python experiments/107_composition_vs_lid.py --cells flowers:dino
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from supersig.holdouts import n_holdout, run_tag
import argparse
import importlib
import numpy as np
from sklearn.metrics import roc_auc_score

exp44 = importlib.import_module("44_transfer_32d")
exp72 = importlib.import_module("72_residual_discovery")
exp77 = importlib.import_module("77_space_similarity")
exp78 = importlib.import_module("78_lid_verification")
exp100 = importlib.import_module("100_dense_reach")

CELLS = [f"{d}:{b}" for d in ("aircraft", "cars", "flowers", "dtd",
                              "galaxy10") for b in ("dino", "lejepa",
                                                    "visreg")]
K = 20


def comp_scores(Xref, yref, Xq, k=K):
    """Frac-mixed: share of the k nearest refs not of the modal class."""
    D = exp77.sqdist(np.asarray(Xq, np.float64),
                     np.asarray(Xref, np.float64))
    idx = np.argpartition(D, k, axis=1)[:, :k]
    out = np.empty(len(Xq))
    for i in range(len(Xq)):
        labs = yref[idx[i]]
        _, counts = np.unique(labs, return_counts=True)
        out[i] = 1.0 - counts.max() / k
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=",".join(CELLS
                                                + ["cifar10", "cifar100"]))
    ap.add_argument("--emb-dim", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-ref", type=int, default=4000)
    ap.add_argument("--skip-rotation", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    rows = {}
    print(f"  {'cell':<18}{'lid20':>7}{'comp':>7}{'knnd':>7}{'eucl':>7}"
          f"{'ens':>7}{'ens-lid':>8}")
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
        hm = np.isin(yte, list(holdouts))

        lid = exp77.lid_scores(Xr, Xte, k=K)
        comp = comp_scores(Xr, yr, Xte, k=K)
        knnd = exp78.knn_dist(Xr, Xte, k=10)
        cents = np.stack([Xtr[ytr == c].mean(0) for c in seen])
        eu = np.sqrt(exp77.sqdist(Xte, cents)).min(1)
        scores = dict(lid20=lid, comp=comp, knnd=knnd, eucl=eu,
                      ens=exp78.rank(comp) + exp78.rank(knnd),
                      **{"ens-lid": exp78.rank(lid) + exp78.rank(eu)})
        rows[cell] = {k: float(roc_auc_score(hm, s))
                      for k, s in scores.items()}
        r = rows[cell]
        print(f"  {cell:<18}{r['lid20']:>7.3f}{r['comp']:>7.3f}"
              f"{r['knnd']:>7.3f}{r['eucl']:>7.3f}{r['ens']:>7.3f}"
              f"{r['ens-lid']:>8.3f}", flush=True)

    # aggregate + winner
    print("\n===== EXP107 SUMMARY =====")
    names = list(rows)
    for k in ("lid20", "comp", "knnd", "eucl", "ens", "ens-lid"):
        v = [rows[c][k] for c in names]
        print(f"  {k:<8} mean={np.mean(v):.3f}  wins="
              f"{sum(1 for c in names if rows[c][k] >= max(rows[c].values()) - 1e-9)}"
              f"/{len(names)}")
    gaps = [(c, rows[c]["comp"] - rows[c]["lid20"]) for c in names]
    worst = min(gaps, key=lambda t: t[1])
    print(f"  comp-lid gap: mean={np.mean([g for _, g in gaps]):+.3f}  "
          f"worst {worst[0]} {worst[1]:+.3f}")

    rot = {}
    if not args.skip_rotation:
        print("\n  holdout-rotation control (frozen banks, 10 draws)")
        for ds, base in (("flowers", "dino"), ("cars", "dino")):
            cellsp = exp77.transfer_cell(
                argparse.Namespace(emb_dim=args.emb_dim), ds, base)
            if "frozen" not in cellsp:
                print(f"  !! no frozen bank for {ds}:{base}")
                continue
            Xtr, ytr, Xte, yte = cellsp["frozen"]
            Xtr = np.asarray(Xtr, np.float64)
            Xte = np.asarray(Xte, np.float64)
            n_cls = exp44.N_CLASSES[ds]
            res = {"lid": [], "comp": []}
            for d in range(10):
                hold = set(rng.choice(n_cls, 10, replace=False).tolist())
                sn = [c for c in range(n_cls) if c not in hold]
                mm = np.isin(ytr, sn)
                Xr, yr = Xtr[mm], ytr[mm]
                if len(Xr) > args.max_ref:
                    sub = rng.choice(len(Xr), args.max_ref, replace=False)
                    Xr, yr = Xr[sub], yr[sub]
                hm = np.isin(yte, list(hold))
                res["lid"].append(roc_auc_score(
                    hm, exp77.lid_scores(Xr, Xte, k=K)))
                res["comp"].append(roc_auc_score(
                    hm, comp_scores(Xr, yr, Xte, k=K)))
            rot[f"{ds}:{base}"] = {k: (float(np.mean(v)), float(np.std(v)))
                                   for k, v in res.items()}
            print(f"  {ds}:{base}  lid={np.mean(res['lid']):.3f}"
                  f"+-{np.std(res['lid']):.3f}  "
                  f"comp={np.mean(res['comp']):.3f}"
                  f"+-{np.std(res['comp']):.3f}", flush=True)

    os.makedirs(os.path.join("logs", "exp107"), exist_ok=True)
    np.savez(os.path.join("logs", "exp107", f"results{run_tag()}.npz"),
             summary=np.array([repr(dict(rows=rows, rotation=rot))],
                              dtype=object))
    print("EXP107 DONE.")


if __name__ == "__main__":
    main()
