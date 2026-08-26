"""
Experiment 119 (IMPROVEMENT_TESTS.md #119): an 18th cell -- rr_disp out of
sample.

Exp 108 met its pre-registered bar (15/17) for predicting the LID-vs-distance
regime from the within-modal-class radial-ratio dispersion, but stated the
caveat itself: the threshold is chosen IN-SAMPLE on the same 17 cells exp 90
used, and Spearman 0.37 says the relation is threshold-like, not monotone.  As
it stands the supportable claim is "the regime is label-free identifiable", not
"here is a predictor".  Genuinely held-out cells convert the first into the
second, or kill it.

Protocol, and the point is the ORDER of operations:
  1. `--freeze`  : recompute the exp-108 threshold on the original 17 cells and
                   write it to logs/exp119/threshold.json.  Run once.
  2. `--predict` : for each NEW cell, compute rr_disp, apply the FROZEN
                   threshold, and write the predicted sign to
                   logs/exp119/predictions.json.  No novelty labels are touched.
  3. `--score`   : only now measure the actual LID-eucl gap and compare.
The three steps are separate subcommands precisely so the prediction is
committed before the measurement exists.

Prediction: correct sign on >= 4 of 5 held-out cells at the frozen threshold.
Falsifier: <= 3 of 5 -> rr_disp is an in-sample artifact, exp 108 is downgraded
to a descriptive correlation, and per its own stopping rule the regime line
closes.

    python experiments/119_rrdisp_holdout_cell.py --freeze
    python experiments/119_rrdisp_holdout_cell.py --predict --cells food101:dino
    python experiments/119_rrdisp_holdout_cell.py --score
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from supersig.holdouts import n_holdout, run_tag
import argparse
import importlib
import json

import numpy as np

exp77 = importlib.import_module("77_space_similarity")
exp100 = importlib.import_module("100_dense_reach")
exp108 = importlib.import_module("108_regime_predictor2")

OUT = os.path.join("logs", "exp119")


def rr_disp(Xtr, ytr, Xte, seen, k=20):
    """The exp-108 winning feature: dispersion of the within-modal-class mean
    log radial ratio (the quantity LID inverts).  Built on exp-108's own
    primitive so the feature is identical to the one that scored 15/17."""
    ref = np.asarray(Xtr[np.isin(ytr, seen)], np.float32)
    yref = np.asarray(ytr[np.isin(ytr, seen)])
    lr = exp108.same_class_logratio(ref, yref, np.asarray(Xte, np.float32), k=k)
    lr = np.asarray(lr, float)
    lr = lr[np.isfinite(lr)]
    return float(lr.std())


def cell_arrays(ds, base=None):
    """(Xtr, ytr, Xte, yte, seen, holdouts) for a champion cell (incl. the
    CIFAR record concats), or -- for a NEW dataset with no exp-72 champion --
    its frozen ViT bank (that is the only space a genuinely held-out cell can
    have; subsampled so the O(n^2) feature/gap computations stay tractable)."""
    import torch as T
    exp44 = importlib.import_module("44_transfer_32d")
    exp72 = importlib.import_module("72_residual_discovery")
    if base is None:                                  # cifar10 / cifar100
        sp, _ = exp100.cifar_space(ds)
        if sp is None:
            raise RuntimeError(f"missing CIFAR ckpts for {ds}")
        Xtr, ytr, Xte, yte = sp
        n_cls = 10 if ds == "cifar10" else 100
        holdouts = {4}
        seen = [c for c in range(n_cls) if c not in holdouts]
        return (np.asarray(Xtr, np.float32), np.asarray(ytr),
                np.asarray(Xte, np.float32), np.asarray(yte), seen, holdouts)
    if (ds, base) in exp72.WINNERS:
        sp = exp100.champion_space(ds, base)
        if sp is None:
            raise RuntimeError(f"missing banks for {ds}:{base}")
        Xtr, ytr, Xte, yte = sp
    else:
        from supersig.config import DATA_DIR
        fz = os.path.join(DATA_DIR, f"tf_feats_{ds}_{base}_vitb16.pt")
        if not os.path.exists(fz):
            raise RuntimeError(f"missing frozen bank {fz}")
        b = T.load(fz)
        Xtr, ytr = b["train"][0].float().numpy(), b["train"][1].numpy()
        Xte, yte = b["test"][0].float().numpy(), b["test"][1].numpy()
        rng = np.random.default_rng(0)
        keep = np.concatenate([
            rng.choice(np.where(ytr == c)[0],
                       min(60, (ytr == c).sum()), replace=False)
            for c in np.unique(ytr)])
        Xtr, ytr = Xtr[keep], ytr[keep]
        if len(yte) > 6000:
            qi = rng.choice(len(yte), 6000, replace=False)
            Xte, yte = Xte[qi], yte[qi]
    n_cls = 47 if ds == "dtd" else exp44.N_CLASSES[ds]
    nh = n_holdout(ds)
    holdouts = set(range(n_cls - nh, n_cls))
    seen = [c for c in range(n_cls) if c not in holdouts]
    return (np.asarray(Xtr, np.float32), np.asarray(ytr),
            np.asarray(Xte, np.float32), np.asarray(yte), seen, holdouts)


def measured_gap(ds, base, k=20, seed=0):
    """LID AUC minus eucl AUC on a cell -- the response exp 108 regressed."""
    from supersig.discovery import lid_pool_scores
    import torch as T
    Xtr, ytr, Xte, yte, seen, holdouts = cell_arrays(ds, base)
    is_novel = np.isin(yte, list(holdouts)).astype(int)
    z = T.as_tensor(np.concatenate([Xtr[np.isin(ytr, seen)], Xte]))
    is_seen = np.concatenate([np.ones((np.isin(ytr, seen)).sum(), bool),
                              np.zeros(len(Xte), bool)])
    lid = lid_pool_scores(z, is_seen, k=k, seed=seed).cpu().numpy()[-len(Xte):]
    eu = exp77.eucl_auc(Xtr, ytr, Xte, yte, seen, holdouts)
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(is_novel, lid)) - float(eu)


def _load(name):
    p = os.path.join(OUT, name)
    return json.load(open(p)) if os.path.exists(p) else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--predict", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--cells", default="",
                    help="comma list of NEW ds:base cells, e.g. food101:dino")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    if args.freeze:
        # recompute rr_disp and the measured gap on the ORIGINAL 17 cells and
        # store the best in-sample threshold, so it can never drift afterwards.
        rr, gp = [], []
        for cell in exp108.CELLS:
            ds, base = cell.split(":") if ":" in cell else (cell, None)
            try:
                Xtr, ytr, Xte, _yte, seen, _h = cell_arrays(ds, base)
            except Exception as e:
                print(f"  skip {cell}: {e}"); continue
            rr.append(rr_disp(Xtr, ytr, Xte, seen, k=args.k))
            gp.append(measured_gap(ds, base, k=args.k, seed=args.seed))
            print(f"  {cell}: rr_disp={rr[-1]:.5f} gap={gp[-1]:+.3f}")
        rr = np.asarray(rr, float); gp = np.asarray(gp, float)
        order = np.argsort(rr)
        best, thr = -1, None
        for i in range(len(rr) - 1):
            t = 0.5 * (rr[order[i]] + rr[order[i + 1]])
            acc = max(((rr > t) == (gp > 0)).mean(),
                      ((rr < t) == (gp > 0)).mean())
            if acc > best:
                best, thr = acc, float(t)
        sign = 1 if ((rr > thr) == (gp > 0)).mean() >= 0.5 else -1
        json.dump(dict(threshold=thr, sign=sign, in_sample_acc=float(best),
                       n_cells=len(rr), k=args.k),
                  open(os.path.join(OUT, f"threshold{run_tag()}.json"), "w"), indent=1)
        print(f"frozen threshold={thr:.5f} sign={sign:+d} "
              f"in-sample acc={best:.3f} on {len(rr)} cells")
        return

    if args.predict:
        thr = _load("threshold.json")
        if not thr:
            sys.exit("run --freeze first")
        preds = _load("predictions.json")
        for cell in [c for c in args.cells.split(",") if c]:
            ds, base = cell.split(":") if ":" in cell else (cell, None)
            Xtr, ytr, Xte, _yte, seen, _h = cell_arrays(ds, base)
            rr = rr_disp(Xtr, ytr, Xte, seen, k=args.k)
            pred = "LID" if (rr > thr["threshold"]) == (thr["sign"] > 0) \
                else "distance"
            preds[cell] = dict(rr_disp=float(rr), predicted=pred)
            print(f"  {cell}: rr_disp={rr:.5f} -> predict {pred}")
        json.dump(preds, open(os.path.join(OUT, f"predictions{run_tag()}.json"), "w"),
                  indent=1)
        print("\npredictions committed; only now run --score")
        return

    if args.score:
        preds = _load("predictions.json")
        if not preds:
            sys.exit("run --predict first")
        n_ok = 0
        print(f"{'cell':22s}{'predicted':>11s}{'actual':>10s}{'gap':>9s}")
        for cell, p in preds.items():
            ds, base = cell.split(":") if ":" in cell else (cell, None)
            gap = measured_gap(ds, base, k=args.k, seed=args.seed)
            actual = "LID" if gap > 0 else "distance"
            ok = actual == p["predicted"]
            n_ok += ok
            print(f"{cell:22s}{p['predicted']:>11s}{actual:>10s}{gap:>9.3f}"
                  f"  {'OK' if ok else 'MISS'}")
        n = len(preds)
        print(f"\n{n_ok}/{n} correct at the frozen threshold.")
        print("Bar was >=4/5.  " + ("PASSES -- rr_disp is a predictor."
                                    if n and n_ok / n >= 0.8 else
                                    "FAILS -- per exp 108's stopping rule the "
                                    "regime line closes; report rr_disp as a "
                                    "descriptive correlation only."))
        return

    ap.print_help()


if __name__ == "__main__":
    main()
