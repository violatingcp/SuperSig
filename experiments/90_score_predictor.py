"""
Experiment 90 (IMPROVEMENT_TESTS.md #90): an unsupervised predictor for
WHICH novelty score to use (LID vs distance).

LID is superb on flowers (0.95), useless on cars (0.545); the stated
mechanism -- holdouts on-manifold vs off-manifold -- is measurable without
novelty labels.  On every champion space, pool the label-free outlier tail
(test points beyond the 0.95 quantile of the SEEN-TRAIN min-centroid
distance) and compute three candidate predictors:

  dist_ratio   median(pool min-centroid dist) / median(seen-train class
               radius): large = off-manifold (eucl should win), ~1 =
               on-manifold (LID's regime)
  id_gap       TwoNN ID(pool) - TwoNN ID(rest of test): novelty living in
               locally higher-dimensional neighborhoods favors LID
  lid_gap      mean LID(pool) - mean LID(rest): the score's own tail
               contrast, label-free

Response (labels used ONLY here): gap = LID AUC - eucl AUC on the true
holdouts.  Deliverable: Spearman(predictor, gap) and the best-threshold
sign accuracy across the 14 cells (12 transfer champions + the two CIFAR
record concats).  Prediction: the ID gap predicts the sign at >=10/13.
Falsifier: no candidate separates flowers from cars.

Evaluation only, no training.

    python experiments/90_score_predictor.py
    python experiments/90_score_predictor.py --skip-cifar
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from supersig.holdouts import holdout_set, n_holdout, run_tag
import argparse
import importlib
import numpy as np
from sklearn.metrics import roc_auc_score

exp29 = importlib.import_module("29_residual_finetune")
exp44 = importlib.import_module("44_transfer_32d")
exp72 = importlib.import_module("72_residual_discovery")
exp77 = importlib.import_module("77_space_similarity")


def champion_space(ds, base, emb_dim):
    parent, obj, kind = exp72.WINNERS[(ds, base)]
    c_tr = exp77.head_emb(ds, base, f"{parent}_{obj}", emb_dim, "train")
    c_te = exp77.head_emb(ds, base, f"{parent}_{obj}", emb_dim, "test")
    if not (c_tr and c_te):
        return None
    if kind == "residual":
        return c_tr[0], c_tr[1], c_te[0], c_te[1]
    p_tr = exp77.head_emb(ds, base, parent, emb_dim, "train")
    p_te = exp77.head_emb(ds, base, parent, emb_dim, "test")
    if not (p_tr and p_te):
        return None
    return (np.concatenate([p_tr[0], c_tr[0]], 1), p_tr[1],
            np.concatenate([p_te[0], c_te[0]], 1), p_te[1])


def predictors(Xtr, ytr, Xte, seen, rng, q=0.95):
    """Label-free tail diagnostics vs the seen-train population."""
    Xtr = np.asarray(Xtr, np.float64)
    Xte = np.asarray(Xte, np.float64)
    m = np.isin(ytr, seen)
    cents = np.stack([Xtr[ytr == c].mean(0) for c in seen])
    d_tr = np.sqrt(exp77.sqdist(Xtr[m], cents)).min(1)
    d_te = np.sqrt(exp77.sqdist(Xte, cents)).min(1)
    radius = np.median(d_tr)
    tau = np.quantile(d_tr, q)
    pool = d_te > tau
    if pool.sum() < 30 or (~pool).sum() < 30:
        return None
    dist_ratio = float(np.median(d_te[pool]) / max(radius, 1e-12))
    sub = lambda X, n: X[rng.choice(len(X), min(n, len(X)), replace=False)]
    id_pool = exp77.twonn_id(sub(Xte[pool], 2000), rng)
    id_rest = exp77.twonn_id(sub(Xte[~pool], 2000), rng)
    ref = sub(Xtr[m], 4000)
    lid = exp77.lid_scores(ref, Xte, k=20)
    return dict(dist_ratio=dist_ratio, id_gap=float(id_pool - id_rest),
                lid_gap=float(lid[pool].mean() - lid[~pool].mean()),
                pool_n=int(pool.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dim", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-cifar", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    cells = {}
    for (ds, base) in exp72.WINNERS:
        sp = champion_space(ds, base, args.emb_dim)
        if sp is None:
            print(f"!! missing banks for {ds}:{base}, skipping")
            continue
        n_cls = 47 if ds == "dtd" else exp44.N_CLASSES[ds]
        nh = n_holdout(ds)
        cells[f"{ds}:{base}"] = (sp, holdout_set(ds, n_cls), n_cls)
    if not args.skip_cifar:
        ns = argparse.Namespace(dim=32, arms=[], quick=False)
        for ds, dim, cat in (("cifar10", 32, "res-cat"),
                             ("cifar100", 100, "resnplm-cat")):
            ns.dim = dim
            sp = exp77.cifar_cell(ns, ds).get(cat)
            if sp is not None:
                cells[f"{ds}:{cat}"] = (sp, {4}, 10 if ds == "cifar10"
                                        else 100)

    rows = {}
    print(f"\n  {'cell':<22}{'lidAUC':>8}{'euclAUC':>8}{'gap':>8}"
          f"{'d-ratio':>8}{'id_gap':>8}{'lid_gap':>8}{'pool':>6}")
    for cell, ((Xtr, ytr, Xte, yte), holdouts, n_cls) in cells.items():
        seen = [c for c in range(n_cls) if c not in holdouts]
        is_unseen = np.isin(yte, list(holdouts)).astype(int)
        lid_s = exp29.lid_novelty(Xtr, ytr, Xte, seen, k=20)
        lid_auc = float(roc_auc_score(is_unseen, lid_s))
        eucl_auc = exp77.eucl_auc(Xtr, ytr, Xte, yte, seen, holdouts)
        p = predictors(Xtr, ytr, Xte, seen, rng)
        if p is None:
            print(f"  {cell:<22} pool too small, skipped")
            continue
        rows[cell] = dict(lid=lid_auc, eucl=eucl_auc,
                          gap=lid_auc - eucl_auc, **p)
        print(f"  {cell:<22}{lid_auc:>8.4f}{eucl_auc:>8.4f}"
              f"{lid_auc - eucl_auc:>+8.4f}{p['dist_ratio']:>8.3f}"
              f"{p['id_gap']:>8.3f}{p['lid_gap']:>8.2f}{p['pool_n']:>6}")

    from scipy.stats import spearmanr
    names = list(rows)
    gap = np.array([rows[c]["gap"] for c in names])
    print(f"\n===== EXP90 SUMMARY ({len(names)} cells; response = "
          f"LID-eucl AUC gap) =====")
    for pred in ("dist_ratio", "id_gap", "lid_gap"):
        x = np.array([rows[c][pred] for c in names])
        rho = spearmanr(x, gap).correlation
        # best-threshold sign accuracy (either direction)
        best = 0
        for t in x:
            for sgn in (1, -1):
                acc = ((sgn * x > sgn * t) == (gap > 0)).mean()
                acc = max(acc, ((sgn * x >= sgn * t) == (gap > 0)).mean())
                best = max(best, acc)
        print(f"  {pred:<11} Spearman(gap)={rho:+.2f}  best sign acc "
              f"{best * len(names):.0f}/{len(names)}")

    os.makedirs(os.path.join("logs", "exp90"), exist_ok=True)
    np.savez(os.path.join("logs", "exp90", f"results{run_tag()}.npz"),
             summary=np.array([repr(rows)], dtype=object))
    print("EXP90 DONE.")


if __name__ == "__main__":
    main()
