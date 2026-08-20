"""
Experiment 87 (IMPROVEMENT_TESTS.md #87): LID on the residual half.

LID is a local-dimensionality statistic (flowers 0.951-0.955, exp 78); the
residual half is by construction where within-class variation lives (exp 76:
the plain-res residual half scrambles class semantics).  LID computed on the
parent half may be measuring the wrong geometry.  Prediction: residual-half
LID >= concat LID > parent-half LID on flowers, and residual-half LID rescues
the weak concat LID on cars (0.545) where distance scores currently win.
Falsifier: residual-half LID uniformly worse (lower SNR).

Evaluation only, no training: parent/child head embeddings from the cached
exp-70/71 banks (exp-77 loaders) on the exp-72 champion cells, flowers and
cars x {dino, lejepa, visreg}.  Reports LID(k=20) AUC for parent half,
residual half, concat (+ eucl reference), per-holdout-class AUC min/max, and
a pseudo-rotation control (10 random 10-class draws treated as novel at eval
time).  CAVEAT printed with the rotation: unlike exp-78's frozen-trunk
rotation, these spaces were fine-tuned WITH the rotated classes labeled, so
the control is conservative (trained classes are tighter than true novelty).

    python experiments/87_lid_residual_half.py
    python experiments/87_lid_residual_half.py --cells flowers:dino
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import importlib
import numpy as np
from sklearn.metrics import roc_auc_score

exp29 = importlib.import_module("29_residual_finetune")
exp44 = importlib.import_module("44_transfer_32d")
exp72 = importlib.import_module("72_residual_discovery")
exp77 = importlib.import_module("77_space_similarity")

CELLS = [f"{ds}:{b}" for ds in ("flowers", "cars")
         for b in ("dino", "lejepa", "visreg")]


def lid_auc(tr, tr_lab, te, te_lab, seen, holdouts, k=20):
    s = exp29.lid_novelty(tr, tr_lab, te, seen, k=k)
    is_unseen = np.isin(te_lab, list(holdouts)).astype(int)
    auc = float(roc_auc_score(is_unseen, s))
    per = [float(roc_auc_score((te_lab == c).astype(int)[~is_unseen.astype(bool)
                                                        | (te_lab == c)],
                               s[~is_unseen.astype(bool) | (te_lab == c)]))
           for c in sorted(holdouts)]
    return auc, min(per), max(per), s


def eucl_ref(tr, tr_lab, te, te_lab, seen, holdouts):
    cents = np.stack([tr[tr_lab == c].mean(0) for c in seen])
    d = np.sqrt(exp77.sqdist(np.asarray(te, np.float64),
                             np.asarray(cents, np.float64))).min(1)
    return float(roc_auc_score(np.isin(te_lab, list(holdouts)).astype(int), d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=",".join(CELLS))
    ap.add_argument("--emb-dim", type=int, default=100)
    ap.add_argument("--n-rot", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    results = {}
    for cell in args.cells.split(","):
        ds, base = cell.split(":")
        parent, obj, kind = exp72.WINNERS[(ds, base)]
        n_cls = exp44.N_CLASSES[ds]
        holdouts = set(range(n_cls - 10, n_cls))
        seen = [c for c in range(n_cls) if c not in holdouts]
        p_tr = exp77.head_emb(ds, base, parent, args.emb_dim, "train")
        p_te = exp77.head_emb(ds, base, parent, args.emb_dim, "test")
        c_tr = exp77.head_emb(ds, base, f"{parent}_{obj}", args.emb_dim,
                              "train")
        c_te = exp77.head_emb(ds, base, f"{parent}_{obj}", args.emb_dim,
                              "test")
        if not (p_tr and p_te and c_tr and c_te):
            print(f"!! [{cell}] missing banks, skipping")
            continue
        (Ptr, ytr), (Pte, yte) = p_tr, p_te
        (Rtr, _), (Rte, _) = c_tr, c_te
        halves = {
            "parent": (Ptr, Pte),
            "residual": (Rtr, Rte),
            "concat": (np.concatenate([Ptr, Rtr], 1),
                       np.concatenate([Pte, Rte], 1)),
        }
        print(f"\n######## [{cell}] {parent}->{obj} ########")
        print(f"  {'half':<10}{'lid AUC':>9}{'pc min':>8}{'pc max':>8}"
              f"{'eucl':>8}")
        cellres = {}
        for name, (tr, te) in halves.items():
            auc, mn, mx, _ = lid_auc(tr, ytr, te, yte, seen, holdouts)
            eu = eucl_ref(tr, ytr, te, yte, seen, holdouts)
            cellres[name] = dict(lid=auc, pc_min=mn, pc_max=mx, eucl=eu)
            print(f"  {name:<10}{auc:>9.4f}{mn:>8.3f}{mx:>8.3f}{eu:>8.4f}")

        # pseudo-rotation control (conservative -- see module docstring)
        rng = np.random.default_rng(args.seed)
        rot = {name: [] for name in halves}
        for r in range(args.n_rot):
            fake = set(rng.choice(seen, size=10, replace=False).tolist())
            fseen = [c for c in seen if c not in fake]
            for name, (tr, te) in halves.items():
                m = ~np.isin(yte, list(holdouts))   # true holdouts excluded
                s = exp29.lid_novelty(tr[np.isin(ytr, fseen)],
                                      ytr[np.isin(ytr, fseen)],
                                      te[m], fseen, k=20, seed=args.seed + r)
                rot[name].append(float(roc_auc_score(
                    np.isin(yte[m], list(fake)).astype(int), s)))
        print("  pseudo-rotation (10 random seen-class draws; conservative "
              "-- classes were labeled during ft):")
        for name in halves:
            a = np.array(rot[name])
            cellres[name]["rot_mean"] = float(a.mean())
            cellres[name]["rot_sd"] = float(a.std())
            print(f"    {name:<10}{a.mean():.4f}+-{a.std():.4f} "
                  f"min={a.min():.4f}")
        results[cell] = cellres

    print("\n===== EXP87 SUMMARY (LID by half; eucl for reference) =====")
    print(f"  {'cell':<18}{'parent':>9}{'residual':>9}{'concat':>9}"
          f"{'eucl-best':>10}")
    for cell, r in results.items():
        eu = max(v["eucl"] for v in r.values())
        print(f"  {cell:<18}{r['parent']['lid']:>9.4f}"
              f"{r['residual']['lid']:>9.4f}{r['concat']['lid']:>9.4f}"
              f"{eu:>10.4f}")

    os.makedirs(os.path.join("logs", "exp87"), exist_ok=True)
    np.savez(os.path.join("logs", "exp87", "results.npz"),
             summary=np.array([repr(results)], dtype=object),
             **{f"{c.replace(':', '_')}_{h}_{k}": v
                for c, cr in results.items() for h, hv in cr.items()
                for k, v in hv.items()})
    print("EXP87 DONE.")


if __name__ == "__main__":
    main()
