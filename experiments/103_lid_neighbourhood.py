"""
Experiment 103 (IMPROVEMENT_TESTS.md #103): where does the LID class-level
signal actually live?

Exp 87 falsified the within-class story (flowers LID lives in the parent
half); exp 90 falsified the tail diagnostics.  Third mechanism probe:
decompose LID by NEIGHBOURHOOD COMPOSITION.  For every test query, take
its k=20 nearest seen-train references, find the modal reference class m,
and recompute the Levina-Bickel estimate separately on the same-class
subset (neighbours with label m) and the mixed subset (the rest), each
against its own max radius.  If flowers novelty is a class-level anomaly
(the point sits BETWEEN seen classes), the mixed-neighbourhood component
should carry the discrimination and same-class should not; on cars
neither should.

Falsifier: both components discriminate equally -> "class-level" is not
the right description; third failed mechanism story, cap the line.

Evaluation-only, champion concat + parent half, flowers vs cars x 3
bases.

    python experiments/103_lid_neighbourhood.py
    python experiments/103_lid_neighbourhood.py --cells flowers:dino
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import importlib
import numpy as np
from sklearn.metrics import roc_auc_score

exp44 = importlib.import_module("44_transfer_32d")
exp72 = importlib.import_module("72_residual_discovery")
exp77 = importlib.import_module("77_space_similarity")

CELLS = [f"{d}:{b}" for d in ("flowers", "cars")
         for b in ("dino", "lejepa", "visreg")]
K = 20
MIN_SUB = 5


def lb(P_sub):
    """Levina-Bickel on one neighbour-distance subset (own max radius)."""
    if len(P_sub) < MIN_SUB:
        return np.nan
    rk = max(P_sub[-1], 1e-12)
    m = np.mean(np.log(np.maximum(P_sub[:-1], 1e-12) / rk))
    return -1.0 / min(m, -1e-3)


def decompose(Xref, yref, Xq, k=K):
    """(lid_full, lid_same, lid_mixed, frac_mixed) per query."""
    D = np.sqrt(exp77.sqdist(np.asarray(Xq, np.float64),
                             np.asarray(Xref, np.float64)))
    idx = np.argsort(D, axis=1)[:, :k]
    out = np.full((len(Xq), 4), np.nan)
    for i in range(len(Xq)):
        d = D[i, idx[i]]
        labs = yref[idx[i]]
        vals, counts = np.unique(labs, return_counts=True)
        modal = vals[np.argmax(counts)]
        same = np.sort(d[labs == modal])
        mixed = np.sort(d[labs != modal])
        out[i] = (lb(np.sort(d)), lb(same), lb(mixed),
                  float((labs != modal).mean()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=",".join(CELLS))
    ap.add_argument("--emb-dim", type=int, default=100)
    ap.add_argument("--max-ref", type=int, default=4000)
    ap.add_argument("--max-q", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    print(f"  {'cell':<18}{'half':<8}{'full':>7}{'same':>7}{'mixed':>7}"
          f"{'fracmix-AUC':>12}")
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
            print(f"  !! [{cell}] missing banks")
            continue
        halves = {"parent": (p_tr[0], p_tr[1], p_te[0], p_te[1]),
                  "concat": (np.concatenate([p_tr[0], c_tr[0]], 1), p_tr[1],
                             np.concatenate([p_te[0], c_te[0]], 1),
                             p_te[1])}
        for half, (Xtr, ytr, Xte, yte) in halves.items():
            m = np.isin(ytr, seen)
            Xr, yr = Xtr[m], ytr[m]
            if len(Xr) > args.max_ref:
                sub = rng.choice(len(Xr), args.max_ref, replace=False)
                Xr, yr = Xr[sub], yr[sub]
            qi = (rng.choice(len(yte), args.max_q, replace=False)
                  if len(yte) > args.max_q else np.arange(len(yte)))
            comp = decompose(Xr, yr, Xte[qi])
            is_unseen = np.isin(yte[qi], list(holdouts)).astype(int)
            aucs = {}
            for j, name in enumerate(("full", "same", "mixed", "fracmix")):
                v = comp[:, j]
                ok = np.isfinite(v)
                aucs[name] = (float(roc_auc_score(is_unseen[ok], v[ok]))
                              if is_unseen[ok].sum() and
                              (1 - is_unseen[ok]).sum() else float("nan"))
            results[f"{cell}:{half}"] = aucs
            print(f"  {cell:<18}{half:<8}{aucs['full']:>7.3f}"
                  f"{aucs['same']:>7.3f}{aucs['mixed']:>7.3f}"
                  f"{aucs['fracmix']:>12.3f}", flush=True)

    os.makedirs(os.path.join("logs", "exp103"), exist_ok=True)
    np.savez(os.path.join("logs", "exp103", "results.npz"),
             summary=np.array([repr(results)], dtype=object))
    print("EXP103 DONE.")


if __name__ == "__main__":
    main()
