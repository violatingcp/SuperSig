"""
Experiment 78: verification of the exp-77 flowers LID-novelty result
(Levina-Bickel LID AUC 0.95-0.96 in every base -- real, artifact, or
rebranded distance?).

Per base (dino/lejepa/visreg) and space (frozen, supcon-ft, ss-ft,
supcon-ft-resnplm-cat):
  - AUC of LID(k=20) with 1000-resample bootstrap 95% CI
  - k-sensitivity: LID AUC at k = 5 / 10 / 20 / 50
  - CONTROLS: kNN-distance AUC (mean dist to 10 nearest seen refs --
    the classic unsupervised local-density baseline our battery never
    had), eucl-to-centroid AUC, Spearman(LID, kNN-dist) on test points
  - per-holdout-class AUC (each holdout class vs seen test), min/max
  - rank-average ensemble LID + eucl

Holdout-rotation test (frozen trunk only, where every class is
legitimately unseen): 10 random 10-class holdouts, LID and kNN-dist AUC
distribution -- are classes 92-101 special?

Contrast cell: cars under the identical protocol (a weak-LID cell).

    python experiments/78_lid_verification.py
    python experiments/78_lid_verification.py --datasets flowers
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from supersig.holdouts import holdout_set, n_holdout, run_tag
import argparse
import importlib
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

exp44 = importlib.import_module("44_transfer_32d")
exp77 = importlib.import_module("77_space_similarity")

SPACES = ["frozen", "supcon-ft", "ss-ft", "supcon-ft-resnplm-cat"]


def knn_dist(Xref, Xq, k=10):
    D = np.sqrt(exp77.sqdist(Xq, Xref))
    return np.sort(D, axis=1)[:, :k].mean(1)


def boot_ci(y, s, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    aucs = []
    idx = np.arange(len(y))
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if y[b].any() and not y[b].all():
            aucs.append(roc_auc_score(y[b], s[b]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def rank(s):
    r = np.empty(len(s))
    r[np.argsort(s)] = np.arange(len(s))
    return r / (len(s) - 1)


def cell_battery(spaces, seen, holdouts, rng, tag):
    print(f"\n===== [{tag}] LID verification =====")
    print(f"  {'space':<26}{'LID20':>7}{'CI':>15}{'kNNd':>7}{'eucl':>7}"
          f"{'ens':>7}{'rho':>6}{'k5':>6}{'k10':>6}{'k50':>6}"
          f"{'clsMin':>7}{'clsMax':>7}")
    out = {}
    for name, (Xtr, ytr, Xte, yte) in spaces.items():
        Xtr = np.asarray(Xtr, np.float64); Xte = np.asarray(Xte, np.float64)
        ref = Xtr[np.isin(ytr, seen)]
        if len(ref) > 4000:
            ref = ref[rng.choice(len(ref), 4000, replace=False)]
        hm = np.isin(yte, list(holdouts))
        lid = {k: exp77.lid_scores(ref, Xte, k=k) for k in (5, 10, 20, 50)}
        kd = knn_dist(ref, Xte, k=10)
        cents = np.stack([Xtr[ytr == c].mean(0) for c in seen])
        eu = np.sqrt(exp77.sqdist(Xte, cents)).min(1)
        auc = {f"lid{k}": float(roc_auc_score(hm, lid[k])) for k in lid}
        auc["knnd"] = float(roc_auc_score(hm, kd))
        auc["eucl"] = float(roc_auc_score(hm, eu))
        ens = rank(lid[20]) + rank(eu)
        auc["ens"] = float(roc_auc_score(hm, ens))
        lo, hi = boot_ci(hm, lid[20])
        rho = float(spearmanr(lid[20], kd).correlation)
        per = {}
        for h in holdouts:
            m = np.isin(yte, seen) | (yte == h)
            if (yte[m] == h).any():
                per[h] = float(roc_auc_score(yte[m] == h, lid[20][m]))
        pv = list(per.values())
        out[name] = dict(auc=auc, ci=(lo, hi), rho=rho, per=per)
        print(f"  {name:<26}{auc['lid20']:>7.3f}"
              f"  [{lo:.3f},{hi:.3f}]{auc['knnd']:>7.3f}{auc['eucl']:>7.3f}"
              f"{auc['ens']:>7.3f}{rho:>6.2f}{auc['lid5']:>6.3f}"
              f"{auc['lid10']:>6.3f}{auc['lid50']:>6.3f}"
              f"{min(pv):>7.3f}{max(pv):>7.3f}")
    return out


def rotation_test(Xtr, ytr, Xte, yte, n_cls, n_hold, rng, n_draws=10):
    res = {"lid": [], "knnd": []}
    for d in range(n_draws):
        hold = set(rng.choice(n_cls, n_hold, replace=False).tolist())
        seen = [c for c in range(n_cls) if c not in hold]
        ref = Xtr[np.isin(ytr, seen)]
        if len(ref) > 4000:
            ref = ref[rng.choice(len(ref), 4000, replace=False)]
        hm = np.isin(yte, list(hold))
        res["lid"].append(roc_auc_score(hm, exp77.lid_scores(ref, Xte,
                                                             k=20)))
        res["knnd"].append(roc_auc_score(hm, knn_dist(ref, Xte, k=10)))
    return {k: (float(np.mean(v)), float(np.std(v)),
                float(np.min(v)), float(np.max(v))) for k, v in res.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["flowers", "cars"])
    ap.add_argument("--emb-dim", type=int, default=100)
    ap.add_argument("--n-draws", type=int, default=10)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    results = {}
    for ds in args.datasets:
        n_cls = 47 if ds == "dtd" else exp44.N_CLASSES[ds]
        nh = n_holdout(ds)
        holdouts = holdout_set(ds, n_cls)
        seen = [c for c in range(n_cls) if c not in holdouts]
        for base in ("dino", "lejepa", "visreg"):
            cell = exp77.transfer_cell(args, ds, base)
            spaces = {k: v for k, v in cell.items() if k in SPACES}
            results[f"{ds}_{base}"] = cell_battery(
                spaces, seen, holdouts, rng, f"{ds}/{base}")
            if base == "dino" and "frozen" in cell:
                Xtr, ytr, Xte, yte = cell["frozen"]
                rot = rotation_test(np.asarray(Xtr, np.float64), ytr,
                                    np.asarray(Xte, np.float64), yte,
                                    n_cls, nh, rng, args.n_draws)
                results[f"{ds}_rotation"] = rot
                print(f"\n  [{ds} frozen holdout-rotation, "
                      f"{args.n_draws} draws of {nh} classes]")
                for k, (m, s, lo, hi) in rot.items():
                    print(f"    {k:<6} AUC {m:.3f}+-{s:.3f} "
                          f"(min {lo:.3f}, max {hi:.3f})")
    os.makedirs(os.path.join("logs", "exp78"), exist_ok=True)
    np.savez(os.path.join("logs", "exp78", f"results{run_tag()}.npz"),
             summary=np.array([repr(results)], dtype=object))
    print("\nDone.")


if __name__ == "__main__":
    main()
