"""
Experiment 97 (IMPROVEMENT_TESTS.md #97): M and the bandwidth schedule
versus intrinsic dimension.

Protocol debt for the SparKer battery: M=16 and the sigma0 -> sigma0/10
anneal have never been scanned, while exp 77 measured TwoNN intrinsic
dimensions spanning 2-13 across arms.  Scan M in {4, 16, 64} x
sigma_ratio in {3, 10, 30} on the three cars/dino arms that span the ID
range (nplm-sup-ft ~2-3, sigreg-ssl-ft ~5-7, supcon-ft ~9-13; cached
exp-70 banks, no training).  Per config: toy-calibrated power at f=0.05
(n_null=100, 25 signal toys, alpha=0.05, annealed sigma, 3 checkpoints)
and the sigma-checkpoint at which the mean signal statistic peaks.

Prediction: the required sigma range widens with intrinsic dimension and
M=16 under-resolves multi-class novelty.  Falsifier: power flat in both
knobs -- the annealed default is robust and needs no caveat.

    python experiments/97_sparker_systematics.py
    python experiments/97_sparker_systematics.py --quick --arms supcon-ft
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import importlib
import numpy as np
import torch

from supersig.config import DEVICE
from supersig.sparker import (aggregate_pvalues, clopper_pearson,
                              median_pairwise, np_test_stats)

exp31 = importlib.import_module("31_sparker_power")
exp44 = importlib.import_module("44_transfer_32d")
exp77 = importlib.import_module("77_space_similarity")

ARMS = ["nplm-sup-ft", "sigreg-ssl-ft", "supcon-ft"]   # TwoNN ID ~2/6/11
M_GRID = [4, 16, 64]
RATIO_GRID = [3.0, 10.0, 30.0]


def config_power(bg, sg, R, M, ratio, frac, n_d, n_null, n_toys, alpha,
                 seed):
    kw = dict(M=M, steps=300, sigma_ratio=ratio, n_checkpoints=3)
    sigma0 = median_pairwise(bg, seed=seed)
    rng = np.random.default_rng(seed)
    null_ts = []
    for i in range(n_null):
        b, _ = exp31.toy_indices(rng, len(bg), len(sg), n_d, 0)
        null_ts.append(np_test_stats(bg[torch.as_tensor(b, device=DEVICE)],
                                     R, sigma0=sigma0, seed=seed + i, **kw))
    null_ts = np.array(null_ts)
    null_agg = np.array([aggregate_pvalues(null_ts[i],
                                           np.delete(null_ts, i, axis=0))
                         for i in range(n_null)])
    thr = np.quantile(null_agg, 1.0 - alpha)
    n_sig = int(round(frac * n_d))
    det, sig_ts = 0, []
    for j in range(n_toys):
        b, s = exp31.toy_indices(rng, len(bg), len(sg), n_d, n_sig)
        D = torch.cat([bg[torch.as_tensor(b, device=DEVICE)],
                       sg[torch.as_tensor(s, device=DEVICE)]])
        ts = np_test_stats(D, R, sigma0=sigma0, seed=seed + 7919 + j, **kw)
        sig_ts.append(ts)
        det += int(aggregate_pvalues(ts, null_ts) > thr)
    power = det / n_toys
    lo, hi = clopper_pearson(det, n_toys)
    # which sigma checkpoint carries the signal (background-subtracted)
    excess = np.mean(sig_ts, axis=0) - null_ts.mean(axis=0)
    return power, (lo, hi), int(np.argmax(excess)), excess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cars")
    ap.add_argument("--base", default="dino")
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--emb-dim", type=int, default=100)
    ap.add_argument("--frac", type=float, default=0.05)
    ap.add_argument("--n-d", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    ds, base = args.dataset, args.base
    n_null = 20 if args.quick else 100
    n_toys = 10 if args.quick else 25
    n_cls = 47 if ds == "dtd" else exp44.N_CLASSES[ds]
    holdouts = set(range(n_cls - 10, n_cls))
    seen = [c for c in range(n_cls) if c not in holdouts]

    results = {}
    for arm in args.arms:
        r_tr = exp77.head_emb(ds, base, arm, args.emb_dim, "train")
        r_te = exp77.head_emb(ds, base, arm, args.emb_dim, "test")
        if not (r_tr and r_te):
            print(f"!! missing banks for {arm}, skipping")
            continue
        (Xtr, ytr), (Xte, yte) = r_tr, r_te
        rng = np.random.default_rng(0)
        twonn = exp77.twonn_id(np.asarray(Xtr, np.float64)[
            rng.choice(len(Xtr), min(2000, len(Xtr)), replace=False)], rng)
        R = torch.as_tensor(Xtr[np.isin(ytr, seen)][:20000],
                            dtype=torch.float32, device=DEVICE)
        bg = torch.as_tensor(Xte[np.isin(yte, seen)], dtype=torch.float32,
                             device=DEVICE)
        sg = torch.as_tensor(Xte[np.isin(yte, list(holdouts))],
                             dtype=torch.float32, device=DEVICE)
        print(f"\n######## [{ds}:{base}/{arm}] TwoNN-ID={twonn:.1f} "
              f"########", flush=True)
        for M in M_GRID:
            for ratio in RATIO_GRID:
                p, (lo, hi), peak, excess = config_power(
                    bg, sg, R, M, ratio, args.frac, args.n_d, n_null,
                    n_toys, args.alpha, args.seed)
                results[f"{arm}:M{M}:r{ratio:g}"] = dict(
                    power=p, lo=lo, hi=hi, peak=peak, twonn=twonn,
                    excess=list(excess))
                print(f"  M={M:<3} ratio={ratio:<5g} power={p:.3f} "
                      f"[{lo:.3f},{hi:.3f}]  peak-ckpt={peak} "
                      f"excess=" + "/".join(f"{e:.0f}" for e in excess),
                      flush=True)

    print(f"\n===== EXP97 SUMMARY (power@f={args.frac}, {ds}:{base}) =====")
    print(f"  {'arm (ID)':<22}" + "".join(f"{'M' + str(M):>18}"
                                          for M in M_GRID))
    for arm in args.arms:
        keys = [k for k in results if k.startswith(arm + ":")]
        if not keys:
            continue
        tw = results[keys[0]]["twonn"]
        row = f"  {arm + f' ({tw:.0f})':<22}"
        for M in M_GRID:
            cell = "/".join(f"{results[f'{arm}:M{M}:r{r:g}']['power']:.2f}"
                            for r in RATIO_GRID)
            row += f"{cell:>18}"
        print(row + "   (ratio 3/10/30)")

    os.makedirs(os.path.join("logs", "exp97"), exist_ok=True)
    np.savez(os.path.join("logs", "exp97", "results.npz"),
             summary=np.array([repr(results)], dtype=object))
    print("EXP97 DONE.")


if __name__ == "__main__":
    main()
