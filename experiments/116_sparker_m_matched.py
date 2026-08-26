"""
Experiment 116 (IMPROVEMENT_TESTS.md #116): SparKer M matched to intrinsic
dimension, applied to the reach table.

Exp 97 found that the kernel count M must be matched INVERSELY to intrinsic
dimension (high-ID spaces want FEWER kernels -- more kernels overfit the null
in high dimension and raise the threshold), and that the default M=16 leaves up
to 0.16 of power on the table on high-ID spaces.  Exp 100 then measured the
entire reach table at M=16.  This re-measures f95 with M set from each space's
own TwoNN intrinsic dimension.

Prediction: reach improves on the high-ID (softmax-parent) cells and several of
the seven `>0.1` cells cross for the first time; the dataset ORDERING is
preserved.
Falsifier: the ordering changes -> our sensitivity ranking was an artifact of a
fixed kernel budget and exp 100's headline needs restating.

    python experiments/116_sparker_m_matched.py
    python experiments/116_sparker_m_matched.py --cells cars:visreg --quick
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from supersig.holdouts import holdout_set, n_holdout, run_tag
import argparse
import importlib

import numpy as np
import torch

from supersig.config import DEVICE

exp31 = importlib.import_module("31_sparker_power")
exp44 = importlib.import_module("44_transfer_32d")
exp77 = importlib.import_module("77_space_similarity")
exp99 = importlib.import_module("99_discovery_reach")
exp100 = importlib.import_module("100_dense_reach")

# exp-97 rule: high intrinsic dimension -> fewer kernels.
def m_for_id(twonn_id):
    if twonn_id >= 9.0:
        return 4
    if twonn_id >= 5.0:
        return 16
    return 64


# Intrinsic dimension via exp-77's TwoNN, so the ID scale here is exactly the
# one exp 97 derived its M rule on.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", default=None,
                    help="ds:base pairs; default = the exp-100 champion cells")
    ap.add_argument("--fractions", default="0.02,0.03,0.04,0.05,0.06,0.08,0.10")
    ap.add_argument("--n-null", type=int, default=100)
    ap.add_argument("--n-sig", type=int, default=50)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    fracs = [float(x) for x in args.fractions.split(",")]
    cells = args.cells or [f"{ds}:{b}" for ds in
                           ("aircraft", "cars", "flowers", "dtd", "galaxy10")
                           for b in ("dino", "lejepa", "visreg")]
    out = os.path.join("logs", "exp116")
    os.makedirs(out, exist_ok=True)
    res = {}

    for cell in cells:
        ds, base = cell.split(":")
        sp = exp100.champion_space(ds, base)
        if sp is None:
            print(f"  !! [{cell}] missing banks, skipping")
            continue
        Xtr, ytr, Xte, yte = sp
        n_cls = 47 if ds == "dtd" else exp44.N_CLASSES[ds]
        nh = n_holdout(ds)
        holdouts = holdout_set(ds, n_cls)
        seen = [c for c in range(n_cls) if c not in holdouts]
        n_d = 2000 if ds in ("cars", "galaxy10") else 1000
        Xtr = np.asarray(Xtr, np.float32); Xte = np.asarray(Xte, np.float32)

        idim = exp77.twonn_id(Xtr[np.isin(ytr, seen)][:4000],
                              np.random.default_rng(args.seed))
        M = m_for_id(idim)
        print(f"\n----- {cell}: TwoNN ID={idim:.1f} -> M={M} -----", flush=True)

        R = torch.as_tensor(Xtr[np.isin(ytr, seen)][:20000], device=DEVICE)
        bg = torch.as_tensor(Xte[np.isin(yte, seen)], device=DEVICE)
        sg = torch.as_tensor(Xte[np.isin(yte, list(holdouts))], device=DEVICE)
        pw, _ = exp31.run_test_battery(
            bg, sg, R, fracs, n_d,
            20 if args.quick else args.n_null,
            5 if args.quick else args.n_sig,
            args.alpha, args.seed, dict(M=M), tag=f"{ds}_{base}_M{M}")
        f95v, flag = exp99.f95(fracs, pw)
        key = f"{ds}_{base}"
        res[f"{key}_power"] = np.asarray(pw, float)
        res[f"{key}_id"] = np.float64(idim)
        res[f"{key}_M"] = np.int64(M)
        res[f"{key}_f95"] = np.float64(f95v if f95v is not None else np.nan)
        np.savez(os.path.join(out, f"results{run_tag()}.npz"),
                 fractions=np.asarray(fracs), **res)
        print(f"  f95={'>0.1' if f95v is None else round(f95v, 4)} ({flag})",
              flush=True)

    print("\n=== reach at ID-matched M (compare against exp-100 M=16) ===")
    for k in sorted(res):
        if k.endswith("_f95"):
            stem = k[:-4]
            v = float(res[k])
            print(f"  {stem:44s} ID={float(res[stem+'_id']):5.1f} "
                  f"M={int(res[stem+'_M']):3d} "
                  f"f95={'>0.1' if np.isnan(v) else f'{v:.4f}'}")


if __name__ == "__main__":
    main()
