"""
Experiment 106 (IMPROVEMENT_TESTS.md #106): panel control -- unfaithful
space, or unestimable covariance?

Exp 104's transfer numbers are shrinkage-dominated (10-40 samples/class in
100-D).  Three controls on our own instrument:

  (a) subsample the CIFAR cells to 10/20/40 samples/class and re-run the
      panel: how much of the transfer-cell degradation is reproduced on a
      space we KNOW is well-estimated at full n?
  (b) re-run the transfer panel at estimable dimension.  The 100-D ft heads
      have no 16/32-D twins on disk and retraining them would not be
      evaluation-only, so the dimension knob here is a PCA projection of the
      same space to 16/32-D -- this holds the space fixed and varies only the
      estimation burden, which is exactly the confound being audited.
  (c) the panel under isotropic and diagonal reference densities as
      shrinkage bounds (cov_mode of the exp-104 panel).

Prediction: the CIFAR subsample reproduces most of the r_llr degradation
but NOT the width (rms) result; rms is first-moment-ish and stable at small
n.  Falsifier: subsampled CIFAR reproduces the width numbers too -> the
0.13-0.79 widths are partly an estimation artifact and the paper's central
negative needs qualifying.

    python experiments/106_panel_control.py
    python experiments/106_panel_control.py --cells cifar10 --quick
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import importlib
import numpy as np

exp44 = importlib.import_module("44_transfer_32d")
exp77 = importlib.import_module("77_space_similarity")
exp104 = importlib.import_module("104_interpretability_panel")

TRANSFER_CELLS = [f"{d}:{b}" for d in ("aircraft", "cars", "flowers",
                                       "dtd", "galaxy10")
                  for b in ("dino", "lejepa", "visreg")]
FIELDS = ("r_ll", "slope", "r_llr", "ece", "sw", "rms", "sep")


def fmt(r):
    return "  ".join(f"{k}={r[k]:.3f}" for k in FIELDS if k in r)


def subsample_per_class(z, y, n_per, rng):
    idx = []
    for c in np.unique(y):
        w = np.where(y == c)[0]
        idx.append(rng.choice(w, min(n_per, len(w)), replace=False))
    idx = np.concatenate(idx)
    return z[idx], y[idx]


def pca_project(ztr, k):
    mu = ztr.mean(0)
    X = ztr - mu
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    return lambda z: (z - mu) @ Vt[:k].T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=",".join(["cifar10", "cifar100"]
                                                + TRANSFER_CELLS))
    ap.add_argument("--emb-dim", type=int, default=100)
    ap.add_argument("--max-rows", type=int, default=8000)
    ap.add_argument("--shrink", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    rows = {}

    for cell in args.cells.split(","):
        if cell.startswith("cifar"):
            ds = cell
            ns = argparse.Namespace(dim=32 if ds == "cifar10" else 100,
                                    arms=[], quick=args.quick)
            spaces = exp77.cifar_cell(ns, ds)
            n_cls = 10 if ds == "cifar10" else 100
            holdouts = {4}
            key_pre = ds
            grids = [10, 20, 40, None]          # (a) subsample grid
        else:
            ds, base = cell.split(":")
            ns = argparse.Namespace(emb_dim=args.emb_dim)
            spaces = exp77.transfer_cell(ns, ds, base)
            spaces.pop("frozen", None)
            n_cls = 47 if ds == "dtd" else exp44.N_CLASSES[ds]
            nh = 1 if ds == "galaxy10" else 10
            holdouts = set(range(n_cls - nh, n_cls))
            key_pre = f"{ds}_{base}"
            grids = [None]                       # transfer: dims + refs only
        seen = np.array([c for c in range(n_cls) if c not in holdouts])
        print(f"\n######## {cell}: {len(spaces)} spaces ########", flush=True)
        for name, (z_tr, y_tr, _, _) in spaces.items():
            z0 = np.asarray(z_tr, dtype=np.float64)
            y0 = np.asarray(y_tr)
            keep = np.isin(y0, seen)
            z0, y0 = z0[keep], y0[keep]
            if len(z0) > args.max_rows:
                idx = rng.choice(len(z0), args.max_rows, replace=False)
                z0, y0 = z0[idx], y0[idx]

            # (a) sample-size grid (CIFAR only) / full-n reference row
            for n_per in grids:
                if n_per is None:
                    z, y = z0, y0
                    tag = "full"
                else:
                    z, y = subsample_per_class(z0, y0, n_per, rng)
                    tag = f"n{n_per}"
                r = exp104.panel(z, y, classes=seen, shrink=args.shrink,
                                 seed=args.seed)
                rows[f"{key_pre}_{name}_{tag}"] = r
                print(f"  {name:<24}{tag:<6}{fmt(r)}", flush=True)

            # (b) PCA dimension grid (transfer only; CIFAR is estimable)
            if not cell.startswith("cifar"):
                for k in (16, 32):
                    if k >= z0.shape[1]:
                        continue
                    proj = pca_project(z0, k)
                    r = exp104.panel(proj(z0), y0, classes=seen,
                                     shrink=args.shrink, seed=args.seed)
                    rows[f"{key_pre}_{name}_pca{k}"] = r
                    print(f"  {name:<24}pca{k:<3}{fmt(r)}", flush=True)

            # (c) reference-density bounds (skip sw: unchanged by cov_mode)
            for cm in ("diag", "iso"):
                r = exp104.panel(z0, y0, classes=seen, shrink=args.shrink,
                                 seed=args.seed, sw=False, cov_mode=cm)
                rows[f"{key_pre}_{name}_{cm}"] = r
                print(f"  {name:<24}{cm:<6}{fmt(r)}", flush=True)

    os.makedirs(os.path.join("logs", "exp106"), exist_ok=True)
    np.savez(os.path.join("logs", "exp106", "results.npz"),
             summary=np.array([repr(rows)], dtype=object))

    # summary: CIFAR rms/r_llr trajectories vs n, transfer full vs pca
    print("\n===== EXP106 SUMMARY (rms | r_llr by condition) =====")
    tags_c = ["n10", "n20", "n40", "full"]
    for k in sorted({r.rsplit("_", 1)[0] for r in rows
                     if r.rsplit("_", 1)[1] in tags_c
                     and r.split("_")[0].startswith("cifar")}):
        tr = [rows.get(f"{k}_{t}") for t in tags_c]
        if all(tr):
            print(f"  {k:<34} rms  " +
                  " ".join(f"{t}={r['rms']:.2f}" for t, r in zip(tags_c, tr))
                  + "   r_llr " +
                  " ".join(f"{t}={r['r_llr']:.2f}"
                           for t, r in zip(tags_c, tr)))
    print("EXP106 DONE.")


if __name__ == "__main__":
    main()
