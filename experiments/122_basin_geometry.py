"""
Experiment 122 (IMPROVEMENT_TESTS.md #122): what IS the basin geometry?

We have the best single-loss C100 both-currencies space on record (tau>=0.3
with a SIGReg marginal, exps 110/113) and no account of why it works beyond
"a looser interaction lets the marginal shape the covariance".  Worse, it
CONTRADICTS the program's stated ideal: exp 110 measured it at rms~0.50 with
slope 10-18, i.e. narrow AND strongly anisotropic, while exp 105 showed that
forcing the unit-width isotropic ideal is decorative.  The best space we have
is not the space the theory asked for, and nobody has characterised it.

Hypothesis: the basin spaces are anisotropic in a STRUCTURED way -- high
variance along between-class directions, low along within-class ones -- which
is exactly what makes a tied-covariance Mahalanobis score work while unit width
does not.

Measures, per tau, on the seen classes:
  eig profile   : eigenvalues of the pooled within-class covariance (sorted)
  aniso         : lambda_max / lambda_min of that covariance
  align         : fraction of the between-class centroid subspace energy that
                  lands in the TOP-k within-class eigenvectors.  Under the
                  hypothesis this should FALL as the basin is entered -- the
                  between-class signal moving into the LOW-variance directions
                  is what a whitened distance rewards.
  wb_ratio      : mean within-class variance / mean between-centroid variance
Needs exp 113 re-run with --save-embs.

Falsifier: anisotropy grows but `align` does not fall -> the anisotropy is
unstructured, "looser interaction" is the whole story, and there is nothing
further to characterise.

    python experiments/122_basin_geometry.py --cell cifar100_on
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import glob
import json

import numpy as np


def geometry(tr, tr_lab, seen, topk=None):
    X = np.asarray(tr, np.float64)
    y = np.asarray(tr_lab)
    m = np.isin(y, seen)
    X, y = X[m], y[m]
    d = X.shape[1]
    topk = topk or max(1, d // 4)

    mus = np.stack([X[y == c].mean(0) for c in seen])
    W = np.zeros((d, d))
    n = 0
    for i, c in enumerate(seen):
        Z = X[y == c] - mus[i]
        W += Z.T @ Z
        n += len(Z)
    W /= max(n - len(seen), 1)
    evals, evecs = np.linalg.eigh(W)                 # ascending
    evals = np.clip(evals, 1e-12, None)

    B = mus - mus.mean(0)
    # energy of the between-class subspace in the TOP-k within-class directions
    top = evecs[:, -topk:]
    align = float((B @ top).var(0).sum() / max(B.var(0).sum(), 1e-12))
    return dict(
        aniso=float(evals[-1] / evals[0]),
        eig_top=float(evals[-1]), eig_bot=float(evals[0]),
        eig_ratio_q=float(evals[int(0.75 * d)] / evals[int(0.25 * d)]),
        align_topk=align, topk=int(topk),
        wb_ratio=float(np.mean(evals) / max(B.var(0).sum() / d, 1e-12)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default="logs/exp113/embs")
    ap.add_argument("--cell", default="cifar100_on")
    ap.add_argument("--holdout", type=int, default=4)
    ap.add_argument("--out", default="logs/exp122")
    args = ap.parse_args()

    ds, marg = args.cell.rsplit("_", 1)
    files = sorted(glob.glob(os.path.join(args.archive,
                                          f"{ds}_tau*_{marg}_s*.npz")))
    if not files:
        sys.exit(f"no embeddings under {args.archive}; run exp 113 --save-embs")
    by_tau = {}
    for fn in files:
        tau = float(os.path.basename(fn).split("_tau")[1].split("_")[0])
        by_tau.setdefault(tau, []).append(fn)

    rows = {}
    print(f"{'tau':>6}{'aniso':>10}{'eig_top':>10}{'eig_bot':>10}"
          f"{'align_topk':>12}{'wb_ratio':>10}")
    for tau in sorted(by_tau):
        gs = []
        for fn in by_tau[tau]:
            d = np.load(fn, allow_pickle=True)
            seen = sorted(set(int(c) for c in np.unique(d["tr_lab"]))
                          - {args.holdout})
            gs.append(geometry(d["tr"], d["tr_lab"], seen))
        g = {k: float(np.mean([x[k] for x in gs])) for k in gs[0]}
        rows[tau] = g
        print(f"{tau:>6}{g['aniso']:>10.1f}{g['eig_top']:>10.3f}"
              f"{g['eig_bot']:>10.4f}{g['align_topk']:>12.3f}"
              f"{g['wb_ratio']:>10.2f}")

    os.makedirs(args.out, exist_ok=True)
    json.dump(rows, open(os.path.join(args.out, f"{args.cell}.json"), "w"),
              indent=1)
    taus = sorted(rows)
    lo, hi = taus[0], taus[-1]
    d_aniso = rows[hi]["aniso"] - rows[lo]["aniso"]
    d_align = rows[hi]["align_topk"] - rows[lo]["align_topk"]
    print(f"\nacross the sweep: d(aniso)={d_aniso:+.1f}  "
          f"d(align_topk)={d_align:+.3f}")
    print("\n=== verdict ===")
    if d_aniso > 0 and d_align < -0.05:
        print("  Anisotropy grows AND the between-class subspace moves out of "
              "the high-variance directions: the basin geometry is STRUCTURED, "
              "as hypothesised -- a whitened distance is rewarded.")
    elif d_aniso > 0:
        print("  Anisotropy grows but the between-class alignment does not "
              "fall: the falsifier fires -- the anisotropy is unstructured and "
              "'looser interaction' is the whole story.")
    else:
        print("  Anisotropy does not grow across the sweep: the exp-110 "
              "slope reading does not reproduce here; investigate before "
              "citing either.")


if __name__ == "__main__":
    main()
