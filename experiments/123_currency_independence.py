"""
Experiment 123 (IMPROVEMENT_TESTS.md #123): are the four currencies actually
independent?

The paper's organising claim is that a discovery-ready space must be audited in
four currencies -- detection, calibration, discoverability, interpretability.
We have never tested whether they are empirically separable.  If detection and
interpretability correlate at 0.9 across the campaign's spaces, the battery is
over-specified and the paper should say so.

Analysis-only: harvests every scalar metric on every space across the archives,
builds a (space x metric) matrix, and asks how many latent dimensions the
battery actually spans -- by correlation structure, by PCA scree, and by the
average within- vs between-currency correlation.

Prediction: THREE latent factors, not four -- probe and semantic agreement load
together (both track supervision), leaving detection, calibration and
interpretability-as-fidelity as the separable axes.
Falsifier: four or more clean factors -> the framing is vindicated as stated.

    python experiments/123_currency_independence.py
    python experiments/123_currency_independence.py --min-cov 0.5
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import collections
import glob
import json
import re

import numpy as np

# metric -> currency, per the paper's taxonomy
CURRENCY = {
    "probe": "detection", "acc": "detection", "sup_auc": "detection",
    "eucl": "calibration", "mahaT": "calibration", "mahaPC": "calibration",
    "perevt": "calibration", "sparker": "calibration", "lid": "calibration",
    "purity": "discoverability", "margin": "discoverability",
    "khat": "discoverability", "n_anchors": "discoverability",
    "p_rms": "interpretability", "p_sw": "interpretability",
    "p_slope": "interpretability", "p_ece": "interpretability",
    "p_ll": "interpretability", "p_llr": "interpretability",
    "p_sep": "interpretability",
}
SEED_RE = re.compile(r"^(?P<stem>.+?)_s\d+(?P<tail>_.*)?$")


def harvest():
    """(space -> {metric: value}) across every archive, seeds averaged."""
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for fn in sorted(glob.glob("logs/exp*/*.npz")):
        try:
            d = np.load(fn, allow_pickle=True)
        except Exception:
            continue
        exp = fn.split(os.sep)[1]
        for k in d.files:
            v = np.asarray(d[k])
            if v.shape != () or not np.issubdtype(v.dtype, np.number):
                continue
            for met in CURRENCY:
                if k.endswith("_" + met) or k == met:
                    stem = k[: -(len(met) + 1)] if k != met else "root"
                    m = SEED_RE.match(stem)
                    if m:
                        stem = m.group("stem") + (m.group("tail") or "")
                    val = float(v)
                    if np.isfinite(val):
                        acc[f"{exp}/{stem}"][met].append(val)
                    break
    return {sp: {m: float(np.mean(vs)) for m, vs in mm.items()}
            for sp, mm in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-cov", type=float, default=0.5,
                    help="keep metrics present on >= this fraction of spaces")
    ap.add_argument("--out", default="logs/exp123")
    args = ap.parse_args()

    spaces = harvest()
    if not spaces:
        sys.exit("no archives found")
    mets = collections.Counter()
    for mm in spaces.values():
        mets.update(mm)
    keep = [m for m, c in mets.items() if c >= args.min_cov * len(spaces)]
    keep.sort(key=lambda m: (CURRENCY[m], m))
    rows = [sp for sp, mm in spaces.items() if all(m in mm for m in keep)]
    if len(rows) < 10 or len(keep) < 4:
        print(f"only {len(rows)} complete spaces x {len(keep)} metrics at "
              f"--min-cov {args.min_cov}; lower it")
        print("metric coverage:", dict(mets.most_common(20)))
        return
    X = np.array([[spaces[sp][m] for m in keep] for sp in rows], float)
    print(f"{len(rows)} spaces x {len(keep)} metrics")
    print("metrics:", ", ".join(f"{m}({CURRENCY[m][:4]})" for m in keep))

    Z = (X - X.mean(0)) / (X.std(0) + 1e-12)
    C = np.corrcoef(Z.T)

    # within- vs between-currency mean |correlation|
    wi, bt = [], []
    for i in range(len(keep)):
        for j in range(i + 1, len(keep)):
            (wi if CURRENCY[keep[i]] == CURRENCY[keep[j]] else bt).append(
                abs(C[i, j]))
    print(f"\nmean |corr| within currency : {np.mean(wi):.3f}  (n={len(wi)})")
    print(f"mean |corr| between currency: {np.mean(bt):.3f}  (n={len(bt)})")
    print("  -> currencies are separable iff within >> between")

    ev = np.linalg.eigvalsh(C)[::-1]
    ev = np.clip(ev, 0, None)
    frac = ev / ev.sum()
    print("\nPCA scree (fraction of variance):")
    print("  " + "  ".join(f"{f:.2f}" for f in frac[:8]))
    k80 = int(np.searchsorted(np.cumsum(frac), 0.80) + 1)
    kaiser = int((ev > 1.0).sum())
    print(f"  components for 80% variance: {k80};  eigenvalue>1 (Kaiser): "
          f"{kaiser}")

    os.makedirs(args.out, exist_ok=True)
    json.dump(dict(metrics=keep, n_spaces=len(rows),
                   within=float(np.mean(wi)), between=float(np.mean(bt)),
                   scree=[float(x) for x in frac], k80=k80, kaiser=kaiser,
                   corr=C.tolist()),
              open(os.path.join(args.out, "results.json"), "w"), indent=1)

    print("\n=== verdict ===")
    n_cur = len({CURRENCY[m] for m in keep})
    if kaiser >= n_cur and np.mean(wi) > np.mean(bt) + 0.1:
        print(f"  {kaiser} factors for {n_cur} currencies represented, and "
              "within-currency correlation dominates: the framing is "
              "VINDICATED as stated.")
    elif kaiser < n_cur:
        print(f"  Only {kaiser} factors span {n_cur} currencies: the battery "
              "is OVER-SPECIFIED and the paper should collapse the "
              "correlated axes.")
    else:
        print("  Factor count matches but within-currency correlation does "
              "NOT dominate: the currency LABELS may not carve the metric "
              "space at its joints, even if the dimension count is right.")
    print(f"  (scope: metrics present on >={args.min_cov:.0%} of spaces)")


if __name__ == "__main__":
    main()
