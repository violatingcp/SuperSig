"""
Experiment 120 (the thread exp 115 left open): can an OPEN-WORLD-LEGAL tuner
find the temperature calibration basin?

Exp 110/113 found that tau=1.0 with a SIGReg marginal breaks the C100
calibration ceiling.  Exp 115 then showed the discovery is operationally
inert: tuning on seen-class ACCURACY -- the only signal an open-world tuner
was given -- returns tau=0.1 and never enters the basin, because entering it
costs seen accuracy.  The basin is real and invisible to deployment.

But accuracy is not the only legal criterion.  The interpretability panel
(exp 104) is computed entirely from SEEN classes: class-conditional width
(rms), Gaussianity (sw), the log-likelihood-fidelity columns (ll/llr/slope)
and separation (sep) require no novel example whatsoever.  If any of them
selects the basin, the accidental discovery becomes a deployable recipe.

This is ANALYSIS-ONLY: it re-reads the exp-113 sweep archive (tau x marginal
x seeds x {c10, c100}, full battery + panel per run) and asks, for each
candidate seen-only criterion, which tau it would have chosen and what
novelty performance followed.  Nothing is retrained.

LEGAL criteria (seen classes only):   rms->1, sw->1, slope->1, ece->0, sep max
ILLEGAL (used only to score, never to select): probe, mahaT, eucl, perevt, lid

Prediction: `sw` (Gaussianity) selects the basin on C100 while `rms` does NOT
-- exp 110 noted the basin winner sits at rms~0.50 while the marginal-free
arm reaches rms~1.0 with poor mahaT, so a width-seeking tuner is actively
misled.
Falsifier: no legal criterion beats accuracy's tau=0.1 choice -> the basin is
unreachable by any seen-only selection we can construct, and exp 115's verdict
hardens from "our tuner missed it" to "it is structurally invisible".

    python experiments/120_seen_only_tuner.py
    python experiments/120_seen_only_tuner.py --dataset cifar100 --marginal on
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import glob
import json

import numpy as np

# criterion -> (panel key, how to score a value: lower is better)
LEGAL = {
    "rms->1":   ("p_rms",   lambda v: abs(v - 1.0)),
    "sw->1":    ("p_sw",    lambda v: abs(v - 1.0)),
    "slope->1": ("p_slope", lambda v: abs(v - 1.0)),
    "ece->0":   ("p_ece",   lambda v: abs(v)),
    "sep max":  ("p_sep",   lambda v: -v),
}
SCORED = ["mahaT", "perevt", "probe"]


def load(paths):
    d = {}
    for p in paths:
        z = np.load(p, allow_pickle=True)
        for k in z.files:
            d[k] = z[k]
    return d


def series(done, ds, tau, marg, metric, seeds=3):
    vals = [float(done[f"{ds}_tau{tau}_{marg}_s{s}_{metric}"])
            for s in range(seeds)
            if f"{ds}_tau{tau}_{marg}_s{s}_{metric}" in done]
    return np.mean(vals) if vals else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default="logs/exp113")
    ap.add_argument("--dataset", nargs="+", default=["cifar10", "cifar100"])
    ap.add_argument("--marginal", nargs="+", default=["on", "off"])
    ap.add_argument("--taus", nargs="+", type=float,
                    default=[0.05, 0.1, 0.3, 1.0, 3.0])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--probe-tol", type=float, default=0.05,
                    help="oracle must keep probe within this of the best")
    ap.add_argument("--out", default="logs/exp120")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.archive, "*.npz")))
    if not paths:
        sys.exit(f"no exp-113 archive under {args.archive}; run exp 113 first")
    done = load(paths)
    os.makedirs(args.out, exist_ok=True)
    report = {}

    for ds in args.dataset:
        for marg in args.marginal:
            taus = [t for t in args.taus
                    if not np.isnan(series(done, ds, t, marg, "mahaT",
                                           args.seeds))]
            if not taus:
                continue
            print(f"\n######## {ds}  marginal={marg} ########")
            hdr = f"{'tau':>6}" + "".join(f"{k:>10}" for k in
                                          list(LEGAL) + SCORED)
            print(hdr); print("-" * len(hdr))
            table = {}
            for t in taus:
                row = {}
                for name, (key, _) in LEGAL.items():
                    row[name] = series(done, ds, t, marg, key, args.seeds)
                for m in SCORED:
                    row[m] = series(done, ds, t, marg, m, args.seeds)
                table[t] = row
                print(f"{t:>6}" + "".join(
                    f"{row[k]:>10.3f}" if np.isfinite(row[k]) else f"{'--':>10}"
                    for k in list(LEGAL) + SCORED))

            # Oracle: the tau a novelty-aware tuner would pick.  Constrained
            # to spaces that remain USABLE -- exp 113 showed high tau on C10
            # buys mahaT while collapsing the probe, and a space nobody would
            # deploy is not a meaningful target.
            best_probe = max(table[t]["probe"] for t in taus)
            usable = [t for t in taus
                      if table[t]["probe"] >= best_probe - args.probe_tol]
            oracle = max(usable, key=lambda t: table[t]["mahaT"])
            print(f"\n  ORACLE (illegal; best mahaT with probe within "
                  f"{args.probe_tol} of best): tau={oracle} "
                  f"-> mahaT {table[oracle]['mahaT']:.3f} "
                  f"perevt {table[oracle]['perevt']:.3f} "
                  f"probe {table[oracle]['probe']:.3f}")

            picks = {}
            for name, (key, score) in LEGAL.items():
                ok = [t for t in taus if np.isfinite(table[t][name])]
                # note: selection ranges over ALL taus -- a legal tuner has no
                # probe-retention oracle either
                if not ok:
                    continue
                pick = min(ok, key=lambda t: score(table[t][name]))
                picks[name] = dict(
                    tau=float(pick), mahaT=float(table[pick]["mahaT"]),
                    perevt=float(table[pick]["perevt"]),
                    probe=float(table[pick]["probe"]),
                    found_basin=bool(pick == oracle))
                flag = "  <-- FINDS THE ORACLE" if pick == oracle else ""
                print(f"  legal '{name:9s}' picks tau={pick:<5} "
                      f"-> mahaT {table[pick]['mahaT']:.3f} "
                      f"perevt {table[pick]['perevt']:.3f} "
                      f"probe {table[pick]['probe']:.3f}{flag}")
            report[f"{ds}_{marg}"] = dict(oracle=float(oracle), picks=picks)

    json.dump(report, open(os.path.join(args.out, "tuner.json"), "w"), indent=1)
    print("\n=== verdict ===")
    for cell, r in report.items():
        hits = [n for n, p in r["picks"].items() if p["found_basin"]]
        print(f"  {cell:16s} oracle tau={r['oracle']:<5} "
              f"legal criteria that find it: {hits or 'NONE'}")
    # The decisive cell is the one where the basin actually exists: C100 WITH
    # the marginal (exps 110/113).  Hits elsewhere are not evidence about it.
    key = "cifar100_on"
    if key in report:
        hits = [n for n, p in report[key]["picks"].items() if p["found_basin"]]
        print(f"\nDECISIVE CELL ({key}, where exps 110/113 put the basin):")
        print("  " + ("legal criteria reaching the oracle: " + ", ".join(hits)
                      + " -- the basin IS reachable by legal selection."
                      if hits else
                      "NONE.  Exp 115's verdict HARDENS: the basin is not "
                      "merely missed by an accuracy tuner, it is invisible to "
                      "every seen-only criterion we can construct."))
    print(f"wrote {args.out}/tuner.json")


if __name__ == "__main__":
    main()
