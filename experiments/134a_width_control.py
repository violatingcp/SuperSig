"""
Experiment 134a: is the residual concat's gain a construction, or just width?

Question (A) of exp 134.  The concat [parent ; child] is TWO full networks --
it doubles width and parameter count -- and the flagship number (cars/VISReg
+0.148 +- 0.004 over its parent, exp 75) has never been compared against a
same-width NON-residual control.  Exp 85 already found a capacity effect on
this cell (width-matched single residual ties the 3-way concat).

CONTROL.  The parent's own objective (supcon-ft) fine-tuned with a 200-D head
instead of 100-D: same trunk, same corpus, same epochs, same recipe, twice the
head width -- and NO residual.  Produced by exp 70 itself:

    python experiments/70_cars_ft_suite.py --dataset cars --base visreg \
        --emb-dim 200 --arms supcon-ft --skip-discovery --seed S

which now writes artifacts tagged `_e200` (exp70.seed_sfx) instead of
overwriting the archived 100-D parents.  This script only COMPARES, across
whatever seeds exist:

    parent 100-D            results_{ds}_{base}_ft70[_sS].npz     probe_supcon-ft
    concat 100+100 (res-nplm) results_{ds}_{base}_ft71[_sS].npz   supcon-ft-res-nplm_(concat)__probe
    control 200-D           results_{ds}_{base}_ft70[_sS]_e200.npz probe_supcon-ft

and reports probe / eucl / mahaT for each, paired by seed, with the exp-132
TIE guard (a gap under max(0.017, sd_a + sd_b) is a tie).

PREDICTION.  The 200-D control recovers only a small part of the residual
gain on cars/VISReg (exp 85: +0.05 of the +0.15 was capacity) -- the concat
beats it by more than the floor, so the construction is real.

FALSIFIER.  control >= concat on the probe -> the flagship gain is width, the
residual is a way of spending parameters, and the paper drops the residual as
a construction (it keeps the CHILD-alone detector result, which is a
half-width claim and unaffected).

    python experiments/134a_width_control.py --dataset cars --base visreg
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from supersig.holdouts import run_tag
import argparse
import glob
import json
import re

import numpy as np

FLOOR = 0.017          # exp 52 seed spread; exp 132's TIE guard


def load(pattern, key):
    """{seed: value} over every seed file matching the pattern."""
    out = {}
    for f in sorted(glob.glob(pattern)):
        m = re.search(r"_s(\d+)", os.path.basename(f))
        s = int(m.group(1)) if m else 0
        d = np.load(f, allow_pickle=True)
        if key in d.files:
            out[s] = float(d[key])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cars")
    ap.add_argument("--base", default="visreg")
    ap.add_argument("--parent", default="supcon-ft")
    ap.add_argument("--concat", default="supcon-ft-res-nplm",
                    help="exp-71 concat key (supcon-ft-res | supcon-ft-res-nplm)")
    ap.add_argument("--dim", type=int, default=200)
    ap.add_argument("--out", default="logs/exp134")
    args = ap.parse_args()
    ds, base, tag = args.dataset, args.base, run_tag()

    KEYS = {"probe": (f"probe_{args.parent}", f"{args.concat}_(concat)__probe"),
            "eucl": (f"eucl_{args.parent}", f"{args.concat}_(concat)__eucl"),
            "mahaT": (f"mahaT_{args.parent}", f"{args.concat}_(concat)__mahaT")}

    def files70(dim):
        """exp-70 result files of this cell at one head width, keyed by seed."""
        out = {}
        for f in sorted(glob.glob(f"logs/exp70/results_{ds}_{base}_ft70{tag}*.npz")):
            b = os.path.basename(f)
            if "_quick" in b:
                continue
            md = re.search(r"_e(\d+)", b)
            if (int(md.group(1)) if md else 100) != dim:
                continue
            ms = re.search(r"_s(\d+)", b)
            out[int(ms.group(1)) if ms else 0] = f
        return out

    def files71():
        out = {}
        for f in sorted(glob.glob(f"logs/exp71/results_{ds}_{base}_ft71{tag}*.npz")):
            b = os.path.basename(f)
            if "_quick" in b or re.search(r"_e\d+", b):
                continue
            ms = re.search(r"_s(\d+)", b)
            out[int(ms.group(1)) if ms else 0] = f
        return out

    def read(fmap, key):
        vals = {}
        for s, f in fmap.items():
            d = np.load(f, allow_pickle=True)
            if key in d.files:
                vals[s] = float(d[key])
        return vals

    f_par, f_ctl, f_cat = files70(100), files70(args.dim), files71()
    rows = {m: dict(parent=read(f_par, k70), concat=read(f_cat, k71),
                    control=read(f_ctl, k70))
            for m, (k70, k71) in KEYS.items()}

    print(f"exp134a [{ds}:{base}{tag}] width-matched control "
          f"({args.parent} @ {args.dim}-D) vs residual concat "
          f"({args.concat}, 100+100)\n")
    print(f"  {'metric':<7}{'arm':<10}{'seeds':>6}{'mean+-sd':>16}   per-seed")
    verdicts = {}
    for metric, arms in rows.items():
        for arm, vals in arms.items():
            v = np.array(list(vals.values()))
            if len(v):
                print(f"  {metric:<7}{arm:<10}{len(v):>6}"
                      f"{v.mean():>10.4f}+-{v.std():<5.4f}   "
                      + " ".join(f"s{s}={x:.4f}" for s, x in sorted(vals.items())))
            else:
                print(f"  {metric:<7}{arm:<10}{0:>6}{'--':>16}")
        c, k = arms["concat"], arms["control"]
        common = sorted(set(c) & set(k))
        if common:
            gaps = np.array([c[s] - k[s] for s in common])
            thr = max(FLOOR, np.std(list(c.values())) + np.std(list(k.values())))
            w = ("concat" if gaps.mean() > thr else
                 "control" if gaps.mean() < -thr else "TIE")
            verdicts[metric] = dict(gap=float(gaps.mean()), sd=float(gaps.std()),
                                    thresh=float(thr), n=len(common), winner=w)
            print(f"  -> {metric}: concat - control = {gaps.mean():+.4f}"
                  f"+-{gaps.std():.4f} over {len(common)} paired seeds "
                  f"(thresh {thr:.3f}) => {w}")
    if "probe" in verdicts:
        w = verdicts["probe"]["winner"]
        print("\n  VERDICT: " + {
            "concat": "PREDICTION HOLDS -- the residual gain exceeds a same-width control.",
            "control": "FALSIFIER FIRES -- a same-width non-residual head matches/beats the concat; the gain was width.",
            "TIE": "TIE -- the concat and the width control are inside the noise floor; the residual gain is not distinguishable from capacity."}[w])
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"width_control_{ds}-{base}{tag}.json"), "w") as fh:
        json.dump(dict(rows=rows, verdicts=verdicts), fh, indent=1, default=float)


if __name__ == "__main__":
    main()
