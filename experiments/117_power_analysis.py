"""
Experiment 117 (IMPROVEMENT_TESTS.md #117): are our comparisons adequately
powered?

Protocol hygiene the campaign has never done.  Many claims rest on 3-5 seeds,
and the "gaps below 0.02 need 3-5 seeds" rule was adopted by folklore (a
single-seed spread of ~0.017 observed once, exp 52) rather than by calculation.
This estimates the per-cell seed standard deviation from every multi-seed
archive we have, then reports the MINIMUM DETECTABLE EFFECT at n = 3, 5, 10
seeds, so each comparative claim in the paper can be checked against the
effect size it would need.

Analysis-only: reads archived npz, trains nothing.

MDE for a PAIRED comparison at two-sided alpha and power 1-beta is
    MDE = (z_{1-alpha/2} + z_{1-beta}) * sd_paired / sqrt(n)
and for an UNPAIRED comparison of two arms with equal sd,
    MDE = (z_{1-alpha/2} + z_{1-beta}) * sd * sqrt(2/n).
The campaign's protocol is paired (identical seeds across arms), so the paired
form is the relevant one and the unpaired form is reported as the penalty for
not pairing.

Prediction: several sub-0.02 claims are underpowered at 3 seeds, and the
transfer cells need more seeds than CIFAR (their spreads are wider).

    python experiments/117_power_analysis.py
    python experiments/117_power_analysis.py --metric probe --alpha 0.05
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import collections
import glob
import json
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Z = {0.80: 0.8416, 0.90: 1.2816}          # z_{1-beta}
ZA = {0.05: 1.9600, 0.10: 1.6449}         # z_{1-alpha/2}

# key patterns that encode a seed index, e.g. "..._s0_probe" or "..._s0"
SEED_RE = re.compile(r"^(?P<stem>.+?)_s(?P<seed>\d+)(?P<tail>_.*)?$")


def harvest(metric):
    """{(file, stem, tail): [values across seeds]} for a metric substring."""
    groups = collections.defaultdict(dict)
    for fn in sorted(glob.glob(os.path.join(ROOT, "logs", "exp*", "*.npz"))):
        try:
            d = np.load(fn, allow_pickle=True)
        except Exception:
            continue
        for k in d.files:
            if metric not in k:
                continue
            m = SEED_RE.match(k)
            if not m:
                continue
            try:
                v = float(np.asarray(d[k]).ravel()[0])
            except Exception:
                continue
            if not np.isfinite(v):
                continue
            cell = (os.path.relpath(fn, ROOT), m.group("stem"),
                    m.group("tail") or "")
            groups[cell][int(m.group("seed"))] = v
    return {c: [v for _, v in sorted(s.items())]
            for c, s in groups.items() if len(s) >= 2}


def mde(sd, n, alpha=0.05, power=0.80, paired=True):
    f = 1.0 if paired else np.sqrt(2.0)
    return (ZA[alpha] + Z[power]) * sd * f / np.sqrt(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="probe")
    ap.add_argument("--alpha", type=float, default=0.05, choices=[0.05, 0.10])
    ap.add_argument("--power", type=float, default=0.80, choices=[0.80, 0.90])
    ap.add_argument("--out", default="logs/exp117")
    args = ap.parse_args()

    groups = harvest(args.metric)
    if not groups:
        print(f"no multi-seed series found for metric '{args.metric}'")
        return

    rows = []
    for (fn, stem, tail), vals in groups.items():
        a = np.asarray(vals, float)
        rows.append(dict(file=fn, stem=stem, tail=tail, n=len(a),
                         mean=float(a.mean()), sd=float(a.std(ddof=1))))
    rows.sort(key=lambda r: -r["sd"])

    sds = np.array([r["sd"] for r in rows])
    # datasets are encoded in the file path; split CIFAR vs transfer
    def fam(f):
        return "cifar" if ("cifar" in f or "c10" in f or "c100" in f) else "transfer"
    byfam = collections.defaultdict(list)
    for r in rows:
        byfam[fam(r["file"])].append(r["sd"])

    print(f"metric='{args.metric}'  series with >=2 seeds: {len(rows)}")
    print(f"seed sd: median {np.median(sds):.4f}  p90 {np.percentile(sds,90):.4f}"
          f"  max {sds.max():.4f}\n")
    for f, v in sorted(byfam.items()):
        v = np.asarray(v)
        print(f"  {f:9s} n={len(v):3d}  median sd {np.median(v):.4f}  "
              f"p90 {np.percentile(v,90):.4f}")

    print(f"\nMinimum detectable effect (alpha={args.alpha}, "
          f"power={args.power:.0%}):")
    print(f"  {'basis':22s}{'n=3':>9s}{'n=5':>9s}{'n=10':>9s}")
    for label, sd in [("median sd (paired)", float(np.median(sds))),
                      ("p90 sd (paired)", float(np.percentile(sds, 90))),
                      ("median sd (unpaired)", float(np.median(sds)))]:
        paired = "unpaired" not in label
        vals = [mde(sd, n, args.alpha, args.power, paired) for n in (3, 5, 10)]
        print(f"  {label:22s}" + "".join(f"{v:9.4f}" for v in vals))

    print("\nWidest-spread series (these bound what their cells can claim):")
    for r in rows[:10]:
        print(f"  sd={r['sd']:.4f} n={r['n']}  {r['stem'][:38]:39s} "
              f"{os.path.basename(r['file'])}")

    os.makedirs(args.out, exist_ok=True)
    json.dump(dict(metric=args.metric, alpha=args.alpha, power=args.power,
                   rows=rows,
                   mde={f"n{n}": mde(float(np.median(sds)), n, args.alpha,
                                     args.power) for n in (3, 5, 10)}),
              open(os.path.join(args.out, f"power_{args.metric}.json"), "w"),
              indent=1)
    print(f"\nwrote {args.out}/power_{args.metric}.json")


if __name__ == "__main__":
    main()
