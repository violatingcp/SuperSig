"""
Experiment 99: discovery reach -- the injected signal fraction f95 at which
SparKer power crosses 0.95, for every archived configuration.

Motivation: the campaign reports power on a coarse fraction grid, which makes
"how strong a signal does this space need?" hard to read off.  f95 inverts the
power curve into a single sensitivity number in the units a search actually
cares about (signal fraction), so spaces can be ranked by REACH rather than by
power at an arbitrary fraction.  Lower f95 = more sensitive.

Method: power(f) is estimated from n_sig=50 toys per fraction, so it is
quantised in steps of 0.02 and carries a Clopper-Pearson uncertainty of
roughly +-0.03 near 0.95.  We take the FIRST upward crossing of 0.95 and
interpolate linearly in log f between the bracketing grid points (power curves
are sigmoidal in log f, so log-interpolation is the milder assumption).

Honest reporting rules, all enforced here:
  * if max(power) < 0.95 the curve never crosses inside the measured grid and
    we report ">fmax" -- NEVER an extrapolation;
  * if the crossing bracket spans the top grid interval the estimate depends
    on one noisy endpoint, and is flagged;
  * non-monotone curves (power dips after crossing) are flagged;
  * f95 below the smallest measured fraction is reported as "<fmin".

Evaluation-only: reads archived npz, trains nothing.

    python experiments/99_discovery_reach.py
    python experiments/99_discovery_reach.py --stat sparker --latex
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import glob
import json
import numpy as np

TARGET = 0.95
N_SIG_TOYS = 50            # power quantum = 1/50 = 0.02


def clopper_pearson(k, n, cl=0.68):
    from scipy.stats import beta
    lo = beta.ppf((1 - cl) / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - (1 - cl) / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def f95(fractions, power, target=TARGET):
    """First upward crossing of `target`, interpolated in log f.

    Returns (value, flag) where value is None when the curve never crosses.
    flag in {"ok", "top-bracket", "non-monotone", "below-grid", "never"}.
    """
    fr = np.asarray(fractions, float)
    pw = np.asarray(power, float)
    if fr.size != pw.size or fr.size < 2:
        return None, "bad-shape"
    order = np.argsort(fr)
    fr, pw = fr[order], pw[order]

    if pw.max() < target:
        return None, "never"
    if pw[0] >= target:
        return fr[0], "below-grid"

    i = int(np.argmax(pw >= target))          # first index at/above target
    f_lo, f_hi = fr[i - 1], fr[i]
    p_lo, p_hi = pw[i - 1], pw[i]
    if p_hi <= p_lo:
        return f_hi, "non-monotone"
    t = (target - p_lo) / (p_hi - p_lo)
    val = float(np.exp(np.log(f_lo) + t * (np.log(f_hi) - np.log(f_lo))))

    flag = "ok"
    if i == fr.size - 1:
        flag = "top-bracket"                  # relies on the last, noisiest point
    if np.any(pw[i:] < target - 1e-9):
        flag = "non-monotone"
    return val, flag


def band(fractions, power, target=TARGET):
    """Crossing range implied by +-1 Clopper-Pearson sigma on each power point.

    Gives an honest sense of how well f95 is pinned by 50 toys.
    """
    pw = np.asarray(power, float)
    k = np.round(pw * N_SIG_TOYS).astype(int)
    lo_pw, hi_pw = [], []
    for kk in k:
        lo, hi = clopper_pearson(int(kk), N_SIG_TOYS)
        lo_pw.append(lo); hi_pw.append(hi)
    # optimistic curve (upper band) crosses earliest; pessimistic crosses latest
    f_opt, _ = f95(fractions, hi_pw, target)
    f_pes, _ = f95(fractions, lo_pw, target)
    return f_opt, f_pes


def fmt(val, flag, fmax):
    if val is None:
        return f">{fmax:g}" if flag == "never" else "--"
    s = f"{val:.3f}"
    if flag == "top-bracket":
        s += "*"
    elif flag == "non-monotone":
        s += "~"
    elif flag == "below-grid":
        s = f"<{val:g}"
    return s


def harvest(stat="sparker"):
    """Collect every (file, arm, phase) power series for `stat`."""
    rows = []
    for fn in sorted(glob.glob("logs/exp*/*.npz")):
        try:
            d = np.load(fn, allow_pickle=True)
        except Exception:
            continue
        keys = [k for k in d.files if stat in k.lower()]
        if not keys:
            continue
        for k in keys:
            phase = "post" if "post" in k else ("pre" if "pre" in k else "?")
            # locate the matching fraction grid
            for cand in (f"{phase}_fractions", "fractions", "pre_fractions"):
                if cand in d.files:
                    fr = np.asarray(d[cand], float)
                    break
            else:
                continue
            pw = np.asarray(d[k], float)
            if pw.shape != fr.shape:
                # exp-70-style: post grid is shorter than pre
                alt = "post_fractions" if phase == "post" else "pre_fractions"
                if alt in d.files and np.asarray(d[alt]).shape == pw.shape:
                    fr = np.asarray(d[alt], float)
                else:
                    continue
            arm = (k.replace(f"{stat}_", "").replace(f"_{phase}", "")
                   .replace(f"{phase}_", ""))
            exp = fn.split("/")[1]
            cell = os.path.basename(fn).replace("results_", "").replace(".npz", "")
            val, flag = f95(fr, pw)
            f_opt, f_pes = band(fr, pw)
            rows.append(dict(exp=exp, cell=cell, arm=arm, phase=phase,
                             fractions=fr.tolist(), power=pw.tolist(),
                             f95=val, flag=flag, fmax=float(fr.max()),
                             pmax=float(pw.max()),
                             f95_opt=f_opt, f95_pes=f_pes))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stat", default="sparker")
    ap.add_argument("--exps", default="", help="comma list, e.g. exp80,exp70")
    ap.add_argument("--out", default="logs/exp99")
    ap.add_argument("--latex", action="store_true")
    args = ap.parse_args()

    rows = harvest(args.stat)
    if args.exps:
        keep = set(args.exps.split(","))
        rows = [r for r in rows if r["exp"] in keep]

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"f95_{args.stat}.json"), "w") as fh:
        json.dump(rows, fh, indent=1)

    n_cross = sum(r["f95"] is not None for r in rows)
    print(f"{len(rows)} series, {n_cross} cross {TARGET} inside the measured grid "
          f"({100*n_cross/max(len(rows),1):.0f}%)\n")

    hdr = f"{'exp':7s}{'cell':26s}{'arm':26s}{'ph':5s}{'f95':>9s}{'pmax':>7s}{'flag':>14s}"
    print(hdr); print("-" * len(hdr))
    for r in sorted(rows, key=lambda r: (r["f95"] is None, r["f95"] or 9e9)):
        print(f"{r['exp']:7s}{r['cell'][:25]:26s}{r['arm'][:25]:26s}{r['phase']:5s}"
              f"{fmt(r['f95'], r['flag'], r['fmax']):>9s}{r['pmax']:>7.2f}"
              f"{r['flag']:>14s}")

    print("\n* = crossing sits in the top grid interval (one noisy endpoint)")
    print("~ = non-monotone power curve")
    print(f"> = never reaches {TARGET} within the measured grid")


if __name__ == "__main__":
    main()
