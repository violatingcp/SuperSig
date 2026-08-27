"""Exp 125 (residual half): aggregate exp-71 residual spaces and exp-72
residual discovery across SUPERSIG_HOLDOUT_DRAW for one galaxy10 base.

usage: python experiments/125_aggregate_residual_draws.py <dataset> <base>
Prints mean +- sd across draws per space, the PAIRED per-draw deltas of each
concat over its own parent, the exp-72 discovery block on the winner concat,
and the archived alphabetical draw as a reference column.
"""
import glob
import os
import re
import sys

import numpy as np

ds, base = sys.argv[1], sys.argv[2]
SPACES = ["supcon-ft_(parent)", "supcon-ft-res_(residual)", "supcon-ft-res_(concat)",
          "supcon-ft-res-nplm_(residual)", "supcon-ft-res-nplm_(concat)",
          "ss-ft_(parent)", "ss-ft-res_(residual)", "ss-ft-res_(concat)"]
METRICS = ["probe", "acc", "eucl", "mahaT", "perevt"]
PAIRS = [("supcon-ft-res_(concat)", "supcon-ft_(parent)"),
         ("supcon-ft-res-nplm_(concat)", "supcon-ft_(parent)"),
         ("ss-ft-res_(concat)", "ss-ft_(parent)")]

f71 = sorted(glob.glob(f"logs/exp71/results_{ds}_{base}_ft71_h1_d*.npz"))
draws = [int(re.search(r"_d(\d+)\.npz", f).group(1)) for f in f71]
per = {s: {m: [] for m in METRICS} for s in SPACES}
for f in f71:
    z = np.load(f, allow_pickle=True)
    for s in SPACES:
        for m in METRICS:
            k = f"{s}__{m}"
            per[s][m].append(float(z[k]) if k in z.files else np.nan)
ref = f"logs/exp71/results_{ds}_{base}_ft71.npz"
refz = np.load(ref, allow_pickle=True) if os.path.exists(ref) else None


def ms(v):
    v = np.asarray(v, float)
    return f"{np.nanmean(v):.3f}+-{np.nanstd(v):.3f}"


print(f"## {ds} residual spaces (exp 125/71; {base}, 100(+100)d, holdout 1 "
      f"[single-holdout], {len(draws)} draws d{draws}, 20 ep)\n")
print("| space | " + " | ".join(METRICS) + " | archived d{9} probe |")
print("|---|" + "---|" * (len(METRICS) + 1))
for s in SPACES:
    r = f"{float(refz[f'{s}__probe']):.3f}" if refz is not None and f"{s}__probe" in refz.files else "--"
    print(f"| {s.replace('_', ' ')} | " + " | ".join(ms(per[s][m]) for m in METRICS) + f" | {r} |")

print("\nPAIRED per-draw deltas (concat - its parent), draws " + str(draws))
for c, p in PAIRS:
    for m in ("probe", "eucl", "mahaT", "perevt"):
        dv = np.asarray(per[c][m]) - np.asarray(per[p][m])
        wins = int(np.nansum(dv > 0))
        print(f"  {c:28s} {m:7s} {np.nanmean(dv):+.3f}+-{np.nanstd(dv):.3f}  wins {wins}/{len(dv)}  "
              + " ".join(f"{x:+.3f}" for x in dv))

# exp 72: discovery on the winner concat
f72 = sorted(glob.glob(f"logs/exp72/residual_discovery_h1_d*_{ds}_{base}.npz"))
if f72:
    key = f"{ds}_{base}"
    acc = {}
    for f in f72:
        z = np.load(f, allow_pickle=True)
        for k in ("probe_pre", "probe_post", "eucl_pre", "eucl_post", "maha_pre", "maha_post"):
            acc.setdefault(k, []).append(float(z[f"{key}_{k}"]))
        pur = np.asarray(z[f"{key}_purity"], float)
        acc.setdefault("purity_r1", []).append(pur[0] if len(pur) else np.nan)
        acc.setdefault("purity_r2", []).append(pur[1] if len(pur) > 1 else np.nan)
        pe = np.asarray(z[f"{key}_post_perevent"], float)
        acc.setdefault("perevt_post@.05", []).append(pe[-1] if len(pe) else np.nan)
        spk = np.asarray(z[f"{key}_post_sparker"], float)
        acc.setdefault("spk_post@.05", []).append(spk[-1] if len(spk) else np.nan)
    print(f"\nexp 72 discovery on the winner concat ({len(f72)} draws):")
    for k, v in acc.items():
        print(f"  {k:16s} {ms(v)}   per-draw " + " ".join(f"{x:.3f}" for x in v))
