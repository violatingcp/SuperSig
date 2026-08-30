"""Exp 136: aggregate the scratch master battery across holdouts (draws).

usage: python experiments/136_aggregate.py <dataset>
Prints, per arm, mean +- sd across every logs/exp136/master_<ds>*.json
(the archived holdout 4 plus the _h<N> draws) for the pre battery, the
frozen pools, the legal cut, and the exp-68 discovery columns.
"""
import glob
import json
import re
import sys

import numpy as np

ds = sys.argv[1]
files = sorted(f for f in glob.glob(f"logs/exp136/master_{ds}*.json")
               if re.fullmatch(rf"logs/exp136/master_{ds}(_h\d+)?\.json", f))   # cifar10 must not match cifar100
holds = [int((re.search(r"_h(\d+)\.json", f) or [None, 4])[1]) for f in files]
ARMS = ["supcon", "ssig", "nplmsd", "nplmcw", "supsig", "simclr", "visreg", "nplm"]
COLS = [("probe", lambda r: r["pre"]["probe"]), ("top1", lambda r: r["pre"]["top1"]),
        ("acc", lambda r: r["pre"]["acc"]), ("eucl", lambda r: r["pre"]["eucl"]),
        ("mahaT", lambda r: r["pre"]["mahaT"]), ("lid", lambda r: r["pre"]["lid"]),
        ("perevt", lambda r: r["pre"]["perevt"]),
        ("spk@.02", lambda r: r["pre_power"]["sparker"][r["pre_power"]["fractions"].index(0.02)]),
        ("mmd@.02", lambda r: r["pre_power"]["mmd"][r["pre_power"]["fractions"].index(0.02)]),
        ("frz-dist r1", lambda r: r["frozen"]["dist|frozen"]["purity"][0]),
        ("frz-np r1", lambda r: r["frozen"]["np|frozen"]["purity"][0]),
        ("frz-np r2", lambda r: r["frozen"]["np|frozen"]["purity"][1]),
        ("np bn-adapt r2", lambda r: r["frozen"]["np|bn-adapt"]["purity"][1]),
        ("legal pur", lambda r: r["legal_cut"]["purity"] if r["legal_cut"]["ok"] else np.nan),
        ("68 probe post", lambda r: r["discovery68"]["probe_post"]),
        ("68 pur r1", lambda r: r["discovery68"]["purity"][0]),
        ("68 post eucl", lambda r: r["discovery68"]["post"]["eucl"])]

print(f"## {ds} from-scratch master (exp 136; 100-D, pretrain=None, holdouts {holds}, "
      f"n={len(files)} [single-holdout regime])\n")
print("| arm | " + " | ".join(c for c, _ in COLS) + " |")
print("|---|" + "---|" * len(COLS))
data = [json.load(open(f)) for f in files]
for arm in ARMS:
    cells = []
    for name, fn in COLS:
        vals = []
        for d in data:
            try:
                vals.append(float(fn(d[arm])))
            except (KeyError, TypeError, ValueError, IndexError):
                pass
        cells.append(f"{np.nanmean(vals):.3f}+-{np.nanstd(vals):.3f} (n={len(vals)})" if vals else "--")
    print(f"| {arm} | " + " | ".join(cells) + " |")
