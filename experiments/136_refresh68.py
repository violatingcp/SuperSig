"""Refresh the `discovery68` block of every logs/exp136/master_<ds>[_hN].json
from the CURRENT logs/exp68/scratch_discovery_<ds>[_hN].npz (Block S5 re-pass:
per-injected-fraction post metrics).  Same merge as 136_scratch_master.

    python experiments/136_refresh68.py [cifar10 cifar100]
"""
import os, sys, re, glob, json, importlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
m136 = importlib.import_module("136_scratch_master")
STATS = m136.STATS

for ds in (sys.argv[1:] or ["cifar10", "cifar100"]):
    for mj in sorted(glob.glob(f"logs/exp136/master_{ds}*.json")):
        m = re.fullmatch(rf"logs/exp136/master_{ds}(_h(\d+))?\.json", mj)
        if not m:
            continue
        tag, hold = (m.group(1) or ""), int(m.group(2) or 4)
        npz = f"logs/exp68/scratch_discovery_{ds}{tag}.npz"
        if not os.path.exists(npz):
            print(f"{mj}: no npz, skipped"); continue
        d68 = np.load(npz, allow_pickle=True)
        if not any(k.startswith("postf_") for k in d68.files):
            print(f"{mj}: npz predates postf_, skipped"); continue
        res = json.load(open(mj)); n = 0
        for arm, r in res.items():
            if f"probe_{arm}" not in d68.files:
                continue
            pp = d68[f"probe_{arm}"]
            r["discovery68"] = dict(
                probe_pre=float(pp[0]), probe_post=float(pp[1]),
                purity=[float(x) for x in d68[f"purity_{arm}"]] if f"purity_{arm}" in d68.files else None,
                post={k: float(d68[f"post_{k}_{arm}"]) for k in ("acc", "eucl", "maha_tied", "maha_pc", "lid")
                      if f"post_{k}_{arm}" in d68.files},
                post_power={s: [float(x) for x in d68[f"{s}_{arm}_post"]] for s in STATS
                            if f"{s}_{arm}_post" in d68.files},
                fractions=[float(x) for x in d68["fractions"]],
                postf={k: [float(x) for x in d68[f"postf_{k}_{arm}"]]
                       for k in ("probe", "eucl", "mahaT", "mahaPC") if f"postf_{k}_{arm}" in d68.files},
                purity_inj=m136.exp68_purity_inj(ds, hold).get(arm, {}))
            n += 1
        json.dump(res, open(mj, "w"), indent=1, default=float)
        print(f"{mj}: refreshed {n} arms from {npz}")
