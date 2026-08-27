"""Exp 125: aggregate exp-70 single-holdout results across SUPERSIG_HOLDOUT_DRAW.

usage: python experiments/125_aggregate_draws.py <dataset> <base> [draws...]
Prints mean +- sd across draws per arm, per-draw probe/mahaT/purity spread,
and the archived alphabetical draw as a reference column.
"""
import re, sys, glob, os
import numpy as np

ARMS = ["simclr-ft", "sigreg-ssl-ft", "nplm-bil-ft", "supcon-ft", "ss-ft",
        "nplm-sup-ft"]
ds, base = sys.argv[1], sys.argv[2]
draws = [int(x) for x in sys.argv[3:]] or None
files = sorted(glob.glob(f"logs/exp70/results_{ds}_{base}_ft70_h1_d*.npz"))
if draws is not None:
    files = [f for f in files if int(re.search(r"_d(\d+)\.npz", f).group(1)) in draws]
dlist = [int(re.search(r"_d(\d+)\.npz", f).group(1)) for f in files]


def purity_from_log(d):
    fn = f"logs/exp70_{ds}_{base}_h1_d{d}.log"
    if not os.path.exists(fn):
        fn = f"logs/exp70_g10_{base}_h1_d{d}.log"
    out, arm = {}, None
    for line in open(fn):
        if line.startswith(("-----", "=====")):
            arm = None                      # any other block ends the arm
        m = re.match(r"----- natural discovery: (\S+) -----", line)
        if m:
            arm = m.group(1); out[arm] = {}
        m = re.match(r"\s+round (\d): pool=\d+ purity=([\d.]+)", line)
        if m and arm:
            out[arm][int(m.group(1))] = float(m.group(2))
    return out


def fidx(fr, f):
    return int(np.argmin(np.abs(np.asarray(fr) - f)))


rows = {a: {} for a in ARMS}
for f, d in zip(files, dlist):
    z = np.load(f, allow_pickle=True)
    pur = purity_from_log(d)
    pre, post = z["pre_fractions"], z["post_fractions"]
    for a in ARMS:
        r = rows[a]
        def add(k, v): r.setdefault(k, []).append(float(v))
        add("probe", z[f"probe_{a}"]); add("post_probe", z[f"post_probe_{a}"])
        add("acc", z[f"acc_{a}"]); add("eucl", z[f"eucl_{a}"])
        add("mahaT", z[f"mahaT_{a}"]); add("post_mahaT", z[f"post_mahaT_{a}"])
        add("perevt", z[f"perevent_{a}_pre"][0])
        add("post_perevt", z[f"perevent_{a}_post"][fidx(post, 0.02)])
        add("spk05", z[f"sparker_{a}_pre"][fidx(pre, 0.05)])
        add("post_spk05", z[f"sparker_{a}_post"][fidx(post, 0.05)])
        add("mmd05", z[f"mmd_{a}_pre"][fidx(pre, 0.05)])
        add("post_mmd05", z[f"mmd_{a}_post"][fidx(post, 0.05)])
        add("pur1", pur.get(a, {}).get(1, np.nan))
        add("pur2", pur.get(a, {}).get(2, np.nan))

ref = f"logs/exp70/results_{ds}_{base}_ft70.npz"
refz = np.load(ref, allow_pickle=True) if os.path.exists(ref) else None

print(f"## {ds} (exp 125; {base}, 100d, holdout 1 [single-holdout], "
      f"{len(files)} draws d{dlist}, 20 ep)\n")
COLS = [("probe", "probe"), ("post_probe", "probe post"), ("acc", "acc"),
        ("eucl", "eucl"), ("mahaT", "mahaT"), ("post_mahaT", "mahaT post"),
        ("perevt", "perevt"), ("post_perevt", "perevt post@.02"),
        ("pur1", "purity r1"), ("pur2", "purity r2"),
        ("spk05", "SpK@.05"), ("post_spk05", "SpK@.05 post"),
        ("mmd05", "MMD@.05"), ("post_mmd05", "MMD@.05 post")]
hdr = "| arm | " + " | ".join(c for _, c in COLS) + " |"
print(hdr); print("|" + "---|" * (len(COLS) + 1))
for a in ARMS:
    cells = []
    for k, _ in COLS:
        v = np.asarray(rows[a][k])
        cells.append(f"{np.nanmean(v):.3f}+-{np.nanstd(v):.3f}")
    print(f"| {a} | " + " | ".join(cells) + " |")

print("\nper-draw spread (probe / mahaT / purity r1), draws " + str(dlist))
for a in ARMS:
    p = rows[a]["probe"]; m = rows[a]["mahaT"]; u = rows[a]["pur1"]
    rp = f"{np.min(p):.3f}-{np.max(p):.3f}"; rm = f"{np.min(m):.3f}-{np.max(m):.3f}"
    ru = f"{np.nanmin(u):.3f}-{np.nanmax(u):.3f}"
    refs = ""
    if refz is not None:
        refs = f"   archived d{{9}}: probe {float(refz[f'probe_{a}']):.3f} mahaT {float(refz[f'mahaT_{a}']):.3f}"
    print(f"  {a:14s} probe {rp}  mahaT {rm}  purity {ru}{refs}")
