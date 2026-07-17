"""Regenerate batch-2 figures with holdout-tagged names (the runs' own
outputs collided between k=1 and k=2). Parses the four POWER SUMMARY blocks
of each log and redraws per-run figures + npz."""
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOGDIR = "/home/pharris/sigreg/SuperSig/logs/exp33"
PLOTS = "/home/pharris/sigreg/SuperSig/plots"
COLORS = {"sup->res": "#2a78d6", "ssl->supres": "#1baf7a", "joint": "#eda100",
          "sup": "#008300", "supcon": "#4a3aa7", "supcon+simclr": "#e34948"}
ARMS = list(COLORS)
STATS = ["PEREVENT", "SPARKER", "MAHA", "MMD"]
RUNS = [
    ("exp33_cifar10_16p16.log", "cifar10", "k1",
     [0.001, 0.003, 0.01, 0.02, 0.03, 0.1]),
    ("exp33_cifar10_16p16_k2.log", "cifar10", "k2",
     [0.001, 0.003, 0.01, 0.02, 0.03, 0.1]),
    ("exp33_cifar100_16p16.log", "cifar100", "k1",
     [0.001, 0.003, 0.01, 0.02, 0.05]),
    ("exp33_cifar100_16p16_k2.log", "cifar100", "k2",
     [0.001, 0.003, 0.01, 0.02, 0.05]),
]


def parse(path, stat):
    text = open(path).read()
    header = f"===== EXP33 {stat} POWER SUMMARY"
    block = text[text.index(header):]
    out = {}
    for line in block.splitlines()[2:]:
        m = re.match(r"\s+(\S+)\s+(pre|post)\s+([\d.\s]+)$", line)
        if m:
            out[(m.group(1), m.group(2))] = [float(x)
                                             for x in m.group(3).split()]
        elif "saved" in line or "SUMMARY" in line:
            break
    return out


for fname, ds, ktag, fracs in RUNS:
    npz = {"fractions": np.array(fracs)}
    for stat in STATS:
        data = parse(f"{LOGDIR}/{fname}", stat)
        plt.figure(figsize=(8, 6.5))
        for arm in ARMS:
            c = COLORS[arm]
            plt.plot(fracs, data[(arm, "pre")], "--o", color=c, lw=1.4, ms=5,
                     alpha=0.75, label=f"{arm} pre")
            plt.plot(fracs, data[(arm, "post")], "-o", color=c, lw=2, ms=6,
                     label=f"{arm} post")
            npz[f"{stat.lower()}_{arm}_pre"] = np.array(data[(arm, "pre")])
            npz[f"{stat.lower()}_{arm}_post"] = np.array(data[(arm, "post")])
        plt.xscale("log")
        plt.axhline(0.05, color="gray", lw=1, ls=":")
        plt.xlabel("injected anomaly fraction")
        plt.ylabel("power at alpha=0.05")
        note = (" (train-side clamp above f~0.01/anomaly-class)"
                if ds == "cifar100" else "")
        plt.title(f"exp33 [{ds}, {ktag}] 32-scan: {stat.lower()} power "
                  f"vs fraction{note}")
        plt.grid(alpha=0.25, which="both")
        plt.legend(loc="upper left", fontsize=8, ncol=2)
        plt.tight_layout()
        out = f"{PLOTS}/exp33_{stat.lower()}_power_{ds}_16p16_{ktag}.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print("saved", out)
    np.savez(f"{LOGDIR}/power_data_{ds}_16p16_{ktag}.npz", **npz)
    print(f"saved {LOGDIR}/power_data_{ds}_16p16_{ktag}.npz")
