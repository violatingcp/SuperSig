"""Trend vs embedding dimension for the exp-28 spaces (parses run logs)."""
import os
import re
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRATCH = os.path.dirname(os.path.abspath(__file__))
OUT = "/home/pharris/sigreg/SuperSig/plots/exp28_dim_trend.png"

LOGS = {  # (dataset, dim) -> log file
    ("cifar10", 4): "exp28_cifar.log",
    ("mnist", 4): "exp28_mnist.log",
    ("cifar10", 8): "exp28_cifar_8d.log",
    ("mnist", 8): "exp28_mnist_8d.log",
    ("cifar10", 16): "exp28_cifar_16d.log",
    ("mnist", 16): "exp28_mnist_16d.log",
    ("cifar100", 8): "exp28_cifar100_8d.log",
    ("cifar100", 16): "exp28_cifar100_16d.log",
    ("cifar100", 32): "exp28_cifar100_32d.log",
    ("cifar10", 64): "exp28_cifar_64d.log",
    ("mnist", 64): "exp28_mnist_64d.log",
    ("cifar100", 64): "exp28_cifar100_64d.log",
    ("cifar100", 100): "exp28_cifar100_100d.log",
}

# fixed categorical assignment (validated reference palette, light mode)
SPACES = ["sup", "ssl", "concat", "res", "supres", "res-cw"]
COLORS = {"sup": "#2a78d6", "ssl": "#1baf7a", "concat": "#eda100",
          "res": "#008300", "supres": "#4a3aa7", "res-cw": "#e34948"}

# classwise-residual reruns: their [res] rows are the "res-cw" series
RESC_LOGS = {
    ("cifar10", 4): "exp28resc_cifar_4d.log",
    ("cifar10", 8): "exp28resc_cifar_8d.log",
    ("cifar10", 16): "exp28resc_cifar_16d.log",
    ("cifar10", 64): "exp28resc_cifar_64d.log",
    ("mnist", 4): "exp28resc_mnist_4d.log",
    ("mnist", 8): "exp28resc_mnist_8d.log",
    ("mnist", 16): "exp28resc_mnist_16d.log",
    ("mnist", 64): "exp28resc_mnist_64d.log",
    ("cifar100", 8): "exp28resc_cifar100_8d.log",
    ("cifar100", 16): "exp28resc_cifar100_16d.log",
    ("cifar100", 32): "exp28resc_cifar100_32d.log",
    ("cifar100", 64): "exp28resc_cifar100_64d.log",
    ("cifar100", 100): "exp28resc_cifar100_100d.log",
}

PRE_RE = re.compile(r"\[(\w+)\s*\] pre: acc=([\d.]+) novelty-AUC=([\d.]+)")
RD_RE = re.compile(r"round (\d): purity=[\d.nan]+ anchors=\d+\s+"
                   r"margin=([\d.]+)\s+mean-anchor=([\d.]+)")


def parse(path):
    """{space: {acc, nov, margin2, anchor2}} from a log's SUMMARY block."""
    text = open(path).read()
    block = text.split("EXP28 SUMMARY")[-1]
    out, cur = {}, None
    for line in block.splitlines():
        m = PRE_RE.search(line)
        if m:
            cur = m.group(1)
            out[cur] = {"acc": float(m.group(2)), "nov": float(m.group(3))}
            continue
        m = RD_RE.search(line)
        if m and cur and m.group(1) == "2":
            out[cur]["margin2"] = float(m.group(2))
            out[cur]["anchor2"] = float(m.group(3))
    return out


data = {}
for (ds, dim), fname in LOGS.items():
    path = os.path.join(SCRATCH, fname)
    if os.path.exists(path):
        data[(ds, dim)] = parse(path)
    else:
        print(f"  missing {fname} -- skipping ({ds}, {dim}d)")
for (ds, dim), fname in RESC_LOGS.items():
    path = os.path.join(SCRATCH, fname)
    if os.path.exists(path):
        r = parse(path).get("res")
        if r:
            data.setdefault((ds, dim), {})["res-cw"] = r
    else:
        print(f"  missing {fname} -- skipping res-cw ({ds}, {dim}d)")

METRICS = [("acc", "seen nearest-anchor acc"),
           ("nov", "novelty AUC (pre-discovery)"),
           ("margin2", "margin AUC (round 2)"),
           ("anchor2", "mean anchor AUC (round 2)")]
DATASETS = {"cifar10": [4, 8, 16, 64], "mnist": [4, 8, 16, 64],
            "cifar100": [8, 16, 32, 64, 100]}

fig, axes = plt.subplots(len(DATASETS), 4, figsize=(17, 12))
for i, (ds, dims) in enumerate(DATASETS.items()):
    for j, (key, label) in enumerate(METRICS):
        ax = axes[i, j]
        for sp in SPACES:
            xs, ys = [], []
            for d in dims:
                v = data.get((ds, d), {}).get(sp, {}).get(key)
                if v is not None:
                    xs.append(d); ys.append(v)
            if xs:
                ax.plot(xs, ys, "-o", color=COLORS[sp], lw=2, ms=6, label=sp)
        ax.set_xscale("log", base=2)
        ax.set_xticks(dims); ax.set_xticklabels([str(d) for d in dims])
        ax.grid(alpha=0.25)
        if i == 0:
            ax.set_title(label, fontsize=11)
        if i == len(DATASETS) - 1:
            ax.set_xlabel("embedding dimension")
        if j == 0:
            ax.set_ylabel(f"{ds}\n", fontsize=12)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=11,
           bbox_to_anchor=(0.5, 1.0))
fig.suptitle("exp 28: anomaly metrics vs embedding dimension "
             "(holdout 4, seed 0; supres has no 4-D run)", y=0.95)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(OUT, dpi=150)
print(f"saved {OUT}")
