"""Merge the f=0.02 top-up points into the four 16-D power figures."""
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOGDIR = "/home/pharris/sigreg/SuperSig/logs"
PLOTS = "/home/pharris/sigreg/SuperSig/plots"
COLORS = {"sup->res": "#2a78d6", "ssl->supres": "#1baf7a", "joint": "#eda100",
          "sup": "#008300", "supcon": "#4a3aa7", "supcon+simclr": "#e34948"}
ARMS = list(COLORS)
F_ORIG = [0.001, 0.003, 0.01, 0.03, 0.1]
F_ALL = [0.001, 0.003, 0.01, 0.02, 0.03, 0.1]
INS = 3   # index where 0.02 inserts


def parse_block(path, header):
    """{(arm, kind): [values]} from a POWER SUMMARY table."""
    text = open(path).read()
    block = text[text.index(header):]
    out = {}
    for line in block.splitlines()[2:]:
        m = re.match(r"\s+(\S+)\s+(pre|post)\s+([\d.\s]+)$", line)
        if not m:
            m2 = re.match(r"\s+(\S+)\s+([\d.]+(?:\s+[\d.]+)+)$", line)
            if m2 and m2.group(1) in ARMS:   # exp30 style: arm pre f...
                vals = [float(x) for x in m2.group(2).split()]
                out[(m2.group(1), "pre")] = [vals[0]]
                out[(m2.group(1), "post")] = vals[1:]
                continue
            if line.strip().startswith("saved") or "=====" in line[2:]:
                break
            continue
        out[(m.group(1), m.group(2))] = [float(x) for x in m.group(3).split()]
    return out


def merged(orig, top, arm, kind):
    o = orig[(arm, kind)]
    t = top[(arm, kind)]
    if len(o) == 1:                       # exp30 pre: constant
        return [o[0]] * len(F_ALL)
    vals = list(o)
    vals.insert(INS, t[-1])
    return vals


def draw(orig_path, top_path, header, title, out, pre_constant=False):
    orig = parse_block(orig_path, header)
    top = parse_block(top_path, header)
    plt.figure(figsize=(8, 6.5))
    for arm in ARMS:
        c = COLORS[arm]
        post = merged(orig, top, arm, "post")
        plt.plot(F_ALL, post, "-o", color=c, lw=2, ms=6, label=f"{arm} post")
        pre = merged(orig, top, arm, "pre")
        if pre_constant:
            plt.axhline(pre[0], color=c, lw=1.2, ls="--", alpha=0.6)
        else:
            plt.plot(F_ALL, pre, "--o", color=c, lw=1.4, ms=5, alpha=0.75,
                     label=f"{arm} pre")
    plt.xscale("log")
    plt.axhline(0.05, color="gray", lw=1, ls=":")
    plt.xlabel("injected anomaly fraction")
    plt.ylabel("power at alpha=0.05")
    plt.title(title)
    plt.grid(alpha=0.25, which="both")
    plt.legend(loc="upper left", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print("saved", out)


draw(f"{LOGDIR}/exp30/exp30_cifar10_16d.log",
     f"{LOGDIR}/exp30/exp30_cifar10_16d_f002.log",
     "===== EXP30 POWER SUMMARY",
     "exp30: per-event power vs injection fraction (16+16d, incl. f=0.02)\n"
     "(solid = post-discovery margin, dashed = pre-discovery distance)",
     f"{PLOTS}/exp30_power_curves_16d.png", pre_constant=True)
draw(f"{LOGDIR}/exp31/exp31_cifar10_16d.log",
     f"{LOGDIR}/exp31/exp31_cifar10_16d_f002.log",
     "===== EXP31 SPARKER POWER SUMMARY",
     "exp31: SparKer (annealed) dataset-level power (16+16d, incl. f=0.02)",
     f"{PLOTS}/exp31_sparker_power_16d.png")
draw(f"{LOGDIR}/exp32/exp32_cifar10_16d.log",
     f"{LOGDIR}/exp32/exp32_cifar10_16d_f002.log",
     "===== EXP32 MAHA POWER SUMMARY",
     "exp32: Mahalanobis dataset-level power (16+16d, incl. f=0.02)",
     f"{PLOTS}/exp32_maha_power_16d.png")
draw(f"{LOGDIR}/exp32/exp32_cifar10_16d.log",
     f"{LOGDIR}/exp32/exp32_cifar10_16d_f002.log",
     "===== EXP32 MMD POWER SUMMARY",
     "exp32: MMD dataset-level power (16+16d, incl. f=0.02)",
     f"{PLOTS}/exp32_mmd_power_16d.png")
