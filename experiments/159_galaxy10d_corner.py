"""
Experiment 159: corner plots of the exp-157 10-D LeJEPA/Galaxy10 latent.

A full pairwise scatter matrix of the 10 latent dimensions, coloured by
class: the 9 labelled (seen) classes each a distinct colour, and the
UNLABELLED novel class (the held-out one, which arrives in the corpus
without a label) drawn on top in bold black.  Diagonal = per-dimension
histograms (labelled-pool grey vs unlabelled-novel red).  Discovered
anchors are marked as gold stars on every off-diagonal panel.

    python experiments/159_galaxy10d_corner.py --arms supcon-ft supcon-ft_res
    python experiments/159_galaxy10d_corner.py --state pre
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN = os.path.join(REPO, "logs", "exp157")
OUT = os.path.join(REPO, "plots")
NAMES = {0: "disturbed", 1: "merging", 2: "round smooth",
         3: "in-between", 4: "cigar smooth", 5: "barred spiral",
         6: "tight spiral", 7: "loose spiral", 8: "edge-on no bulge",
         9: "edge-on bulge"}


def corner(Z, lab, seen, novel, anchors, title, per_seen=120, per_nov=600):
    d = Z.shape[1]
    rng = np.random.default_rng(0)
    palette = plt.cm.tab10(np.linspace(0, 1, 10))
    # limits per dim from robust quantiles
    lo = np.percentile(Z, 1, axis=0); hi = np.percentile(Z, 99, axis=0)
    fig, ax = plt.subplots(d, d, figsize=(1.5 * d, 1.5 * d))
    for i in range(d):
        for j in range(d):
            a = ax[i, j]
            if j > i:
                a.axis("off"); continue
            if i == j:
                # diagonal: labelled pool (grey) vs unlabelled novel (red)
                seenmask = np.isin(lab, seen)
                a.hist(Z[seenmask, i], bins=40, color="#9a9a9a", alpha=0.7,
                       density=True, range=(lo[i], hi[i]))
                a.hist(Z[lab == novel, i], bins=40, histtype="step",
                       color="#d62728", lw=1.4, density=True,
                       range=(lo[i], hi[i]))
                a.set_yticks([]); a.set_xlim(lo[i], hi[i])
            else:
                for k, c in enumerate(seen):        # labelled seen classes
                    idx = np.where(lab == c)[0]
                    if len(idx) > per_seen:
                        idx = rng.choice(idx, per_seen, replace=False)
                    a.scatter(Z[idx, j], Z[idx, i], s=3,
                              color=palette[c], alpha=0.55, linewidths=0)
                idx = np.where(lab == novel)[0]     # unlabelled novel on top
                if len(idx) > per_nov:
                    idx = rng.choice(idx, per_nov, replace=False)
                a.scatter(Z[idx, j], Z[idx, i], s=6, color="k", alpha=0.8,
                          linewidths=0, zorder=5)
                if len(anchors):
                    a.scatter(anchors[:, j], anchors[:, i], s=140,
                              marker="*", c="#ffd400", edgecolors="k",
                              linewidths=0.8, zorder=6)
                a.set_xlim(lo[j], hi[j]); a.set_ylim(lo[i], hi[i])
            if i == d - 1:
                a.set_xlabel(f"z{j}", fontsize=7)
            else:
                a.set_xticklabels([])
            if j == 0 and i != 0:
                a.set_ylabel(f"z{i}", fontsize=7)
            else:
                a.set_yticklabels([])
            a.tick_params(labelsize=5)
    handles = [Line2D([0], [0], marker="o", ls="", color=palette[c],
                      label=f"z{c} {NAMES.get(c,'')}", ms=6) for c in seen]
    handles += [Line2D([0], [0], marker="o", ls="", color="k",
                       label=f"UNLABELLED novel: class {novel} "
                             f"({NAMES.get(novel,'')})", ms=6),
                Line2D([0], [0], marker="*", ls="", color="#ffd400",
                       mec="k", label="discovered anchor", ms=12)]
    fig.legend(handles=handles, loc="upper right", fontsize=9,
               framealpha=0.9, bbox_to_anchor=(0.98, 0.98))
    fig.suptitle(title, fontsize=13, x=0.35)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="lejepa")
    ap.add_argument("--draw", type=int, default=0)
    ap.add_argument("--emb-dim", type=int, default=10)
    ap.add_argument("--arms", nargs="+",
                    default=["supcon-ft", "supcon-ft_res"])
    ap.add_argument("--state", default="post", choices=["pre", "post"])
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for arm in args.arms:
        f = os.path.join(IN, f"viz_galaxy10_{args.base}_d{args.draw}_"
                         f"e{args.emb_dim}_{arm}.npz")
        if not os.path.exists(f):
            print(f"[miss] {arm}"); continue
        d = np.load(f)
        Z = d["te_post"] if args.state == "post" else d["te"]
        anchors = d["anchors"] if args.state == "post" else np.empty((0,
                                                        args.emb_dim))
        novel = int(d["holdout"][0])
        title = (f"Galaxy10/{args.base} 10-D latent  --  {arm}  "
                 f"({args.state.upper()}"
                 f"{' s/√b discovery' if args.state=='post' else '-discovery'})")
        fig = corner(Z, d["te_lab"], d["seen"].tolist(), novel, anchors,
                     title)
        out = os.path.join(OUT, f"corner_galaxy10_{args.base}_d{args.draw}_"
                           f"e{args.emb_dim}_{arm}_{args.state}.png")
        fig.savefig(out, dpi=130); plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
