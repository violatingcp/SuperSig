"""
Experiment 158: visualize the exp-157 10-D LeJEPA/Galaxy10 latents.

Per arm, a two-panel PCA(2) scatter of the test embeddings -- PRE vs POST
s/sqrt(b) discovery -- with the 9 seen classes muted, the novel class
highlighted, the discovered anchors as stars and the seen centroids as X.
One combined PDF over all arms, plus a single-figure overview.

    python experiments/158_galaxy10d_plot.py --base lejepa --draw 0
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.decomposition import PCA

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN = os.path.join(REPO, "logs", "exp157")
OUT = os.path.join(REPO, "plots")
GAL_NAMES = {0: "disturbed", 1: "merging", 2: "round smooth",
             3: "in-between smooth", 4: "cigar smooth", 5: "barred spiral",
             6: "unbarred tight spiral", 7: "unbarred loose spiral",
             8: "edge-on w/o bulge", 9: "edge-on w/ bulge"}


def panel(ax, Z, lab, seen, hold, anchors, cents, title, rng, per=400):
    p = PCA(n_components=2, random_state=0).fit(Z)
    P = p.transform(Z)
    A = p.transform(anchors) if len(anchors) else None
    C = p.transform(cents) if len(cents) else None
    for c in seen:                                   # muted seen classes
        idx = np.where(lab == c)[0]
        if len(idx) > per:
            idx = rng.choice(idx, per, replace=False)
        ax.scatter(P[idx, 0], P[idx, 1], s=4, c="#c7c7c7", alpha=0.5,
                   linewidths=0)
    h = list(hold)[0]
    idx = np.where(lab == h)[0]
    if len(idx) > per * 2:
        idx = rng.choice(idx, per * 2, replace=False)
    ax.scatter(P[idx, 0], P[idx, 1], s=8, c="#d62728", alpha=0.7,
               linewidths=0, label=f"novel: {GAL_NAMES.get(h, h)}")
    if C is not None:
        ax.scatter(C[:, 0], C[:, 1], s=70, marker="X", c="#1f77b4",
                   edgecolors="k", linewidths=0.6, label="seen centroids")
    if A is not None:
        ax.scatter(A[:, 0], A[:, 1], s=180, marker="*", c="#ffd400",
                   edgecolors="k", linewidths=0.9,
                   label="discovered anchor")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=6, loc="best", framealpha=0.6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="lejepa")
    ap.add_argument("--draw", type=int, default=0)
    ap.add_argument("--emb-dim", type=int, default=10)
    ap.add_argument("--split", default="te", choices=["te", "tr"])
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(0)
    files = sorted(glob.glob(os.path.join(
        IN, f"viz_galaxy10_{args.base}_d{args.draw}_e{args.emb_dim}_*.npz")))
    if not files:
        print("no npz found"); return
    tag = f"galaxy10_{args.base}_d{args.draw}_e{args.emb_dim}"
    pdf_path = os.path.join(OUT, f"viz_{tag}.pdf")
    arms = []
    with PdfPages(pdf_path) as pdf:
        for f in files:
            arm = os.path.basename(f).split(f"e{args.emb_dim}_")[1][:-4]
            arms.append((arm, f))
            d = np.load(f)
            seen, hold = d["seen"], d["holdout"]
            fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
            panel(axes[0], d[f"{args.split}"], d[f"{args.split}_lab"], seen,
                  hold, np.empty((0, args.emb_dim)), d["seen_centroids"],
                  f"{arm}  PRE-discovery", rng)
            panel(axes[1], d[f"{args.split}_post"], d[f"{args.split}_lab"],
                  seen, hold, d["anchors"], d["seen_centroids"],
                  f"{arm}  POST s/√b  ({len(d['anchors'])} anchor"
                  f"{'s' if len(d['anchors'])!=1 else ''})", rng)
            fig.suptitle(f"Galaxy10 / {args.base} / 10-D latent / draw "
                         f"{args.draw} (novel = class {list(hold)[0]})",
                         fontsize=11)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig, dpi=140)
            png = os.path.join(OUT, f"pca_{tag}_{arm}.png")
            fig.savefig(png, dpi=140); plt.close(fig)
            print(f"  wrote {png}")
    print(f"wrote {pdf_path} ({len(arms)} arms)")

    # overview: POST panels for the discovery-relevant arms, one grid
    key = [a for a in ("supcon-ft", "ss-ft", "nplm-sup-ft", "supcon-ft_res",
                       "supcon-ft_resnplm", "ss-ft_res")
           if any(a == x for x, _ in arms)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
    for ax, arm in zip(axes.ravel(), key):
        d = np.load(os.path.join(
            IN, f"viz_{tag}_{arm}.npz"))
        panel(ax, d[f"{args.split}_post"], d[f"{args.split}_lab"], d["seen"],
              d["holdout"], d["anchors"], d["seen_centroids"],
              f"{arm}  ({len(d['anchors'])} anchor)", rng)
    for ax in axes.ravel()[len(key):]:
        ax.axis("off")
    fig.suptitle(f"Galaxy10 / {args.base} 10-D latent, POST s/√b "
                 f"discovery (novel = class {int(np.load(files[0])['holdout'][0])})",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    ov = os.path.join(OUT, f"viz_{tag}_overview.png")
    fig.savefig(ov, dpi=150); plt.close(fig)
    print(f"wrote {ov}")


if __name__ == "__main__":
    main()
