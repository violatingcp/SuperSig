"""
Experiment 27: DualSuperVisReg vs the settled default pipeline.

Head-to-head embedding comparison with the discovery machinery held fixed:

  default : classwise SIGReg + proto + repulsed means (the exp-26 recipe)
            -> run_discovery
  dual    : DualSuperVisReg alone (anchor-free; per-class isotropic Gaussians
            whose batch centroids live in a global Gaussian) -> empirical
            class centroids as means -> the same run_discovery

Both arms are evaluated pre-discovery (nearest-mean accuracy on seen test
classes, novelty AUC from distance to the nearest seen mean) and through the
standard discovery rounds (purity / margin AUC / mean per-anchor AUC).  A
Gaussianity table (per-class covariance eigenspectrum range, per-dim class
RMS, worst inter-dimension correlation, calibrated sliced-Wasserstein shape
ratio, centroid separation; supersig.metrics.gaussianity_summary) is printed
for each arm in both the pre-trained space (after embedding training) and the
clustered fine-tuned space (after the discovery rounds).  The
dual arm has no trained anchor for the holdout rows, so those rows are filled
with the default fixed anchors -- same structural treatment as the default
arm, where the holdout anchor also never sees data.

Datasets: cifar10 (resnet20 backbone, exp-26 recipe) and mnist (ConvBackbone,
same hyperparameters; MNIST has no settled recipe, so the cifar10 one is
reused).  Holdout class 4 = deer / digit 4, matching the historical series.

    python experiments/27_dual_visreg.py                    # CIFAR-10, k=1
    python experiments/27_dual_visreg.py --dataset mnist
    python experiments/27_dual_visreg.py --ks 1,2,3
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from supersig.config import DATA_DIR, DEVICE, plot_path
from supersig.data import (get_cifar_loaders, get_loaders, cifar_balanced_loader,
                           mnist_balanced_loader, _cifar_spec, TF_PLAIN)
from supersig.losses import DualSuperVisReg, make_anchors
from supersig.models import CIFARResNetBackbone, ConvBackbone
from supersig.recipes import supervised_embedding, recipe
from supersig.discovery import run_discovery
from supersig.metrics import gaussianity_summary
from supersig.plotting import plot_corner
from supersig.train import train_dual_visreg, train_sigreg_hybrid, collect_embeddings

HOLDOUT_SETS = {1: [4], 2: [4, 9], 3: [0, 4, 9]}
CIFAR_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

GAUSS_ROWS = [
    ("eig min (class cov)", "eig_min", ".3f"),
    ("eig max (class cov)", "eig_max", ".3f"),
    ("eig cond worst", "eig_cond_max", ".1f"),
    ("class RMS min", "rms_min", ".3f"),
    ("class RMS mean", "rms_mean", ".3f"),
    ("class RMS max", "rms_max", ".3f"),
    ("max |corr| (worst class)", "corr_max", ".3f"),
    ("SW ratio mean", "sw_ratio_mean", ".2f"),
    ("SW ratio worst", "sw_ratio_max", ".2f"),
    ("|skew| mean", "skew_mean", ".3f"),
    ("|ex-kurt| mean", "kurt_mean", ".3f"),
    ("centroid dist min", "cdist_min", ".2f"),
    ("centroid dist mean", "cdist_mean", ".2f"),
    ("separation (min d/RMS)", "separation", ".2f"),
]


def print_gauss_table(spaces):
    """spaces: {column name: gaussianity_summary dict} printed metric-per-row."""
    print(f"  {'metric':<26}" + "".join(f"{n:>16}" for n in spaces))
    for label, key, fmt in GAUSS_ROWS:
        print(f"  {label:<26}"
              + "".join(f"{spaces[n][key]:>16{fmt}}" for n in spaces))


def plot_latent_overview(spaces, holdouts, names, out_path):
    """
    2x2 PCA scatter of the four spaces (default/dual x pre/ft).  Each panel is
    PCA-projected independently; seen classes in tab10, holdouts as black x.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 12))
    cmap = plt.get_cmap("tab10")
    for ax, (name, (embs, lab)) in zip(axes.flat, spaces.items()):
        p = PCA(n_components=2).fit_transform(embs)
        for c in range(10):
            m = lab == c
            if c in holdouts:
                ax.scatter(p[m, 0], p[m, 1], s=6, c="k", marker="x",
                           alpha=0.5, label=f"{names[c]} (holdout)")
            else:
                ax.scatter(p[m, 0], p[m, 1], s=3, color=cmap(c % 10),
                           alpha=0.35, label=names[c])
        ax.set_title(name)
        ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=10, fontsize=8,
               markerscale=3, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Latent spaces: test embeddings, PCA per panel", y=0.955)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"  saved {out_path}")


def make_backbone(dataset, cfg, scratch=False):
    if dataset == "mnist":
        return ConvBackbone(cfg["emb_dim"]).to(DEVICE)   # always random init
    return CIFARResNetBackbone(cfg["emb_dim"], arch=cfg["arch"],
                               pretrain="none" if scratch else "cifar10").to(DEVICE)


def balanced_loader(dataset, holdouts, quick, limit, per_class=24):
    if dataset == "mnist":
        return mnist_balanced_loader(holdout=holdouts, quick=quick, limit=limit,
                                     per_class=per_class)
    return cifar_balanced_loader("cifar10", holdout=holdouts, quick=quick,
                                 limit=limit, per_class=per_class)


def default_embedding(dataset, holdouts, cfg, quick=False, limit=None, seed=0,
                      scratch=False):
    """The exp-26 recipe embedding (classwise SIGReg + proto + repulsed means)."""
    if dataset == "cifar10":
        backbone, means, _ = supervised_embedding(
            "cifar10", holdouts=holdouts, quick=quick, limit=limit, seed=seed,
            pretrain="none" if scratch else None,
            ssl_epochs=cfg["ssl_epochs"], emb_dim=cfg["emb_dim"])
        return backbone, means
    torch.manual_seed(seed); np.random.seed(seed)
    backbone = make_backbone(dataset, cfg)
    means = make_anchors(cfg["pair_dist"] / math.sqrt(2.0),
                         emb_dim=cfg["emb_dim"],
                         n_classes=cfg["n_classes"]).clone()
    loader = balanced_loader(dataset, holdouts, quick, limit)
    train_sigreg_hybrid(backbone, loader, cfg["ssl_epochs"], means,
                        mode="repulse", disc="proto", alpha=1.0,
                        rep_weight=cfg["rep_weight"],
                        sigreg_weight=cfg["sigreg_weight"],
                        n_slices=cfg["n_slices"])
    return backbone, means


def dual_embedding(dataset, holdouts, cfg, quick=False, limit=None, seed=0,
                   train_eval_loader=None, global_scale=None, scratch=False,
                   per_class=24):
    """Train a fresh backbone with DualSuperVisReg; return (backbone, means).

    Loss hyperparameters (projections, scales, term weights) follow the
    DualSuperVisReg defaults unless global_scale is given explicitly.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    backbone = make_backbone(dataset, cfg, scratch=scratch)
    kw = {} if global_scale is None else {"global_scale": global_scale}
    loss_fn = DualSuperVisReg(num_classes=cfg["n_classes"] - len(holdouts),
                              embed_dim=cfg["emb_dim"], **kw).to(DEVICE)
    loader = balanced_loader(dataset, holdouts, quick, limit,
                             per_class=per_class)
    train_dual_visreg(backbone, loader, cfg["ssl_epochs"], loss_fn)

    # Anchor-free training: read the class means off the (seen) training data.
    embs, lab = collect_embeddings(backbone, train_eval_loader)
    means = make_anchors(cfg["pair_dist"] / math.sqrt(2.0),
                         emb_dim=cfg["emb_dim"],
                         n_classes=cfg["n_classes"]).clone()
    for c in range(cfg["n_classes"]):
        if c in holdouts:
            continue                     # no data: keep the fixed-anchor fill
        means[c] = torch.as_tensor(embs[lab == c].mean(axis=0), device=DEVICE)
    return backbone, means


def pre_discovery_eval(backbone, means, test_loader, seen, holdouts):
    """Nearest-mean accuracy on seen test classes + novelty AUC (pre-discovery)."""
    embs, lab = collect_embeddings(backbone, test_loader)
    z = torch.as_tensor(embs, device=DEVICE)
    d = torch.cdist(z, means.detach()[seen])
    pred = np.array(seen)[d.argmin(1).cpu().numpy()]
    seen_mask = np.isin(lab, seen)
    acc = (pred[seen_mask] == lab[seen_mask]).mean()
    is_unseen = np.isin(lab, list(holdouts)).astype(int)
    auc = roc_auc_score(is_unseen, d.min(1).values.cpu().numpy())
    return float(acc), float(auc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar10", "mnist"], default="cifar10")
    ap.add_argument("--ks", default="1")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--global-scale", type=float, default=None,
                    help="override DualSuperVisReg.global_scale (default: class default)")
    ap.add_argument("--dual-per-class", type=int, default=24,
                    help="samples per class per batch for dual-arm training "
                         "(batch = n_seen_classes x this)")
    ap.add_argument("--plots", action="store_true",
                    help="save PCA overview + corner plots of all four spaces")
    ap.add_argument("--scratch", action="store_true",
                    help="random-init backbone (no torch.hub pretrained weights;"
                         " MNIST's ConvBackbone is always random-init)")
    ap.add_argument("--epochs", type=int, default=None,
                    help="override ssl_epochs (useful for from-scratch runs)")
    ap.add_argument("--emb-dim", type=int, default=None,
                    help="override the recipe embedding dimension (default 16)")
    args = ap.parse_args()
    ds = args.dataset
    ks = [int(x) for x in args.ks.split(",")]
    cfg = recipe("cifar10", emb_dim=args.emb_dim)   # MNIST reuses this recipe
    if args.epochs:
        cfg["ssl_epochs"] = args.epochs
    names = [str(d) for d in range(10)] if ds == "mnist" else CIFAR_NAMES

    if ds == "mnist":
        train_loader, test_loader = get_loaders(batch_size=256, quick=args.quick)
        base = datasets.MNIST(DATA_DIR, train=True, download=True,
                              transform=TF_PLAIN)
    else:
        train_loader, test_loader = get_cifar_loaders(quick=args.quick,
                                                      limit=args.limit)
        cls, plain, _ = _cifar_spec("cifar10")
        base = cls(DATA_DIR, train=True, download=True, transform=plain)
    train_eval_loader = DataLoader(train_loader.dataset, batch_size=256,
                                   shuffle=False, num_workers=2)

    print(f"recipe [{ds}]{' [scratch]' if args.scratch else ''}: "
          + "  ".join(f"{k}={v}" for k, v in cfg.items()))
    summary = {}
    for k in ks:
        holdouts = set(HOLDOUT_SETS[k])
        seen = [c for c in range(cfg["n_classes"]) if c not in holdouts]
        print(f"\n===== k={k} ({', '.join(names[c] for c in sorted(holdouts))}) =====")
        summary[k] = {}
        for arm in ("default", "dual"):
            print(f"\n----- arm: {arm} -----")
            if arm == "default":
                backbone, means = default_embedding(
                    ds, holdouts, cfg, quick=args.quick, limit=args.limit,
                    seed=args.seed, scratch=args.scratch)
            else:
                backbone, means = dual_embedding(
                    ds, holdouts, cfg, quick=args.quick, limit=args.limit,
                    seed=args.seed, train_eval_loader=train_eval_loader,
                    global_scale=args.global_scale, scratch=args.scratch,
                    per_class=args.dual_per_class)
            acc, auc = pre_discovery_eval(backbone, means, test_loader,
                                          seen, holdouts)
            print(f"  pre-discovery: seen nearest-mean acc={acc:.4f}  "
                  f"novelty AUC={auc:.4f}")
            e_pre, l_pre = collect_embeddings(backbone, test_loader)
            g_pre = gaussianity_summary(e_pre, l_pre, seen, seed=args.seed)
            ft_epochs = 1 if args.quick else cfg["ft_epochs"]
            _, history = run_discovery(
                backbone, means, base_ds=base,
                train_eval_loader=train_eval_loader, test_loader=test_loader,
                seen=seen, holdouts=holdouts, dataset_name=ds,
                rep_weight=cfg["rep_weight"], sigreg_weight=cfg["sigreg_weight"],
                n_slices=cfg["n_slices"], rounds=args.rounds,
                ft_epochs=ft_epochs, names=names, seed=args.seed)
            e_ft, l_ft = collect_embeddings(backbone, test_loader)
            g_ft = gaussianity_summary(e_ft, l_ft, seen, seed=args.seed)
            summary[k][arm] = dict(acc=acc, auc=auc, history=history,
                                   gauss={"pre": g_pre, "ft": g_ft},
                                   embs={"pre": (e_pre, l_pre),
                                         "ft": (e_ft, l_ft)})
        print(f"\n  --- gaussianity (seen classes, test set) k={k} ---")
        print_gauss_table({f"{arm}/{sp}": summary[k][arm]["gauss"][sp]
                           for arm in ("default", "dual")
                           for sp in ("pre", "ft")})
        if args.plots:
            tag = (f"{ds}_exp27{'_scratch' if args.scratch else ''}"
                   f"{f'_{args.emb_dim}d' if args.emb_dim else ''}_k{k}")
            spaces = {f"{arm}/{sp}": summary[k][arm]["embs"][sp]
                      for arm in ("default", "dual") for sp in ("pre", "ft")}
            plot_latent_overview(spaces, holdouts, names,
                                 plot_path(f"latent_{tag}.png"))
            for name, (embs, lab) in spaces.items():
                plot_corner(embs[:, :6], lab,
                            plot_path(f"corner_{tag}_"
                                      f"{name.replace('/', '_')}.png"),
                            title=f"{ds} exp27 {name} (first "
                                  f"{min(6, embs.shape[1])} of "
                                  f"{embs.shape[1]} dims)")

    print(f"\n===== DUAL-VISREG vs DEFAULT SUMMARY [{ds}] =====")
    for k in ks:
        for arm in ("default", "dual"):
            s = summary[k][arm]
            print(f"  k={k} [{arm:7s}] pre: acc={s['acc']:.4f} "
                  f"novelty-AUC={s['auc']:.4f}")
            for h in s["history"]:
                print(f"          round {h['round']}: purity={h['purity']:.3f} "
                      f"anchors={h['n_anchors']}  margin={h['margin']:.4f}  "
                      f"mean-anchor={h['mean_pc']:.4f}")
    print("Done.")


if __name__ == "__main__":
    main()
