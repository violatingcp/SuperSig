"""
Experiment 54: visualize an exp-50 arm's embedding space.

Retrains a single exp-50 arm bit-for-bit (same per-arm seed offset, loader,
epochs), caches the train/test embeddings to logs/exp54/, and renders:
  1. PCA scatter of the test set (classes = tab10, holdout = black x)
  2. corner plot of the first 6 dims
  3. min-distance-to-seen-centroid histograms (seen vs holdout test), the
     direct view of the calibrated-distance property.
Reuses a cached embeddings npz on rerun instead of retraining.

    python experiments/54_plot_space.py --arm nplm_sup_dist --dataset cifar10
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import importlib
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from supersig.config import DEVICE, plot_path
from supersig.data import get_cifar_loaders, cifar_two_view_loader
from supersig.models import CIFARResNetBackbone
from supersig.plotting import plot_latent_panels, plot_corner
from supersig.recipes import recipe
from supersig.train import collect_embeddings

exp28 = importlib.import_module("28_concat_residual")
exp29 = importlib.import_module("29_residual_finetune")
exp34h = importlib.import_module("34h_hybrid_nplm_cifar")
exp50 = importlib.import_module("50_nplm_cifar10_suite")
exp53 = importlib.import_module("53_nplm_classwise")
exp55 = importlib.import_module("55_nplm_discovery")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="nplm_sup_dist",
                    choices=list(exp50.ARMS) + list(exp53.ARMS))
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--holdout", type=int, default=4)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=1.0)
    args = ap.parse_args()
    ds = args.dataset

    cfg = recipe(ds, emb_dim=args.dim)
    n_cls = cfg["n_classes"]
    holdouts = {args.holdout}
    seen = [c for c in range(n_cls) if c not in holdouts]
    con_ep = args.epochs or (2 if args.quick else 20)
    names = (exp29.CIFAR_NAMES if ds == "cifar10"
             else [str(c) for c in range(n_cls)])
    os.makedirs(os.path.join("logs", "exp54"), exist_ok=True)
    cache = os.path.join("logs", "exp54", f"embs_{args.arm}_{ds}.npz")

    if os.path.exists(cache):
        print(f"loading cached embeddings {cache}")
        d = np.load(cache)
        tr, tr_lab, te, te_lab = d["tr"], d["tr_lab"], d["te"], d["te_lab"]
    else:
        print(f"retraining {args.arm} [{ds}] with its original seed")
        net = exp55.train_arm(args.arm, ds, cfg, args, con_ep, holdouts)
        train_loader, test_loader = get_cifar_loaders(quick=args.quick,
                                                      dataset=ds)
        train_eval_loader = DataLoader(train_loader.dataset, batch_size=256,
                                       shuffle=False, num_workers=2)
        tr, tr_lab = collect_embeddings(net, train_eval_loader)
        te, te_lab = collect_embeddings(net, test_loader)
        np.savez(cache, tr=tr, tr_lab=tr_lab, te=te, te_lab=te_lab)
        print(f"saved embeddings {cache}")

    tag = f"{args.arm}_{ds}"
    plot_latent_panels({args.arm: (te, te_lab)}, holdouts, names,
                       plot_path(f"exp54_latent_{tag}.png"),
                       title=f"exp54: {args.arm} test space ({ds})")
    print("saved", plot_path(f"exp54_latent_{tag}.png"))
    plot_corner(te[:, :6], te_lab, plot_path(f"exp54_corner_{tag}.png"),
                title=f"exp54 {args.arm} ({ds}, first 6 of {te.shape[1]})")
    print("saved", plot_path(f"exp54_corner_{tag}.png"))

    m = np.isin(tr_lab, seen)
    cents = exp28.class_centroids(tr[m], tr_lab[m], seen)
    d = torch.cdist(torch.as_tensor(te, dtype=torch.float32, device=DEVICE),
                    torch.as_tensor(cents, dtype=torch.float32,
                                    device=DEVICE))
    s = d.min(1).values.cpu().numpy()
    sm, hm = np.isin(te_lab, seen), np.isin(te_lab, list(holdouts))
    plt.figure(figsize=(7.5, 5))
    bins = np.linspace(0, np.percentile(s, 99.5), 60)
    plt.hist(s[sm], bins=bins, density=True, alpha=0.6, color="#2a78d6",
             label=f"seen classes (n={sm.sum()})")
    plt.hist(s[hm], bins=bins, density=True, alpha=0.6, color="#e34948",
             label=f"holdout {sorted(holdouts)} (n={hm.sum()})")
    plt.xlabel("min distance to seen-class centroid")
    plt.ylabel("density")
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(hm[sm | hm], s[sm | hm])
    plt.title(f"exp54 {args.arm} ({ds}): per-event novelty distance "
              f"(AUC={auc:.3f})")
    plt.legend(); plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(plot_path(f"exp54_dist_{tag}.png"), dpi=150)
    plt.close()
    print("saved", plot_path(f"exp54_dist_{tag}.png"))


if __name__ == "__main__":
    main()
