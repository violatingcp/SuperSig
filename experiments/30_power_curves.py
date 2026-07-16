"""
Experiment 30: power curves at alpha = 0.05 vs injected anomaly fraction.

Physics-style hidden-signal evaluation on the exp-29 arms (CIFAR-10, deer as
the signal).  The embeddings are trained once on seen classes only (identical
seeds to exp 29).  Then, for each injected fraction f, the unlabeled stream
handed to discovery is 45k seen train images plus N deer sampled such that
N / (45k + N) = f, and the discovery loop (pool -> BIC k-means -> anchors ->
fine-tune rounds) runs from scratch on that stream.

Power = TPR on the deer TEST images at the score threshold giving false
positive rate alpha = 0.05 on the seen TEST images.

  pre-discovery score  : min distance to the seen anchors (constant in f)
  post-discovery score : margin d(nearest seen anchor) - d(nearest discovered
                         anchor) in the fine-tuned space

    python experiments/30_power_curves.py
    python experiments/30_power_curves.py --fractions 0.01,0.1 --quick
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import copy
import importlib
import math
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from supersig.config import DATA_DIR, DEVICE, plot_path
from supersig.data import (get_cifar_loaders, cifar_balanced_loader,
                           cifar_two_view_loader, cifar_two_view_balanced_loader,
                           _cifar_spec)
from supersig.losses import make_anchors
from supersig.models import CIFARResNetBackbone
from supersig.recipes import supervised_embedding, recipe
from supersig.discovery import run_discovery
from supersig.train import (train_sigreg_ssl, train_sigreg_hybrid,
                            train_sigreg_hybrid_aug, train_sigreg_residual_ssl,
                            train_supcon, train_simclr, collect_embeddings)

exp28 = importlib.import_module("28_concat_residual")

CIFAR_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]
COLORS = {"sup->res": "#2a78d6", "ssl->supres": "#1baf7a", "joint": "#eda100",
          "sup": "#008300", "supcon": "#4a3aa7", "supcon+simclr": "#e34948"}


def power_at_alpha(seen_scores, sig_scores, alpha=0.05):
    """TPR on the signal at the (1-alpha) quantile of the background scores."""
    thr = np.quantile(seen_scores, 1.0 - alpha)
    return float((sig_scores > thr).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--emb-dim", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--fractions", default="0.001,0.003,0.01,0.03,0.1")
    ap.add_argument("--arms", default=None,
                    help="comma subset of the exp-29 arm names (default all)")
    args = ap.parse_args()
    ds = "cifar10"
    cfg = recipe(ds, emb_dim=args.emb_dim)
    ssl_ep = 2 if args.quick else 20
    sup_ep = cfg["ssl_epochs"]
    res_ep = 2 if args.quick else 10
    ft_ep = 1 if args.quick else cfg["ft_epochs"]
    holdouts = {args.holdout}
    seen = [c for c in range(cfg["n_classes"]) if c not in holdouts]
    fractions = [float(x) for x in args.fractions.split(",")]
    n_cls = cfg["n_classes"]
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    train_loader, test_loader = get_cifar_loaders(quick=args.quick)
    train_eval_loader = DataLoader(train_loader.dataset, batch_size=256,
                                   shuffle=False, num_workers=2)
    cls, plain, _ = _cifar_spec(ds)
    base = cls(DATA_DIR, train=True, download=True, transform=plain)
    base_targets = np.array(base.targets)
    n_base = 8000 if args.quick else len(base)
    seen_idx = np.where(np.isin(base_targets[:n_base], seen))[0]
    sig_idx_all = np.where(base_targets[:n_base] == args.holdout)[0]
    print(f"exp30 [{ds}] emb_dim={cfg['emb_dim']} holdout={sorted(holdouts)} "
          f"alpha={args.alpha} fractions={fractions}")

    def cents_of(net):
        e, l = collect_embeddings(net, train_eval_loader)
        m = np.isin(l, seen)
        return exp28.class_centroids(e[m], l[m], seen)

    # ----- networks (identical seeds/config to exp 29) ----------------------
    print("\n===== training: sup =====")
    sup, means_sup, _ = supervised_embedding(ds, holdouts=holdouts,
                                             quick=args.quick, seed=args.seed,
                                             emb_dim=cfg["emb_dim"])
    means_sup = means_sup.detach()
    print("\n===== training: res =====")
    torch.manual_seed(args.seed + 1); np.random.seed(args.seed + 1)
    res = copy.deepcopy(sup)
    train_sigreg_residual_ssl(
        res, cifar_two_view_balanced_loader(ds, holdout=holdouts,
                                            quick=args.quick),
        res_ep, means_sup, n_slices=cfg["n_slices"], classwise=True)
    print("\n===== training: ssl =====")
    torch.manual_seed(args.seed + 2); np.random.seed(args.seed + 2)
    trunk = CIFARResNetBackbone(cfg["emb_dim"], arch=cfg["arch"],
                                pretrain=ds).to(DEVICE)
    train_sigreg_ssl(trunk, cifar_two_view_loader(quick=args.quick,
                                                  labeled=False,
                                                  holdout=holdouts), ssl_ep)
    ssl_cents = cents_of(trunk)
    print("\n===== training: supres =====")
    torch.manual_seed(args.seed + 3); np.random.seed(args.seed + 3)
    supres = copy.deepcopy(trunk)
    means_supres = exp28.fill_means(ssl_cents, seen, cfg).clone()
    train_sigreg_hybrid(supres, cifar_balanced_loader(ds, holdout=holdouts,
                                                      quick=args.quick),
                        sup_ep, means_supres, mode="repulse", disc="proto",
                        alpha=1.0, rep_weight=cfg["rep_weight"],
                        sigreg_weight=cfg["sigreg_weight"],
                        n_slices=cfg["n_slices"])
    means_supres = means_supres.detach()
    print("\n===== training: joint =====")
    torch.manual_seed(args.seed + 4); np.random.seed(args.seed + 4)
    joint = CIFARResNetBackbone(cfg["emb_dim"], arch=cfg["arch"],
                                pretrain=ds).to(DEVICE)
    means_joint = make_anchors(cfg["pair_dist"] / math.sqrt(2.0),
                               emb_dim=cfg["emb_dim"],
                               n_classes=n_cls).clone()
    train_sigreg_hybrid_aug(joint,
                            cifar_two_view_balanced_loader(ds, holdout=holdouts,
                                                           quick=args.quick),
                            sup_ep, means_joint, rep_weight=cfg["rep_weight"],
                            sigreg_weight=cfg["sigreg_weight"],
                            n_slices=cfg["n_slices"])
    means_joint = means_joint.detach()
    print("\n===== training: supcon =====")
    torch.manual_seed(args.seed + 5); np.random.seed(args.seed + 5)
    supcon = CIFARResNetBackbone(cfg["emb_dim"], arch=cfg["arch"],
                                 pretrain=ds).to(DEVICE)
    train_supcon(supcon, cifar_two_view_loader(quick=args.quick, labeled=True,
                                               holdout=holdouts), sup_ep)
    supcon_cents = cents_of(supcon)
    print("\n===== training: simclr =====")
    torch.manual_seed(args.seed + 6); np.random.seed(args.seed + 6)
    simclr = CIFARResNetBackbone(cfg["emb_dim"], arch=cfg["arch"],
                                 pretrain=ds).to(DEVICE)
    train_simclr(simclr, cifar_two_view_loader(quick=args.quick, labeled=False,
                                               holdout=holdouts), ssl_ep)
    simclr_cents = cents_of(simclr)
    res_cents = cents_of(res)

    # arm -> (sup net, sup means matrix, aug net or None, aug cents or None)
    ARMS = {
        "sup->res": (sup, means_sup, res, res_cents),
        "ssl->supres": (supres, means_supres, trunk, ssl_cents),
        "joint": (joint, means_joint, None, None),
        "sup": (sup, means_sup, None, None),
        "supcon": (supcon, exp28.fill_means(supcon_cents, seen, cfg),
                   None, None),
        "supcon+simclr": (supcon, exp28.fill_means(supcon_cents, seen, cfg),
                          simclr, simclr_cents),
    }
    arm_names = (args.arms.split(",") if args.arms else list(ARMS))

    # test embeddings of the frozen aug halves + labels
    te_lab = None
    aug_te = {}
    for name in arm_names:
        _, _, aug, _ = ARMS[name]
        if aug is not None and id(aug) not in aug_te:
            e, te_lab = collect_embeddings(aug, test_loader)
            aug_te[id(aug)] = e
    if te_lab is None:
        _, te_lab = collect_embeddings(sup, test_loader)
    te_seen = np.isin(te_lab, seen)
    te_sig = te_lab == args.holdout

    # ----- pre-discovery power (constant in f) -------------------------------
    print("\n===== pre-discovery power =====")
    pre_power = {}
    for name in arm_names:
        net, means, aug, cents = ARMS[name]
        e, _ = collect_embeddings(net, test_loader)
        anchors = means[seen]
        if aug is not None:
            e = np.concatenate([e, aug_te[id(aug)]], axis=1)
            anchors = torch.cat([anchors, cents], dim=1)
        d = torch.cdist(torch.as_tensor(e, device=DEVICE),
                        torch.as_tensor(anchors, device=DEVICE))
        s = d.min(1).values.cpu().numpy()
        pre_power[name] = power_at_alpha(s[te_seen], s[te_sig], args.alpha)
        print(f"  [{name:<14}] power={pre_power[name]:.4f}")

    # ----- post-discovery power per injected fraction ------------------------
    power = {name: [] for name in arm_names}
    for i_f, f in enumerate(fractions):
        n_sig = int(round(f * len(seen_idx) / (1.0 - f)))
        rng = np.random.default_rng(args.seed * 1000 + i_f)
        inj = rng.choice(sig_idx_all, size=min(n_sig, len(sig_idx_all)),
                         replace=False)
        idx = np.concatenate([seen_idx, inj])
        sub = Subset(base, idx.tolist())
        tel = DataLoader(sub, batch_size=256, shuffle=False, num_workers=2)
        print(f"\n===== fraction f={f} ({len(inj)} injected deer) =====")
        for name in arm_names:
            net, means, aug, cents = ARMS[name]
            bb = copy.deepcopy(net)
            if aug is None:
                cur_means, _ = run_discovery(
                    bb, means.clone(), base_ds=sub, train_eval_loader=tel,
                    test_loader=test_loader, seen=seen, holdouts=holdouts,
                    dataset_name=ds, rep_weight=cfg["rep_weight"],
                    sigreg_weight=cfg["sigreg_weight"],
                    n_slices=cfg["n_slices"], rounds=args.rounds,
                    ft_epochs=ft_ep, names=CIFAR_NAMES, seed=args.seed)
                zt = torch.as_tensor(collect_embeddings(bb, test_loader)[0],
                                     device=DEVICE)
                d_seen = torch.cdist(zt, cur_means[seen]).min(1).values
                d_disc = torch.cdist(zt, cur_means[n_cls:]).min(1).values
            else:
                _, extras = exp28.run_concat_discovery(
                    bb, aug, means.clone(), cents, base=sub,
                    dim=cfg["emb_dim"], train_eval_loader=tel,
                    test_loader=test_loader, seen=seen, holdouts=holdouts,
                    cfg=cfg, rounds=args.rounds, ft_epochs=ft_ep,
                    names=CIFAR_NAMES, seed=args.seed)
                cur_means, disc_ssl = extras["cur_means"], extras["disc_ssl"]
                zt = torch.cat(
                    [torch.as_tensor(collect_embeddings(bb, test_loader)[0],
                                     device=DEVICE),
                     torch.as_tensor(aug_te[id(aug)], device=DEVICE)], dim=1)
                seen_anc = torch.cat([cur_means[seen], cents], dim=1)
                disc_anc = torch.cat([cur_means[n_cls:], disc_ssl], dim=1)
                d_seen = torch.cdist(zt, seen_anc).min(1).values
                d_disc = torch.cdist(zt, disc_anc).min(1).values
            s = (d_seen - d_disc).cpu().numpy()
            p = power_at_alpha(s[te_seen], s[te_sig], args.alpha)
            power[name].append(p)
            print(f"  [{name:<14}] f={f}: power={p:.4f}")

    # ----- report ------------------------------------------------------------
    print(f"\n===== EXP30 POWER SUMMARY (alpha={args.alpha}) =====")
    hdr = f"  {'arm':<16}{'pre':>8}" + "".join(f"{f:>9}" for f in fractions)
    print(hdr)
    for name in arm_names:
        print(f"  {name:<16}{pre_power[name]:>8.3f}"
              + "".join(f"{p:>9.3f}" for p in power[name]))

    plt.figure(figsize=(8, 6.5))
    for name in arm_names:
        c = COLORS[name]
        plt.plot(fractions, power[name], "-o", color=c, lw=2, ms=6,
                 label=f"{name} post")
        plt.axhline(pre_power[name], color=c, lw=1.2, ls="--", alpha=0.6)
    plt.xscale("log")
    plt.xlabel("injected anomaly fraction")
    plt.ylabel(f"power at alpha={args.alpha}")
    plt.title("exp30: hidden-signal power vs injection fraction\n"
              "(solid = post-discovery margin, dashed = pre-discovery "
              "distance, constant in f)")
    plt.grid(alpha=0.25, which="both")
    plt.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(plot_path(f"exp30_power_curves_{cfg['emb_dim']}d.png"), dpi=150)
    print("  saved "
          + plot_path(f"exp30_power_curves_{cfg['emb_dim']}d.png"))
    print("Done.")


if __name__ == "__main__":
    main()
