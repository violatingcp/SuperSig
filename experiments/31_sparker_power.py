"""
Experiment 31: SparKer-style dataset-level power curves (arXiv:2511.03095)
on the exp-29 latent spaces -- the redo of exp 30 with the paper's score.

Instead of per-event thresholds, each point is a hypothesis test: a data
sample D (N_D events: seen test images + deer injected at fraction f) is
tested against an anomaly-free reference R (seen train embeddings) with the
sparse Gaussian-kernel Neyman-Pearson ensemble (supersig.sparker).  Per-sigma
p-values are calibrated on anomaly-free toys, aggregated across scales, and
power = fraction of signal toys whose aggregate exceeds the (1-alpha) null
quantile.  Clopper-Pearson 68% intervals.

  pre  : the arm's frozen embedding (one null calibration per arm)
  post : the space after the settled discovery has run on the injected
         train stream (per arm x fraction, as exp 30)

    python experiments/31_sparker_power.py
    python experiments/31_sparker_power.py --quick --fractions 0.01,0.1
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
from supersig.sparker import (np_test_stats, aggregate_pvalues,
                              clopper_pearson, median_pairwise)
from supersig.train import (train_sigreg_ssl, train_sigreg_hybrid,
                            train_sigreg_hybrid_aug, train_sigreg_residual_ssl,
                            train_supcon, train_simclr, collect_embeddings)

exp28 = importlib.import_module("28_concat_residual")

CIFAR_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]
COLORS = {"sup->res": "#2a78d6", "ssl->supres": "#1baf7a", "joint": "#eda100",
          "sup": "#008300", "supcon": "#4a3aa7", "supcon+simclr": "#e34948"}


def toy_indices(rng, n_bg_pool, n_sig_pool, N_D, n_sig):
    """Index draws (into the bg / sig pools) for one toy realization.
    Falls back to bootstrap (with replacement) if a pool is smaller than
    the requested draw (only happens in --quick runs)."""
    n_bg = N_D - n_sig
    bg = rng.choice(n_bg_pool, size=n_bg, replace=n_bg > n_bg_pool)
    sig = (rng.choice(n_sig_pool, size=n_sig, replace=n_sig > n_sig_pool)
           if n_sig > 0 else np.empty(0, dtype=int))
    return bg, sig


def run_test_battery(bg_pool, sig_pool, R, fractions, N_D, n_null, n_sig_toys,
                     alpha, seed, sparker_kw, tag=""):
    """
    Null calibration + power per fraction in ONE fixed space.
    bg_pool/sig_pool/R: torch tensors on DEVICE.  Returns
    (powers, (lo, hi) bands, null aggregate threshold diagnostics).
    sparker_kw may carry a fixed "sigma0" (e.g. 1.0 with sigma_ratio=1.0 and
    n_checkpoints=1 for a fixed-width, no-annealing test); default is the
    median-pairwise heuristic of the background pool.
    """
    sparker_kw = dict(sparker_kw)
    sigma0 = sparker_kw.pop("sigma0", None) or median_pairwise(bg_pool,
                                                               seed=seed)
    rng = np.random.default_rng(seed)
    null_ts = []
    for i in range(n_null):
        bg, _ = toy_indices(rng, len(bg_pool), len(sig_pool), N_D, 0)
        D = bg_pool[torch.as_tensor(bg, device=DEVICE)]
        null_ts.append(np_test_stats(D, R, sigma0=sigma0, seed=seed + i,
                                     **sparker_kw))
    null_ts = np.array(null_ts)
    null_agg = np.array([
        aggregate_pvalues(null_ts[i],
                          np.delete(null_ts, i, axis=0))
        for i in range(n_null)])
    thr = np.quantile(null_agg, 1.0 - alpha)
    powers, bands = [], []
    for f in fractions:
        n_sig = int(round(f * N_D))
        det = 0
        for j in range(n_sig_toys):
            bg, sg = toy_indices(rng, len(bg_pool), len(sig_pool), N_D, n_sig)
            D = torch.cat([bg_pool[torch.as_tensor(bg, device=DEVICE)],
                           sig_pool[torch.as_tensor(sg, device=DEVICE)]])
            ts = np_test_stats(D, R, sigma0=sigma0,
                               seed=seed + 7919 + j, **sparker_kw)
            agg = aggregate_pvalues(ts, null_ts)
            det += int(agg > thr)
        p = det / n_sig_toys
        powers.append(p)
        bands.append(clopper_pearson(det, n_sig_toys))
        print(f"    {tag} f={f}: power={p:.3f} "
              f"[{bands[-1][0]:.3f},{bands[-1][1]:.3f}] ({det}/{n_sig_toys})")
    return powers, bands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--emb-dim", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--fractions", default="0.001,0.003,0.01,0.03,0.1")
    ap.add_argument("--arms", default=None)
    ap.add_argument("--n-d", type=int, default=5000)
    ap.add_argument("--n-null", type=int, default=None,
                    help="null toys (default 200 pre / 100 post)")
    ap.add_argument("--n-sig-toys", type=int, default=None,
                    help="signal toys per fraction (default 50)")
    ap.add_argument("--kernels", type=int, default=16)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--no-post", action="store_true",
                    help="skip the post-discovery battery")
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
    n_null_pre = args.n_null or (20 if args.quick else 200)
    n_null_post = args.n_null or (20 if args.quick else 100)
    n_sig_toys = args.n_sig_toys or (10 if args.quick else 50)
    n_cls = cfg["n_classes"]
    sparker_kw = dict(M=args.kernels, steps=args.steps)
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
    print(f"exp31 [sparker] emb_dim={cfg['emb_dim']} holdout={sorted(holdouts)}"
          f" alpha={args.alpha} N_D={args.n_d} M={args.kernels}"
          f" nulls={n_null_pre}/{n_null_post} sig-toys={n_sig_toys}")

    def cents_of(net):
        e, l = collect_embeddings(net, train_eval_loader)
        m = np.isin(l, seen)
        return exp28.class_centroids(e[m], l[m], seen)

    # ----- networks (identical seeds/config to exps 29/30) -------------------
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
                               emb_dim=cfg["emb_dim"], n_classes=n_cls).clone()
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

    def space_embs(net, aug, loader):
        e, l = collect_embeddings(net, loader)
        if aug is not None:
            ea, _ = collect_embeddings(aug, loader)
            e = np.concatenate([e, ea], axis=1)
        return e, l

    # pools per arm (pre space): reference from train seen, toys from test
    def pools(net, aug):
        tr, trl = space_embs(net, aug, train_eval_loader)
        te, tel = space_embs(net, aug, test_loader)
        R = torch.as_tensor(tr[np.isin(trl, seen)][:20000], dtype=torch.float32,
                            device=DEVICE)
        bg = torch.as_tensor(te[np.isin(tel, seen)], dtype=torch.float32,
                             device=DEVICE)
        sg = torch.as_tensor(te[tel == args.holdout], dtype=torch.float32,
                             device=DEVICE)
        return R, bg, sg

    print("\n===== PRE-DISCOVERY sparker battery =====")
    pre_power, pre_band = {}, {}
    for name in arm_names:
        net, means, aug, cents = ARMS[name]
        R, bg, sg = pools(net, aug)
        print(f"  [{name}]")
        pre_power[name], pre_band[name] = run_test_battery(
            bg, sg, R, fractions, args.n_d, n_null_pre, n_sig_toys,
            args.alpha, args.seed, sparker_kw, tag="pre")

    post_power, post_band = ({n: [] for n in arm_names},
                             {n: [] for n in arm_names})
    if not args.no_post:
        for i_f, f in enumerate(fractions):
            n_inj = int(round(f * len(seen_idx) / (1.0 - f)))
            rng = np.random.default_rng(args.seed * 1000 + i_f)
            inj = rng.choice(sig_idx_all, size=min(n_inj, len(sig_idx_all)),
                             replace=False)
            idx = np.concatenate([seen_idx, inj])
            sub = Subset(base, idx.tolist())
            tel = DataLoader(sub, batch_size=256, shuffle=False, num_workers=2)
            print(f"\n===== POST-DISCOVERY battery, f={f} "
                  f"({len(inj)} injected) =====")
            for name in arm_names:
                net, means, aug, cents = ARMS[name]
                bb = copy.deepcopy(net)
                if aug is None:
                    run_discovery(bb, means.clone(), base_ds=sub,
                                  train_eval_loader=tel,
                                  test_loader=test_loader, seen=seen,
                                  holdouts=holdouts, dataset_name=ds,
                                  rep_weight=cfg["rep_weight"],
                                  sigreg_weight=cfg["sigreg_weight"],
                                  n_slices=cfg["n_slices"],
                                  rounds=args.rounds, ft_epochs=ft_ep,
                                  names=CIFAR_NAMES, seed=args.seed)
                else:
                    exp28.run_concat_discovery(
                        bb, aug, means.clone(), cents, base=sub,
                        dim=cfg["emb_dim"], train_eval_loader=tel,
                        test_loader=test_loader, seen=seen, holdouts=holdouts,
                        cfg=cfg, rounds=args.rounds, ft_epochs=ft_ep,
                        names=CIFAR_NAMES, seed=args.seed)
                R, bg, sg = pools(bb, aug)
                print(f"  [{name}] (post, f={f})")
                p, b = run_test_battery(bg, sg, R, [f], args.n_d, n_null_post,
                                        n_sig_toys, args.alpha,
                                        args.seed + i_f, sparker_kw,
                                        tag="post")
                post_power[name].append(p[0])
                post_band[name].append(b[0])

    print(f"\n===== EXP31 SPARKER POWER SUMMARY (alpha={args.alpha}) =====")
    hdr = f"  {'arm':<16}{'kind':>6}" + "".join(f"{f:>9}" for f in fractions)
    print(hdr)
    for name in arm_names:
        print(f"  {name:<16}{'pre':>6}"
              + "".join(f"{p:>9.3f}" for p in pre_power[name]))
        if post_power[name]:
            print(f"  {name:<16}{'post':>6}"
                  + "".join(f"{p:>9.3f}" for p in post_power[name]))

    plt.figure(figsize=(8, 6.5))
    for name in arm_names:
        c = COLORS[name]
        lo = [b[0] for b in pre_band[name]]
        hi = [b[1] for b in pre_band[name]]
        plt.plot(fractions, pre_power[name], "--o", color=c, lw=1.4, ms=5,
                 alpha=0.75, label=f"{name} pre")
        plt.fill_between(fractions, lo, hi, color=c, alpha=0.10)
        if post_power[name]:
            lo = [b[0] for b in post_band[name]]
            hi = [b[1] for b in post_band[name]]
            plt.plot(fractions, post_power[name], "-o", color=c, lw=2, ms=6,
                     label=f"{name} post")
            plt.fill_between(fractions, lo, hi, color=c, alpha=0.15)
    plt.xscale("log")
    plt.axhline(args.alpha, color="gray", lw=1, ls=":")
    plt.xlabel("injected anomaly fraction")
    plt.ylabel(f"power at alpha={args.alpha}")
    plt.title("exp31: SparKer dataset-level power vs injection fraction\n"
              "(sparse-kernel NP test, arXiv:2511.03095; bands = "
              "Clopper-Pearson 68%)")
    plt.grid(alpha=0.25, which="both")
    plt.legend(loc="upper left", fontsize=8, ncol=2)
    plt.tight_layout()
    out = plot_path(f"exp31_sparker_power_{cfg['emb_dim']}d.png")
    plt.savefig(out, dpi=150)
    print("  saved " + out)
    os.makedirs(os.path.join("logs", "exp31"), exist_ok=True)
    np.savez(os.path.join("logs", "exp31", "power_data.npz"),
             fractions=np.array(fractions),
             **{f"{n}_pre": np.array(pre_power[n]) for n in arm_names},
             **{f"{n}_post": np.array(post_power[n]) for n in arm_names
                if post_power[n]})
    print("Done.")


if __name__ == "__main__":
    main()
