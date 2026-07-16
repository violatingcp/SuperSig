"""
Experiment 32: dataset-level power curves with Mahalanobis and MMD statistics
-- exp 31's SparKer testing machinery (toy-calibrated null, multi-scale
min/mean-log-p aggregation, alpha on the null quantile, Clopper-Pearson
intervals) with closed-form statistics in place of the trained kernel
ensemble:

  maha : mean over the data sample of the per-event min Mahalanobis distance
         to the seen-class Gaussians (per-class covariances, shrinkage 0.1),
         fitted once on the reference; single-scale.
  mmd  : unbiased MMD^2(D, R) with Gaussian kernels at 1/2x, 1x, 2x the
         median-heuristic bandwidth; three scales aggregated like SparKer's
         sigma checkpoints.

Same arms, injection grid, toy counts and seeds as exp 31; the discovery
reruns per (arm, fraction) provide the post spaces, and both statistics are
evaluated on each.  Post test-pool embeddings are archived.

    python experiments/32_maha_mmd_power.py
    python experiments/32_maha_mmd_power.py --quick --fractions 0.01,0.1
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
from supersig.metrics import mahalanobis_novelty
from supersig.recipes import supervised_embedding, recipe
from supersig.discovery import run_discovery
from supersig.sparker import (aggregate_pvalues, clopper_pearson,
                              median_pairwise, krr_term, mmd2_multi_stats)
from supersig.train import (train_sigreg_ssl, train_sigreg_hybrid,
                            train_sigreg_hybrid_aug, train_sigreg_residual_ssl,
                            train_supcon, train_simclr, collect_embeddings)

exp28 = importlib.import_module("28_concat_residual")
exp31 = importlib.import_module("31_sparker_power")

CIFAR_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]
COLORS = exp31.COLORS


def battery(stats_fn, n_bg_pool, n_sig_pool, fractions, N_D, n_null,
            n_sig_toys, alpha, seed, tag=""):
    """Toy-calibrated power scan; stats_fn(bg_idx, sig_idx) -> [stats]."""
    rng = np.random.default_rng(seed)
    null_ts = []
    for i in range(n_null):
        bg, sg = exp31.toy_indices(rng, n_bg_pool, n_sig_pool, N_D, 0)
        null_ts.append(stats_fn(bg, sg))
    null_ts = np.array(null_ts)
    null_agg = np.array([
        aggregate_pvalues(null_ts[i], np.delete(null_ts, i, axis=0))
        for i in range(n_null)])
    thr = np.quantile(null_agg, 1.0 - alpha)
    powers, bands = [], []
    for f in fractions:
        n_sig = int(round(f * N_D))
        det = 0
        for j in range(n_sig_toys):
            bg, sg = exp31.toy_indices(rng, n_bg_pool, n_sig_pool, N_D, n_sig)
            agg = aggregate_pvalues(stats_fn(bg, sg), null_ts)
            det += int(agg > thr)
        p = det / n_sig_toys
        powers.append(p)
        bands.append(clopper_pearson(det, n_sig_toys))
        print(f"    {tag} f={f}: power={p:.3f} "
              f"[{bands[-1][0]:.3f},{bands[-1][1]:.3f}] ({det}/{n_sig_toys})")
    return powers, bands


def make_stats_fns(tr, trl, te, tel, seen, holdout, seed):
    """Build (maha_fn, mmd_fn, n_bg, n_sig) for one space."""
    bg_mask = np.isin(tel, seen)
    sig_mask = tel == holdout
    # mahalanobis: per-event scores precomputed once on the whole test pool
    _, pc, _ = mahalanobis_novelty(tr, trl, te, seen)
    s_bg, s_sig = pc[bg_mask], pc[sig_mask]

    def maha_fn(bg_idx, sig_idx):
        s = (np.concatenate([s_bg[bg_idx], s_sig[sig_idx]])
             if len(sig_idx) else s_bg[bg_idx])
        return [float(s.mean())]

    # mmd: fixed reference subsample + bandwidths from the background pool
    bg_t = torch.as_tensor(te[bg_mask], dtype=torch.float32, device=DEVICE)
    sig_t = torch.as_tensor(te[sig_mask], dtype=torch.float32, device=DEVICE)
    R_pool = tr[np.isin(trl, seen)]
    g = np.random.default_rng(seed)
    R = torch.as_tensor(
        R_pool[g.choice(len(R_pool), size=min(5000, len(R_pool)),
                        replace=False)], dtype=torch.float32, device=DEVICE)
    med = median_pairwise(bg_t, seed=seed)
    sigmas = [0.5 * med, med, 2.0 * med]
    krr = krr_term(R, sigmas)

    def mmd_fn(bg_idx, sig_idx):
        D = (torch.cat([bg_t[torch.as_tensor(bg_idx, device=DEVICE)],
                        sig_t[torch.as_tensor(sig_idx, device=DEVICE)]])
             if len(sig_idx) else bg_t[torch.as_tensor(bg_idx, device=DEVICE)])
        return mmd2_multi_stats(D, R, sigmas, krr)

    return maha_fn, mmd_fn, int(bg_mask.sum()), int(sig_mask.sum())


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
    ap.add_argument("--n-null", type=int, default=None)
    ap.add_argument("--n-sig-toys", type=int, default=None)
    ap.add_argument("--no-post", action="store_true")
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
    print(f"exp32 [maha+mmd] emb_dim={cfg['emb_dim']} holdout={sorted(holdouts)}"
          f" alpha={args.alpha} N_D={args.n_d}"
          f" nulls={n_null_pre}/{n_null_post} sig-toys={n_sig_toys}")

    def cents_of(net):
        e, l = collect_embeddings(net, train_eval_loader)
        m = np.isin(l, seen)
        return exp28.class_centroids(e[m], l[m], seen)

    # ----- networks (identical to exps 29/30/31) -----------------------------
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
    STATS = ["maha", "mmd"]

    def space_embs(net, aug, loader):
        e, l = collect_embeddings(net, loader)
        if aug is not None:
            ea, _ = collect_embeddings(aug, loader)
            e = np.concatenate([e, ea], axis=1)
        return e, l

    def run_both(net, aug, seed, tag):
        tr, trl = space_embs(net, aug, train_eval_loader)
        te, tel = space_embs(net, aug, test_loader)
        maha_fn, mmd_fn, n_bg, n_sig = make_stats_fns(
            tr, trl, te, tel, seen, args.holdout, seed)
        return te, tel, maha_fn, mmd_fn, n_bg, n_sig

    print("\n===== PRE batteries (maha + mmd) =====")
    pre_power = {s: {} for s in STATS}
    pre_band = {s: {} for s in STATS}
    for name in arm_names:
        net, means, aug, cents = ARMS[name]
        _, _, maha_fn, mmd_fn, n_bg, n_sig = run_both(net, aug, args.seed,
                                                      name)
        print(f"  [{name}] maha")
        pre_power["maha"][name], pre_band["maha"][name] = battery(
            maha_fn, n_bg, n_sig, fractions, args.n_d, n_null_pre,
            n_sig_toys, args.alpha, args.seed, tag="pre-maha")
        print(f"  [{name}] mmd")
        pre_power["mmd"][name], pre_band["mmd"][name] = battery(
            mmd_fn, n_bg, n_sig, fractions, args.n_d, n_null_pre,
            n_sig_toys, args.alpha, args.seed, tag="pre-mmd")

    post_power = {s: {n: [] for n in arm_names} for s in STATS}
    post_band = {s: {n: [] for n in arm_names} for s in STATS}
    post_te = {}
    if not args.no_post:
        for i_f, f in enumerate(fractions):
            n_inj = int(round(f * len(seen_idx) / (1.0 - f)))
            rng = np.random.default_rng(args.seed * 1000 + i_f)
            inj = rng.choice(sig_idx_all, size=min(n_inj, len(sig_idx_all)),
                             replace=False)
            idx = np.concatenate([seen_idx, inj])
            sub = Subset(base, idx.tolist())
            tel_loader = DataLoader(sub, batch_size=256, shuffle=False,
                                    num_workers=2)
            print(f"\n===== POST batteries, f={f} ({len(inj)} injected) =====")
            for name in arm_names:
                net, means, aug, cents = ARMS[name]
                bb = copy.deepcopy(net)
                if aug is None:
                    run_discovery(bb, means.clone(), base_ds=sub,
                                  train_eval_loader=tel_loader,
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
                        dim=cfg["emb_dim"], train_eval_loader=tel_loader,
                        test_loader=test_loader, seen=seen, holdouts=holdouts,
                        cfg=cfg, rounds=args.rounds, ft_epochs=ft_ep,
                        names=CIFAR_NAMES, seed=args.seed)
                te, tel, maha_fn, mmd_fn, n_bg, n_sig = run_both(
                    bb, aug, args.seed + i_f, name)
                post_te[f"{name}_f{f}"] = te.astype(np.float16)
                print(f"  [{name}] (post, f={f}) maha")
                p, b = battery(maha_fn, n_bg, n_sig, [f], args.n_d,
                               n_null_post, n_sig_toys, args.alpha,
                               args.seed + i_f, tag="post-maha")
                post_power["maha"][name].append(p[0])
                post_band["maha"][name].append(b[0])
                print(f"  [{name}] (post, f={f}) mmd")
                p, b = battery(mmd_fn, n_bg, n_sig, [f], args.n_d,
                               n_null_post, n_sig_toys, args.alpha,
                               args.seed + i_f, tag="post-mmd")
                post_power["mmd"][name].append(p[0])
                post_band["mmd"][name].append(b[0])

    for stat in STATS:
        print(f"\n===== EXP32 {stat.upper()} POWER SUMMARY "
              f"(alpha={args.alpha}) =====")
        print(f"  {'arm':<16}{'kind':>6}"
              + "".join(f"{f:>9}" for f in fractions))
        for name in arm_names:
            print(f"  {name:<16}{'pre':>6}"
                  + "".join(f"{p:>9.3f}" for p in pre_power[stat][name]))
            if post_power[stat][name]:
                print(f"  {name:<16}{'post':>6}"
                      + "".join(f"{p:>9.3f}" for p in post_power[stat][name]))

        plt.figure(figsize=(8, 6.5))
        for name in arm_names:
            c = COLORS[name]
            plt.plot(fractions, pre_power[stat][name], "--o", color=c, lw=1.4,
                     ms=5, alpha=0.75, label=f"{name} pre")
            lo = [b[0] for b in pre_band[stat][name]]
            hi = [b[1] for b in pre_band[stat][name]]
            plt.fill_between(fractions, lo, hi, color=c, alpha=0.10)
            if post_power[stat][name]:
                plt.plot(fractions, post_power[stat][name], "-o", color=c,
                         lw=2, ms=6, label=f"{name} post")
                lo = [b[0] for b in post_band[stat][name]]
                hi = [b[1] for b in post_band[stat][name]]
                plt.fill_between(fractions, lo, hi, color=c, alpha=0.15)
        plt.xscale("log")
        plt.axhline(args.alpha, color="gray", lw=1, ls=":")
        plt.xlabel("injected anomaly fraction")
        plt.ylabel(f"power at alpha={args.alpha}")
        title = ("mean min-Mahalanobis statistic" if stat == "maha"
                 else "multi-bandwidth unbiased MMD^2 statistic")
        plt.title(f"exp32: dataset-level power vs injection fraction\n"
                  f"({title}; exp-31 testing machinery; CP 68% bands)")
        plt.grid(alpha=0.25, which="both")
        plt.legend(loc="upper left", fontsize=8, ncol=2)
        plt.tight_layout()
        out = plot_path(f"exp32_{stat}_power_{cfg['emb_dim']}d.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print("  saved " + out)

    os.makedirs(os.path.join("logs", "exp32"), exist_ok=True)
    np.savez(os.path.join("logs", "exp32", "power_data.npz"),
             fractions=np.array(fractions),
             **{f"{s}_{n}_pre": np.array(pre_power[s][n])
                for s in STATS for n in arm_names},
             **{f"{s}_{n}_post": np.array(post_power[s][n])
                for s in STATS for n in arm_names if post_power[s][n]})
    np.savez_compressed(os.path.join("logs", "exp32", "post_te_embs.npz"),
                        **post_te)
    print("Done.")


if __name__ == "__main__":
    main()
