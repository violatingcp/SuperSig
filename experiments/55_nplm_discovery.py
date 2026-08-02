"""
Experiment 55: the exp-33/36 discovery protocol on the NPLM spaces.

Exps 50/53 evaluated the NPLM arms pre-discovery only.  This runs the settled
open-world loop on them (discovery.run_discovery: pool events past the 0.95
seen-distance quantile, BIC k-means, pseudo-label, fine-tune with the proto/
repulse hybrid, 2 rounds) for three NPLM spaces per dataset:

  nplm_bilinear     exp-50 corner, instance/bilinear/nplm + global sigreg
  nplm_sup_dist     exp-50 corner, supervised/distance/nplm + global sigreg
  nplm_dist_sup_cw  exp-53 corner, supervised/distance/nplm + classwise sigreg

Arms are retrained bit-for-bit with their original per-arm seeds (no
checkpoints were saved).  Part A: natural-fraction discovery -- purity,
anchors, margin, probe pre/post.  Part B: exp-33 POST power grid -- per
injected fraction, one discovery rerun, then per-event margin (d_seen -
d_disc), SparKer, Mahalanobis and MMD batteries on the updated space.  The
discovery fine-tune is the settled supervised-SIGReg loop, so only the
starting space differs from exps 33/36; their post curves overlay from the
exp-33 npz.

    python experiments/55_nplm_discovery.py --dataset cifar10
    python experiments/55_nplm_discovery.py --dataset cifar100 --quick
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
from supersig.data import (get_cifar_loaders, cifar_two_view_loader,
                           cifar_two_view_balanced_loader, _cifar_spec)
from supersig.losses import make_anchors
from supersig.models import CIFARResNetBackbone
from supersig.recipes import recipe
from supersig.discovery import run_discovery
from supersig.train import collect_embeddings

exp28 = importlib.import_module("28_concat_residual")
exp29 = importlib.import_module("29_residual_finetune")
exp30 = importlib.import_module("30_power_curves")
exp31 = importlib.import_module("31_sparker_power")
exp32 = importlib.import_module("32_maha_mmd_power")
exp34h = importlib.import_module("34h_hybrid_nplm_cifar")
exp50 = importlib.import_module("50_nplm_cifar10_suite")
exp53 = importlib.import_module("53_nplm_classwise")

STATS = ["perevent", "sparker", "maha", "mmd"]
ARMS = ["nplm_bilinear", "nplm_sup_dist", "nplm_dist_sup_cw"]
COLORS = {"nplm_bilinear": "#1baf7a", "nplm_sup_dist": "#8c2d9e",
          "nplm_dist_sup_cw": "#d62728"}
REF_ARMS = {"sup": ":", "supcon": "--"}


def train_arm(name, ds, cfg, args, con_ep, holdouts):
    """Reproduce an exp-50 / exp-53 NPLM arm with its original seed."""
    if name in exp50.ARMS:
        kind, spec, labeled = exp50.ARMS[name]
        seed = args.seed + 20 + list(exp50.ARMS).index(name)
        torch.manual_seed(seed); np.random.seed(seed)
        net = CIFARResNetBackbone(args.dim, arch=cfg["arch"],
                                  pretrain=ds).to(DEVICE)
        loader = cifar_two_view_loader(quick=args.quick, labeled=labeled,
                                       holdout=holdouts, dataset=ds)
        exp34h.train_hybrid(net, loader, con_ep, spec, labeled,
                            lam=args.lam, n_slices=cfg["n_slices"])
        return net
    positives, critic = exp53.ARMS[name]
    seed = args.seed + 20 + list(exp53.ARMS).index(name)
    torch.manual_seed(seed); np.random.seed(seed)
    net = CIFARResNetBackbone(args.dim, arch=cfg["arch"],
                              pretrain=ds).to(DEVICE)
    loader = cifar_two_view_balanced_loader(ds, holdout=holdouts,
                                            quick=args.quick)
    cw_means = make_anchors(cfg["pair_dist"] / math.sqrt(2.0),
                            emb_dim=args.dim,
                            n_classes=cfg["n_classes"]).detach()
    exp53.train_nplm_classwise(net, loader, con_ep, positives, critic,
                               cw_means, tau=args.tau, lam=args.lam,
                               n_slices=cfg["n_slices"])
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--holdout", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--fractions", default=None)
    ap.add_argument("--n-d", type=int, default=5000)
    ap.add_argument("--kernels", type=int, default=16)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--sparker-sigma", type=float, default=1.0)
    ap.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    ap.add_argument("--out-tag", default="")
    args = ap.parse_args()
    ds = args.dataset
    if args.fractions is None:
        args.fractions = ("0.001,0.003,0.01,0.02,0.03,0.1" if ds == "cifar10"
                          else "0.001,0.003,0.01,0.02,0.05")

    cfg = recipe(ds, emb_dim=args.dim)
    n_cls = cfg["n_classes"]
    holdouts = {args.holdout}
    seen = [c for c in range(n_cls) if c not in holdouts]
    con_ep = args.epochs or (2 if args.quick else 20)
    ft_ep = 1 if args.quick else cfg["ft_epochs"]
    fractions = [float(x) for x in args.fractions.split(",")]
    n_null_post = 20 if args.quick else 100
    n_sig_toys = 10 if args.quick else 50
    sparker_kw = dict(M=args.kernels, steps=args.steps)
    if args.sparker_sigma > 0:
        sparker_kw.update(sigma0=args.sparker_sigma, sigma_ratio=1.0,
                          n_checkpoints=1)
    names = (exp29.CIFAR_NAMES if ds == "cifar10"
             else [str(c) for c in range(n_cls)])
    print(f"exp55 [{ds}] NPLM discovery, dim={args.dim}, epochs={con_ep}, "
          f"holdout={sorted(holdouts)}, rounds={args.rounds}, "
          f"arms={args.arms}")

    train_loader, test_loader = get_cifar_loaders(quick=args.quick, dataset=ds)
    train_eval_loader = DataLoader(train_loader.dataset, batch_size=256,
                                   shuffle=False, num_workers=2)
    cls, plain, _ = _cifar_spec(ds)
    base = cls(DATA_DIR, train=True, download=True, transform=plain)
    base_targets = np.array(base.targets)
    n_base = 8000 if args.quick else len(base)
    seen_idx = np.where(np.isin(base_targets[:n_base], seen))[0]
    sig_idx_all = np.where(np.isin(base_targets[:n_base], list(holdouts)))[0]

    # ===== train arms, natural-fraction discovery ===========================
    nets, means_of, trains, tests, hist, probe = {}, {}, {}, {}, {}, {}
    tr_lab = te_lab = None
    for name in args.arms:
        print(f"\n===== training: {name} =====")
        nets[name] = train_arm(name, ds, cfg, args, con_ep, holdouts)
        tr, tr_lab = collect_embeddings(nets[name], train_eval_loader)
        te, te_lab = collect_embeddings(nets[name], test_loader)
        trains[name], tests[name] = tr, te
        m = np.isin(tr_lab, seen)
        cents = exp28.class_centroids(tr[m], tr_lab[m], seen)
        means_of[name] = exp28.fill_means(cents, seen, cfg).detach()

        print(f"\n----- natural discovery: {name} -----")
        bb = copy.deepcopy(nets[name])
        _, hist[name] = run_discovery(
            bb, means_of[name].clone(), base_ds=base,
            train_eval_loader=train_eval_loader, test_loader=test_loader,
            seen=seen, holdouts=holdouts, dataset_name=ds,
            rep_weight=cfg["rep_weight"], sigreg_weight=cfg["sigreg_weight"],
            n_slices=cfg["n_slices"], rounds=args.rounds, ft_epochs=ft_ep,
            names=names, seed=args.seed)
        tr_post, _ = collect_embeddings(bb, train_eval_loader)
        te_post, _ = collect_embeddings(bb, test_loader)
        a_pre, _, _ = exp29.linear_probe_novelty(tr, tr_lab, te, te_lab,
                                                 holdouts)
        a_post, _, _ = exp29.linear_probe_novelty(tr_post, tr_lab, te_post,
                                                  te_lab, holdouts)
        probe[name] = (a_pre, a_post)
        print(f"  probe pre={a_pre:.4f} post={a_post:.4f}")
        del bb
        torch.cuda.empty_cache()

    # ===== POST power grid ==================================================
    post_power = {s: {n: [] for n in args.arms} for s in STATS}
    for i_f, f in enumerate(fractions):
        n_inj = int(round(f * len(seen_idx) / (1.0 - f)))
        rng = np.random.default_rng(args.seed * 1000 + i_f)
        if n_inj > len(sig_idx_all):
            print(f"  NOTE: requested {n_inj} injected but only "
                  f"{len(sig_idx_all)} available -- fraction clamped")
        inj = rng.choice(sig_idx_all, size=min(n_inj, len(sig_idx_all)),
                         replace=False)
        sub = Subset(base, np.concatenate([seen_idx, inj]).tolist())
        tel_loader = DataLoader(sub, batch_size=256, shuffle=False,
                                num_workers=2)
        print(f"\n===== POST grid, f={f} ({len(inj)} injected) =====")
        for name in args.arms:
            bb = copy.deepcopy(nets[name])
            cur_means, _ = run_discovery(
                bb, means_of[name].clone(), base_ds=sub,
                train_eval_loader=tel_loader, test_loader=test_loader,
                seen=seen, holdouts=holdouts, dataset_name=ds,
                rep_weight=cfg["rep_weight"],
                sigreg_weight=cfg["sigreg_weight"],
                n_slices=cfg["n_slices"], rounds=args.rounds,
                ft_epochs=ft_ep, names=names, seed=args.seed)
            te_post, tel_post = collect_embeddings(bb, test_loader)
            tr_post, trl_post = collect_embeddings(bb, train_eval_loader)
            zt = torch.as_tensor(te_post, dtype=torch.float32, device=DEVICE)
            d_seen = torch.cdist(zt, cur_means[seen]).min(1).values
            d_disc = (torch.cdist(zt, cur_means[n_cls:]).min(1).values
                      if cur_means.size(0) > n_cls else
                      torch.full_like(d_seen, float("inf")))
            bg_mask = np.isin(tel_post, seen)
            sig_mask = np.isin(tel_post, list(holdouts))
            s = (d_seen - d_disc).cpu().numpy()
            pe = exp30.power_at_alpha(s[bg_mask], s[sig_mask], args.alpha)
            post_power["perevent"][name].append(pe)
            print(f"  [{name}] per-event post f={f}: power={pe:.3f}")
            R = torch.as_tensor(tr_post[np.isin(trl_post, seen)][:20000],
                                dtype=torch.float32, device=DEVICE)
            bg = torch.as_tensor(te_post[bg_mask], dtype=torch.float32,
                                 device=DEVICE)
            sg = torch.as_tensor(te_post[sig_mask], dtype=torch.float32,
                                 device=DEVICE)
            print(f"  [{name}] sparker (post)")
            p, _ = exp31.run_test_battery(bg, sg, R, [f], args.n_d,
                                          n_null_post, n_sig_toys, args.alpha,
                                          args.seed + i_f, sparker_kw,
                                          tag="post-spk")
            post_power["sparker"][name].append(p[0])
            maha_fn, mmd_fn, n_bg, n_sig = exp32.make_stats_fns(
                tr_post, trl_post, te_post, tel_post, seen, holdouts,
                args.seed + i_f)
            print(f"  [{name}] maha (post)")
            p, _ = exp32.battery(maha_fn, n_bg, n_sig, [f], args.n_d,
                                 n_null_post, n_sig_toys, args.alpha,
                                 args.seed + i_f, tag="post-maha")
            post_power["maha"][name].append(p[0])
            print(f"  [{name}] mmd (post)")
            p, _ = exp32.battery(mmd_fn, n_bg, n_sig, [f], args.n_d,
                                 n_null_post, n_sig_toys, args.alpha,
                                 args.seed + i_f, tag="post-mmd")
            post_power["mmd"][name].append(p[0])
            del bb
            torch.cuda.empty_cache()

    # ===== report ===========================================================
    ref_path = os.path.join("logs", "exp33",
                            f"power_data_{ds}_16p16_k1.npz")
    ref = np.load(ref_path) if os.path.exists(ref_path) else None
    for stat in STATS:
        print(f"\n===== EXP55 {stat.upper()} POST POWER "
              f"(alpha={args.alpha}) =====")
        print(f"  {'arm':<18}" + "".join(f"{f:>9}" for f in fractions))
        for name in args.arms:
            print(f"  {name:<18}"
                  + "".join(f"{p:>9.3f}" for p in post_power[stat][name]))
        plt.figure(figsize=(8, 6.5))
        for name in args.arms:
            plt.plot(fractions, post_power[stat][name], "-o",
                     color=COLORS[name], lw=2, ms=5, label=f"{name} post")
        if ref is not None:
            for rname, style in REF_ARMS.items():
                k = f"{stat}_{rname}_post"
                if (k in ref.files and len(ref["fractions"]) ==
                        len(fractions) and
                        np.allclose(ref["fractions"], fractions)):
                    plt.plot(fractions, ref[k], style, color="black", lw=1.2,
                             alpha=0.6, label=f"{rname} post (exp33)")
        plt.xscale("log")
        plt.axhline(args.alpha, color="gray", lw=1, ls=":")
        plt.xlabel("injected anomaly fraction")
        plt.ylabel(f"power at alpha={args.alpha}")
        plt.title(f"exp55 NPLM discovery ({ds} {args.dim}d): "
                  f"{stat} post power")
        plt.grid(alpha=0.25, which="both")
        plt.legend(loc="upper left", fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(plot_path(f"exp55_{stat}_power_{ds}{args.out_tag}.png"),
                    dpi=150)
        plt.close()
        print("  saved "
              + plot_path(f"exp55_{stat}_power_{ds}{args.out_tag}.png"))

    print(f"\n===== EXP55 SUMMARY [{ds}] =====")
    for name in args.arms:
        print(f"  [{name:<16}] probe pre={probe[name][0]:.4f} "
              f"post={probe[name][1]:.4f}")
        for h in hist[name]:
            print(f"          round {h['round']}: purity={h['purity']:.3f} "
                  f"anchors={h['n_anchors']}  margin={h['margin']:.4f}  "
                  f"mean-anchor={h['mean_pc']:.4f}")
    os.makedirs(os.path.join("logs", "exp55"), exist_ok=True)
    np.savez(os.path.join("logs", "exp55",
                          f"discovery_nplm_{ds}{args.out_tag}.npz"),
             fractions=np.array(fractions), arms=np.array(args.arms),
             **{f"probe_{n}": np.array(probe[n]) for n in args.arms},
             **{f"{s}_{n}_post": np.array(post_power[s][n]) for s in STATS
                for n in args.arms})
    print("Done.")


if __name__ == "__main__":
    main()
