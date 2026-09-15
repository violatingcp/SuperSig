"""
Experiment 35: two JEPAMatch-inspired discovery-loop upgrades (paper
arXiv:2604.21046), evaluated on the CIFAR-100 16+16 post-discovery power
collapse they target (exp 34f/g finding: post power at f<=0.01 falls to ~0
because impure pseudo-labels drag background into the discovered region).

Stage A -- confidence-masked classwise SIGReg: pooled events enter the
  fine-tune (and centroid refresh) only if the proto-posterior probability
  of their assigned discovered anchor exceeds conf_thresh (JEPAMatch's M_i
  mask; unconfident events stay unlabeled).
Stage B -- Stage A + asymmetric variance annealing: the SIGReg target std of
  DISCOVERED classes anneals 1.0 -> 0.3 across each fine-tune while seen
  classes stay at sigma=1 (concentrate the anomaly, preserve calibration).

Arms: supcon+hybrid[lam5] and ss[lam5]+hybrid (the post-discovery champions).
Baselines: unmasked post curves from exps 34f/34g (identical seeds + toys,
loaded from logs/exp34/power_data_{hybrid,ss_hybrid}.npz).

    python experiments/35_confmask_anneal.py [--conf-thresh 0.5 --sigma-end 0.3]
    python experiments/35_confmask_anneal.py --quick --fractions 0.01,0.05
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import copy
import importlib
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from supersig.config import DATA_DIR, DEVICE, plot_path
from supersig.data import get_cifar_loaders, cifar_two_view_loader, _cifar_spec
from supersig.models import CIFARResNetBackbone
from supersig.recipes import recipe
from supersig.train import (train_supcon, train_simclr_sigreg,
                            train_supcon_sigreg, collect_embeddings)

exp28 = importlib.import_module("28_concat_residual")
exp29 = importlib.import_module("29_residual_finetune")
exp30 = importlib.import_module("30_power_curves")
exp31 = importlib.import_module("31_sparker_power")
exp32 = importlib.import_module("32_maha_mmd_power")

STATS = ["perevent", "sparker", "maha", "mmd"]
LOGDIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "logs", "exp34")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar10", "cifar100"],
                    default="cifar100")
    ap.add_argument("--holdout", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dim-half", type=int, default=16)
    ap.add_argument("--conf-thresh", type=float, default=0.5)
    ap.add_argument("--sigma-end", type=float, default=0.3)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--fractions", default=None,
                    help="defaults per dataset: cifar100 0.001..0.05, "
                         "cifar10 0.001..0.1")
    ap.add_argument("--n-d", type=int, default=5000)
    ap.add_argument("--kernels", type=int, default=16)
    ap.add_argument("--steps", type=int, default=300)
    args = ap.parse_args()
    ds = args.dataset
    if args.fractions is None:
        args.fractions = ("0.001,0.003,0.01,0.02,0.03,0.1" if ds == "cifar10"
                          else "0.001,0.003,0.01,0.02,0.05")
    cfgH = recipe(ds, emb_dim=args.dim_half)
    n_cls = cfgH["n_classes"]
    holdouts = {args.holdout}
    seen = [c for c in range(n_cls) if c not in holdouts]
    fractions = [float(x) for x in args.fractions.split(",")]
    sup_ep = 2 if args.quick else 10
    con_ep = 2 if args.quick else 20
    ft_ep = 1 if args.quick else cfgH["ft_epochs"]
    n_null_post = 20 if args.quick else 100
    n_sig_toys = 10 if args.quick else 50
    sparker_kw = dict(M=args.kernels, steps=args.steps)   # annealed widths
    names = (exp29.CIFAR_NAMES if ds == "cifar10"
             else [str(c) for c in range(n_cls)])
    STAGES = [("confmask", dict(conf_thresh=args.conf_thresh)),
              ("confmask+anneal", dict(conf_thresh=args.conf_thresh,
                                       disc_sigma_end=args.sigma_end))]
    print(f"exp35 [{ds}] conf-masked discovery + asymmetric annealing, "
          f"holdout={sorted(holdouts)}, thresh={args.conf_thresh}, "
          f"sigma_end={args.sigma_end}")

    train_loader, test_loader = get_cifar_loaders(quick=args.quick, dataset=ds)
    train_eval_loader = DataLoader(train_loader.dataset, batch_size=256,
                                   shuffle=False, num_workers=2)
    cls, plain, _ = _cifar_spec(ds)
    base = cls(DATA_DIR, train=True, download=True, transform=plain)
    base_targets = np.array(base.targets)
    n_base = 8000 if args.quick else len(base)
    seen_idx = np.where(np.isin(base_targets[:n_base], seen))[0]
    sig_idx_all = np.where(np.isin(base_targets[:n_base], list(holdouts)))[0]

    def cents_of(net):
        e, l = collect_embeddings(net, train_eval_loader)
        m = np.isin(l, seen)
        return exp28.class_centroids(e[m], l[m], seen)

    def backbone():
        return CIFARResNetBackbone(args.dim_half, arch=cfgH["arch"],
                                   pretrain=ds).to(DEVICE)

    def space_embs(net, aug, loader):
        e, l = collect_embeddings(net, loader)
        ea, _ = collect_embeddings(aug, loader)
        return np.concatenate([e, ea], axis=1), l

    # ----- networks (exp-34e/f/g seeds) -------------------------------------
    print("\n----- supcon16 -----")
    torch.manual_seed(args.seed + 15); np.random.seed(args.seed + 15)
    supcon = backbone()
    train_supcon(supcon, cifar_two_view_loader(quick=args.quick, labeled=True,
                                               holdout=holdouts, dataset=ds),
                 sup_ep)
    supcon_cents = cents_of(supcon)
    print("\n----- hybrid16 simclr+sigreg (lam=5) -----")
    torch.manual_seed(args.seed + 16); np.random.seed(args.seed + 16)
    hybrid = backbone()
    train_simclr_sigreg(hybrid, cifar_two_view_loader(quick=args.quick,
                                                      labeled=False,
                                                      holdout=holdouts,
                                                      dataset=ds),
                        con_ep, lam=5.0, n_slices=cfgH["n_slices"])
    hybrid_cents = cents_of(hybrid)
    print("\n----- ss16 supcon+sigreg (lam=5) -----")
    torch.manual_seed(args.seed + 15); np.random.seed(args.seed + 15)
    ss = backbone()
    train_supcon_sigreg(ss, cifar_two_view_loader(quick=args.quick,
                                                  labeled=True,
                                                  holdout=holdouts,
                                                  dataset=ds),
                        sup_ep, lam=5.0, n_slices=cfgH["n_slices"])
    ss_cents = cents_of(ss)

    ARMS = {
        "supcon+hybrid": (supcon, supcon_cents),
        "ss[lam5]+hybrid": (ss, ss_cents),
    }
    arm_names = list(ARMS)

    # ----- post grids per stage ---------------------------------------------
    power = {st: {s: {n: [] for n in arm_names} for s in STATS}
             for st, _ in STAGES}
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
        for st_name, st_kw in STAGES:
            print(f"\n===== POST [{st_name}], f={f} ({len(inj)} injected) "
                  f"=====")
            for name in arm_names:
                c_net, c_cents = ARMS[name]
                bb = copy.deepcopy(c_net)
                _, extras = exp28.run_concat_discovery(
                    bb, hybrid, exp28.fill_means(c_cents, seen, cfgH).clone(),
                    hybrid_cents, base=sub, dim=args.dim_half,
                    train_eval_loader=tel_loader, test_loader=test_loader,
                    seen=seen, holdouts=holdouts, cfg=cfgH,
                    rounds=args.rounds, ft_epochs=ft_ep, names=names,
                    seed=args.seed, **st_kw)
                cur_means, disc_ssl = extras["cur_means"], extras["disc_ssl"]
                te_post, tel_post = space_embs(bb, hybrid, test_loader)
                zt = torch.as_tensor(te_post, dtype=torch.float32,
                                     device=DEVICE)
                seen_anc = torch.cat([cur_means[seen], hybrid_cents], dim=1)
                disc_anc = torch.cat([cur_means[n_cls:], disc_ssl], dim=1)
                d_seen = torch.cdist(zt, seen_anc).min(1).values
                d_disc = torch.cdist(zt, disc_anc).min(1).values
                tr_post, trl_post = space_embs(bb, hybrid, train_eval_loader)
                bgm = np.isin(tel_post, seen)
                sgm = np.isin(tel_post, list(holdouts))
                s = (d_seen - d_disc).cpu().numpy()
                pe = exp30.power_at_alpha(s[bgm], s[sgm], args.alpha)
                power[st_name]["perevent"][name].append(pe)
                print(f"  [{name}] per-event post f={f}: power={pe:.3f}")
                R = torch.as_tensor(tr_post[np.isin(trl_post, seen)][:20000],
                                    dtype=torch.float32, device=DEVICE)
                bg = torch.as_tensor(te_post[bgm], dtype=torch.float32,
                                     device=DEVICE)
                sg = torch.as_tensor(te_post[sgm], dtype=torch.float32,
                                     device=DEVICE)
                print(f"  [{name}] sparker (post)")
                p, _ = exp31.run_test_battery(bg, sg, R, [f], args.n_d,
                                              n_null_post, n_sig_toys,
                                              args.alpha, args.seed + i_f,
                                              sparker_kw, tag="post-spk")
                power[st_name]["sparker"][name].append(p[0])
                maha_fn, mmd_fn, n_bg, n_sig = exp32.make_stats_fns(
                    tr_post, trl_post, te_post, tel_post, seen, holdouts,
                    args.seed + i_f)
                print(f"  [{name}] maha (post)")
                p, _ = exp32.battery(maha_fn, n_bg, n_sig, [f], args.n_d,
                                     n_null_post, n_sig_toys, args.alpha,
                                     args.seed + i_f, tag="post-maha")
                power[st_name]["maha"][name].append(p[0])
                print(f"  [{name}] mmd (post)")
                p, _ = exp32.battery(mmd_fn, n_bg, n_sig, [f], args.n_d,
                                     n_null_post, n_sig_toys, args.alpha,
                                     args.seed + i_f, tag="post-mmd")
                power[st_name]["mmd"][name].append(p[0])

    # ----- report vs stored baselines ---------------------------------------
    if ds == "cifar10":
        base34i = np.load(os.path.join(LOGDIR, "arc_cifar10.npz"),
                          allow_pickle=True)
        basef = list(base34i["fractions"])

        def baseline(stat, arm):
            key = ("supcon+hybrid[lam5]" if arm == "supcon+hybrid" else arm)
            row = base34i[f"{stat}_{key}_post"]
            return [float(row[basef.index(f)]) if f in basef
                    else float("nan") for f in fractions]
    else:
        base34f = np.load(os.path.join(LOGDIR, "power_data_hybrid.npz"),
                          allow_pickle=True)
        base34g = np.load(os.path.join(LOGDIR, "power_data_ss_hybrid.npz"),
                          allow_pickle=True)
        basef = list(base34f["fractions"])

        def baseline(stat, arm):
            if arm == "supcon+hybrid":
                row = base34f[f"{stat}_supcon+hybrid_post"]
            else:
                row = base34g[f"{stat}_post"]
            return [float(row[basef.index(f)]) if f in basef
                    else float("nan") for f in fractions]

    npz = {"fractions": np.array(fractions)}
    for stat in STATS:
        print(f"\n===== EXP35 {stat.upper()} POST POWER "
              f"(alpha={args.alpha}) =====")
        print(f"  {'arm':<18}{'variant':>18}"
              + "".join(f"{f:>9}" for f in fractions))
        for name in arm_names:
            b = baseline(stat, name)
            print(f"  {name:<18}{'baseline(34f/g)':>18}"
                  + "".join(f"{p:>9.3f}" for p in b))
            for st_name, _ in STAGES:
                print(f"  {name:<18}{st_name:>18}"
                      + "".join(f"{p:>9.3f}"
                                for p in power[st_name][stat][name]))
                npz[f"{stat}_{name}_{st_name}"] = np.array(
                    power[st_name][stat][name])
            npz[f"{stat}_{name}_baseline"] = np.array(b)
        plt.figure(figsize=(8.5, 6.5))
        for name, c in zip(arm_names, ["#d62728", "#2a78d6"]):
            plt.plot(fractions, baseline(stat, name), ":", color=c, lw=1.5,
                     label=f"{name} baseline")
            plt.plot(fractions, power["confmask"][stat][name], "--o",
                     color=c, lw=1.6, ms=5, label=f"{name} confmask")
            plt.plot(fractions, power["confmask+anneal"][stat][name], "-o",
                     color=c, lw=2, ms=6, label=f"{name} confmask+anneal")
        plt.xscale("log")
        plt.axhline(0.05, color="gray", lw=1, ls=":")
        plt.xlabel("injected anomaly fraction")
        plt.ylabel("power at alpha=0.05 (post-discovery)")
        plt.title(f"exp35 [{ds} 16+16] conf-masked / annealed discovery: "
                  f"{stat}")
        plt.grid(alpha=0.25, which="both")
        plt.legend(loc="upper left", fontsize=7)
        plt.tight_layout()
        out = plot_path(f"exp35_{stat}_power_{ds}.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print("saved", out)
    outdir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "logs", "exp35")
    os.makedirs(outdir, exist_ok=True)
    np.savez(os.path.join(outdir, f"power_data_confmask_{ds}.npz"), **npz)
    print(f"saved {outdir}/power_data_confmask_{ds}.npz")


if __name__ == "__main__":
    main()
