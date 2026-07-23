"""
Experiment 36b: confidence-masked discovery (exp-35 conf_thresh, JEPAMatch
M_i mask; annealing OFF per the exp-35 verdict) applied to the CIFAR-10
sup->res arms of exp 36:

  sup->res        : [sup16 ; res16(classwise sigreg residual)]
  sup->res-hybrid : [sup16 ; res-hybrid16(NT-Xent+sigreg residual)]

Networks rebuilt with the exp-36 seeds; unmasked baselines loaded from
logs/exp36/power_data_supres_hybrid.npz (same seeds + toys).  Post grid
only -- pre-discovery power is unaffected by the mask.

    python experiments/36b_supres_confmask.py [--conf-thresh 0.5]
    python experiments/36b_supres_confmask.py --quick --fractions 0.01,0.1
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
from supersig.data import (get_cifar_loaders, cifar_two_view_loader,
                           cifar_two_view_balanced_loader, _cifar_spec)
from supersig.recipes import supervised_embedding, recipe
from supersig.train import (train_sigreg_residual_ssl, train_simclr_residual,
                            collect_embeddings)

exp28 = importlib.import_module("28_concat_residual")
exp29 = importlib.import_module("29_residual_finetune")
exp30 = importlib.import_module("30_power_curves")
exp31 = importlib.import_module("31_sparker_power")
exp32 = importlib.import_module("32_maha_mmd_power")

STATS = ["perevent", "sparker", "maha", "mmd"]
BASE_NPZ = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "logs", "exp36",
    "power_data_supres_hybrid.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dim-half", type=int, default=16)
    ap.add_argument("--res-lam", type=float, default=5.0)
    ap.add_argument("--conf-thresh", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--fractions", default="0.001,0.003,0.01,0.02,0.03,0.1")
    ap.add_argument("--n-d", type=int, default=5000)
    ap.add_argument("--kernels", type=int, default=16)
    ap.add_argument("--steps", type=int, default=300)
    args = ap.parse_args()
    ds = "cifar10"
    cfgH = recipe(ds, emb_dim=args.dim_half)
    n_cls = cfgH["n_classes"]
    holdouts = {args.holdout}
    seen = [c for c in range(n_cls) if c not in holdouts]
    fractions = [float(x) for x in args.fractions.split(",")]
    res_ep = 2 if args.quick else 10
    con_ep = 2 if args.quick else 20
    ft_ep = 1 if args.quick else cfgH["ft_epochs"]
    n_null_post = 20 if args.quick else 100
    n_sig_toys = 10 if args.quick else 50
    sparker_kw = dict(M=args.kernels, steps=args.steps)   # annealed widths
    names = exp29.CIFAR_NAMES
    print(f"exp36b [{ds}] conf-masked sup->res discovery, "
          f"holdout={sorted(holdouts)}, thresh={args.conf_thresh}")

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

    def space_embs(net, aug, loader):
        e, l = collect_embeddings(net, loader)
        ea, _ = collect_embeddings(aug, loader)
        return np.concatenate([e, ea], axis=1), l

    # ----- networks (exp-36 seeds) ------------------------------------------
    print("\n----- sup16 (settled recipe) -----")
    sup, means_sup, _ = supervised_embedding(ds, holdouts=holdouts,
                                             quick=args.quick,
                                             seed=args.seed + 10,
                                             emb_dim=args.dim_half)
    means_sup = means_sup.detach()
    print("\n----- res16 classic -----")
    torch.manual_seed(args.seed + 11); np.random.seed(args.seed + 11)
    res = copy.deepcopy(sup)
    train_sigreg_residual_ssl(
        res, cifar_two_view_balanced_loader(ds, holdout=holdouts,
                                            quick=args.quick),
        res_ep, means_sup, n_slices=cfgH["n_slices"], classwise=True)
    res_cents = cents_of(res)
    print("\n----- res-hybrid16 -----")
    torch.manual_seed(args.seed + 11); np.random.seed(args.seed + 11)
    resh = copy.deepcopy(sup)
    train_simclr_residual(resh,
                          cifar_two_view_loader(quick=args.quick,
                                                labeled=True,
                                                holdout=holdouts, dataset=ds),
                          con_ep, means_sup, lam=args.res_lam,
                          n_slices=cfgH["n_slices"])
    resh_cents = cents_of(resh)

    ARMS = {
        "sup->res": (res, res_cents),
        "sup->res-hybrid": (resh, resh_cents),
    }
    arm_names = list(ARMS)

    # ----- conf-masked post grid --------------------------------------------
    post_power = {s: {n: [] for n in arm_names} for s in STATS}
    for i_f, f in enumerate(fractions):
        n_inj = int(round(f * len(seen_idx) / (1.0 - f)))
        rng = np.random.default_rng(args.seed * 1000 + i_f)
        inj = rng.choice(sig_idx_all, size=min(n_inj, len(sig_idx_all)),
                         replace=False)
        sub = Subset(base, np.concatenate([seen_idx, inj]).tolist())
        tel_loader = DataLoader(sub, batch_size=256, shuffle=False,
                                num_workers=2)
        print(f"\n===== POST [confmask], f={f} ({len(inj)} injected) =====")
        for name in arm_names:
            aug, cents = ARMS[name]
            bb = copy.deepcopy(sup)
            _, extras = exp28.run_concat_discovery(
                bb, aug, means_sup.clone(), cents, base=sub,
                dim=args.dim_half, train_eval_loader=tel_loader,
                test_loader=test_loader, seen=seen, holdouts=holdouts,
                cfg=cfgH, rounds=args.rounds, ft_epochs=ft_ep, names=names,
                seed=args.seed, conf_thresh=args.conf_thresh)
            cur_means, disc_ssl = extras["cur_means"], extras["disc_ssl"]
            te_post, tel_post = space_embs(bb, aug, test_loader)
            zt = torch.as_tensor(te_post, dtype=torch.float32, device=DEVICE)
            seen_anc = torch.cat([cur_means[seen], cents], dim=1)
            disc_anc = torch.cat([cur_means[n_cls:], disc_ssl], dim=1)
            d_seen = torch.cdist(zt, seen_anc).min(1).values
            d_disc = torch.cdist(zt, disc_anc).min(1).values
            tr_post, trl_post = space_embs(bb, aug, train_eval_loader)
            bgm = np.isin(tel_post, seen)
            sgm = np.isin(tel_post, list(holdouts))
            s = (d_seen - d_disc).cpu().numpy()
            pe = exp30.power_at_alpha(s[bgm], s[sgm], args.alpha)
            post_power["perevent"][name].append(pe)
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

    # ----- report vs exp36 unmasked baselines -------------------------------
    ref = np.load(BASE_NPZ, allow_pickle=True) if os.path.exists(BASE_NPZ) \
        else None
    npz = {"fractions": np.array(fractions)}
    for stat in STATS:
        print(f"\n===== EXP36B {stat.upper()} POST POWER "
              f"(alpha={args.alpha}) =====")
        print(f"  {'arm':<18}{'variant':>16}"
              + "".join(f"{f:>9}" for f in fractions))
        for name in arm_names:
            if ref is not None:
                rf = list(ref["fractions"])
                b = [float(ref[f"{stat}_{name}_post"][rf.index(f)])
                     if f in rf else float("nan") for f in fractions]
                print(f"  {name:<18}{'unmasked(36)':>16}"
                      + "".join(f"{p:>9.3f}" for p in b))
                npz[f"{stat}_{name}_baseline"] = np.array(b)
            print(f"  {name:<18}{'confmask':>16}"
                  + "".join(f"{p:>9.3f}"
                            for p in post_power[stat][name]))
            npz[f"{stat}_{name}_confmask"] = np.array(post_power[stat][name])
        plt.figure(figsize=(8, 6.5))
        for name, c in zip(arm_names, ["#2a78d6", "#d62728"]):
            if ref is not None:
                plt.plot(fractions, npz[f"{stat}_{name}_baseline"], ":",
                         color=c, lw=1.5, label=f"{name} unmasked (36)")
            plt.plot(fractions, post_power[stat][name], "-o", color=c, lw=2,
                     ms=6, label=f"{name} confmask")
        plt.xscale("log")
        plt.axhline(0.05, color="gray", lw=1, ls=":")
        plt.xlabel("injected anomaly fraction")
        plt.ylabel("power at alpha=0.05 (post-discovery)")
        plt.title(f"exp36b [cifar10] conf-masked sup->res discovery: {stat}")
        plt.grid(alpha=0.25, which="both")
        plt.legend(loc="upper left", fontsize=8)
        plt.tight_layout()
        out = plot_path(f"exp36b_{stat}_power_cifar10.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print("saved", out)
    outdir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "logs", "exp36")
    os.makedirs(outdir, exist_ok=True)
    np.savez(os.path.join(outdir, "power_data_supres_confmask.npz"), **npz)
    print(f"saved {outdir}/power_data_supres_confmask.npz")


if __name__ == "__main__":
    main()
