"""
Experiment 34f: the exp-33 four-statistic power battery (per-event margin,
annealed SparKer, Mahalanobis, MMD; toy-calibrated dataset-level NP tests)
on the two winning exp-34e spaces:

  supcon+hybrid : [supcon16 ; simclr+sigreg16(lam5)]  -- best probe (0.9423)
  hybrid->supres: [supres16 ; simclr+sigreg16(lam1)]  -- best Mahalanobis
                   (supres warm-started from the lam1 hybrid trunk)

Networks retrained with the exp-34e seeds; same protocol as the exp-33
CIFAR-100 16+16 run (holdout 4, fractions to 0.05, annealed SparKer,
one shared discovery rerun per (arm, fraction)).  Exp-33 reference curves
(supcon, supcon+simclr, sup->res) are overlaid from the saved npz.

    python experiments/34f_hybrid_power.py
    python experiments/34f_hybrid_power.py --quick --fractions 0.01,0.05
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
from supersig.data import (get_cifar_loaders, cifar_balanced_loader,
                           cifar_two_view_loader, _cifar_spec)
from supersig.models import CIFARResNetBackbone
from supersig.recipes import recipe
from supersig.train import (train_sigreg_hybrid, train_supcon,
                            train_simclr_sigreg, collect_embeddings)

exp28 = importlib.import_module("28_concat_residual")
exp29 = importlib.import_module("29_residual_finetune")
exp30 = importlib.import_module("30_power_curves")
exp31 = importlib.import_module("31_sparker_power")
exp32 = importlib.import_module("32_maha_mmd_power")

STATS = ["perevent", "sparker", "maha", "mmd"]
CPB, PC = 99, 24
REF_NPZ = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "logs", "exp33",
    "power_data_cifar100_16p16_k1.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dim-half", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--fractions", default="0.001,0.003,0.01,0.02,0.05")
    ap.add_argument("--n-d", type=int, default=5000)
    ap.add_argument("--kernels", type=int, default=16)
    ap.add_argument("--steps", type=int, default=300)
    args = ap.parse_args()
    ds = "cifar100"
    cfgH = recipe(ds, emb_dim=args.dim_half)
    n_cls = cfgH["n_classes"]
    holdouts = {args.holdout}
    seen = [c for c in range(n_cls) if c not in holdouts]
    fractions = [float(x) for x in args.fractions.split(",")]
    sup_ep = 2 if args.quick else 10
    con_ep = 2 if args.quick else 20
    ft_ep = 1 if args.quick else cfgH["ft_epochs"]
    n_null_pre = 20 if args.quick else 200
    n_null_post = 20 if args.quick else 100
    n_sig_toys = 10 if args.quick else 50
    sparker_kw = dict(M=args.kernels, steps=args.steps)   # annealed widths
    names = [str(c) for c in range(n_cls)]
    print(f"exp34f [{ds}] power battery on exp-34e hybrid spaces, "
          f"holdout={sorted(holdouts)}, alpha={args.alpha}")

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

    # ----- networks (exp-34e seeds) -----------------------------------------
    print("\n----- supcon16 -----")
    torch.manual_seed(args.seed + 15); np.random.seed(args.seed + 15)
    supcon = backbone()
    train_supcon(supcon, cifar_two_view_loader(quick=args.quick, labeled=True,
                                               holdout=holdouts, dataset=ds),
                 sup_ep)
    supcon_cents = cents_of(supcon)

    hybrids = {}
    for lam in (1.0, 5.0):
        print(f"\n----- hybrid16 simclr+sigreg (lam={lam}) -----")
        torch.manual_seed(args.seed + 16); np.random.seed(args.seed + 16)
        h = backbone()
        train_simclr_sigreg(h, cifar_two_view_loader(quick=args.quick,
                                                     labeled=False,
                                                     holdout=holdouts,
                                                     dataset=ds),
                            con_ep, lam=lam, n_slices=cfgH["n_slices"])
        hybrids[lam] = (h, cents_of(h))

    print("\n----- supres16 from hybrid(lam=1) trunk -----")
    torch.manual_seed(args.seed + 13); np.random.seed(args.seed + 13)
    supres = copy.deepcopy(hybrids[1.0][0])
    means_supres = exp28.fill_means(hybrids[1.0][1], seen, cfgH).clone()
    train_sigreg_hybrid(supres,
                        cifar_balanced_loader(ds, holdout=holdouts,
                                              quick=args.quick,
                                              classes_per_batch=CPB,
                                              per_class=PC),
                        sup_ep, means_supres, mode="repulse", disc="proto",
                        alpha=1.0, rep_weight=cfgH["rep_weight"],
                        sigreg_weight=1.0, n_slices=cfgH["n_slices"])
    means_supres = means_supres.detach()

    # arm -> (sup net, full means, aug net, aug cents, cfg)
    ARMS = {
        "supcon+hybrid": (supcon, exp28.fill_means(supcon_cents, seen, cfgH),
                          hybrids[5.0][0], hybrids[5.0][1], cfgH),
        "hybrid->supres": (supres, means_supres,
                           hybrids[1.0][0], hybrids[1.0][1], cfgH),
    }
    arm_names = list(ARMS)

    def space_embs(net, aug, loader):
        e, l = collect_embeddings(net, loader)
        if aug is not None:
            ea, _ = collect_embeddings(aug, loader)
            e = np.concatenate([e, ea], axis=1)
        return e, l

    trains, tests = {}, {}
    tr_lab = te_lab = None
    for name in arm_names:
        net, means, aug, cents, acfg = ARMS[name]
        trains[name], tr_lab = space_embs(net, aug, train_eval_loader)
        tests[name], te_lab = space_embs(net, aug, test_loader)

    # ===== PRE batteries ====================================================
    print("\n===== PRE power batteries =====")
    pre_power = {s: {} for s in STATS}
    for name in arm_names:
        net, means, aug, cents, acfg = ARMS[name]
        tr, te = trains[name], tests[name]
        anchors = torch.cat([means[seen], cents], dim=1)
        bg_mask = np.isin(te_lab, seen)
        sig_mask = np.isin(te_lab, list(holdouts))
        d = torch.cdist(torch.as_tensor(te, dtype=torch.float32,
                                        device=DEVICE),
                        torch.as_tensor(anchors, dtype=torch.float32,
                                        device=DEVICE))
        s = d.min(1).values.cpu().numpy()
        pe = exp30.power_at_alpha(s[bg_mask], s[sig_mask], args.alpha)
        pre_power["perevent"][name] = [pe] * len(fractions)
        print(f"  [{name}] per-event pre power={pe:.3f}")
        R = torch.as_tensor(tr[np.isin(tr_lab, seen)][:20000],
                            dtype=torch.float32, device=DEVICE)
        bg = torch.as_tensor(te[bg_mask], dtype=torch.float32, device=DEVICE)
        sg = torch.as_tensor(te[sig_mask], dtype=torch.float32, device=DEVICE)
        print(f"  [{name}] sparker")
        pre_power["sparker"][name], _ = exp31.run_test_battery(
            bg, sg, R, fractions, args.n_d, n_null_pre, n_sig_toys,
            args.alpha, args.seed, sparker_kw, tag="pre-spk")
        maha_fn, mmd_fn, n_bg, n_sig = exp32.make_stats_fns(
            tr, tr_lab, te, te_lab, seen, holdouts, args.seed)
        print(f"  [{name}] maha")
        pre_power["maha"][name], _ = exp32.battery(
            maha_fn, n_bg, n_sig, fractions, args.n_d, n_null_pre,
            n_sig_toys, args.alpha, args.seed, tag="pre-maha")
        print(f"  [{name}] mmd")
        pre_power["mmd"][name], _ = exp32.battery(
            mmd_fn, n_bg, n_sig, fractions, args.n_d, n_null_pre,
            n_sig_toys, args.alpha, args.seed, tag="pre-mmd")

    # ===== POST grid: one discovery per (arm, fraction) =====================
    post_power = {s: {n: [] for n in arm_names} for s in STATS}
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
        for name in arm_names:
            net, means, aug, cents, acfg = ARMS[name]
            bb = copy.deepcopy(net)
            _, extras = exp28.run_concat_discovery(
                bb, aug, means.clone(), cents, base=sub, dim=args.dim_half,
                train_eval_loader=tel_loader, test_loader=test_loader,
                seen=seen, holdouts=holdouts, cfg=acfg, rounds=args.rounds,
                ft_epochs=ft_ep, names=names, seed=args.seed)
            cur_means, disc_ssl = extras["cur_means"], extras["disc_ssl"]
            te_post, tel_post = space_embs(bb, aug, test_loader)
            zt = torch.as_tensor(te_post, dtype=torch.float32, device=DEVICE)
            seen_anc = torch.cat([cur_means[seen], cents], dim=1)
            disc_anc = torch.cat([cur_means[n_cls:], disc_ssl], dim=1)
            d_seen = torch.cdist(zt, seen_anc).min(1).values
            d_disc = torch.cdist(zt, disc_anc).min(1).values
            tr_post, trl_post = space_embs(bb, aug, train_eval_loader)
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

    # ===== report ===========================================================
    ref = np.load(REF_NPZ, allow_pickle=True) if os.path.exists(REF_NPZ) \
        else None
    npz = {"fractions": np.array(fractions)}
    for stat in STATS:
        print(f"\n===== EXP34F {stat.upper()} POWER SUMMARY "
              f"(alpha={args.alpha}) =====")
        print(f"  {'arm':<16}{'kind':>6}"
              + "".join(f"{f:>9}" for f in fractions))
        for name in arm_names:
            print(f"  {name:<16}{'pre':>6}"
                  + "".join(f"{p:>9.3f}" for p in pre_power[stat][name]))
            print(f"  {name:<16}{'post':>6}"
                  + "".join(f"{p:>9.3f}" for p in post_power[stat][name]))
            npz[f"{stat}_{name}_pre"] = np.array(pre_power[stat][name])
            npz[f"{stat}_{name}_post"] = np.array(post_power[stat][name])
        plt.figure(figsize=(8, 6.5))
        for name, c in zip(arm_names, ["#d62728", "#2a78d6"]):
            plt.plot(fractions, pre_power[stat][name], "--o", color=c, lw=1.4,
                     ms=5, alpha=0.75, label=f"{name} pre")
            plt.plot(fractions, post_power[stat][name], "-o", color=c, lw=2,
                     ms=6, label=f"{name} post")
        if ref is not None:
            rf = ref["fractions"]
            for rname, ls in [("supcon+simclr", ":"), ("supcon", "-."),
                              ("sup->res", "--")]:
                key = f"{stat}_{rname}_post"
                if key in ref:
                    plt.plot(rf, ref[key], ls, color="gray", lw=1.3,
                             label=f"{rname} post (exp33)")
        plt.xscale("log")
        plt.axhline(0.05, color="gray", lw=1, ls=":")
        plt.xlabel("injected anomaly fraction")
        plt.ylabel("power at alpha=0.05")
        plt.title(f"exp34f [cifar100] hybrid spaces: {stat} power vs fraction"
                  " (train-side clamp above f~0.01)")
        plt.grid(alpha=0.25, which="both")
        plt.legend(loc="upper left", fontsize=8)
        plt.tight_layout()
        out = plot_path(f"exp34f_{stat}_power_cifar100.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print("saved", out)
    outdir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "logs", "exp34")
    os.makedirs(outdir, exist_ok=True)
    np.savez(os.path.join(outdir, "power_data_hybrid.npz"), **npz)
    print(f"saved {outdir}/power_data_hybrid.npz")


if __name__ == "__main__":
    main()
