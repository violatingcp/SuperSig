"""
Experiment 36: sup->res with a HYBRID residual half (CIFAR-10, 16+16).

The classic sup->res champion pairs the supervised SIGReg half with a
classwise-SIGReg residual (aug-invariance + per-class N(0,I) on z - mean_y).
Here the residual half is instead trained with the exp-34e hybrid objective
on the residuals: SimCLR NT-Xent + SIGReg (lam=5) on z - mean_y with the
sup half's learned means frozen.  Contrastive features from whatever the
class atom cannot explain, in a calibrated geometry.

Arms (both rebuilt in-run for a clean A/B; exp-33 curves overlaid):
  sup->res        : [sup16 ; res16(classwise sigreg residual)]   (control)
  sup->res-hybrid : [sup16 ; res-hybrid16(NT-Xent+sigreg residual)]

Usual suite: probe/acc/eucl/maha + gaussianity (Part A), four-statistic
power scan pre + post-discovery (Part B), fractions to 0.1.

    python experiments/36_supres_hybrid.py
    python experiments/36_supres_hybrid.py --quick --fractions 0.01,0.1
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
from supersig.metrics import gaussianity_summary
from supersig.recipes import supervised_embedding, recipe
from supersig.train import (train_sigreg_residual_ssl, train_simclr_residual,
                            collect_embeddings)

exp28 = importlib.import_module("28_concat_residual")
exp29 = importlib.import_module("29_residual_finetune")
exp30 = importlib.import_module("30_power_curves")
exp31 = importlib.import_module("31_sparker_power")
exp32 = importlib.import_module("32_maha_mmd_power")

STATS = ["perevent", "sparker", "maha", "mmd"]
REF_NPZ = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "logs", "exp33",
    "power_data_cifar10_16p16_k1.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dim-half", type=int, default=16)
    ap.add_argument("--res-lam", type=float, default=5.0)
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
    n_null_pre = 20 if args.quick else 200
    n_null_post = 20 if args.quick else 100
    n_sig_toys = 10 if args.quick else 50
    sparker_kw = dict(M=args.kernels, steps=args.steps)   # annealed widths
    names = exp29.CIFAR_NAMES
    print(f"exp36 [{ds}] sup->res with hybrid residual, "
          f"holdout={sorted(holdouts)}, res_lam={args.res_lam}")

    train_loader, test_loader = get_cifar_loaders(quick=args.quick, dataset=ds)
    train_eval_loader = DataLoader(train_loader.dataset, batch_size=256,
                                   shuffle=False, num_workers=2)
    cls, plain, _ = _cifar_spec(ds)
    base = cls(DATA_DIR, train=True, download=True, transform=plain)
    base_targets = np.array(base.targets)
    n_base = 8000 if args.quick else len(base)
    seen_idx = np.where(np.isin(base_targets[:n_base], seen))[0]
    sig_idx_all = np.where(np.isin(base_targets[:n_base], list(holdouts)))[0]

    def probe_stat(tr, tr_lab, te, te_lab, n_rep=3):
        aucs = []
        for s in range(n_rep):
            torch.manual_seed(1000 + s)
            a, _, _ = exp29.linear_probe_novelty(tr, tr_lab, te, te_lab,
                                                 holdouts)
            aucs.append(a)
        return float(np.mean(aucs)), float(np.std(aucs))

    def cents_of(net):
        e, l = collect_embeddings(net, train_eval_loader)
        m = np.isin(l, seen)
        return exp28.class_centroids(e[m], l[m], seen)

    def space_embs(net, aug, loader):
        e, l = collect_embeddings(net, loader)
        ea, _ = collect_embeddings(aug, loader)
        return np.concatenate([e, ea], axis=1), l

    # ----- networks (exp-33 16p16 seeds for the sup half) -------------------
    print("\n----- sup16 (settled recipe) -----")
    sup, means_sup, _ = supervised_embedding(ds, holdouts=holdouts,
                                             quick=args.quick,
                                             seed=args.seed + 10,
                                             emb_dim=args.dim_half)
    means_sup = means_sup.detach()
    print("\n----- res16 classic (classwise sigreg residual) -----")
    torch.manual_seed(args.seed + 11); np.random.seed(args.seed + 11)
    res = copy.deepcopy(sup)
    train_sigreg_residual_ssl(
        res, cifar_two_view_balanced_loader(ds, holdout=holdouts,
                                            quick=args.quick),
        res_ep, means_sup, n_slices=cfgH["n_slices"], classwise=True)
    res_cents = cents_of(res)
    print(f"\n----- res-hybrid16 (NT-Xent + sigreg lam={args.res_lam} "
          f"on sup residual) -----")
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

    # ----- Part A -----------------------------------------------------------
    print("\n===== Part A: probe suite =====")
    trains, tests = {}, {}
    tr_lab = te_lab = None
    resultsA = {}
    for name, (aug, cents) in ARMS.items():
        trains[name], tr_lab = space_embs(sup, aug, train_eval_loader)
        tests[name], te_lab = space_embs(sup, aug, test_loader)
        anchors = torch.cat([means_sup[seen], cents], dim=1)
        r = exp29.evaluate_space(trains[name], tr_lab, tests[name], te_lab,
                                 anchors, seen, holdouts)
        pm, psd = probe_stat(trains[name], tr_lab, tests[name], te_lab)
        g = gaussianity_summary(tests[name], te_lab, seen, seed=args.seed)
        print(f"  [{name:<16}] probe={pm:.4f}+-{psd:.4f} acc={r['acc']:.4f} "
              f"supAUC={r['sup_auc']:.4f} eucl={r['eucl']:.4f} "
              f"mahaT={r['maha_tied']:.4f} mahaPC={r['maha_pc']:.4f}")
        resultsA[name] = dict(probe=pm, probe_sd=psd, acc=r["acc"],
                              eucl=r["eucl"], mahaT=r["maha_tied"],
                              mahaPC=r["maha_pc"], gauss=g)
    print("\n===== gaussianity =====")
    exp28.print_gauss_table({n: resultsA[n]["gauss"] for n in resultsA})

    # ----- Part B: power scan -----------------------------------------------
    pre_power = {s: {} for s in STATS}
    for name in arm_names:
        aug, cents = ARMS[name]
        tr, te = trains[name], tests[name]
        anchors = torch.cat([means_sup[seen], cents], dim=1)
        bg_mask = np.isin(te_lab, seen)
        sig_mask = np.isin(te_lab, list(holdouts))
        print(f"\n===== PRE power scan: {name} =====")
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
        print(f"\n===== POST scan, f={f} ({len(inj)} injected) =====")
        for name in arm_names:
            aug, cents = ARMS[name]
            bb = copy.deepcopy(sup)
            _, extras = exp28.run_concat_discovery(
                bb, aug, means_sup.clone(), cents, base=sub,
                dim=args.dim_half, train_eval_loader=tel_loader,
                test_loader=test_loader, seen=seen, holdouts=holdouts,
                cfg=cfgH, rounds=args.rounds, ft_epochs=ft_ep, names=names,
                seed=args.seed)
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

    # ----- report -----------------------------------------------------------
    ref = np.load(REF_NPZ, allow_pickle=True) if os.path.exists(REF_NPZ) \
        else None
    npz = {"fractions": np.array(fractions)}
    for name in arm_names:
        npz[f"probe_{name}"] = np.array([resultsA[name]["probe"],
                                         resultsA[name]["probe_sd"]])
        for k in ("acc", "eucl", "mahaT", "mahaPC"):
            npz[f"{k}_{name}"] = np.array(resultsA[name][k])
    for stat in STATS:
        print(f"\n===== EXP36 {stat.upper()} POWER SUMMARY "
              f"(alpha={args.alpha}) =====")
        print(f"  {'arm':<18}{'kind':>6}"
              + "".join(f"{f:>9}" for f in fractions))
        for name in arm_names:
            print(f"  {name:<18}{'pre':>6}"
                  + "".join(f"{p:>9.3f}" for p in pre_power[stat][name]))
            print(f"  {name:<18}{'post':>6}"
                  + "".join(f"{p:>9.3f}" for p in post_power[stat][name]))
            npz[f"{stat}_{name}_pre"] = np.array(pre_power[stat][name])
            npz[f"{stat}_{name}_post"] = np.array(post_power[stat][name])
        plt.figure(figsize=(8, 6.5))
        for name, c in zip(arm_names, ["#2a78d6", "#d62728"]):
            plt.plot(fractions, pre_power[stat][name], "--o", color=c,
                     lw=1.4, ms=5, alpha=0.75, label=f"{name} pre")
            plt.plot(fractions, post_power[stat][name], "-o", color=c, lw=2,
                     ms=6, label=f"{name} post")
        if ref is not None:
            rf = ref["fractions"]
            key = f"{stat}_sup->res_post"
            if key in ref:
                plt.plot(rf, ref[key], ":", color="gray", lw=1.4,
                         label="sup->res post (exp33)")
        plt.xscale("log")
        plt.axhline(0.05, color="gray", lw=1, ls=":")
        plt.xlabel("injected anomaly fraction")
        plt.ylabel("power at alpha=0.05")
        plt.title(f"exp36 [cifar10] sup->res residual objectives: {stat}")
        plt.grid(alpha=0.25, which="both")
        plt.legend(loc="upper left", fontsize=8)
        plt.tight_layout()
        out = plot_path(f"exp36_{stat}_power_cifar10.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print("saved", out)
    outdir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "logs", "exp36")
    os.makedirs(outdir, exist_ok=True)
    np.savez(os.path.join(outdir, "power_data_supres_hybrid.npz"), **npz)
    print(f"saved {outdir}/power_data_supres_hybrid.npz")


if __name__ == "__main__":
    main()
