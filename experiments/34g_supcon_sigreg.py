"""
Experiment 34g: fully SIGReg-calibrated contrastive space
[supcon+sigreg16 ; simclr+sigreg16(lam5)] -- does the calibration trick that
lifted the SimCLR feature half (exp 34e) transfer to the supervised half?

Part A: train supcon+sigreg16 at lam {1, 5}, evaluate the concat with the
lam5 hybrid feature half on the probe suite (probe / acc / eucl / maha /
gaussianity); pick the better lam by probe.
Part B: the exp-33 four-statistic power scan (per-event, annealed SparKer,
Mahalanobis, MMD; pre + post-discovery) on the winning space, overlaid on
the exp-34f supcon+hybrid curves.

References (exps 34e/34f): supcon+hybrid[lam5] probe 0.9423, SparKer/MMD
1.00 at f=0.05, MMD post 0.88 at f=0.02; plain supcon probe 0.9268.

    python experiments/34g_supcon_sigreg.py
    python experiments/34g_supcon_sigreg.py --quick --fractions 0.01,0.05
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
                           _cifar_spec)
from supersig.models import CIFARResNetBackbone
from supersig.metrics import gaussianity_summary
from supersig.recipes import recipe
from supersig.train import (train_supcon_sigreg, train_simclr_sigreg,
                            collect_embeddings)

exp28 = importlib.import_module("28_concat_residual")
exp29 = importlib.import_module("29_residual_finetune")
exp30 = importlib.import_module("30_power_curves")
exp31 = importlib.import_module("31_sparker_power")
exp32 = importlib.import_module("32_maha_mmd_power")

STATS = ["perevent", "sparker", "maha", "mmd"]
REF = {"supcon (r1)": 0.9268, "supcon+simclr (r1)": 0.9394,
       "supcon+hybrid[lam5] (34e)": 0.9423}
F_NPZ = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "logs", "exp34", "power_data_hybrid.npz")


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
    print(f"exp34g [{ds}] supcon+sigreg half, holdout={sorted(holdouts)}")
    print("  refs: " + ", ".join(f"{k}={v:.4f}" for k, v in REF.items()))

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

    def backbone():
        return CIFARResNetBackbone(args.dim_half, arch=cfgH["arch"],
                                   pretrain=ds).to(DEVICE)

    def space_embs(net, aug, loader):
        e, l = collect_embeddings(net, loader)
        if aug is not None:
            ea, _ = collect_embeddings(aug, loader)
            e = np.concatenate([e, ea], axis=1)
        return e, l

    # ----- networks ---------------------------------------------------------
    print("\n----- hybrid16 simclr+sigreg (lam=5, feature half) -----")
    torch.manual_seed(args.seed + 16); np.random.seed(args.seed + 16)
    hybrid = backbone()
    train_simclr_sigreg(hybrid, cifar_two_view_loader(quick=args.quick,
                                                      labeled=False,
                                                      holdout=holdouts,
                                                      dataset=ds),
                        con_ep, lam=5.0, n_slices=cfgH["n_slices"])
    hybrid_cents = cents_of(hybrid)

    sshalves = {}
    for lam in (1.0, 5.0):
        print(f"\n----- supcon+sigreg16 (lam={lam}) -----")
        torch.manual_seed(args.seed + 15); np.random.seed(args.seed + 15)
        ss = backbone()
        train_supcon_sigreg(ss, cifar_two_view_loader(quick=args.quick,
                                                      labeled=True,
                                                      holdout=holdouts,
                                                      dataset=ds),
                            sup_ep, lam=lam, n_slices=cfgH["n_slices"])
        sshalves[lam] = (ss, cents_of(ss))

    # ----- Part A: probe suite ----------------------------------------------
    print("\n===== Part A: probe suite =====")
    results = {}
    for lam in (1.0, 5.0):
        name = f"ss[lam{lam:g}]+hybrid"
        ss, ss_cents = sshalves[lam]
        tr, tr_lab = space_embs(ss, hybrid, train_eval_loader)
        te, te_lab = space_embs(ss, hybrid, test_loader)
        anchors = torch.cat([ss_cents, hybrid_cents], dim=1)
        r = exp29.evaluate_space(tr, tr_lab, te, te_lab, anchors, seen,
                                 holdouts)
        pm, psd = probe_stat(tr, tr_lab, te, te_lab)
        g = gaussianity_summary(te, te_lab, seen, seed=args.seed)
        print(f"  [{name:<18}] probe={pm:.4f}+-{psd:.4f} acc={r['acc']:.4f} "
              f"supAUC={r['sup_auc']:.4f} eucl={r['eucl']:.4f} "
              f"mahaT={r['maha_tied']:.4f} mahaPC={r['maha_pc']:.4f}")
        results[name] = dict(probe=pm, probe_sd=psd, acc=r["acc"],
                             mahaT=r["maha_tied"], gauss=g, lam=lam)
    print("\n===== gaussianity =====")
    exp28.print_gauss_table({n: results[n]["gauss"] for n in results})

    win = max(results, key=lambda n: results[n]["probe"])
    lam_w = results[win]["lam"]
    ss, ss_cents = sshalves[lam_w]
    print(f"\n  Part B space: {win} (probe {results[win]['probe']:.4f})")

    # ----- Part B: power scan on the winner ---------------------------------
    ARM = (ss, exp28.fill_means(ss_cents, seen, cfgH), hybrid, hybrid_cents,
           cfgH)
    trains, tr_lab = space_embs(ss, hybrid, train_eval_loader)
    tests, te_lab = space_embs(ss, hybrid, test_loader)
    pre_power, post_power = {}, {s: [] for s in STATS}
    net, means, aug, cents, acfg = ARM
    anchors = torch.cat([means[seen], cents], dim=1)
    bg_mask = np.isin(te_lab, seen)
    sig_mask = np.isin(te_lab, list(holdouts))
    print("\n===== PRE power scan =====")
    d = torch.cdist(torch.as_tensor(tests, dtype=torch.float32, device=DEVICE),
                    torch.as_tensor(anchors, dtype=torch.float32,
                                    device=DEVICE))
    s = d.min(1).values.cpu().numpy()
    pe = exp30.power_at_alpha(s[bg_mask], s[sig_mask], args.alpha)
    pre_power["perevent"] = [pe] * len(fractions)
    print(f"  per-event pre power={pe:.3f}")
    R = torch.as_tensor(trains[np.isin(tr_lab, seen)][:20000],
                        dtype=torch.float32, device=DEVICE)
    bg = torch.as_tensor(tests[bg_mask], dtype=torch.float32, device=DEVICE)
    sg = torch.as_tensor(tests[sig_mask], dtype=torch.float32, device=DEVICE)
    print("  sparker")
    pre_power["sparker"], _ = exp31.run_test_battery(
        bg, sg, R, fractions, args.n_d, n_null_pre, n_sig_toys, args.alpha,
        args.seed, sparker_kw, tag="pre-spk")
    maha_fn, mmd_fn, n_bg, n_sig = exp32.make_stats_fns(
        trains, tr_lab, tests, te_lab, seen, holdouts, args.seed)
    print("  maha")
    pre_power["maha"], _ = exp32.battery(
        maha_fn, n_bg, n_sig, fractions, args.n_d, n_null_pre, n_sig_toys,
        args.alpha, args.seed, tag="pre-maha")
    print("  mmd")
    pre_power["mmd"], _ = exp32.battery(
        mmd_fn, n_bg, n_sig, fractions, args.n_d, n_null_pre, n_sig_toys,
        args.alpha, args.seed, tag="pre-mmd")

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
        bb = copy.deepcopy(net)
        _, extras = exp28.run_concat_discovery(
            bb, aug, means.clone(), cents, base=sub, dim=args.dim_half,
            train_eval_loader=tel_loader, test_loader=test_loader, seen=seen,
            holdouts=holdouts, cfg=acfg, rounds=args.rounds, ft_epochs=ft_ep,
            names=names, seed=args.seed)
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
        post_power["perevent"].append(pe)
        print(f"  per-event post f={f}: power={pe:.3f}")
        R = torch.as_tensor(tr_post[np.isin(trl_post, seen)][:20000],
                            dtype=torch.float32, device=DEVICE)
        bg = torch.as_tensor(te_post[bgm], dtype=torch.float32, device=DEVICE)
        sg = torch.as_tensor(te_post[sgm], dtype=torch.float32, device=DEVICE)
        print("  sparker (post)")
        p, _ = exp31.run_test_battery(bg, sg, R, [f], args.n_d, n_null_post,
                                      n_sig_toys, args.alpha, args.seed + i_f,
                                      sparker_kw, tag="post-spk")
        post_power["sparker"].append(p[0])
        maha_fn, mmd_fn, n_bg, n_sig = exp32.make_stats_fns(
            tr_post, trl_post, te_post, tel_post, seen, holdouts,
            args.seed + i_f)
        print("  maha (post)")
        p, _ = exp32.battery(maha_fn, n_bg, n_sig, [f], args.n_d, n_null_post,
                             n_sig_toys, args.alpha, args.seed + i_f,
                             tag="post-maha")
        post_power["maha"].append(p[0])
        print("  mmd (post)")
        p, _ = exp32.battery(mmd_fn, n_bg, n_sig, [f], args.n_d, n_null_post,
                             n_sig_toys, args.alpha, args.seed + i_f,
                             tag="post-mmd")
        post_power["mmd"].append(p[0])

    # ----- report -----------------------------------------------------------
    ref34f = np.load(F_NPZ, allow_pickle=True) if os.path.exists(F_NPZ) \
        else None
    npz = {"fractions": np.array(fractions)}
    for stat in STATS:
        print(f"\n===== EXP34G {stat.upper()} POWER SUMMARY "
              f"(alpha={args.alpha}) =====")
        print(f"  {'arm':<18}{'kind':>6}"
              + "".join(f"{f:>9}" for f in fractions))
        print(f"  {win:<18}{'pre':>6}"
              + "".join(f"{p:>9.3f}" for p in pre_power[stat]))
        print(f"  {win:<18}{'post':>6}"
              + "".join(f"{p:>9.3f}" for p in post_power[stat]))
        npz[f"{stat}_pre"] = np.array(pre_power[stat])
        npz[f"{stat}_post"] = np.array(post_power[stat])
        plt.figure(figsize=(8, 6.5))
        plt.plot(fractions, pre_power[stat], "--o", color="#d62728", lw=1.4,
                 ms=5, alpha=0.75, label=f"{win} pre")
        plt.plot(fractions, post_power[stat], "-o", color="#d62728", lw=2,
                 ms=6, label=f"{win} post")
        if ref34f is not None:
            rf = ref34f["fractions"]
            for rname, ls in [("supcon+hybrid", ":"),
                              ("hybrid->supres", "-.")]:
                key = f"{stat}_{rname}_post"
                if key in ref34f:
                    plt.plot(rf, ref34f[key], ls, color="gray", lw=1.3,
                             label=f"{rname} post (34f)")
        plt.xscale("log")
        plt.axhline(0.05, color="gray", lw=1, ls=":")
        plt.xlabel("injected anomaly fraction")
        plt.ylabel("power at alpha=0.05")
        plt.title(f"exp34g [cifar100] {win}: {stat} power vs fraction"
                  " (train-side clamp above f~0.01)")
        plt.grid(alpha=0.25, which="both")
        plt.legend(loc="upper left", fontsize=8)
        plt.tight_layout()
        out = plot_path(f"exp34g_{stat}_power_cifar100.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print("saved", out)
    outdir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "logs", "exp34")
    os.makedirs(outdir, exist_ok=True)
    np.savez(os.path.join(outdir, "power_data_ss_hybrid.npz"), **npz)
    print(f"saved {outdir}/power_data_ss_hybrid.npz")


if __name__ == "__main__":
    main()
