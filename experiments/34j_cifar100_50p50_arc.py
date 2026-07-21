"""
Experiment 34j: the calibrated-contrastive arc on CIFAR-100 at HIGH dimension
-- 50+50 halves (100-D concat spaces, matching the exp-33 100-D scan where
C100 accuracy peaked) to test whether more room relieves the class crowding
that shaped the 16+16 results.

Part A (probe suite, 9 spaces) and Part B (four-statistic power scan on the
five arms) exactly as exp 34i, with the 16+16 CIFAR-100 numbers (exps
34e/f/g/h) as the comparison point:
  probe: supcon+simclr 0.9394, supcon+hybrid[lam5] 0.9423, ss[lam5]+hybrid
  0.9235, hybrid->supres 0.9263, cls->resfeat 0.8745, feat->rescls 0.8196.
  power at f=0.02 (pre/post): supcon+hybrid spk 0.06/0.30 mmd 0.30/0.88;
  ss+hybrid spk 0.10/0.52 mmd 0.38/0.86; cls->resfeat spk 0.50/0.44
  mmd 0.52/0.74; maha dead everywhere.

    python experiments/34j_cifar100_50p50_arc.py
    python experiments/34j_cifar100_50p50_arc.py --quick --fractions 0.01,0.05
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
from supersig.metrics import gaussianity_summary
from supersig.recipes import recipe
from supersig.train import (train_sigreg_hybrid, train_supcon, train_simclr,
                            train_supcon_sigreg, train_simclr_sigreg,
                            train_simclr_residual,
                            train_supcon_sigreg_residual, collect_embeddings)

exp28 = importlib.import_module("28_concat_residual")
exp29 = importlib.import_module("29_residual_finetune")
exp30 = importlib.import_module("30_power_curves")
exp31 = importlib.import_module("31_sparker_power")
exp32 = importlib.import_module("32_maha_mmd_power")

STATS = ["perevent", "sparker", "maha", "mmd"]
POWER_ARMS = ["supcon+hybrid[lam5]", "ss[lam5]+hybrid", "hybrid->supres",
              "cls->resfeat", "feat->rescls"]
CPB, PC = 99, 24


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dim-half", type=int, default=50)
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
    print(f"exp34j [{ds}] calibrated-contrastive arc at "
          f"{args.dim_half}+{args.dim_half}d, holdout={sorted(holdouts)}")

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

    def two_view(labeled):
        return cifar_two_view_loader(quick=args.quick, labeled=labeled,
                                     holdout=holdouts, dataset=ds)

    # ----- networks ---------------------------------------------------------
    print("\n----- supcon50 -----")
    torch.manual_seed(args.seed + 15); np.random.seed(args.seed + 15)
    supcon = backbone()
    train_supcon(supcon, two_view(True), sup_ep)
    supcon_cents = cents_of(supcon)
    print("\n----- simclr50 -----")
    torch.manual_seed(args.seed + 16); np.random.seed(args.seed + 16)
    simclr = backbone()
    train_simclr(simclr, two_view(False), con_ep)
    simclr_cents = cents_of(simclr)
    hybrids = {}
    for lam in (1.0, 5.0):
        print(f"\n----- hybrid50 simclr+sigreg (lam={lam}) -----")
        torch.manual_seed(args.seed + 16); np.random.seed(args.seed + 16)
        h = backbone()
        train_simclr_sigreg(h, two_view(False), con_ep, lam=lam,
                            n_slices=cfgH["n_slices"])
        hybrids[lam] = (h, cents_of(h))
    sshalves = {}
    for lam in (1.0, 5.0):
        print(f"\n----- ss50 supcon+sigreg (lam={lam}) -----")
        torch.manual_seed(args.seed + 15); np.random.seed(args.seed + 15)
        ss = backbone()
        train_supcon_sigreg(ss, two_view(True), sup_ep, lam=lam,
                            n_slices=cfgH["n_slices"])
        sshalves[lam] = (ss, cents_of(ss))
    print("\n----- supres50 from hybrid(lam=1) trunk (w=1, batch 99x24) -----")
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
                        sigreg_weight=cfgH["sigreg_weight"],
                        n_slices=cfgH["n_slices"])
    means_supres = means_supres.detach()
    print("\n----- res-simclr50 (SimCLR on supcon residual) -----")
    torch.manual_seed(args.seed + 17); np.random.seed(args.seed + 17)
    res_simclr = copy.deepcopy(supcon)
    train_simclr_residual(res_simclr, two_view(True), con_ep,
                          exp28.fill_means(supcon_cents, seen,
                                           cfgH).to(DEVICE))
    res_simclr_cents = cents_of(res_simclr)
    print("\n----- resfeat50 (simclr+sigreg on ss5 residual, lam=5) -----")
    torch.manual_seed(args.seed + 17); np.random.seed(args.seed + 17)
    resfeat = copy.deepcopy(sshalves[5.0][0])
    train_simclr_residual(resfeat, two_view(True), con_ep,
                          exp28.fill_means(sshalves[5.0][1], seen,
                                           cfgH).to(DEVICE),
                          lam=5.0, n_slices=cfgH["n_slices"])
    resfeat_cents = cents_of(resfeat)
    print("\n----- rescls50 (supcon+sigreg on hybrid5 residual, lam=5) -----")
    torch.manual_seed(args.seed + 18); np.random.seed(args.seed + 18)
    rescls = copy.deepcopy(hybrids[5.0][0])
    train_supcon_sigreg_residual(rescls, two_view(True), sup_ep,
                                 exp28.fill_means(hybrids[5.0][1], seen,
                                                  cfgH).to(DEVICE),
                                 lam=5.0, n_slices=cfgH["n_slices"])
    rescls_cents = cents_of(rescls)

    ARMS = {
        "supcon+simclr": (supcon, supcon_cents, simclr, simclr_cents),
        "supcon+hybrid[lam1]": (supcon, supcon_cents,
                                hybrids[1.0][0], hybrids[1.0][1]),
        "supcon+hybrid[lam5]": (supcon, supcon_cents,
                                hybrids[5.0][0], hybrids[5.0][1]),
        "ss[lam1]+hybrid": (sshalves[1.0][0], sshalves[1.0][1],
                            hybrids[5.0][0], hybrids[5.0][1]),
        "ss[lam5]+hybrid": (sshalves[5.0][0], sshalves[5.0][1],
                            hybrids[5.0][0], hybrids[5.0][1]),
        "hybrid->supres": (supres, means_supres[seen],
                           hybrids[1.0][0], hybrids[1.0][1]),
        "supcon+res-simclr": (supcon, supcon_cents,
                              res_simclr, res_simclr_cents),
        "cls->resfeat": (sshalves[5.0][0], sshalves[5.0][1],
                         resfeat, resfeat_cents),
        "feat->rescls": (rescls, rescls_cents,
                         hybrids[5.0][0], hybrids[5.0][1]),
    }

    # ----- Part A: probe suite ----------------------------------------------
    print("\n===== Part A: probe suite =====")
    trains, tests = {}, {}
    tr_lab = te_lab = None
    resultsA = {}
    for name, (c_net, c_anc, f_net, f_cents) in ARMS.items():
        trains[name], tr_lab = space_embs(c_net, f_net, train_eval_loader)
        tests[name], te_lab = space_embs(c_net, f_net, test_loader)
        anchors = torch.cat([c_anc, f_cents], dim=1)
        r = exp29.evaluate_space(trains[name], tr_lab, tests[name], te_lab,
                                 anchors, seen, holdouts)
        pm, psd = probe_stat(trains[name], tr_lab, tests[name], te_lab)
        g = gaussianity_summary(tests[name], te_lab, seen, seed=args.seed)
        print(f"  [{name:<20}] probe={pm:.4f}+-{psd:.4f} acc={r['acc']:.4f} "
              f"supAUC={r['sup_auc']:.4f} eucl={r['eucl']:.4f} "
              f"mahaT={r['maha_tied']:.4f} mahaPC={r['maha_pc']:.4f}")
        resultsA[name] = dict(probe=pm, probe_sd=psd, acc=r["acc"],
                              sup_auc=r["sup_auc"], eucl=r["eucl"],
                              mahaT=r["maha_tied"], mahaPC=r["maha_pc"],
                              gauss=g)
    print("\n===== gaussianity =====")
    exp28.print_gauss_table({n: resultsA[n]["gauss"] for n in resultsA})

    # ----- Part B: power scan -----------------------------------------------
    pre_power = {s: {} for s in STATS}
    for name in POWER_ARMS:
        c_net, c_anc, f_net, f_cents = ARMS[name]
        tr, te = trains[name], tests[name]
        anchors = torch.cat([c_anc, f_cents], dim=1)
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

    post_power = {s: {n: [] for n in POWER_ARMS} for s in STATS}
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
        for name in POWER_ARMS:
            c_net, c_anc, f_net, f_cents = ARMS[name]
            bb = copy.deepcopy(c_net)
            if name == "hybrid->supres":
                init_means = means_supres.clone()
            else:
                init_means = exp28.fill_means(
                    torch.as_tensor(c_anc, device=DEVICE), seen, cfgH).clone()
            _, extras = exp28.run_concat_discovery(
                bb, f_net, init_means, f_cents, base=sub, dim=args.dim_half,
                train_eval_loader=tel_loader, test_loader=test_loader,
                seen=seen, holdouts=holdouts, cfg=cfgH, rounds=args.rounds,
                ft_epochs=ft_ep, names=names, seed=args.seed)
            cur_means, disc_ssl = extras["cur_means"], extras["disc_ssl"]
            te_post, tel_post = space_embs(bb, f_net, test_loader)
            zt = torch.as_tensor(te_post, dtype=torch.float32, device=DEVICE)
            seen_anc = torch.cat([cur_means[seen], f_cents], dim=1)
            disc_anc = torch.cat([cur_means[n_cls:], disc_ssl], dim=1)
            d_seen = torch.cdist(zt, seen_anc).min(1).values
            d_disc = torch.cdist(zt, disc_anc).min(1).values
            tr_post, trl_post = space_embs(bb, f_net, train_eval_loader)
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

    # ----- report -----------------------------------------------------------
    npz = {"fractions": np.array(fractions)}
    for name in ARMS:
        npz[f"probe_{name}"] = np.array([resultsA[name]["probe"],
                                         resultsA[name]["probe_sd"]])
        for k in ("acc", "eucl", "mahaT", "mahaPC"):
            npz[f"{k}_{name}"] = np.array(resultsA[name][k])
    for stat in STATS:
        print(f"\n===== EXP34J {stat.upper()} POWER SUMMARY "
              f"(alpha={args.alpha}, {args.dim_half}+{args.dim_half}d) =====")
        print(f"  {'arm':<22}{'kind':>6}"
              + "".join(f"{f:>9}" for f in fractions))
        for name in POWER_ARMS:
            print(f"  {name:<22}{'pre':>6}"
                  + "".join(f"{p:>9.3f}" for p in pre_power[stat][name]))
            print(f"  {name:<22}{'post':>6}"
                  + "".join(f"{p:>9.3f}" for p in post_power[stat][name]))
            npz[f"{stat}_{name}_pre"] = np.array(pre_power[stat][name])
            npz[f"{stat}_{name}_post"] = np.array(post_power[stat][name])
        plt.figure(figsize=(8.5, 6.5))
        cols = ["#d62728", "#2a78d6", "#1baf7a", "#eda100", "#4a3aa7"]
        for name, c in zip(POWER_ARMS, cols):
            plt.plot(fractions, pre_power[stat][name], "--o", color=c, lw=1.3,
                     ms=4, alpha=0.7, label=f"{name} pre")
            plt.plot(fractions, post_power[stat][name], "-o", color=c, lw=2,
                     ms=5, label=f"{name} post")
        plt.xscale("log")
        plt.axhline(0.05, color="gray", lw=1, ls=":")
        plt.xlabel("injected anomaly fraction")
        plt.ylabel("power at alpha=0.05")
        plt.title(f"exp34j [cifar100 {args.dim_half}+{args.dim_half}d]: "
                  f"{stat} power (train-side clamp above f~0.01)")
        plt.grid(alpha=0.25, which="both")
        plt.legend(loc="upper left", fontsize=7, ncol=2)
        plt.tight_layout()
        out = plot_path(f"exp34j_{stat}_power_cifar100_50p50.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print("saved", out)
    outdir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "logs", "exp34")
    os.makedirs(outdir, exist_ok=True)
    np.savez(os.path.join(outdir, "arc_cifar100_50p50.npz"), **npz)
    print(f"saved {outdir}/arc_cifar100_50p50.npz")


if __name__ == "__main__":
    main()
