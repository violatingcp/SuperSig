"""
Experiment 33: the exp-29..32 study suite with dimension-matched evaluation
spaces -- 8-D supervised + 8-D feature halves for the concatenated arms and
16-D single networks, so every evaluated space is 16-dimensional.

Networks: sup16, supcon16, joint16 (single arms); sup8 (+res8 from it), ssl8
(+supres8 from it), supcon8, simclr8 (concat halves).  Arms match exp 29:

  sup->res      : [sup8 ; res8]           ssl->supres : [supres8 ; ssl8]
  joint         : joint16                 sup         : sup16
  supcon        : supcon16                supcon+simclr : [supcon8 ; simclr8]

Part A (exp-29 suite): performance/novelty table, gaussianity, latent/ROC/
corner figures, natural-fraction discovery, 1-layer-NN probe pre/post.
Part B (power grid): per injected fraction, ONE discovery rerun per arm,
then per-event margin power (exp 30), SparKer battery (exp 31) and
Mahalanobis/MMD batteries (exp 32) all on the same pre/post spaces.

    python experiments/33_dim_matched_suite.py
    python experiments/33_dim_matched_suite.py --quick --fractions 0.01,0.1
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
from sklearn.metrics import roc_curve

from supersig.config import DATA_DIR, DEVICE, plot_path
from supersig.data import (get_cifar_loaders, cifar_balanced_loader,
                           cifar_two_view_loader, cifar_two_view_balanced_loader,
                           _cifar_spec)
from supersig.losses import make_anchors
from supersig.models import CIFARResNetBackbone
from supersig.metrics import gaussianity_summary
from supersig.plotting import plot_latent_panels, plot_corner
from supersig.recipes import supervised_embedding, recipe
from supersig.discovery import run_discovery
from supersig.sparker import median_pairwise
from supersig.train import (train_sigreg_ssl, train_sigreg_hybrid,
                            train_sigreg_hybrid_aug, train_sigreg_residual_ssl,
                            train_supcon, train_simclr, collect_embeddings)

exp28 = importlib.import_module("28_concat_residual")
exp29 = importlib.import_module("29_residual_finetune")
exp30 = importlib.import_module("30_power_curves")
exp31 = importlib.import_module("31_sparker_power")
exp32 = importlib.import_module("32_maha_mmd_power")

CIFAR_NAMES = exp29.CIFAR_NAMES
COLORS = exp31.COLORS
STATS = ["perevent", "sparker", "maha", "mmd"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar10", "cifar100"],
                    default="cifar10")
    ap.add_argument("--holdout", type=int, default=4)
    ap.add_argument("--holdouts", default=None,
                    help="comma list of holdout classes (overrides --holdout)")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dim-single", type=int, default=16)
    ap.add_argument("--dim-half", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--fractions", default="0.001,0.003,0.01,0.03,0.1")
    ap.add_argument("--n-d", type=int, default=5000)
    ap.add_argument("--kernels", type=int, default=16)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--sparker-sigma", type=float, default=1.0,
                    help="fixed kernel width for SparKer (no annealing); "
                         "0 restores the annealed median-heuristic schedule")
    args = ap.parse_args()
    ds = args.dataset
    cfg16 = recipe(ds, emb_dim=args.dim_single)
    cfg8 = recipe(ds, emb_dim=args.dim_half)
    n_cls = cfg16["n_classes"]
    ssl_ep = 2 if args.quick else 20
    sup_ep = cfg16["ssl_epochs"]
    res_ep = 2 if args.quick else 10
    ft_ep = 1 if args.quick else cfg16["ft_epochs"]
    holdouts = ({int(x) for x in args.holdouts.split(",")}
                if args.holdouts else {args.holdout})
    seen = [c for c in range(n_cls) if c not in holdouts]
    fractions = [float(x) for x in args.fractions.split(",")]
    n_null_pre = 20 if args.quick else 200
    n_null_post = 20 if args.quick else 100
    n_sig_toys = 10 if args.quick else 50
    sparker_kw = dict(M=args.kernels, steps=args.steps)
    if args.sparker_sigma > 0:      # fixed width, no annealing, single scale
        sparker_kw.update(sigma0=args.sparker_sigma, sigma_ratio=1.0,
                          n_checkpoints=1)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    train_loader, test_loader = get_cifar_loaders(quick=args.quick,
                                                  dataset=ds)
    train_eval_loader = DataLoader(train_loader.dataset, batch_size=256,
                                   shuffle=False, num_workers=2)
    cls, plain, _ = _cifar_spec(ds)
    base = cls(DATA_DIR, train=True, download=True, transform=plain)
    base_targets = np.array(base.targets)
    n_base = 8000 if args.quick else len(base)
    seen_idx = np.where(np.isin(base_targets[:n_base], seen))[0]
    sig_idx_all = np.where(np.isin(base_targets[:n_base],
                               list(holdouts)))[0]
    names = (CIFAR_NAMES if ds == "cifar10" else
             [str(c) for c in range(n_cls)])
    print(f"exp33 [dim-matched] [{ds}] single={args.dim_single}d "
          f"halves={args.dim_half}+{args.dim_half}d holdout={sorted(holdouts)} "
          f"alpha={args.alpha}")

    def cents_of(net):
        e, l = collect_embeddings(net, train_eval_loader)
        m = np.isin(l, seen)
        return exp28.class_centroids(e[m], l[m], seen)

    def backbone(dim):
        return CIFARResNetBackbone(dim, arch=cfg16["arch"],
                                   pretrain=ds).to(DEVICE)

    # ----- networks ----------------------------------------------------------
    print("\n===== training: sup16 =====")
    sup16, means_sup16, _ = supervised_embedding(
        ds, holdouts=holdouts, quick=args.quick, seed=args.seed,
        emb_dim=args.dim_single)
    means_sup16 = means_sup16.detach()
    print("\n===== training: sup8 =====")
    sup8, means_sup8, _ = supervised_embedding(
        ds, holdouts=holdouts, quick=args.quick, seed=args.seed + 10,
        emb_dim=args.dim_half)
    means_sup8 = means_sup8.detach()
    print("\n===== training: res8 (classwise residual post sup8) =====")
    torch.manual_seed(args.seed + 11); np.random.seed(args.seed + 11)
    res8 = copy.deepcopy(sup8)
    train_sigreg_residual_ssl(
        res8, cifar_two_view_balanced_loader(ds, holdout=holdouts,
                                             quick=args.quick),
        res_ep, means_sup8, n_slices=cfg8["n_slices"], classwise=True)
    print("\n===== training: ssl8 =====")
    torch.manual_seed(args.seed + 12); np.random.seed(args.seed + 12)
    ssl8 = backbone(args.dim_half)
    train_sigreg_ssl(ssl8, cifar_two_view_loader(quick=args.quick,
                                                 labeled=False,
                                                 holdout=holdouts,
                                                 dataset=ds), ssl_ep)
    ssl8_cents = cents_of(ssl8)
    print("\n===== training: supres8 (supervised post ssl8) =====")
    torch.manual_seed(args.seed + 13); np.random.seed(args.seed + 13)
    supres8 = copy.deepcopy(ssl8)
    means_supres8 = exp28.fill_means(ssl8_cents, seen, cfg8).clone()
    train_sigreg_hybrid(supres8, cifar_balanced_loader(ds, holdout=holdouts,
                                                       quick=args.quick),
                        sup_ep, means_supres8, mode="repulse", disc="proto",
                        alpha=1.0, rep_weight=cfg8["rep_weight"],
                        sigreg_weight=cfg8["sigreg_weight"],
                        n_slices=cfg8["n_slices"])
    means_supres8 = means_supres8.detach()
    print("\n===== training: joint16 =====")
    torch.manual_seed(args.seed + 4); np.random.seed(args.seed + 4)
    joint16 = backbone(args.dim_single)
    means_joint16 = make_anchors(cfg16["pair_dist"] / math.sqrt(2.0),
                                 emb_dim=args.dim_single, n_classes=n_cls).clone()
    train_sigreg_hybrid_aug(joint16,
                            cifar_two_view_balanced_loader(ds, holdout=holdouts,
                                                           quick=args.quick),
                            sup_ep, means_joint16,
                            rep_weight=cfg16["rep_weight"],
                            sigreg_weight=cfg16["sigreg_weight"],
                            n_slices=cfg16["n_slices"])
    means_joint16 = means_joint16.detach()
    print("\n===== training: supcon16 =====")
    torch.manual_seed(args.seed + 5); np.random.seed(args.seed + 5)
    supcon16 = backbone(args.dim_single)
    train_supcon(supcon16, cifar_two_view_loader(quick=args.quick,
                                                 labeled=True,
                                                 holdout=holdouts,
                                                 dataset=ds), sup_ep)
    supcon16_cents = cents_of(supcon16)
    print("\n===== training: supcon8 =====")
    torch.manual_seed(args.seed + 15); np.random.seed(args.seed + 15)
    supcon8 = backbone(args.dim_half)
    train_supcon(supcon8, cifar_two_view_loader(quick=args.quick,
                                                labeled=True,
                                                holdout=holdouts,
                                                dataset=ds), sup_ep)
    supcon8_cents = cents_of(supcon8)
    print("\n===== training: simclr8 =====")
    torch.manual_seed(args.seed + 16); np.random.seed(args.seed + 16)
    simclr8 = backbone(args.dim_half)
    train_simclr(simclr8, cifar_two_view_loader(quick=args.quick,
                                                labeled=False,
                                                holdout=holdouts,
                                                dataset=ds), ssl_ep)
    simclr8_cents = cents_of(simclr8)
    res8_cents = cents_of(res8)

    # arm -> (sup net, means, aug net, aug cents, cfg of the sup half)
    ARMS = {
        "sup->res": (sup8, means_sup8, res8, res8_cents, cfg8),
        "ssl->supres": (supres8, means_supres8, ssl8, ssl8_cents, cfg8),
        "joint": (joint16, means_joint16, None, None, cfg16),
        "sup": (sup16, means_sup16, None, None, cfg16),
        "supcon": (supcon16, exp28.fill_means(supcon16_cents, seen, cfg16),
                   None, None, cfg16),
        "supcon+simclr": (supcon8, exp28.fill_means(supcon8_cents, seen, cfg8),
                          simclr8, simclr8_cents, cfg8),
    }
    arm_names = list(ARMS)

    def space_embs(net, aug, loader):
        e, l = collect_embeddings(net, loader)
        if aug is not None:
            ea, _ = collect_embeddings(aug, loader)
            e = np.concatenate([e, ea], axis=1)
        return e, l

    def arm_anchors(name):
        net, means, aug, cents, acfg = ARMS[name]
        return (torch.cat([means[seen], cents], dim=1) if aug is not None
                else means[seen])

    # ===== Part A: exp-29 metric suite ======================================
    sup_tr = {}
    tr_lab = te_lab = None
    tests, trains = {}, {}
    for name in arm_names:
        net, means, aug, cents, acfg = ARMS[name]
        trains[name], tr_lab = space_embs(net, aug, train_eval_loader)
        tests[name], te_lab = space_embs(net, aug, test_loader)

    print("\n===== performance / novelty table =====")
    print(f"  {'space':<16}{'acc':>8}{'supAUC':>8}{'eucl':>8}"
          f"{'mahaT':>8}{'mahaPC':>8}")
    perf = {}
    for name in arm_names:
        r = exp29.evaluate_space(trains[name], tr_lab, tests[name], te_lab,
                                 arm_anchors(name), seen, holdouts)
        perf[name] = r
        print(f"  {name:<16}{r['acc']:>8.4f}{r['sup_auc']:>8.4f}"
              f"{r['eucl']:>8.4f}{r['maha_tied']:>8.4f}{r['maha_pc']:>8.4f}")

    print("\n===== gaussianity (seen classes, test set) =====")
    gauss = {n: gaussianity_summary(tests[n], te_lab, seen, seed=args.seed)
             for n in arm_names}
    exp28.print_gauss_table(gauss)

    tag = f"{args.dim_half}p{args.dim_half}"
    if n_cls > 10:
        blab = np.isin(te_lab, list(holdouts)).astype(int)
        plot_latent_panels({n: (tests[n], blab) for n in arm_names}, {1},
                           ["seen", "holdout"],
                           plot_path(f"latent_{ds}_exp33_{tag}.png"),
                           title="exp33 dim-matched: " + " / ".join(arm_names))
    else:
        plot_latent_panels({n: (tests[n], te_lab) for n in arm_names},
                           holdouts, CIFAR_NAMES,
                           plot_path(f"latent_{ds}_exp33_{tag}.png"),
                           title="exp33 dim-matched: " + " / ".join(arm_names))
    colors6 = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948"]
    plt.figure(figsize=(7.5, 7))
    for (name, r), c in zip(perf.items(), colors6):
        s = r["scores"]
        fpr, tpr, _ = roc_curve(s["is_unseen"], s["maha_pc"])
        plt.plot(fpr, tpr, color=c, lw=2, label=f"{name} maha-PC ({r['maha_pc']:.3f})")
        fpr, tpr, _ = roc_curve(s["is_unseen"], s["eucl"])
        plt.plot(fpr, tpr, color=c, lw=1.2, ls="--", alpha=0.7,
                 label=f"{name} eucl ({r['eucl']:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=1, ls=":")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title(f"exp33 novelty ROC (dim-matched {tag})")
    plt.legend(loc="lower right", fontsize=8); plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(plot_path(f"exp33_novelty_roc_{ds}_{tag}.png"), dpi=150); plt.close()
    for name in (arm_names if n_cls <= 10 else []):
        te = tests[name]
        if te.shape[1] > args.dim_half and ARMS[name][2] is not None:
            sl = np.concatenate([te[:, :3], te[:, args.dim_half:args.dim_half + 3]],
                                axis=1)
            note = "3 sup + 3 aug dims"
        else:
            sl = te[:, :6]
            note = f"first 6 of {te.shape[1]}"
        plot_corner(sl, te_lab,
                    plot_path(f"corner_{ds}_exp33_{tag}_"
                              f"{name.replace('->', '_').replace('+', '_')}.png"),
                    title=f"exp33 {name} ({note})")

    # natural-fraction discovery + probe (exp-29 protocol)
    print("\n===== natural-fraction discovery + probe =====")
    hist, probe = {}, {}
    for name in arm_names:
        net, means, aug, cents, acfg = ARMS[name]
        bb = copy.deepcopy(net)
        print(f"\n----- discovery: {name} -----")
        if aug is None:
            _, hist[name] = run_discovery(
                bb, means.clone(), base_ds=base,
                train_eval_loader=train_eval_loader, test_loader=test_loader,
                seen=seen, holdouts=holdouts, dataset_name=ds,
                rep_weight=acfg["rep_weight"], sigreg_weight=acfg["sigreg_weight"],
                n_slices=acfg["n_slices"], rounds=args.rounds, ft_epochs=ft_ep,
                names=names, seed=args.seed)
        else:
            hist[name], _ = exp28.run_concat_discovery(
                bb, aug, means.clone(), cents, base=base, dim=args.dim_half,
                train_eval_loader=train_eval_loader, test_loader=test_loader,
                seen=seen, holdouts=holdouts, cfg=acfg, rounds=args.rounds,
                ft_epochs=ft_ep, names=names, seed=args.seed)
        tr_post, _ = space_embs(bb, aug, train_eval_loader)
        te_post, _ = space_embs(bb, aug, test_loader)
        a_pre, _, _ = exp29.linear_probe_novelty(trains[name], tr_lab,
                                                 tests[name], te_lab, holdouts)
        a_post, _, _ = exp29.linear_probe_novelty(tr_post, tr_lab, te_post,
                                                  te_lab, holdouts)
        probe[name] = (a_pre, a_post)
        print(f"  probe pre={a_pre:.4f} post={a_post:.4f}")

    # ===== Part B-D: power grid, one discovery per (arm, fraction) ==========
    print("\n===== PRE power batteries (all statistics) =====")
    pre_power = {s: {} for s in STATS}
    for name in arm_names:
        net, means, aug, cents, acfg = ARMS[name]
        tr, trl = trains[name], tr_lab
        te, tel = tests[name], te_lab
        anchors = arm_anchors(name)
        bg_mask = np.isin(tel, seen)
        # per-event (constant in f)
        d = torch.cdist(torch.as_tensor(te, dtype=torch.float32, device=DEVICE),
                        torch.as_tensor(anchors, dtype=torch.float32,
                                        device=DEVICE))
        s = d.min(1).values.cpu().numpy()
        pe = exp30.power_at_alpha(s[bg_mask], s[np.isin(tel, list(holdouts))],
                                  args.alpha)
        pre_power["perevent"][name] = [pe] * len(fractions)
        print(f"  [{name}] per-event pre power={pe:.3f}")
        # sparker
        R = torch.as_tensor(tr[np.isin(trl, seen)][:20000], dtype=torch.float32,
                            device=DEVICE)
        bg = torch.as_tensor(te[bg_mask], dtype=torch.float32, device=DEVICE)
        sg = torch.as_tensor(te[np.isin(tel, list(holdouts))],
                     dtype=torch.float32,
                             device=DEVICE)
        print(f"  [{name}] sparker")
        pre_power["sparker"][name], _ = exp31.run_test_battery(
            bg, sg, R, fractions, args.n_d, n_null_pre, n_sig_toys,
            args.alpha, args.seed, sparker_kw, tag="pre-spk")
        # maha + mmd
        maha_fn, mmd_fn, n_bg, n_sig = exp32.make_stats_fns(
            tr, trl, te, tel, seen, holdouts, args.seed)
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
        print(f"\n===== POST grid, f={f} ({len(inj)} injected) =====")
        for name in arm_names:
            net, means, aug, cents, acfg = ARMS[name]
            bb = copy.deepcopy(net)
            if aug is None:
                cur_means, _ = run_discovery(
                    bb, means.clone(), base_ds=sub,
                    train_eval_loader=tel_loader, test_loader=test_loader,
                    seen=seen, holdouts=holdouts, dataset_name=ds,
                    rep_weight=acfg["rep_weight"],
                    sigreg_weight=acfg["sigreg_weight"],
                    n_slices=acfg["n_slices"], rounds=args.rounds,
                    ft_epochs=ft_ep, names=names, seed=args.seed)
                te_post, tel_post = space_embs(bb, None, test_loader)
                zt = torch.as_tensor(te_post, dtype=torch.float32,
                                     device=DEVICE)
                d_seen = torch.cdist(zt, cur_means[seen]).min(1).values
                d_disc = torch.cdist(zt, cur_means[n_cls:]).min(1).values
            else:
                _, extras = exp28.run_concat_discovery(
                    bb, aug, means.clone(), cents, base=sub,
                    dim=args.dim_half, train_eval_loader=tel_loader,
                    test_loader=test_loader, seen=seen, holdouts=holdouts,
                    cfg=acfg, rounds=args.rounds, ft_epochs=ft_ep,
                    names=names, seed=args.seed)
                cur_means, disc_ssl = extras["cur_means"], extras["disc_ssl"]
                te_post, tel_post = space_embs(bb, aug, test_loader)
                zt = torch.as_tensor(te_post, dtype=torch.float32,
                                     device=DEVICE)
                seen_anc = torch.cat([cur_means[seen], cents], dim=1)
                disc_anc = torch.cat([cur_means[n_cls:], disc_ssl], dim=1)
                d_seen = torch.cdist(zt, seen_anc).min(1).values
                d_disc = torch.cdist(zt, disc_anc).min(1).values
            tr_post, trl_post = space_embs(bb, aug, train_eval_loader)
            bg_mask = np.isin(tel_post, seen)
            # per-event margin
            s = (d_seen - d_disc).cpu().numpy()
            pe = exp30.power_at_alpha(s[bg_mask],
                                      s[np.isin(tel_post, list(holdouts))], args.alpha)
            post_power["perevent"][name].append(pe)
            print(f"  [{name}] per-event post f={f}: power={pe:.3f}")
            # sparker
            R = torch.as_tensor(tr_post[np.isin(trl_post, seen)][:20000],
                                dtype=torch.float32, device=DEVICE)
            bg = torch.as_tensor(te_post[bg_mask], dtype=torch.float32,
                                 device=DEVICE)
            sg = torch.as_tensor(te_post[np.isin(tel_post, list(holdouts))],
                                 dtype=torch.float32, device=DEVICE)
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

    # ===== report ============================================================
    for stat in STATS:
        print(f"\n===== EXP33 {stat.upper()} POWER SUMMARY "
              f"(alpha={args.alpha}, {tag}) =====")
        print(f"  {'arm':<16}{'kind':>6}"
              + "".join(f"{f:>9}" for f in fractions))
        for name in arm_names:
            print(f"  {name:<16}{'pre':>6}"
                  + "".join(f"{p:>9.3f}" for p in pre_power[stat][name]))
            print(f"  {name:<16}{'post':>6}"
                  + "".join(f"{p:>9.3f}" for p in post_power[stat][name]))
        plt.figure(figsize=(8, 6.5))
        for name in arm_names:
            c = COLORS[name]
            plt.plot(fractions, pre_power[stat][name], "--o", color=c, lw=1.4,
                     ms=5, alpha=0.75, label=f"{name} pre")
            plt.plot(fractions, post_power[stat][name], "-o", color=c, lw=2,
                     ms=6, label=f"{name} post")
        plt.xscale("log")
        plt.axhline(args.alpha, color="gray", lw=1, ls=":")
        plt.xlabel("injected anomaly fraction")
        plt.ylabel(f"power at alpha={args.alpha}")
        plt.title(f"exp33 dim-matched ({tag}): {stat} power vs fraction")
        plt.grid(alpha=0.25, which="both")
        plt.legend(loc="upper left", fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(plot_path(f"exp33_{stat}_power_{ds}_{tag}.png"), dpi=150)
        plt.close()
        print("  saved " + plot_path(f"exp33_{stat}_power_{ds}_{tag}.png"))

    print(f"\n===== EXP33 SUMMARY (part A, {tag}) =====")
    for name in arm_names:
        r = perf[name]
        print(f"  [{name:<14}] acc={r['acc']:.4f} supAUC={r['sup_auc']:.4f} "
              f"eucl={r['eucl']:.4f} mahaT={r['maha_tied']:.4f} "
              f"mahaPC={r['maha_pc']:.4f} probe={probe[name][0]:.4f}"
              f"/{probe[name][1]:.4f}")
        for h in hist[name]:
            print(f"          round {h['round']}: purity={h['purity']:.3f} "
                  f"anchors={h['n_anchors']}  margin={h['margin']:.4f}  "
                  f"mean-anchor={h['mean_pc']:.4f}")
    os.makedirs(os.path.join("logs", "exp33"), exist_ok=True)
    np.savez(os.path.join("logs", "exp33", f"power_data_{ds}_{tag}.npz"),
             fractions=np.array(fractions),
             **{f"{s}_{n}_pre": np.array(pre_power[s][n])
                for s in STATS for n in arm_names},
             **{f"{s}_{n}_post": np.array(post_power[s][n])
                for s in STATS for n in arm_names})
    print("Done.")


if __name__ == "__main__":
    main()
