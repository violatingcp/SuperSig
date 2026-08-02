"""
Experiment 59: NPLM residual and concatenated spaces (CIFAR-10, 16+16).

Extends the exp-33/36 space constructions with NPLM ingredients.  Halves
(16-D each; total evaluated spaces 32-D):

  sup16          supervised SIGReg recipe (supervised_embedding)
  supcon16       SupCon
  nplm_bil16     instance / bilinear / nplm / sigreg   (label-free)
  nplm_sup16     supervised / distance / nplm / sigreg
  res_nplm16     deepcopy(sup16) + NPLM residual: instance/bilinear/nplm +
                 global sigreg on r = z - means_sup[y]   (exp-36 pattern,
                 NPLM replacing NT-Xent)
  res_cls16      deepcopy(nplm_sup16) + classic classwise-SIGReg residual
                 around nplm_sup16's centroids

Arms:
  sup->res-nplm   [sup16 ; res_nplm16]
  nplmsup->res    [nplm_sup16 ; res_cls16]
  supcon+nplm     [supcon16 ; nplm_bil16]
  sup+nplm        [sup16 ; nplm_bil16]
  nplmsup+nplm    [nplm_sup16 ; nplm_bil16]

Part A (exp-29 suite, 3-seed probe) + pre power batteries + natural
discovery (probe pre/post) + exp-33-style concat post-discovery grid.
SparKer uses the annealed median-heuristic sigma throughout (exp-57 rule).

    python experiments/59_nplm_residual_concat.py
    python experiments/59_nplm_residual_concat.py --quick --arms supcon+nplm --fractions 0.01,0.1
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
from supersig.losses import HybridContrastiveLoss, make_anchors
from supersig.models import CIFARResNetBackbone
from supersig.metrics import gaussianity_summary
from supersig.recipes import supervised_embedding, recipe
from supersig.train import (train_supcon, train_sigreg_residual_ssl,
                            collect_embeddings)

exp28 = importlib.import_module("28_concat_residual")
exp29 = importlib.import_module("29_residual_finetune")
exp30 = importlib.import_module("30_power_curves")
exp31 = importlib.import_module("31_sparker_power")
exp32 = importlib.import_module("32_maha_mmd_power")
exp34h = importlib.import_module("34h_hybrid_nplm_cifar")
exp53 = importlib.import_module("53_nplm_classwise")

STATS = ["perevent", "sparker", "maha", "mmd"]
COLORS = {"sup->res-nplm": "#2a78d6", "nplmsup->res": "#8c2d9e",
          "supcon+nplm": "#eda100", "sup+nplm": "#1baf7a",
          "nplmsup+nplm": "#d62728"}


def train_nplm_residual(backbone, loader, epochs, means, tau=1.0, lam=1.0,
                        n_slices=64, lr=1e-3):
    """NPLM interaction + global SIGReg on residuals r = z - means[y]."""
    loss_fn = HybridContrastiveLoss(positives="instance", critic="bilinear",
                                    estimator="nplm", marginal="sigreg",
                                    tau=tau, lam=lam, n_slices=n_slices)
    opt = torch.optim.Adam(backbone.parameters(), lr=lr)
    backbone.train()
    for ep in range(epochs):
        int_run, marg_run, n = 0.0, 0.0, 0
        for v1, v2, y in loader:
            v1, v2, y = v1.to(DEVICE), v2.to(DEVICE), y.to(DEVICE)
            inst = torch.arange(v1.size(0), device=DEVICE)
            labels = torch.cat([inst, inst])
            opt.zero_grad()
            z = backbone(torch.cat([v1, v2]))
            r = z - means[torch.cat([y, y])]
            loss, parts = loss_fn(r, labels)
            loss.backward()
            opt.step()
            int_run += parts["interaction"].item() * v1.size(0)
            marg_run += parts["marginal"].item() * v1.size(0)
            n += v1.size(0)
        n = max(n, 1)
        print(f"  [nplm-res] epoch {ep+1}/{epochs}  "
              f"interaction={int_run/n:.4f}  marginal={marg_run/n:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--holdout", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dim-half", type=int, default=16)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--fractions", default="0.001,0.003,0.01,0.02,0.03,0.1")
    ap.add_argument("--n-d", type=int, default=5000)
    ap.add_argument("--kernels", type=int, default=16)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--arms", nargs="+", default=list(COLORS),
                    choices=list(COLORS))
    ap.add_argument("--cw-lam", type=float, default=None,
                    help="use classwise-NPLM (this lam) for the nplm_sup "
                         "half instead of the global-marginal variant")
    ap.add_argument("--skip-power", action="store_true")
    args = ap.parse_args()
    ds = args.dataset
    dtag = ds
    if args.dim_half != 16:
        dtag += f"_{2 * args.dim_half}d"
    if args.cw_lam is not None:
        dtag += f"_cwlam{args.cw_lam:g}"

    cfgH = recipe(ds, emb_dim=args.dim_half)
    n_cls = cfgH["n_classes"]
    holdouts = {args.holdout}
    seen = [c for c in range(n_cls) if c not in holdouts]
    con_ep = 2 if args.quick else 20
    res_ep = 2 if args.quick else 10
    ft_ep = 1 if args.quick else cfgH["ft_epochs"]
    fractions = [float(x) for x in args.fractions.split(",")]
    n_null_pre = 20 if args.quick else 200
    n_null_post = 20 if args.quick else 100
    n_sig_toys = 10 if args.quick else 50
    sparker_kw = dict(M=args.kernels, steps=args.steps)  # annealed sigma
    names = exp29.CIFAR_NAMES if ds == "cifar10" else None
    print(f"exp59 [{ds}] NPLM residual+concat, halves={args.dim_half}d, "
          f"holdout={sorted(holdouts)}, arms={args.arms}")

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

    # ----- halves -----------------------------------------------------------
    print("\n===== training: sup16 =====")
    sup16, means_sup, _ = supervised_embedding(
        ds, holdouts=holdouts, quick=args.quick, seed=args.seed + 10,
        emb_dim=args.dim_half)
    means_sup = means_sup.detach()

    print("\n===== training: res_nplm16 (NPLM residual post sup16) =====")
    torch.manual_seed(args.seed + 11); np.random.seed(args.seed + 11)
    res_nplm16 = copy.deepcopy(sup16)
    train_nplm_residual(res_nplm16,
                        cifar_two_view_loader(quick=args.quick, labeled=True,
                                              holdout=holdouts, dataset=ds),
                        con_ep, means_sup, tau=args.tau, lam=args.lam,
                        n_slices=cfgH["n_slices"])

    print("\n===== training: supcon16 =====")
    torch.manual_seed(args.seed + 15); np.random.seed(args.seed + 15)
    supcon16 = CIFARResNetBackbone(args.dim_half, arch=cfgH["arch"],
                                   pretrain=ds).to(DEVICE)
    train_supcon(supcon16, cifar_two_view_loader(quick=args.quick,
                                                 labeled=True,
                                                 holdout=holdouts,
                                                 dataset=ds), con_ep)

    print("\n===== training: nplm_bil16 =====")
    torch.manual_seed(args.seed + 16); np.random.seed(args.seed + 16)
    nplm_bil16 = CIFARResNetBackbone(args.dim_half, arch=cfgH["arch"],
                                     pretrain=ds).to(DEVICE)
    exp34h.train_hybrid(nplm_bil16,
                        cifar_two_view_loader(quick=args.quick, labeled=False,
                                              holdout=holdouts, dataset=ds),
                        con_ep, dict(positives="instance", critic="bilinear",
                                     estimator="nplm", marginal="sigreg",
                                     tau=args.tau), False,
                        lam=args.lam, n_slices=cfgH["n_slices"])

    print("\n===== training: nplm_sup16 =====")
    torch.manual_seed(args.seed + 17); np.random.seed(args.seed + 17)
    nplm_sup16 = CIFARResNetBackbone(args.dim_half, arch=cfgH["arch"],
                                     pretrain=ds).to(DEVICE)
    if args.cw_lam is not None:
        cw_means = make_anchors(cfgH["pair_dist"] / math.sqrt(2.0),
                                emb_dim=args.dim_half,
                                n_classes=n_cls).detach()
        exp53.train_nplm_classwise(
            nplm_sup16,
            cifar_two_view_balanced_loader(ds, holdout=holdouts,
                                           quick=args.quick),
            con_ep, "supervised", "distance", cw_means, tau=args.tau,
            lam=args.cw_lam, n_slices=cfgH["n_slices"])
    else:
        exp34h.train_hybrid(nplm_sup16,
                            cifar_two_view_loader(quick=args.quick,
                                                  labeled=True,
                                                  holdout=holdouts,
                                                  dataset=ds),
                            con_ep,
                            dict(positives="supervised", critic="distance",
                                 estimator="nplm", marginal="sigreg",
                                 tau=args.tau), True,
                            lam=args.lam, n_slices=cfgH["n_slices"])
    means_nplmsup = exp28.fill_means(cents_of(nplm_sup16), seen, cfgH).detach()

    print("\n===== training: res_cls16 (classwise residual post nplm_sup16) "
          "=====")
    torch.manual_seed(args.seed + 12); np.random.seed(args.seed + 12)
    res_cls16 = copy.deepcopy(nplm_sup16)
    train_sigreg_residual_ssl(
        res_cls16, cifar_two_view_balanced_loader(ds, holdout=holdouts,
                                                  quick=args.quick),
        res_ep, means_nplmsup, n_slices=cfgH["n_slices"], classwise=True)

    means_supcon = exp28.fill_means(cents_of(supcon16), seen, cfgH).detach()
    cents = {n: torch.as_tensor(cents_of(net), dtype=torch.float32,
                                device=DEVICE)
             for n, net in (("res_nplm", res_nplm16), ("res_cls", res_cls16),
                            ("nplm_bil", nplm_bil16))}

    # arm -> (sup net, means, aug net, aug cents)
    ARMS = {
        "sup->res-nplm": (sup16, means_sup, res_nplm16, cents["res_nplm"]),
        "nplmsup->res": (nplm_sup16, means_nplmsup, res_cls16,
                         cents["res_cls"]),
        "supcon+nplm": (supcon16, means_supcon, nplm_bil16,
                        cents["nplm_bil"]),
        "sup+nplm": (sup16, means_sup, nplm_bil16, cents["nplm_bil"]),
        "nplmsup+nplm": (nplm_sup16, means_nplmsup, nplm_bil16,
                         cents["nplm_bil"]),
    }

    def space_embs(net, aug, loader):
        e, l = collect_embeddings(net, loader)
        ea, _ = collect_embeddings(aug, loader)
        return np.concatenate([e, ea], axis=1), l

    def arm_anchors(name):
        _, means, _, c = ARMS[name]
        return torch.cat([means[seen], c], dim=1)

    def probe_stat(tr, tr_lab, te, te_lab, n_rep=3):
        aucs = []
        for s in range(n_rep):
            torch.manual_seed(1000 + s)
            a, _, _ = exp29.linear_probe_novelty(tr, tr_lab, te, te_lab,
                                                 holdouts)
            aucs.append(a)
        return float(np.mean(aucs)), float(np.std(aucs))

    # ===== Part A ===========================================================
    trains, tests, results = {}, {}, {}
    tr_lab = te_lab = None
    for name in args.arms:
        net, means, aug, c = ARMS[name]
        trains[name], tr_lab = space_embs(net, aug, train_eval_loader)
        tests[name], te_lab = space_embs(net, aug, test_loader)
        r = exp29.evaluate_space(trains[name], tr_lab, tests[name], te_lab,
                                 arm_anchors(name), seen, holdouts)
        pm, psd = probe_stat(trains[name], tr_lab, tests[name], te_lab)
        g = gaussianity_summary(tests[name], te_lab, seen, seed=args.seed)
        print(f"  [{name:<14}] probe={pm:.4f}+-{psd:.4f} acc={r['acc']:.4f} "
              f"supAUC={r['sup_auc']:.4f} eucl={r['eucl']:.4f} "
              f"mahaT={r['maha_tied']:.4f}")
        results[name] = dict(probe=pm, probe_sd=psd, acc=r["acc"],
                             sup_auc=r["sup_auc"], eucl=r["eucl"],
                             mahaT=r["maha_tied"], mahaPC=r["maha_pc"],
                             gauss=g)

    print("\n===== performance / novelty table =====")
    print(f"  {'arm':<16}{'probe':>16}{'acc':>8}{'eucl':>8}{'mahaT':>8}")
    for name in args.arms:
        r = results[name]
        print(f"  {name:<16}{r['probe']:>9.4f}+-{r['probe_sd']:.4f}"
              f"{r['acc']:>8.4f}{r['eucl']:>8.4f}{r['mahaT']:>8.4f}")
    print("\n===== gaussianity =====")
    exp28.print_gauss_table({n: results[n]["gauss"] for n in args.arms})

    # ===== natural discovery: probe pre/post ================================
    probe_post = {}
    for name in args.arms:
        net, means, aug, c = ARMS[name]
        print(f"\n----- natural discovery: {name} -----")
        bb = copy.deepcopy(net)
        exp28.run_concat_discovery(
            bb, aug, means.clone(), c, base=base, dim=args.dim_half,
            train_eval_loader=train_eval_loader, test_loader=test_loader,
            seen=seen, holdouts=holdouts, cfg=cfgH, rounds=args.rounds,
            ft_epochs=ft_ep, names=names, seed=args.seed)
        tr_post, _ = space_embs(bb, aug, train_eval_loader)
        te_post, _ = space_embs(bb, aug, test_loader)
        a_post, _, _ = exp29.linear_probe_novelty(tr_post, tr_lab, te_post,
                                                  te_lab, holdouts)
        probe_post[name] = a_post
        print(f"  probe pre={results[name]['probe']:.4f} post={a_post:.4f}")
        del bb
        torch.cuda.empty_cache()

    # ===== power: pre + post grid ===========================================
    pre_power = {s: {} for s in STATS}
    post_power = {s: {n: [] for n in args.arms} for s in STATS}
    if not args.skip_power:
        print("\n===== PRE power batteries =====")
        for name in args.arms:
            tr, te = trains[name], tests[name]
            bg_mask = np.isin(te_lab, seen)
            sig_mask = np.isin(te_lab, list(holdouts))
            d = torch.cdist(torch.as_tensor(te, dtype=torch.float32,
                                            device=DEVICE),
                            arm_anchors(name))
            s = d.min(1).values.cpu().numpy()
            pe = exp30.power_at_alpha(s[bg_mask], s[sig_mask], args.alpha)
            pre_power["perevent"][name] = [pe] * len(fractions)
            print(f"  [{name}] per-event pre power={pe:.3f}")
            R = torch.as_tensor(tr[np.isin(tr_lab, seen)][:20000],
                                dtype=torch.float32, device=DEVICE)
            bg = torch.as_tensor(te[bg_mask], dtype=torch.float32,
                                 device=DEVICE)
            sg = torch.as_tensor(te[sig_mask], dtype=torch.float32,
                                 device=DEVICE)
            print(f"  [{name}] sparker (annealed)")
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

        for i_f, f in enumerate(fractions):
            n_inj = int(round(f * len(seen_idx) / (1.0 - f)))
            rng = np.random.default_rng(args.seed * 1000 + i_f)
            inj = rng.choice(sig_idx_all, size=min(n_inj, len(sig_idx_all)),
                             replace=False)
            sub = Subset(base, np.concatenate([seen_idx, inj]).tolist())
            tel_loader = DataLoader(sub, batch_size=256, shuffle=False,
                                    num_workers=2)
            print(f"\n===== POST grid, f={f} ({len(inj)} injected) =====")
            for name in args.arms:
                net, means, aug, c = ARMS[name]
                bb = copy.deepcopy(net)
                _, extras = exp28.run_concat_discovery(
                    bb, aug, means.clone(), c, base=sub, dim=args.dim_half,
                    train_eval_loader=tel_loader, test_loader=test_loader,
                    seen=seen, holdouts=holdouts, cfg=cfgH,
                    rounds=args.rounds, ft_epochs=ft_ep, names=names,
                    seed=args.seed)
                cur_means, disc_ssl = extras["cur_means"], extras["disc_ssl"]
                te_post, tel_post = space_embs(bb, aug, test_loader)
                tr_post, trl_post = space_embs(bb, aug, train_eval_loader)
                zt = torch.as_tensor(te_post, dtype=torch.float32,
                                     device=DEVICE)
                seen_anc = torch.cat([cur_means[seen], c], dim=1)
                disc_anc = torch.cat([cur_means[n_cls:], disc_ssl], dim=1)
                d_seen = torch.cdist(zt, seen_anc).min(1).values
                d_disc = torch.cdist(zt, disc_anc).min(1).values
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
                print(f"  [{name}] sparker (post, annealed)")
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
                del bb
                torch.cuda.empty_cache()

        for stat in STATS:
            print(f"\n===== EXP59 {stat.upper()} POWER (annealed sigma) "
                  f"=====")
            print(f"  {'arm':<16}{'kind':>6}"
                  + "".join(f"{f:>9}" for f in fractions))
            for name in args.arms:
                print(f"  {name:<16}{'pre':>6}"
                      + "".join(f"{p:>9.3f}" for p in pre_power[stat][name]))
                print(f"  {name:<16}{'post':>6}"
                      + "".join(f"{p:>9.3f}" for p in post_power[stat][name]))
            plt.figure(figsize=(8, 6.5))
            for name in args.arms:
                plt.plot(fractions, pre_power[stat][name], "--o",
                         color=COLORS[name], lw=1.4, ms=5, alpha=0.75,
                         label=f"{name} pre")
                plt.plot(fractions, post_power[stat][name], "-o",
                         color=COLORS[name], lw=2, ms=6,
                         label=f"{name} post")
            plt.xscale("log")
            plt.axhline(args.alpha, color="gray", lw=1, ls=":")
            plt.xlabel("injected anomaly fraction")
            plt.ylabel(f"power at alpha={args.alpha}")
            plt.title(f"exp59 NPLM residual/concat ({ds}): {stat}")
            plt.grid(alpha=0.25, which="both")
            plt.legend(loc="upper left", fontsize=8, ncol=2)
            plt.tight_layout()
            plt.savefig(plot_path(f"exp59_{stat}_power_{dtag}.png"), dpi=150)
            plt.close()
            print("  saved " + plot_path(f"exp59_{stat}_power_{dtag}.png"))

    print(f"\n===== EXP59 SUMMARY [{ds}] =====")
    for name in args.arms:
        r = results[name]
        print(f"  [{name:<14}] probe={r['probe']:.4f}/{probe_post[name]:.4f} "
              f"acc={r['acc']:.4f} eucl={r['eucl']:.4f} "
              f"mahaT={r['mahaT']:.4f}")
    os.makedirs(os.path.join("logs", "exp59"), exist_ok=True)
    np.savez(os.path.join("logs", "exp59", f"nplm_residual_concat_{dtag}.npz"),
             fractions=np.array(fractions), arms=np.array(args.arms),
             **{f"{k}_{n}": np.array(results[n][k]) for n in args.arms
                for k in ("probe", "probe_sd", "acc", "sup_auc", "eucl",
                          "mahaT", "mahaPC")},
             **{f"probe_post_{n}": np.array(probe_post[n])
                for n in args.arms},
             **{f"{s}_{n}_pre": np.array(pre_power[s][n]) for s in STATS
                for n in args.arms if n in pre_power[s]},
             **{f"{s}_{n}_post": np.array(post_power[s][n]) for s in STATS
                for n in args.arms if post_power[s][n]})
    print("Done.")


if __name__ == "__main__":
    main()
