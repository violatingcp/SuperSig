"""
Experiment 56: confidence-masked discovery on the CIFAR-100 NPLM spaces.

Exp 55 found the discovery loop neutral-to-harmful on CIFAR-100: at
tau_quantile=0.95 the pooled tail is ~2500 events against only 500 holdout
images, so pool purity is ~0.003-0.013 and the pseudo-label fine-tune trains
on noise.  This applies the exp-36b JEPAMatch-style confidence mask to the
single-net discovery loop: after BIC k-means assignment, a pooled event is
kept for fine-tuning only if its proto-posterior probability (softmax over
-0.5 d^2 to [seen anchors; discovered anchors]) at the assigned discovered
anchor is >= conf_thresh.  `run_discovery_conf` below is discovery.
run_discovery with that one block added (annealing off, per the exp-35
verdict); pooling, BIC, anchors and scoring are unchanged, so exp-55 npz
curves are the exact no-mask baseline.

    python experiments/56_nplm_confmask.py
    python experiments/56_nplm_confmask.py --quick --arms nplm_bilinear
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
from sklearn.metrics import roc_auc_score

from supersig.config import DATA_DIR, DEVICE, plot_path
from supersig.data import get_cifar_loaders, BalancedBatchSampler, _cifar_spec
from supersig.recipes import recipe
from supersig.discovery import (PseudoDataset, bic_select, merge_anchors)
from supersig.train import train_sigreg_hybrid, collect_embeddings

exp28 = importlib.import_module("28_concat_residual")
exp29 = importlib.import_module("29_residual_finetune")
exp30 = importlib.import_module("30_power_curves")
exp31 = importlib.import_module("31_sparker_power")
exp32 = importlib.import_module("32_maha_mmd_power")
exp55 = importlib.import_module("55_nplm_discovery")

STATS = exp55.STATS
ARMS = exp55.ARMS
COLORS = exp55.COLORS


def run_discovery_conf(backbone, means, *, base_ds, train_eval_loader,
                       test_loader, seen, holdouts, dataset_name, rep_weight,
                       sigreg_weight, n_slices, conf_thresh, rounds=2,
                       ft_epochs=5, tau_quantile=0.95, kmax=None,
                       merge_dist=3.0, names=None, seed=0):
    """discovery.run_discovery + the exp-28/36b proto-posterior conf mask."""
    n_classes = means.size(0)
    cur_means = means.detach().clone()
    pooled = np.zeros(len(train_eval_loader.dataset), dtype=bool)
    history = []
    for r in range(1, rounds + 1):
        tr_embs, tr_lab = collect_embeddings(backbone, train_eval_loader)
        z = torch.as_tensor(tr_embs, device=DEVICE)
        anchor_mat = torch.cat([cur_means[seen], cur_means[n_classes:]]) \
            if cur_means.size(0) > n_classes else cur_means[seen]
        dmin = torch.cdist(z, anchor_mat).min(1).values
        is_seen_lab = np.isin(tr_lab, seen)
        tau = torch.quantile(dmin[torch.as_tensor(is_seen_lab, device=DEVICE)],
                             tau_quantile)
        pool = (dmin > tau).cpu().numpy()
        purity = (~is_seen_lab[pool]).mean() if pool.any() else float("nan")
        km = kmax or max(4, len(holdouts) + 2)
        khat, centers, _ = bic_select(z[torch.as_tensor(pool, device=DEVICE)],
                                      kmax=km, seed=seed + r)
        cur_means = torch.cat([cur_means, centers.detach()], dim=0)
        pooled |= pool
        disc = cur_means[n_classes:]
        memb = torch.cdist(z[torch.as_tensor(pooled, device=DEVICE)],
                           disc).argmin(1)
        disc = merge_anchors(disc, memb, merge_dist)
        cur_means = torch.cat([cur_means[:n_classes], disc], dim=0)
        p_idx = np.where(pooled)[0]
        p_lab = n_classes + torch.cdist(
            z[torch.as_tensor(pooled, device=DEVICE)],
            disc).argmin(1).cpu().numpy()
        # ---- exp-36b confidence mask (the only change vs run_discovery) ----
        keep_idx, keep_lab = p_idx, p_lab
        kept_pur = float("nan")
        if conf_thresh is not None and len(p_idx):
            all_anc = torch.cat([cur_means[seen], disc])
            logits = -0.5 * torch.cdist(
                z[torch.as_tensor(pooled, device=DEVICE)], all_anc).pow(2)
            conf = torch.softmax(logits, dim=1)
            assigned = len(seen) + torch.as_tensor(p_lab - n_classes,
                                                   device=DEVICE)
            ca = conf[torch.arange(len(p_lab), device=DEVICE), assigned]
            keep = (ca >= conf_thresh).cpu().numpy()
            kept_pur = (float((~is_seen_lab[p_idx[keep]]).mean())
                        if keep.any() else float("nan"))
            print(f"    conf-mask(>{conf_thresh}): kept {int(keep.sum())}"
                  f"/{len(p_lab)} pooled, kept-purity={kept_pur:.3f}")
            keep_idx, keep_lab = p_idx[keep], p_lab[keep]
        lab_idx = np.where(is_seen_lab)[0]
        ft_idx = np.concatenate([lab_idx, keep_idx])
        ft_lab = np.concatenate([tr_lab[lab_idx], keep_lab])
        n_pb = len(seen) + disc.size(0) if n_classes <= 10 else 25
        sampler = BalancedBatchSampler(list(ft_lab), n_classes=n_pb,
                                       n_per_class=24)
        ft_loader = DataLoader(PseudoDataset(base_ds, ft_idx, ft_lab),
                               batch_sampler=sampler, num_workers=2)
        train_sigreg_hybrid(backbone, ft_loader, ft_epochs, cur_means,
                            mode="repulse", disc="proto", alpha=1.0,
                            rep_weight=rep_weight,
                            sigreg_weight=sigreg_weight, n_slices=n_slices,
                            rep_exempt_from=n_classes)
        cur_means = cur_means.detach()

        te_embs, te_lab = collect_embeddings(backbone, test_loader)
        zt = torch.as_tensor(te_embs, device=DEVICE)
        d_seen = torch.cdist(zt, cur_means[seen]).min(1).values
        d_each = torch.cdist(zt, cur_means[n_classes:])
        is_unseen = np.isin(te_lab, list(holdouts)).astype(int)
        margin = roc_auc_score(is_unseen,
                               (d_seen - d_each.min(1).values).cpu().numpy())
        per_class = {}
        for c in sorted(holdouts):
            counts = [int(((te_lab == c) &
                           (d_each.argmin(1).cpu().numpy() == j)).sum())
                      for j in range(d_each.size(1))]
            j = int(np.argmax(counts))
            per_class[c] = roc_auc_score((te_lab == c).astype(int),
                                         (-d_each[:, j]).cpu().numpy())
        history.append(dict(round=r, pool=int(pool.sum()),
                            purity=float(purity), kept_purity=kept_pur,
                            khat=khat, n_anchors=int(disc.size(0)),
                            margin=float(margin),
                            mean_pc=float(np.mean(list(per_class.values())))))
        h = history[-1]
        print(f"  round {r}: pool={h['pool']} purity={h['purity']:.3f} "
              f"kept-purity={h['kept_purity']:.3f} k-hat={h['khat']} "
              f"anchors={h['n_anchors']}  margin={h['margin']:.4f}  "
              f"mean-anchor={h['mean_pc']:.4f}")
    return cur_means, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--holdout", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--conf-thresh", type=float, default=0.5)
    ap.add_argument("--tau-quantile", type=float, default=0.95)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--fractions", default=None)
    ap.add_argument("--n-d", type=int, default=5000)
    ap.add_argument("--kernels", type=int, default=16)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--sparker-sigma", type=float, default=1.0)
    ap.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
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
    print(f"exp56 [{ds}] NPLM conf-mask discovery, conf={args.conf_thresh}, "
          f"tauq={args.tau_quantile}, holdout={sorted(holdouts)}, "
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

    disc_kw = dict(seen=seen, holdouts=holdouts, dataset_name=ds,
                   rep_weight=cfg["rep_weight"],
                   sigreg_weight=cfg["sigreg_weight"],
                   n_slices=cfg["n_slices"], conf_thresh=args.conf_thresh,
                   rounds=args.rounds, ft_epochs=ft_ep,
                   tau_quantile=args.tau_quantile, names=names,
                   seed=args.seed)

    nets, means_of, hist, probe = {}, {}, {}, {}
    tr_lab = te_lab = None
    for name in args.arms:
        print(f"\n===== training: {name} =====")
        nets[name] = exp55.train_arm(name, ds, cfg, args, con_ep, holdouts)
        tr, tr_lab = collect_embeddings(nets[name], train_eval_loader)
        te, te_lab = collect_embeddings(nets[name], test_loader)
        m = np.isin(tr_lab, seen)
        cents = exp28.class_centroids(tr[m], tr_lab[m], seen)
        means_of[name] = exp28.fill_means(cents, seen, cfg).detach()

        print(f"\n----- conf-mask discovery: {name} -----")
        bb = copy.deepcopy(nets[name])
        _, hist[name] = run_discovery_conf(
            bb, means_of[name].clone(), base_ds=base,
            train_eval_loader=train_eval_loader, test_loader=test_loader,
            **disc_kw)
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

    post_power = {s: {n: [] for n in args.arms} for s in STATS}
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
            bb = copy.deepcopy(nets[name])
            cur_means, _ = run_discovery_conf(
                bb, means_of[name].clone(), base_ds=sub,
                train_eval_loader=tel_loader, test_loader=test_loader,
                **disc_kw)
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

    ref_path = os.path.join("logs", "exp55", f"discovery_nplm_{ds}.npz")
    ref = np.load(ref_path) if os.path.exists(ref_path) else None
    for stat in STATS:
        print(f"\n===== EXP56 {stat.upper()} POST POWER "
              f"(conf={args.conf_thresh}) =====")
        print(f"  {'arm':<18}" + "".join(f"{f:>9}" for f in fractions))
        for name in args.arms:
            print(f"  {name:<18}"
                  + "".join(f"{p:>9.3f}" for p in post_power[stat][name]))
        plt.figure(figsize=(8, 6.5))
        for name in args.arms:
            plt.plot(fractions, post_power[stat][name], "-o",
                     color=COLORS[name], lw=2, ms=5,
                     label=f"{name} conf-mask")
            if (ref is not None and f"{stat}_{name}_post" in ref.files
                    and len(ref["fractions"]) == len(fractions)
                    and np.allclose(ref["fractions"], fractions)):
                plt.plot(fractions, ref[f"{stat}_{name}_post"], "--o",
                         color=COLORS[name], lw=1.2, ms=4, alpha=0.6,
                         label=f"{name} no-mask (exp55)")
        plt.xscale("log")
        plt.axhline(args.alpha, color="gray", lw=1, ls=":")
        plt.xlabel("injected anomaly fraction")
        plt.ylabel(f"power at alpha={args.alpha}")
        plt.title(f"exp56 conf-mask discovery ({ds} {args.dim}d): "
                  f"{stat} post power")
        plt.grid(alpha=0.25, which="both")
        plt.legend(loc="upper left", fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(plot_path(f"exp56_{stat}_power_{ds}.png"), dpi=150)
        plt.close()
        print("  saved " + plot_path(f"exp56_{stat}_power_{ds}.png"))

    print(f"\n===== EXP56 SUMMARY [{ds}] conf={args.conf_thresh} =====")
    for name in args.arms:
        base_post = (float(ref[f"probe_{name}"][1])
                     if ref is not None and f"probe_{name}" in ref.files
                     else float("nan"))
        print(f"  [{name:<16}] probe pre={probe[name][0]:.4f} "
              f"post={probe[name][1]:.4f} (no-mask post={base_post:.4f})")
        for h in hist[name]:
            print(f"          round {h['round']}: purity={h['purity']:.3f} "
                  f"kept-purity={h['kept_purity']:.3f} "
                  f"anchors={h['n_anchors']}  margin={h['margin']:.4f}  "
                  f"mean-anchor={h['mean_pc']:.4f}")
    os.makedirs(os.path.join("logs", "exp56"), exist_ok=True)
    np.savez(os.path.join("logs", "exp56", f"confmask_nplm_{ds}.npz"),
             fractions=np.array(fractions), arms=np.array(args.arms),
             conf_thresh=args.conf_thresh,
             **{f"probe_{n}": np.array(probe[n]) for n in args.arms},
             **{f"{s}_{n}_post": np.array(post_power[s][n]) for s in STATS
                for n in args.arms})
    print("Done.")


if __name__ == "__main__":
    main()
