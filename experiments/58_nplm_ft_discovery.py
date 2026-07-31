"""
Experiment 58: NPLM+SIGReg as the discovery fine-tune objective (CIFAR-10).

Exp 55 updated the space with the settled proto/repulse hybrid.  Here the
same discovery loop (pool past the 0.95 quantile, BIC k-means, pseudo-label)
fine-tunes with the NPLM loss itself: supervised positives over seen labels
+ discovered pseudo-labels (single-view balanced batches -- same-class pairs
are the positives), calibrated NPLM interaction, global SIGReg marginal.
Anchors are recomputed as class/cluster centroids after each round (the NPLM
loss has no anchor parameters).  A/B per arm: ft=proto (exp-55 loop, rerun)
vs ft=nplm.  Per the exp-57 lesson all SparKer scoring uses the ANNEALED
median-heuristic sigma (exp-55's fixed sigma=1 post numbers were artifacts),
so the proto arm is rerun here to get comparable SparKer values.

    python experiments/58_nplm_ft_discovery.py
    python experiments/58_nplm_ft_discovery.py --quick --arms nplm_sup_dist --fts nplm
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
from supersig.losses import HybridContrastiveLoss
from supersig.recipes import recipe
from supersig.discovery import (PseudoDataset, bic_select, merge_anchors,
                                run_discovery)
from supersig.train import collect_embeddings

exp28 = importlib.import_module("28_concat_residual")
exp29 = importlib.import_module("29_residual_finetune")
exp30 = importlib.import_module("30_power_curves")
exp31 = importlib.import_module("31_sparker_power")
exp32 = importlib.import_module("32_maha_mmd_power")
exp55 = importlib.import_module("55_nplm_discovery")

STATS = exp55.STATS
ARMS = ["nplm_sup_dist", "nplm_dist_sup_cw"]
FTS = ["proto", "nplm"]
COLORS = {("nplm_sup_dist", "proto"): "#8c2d9e",
          ("nplm_sup_dist", "nplm"): "#d62728",
          ("nplm_dist_sup_cw", "proto"): "#2a78d6",
          ("nplm_dist_sup_cw", "nplm"): "#1baf7a"}


def train_nplm_ft(backbone, loader, epochs, critic, tau, lam, n_slices,
                  lr=1e-3):
    """Single-view supervised-NPLM fine-tune on pseudo-labelled batches."""
    loss_fn = HybridContrastiveLoss(positives="supervised", critic=critic,
                                    estimator="nplm", marginal="sigreg",
                                    tau=tau, lam=lam, n_slices=n_slices)
    opt = torch.optim.Adam(backbone.parameters(), lr=lr)
    backbone.train()
    for ep in range(epochs):
        int_run, marg_run, n = 0.0, 0.0, 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            z = backbone(x)
            total, parts = loss_fn(z, y)
            total.backward()
            opt.step()
            int_run += parts["interaction"].item() * x.size(0)
            marg_run += parts["marginal"].item() * x.size(0)
            n += x.size(0)
        n = max(n, 1)
        print(f"  [nplm-ft/{critic}] epoch {ep+1}/{epochs}  "
              f"interaction={int_run/n:.4f}  marginal={marg_run/n:.4f}")


def run_discovery_nplm_ft(backbone, means, *, base_ds, train_eval_loader,
                          test_loader, seen, holdouts, critic, tau, lam,
                          n_slices, rounds=2, ft_epochs=5, tau_quantile=0.95,
                          kmax=None, merge_dist=3.0, seed=0):
    """discovery.run_discovery with the proto ft swapped for supervised-NPLM;
    anchors recomputed as centroids after each round."""
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
        tau_pool = torch.quantile(
            dmin[torch.as_tensor(is_seen_lab, device=DEVICE)], tau_quantile)
        pool = (dmin > tau_pool).cpu().numpy()
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
        lab_idx = np.where(is_seen_lab)[0]
        ft_idx = np.concatenate([lab_idx, p_idx])
        ft_lab = np.concatenate([tr_lab[lab_idx], p_lab])
        n_pb = len(seen) + disc.size(0) if n_classes <= 10 else 25
        sampler = BalancedBatchSampler(list(ft_lab), n_classes=n_pb,
                                       n_per_class=24)
        ft_loader = DataLoader(PseudoDataset(base_ds, ft_idx, ft_lab),
                               batch_sampler=sampler, num_workers=2)
        train_nplm_ft(backbone, ft_loader, ft_epochs, critic, tau, lam,
                      n_slices)
        # recompute anchors as centroids in the updated space
        tr_embs, tr_lab = collect_embeddings(backbone, train_eval_loader)
        z = torch.as_tensor(tr_embs, device=DEVICE)
        seen_cents = exp28.class_centroids(tr_embs[is_seen_lab],
                                           tr_lab[is_seen_lab], seen)
        for i, c in enumerate(seen):
            cur_means[c] = seen_cents[i]
        disc_ids = sorted(set(p_lab))
        for j, d_id in enumerate(disc_ids):
            mask = np.zeros(len(tr_lab), dtype=bool)
            mask[ft_idx[len(lab_idx):][p_lab == d_id]] = True
            if mask.any():
                cur_means[n_classes + j] = z[
                    torch.as_tensor(mask, device=DEVICE)].mean(0)
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
                            purity=float(purity), khat=khat,
                            n_anchors=int(disc.size(0)), margin=float(margin),
                            mean_pc=float(np.mean(list(per_class.values())))))
        h = history[-1]
        print(f"  round {r}: pool={h['pool']} purity={h['purity']:.3f} "
              f"k-hat={h['khat']} anchors={h['n_anchors']}  "
              f"margin={h['margin']:.4f}  mean-anchor={h['mean_pc']:.4f}")
    return cur_means, history


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
    ap.add_argument("--fractions", default="0.001,0.003,0.01,0.02,0.03,0.1")
    ap.add_argument("--n-d", type=int, default=5000)
    ap.add_argument("--kernels", type=int, default=16)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    ap.add_argument("--fts", nargs="+", default=FTS, choices=FTS)
    args = ap.parse_args()
    ds = args.dataset

    cfg = recipe(ds, emb_dim=args.dim)
    n_cls = cfg["n_classes"]
    holdouts = {args.holdout}
    seen = [c for c in range(n_cls) if c not in holdouts]
    con_ep = args.epochs or (2 if args.quick else 20)
    ft_ep = 1 if args.quick else cfg["ft_epochs"]
    fractions = [float(x) for x in args.fractions.split(",")]
    n_null_post = 20 if args.quick else 100
    n_sig_toys = 10 if args.quick else 50
    sparker_kw = dict(M=args.kernels, steps=args.steps)  # annealed sigma
    names = exp29.CIFAR_NAMES if ds == "cifar10" else None
    configs = [(a, f) for a in args.arms for f in args.fts]
    print(f"exp58 [{ds}] NPLM-ft discovery A/B, arms={args.arms}, "
          f"fts={args.fts}, annealed-sigma sparker")

    train_loader, test_loader = get_cifar_loaders(quick=args.quick, dataset=ds)
    train_eval_loader = DataLoader(train_loader.dataset, batch_size=256,
                                   shuffle=False, num_workers=2)
    cls, plain, _ = _cifar_spec(ds)
    base = cls(DATA_DIR, train=True, download=True, transform=plain)
    base_targets = np.array(base.targets)
    n_base = 8000 if args.quick else len(base)
    seen_idx = np.where(np.isin(base_targets[:n_base], seen))[0]
    sig_idx_all = np.where(np.isin(base_targets[:n_base], list(holdouts)))[0]

    def critic_of(arm):
        return "distance"    # both supervised NPLM arms use the distance critic

    def discover(bb, means, base_ds, tel_loader, ft):
        if ft == "proto":
            return run_discovery(
                bb, means, base_ds=base_ds, train_eval_loader=tel_loader,
                test_loader=test_loader, seen=seen, holdouts=holdouts,
                dataset_name=ds, rep_weight=cfg["rep_weight"],
                sigreg_weight=cfg["sigreg_weight"],
                n_slices=cfg["n_slices"], rounds=args.rounds,
                ft_epochs=ft_ep, names=names, seed=args.seed)
        return run_discovery_nplm_ft(
            bb, means, base_ds=base_ds, train_eval_loader=tel_loader,
            test_loader=test_loader, seen=seen, holdouts=holdouts,
            critic=critic_of(None), tau=args.tau, lam=args.lam,
            n_slices=cfg["n_slices"], rounds=args.rounds, ft_epochs=ft_ep,
            seed=args.seed)

    nets, means_of, hist, probe = {}, {}, {}, {}
    tr_lab = te_lab = None
    for arm in args.arms:
        print(f"\n===== training: {arm} =====")
        nets[arm] = exp55.train_arm(arm, ds, cfg, args, con_ep, holdouts)
        tr, tr_lab = collect_embeddings(nets[arm], train_eval_loader)
        te, te_lab = collect_embeddings(nets[arm], test_loader)
        m = np.isin(tr_lab, seen)
        cents = exp28.class_centroids(tr[m], tr_lab[m], seen)
        means_of[arm] = exp28.fill_means(cents, seen, cfg).detach()
        a_pre, _, _ = exp29.linear_probe_novelty(tr, tr_lab, te, te_lab,
                                                 holdouts)
        for ft in args.fts:
            print(f"\n----- natural discovery: {arm} ft={ft} -----")
            bb = copy.deepcopy(nets[arm])
            _, hist[(arm, ft)] = discover(bb, means_of[arm].clone(), base,
                                          train_eval_loader, ft)
            tr_post, _ = collect_embeddings(bb, train_eval_loader)
            te_post, _ = collect_embeddings(bb, test_loader)
            a_post, _, _ = exp29.linear_probe_novelty(tr_post, tr_lab,
                                                      te_post, te_lab,
                                                      holdouts)
            probe[(arm, ft)] = (a_pre, a_post)
            print(f"  probe pre={a_pre:.4f} post={a_post:.4f}")
            del bb
            torch.cuda.empty_cache()

    post_power = {s: {c: [] for c in configs} for s in STATS}
    for i_f, f in enumerate(fractions):
        n_inj = int(round(f * len(seen_idx) / (1.0 - f)))
        rng = np.random.default_rng(args.seed * 1000 + i_f)
        inj = rng.choice(sig_idx_all, size=min(n_inj, len(sig_idx_all)),
                         replace=False)
        sub = Subset(base, np.concatenate([seen_idx, inj]).tolist())
        tel_loader = DataLoader(sub, batch_size=256, shuffle=False,
                                num_workers=2)
        print(f"\n===== POST grid, f={f} ({len(inj)} injected) =====")
        for arm, ft in configs:
            bb = copy.deepcopy(nets[arm])
            cur_means, _ = discover(bb, means_of[arm].clone(), sub,
                                    tel_loader, ft)
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
            post_power["perevent"][(arm, ft)].append(pe)
            print(f"  [{arm}/{ft}] per-event post f={f}: power={pe:.3f}")
            R = torch.as_tensor(tr_post[np.isin(trl_post, seen)][:20000],
                                dtype=torch.float32, device=DEVICE)
            bg = torch.as_tensor(te_post[bg_mask], dtype=torch.float32,
                                 device=DEVICE)
            sg = torch.as_tensor(te_post[sig_mask], dtype=torch.float32,
                                 device=DEVICE)
            print(f"  [{arm}/{ft}] sparker (post, annealed)")
            p, _ = exp31.run_test_battery(bg, sg, R, [f], args.n_d,
                                          n_null_post, n_sig_toys, args.alpha,
                                          args.seed + i_f, sparker_kw,
                                          tag="post-spk")
            post_power["sparker"][(arm, ft)].append(p[0])
            maha_fn, mmd_fn, n_bg, n_sig = exp32.make_stats_fns(
                tr_post, trl_post, te_post, tel_post, seen, holdouts,
                args.seed + i_f)
            print(f"  [{arm}/{ft}] maha (post)")
            p, _ = exp32.battery(maha_fn, n_bg, n_sig, [f], args.n_d,
                                 n_null_post, n_sig_toys, args.alpha,
                                 args.seed + i_f, tag="post-maha")
            post_power["maha"][(arm, ft)].append(p[0])
            print(f"  [{arm}/{ft}] mmd (post)")
            p, _ = exp32.battery(mmd_fn, n_bg, n_sig, [f], args.n_d,
                                 n_null_post, n_sig_toys, args.alpha,
                                 args.seed + i_f, tag="post-mmd")
            post_power["mmd"][(arm, ft)].append(p[0])
            del bb
            torch.cuda.empty_cache()

    for stat in STATS:
        print(f"\n===== EXP58 {stat.upper()} POST POWER "
              f"(annealed sigma) =====")
        print(f"  {'config':<26}" + "".join(f"{f:>9}" for f in fractions))
        for c in configs:
            print(f"  {c[0]+'/'+c[1]:<26}"
                  + "".join(f"{p:>9.3f}" for p in post_power[stat][c]))
        plt.figure(figsize=(8, 6.5))
        for c in configs:
            plt.plot(fractions, post_power[stat][c], "-o", color=COLORS[c],
                     lw=2, ms=5, label=f"{c[0]} ft={c[1]}")
        plt.xscale("log")
        plt.axhline(args.alpha, color="gray", lw=1, ls=":")
        plt.xlabel("injected anomaly fraction")
        plt.ylabel(f"power at alpha={args.alpha}")
        plt.title(f"exp58 NPLM-ft vs proto-ft discovery ({ds}): {stat} post")
        plt.grid(alpha=0.25, which="both")
        plt.legend(loc="upper left", fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(plot_path(f"exp58_{stat}_power_{ds}.png"), dpi=150)
        plt.close()
        print("  saved " + plot_path(f"exp58_{stat}_power_{ds}.png"))

    print(f"\n===== EXP58 SUMMARY [{ds}] =====")
    for arm, ft in configs:
        print(f"  [{arm}/{ft:<6}] probe pre={probe[(arm, ft)][0]:.4f} "
              f"post={probe[(arm, ft)][1]:.4f}")
        for h in hist[(arm, ft)]:
            print(f"          round {h['round']}: purity={h['purity']:.3f} "
                  f"anchors={h['n_anchors']}  margin={h['margin']:.4f}  "
                  f"mean-anchor={h['mean_pc']:.4f}")
    os.makedirs(os.path.join("logs", "exp58"), exist_ok=True)
    np.savez(os.path.join("logs", "exp58", f"nplm_ft_discovery_{ds}.npz"),
             fractions=np.array(fractions),
             **{f"probe_{a}_{f}": np.array(probe[(a, f)])
                for a, f in configs},
             **{f"{s}_{a}_{f}_post": np.array(post_power[s][(a, f)])
                for s in STATS for a, f in configs})
    print("Done.")


if __name__ == "__main__":
    main()
