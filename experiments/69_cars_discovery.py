"""
Experiment 69: discovery phase on Stanford Cars (supervised arms, feature
space).

First transfer-suite discovery run: because the exp-51 "backbone" is just a
FeatureHead on frozen DINO features, the settled discovery loop
(discovery.run_discovery -- pool tail, BIC k-means, pseudo-label,
proto/repulse fine-tune) runs unchanged with TensorDatasets of cached
features as the base dataset (PseudoDataset and collect_embeddings are
input-agnostic).  Arms: supcon / supcon_sigreg / nplm_sup_dist heads
(32-D, exp-51 protocol, holdouts 186-195).  Natural discovery (probe
pre/post, purity) + injected post-power grid (annealed-sigma SparKer).
Cars has ~41 train images per class (~410 holdout events vs a ~390-event
0.95-quantile tail), so pool purity CAN be high if the space separates.

    python experiments/69_cars_discovery.py
    python experiments/69_cars_discovery.py --quick --arms supcon_sigreg --fractions 0.01,0.05
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import copy
import importlib
import math
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from supersig.config import DEVICE, plot_path
from supersig.losses import make_anchors
from supersig.discovery import run_discovery
from supersig.train import train_supcon, collect_embeddings

exp28 = importlib.import_module("28_concat_residual")
exp29 = importlib.import_module("29_residual_finetune")
exp30 = importlib.import_module("30_power_curves")
exp31 = importlib.import_module("31_sparker_power")
exp32 = importlib.import_module("32_maha_mmd_power")
exp34h = importlib.import_module("34h_hybrid_nplm_cifar")
exp37 = importlib.import_module("37_dtd_vit")
exp38 = importlib.import_module("38_dtd_arc")
exp44 = importlib.import_module("44_transfer_32d")
exp51 = importlib.import_module("51_nplm_aircraft_suite")

STATS = ["perevent", "sparker", "maha", "mmd"]
ARMS = {
    "supcon": (None, True),
    "supcon_sigreg": (dict(positives="supervised", critic="cosine",
                           estimator="softmax", marginal="sigreg",
                           tau=0.1), True),
    "nplm_sup_dist": (dict(positives="supervised", critic="distance",
                           estimator="nplm", marginal="sigreg",
                           tau=1.0), True),
}
COLORS = {"supcon": "#eda100", "supcon_sigreg": "#008300",
          "nplm_sup_dist": "#8c2d9e"}
REP_WEIGHT = 20.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cars")
    ap.add_argument("--base", default="dino")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--n-slices", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--aug-reps", type=int, default=8)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--fractions", default="0.003,0.01,0.02,0.05")
    ap.add_argument("--n-d", type=int, default=2000)
    ap.add_argument("--kernels", type=int, default=16)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--arms", nargs="+", default=list(ARMS),
                    choices=list(ARMS))
    args = ap.parse_args()
    ds = args.dataset

    N_CLS = exp44.N_CLASSES[ds]
    holdouts = set(range(N_CLS - 10, N_CLS))
    seen = [c for c in range(N_CLS) if c not in holdouts]
    con_ep = args.epochs or (5 if args.quick else 120)
    ft_ep = 1 if args.quick else 5
    fractions = [float(x) for x in args.fractions.split(",")]
    n_null_post = 20 if args.quick else 100
    n_sig_toys = 10 if args.quick else 50
    sparker_kw = dict(M=args.kernels, steps=args.steps)   # annealed
    cfg = dict(n_classes=N_CLS, pair_dist=5.0, sigreg_weight=1.0,
               n_slices=args.n_slices,
               rep_weight=REP_WEIGHT * 45.0 / (N_CLS * (N_CLS - 1) / 2))
    tag = f"{ds}_{args.base}"
    print(f"exp69 [{tag}] feature-space discovery, supervised arms "
          f"{args.arms}, dim={args.dim}, holdouts "
          f"{min(holdouts)}-{max(holdouts)}")

    plain, bank = exp44.build_features(ds, args.base, args)
    (Xtr, ytr), (Xte, yte) = plain["train"], plain["test"]
    tr_lab, te_lab = ytr.numpy(), yte.numpy()
    seen_bank = exp51.filter_bank(bank, ~np.isin(bank["labels"].numpy(),
                                                 list(holdouts)))
    base_feats = TensorDataset(Xtr.float(), ytr)
    train_eval_loader = DataLoader(base_feats, batch_size=512, shuffle=False)
    test_loader = DataLoader(TensorDataset(Xte.float(), yte),
                             batch_size=512, shuffle=False)
    seen_idx = np.where(np.isin(tr_lab, seen))[0]
    sig_idx_all = np.where(np.isin(tr_lab, list(holdouts)))[0]

    nets, means_of, hist, probe = {}, {}, {}, {}
    for i, name in enumerate(args.arms):
        spec, labeled = ARMS[name]
        print(f"\n===== training: {name} =====")
        torch.manual_seed(args.seed + 20 + i)
        np.random.seed(args.seed + 20 + i)
        head = exp37.FeatureHead(args.dim).to(DEVICE)
        loader = DataLoader(exp37.TwoViewFeatures(seen_bank),
                            batch_size=args.batch_size, shuffle=True,
                            drop_last=True)
        if spec is None:
            train_supcon(head, loader, con_ep)
        else:
            exp34h.train_hybrid(head, loader, con_ep, spec, labeled,
                                lam=args.lam, n_slices=args.n_slices)
        nets[name] = head
        tr = exp37.embed(head, Xtr).numpy()
        te = exp37.embed(head, Xte).numpy()
        m = np.isin(tr_lab, seen)
        cents = exp28.class_centroids(tr[m], tr_lab[m], seen)
        means_of[name] = exp28.fill_means(cents, seen, cfg).detach()

        print(f"----- natural discovery: {name} -----")
        bb = copy.deepcopy(head)
        _, hist[name] = run_discovery(
            bb, means_of[name].clone(), base_ds=base_feats,
            train_eval_loader=train_eval_loader, test_loader=test_loader,
            seen=seen, holdouts=holdouts, dataset_name=ds,
            rep_weight=cfg["rep_weight"], sigreg_weight=cfg["sigreg_weight"],
            n_slices=cfg["n_slices"], rounds=args.rounds, ft_epochs=ft_ep,
            names=None, seed=args.seed)
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
        sub_idx = np.concatenate([seen_idx, inj])
        sub = TensorDataset(Xtr[sub_idx].float(), ytr[sub_idx])
        tel_loader = DataLoader(sub, batch_size=512, shuffle=False)
        print(f"\n===== POST grid, f={f} ({len(inj)} injected) =====")
        for name in args.arms:
            bb = copy.deepcopy(nets[name])
            cur_means, _ = run_discovery(
                bb, means_of[name].clone(), base_ds=sub,
                train_eval_loader=tel_loader, test_loader=test_loader,
                seen=seen, holdouts=holdouts, dataset_name=ds,
                rep_weight=cfg["rep_weight"],
                sigreg_weight=cfg["sigreg_weight"],
                n_slices=cfg["n_slices"], rounds=args.rounds,
                ft_epochs=ft_ep, names=None, seed=args.seed)
            te_post, tel_post = collect_embeddings(bb, test_loader)
            tr_post, trl_post = collect_embeddings(bb, train_eval_loader)
            zt = torch.as_tensor(te_post, dtype=torch.float32, device=DEVICE)
            d_seen = torch.cdist(zt, cur_means[seen]).min(1).values
            d_disc = (torch.cdist(zt, cur_means[N_CLS:]).min(1).values
                      if cur_means.size(0) > N_CLS else
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
        print(f"\n===== EXP69 {stat.upper()} POST POWER =====")
        print(f"  {'arm':<15}" + "".join(f"{f:>9}" for f in fractions))
        for name in args.arms:
            print(f"  {name:<15}"
                  + "".join(f"{p:>9.3f}" for p in post_power[stat][name]))
        plt.figure(figsize=(8, 6.5))
        for name in args.arms:
            plt.plot(fractions, post_power[stat][name], "-o",
                     color=COLORS[name], lw=2, ms=5, label=f"{name} post")
        plt.xscale("log")
        plt.axhline(args.alpha, color="gray", lw=1, ls=":")
        plt.xlabel("injected anomaly fraction")
        plt.ylabel(f"power at alpha={args.alpha}")
        plt.title(f"exp69 cars discovery ({tag}): {stat}")
        plt.grid(alpha=0.25, which="both")
        plt.legend(loc="upper left", fontsize=8)
        plt.tight_layout()
        plt.savefig(plot_path(f"exp69_{stat}_power_{tag}.png"), dpi=150)
        plt.close()
        print("  saved " + plot_path(f"exp69_{stat}_power_{tag}.png"))

    print(f"\n===== EXP69 SUMMARY [{tag}] =====")
    for name in args.arms:
        print(f"  [{name:<14}] probe pre={probe[name][0]:.4f} "
              f"post={probe[name][1]:.4f}")
        for h in hist[name]:
            print(f"          round {h['round']}: purity={h['purity']:.3f} "
                  f"anchors={h['n_anchors']}  margin={h['margin']:.4f}  "
                  f"mean-anchor={h['mean_pc']:.4f}")
    os.makedirs(os.path.join("logs", "exp69"), exist_ok=True)
    np.savez(os.path.join("logs", "exp69", f"cars_discovery_{tag}.npz"),
             fractions=np.array(fractions), arms=np.array(args.arms),
             **{f"probe_{n}": np.array(probe[n]) for n in args.arms},
             **{f"{s}_{n}_post": np.array(post_power[s][n]) for s in STATS
                for n in args.arms})
    print("Done.")


if __name__ == "__main__":
    main()
