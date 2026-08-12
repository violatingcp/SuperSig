"""
Experiment 74: discovery phase on the exp-73 CIFAR residual spaces.

Reproduces the exp-73 constructions bit-for-bit (same seeds; parents via
exp55.train_arm, children via exp73.residual_ft) -- no checkpoints were
saved -- then runs the settled image-space discovery loop (exp-55
protocol) on the CONCAT space with a two-net backbone: forward =
cat([parent(x), child(x)]), both resnets fine-tune in the discovery
proto/repulse step.  Arms per dataset: supcon->res concat (the exp-73
record space) and supcon->res-nplm concat (the calibrated variant /
exp-59 post-discovery lineage).

Reports probe/eucl/mahaT pre -> post + pool purity, then the injected
POST power grid (per-event / SparKer annealed / Maha / MMD) at the
dataset's standard fractions (n_d=5000).

    python experiments/74_cifar_residual_discovery.py --dataset cifar10 --dim 32
    python experiments/74_cifar_residual_discovery.py --dataset cifar100 --dim 100 --quick
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import copy
import importlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from supersig.config import DATA_DIR, DEVICE, plot_path
from supersig.data import cifar_two_view_loader, get_cifar_loaders, _cifar_spec
from supersig.recipes import recipe
from supersig.discovery import run_discovery
from supersig.train import collect_embeddings

exp28 = importlib.import_module("28_concat_residual")
exp29 = importlib.import_module("29_residual_finetune")
exp30 = importlib.import_module("30_power_curves")
exp31 = importlib.import_module("31_sparker_power")
exp32 = importlib.import_module("32_maha_mmd_power")
exp55 = importlib.import_module("55_nplm_discovery")
exp73 = importlib.import_module("73_cifar_residual_ft")

STATS = ["perevent", "sparker", "maha", "mmd"]
ARMS = ["res", "res-nplm"]
COLORS = {"res": "#008300", "res-nplm": "#8c2d9e"}


class ConcatNets(nn.Module):
    def __init__(self, parent, child):
        super().__init__()
        self.p, self.c = parent, child

    def forward(self, x):
        return torch.cat([self.p(x), self.c(x)], dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10",
                    choices=["cifar10", "cifar100"])
    ap.add_argument("--holdout", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--fractions", default=None)
    ap.add_argument("--n-d", type=int, default=5000)
    ap.add_argument("--kernels", type=int, default=16)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    ap.add_argument("--skip-power", action="store_true")
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
    n_null = 20 if args.quick else 100
    n_sig = 10 if args.quick else 50
    sparker_kw = dict(M=args.kernels, steps=args.steps)   # annealed
    tag = f"{ds}_{args.dim}d"
    print(f"exp74 [{tag}] discovery on exp-73 residual concats, "
          f"arms={args.arms}, holdout={sorted(holdouts)}")

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

    # ---- reproduce the exp-73 parent + children -----------------------------
    print("\n===== reproducing exp-73 supcon parent =====")
    parent = exp55.train_arm("supcon", ds, cfg, args, con_ep, holdouts)
    ptr, tr_lab = collect_embeddings(parent, train_eval_loader)
    m = np.isin(tr_lab, seen)
    cents_full = torch.zeros(n_cls, args.dim, device=DEVICE)
    cents_full[torch.as_tensor(seen, device=DEVICE)] = \
        exp28.class_centroids(ptr[m], tr_lab[m],
                              seen).detach().float().to(DEVICE)

    children = {}
    for obj in args.arms:
        print(f"\n===== reproducing exp-73 supcon->{obj} child =====")
        torch.manual_seed(args.seed + 7); np.random.seed(args.seed + 7)
        child = copy.deepcopy(parent)
        loader = cifar_two_view_loader(quick=args.quick, labeled=True,
                                       holdout=holdouts, dataset=ds)
        step = (exp73.make_res_step(cents_full, 5.0) if obj == "res"
                else exp73.make_res_nplm_step(cents_full, args.lam,
                                              cfg["n_slices"]))
        exp73.residual_ft(child, loader, con_ep, step, f"supcon-{obj}")
        children[obj] = child

    def space_scores(tr, te, tr_lab, te_lab):
        m2 = np.isin(tr_lab, seen)
        cents = exp28.class_centroids(tr[m2], tr_lab[m2], seen)
        anch = cents.detach().float().to(DEVICE)
        r = exp29.evaluate_space(tr, tr_lab, te, te_lab, anch, seen,
                                 holdouts)
        aucs = []
        for s in range(3):
            torch.manual_seed(1000 + s)
            a, _, _ = exp29.linear_probe_novelty(tr, tr_lab, te, te_lab,
                                                 holdouts)
            aucs.append(a)
        return (float(np.mean(aucs)), float(np.std(aucs)), r["eucl"],
                r["maha_tied"])

    # cfg for the 2*dim concat space
    cfg2 = dict(cfg)
    all_out = {}
    for obj in args.arms:
        key = f"supcon->{obj} concat"
        print(f"\n----- natural discovery: {key} -----")
        bb0 = ConcatNets(copy.deepcopy(parent), copy.deepcopy(children[obj]))
        tr0, tr_lab = collect_embeddings(bb0, train_eval_loader)
        te0, te_lab = collect_embeddings(bb0, test_loader)
        pr0, sd0, eu0, ma0 = space_scores(tr0, te0, tr_lab, te_lab)
        m2 = np.isin(tr_lab, seen)
        means0 = exp28.fill_means(
            exp28.class_centroids(tr0[m2], tr_lab[m2], seen), seen,
            cfg2).detach()
        bb = copy.deepcopy(bb0)
        _, hist = run_discovery(
            bb, means0.clone(), base_ds=base,
            train_eval_loader=train_eval_loader, test_loader=test_loader,
            seen=seen, holdouts=holdouts, dataset_name=ds,
            rep_weight=cfg["rep_weight"], sigreg_weight=cfg["sigreg_weight"],
            n_slices=cfg["n_slices"], rounds=args.rounds, ft_epochs=ft_ep,
            names=None, seed=args.seed)
        trp, _ = collect_embeddings(bb, train_eval_loader)
        tep, _ = collect_embeddings(bb, test_loader)
        pr1, sd1, eu1, ma1 = space_scores(trp, tep, tr_lab, te_lab)
        print(f"  [{key}] probe {pr0:.4f}+-{sd0:.4f} -> {pr1:.4f}+-{sd1:.4f}"
              f"  eucl {eu0:.4f} -> {eu1:.4f}  mahaT {ma0:.4f} -> {ma1:.4f}")
        for h in hist:
            print(f"          round {h['round']}: purity={h['purity']:.3f} "
                  f"anchors={h['n_anchors']}  margin={h['margin']:.4f}")
        out = dict(probe_pre=pr0, probe_post=pr1, eucl_pre=eu0,
                   eucl_post=eu1, maha_pre=ma0, maha_post=ma1,
                   purity=[h["purity"] for h in hist])
        del bb
        torch.cuda.empty_cache()

        if not args.skip_power:
            post_power = {s: [] for s in STATS}
            for i_f, f in enumerate(fractions):
                n_inj = int(round(f * len(seen_idx) / (1.0 - f)))
                rng = np.random.default_rng(args.seed * 1000 + i_f)
                inj = rng.choice(sig_idx_all,
                                 size=min(n_inj, len(sig_idx_all)),
                                 replace=False)
                sub_idx = np.concatenate([seen_idx, inj])
                sub = torch.utils.data.Subset(base, sub_idx.tolist())
                sub_loader = DataLoader(sub, batch_size=256, shuffle=False,
                                        num_workers=2)
                bb = copy.deepcopy(bb0)
                cur_means, _ = run_discovery(
                    bb, means0.clone(), base_ds=sub,
                    train_eval_loader=sub_loader, test_loader=test_loader,
                    seen=seen, holdouts=holdouts, dataset_name=ds,
                    rep_weight=cfg["rep_weight"],
                    sigreg_weight=cfg["sigreg_weight"],
                    n_slices=cfg["n_slices"], rounds=args.rounds,
                    ft_epochs=ft_ep, names=None, seed=args.seed)
                tep2, tel2 = collect_embeddings(bb, test_loader)
                trp2, trl2 = collect_embeddings(bb, train_eval_loader)
                zt = torch.as_tensor(tep2, dtype=torch.float32,
                                     device=DEVICE)
                d_seen = torch.cdist(zt, cur_means[seen]).min(1).values
                d_disc = (torch.cdist(zt, cur_means[n_cls:]).min(1).values
                          if cur_means.size(0) > n_cls else
                          torch.full_like(d_seen, float("inf")))
                bg_m = np.isin(tel2, seen)
                sg_m = np.isin(tel2, list(holdouts))
                s_ = (d_seen - d_disc).cpu().numpy()
                post_power["perevent"].append(
                    exp30.power_at_alpha(s_[bg_m], s_[sg_m], args.alpha))
                R = torch.as_tensor(trp2[np.isin(trl2, seen)][:20000],
                                    dtype=torch.float32, device=DEVICE)
                bg = torch.as_tensor(tep2[bg_m], dtype=torch.float32,
                                     device=DEVICE)
                sg = torch.as_tensor(tep2[sg_m], dtype=torch.float32,
                                     device=DEVICE)
                p, _ = exp31.run_test_battery(bg, sg, R, [f], args.n_d,
                                              n_null, n_sig, args.alpha,
                                              args.seed + i_f, sparker_kw,
                                              tag="post-spk")
                post_power["sparker"].append(p[0])
                maha_fn, mmd_fn, n_bg, n_sg = exp32.make_stats_fns(
                    trp2, trl2, tep2, tel2, seen, holdouts, args.seed + i_f)
                p, _ = exp32.battery(maha_fn, n_bg, n_sg, [f], args.n_d,
                                     n_null, n_sig, args.alpha,
                                     args.seed + i_f, tag="post-maha")
                post_power["maha"].append(p[0])
                p, _ = exp32.battery(mmd_fn, n_bg, n_sg, [f], args.n_d,
                                     n_null, n_sig, args.alpha,
                                     args.seed + i_f, tag="post-mmd")
                post_power["mmd"].append(p[0])
                print(f"  [{key}] post f={f}: " + "  ".join(
                    f"{s}={post_power[s][-1]:.3f}" for s in STATS))
                del bb
                torch.cuda.empty_cache()
            out["post_power"] = post_power
        all_out[obj] = out
        del bb0
        torch.cuda.empty_cache()

    print(f"\n===== EXP74 SUMMARY [{tag}] =====")
    for obj, o in all_out.items():
        print(f"  [supcon->{obj} concat] probe {o['probe_pre']:.4f} -> "
              f"{o['probe_post']:.4f}  eucl {o['eucl_pre']:.4f} -> "
              f"{o['eucl_post']:.4f}  mahaT {o['maha_pre']:.4f} -> "
              f"{o['maha_post']:.4f}  purity r1 {o['purity'][0]:.3f}")
        if "post_power" in o:
            for s in STATS:
                print(f"          {s} post: " + " ".join(
                    f"{p:.3f}" for p in o["post_power"][s]))

    xs = np.arange(len(args.arms))
    plt.figure(figsize=(6, 5))
    w = 0.38
    plt.bar(xs - w / 2, [all_out[a]["probe_pre"] for a in args.arms], w,
            label="pre", color="#eda100")
    plt.bar(xs + w / 2, [all_out[a]["probe_post"] for a in args.arms], w,
            label="post-discovery", color="#008300")
    plt.xticks(xs, [f"supcon->{a}" for a in args.arms])
    plt.ylabel("holdout probe ROC AUC")
    plt.title(f"exp74: discovery on exp-73 concats ({tag})")
    plt.legend()
    plt.grid(alpha=0.25, axis="y")
    plt.tight_layout()
    out = plot_path(f"exp74_probe_{tag}.png")
    plt.savefig(out, dpi=150); plt.close()
    print("saved", out)

    os.makedirs(os.path.join("logs", "exp74"), exist_ok=True)
    np.savez(os.path.join("logs", "exp74", f"results_{tag}.npz"),
             arms=np.array(args.arms), fractions=np.array(fractions),
             **{f"{a}_{f}": np.array(all_out[a][f]) for a in args.arms
                for f in ("probe_pre", "probe_post", "eucl_pre", "eucl_post",
                          "maha_pre", "maha_post", "purity")},
             **{f"{a}_post_{s}": np.array(all_out[a]["post_power"][s])
                for a in args.arms for s in STATS
                if "post_power" in all_out[a]})
    print("Done.")


if __name__ == "__main__":
    main()
