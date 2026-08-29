"""
Experiment 68: discovery phase on the from-scratch supervised bases.

Loads the exp-67 supervised scratch checkpoints (supcon / supsig / nplmcw,
CIFAR-100, 100-D, holdout 4 never seen) and runs the settled discovery
loop on each base space directly (exp-55 protocol): natural discovery
(probe pre/post, purity) + the injected post-power grid (per-event,
SparKer annealed-sigma, Maha, MMD).  supsig uses its learned means from
the checkpoint; the others use centroid-filled means.

    python experiments/68_scratch_discovery.py
    python experiments/68_scratch_discovery.py --quick --bases nplmcw --fractions 0.01,0.05
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
from supersig.data import get_cifar_loaders, _cifar_spec
from supersig.models import CIFARResNetBackbone
from supersig.recipes import recipe
from supersig.discovery import run_discovery
from supersig.train import collect_embeddings

exp28 = importlib.import_module("28_concat_residual")
exp29 = importlib.import_module("29_residual_finetune")
exp30 = importlib.import_module("30_power_curves")
exp31 = importlib.import_module("31_sparker_power")
exp32 = importlib.import_module("32_maha_mmd_power")

STATS = ["perevent", "sparker", "maha", "mmd"]
COLORS = {"supcon": "#eda100", "supsig": "#2a78d6", "nplmcw": "#d62728"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--holdout", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dim", type=int, default=100)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--fractions", default="0.001,0.003,0.01,0.02,0.05")
    ap.add_argument("--n-d", type=int, default=5000)
    ap.add_argument("--kernels", type=int, default=16)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--bases",
                    default="simclr,visreg,nplm,supcon,supsig,nplmcw,ssig,nplmsd")
    args = ap.parse_args()
    ds = args.dataset
    bases = args.bases.split(",")

    cfg = recipe(ds, emb_dim=args.dim)
    n_cls = cfg["n_classes"]
    holdouts = {args.holdout}
    seen = [c for c in range(n_cls) if c not in holdouts]
    ft_ep = 1 if args.quick else cfg["ft_epochs"]
    fractions = [float(x) for x in args.fractions.split(",")]
    n_null_post = 20 if args.quick else 100
    n_sig_toys = 10 if args.quick else 50
    sparker_kw = dict(M=args.kernels, steps=args.steps)   # annealed sigma
    print(f"exp68 [{ds}] discovery on scratch bases {bases}, "
          f"dim={args.dim}, holdout={sorted(holdouts)}")

    train_loader, test_loader = get_cifar_loaders(quick=args.quick, dataset=ds)
    train_eval_loader = DataLoader(train_loader.dataset, batch_size=256,
                                   shuffle=False, num_workers=2)
    cls, plain, _ = _cifar_spec(ds)
    base_ds = cls(DATA_DIR, train=True, download=True, transform=plain)
    base_targets = np.array(base_ds.targets)
    n_base = 8000 if args.quick else len(base_ds)
    seen_idx = np.where(np.isin(base_targets[:n_base], seen))[0]
    sig_idx_all = np.where(np.isin(base_targets[:n_base], list(holdouts)))[0]

    def load_base(name):
        net = CIFARResNetBackbone(args.dim, arch=cfg["arch"],
                                  pretrain=None).to(DEVICE)
        htag = "" if args.holdout == 4 else f"_h{args.holdout}"
        ck = torch.load(os.path.join("checkpoints",
                                     f"scratch_{name}_{ds}_{args.dim}d{htag}.pt"),
                        map_location=DEVICE)
        net.load_state_dict(ck["state_dict"])
        return net, ck

    nets, means_of, hist, probe, geo_post = {}, {}, {}, {}, {}
    tr_lab = te_lab = None
    for name in bases:
        print(f"\n===== base: {name} =====")
        nets[name], ck = load_base(name)
        tr, tr_lab = collect_embeddings(nets[name], train_eval_loader)
        te, te_lab = collect_embeddings(nets[name], test_loader)
        if name == "supsig" and "means" in ck:
            means_of[name] = ck["means"].to(DEVICE).detach()
            print("  using learned means from checkpoint")
        else:
            m = np.isin(tr_lab, seen)
            cents = exp28.class_centroids(tr[m], tr_lab[m], seen)
            means_of[name] = exp28.fill_means(cents, seen, cfg).detach()

        print(f"----- natural discovery: {name} -----")
        bb = copy.deepcopy(nets[name])
        _, hist[name] = run_discovery(
            bb, means_of[name].clone(), base_ds=base_ds,
            train_eval_loader=train_eval_loader, test_loader=test_loader,
            seen=seen, holdouts=holdouts, dataset_name=ds,
            rep_weight=cfg["rep_weight"], sigreg_weight=cfg["sigreg_weight"],
            n_slices=cfg["n_slices"], rounds=args.rounds, ft_epochs=ft_ep,
            names=None, seed=args.seed)
        tr_post, _ = collect_embeddings(bb, train_eval_loader)
        te_post, _ = collect_embeddings(bb, test_loader)
        m_post = np.isin(tr_lab, seen)
        anch_post = torch.as_tensor(
            exp28.class_centroids(tr_post[m_post], tr_lab[m_post], seen),
            dtype=torch.float32, device=DEVICE)
        geo_post[name] = exp29.evaluate_space(tr_post, tr_lab, te_post,
                                              te_lab, anch_post, seen, holdouts)
        a_pre, _, _ = exp29.linear_probe_novelty(tr, tr_lab, te, te_lab,
                                                 holdouts)
        a_post, _, _ = exp29.linear_probe_novelty(tr_post, tr_lab, te_post,
                                                  te_lab, holdouts)
        probe[name] = (a_pre, a_post)
        print(f"  probe pre={a_pre:.4f} post={a_post:.4f}")
        del bb
        torch.cuda.empty_cache()

    post_power = {s: {n: [] for n in bases} for s in STATS}
    # per-injected-fraction post metrics (the small-new-sample discovery regime)
    post_f = {n: {k: [] for k in ("probe", "eucl", "mahaT", "mahaPC", "purity1")}
              for n in bases}
    for i_f, f in enumerate(fractions):
        n_inj = int(round(f * len(seen_idx) / (1.0 - f)))
        rng = np.random.default_rng(args.seed * 1000 + i_f)
        inj = rng.choice(sig_idx_all, size=min(n_inj, len(sig_idx_all)),
                         replace=False)
        sub = Subset(base_ds, np.concatenate([seen_idx, inj]).tolist())
        tel_loader = DataLoader(sub, batch_size=256, shuffle=False,
                                num_workers=2)
        print(f"\n===== POST grid, f={f} ({len(inj)} injected) =====")
        for name in bases:
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
            mf = np.isin(trl_post, seen)
            anch_f = torch.as_tensor(
                exp28.class_centroids(tr_post[mf], trl_post[mf], seen),
                dtype=torch.float32, device=DEVICE)
            rpf = exp29.evaluate_space(tr_post, trl_post, te_post, tel_post,
                                       anch_f, seen, holdouts)
            torch.manual_seed(1000)
            pf, _, _ = exp29.linear_probe_novelty(tr_post, trl_post, te_post,
                                                  tel_post, holdouts)
            post_f[name]["probe"].append(float(pf))
            post_f[name]["eucl"].append(rpf["eucl"])
            post_f[name]["mahaT"].append(rpf["maha_tied"])
            post_f[name]["mahaPC"].append(rpf["maha_pc"])
            print(f"  [{name}] post f={f}: probe={pf:.4f} eucl={rpf['eucl']:.4f} "
                  f"mahaT={rpf['maha_tied']:.4f}")
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
        print(f"\n===== EXP68 {stat.upper()} POST POWER =====")
        print(f"  {'base':<10}" + "".join(f"{f:>9}" for f in fractions))
        for name in bases:
            print(f"  {name:<10}"
                  + "".join(f"{p:>9.3f}" for p in post_power[stat][name]))
        plt.figure(figsize=(8, 6.5))
        for name in bases:
            plt.plot(fractions, post_power[stat][name], "-o",
                     color=COLORS.get(name, "#666"), lw=2, ms=5,
                     label=f"{name} post")
        plt.xscale("log")
        plt.axhline(args.alpha, color="gray", lw=1, ls=":")
        plt.xlabel("injected anomaly fraction")
        plt.ylabel(f"power at alpha={args.alpha}")
        plt.title(f"exp68 discovery on scratch bases ({ds}): {stat}")
        plt.grid(alpha=0.25, which="both")
        plt.legend(loc="upper left", fontsize=8)
        plt.tight_layout()
        plt.savefig(plot_path(f"exp68_{stat}_power_{ds}.png"), dpi=150)
        plt.close()
        print("  saved " + plot_path(f"exp68_{stat}_power_{ds}.png"))

    print(f"\n===== EXP68 SUMMARY [{ds}] =====")
    for name in bases:
        print(f"  [{name:<8}] probe pre={probe[name][0]:.4f} "
              f"post={probe[name][1]:.4f}")
        for h in hist[name]:
            print(f"          round {h['round']}: purity={h['purity']:.3f} "
                  f"anchors={h['n_anchors']}  margin={h['margin']:.4f}  "
                  f"mean-anchor={h['mean_pc']:.4f}")
    os.makedirs(os.path.join("logs", "exp68"), exist_ok=True)
    htag = "" if args.holdout == 4 else f"_h{args.holdout}"
    np.savez(os.path.join("logs", "exp68", f"scratch_discovery_{ds}{htag}.npz"),
             fractions=np.array(fractions), bases=np.array(bases),
             **{f"probe_{n}": np.array(probe[n]) for n in bases},
             **{f"purity_{n}": np.array([h["purity"] for h in hist[n]]) for n in bases},
             **{f"post_{k}_{n}": np.array(geo_post[n][k]) for n in bases
                for k in ("acc", "eucl", "maha_tied", "maha_pc", "lid")},
             **{f"{s}_{n}_post": np.array(post_power[s][n]) for s in STATS
                for n in bases},
             **{f"postf_{k}_{n}": np.array(post_f[n][k]) for n in bases
                for k in ("probe", "eucl", "mahaT", "mahaPC") if post_f[n][k]})
    print("Done.")


if __name__ == "__main__":
    main()
