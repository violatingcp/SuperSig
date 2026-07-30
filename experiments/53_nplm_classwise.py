"""
Experiment 53: NPLM interaction + class-wise SIGReg Gaussian constraint
(CIFAR-10 / CIFAR-100).

Exp 50 ran the NPLM corners with the *global* SIGReg marginal.  This swaps in
`classwise_sigreg_loss` -- each seen class pulled toward N(anchor_c, I) with
fixed `make_anchors` means (pair_dist/sqrt(2) scale, the supervised-recipe
geometry) -- completing the "class-wise NPLM" corner of the
HybridContrastiveLoss docstring.  Three arms:

  nplm_bil_cw       instance positives / bilinear / nplm  + classwise SIGReg
                    (labels shape the marginal only, not the positives)
  nplm_bil_sup_cw   supervised positives / bilinear / nplm + classwise SIGReg
  nplm_dist_sup_cw  supervised positives / distance / nplm + classwise SIGReg
                    (the docstring's class-wise NPLM recipe)

The class-wise term needs per-class batch statistics (MIN_PER_CLASS=8), so
training uses cifar_two_view_balanced_loader (24/class/view).  Evaluation is
the exp-50 protocol verbatim (Part A + pre-discovery power batteries), with
the exp-50 nplm/supcon reference curves overlaid from its npz.

    python experiments/53_nplm_classwise.py --dataset cifar10
    python experiments/53_nplm_classwise.py --dataset cifar100 --quick
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import importlib
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from supersig.config import DEVICE, plot_path
from supersig.data import get_cifar_loaders, cifar_two_view_balanced_loader
from supersig.models import CIFARResNetBackbone
from supersig.metrics import gaussianity_summary
from supersig.recipes import recipe
from supersig.losses import (HybridContrastiveLoss, classwise_sigreg_loss,
                             make_anchors)
from supersig.train import collect_embeddings

exp28 = importlib.import_module("28_concat_residual")
exp29 = importlib.import_module("29_residual_finetune")
exp30 = importlib.import_module("30_power_curves")
exp31 = importlib.import_module("31_sparker_power")
exp32 = importlib.import_module("32_maha_mmd_power")

STATS = ["perevent", "sparker", "maha", "mmd"]
# arm -> (positives, critic)
ARMS = {
    "nplm_bil_cw": ("instance", "bilinear"),
    "nplm_bil_sup_cw": ("supervised", "bilinear"),
    "nplm_dist_sup_cw": ("supervised", "distance"),
}
COLORS = {"nplm_bil_cw": "#1baf7a", "nplm_bil_sup_cw": "#2a78d6",
          "nplm_dist_sup_cw": "#8c2d9e"}
REF_ARMS = ["nplm_bilinear", "nplm_sup_dist", "supcon"]
REF_STYLE = {"nplm_bilinear": ":", "nplm_sup_dist": "--", "supcon": "-."}


def train_nplm_classwise(backbone, loader, epochs, positives, critic, means,
                         tau, lam, n_slices, lr=1e-3):
    """NPLM interaction (marginal='none') + lam * classwise SIGReg on raw z."""
    loss_fn = HybridContrastiveLoss(positives=positives, critic=critic,
                                    estimator="nplm", marginal="none",
                                    tau=tau)
    opt = torch.optim.Adam(backbone.parameters(), lr=lr)
    backbone.train()
    for ep in range(epochs):
        int_run, marg_run, n = 0.0, 0.0, 0
        for v1, v2, y in loader:
            v1, v2, y = v1.to(DEVICE), v2.to(DEVICE), y.to(DEVICE)
            cls_lab = torch.cat([y, y])
            if positives == "instance":
                inst = torch.arange(v1.size(0), device=DEVICE)
                pos_lab = torch.cat([inst, inst])
            else:
                pos_lab = cls_lab
            opt.zero_grad()
            z = backbone(torch.cat([v1, v2]))
            inter, _ = loss_fn(z, pos_lab)
            marg = classwise_sigreg_loss(z, cls_lab, means, n_slices=n_slices)
            (inter + lam * marg).backward()
            opt.step()
            int_run += inter.item() * v1.size(0)
            marg_run += marg.item() * v1.size(0)
            n += v1.size(0)
        n = max(n, 1)
        print(f"  [nplm/{critic}/cw] epoch {ep+1}/{epochs}  "
              f"interaction={int_run/n:.4f}  classwise={marg_run/n:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--holdout", type=int, default=4)
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
    ap.add_argument("--sparker-sigma", type=float, default=1.0)
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
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
    fractions = [float(x) for x in args.fractions.split(",")]
    n_null_pre = 20 if args.quick else 200
    n_sig_toys = 10 if args.quick else 50
    sparker_kw = dict(M=args.kernels, steps=args.steps)
    if args.sparker_sigma > 0:
        sparker_kw.update(sigma0=args.sparker_sigma, sigma_ratio=1.0,
                          n_checkpoints=1)
    means = make_anchors(cfg["pair_dist"] / math.sqrt(2.0), emb_dim=args.dim,
                         n_classes=n_cls).detach()
    print(f"exp53 [{ds}] NPLM + classwise SIGReg, dim={args.dim}, "
          f"epochs={con_ep}, holdout={sorted(holdouts)}, lam={args.lam}, "
          f"tau={args.tau}, arms={args.arms}")

    train_loader, test_loader = get_cifar_loaders(quick=args.quick, dataset=ds)
    train_eval_loader = DataLoader(train_loader.dataset, batch_size=256,
                                   shuffle=False, num_workers=2)

    def probe_stat(tr, tr_lab, te, te_lab, n_rep=3):
        aucs = []
        for s in range(n_rep):
            torch.manual_seed(1000 + s)
            a, _, _ = exp29.linear_probe_novelty(tr, tr_lab, te, te_lab,
                                                 holdouts)
            aucs.append(a)
        return float(np.mean(aucs)), float(np.std(aucs))

    results, trains, tests, anchors_of = {}, {}, {}, {}
    tr_lab = te_lab = None
    for i, name in enumerate(args.arms):
        positives, critic = ARMS[name]
        print(f"\n----- {name} (positives={positives}, critic={critic}) -----")
        torch.manual_seed(args.seed + 20 + i); np.random.seed(args.seed + 20 + i)
        net = CIFARResNetBackbone(args.dim, arch=cfg["arch"],
                                  pretrain=ds).to(DEVICE)
        loader = cifar_two_view_balanced_loader(ds, holdout=holdouts,
                                                quick=args.quick)
        train_nplm_classwise(net, loader, con_ep, positives, critic, means,
                             tau=args.tau, lam=args.lam,
                             n_slices=cfg["n_slices"])

        tr, tr_lab = collect_embeddings(net, train_eval_loader)
        te, te_lab = collect_embeddings(net, test_loader)
        trains[name], tests[name] = tr, te
        m = np.isin(tr_lab, seen)
        cents = exp28.class_centroids(tr[m], tr_lab[m], seen)
        anchors_of[name] = torch.as_tensor(cents, dtype=torch.float32,
                                           device=DEVICE)
        r = exp29.evaluate_space(tr, tr_lab, te, te_lab, anchors_of[name],
                                 seen, holdouts)
        pm, psd = probe_stat(tr, tr_lab, te, te_lab)
        g = gaussianity_summary(te, te_lab, seen, seed=args.seed)
        d_anchor = float((anchors_of[name] - means[seen]).norm(dim=1)
                         .mean())
        print(f"  [{name:<16}] probe={pm:.4f}+-{psd:.4f} acc={r['acc']:.4f} "
              f"supAUC={r['sup_auc']:.4f} eucl={r['eucl']:.4f} "
              f"mahaT={r['maha_tied']:.4f} mahaPC={r['maha_pc']:.4f} "
              f"cent->anchor={d_anchor:.2f}")
        results[name] = dict(probe=pm, probe_sd=psd, acc=r["acc"],
                             sup_auc=r["sup_auc"], eucl=r["eucl"],
                             mahaT=r["maha_tied"], mahaPC=r["maha_pc"],
                             d_anchor=d_anchor, gauss=g)
        del net
        torch.cuda.empty_cache()

    print("\n===== performance / novelty table =====")
    print(f"  {'arm':<18}{'probe':>16}{'acc':>8}{'supAUC':>8}{'eucl':>8}"
          f"{'mahaT':>8}{'mahaPC':>8}")
    for name in args.arms:
        r = results[name]
        print(f"  {name:<18}{r['probe']:>9.4f}+-{r['probe_sd']:.4f}"
              f"{r['acc']:>8.4f}{r['sup_auc']:>8.4f}{r['eucl']:>8.4f}"
              f"{r['mahaT']:>8.4f}{r['mahaPC']:>8.4f}")

    print("\n===== gaussianity (seen classes, test set) =====")
    exp28.print_gauss_table({n: results[n]["gauss"] for n in args.arms})

    # ===== pre-discovery power batteries ====================================
    pre_power = {s: {} for s in STATS}
    if not args.skip_power:
        print("\n===== PRE power batteries (all statistics) =====")
        for name in args.arms:
            tr, te = trains[name], tests[name]
            bg_mask = np.isin(te_lab, seen)
            sig_mask = np.isin(te_lab, list(holdouts))
            d = torch.cdist(torch.as_tensor(te, dtype=torch.float32,
                                            device=DEVICE), anchors_of[name])
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

        ref_path = os.path.join("logs", "exp50", f"results_nplm_{ds}.npz")
        ref = np.load(ref_path) if os.path.exists(ref_path) else None
        for stat in STATS:
            print(f"\n===== EXP53 {stat.upper()} PRE POWER "
                  f"(alpha={args.alpha}) =====")
            print(f"  {'arm':<18}" + "".join(f"{f:>9}" for f in fractions))
            for name in args.arms:
                print(f"  {name:<18}"
                      + "".join(f"{p:>9.3f}" for p in pre_power[stat][name]))
            plt.figure(figsize=(8, 6.5))
            for name in args.arms:
                plt.plot(fractions, pre_power[stat][name], "-o",
                         color=COLORS[name], lw=2, ms=5, label=name)
            if ref is not None:
                for rname in REF_ARMS:
                    k = f"{stat}_{rname}_pre"
                    if (k in ref.files and len(ref["fractions"]) ==
                            len(fractions) and
                            np.allclose(ref["fractions"], fractions)):
                        plt.plot(fractions, ref[k], REF_STYLE[rname],
                                 color="black", lw=1.2, alpha=0.6,
                                 label=f"{rname} (exp50, global sigreg)")
            plt.xscale("log")
            plt.axhline(args.alpha, color="gray", lw=1, ls=":")
            plt.xlabel("injected anomaly fraction")
            plt.ylabel(f"power at alpha={args.alpha}")
            plt.title(f"exp53 NPLM+classwise ({ds} {args.dim}d): "
                      f"{stat} pre power")
            plt.grid(alpha=0.25, which="both")
            plt.legend(loc="upper left", fontsize=8, ncol=2)
            plt.tight_layout()
            plt.savefig(plot_path(f"exp53_{stat}_power_{ds}.png"), dpi=150)
            plt.close()
            print("  saved " + plot_path(f"exp53_{stat}_power_{ds}.png"))

    os.makedirs(os.path.join("logs", "exp53"), exist_ok=True)
    np.savez(os.path.join("logs", "exp53", f"results_classwise_{ds}.npz"),
             fractions=np.array(fractions), arms=np.array(args.arms),
             **{f"{k}_{n}": np.array(results[n][k]) for n in args.arms
                for k in ("probe", "probe_sd", "acc", "sup_auc", "eucl",
                          "mahaT", "mahaPC", "d_anchor")},
             **{f"{s}_{n}_pre": np.array(pre_power[s][n]) for s in STATS
                for n in args.arms if n in pre_power[s]})

    print("\n===== EXP53 SUMMARY =====")
    for name in args.arms:
        r = results[name]
        print(f"  [{name:<16}] probe={r['probe']:.4f}+-{r['probe_sd']:.4f}  "
              f"acc={r['acc']:.4f}  eucl={r['eucl']:.4f}  "
              f"mahaT={r['mahaT']:.4f}  cent->anchor={r['d_anchor']:.2f}")
    print("Done.")


if __name__ == "__main__":
    main()
