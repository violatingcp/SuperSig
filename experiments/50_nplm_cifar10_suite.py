"""
Experiment 50: NPLM loss suite on CIFAR-10 (exp-36 study protocol, standalone
spaces).

Compares the calibrated NPLM interaction (supersig.losses.HybridContrastiveLoss,
exps 34f/34h) against its softmax counterparts and the plain augmentation-only
baselines, each trained as a standalone 32-D space on the settled CIFAR-10 base
model (CIFAR-pretrained ResNet-20 -> 128 -> emb):

  simclr         instance / cosine / softmax / --        (augmentations only)
  lejepa         MSE invariance + global SIGReg          (augmentations only)
  simclr_sigreg  instance / cosine / softmax / sigreg    ("ss" hybrid, exp 34e)
  supcon         supervised / cosine / softmax / --      (Khosla SupCon)
  supcon_sigreg  supervised / cosine / softmax / sigreg  (exp 34g)
  nplm_bilinear  instance / bilinear / nplm / sigreg
  nplm_distance  instance / distance / nplm / sigreg
  nplm_sup_dist  supervised / distance / nplm / sigreg

Part A: exp-29/36 metric suite (acc, supAUC, eucl/maha novelty, 3-seed holdout
probe, gaussianity table).  Part B: pre-discovery power batteries (per-event,
SparKer, Mahalanobis, MMD) at the exp-36 CIFAR-10 fractions, with the exp-33
32-D sup/supcon reference curves overlaid.  Post-discovery is out of scope
here: the discovery loop is the supervised-anchor pipeline, orthogonal to the
loss comparison.

    python experiments/50_nplm_cifar10_suite.py
    python experiments/50_nplm_cifar10_suite.py --quick --arms nplm_bilinear simclr
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import importlib
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from supersig.config import DEVICE, plot_path
from supersig.data import get_cifar_loaders, cifar_two_view_loader
from supersig.models import CIFARResNetBackbone
from supersig.metrics import gaussianity_summary
from supersig.recipes import recipe
from supersig.train import (train_supcon, train_simclr, train_sigreg_ssl,
                            collect_embeddings)

exp28 = importlib.import_module("28_concat_residual")
exp29 = importlib.import_module("29_residual_finetune")
exp30 = importlib.import_module("30_power_curves")
exp31 = importlib.import_module("31_sparker_power")
exp32 = importlib.import_module("32_maha_mmd_power")
exp34h = importlib.import_module("34h_hybrid_nplm_cifar")

STATS = ["perevent", "sparker", "maha", "mmd"]


def ref_npz(ds):
    return os.path.join("logs", "exp33", f"power_data_{ds}_16p16_k1.npz")

# arm -> ("hybrid", loss_cfg, labeled) driven by exp34h.train_hybrid, or
#        ("plain", train_fn, labeled) for the settled loops in supersig.train
ARMS = {
    "simclr":        ("plain", train_simclr, False),
    "lejepa":        ("plain", train_sigreg_ssl, False),
    "simclr_sigreg": ("hybrid", dict(positives="instance", critic="cosine",
                                     estimator="softmax", marginal="sigreg",
                                     tau=0.5), False),
    "supcon":        ("plain", train_supcon, True),
    "supcon_sigreg": ("hybrid", dict(positives="supervised", critic="cosine",
                                     estimator="softmax", marginal="sigreg",
                                     tau=0.1), True),
    "nplm_bilinear": ("hybrid", dict(positives="instance", critic="bilinear",
                                     estimator="nplm", marginal="sigreg",
                                     tau=1.0), False),
    "nplm_distance": ("hybrid", dict(positives="instance", critic="distance",
                                     estimator="nplm", marginal="sigreg",
                                     tau=1.0), False),
    "nplm_sup_dist": ("hybrid", dict(positives="supervised", critic="distance",
                                     estimator="nplm", marginal="sigreg",
                                     tau=1.0), True),
}
COLORS = {
    "simclr": "#9aa0a6", "lejepa": "#4a3aa7", "simclr_sigreg": "#2a78d6",
    "supcon": "#eda100", "supcon_sigreg": "#008300",
    "nplm_bilinear": "#1baf7a", "nplm_distance": "#d62728",
    "nplm_sup_dist": "#8c2d9e",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--holdout", type=int, default=4)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--fractions", default=None)
    ap.add_argument("--n-d", type=int, default=5000)
    ap.add_argument("--kernels", type=int, default=16)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--sparker-sigma", type=float, default=1.0)
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--scratch-base", default=None,
                    help="init every arm from checkpoints/scratch_{this}_"
                         "{ds}_{dim}d.pt (random-init trunk, exp 67) "
                         "instead of the hub-pretrained trunk")
    ap.add_argument("--skip-power", action="store_true")
    args = ap.parse_args()
    ds = args.dataset
    if args.fractions is None:
        args.fractions = ("0.001,0.003,0.01,0.02,0.03,0.1" if ds == "cifar10"
                          else "0.001,0.003,0.01,0.02,0.05")

    dtag = ds if args.dim == 32 else f"{ds}_{args.dim}d"
    if args.scratch_base:
        dtag += f"_scr-{args.scratch_base}"
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
    print(f"exp50 [{ds}] NPLM suite, dim={args.dim}, epochs={con_ep}, "
          f"holdout={sorted(holdouts)}, lam={args.lam}, arms={args.arms}")

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

    # ===== training + Part A ================================================
    results, trains, tests, anchors_of = {}, {}, {}, {}
    tr_lab = te_lab = None
    for i, name in enumerate(args.arms):
        kind, spec, labeled = ARMS[name]
        print(f"\n----- {name} ({'labels' if labeled else 'augmentations only'})"
              f" -----")
        torch.manual_seed(args.seed + 20 + i); np.random.seed(args.seed + 20 + i)
        net = CIFARResNetBackbone(
            args.dim, arch=cfg["arch"],
            pretrain=None if args.scratch_base else ds).to(DEVICE)
        if args.scratch_base:
            ck = os.path.join("checkpoints",
                              f"scratch_{args.scratch_base}_{ds}"
                              f"_{args.dim}d.pt")
            net.load_state_dict(torch.load(ck,
                                           map_location=DEVICE)["state_dict"])
            print(f"  init from {ck}")
        loader = cifar_two_view_loader(quick=args.quick, labeled=labeled,
                                       holdout=holdouts, dataset=ds)
        if kind == "hybrid":
            exp34h.train_hybrid(net, loader, con_ep, spec, labeled,
                                lam=args.lam, n_slices=cfg["n_slices"])
        else:
            spec(net, loader, con_ep)

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
        print(f"  [{name:<14}] probe={pm:.4f}+-{psd:.4f} acc={r['acc']:.4f} "
              f"supAUC={r['sup_auc']:.4f} eucl={r['eucl']:.4f} "
              f"mahaT={r['maha_tied']:.4f} mahaPC={r['maha_pc']:.4f}")
        results[name] = dict(probe=pm, probe_sd=psd, acc=r["acc"],
                             sup_auc=r["sup_auc"], eucl=r["eucl"],
                             mahaT=r["maha_tied"], mahaPC=r["maha_pc"],
                             gauss=g)
        del net
        torch.cuda.empty_cache()

    print("\n===== performance / novelty table =====")
    print(f"  {'arm':<16}{'probe':>16}{'acc':>8}{'supAUC':>8}{'eucl':>8}"
          f"{'mahaT':>8}{'mahaPC':>8}")
    for name in args.arms:
        r = results[name]
        print(f"  {name:<16}{r['probe']:>9.4f}+-{r['probe_sd']:.4f}"
              f"{r['acc']:>8.4f}{r['sup_auc']:>8.4f}{r['eucl']:>8.4f}"
              f"{r['mahaT']:>8.4f}{r['mahaPC']:>8.4f}")

    print("\n===== gaussianity (seen classes, test set) =====")
    exp28.print_gauss_table({n: results[n]["gauss"] for n in args.arms})

    xs = np.arange(len(args.arms))
    plt.figure(figsize=(9, 5.5))
    plt.bar(xs, [results[n]["probe"] for n in args.arms],
            yerr=[results[n]["probe_sd"] for n in args.arms],
            color=[COLORS[n] for n in args.arms], capsize=3)
    plt.xticks(xs, args.arms, rotation=15, ha="right")
    plt.ylabel("holdout probe ROC AUC (pre-discovery)")
    plt.title(f"exp50: NPLM suite, {ds} {args.dim}d holdout={args.holdout}")
    plt.grid(alpha=0.25, axis="y")
    plt.tight_layout()
    out = plot_path(f"exp50_probe_{dtag}.png")
    plt.savefig(out, dpi=150); plt.close()
    print("saved", out)

    # ===== Part B: pre-discovery power batteries ============================
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

        ref = (np.load(ref_npz(ds)) if os.path.exists(ref_npz(ds))
               else None)
        for stat in STATS:
            print(f"\n===== EXP50 {stat.upper()} PRE POWER "
                  f"(alpha={args.alpha}) =====")
            print(f"  {'arm':<16}" + "".join(f"{f:>9}" for f in fractions))
            for name in args.arms:
                print(f"  {name:<16}"
                      + "".join(f"{p:>9.3f}" for p in pre_power[stat][name]))
            plt.figure(figsize=(8, 6.5))
            for name in args.arms:
                plt.plot(fractions, pre_power[stat][name], "-o",
                         color=COLORS[name], lw=2, ms=5, label=name)
            if ref is not None:
                for rname, style in (("sup", ":"), ("supcon", "--")):
                    k = f"{stat}_{rname}_pre"
                    if (k in ref.files and len(ref["fractions"]) ==
                            len(fractions) and
                            np.allclose(ref["fractions"], fractions)):
                        plt.plot(fractions, ref[k], style, color="black",
                                 lw=1.2, alpha=0.6,
                                 label=f"{rname} 32d (exp33 ref)")
            plt.xscale("log")
            plt.axhline(args.alpha, color="gray", lw=1, ls=":")
            plt.xlabel("injected anomaly fraction")
            plt.ylabel(f"power at alpha={args.alpha}")
            plt.title(f"exp50 NPLM suite ({ds} {args.dim}d): {stat} pre power")
            plt.grid(alpha=0.25, which="both")
            plt.legend(loc="upper left", fontsize=8, ncol=2)
            plt.tight_layout()
            plt.savefig(plot_path(f"exp50_{stat}_power_{dtag}.png"), dpi=150)
            plt.close()
            print("  saved " + plot_path(f"exp50_{stat}_power_{dtag}.png"))

    os.makedirs(os.path.join("logs", "exp50"), exist_ok=True)
    np.savez(os.path.join("logs", "exp50", f"results_nplm_{dtag}.npz"),
             fractions=np.array(fractions), arms=np.array(args.arms),
             **{f"{k}_{n}": np.array(results[n][k]) for n in args.arms
                for k in ("probe", "probe_sd", "acc", "sup_auc", "eucl",
                          "mahaT", "mahaPC")},
             **{f"{s}_{n}_pre": np.array(pre_power[s][n]) for s in STATS
                for n in args.arms if n in pre_power[s]})

    print("\n===== EXP50 SUMMARY =====")
    for name in args.arms:
        r = results[name]
        print(f"  [{name:<14}] probe={r['probe']:.4f}+-{r['probe_sd']:.4f}  "
              f"acc={r['acc']:.4f}  eucl={r['eucl']:.4f}  "
              f"mahaT={r['mahaT']:.4f}")
    print("Done.")


if __name__ == "__main__":
    main()
