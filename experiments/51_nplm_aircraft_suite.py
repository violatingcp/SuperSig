"""
Experiment 51: NPLM loss suite on FGVC-Aircraft (exp-50 protocol on the
transfer-suite infrastructure).

Same eight arms as exp 50 (simclr / lejepa / simclr_sigreg / supcon /
supcon_sigreg / nplm_bilinear / nplm_distance / nplm_sup_dist), but each is a
FeatureHead (768 -> 256 -> emb) trained on frozen ViT-B/16 CLS features with
the cached 8-replica augmented feature bank supplying the two views (exps
44-48 convention).  Aircraft has no settled holdout protocol, so novelty uses
a 10-variant holdout (classes 90-99 by default, ~330 signal test images);
the exp-29 metric suite and the exp-30/31/32 pre-discovery power batteries
run on the head embeddings exactly as in exp 50.  The toy size defaults to
1000 (test pool is only 3333 events).

    python experiments/51_nplm_aircraft_suite.py
    python experiments/51_nplm_aircraft_suite.py --quick --arms nplm_bilinear simclr
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
from supersig.metrics import gaussianity_summary

exp28 = importlib.import_module("28_concat_residual")
exp29 = importlib.import_module("29_residual_finetune")
exp30 = importlib.import_module("30_power_curves")
exp31 = importlib.import_module("31_sparker_power")
exp32 = importlib.import_module("32_maha_mmd_power")
exp34h = importlib.import_module("34h_hybrid_nplm_cifar")
exp37 = importlib.import_module("37_dtd_vit")
exp38 = importlib.import_module("38_dtd_arc")
exp40 = importlib.import_module("40_dtd_bases")
exp44 = importlib.import_module("44_transfer_32d")
exp50 = importlib.import_module("50_nplm_cifar10_suite")

ARMS, COLORS, STATS = exp50.ARMS, exp50.COLORS, exp50.STATS
N_CLS = exp44.N_CLASSES["aircraft"]


def filter_bank(bank, keep_mask):
    return {"feats": bank["feats"][:, keep_mask],
            "labels": bank["labels"][keep_mask]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="dino",
                    choices=list(exp40.CACHE_TAG))
    ap.add_argument("--holdouts",
                    default=",".join(str(c) for c in range(90, 100)))
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
    ap.add_argument("--fractions", default="0.003,0.01,0.02,0.05,0.1")
    ap.add_argument("--n-d", type=int, default=1000)
    ap.add_argument("--kernels", type=int, default=16)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--sparker-sigma", type=float, default=1.0)
    ap.add_argument("--arms", nargs="+", default=list(ARMS),
                    choices=list(ARMS))
    ap.add_argument("--skip-power", action="store_true")
    args = ap.parse_args()

    holdouts = {int(x) for x in args.holdouts.split(",")}
    seen = [c for c in range(N_CLS) if c not in holdouts]
    con_ep = args.epochs or (5 if args.quick else 120)
    fractions = [float(x) for x in args.fractions.split(",")]
    n_null_pre = 20 if args.quick else 200
    n_sig_toys = 10 if args.quick else 50
    sparker_kw = dict(M=args.kernels, steps=args.steps)
    if args.sparker_sigma > 0:
        sparker_kw.update(sigma0=args.sparker_sigma, sigma_ratio=1.0,
                          n_checkpoints=1)
    tag = f"aircraft_{args.base}"
    print(f"exp51 [{tag}] NPLM suite, dim={args.dim}, epochs={con_ep}, "
          f"holdouts={sorted(holdouts)}, lam={args.lam}, arms={args.arms}")

    plain, bank = exp44.build_features("aircraft", args.base, args)
    (Xtr, ytr), (Xte, yte) = plain["train"], plain["test"]
    tr_lab, te_lab = ytr.numpy(), yte.numpy()
    seen_bank = filter_bank(bank, ~np.isin(bank["labels"].numpy(),
                                           list(holdouts)))
    print(f"  train pool {len(tr_lab)} (seen bank {seen_bank['feats'].size(1)})"
          f", test pool {len(te_lab)} "
          f"({int(np.isin(te_lab, list(holdouts)).sum())} holdout)")

    def make_loader(labeled):
        ds = (exp37.TwoViewFeatures(seen_bank) if labeled
              else exp38.TwoViewUnlabeled(seen_bank))
        return DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                          drop_last=True)

    def probe_stat(tr, te, n_rep=3):
        aucs = []
        for s in range(n_rep):
            torch.manual_seed(1000 + s)
            a, _, _ = exp29.linear_probe_novelty(tr, tr_lab, te, te_lab,
                                                 holdouts)
            aucs.append(a)
        return float(np.mean(aucs)), float(np.std(aucs))

    # ===== training + Part A ================================================
    results, trains, tests, anchors_of = {}, {}, {}, {}
    for i, name in enumerate(args.arms):
        kind, spec, labeled = ARMS[name]
        print(f"\n----- {name} ({'labels' if labeled else 'augmentations only'})"
              f" -----")
        torch.manual_seed(args.seed + 20 + i); np.random.seed(args.seed + 20 + i)
        head = exp37.FeatureHead(args.dim).to(DEVICE)
        loader = make_loader(labeled)
        if kind == "hybrid":
            exp34h.train_hybrid(head, loader, con_ep, spec, labeled,
                                lam=args.lam, n_slices=args.n_slices)
        else:
            spec(head, loader, con_ep)

        tr = exp37.embed(head, Xtr).numpy()
        te = exp37.embed(head, Xte).numpy()
        trains[name], tests[name] = tr, te
        m = np.isin(tr_lab, seen)
        cents = exp28.class_centroids(tr[m], tr_lab[m], seen)
        anchors_of[name] = torch.as_tensor(cents, dtype=torch.float32,
                                           device=DEVICE)
        r = exp29.evaluate_space(tr, tr_lab, te, te_lab, anchors_of[name],
                                 seen, holdouts)
        pm, psd = probe_stat(tr, te)
        g = gaussianity_summary(te, te_lab, seen, seed=args.seed)
        print(f"  [{name:<14}] probe={pm:.4f}+-{psd:.4f} acc={r['acc']:.4f} "
              f"supAUC={r['sup_auc']:.4f} eucl={r['eucl']:.4f} "
              f"mahaT={r['maha_tied']:.4f} mahaPC={r['maha_pc']:.4f}")
        results[name] = dict(probe=pm, probe_sd=psd, acc=r["acc"],
                             sup_auc=r["sup_auc"], eucl=r["eucl"],
                             mahaT=r["maha_tied"], mahaPC=r["maha_pc"],
                             gauss=g)
        del head
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
    plt.title(f"exp51: NPLM suite, {tag} {args.dim}d "
              f"holdouts={min(holdouts)}-{max(holdouts)}")
    plt.grid(alpha=0.25, axis="y")
    plt.tight_layout()
    out = plot_path(f"exp51_probe_{tag}.png")
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

        for stat in STATS:
            print(f"\n===== EXP51 {stat.upper()} PRE POWER "
                  f"(alpha={args.alpha}) =====")
            print(f"  {'arm':<16}" + "".join(f"{f:>9}" for f in fractions))
            for name in args.arms:
                print(f"  {name:<16}"
                      + "".join(f"{p:>9.3f}" for p in pre_power[stat][name]))
            plt.figure(figsize=(8, 6.5))
            for name in args.arms:
                plt.plot(fractions, pre_power[stat][name], "-o",
                         color=COLORS[name], lw=2, ms=5, label=name)
            plt.xscale("log")
            plt.axhline(args.alpha, color="gray", lw=1, ls=":")
            plt.xlabel("injected anomaly fraction")
            plt.ylabel(f"power at alpha={args.alpha}")
            plt.title(f"exp51 NPLM suite ({tag} {args.dim}d): {stat} pre power")
            plt.grid(alpha=0.25, which="both")
            plt.legend(loc="upper left", fontsize=8, ncol=2)
            plt.tight_layout()
            plt.savefig(plot_path(f"exp51_{stat}_power_{tag}.png"), dpi=150)
            plt.close()
            print("  saved " + plot_path(f"exp51_{stat}_power_{tag}.png"))

    os.makedirs(os.path.join("logs", "exp51"), exist_ok=True)
    np.savez(os.path.join("logs", "exp51", f"results_nplm_{tag}.npz"),
             fractions=np.array(fractions), arms=np.array(args.arms),
             holdouts=np.array(sorted(holdouts)),
             **{f"{k}_{n}": np.array(results[n][k]) for n in args.arms
                for k in ("probe", "probe_sd", "acc", "sup_auc", "eucl",
                          "mahaT", "mahaPC")},
             **{f"{s}_{n}_pre": np.array(pre_power[s][n]) for s in STATS
                for n in args.arms if n in pre_power[s]})

    print("\n===== EXP51 SUMMARY =====")
    for name in args.arms:
        r = results[name]
        print(f"  [{name:<14}] probe={r['probe']:.4f}+-{r['probe_sd']:.4f}  "
              f"acc={r['acc']:.4f}  eucl={r['eucl']:.4f}  "
              f"mahaT={r['mahaT']:.4f}")
    print("Done.")


if __name__ == "__main__":
    main()
