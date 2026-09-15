"""
Experiment 60: NPLM residual + concat spaces on the transfer suite
(aircraft / cars, frozen ViT-B/16 features, 16+16-D FeatureHead halves).

The exp-59 constructions ported to the exp-51 infrastructure.  The
supervised half is supcon_sigreg (the aircraft champion) or supcon; the
feature half is label-free NPLM-bilinear; the residual half is trained with
NPLM+sigreg on r = z - cent[y] (centroids of the supervised half), warm-
started from it (exp-36/59 pattern).  Arms (16+16 = 32-D):

  supsig->res-nplm  [supcon_sigreg16 ; NPLM-residual16]
  supsig+nplm       [supcon_sigreg16 ; nplm_bil16]
  supcon+nplm       [supcon16 ; nplm_bil16]
  nplmsup+nplm      [nplm_sup16 ; nplm_bil16]

Part A + pre-discovery power batteries only (the discovery loop is
CIFAR-image-coupled; no transfer-suite discovery protocol exists).

    python experiments/60_nplm_residual_transfer.py
    python experiments/60_nplm_residual_transfer.py --quick --arms supcon+nplm
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import copy
import importlib
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from supersig.config import DEVICE, plot_path
from supersig.losses import HybridContrastiveLoss
from supersig.metrics import gaussianity_summary
from supersig.train import train_supcon, collect_embeddings

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
exp51 = importlib.import_module("51_nplm_aircraft_suite")

STATS = ["perevent", "sparker", "maha", "mmd"]
COLORS = {"supsig->res-nplm": "#2a78d6", "supsig+nplm": "#008300",
          "supcon+nplm": "#eda100", "nplmsup+nplm": "#d62728"}


def train_nplm_residual_feat(head, loader, epochs, means, tau=1.0, lam=1.0,
                             n_slices=64, lr=1e-3):
    """NPLM+sigreg on residuals r = z - means[y] (feature-bank two views)."""
    loss_fn = HybridContrastiveLoss(positives="instance", critic="bilinear",
                                    estimator="nplm", marginal="sigreg",
                                    tau=tau, lam=lam, n_slices=n_slices)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    head.train()
    for ep in range(epochs):
        int_run, marg_run, n = 0.0, 0.0, 0
        for f1, f2, y in loader:
            f1, f2, y = f1.to(DEVICE), f2.to(DEVICE), y.to(DEVICE)
            inst = torch.arange(f1.size(0), device=DEVICE)
            labels = torch.cat([inst, inst])
            opt.zero_grad()
            z = head(torch.cat([f1, f2]))
            r = z - means[torch.cat([y, y])]
            loss, parts = loss_fn(r, labels)
            loss.backward()
            opt.step()
            int_run += parts["interaction"].item() * f1.size(0)
            marg_run += parts["marginal"].item() * f1.size(0)
            n += f1.size(0)
        n = max(n, 1)
        if (ep + 1) % 20 == 0 or ep == epochs - 1:
            print(f"  [nplm-res] epoch {ep+1}/{epochs}  "
                  f"interaction={int_run/n:.4f}  marginal={marg_run/n:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="aircraft",
                    choices=list(exp44.N_CLASSES))
    ap.add_argument("--base", default="dino", choices=list(exp40.CACHE_TAG))
    ap.add_argument("--holdouts", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dim-half", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--n-slices", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--aug-reps", type=int, default=8)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--fractions", default="0.003,0.01,0.02,0.05,0.1")
    ap.add_argument("--n-d", type=int, default=None)
    ap.add_argument("--kernels", type=int, default=16)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--sparker-sigma", type=float, default=1.0)
    ap.add_argument("--arms", nargs="+", default=list(COLORS),
                    choices=list(COLORS))
    args = ap.parse_args()

    N_CLS = exp44.N_CLASSES[args.dataset]
    if args.holdouts is None:
        args.holdouts = ",".join(str(c) for c in range(N_CLS - 10, N_CLS))
    if args.n_d is None:
        args.n_d = 2000 if args.dataset == "cars" else 1000
    holdouts = {int(x) for x in args.holdouts.split(",")}
    seen = [c for c in range(N_CLS) if c not in holdouts]
    con_ep = args.epochs or (5 if args.quick else 120)
    res_ep = 3 if args.quick else 60
    fractions = [float(x) for x in args.fractions.split(",")]
    n_null_pre = 20 if args.quick else 200
    n_sig_toys = 10 if args.quick else 50
    sparker_kw = dict(M=args.kernels, steps=args.steps)
    if args.sparker_sigma > 0:
        sparker_kw.update(sigma0=args.sparker_sigma, sigma_ratio=1.0,
                          n_checkpoints=1)
    tag = f"{args.dataset}_{args.base}"
    print(f"exp60 [{tag}] NPLM residual/concat, halves={args.dim_half}d, "
          f"holdouts={min(holdouts)}-{max(holdouts)}, arms={args.arms}")

    plain, bank = exp44.build_features(args.dataset, args.base, args)
    (Xtr, ytr), (Xte, yte) = plain["train"], plain["test"]
    tr_lab, te_lab = ytr.numpy(), yte.numpy()
    seen_bank = exp51.filter_bank(bank, ~np.isin(bank["labels"].numpy(),
                                                 list(holdouts)))

    def make_loader(labeled):
        ds_ = (exp37.TwoViewFeatures(seen_bank) if labeled
               else exp38.TwoViewUnlabeled(seen_bank))
        return DataLoader(ds_, batch_size=args.batch_size, shuffle=True,
                          drop_last=True)

    def embs_of(head):
        return (exp37.embed(head, Xtr).numpy(),
                exp37.embed(head, Xte).numpy())

    def cents_full(tr):
        """(N_CLS, dim) centroid matrix; holdout rows zero (never used)."""
        m = np.isin(tr_lab, seen)
        c = exp28.class_centroids(tr[m], tr_lab[m], seen)
        full = torch.zeros(N_CLS, c.size(1), device=DEVICE)
        for i, cl in enumerate(seen):
            full[cl] = c[i]
        return full, c

    def new_head(seed_off):
        torch.manual_seed(args.seed + seed_off)
        np.random.seed(args.seed + seed_off)
        return exp37.FeatureHead(args.dim_half).to(DEVICE)

    print("\n===== training: supcon16 =====")
    supcon16 = new_head(15)
    train_supcon(supcon16, make_loader(True), con_ep)
    print("\n===== training: supsig16 (supcon_sigreg) =====")
    supsig16 = new_head(14)
    exp34h.train_hybrid(supsig16, make_loader(True), con_ep,
                        dict(positives="supervised", critic="cosine",
                             estimator="softmax", marginal="sigreg", tau=0.1),
                        True, lam=args.lam, n_slices=args.n_slices)
    print("\n===== training: nplm_bil16 =====")
    nplm_bil16 = new_head(16)
    exp34h.train_hybrid(nplm_bil16, make_loader(False), con_ep,
                        dict(positives="instance", critic="bilinear",
                             estimator="nplm", marginal="sigreg",
                             tau=args.tau), False,
                        lam=args.lam, n_slices=args.n_slices)
    print("\n===== training: nplm_sup16 =====")
    nplm_sup16 = new_head(17)
    exp34h.train_hybrid(nplm_sup16, make_loader(True), con_ep,
                        dict(positives="supervised", critic="distance",
                             estimator="nplm", marginal="sigreg",
                             tau=args.tau), True,
                        lam=args.lam, n_slices=args.n_slices)

    supsig_tr, _ = embs_of(supsig16)
    means_supsig, _ = cents_full(supsig_tr)
    print("\n===== training: res_nplm16 (NPLM residual post supsig16) =====")
    torch.manual_seed(args.seed + 11); np.random.seed(args.seed + 11)
    res_nplm16 = copy.deepcopy(supsig16)
    train_nplm_residual_feat(res_nplm16, make_loader(True), res_ep,
                             means_supsig, tau=args.tau, lam=args.lam,
                             n_slices=args.n_slices)

    HALves = {"supcon": supcon16, "supsig": supsig16, "nplm_bil": nplm_bil16,
              "nplm_sup": nplm_sup16, "res_nplm": res_nplm16}
    embs = {n: embs_of(h) for n, h in HALves.items()}
    ARMS = {
        "supsig->res-nplm": ("supsig", "res_nplm"),
        "supsig+nplm": ("supsig", "nplm_bil"),
        "supcon+nplm": ("supcon", "nplm_bil"),
        "nplmsup+nplm": ("nplm_sup", "nplm_bil"),
    }

    def probe_stat(tr, te, n_rep=3):
        aucs = []
        for s in range(n_rep):
            torch.manual_seed(1000 + s)
            a, _, _ = exp29.linear_probe_novelty(tr, tr_lab, te, te_lab,
                                                 holdouts)
            aucs.append(a)
        return float(np.mean(aucs)), float(np.std(aucs))

    results, trains, tests, anchors_of = {}, {}, {}, {}
    for name in args.arms:
        a, b = ARMS[name]
        tr = np.concatenate([embs[a][0], embs[b][0]], axis=1)
        te = np.concatenate([embs[a][1], embs[b][1]], axis=1)
        trains[name], tests[name] = tr, te
        m = np.isin(tr_lab, seen)
        cents = exp28.class_centroids(tr[m], tr_lab[m], seen)
        anchors_of[name] = torch.as_tensor(cents, dtype=torch.float32,
                                           device=DEVICE)
        r = exp29.evaluate_space(tr, tr_lab, te, te_lab, anchors_of[name],
                                 seen, holdouts)
        pm, psd = probe_stat(tr, te)
        g = gaussianity_summary(te, te_lab, seen, seed=args.seed)
        print(f"  [{name:<16}] probe={pm:.4f}+-{psd:.4f} acc={r['acc']:.4f} "
              f"supAUC={r['sup_auc']:.4f} eucl={r['eucl']:.4f} "
              f"mahaT={r['maha_tied']:.4f}")
        results[name] = dict(probe=pm, probe_sd=psd, acc=r["acc"],
                             sup_auc=r["sup_auc"], eucl=r["eucl"],
                             mahaT=r["maha_tied"], mahaPC=r["maha_pc"],
                             gauss=g)

    print("\n===== performance / novelty table =====")
    print(f"  {'arm':<18}{'probe':>16}{'acc':>8}{'eucl':>8}{'mahaT':>8}")
    for name in args.arms:
        r = results[name]
        print(f"  {name:<18}{r['probe']:>9.4f}+-{r['probe_sd']:.4f}"
              f"{r['acc']:>8.4f}{r['eucl']:>8.4f}{r['mahaT']:>8.4f}")
    print("\n===== gaussianity =====")
    exp28.print_gauss_table({n: results[n]["gauss"] for n in args.arms})

    pre_power = {s: {} for s in STATS}
    print("\n===== PRE power batteries =====")
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
        bg = torch.as_tensor(te[bg_mask], dtype=torch.float32, device=DEVICE)
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
        print(f"\n===== EXP60 {stat.upper()} PRE POWER =====")
        print(f"  {'arm':<18}" + "".join(f"{f:>9}" for f in fractions))
        for name in args.arms:
            print(f"  {name:<18}"
                  + "".join(f"{p:>9.3f}" for p in pre_power[stat][name]))
        plt.figure(figsize=(8, 6.5))
        for name in args.arms:
            plt.plot(fractions, pre_power[stat][name], "-o",
                     color=COLORS[name], lw=2, ms=5, label=name)
        plt.xscale("log")
        plt.axhline(args.alpha, color="gray", lw=1, ls=":")
        plt.xlabel("injected anomaly fraction")
        plt.ylabel(f"power at alpha={args.alpha}")
        plt.title(f"exp60 NPLM residual/concat ({tag}): {stat} pre power")
        plt.grid(alpha=0.25, which="both")
        plt.legend(loc="upper left", fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(plot_path(f"exp60_{stat}_power_{tag}.png"), dpi=150)
        plt.close()
        print("  saved " + plot_path(f"exp60_{stat}_power_{tag}.png"))

    os.makedirs(os.path.join("logs", "exp60"), exist_ok=True)
    np.savez(os.path.join("logs", "exp60", f"residual_concat_{tag}.npz"),
             fractions=np.array(fractions), arms=np.array(args.arms),
             holdouts=np.array(sorted(holdouts)),
             **{f"{k}_{n}": np.array(results[n][k]) for n in args.arms
                for k in ("probe", "probe_sd", "acc", "sup_auc", "eucl",
                          "mahaT", "mahaPC")},
             **{f"{s}_{n}_pre": np.array(pre_power[s][n]) for s in STATS
                for n in args.arms})
    print("\n===== EXP60 SUMMARY =====")
    for name in args.arms:
        r = results[name]
        print(f"  [{name:<16}] probe={r['probe']:.4f}  acc={r['acc']:.4f}  "
              f"eucl={r['eucl']:.4f}  mahaT={r['mahaT']:.4f}")
    print("Done.")


if __name__ == "__main__":
    main()
