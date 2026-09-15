"""
Experiment 81 (IMPROVEMENT_TESTS.md #81, folding in #82): is the NPLM seed
variance a property of the CRITIC, not the estimator or the positives?

App. A of the paper: the NPLM reference gradient is e^g/N(N-1), so the
relative gradient variance grows as exp(s^2)-1 with the critic spread s.
The distance critic (g <= 0, e^g <= 1) is bounded by construction; the
bilinear critic is unbounded above.  2x2 at matched tau=1 on C100 32-D,
5 paired seeds: critic {distance, bilinear} x positives {instance,
supervised}, estimator=NPLM, marginal=global SIGReg.  The probe SD is the
primary statistic; the empirical critic spread s = sd(g_ij) on reference
pairs is logged every epoch (the theory's actual predictor), as is the
calibration residual E_ref[e^g] - 1 (exp 82: a label-free seed-selection
signal -- report Spearman(|resid|, probe) and best-of-5-by-residual).

Predictions: SD is governed by the critic column (bilinear ~0.04, distance
~0.01) independent of positives, and s tracks SD across cells.  Falsifier:
SD splits by positives instead.  Exp-82 falsifier: residual ~0 for every
seed regardless of probe.

    python experiments/81_critic_variance.py
    python experiments/81_critic_variance.py --quick --seeds 2
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import importlib
import numpy as np
import torch
from torch.utils.data import DataLoader

from supersig.config import DEVICE
from supersig.data import get_cifar_loaders, cifar_two_view_loader
from supersig.losses import HybridContrastiveLoss, _critic_matrix
from supersig.models import CIFARResNetBackbone
from supersig.recipes import recipe
from supersig.train import collect_embeddings

exp28 = importlib.import_module("28_concat_residual")
exp29 = importlib.import_module("29_residual_finetune")
exp30 = importlib.import_module("30_power_curves")

ARMS = {  # name -> (critic, positives)
    "dist-inst": ("distance", "instance"),
    "dist-sup": ("distance", "supervised"),
    "bil-inst": ("bilinear", "instance"),
    "bil-sup": ("bilinear", "supervised"),
}


def train_instrumented(net, loader, epochs, critic, positives, lam, n_slices,
                       tau=1.0, lr=1e-3):
    """exp-34h train_hybrid + per-epoch critic spread s and calibration
    residual E_ref[e^g]-1 on reference pairs (clamp 30 as in the loss)."""
    labeled = positives == "supervised"
    loss_fn = HybridContrastiveLoss(positives=positives, critic=critic,
                                    estimator="nplm", marginal="sigreg",
                                    tau=tau, lam=lam, n_slices=n_slices)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    s_hist, se_hist, r_hist = [], [], []
    for ep in range(epochs):
        s_run, se_run, r_run, n = 0.0, 0.0, 0.0, 0
        for batch in loader:
            if labeled:
                v1, v2, y = (t.to(DEVICE) for t in batch)
                labels = torch.cat([y, y])
            else:
                v1, v2 = batch[0].to(DEVICE), batch[1].to(DEVICE)
                inst = torch.arange(v1.size(0), device=DEVICE)
                labels = torch.cat([inst, inst])
            opt.zero_grad()
            z = net(torch.cat([v1, v2]))
            loss, _ = loss_fn(z, labels)
            loss.backward()
            opt.step()
            with torch.no_grad():
                g = _critic_matrix(z.detach(), critic, tau)
                m = torch.eye(g.size(0), dtype=torch.bool, device=g.device)
                pos = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~m
                ref = (~pos) & (~m)
                gr = g[ref]
                eg = torch.exp(gr.clamp(max=30.0))
                s_run += float(gr.std()) * v1.size(0)
                se_run += float(eg.std()) * v1.size(0)   # the App.-A gradient-
                r_run += float((eg - 1.0).mean()) * v1.size(0)  # variance proxy
                n += v1.size(0)
        s_hist.append(s_run / max(n, 1))
        se_hist.append(se_run / max(n, 1))
        r_hist.append(r_run / max(n, 1))
        print(f"    epoch {ep+1}/{epochs}  s={s_hist[-1]:.3f}  "
              f"s_exp={se_hist[-1]:.3f}  resid={r_hist[-1]:+.4f}", flush=True)
    return s_hist, se_hist, r_hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--holdout", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--arms", nargs="+", default=list(ARMS),
                    choices=list(ARMS))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    ds = args.dataset
    cfg = recipe(ds, emb_dim=args.dim)
    holdouts = {args.holdout}
    seen = [c for c in range(cfg["n_classes"]) if c not in holdouts]
    con_ep = args.epochs or (2 if args.quick else 20)
    tag = f"{ds}_{args.dim}d"
    out_path = os.path.join("logs", "exp81", f"results_{tag}.npz")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    done = {}
    if os.path.exists(out_path):
        d = np.load(out_path, allow_pickle=True)
        done = {k: d[k] for k in d.files}
    print(f"exp81 [{tag}] critic-variance 2x2, {args.seeds} paired seeds, "
          f"{con_ep} ep, tau={args.tau} lam={args.lam}", flush=True)

    train_loader, test_loader = get_cifar_loaders(quick=args.quick,
                                                  dataset=ds)
    train_eval_loader = DataLoader(train_loader.dataset, batch_size=256,
                                   shuffle=False, num_workers=2)

    for si in range(args.seeds):
        for name in args.arms:
            key = f"{name}_s{si}"
            if f"{key}_probe" in done:
                print(f"  [{key}] cached, skipping", flush=True)
                continue
            critic, positives = ARMS[name]
            labeled = positives == "supervised"
            print(f"\n----- {key} ({critic}/{positives}) -----", flush=True)
            torch.manual_seed(args.seed + 20 + si)
            np.random.seed(args.seed + 20 + si)
            net = CIFARResNetBackbone(args.dim, arch=cfg["arch"],
                                      pretrain=ds).to(DEVICE)
            loader = cifar_two_view_loader(quick=args.quick, labeled=labeled,
                                           holdout=holdouts, dataset=ds)
            s_hist, se_hist, r_hist = train_instrumented(
                net, loader, con_ep, critic, positives, args.lam,
                cfg["n_slices"], tau=args.tau)

            tr, tr_lab = collect_embeddings(net, train_eval_loader)
            te, te_lab = collect_embeddings(net, test_loader)
            m = np.isin(tr_lab, seen)
            anch = exp28.class_centroids(tr[m], tr_lab[m],
                                         seen).detach().float().to(DEVICE)
            r = exp29.evaluate_space(tr, tr_lab, te, te_lab, anch, seen,
                                     holdouts)
            aucs = []
            for s in range(3):
                torch.manual_seed(1000 + s)
                a, _, _ = exp29.linear_probe_novelty(tr, tr_lab, te, te_lab,
                                                     holdouts)
                aucs.append(a)
            d_te = torch.cdist(torch.as_tensor(te, dtype=torch.float32,
                                               device=DEVICE), anch) \
                .min(1).values.cpu().numpy()
            bg = np.isin(te_lab, seen)
            sg = np.isin(te_lab, list(holdouts))
            pe = exp30.power_at_alpha(d_te[bg], d_te[sg], args.alpha)
            done[f"{key}_probe"] = np.float64(np.mean(aucs))
            done[f"{key}_acc"] = np.float64(r["acc"])
            done[f"{key}_eucl"] = np.float64(r["eucl"])
            done[f"{key}_mahaT"] = np.float64(r["maha_tied"])
            done[f"{key}_lid"] = np.float64(r["lid"])
            done[f"{key}_perevt"] = np.float64(pe)
            done[f"{key}_s_hist"] = np.array(s_hist)
            done[f"{key}_s_exp_hist"] = np.array(se_hist)
            done[f"{key}_resid_hist"] = np.array(r_hist)
            np.savez(out_path, **done)
            print(f"  [{key}] probe={np.mean(aucs):.4f} eucl={r['eucl']:.4f} "
                  f"mahaT={r['maha_tied']:.4f} lid={r['lid']:.4f} "
                  f"perevt={pe:.3f} s_fin={s_hist[-1]:.3f} "
                  f"resid_fin={r_hist[-1]:+.4f}", flush=True)
            del net
            torch.cuda.empty_cache()

    # ===== summary ==========================================================
    from scipy.stats import spearmanr
    print(f"\n===== EXP81 SUMMARY [{tag}] ({args.seeds} paired seeds) =====")
    print(f"  {'arm':<11}{'probe mean+-sd':>17}{'eucl':>7}{'mahaT':>7}"
          f"{'perevt':>7}{'s_fin':>7}{'|resid|':>9}")
    stats = {}
    for name in args.arms:
        pr = [float(done[f"{name}_s{si}_probe"]) for si in range(args.seeds)
              if f"{name}_s{si}_probe" in done]
        if not pr:
            continue
        eu = np.mean([float(done[f"{name}_s{si}_eucl"])
                      for si in range(args.seeds)])
        ma = np.mean([float(done[f"{name}_s{si}_mahaT"])
                      for si in range(args.seeds)])
        pe = np.mean([float(done[f"{name}_s{si}_perevt"])
                      for si in range(args.seeds)])
        sf = np.mean([done[f"{name}_s{si}_s_hist"][-1]
                      for si in range(args.seeds)])
        sef = np.mean([done[f"{name}_s{si}_s_exp_hist"][-1]
                       for si in range(args.seeds)
                       if f"{name}_s{si}_s_exp_hist" in done] or [np.nan])
        rf = [abs(done[f"{name}_s{si}_resid_hist"][-1])
              for si in range(args.seeds)]
        stats[name] = dict(mean=np.mean(pr), sd=np.std(pr), s=sf, s_exp=sef,
                           resid=np.mean(rf))
        print(f"  {name:<11}{np.mean(pr):>9.4f}+-{np.std(pr):.4f}"
              f"{eu:>7.3f}{ma:>7.3f}{pe:>7.3f}{sf:>7.3f}"
              f"{np.mean(rf):>9.4f}  s_exp={sef:.3f}")
        rho = spearmanr(rf, pr).correlation if len(pr) > 2 else float("nan")
        best = pr[int(np.argmin(rf))]
        print(f"    exp82: Spearman(|resid|,probe)={rho:+.2f}  "
              f"best-of-{len(pr)}-by-resid probe={best:.4f} "
              f"(vs mean {np.mean(pr):.4f}, max {np.max(pr):.4f})")
    if len(stats) == 4:
        sd_crit = {c: np.mean([stats[f"{c}-{p}"]["sd"]
                               for p in ("inst", "sup")])
                   for c in ("dist", "bil")}
        sd_pos = {p: np.mean([stats[f"{c}-{p}"]["sd"]
                              for c in ("dist", "bil")])
                  for p in ("inst", "sup")}
        print(f"\n  sd by critic:    dist={sd_crit['dist']:.4f}  "
              f"bil={sd_crit['bil']:.4f}  (ratio "
              f"{sd_crit['bil'] / max(sd_crit['dist'], 1e-9):.1f}x)")
        print(f"  sd by positives: inst={sd_pos['inst']:.4f}  "
              f"sup={sd_pos['sup']:.4f}  (ratio "
              f"{max(sd_pos.values()) / max(min(sd_pos.values()), 1e-9):.1f}x)")
        names = list(stats)
        rho_s = spearmanr([stats[n]["s"] for n in names],
                          [stats[n]["sd"] for n in names]).correlation
        rho_se = spearmanr([stats[n]["s_exp"] for n in names],
                           [stats[n]["sd"] for n in names]).correlation
        print(f"  Spearman(s_fin, probe sd) across cells: {rho_s:+.2f}   "
              f"Spearman(s_exp_fin, probe sd): {rho_se:+.2f}")
    print("EXP81 DONE.")


if __name__ == "__main__":
    main()
