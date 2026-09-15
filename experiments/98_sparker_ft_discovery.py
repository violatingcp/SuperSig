"""
Experiment 98 (IMPROVEMENT_TESTS.md #98): SparKer-ft as the discovery
fine-tune objective — the missing third corner of the exp-58 split.

Exp 58: proto/repulse wins the probe, NPLM-ft wins every power statistic.
SparKer-ft optimizes the EVENT-LEVEL statistic the battery reports:
alternating optimization inside the standard discovery loop — each ft
epoch refits (mu, a, sigma) on the current corpus-vs-reference embeddings
(encoder frozen), then the encoder takes gradient steps on the NP loss
L = E_ref[e^f - 1] - E_corpus[f] with the kernel fixed (+ global SIGReg
guard, lam=1).  Pooling/clustering stay the standard distance loop;
anchors are recomputed as centroids after each round (no anchor params),
exactly like the exp-58 nplm arm.

All three fts (proto / nplm / sparker) run FRESH here under one identical
post battery (probe, acc/eucl/mahaT/lid, per-event on d_seen - d_disc,
annealed-sigma SparKer post power at f in {0.01, 0.02, 0.03}), so the
three-way comparison is exactly paired; exp-58's archived numbers are the
qualitative cross-check.

Prediction: sparker-ft power >= nplm-ft, probe worse than proto, and
better round-2 purity retention (an event-level objective has no reason
to inflate the space).  Falsifier: round-2 purity degrades like proto's
— the inflation is a property of any discovery ft.

    python experiments/98_sparker_ft_discovery.py
    python experiments/98_sparker_ft_discovery.py --quick --arms nplm_sup_dist --fts sparker
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import copy
import importlib
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from supersig.config import DEVICE
from supersig.data import get_cifar_loaders
from supersig.discovery import run_discovery
from supersig.losses import sigreg_loss
from supersig.recipes import recipe
from supersig.sparker import median_pairwise
from supersig.train import collect_embeddings

exp28 = importlib.import_module("28_concat_residual")
exp29 = importlib.import_module("29_residual_finetune")
exp30 = importlib.import_module("30_power_curves")
exp31 = importlib.import_module("31_sparker_power")
exp55 = importlib.import_module("55_nplm_discovery")
exp58 = importlib.import_module("58_nplm_ft_discovery")
exp92 = importlib.import_module("92_sparker_discovery")

ARMS = ["nplm_sup_dist", "nplm_dist_sup_cw"]
FTS = ["proto", "nplm", "sparker"]


def train_sparker_ft(backbone, ft_loader, epochs, n_classes,
                     train_eval_loader, lam=1.0, n_slices=64, lr=1e-3,
                     inner_steps=150):
    """Alternating SparKer-ft (module docstring)."""
    opt = torch.optim.Adam(backbone.parameters(), lr=lr)
    for ep in range(epochs):
        # inner: refit the kernel test on the current embeddings
        backbone.eval()
        tr_embs, tr_lab = collect_embeddings(backbone, train_eval_loader)
        z = torch.as_tensor(tr_embs, dtype=torch.float32, device=DEVICE)
        ref = z[torch.as_tensor(tr_lab < n_classes, device=DEVICE)]
        g = torch.Generator().manual_seed(ep)
        sub = lambda X, n: X[torch.randperm(len(X), generator=g)[:n]
                             .to(DEVICE)]
        mu, a, sigma, _ = exp92.fit_sparker(sub(z, 8000), sub(ref, 4000),
                                            M=16, steps=inner_steps,
                                            seed=ep)
        mu, a = mu.detach(), a.detach()

        def f(zz):
            k = torch.exp(-0.5 * torch.cdist(zz, mu).pow(2) / sigma ** 2)
            p = k / (k.sum(dim=1, keepdim=True) + 1e-12)
            return ((p * k) @ a).clamp(-20.0, 20.0)

        # outer: encoder steps on the NP loss with the kernel fixed
        backbone.train()
        np_run, marg_run, n = 0.0, 0.0, 0
        for x, y in ft_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            zb = backbone(x)
            fz = f(zb)
            seen_rows = y < n_classes
            if not seen_rows.any() or seen_rows.all():
                continue
            np_loss = (torch.exp(fz[seen_rows]) - 1.0).mean() \
                - fz.mean()
            marg = sigreg_loss(zb, n_slices=n_slices)
            (np_loss + lam * marg).backward()
            opt.step()
            np_run += np_loss.item() * x.size(0)
            marg_run += marg.item() * x.size(0)
            n += x.size(0)
        n = max(n, 1)
        print(f"  [sparker-ft] epoch {ep+1}/{epochs}  np={np_run/n:+.6f}  "
              f"marginal={marg_run/n:.4f}", flush=True)


def run_discovery_sparker_ft(backbone, means, *, base_ds, train_eval_loader,
                             test_loader, seen, holdouts, lam, n_slices,
                             rounds=2, ft_epochs=5, seed=0):
    """exp-58's nplm-ft discovery loop with the ft step swapped for
    train_sparker_ft (pooling/clustering/anchor-recompute identical)."""
    import types
    saved = exp58.train_nplm_ft

    def shim(bb, loader, epochs, critic, tau, lam_, n_slices_):
        train_sparker_ft(bb, loader, epochs, means.size(0),
                         train_eval_loader, lam=lam_, n_slices=n_slices_)

    exp58.train_nplm_ft = shim
    try:
        return exp58.run_discovery_nplm_ft(
            backbone, means, base_ds=base_ds,
            train_eval_loader=train_eval_loader, test_loader=test_loader,
            seen=seen, holdouts=holdouts, critic="distance", tau=1.0,
            lam=lam, n_slices=n_slices, rounds=rounds, ft_epochs=ft_epochs,
            seed=seed)
    finally:
        exp58.train_nplm_ft = saved


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
    ap.add_argument("--fractions", default="0.01,0.02,0.03")
    ap.add_argument("--n-d", type=int, default=5000)
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
    n_null = 20 if args.quick else 100
    n_toys = 10 if args.quick else 50
    sparker_kw = dict(M=16, steps=50 if args.quick else 300)  # annealed
    print(f"exp98 [{ds}] SparKer-ft vs proto vs nplm discovery, "
          f"arms={args.arms}", flush=True)

    train_loader, test_loader = get_cifar_loaders(quick=args.quick,
                                                  dataset=ds)
    tel = DataLoader(train_loader.dataset, batch_size=256, shuffle=False,
                     num_workers=2)
    results = {}
    for arm in args.arms:
        print(f"\n===== training: {arm} =====", flush=True)
        net = exp55.train_arm(arm, ds, cfg, args, con_ep, holdouts)
        tr, tr_lab = collect_embeddings(net, tel)
        te, te_lab = collect_embeddings(net, tel if False else test_loader)
        m = np.isin(tr_lab, seen)
        means0 = exp28.fill_means(
            exp28.class_centroids(tr[m], tr_lab[m], seen), seen,
            cfg).detach()
        torch.manual_seed(1000)
        a_pre, _, _ = exp29.linear_probe_novelty(tr, tr_lab, te, te_lab,
                                                 holdouts)

        for ft in args.fts:
            print(f"\n----- {arm} ft={ft} -----", flush=True)
            bb = copy.deepcopy(net)
            if ft == "proto":
                cur_means, hist = run_discovery(
                    bb, means0.clone(), base_ds=train_loader.dataset,
                    train_eval_loader=tel, test_loader=test_loader,
                    seen=seen, holdouts=holdouts, dataset_name=ds,
                    rep_weight=cfg["rep_weight"],
                    sigreg_weight=cfg["sigreg_weight"],
                    n_slices=cfg["n_slices"], rounds=args.rounds,
                    ft_epochs=ft_ep, names=None, seed=args.seed)
            elif ft == "nplm":
                cur_means, hist = exp58.run_discovery_nplm_ft(
                    bb, means0.clone(), base_ds=train_loader.dataset,
                    train_eval_loader=tel, test_loader=test_loader,
                    seen=seen, holdouts=holdouts, critic="distance",
                    tau=args.tau, lam=args.lam, n_slices=cfg["n_slices"],
                    rounds=args.rounds, ft_epochs=ft_ep, seed=args.seed)
            else:
                cur_means, hist = run_discovery_sparker_ft(
                    bb, means0.clone(), base_ds=train_loader.dataset,
                    train_eval_loader=tel, test_loader=test_loader,
                    seen=seen, holdouts=holdouts, lam=args.lam,
                    n_slices=cfg["n_slices"], rounds=args.rounds,
                    ft_epochs=ft_ep, seed=args.seed)

            trp, _ = collect_embeddings(bb, tel)
            tep, _ = collect_embeddings(bb, test_loader)
            anch = exp28.class_centroids(trp[m], tr_lab[m],
                                         seen).detach().float().to(DEVICE)
            r = exp29.evaluate_space(trp, tr_lab, tep, te_lab, anch, seen,
                                     holdouts)
            torch.manual_seed(1000)
            a_post, _, _ = exp29.linear_probe_novelty(trp, tr_lab, tep,
                                                      te_lab, holdouts)
            zt = torch.as_tensor(tep, dtype=torch.float32, device=DEVICE)
            d_seen = torch.cdist(zt, cur_means[seen].to(DEVICE)) \
                .min(1).values
            d_disc = torch.cdist(zt, cur_means[n_cls:].to(DEVICE)) \
                .min(1).values if cur_means.size(0) > n_cls else d_seen * 0
            s = (d_seen - d_disc).cpu().numpy()
            bgm = np.isin(te_lab, seen)
            sgm = np.isin(te_lab, list(holdouts))
            pe = exp30.power_at_alpha(s[bgm], s[sgm], args.alpha)
            R = torch.as_tensor(trp[m][:20000], dtype=torch.float32,
                                device=DEVICE)
            spk, _ = exp31.run_test_battery(
                torch.as_tensor(tep[bgm], dtype=torch.float32,
                                device=DEVICE),
                torch.as_tensor(tep[sgm], dtype=torch.float32,
                                device=DEVICE),
                R, fractions, args.n_d, n_null, n_toys, args.alpha,
                args.seed, sparker_kw, tag=f"{arm}:{ft}")
            results[f"{arm}:{ft}"] = dict(
                probe_pre=a_pre, probe_post=a_post, eucl=r["eucl"],
                mahaT=r["maha_tied"], lid=r["lid"], perevt=pe,
                sparker=spk, purity=[h["purity"] for h in hist],
                margin=[h["margin"] for h in hist])
            print(f"  [{arm}:{ft}] probe {a_pre:.4f}->{a_post:.4f}  "
                  f"mahaT={r['maha_tied']:.4f}  perevt={pe:.3f}  "
                  f"spk={'/'.join(f'{p:.2f}' for p in spk)}  purity=" +
                  "/".join(f"{p:.3f}" for p in
                           results[f'{arm}:{ft}']['purity']), flush=True)
            del bb
            torch.cuda.empty_cache()
        del net
        torch.cuda.empty_cache()

    print(f"\n===== EXP98 SUMMARY (fracs {args.fractions}) =====")
    print(f"  {'arm:ft':<26}{'probe post':>11}{'mahaT':>8}{'perevt':>8}"
          f"{'spk@.02':>9}{'pur r1/r2':>12}")
    for k, r in results.items():
        pur = r["purity"] + [float("nan")] * (2 - len(r["purity"]))
        spk02 = r["sparker"][1] if len(r["sparker"]) > 1 else float("nan")
        print(f"  {k:<26}{r['probe_post']:>11.4f}{r['mahaT']:>8.3f}"
              f"{r['perevt']:>8.3f}{spk02:>9.2f}"
              f"{pur[0]:>6.3f}/{pur[1]:.3f}")

    os.makedirs(os.path.join("logs", "exp98"), exist_ok=True)
    np.savez(os.path.join("logs", "exp98", f"results_{ds}.npz"),
             summary=np.array([repr(results)], dtype=object))
    print("EXP98 DONE.")


if __name__ == "__main__":
    main()
