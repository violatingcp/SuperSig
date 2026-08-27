"""
Experiment 92 (IMPROVEMENT_TESTS.md #92): SparKer centres as the discovery
clustering.

Cars pools at purity <=0.14 because novel models are not geometrically
outlying -- but "not outlying in distance" does not imply "not outlying in
density ratio": p_corpus/p_seen can be >> 1 at ordinary centroid distance.
SparKer's gated kernels place their M trainable centres exactly where that
ratio is large.  This replaces steps (ii)-(iv) of the discovery loop
(quantile pool -> BIC k-means -> merge) with:

  fit SparKer f on (unlabeled corpus D vs seen-train reference R) ->
  keep the trained centres mu_i with f(mu_i) above the 0.95 seen-quantile
  of f -> pool = {x : f(x) > same quantile} -> pseudo-label by nearest
  kept centre -> merge + train_sigreg_hybrid as usual.

A/B against the standard distance loop (run_discovery, identical
recipe/seed) on the three cells spanning the purity range: cars:dino
(0.14, fails), aircraft:visreg (0.53, purity-adequate but probe-negative),
flowers:dino (0.61, works).  PRIMARY readout: round-1 purity.
Prediction: purity rises most on cars; parity on flowers.  Falsifier:
SparKer centres are no purer -- density ratio does not see what distance
misses at these sample sizes.

    python experiments/92_sparker_discovery.py
    python experiments/92_sparker_discovery.py --cells flowers:dino --quick
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from supersig.holdouts import holdout_set, n_holdout, run_tag
import argparse
import copy
import importlib
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from supersig.config import DEVICE
from supersig.discovery import (PseudoDataset, merge_anchors, run_discovery)
from supersig.data import BalancedBatchSampler
from supersig.sparker import median_pairwise
from supersig.train import collect_embeddings, train_sigreg_hybrid

exp28 = importlib.import_module("28_concat_residual")
exp29 = importlib.import_module("29_residual_finetune")
exp44 = importlib.import_module("44_transfer_32d")
exp72 = importlib.import_module("72_residual_discovery")

CELLS = ["cars:dino", "aircraft:visreg", "flowers:dino"]


def fit_sparker(D, R, M=16, steps=300, sigma_ratio=10.0, lr=0.05, seed=0):
    """SparKer NP fit that RETURNS the trained model (mu, a, sigma, f)."""
    torch.manual_seed(seed)
    w = len(D) / len(R)
    sigma0 = median_pairwise(D, seed=seed)
    sigmaT = sigma0 / sigma_ratio
    g = torch.Generator().manual_seed(seed)
    mu = D[torch.randperm(len(D), generator=g)[:M].to(D.device)] \
        .clone().requires_grad_(True)
    a = torch.zeros(M, device=D.device, requires_grad=True)
    opt = torch.optim.Adam([mu, a], lr=lr)

    def f(X, sigma):
        k = torch.exp(-0.5 * torch.cdist(X, mu).pow(2) / sigma ** 2)
        p = k / (k.sum(dim=1, keepdim=True) + 1e-12)
        return ((p * k) @ a).clamp(-20.0, 20.0)

    sigma = sigma0
    for t in range(1, steps + 1):
        sigma = sigma0 + (sigmaT - sigma0) * t / steps
        loss = w * (torch.exp(f(R, sigma)) - 1).sum() - f(D, sigma).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return mu.detach(), a.detach(), sigma, \
        lambda X: f(X, sigma).detach()


def sparker_discovery(backbone, means, *, base_ds, train_eval_loader,
                      test_loader, seen, holdouts, rep_weight, sigreg_weight,
                      n_slices, rounds=2, ft_epochs=5, tau_quantile=0.95,
                      M=16, steps=300, merge_dist=3.0, seed=0,
                      bn_adapt=False):
    """The exp-24 discovery loop with the anchor-proposal step swapped for
    SparKer density-ratio centres (module docstring)."""
    n_classes = means.size(0)
    cur_means = means.detach().clone()
    pooled = np.zeros(len(train_eval_loader.dataset), dtype=bool)
    history = []
    for r in range(1, rounds + 1):
        tr_embs, tr_lab = collect_embeddings(backbone, train_eval_loader)
        z = torch.as_tensor(tr_embs, device=DEVICE)
        is_seen_lab = np.isin(tr_lab, seen)
        ref = z[torch.as_tensor(is_seen_lab, device=DEVICE)]
        if len(ref) > 4000:
            g = torch.Generator().manual_seed(seed + r)
            ref = ref[torch.randperm(len(ref), generator=g)[:4000]
                      .to(DEVICE)]
        mu, a, sigma, f_fn = fit_sparker(z, ref, M=M, steps=steps,
                                         seed=seed + r)
        f_all = f_fn(z)
        tau = torch.quantile(
            f_all[torch.as_tensor(is_seen_lab, device=DEVICE)], tau_quantile)
        keep = f_fn(mu) > tau
        centers = mu[keep]
        pool = (f_all > tau).cpu().numpy()
        purity = (~is_seen_lab[pool]).mean() if pool.any() else float("nan")
        if centers.size(0) == 0:
            print(f"  round {r}: no centres above tau, stopping")
            history.append(dict(round=r, pool=int(pool.sum()),
                                purity=float(purity), khat=0, n_anchors=0,
                                margin=float("nan"), mean_pc=float("nan"),
                                per_class={}))
            break
        cur_means = torch.cat([cur_means, centers], dim=0)
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
        train_sigreg_hybrid(backbone, ft_loader, ft_epochs, cur_means,
                            mode="repulse", disc="proto", alpha=1.0,
                            rep_weight=rep_weight,
                            sigreg_weight=sigreg_weight, n_slices=n_slices,
                            rep_exempt_from=n_classes, bn_adapt=bn_adapt)
        cur_means = cur_means.detach()

        from sklearn.metrics import roc_auc_score
        te_embs, te_lab = collect_embeddings(backbone, test_loader)
        zt = torch.as_tensor(te_embs, device=DEVICE)
        d_seen = torch.cdist(zt, cur_means[seen]).min(1).values
        d_each = torch.cdist(zt, cur_means[n_classes:])
        is_unseen = np.isin(te_lab, list(holdouts)).astype(int)
        margin = roc_auc_score(is_unseen,
                               (d_seen - d_each.min(1).values).cpu().numpy())
        per_class = {}
        for c in sorted(holdouts):
            counts = [int(((te_lab == c)
                           & (d_each.argmin(1).cpu().numpy() == j)).sum())
                      for j in range(d_each.size(1))]
            j = int(np.argmax(counts))
            per_class[c] = roc_auc_score((te_lab == c).astype(int),
                                         (-d_each[:, j]).cpu().numpy())
        history.append(dict(round=r, pool=int(pool.sum()),
                            purity=float(purity), khat=int(keep.sum()),
                            n_anchors=int(disc.size(0)),
                            margin=float(margin),
                            mean_pc=float(np.mean(list(per_class.values())))))
        h = history[-1]
        print(f"  round {r}: pool={h['pool']} purity={h['purity']:.3f} "
              f"centres-kept={h['khat']} anchors={h['n_anchors']}  "
              f"margin={h['margin']:.4f}  mean-anchor={h['mean_pc']:.4f}")
    return cur_means, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=",".join(CELLS))
    ap.add_argument("--variants", nargs="+",
                    default=["distance", "sparker"],
                    choices=["distance", "sparker", "distance-frozen",
                             "sparker-frozen"],
                    help="-frozen = freeze the whole backbone (exp-86 "
                         "freeze-both): only the anchors train")
    ap.add_argument("--tag", default="",
                    help="suffix for the results npz (avoid clobbering)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--emb-dim", type=int, default=100)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--n-slices", type=int, default=64)
    ap.add_argument("--kernels", type=int, default=16)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    ft_ep = 1 if args.quick else 5
    steps = 50 if args.quick else args.steps

    results = {}
    for cell in args.cells.split(","):
        ds, base = cell.split(":")
        parent, obj, kind = exp72.WINNERS[(ds, base)]
        key = f"{ds}_{base}"
        N_CLS = 47 if ds == "dtd" else exp44.N_CLASSES[ds]
        n_hold = n_holdout(ds)
        holdouts = holdout_set(ds, N_CLS)
        seen = [c for c in range(N_CLS) if c not in holdouts]
        cfg = dict(n_classes=N_CLS, pair_dist=5.0, sigreg_weight=1.0,
                   n_slices=args.n_slices,
                   rep_weight=exp72.REP_WEIGHT * 45.0
                   / (N_CLS * (N_CLS - 1) / 2))
        print(f"\n######## [{key}] SparKer-centre vs distance discovery on "
              f"{parent}->{obj} {kind} ########", flush=True)
        bb0, Xtr, ytr, Xte, yte = exp72.load_cell(ds, base, parent, obj,
                                                  args)
        tr_lab, te_lab = ytr.numpy(), yte.numpy()
        base_feats = TensorDataset(Xtr, ytr)
        tr_loader = DataLoader(base_feats, batch_size=512, shuffle=False)
        te_loader = DataLoader(TensorDataset(Xte, yte), batch_size=512,
                               shuffle=False)

        def battery(tr, te):
            m = np.isin(tr_lab, seen)
            anch = exp28.class_centroids(tr[m], tr_lab[m],
                                         seen).detach().float().to(DEVICE)
            r = exp29.evaluate_space(tr, tr_lab, te, te_lab, anch, seen,
                                     holdouts)
            aucs = []
            for s in range(3):
                torch.manual_seed(1000 + s)
                a, _, _ = exp29.linear_probe_novelty(tr, tr_lab, te,
                                                     te_lab, holdouts)
                aucs.append(a)
            return dict(probe=float(np.mean(aucs)), eucl=r["eucl"],
                        mahaT=r["maha_tied"], lid=r["lid"])

        tr0, _ = collect_embeddings(bb0, tr_loader)
        te0, _ = collect_embeddings(bb0, te_loader)
        pre = battery(tr0, te0)
        m = np.isin(tr_lab, seen)
        means0 = exp28.fill_means(
            exp28.class_centroids(tr0[m], tr_lab[m], seen), seen,
            cfg).detach()
        print(f"  pre: probe={pre['probe']:.4f} eucl={pre['eucl']:.4f} "
              f"mahaT={pre['mahaT']:.4f}", flush=True)

        for variant in args.variants:
            bb = copy.deepcopy(bb0)
            if variant.endswith("-frozen"):
                for p in bb.parameters():
                    p.requires_grad_(False)
            kw = dict(base_ds=base_feats, train_eval_loader=tr_loader,
                      test_loader=te_loader, seen=seen, holdouts=holdouts,
                      rep_weight=cfg["rep_weight"],
                      sigreg_weight=cfg["sigreg_weight"],
                      n_slices=cfg["n_slices"], rounds=args.rounds,
                      ft_epochs=ft_ep, seed=args.seed)
            if variant.startswith("distance"):
                _, hist = run_discovery(bb, means0.clone(),
                                        dataset_name=ds, names=None, **kw)
            else:
                _, hist = sparker_discovery(bb, means0.clone(),
                                            M=args.kernels, steps=steps,
                                            **kw)
            trp, _ = collect_embeddings(bb, tr_loader)
            tep, _ = collect_embeddings(bb, te_loader)
            post = battery(trp, tep)
            results[f"{key}:{variant}"] = dict(
                pre=pre, post=post,
                purity=[h["purity"] for h in hist],
                pool=[h["pool"] for h in hist],
                margin=[h.get("margin", float("nan")) for h in hist])
            print(f"  [{variant:<15}] purity " +
                  " ".join(f"r{h['round']}={h['purity']:.3f}"
                           f"(n={h['pool']})" for h in hist) +
                  f"  probe {pre['probe']:.4f}->{post['probe']:.4f}"
                  f"  mahaT {pre['mahaT']:.4f}->{post['mahaT']:.4f}",
                  flush=True)
            del bb
            torch.cuda.empty_cache()
        del bb0
        torch.cuda.empty_cache()

    print("\n===== EXP92 SUMMARY (distance vs SparKer-centre discovery) ====")
    print(f"  {'cell':<18}{'variant':<10}{'pur r1':>8}{'pur r2':>8}"
          f"{'probe post':>11}{'mahaT post':>11}")
    for k, r in results.items():
        cell, var = k.rsplit(":", 1)
        pur = r["purity"] + [float("nan")] * (2 - len(r["purity"]))
        print(f"  {cell:<18}{var:<10}{pur[0]:>8.3f}{pur[1]:>8.3f}"
              f"{r['post']['probe']:>11.4f}{r['post']['mahaT']:>11.4f}")

    os.makedirs(os.path.join("logs", "exp92"), exist_ok=True)
    np.savez(os.path.join("logs", "exp92", f"results{args.tag}.npz"),
             summary=np.array([repr(results)], dtype=object))
    print("EXP92 DONE.")


if __name__ == "__main__":
    main()
