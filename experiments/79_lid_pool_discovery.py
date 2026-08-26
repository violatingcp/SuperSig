"""
Experiment 79: LID as the discovery pool scorer on flowers (the exp-78
follow-up).  Exp-72 showed discovery-on-residuals is purity-gated;
exp-78 showed LID separates flowers novelty at 0.95 AUC where distance
scores sit ~0.83-0.93.  Hypothesis: swapping the outlier-pool scorer
from min-anchor-distance to LID raises pool purity and therefore the
post-discovery battery.

A/B per flowers winner cell (dino/lejepa/visreg, exp-72 WINNERS,
identical seed and recipe): run_discovery(pool_score="dist") vs
run_discovery(pool_score="lid").  Natural discovery only (no injected
power grid); reports per-round pool size + purity and the pre -> post
probe / eucl / mahaT / lid battery.

    python experiments/79_lid_pool_discovery.py
    python experiments/79_lid_pool_discovery.py --cells flowers:dino --quick
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from supersig.holdouts import n_holdout, run_tag
import argparse
import copy
import importlib
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from supersig.config import DEVICE
from supersig.discovery import run_discovery
from supersig.train import collect_embeddings

exp28 = importlib.import_module("28_concat_residual")
exp29 = importlib.import_module("29_residual_finetune")
exp44 = importlib.import_module("44_transfer_32d")
exp72 = importlib.import_module("72_residual_discovery")

REP_WEIGHT = exp72.REP_WEIGHT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="flowers:dino,flowers:lejepa,"
                                       "flowers:visreg")
    ap.add_argument("--scorers", nargs="+", default=["dist", "lid"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--emb-dim", type=int, default=100)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--n-slices", type=int, default=64)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    cells = [tuple(c.split(":")) for c in args.cells.split(",")]
    ft_ep = 1 if args.quick else 5

    results = {}
    for ds, base in cells:
        parent, obj, kind = exp72.WINNERS[(ds, base)]
        key = f"{ds}_{base}"
        space = f"{parent}->{obj} {kind}"
        N_CLS = 47 if ds == "dtd" else exp44.N_CLASSES[ds]
        n_hold = n_holdout(ds)
        holdouts = set(range(N_CLS - n_hold, N_CLS))
        seen = [c for c in range(N_CLS) if c not in holdouts]
        cfg = dict(n_classes=N_CLS, pair_dist=5.0, sigreg_weight=1.0,
                   n_slices=args.n_slices,
                   rep_weight=REP_WEIGHT * 45.0 / (N_CLS * (N_CLS - 1) / 2))
        print(f"\n######## [{key}] LID-pool A/B on {space} ########")
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
              f"mahaT={pre['mahaT']:.4f} lid={pre['lid']:.4f}")

        for scorer in args.scorers:
            bb = copy.deepcopy(bb0)
            _, hist = run_discovery(
                bb, means0.clone(), base_ds=base_feats,
                train_eval_loader=tr_loader, test_loader=te_loader,
                seen=seen, holdouts=holdouts, dataset_name=ds,
                rep_weight=cfg["rep_weight"],
                sigreg_weight=cfg["sigreg_weight"],
                n_slices=cfg["n_slices"], rounds=args.rounds,
                ft_epochs=ft_ep, names=None, seed=args.seed,
                pool_score=scorer)
            trp, _ = collect_embeddings(bb, tr_loader)
            tep, _ = collect_embeddings(bb, te_loader)
            post = battery(trp, tep)
            results[f"{key}:{scorer}"] = dict(
                pre=pre, post=post,
                pool=[h["pool"] for h in hist],
                purity=[h["purity"] for h in hist])
            print(f"  [{scorer:<4}] purity " +
                  " ".join(f"r{h['round']}={h['purity']:.3f}"
                           f"(n={h['pool']})" for h in hist) +
                  f"  probe {pre['probe']:.4f}->{post['probe']:.4f}"
                  f"  eucl {pre['eucl']:.4f}->{post['eucl']:.4f}"
                  f"  mahaT {pre['mahaT']:.4f}->{post['mahaT']:.4f}"
                  f"  lid {pre['lid']:.4f}->{post['lid']:.4f}")
            del bb
            torch.cuda.empty_cache()

    print("\n===== EXP79 SUMMARY (dist vs lid pool scorer) =====")
    print(f"  {'cell':<18}{'scorer':<7}{'pur r1':>8}{'pur r2':>8}"
          f"{'probe post':>11}{'eucl post':>10}{'mahaT post':>11}")
    for k, r in results.items():
        cell, scorer = k.rsplit(":", 1)
        pur = r["purity"] + [float("nan")] * (2 - len(r["purity"]))
        print(f"  {cell:<18}{scorer:<7}{pur[0]:>8.3f}{pur[1]:>8.3f}"
              f"{r['post']['probe']:>11.4f}{r['post']['eucl']:>10.4f}"
              f"{r['post']['mahaT']:>11.4f}")

    os.makedirs(os.path.join("logs", "exp79"), exist_ok=True)
    np.savez(os.path.join("logs", "exp79", "results.npz"),
             summary=np.array([repr(results)], dtype=object))
    print("Done.")


if __name__ == "__main__":
    main()
