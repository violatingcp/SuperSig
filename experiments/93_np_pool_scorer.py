"""
Experiment 93 (IMPROVEMENT_TESTS.md #93): the NP score as the pool scorer —
the three-way dist vs LID vs f comparison inside the STANDARD discovery loop.

Exp 79 showed the pool scorer matters (scale-free LID dominates distance);
the NP lemma says the learned density-ratio f should dominate both.  Same
harness as exp 79 (run_discovery with pool_score in {dist, lid, np};
`discovery.np_pool_scores` fits the SparKer critic per round), identical
recipe/seed, on the three exp-79 flowers cells plus cars:dino (the
below-the-gate cell where the scorer could actually matter).

Prediction: np >= lid > dist on purity in every cell; probe unmoved on
flowers (above the gate), cars is the interesting cell.  Falsifier: np
underperforms lid (finite-sample estimation variance beats asymptotic
optimality).

    python experiments/93_np_pool_scorer.py
    python experiments/93_np_pool_scorer.py --cells flowers:dino --quick
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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

CELLS = ["flowers:dino", "flowers:lejepa", "flowers:visreg", "cars:dino"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=",".join(CELLS))
    ap.add_argument("--scorers", nargs="+", default=["dist", "lid", "np"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--emb-dim", type=int, default=100)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--n-slices", type=int, default=64)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    ft_ep = 1 if args.quick else 5

    results = {}
    for cell in args.cells.split(","):
        ds, base = cell.split(":")
        parent, obj, kind = exp72.WINNERS[(ds, base)]
        key = f"{ds}_{base}"
        N_CLS = 47 if ds == "dtd" else exp44.N_CLASSES[ds]
        n_hold = 1 if ds == "galaxy10" else 10
        holdouts = set(range(N_CLS - n_hold, N_CLS))
        seen = [c for c in range(N_CLS) if c not in holdouts]
        cfg = dict(n_classes=N_CLS, pair_dist=5.0, sigreg_weight=1.0,
                   n_slices=args.n_slices,
                   rep_weight=exp72.REP_WEIGHT * 45.0
                   / (N_CLS * (N_CLS - 1) / 2))
        print(f"\n######## [{key}] pool-scorer 3-way on {parent}->{obj} "
              f"{kind} ########", flush=True)
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
                purity=[h["purity"] for h in hist],
                margin=[h["margin"] for h in hist])
            print(f"  [{scorer:<4}] purity " +
                  " ".join(f"r{h['round']}={h['purity']:.3f}"
                           f"(n={h['pool']})" for h in hist) +
                  f"  probe {pre['probe']:.4f}->{post['probe']:.4f}"
                  f"  mahaT {pre['mahaT']:.4f}->{post['mahaT']:.4f}",
                  flush=True)
            del bb
            torch.cuda.empty_cache()
        del bb0
        torch.cuda.empty_cache()

    print("\n===== EXP93 SUMMARY (dist vs lid vs np pool scorer) =====")
    print(f"  {'cell':<18}{'scorer':<7}{'pur r1':>8}{'pur r2':>8}"
          f"{'probe post':>11}{'mahaT post':>11}")
    for k, r in results.items():
        cell, scorer = k.rsplit(":", 1)
        pur = r["purity"] + [float("nan")] * (2 - len(r["purity"]))
        print(f"  {cell:<18}{scorer:<7}{pur[0]:>8.3f}{pur[1]:>8.3f}"
              f"{r['post']['probe']:>11.4f}{r['post']['mahaT']:>11.4f}")

    os.makedirs(os.path.join("logs", "exp93"), exist_ok=True)
    np.savez(os.path.join("logs", "exp93", "results.npz"),
             summary=np.array([repr(results)], dtype=object))
    print("EXP93 DONE.")


if __name__ == "__main__":
    main()
