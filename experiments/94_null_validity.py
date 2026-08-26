"""
Experiment 94 (IMPROVEMENT_TESTS.md #94): null validity when the encoder has
seen the test data — the prerequisite for exps 95/98.

Data-snooping check: when the embedding is produced by a NOVELTY-SEEKING
fine-tune (the discovery ft; exps 92/95/98 sharpen this), the encoder can
manufacture separation on the very corpus it optimized, shifting the test
statistic on that data relative to fresh draws and making the nominal alpha
anticonservative.

Design (per cell): background = seen-class events only, everywhere.
  null calibration : SparKer aggregates of n_null pure-bg toys drawn from
                     the TEST-set background (data no ft ever touched),
                     threshold = (1-alpha) quantile.
  FPR measurement  : n_test pure-bg toys drawn from the TRAIN-corpus
                     background (the data the ft optimized), fraction
                     above the threshold.  Clopper-Pearson 68% interval.
  regimes          : frozen    the champion space as loaded (current
                               protocol);
                     ft-full   discovery ft on the full train corpus
                               (the exp-72 recipe -- toys draw from data
                               the ft pooled/clustered/refit);
                     ft-split  discovery ft on half A of the corpus,
                               FPR toys drawn from half B only.

Prediction: frozen ~ alpha, ft-full > alpha, ft-split ~ alpha (split
disjointness is the required protocol).  Falsifier: ft-full also ~ alpha
-- the encoder cannot manufacture separation at these sizes and the
concern is theoretical.

    python experiments/94_null_validity.py
    python experiments/94_null_validity.py --cells flowers:dino --quick
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
from supersig.sparker import (aggregate_pvalues, clopper_pearson,
                              median_pairwise, np_test_stats)
from supersig.train import collect_embeddings

exp28 = importlib.import_module("28_concat_residual")
exp44 = importlib.import_module("44_transfer_32d")
exp72 = importlib.import_module("72_residual_discovery")

CELLS = ["flowers:dino", "cars:dino"]


def toy_aggregates(pool, R, n_toys, n_d, rng, sparker_kw, seed, null_ts=None):
    """SparKer aggregate score of n_toys pure-background draws from pool."""
    ts = []
    for i in range(n_toys):
        idx = rng.choice(len(pool), size=min(n_d, len(pool)),
                         replace=n_d > len(pool))
        D = pool[torch.as_tensor(idx, device=DEVICE)]
        ts.append(np_test_stats(D, R, seed=seed + i, **sparker_kw))
    ts = np.array(ts)
    if null_ts is None:                      # self-calibrated (leave-one-out)
        agg = np.array([aggregate_pvalues(ts[i], np.delete(ts, i, axis=0))
                        for i in range(len(ts))])
        return ts, agg
    return ts, np.array([aggregate_pvalues(t, null_ts) for t in ts])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=",".join(CELLS))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--emb-dim", type=int, default=100)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--n-null", type=int, default=None)
    ap.add_argument("--n-test", type=int, default=None)
    ap.add_argument("--n-d", type=int, default=1000)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--n-slices", type=int, default=64)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    n_null = args.n_null or (20 if args.quick else 100)
    n_test = args.n_test or (20 if args.quick else 100)
    sparker_kw = dict(M=16, steps=50 if args.quick else 300)
    ft_ep = 1 if args.quick else 5

    results = {}
    for cell in args.cells.split(","):
        ds, base = cell.split(":")
        parent, obj, kind = exp72.WINNERS[(ds, base)]
        key = f"{ds}_{base}"
        N_CLS = 47 if ds == "dtd" else exp44.N_CLASSES[ds]
        n_hold = n_holdout(ds)
        holdouts = set(range(N_CLS - n_hold, N_CLS))
        seen = [c for c in range(N_CLS) if c not in holdouts]
        cfg = dict(n_classes=N_CLS, pair_dist=5.0, sigreg_weight=1.0,
                   n_slices=args.n_slices,
                   rep_weight=exp72.REP_WEIGHT * 45.0
                   / (N_CLS * (N_CLS - 1) / 2))
        print(f"\n######## [{key}] null validity on {parent}->{obj} "
              f"########", flush=True)
        bb0, Xtr, ytr, Xte, yte = exp72.load_cell(ds, base, parent, obj,
                                                  args)
        tr_lab, te_lab = ytr.numpy(), yte.numpy()
        tr_bg_idx = np.where(np.isin(tr_lab, seen))[0]
        rng_split = np.random.default_rng(args.seed)
        perm = rng_split.permutation(tr_bg_idx)
        half_a, half_b = perm[: len(perm) // 2], perm[len(perm) // 2:]

        def embed(bb, X, y):
            loader = DataLoader(TensorDataset(X, y), batch_size=512,
                                shuffle=False)
            e, _ = collect_embeddings(bb, loader)
            return torch.as_tensor(e, dtype=torch.float32, device=DEVICE)

        def regime_fpr(bb, fpr_idx, tag):
            """Null from TEST-set bg; FPR from the given train-bg subset."""
            ztr = embed(bb, Xtr, ytr)
            zte = embed(bb, Xte, yte)
            R = ztr[torch.as_tensor(tr_bg_idx[:4000], device=DEVICE)]
            te_bg = zte[torch.as_tensor(
                np.where(np.isin(te_lab, seen))[0], device=DEVICE)]
            rng = np.random.default_rng(args.seed + 1)
            null_ts, _ = toy_aggregates(te_bg, R, n_null, args.n_d, rng,
                                        sparker_kw, args.seed)
            null_agg = np.array([
                aggregate_pvalues(null_ts[i], np.delete(null_ts, i, axis=0))
                for i in range(n_null)])
            thr = np.quantile(null_agg, 1.0 - args.alpha)
            pool = ztr[torch.as_tensor(fpr_idx, device=DEVICE)]
            _, agg = toy_aggregates(pool, R, n_test, args.n_d, rng,
                                    sparker_kw, args.seed + 7919,
                                    null_ts=null_ts)
            k = int((agg > thr).sum())
            fpr = k / n_test
            lo, hi = clopper_pearson(k, n_test)
            print(f"  [{tag:<9}] FPR@{args.alpha} = {fpr:.3f} "
                  f"[{lo:.3f},{hi:.3f}]  ({k}/{n_test})", flush=True)
            return dict(fpr=fpr, lo=lo, hi=hi, k=k, n=n_test)

        # regime 1: frozen champion space
        results[f"{key}:frozen"] = regime_fpr(bb0, tr_bg_idx, "frozen")

        # regime 2: discovery ft on the FULL corpus, FPR on that corpus
        m = np.isin(tr_lab, seen)
        tr0 = embed(bb0, Xtr, ytr).cpu().numpy()
        means0 = exp28.fill_means(
            exp28.class_centroids(tr0[m], tr_lab[m], seen), seen,
            cfg).detach()
        base_feats = TensorDataset(Xtr, ytr)
        tr_loader = DataLoader(base_feats, batch_size=512, shuffle=False)
        te_loader = DataLoader(TensorDataset(Xte, yte), batch_size=512,
                               shuffle=False)
        bb_full = copy.deepcopy(bb0)
        run_discovery(bb_full, means0.clone(), base_ds=base_feats,
                      train_eval_loader=tr_loader, test_loader=te_loader,
                      seen=seen, holdouts=holdouts, dataset_name=ds,
                      rep_weight=cfg["rep_weight"],
                      sigreg_weight=cfg["sigreg_weight"],
                      n_slices=cfg["n_slices"], rounds=args.rounds,
                      ft_epochs=ft_ep, names=None, seed=args.seed)
        results[f"{key}:ft-full"] = regime_fpr(bb_full, tr_bg_idx,
                                               "ft-full")
        del bb_full
        torch.cuda.empty_cache()

        # regime 3: discovery ft on half A only, FPR toys from half B
        sub_idx = np.sort(np.concatenate(
            [half_a, np.where(np.isin(tr_lab, list(holdouts)))[0]]))
        sub = TensorDataset(Xtr[sub_idx], ytr[sub_idx])
        sub_loader = DataLoader(sub, batch_size=512, shuffle=False)
        bb_split = copy.deepcopy(bb0)
        run_discovery(bb_split, means0.clone(), base_ds=sub,
                      train_eval_loader=sub_loader, test_loader=te_loader,
                      seen=seen, holdouts=holdouts, dataset_name=ds,
                      rep_weight=cfg["rep_weight"],
                      sigreg_weight=cfg["sigreg_weight"],
                      n_slices=cfg["n_slices"], rounds=args.rounds,
                      ft_epochs=ft_ep, names=None, seed=args.seed)
        results[f"{key}:ft-split"] = regime_fpr(bb_split, half_b,
                                                "ft-split")
        del bb_split, bb0
        torch.cuda.empty_cache()

    print(f"\n===== EXP94 SUMMARY (realized FPR at nominal "
          f"alpha={args.alpha}) =====")
    print(f"  {'cell':<18}{'regime':<10}{'FPR':>7}{'CP68':>18}")
    for k, r in results.items():
        cell, reg = k.rsplit(":", 1)
        print(f"  {cell:<18}{reg:<10}{r['fpr']:>7.3f}"
              f"   [{r['lo']:.3f},{r['hi']:.3f}]")

    os.makedirs(os.path.join("logs", "exp94"), exist_ok=True)
    np.savez(os.path.join("logs", "exp94", "results.npz"),
             summary=np.array([repr(results)], dtype=object))
    print("EXP94 DONE.")


if __name__ == "__main__":
    main()
