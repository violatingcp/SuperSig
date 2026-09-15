"""
Experiment 86 (IMPROVEMENT_TESTS.md #86): fix aircraft discovery by freezing
the probe directions.

Exp 72: discovery on the aircraft residual-concat champions buys geometry
but costs probe (-0.012 to -0.037) at fine purity (0.32-0.53); exp 79 ruled
out the pool scorer as the lever.  The paper's diagnosis: the discovery ft
erodes the res-nplm concat's probe directions.  The diagnosis implies its
own fix: do not let it.

Three variants per aircraft cell (dino/lejepa/visreg winner concats,
identical recipe/seed to exp 72):
  unfrozen      exp-72 baseline, rerun here for exact pairing
  freeze-parent parent half frozen, residual half + anchors fine-tune
  freeze-both   both halves frozen, ONLY the anchors train
                (train_sigreg_hybrid always learns means)

Report pre -> post probe / eucl / mahaT / lid and the simple per-event
power (d_seen - d_disc at alpha=0.05), plus per-round pool purity.
Prediction: freeze-parent holds the probe at the pre-discovery record
while keeping the geometry/per-event gains.  Falsifier: the geometry gains
disappear along with the probe loss (the trade is not separable).

    python experiments/86_frozen_parent_discovery.py
    python experiments/86_frozen_parent_discovery.py --cells aircraft:dino --quick
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
from supersig.discovery import run_discovery
from supersig.train import collect_embeddings

exp28 = importlib.import_module("28_concat_residual")
exp29 = importlib.import_module("29_residual_finetune")
exp30 = importlib.import_module("30_power_curves")
exp44 = importlib.import_module("44_transfer_32d")
exp72 = importlib.import_module("72_residual_discovery")

VARIANTS = ["unfrozen", "freeze-parent", "freeze-both"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="aircraft:dino,aircraft:lejepa,"
                                       "aircraft:visreg")
    ap.add_argument("--variants", nargs="+", default=VARIANTS,
                    choices=VARIANTS)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--emb-dim", type=int, default=100)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--n-slices", type=int, default=64)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    ft_ep = 1 if args.quick else 5

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
        print(f"\n######## [{key}] frozen-variant discovery on "
              f"{parent}->{obj} {kind} ########", flush=True)
        bb0, Xtr, ytr, Xte, yte = exp72.load_cell(ds, base, parent, obj,
                                                  args)
        tr_lab, te_lab = ytr.numpy(), yte.numpy()
        base_feats = TensorDataset(Xtr, ytr)
        tr_loader = DataLoader(base_feats, batch_size=512, shuffle=False)
        te_loader = DataLoader(TensorDataset(Xte, yte), batch_size=512,
                               shuffle=False)

        def battery(tr, te, means=None):
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
            out = dict(probe=float(np.mean(aucs)), eucl=r["eucl"],
                       mahaT=r["maha_tied"], lid=r["lid"])
            if means is not None and means.size(0) > N_CLS:
                zt = torch.as_tensor(te, device=DEVICE)
                d_seen = torch.cdist(zt, means[seen]).min(1).values
                d_disc = torch.cdist(zt, means[N_CLS:]).min(1).values
                s = (d_seen - d_disc).cpu().numpy()
                bg = np.isin(te_lab, seen)
                sg = np.isin(te_lab, list(holdouts))
                out["perevt"] = exp30.power_at_alpha(s[bg], s[sg],
                                                     args.alpha)
            return out

        tr0, _ = collect_embeddings(bb0, tr_loader)
        te0, _ = collect_embeddings(bb0, te_loader)
        pre = battery(tr0, te0)
        m = np.isin(tr_lab, seen)
        means0 = exp28.fill_means(
            exp28.class_centroids(tr0[m], tr_lab[m], seen), seen,
            cfg).detach()
        print(f"  pre: probe={pre['probe']:.4f} eucl={pre['eucl']:.4f} "
              f"mahaT={pre['mahaT']:.4f} lid={pre['lid']:.4f}", flush=True)

        for var in args.variants:
            bb = copy.deepcopy(bb0)
            if var in ("freeze-parent", "freeze-both"):
                for p in bb.p.parameters():
                    p.requires_grad_(False)
            if var == "freeze-both":
                for p in bb.c.parameters():
                    p.requires_grad_(False)
            cur_means, hist = run_discovery(
                bb, means0.clone(), base_ds=base_feats,
                train_eval_loader=tr_loader, test_loader=te_loader,
                seen=seen, holdouts=holdouts, dataset_name=ds,
                rep_weight=cfg["rep_weight"],
                sigreg_weight=cfg["sigreg_weight"],
                n_slices=cfg["n_slices"], rounds=args.rounds,
                ft_epochs=ft_ep, names=None, seed=args.seed)
            trp, _ = collect_embeddings(bb, tr_loader)
            tep, _ = collect_embeddings(bb, te_loader)
            post = battery(trp, tep, means=cur_means)
            results[f"{key}:{var}"] = dict(
                pre=pre, post=post,
                purity=[h["purity"] for h in hist],
                margin=[h["margin"] for h in hist])
            print(f"  [{var:<13}] probe {pre['probe']:.4f}->"
                  f"{post['probe']:.4f}  eucl {pre['eucl']:.4f}->"
                  f"{post['eucl']:.4f}  mahaT {pre['mahaT']:.4f}->"
                  f"{post['mahaT']:.4f}  lid {pre['lid']:.4f}->"
                  f"{post['lid']:.4f}  perevt={post.get('perevt', 0):.3f}  "
                  f"purity " + " ".join(f"r{i+1}={p:.3f}" for i, p in
                                        enumerate(results[f'{key}:{var}']
                                                  ['purity'])),
                  flush=True)
            del bb
            torch.cuda.empty_cache()
        del bb0
        torch.cuda.empty_cache()

    print("\n===== EXP86 SUMMARY (frozen-parent aircraft discovery) =====")
    print(f"  {'cell':<18}{'variant':<15}{'probe pre->post':>18}"
          f"{'mahaT post':>11}{'perevt':>8}{'pur r2':>8}")
    for k, r in results.items():
        cell, var = k.rsplit(":", 1)
        pur = r["purity"] + [float("nan")] * (2 - len(r["purity"]))
        print(f"  {cell:<18}{var:<15}"
              f"{r['pre']['probe']:>9.4f}->{r['post']['probe']:.4f}"
              f"{r['post']['mahaT']:>11.4f}"
              f"{r['post'].get('perevt', float('nan')):>8.3f}{pur[1]:>8.3f}")

    os.makedirs(os.path.join("logs", "exp86"), exist_ok=True)
    np.savez(os.path.join("logs", "exp86", "results.npz"),
             summary=np.array([repr(results)], dtype=object))
    print("EXP86 DONE.")


if __name__ == "__main__":
    main()
