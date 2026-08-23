"""
Experiment 101 (IMPROVEMENT_TESTS.md #101): does frozen-space discovery
generalize beyond aircraft?

Exp 86: anchors-only discovery in a fully frozen space keeps 90-99% of
the per-event gain at zero probe cost and stops the round-2 purity
collapse -- tested on aircraft only.  Here: freeze-both discovery on the
exp-71/72 champions of cars/flowers/dtd/galaxy10 x 3 bases plus the
CIFAR-10/100 record concats (exp-76 ckpts), same recipe and seed;
compare against the archived unfrozen exp-72/74 numbers.

Prediction: probe cost ~0 everywhere; per-event retention high on
fine-grained cells, LOWER on coarse ones (galaxy10, CIFAR-10) where the
space update does real work.  Falsifier: frozen discovery zeroes the
per-event GAIN on some cells -> exp 86 is a regime result, not a recipe.

    python experiments/101_frozen_generality.py
    python experiments/101_frozen_generality.py --cells cars:dino --quick
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
from supersig.data import get_cifar_loaders
from supersig.discovery import run_discovery
from supersig.models import CIFARResNetBackbone
from supersig.recipes import recipe
from supersig.train import collect_embeddings

exp28 = importlib.import_module("28_concat_residual")
exp29 = importlib.import_module("29_residual_finetune")
exp30 = importlib.import_module("30_power_curves")
exp44 = importlib.import_module("44_transfer_32d")
exp72 = importlib.import_module("72_residual_discovery")
exp74 = importlib.import_module("74_cifar_residual_discovery")

TRANSFER_CELLS = [f"{d}:{b}" for d in ("cars", "flowers", "dtd",
                                       "galaxy10")
                  for b in ("dino", "lejepa", "visreg")]


def load_cifar_concat(ds):
    dim = 32 if ds == "cifar10" else 100
    obj = "res" if ds == "cifar10" else "resnplm"
    cfg = recipe(ds, emb_dim=dim)
    nets = {}
    for name in ("supcon", obj):
        ck = os.path.join("checkpoints", f"exp76_{ds}_{dim}d_{name}.pt")
        net = CIFARResNetBackbone(dim, arch=cfg["arch"],
                                  pretrain=ds).to(DEVICE)
        net.load_state_dict(torch.load(ck, map_location=DEVICE))
        nets[name] = net
    return exp74.ConcatNets(nets["supcon"], nets[obj]).to(DEVICE), cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=",".join(TRANSFER_CELLS
                                                + ["cifar10", "cifar100"]))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--emb-dim", type=int, default=100)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--n-slices", type=int, default=64)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    results = {}
    for cell in args.cells.split(","):
        key = cell.replace(":", "_")
        if cell.startswith("cifar"):
            ds = cell
            cfg0 = recipe(ds, emb_dim=32 if ds == "cifar10" else 100)
            n_cls = cfg0["n_classes"]
            holdouts = {4}
            bb0, cfg0 = load_cifar_concat(ds)
            ft_ep = 1 if args.quick else cfg0["ft_epochs"]
            train_loader, test_loader = get_cifar_loaders(
                quick=args.quick, dataset=ds)
            base_ds = train_loader.dataset
            tr_loader = DataLoader(base_ds, batch_size=256, shuffle=False,
                                   num_workers=2)
            te_loader = test_loader
            cfg = dict(rep_weight=cfg0["rep_weight"],
                       sigreg_weight=cfg0["sigreg_weight"],
                       n_slices=cfg0["n_slices"], pair_dist=cfg0["pair_dist"],
                       n_classes=n_cls)
            space = f"{'res' if ds == 'cifar10' else 'resnplm'}-cat"
        else:
            ds, base = cell.split(":")
            parent, obj, kind = exp72.WINNERS[(ds, base)]
            n_cls = 47 if ds == "dtd" else exp44.N_CLASSES[ds]
            nh = 1 if ds == "galaxy10" else 10
            holdouts = set(range(n_cls - nh, n_cls))
            ft_ep = 1 if args.quick else 5
            bb0, Xtr, ytr, Xte, yte = exp72.load_cell(ds, base, parent,
                                                      obj, args)
            base_ds = TensorDataset(Xtr, ytr)
            tr_loader = DataLoader(base_ds, batch_size=512, shuffle=False)
            te_loader = DataLoader(TensorDataset(Xte, yte), batch_size=512,
                                   shuffle=False)
            cfg = dict(n_classes=n_cls, pair_dist=5.0, sigreg_weight=1.0,
                       n_slices=args.n_slices,
                       rep_weight=exp72.REP_WEIGHT * 45.0
                       / (n_cls * (n_cls - 1) / 2))
            space = f"{parent}->{obj} {kind}"
        seen = [c for c in range(n_cls) if c not in holdouts]
        print(f"\n######## [{cell}] frozen discovery on {space} ########",
              flush=True)

        tr0, tr_lab = collect_embeddings(bb0, tr_loader)
        te0, te_lab = collect_embeddings(bb0, te_loader)
        torch.manual_seed(1000)
        a_pre, _, _ = exp29.linear_probe_novelty(tr0, tr_lab, te0, te_lab,
                                                 holdouts)
        m = np.isin(tr_lab, seen)
        means0 = exp28.fill_means(
            exp28.class_centroids(tr0[m], tr_lab[m], seen), seen,
            cfg).detach()
        print(f"  pre probe={a_pre:.4f}", flush=True)

        bb = copy.deepcopy(bb0)
        for p in bb.parameters():
            p.requires_grad_(False)
        cur_means, hist = run_discovery(
            bb, means0.clone(), base_ds=base_ds,
            train_eval_loader=tr_loader, test_loader=te_loader, seen=seen,
            holdouts=holdouts, dataset_name=ds,
            rep_weight=cfg["rep_weight"],
            sigreg_weight=cfg["sigreg_weight"], n_slices=cfg["n_slices"],
            rounds=args.rounds, ft_epochs=ft_ep, names=None,
            seed=args.seed)
        # space frozen: probe post == pre by construction; per-event from
        # the discovered anchors
        zt = torch.as_tensor(te0, dtype=torch.float32, device=DEVICE)
        d_seen = torch.cdist(zt, cur_means[seen].to(DEVICE)).min(1).values
        d_disc = torch.cdist(zt, cur_means[n_cls:].to(DEVICE)) \
            .min(1).values if cur_means.size(0) > n_cls else d_seen * 0
        s = (d_seen - d_disc).cpu().numpy()
        bgm = np.isin(te_lab, seen)
        sgm = np.isin(te_lab, list(holdouts))
        pe = exp30.power_at_alpha(s[bgm], s[sgm], args.alpha)
        results[cell] = dict(probe=a_pre, perevt=pe,
                             purity=[h["purity"] for h in hist],
                             margin=[h["margin"] for h in hist])
        print(f"  [{cell}] probe={a_pre:.4f} (frozen)  perevt={pe:.3f}  "
              f"purity " + " ".join(f"r{h['round']}={h['purity']:.3f}"
                                    for h in hist) +
              f"  margin r2={hist[-1]['margin']:.4f}", flush=True)
        del bb, bb0
        torch.cuda.empty_cache()

    # archived unfrozen comparisons
    arch = {}
    try:
        d = np.load("logs/exp72/residual_discovery.npz", allow_pickle=True)
        for k in d.files:
            if k.endswith("_probe_post") or k.endswith("_probe_pre"):
                arch[k] = float(d[k])
    except Exception:
        pass
    print("\n===== EXP101 SUMMARY (frozen discovery; archived unfrozen "
          "probe pre->post in parens) =====")
    print(f"  {'cell':<14}{'probe':>8}{'perevt':>8}{'pur r1/r2':>12}"
          f"{'unfrozen (arch)':>20}")
    for cell, r in results.items():
        key = cell.replace(":", "_")
        pur = r["purity"] + [float("nan")] * (2 - len(r["purity"]))
        a = (f"{arch.get(key + '_probe_pre', float('nan')):.3f}->"
             f"{arch.get(key + '_probe_post', float('nan')):.3f}"
             if key + "_probe_post" in arch else "--")
        print(f"  {cell:<14}{r['probe']:>8.4f}{r['perevt']:>8.3f}"
              f"{pur[0]:>6.3f}/{pur[1]:.3f}{a:>20}")

    os.makedirs(os.path.join("logs", "exp101"), exist_ok=True)
    np.savez(os.path.join("logs", "exp101", "results.npz"),
             summary=np.array([repr(results)], dtype=object))
    print("EXP101 DONE.")


if __name__ == "__main__":
    main()
