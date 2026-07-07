"""
The settled default pipeline: supervised SIGReg embedding (per-dataset recipe)
+ iterated open-world anchor discovery with repulsion-exempt anchors.

This is the canonical entry point going forward; the numbered experiments
before it are the historical record that selected these defaults.

    python experiments/26_default_pipeline.py                       # CIFAR-10
    python experiments/26_default_pipeline.py --dataset cifar100 --ks 10,20

Library surface for new tests:
    from supersig.recipes import supervised_embedding, recipe, RECIPES
    from supersig.discovery import run_discovery
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from supersig.config import DATA_DIR
from supersig.data import get_cifar_loaders, _cifar_spec
from supersig.recipes import supervised_embedding, recipe
from supersig.discovery import run_discovery

HOLDOUT_SETS = {
    "cifar10": {1: [4], 2: [4, 9], 3: [0, 4, 9]},
    "cifar100": {1: [4], 3: [4, 30, 70],
                 10: [4, 14, 24, 34, 44, 54, 64, 74, 84, 94],
                 20: [4 + 5 * i for i in range(20)]},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    ap.add_argument("--ks", default=None)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--emb-dim", type=int, default=None)
    ap.add_argument("--sigreg-weight", type=float, default=None)
    args = ap.parse_args()
    ds = args.dataset
    ks = [int(x) for x in (args.ks or ("1,2,3" if ds == "cifar10"
                                       else "10,20")).split(",")]
    cfg = recipe(ds, emb_dim=args.emb_dim, sigreg_weight=args.sigreg_weight)
    if ds == "cifar100":
        from torchvision import datasets as tvd
        names = tvd.CIFAR100(DATA_DIR, train=False, download=True).classes
    else:
        names = ["airplane", "automobile", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck"]

    train_loader, test_loader = get_cifar_loaders(quick=args.quick,
                                                  limit=args.limit, dataset=ds)
    train_eval_loader = DataLoader(train_loader.dataset, batch_size=256,
                                   shuffle=False, num_workers=2)
    cls, plain, _ = _cifar_spec(ds)
    base = cls(DATA_DIR, train=True, download=True, transform=plain)

    print(f"settled recipe [{ds}]: " + "  ".join(
        f"{k}={v}" for k, v in cfg.items()))
    summary = {}
    for k in ks:
        holdouts = set(HOLDOUT_SETS[ds][k])
        seen = [c for c in range(cfg["n_classes"]) if c not in holdouts]
        print(f"\n===== k={k} ({', '.join(names[c] for c in sorted(holdouts))}) =====")
        backbone, means, _ = supervised_embedding(
            ds, holdouts=holdouts, quick=args.quick, limit=args.limit,
            seed=args.seed, emb_dim=args.emb_dim,
            sigreg_weight=args.sigreg_weight)
        if args.quick:
            cfg["ft_epochs"] = 1
        _, history = run_discovery(
            backbone, means, base_ds=base, train_eval_loader=train_eval_loader,
            test_loader=test_loader, seen=seen, holdouts=holdouts,
            dataset_name=ds, rep_weight=cfg["rep_weight"],
            sigreg_weight=cfg["sigreg_weight"], n_slices=cfg["n_slices"],
            rounds=args.rounds, ft_epochs=cfg["ft_epochs"], names=names,
            seed=args.seed)
        summary[k] = history

    print("\n===== DEFAULT-PIPELINE SUMMARY =====")
    for k in ks:
        for h in summary[k]:
            print(f"  k={k} round {h['round']}: purity={h['purity']:.3f} "
                  f"anchors={h['n_anchors']}  margin={h['margin']:.4f}  "
                  f"mean-anchor={h['mean_pc']:.4f}")
    print("Done.")


if __name__ == "__main__":
    main()
