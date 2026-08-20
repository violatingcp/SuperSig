"""
Experiment 80: close the SparKer coverage gap — the pre-discovery SparKer
power battery for every campaign space that never had one.

Audit (2026-08-20): exp-70 ran SparKer for the e2e arms on cars/flowers/
dtd/galaxy10 but the aircraft cells were run with --skip-power; exps 71/73
never ran power batteries, so the residual children and the record concat
spaces have no SparKer numbers anywhere (exp-72/74 cover only the
post-discovery winners); exp-50 covers the CIFAR suite arms.  Missing and
loadable from cache (no retraining):

  aircraft x {dino,lejepa,visreg} : supcon-ft, ss-ft + 3 children + 3 concats
  {cars,flowers,dtd,galaxy10} x 3 : 3 children + 3 concats (exp-71 spaces)
  cifar10 32d / cifar100 100d     : supcon parent, res, resnplm + 2 concats

Protocol = exp-70 pre-discovery battery exactly (annealed-sigma SparKer,
M=16, steps=300, n_null=200, 50 signal toys, alpha=0.05; R = seen-train
embeddings [:20000]; transfer fractions 0.003-0.1, n_d 1000/2000; CIFAR
fractions per exp-74, n_d 5000, holdout 4).  Embeddings via the exp-77
loaders (cached banks + heads / exp-76 ckpts).  Resumable: one npz per
cell, rewritten after every space; existing spaces are skipped.

    python experiments/80_sparker_all_spaces.py
    python experiments/80_sparker_all_spaces.py --cells aircraft:dino --quick
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import importlib
import numpy as np
import torch

from supersig.config import DEVICE

exp31 = importlib.import_module("31_sparker_power")
exp44 = importlib.import_module("44_transfer_32d")
exp77 = importlib.import_module("77_space_similarity")

TRANSFER_CELLS = [f"{ds}:{b}" for ds in
                  ("aircraft", "cars", "flowers", "dtd", "galaxy10")
                  for b in ("dino", "lejepa", "visreg")]
CIFAR_CELLS = ["cifar10", "cifar100"]
E2E_ARMS = set(exp77.ARMS_70)


def cell_spaces(cell, args):
    """(spaces dict, seen, holdouts, fractions, n_d) for one cell."""
    if cell.startswith("cifar"):
        ds = cell
        args.dim = 32 if ds == "cifar10" else 100
        args.arms = []                       # residual program spaces only
        spaces = exp77.cifar_cell(args, ds)
        n_cls = 10 if ds == "cifar10" else 100
        holdouts = {4}
        fracs = ([0.001, 0.003, 0.01, 0.02, 0.03, 0.1] if ds == "cifar10"
                 else [0.001, 0.003, 0.01, 0.02, 0.05])
        n_d = 5000
    else:
        ds, base = cell.split(":")
        spaces = exp77.transfer_cell(args, ds, base)
        spaces.pop("frozen", None)
        if ds != "aircraft":                 # e2e arms already have SparKer
            spaces = {k: v for k, v in spaces.items() if k not in E2E_ARMS}
        n_cls = 47 if ds == "dtd" else exp44.N_CLASSES[ds]
        nh = 1 if ds == "galaxy10" else 10
        holdouts = set(range(n_cls - nh, n_cls))
        fracs = [0.003, 0.01, 0.02, 0.05, 0.1]
        n_d = 2000 if ds in ("cars", "galaxy10") else 1000
    seen = [c for c in range(n_cls) if c not in holdouts]
    return spaces, seen, holdouts, fracs, n_d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=",".join(TRANSFER_CELLS + CIFAR_CELLS))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--kernels", type=int, default=16)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--emb-dim", type=int, default=100)
    ap.add_argument("--refresh", action="store_true",
                    help="recompute spaces already in the cell npz")
    args = ap.parse_args()
    n_null = 20 if args.quick else 200
    n_sig_toys = 10 if args.quick else 50
    sparker_kw = dict(M=args.kernels,
                      steps=50 if args.quick else args.steps)  # annealed
    os.makedirs(os.path.join("logs", "exp80"), exist_ok=True)

    for cell in args.cells.split(","):
        tag = cell.replace(":", "_")
        out_path = os.path.join("logs", "exp80", f"results_{tag}.npz")
        done = {}
        if os.path.exists(out_path) and not args.refresh:
            d = np.load(out_path, allow_pickle=True)
            done = {k: d[k] for k in d.files}
        spaces, seen, holdouts, fracs, n_d = cell_spaces(cell, args)
        todo = [s for s in spaces if f"sparker_{s}_pre" not in done]
        print(f"\n######## [{cell}] {len(todo)} spaces to run "
              f"({len(spaces) - len(todo)} cached): {todo} ########",
              flush=True)
        for s in todo:
            Xtr, ytr, Xte, yte = spaces[s]
            Xtr = np.asarray(Xtr, dtype=np.float32)
            Xte = np.asarray(Xte, dtype=np.float32)
            R = torch.as_tensor(Xtr[np.isin(ytr, seen)][:20000],
                                device=DEVICE)
            bg = torch.as_tensor(Xte[np.isin(yte, seen)], device=DEVICE)
            sg = torch.as_tensor(Xte[np.isin(yte, list(holdouts))],
                                 device=DEVICE)
            print(f"  [{cell}/{s}] dim={Xtr.shape[1]} R={len(R)} "
                  f"bg={len(bg)} sig={len(sg)}", flush=True)
            powers, _ = exp31.run_test_battery(
                bg, sg, R, fracs, n_d, n_null, n_sig_toys, args.alpha,
                args.seed, sparker_kw, tag=f"{tag}:{s}")
            done[f"sparker_{s}_pre"] = np.array(powers)
            done["fractions"] = np.array(fracs)
            np.savez(out_path, **done)           # incremental save
            print(f"  [{cell}/{s}] sparker " +
                  " ".join(f"{f}={p:.3f}" for f, p in zip(fracs, powers)),
                  flush=True)
        del spaces
        torch.cuda.empty_cache()

    print("\n===== EXP80 SUMMARY (SparKer pre-discovery, new spaces) =====")
    for cell in args.cells.split(","):
        tag = cell.replace(":", "_")
        p = os.path.join("logs", "exp80", f"results_{tag}.npz")
        if not os.path.exists(p):
            continue
        d = np.load(p, allow_pickle=True)
        fr = d["fractions"]
        for k in sorted(d.files):
            if k.startswith("sparker_"):
                s = k[len("sparker_"):-len("_pre")]
                row = " ".join(f"{f}:{v:.2f}" for f, v in zip(fr, d[k]))
                print(f"  {tag:<18}{s:<26}{row}")
    print("EXP80 DONE.")


if __name__ == "__main__":
    main()
