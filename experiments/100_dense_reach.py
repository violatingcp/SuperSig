"""
Experiment 100 (IMPROVEMENT_TESTS.md #100): dense fraction scan — turn f95
from a bound into a measurement.

Exp 99: of 106 reach numbers only four are cleanly bracketed; the coarse
fraction grid, not the spaces, is the limiting factor.  Dense scan
f in {0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10} on the CHAMPION space of
each of the 17 cells (exp-72 winner concats; CIFAR record concats, which
also get the finer {0.005, 0.01, 0.015} since c10 brackets at 0.019),
annealed sigma, 50 toys, cached embeddings, no retraining.  f95 with
Clopper-Pearson bands via the exp-99 inverter.

Prediction: every starred cell resolves to a crossing bracketed within
<=0.02; flowers stays >0.1 on all bases.  Falsifier: curves shallow or
non-monotone in [0.02, 0.10] -> reach is the wrong summary, keep
power-at-fixed-f.

    python experiments/100_dense_reach.py
    python experiments/100_dense_reach.py --cells flowers:dino --quick
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
exp72 = importlib.import_module("72_residual_discovery")
exp77 = importlib.import_module("77_space_similarity")
exp99 = importlib.import_module("99_discovery_reach")

TRANSFER_CELLS = [f"{d}:{b}" for d in ("aircraft", "cars", "flowers",
                                       "dtd", "galaxy10")
                  for b in ("dino", "lejepa", "visreg")]
FRACS = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]
FRACS_FINE = [0.005, 0.01, 0.015] + FRACS


def champion_space(ds, base, emb_dim=100):
    parent, obj, kind = exp72.WINNERS[(ds, base)]
    c_tr = exp77.head_emb(ds, base, f"{parent}_{obj}", emb_dim, "train")
    c_te = exp77.head_emb(ds, base, f"{parent}_{obj}", emb_dim, "test")
    if not (c_tr and c_te):
        return None
    if kind == "residual":
        return c_tr[0], c_tr[1], c_te[0], c_te[1]
    p_tr = exp77.head_emb(ds, base, parent, emb_dim, "train")
    p_te = exp77.head_emb(ds, base, parent, emb_dim, "test")
    return (np.concatenate([p_tr[0], c_tr[0]], 1), p_tr[1],
            np.concatenate([p_te[0], c_te[0]], 1), p_te[1])


def cifar_space(ds):
    ns = argparse.Namespace(dim=32 if ds == "cifar10" else 100, arms=[],
                            quick=False)
    cat = "res-cat" if ds == "cifar10" else "resnplm-cat"
    return exp77.cifar_cell(ns, ds).get(cat), cat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=",".join(TRANSFER_CELLS
                                                + ["cifar10", "cifar100"]))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    n_null = 20 if args.quick else 200
    n_toys = 10 if args.quick else 50
    sparker_kw = dict(M=16, steps=50 if args.quick else 300)
    os.makedirs(os.path.join("logs", "exp100"), exist_ok=True)
    out_path = os.path.join("logs", "exp100", "results.npz")
    done = {}
    if os.path.exists(out_path) and not args.refresh:
        d = np.load(out_path, allow_pickle=True)
        done = {k: d[k] for k in d.files}

    for cell in args.cells.split(","):
        key = cell.replace(":", "_")
        if f"sparker_{key}" in done:
            print(f"  [{cell}] cached, skipping", flush=True)
            continue
        if cell.startswith("cifar"):
            sp, name = cifar_space(cell)
            n_cls = 10 if cell == "cifar10" else 100
            holdouts, n_d = {4}, 5000
            fracs = FRACS_FINE if cell == "cifar10" else FRACS
        else:
            ds, base = cell.split(":")
            sp = champion_space(ds, base)
            name = "champion"
            n_cls = 47 if ds == "dtd" else exp44.N_CLASSES[ds]
            nh = 1 if ds == "galaxy10" else 10
            holdouts = set(range(n_cls - nh, n_cls))
            n_d = 2000 if ds in ("cars", "galaxy10") else 1000
            fracs = FRACS
        if sp is None:
            print(f"  !! [{cell}] missing banks, skipping")
            continue
        Xtr, ytr, Xte, yte = sp
        seen = [c for c in range(n_cls) if c not in holdouts]
        Xtr = np.asarray(Xtr, np.float32)
        Xte = np.asarray(Xte, np.float32)
        R = torch.as_tensor(Xtr[np.isin(ytr, seen)][:20000], device=DEVICE)
        bg = torch.as_tensor(Xte[np.isin(yte, seen)], device=DEVICE)
        sg = torch.as_tensor(Xte[np.isin(yte, list(holdouts))],
                             device=DEVICE)
        print(f"\n######## [{cell}] {name} dense reach ########", flush=True)
        powers, _ = exp31.run_test_battery(bg, sg, R, fracs, n_d, n_null,
                                           n_toys, args.alpha, args.seed,
                                           dict(sparker_kw), tag=key)
        done[f"sparker_{key}"] = np.array(powers)
        done[f"fractions_{key}"] = np.array(fracs)
        np.savez(out_path, **done)
        val, flag = exp99.f95(fracs, powers)
        f_opt, f_pes = exp99.band(fracs, powers)
        print(f"  [{cell}] f95={exp99.fmt(val, flag, max(fracs))} "
              f"band=[{f_opt if f_opt else '--'},"
              f"{f_pes if f_pes else '--'}]  powers=" +
              "/".join(f"{p:.2f}" for p in powers), flush=True)

    print("\n===== EXP100 SUMMARY (dense f95 per champion) =====")
    print(f"  {'cell':<18}{'f95':>10}{'flag':>14}{'band':>22}")
    rows = []
    for cell in args.cells.split(","):
        key = cell.replace(":", "_")
        if f"sparker_{key}" not in done:
            continue
        fr = list(done[f"fractions_{key}"])
        pw = list(done[f"sparker_{key}"])
        val, flag = exp99.f95(fr, pw)
        f_opt, f_pes = exp99.band(fr, pw)
        rows.append((cell, val, flag))
        bs = (f"[{f_opt:.3f},{f_pes:.3f}]"
              if f_opt is not None and f_pes is not None else "--")
        print(f"  {cell:<18}{exp99.fmt(val, flag, max(fr)):>10}"
              f"{flag:>14}{bs:>22}")
    n_ok = sum(1 for _, v, f in rows if v is not None and f == "ok")
    print(f"\n  {n_ok}/{len(rows)} cleanly bracketed (exp-99 baseline: "
          f"4/106 over all spaces)")
    print("EXP100 DONE.")


if __name__ == "__main__":
    main()
