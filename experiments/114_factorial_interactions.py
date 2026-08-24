"""
Experiment 114 (IMPROVEMENT_TESTS.md #114): a factorial, not a scan -- which
knobs INTERACT?

Exp 110's effect was invisible to every one-at-a-time scan the campaign ran,
because it lives entirely in a tau x marginal interaction.  We have never run a
factorial design over the continuous knobs, so we do not know which others
interact.

Design: a 2^(5-1) RESOLUTION IV fractional factorial (16 runs per cell) over
  A tau           {0.1, 1.0}
  B lam           {0.3, 3.0}      (marginal weight)
  C n_slices      {64, 256}
  D emb dim       {32, 100}
  E ft epochs     {10, 20}
with generator E = ABCD.  Resolution IV means main effects are NOT aliased with
two-factor interactions (they are aliased with three-factor ones), which is
exactly what is needed to read interactions honestly -- and it is why a
resolution III design would be useless here.

Aliasing that remains, and which we report rather than hide: two-factor
interactions are aliased in PAIRS (AB=CE, AC=BE, AD=... etc. under E=ABCD), so
a large estimated AB is "AB or its alias".  Disambiguating needs the fold-over
(`--foldover`), which adds 16 more runs.

Prediction: tau x marginal is the largest interaction; most main effects are
small; n_slices is inert (it is a Monte-Carlo budget, not a modelling choice).
Falsifier: several large interactions -> the space is not approximately
separable in its knobs and single-arm comparisons are unsafe as a methodology.

    python experiments/114_factorial_interactions.py --cells cifar100
    python experiments/114_factorial_interactions.py --quick --foldover
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import importlib
import itertools

import numpy as np
import torch
from torch.utils.data import DataLoader

from supersig.config import DEVICE
from supersig.data import get_cifar_loaders, cifar_two_view_loader
from supersig.models import CIFARResNetBackbone
from supersig.recipes import recipe
from supersig.train import collect_embeddings

exp34h = importlib.import_module("34h_hybrid_nplm_cifar")
exp105 = importlib.import_module("105_width_penalty")

FACTORS = ["tau", "lam", "n_slices", "dim", "epochs"]
LEVELS = {"tau": (0.1, 1.0), "lam": (0.3, 3.0), "n_slices": (64, 256),
          "dim": (32, 100), "epochs": (10, 20)}


def design(foldover=False):
    """2^(5-1) resolution-IV design, E = ABCD, in +-1 coding."""
    rows = []
    for a, b, c, d in itertools.product((-1, 1), repeat=4):
        rows.append([a, b, c, d, a * b * c * d])
    if foldover:
        rows += [[-v for v in r] for r in rows]
    return np.array(rows, float)


def effects(X, y):
    """Main effects and all two-factor interaction estimates (+ alias note)."""
    out = {}
    n, k = X.shape
    for i in range(k):
        out[FACTORS[i]] = float(2.0 * (X[:, i] * y).mean())
    for i, j in itertools.combinations(range(k), 2):
        col = X[:, i] * X[:, j]
        out[f"{FACTORS[i]}x{FACTORS[j]}"] = float(2.0 * (col * y).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", default=["cifar100", "cifar10"])
    ap.add_argument("--metrics", nargs="+",
                    default=["probe", "mahaT", "perevt"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--foldover", action="store_true",
                    help="add the mirror design to de-alias 2-factor terms")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tag", default="",
                    help="npz suffix so concurrent invocations don't clobber")
    args = ap.parse_args()

    D = design(args.foldover)
    holdouts = {args.holdout}
    sfx = ("_" + args.tag if args.tag else "") + \
        ("_fold" if args.foldover else "") + \
        ("_quick" if args.quick else "")
    out = os.path.join("logs", "exp114", f"results{sfx}.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = {}
    if os.path.exists(out):
        d = np.load(out, allow_pickle=True); done = {k: d[k] for k in d.files}
    rng = np.random.default_rng(args.seed)

    for ds in args.cells:
        base = recipe(ds)
        train_loader, test_loader = get_cifar_loaders(quick=args.quick,
                                                      dataset=ds)
        tel = DataLoader(train_loader.dataset, batch_size=256, shuffle=False,
                         num_workers=2)
        for ri, row in enumerate(D):
            settings = {f: LEVELS[f][0 if v < 0 else 1]
                        for f, v in zip(FACTORS, row)}
            cfg = recipe(ds, emb_dim=settings["dim"])
            seen = [c for c in range(cfg["n_classes"]) if c not in holdouts]
            ep = 2 if args.quick else settings["epochs"]
            for si in range(args.seeds):
                key = f"{ds}_r{ri}_s{si}"
                if f"{key}_probe" in done:
                    continue
                print(f"\n----- {key}: {settings} -----", flush=True)
                torch.manual_seed(args.seed + 20 + si)
                np.random.seed(args.seed + 20 + si)
                net = CIFARResNetBackbone(settings["dim"], arch=cfg["arch"],
                                          pretrain=ds).to(DEVICE)
                loader = cifar_two_view_loader(quick=args.quick, labeled=True,
                                               holdout=holdouts, dataset=ds)
                spec = dict(positives="supervised", critic="cosine",
                            estimator="softmax", marginal="sigreg",
                            tau=settings["tau"])
                exp34h.train_hybrid(net, loader, ep, spec, True,
                                    lam=settings["lam"],
                                    n_slices=settings["n_slices"])
                tr, tr_lab = collect_embeddings(net, tel)
                te, te_lab = collect_embeddings(net, test_loader)
                del net; torch.cuda.empty_cache()
                r = exp105.battery(tr, tr_lab, te, te_lab, seen, holdouts,
                                   args.alpha, rng)
                for k, v in r.items():
                    done[f"{key}_{k}"] = np.float64(v)
                np.savez(out, **done)

        print(f"\n=== {ds}: effect estimates (change in metric per factor "
              f"low->high) ===")
        for metric in args.metrics:
            y = []
            for ri in range(len(D)):
                vals = [float(done[f"{ds}_r{ri}_s{s}_{metric}"])
                        for s in range(args.seeds)
                        if f"{ds}_r{ri}_s{s}_{metric}" in done]
                y.append(np.mean(vals) if vals else np.nan)
            y = np.asarray(y)
            if np.isnan(y).any():
                print(f"  {metric}: incomplete, skipping"); continue
            e = effects(D, y)
            main_e = {k: v for k, v in e.items() if "x" not in k}
            inter = {k: v for k, v in e.items() if "x" in k}
            print(f"  -- {metric} --")
            print("     main:  " + "  ".join(
                f"{k}={v:+.4f}" for k, v in sorted(main_e.items(),
                                                   key=lambda kv: -abs(kv[1]))))
            top = sorted(inter.items(), key=lambda kv: -abs(kv[1]))[:4]
            print("     inter: " + "  ".join(f"{k}={v:+.4f}" for k, v in top))
            if not args.foldover:
                print("     (2-factor terms are alias PAIRS under E=ABCD; "
                      "rerun with --foldover to de-alias)")


if __name__ == "__main__":
    main()
