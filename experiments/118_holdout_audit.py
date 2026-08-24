"""
Experiment 118 (IMPROVEMENT_TESTS.md #118): the holdout-selection audit.

The campaign holds out "the last N classes" on every dataset.  On several of
them the class order is ALPHABETICAL, so the holdout set is not random with
respect to semantics -- and exp 111 found that under this rule the number of
scorable superclasses can collapse to 0-15 out of 61-137, meaning some absolute
superclass-agreement figures rest on a handful of classes.

This re-runs the headline probe numbers and the exp-76 interpretability battery
under RANDOM holdout draws, reporting the scorable-class count each time, and
compares against the archived alphabetical numbers.

Prediction: probe numbers are stable (exp 78's holdout-rotation control already
suggests this) but the superclass-agreement figures move materially and should
be republished as random-draw means.
Falsifier: probe numbers also move -> the alphabetical holdout is a confound in
the headline results, not only in the interpretability ones.

    python experiments/118_holdout_audit.py --dataset cifar100 --draws 5
    python experiments/118_holdout_audit.py --quick
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import importlib

import numpy as np
import torch
from torch.utils.data import DataLoader

from supersig.config import DEVICE
from supersig.data import get_cifar_loaders, cifar_two_view_loader
from supersig.models import CIFARResNetBackbone
from supersig.recipes import recipe
from supersig.train import collect_embeddings, train_supcon

exp105 = importlib.import_module("105_width_penalty")
try:
    exp76 = importlib.import_module("76_interpretability")
except Exception:
    exp76 = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--n-holdout", type=int, default=None,
                    help="default: match the archived protocol (1 on CIFAR)")
    ap.add_argument("--dim", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    ds = args.dataset
    cfg = recipe(ds, emb_dim=args.dim) if args.dim else recipe(ds)
    dim = args.dim or cfg["emb_dim"]
    n_cls = cfg["n_classes"]
    n_h = args.n_holdout or 1
    ep = args.epochs or (2 if args.quick else 20)
    sfx = "_quick" if args.quick else ""
    out = os.path.join("logs", "exp118", f"results_{ds}{sfx}.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = {}
    if os.path.exists(out):
        d = np.load(out, allow_pickle=True); done = {k: d[k] for k in d.files}
    rng = np.random.default_rng(args.seed)

    train_loader, test_loader = get_cifar_loaders(quick=args.quick, dataset=ds)
    tel = DataLoader(train_loader.dataset, batch_size=256, shuffle=False,
                     num_workers=2)

    # draw 0 reproduces the archived protocol (the LAST n_h classes);
    # draws 1.. are random, matched in size.
    draws = [tuple(range(n_cls - n_h, n_cls))]
    g = np.random.default_rng(1234 + args.seed)
    while len(draws) < args.draws + 1:
        cand = tuple(sorted(g.choice(n_cls, size=n_h, replace=False).tolist()))
        if cand not in draws:
            draws.append(cand)

    for di, hold in enumerate(draws):
        holdouts = set(hold)
        seen = [c for c in range(n_cls) if c not in holdouts]
        label = "archived(last)" if di == 0 else f"random{di}"
        for si in range(args.seeds):
            key = f"{ds}_d{di}_s{si}"
            if f"{key}_probe" in done:
                continue
            print(f"\n----- {key} [{label}] holdouts={sorted(holdouts)} -----",
                  flush=True)
            torch.manual_seed(args.seed + 20 + si)
            np.random.seed(args.seed + 20 + si)
            net = CIFARResNetBackbone(dim, arch=cfg["arch"],
                                      pretrain=ds).to(DEVICE)
            loader = cifar_two_view_loader(quick=args.quick, labeled=True,
                                           holdout=holdouts, dataset=ds)
            train_supcon(net, loader, ep)
            tr, tr_lab = collect_embeddings(net, tel)
            te, te_lab = collect_embeddings(net, test_loader)
            del net; torch.cuda.empty_cache()
            r = exp105.battery(tr, tr_lab, te, te_lab, seen, holdouts,
                               args.alpha, rng)
            for k, v in r.items():
                done[f"{key}_{k}"] = np.float64(v)
            # scorable-superclass count: how many coarse groups survive with
            # >=2 seen members -- the quantity exp 111 found collapsing
            if exp76 is not None:
                try:
                    _names, sup = exp76.class_names_and_sup(ds)
                    if sup is not None:
                        grp = {}
                        for c in seen:
                            grp.setdefault(sup[c], []).append(c)
                        done[f"{key}_scorable"] = np.int64(
                            sum(1 for v in grp.values() if len(v) >= 2))
                except Exception as e:
                    print(f"    (superclass count unavailable: {e})")
            np.savez(out, **done)
            print(f"  probe={r['probe']:.4f} mahaT={r['mahaT']:.3f}",
                  flush=True)

    print(f"\n{'draw':16s}{'probe mean+-sd':>18s}{'mahaT':>9s}{'scorable':>10s}")
    for di, hold in enumerate(draws):
        label = "archived(last)" if di == 0 else f"random{di}"
        pr = [float(done[f"{ds}_d{di}_s{s}_probe"]) for s in range(args.seeds)
              if f"{ds}_d{di}_s{s}_probe" in done]
        mh = [float(done[f"{ds}_d{di}_s{s}_mahaT"]) for s in range(args.seeds)
              if f"{ds}_d{di}_s{s}_mahaT" in done]
        sc = done.get(f"{ds}_d{di}_s0_scorable")
        if not pr:
            continue
        print(f"{label:16s}{np.mean(pr):>11.4f}+-{np.std(pr, ddof=1) if len(pr)>1 else 0:.4f}"
              f"{np.mean(mh):>9.3f}{('' if sc is None else int(sc)):>10}")
    print("\nFalsifier check: if the random-draw probe mean differs from the "
          "archived draw by more than the exp-117 MDE, the alphabetical "
          "holdout is a confound in the HEADLINE results, not just the "
          "interpretability ones.")


if __name__ == "__main__":
    main()
