"""
Experiment 113 (IMPROVEMENT_TESTS.md #113): is the tau x marginal basin
general, or a many-class phenomenon?

Exp 110 established on C100 that only tau=1.0 AND the SIGReg marginal together
break the mahaT ceiling (each alone stays under it), and that the effect is
present already at 32-D.  Exp 105 reached numerically the same place by an
independent route (a width penalty) and noted that C10 shows the OPPOSITE sign.
If the basin is general it is a recipe change for the whole program; if it is
many-class only it is a regime rule.  This decides whether every softmax row in
the paper needs re-running.

Grid: tau in {0.05, 0.1, 0.3, 1.0, 3.0} x {marginal on, off} x 3 paired seeds,
on c10 (10 classes) and c100 (100 classes).  Full battery + exp-104 panel.

Prediction: many-class only -- the basin appears on c100 and inverts on c10,
i.e. it tracks the CLASS-COUNT axis, not the on-manifold axis.
Falsifier: the basin appears on c10 too -> tau=0.1 was simply wrong throughout
and every softmax row in the paper is under-tuned.

    python experiments/113_tau_generality.py
    python experiments/113_tau_generality.py --datasets cifar10 --taus 1.0 --quick
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

exp34h = importlib.import_module("34h_hybrid_nplm_cifar")
exp105 = importlib.import_module("105_width_penalty")

TAUS = [0.05, 0.1, 0.3, 1.0, 3.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["cifar10", "cifar100"])
    ap.add_argument("--taus", nargs="+", type=float, default=TAUS)
    ap.add_argument("--marginal", nargs="+", default=["on", "off"])
    ap.add_argument("--dim", type=int, default=None,
                    help="default: the recipe dim for each dataset")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tag", default="",
                    help="npz suffix so concurrent invocations don't clobber")
    args = ap.parse_args()

    holdouts = {args.holdout}
    ep = args.epochs or (2 if args.quick else 20)
    sfx = ("_" + args.tag if args.tag else "") + \
        ("_quick" if args.quick else "")
    out = os.path.join("logs", "exp113", f"results{sfx}.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = {}
    if os.path.exists(out):
        d = np.load(out, allow_pickle=True)
        done = {k: d[k] for k in d.files}
    rng = np.random.default_rng(args.seed)

    for ds in args.datasets:
        cfg = recipe(ds, emb_dim=args.dim) if args.dim else recipe(ds)
        dim = args.dim or cfg["emb_dim"]
        seen = [c for c in range(cfg["n_classes"]) if c not in holdouts]
        train_loader, test_loader = get_cifar_loaders(quick=args.quick,
                                                      dataset=ds)
        tel = DataLoader(train_loader.dataset, batch_size=256, shuffle=False,
                         num_workers=2)
        for tau in args.taus:
            for marg in args.marginal:
                for si in range(args.seeds):
                    key = f"{ds}_tau{tau}_{marg}_s{si}"
                    if f"{key}_probe" in done:
                        print(f"  [{key}] cached", flush=True); continue
                    print(f"\n----- {key} ({ep} ep, {dim}d) -----", flush=True)
                    torch.manual_seed(args.seed + 20 + si)
                    np.random.seed(args.seed + 20 + si)
                    net = CIFARResNetBackbone(dim, arch=cfg["arch"],
                                              pretrain=ds).to(DEVICE)
                    loader = cifar_two_view_loader(quick=args.quick,
                                                   labeled=True,
                                                   holdout=holdouts,
                                                   dataset=ds)
                    if marg == "off":
                        # plain SupCon at this temperature (no marginal)
                        train_supcon(net, loader, ep, temp=tau)
                    else:
                        spec = dict(positives="supervised", critic="cosine",
                                    estimator="softmax", marginal="sigreg",
                                    tau=tau)
                        exp34h.train_hybrid(net, loader, ep, spec, True,
                                            lam=args.lam,
                                            n_slices=cfg["n_slices"])
                    tr, tr_lab = collect_embeddings(net, tel)
                    te, te_lab = collect_embeddings(net, test_loader)
                    del net; torch.cuda.empty_cache()
                    r = exp105.battery(tr, tr_lab, te, te_lab, seen, holdouts,
                                       args.alpha, rng)
                    for k, v in r.items():
                        done[f"{key}_{k}"] = np.float64(v)
                    np.savez(out, **done)
                    print(f"  [{key}] probe={r['probe']:.4f} "
                          f"mahaT={r['mahaT']:.4f} perevt={r['perevt']:.3f}",
                          flush=True)

    # verdict: does the basin (mahaT above the 0.47-0.49 band) appear per dataset?
    print("\n=== basin check (mean mahaT over seeds) ===")
    for ds in args.datasets:
        print(f"  {ds}:")
        for tau in args.taus:
            row = []
            for marg in args.marginal:
                v = [done.get(f"{ds}_tau{tau}_{marg}_s{s}_mahaT")
                     for s in range(args.seeds)]
                v = [float(x) for x in v if x is not None]
                row.append(f"{marg}={np.mean(v):.3f}" if v else f"{marg}=--")
            print(f"    tau={tau:<5} " + "  ".join(row))


if __name__ == "__main__":
    main()
