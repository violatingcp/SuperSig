"""
Experiment 115 (IMPROVEMENT_TESTS.md #115): does the probe/calibration
dissociation survive per-arm tuning?

The uncomfortable one.  Every comparative claim in the paper compares arms AT
THEIR INHERITED DEFAULTS, and exp 110 showed at least one arm was badly served
by its default (tau=0.1 vs the 1.0 that broke the C100 ceiling).  If each arm
is given its own best tau, does the dissociation survive?

Two stages.
  STAGE 1 (tuning): for each arm, scan tau on a VALIDATION split built from
  SEEN classes only -- no holdout-class access -- so the tuning is open-world
  legal.  Selection criterion is deliberately NEUTRAL between the two
  currencies (`--select`): `probe_seen` uses a seen-class-only proxy probe,
  `acc` uses nearest-centroid accuracy.  Neither looks at novelty.
  STAGE 2 (comparison): re-run the dissociation table at each arm's tuned tau
  and report it beside the archived default-tau table.

Prediction: the dissociation NARROWS but survives -- softmax arms gain
per-event from ~0 to 0.05-0.10 while NPLM arms keep a clear per-event edge, and
the probe ordering is unchanged.
Falsifier: the dissociation disappears -> the paper's central empirical claim
was substantially a statement about defaults and must be rewritten as such.

    python experiments/115_tuning_fairness.py --dataset cifar100
    python experiments/115_tuning_fairness.py --quick --taus 0.1 1.0
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
from supersig.train import collect_embeddings, train_supcon, train_linear_probe

exp34h = importlib.import_module("34h_hybrid_nplm_cifar")
exp105 = importlib.import_module("105_width_penalty")

# the arms whose comparison constitutes the dissociation table
SPECS = {
    "supcon":        None,                       # plain SupCon, temp = tau
    "supcon_sigreg": dict(positives="supervised", critic="cosine",
                          estimator="softmax", marginal="sigreg"),
    "nplm_sup_dist": dict(positives="supervised", critic="distance",
                          estimator="nplm", marginal="sigreg"),
    "nplm_bilinear": dict(positives="instance", critic="bilinear",
                          estimator="nplm", marginal="sigreg"),
}
TAUS = [0.05, 0.1, 0.3, 1.0, 3.0]


def _train(arm, net, loader, ep, tau, lam, n_slices):
    if arm == "supcon":
        train_supcon(net, loader, ep, temp=tau)
    else:
        spec = dict(SPECS[arm]); spec["tau"] = tau
        exp34h.train_hybrid(net, loader, ep, spec, spec["positives"] == "supervised",
                            lam=lam, n_slices=n_slices)


def seen_only_score(tr, tr_lab, te, te_lab, seen, how):
    """Selection criterion that never sees a holdout class."""
    import torch as T
    m_tr = np.isin(tr_lab, seen); m_te = np.isin(te_lab, seen)
    X, y = tr[m_tr], tr_lab[m_tr]
    Xte, yte = te[m_te], te_lab[m_te]
    if how == "acc":
        cents = T.stack([T.as_tensor(X[y == c]).mean(0) for c in seen])
        d = T.cdist(T.as_tensor(Xte), cents)
        pred = np.asarray(seen)[d.argmin(1).cpu().numpy()]
        return float((pred == yte).mean())
    # seen-class linear probe accuracy (a proxy for readability, not novelty)
    remap = {c: i for i, c in enumerate(seen)}
    acc = train_linear_probe(X, np.array([remap[int(v)] for v in y]),
                             Xte, np.array([remap[int(v)] for v in yte]),
                             n_classes=len(seen))
    return float(acc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--arms", nargs="+", default=list(SPECS))
    ap.add_argument("--taus", nargs="+", type=float, default=TAUS)
    ap.add_argument("--select", default="acc", choices=["acc", "probe_seen"])
    ap.add_argument("--dim", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    ds = args.dataset
    holdouts = {args.holdout}
    cfg = recipe(ds, emb_dim=args.dim) if args.dim else recipe(ds)
    dim = args.dim or cfg["emb_dim"]
    seen = [c for c in range(cfg["n_classes"]) if c not in holdouts]
    ep = args.epochs or (2 if args.quick else 20)
    sfx = "_quick" if args.quick else ""
    out = os.path.join("logs", "exp115", f"results_{ds}{sfx}.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = {}
    if os.path.exists(out):
        d = np.load(out, allow_pickle=True); done = {k: d[k] for k in d.files}
    rng = np.random.default_rng(args.seed)

    train_loader, test_loader = get_cifar_loaders(quick=args.quick, dataset=ds)
    tel = DataLoader(train_loader.dataset, batch_size=256, shuffle=False,
                     num_workers=2)

    def run(arm, tau, si):
        key = f"{arm}_tau{tau}_s{si}"
        if f"{key}_probe" in done:
            return key
        torch.manual_seed(args.seed + 20 + si); np.random.seed(args.seed + 20 + si)
        net = CIFARResNetBackbone(dim, arch=cfg["arch"], pretrain=ds).to(DEVICE)
        loader = cifar_two_view_loader(quick=args.quick, labeled=True,
                                       holdout=holdouts, dataset=ds)
        _train(arm, net, loader, ep, tau, args.lam, cfg["n_slices"])
        tr, tr_lab = collect_embeddings(net, tel)
        te, te_lab = collect_embeddings(net, test_loader)
        del net; torch.cuda.empty_cache()
        r = exp105.battery(tr, tr_lab, te, te_lab, seen, holdouts,
                           args.alpha, rng)
        r["select"] = seen_only_score(tr, tr_lab, te, te_lab, seen, args.select)
        for k, v in r.items():
            done[f"{key}_{k}"] = np.float64(v)
        np.savez(out, **done)
        print(f"  [{key}] select={r['select']:.4f} probe={r['probe']:.4f} "
              f"perevt={r['perevt']:.3f}", flush=True)
        return key

    # ---- stage 1: tune tau per arm on the seen-only criterion, seed 0
    print("=== stage 1: per-arm tuning (seen classes only) ===", flush=True)
    best = {}
    for arm in args.arms:
        scores = {}
        for tau in args.taus:
            k = run(arm, tau, 0)
            scores[tau] = float(done[f"{k}_select"])
        best[arm] = max(scores, key=scores.get)
        print(f"  {arm}: tuned tau={best[arm]}  "
              + " ".join(f"{t}:{s:.3f}" for t, s in scores.items()), flush=True)

    # ---- stage 2: full paired comparison at tuned tau vs the default
    print("\n=== stage 2: dissociation at tuned vs default tau ===", flush=True)
    DEFAULT = {"supcon": 0.1, "supcon_sigreg": 0.1,
               "nplm_sup_dist": 1.0, "nplm_bilinear": 1.0}
    for arm in args.arms:
        for tau in {DEFAULT[arm], best[arm]}:
            for si in range(args.seeds):
                run(arm, tau, si)

    print(f"\n{'arm':16s}{'tau':>6s}{'probe':>9s}{'mahaT':>8s}{'perevt':>8s}")
    for arm in args.arms:
        for label, tau in (("default", DEFAULT[arm]), ("tuned", best[arm])):
            g = lambda m: np.mean([float(done[f"{arm}_tau{tau}_s{s}_{m}"])
                                   for s in range(args.seeds)
                                   if f"{arm}_tau{tau}_s{s}_{m}" in done] or [np.nan])
            print(f"{arm+' ('+label+')':16s}{tau:>6}{g('probe'):>9.4f}"
                  f"{g('mahaT'):>8.3f}{g('perevt'):>8.3f}")
    print("\nFalsifier check: if the softmax arms' perevt is now comparable to "
          "the NPLM arms', the dissociation was a statement about defaults.")


if __name__ == "__main__":
    main()
