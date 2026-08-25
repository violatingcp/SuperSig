"""
Experiment 121 (IMPROVEMENT_TESTS.md #121): can a TRANSDUCTIVE criterion see
calibration quality that seen-only criteria cannot?

Exp 115 showed the temperature basin is invisible to a tuner selecting on
seen-class accuracy; exp 120 hardened that to all five seen-only panel
criteria.  Both restricted selection to the seen TRAINING classes.  But a
deployed pipeline does possess the unlabelled evaluation pool -- it simply has
no labels for it.  Criteria computed from (seen train + unlabelled pool) are a
strictly larger and still-legal class.

The key observation that makes this a valid selection rule: the pool is the
SAME set of events in every candidate space.  So a statistic measuring how far
the pool departs from the seen reference is comparable across spaces, and the
space that scores highest is the one that best separates whatever is actually
anomalous -- without ever seeing a label.

SCOPE, stated plainly.  The exp-113 tau sweep archived only scalar metrics --
no embeddings, no checkpoints -- so the tau-basin selection CANNOT be re-tested
analysis-only.  This runs the same question one level more general: across the
six cached CIFAR-10 NPLM spaces (exp 54), does the transductive criterion class
rank spaces by novelty calibration better than the seen-only class does?  That
tests the MECHANISM, not the tau selection.  For the tau test itself, exp 113
must be re-run with --save-embs (added in this commit) on the GPU box.

  seen-only  (exp-120 class): panel rms->1, sw->1, slope->1, ece->0, sep max
  transductive (this exp)    : SparKer t_NP(pool vs seen), MMD(pool, seen),
                               pool LID dispersion, pool tail mass past the
                               seen 95th-percentile distance
  outcome (illegal, scoring only): eucl / mahaT / LID novelty AUC

Prediction: the transductive criteria correlate with the outcome ranking
(Spearman >= 0.6) where the seen-only criteria do not -- making the recipe
deployable after all.
Falsifier: transductive criteria are no better than seen-only -> the basin is
invisible to anything short of labelled novelty, and the tau finding is
permanently a curiosity.

    python experiments/121_transductive_tuner.py
    python experiments/121_transductive_tuner.py --holdout 4 --quick
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import glob
import importlib
import json

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from supersig.config import DEVICE
from supersig.discovery import lid_pool_scores
from supersig.metrics import mahalanobis_novelty
from supersig.sparker import np_test_stats, median_pairwise, mmd2_multi_stats, krr_term

exp104 = importlib.import_module("104_interpretability_panel")


# ----------------------------------------------------------------- outcomes
def outcomes(tr, tr_lab, te, te_lab, seen, holdouts):
    """Novelty AUCs.  ILLEGAL as selection signals -- used only to score."""
    y = np.isin(te_lab, list(holdouts)).astype(int)
    cents = torch.stack([torch.as_tensor(tr[tr_lab == c]).mean(0)
                         for c in seen]).to(DEVICE)
    d = torch.cdist(torch.as_tensor(te).to(DEVICE), cents).min(1).values
    eucl = roc_auc_score(y, d.cpu().numpy())
    tied, _perclass, _eigs = mahalanobis_novelty(tr, tr_lab, te, seen)
    mt = float(roc_auc_score(y, np.asarray(tied)))
    z = torch.as_tensor(np.concatenate([tr[np.isin(tr_lab, seen)], te]))
    is_seen = np.concatenate([np.ones(int(np.isin(tr_lab, seen).sum()), bool),
                              np.zeros(len(te), bool)])
    lid = lid_pool_scores(z, is_seen, k=20).cpu().numpy()[-len(te):]
    return dict(eucl=float(eucl), mahaT=mt, lid=float(roc_auc_score(y, lid)))


# ------------------------------------------------------------ transductive
def transductive(tr, tr_lab, te, seen, seed=0, steps=300, max_ref=8000):
    """Label-free statistics computed from (seen train, unlabelled pool)."""
    ref_np = tr[np.isin(tr_lab, seen)][:max_ref]
    R = torch.as_tensor(ref_np).to(DEVICE)
    D = torch.as_tensor(te).to(DEVICE)

    # 1) SparKer NP statistic: the fitted log density ratio pool-vs-reference.
    t_np = float(np.mean(np_test_stats(D, R, steps=steps, seed=seed)))

    # 2) multi-bandwidth MMD^2 between pool and reference
    s0 = median_pairwise(R, seed=seed)
    sig = [s0 / 2, s0, s0 * 2]
    mmd = float(np.mean(mmd2_multi_stats(D, R, sig, krr_term(R, sig))))

    # 3) dispersion of pool LID (scale-free, label-free)
    z = torch.as_tensor(np.concatenate([ref_np, te]))
    is_seen = np.concatenate([np.ones(len(ref_np), bool),
                              np.zeros(len(te), bool)])
    lid = lid_pool_scores(z, is_seen, k=20, seed=seed).cpu().numpy()[-len(te):]
    lid_disp = float(np.std(lid[np.isfinite(lid)]))

    # 4) pool mass past the seen 95th-percentile nearest-centroid distance
    cents = torch.stack([torch.as_tensor(tr[tr_lab == c]).mean(0)
                         for c in seen]).to(DEVICE)
    d_ref = torch.cdist(R, cents).min(1).values.cpu().numpy()
    d_pool = torch.cdist(D, cents).min(1).values.cpu().numpy()
    tail = float((d_pool > np.quantile(d_ref, 0.95)).mean())
    return dict(t_np=t_np, mmd=mmd, lid_disp=lid_disp, tail_mass=tail)


def tau_axis(args, holdouts):
    """The test exp 121 was written for: does a transductive criterion select
    the temperature basin, where every seen-only criterion (exp 120) fails?"""
    ds, marg = args.cell.rsplit("_", 1)
    pat = os.path.join(args.tau_archive, f"{ds}_tau*_{marg}_s*.npz")
    files = sorted(glob.glob(pat))
    if not files:
        sys.exit(f"no tau embeddings matching {pat}\n"
                 f"run: python experiments/113_tau_generality.py --save-embs")
    by_tau = {}
    for fn in files:
        tau = float(os.path.basename(fn).split("_tau")[1].split("_")[0])
        by_tau.setdefault(tau, []).append(fn)

    rows = {}
    for tau in sorted(by_tau):
        crit, outc = [], []
        for fn in by_tau[tau]:                       # average over seeds
            d = np.load(fn, allow_pickle=True)
            tr, tr_lab, te, te_lab = d["tr"], d["tr_lab"], d["te"], d["te_lab"]
            seen = sorted(set(int(c) for c in np.unique(tr_lab)) - holdouts)
            crit.append(transductive(tr, tr_lab, te, seen, seed=args.seed,
                                     steps=50 if args.quick else args.steps))
            outc.append(outcomes(tr, tr_lab, te, te_lab, seen, holdouts))
        agg = lambda ds_: {k: float(np.mean([x[k] for x in ds_]))
                           for k in ds_[0]}
        rows[tau] = dict(transductive=agg(crit), outcome=agg(outc),
                         n_seeds=len(by_tau[tau]))
        print(f"  tau={tau:<5} " +
              " ".join(f"{k}={v:.4g}" for k, v in rows[tau]["transductive"].items())
              + "  | mahaT=%.3f perevt=%.3f" % (rows[tau]["outcome"]["mahaT"],
                                                rows[tau]["outcome"].get("perevt", float("nan"))
                                                if "perevt" in rows[tau]["outcome"] else float("nan")))

    taus = sorted(rows)
    oracle = max(taus, key=lambda t: rows[t]["outcome"]["mahaT"])
    print(f"\n  ORACLE (illegal, best mahaT): tau={oracle}")
    hits = []
    for crit_name in ("t_np", "mmd", "lid_disp", "tail_mass"):
        pick = max(taus, key=lambda t: rows[t]["transductive"][crit_name])
        ok = pick == oracle
        hits.append(crit_name) if ok else None
        print(f"  transductive '{crit_name:9s}' picks tau={pick:<5}"
              f"{'  <-- FINDS THE ORACLE' if ok else ''}")
    os.makedirs(args.out, exist_ok=True)
    json.dump(rows, open(os.path.join(args.out, f"tau_{args.cell}.json"), "w"),
              indent=1)
    print("\n=== verdict (tau axis, the test exp 121 was written for) ===")
    print("  " + ("transductive criteria reaching the oracle: "
                  + ", ".join(hits) + " -- the basin IS reachable without "
                  "novelty labels, and the tau recipe is deployable."
                  if hits else
                  "NONE.  With exp 120 (five seen-only criteria) this closes "
                  "the question: the basin is invisible to every label-free "
                  "selection rule we can construct, transductive included."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="logs/exp54/embs_*_cifar10.npz")
    ap.add_argument("--tau-archive", default=None,
                    help="logs/exp113/embs -- switches to the TAU-AXIS test: "
                         "rank taus within a cell instead of ranking arms. "
                         "This is the test exp 121 was written for; it needs "
                         "exp 113 re-run with --save-embs.")
    ap.add_argument("--cell", default="cifar100_on",
                    help="tau-axis mode: {dataset}_{on|off}")
    ap.add_argument("--holdout", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--target", default="mahaT",
                    choices=["eucl", "mahaT", "lid"],
                    help="outcome to judge criteria against (exp-120 used mahaT)")
    ap.add_argument("--out", default="logs/exp121")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    holdouts = {args.holdout}
    if args.tau_archive:
        return tau_axis(args, holdouts)
    files = sorted(glob.glob(args.glob))
    if not files:
        sys.exit(f"no cached spaces matching {args.glob}")
    rows = {}
    for fn in files:
        name = os.path.basename(fn).replace("embs_", "").replace(
            "_cifar10.npz", "")
        d = np.load(fn, allow_pickle=True)
        tr, tr_lab = d["tr"], d["tr_lab"]
        te, te_lab = d["te"], d["te_lab"]
        seen = sorted(set(int(c) for c in np.unique(tr_lab)) - holdouts)
        # the space was trained without the holdout, but the cached train bank
        # may still contain it; selection and reference use SEEN rows only.
        print(f"\n----- {name} -----", flush=True)
        pan = exp104.panel(tr[np.isin(tr_lab, seen)][:20000],
                           tr_lab[np.isin(tr_lab, seen)][:20000],
                           sw=True, seed=args.seed)
        tra = transductive(tr, tr_lab, te, seen, seed=args.seed,
                           steps=50 if args.quick else args.steps)
        out = outcomes(tr, tr_lab, te, te_lab, seen, holdouts)
        rows[name] = dict(seen_only={k: pan[k] for k in
                                     ("rms", "sw", "slope", "ece", "sep")},
                          transductive=tra, outcome=out)
        print(f"  seen-only : " + " ".join(f"{k}={pan[k]:.3f}" for k in
                                           ("rms", "sw", "slope", "ece", "sep")))
        print(f"  transduct.: " + " ".join(f"{k}={v:.4g}" for k, v in tra.items()))
        print(f"  OUTCOME   : " + " ".join(f"{k}={v:.3f}" for k, v in out.items()))

    names = list(rows)
    os.makedirs(args.out, exist_ok=True)
    json.dump(rows, open(os.path.join(args.out, "results.json"), "w"), indent=1)

    # rank correlation of each criterion against each outcome
    SEEN_DIR = dict(rms=-1, sw=-1, slope=-1, ece=-1, sep=+1)   # -1 = |v-target|
    TARGET = dict(rms=1.0, sw=1.0, slope=1.0, ece=0.0, sep=None)
    print(f"\n=== Spearman(criterion, outcome) over {len(names)} spaces ===")
    print(f"{'criterion':16s}{'class':>14s}" +
          "".join(f"{o:>13}" for o in ("eucl", "mahaT", "lid")) +
          "     [rho(p)]")
    summary = {}
    for crit in ("rms", "sw", "slope", "ece", "sep"):
        v = np.array([rows[n]["seen_only"][crit] for n in names], float)
        score = -np.abs(v - TARGET[crit]) if TARGET[crit] is not None else v
        sp = [spearmanr(score, [rows[n]["outcome"][o] for n in names])
              for o in ("eucl", "mahaT", "lid")]
        r = [x.correlation for x in sp]; pv = [x.pvalue for x in sp]
        summary[crit] = dict(cls="seen-only", rho=r, p=pv)
        print(f"{crit:16s}{'seen-only':>14s}" +
              "".join(f"{x:>7.2f}({p:.2f})" for x, p in zip(r, pv)))
    for crit in ("t_np", "mmd", "lid_disp", "tail_mass"):
        v = np.array([rows[n]["transductive"][crit] for n in names], float)
        sp = [spearmanr(v, [rows[n]["outcome"][o] for n in names])
              for o in ("eucl", "mahaT", "lid")]
        r = [x.correlation for x in sp]; pv = [x.pvalue for x in sp]
        summary[crit] = dict(cls="transductive", rho=r, p=pv)
        print(f"{crit:16s}{'transductive':>14s}" +
              "".join(f"{x:>7.2f}({p:.2f})" for x, p in zip(r, pv)))

    # NOTE: the three outcomes disagree in SIGN on these spaces (LID AUC
    # anti-correlates with eucl/mahaT -- the on-manifold/off-manifold split
    # appearing across arms within one dataset), so averaging |rho| over them
    # is meaningless.  Judge against ONE target: mahaT, matching the oracle
    # exp 120 used, with the others reported for context.
    OI = {"eucl": 0, "mahaT": 1, "lid": 2}[args.target]
    best_seen = max((abs(s["rho"][OI]), c) for c, s in summary.items()
                    if s["cls"] == "seen-only")
    best_tran = max((abs(s["rho"][OI]), c) for c, s in summary.items()
                    if s["cls"] == "transductive")
    print(f"\ntarget outcome    : {args.target}")
    print(f"best seen-only    : {best_seen[1]} (|rho| {best_seen[0]:.2f})")
    print(f"best transductive : {best_tran[1]} (|rho| {best_tran[0]:.2f})")
    print("\n=== verdict ===")
    if best_tran[0] >= 0.6 and best_tran[0] > best_seen[0]:
        print("  Transductive criteria track novelty calibration where "
              "seen-only criteria do not: the criterion class is viable and "
              "the tau test should be re-run with saved embeddings.")
    elif best_tran[0] > best_seen[0]:
        print("  Transductive criteria are BETTER but below the 0.6 bar -- "
              "suggestive, not sufficient.")
    else:
        print("  Transductive criteria are NO BETTER than seen-only: the "
              "falsifier fires and the tau finding stays a curiosity.")
    print(f"  POWER: n={len(names)} spaces.  At n=6 Spearman needs |rho|>=0.83 "
          "for p<0.05, so only the strongest row is individually significant "
          "and the CLASS comparison is suggestive, not decisive.")
    print("  SCOPE: tested across ARMS on cifar10, not across tau -- the "
          "exp-113 sweep saved no embeddings.  Re-run exp 113 with --save-embs "
          "for the tau test proper.")
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"),
              indent=1, default=float)


if __name__ == "__main__":
    main()
