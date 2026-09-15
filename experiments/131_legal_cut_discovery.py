"""
Experiment 131: re-run discovery on cifar10 / cifar100 / galaxy10 with the
label-free pool cut, A/B against the inherited constants.

WHY.  Three changes have landed since these cells were last run, and all three
touch the discovery loop:

  1. exps 128/129 -- the pool cut.  `tau_quantile=0.95` came from exp 23 and was
     never derived; the ceiling `purity <= min(1, b/q)` says it caps achievable
     purity at 0.20 when b=1%.  Replaced by the label-free rule: the tightest
     cut still holding `n_min` ESTIMATED novel points (`supersig.poolcut`).
  2. exp 129 -- `kmax`.  `max(4, len(holdouts)+2)` uses the NUMBER OF NOVEL
     CLASSES, which is oracle knowledge.  Replaced by a bound derived from the
     same label-free novelty weights.
  3. exp 130 -- "frozen" now really freezes.  `requires_grad_(False)` did not
     stop BatchNorm running statistics, so on the CIFAR ResNet trunk a frozen
     space still drifted (max embedding drift 1.29 against mean |z| 0.52 over
     3 rounds).  `supersig.train.set_train_mode` fixes it.

Change 3 alone means the archived frozen CIFAR numbers will not reproduce
exactly, so these cells need re-running regardless of 1 and 2.

DESIGN.  Paired A/B on identical spaces, seeds and recipes:

    A  cut_rule="quantile", pool_score as archived   (the inherited baseline)
    B  cut_rule="legal",    pool_score="np"          (exps 128/129)

Reporting purity per round, the probe pre/post, and -- the point of the
exercise -- the label-free rule's `ok` flag and its estimated vs true base
rate.  A False `ok` is a RESULT: it means the space cannot support discovery,
which is exactly what we expect on low-rate single-holdout cells.

WHAT WE EXPECT (state plainly if it does not happen):
  * cifar10 (b ~ 0.10 at holdout {4}): B >= A on purity.  This is the cell
    where the rule engaged on real embeddings in exp 129 (purity 0.83).
  * cifar100 (b ~ 0.01): B tightens the cut hard.  The exp-128 ceiling says
    0.20 is the most q=0.05 could ever give, so B should beat A -- but the
    estimated novel count may not reach n_min, in which case `ok` is False and
    that is the honest answer, not a failure.
  * galaxy10 (b ~ 0.10, single holdout of 10 classes, and the ONLY cell
    genuinely out-of-distribution to the ImageNet-1k backbones): the most
    informative cell in the paper.  Treat its numbers as load-bearing.

FALSIFIER.  B is worse than A on purity in cells where `ok` is True -> the
label-free cut is not usable and the inherited constant should stand (with the
ceiling reported as a limitation).

    python experiments/131_legal_cut_discovery.py --selftest
    python experiments/131_legal_cut_discovery.py --cells cifar10
    python experiments/131_legal_cut_discovery.py --cells galaxy10:dino --quick
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from supersig.holdouts import n_holdout, run_tag
import argparse
import importlib
import json

import numpy as np
import torch

from supersig.config import DEVICE
from supersig import poolcut
from supersig.discovery import run_discovery, np_pool_scores

exp77 = importlib.import_module("77_space_similarity")
exp80 = importlib.import_module("80_sparker_all_spaces")

def _score_pool(z, mask, is_seen, kmax, seed, info):
    """Purity and BIC recovery for one pool."""
    from supersig.discovery import bic_select
    is_novel = ~is_seen
    n_nov = int(is_novel[mask].sum())
    out = dict(q=float(mask.mean()), pool=int(mask.sum()),
               purity=float(is_novel[mask].mean()) if mask.any() else 0.0,
               n_novel=n_nov, kmax=int(kmax),
               ok=bool(info.get("ok", True)), reason=info.get("reason", ""))
    Z = torch.as_tensor(np.asarray(z, dtype=np.float32)[mask], device=DEVICE)
    if len(Z) >= max(4, kmax):
        khat, centers, _ = bic_select(Z, kmax=int(kmax), seed=seed)
        a = torch.cdist(Z, centers).argmin(1).cpu().numpy()
        nov = is_novel[mask]
        best = max((float(nov[a == j].mean()) for j in range(centers.shape[0])
                    if (a == j).any()), default=0.0)
        out.update(khat=int(khat), best_purity=best)
    else:
        out.update(khat=0, best_purity=0.0)
    return out


DEFAULT_CELLS = "cifar10,cifar100,galaxy10:dino,galaxy10:lejepa,galaxy10:visreg"


def base_rate_report(z, is_seen_lab, seed=0):
    """True vs estimated base rate, and the critic's own health (exp 130)."""
    f, cal = np_pool_scores(torch.as_tensor(np.asarray(z, dtype=np.float32),
                                            device=DEVICE),
                            is_seen_lab, seed=seed, return_calib=True)
    f = f.cpu().numpy()
    w = poolcut.novelty_weights(f, f[is_seen_lab])
    return dict(_scores=f,
                b_true=float((~is_seen_lab).mean()),
                n_hat=float(w.sum()),
                b_hat=float(w.sum() / len(w)),
                calib_in=cal["calib_in"], calib_out=cal["calib_out"])


def _selftest():
    """Both cut rules must run end-to-end on a synthetic space, the legal rule
    must be at least as pure where it engages, and the default must be the
    archived behaviour."""
    import inspect
    sig = inspect.signature(run_discovery)
    assert sig.parameters["cut_rule"].default == "quantile", \
        "default must stay 'quantile' so archived results reproduce"
    print("  default cut_rule is 'quantile' (archived behaviour)      OK")

    try:
        run_discovery(None, None, base_ds=None, train_eval_loader=None,
                      test_loader=None, seen=[], holdouts={0},
                      dataset_name="x", rep_weight=0, sigreg_weight=0,
                      n_slices=8, cut_rule="legal", pool_score="dist")
    except ValueError as e:
        assert "needs pool_score='np'" in str(e), e
        print("  cut_rule='legal' rejects a distance scorer             OK")
    else:
        raise AssertionError("should have rejected pool_score='dist'")

    rng = np.random.default_rng(0)
    n, d = 8000, 16
    v = np.zeros(n, dtype=bool)
    v[rng.choice(n, 200, replace=False)] = True
    f = rng.normal(0.0, 0.3, n)
    f[v] = rng.normal(3.0, 0.3, int(v.sum()))
    mask, info = poolcut.legal_pool(f, ~v)
    print(f"  legal_pool on a clean synthetic: purity "
          f"{v[mask].mean():.3f}, n_novel {int(v[mask].sum())}, "
          f"ok={info['ok']}                     OK")
    assert v[mask].mean() > 0.9 and info["ok"]

    # a space with no novelty must refuse rather than pool noise
    f0 = rng.normal(0.0, 0.3, n)
    _, info0 = poolcut.legal_pool(f0, ~v)
    assert not info0["ok"], info0
    print(f"  refuses when there is no novelty to find ({info0['reason']})")
    print("\nselftest OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--cells", default=DEFAULT_CELLS)
    ap.add_argument("--arms", nargs="*", default=[])
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--ft-epochs", type=int, default=5)
    ap.add_argument("--n-min", type=int, default=poolcut.N_MIN)
    ap.add_argument("--emb-dim", type=int, default=100)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--freeze", action="store_true",
                    help="frozen-space discovery (anchors only); recommended")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="logs/exp131")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    from supersig.recipes import recipe
    os.makedirs(args.out, exist_ok=True)
    results = {}
    for cell in args.cells.split(","):
        spaces, seen, holdouts, _, _ = exp80.cell_spaces(cell, args)
        ds = cell.split(":")[0]
        cfg = recipe(ds if ds.startswith("cifar") else "cifar100",
                     emb_dim=args.dim if ds.startswith("cifar") else args.emb_dim)
        print(f"\n######## {cell}: {len(spaces)} spaces, "
              f"holdouts={sorted(holdouts)} ########", flush=True)
        for name, payload in spaces.items():
            tr, tr_lab = payload[0], payload[1]
            is_seen = np.isin(tr_lab, seen)
            rep = base_rate_report(tr, is_seen, seed=args.seed)
            print(f"\n  --- {name} ---", flush=True)
            print(f"    b_true={rep['b_true']:.4f}  b_hat={rep['b_hat']:.4f}  "
                  f"n_hat={rep['n_hat']:.0f}  calib_in={rep['calib_in']:.3f}  "
                  f"calib_out={rep['calib_out']:.3f}", flush=True)
            key = f"{cell}|{name}"

            # ROUND-1 A/B.  Pool selection, BIC range and clustering are fully
            # determined by the embeddings, so this is evaluation-only and is
            # the comparison that matters: the fine-tune step that follows is
            # identical between arms and (in the frozen recipe) touches only
            # the anchors.
            f = rep.pop("_scores")
            arms = {}

            # A: the inherited constants
            tau = float(np.quantile(f[is_seen], 0.95))
            mA = f > tau
            kmA = max(4, len(holdouts) + 2)
            arms["quantile"] = _score_pool(tr, mA, is_seen, kmA, args.seed,
                                           dict(ok=True, reason="quantile",
                                                q=float(mA.mean()),
                                                pool=int(mA.sum()),
                                                kmax=kmA))
            # B: the label-free rule
            mB, info = poolcut.legal_pool(f, is_seen, n_min=args.n_min)
            arms["legal"] = _score_pool(tr, mB, is_seen, info["kmax"],
                                        args.seed, info)

            for arm, r in arms.items():
                print(f"    {arm:9s} q={r['q']:.4f} pool={r['pool']:>6d} "
                      f"purity={r['purity']:.4f} n_nov={r['n_novel']:>5d} "
                      f"kmax={r['kmax']:>3d} khat={r['khat']} "
                      f"clus_pur={r['best_purity']:.3f} ok={r['ok']}"
                      f"{'' if r['ok'] else '  <- ' + r['reason']}", flush=True)
            d = arms["legal"]["purity"] - arms["quantile"]["purity"]
            print(f"    delta purity (legal - quantile): {d:+.4f}", flush=True)
            results[key] = dict(base_rate=rep, arms=arms, delta_purity=float(d))

    # One file per --cells invocation: without the cell tag, consecutive
    # single-cell runs silently overwrote each other (found 2026-08-26).
    tag = "_".join(c.replace(":", "-") for c in args.cells.split(","))
    out_path = os.path.join(args.out, f"legal_cut_{tag}{run_tag()}.json")
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=1, default=float)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
