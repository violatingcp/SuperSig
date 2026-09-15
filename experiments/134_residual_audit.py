"""
Experiment 134: what is the residual construction actually doing?

Three questions, one script.  All three must be answered before the residual
can carry a results section, because the audit of the existing evidence found a
pattern that is suspiciously specific: the residual concat wins the novelty
probe in 30/30 cells, and LOSES `eucl` in 14/15 and `mahaT` in 11/15 against
its own parent.  A construction that helps one currency and hurts the other two
is either a real trade-off or an artefact of how the halves are combined.

-------------------------------------------------------------------------
(A) IS THE GAIN A CONSTRUCTION OR JUST WIDTH?
-------------------------------------------------------------------------
The concat is TWO FULL NETWORKS (`71_residual_ft_grid.py:197-199`, plain
`np.concatenate`), so it doubles both width and parameter count relative to its
parent.  The flagship number, cars/VISReg +0.148 +- 0.004 (3 seeds, exp 75), has
never been compared against a same-width NON-residual control.

Exp 85 already fired a related falsifier: with width matched, "the width-matched
single residual matches/beats the 3-way (+0.051 vs +0.048)" on cars/VISReg --
i.e. on the flagship cell the extra gain was capacity.  That was 3-way vs 2-way;
nobody has run 2-way vs a same-width single head.  A referee will ask.

  arms: parent (1x width) | residual concat (2x) | WIDTH-MATCHED non-residual
        control (2x width, same objective as the parent, no residual)

-------------------------------------------------------------------------
(B) IS THE GEOMETRY LOSS REAL, OR A SCALE ARTEFACT OF THE CONCAT?
-------------------------------------------------------------------------
The halves are concatenated RAW.  Nothing normalises them, and they have no
reason to share a scale: the parent is a softmax-contrastive space (scale-free
by construction) while the child carries a SIGReg marginal that pins unit
variance.  Concatenating at mismatched scale has a sharply asymmetric effect:

  * a LINEAR PROBE is scale-robust -- per-feature weights absorb the mismatch;
  * EUCLIDEAN DISTANCE is not -- the larger half dominates and the smaller
    half's information is simply not in the metric.

Measured on a synthetic where the novelty lives ONLY in the child half:

    space                  probe    eucl
    parent alone          0.9493  0.4983
    child alone           1.0000  1.0000
    concat, child x0.05   0.9999  0.5143   <- probe sees it, distance does not
    concat, child x0.2    1.0000  0.7313
    concat, child x1      1.0000  1.0000
    concat, standardised  1.0000  1.0000

That is exactly the reported pattern: probe wins everywhere, distance currencies
lose.  So the campaign's "the residual only helps the probe" conclusion may be
an artefact of the combination rule rather than a property of the construction.

  arms: raw concat (as archived) | per-half standardised | per-half unit-norm
        | whitened.  Report probe AND eucl/mahaT AND purity for each.

  >>> PARTIAL RESULT 2026-08-27, real exp-54 CIFAR-10 spaces, untrained
  >>> residual r(x) = x - nearest seen centroid (the LINEAR part of the
  >>> construction; no child network was trained).
  >>> logs/exp134/untrained_residual_cifar10.json
  >>>
  >>> CONFIRMED: the RMS ratio predicts whether the combiner matters at all.
  >>>   At ratio 0.945 (nplm_bilinear) every combiner agrees to 3 decimals.
  >>>   At ratio 0.156-0.194 the combiner moves the probe by up to 0.13.
  >>>
  >>> NOT CONFIRMED -- and this retracts the strong form of the hypothesis:
  >>>   standardising does NOT rescue eucl.  It sometimes HURTS it
  >>>   (nplm_dist_sup_cw 0.7871 raw -> 0.7092 standardised) and whitening
  >>>   hurts it everywhere (0.6737 / 0.6034 / 0.4193).  On this evidence the
  >>>   geometry trade-off is REAL, not a scale artefact, and the residual's
  >>>   story stays probe-centric.
  >>>
  >>>   space (ratio)        parent      raw   stand.  whiten   [probe]
  >>>   nplm_sup_dist  .156  0.7894   0.8340   0.8781  0.9151
  >>>   nplm_dist_sup  .194  0.8861   0.9091   0.9202  0.9187
  >>>   nplm_bilinear  .945  0.8551   0.8562   0.8541  0.8532
  >>>
  >>> INCIDENTAL AND WORTH FOLLOWING UP: the UNTRAINED linear residual already
  >>> lifts the probe 0.7894 -> 0.9151.  Some of the residual construction's
  >>> value may not require training the child at all.
  >>>
  >>> STILL OPEN.  The archived 14/15 eucl losses are TRAINED children on
  >>> TRANSFER cells; this proxy is an untrained residual on CIFAR NPLM spaces.
  >>> Also note the trained child is predicted to be LARGER than its parent,
  >>> not smaller: SIGReg(lam=5) drives unit per-dim variance, i.e. RMS =
  >>> sqrt(d), which on these spaces is 1.6x-3.8x the parent's RMS.  So the
  >>> real pairs may sit on the opposite side of ratio 1 from this proxy.
  >>> FIRST THING TO RUN: the child/parent RMS ratio on a real trained pair.
  >>> If it is near 1, question (B) is closed and only (A) and (C) matter.

-------------------------------------------------------------------------
(C) SHOULD THE RESIDUAL COME AFTER DISCOVERY RATHER THAN BEFORE?
-------------------------------------------------------------------------
Every residual in the campaign is built on the PRE-discovery parent, and
discovery is then run on the concat (`28_concat_residual.run_concat_discovery`).
The reverse order has never been tried, and there is a reason to expect it to be
better: the residual is defined against the class centroids, so it removes the
variance the ANCHOR SET explains.  After discovery the anchor set is strictly
richer -- it contains the discovered anchors too -- so the same construction
would be removing more, and the child would carry correspondingly more of what
is genuinely unexplained.

  orders: parent -> child -> concat -> discovery      (archived)
          parent -> discovery -> child -> concat      (NEW)
          parent -> discovery -> child -> concat -> discovery   (both)

Prediction: the new order raises child informativeness where discovery actually
worked (purity above the gate) and changes nothing where it did not -- which
would make it a free improvement in exactly the cells the paper cares about,
and a no-op elsewhere.  A clean falsifier: if it helps where purity was ~0, the
mechanism story is wrong.

-------------------------------------------------------------------------
NOTE ON A DOC ERROR found while auditing: `docs/PAPER_PLAN.md` claims
"universal residual parent 12/12".  It is 10/12 -- ss-ft parents win
flowers/lejepa and dtd/lejepa (`docs/AIRCRAFT_MASTER_TABLE.md:368-372`).  The
12/12 that does hold is against the DISCOVERY PIPELINE, not against the best
known space; dtd/VISReg residual 0.847 still loses to simclr-ft 0.854.

    python experiments/134_residual_audit.py --selftest
    python experiments/134_residual_audit.py --combine-scan --embs parent.npz child.npz
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from supersig.holdouts import n_holdout, run_tag
import argparse
import json

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from supersig.config import DEVICE

COMBINERS = ("raw", "standardize", "unitnorm", "whiten")


# ----------------------------------------------------------------- combining

def combine(parent, child, mode="raw", ref=None):
    """Join two halves under one of the candidate rules.

    `ref` supplies the statistics (fit on train, applied to test) so nothing
    leaks across the split.  Returns (combined, stats).
    """
    P, C = np.asarray(parent, np.float64), np.asarray(child, np.float64)
    st = {}
    if mode == "raw":
        pass
    elif mode == "standardize":
        for nm, X in (("p", P), ("c", C)):
            mu = (ref[nm]["mu"] if ref else X.mean(0, keepdims=True))
            sd = (ref[nm]["sd"] if ref else X.std(0, keepdims=True) + 1e-8)
            st[nm] = dict(mu=mu, sd=sd)
        P = (P - st["p"]["mu"]) / st["p"]["sd"]
        C = (C - st["c"]["mu"]) / st["c"]["sd"]
    elif mode == "unitnorm":
        # equalise the OVERALL scale of each half, preserving its internal shape
        for nm, X in (("p", P), ("c", C)):
            s = (ref[nm]["s"] if ref else float(np.sqrt((X ** 2).sum(1).mean())))
            st[nm] = dict(s=s if s > 0 else 1.0)
        P = P / st["p"]["s"]
        C = C / st["c"]["s"]
    elif mode == "whiten":
        for nm, X in (("p", P), ("c", C)):
            if ref:
                st[nm] = ref[nm]
            else:
                mu = X.mean(0, keepdims=True)
                cov = np.cov((X - mu).T) + 1e-6 * np.eye(X.shape[1])
                w, V = np.linalg.eigh(cov)
                st[nm] = dict(mu=mu, W=V @ np.diag(w ** -0.5) @ V.T)
        P = (P - st["p"]["mu"]) @ st["p"]["W"]
        C = (C - st["c"]["mu"]) @ st["c"]["W"]
    else:
        raise ValueError(mode)
    return np.concatenate([P, C], 1), st


def half_scale_ratio(parent, child):
    """RMS norm of the child half divided by the parent half.

    Far from 1 means the raw concat is effectively one-sided for any
    distance-based metric.  This single number predicts whether (B) matters.
    """
    p = float(np.sqrt((np.asarray(parent, np.float64) ** 2).sum(1).mean()))
    c = float(np.sqrt((np.asarray(child, np.float64) ** 2).sum(1).mean()))
    return c / p if p > 0 else float("nan")


# ------------------------------------------------------------------- metrics

def eucl_auc(tr, tr_lab, te, te_lab, seen, holdouts):
    cent = np.stack([tr[tr_lab == c].mean(0) for c in seen
                     if (tr_lab == c).any()])
    d = np.linalg.norm(te[:, None, :] - cent[None], axis=2).min(1)
    return float(roc_auc_score(np.isin(te_lab, list(holdouts)).astype(int), d))


def probe_auc(tr, tr_lab, te, te_lab, holdouts, epochs=10, lr=1e-2, seed=0):
    torch.manual_seed(seed)
    Xtr = torch.as_tensor(np.ascontiguousarray(tr), dtype=torch.float32,
                          device=DEVICE)
    ytr = torch.as_tensor(np.isin(tr_lab, list(holdouts)).astype(np.int64),
                          device=DEVICE)
    Xte = torch.as_tensor(np.ascontiguousarray(te), dtype=torch.float32,
                          device=DEVICE)
    head = torch.nn.Linear(Xtr.size(1), 2).to(DEVICE)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    for _ in range(epochs):
        perm = torch.randperm(len(Xtr), device=DEVICE)
        for i in range(0, len(Xtr), 512):
            j = perm[i:i + 512]
            loss = torch.nn.functional.cross_entropy(head(Xtr[j]), ytr[j])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        s = torch.softmax(head(Xte), 1)[:, 1].cpu().numpy()
    return float(roc_auc_score(np.isin(te_lab, list(holdouts)).astype(int), s))


def combine_scan(p_tr, c_tr, p_te, c_te, tr_lab, te_lab, seen, holdouts):
    """(B): the same halves under every combination rule."""
    out = {}
    for mode in COMBINERS:
        tr, st = combine(p_tr, c_tr, mode)
        te, _ = combine(p_te, c_te, mode, ref=st)
        out[mode] = dict(
            probe=probe_auc(tr, tr_lab, te, te_lab, holdouts),
            eucl=eucl_auc(tr, tr_lab, te, te_lab, seen, holdouts),
            dim=int(tr.shape[1]))
    return out


# ------------------------------------------------------------------ selftest

def _selftest():
    rng = np.random.default_rng(0)
    C, d, n = 10, 16, 400
    y = np.repeat(np.arange(C), n)
    hold = {9}
    seen = [c for c in range(C) if c not in hold]

    # Parent: the novel class sits ON TOP of a seen class -> invisible here.
    pm = rng.normal(0, 3.0, (C, d)); pm[9] = pm[0]
    P = np.concatenate([pm[c] + rng.normal(0, 1, (n, d)) for c in range(C)])
    # Child: the novel class IS separated -> the information lives here.
    cm = rng.normal(0, 3.0, (C, d))
    Ch = np.concatenate([cm[c] + rng.normal(0, 1, (n, d)) for c in range(C)])

    print("1. THE SCALE ARTEFACT (question B)")
    print("   novelty lives only in the child half; can the concat see it?")
    print(f"   {'child scale':>13s}{'ratio':>8s}{'probe':>9s}{'eucl':>9s}")
    eu = {}
    for s in (0.05, 0.2, 1.0, 5.0):
        tr, _ = combine(P, Ch * s, "raw")
        r = half_scale_ratio(P, Ch * s)
        pr = probe_auc(tr, y, tr, y, hold)
        ec = eucl_auc(tr, y, tr, y, seen, hold)
        eu[s] = ec
        print(f"   {s:>13g}{r:>8.3f}{pr:>9.4f}{ec:>9.4f}")
    # the probe is robust to the mismatch; euclidean distance is not
    assert eu[0.05] < 0.7 and eu[1.0] > 0.9, eu
    print("   -> probe is scale-robust, euclidean distance is NOT")

    print("\n2. THE COMBINERS REPAIR IT (question B)")
    scan = combine_scan(P, Ch * 0.05, P, Ch * 0.05, y, y, seen, hold)
    print(f"   {'mode':>13s}{'probe':>9s}{'eucl':>9s}")
    for m, r in scan.items():
        print(f"   {m:>13s}{r['probe']:>9.4f}{r['eucl']:>9.4f}")
    assert scan["standardize"]["eucl"] > scan["raw"]["eucl"] + 0.2, scan
    print("   -> standardising recovers the geometry the raw concat threw away")

    print("\n3. COMBINERS ARE FIT ON TRAIN AND APPLIED TO TEST (no leakage)")
    tr, st = combine(P, Ch, "standardize")
    te, _ = combine(P + 5.0, Ch, "standardize", ref=st)
    # Applying TRAIN statistics to a shifted test set must leave a large mean.
    # If the stats were (wrongly) recomputed on test, the mean would be ~0.
    shifted_mean = abs(float(te[:, :P.shape[1]].mean()))
    refit_mean = abs(float(combine(P + 5.0, Ch, "standardize")[0]
                           [:, :P.shape[1]].mean()))
    print(f"   train-stats applied: |mean| = {shifted_mean:.3f}   "
          f"(refit on test would give {refit_mean:.3f})")
    assert shifted_mean > 1.0 and refit_mean < 1e-6, (shifted_mean, refit_mean)
    print("   -> test statistics are NOT recomputed on test")

    print("\n4. half_scale_ratio flags a one-sided concat")
    for s in (0.01, 1.0, 100.0):
        print(f"   child x{s:<6g} -> ratio {half_scale_ratio(P, Ch * s):.4f}")
    assert half_scale_ratio(P, Ch * 0.01) < 0.05
    assert half_scale_ratio(P, Ch * 100.0) > 20

    print("\nselftest OK")


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--parent", default="", help="npz with tr/tr_lab/te/te_lab")
    ap.add_argument("--child", default="")
    ap.add_argument("--holdouts", default="")
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--cells", default="",
                    help="comma list ds:base -- run (B) on every TRAINED "
                         "parent/child pair of the cell via "
                         "exp77.transfer_cell (honours SUPERSIG_NH / "
                         "SUPERSIG_HOLDOUT_DRAW)")
    ap.add_argument("--emb-dim", type=int, default=100)
    ap.add_argument("--out", default="logs/exp134")
    args = ap.parse_args()

    if args.selftest:
        _selftest(); return
    if not ((args.parent and args.child) or args.cells):
        ap.error("need --parent and --child, or --cells (or --selftest)")

    # (name, parent(tr,te), child(tr,te), tr_lab, te_lab, dataset)
    pairs = []
    if args.parent:
        dp, dc = (np.load(args.parent, allow_pickle=True),
                  np.load(args.child, allow_pickle=True))
        nm = (os.path.basename(args.parent).replace(".npz", "") + "->" +
              os.path.basename(args.child).replace(".npz", ""))
        pairs.append((nm, (dp["tr"], dp["te"]), (dc["tr"], dc["te"]),
                      np.asarray(dp["tr_lab"]), np.asarray(dp["te_lab"]),
                      args.dataset))
    if args.cells:
        import importlib
        exp77 = importlib.import_module("77_space_similarity")
        a77 = argparse.Namespace(emb_dim=args.emb_dim, dim=args.emb_dim,
                                 arms=None, quick=False)
        for cell in args.cells.split(","):
            ds, base = cell.split(":")
            got = exp77.transfer_cell(a77, ds, base)
            for parent, obj in exp77.CHILDREN_71:
                ck = f"{parent}-{obj}"
                if parent in got and ck in got:
                    Ptr, ytr, Pte, yte = got[parent]
                    Ctr, _, Cte, _ = got[ck]
                    pairs.append((f"{cell}|{parent}->{obj}", (Ptr, Pte),
                                  (Ctr, Cte), ytr, yte, ds))
            print(f"  {cell}: {sum(p[0].startswith(cell) for p in pairs)} "
                  f"trained pairs (tag '{run_tag()}')")

    from supersig.holdouts import holdout_set
    os.makedirs(args.out, exist_ok=True)
    results = {}
    for nm, (Ptr, Pte), (Ctr, Cte), tr_lab, te_lab, ds in pairs:
        n_cls = int(max(tr_lab.max(), te_lab.max())) + 1
        if args.holdouts:
            hold = set(int(x) for x in args.holdouts.split(",") if x != "")
        elif "|" in nm:
            hold = set(holdout_set(ds, n_cls))
        else:
            hold = set(range(n_cls - n_holdout(ds), n_cls))
        seen = [c for c in range(n_cls) if c not in hold]
        ratio = half_scale_ratio(Ptr, Ctr)
        warn = ("   <- ONE-SIDED: the raw concat is effectively the larger half"
                if (ratio < 0.5 or ratio > 2.0) else "")
        print(f"\n=== {nm}  holdouts={sorted(hold)}")
        print(f"child/parent RMS-norm ratio: {ratio:.4f}{warn}")
        scan = combine_scan(Ptr, Ctr, Pte, Cte, tr_lab, te_lab, seen, hold)
        # reference rows: each half alone, so the concat's gain is legible
        scan["parent-only"] = dict(
            probe=probe_auc(Ptr, tr_lab, Pte, te_lab, hold),
            eucl=eucl_auc(Ptr, tr_lab, Pte, te_lab, seen, hold),
            dim=int(Ptr.shape[1]))
        scan["child-only"] = dict(
            probe=probe_auc(Ctr, tr_lab, Cte, te_lab, hold),
            eucl=eucl_auc(Ctr, tr_lab, Cte, te_lab, seen, hold),
            dim=int(Ctr.shape[1]))
        print(f"{'mode':>13s}{'probe':>9s}{'eucl':>9s}{'dim':>6s}")
        for m, r in scan.items():
            print(f"{m:>13s}{r['probe']:>9.4f}{r['eucl']:>9.4f}{r['dim']:>6d}")
        results[nm] = dict(half_scale_ratio=ratio, holdouts=sorted(hold),
                           scan=scan)

    src = (args.cells.replace(":", "-").replace(",", "_") if args.cells
           else "pair")
    out_path = os.path.join(args.out, f"combine_{src}{run_tag()}.json")
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=1, default=float)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
