"""Experiment 134d: the discovery x construction 2x2, and its run plan.

Exp 134c established that building the residual AFTER discovery beats building
it before.  It did not establish WHY that matters, because it never formed the
interaction.  The interaction is the paper's capability claim:

    per-event power, dtd/dino, construction `res`
        parent, no discovery   A = 0.040     (alpha = 0.05: nominal, no ability)
        parent, + discovery    B = 0.068
        residual, no discovery C = 0.023     (BELOW the parent -- it hurts)
        residual, + discovery  D = 0.253     (6x nominal)

    main effect of discovery      +0.028
    main effect of construction   -0.018
    INTERACTION                   +0.203

Neither ingredient does anything alone.  Composed, they take detection of an
individual novel sample from chance to 6x nominal.  That is an ability, not a
delta, and it is only available to a method that HAS a discovery stage to
compose after -- which is the novelty.

This script does three things:

  --aggregate   form the 2x2 from the 134c JSONs + the exp-71 npz files and
                rank the interactions.  Pure CPU re-aggregation.  It also
                REPAIRS the two galaxy10 cells whose archived rows are `{}`
                in the JSON: 134c snapshotted the exp-71 npz before the
                galaxy10 h1 draw sweep re-wrote it, so the pre-discovery
                children exist on disk and were never missing.

  --plan        emit the GPU run plan that turns the finding from n=1 into a
                headline (seeds on the interaction; the two absent 134c
                cells; the 2x2 completed on the cells that lack a pre child).

  --selftest    CPU checks on the algebra and the repair path.

    python experiments/134d_2x2_ablation.py --aggregate
    python experiments/134d_2x2_ablation.py --plan
    python experiments/134d_2x2_ablation.py --selftest

READ THE RESULT THIS WAY.  A large `total` with a small `interaction` is an
additive stack -- worth reporting but not novel.  A large `interaction` is the
capability claim.  Per-event is the metric that matters most here because it
is bounded below by alpha: A near 0.05 means the parent could not flag a
single novel event, so a positive interaction is a threshold crossing rather
than an improvement.
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from supersig.ablation2x2 import (ALPHA, METRICS, archived_rows, cell_2x2,
                                  cell_name, load_cell, r1_purity)

# (dataset, base) for each 134c JSON stem.
CELLS = {
    "cars_visreg": ("cars", "visreg"),
    "dtd_dino": ("dtd", "dino"),
    "galaxy10_dino": ("galaxy10", "dino"),
    "galaxy10_lejepa": ("galaxy10", "lejepa"),
    "galaxy10_visreg": ("galaxy10", "visreg"),
}


def aggregate(args):
    paths = sorted(glob.glob(os.path.join(args.logdir, "postdisc_*_ft134c.json")))
    if not paths:
        print(f"no 134c JSONs under {args.logdir}")
        return
    repaired, missing, rows = [], [], []
    for p in paths:
        name = cell_name(p)
        j = load_cell(p)
        ds, base = CELLS.get(name, (None, None))
        if ds is None:
            print(f"  [skip] unknown cell {name}")
            continue
        pre, npz = archived_rows(ds, base, parent=args.parent, tag=args.tag)
        stale = j.get("archived_exp71", {})
        n_stale = sum(1 for k, v in stale.items() if not v)
        n_now = sum(1 for k, v in pre.items() if v)
        if n_stale and n_now > sum(1 for k, v in stale.items() if v):
            repaired.append((name, n_stale, os.path.basename(npz)))
        if not pre:
            missing.append((name, npz))
        for r in cell_2x2(j, pre, parent=args.parent):
            r.update(cell=name, purity=r1_purity(j))
            rows.append(r)

    if repaired:
        print("REPAIRED (134c snapshotted exp-71 before it was re-run; "
              "no GPU work needed):")
        for name, n, f in repaired:
            print(f"  {name:<18} {n} empty archived rows -> re-read from {f}")
        print()
    if missing:
        print("MISSING exp-71 npz (cannot form 2x2):")
        for name, f in missing:
            print(f"  {name:<18} {f}")
        print()

    for m in (args.metric,) if args.metric else METRICS:
        sub = [r for r in rows if r["metric"] == m]
        done = [r for r in sub if not r["missing"]]
        gone = [r for r in sub if r["missing"]]
        print(f"===== {m}  (2x2; alpha={ALPHA} for perevt) =====")
        print(f"  {'cell':<17}{'ctor':<10}{'kind':<10}{'r1pur':>7}"
              f"{'A':>8}{'B':>8}{'C':>8}{'D':>8}"
              f"{'disc':>8}{'ctor':>8}{'INTER':>8}{'total':>8}")
        for r in sorted(done, key=lambda r: -r["interaction"]):
            flag = " *" if (m == "perevt" and r["A"] <= ALPHA * 1.5
                            and r["D"] > 4 * ALPHA) else ""
            print(f"  {r['cell']:<17}{r['obj']:<10}{r['kind']:<10}"
                  f"{r['purity']:>7.3f}{r['A']:>8.3f}{r['B']:>8.3f}"
                  f"{r['C']:>8.3f}{r['D']:>8.3f}{r['main_disc']:>+8.3f}"
                  f"{r['main_ctor']:>+8.3f}{r['interaction']:>+8.3f}"
                  f"{r['total']:>+8.3f}{flag}")
        for r in gone:
            print(f"  {r['cell']:<17}{r['obj']:<10}{r['kind']:<10}"
                  f"{'':>7}{'-- no pre-discovery child --':>40}")
        if m == "perevt" and done:
            star = [r for r in done if r["A"] <= ALPHA * 1.5 and r["D"] > 4 * ALPHA]
            print(f"\n  * = threshold crossing: parent at or below nominal "
                  f"({ALPHA}) -> composed above 4x nominal.  {len(star)}/{len(done)} rows.")
            sup = [r for r in done if r["interaction"] > 0]
            print(f"  superadditive (interaction > 0): {len(sup)}/{len(done)}; "
                  f"max {max(r['interaction'] for r in done):+.3f} "
                  f"({max(done, key=lambda r: r['interaction'])['cell']}"
                  f"/{max(done, key=lambda r: r['interaction'])['obj']})")
            print("  CAVEAT: every row is a single seed.  No error bar on any "
                  "interaction.  See --plan.")
        print()


PLAN = """\
GPU RUN PLAN -- turning the interaction from n=1 into a headline
================================================================
Ordered by what each buys per GPU-hour.  Items 1 and 2 are the ones that
decide whether this can be the paper's main claim.

--- 0. FREE (CPU, already done by --aggregate) ------------------------------
The galaxy10 dino/lejepa archived rows were never missing: exp 134c
snapshotted the exp-71 npz before the galaxy10 h1 draw sweep re-wrote it.
Re-aggregating recovers them.  No GPU time.  This was previously scoped as a
re-run; it is not.

--- 1. SEEDS ON THE INTERACTION  (decides the headline) ---------------------
The 0.023 -> 0.253 per-event interaction on dtd/dino `res` is ONE seed.  The
2x2 needs seeds on the D cell (134c) and the C cell (exp 71) together, since
the interaction is a difference of differences and inherits both variances.

  for S in 1 2; do
    python experiments/71_residual_suite.py  --dataset dtd --base dino \\
        --parent supcon-ft --objs res --seed $S
    python experiments/134c_residual_after_discovery.py --dataset dtd \\
        --base dino --objs res --seed $S
  done

  Cost: 4 child fine-tunes (~13-40 min each on the ViT) + 2 feature-space
  discoveries.  ~2-4 GPU-hours.
  DECIDES: if the interaction sd is comparable to its 0.203 size, this is a
  single-seed artefact and the thesis reverts to the metrics framing.

--- 2. THE TWO ABSENT 134c CELLS  (independent replication) -----------------
134c has run logs for only 3 of 5 cells.  galaxy10/dino and galaxy10/lejepa
have JSONs but no logs; cars/flowers/aircraft were never attempted.  The
cheapest independent test of the interaction is a cell where the parent is
also at nominal per-event.

  python experiments/134c_residual_after_discovery.py --dataset flowers --base dino
  python experiments/134c_residual_after_discovery.py --dataset aircraft --base dino

  Cost: ~2 cells x 2 objectives x one child fine-tune.  ~2-3 GPU-hours.
  DECIDES: whether the crossing is dtd-specific.

--- 3. COMPLETE THE 2x2 WHERE THE PRE CHILD IS ABSENT -----------------------
Any row --aggregate prints as "-- no pre-discovery child --" has D but no C,
so no interaction.  Fill C by running exp 71 for that (cell, objective) at the
same holdout tag; no 134c re-run is needed.

  python experiments/71_residual_suite.py --dataset <ds> --base <base> \\
      --parent supcon-ft --objs <obj>

--- WHAT WOULD FALSIFY THE CLAIM --------------------------------------------
* interaction sd >= its magnitude under item 1  -> single-seed artefact.
* the interaction appears where round-1 purity is ~0 -> the mechanism story
  (a richer, CORRECT anchor set is what makes the residual informative) is
  wrong, and something else (simply more fine-tuning) is doing the work.
  galaxy10/lejepa is the case to watch: r2 purity is exactly 0.000 with
  n_pseudo=9, so a large interaction there is evidence AGAINST the mechanism.
* it survives on `res` but never on `res-nplm` in any cell -> it is a property
  of one construction, not of composition, and should be reported as such.
"""


def plan(args):
    print(PLAN)


def selftest(args):
    import numpy as np
    ok = True

    # --- the algebra: interaction is a difference of differences ----------
    j = {"results": {
        "parent (pre)": {"perevt": 0.040, "eucl": 0.70},
        "parent (post-discovery)": {"perevt": 0.068, "eucl": 0.73},
        "p->res (post-discovery) residual": {"perevt": 0.253, "eucl": 0.71}},
        "discovery": [{"purity": 0.129}]}
    pre = {"p->res (residual)": {"perevt": 0.0225, "eucl": 0.56}}
    rows = {r["metric"]: r for r in cell_2x2(j, pre, parent="p")}
    r = rows["perevt"]
    assert abs(r["main_disc"] - 0.028) < 1e-9, r
    assert abs(r["main_ctor"] - (-0.0175)) < 1e-9, r
    assert abs(r["interaction"] - 0.2025) < 1e-9, r
    assert abs(r["total"] - 0.213) < 1e-9, r
    # total = sum of the three contrasts, always
    assert abs(r["total"] - (r["main_disc"] + r["main_ctor"] + r["interaction"])) < 1e-12
    print("  [ok] 2x2 algebra: total == disc + ctor + interaction")

    # an additive stack must show ZERO interaction
    j2 = {"results": {
        "parent (pre)": {"perevt": 0.10},
        "parent (post-discovery)": {"perevt": 0.15},
        "p->res (post-discovery) residual": {"perevt": 0.27}},
        "discovery": [{"purity": 0.3}]}
    r2 = {x["metric"]: x for x in cell_2x2(j2, {"p->res (residual)": {"perevt": 0.22}},
                                           parent="p")}["perevt"]
    assert abs(r2["interaction"]) < 1e-12, r2
    print("  [ok] purely additive cell gives interaction == 0")

    # --- missing pre child is reported, not dropped -----------------------
    rm = {x["metric"]: x for x in cell_2x2(j, {"p->res (residual)": {}}, parent="p")}
    assert rm["perevt"]["missing"] is True
    assert "interaction" not in rm["perevt"]
    assert len(rm) == len(METRICS), "every metric must still appear as a row"
    print("  [ok] absent pre-discovery child -> missing=True, row retained")

    # --- npz key mapping round-trips the exp-71 naming --------------------
    from supersig.ablation2x2 import npz_key
    assert npz_key("supcon-ft->res (residual)") == "supcon-ft-res_(residual)"
    assert npz_key("supcon-ft->res-nplm (concat)") == "supcon-ft-res-nplm_(concat)"
    assert npz_key("supcon-ft (parent)") == "supcon-ft_(parent)"
    print("  [ok] npz key mapping matches the exp-71 archive")

    # --- the repair path: real files on disk ------------------------------
    for ds, base in [("galaxy10", "dino"), ("galaxy10", "lejepa")]:
        rows_, path = archived_rows(ds, base)
        if not rows_:
            print(f"  [skip] {ds}/{base}: {path} absent")
            continue
        kids = [k for k, v in rows_.items() if "parent" not in k and v]
        if not kids:
            print(f"  [FAIL] {ds}/{base}: npz has no child metrics")
            ok = False
        else:
            print(f"  [ok] {ds}/{base}: {len(kids)} pre-discovery children "
                  f"recoverable from {os.path.basename(path)}")

    # --- perevt is bounded below by alpha under the null -------------------
    assert ALPHA == 0.05
    print(f"  [ok] alpha={ALPHA}; a parent at {ALPHA} has no per-event ability")
    print("SELFTEST PASS" if ok else "SELFTEST FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--logdir", default=os.path.join("logs", "exp134"))
    ap.add_argument("--parent", default="supcon-ft")
    ap.add_argument("--tag", default="", help="exp-71 archive tag, e.g. _h1_d0")
    ap.add_argument("--metric", default=None, choices=list(METRICS))
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest(args))
    if args.plan:
        plan(args)
    if args.aggregate or not (args.plan or args.selftest):
        aggregate(args)


if __name__ == "__main__":
    main()
