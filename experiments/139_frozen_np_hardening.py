"""
Experiment 139: harden the frozen-head density-ratio pooling result.

WHAT IS BEING HARDENED.  Exp 135's arm A (head frozen, pool scored by the NP
density ratio, no fine-tune) clears the 0.15 purity gate on **45/45 galaxy10
draw cells** -- every arm, every base, every held-out class -- at r1 purity
0.40-0.57, with no round-2 collapse.  The exp-70/125 fine-tuning loop on the
SAME parents clears it on 0/5 (dino) to 5/5 (visreg only).  It is the only
base- AND draw-independent discovery result in the campaign, and it is
therefore the paper's headline.

It rests on ONE SEED PER DRAW.  That is the gap.

WHY SEED VARIANCE IS NOT OBVIOUSLY SMALL HERE.  The head is frozen, so there is
no training randomness -- but the seed still controls the NP critic fit (kernel
centre initialisation and its optimisation) and the BIC k-means.  Exp 128 saw
seed spread reach +-0.34 on tight cuts, where the pool is only 100-200 points.
Exp 118's "draw variance exceeds seed variance" was measured on the FINE-TUNING
loop and does not automatically transfer to a frozen pool whose only stochastic
component is the critic.  So the two variance components have to be separated,
not assumed.

WHAT THIS SCRIPT DOES.  It is an orchestrator plus an estimator; the physics is
all in exp 135 (arm A, np scorer), which it calls unchanged.

  --plan       emit the exact seed x draw x cell command grid to run
  --aggregate  read the resulting JSONs and do the statistics

THE STATISTICS, which are the actual contribution here:

  1. ONE-WAY VARIANCE DECOMPOSITION over the (draw, seed) grid.
         within-draw (seed) variance   = mean over draws of var over seeds
         between-draw variance         = var over draws of per-draw means
     These answer different questions.  Seed variance says "would I get this
     number again on the same held-out class"; draw variance says "would I get
     it on a different one".  The paper needs both, and exp 118 established
     that quoting only the seed interval understates the uncertainty.

  2. GATE CLEARANCE AS A PROPORTION WITH A CLOPPER-PEARSON INTERVAL.  "45/45"
     is a point estimate on 45 Bernoulli trials; the honest statement is
     45/45 with a 95% lower bound of 0.921, and that bound is what should
     appear in the paper.  Reporting a bare 45/45 invites the reviewer to
     compute it themselves and find it weaker than it looked.

  3. THE PAIRED FROZEN-vs-LOOP CONTRAST, per (draw, seed), since both arms see
     the identical held-out class.  Paired-by-draw comparisons are decisive in
     this campaign where unpaired arm comparisons are ties.

PREDICTION.  Seed variance is small relative to draw variance (the critic is
fit on thousands of points and only its ranking is used), so the 45/45 survives
and the interval is dominated by the draw component.  If instead seed variance
is comparable to draw variance, the headline needs restating as a mean over
seeds AND draws, and every single-seed purity in the campaign inherits that
caveat.

FALSIFIER.  Any (draw, seed) cell falls below the gate -> "45/45" becomes
"k/N with a Clopper-Pearson interval", which is a materially weaker claim and
must be written as such.

    python experiments/139_frozen_np_hardening.py --selftest
    python experiments/139_frozen_np_hardening.py --plan
    python experiments/139_frozen_np_hardening.py --aggregate
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import glob
import json
from collections import defaultdict

import numpy as np

GATE = 0.15
DEFAULT_CELLS = "galaxy10:dino,galaxy10:lejepa,galaxy10:visreg"
DEFAULT_DRAWS = (0, 3, 5, 7, 8)
DEFAULT_SEEDS = (0, 1, 2)
ARMS = ("supcon-ft", "ss-ft", "nplm-sup-ft")


# ------------------------------------------------------------- the statistics

def variance_decomposition(by_draw):
    """One-way decomposition of a metric measured on a (draw, seed) grid.

    `by_draw` maps draw -> list of per-seed values.

        within  = mean_d Var_s( x[d, s] )        reproducibility on one class
        between = Var_d( mean_s x[d, s] )        transfer to another class

    Var_d is taken on the per-draw MEANS, so `between` is not inflated by seed
    noise the way a naive Var over all cells would be.  Reported alongside the
    grand mean and the total spread so a reader can see which term dominates.
    """
    # Archived (default-draw) files carry no _dN suffix, so their key is None.
    # Sort None first rather than crashing on int-vs-None comparison.
    draws = sorted(by_draw, key=lambda d: (d is not None, d))
    per_draw = [np.asarray(by_draw[d], float) for d in draws]
    per_draw = [x for x in per_draw if len(x)]
    if not per_draw:
        return None
    means = np.array([x.mean() for x in per_draw])
    within = float(np.mean([x.var(ddof=1) if len(x) > 1 else 0.0
                            for x in per_draw]))
    between = float(means.var(ddof=1)) if len(means) > 1 else 0.0
    allv = np.concatenate(per_draw)
    return dict(grand_mean=float(allv.mean()),
                sd_within_seed=float(np.sqrt(within)),
                sd_between_draw=float(np.sqrt(between)),
                sd_total=float(allv.std(ddof=1)) if len(allv) > 1 else 0.0,
                n_draws=len(per_draw), n_cells=int(len(allv)),
                draw_dominates=bool(between > within))


def clopper_pearson(k, n, cl=0.95):
    """Exact binomial interval -- k/n is a point estimate, not a guarantee."""
    from scipy.stats import beta
    lo = beta.ppf((1 - cl) / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - (1 - cl) / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def gate_report(values, gate=GATE, cl=0.95):
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    k = int((v >= gate).sum())
    lo, hi = clopper_pearson(k, len(v), cl) if len(v) else (0.0, 1.0)
    return dict(k=k, n=int(len(v)), frac=float(k / len(v)) if len(v) else np.nan,
                lo=lo, hi=hi, gate=gate,
                min=float(v.min()) if len(v) else np.nan,
                median=float(np.median(v)) if len(v) else np.nan)


def paired_delta(a_by_cell, b_by_cell):
    """Mean paired difference over cells present in BOTH, with its sd."""
    keys = sorted(set(a_by_cell) & set(b_by_cell))
    d = np.array([a_by_cell[k] - b_by_cell[k] for k in keys], float)
    d = d[~np.isnan(d)]
    if not len(d):
        return None
    return dict(n=int(len(d)), mean=float(d.mean()),
                sd=float(d.std(ddof=1)) if len(d) > 1 else 0.0,
                wins=int((d > 0).sum()))


# ------------------------------------------------------------------- the plan

def plan(cells, draws, seeds, out="logs/exp135"):
    lines = []
    for cell in cells:
        for d in draws:
            for s in seeds:
                lines.append(
                    f"SUPERSIG_NH=1 SUPERSIG_HOLDOUT_DRAW={d} "
                    f"python experiments/135_corpus_norm_everywhere.py "
                    f"--cells {cell} --scorers np --seed {s} --out {out}")
    return lines


# -------------------------------------------------------------- the aggregate

def load(out="logs/exp135", cells=None, scorer="np", variant="frozen"):
    """Harvest exp-135 JSONs into rows keyed by (cell, arm, draw, seed)."""
    rows = []
    for fn in sorted(glob.glob(os.path.join(out, "corpus_norm_*.json"))):
        base = os.path.basename(fn)[len("corpus_norm_"):-len(".json")]
        seed = 0
        if "_s" in base:
            base, s = base.rsplit("_s", 1)
            seed = int(s) if s.isdigit() else 0
        draw = None
        if "_d" in base:
            base, d = base.rsplit("_d", 1)
            draw = int(d) if d.isdigit() else None
        base = base.replace("_h1", "")
        try:
            d_json = json.load(open(fn))
        except Exception:
            continue
        for key, v in d_json.items():
            parts = key.split("|")
            if len(parts) != 4:
                continue
            cell, arm, sc, var = parts
            if sc != scorer or var != variant:
                continue
            if cells and cell not in cells:
                continue
            pur = v.get("purity") or []
            rows.append(dict(cell=cell, arm=arm, draw=draw, seed=seed,
                             r1=float(pur[0]) if len(pur) > 0 else np.nan,
                             r2=float(pur[1]) if len(pur) > 1 else np.nan,
                             margin=(float(v["margin"][0])
                                     if v.get("margin") else np.nan),
                             file=os.path.basename(fn)))
    return rows


def analyse(rows, gate=GATE):
    """Gate clearance, and a variance decomposition computed WITHIN strata.

    IMPORTANT.  The decomposition must be run inside each (cell, arm) stratum
    and then pooled.  If you group by draw alone, the "within-draw" replicates
    are different arms and bases rather than different seeds, and the number
    reported as seed variance is actually arm/base variance -- which is both
    larger and answering a different question.  With one seed per draw the
    seed term is simply UNMEASURED, and this reports it as such rather than
    silently substituting the wrong quantity.
    """
    out = {}
    for metric in ("r1", "r2"):
        out[f"{metric}_gate"] = gate_report([r[metric] for r in rows], gate)

        strata = defaultdict(lambda: defaultdict(list))
        for r in rows:
            if not np.isnan(r[metric]):
                strata[(r["cell"], r["arm"])][r["draw"]].append(r[metric])
        withins, betweens, n_seed_reps = [], [], []
        for _, by_draw in strata.items():
            v = variance_decomposition(by_draw)
            if v is None:
                continue
            reps = [len(x) for x in by_draw.values()]
            n_seed_reps.append(max(reps))
            if max(reps) > 1:
                withins.append(v["sd_within_seed"] ** 2)
            if v["n_draws"] > 1:
                betweens.append(v["sd_between_draw"] ** 2)
        measured = bool(withins)
        out[f"{metric}_variance"] = dict(
            grand_mean=float(np.nanmean([r[metric] for r in rows])),
            sd_within_seed=(float(np.sqrt(np.mean(withins))) if measured
                            else None),
            sd_between_draw=(float(np.sqrt(np.mean(betweens)))
                             if betweens else None),
            n_strata=len(strata),
            max_seeds_per_cell=int(max(n_seed_reps)) if n_seed_reps else 0,
            seed_term_measured=measured,
            draw_dominates=(bool(np.mean(betweens) > np.mean(withins))
                            if (measured and betweens) else None))

    per_cell = defaultdict(lambda: defaultdict(list))
    for r in rows:
        per_cell[r["cell"]][r["arm"]].append(r["r1"])
    out["per_cell"] = {c: {a: dict(mean=float(np.nanmean(v)),
                                   sd=float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0,
                                   n=len(v),
                                   min=float(np.nanmin(v)))
                           for a, v in d.items()}
                       for c, d in per_cell.items()}
    return out


# ------------------------------------------------------------------ selftest

def _selftest():
    rng = np.random.default_rng(0)

    print("1. VARIANCE DECOMPOSITION separates the two components")
    for name, (dsd, ssd) in [("draw-dominated", (0.10, 0.01)),
                             ("seed-dominated", (0.01, 0.10)),
                             ("balanced", (0.05, 0.05))]:
        centres = 0.5 + rng.normal(0, dsd, 8)
        by_draw = {d: list(centres[d] + rng.normal(0, ssd, 6)) for d in range(8)}
        v = variance_decomposition(by_draw)
        print(f"   {name:16s} sd_seed={v['sd_within_seed']:.4f} "
              f"sd_draw={v['sd_between_draw']:.4f}  "
              f"draw_dominates={v['draw_dominates']}")
        assert v["draw_dominates"] == (dsd > ssd), name
    print("   -> recovers which component dominates")

    print("\n2. between-draw is NOT inflated by seed noise")
    centres = 0.5 + rng.normal(0, 0.001, 10)          # essentially no draw effect
    by_draw = {d: list(centres[d] + rng.normal(0, 0.20, 30)) for d in range(10)}
    v = variance_decomposition(by_draw)
    print(f"   true draw sd ~0.001, seed sd 0.20 -> "
          f"sd_draw={v['sd_between_draw']:.4f}, sd_seed={v['sd_within_seed']:.4f}")
    assert v["sd_between_draw"] < 0.06, v
    assert v["draw_dominates"] is False
    print("   -> averaging over seeds first keeps the draw term honest")

    print("\n3. 45/45 IS NOT 1.0 -- the interval the paper must quote")
    for k, n in ((45, 45), (44, 45), (40, 45), (9, 9)):
        lo, hi = clopper_pearson(k, n)
        print(f"   {k}/{n} -> 95% CI [{lo:.3f}, {hi:.3f}]")
    lo, _ = clopper_pearson(45, 45)
    assert 0.90 < lo < 0.94, lo
    print("   -> 45/45 has a 95% lower bound of 0.921, not 1.0")

    print("\n4. gate_report flags the WORST cell, not just the mean")
    good = list(rng.uniform(0.40, 0.57, 45))
    g = gate_report(good)
    print(f"   all above gate: {g['k']}/{g['n']}  min={g['min']:.3f}  "
          f"lo={g['lo']:.3f}")
    assert g["k"] == 45
    bad = good[:44] + [0.05]
    g2 = gate_report(bad)
    print(f"   one below     : {g2['k']}/{g2['n']}  min={g2['min']:.3f}  "
          f"lo={g2['lo']:.3f}")
    assert g2["k"] == 44 and g2["lo"] < g["lo"]
    print("   -> a single failure moves the lower bound materially")

    print("\n5. paired_delta pairs by cell and reports wins")
    a = {f"c{i}": 0.5 + 0.1 * i for i in range(6)}
    b = {f"c{i}": 0.4 + 0.1 * i for i in range(6)}
    p = paired_delta(a, b)
    print(f"   n={p['n']} mean={p['mean']:+.4f} sd={p['sd']:.4f} wins={p['wins']}")
    assert p["n"] == 6 and abs(p["mean"] - 0.1) < 1e-9 and p["wins"] == 6

    print("\n6. archived rows (draw=None) mix with drawn rows without crashing")
    mixed = {None: [0.50, 0.52], 0: [0.44, 0.46], 3: [0.55]}
    vm = variance_decomposition(mixed)
    print(f"   {vm['n_draws']} draw groups, {vm['n_cells']} cells, "
          f"mean {vm['grand_mean']:.3f}")
    assert vm["n_draws"] == 3 and vm["n_cells"] == 5
    gm = gate_report([v for vs in mixed.values() for v in vs])
    print(f"   gate {gm['k']}/{gm['n']}  lo={gm['lo']:.3f}")

    print("\n7. the plan is the full grid")
    lines = plan(DEFAULT_CELLS.split(","), DEFAULT_DRAWS, DEFAULT_SEEDS)
    print(f"   {len(lines)} runs = {len(DEFAULT_CELLS.split(','))} cells x "
          f"{len(DEFAULT_DRAWS)} draws x {len(DEFAULT_SEEDS)} seeds")
    assert len(lines) == 45
    assert "SUPERSIG_HOLDOUT_DRAW=" in lines[0] and "--seed" in lines[0]

    print("\nselftest OK")


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--cells", default=DEFAULT_CELLS)
    ap.add_argument("--draws", default=",".join(map(str, DEFAULT_DRAWS)))
    ap.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    ap.add_argument("--src", default="logs/exp135")
    ap.add_argument("--out", default="logs/exp139")
    args = ap.parse_args()

    if args.selftest:
        _selftest(); return

    cells = args.cells.split(",")
    draws = [int(x) for x in args.draws.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    if args.plan:
        lines = plan(cells, draws, seeds, args.src)
        print(f"# {len(lines)} runs: {len(cells)} cells x {len(draws)} draws "
              f"x {len(seeds)} seeds  (exp 135 arm A, np scorer)")
        print("\n".join(lines))
        return

    if not args.aggregate:
        ap.error("pick --plan, --aggregate or --selftest")

    rows = load(args.src, set(cells))
    if not rows:
        print(f"no frozen/np rows found under {args.src}")
        return
    res = analyse(rows)
    os.makedirs(args.out, exist_ok=True)

    print(f"harvested {len(rows)} (cell, arm, draw, seed) cells\n")
    for m in ("r1", "r2"):
        g, v = res[f"{m}_gate"], res[f"{m}_variance"]
        print(f"{m}: {g['k']}/{g['n']} above gate {g['gate']}  "
              f"95% CI [{g['lo']:.3f}, {g['hi']:.3f}]  "
              f"min={g['min']:.3f} median={g['median']:.3f}")
        sd_s = ("UNMEASURED (1 seed/cell)" if not v["seed_term_measured"]
                else f"{v['sd_within_seed']:.4f}")
        sd_d = ("--" if v["sd_between_draw"] is None
                else f"{v['sd_between_draw']:.4f}")
        dom = ("" if v["draw_dominates"] is None
               else ("   DRAW dominates" if v["draw_dominates"]
                     else "   SEED dominates"))
        print(f"    mean {v['grand_mean']:.4f}   sd_seed {sd_s}   "
              f"sd_draw {sd_d}   (max {v['max_seeds_per_cell']} seed(s) per "
              f"cell, {v['n_strata']} strata){dom}")
    print("\nper cell/arm (r1 mean +- sd over draws x seeds, and the worst cell):")
    for c, d in sorted(res["per_cell"].items()):
        for a, s in sorted(d.items()):
            print(f"  {c:20s} {a:14s} {s['mean']:.3f} +- {s['sd']:.3f}  "
                  f"min {s['min']:.3f}  (n={s['n']})")

    with open(os.path.join(args.out, "hardening.json"), "w") as fh:
        json.dump(dict(rows=rows, analysis=res), fh, indent=1, default=float)
    print(f"\nwrote {args.out}/hardening.json")


if __name__ == "__main__":
    main()
