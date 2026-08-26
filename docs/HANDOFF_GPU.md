# Handoff — GPU runs for the ICLR discovery paper

Written 2026-08-26.  Point a session on the GPU machine at this file.  It is
self-contained: read it, then `docs/IMPROVEMENT_TESTS.md` Tier 9 for the
prediction/falsifier of each run, then `docs/PAPER_PLAN.md` for why.

**Do not re-derive the plan.  Run the four blocks below in order and append
results to `logs/SUMMARY_TABLES.md` in the house style (`docs/METRICS.md`).**

---

## 0. Read first (5 minutes, prevents every known footgun)

1. `supersig/holdouts.py` — the two env vars that control holdouts.
2. `docs/METRICS.md` "Known pitfalls checklist" — all 10.
3. This section.

### The two environment variables

```bash
SUPERSIG_NH=1               # hold out 1 class per dataset (single-holdout regime)
SUPERSIG_HOLDOUT_DRAW=3     # WHICH class(es): reproducible random draw #3
```

With **both unset**, the campaign default is reproduced byte-for-byte — same
holdout sets, same filenames.  Set either one and every artifact this run
writes is tagged (`_h1`, `_h1_d3`), so runs cannot overwrite each other.
This matters: before this mechanism existed, a single-holdout run would have
destroyed the multi-holdout fine-tune checkpoints, whose `_seen` suffix means
"trained excluding the holdouts" — contents differ, filename did not.

### Five rules

1. **Never pool single- and multi-holdout numbers in one table.**  They reach
   different conclusions (exps 89/109: the purity gate is reachable at h5/h10
   but rate-floored at h1, and the quantile-strictness verdict *inverts*).
   Every table is split by regime.
2. **Single-holdout numbers need an interval over DRAWS, not seeds.**  Exp 118:
   draw-to-draw spread exceeds seed spread (probe sd ~0.019; mahaT 0.391-0.569
   across draws vs 0.001-0.010 across seeds).  Under nh=1 the entire result
   rests on one class.  Run >= 3 draws, report mean +- sd across draws.
3. **Naming hazard — do not pool these three:**
   - `ss-ft` (exp 70) = SupCon + global SIGReg **lam=5**
   - `supcon_sigreg` (exp 50) = SupCon + global SIGReg **lam=1** (the
     `HybridContrastiveLoss` default)
   - `"ss"` (`docs/LOSSES.md`) = **SimCLR** + SIGReg, unsupervised
   `ssig` in exp 67 is lam=5, i.e. the `ss-ft` twin.
4. **`--quick` numbers are pipeline checks, never results.**
5. **Frozen-space runs changed on 2026-08-26.**  "Frozen" now really means
   frozen: `supersig.train.set_train_mode` puts a fully-frozen backbone in
   eval() so BatchNorm running statistics stop drifting.  Before the fix the
   CIFAR ResNet-20 trunk moved embeddings by up to 1.29 (mean |z| 0.52) over 3
   "frozen" rounds.  Consequence: **archived frozen CIFAR numbers (exps
   86/92b/109) will not reproduce exactly.**  If you re-run any of them, report
   the new number and note the change rather than treating a mismatch as a bug.
   Transfer cells (ViT/LayerNorm) are unaffected and should reproduce.

### Sanity check before starting

```bash
python -m pytest tests/ -q          # expect 146 passed
python supersig/holdouts.py         # prints the default vs SUPERSIG_NH=1 sets
```

If the suite is not green, stop and report rather than running anything.

---

## Block A — Exp 124: leakage-free shortlist (run this first)

**Why first:** cheapest, and it decides whether the paper's workhorse claim has
any leakage-free support at all.

Every CIFAR cell built with `pretrain=ds` uses hub weights that already saw the
held-out class (`supersig/models.py:80-82`).  Exps 67/68 are the clean lineage
(`pretrain=None`).  The two arms below were added 2026-08-25 and verified
numerically identical to their exp-70 fine-tune twins.

```bash
python experiments/67_scratch_pretrain.py --arms supcon ssig nplmsd
python experiments/68_scratch_discovery.py --bases supcon,ssig,nplmsd
```

CIFAR-100, 100-D, single holdout {4}, 200 ep, random init.  Cost: 2 x 200-epoch
pretrains + one discovery pass.

**Report:** probe/acc/eucl/mahaT per arm (exp 67); probe pre->post, per-round
purity, and the per-event/SparKer/Maha/MMD power grid (exp 68).

**What we expect** (say so plainly if it does not happen): purities stay
0.00-0.01 and per-event stays dead — C100 at h1 is rate-floored.  The
informative comparison is **`ssig` vs `supcon` on probe**: SIGReg beats plain
SupCon on 2/3 clean bases in exp 50 but loses on hub-init trunks.  If `ssig`
does *not* beat `supcon` here, that is a real falsification — flag it loudly,
do not bury it.

---

## Block B — Exp 125: the single-holdout battery (the bulk of the budget)

Run in this priority order.  **Stop and report after galaxy10 and dtd** — those
two decide whether the rest is worth the hours.

```bash
for D in 0 1 2 3 4; do
  SUPERSIG_NH=1 SUPERSIG_HOLDOUT_DRAW=$D \
    python experiments/70_cars_ft_suite.py --dataset galaxy10 --base dino
done
```

Then the same loop for: `galaxy10 x {lejepa, visreg}`, then `dtd x {dino,
lejepa, visreg}`, then flowers, cars, aircraft.  After the exp-70 fine-tunes
exist for a cell, run the downstream evaluations on it:

```bash
SUPERSIG_NH=1 SUPERSIG_HOLDOUT_DRAW=$D python experiments/71_residual_ft_grid.py --dataset dtd --base dino
SUPERSIG_NH=1 SUPERSIG_HOLDOUT_DRAW=$D python experiments/72_residual_discovery.py
SUPERSIG_NH=1 SUPERSIG_HOLDOUT_DRAW=$D python experiments/80_sparker_all_spaces.py
SUPERSIG_NH=1 SUPERSIG_HOLDOUT_DRAW=$D python experiments/100_dense_reach.py
SUPERSIG_NH=1 SUPERSIG_HOLDOUT_DRAW=$D python experiments/104_interpretability_panel.py
```

Artifacts land tagged `_h1_d{D}`.  **Report mean +- sd across draws**, never a
single draw.

**What we expect:** discovery survives on galaxy10 (novel-class rate 1/10) and
dies on the label-rich cells (1/196 cars, 1/102 flowers) — the rate floor
generalizing beyond C100.  Draw sd will likely exceed most arm-to-arm gaps; if
so, say that explicitly, because it means several single-holdout comparisons
are underpowered and must be reported as ties.

**Note:** galaxy10 is the only dataset genuinely out-of-distribution to the
backbones (all three are ImageNet-1k SSL ViT-B/16; cars/flowers/aircraft are
fine-grained partitions of regions ImageNet already covers).  Its numbers carry
the most weight in the paper.  Treat them accordingly.

---

## Block C — Exp 126: multi-holdout, restricted

Multi-holdout is kept **only** for the label-rich datasets: cars (196),
flowers (102), aircraft (100), cifar100 (100), dtd (47).  Not galaxy10 (10
classes, already nh=1).

Most of this data already exists at the campaign default (unset both env vars).
The work is mostly **re-tabulation into regime-separated tables**.  Add draws
where a headline number currently rests on a single alphabetical draw —
especially the ss-ft DTD purities (0.795/0.803/0.811), which are the paper's
workhorse evidence and currently rest on one draw:

```bash
for D in 0 1 2; do
  SUPERSIG_HOLDOUT_DRAW=$D python experiments/70_cars_ft_suite.py --dataset dtd --base dino
done
```

(no `SUPERSIG_NH` here — multi-holdout keeps the default count of 10)

**Falsifier worth taking seriously:** if the DTD purities move outside their
draw interval, the workhorse claim was a favourable alphabetical draw — exactly
the hazard exp 118 identified.

---

## Block D — leftovers from Tier 8

Lower priority; run only if A-C finish.

First, the cheap re-validation (exp 127): re-run the frozen density-ratio
pool now that BN is actually frozen, and compare to archived `logs/exp109/`.
It is evaluation-only and it re-validates the paper's second construction.

```bash
python experiments/109_c100_density_pool.py

python experiments/113_tau_generality.py --save-embs
python experiments/121_transductive_tuner.py --tau-archive logs/exp113/embs --cell cifar100_on
python experiments/122_basin_geometry.py --cell cifar100_on
```

---

## Reporting

Append to `logs/SUMMARY_TABLES.md` in the house style (`docs/METRICS.md`
"Reporting template"), with the regime named in the header:

```markdown
## dtd (exp 125; dino, 100d, holdout 1 [single-holdout], 5 draws, 30 ep)
```

Commit code + logs + npz + plots together.  If a program-level lesson changed,
update `docs/` too.

**Honesty rules that matter more than the numbers:**
- Report negative results as prominently as positive ones.  Exp 124 is expected
  to be largely negative and that is a *result* (the rate floor), not a failure.
- If a prediction in Tier 9 is falsified, say so in the summary bullet.  The
  campaign has retracted claims before (exp 61 retracted exp 50's headline
  after 5 paired seeds) and that is the norm here, not an embarrassment.
- Never report a `--quick` number as a result.
- If draw sd swamps an arm-to-arm gap, report a tie.
