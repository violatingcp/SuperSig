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
python -m pytest tests/ -q          # expect 173 passed
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

## Block E — Exp 128: the pool cut (run on every cell that yields embeddings)

Evaluation-only, minutes.  `tau_quantile=0.95` was inherited from exp 23 and
never derived; this sweeps it densely against the analytic ceiling
`purity <= min(1, b/q)`.

```bash
python experiments/128_pool_cut_optimization.py --selftest        # expect 8 OK
python experiments/128_pool_cut_optimization.py \
    --embs logs/exp113/embs/cifar100_on.npz --holdouts 99 --bic
```

It needs an npz with `tr` / `tr_lab`, which is what `113 --save-embs` writes.
**If any run in Blocks A/B can cheaply dump its train embeddings in that
format, do so** -- this analysis is free once the bank exists and it tells us
whether our purity numbers are limited by the scorer or by an arbitrary cut.

Report per scorer: the best usable cut (highest purity with n_novel >= 100),
its headroom (purity/ceiling), and the cut-free AUC.

Then exp 129 -- the label-free cut rule, which is what we would actually ship:

```bash
python experiments/129_legal_pool_cut.py --selftest                 # expect 7 OK
python experiments/129_legal_pool_cut.py --embs <same npz> --holdouts 99
```

**Exp 130 step (a) is DONE and came back clean** -- the NP critic converges.
In-sample calibration `E_ref[e^f]` is 0.997-1.019 on six real exp-54 CIFAR-10
spaces (ideal 1.000), and no point anywhere reaches the +-20 clamp.  An earlier
worry that the critic was diverging was my measurement error (out-of-sample vs
in-sample); it is retracted in `IMPROVEMENT_TESTS.md`.  **No archived SparKer
number is in question.**

Still worth printing on real runs: `np_pool_scores(..., return_calib=True)`
now returns `dict(calib_in, calib_out)`.  `calib_out` (over all seen points,
not just the fitted reference subsample) is the one that can drift -- it ranges
1.000-2.706 across the six spaces -- and it is the value the scores are
actually used at.

---

## Block F — Exp 131: re-run cifar10 / cifar100 / galaxy10 (REQUESTED)

These three cells must be re-run: the frozen-BN fix alone means archived frozen
CIFAR numbers will not reproduce, and the pool cut and kmax both changed.

```bash
python experiments/131_legal_cut_discovery.py --selftest
python experiments/131_legal_cut_discovery.py --cells cifar10,cifar100
python experiments/131_legal_cut_discovery.py --cells galaxy10:dino,galaxy10:lejepa,galaxy10:visreg
```

`run_discovery` defaults are UNCHANGED (`cut_rule="quantile"`), so anything you
re-run without the flag reproduces the archive.  Pass `cut_rule="legal"` with
`pool_score="np"` for the new rule.

**Report the `ok` flag.**  False means the rule refused -- the estimated
novelty could not support a detectable cluster.  That is a RESULT ("this space
cannot support discovery at this rate"), not a crash, and we expect it on the
b~1% cells.

---

## Block H — Exp 133: BN adaptation A/B (after exp 132)

Paired on the cached exp-89 spaces exp 127 built; evaluation-scale.

```bash
python experiments/133_bn_adaptation_ab.py --selftest
python experiments/133_bn_adaptation_ab.py            # do NOT pass --refresh
```

Report the paired table (frozen vs bn-adapt purity r1/r2, drift).  A False
prediction here is fine and expected to be possible: it retires the idea.

**Loader fix that gates Block B's downstream evals:** `exp77.head_emb` was
draw-blind until 2026-08-27 (no `run_tag()` in the checkpoint name).  Exps
80/100/104/111 on the `_h1_d*` cells only mean something from that commit on.

## Block J — Exps 134a / 134c (after Block I)

```bash
# 134a: the width-matched control (tagged _e200, does not touch archived parents)
for S in 0 1 2; do
  python experiments/70_cars_ft_suite.py --dataset cars --base visreg --emb-dim 200 \
      --arms supcon-ft --skip-discovery --seed $S
done
python experiments/134a_width_control.py --dataset cars --base visreg
# 134c: residual AFTER discovery
python experiments/134c_residual_after_discovery.py --dataset galaxy10 --base dino
```

Report 134a's paired verdict verbatim (it prints TIE when inside the floor)
and 134c's summary table beside the archived exp-71 rows, with the discovery
purity in the header.

## Block K — Exp 135: corpus-adaptive normalisation everywhere

```bash
python experiments/135_corpus_norm_everywhere.py --selftest
python experiments/135_corpus_norm_everywhere.py --cells galaxy10:dino,galaxy10:lejepa,galaxy10:visreg
SUPERSIG_NH=1 SUPERSIG_HOLDOUT_DRAW=3 python experiments/135_corpus_norm_everywhere.py --cells galaxy10:dino
```

Paired table per cell/arm/scorer: frozen vs corpus-norm round-2 purity, post
probe, drift.  Report the falsifier cells (helps at purity ~0) as loudly as
the wins.

## Block L — Exp 136: the from-scratch CIFAR master grid

```bash
python experiments/136_scratch_master.py --selftest
# tier 1 (checkpoints exist)
python experiments/68_scratch_discovery.py --dataset cifar100            # all 8 bases
python experiments/136_scratch_master.py --dataset cifar100
# tier 2
python experiments/67_scratch_pretrain.py --dataset cifar10 --resume     # all 8 arms
python experiments/68_scratch_discovery.py --dataset cifar10
python experiments/136_scratch_master.py --dataset cifar10
# tier 3: draws (holdout = the class, from holdouts.py draws 0/1/2)
python experiments/67_scratch_pretrain.py --dataset cifar10 --holdout 8 --arms supcon ssig nplmsd nplmcw
python experiments/68_scratch_discovery.py --dataset cifar10 --holdout 8 --bases supcon,ssig,nplmsd,nplmcw
python experiments/136_scratch_master.py --dataset cifar10 --holdout 8 --arms supcon ssig nplmsd nplmcw
python experiments/136_aggregate.py cifar10
```

Artifacts at a non-default holdout are tagged `_h{N}`; holdout 4 keeps the
archived names.  `logs/exp68/scratch_discovery_cifar100_3bases_archived.npz`
is the pre-Block-L exp-68 file.

## Block G — Exp 132: supervised linear probe (piggyback on Block B)

The campaign has NO supervised linear probe -- the `probe` column everywhere is
holdout-vs-rest novelty AUC.  This adds the missing metric so the paper can
make its "discovery costs nothing" argument honestly.

```bash
python experiments/132_supervised_probe.py --selftest        # expect 5 OK
python experiments/132_supervised_probe.py --glob 'logs/expNN/embs_*.npz' \
    --holdouts <h> --seeds 3 --baseline supcon-ft
```

Needs npz with `tr`/`tr_lab`/`te`/`te_lab`.  **If Block B runs can dump train
AND test embeddings, this and exp 128/129 all come for free.**

Report top-1 mean +- sd over >= 3 seeds.  The script prints TIE when a gap is
under the noise floor -- **report the TIE**, do not round it into a win.  The
comparison that matters is ss-ft vs supcon-ft (does SIGReg cost supervised
accuracy?), NOT our arms vs the SSL arms: ours use labels and theirs do not.

---

## Block I — Exp 134: what the residual is actually doing

The residual is a candidate for main results, but the evidence audit found it
wins the novelty probe 30/30 while LOSING eucl 14/15 and mahaT 11/15, and never
improving purity.  Three questions:

**(B) first — it is nearly free.**  Load a real TRAINED parent/child pair and
print the RMS-norm ratio:

```bash
python experiments/134_residual_audit.py --parent <parent.npz> --child <child.npz> --holdouts <h>
# or, for every trained pair of a cell (honours the draw env):
python experiments/134_residual_audit.py --cells cars:visreg,galaxy10:dino
```

If the ratio is near 1, question (B) is closed and you can skip the combiner
scan.  If it is far from 1, the scan (raw / standardize / unitnorm / whiten)
says whether the archived geometry losses are a combination artefact.

Partial result already in from an UNTRAINED proxy on CIFAR
(`logs/exp134/untrained_residual_cifar10.json`): the ratio does predict
sensitivity, but standardising does NOT rescue eucl -- so expect the geometry
trade-off to be real.  Note the trained child is predicted to be LARGER than
its parent (SIGReg lam=5 -> unit per-dim variance), i.e. on the opposite side of
ratio 1 from the proxy.

**(A) and (C) need training:** (A) a width-matched non-residual control against
the 2-way concat on cars/VISReg -- the flagship +0.148 has never had one, and
exp 85 already showed a capacity effect on that cell.  (C) build the residual
AFTER discovery instead of before, so it removes what the enlarged anchor set
explains.

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
