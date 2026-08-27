# Proposed tests to improve performance — exps 81–134

Companion to [QUESTIONS.md](QUESTIONS.md) (whose "open items as of exp 58" list
this supersedes).  Every entry follows the standard protocol in
[METRICS.md](METRICS.md): `experiments/NN_<topic>.py` with `--quick`, npz to
`logs/expNN/`, paired seeds across cells, full metric row (probe AND geometry),
append to `logs/SUMMARY_TABLES.md`.

Each test states a **prediction** and a **falsifier**.  A test whose falsifier
is not stated is not a test; a run that cannot come out the other way is not
worth the GPU hours.  Cost is wall-clock on one GPU, rough.

Ordering is by expected value per unit cost, not by topic.

Numbering starts at 81: exp 80 is taken by `80_sparker_all_spaces.py` (the
SparKer coverage audit for the exp-71/73 residual and concat spaces).

**Already closed:** the unit suite gained `tests/test_calibration.py`, which
pins the NPLM calibration property and the gauge identities of App. A
(softmax exactly invariant to a per-anchor shift, NPLM strictly convex along
it with its minimum at `E_ref[e^g] = 1`, and max-subtraction breaking that
while clamping preserves it).  Executed 2026-08-21 on the campaign machine:
full suite green (103 passed incl. test_calibration).

---

## Status board (2026-08-21)

| exp | what it asked | outcome |
|---|---|---|
| 80 | SparKer coverage for the uncovered spaces | **done** — 106 spaces / 17 cells; residual *child* out-detects its parent |
| 81 | is NPLM seed variance a property of the critic? | **CONFIRMED** — 3.1× by critic vs 1.6× by positives; `dist-sup` stable at ±0.0125 |
| 82 | calibration residual as a label-free seed selector | **FALSIFIED** — Spearman +0.30/−0.10; best-of-5 *underperforms* the mean |
| 86 | fix aircraft discovery by freezing | **CONFIRMED, sharper** — freeze *everything*: 90–99% of per-event gain at zero probe cost |
| 87 | LID on the residual half | **FALSIFIED** on flowers — the signal is class-level, in the parent half |
| 90 | unsupervised predictor of the LID-vs-distance regime | **FALSIFIED** — 3 diagnostics, best 12/17 vs an 11/17 base rate |
| 92 | SparKer centres as the discovery clustering | **PARTIALLY CONFIRMED** — cars purity +33% rel (0.161→0.214) but still below the gate; unpredicted win: round-2 purity immune to collapse |
| 92b | 92+86 combined (density-ratio pool, frozen space) | **DOMINATES** — wins all six round measurements, zero probe cost; exposes *anchor absorption* as a second failure mode |
| 93 | NP score as pool scorer | **FALSIFIED as stated** — lid > np on flowers by 0.03–0.06 (finite-sample, the named falsifier); np *best* below the gate (cars r1 0.214) |
| 94 | null validity under novelty-seeking ft | **FALSIFIER FIRED = good outcome** — FPR at or below nominal everywhere; nulls valid, split-disjointness not needed, exp 95 cleared |
| 96 | does the pairwise critic warm-start the test? | **FALSIFIED, strong form** — warm start neutral-to-harmful (0.800→0.560 at 50 steps); class structure is where novelty *isn't* |
| 83 | variance-reduced NPLM (distance+bias) | **FALSIFIED, twice over** — the fixes *do* achieve calibration (resid 1.77→0.03) and it buys nothing; per-event falls 0.042→0.008 |
| 84 | two-stage under strict open-world | **CONFIRMED** — gains survive within 0.007–0.016; reading #8 graduates from caveat to citable |
| 85 | iterated residuals | **SPLIT + TWO RECORDS** — flowers decomposition 0.910±0.012 (beats discovery, without discovery); cars is capacity, 0.884±0.003 |
| 89 | C100 purity-gate grid | **SHARPENED FALSIFIER** — C100 is GEOMETRY-blocked, not rate-blocked; stricter quantiles make purity worse |
| 100 | dense fraction scan | **PARTIAL** — 8/17 citable (vs 4/106); 7 cells genuinely need f>0.1, so the bound *is* the reach |
| 101 | frozen discovery generality | **CONFIRMED incl. the coarse clause** — regime-scoped: freeze on-manifold, fine-tune where novelty is outlying |
| 102 | residual child as standalone detector | **YES, and cheaper than predicted** — best transfer reach anywhere (cars/visreg f95 0.049); semantics legible, not scrambled |
| 103 | LID neighbourhood decomposition | **NEITHER prediction nor falsifier** — same-class component carries it; bare composition nearly matches LID |
| 104 | interpretability panel | **RUN** — no space achieves unit width (0.13–0.79); distances overstate ΔlogL by 4–37× |
| 113 | is the tau x marginal basin general? | **MANY-CLASS ONLY** (prediction held) — plateau from tau>=0.3 on C100; on C10 it costs 0.94->0.65 probe. No campaign-wide re-run |
| 116 | SparKer M matched to intrinsic dimension | **PARTIAL** — helps at the ID extremes (2 new crossings), hurts mid-ID cars; ordering preserved so M=16 stays default |
| 118 | holdout-selection audit | **CONFIRMED + refinement** — alphabetical draw is not a confound, but DRAW variance (sd 0.019) exceeds SEED variance (0.001-0.010) |
| 119 | 18th cell: rr_disp out of sample | **FAILED at the frozen threshold (2/3)** — and returned one answer for three inputs; regime line CLOSES per the pre-registered stopping rule |
| 112 | inherited-constants audit | **done** — 9 never-swept constants sit under named claims (predicted >=3) |
| 117 | power analysis | **half falsified** — paired MDE 0.0180 at n=3; CIFAR spreads are WIDER than transfer, opposite to prediction |
| 99 | SparKer discovery reach `f95` | **done** — and mostly *unresolvable* on the current fraction grid (see exp 100) |
| 88 | classwise-realizable C100 | **BOTH CLAUSES FALSIFIED** — cent->anchor stall (~3.5) dim-independent at 100/128/200-D (optimization equilibrium, Q4 restated); unpredicted: supcon_sigreg control posts mahaT 0.545–0.558 at 100–128-D with probe 0.90–0.91, above the old ceiling |
| 91 | multi-seed the records | **done** — all citable, no ordering flips: aircraft 0.866±0.002, flowers 0.887±0.019, dtd 0.867±0.004 (seed-2 new best), galaxy10 0.971±0.004 |
| 97 | M / sigma-schedule systematics | **split** — sigma_ratio flat (annealed schedule robust); M INVERTS the prediction: high-ID wants FEWER kernels (supcon M=4 0.60 vs M=64 0.24) |
| 98 | SparKer-ft as the discovery ft objective | **FAILS beyond its falsifier** — worse than proto AND nplm on every column incl. its own statistic (post SparKer 0.06–0.12 vs 1.00); statistic-chasing destroys separation; with 96, contraindicates exp 95 as specified |


| 95 | SparKer as a training loss | **CONTRAINDICATED** by 96+98 — do not run in the alternating form |

Tier 6 (run 2026-08-22/23; verdicts in `logs/SUMMARY_TABLES.md`):

| exp | what it asked | outcome |
|---|---|---|
| 106 | panel control: unfaithful space or unestimable covariance? | **PREDICTION HOLDS** — rms stable within ±0.05 down to n=10/class (the width headline survives); r_llr collapses at small n and rises to 0.73–0.91 under diag/iso references → cite transfer r_llr only as [full, iso] bounds |
| 105 | direct class-conditional width penalty | **WATCHED FALSIFIER FIRES** (5th meta-lesson confirmation) — rms→1 on demand at λ_w=1, but slope stalls at 2.3–4.6 (shape, not scale), r_llr flat-to-down, ECE worse; safe, decorative. Exceptions: c10 res-cat improves (probe 0.9593, sep 10.9); c100 supcon_sigreg jumps mahaT 0.389→0.574 — the exp-110 ceiling break by an independent route |
| 107 | is neighbourhood composition enough? | **FALSIFIER DOES NOT FIRE, LID NOT DEMOTED** — comp ties LID on flowers (−0.016), beats it on dtd/c10 (0.865 vs 0.672), but is ANTI-correlated on aircraft (0.42) and loses the rotation control (0.818 vs 0.875). Battery: add comp, keep lid |
| 109 | density-ratio pooling on C100 | **PREDICTION HOLDS — GEOMETRY BLOCK BROKEN** — sparker-frozen r1 purity 0.358 at h10:q0.99 (distance ceiling 0.121; matched point 0.002), r2 RISES to 0.418; strict quantiles now help (f-tail owned by novelty). h1 stays blocked: rate floor, not geometry |
| 108 | on-manifold predictor, second attempt | **PREDICTION MET** — rr_disp (the mechanism feature) 15/17 sign accuracy vs 11/17 base; comp_gap at base rate. In-sample caveat stands; line closes successfully, no third attempt |
| 110 | why did the softmax control break the C100 ceiling? | **CAUSE FOUND: τ×marginal interaction** — exp-88's arm accidentally ran at τ=1.0; only τ=1.0 AND SIGReg break the ceiling (0.539–0.572 at 32–128-D), each alone ≤0.43. Dimension incidental (present at 32-D); NOT a width effect (winner rms~0.5). New designed record mahaT 0.572±0.031 @128-D, probe 0.910 |
| 111 | child-only deployment | **PREDICTION HOLDS, DEPLOYABLE** — child-only detection ties/beats concat (aircraft:visreg +0.048 per-event) at half width with better f95; parent-only explanations overlap concat's at 0.73–0.85 top-1. Protocol limit: alphabetical holdouts make absolute superclass agreement near-unmeasurable |

Tier 7 (run 2026-08-24; 112/117 on the laptop, 113-116/118/119 on the GPU box; verdicts in `logs/SUMMARY_TABLES.md`):

| exp | what it asked | outcome |
|---|---|---|
| 112 | inherited-constants audit | **PREDICTION EXCEEDED** — nine never-swept constants under named claims (predicted ≥3); `logs/exp112/constants.md` |
| 117 | are our comparisons powered? | **HALF FALSIFIED** — see laptop write-up; feeds the exp-118 draw-variance finding |
| 113 | is the τ×marginal basin general? | **MANY-CLASS ONLY** — C100 plateau from τ≥0.3 (mahaT ~0.51, per-event turns on); on C10 high τ collapses probe 0.94→0.64 for +0.03 mahaT. No campaign-wide re-run needed |
| 114 | which knobs interact? | **PREDICTION HOLDS** — τ×lam is the only large interaction (c100 mahaT −0.058, c10 probe +0.090); τ the one big main effect; n_slices inert; knob space separable, cube methodology stands |
| 115 | does the dissociation survive per-arm tuning? | **SURVIVES, STRONGER THAN PREDICTED** — seen-only (open-world-legal) tuning selects the default τ for softmax arms: the calibration basin costs seen accuracy and is invisible to legal model selection. NPLM keeps its per-event edge. One line left open: a seen-only *calibration* criterion was not tried |
| 116 | M matched to intrinsic dimension | **PARTIAL** — two first-time crossings (galaxy10:dino, dtd:lejepa) but regressions on mid-ID cars; ordering preserved; M=16 stays default, matched-M cited as sensitivity (citable reach 8/17 → 10/17 best-of-two) |
| 118 | holdout-selection audit | **PROBES NOT CONFOUNDED** — archived draw sits inside the random-draw distribution; but draw variance (sd ~0.019) exceeds seed variance → draw-resampled intervals for cross-holdout claims; mahaT is draw-dependent (0.39–0.57) |
| 119 | rr_disp out of sample | **FAILS (pre-registered)** — 2/3 at the frozen threshold on food101×3 bases (predictions committed before scoring, commit 2131998); regime line closes per exp 108's stopping rule. All held-out gaps within ±0.03 (boundary dataset) — context, not exculpation |

Across Tiers 1-6, fifteen of twenty-four predictions were falsified or
materially revised.  **Tier 7 inverted that rate**: of its eight tests, three
held cleanly (113, 114, 118), one exceeded its prediction (112), one held for a
reason nobody predicted (115), two came back partial (116, 117) and one failed
outright (119).  Predictions got better as the mechanisms got clearer, which is
the outcome an experimental program should want.  That is the intended hit rate: the
entries below are written to be wrong in a specific way, and two of them
(82, 87) taught us more by failing than they would have by passing.  Where a
result changed the motivation for a pending test, the entry has been rewritten
rather than left standing.

---

## Tier 1 — cheap, decisive, and targeting a known loss of performance

These three all attack the same thing: the label-free NPLM arm loses ~0.085
probe to supcon on C100 *on average* while its best seeds are competitive
(0.9349 archived vs 0.8548±0.042 over 5 seeds).  That is not a weak method, it
is a **high-variance** method, and variance is recoverable performance.  The
gradient analysis in `discovery_metrics_iclr.tex` App. A predicts where the
variance comes from and hands us two interventions.

### Exp 81 — Is the seed variance a property of the CRITIC, not the estimator?

> **DONE 2026-08-20 — CONFIRMED.**  sd splits 3.1x by critic (bil 0.055, dist 0.018) vs 1.6x by positives; `dist-sup` is the stable corner (0.8366±0.0125).  One refinement: the proposed proxy `s = sd(g)` ANTI-correlates with probe sd (-0.80); the `e^g` spread is the real predictor (+0.80).  Log `sd(e^g)`, not `sd(g)`.

**Motivation.**  App. A: the NPLM reference gradient is `e^g/N(N-1)`, so for
approximately-Gaussian critic values with spread `s` the relative gradient
variance is `exp(s^2) - 1`.  The distance critic has `g = -½||z-z'||²/τ ≤ 0`,
hence `e^g ≤ 1` **bounded by construction**; the bilinear critic
`g = <z,z'>/τ` is unbounded above.  The campaign already contains the
signature: `nplm_bilinear` 0.855±0.042 (heavy right tail) vs `nplm_sup_dist`
the most base-robust arm in the program (probe spread 0.009 across three
trunks vs supcon's 0.076).  But those two differ in *positives* (instance vs
supervised) as well as critic, so the attribution is currently confounded.

**Protocol.**  2×2 at matched τ=1, C100 32-D, 5 paired seeds each:
critic ∈ {distance, bilinear} × positives ∈ {instance, supervised},
estimator fixed = NPLM, marginal fixed = global SIGReg.  Report probe
mean±sd, and the **sd** as the primary statistic.  Also log the empirical
critic spread `s = sd(g_ij)` on reference pairs each epoch — that is the
theory's actual predictor and it is free to record.

**Prediction.**  sd is governed by the critic column, not the positives
column: bilinear ~0.04, distance ~0.01, independent of positives.  Further,
`s` should track sd across all four cells.

**Falsifier.**  sd splits by positives instead of critic (→ the variance is
about supervision, not the exponential, and App. A's §A.3 argument is wrong as
an explanation of this data).

**Cost.** ~20 runs × 20 ep on C100 32-D — the cheapest test in this document.
**Payoff if confirmed.** A one-line rule: *never use the bilinear critic under
NPLM*, and the label-free calibrated arm becomes usable.

### Exp 82 — Calibration residual as a free seed-selection and early-stop signal

> **DONE 2026-08-20 — FALSIFIED.**  Spearman(|resid|, probe) +0.30/-0.10 against a predicted < -0.6, and best-of-5-by-residual *underperforms* the random mean (0.839 vs 0.891).  The falsifier branch fired as guessed: the variance lives in the SIGReg/interaction balance.  Structural cause feeds exp 83.

**Motivation.**  App. A eq. (10): the total NPLM critic gradient is exactly
`E_ref[e^g] - 1`, the calibration residual.  This is computable at zero extra
cost during training and **requires no labels and no validation set**.  If bad
seeds are bad because they never calibrate, the residual identifies them.

**Protocol.**  Re-run the exp-61 5-seed C100 `nplm_bilinear` cells, logging
`E_ref[e^g] - 1` per epoch alongside the final probe.  Correlate
final-epoch |residual| against probe.  Then test the operational version:
*select* the seed with the smallest terminal residual out of k and report the
resulting probe against the mean of k random seeds.

**Prediction.**  |residual| anti-correlates with probe (Spearman < -0.6).
Best-of-5 by residual should recover much of the 0.855 → 0.935 gap that random
seeding averages away — a genuine performance gain with zero label cost.

**Falsifier.**  Residual converges to ~0 for every seed regardless of probe
(→ calibration is achieved by all runs and the variance lives elsewhere,
probably in the SIGReg/interaction balance; still a useful negative).

**Cost.** ~5 runs, or free if folded into exp 81.
**Payoff.** An unsupervised model-selection criterion for the whole NPLM
family — and, if it works, it should be added to the standard battery as a
reported column.

### Exp 83 — Variance-reduced NPLM: bound the exponent instead of clamping it

**PROMOTED to the top of the queue by the exp-82 result.**  What began as a
variance fix is now a correctness fix, and it is the single most important
pending experiment in this document.

**Motivation.**  Exp 82 exposed a structural defect, not just a failed
heuristic.  For the distance critic as implemented, `g = -½||z-z'||²/τ ≤ 0`,
so `e^g ≤ 1`, so `E_ref[e^g] ≤ 1` **always**: the calibration condition
`E_ref[e^g] = 1` is *unreachable*, and the residual is structurally negative
for every seed.  **A bare distance critic cannot calibrate at all.**  Since
calibration is the entire justification for preferring NPLM over a softmax
estimator, this undercuts the paper's central claim for exactly the arms
(`nplm_sup_dist`, `nplm_distance`) that carry its best per-event and
kernel-test numbers — their strength is currently evidence of good *scaling*,
not of achieved calibration.

The fix is the bias term the theory always specified and the implementation
omits: `g = -½||z-z'||² + b(x) + b(x')`, which restores the positive range.
A second, independent candidate: (b) **self-normalized importance weighting**,
tracking a running estimate of `E_ref[e^g]` and rescaling — a *known constant
shift*, therefore correctable rather than gauge-destroying, unlike
max-subtraction.

**Protocol.**  C100 32-D, 5 paired seeds, five arms: current bilinear+clamp;
current distance (control, expected residual < 0 always); distance+bias;
distance+bias+running-normalizer; bilinear+running-normalizer.  Report probe
mean±sd, **the calibration residual `E_ref[e^g] - 1` and whether it reaches
0**, per-event power, and `sd(e^g)` (the corrected variance diagnostic — exp 81
showed `sd(g)` anti-correlates).

**Prediction.**  Distance+bias reaches residual ≈ 0 where bare distance cannot,
and its per-event power *improves* because thresholds become genuinely
transferable.  Variance drops ≥2× at equal-or-better mean probe.

**Falsifier.**  Variance drops but per-event power drops with it → the
normalizer has silently removed the absolute scale, i.e. we have re-derived
InfoNCE the hard way.  **This falsifier is the one to watch**; report per-event
power on every cell.

**Cost.** ~20 runs.

---

## Tier 2 — attacking the standing records and the one reliable failure

### Exp 84 — Two-stage recipe under the strict open-world protocol

**Motivation.**  `AIRCRAFT_MASTER_TABLE.md` reading #8 flags a two-stage recipe
(NPLM-sup trunk ft → supcon_sigreg head) with the campaign's best aircraft
numbers: LeJEPA-ft + supcon_sigreg 0.812/0.638/0.800, VISReg-ft nplm_sup_dist
mahaT 0.812.  **But those trunks were fine-tuned on images and labels of all
100 variants including the holdouts**, so the novelty columns are optimistic
and the result is currently uncitable.  It is the single most promising
uncited number in the program.

**Protocol.**  Rerun the exp-62 trunk ft with holdouts 90–99 excluded from the
corpus (exp-70 protocol), then the exp-51 8-arm head suite on the resulting
banks.  Compare against the exp-71 residual champion on the same base (0.863
visreg).

**Prediction.**  The gains shrink but survive on LeJEPA/VISReg (the weak trunks
that had the most to gain), and the two-stage recipe lands between the exp-70
parents and the exp-71 residual champions.

**Falsifier.**  Gains vanish entirely → the effect was contamination, and
reading #8 should be struck from the master table.  Either outcome is worth
having; right now the table carries a caveat instead of a number.

**Cost.** 3 trunk fts + 24 head runs. Moderate.

### Exp 85 — Iterated residuals (does the residual trick stack with itself?)

**Motivation.**  Residual fine-tuning is the campaign's largest and most
seed-stable effect (12/12 cells, cars +0.148±0.004 over 3 seeds, every dataset
record).  It has only ever been applied **once**.  Exp 77 shows the residual
child leaves 5–10% genuinely new variance on fine-grained data
(ridge R² 0.91–0.95, mutual-kNN ~0.2) — which is precisely the condition under
which a *second* residual should still have something to extract.

**Protocol.**  On the exp-71 champions for cars/visreg and flowers/dino (the
two biggest residual gains), train a second residual head on `r₂ = r₁ - mean_y(r₁)`
and evaluate the 3-way concat `[z; r₁; r₂]`.  Control for width: also run a
parent + single-residual-of-double-width, so that any gain is attributable to
the *decomposition* and not to parameter count.  3 seeds.

**Prediction.**  Diminishing but positive: +0.01–0.03 on fine-grained cells,
~0 on coarse.  The width control is what makes this interpretable.

**Falsifier.**  The double-width single residual matches the 3-way concat →
residual fine-tuning is a capacity effect, not a decomposition effect.  That
would be a significant reinterpretation of the paper's §5 and is worth knowing.

**Cost.** ~6 e2e ft runs + controls. Moderate.

### Exp 86 — Fix aircraft discovery by freezing the probe directions

> **DONE 2026-08-20 — CONFIRMED, and sharper than proposed.**  Freezing the parent half only works on DINO; freezing EVERYTHING keeps 90-99% of the per-event gain at exactly zero probe cost, with round-2 purity no longer collapsing (0.176-0.346 vs 0.023-0.065).  Aircraft record 0.8634 retained WITH per-event 0.523.  Generalization beyond aircraft is exp 101.

**Motivation.**  `SPACE_GEOMETRY.md` states the aircraft failure explicitly:
purity is fine (0.32–0.53) yet discovery still costs probe (−0.012 to −0.037)
while buying geometry — and exp 79 showed LID pooling cannot help because LID
separation is weak exactly there, concluding "the aircraft failure needs a
different lever."  The paper's diagnosis (§5) is that on fine-grained data the
discovery ft **erodes the res-nplm concat's probe directions**.  That diagnosis
implies its own fix: do not let it.

**Protocol.**  Discovery on the exp-72 aircraft cells with the parent half of
the concat **frozen**, fine-tuning only the residual half (and a variant
freezing both and training only new anchors).  Compare against exp-72's
unfrozen numbers on probe, mahaT and per-event.

**Prediction.**  Probe holds at the pre-discovery record (0.863) while the
geometry/per-event gains that exp 72 already demonstrated (visreg per-event
0.30–0.50, mahaT +0.017) survive — i.e. the discovery step becomes
strictly non-negative for the first time on fine-grained data.

**Falsifier.**  The geometry gains disappear along with the probe loss → the
two were the same update and the trade is not separable.  This is the single
cleanest test of the paper's "choose by consumer" claim.

**Cost.** ~6 discovery runs. Cheap — the parents already exist.

### Exp 87 — LID on the residual half

> **DONE 2026-08-20 — FALSIFIED on flowers.**  Residual-half LID is uniformly worse than the parent half (0.923/0.825/0.733 vs 0.951/0.893/0.954): the signal is a CLASS-level dimensional anomaly, not within-class.  Half-holds on cars (visreg 0.752, best cars LID on record) but eucl 0.747 matches it, so no regime flip.  Mechanism follow-up is exp 103.

**Motivation.**  Two of the campaign's strongest results have never been
combined.  LID is a *local dimensionality* statistic and hits 0.951–0.955 on
flowers; the residual half is by construction where **within-class variation**
lives (exp 76: the plain-res residual half scrambles class semantics entirely,
purity 0.365).  LID computed on the parent half is measuring the wrong
geometry.

**Protocol.**  On the exp-71/72 flowers and cars champions, compute LID(k=20)
separately on the parent half, the residual half, and the concat.  Also rerun
the exp-78 holdout-rotation control on the residual half.

**Prediction.**  Residual-half LID ≥ concat LID > parent-half LID on flowers,
and — the interesting case — residual-half LID beats the weak concat LID on
**cars** (0.545), where distance scores currently win.  Cars is the dataset
where novelty is on-manifold, which is LID's stated regime.

**Falsifier.**  Residual-half LID is uniformly worse (plausible: the residual
half is lower-SNR).  Cheap enough to be worth the risk.

**Cost.** Evaluation only, no training. **Hours, not days — do this first.**

---

## Tier 3 — regime rules and protocol debt

### Exp 88 — Classwise SIGReg where the anchors are actually realizable

> **DONE 2026-08-21 — BOTH CLAUSES FALSIFIED, and the control stole the show.**  cent->anchor NEVER collapses: 3.46/3.47/3.51 at 100/128/200-D, identical to the 32-D stall, across all nine runs.  The anchors are unreachable for an OPTIMIZATION reason (the equilibrium parks centroids ~3.5 out regardless of feasibility) -- the falsifier branch this entry called 'more interesting'.  Classwise mahaT DECLINES with dim (0.468 -> 0.326), so the archived 100-D 0.463 was a fair draw, not a door.  UNPREDICTED: the softmax control (supcon_sigreg) posts mahaT 0.545-0.558 at 100-128-D while holding probe 0.90-0.91 -- ABOVE the 0.47-0.49 C100 ceiling.  Q4 restated in the paper: dimension is neither the enabler nor the fix.

**Motivation.**  `QUESTIONS.md` Q4: classwise SIGReg is strictly better iff
anchors are realizable (`dim ≥ n_classes`), and harmful otherwise — 100 anchors
in 32-D or 64-D stall at cent→anchor ≈3.5.  The boundary case *has* been run
once: `nplm_dist_sup_cw cw-lam1` at 100-D scores probe 0.8440 / **mahaT 0.463**
— the best mahaT of any 100-D row and close to the program's C100 calibration
ceiling (~0.47–0.49), while its 32-D counterpart sits at 0.372.  That is one
datapoint pointing the way the rule predicts, and it has never been followed
up.  Two gaps: cent→anchor was **never reported at 100-D**, so we do not
actually know realizability was achieved rather than merely dimensionally
possible; and the strict case `dim > n_classes` (128-D, 200-D) is untested.

**Protocol.**  C100 classwise arms at 100-D, 128-D and 200-D against
global-SIGReg and softmax controls at matched dim, 3 seeds.  **Report
cent→anchor at every dim** — that is the actual independent variable, and its
absence is why the existing 100-D row cannot be interpreted.

**Prediction.**  cent→anchor collapses from ~3.5 (32/64-D) toward ~0 once
`dim ≥ n_classes`, and mahaT rises with it, breaking the C100 calibration
ceiling.  The probe stays low — this is a calibration-side test, and proposing
it as a probe win would contradict the paper's own dissociation.

**Falsifier.**  cent→anchor collapses but mahaT does not move → realizability
is necessary and not sufficient, and Q4 needs restating.  Alternatively
cent→anchor stays high at 100-D → the anchors are unreachable for an optimization
reason rather than a geometric one, which would be a more interesting finding
than the original hypothesis.

**Cost.** ~15 runs.  **The most likely source of a new C100 calibration
record**, and it repairs a diagnostic gap in the existing table either way.

### Exp 89 — CIFAR-100 discovery rate unblocking

**Motivation.**  Long-standing open item.  C100 discovery is rate-blocked, not
quality-blocked: 500 holdout images inside a ~2500-event tail gives purity
0.003–0.013.  Never tested: stricter `tau_quantile` (0.995+), multi-class
holdouts, larger holdout fraction.

**Protocol.**  Grid `tau_quantile ∈ {0.95, 0.99, 0.995}` × holdout size
∈ {1, 5, 10 classes} on the exp-73 concats.  Purity is the primary readout;
probe pre/post secondary.

**Prediction.**  Purity scales roughly with holdout fraction and with quantile
strictness; the probe only moves once purity clears ~0.15–0.3 (the gate
observed on flowers/dtd).  This test mostly **calibrates the gate** rather than
setting a record, which is its value: the gate threshold is currently inferred
from four datasets and never measured directly.

**Falsifier.**  Purity rises past 0.3 and the probe still does not move → the
gate is not the mechanism on C100 and something dataset-specific is blocking.

**Cost.** ~9 discovery runs. Cheap.

### Exp 90 — A predictor for *which* novelty score to use

> **DONE 2026-08-20 — FALSIFIED.**  None of the three label-free tail diagnostics tracks the LID-eucl gap (Spearman +0.07/+0.09/-0.16; best sign-accuracy 12/17 vs an 11/17 always-say-LID base rate).  The regime rule stays empirical.  Side-finding: LID wins 12/17 champion cells, and on cifar100 res-nplm-cat it is the only usable score (0.738 vs eucl 0.356).

**Motivation.**  The paper's LID section currently ends on an unsatisfying
note: LID is superb on flowers (0.95), useless on cars (0.545), and we can only
say the regime is "a dataset property."  For a practitioner that is not
actionable.  But the stated mechanism — holdouts sitting **on-manifold**
(small distances, mediocre eucl) vs **off-manifold** — is itself measurable
*without novelty labels*, e.g. via the ratio of holdout-to-nearest-seen-centroid
distance against the seen-class radius, or the TwoNN ID gap between pooled tail
and seen population.

**Protocol.**  Across all 12 transfer cells plus CIFAR, compute candidate
unsupervised predictors and regress them against the measured (LID − eucl) AUC
gap.  Deliverable: a single diagnostic that tells you which score to threshold
on, before you have labels.

**Prediction.**  The tail-vs-seen ID gap predicts the sign of (LID − eucl) with
≥10/13 accuracy.

**Falsifier.**  No candidate predictor separates flowers from cars → report
that honestly; the regime rule stays empirical.

**Cost.** Evaluation only.  Cheap, and the highest-value *paper* contribution
here — it converts a caveat into a method.

### Exp 91 — Multi-seed the uncited records

> **DONE 2026-08-21 — done, and milder than predicted.**  aircraft 0.866+-0.002, galaxy10 0.971+-0.004, dtd 0.867+-0.004, flowers 0.887+-0.019, cars 0.833+-0.017.  The archived aircraft and dtd numbers were the LOW draws (seed 2 sets a new dtd best); only flowers falls materially, and its pre-discovery spread 0.787-0.885 is the campaign's widest -- discovery compensates weak parents (+0.106 on the weakest seed vs +0.021 on the strongest), so part of its apparent benefit is variance reduction.  NO ordering flipped; all records now citable.

**Motivation.**  Protocol debt.  Exp 75 multi-seeded cars/visreg and CIFAR and
found one objective ranking **flipped** under averaging (C100 plain-res vs
res-nplm) and that several single-seed peaks were fair-but-high draws.  The
remaining records — aircraft 0.863, flowers 0.906, dtd 0.862, galaxy10 0.975 —
are still single-seed.

**Protocol.**  3 paired seeds on each of the four record cells.  Report
mean±sd and update the master tables to cite records with uncertainties.

**Prediction.**  Records shift down by 0.01–0.03; the *ordering* of
constructions holds.  No new performance, but every claim in the paper becomes
citable.

**Falsifier.**  An ordering flips (as it did on C100) → the corresponding
verdict in `AIRCRAFT_MASTER_TABLE.md` needs rewriting.

**Cost.** 12 e2e ft runs.  Boring, unavoidable, do it before any submission.

---

## Tier 4 — SparKer as a loss, not just a test

The paper's §5 establishes that SparKer's NP loss
`sum_R w(e^f - 1) - sum_D f` **is** the NPLM contrastive objective with pairs
replaced by events.  The campaign has only ever used it on the right-hand side
of that identity: as a post-hoc statistic on frozen embeddings.  Everything
below follows from taking the left-hand side seriously.

Two structural facts drive the tier.  (i) SparKer's `M=16` kernel centres
`mu_i` are **trainable and initialized from the data sample** — SparKer is
already doing a clustering, but placing centres by *density ratio* rather than
by distance.  (ii) By the Neyman–Pearson lemma the learned `f` is the *most
powerful* novelty score available at any threshold, which makes it the
principled choice for the one job we currently do with a distance quantile.

These are more speculative than Tiers 1–3 and are labelled where so.

### Exp 92 — SparKer centres as the discovery clustering

> **DONE 2026-08-20 — PARTIALLY CONFIRMED, plus an unpredicted win.**  Cars round-1 purity rises 0.161 -> 0.214 (+33% rel) exactly as the density-ratio mechanism predicts, but stays below the ~0.3 gate: an improvement, not the rescue.  The headline nobody predicted: SparKer pooling makes round-2 purity immune to collapse in every cell (0.21-0.62 vs 0.06-0.24) AND proposes the anchors.  Cost: purer pools feed more ft pressure, so the aircraft probe cost worsens -- which motivated 92b (frozen space), where the combination dominates outright.

**Motivation.**  The campaign's cleanest negative result is that Cars pools at
purity ≤0.14 because novel car models are *not geometrically outlying* — they
sit among the seen classes, so a distance quantile cannot reach them.  But
"not outlying in distance" does not imply "not outlying in density ratio":
a region can have p_data/p_ref ≫ 1 while sitting at perfectly ordinary
distance from the nearest centroid.  SparKer's gated kernels place `mu_i`
exactly where that ratio is large.  This is the first proposal in the document
with a mechanism that *predicts a fix for the Cars failure* rather than
working around it.

**Protocol.**  Replace steps (ii)–(iv) of the discovery loop (quantile pool →
BIC k-means → merge) with: fit SparKer on (unlabeled corpus vs seen-train
reference), take the `M` trained centres, keep those whose local density ratio
`f` exceeds a threshold, merge as usual.  A/B against the distance loop on the
three cells that span the purity range: cars/dino (0.14, fails), aircraft/visreg
(0.525, purity-adequate but probe-negative), flowers/dino (0.61, works).
Primary readout is **round-1 purity**, not probe.

**Prediction.**  Purity rises most on cars — the cell where distance pooling is
mechanistically wrong and density ratio is not.  On flowers, where distance
pooling already works, expect parity.

**Falsifier.**  Cars purity stays ≤0.15 → fine-grained novelty is invisible to
the density ratio too, not merely to distance, and the "novelty is not
outlying" verdict generalizes from geometry to likelihood.  That is a stronger
and more useful negative than the one we currently report.

**Cost.**  SparKer fits are cheap (300 steps, M=16) and the parents exist;
~9 discovery runs.  **Highest-value test in this tier.**

### Exp 93 — The NP score as the pool scorer (dist vs LID vs f)

> **DONE 2026-08-21 — FALSIFIED as stated, but np owns the regime that matters.**  On flowers (LID's regime) lid beats np by 0.03-0.06 in all three cells: the NP lemma is asymptotic and loses to a closed-form ratio statistic at these pool sizes -- precisely the falsifier this entry named.  Below the gate (cars) np is the BEST r1 scorer (0.214 vs dist 0.161, lid 0.133).  Rule: lid on-manifold, np below the gate, either one in a frozen space.

**Motivation.**  Exp 79 already showed the pool scorer matters and that
scale-free LID strictly dominates distance.  The NP lemma says the learned `f`
should dominate *both*: it is the optimal statistic for exactly this decision.
This is the natural third arm of a comparison that is currently two-armed.

**Protocol.**  Three-way A/B (`pool_score` ∈ {dist, lid, np}) on the exp-79
flowers cells plus one cars cell, identical recipe and seed.  Report round-1
and round-2 purity and the post battery.

**Prediction.**  `np` ≥ `lid` > `dist` on purity in every cell.  Following exp
79, expect the *probe* to be unmoved on flowers (above the gate purity is not
binding) — the interesting cell is cars, below the gate.

**Falsifier.**  `np` underperforms `lid`.  The likely reason would be
estimation variance: `f` is fitted from finite samples whereas LID is a
closed-form ratio statistic, so the NP lemma's optimality is asymptotic and may
not survive at these pool sizes.  Worth knowing either way.

**Cost.**  ~8 discovery runs.

### Exp 94 — Null validity when the encoder has seen the test data

> **DONE 2026-08-20 — FALSIFIER FIRED, and that is the good outcome.**  Realized FPR at nominal 0.05 is at or below nominal in every regime (cars/dino 0.040 frozen, 0.040 after full ft; flowers 0.000 throughout).  The discovery ft cannot manufacture false positives at these sample sizes, so the campaign's nulls are VALID and split-disjointness is not required.  Exp 95 is formally cleared -- though its directly adversarial objective should rerun this check on its own encoder.

**Motivation.**  A validity guard, and the prerequisite for citing anything in
this tier.  Every power number in the campaign calibrates its null on
anomaly-free toys **using an encoder trained on that same corpus**.  For the
frozen-trunk batteries that is defensible.  The moment the embedding is trained
with a novelty-seeking objective (exps 92, 95) it is not: the encoder can
manufacture separation on pure-background data, the null distribution shifts,
and the test becomes anticonservative.  This is the standard data-snooping
failure and it would silently invalidate results rather than announce itself.

**Protocol.**  Measure the realized false-positive rate at nominal α=0.05 on
**pure-background** toys (no injected signal) under three regimes: frozen
encoder (the current protocol); encoder fine-tuned on the full corpus; encoder
fine-tuned on a disjoint split from the one the test uses.  Report FPR with
Clopper–Pearson intervals.

**Prediction.**  Regime 1 ≈ 0.05 (the current numbers are safe); regime 2
> 0.05, possibly badly; regime 3 ≈ 0.05, establishing split-disjointness as
the required protocol for any trained-embedding test.

**Falsifier.**  Regime 2 also gives 0.05 → the encoder cannot manufacture
separation at these sample sizes and the concern is theoretical.  Cheap
insurance either way; **run this before exp 95, not after.**

**Cost.**  Toy-level only, no training beyond one extra ft.  Cheap.

### Exp 95 — SparKer as a differentiable training objective  *(speculative)*

> **CONTRAINDICATED 2026-08-21 by exp 98.**  The alternating scheme was tested at discovery-ft scale and failed on every column including the statistic it optimizes.  Do not run as specified; a redesign would need an explicit class-structure-preserving term.

**Motivation.**  The big swing.  If `t_NP` is the quantity we ultimately care
about, optimize it directly: make `z` trainable inside the SparKer fit and
maximize the detection statistic end to end, instead of training a proxy
objective and hoping the statistic follows.  The unification of §5 says this is
not a new loss, only a new *scale* at which to apply the existing one.

**Protocol.**  Alternating optimization: inner loop fits `(mu, a)` on the NP
loss with `z` fixed; outer loop takes gradient steps on the encoder to increase
`t_NP` between the unlabeled pool and the seen reference.  Anneal σ as usual.
Compare against the exp-71 residual champion and the proto/repulse discovery ft
on cars/visreg and galaxy10/dino.  **Requires exp 94 to have established a
valid null**; report power against a null recalibrated with the *trained*
encoder, never the frozen one.

**Prediction.**  Genuine per-event and SparKer power gains, with a probe cost
— it is a pure-detection objective and by the dissociation should behave like
the NPLM-ft arm of exp 58 (best on every power statistic, worst on probe).

**Falsifier — and the one to watch.**  `t_NP` rises while the *recalibrated*
power does not.  That is the signature of the encoder overfitting the test
statistic: the model separates the particular data sample rather than the
population, inflating the statistic on both signal and null toys alike.  If
that happens the approach is dead in this form and should be reported as such,
not rescued by tuning.

**Cost.**  Highest in the document; needs a new trainer.  Do not start it until
92–94 have reported.

### Exp 96 — Does the pairwise critic warm-start the event-level test?

> **DONE 2026-08-21 — FALSIFIED in the strong form.**  Warm-starting SparKer's centres at the trained class centroids is neutral-to-HARMFUL (C10, f=0.02, 50-step budget: nplm_sup_dist 0.800 cold vs 0.560 warm); full-budget trajectories indistinguishable.  Mechanism: class centroids sit where the density ratio is ~1, so warm centres must migrate AWAY, while data-sample init sometimes seeds a centre on a novel event.  As this entry required, the paper's section-5 claim has been weakened in print: the two scales share the functional form and loss, NOT transferable learned content.

**Motivation.**  A direct empirical test of the paper's central claim.  If the
NPLM critic learned during representation training and the SparKer `f` learned
at test time really are the same object at two scales, then the former should
contain what the latter re-learns from scratch.

**Protocol.**  Initialize SparKer's centres `mu` at the trained NPLM class
anchors (and `a` from the critic scale) versus the default data-sample init.
Compare `t_NP` trajectories at matched steps and power at fixed budget, on the
CIFAR-10 NPLM cells where both objects exist.

**Prediction.**  Warm start converges in materially fewer steps at equal or
better power — the cheapest available evidence for the §5 unification, and a
free speedup if it holds.

**Falsifier.**  No convergence difference → the two scales share a functional
form but not learned content, which would weaken the paper's framing and should
be said plainly in §5.

**Cost.**  Cheap; evaluation plus a modified init.

### Exp 97 — M and the bandwidth schedule versus intrinsic dimension

> **DONE 2026-08-21 — half confirmed, half INVERTED.**  sigma_ratio is flat within toy noise everywhere: the anneal is genuinely robust and the exp-57 lesson was about sigma0 matching.  M is not flat and moves OPPOSITE to the prediction -- high-ID supcon (ID~12) wants FEWER kernels (0.60 at M=4 vs 0.24 at M=64; more kernels overfit the null in high dimension), low-ID nplm-sup (ID~3) mildly prefers more.  Default M=16 leaves up to 0.16 power on the table.  Rule: match M INVERSELY to intrinsic dimension.

**Motivation.**  Protocol debt for the new loss.  `M=16` kernels and a fixed
σ0→σ0/10 anneal have never been scanned, yet exp 77 measured intrinsic
dimensions spanning 2–13 across the arms, and a kernel model's resolution
requirement scales with dimension.  The exp-57 lesson (fixed σ goes blind on a
rescaled space) says the *schedule* matters; it does not say the current
schedule is right.

**Protocol.**  Scan `M ∈ {4, 16, 64}` × `sigma_ratio ∈ {3, 10, 30}` on three
cells chosen to span TwoNN ID (nplm-sup-ft ≈2–3, sigreg ≈5–7, supcon ≈9–13).
Report power at fixed fraction, and the σ-checkpoint at which the statistic
peaks.

**Prediction.**  The required σ range widens with intrinsic dimension, and
`M=16` under-resolves when several novel classes are present — the SparKer
analogue of the BIC fragmentation discussed in §5.2.

**Falsifier.**  Power is flat in both knobs → the annealed schedule is
genuinely robust and the current default needs no caveat, which is worth being
able to state.

**Cost.**  ~27 battery runs on cached embeddings, no training.

### Exp 98 — SparKer-ft as the discovery fine-tune objective

> **DONE 2026-08-21 — FAILS, beyond the stated falsifier.**  SparKer-ft is worse than both proto and NPLM ft on EVERY column including the event-level statistic it optimizes (post SparKer power ~0.1 vs 1.00).  Statistic-chasing: pseudo-novel points climb the current f surface, class structure smears, and the refitted kernel finds less real separation.  With exp 96 this bounds the paper's section-5 claim in both directions -- the NP loss transfers as a FORM, but neither its parameters nor its optimization transfer.

**Motivation.**  Exp 58 established the ft-objective split: proto/repulse wins
the probe, NPLM+sigreg wins every power statistic.  SparKer-ft is the missing
third corner, and unlike the other two it optimizes the *event-level*
statistic that the battery actually reports.

**Protocol.**  Third arm alongside proto and NPLM-ft on the CIFAR-10 discovery
cells, same rounds and epochs.  Full battery, and pool purity by round.

**Prediction.**  Power comparable to or better than NPLM-ft, probe worse than
proto — and, the interesting part, better **round-2 purity retention**, since
an event-level objective has no reason to inflate the space the way the
prototype fine-tune does (the inflation that broke distance pooling in exp 79).

**Falsifier.**  Round-2 purity degrades like proto's → the inflation is a
property of any discovery fine-tune, not of the prototype objective
specifically.

**Cost.**  Moderate; reuses the exp-58 harness.

---

## Tier 5 — implied by the 80/81/86/87/90/99 results (exps 100+)

Four tests that did not exist before the last batch reported.  The first two
are the highest-value runs in the whole document.

### Exp 100 — Dense fraction scan: turn `f95` from a bound into a measurement

**Motivation.**  Exp 99 inverted every archived power curve into a discovery
reach `f95` (the injected fraction at which SparKer power crosses 0.95) and
found that **the fraction grid, not the spaces, is the limiting factor**.  The
transfer grid is {0.003, 0.01, 0.02, 0.05, 0.1}; power typically climbs from
~0.1 to ~1.0 across the single interval [0.05, 0.1], so of 106 spaces only
**four** have a cleanly bracketed crossing.  Every other number in the reach
table is a bound, not a measurement.  This is the binding constraint on the
question we most care about — sensitivity at small signal fractions.

**Protocol.**  Dense scan `f ∈ {0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10}` on
the champion space of each of the 17 cells, existing 50 toys, annealed sigma,
cached embeddings.  **No retraining.**  Report `f95` with Clopper-Pearson
bands.  Optionally add `f ∈ {0.005, 0.01, 0.015}` on CIFAR-10, the only cell
whose current grid already brackets its crossing (`f95 = 0.019`).

**Prediction.**  Every currently-starred cell resolves to a crossing bracketed
within ≤0.02, and the dataset ordering by reach becomes citable.  Flowers stays
`>0.1` on all bases.

**Falsifier.**  Curves turn out non-monotone or shallow enough in [0.02, 0.10]
that even a dense grid cannot bracket 0.95 → the reach statistic is the wrong
summary for these spaces and power-at-fixed-`f` should be retained instead.

**Cost.**  Battery only, ~17 cells × 7 fractions.  Cheap, no training.
**Do this first.**

### Exp 101 — Does frozen-space discovery generalize beyond aircraft?

**Motivation.**  Exp 86 is the sharpest operational result the campaign has:
discovering anchors in a *completely frozen* space keeps 90–99% of the
per-event gain at exactly zero probe cost, keeps round-2 purity from
collapsing, and makes the discovery step strictly non-negative — so the
aircraft record 0.8634 is retained *with* per-event 0.523.  It was tested on
**aircraft only, three cells**.  If it generalizes it is not an aircraft fix,
it is the default recipe, and the paper's "choose by consumer" framing weakens
much further than currently stated.

**Protocol.**  Freeze-both discovery on the exp-71/72 champions of all
remaining cells — cars, flowers, dtd, galaxy10 × 3 bases — plus the CIFAR-10/100
concats of exps 73/74.  Compare against the archived unfrozen numbers on
probe, mahaT, per-event and round-2 purity.  Same recipe and seeds.

**Prediction.**  Probe cost goes to ~0 everywhere; per-event retention is
high on fine-grained cells (cars, flowers) and *lower* on coarse ones
(galaxy10, CIFAR-10), where the space update plausibly does real work because
the novelty is genuinely outlying and the anchors alone cannot capture it.

**Falsifier.**  Frozen discovery also zeroes the per-event *gain* on some
cells → the anchors-vs-update split is aircraft-specific, and exp 86 is a
regime result rather than a recipe.

**Cost.**  ~15 discovery runs, no retraining of parents.  **Highest-value
training-adjacent test.**

### Exp 102 — Is the residual child a standalone detector?

**Motivation.**  Exp 80's surprise: the res-nplm residual *child* out-detects
its own parent (aircraft/visreg 0.82@0.05 vs 0.44; cars/visreg 0.96@0.05).
Every deployment in the campaign uses the child only inside a concatenation,
where the parent supplies the probe.  If the child alone is the better
detector, a detection-only pipeline could drop the parent entirely — halving
the embedding and removing the supervised half.

**Protocol.**  Full battery on the residual halves alone across the 12 cells:
probe, eucl, mahaT, per-event, SparKer `f95` (exp 100 grid), plus the exp-76
semantic metrics — the last matter because exp 76 showed the residual half
*scrambles* class semantics (purity 0.365), so a detector built on it may be
undiagnosable even when it fires.

**Prediction.**  Child-alone matches or beats the concat on dataset-level
power at a large probe cost, and is semantically illegible.  The useful
deliverable is the explicit trade curve, not a winner.

**Falsifier.**  Child-alone power is an artifact of concatenation scaling
(i.e. it does not survive being scored on its own) → the exp-80 reading is
wrong and the concat is doing the work.

**Cost.**  Evaluation-only on cached banks.  Cheap.

### Exp 103 — Where does the LID class-level signal actually live?

**Motivation.**  Exp 87 falsified the within-class story: flowers LID lives in
the *parent* (class-structure) half, so novelty there is a **class-level**
dimensional anomaly.  We now have a mechanism claim with no direct test, and
exp 90 showed we cannot yet predict the regime from tail diagnostics.  A
sharper probe of the mechanism is the remaining route to making the LID regime
rule predictive rather than descriptive.

**Protocol.**  Decompose LID by neighbourhood composition on flowers vs cars:
for each holdout point, split its `k=20` neighbours into same-seen-class vs
mixed-class sets and recompute the Levina-Bickel estimate on each.  If the
signal is class-level, holdout LID should be driven by *mixed-class*
neighbourhoods (the point sits between seen classes) and the effect should
vanish on cars.

**Prediction.**  Flowers: mixed-neighbourhood LID carries the discrimination,
same-class LID does not.  Cars: neither does, consistent with its `0.545`.

**Falsifier.**  Both components discriminate equally → "class-level" is not
the right description and the mechanism remains open.  Report it as such;
we have now had two failed mechanism stories here (within-class, and the exp-90
tail diagnostics) and a third failure should be taken as evidence that LID's
regime is not simply characterizable.

**Cost.**  Evaluation-only.  Cheap.

---

### Exp 104 — The interpretability panel across every space

**Motivation.**  The program's stated ideal is that distance to a labelled
anchor *is* a delta log-likelihood in a hypothesis test — interpretable and
discriminating at once — and we have never measured it.  Parts A and B report
ten metrics that each capture a shadow of this; none states it directly, which
is why alignment has been hard to gauge.  `104_interpretability_panel.py`
collapses it to six numbers with ideal value 1 (0 for ECE): `r_ll`, `slope`,
`r_llr`, `ece`, `sw` (component Gaussianity), `rms` (component width vs
sigma=1), plus `sep` for discrimination.  Definitions and the validation table
are in METRICS.md Part C and the paper §7.

**Status.**  Written and **validated on synthetic spaces** (`--selftest`: ideal
returns 0.99/1.01/0.00/0.98/1.00; width 2x drives slope to 0.267 = 1/sigma^2;
t_3 tails drive sw to 13.7 with slope ~1).  **Not yet run on campaign spaces** —
it needs the cached feature banks.

**Protocol.**  Run over every space in the exp-76/77 loaders: all 6 arms x 3
bases x 4 datasets, the exp-71 residual constructions, and the CIFAR cells.
Report the panel next to probe and per-event power in SUMMARY_TABLES.

**Prediction.**  SIGReg/NPLM arms sit near rms=1 with slope near 1; SupCon-family
spaces show slope far from 1 with `r_ll` still high — faithful ordering,
meaningless units, i.e. the probe/calibration dissociation expressed in
log-likelihood terms.  `sw` should track the TwoNN intrinsic-dimension
fingerprint of exp 77.

**Falsifier.**  SupCon spaces score slope ~1 too → distance is already a
log-likelihood without the marginal constraint, and SIGReg's contribution is
not what we claim.  Alternatively every space scores sw >> 1 → the labelled
components are nowhere near Gaussian and the whole Eq.-(dll) framing is
aspirational rather than achieved.

**Cost.**  Evaluation-only, no training.  **This is the measurement the paper
most needs.**

---

## Tier 6 — what the completed campaign opened (exps 105+)

The queue from Tiers 1-5 is empty: everything specified has run, except exp 95
(contraindicated).  What follows are the questions the results actually raised,
written to the same standard.

**A meta-lesson to design against.**  Four times now a theoretical defect has
been identified, fixed, and the fix has bought nothing: the calibration residual
did not select seeds (82), achieving calibration cost probe and per-event (83),
warm-starting the test from the representation hurt (96), and optimizing the
representation through the test destroyed the separation it measures (98).  The
campaign's most reliable pattern is that **repairing an identity is not the same
as improving a space.**  Exp 105 is another identity repair, so its watched
falsifier is that same pattern; we state it up front rather than discovering it
again.

### Exp 105 — A direct class-conditional width penalty

**Motivation.**  Exp 104's headline: no space achieves unit class-conditional
width (0.13-0.79 against a target of 1), so squared distances overstate
delta-logL by 4-37x and the paper's central identity holds only up to a large
unknown factor.  SIGReg pins the AGGREGATE marginal, which demonstrably does not
pin the conditionals.  The classwise marginal was supposed to, and exp 88 showed
it never reaches its anchors for an optimization reason.  But a *width* penalty
needs no anchors at all: regularise `E||z - mu_y||^2 / d` toward 1 directly,
with `mu_y` the running class centroid.  This is the smallest intervention that
targets the stated ideal, and it has never been tried.

**Protocol.**  C10 32-D and C100 100-D (the panel's quantitative cells), 5
paired seeds, arms: baseline supcon / supcon_sigreg / res-cat; each with and
without the width term at lambda_w in {0.1, 1, 10}.  Report the FULL panel
(r_ll, slope, r_llr, ece, sw, rms, sep) plus probe, per-event and f95.

**Prediction.**  rms -> 1 and slope -> 1 (nearly by construction); r_llr and ECE
improve materially; sep degrades somewhat as classes are forced to a common
width.

**Falsifier — the one to watch.**  rms reaches 1 and *nothing else moves*, or
probe/per-event fall as they did in exp 83.  That would establish that the
distance-as-log-likelihood identity is decorative in this program: satisfiable
on demand, and not the thing that makes a space work.  Given the meta-lesson
above this is a live possibility and arguably the modal outcome; it would be a
significant negative result and should be published as one.

**Second falsifier.**  rms -> 1 but slope does not follow → the departure is
anisotropy or shape, not scale, and the panel's `cond`/`sw` columns will say
which.

**Cost.**  ~30 short runs.  **The most direct test of the paper's stated ideal.**

### Exp 106 — Panel control: unfaithful space, or unestimable covariance?

**Motivation.**  A control on our own instrument.  Exp 104's transfer numbers
are shrinkage-dominated (10-40 samples/class in 100-D, cond 1e6-1e13), so we
reported them qualitatively.  We cannot presently tell "this space is
unfaithful" from "this covariance is unestimable at this sample size" — and the
paper's width claim would be much weaker if the transfer cells were an artifact.

**Protocol.**  (a) Subsample the CIFAR cells to 10, 20, 40 samples/class and
re-run the panel: how much of the transfer-cell degradation is reproduced on a
space we know is well-estimated?  (b) Re-run the transfer panel at head dim 16
and 32 as well as 100, where the covariance is estimable.  (c) Report the panel
under isotropic and diagonal reference densities as shrinkage bounds.

**Prediction.**  The CIFAR subsample reproduces most of the r_llr degradation
but NOT the width result: `rms` is a first-moment-style quantity and should be
stable at small n, whereas r_llr and ECE depend on the full covariance.  If so,
the width headline survives for the transfer cells and the correlation columns
do not.

**Falsifier.**  Subsampled CIFAR reproduces the width numbers too → the
0.13-0.79 widths are partly an estimation artifact, and the paper's central
negative needs qualifying.  **Run this before leaning further on exp 104.**

**Cost.**  Evaluation-only.  Cheap and it audits our own headline.

### Exp 107 — Is neighbourhood composition enough? (LID demotion test)

**Motivation.**  Exp 103's sobering coda: the bare fraction of a point's
neighbours that are not of the modal class scores 0.83-0.94 against LID's
0.87-0.96.  A parameter-free counting statistic nearly matches the ratio
estimator we have been recommending.  If it ties across the grid, the standard
battery should carry the simpler statistic.

**Protocol.**  Head-to-head across all 17 champion cells plus the CIFAR
concats: LID(k=20), neighbourhood composition, kNN-distance, `eucl`, and the
rank ensemble of composition+distance.  Report novelty AUC, and the exp-78
holdout-rotation control for the winner.

**Prediction.**  Composition ties LID within 0.02 on the on-manifold cells and
is no worse elsewhere; the composition+distance ensemble beats both, because
composition is explicitly scale-free and distance is explicitly not.

**Falsifier.**  Composition trails by >0.05 on flowers → LID's radial-ratio
information is doing real work beyond neighbour counting, and the extra
machinery is justified.

**Cost.**  Evaluation-only.  Cheap, and it could simplify the battery.

### Exp 108 — The on-manifold diagnostic, second attempt

**Motivation.**  Exp 90 tried to predict the LID-vs-distance regime from generic
tail diagnostics and failed (best 12/17 against an 11/17 base rate).  It used
features chosen before we understood the mechanism.  Exp 103 has since supplied
one: novelty sits ON a seen class by distance but OFF that class's local sheet,
and its neighbourhood is more mixed than a true member's.  Those are directly
measurable on unlabelled data and were not among the candidates exp 90 tried.
Since the on-manifold/off-manifold axis now decides six downstream choices
(§ the organizing axis), a working predictor is worth a second attempt.

**Protocol.**  Candidate label-free features: mean neighbourhood-composition gap
between the pool tail and the seen population; within-modal-class radial-ratio
dispersion; the ratio of these two.  Regress against the measured LID-minus-eucl
gap over 17 cells, as in exp 90.  Report sign accuracy against the same base
rate, and be explicit that this is a second attempt on the same 17 points.

**Prediction.**  Sign accuracy >= 15/17, materially above exp 90's 12/17.

**Falsifier.**  Still at or near base rate.  Then we stop: two mechanism-driven
attempts and one generic attempt having failed, the regime should be declared
empirically identifiable but not predictable, and the paper should say so
without a third try.  **Pre-committing to that stopping rule here.**

**Cost.**  Evaluation-only.

### Exp 109 — Density-ratio pooling on CIFAR-100

**Motivation.**  Exp 89 established that C100 discovery is geometry-blocked, not
rate-blocked: the distance pool is the wrong instrument because the extreme tail
is owned by background outliers.  Exp 92/92b established that the density-ratio
pool is immune to exactly the failure modes that cause this, and that in a
frozen space it costs nothing.  The combination has never been run on C100,
which is the campaign's one fully-blocked discovery dataset.

**Protocol.**  Exp-92b recipe (SparKer pooling, frozen space, anchors only) on
the exp-73 C100 concats, against the exp-89 distance grid at matched holdout
sizes.  Purity r1/r2 is the primary readout; probe and per-event secondary.

**Prediction.**  Round-1 purity clears 0.15 for the first time on C100 (against
the 0.121 ceiling of the distance grid), with round-2 holding rather than
collapsing.

**Falsifier.**  Purity stays below 0.15 → C100 novelty is invisible to the
density ratio as well as to distance, and the dataset is genuinely undiscoverable
by tail-pooling of any kind.  That is a clean and citable stopping point for the
C100 thread.

**Cost.**  ~6 discovery runs, no retraining.

### Exp 110 — Why did the softmax control break the C100 ceiling?

**Motivation.**  Exp 88's unpredicted headline is unexplained: plain
`supcon_sigreg` at 100-128-D posts mahaT 0.545-0.558 while holding probe
0.90-0.91, above the 0.47-0.49 band that had bounded every NPLM arm.  It is now
the best single-loss both-currencies C100 space on record and nobody designed
it.  We do not know whether the cause is the dimension, the SIGReg marginal, the
supervised positives, or an interaction.

**Protocol.**  Factorial at 32/64/100/128/200-D: {supcon, supcon_sigreg} x
{global sigreg on/off} x 3 seeds, full battery plus the exp-104 panel.  The
panel matters here: if the ceiling break is a width effect it will show up as
`rms` moving toward 1 with dimension.

**Prediction.**  The marginal is necessary (plain supcon does not break it) and
the effect grows with dimension up to ~128-D then saturates, tracking `rms`.

**Falsifier.**  Plain supcon breaks it too → the effect is dimensional and the
SIGReg marginal is incidental, which would further weaken the marginal's role
in this paper.

**Cost.**  ~30 runs.  Explains the best C100 space we have.

### Exp 111 — Child-only deployment study

**Motivation.**  Exp 102 showed the res-nplm child alone beats its concat's
reach on fine-grained VISReg/DINO cells (cars/visreg f95 0.049, the best
transfer reach measured anywhere) at a probe cost of only 0.005-0.034, and
stays semantically legible.  That suggests a deployable configuration nobody has
evaluated end to end: detect with the child, keep the parent only for
explanation.

**Protocol.**  Full pipeline on the three qualifying cells: child-only detection
(f95, per-event, SparKer), parent consulted only for the nearest-centroid
explanation of flagged events; measure the explanation quality (agree@1,
top-5 tables) and the total embedding cost against the concat baseline.

**Prediction.**  Equal or better detection at half the embedding width, with
explanation quality within 0.05 agree@1 of the concat.

**Falsifier.**  Explanations degrade materially when the parent is not part of
the scored space → the child's legibility depends on the concat context and the
split is not deployable.

**Cost.**  Evaluation-only on existing checkpoints.

---

## Tier 7 — the tuning audit (exps 112+)

**Should we redo things with more tuning?  Partly yes, and the reason is
specific.**  Exp 110 found that the best C100 both-currencies space in the
campaign came from a configuration slip: `tau` was never set, so the arm ran at
the loss-class default 1.0 instead of the inherited SupCon 0.1.  Neither knob
alone breaks the ceiling — it is a `tau x marginal` INTERACTION — which is
exactly the thing a one-at-a-time scan cannot see.

That points at a structural blind spot rather than a tuning oversight.  The
campaign's method has been to vary the DISCRETE corners of the design cube
(positives / critic / estimator / marginal) while holding the CONTINUOUS knobs
at inherited defaults.  The cube was swept exhaustively; the constants were
never swept at all.

But this is not an argument for re-running everything.  Note the contrast with
Tier 6's meta-lesson: the five failed interventions were all *identity repairs*
— theoretically motivated fixes.  `tau` is not a repair, it is an empirical
capacity knob, and it worked.  The distinction is worth respecting: sweep the
constants, don't re-derive the theory.  Tier 7 is therefore a scoped audit, not
a redo, and it is ordered so the cheap correctness checks come before any
retraining.

### Exp 112 — The inherited-constants audit

**Motivation.**  Before deciding what to re-run we should know what was never
chosen.  Cheap, and it scopes everything below.

**Protocol.**  Enumerate every continuous constant in the pipeline with its
value, its provenance (measured / inherited / arbitrary), and whether it was
ever swept.  Candidates already known: `tau` 0.1 for softmax arms (inherited,
NEVER swept — exp 110), `lam` sigreg weight (20 on c10, 1 on c100 — provenance
unclear), `n_slices` 256/64, NPLM `clamp` 30, SparKer `M` 16 (swept only at exp
97, and the answer inverted the default), `sigma_ratio` 10 (flat, exp 97), LID
`k` 20 (swept, exp 78), Mahalanobis `shrink` 0.1, `tau_quantile` 0.95 (swept
exp 89, distance only), BIC `kmax` 4, `merge_dist`, ft epochs/lr.  Deliverable:
a table in METRICS.md, and a shortlist of never-swept constants that sit under
load-bearing claims.

**Prediction.**  At least three more never-swept constants sit under headline
claims.

**Cost.**  Code archaeology.  Hours.  **Do first.**

### Exp 113 — Is the `tau x marginal` basin general or many-class?

> **DONE 2026-08-24 — PREDICTION HELD; the expensive falsifier did not fire.**  The basin is many-class only, and is a PLATEAU from tau>=0.3 rather than a knife-edge at 1.0 (C100 mahaT 0.238 -> ~0.506, probe 0.917-0.926, the only cells with nonzero per-event).  On C10 the same move buys 0.03 mahaT (noise) and collapses the probe 0.94 -> 0.64-0.67.  tau=0.1 is correctly tuned for few-class and under-tuned for many-class only -- the cells exp 110 already re-ran.  Unresolved: 0.502 here vs exp-110's 0.562+-0.037 at 100-D (~1.5 combined sd).

**Motivation.**  Exp 110 established the basin on C100; exp 105 reached it by an
independent route and noted C10 shows the OPPOSITE sign.  If the basin is
general it is a recipe change for the whole program; if it is many-class only it
is a regime rule.  This is the highest-value follow-up because it decides
whether every softmax comparison in the paper needs re-running.

**Protocol.**  `tau` in {0.05, 0.1, 0.3, 1.0, 3.0} x {marginal on, off} x 3
seeds on c10, c100, and two transfer cells (cars/visreg fine-grained,
galaxy10/dino coarse).  Full battery plus the exp-104 panel.

**Prediction.**  Many-class only: the basin appears on c100 and cars (196
classes) and inverts on c10/galaxy10 — i.e. it tracks the class-count axis of
§the regime rules, not the on-manifold axis.

**Falsifier.**  The basin appears everywhere → `tau=0.1` was simply wrong
throughout and every softmax row in this paper is under-tuned.  That would be
the most expensive outcome and we should want to know.

**Cost.**  ~60 short runs.

### Exp 114 — A factorial, not a scan: which knobs INTERACT?

**Motivation.**  Exp 110's effect was invisible to every one-at-a-time scan we
ran because it lives entirely in an interaction.  We have never run a factorial
design over the continuous knobs, so we do not know which others interact.

**Protocol.**  Fractional factorial (resolution IV, so two-factor interactions
are not aliased with main effects) over five knobs — `tau`, marginal weight
`lam`, `n_slices`, embedding dim, ft epochs — on two cells (c100, cars/visreg),
3 seeds.  Estimate main effects AND all two-factor interactions on probe,
mahaT, per-event.  Deliverable: an effects table saying which knobs matter,
which interact, and which are genuinely inert.

**Prediction.**  `tau x marginal` is the largest interaction; most main effects
are small; `n_slices` is inert (it is a Monte-Carlo budget, not a modelling
choice).

**Falsifier.**  Several large interactions → the space is not
approximately-separable in its knobs and single-arm comparisons are unsafe as a
methodology, not just under-tuned.

**Cost.**  ~50 runs for a resolution-IV design; far cheaper than the full grid.

### Exp 115 — The fairness audit: does the dissociation survive per-arm tuning?

**Motivation.**  The uncomfortable one, and the reason to do this at all.  Every
comparative claim in the paper — the probe/calibration dissociation above all —
compares arms AT THEIR INHERITED DEFAULTS.  Exp 110 showed at least one arm was
badly served by its default.  If each arm is given its own best `tau` (and
marginal weight), does the dissociation survive?

**Protocol.**  Per-arm tuning on a validation split with NO holdout-class
access (so the tuning is open-world legal), then re-run the c10/c100
dissociation table at each arm's tuned setting.  Report the tuned table beside
the archived one.

**Prediction.**  The dissociation narrows but survives: softmax arms gain
per-event from ~0 to 0.05-0.10 while NPLM arms keep a clear per-event edge, and
the probe ordering is unchanged.

**Falsifier.**  The dissociation disappears under tuning → the paper's central
empirical claim was substantially a statement about defaults, and would need to
be rewritten as such.  **This is the single most important test in Tier 7 and
should be run even though — especially though — it can undermine the paper.**

**Cost.**  Tuning runs + a re-run of the headline table.  Moderate.

### Exp 116 — SparKer M matched to intrinsic dimension, applied

> **DONE 2026-08-24 — PARTIAL; the exp-97 rule does not transfer as a recipe.**  Falsifier does NOT fire: dataset ordering preserved, so exp 100's ranking stands.  But matched M helps only at the extremes (galaxy10:dino >0.1 -> 0.075 at M=4; dtd:lejepa >0.1 -> 0.079 at M=64) and HURTS mid-ID cars (0.063 -> 0.079).  Same-M reruns put measurement noise at 0.012-0.015.  Keep M=16 as default; cite matched-M as a sensitivity band (citable 8/17 -> 10/17 best-of-two).

**Motivation.**  Exp 97 found `M` should be matched INVERSELY to intrinsic
dimension and that the default `M=16` leaves up to 0.16 power on the table on
high-ID spaces.  We then reported the entire reach table (exp 100) at `M=16`.

**Protocol.**  Re-measure `f95` on the 17 champion cells with `M` set from each
space's TwoNN ID (high-ID -> 4, mid -> 16, low -> 64), against the archived
M=16 numbers.

**Prediction.**  Reach improves on the high-ID (softmax-parent) cells; several
of the seven `>0.1` cells cross for the first time.  The dataset ORDERING is
preserved.

**Falsifier.**  The ordering changes → our sensitivity ranking was an artifact
of a fixed kernel budget, and exp 100's headline needs restating.

**Cost.**  Battery only, no retraining.  Cheap.

### Exp 117 — Are our comparisons adequately powered?

**Motivation.**  Protocol hygiene we have never done. Many claims rest on 3-5
seeds with a stated single-seed spread of ~0.017, and we have adopted a "gaps
below 0.02 need 3-5 seeds" rule by folklore rather than by calculation.  A power
analysis would say which comparisons in the paper are actually resolvable.

**Protocol.**  From the multi-seed archives (exps 61, 75, 81, 91, 105, 110),
estimate the per-cell seed variance by arm and dataset, then compute the
minimum detectable effect at 3, 5 and 10 seeds.  Cross-reference against every
comparative claim in the paper and flag the underpowered ones.

**Prediction.**  Several sub-0.02 claims are underpowered at 3 seeds, and the
transfer cells need more seeds than CIFAR (their spreads are wider).

**Cost.**  Analysis-only.  Cheap, and it protects everything else.

### Exp 118 — Holdout-selection audit

> **DONE 2026-08-24 — PREDICTION HELD, with a refinement worth more than the test.**  The archived alphabetical draw sits WITHIN the random-draw distribution, so it is not a headline confound.  But draw-to-draw spread (probe sd ~0.019; mahaT 0.391-0.569) EXCEEDS seed spread (0.001-0.010): WHICH class is held out matters more than the seed.  Cross-holdout claims need draw-resampled intervals, which feeds directly into the exp-117 MDE accounting.  Scope: CIFAR only.

**Motivation.**  Exp 111 found that alphabetically-ordered holdouts can leave
0-15 scorable superclasses out of 61-137, so some absolute superclass-agreement
figures rest on very few classes.  The whole campaign uses "last N classes" as
its holdout rule, which is alphabetical on several datasets and therefore not
random with respect to semantics.

**Protocol.**  Re-run the exp-76 interpretability battery and the headline probe
numbers under 5 RANDOM holdout draws per dataset, with the scorable-class count
reported.  Compare against the archived alphabetical numbers.

**Prediction.**  Probe numbers are stable (exp 78's rotation control already
suggests this), but the superclass-agreement figures move materially and should
be republished as random-draw means.

**Falsifier.**  Probe numbers also move → the alphabetical holdout is a
confound in the headline results, not just the interpretability ones.

**Cost.**  Moderate; the interpretability half is evaluation-only.

### Exp 119 — An 18th cell: `rr_disp` out of sample

> **DONE 2026-08-24 — FAILED; the pre-registered stopping rule TRIGGERS and the regime line closes.**  Threshold frozen at 0.06052 on the original 17 cells, predictions committed in their own commit (2131998) BEFORE scoring.  2/3 on held-out Food-101 cells against an >=80% bar -- and the predictor returned the SAME answer for all three inputs, so it discriminated nothing.  rr_disp is downgraded to a descriptive correlation; the paper may claim in-sample identifiability only; no further attempts.  Deviation stated: only 3 cells were buildable, so the effective bar was 3/3.  Mitigation recorded but not leaned on: all held-out |gap| <= 0.03, i.e. Food-101 sits on the regime boundary.

**Motivation.**  Exp 108 met its pre-registered bar (15/17) but stated the
caveat itself: the threshold is chosen in-sample on the same 17 cells exp 90
used, and Spearman 0.37 says the relation is threshold-like rather than
monotone.  As it stands the claim supportable is "the regime is label-free
identifiable", not "here is a predictor".  One genuinely held-out cell converts
the first into the second, or kills it.

**Protocol.**  Build 3-5 cells not in the original 17 — new dataset (e.g.
Food-101 or CUB, both fine-grained and plausibly on-manifold), or a new base on
an existing dataset — freeze the exp-108 threshold at its in-sample value, and
predict the LID-vs-eucl sign BEFORE measuring it.  Pre-register the predictions
in the log.

**Prediction.**  Correct sign on >= 4 of 5 held-out cells at the frozen
threshold.

**Falsifier.**  <= 3 of 5.  Then `rr_disp` is an in-sample artifact, the exp-108
result is downgraded to a descriptive correlation, and — per exp 108's own
stopping rule — the regime line closes for good.

**Cost.**  One new feature-bank extraction plus evaluation.  Moderate, and it is
the difference between a mechanism and a method.

---

## Tier 8 — after the tuning audit (exps 120+)

### Exp 120 — Can an open-world-legal tuner find the temperature basin?

> **DONE 2026-08-25 — PREDICTION FALSIFIED; exp 115's verdict HARDENS.**
> Analysis-only over the exp-113 archive (no retraining).  In the decisive
> cell (C100 with the marginal, where exps 110/113 put the basin) the oracle
> is tau=3.0 (mahaT 0.510, per-event 0.043, probe 0.926) and **all five
> seen-only criteria pick tau=0.1** (mahaT 0.398, per-event 0.000) -- the same
> answer exp 115's accuracy tuner gave.  I predicted `sw` would find the basin
> and `rms` would not; neither does.  Worse, the panel's notion of fidelity
> *anti-correlates* with novelty calibration here: tau=0.1 is the most
> faithful space by every panel column (ece 0.077, sw 1.44, rms 0.590) and the
> worst detector.  Hits in other cells are not evidence about the basin --
> C10's oracle comes with a collapsed probe, and C100-without-marginal has no
> basin to find.  (The oracle is now probe-constrained; the first run's
> headline was too generous and was corrected.)

**Motivation.**  Exp 115 showed the basin is invisible to a tuner selecting on
seen-class accuracy, and explicitly left open the one criterion it had not
tried: a seen-only *calibration* criterion.  The exp-104 panel supplies five,
all computable without a single novel example.

**Verdict and consequence.**  The basin is not merely missed by one tuner; it
is unreachable by every seen-only criterion available to us.  The paper should
say that the tau=1.0 recipe is a finding about the loss landscape, **not a
deployable recipe** -- it can only be selected by someone who already has
novelty labels, which is precisely who does not need it.  This also sharpens
the meta-lesson from a new direction: seen-class fidelity and novelty
calibration are not merely distinct, they can point in opposite directions.

**What would still change this.**  A criterion using unlabelled TEST data
(transductive: the pool exists at deployment even if unlabelled) rather than
seen-class training data only -- e.g. the stability of the seen/pool density
ratio across tau.  That is a different legality class and is exp 121.

### Exp 121 — A transductive tuner

> **DONE 2026-08-25 — FALSIFIER FIRES, but the test is under-powered and
> off-axis; read it as suggestive.**  The exp-113 sweep archived scalars only
> (no embeddings, no checkpoints), so the tau-basin selection could NOT be
> re-tested analysis-only.  Ran the one level more general question instead: on
> the six cached CIFAR-10 NPLM spaces, does the transductive criterion class
> rank spaces by novelty calibration better than the seen-only class?  Against
> mahaT (exp 120's oracle) the best transductive criterion is MMD at |rho|=0.54
> (p=0.27) and the best seen-only is class separation `sep` at |rho|=0.89
> (p=0.02) — seen-only WINS.  My predicted winner, the fitted `t_np`, is the
> worst criterion in the table (|rho|=0.09) and is unstable in the fit budget
> (0.60 at 50 steps, -0.09 at 300), which is itself consistent with exp 98.
> Caveats that matter: n=6 needs |rho|>=0.83 for p<0.05, so only `sep` is
> individually significant and the CLASS comparison is not decisive; `sep`
> predicting mahaT may be near-tautological (both reward well-separated tight
> seen classes); and the three outcomes DISAGREE IN SIGN (LID AUC
> anti-correlates with eucl/mahaT across these arms — the on-manifold split
> appearing within one dataset), so a single target had to be chosen.
> **Exp 113 has been patched with `--save-embs`; the tau test proper is a GPU
> re-run away and remains the deciding experiment.**

> **TAU-AXIS TEST DONE 2026-08-25 (GPU box) — FALSIFIER FIRES; the selection
> thread is CLOSED.**  Fresh C100 sweep with embeddings (basin replicates:
> on-marginal 0.218→0.512–0.528 plateau, off flat).  On the decisive cell the
> oracle is tau=1.0 and none of the four transductive criteria select it; the
> predicted winner `t_np` picks 0.1 (its third failure as a selector).  The
> control cell closes the loophole: `mmd` and `tail_mass` pick the same tau
> whether or not a basin exists — constant preferences, not detection.  With
> exps 115/120: the basin is invisible to every label-free rule constructed,
> seen-only or transductive.  Permanently a loss-landscape finding.


**Motivation.**  Exp 120 restricted selection to seen TRAINING classes.  But a
deployed pipeline does have the unlabelled evaluation pool -- it just has no
labels for it.  Criteria computable from (seen train + unlabelled pool) are a
strictly larger and still-legal class: pool-vs-seen density-ratio statistics,
the SparKer t_NP fitted on the pool, LID-gap dispersion, the fraction of pool
mass in the seen tail.

**Protocol.**  Re-run the exp-120 selection over the exp-113 archive with
transductive criteria, adding the pool as an input.  Same decisive cell, same
oracle, same probe-retention constraint.

**Prediction.**  The fitted `t_NP` between pool and seen reference DOES track
the basin, because it is measuring exactly the density-ratio structure the
basin creates -- making the recipe deployable after all.

**Falsifier.**  Transductive criteria also pick tau=0.1 -> the basin is
invisible to anything short of labelled novelty, and the tau finding is
permanently a curiosity rather than a method.  Report it as such and stop.

**Cost.**  Analysis-only if the pool statistics can be recomputed from cached
embeddings; one sweep otherwise.  **The natural next run.**

### Exp 122 — What IS the basin geometry?

> **DONE 2026-08-25 (GPU box) — PREDICTION HOLDS, with the control cell
> showing the alternative.**  Basin cell: anisotropy 6.4→2494.6 while the
> between-class alignment with the top eigendirections FALLS (0.732→0.467 at
> tau=0.3; endpoint d=−0.177) — structured anti-isotropy, exactly what the
> tied-covariance whitened distance rewards, and why mahaT works at rms~0.5
> where forcing unit width (exp 105) was decorative.  Control cell (no
> marginal): anisotropy explodes to 4.8e5 with alignment saturating at 1.000 —
> unstructured collapse.  The SIGReg marginal is what STRUCTURES the
> anisotropy; "looser interaction" is not the whole story.  Caveat: align is
> non-monotone on-cell (0.699 at tau=1.0); cleanest row is tau=0.3.

**Motivation.**  We have the best single-loss C100 both-currencies space on
record and no account of *why* it works beyond "a looser interaction lets the
marginal shape the covariance".  Exp 110 noted the winner sits at rms~0.50
with slope 10-18 (strongly anisotropic), i.e. it is NOT the unit-width
isotropic ideal the program was built around -- and exp 105 showed forcing that
ideal is decorative.  So the best space we have contradicts the stated target
and nobody has characterised it.

**Protocol.**  Full panel + exp-76 semantic battery + eigenspectrum on the
tau-sweep spaces, C100, marginal on, tau in {0.1, 0.3, 1.0, 3.0}: how does the
class-conditional covariance change as the basin is entered?  Report the
eigenvalue profile, the anisotropy direction, and whether the anisotropy aligns
with the between-class subspace.

**Prediction.**  The basin spaces are anisotropic in a *structured* way -- high
variance along between-class directions, low along within-class ones -- which
is what makes tied-covariance Mahalanobis work while unit width does not.

**Falsifier.**  The anisotropy is unstructured -> "looser interaction" is the
whole story and there is nothing further to characterise.

**Cost.**  Evaluation-only on the exp-113 checkpoints, if kept.

### Exp 123 — Are the four currencies actually independent?

> **DONE 2026-08-25 — NEITHER prediction nor falsifier; the dimension count is
> roughly right but the LABELS do not carve the space.**  122 spaces x 7-9
> metrics from the archives.  The robust finding, stable at every coverage
> threshold: **within-currency |corr| does NOT exceed between-currency |corr|**
> (0.34 vs 0.30 at min-cov 0.35; 0.28 vs 0.27 at 0.2 and 0.1).  Scree is
> 0.43/0.21/0.12..., i.e. ~3 components for 80% of variance and 2 by Kaiser --
> so the battery spans about three dimensions, close to the predicted three,
> but the metric-to-currency assignment is not what the correlation structure
> says.  Read as: the paper's DIMENSIONALITY claim survives; its TAXONOMY is
> not validated by the data.
> Two caveats that stop this being decisive: (i) DISCOVERABILITY is absent
> entirely -- purity/margin exist only on discovery runs and never clear the
> coverage bar -- so this tests 3 of 4 currencies; (ii) the 122 spaces pool
> heterogeneous datasets, dims and protocols, and pooled correlations across
> such cells can invert within them (Simpson's paradox), so a per-dataset
> replication is needed before this is load-bearing.

> **PER-DATASET REPLICATION DONE 2026-08-25 (`--per-dataset`) — the Simpson
> inversion is REAL and the taxonomy verdict softens.**  Within cifar100
> (n=44) within-currency correlation DOMINATES (0.515 vs 0.280; bar +0.1
> met); cifar10 and the transfer pool show the same positive sign below the
> bar.  Dimensionality is rock-solid in every group (k80 = 3-4, Kaiser 2-3).
> Refined statement: the battery spans ~3 empirical dimensions and the
> currency labels align with them within homogeneous cells (clearly on C100,
> weakly elsewhere); pooled correlations understate the alignment.  Caveat (i)
> stands: 3 of 4 currencies audited.


**Motivation.**  The paper's organising claim is that a discovery-ready space
must be audited in four currencies.  We have never tested whether they are
empirically separable -- if detection and interpretability correlate at 0.9
across the campaign's spaces, the battery is over-specified and the paper
should say so.

**Protocol.**  Assemble every metric on every space in the archives (~100+
spaces x ~15 metrics), compute the correlation matrix and a factor analysis,
and ask how many latent dimensions the battery actually spans.

**Prediction.**  Three latent factors, not four: probe and semantic agreement
load together (both track supervision), leaving detection, calibration and
interpretability-as-fidelity as the separable axes.

**Falsifier.**  Four or more clean factors -> the framing is vindicated as
stated.  Either outcome is publishable and it validates the paper's own
skeleton.

**Cost.**  Analysis-only.  Cheap.

---

## Deliberately not proposed

- **More bases / more datasets.**  The regime rules already replicate on three
  bases and five datasets; another base tests nothing new.  Breadth is not the
  binding constraint.
- **Bigger trunks.**  Would raise every number and change no verdict.
- **Tuning λ.**  App. A explains why it is inert for the first ~2 epochs;
  the empirical scan already found it flat.  τ is the knob that matters.
- **A "best of both" single loss.**  Four attempts (exps 34e/50/53/59) plus the
  gradient argument in App. A (each term is the other's null space) suggest the
  concat/residual construction is not a workaround but the right answer.  Would
  need a new idea, not another sweep.

---

## Suggested order

**Done** (see the status boards above): everything in Tiers 1-6 — 80-94,
96-104 (Tiers 1-5, completed 2026-08-21) and 105-111 (Tier 6, completed
2026-08-23).  Exp 95 remains contraindicated and unrun.

The Tier-6 pass closed every line it opened: the width identity is
repairable-but-decorative (105, fifth meta-lesson confirmation), the width
headline survives its own audit (106), composition joins but does not
replace LID (107), the regime is label-free identifiable via rr_disp
(108, pre-registered bar met), the C100 geometry block is broken by the
frozen density-ratio pool (109), the C100 ceiling break is a designed
tau=1.0 x marginal recipe (110, corroborated by 105's independent width
route), and the child-only deployment split is citable (111).  The two open threads identified at
the close of Tier 6 were an 18th cell to take `rr_disp` out of sample, and a
rotated-holdout protocol if absolute explanation agreement is ever needed.
Both are carried forward: the first as **exp 119**, the second as **exp 118**.

**Tier 7 adds a third thread that the exp-110 accident forces.**  The argument
for it is not "more tuning is generally good" — it is that the campaign swept
the DISCRETE design cube exhaustively and never swept the CONTINUOUS constants
at all, and that the one time a constant varied (by accident) it produced the
best C100 space on record via an INTERACTION that no one-at-a-time scan could
have seen.  That is a methodological gap, not a missed hyperparameter.  It is
scoped as an audit rather than a redo, and ordered so the analysis-only checks
come first:

1. **Exp 112** (inherited-constants audit) — hours of code archaeology; scopes
   everything else by saying what was never chosen.
2. **Exp 117** (power analysis) — analysis-only; says which existing claims are
   even resolvable before we spend GPU time adding more.
3. **Exp 116** (SparKer M matched to ID) — battery-only; exp 100's reach table
   was measured at a kernel budget exp 97 showed to be wrong for high-ID spaces.
4. **Exp 119** (18th cell for `rr_disp`) — the pre-registered out-of-sample test
   exp 108 explicitly deferred.
5. **Exp 115** (does the dissociation survive per-arm tuning?) — the most
   important test in the tier, and the one most able to undermine the paper.
6. **Exp 113** (is the `tau x marginal` basin general or many-class?) — decides
   whether every softmax row needs re-running.
7. **Exp 118** (holdout-selection audit) — the alphabetical-holdout confound.
8. **Exp 114** (fractional factorial) — the general answer to what else our
   one-at-a-time scans are missing.

Not to be run as specified: **Exp 95** (SparKer as a training loss).  Formally
cleared by exp 94, but contraindicated by 96 (warm start harmful) and 98
(SparKer-ft destroys the separation it optimizes); it needs a
class-structure-preserving redesign before any attempt.

Standing item resolved 2026-08-21: `tests/test_calibration.py` HAS been
executed on this machine — full suite green (103 passed) both at merge time
and after; the identities it pins are verified.

---

## Tier 9 — the paper-facing runs (exps 124-126)

Added 2026-08-26 after the paper restructure.  These are not exploratory: each
one exists because a decision about the ICLR draft depends on it.  Unlike
Tiers 1-8 these are **required for the paper to be honest**, not optional
upside.

Two mechanisms landed with this tier (`supersig/holdouts.py`, 34 tests):
`SUPERSIG_NH` selects the holdout REGIME and `SUPERSIG_HOLDOUT_DRAW` selects
WHICH classes are held out.  Both tag every artifact they write, so no run can
overwrite another.  With both unset the campaign default is reproduced exactly.

### Exp 124 — The leakage-free shortlist on a scratch trunk

**Motivation.**  Every CIFAR cell built with `pretrain=ds` uses hub weights
that already saw the HELD-OUT class (`supersig/models.py:80-82`) — backbone
label leakage in exactly the cells used to demonstrate discovery.  Exp 67/68
is the clean lineage but originally carried only one of the three shortlisted
objectives.  `ssig` (= exp-70 `ss-ft`) and `nplmsd` (= `nplm-sup-ft`) were
added 2026-08-25 and verified objective-identical to their fine-tune twins.

**Protocol.**
    python experiments/67_scratch_pretrain.py --arms supcon ssig nplmsd
    python experiments/68_scratch_discovery.py --bases supcon,ssig,nplmsd
CIFAR-100, 100-D, single holdout {4}, 200 ep, `pretrain=None`.

**Prediction.**  Purities stay ~0.00-0.01, far under the 0.15 gate, and
per-event stays dead — C100 at h1 is rate-floored (exp 109).  `ssig` beats
`supcon` on probe, as `supcon_sigreg` does on 2/3 clean bases in exp 50 but
NOT on hub-init trunks.  This is the SIGReg-earns-its-keep-on-honest-trunks
claim, on the objective the paper actually shortlists.

**Falsifier.**  `ssig` does NOT beat `supcon` on a clean trunk → the exp-50
scratch-base result was an artifact of head-on-frozen-base training and the
workhorse claim loses its leakage-free support.

**Cost.**  2 x 200-epoch pretrains + one discovery pass.

### Exp 125 — The single-holdout battery, with draws

> **galaxy10 VISREG DONE 2026-08-27 (5 draws) — the prediction is BASE-
> DEPENDENT.**  On visreg, supcon-ft clears the purity gate on 5/5 random
> draws (0.290+-0.067, min 0.171; nplm-sup-ft 0.262, min 0.151), where dino
> is 0/5 and lejepa 2/5.  Round 2 collapses on every visreg arm (0.29 ->
> 0.08).  Archived d{9} mahaT above the draw range for 6/6 arms.  State as
> "reliable on VISReg, coin-flip on LeJEPA, absent on DINO".
> `logs/SUMMARY_TABLES.md`.

> **Residual half (exps 71/72 on the draws) DONE 2026-08-27, dino + lejepa.**
> The residual concat's probe gain over its OWN parent is the one galaxy10
> effect that survives draw variance: 5/5 draws on both bases for all three
> constructions (supcon->res +0.023/+0.053, res-nplm +0.011/+0.039, ss->res
> +0.065/+0.093; paired sd 0.007-0.09 vs unpaired 0.03-0.06).  On lejepa it
> also wins eucl (4/5) and per-event (5/5).  Archived residual records
> (0.955/0.975) are the top of the draw range.  Discovery on the concat stays
> below the gate on average and is probe-negative.  `logs/SUMMARY_TABLES.md`;
> aggregator `experiments/125_aggregate_residual_draws.py`.

> **galaxy10 dino + lejepa DONE 2026-08-26 (5 draws each) — PREDICTION
> FALSIFIED ON RANDOM DRAWS.**  "Discovery survives on galaxy10" was a class-9
> result: round-1 purity across held-out classes {2,3,4,5,6} is 0.09-0.13
> (dino, every arm below the 0.15 gate) and 0.10-0.19 (lejepa); the archived
> nplm-sup-ft 0.46/0.50 becomes 0.13+-0.04 / 0.19+-0.14.  Archived mahaT sits
> above the random-draw range for 5/6 dino arms.  Draw sd (probe 0.03-0.18,
> mahaT 0.09-0.17) exceeds every arm gap except supcon-ft's probe lead -> report
> arm comparisons as ties.  Tables + verdicts in `logs/SUMMARY_TABLES.md`;
> aggregator `experiments/125_aggregate_draws.py`.  visreg and dtd pending.

**Motivation.**  The paper adopts the REGIME SPLIT: single-holdout is the hard
low-rate regime, multi-holdout the higher-rate regime where pooling unlocks the
gate.  Single-holdout must be run for EVERY dataset to be the uniform backbone.
Critically, exp 118 found draw-to-draw spread EXCEEDS seed spread (probe sd
~0.019, mahaT 0.391-0.569 across draws vs 0.001-0.010 across seeds).  Under
nh=1 the whole result rests on ONE class, so single-holdout numbers need an
interval over DRAWS, not seeds.

**Protocol.**  For each dataset, >= 3 draws (5 preferred):
    for D in 0 1 2 3 4; do
      SUPERSIG_NH=1 SUPERSIG_HOLDOUT_DRAW=$D python experiments/70_cars_ft_suite.py --dataset dtd --base dino
    done
then 71/72 (residual), 80 (SparKer), 100 (reach), 104 (panel).  Artifacts land
tagged `_h1_d{D}`.  Report mean +- sd ACROSS DRAWS.

**Prediction.**  Positive discovery survives on galaxy10 (rate 1/10) and dies on
the label-rich cells (rate 1/196 on cars, 1/102 on flowers) — the rate floor
generalizing beyond C100.  Draw sd will be the dominant uncertainty and larger
than most arm-to-arm gaps in the single-holdout tables.

**Falsifier.**  Label-rich single-holdout cells clear the purity gate → the
rate floor is C100-specific and the regime split is the wrong framing.

**Cost.**  High — this is the bulk of the remaining GPU budget.  Prioritize
galaxy10 (cheapest, most informative) then dtd, then flowers/cars/aircraft.

### Exp 126 — Multi-holdout, restricted and re-tabulated

**Motivation.**  Multi-holdout is kept ONLY for the label-rich datasets, where
10 classes is a small fraction of the label set: cars (196), flowers (102),
aircraft (100), cifar100 (100), dtd (47).  Most of this data already exists at
the campaign default; the work is re-tabulation into regime-separated tables
plus draw intervals where they are missing.

**Protocol.**  Mostly evaluation-only re-harvest of archived npz into the new
table format.  Add draws (`SUPERSIG_HOLDOUT_DRAW`) where a headline number
currently rests on one alphabetical draw.

**Prediction.**  The ss-ft DTD purities (0.795/0.803/0.811) survive draw
resampling with a wider interval; the exp-109 density-ratio result
(0.358 -> 0.418) survives.

**Falsifier.**  The DTD purities move outside their draw interval → the
workhorse claim was a favourable alphabetical draw, exactly the exp-118 hazard.

**Cost.**  Low if re-tabulation only; moderate with draws.

### Exp 127 — Re-run the frozen density-ratio pool with BN actually frozen

> **DONE 2026-08-27 — FALSIFIER HALF-FIRES.**  The gate is still cleared at
> h10 (r1 0.268 / 0.214 at q0.99 / q0.95, vs distance 0.002 / 0.036), so the
> construction stands -- but the headline drops 0.358 -> 0.268 and the
> round-2 rise is NOT reproduced at h10 (0.418 -> 0.237); h5 falls below the
> gate on round 1 (0.198 -> 0.141).  Part of the archived exp-109 gain was BN
> drift.  Confound: 109 rebuilds the exp-89 space (pre-probe 0.954 -> 0.959),
> so retraining noise is mixed in; a same-checkpoint BN-train/eval A/B would
> isolate it.  Tables in `logs/SUMMARY_TABLES.md`; archive kept as
> `logs/exp109/results_preBNfix_archived.npz`.

**Motivation.**  Exps 86/92b/109 froze backbone WEIGHTS but not BatchNorm
running statistics, so on the CIFAR trunk the "frozen" space still drifted (up
to 1.29 in embedding units against mean |z| 0.52 over 3 rounds).  The transfer
cells use a LayerNorm ViT and were exact; only CIFAR is affected.  Fixed
2026-08-26 (`supersig.train.set_train_mode`).

The exp-109 headline — C100 purity 0.121 -> 0.358, round 2 rising to 0.418 —
is the evidence for the second of the paper's two constructions, and it was
produced under the leaky freeze.

**Protocol.**  Re-run `109_c100_density_pool.py` unchanged; the fix is in the
trainer.  Compare against the archived `logs/exp109/` numbers.

**Prediction.**  The result survives.  The mechanism (critic refit + pool
growth + anchor updates) does not depend on BN drift, and the transfer cells
showed the same round-2-rises signature with an exactly frozen ViT.

**Falsifier.**  Purity no longer clears the gate, or round 2 stops rising →
part of the exp-109 gain came from the space quietly moving, and the
"frozen density-ratio pool" construction needs restating.

**Cost.**  Low — evaluation-only, no retraining (the space is frozen).

### Exp 128 — Where should the pool cut actually sit?

> **DONE 2026-08-27 on the 30-cell C100 bank (tau x marginal x 3 seeds, h1,
> b=0.01).**  (a) HOLDS: np optimum q* = 0.004-0.04, far tighter than 0.05.
> (c) FALSIFIED in the good direction: with the SIGReg marginal ON, np clears
> the gate AT h1 with >= 100 novel points on 5/5 tau settings (0.31-0.58;
> tau0.05-on 0.58+-0.08 at q=0.0036).  The "rate floor" was distance + q0.95,
> not the cell.  (b) HOLDS where it matters: dist never exceeds purity 0.05 at
> any cut; lid dead.  UNPREDICTED: the marginal helps np and hurts dist -- a
> fourth instance of the dissociation.  Exp 131's C100 refusals are therefore
> an ESTIMATOR failure (b_hat 100x low), not a space failure.  Tables in
> `logs/SUMMARY_TABLES.md`, JSON `logs/exp128/cutscan.json`.

> **SCRIPT READY, ALGEBRA VALIDATED (2026-08-26).**  `128_pool_cut_optimization.py`,
> evaluation-only, `--selftest` green (8 checks) + `tests/test_pool_cut.py` (9).
> Needs embeddings from a real cell to produce results.

**Motivation.**  `tau_quantile=0.95` has been the pool cut since exp 23 and was
never derived.  Exp 112 lists it as "swept exp 89 (distance only)"; exp 109
tried only {0.95, 0.99} for the density scorer.  Three values on a grid that
skips the whole 0.95-0.99 interval where the h10 optimum evidently lives.

**The algebra.**  With b = base rate, q = pool/N, e_s = signal efficiency:

    purity = e_s*b/q,    E = purity/b = e_s/q,    purity <= min(1, b/q)

At q=0.05, b=0.01 the CEILING is 0.20 (20x) while measured density-ratio purity
at h1 is 0.030 (E=3.0, e_s=0.15).  **This corrects the "rate floor" framing**:
h1 is not information-theoretically blocked, it is SIGNAL-EFFICIENCY blocked,
and the cut is a free parameter sitting far from its own ceiling.  The earlier
"base rate >= 4%" rule silently treated E~3.5 as a property of the problem
rather than of one arbitrary cut.

Second constraint: `n_novel = purity*q*N >= n_min` for BIC to find a component.
h10/q0.99 leaves ~179 novel points; h1/q0.99 leaves ~15.  That is why
tightening helped at h10 (0.232->0.358) and did nothing at h1 (0.030->0.029).
Objective: maximize purity(q) s.t. n_novel(q) >= n_min.

**Protocol.**  Feed archived embeddings (`113 --save-embs` format: `tr`,
`tr_lab`) and sweep q over a dense log+linear grid for all three scorers
(dist/lid/np), reporting purity, ceiling, headroom, E, e_s, n_novel, the
campaign's own tau_q cuts for reference, and (with `--bic`) whether BIC
actually recovers a majority-novel cluster.  Also reports each scorer's
cut-free novel-vs-seen AUC.

**Prediction.**  (a) The optimum for `np` is tighter than 0.95 on every
multi-class cell.  (b) Headroom (purity/ceiling) is well below 1 everywhere,
i.e. the deficit is scorer efficiency, not the cut.  (c) On h1 no q satisfies
both constraints, so the floor is real but for the n_min reason, not the
enrichment reason.

**Falsifier.**  Headroom is already ~1 at q=0.05 → the scorers are at their
ceiling and only the cut was ever wrong; the "improve the scorer" program is
misdirected.

**Validated synthetically already.**  On a hard synthetic bank (novel class
adjacent to a seen class, heavy-tailed background) the tool reproduces the
exp-89/exp-109 inversion from first principles: `dist` has AUC 0.954 yet purity
**0.000** at tight cuts (tail owned by background outliers, tightening HURTS:
0.117 -> 0.000), while `np` at AUC 0.9998 improves monotonically under
tightening (0.168 -> 0.669) with a best usable cut of purity 0.99.

**A caution the synthetic exposed.**  A scorer can have HIGH AUC and still pool
badly, because AUC is blind to tail structure -- pinned by
`test_high_auc_can_still_pool_badly`.  So report the operating curve, not AUC
alone; AUC separates scorer quality from cut choice but does not replace it.

**Cost.**  Minutes, evaluation-only.  Run it on every cell that produces
embeddings in exps 124/125.

#### Exp 129 addendum — n_min swept, kmax made label-free (2026-08-26, REAL data)

Run on three real exp-54 CIFAR-10 spaces with the novel class subsampled to
b in {0.01, 0.02, 0.05, 0.10} (real geometry, realistic rates).
`logs/exp129/n_min_sweep_real.json`.

**n_min: smaller is better, confirmed.**  Purity falls monotonically as n_min
grows.  On `nplm_dist_sup_cw` at b=0.10: purity **0.830** at n_min=20 vs 0.696
at n_min=500.  Default set to **30**.

**Q_MIN was binding and is lowered 0.002 -> 0.0005.**  At every n_min <= 75 the
rule wanted a tighter cut than the clip allowed.  Pushing further shows a
plateau, so the tight end is safe but not unboundedly profitable:

    q       0.0002  0.0005  0.001  0.002  0.005  0.010  0.020  0.050
    purity   1.000   0.800  0.840  0.830  0.788  0.716  0.648  0.565
    recall   0.002   0.004  0.008  0.017  0.039  0.072  0.130  0.283

**kmax is now label-free**: `k_max = clip(floor(sum(w)/n_min), 2, 64)` from the
same novelty weights that drive the cut, replacing
`max(4, len(holdouts) + 2)` (`discovery.py:189`), which uses the NUMBER OF
NOVEL CLASSES -- oracle knowledge.

**FINDING 1 — the oracle leak is inert at single holdout.**  BIC returns
khat=1 in essentially every row, at every kmax from 2 to 63.  With one novel
class that is the CORRECT answer, so kmax cannot matter at h1; the leak only
has teeth in the multi-holdout regime.  It also means the clustering step is a
no-op on single-holdout pools -- the "cluster" is the pool.

**FINDING 2 — b_hat is badly biased downward on real data, and this is the
binding limitation.**  TV(p_D,p_R) = b * TV(p_S,p_R), and real novel classes
overlap the seen manifold heavily, so:

    space              AUC    b_true   b_hat
    nplm_dist_sup_cw  0.766     0.10   0.0255     (4x low)
    nplm_bilinear     0.714     0.10   0.0003   (300x low)
    nplm_bilinear     0.534     0.01   0.0001   (100x low)

On spaces where the critic separates well the rule engages and works.  On weak
spaces b_hat collapses, N_hat never reaches n_min, and q saturates at Q_MAX --
the rule degenerates to a wide pool and its `ok` flag is False.  That flag is
doing its job and MUST be reported: a False there means "this space cannot
support discovery", which is information, not a crash.

**Consequence for the paper.**  The label-free cut is viable where the scorer
is strong (purity 0.83 at b=0.10 on real embeddings, far above the 0.15 gate)
and correctly refuses where it is not.  But the quantity limiting it is scorer
quality, exactly as exp 128 predicted -- the same conclusion from a second,
independent direction.

### Exp 129 — A label-free rule for the pool cut

> **DONE 2026-08-27 on the 30-cell C100 bank — PREDICTION HOLDS.**  Rule B
> beats the inherited q=0.05 in 29/30 cells (2-4x), engages 30/30, and
> tau=0.05-on clears the 0.15 gate LABEL-FREE at h1 on 3/3 seeds (0.39+-0.17,
> 171 novel points) -- first legal single-holdout C100 pool above the gate.
> Oracle gap +0.29 is the base-rate estimator (b_hat 2-4x low -> cut 2-5x too
> wide), not the critic.  The `detect` variant's oracle match is a grid-floor
> artefact (true n_novel 24 < n_min while n_hat says 81) -- do not quote.
> Tables in `logs/SUMMARY_TABLES.md`; JSON `logs/exp129/legal_cut.json`.

> **SCRIPT READY, RULE DERIVED AND VALIDATED (2026-08-26).**  `--selftest` green
> (7 checks) + `tests/test_legal_pool_cut.py` (18).  Needs real embeddings.

**Motivation.**  Exp 128 shows the cut matters and 0.95 is wrong, but `purity`
needs novel labels, so sweeping it per cell is ORACLE TUNING.  The campaign was
careful about exactly this for `tau` (exp 115; exps 120/121) and must not
quietly oracle-tune `q`.  Exp 128 is therefore a diagnostic; this is the
deliverable: ONE global, label-free rule, reported against the oracle.

**Estimating the base rate without labels.**  The NP critic gives
f = log(p_corpus/p_ref).  Three estimators, all validated to recover a known b
to within 10% on an exact critic:
`tv` = E_corpus[max(0, 1-1/r)], `mass` = P_corpus(r>2), `excess` = bump-hunt
excess at the 0.99 reference quantile.

**RULE A (purity target), rejected.**  q* = clip(b_hat / 0.30).  Hits 0.30 by
construction -- and stops there.  Oracle reaches purity 1.00 at a 6x tighter
cut.  Targeting a purity LEVEL is the wrong objective.

**RULE B (detectability-limited), adopted.**  Purity = e_s*b/q rises
monotonically as q shrinks, so the optimum is at the tight end and the binding
constraint is whether a CLUSTER survives.  Estimate the survivor count
label-free: under contamination the posterior that a corpus point is novel is
`1 - 1/r`, so summing it over the pool estimates n_novel without labels.
Then q* = the tightest cut whose ESTIMATED novel count still exceeds n_min.

On an exact critic Rule B **matches the oracle exactly** (q=0.0056, purity
1.000) at b = 0.01/0.02/0.05/0.10, versus 0.30 for Rule A and 0.20-1.00 for the
inherited q=0.05.

**Prediction.**  On real cells Rule B beats the inherited q=0.05 wherever the
base rate differs from ~1.5%, and its oracle gap is dominated by critic quality
rather than by the rule.

**Falsifier.**  Rule B's oracle gap on real cells is large and comparable to
Rule A's → the estimated novel count is too noisy on fitted critics, and q
joins tau as a knob invisible to legal selection.  (Still a publishable
finding, and the campaign's recurring one.)

**Two implementation traps, both hit and both now pinned by test.**
1. The reference-side form E_ref[max(0, r-1)] is algebraically equal to the
   corpus-side TV but USELESS in Monte Carlo: when novelty is disjoint the
   excess lives where p_ref ~ 0, so no reference sample lands there.  Measured
   1.2e-5 against a true b of 0.01.  Always integrate against the sample that
   COVERS the excess.
2. The fitted critic is frequently far from its own calibration identity, so
   ratios must be renormalised by E_ref[e^f] before use.  See exp 130.

### Exp 130 — Is the NP critic converging at all?

> **STEP (a) DONE 2026-08-26 — HYPOTHESIS FALSIFIED, and the measurement that
> opened it was MY ERROR.**  The critic converges.  In-sample calibration on
> six real exp-54 CIFAR-10 spaces is **0.997-1.019** (ideal 1.000).  No
> archived SparKer number is in question.

**What I got wrong.**  Exp 129 reported `E_ref[e^f]` of 60 / 4.7e3 / 6.5e5 /
1.1e6 and concluded the optimisation was diverging.  It was not.  Those were
OUT-OF-SAMPLE values -- E[e^f] over ALL seen points -- while the identity the
NP minimiser enforces is over the reference points USED IN THE FIT (the
`max_ref=4000` subsample).  Measured properly, in-sample calibration is 1.000
at every capacity setting, including the ones I called divergent.  The AUCs I
reported (0.885/0.882/0.851/0.913) reproduce exactly; only the calibration
column was wrong.

The proposed mechanism was also wrong: `f` never approaches the +-20 clamp.
Max |f| is 5.7-11.6 on real embeddings and ~13 on synthetic, **0.00% of points
at the clamp** in every case.  The zero-gradient-at-clamp story is unsupported.

**What is actually there (smaller, real).**  In-sample ~ 1 with out-of-sample
> 1 is textbook OVERFITTING of the critic to its reference subsample -- the
fitted ratio has a tail the 4000 reference points never covered.  It grows with
capacity, as overfitting should:

    M=16,  300 steps:  calib_in 1.000   calib_out 60
    M=64,  300 steps:  calib_in 0.999   calib_out 4.7e3
    M=64,  800 steps:  calib_in 0.998   calib_out 6.5e5   (synthetic, b=1%)

On REAL embeddings the same effect is mild -- calib_out 1.000-2.706 across six
spaces, worst for `nplm_distance` (2.71), best for `nplm_bilinear` (1.000):

    space              calib_in  calib_out   AUC
    nplm_bil_cw           0.998      1.005  0.679
    nplm_bil_sup_cw       1.002      1.059  0.699
    nplm_bilinear         0.998      1.000  0.714
    nplm_dist_sup_cw      0.997      1.047  0.766
    nplm_distance         1.006      2.706  0.746
    nplm_sup_dist         1.019      1.353  0.760

**Instrumentation landed.**  `np_pool_scores(..., return_calib=True)` returns
`dict(calib_in, calib_out)`; `sparker.np_test_stats(..., return_calib=True)`
and `sparker.np_calibration(D, R)` return the in-sample value.  Report
`calib_out` alongside SparKer numbers: it is the one that can drift, and it is
the one the scores are actually used at.

**What this means for exp 128.**  The 3-15% signal efficiency is NOT explained
by a broken critic.  Real novel-vs-seen AUC is 0.68-0.77 -- the critic is
working and the task is simply hard.  Remaining candidates for the low e_s:
kernel init (M kernels sampled uniformly give a rate-b class an expected M*b
kernels; still untested) and genuine class overlap.

**Steps (b)/(c) remain open** but are no longer urgent, and should be motivated
by the overfitting finding (reference subsample size, capacity) rather than by
a convergence failure that does not exist.

**Cost.**  (a) was minutes and is now done.

### Exp 131 — Re-run cifar10 / cifar100 / galaxy10 with the label-free cut

> **ROUND-1 DONE 2026-08-26 — PREDICTION HOLDS WHERE THE RULE ENGAGES; the
> falsifier does not fire.**  cifar10 (b=0.10): legal beats inherited on 4/4
> engaged spaces, +0.17..+0.30 (resnplm-cat 0.586 -> 0.888, res-cat 0.515 ->
> 0.806).  galaxy10 lejepa/visreg: 11/12 engage at purity 0.83-1.00.  cifar100
> (b=0.01): refuses on all 5, as predicted.  UNPREDICTED: two false refusals at
> b=0.10 -- cifar10 supcon (critic calib_out 57, overfit) and ALL SIX galaxy10
> DINO spaces (calib 1.000 but b_hat 7-15x low) -- so `ok=False` diagnoses the
> critic, not the space.  Cost of purity is recall: pools shrink 10-25x
> (n_nov 3189 -> 269).  Multi-round with backbones not yet run.  Tables in
> `logs/SUMMARY_TABLES.md`; JSON in `logs/exp131/`.

> **CODE READY 2026-08-26.**  `--selftest` green; `supersig/poolcut.py` +
> `tests/test_poolcut.py` (11).  NEEDS THE GPU BOX.

**Why these three cells must be re-run.**  Three changes since they were last
measured, each touching the discovery loop:

1. **The pool cut** (exps 128/129).  `tau_quantile=0.95` is inherited from exp
   23 and the ceiling `purity <= min(1, b/q)` caps it at 0.20 when b=1%.
   Replaced by the label-free rule in `supersig.poolcut`.
2. **`kmax`** (exp 129).  `max(4, len(holdouts)+2)` uses the NUMBER OF NOVEL
   CLASSES -- oracle knowledge.  Now derived from label-free novelty weights.
3. **Freezing** (exp 130).  `requires_grad_(False)` did not stop BatchNorm
   running statistics; the CIFAR trunk drifted by 1.29 in embedding units
   (mean |z| 0.52) over 3 "frozen" rounds.  **This alone means archived frozen
   CIFAR numbers will not reproduce**, independently of 1 and 2.

**How it is wired.**  `run_discovery(..., cut_rule=)`: `"quantile"` is the
DEFAULT and reproduces every archived result exactly; `"legal"` uses the new
rule and requires `pool_score="np"`.  Nothing changes unless asked.

**Protocol.**  Paired A/B on identical spaces and seeds, per cell and arm.
Round-1 pool/purity/BIC is evaluation-only and runs from embeddings; the
multi-round loop needs the backbones.

    python experiments/131_legal_cut_discovery.py --selftest
    python experiments/131_legal_cut_discovery.py --cells cifar10,cifar100
    python experiments/131_legal_cut_discovery.py --cells galaxy10:dino,galaxy10:lejepa,galaxy10:visreg

**Evidence so far (real exp-54 CIFAR-10, novel class subsampled).**  The rule
engages where the scorer is strong and REFUSES where it is not:

    space              b     AUC   n_hat       q    pool  purity  n_nov     ok
    nplm_dist_sup_cw  0.10  0.766     717  0.0052    261   0.778    203   True
    nplm_sup_dist     0.10  0.760     690  0.0051    253   0.719    182   True
    nplm_bilinear     0.10  0.714      13  0.2000  10000   0.232   2315  False
    nplm_sup_dist     0.01  0.707      56  0.0273   1239   0.090    111   True
    nplm_dist_sup_cw  0.01  0.687      12  0.2000   9091   0.028    258  False
    nplm_bilinear     0.01  0.638       0  0.2000   9091   0.019    171  False

At b=0.10 the rule lifts purity 0.232 -> 0.778 on the best space.  At b=0.01 it
mostly refuses, which is the honest answer.

**Prediction.**  galaxy10 (b~0.10, single holdout, and the ONLY cell genuinely
OOD to the ImageNet-1k backbones) behaves like the b=0.10 rows -- the rule
engages and beats the inherited cut.  cifar100 (b~0.01) mostly refuses.

**Falsifier.**  Purity is WORSE under `legal` in cells where `ok` is True ->
the label-free cut is not usable and 0.95 should stand, with the ceiling
reported as a limitation.

**A DEFECT FOUND AND FIXED while building this.**  `w = [1-1/r]_+` is a
positive part, so ANY spread in a fitted `f` manufactures novelty: pure-noise
scores produced sum(w) = 665 of 8000 points, enough for the rule to fire on
nothing.  A flat mean subtraction does NOT fix it (405 phantom points survived)
because the top-ranked null points carry far above-average w.  The fix is a
RANK-MATCHED null -- under H0 the top q of the corpus is distributed like the
top q of the reference -- which cancels to ~0 under the null while preserving
real signal.  Pinned by `tests/test_poolcut.py::test_refuses_pure_noise`.

**Cost.**  Round-1 A/B is minutes per cell.  Full multi-round needs backbones.

### Exp 133 — BatchNorm adaptation: a construction or a leak?

> **DONE 2026-08-27 — PREDICTION HOLDS: A CONSTRUCTION.**  Paired on the cached
> exp-89 spaces, BN adaptation raises round-2 purity in 6/6 cells (+0.003..
> +0.062) and restores the round-2 rise at h10:q0.99 (frozen 0.268 -> 0.237;
> adapting 0.268 -> 0.299), with test-set drift ~25-30% of |z| and no probe
> cost.  Apportioning exp 127: the r1 drop (0.358 -> 0.268) is the space
> retrain, the r2 collapse is the freeze.  Paper: state "frozen pooling"
> (0.268, flat) and "corpus-adaptive normalisation" (0.299, rising; CIFAR
> ResNet only, explicit opt-in) as two constructions.  `logs/SUMMARY_TABLES.md`.

> **CODE READY 2026-08-27.**  `133_bn_adaptation_ab.py`, `--selftest` green +
> `tests/test_bn_adapt_switch.py`.  Queued on the GPU box behind exp 132.

**Motivation.**  Exp 127 re-ran exp 109 with BatchNorm actually frozen and the
headline moved (h10 purity 0.358 -> 0.268; the round-2 rise 0.418 -> 0.237
vanished).  But exp 109 REBUILDS the exp-89 space, so two things changed at
once and exp 127 cannot apportion the loss between BN drift and retrain noise.
What the drift *was* is the interesting part: a frozen trunk with BN in train
mode is running unsupervised test-time normalisation adaptation on the
fine-tune loader, which in discovery contains the pooled novel points.  No
labels are used, so it is legitimate -- but it is a DIFFERENT construction
from "freeze the encoder, iterate the anchors", and it ran silently, on the
CIFAR ResNet only (the ViT transfer trunks are LayerNorm).

**Protocol.**  `supersig.train.set_train_mode(backbone, bn_adapt=False)` is
now the explicit switch (plumbed through `train_sigreg_hybrid`,
`run_discovery`, `exp92.sparker_discovery`; default off everywhere).  Paired
A/B on ONE cached checkpoint per holdout size (`checkpoints/exp89_c100_h*`,
the spaces exp 127 built): A frozen, B weights-frozen + BN adapting; same seed,
loader order, critic.  h{1,5,10} x q{0.95,0.99}.  Report purity r1/r2, margin,
post probe, and test-set embedding drift |z_post - z_pre| (max, mean) against
mean |z| -- the direct "did the space move" measurement.

**Prediction.**  B reproduces the archived exp-109 signature on the same space
where A does not: higher r1 purity at h10 and a round-2 rise, with drift of
order the 1.29 exp 130 measured; A drifts ~0.  Then BN adaptation is a real
construction -- "corpus-adaptive normalisation helps the density-ratio pool"
-- and enters the paper under its own name with its own falsifier, never as
the default.

**Falsifier.**  B ~ A (gap inside ~0.03 seed noise) -> the archived 0.358 was
retrain variance, not adaptation; drop the idea, quote exp 127.  B < A ->
adaptation is harmful and the archived numbers were lucky.

**Cost.**  Evaluation-scale: spaces cached; 12 two-round frozen loops (~2x the
discovery half of exp 109).

**Also fixed on the way (2026-08-27).**  `exp77.head_emb` -- the checkpoint
loader behind exps 80/100/102/103/111/131/132 -- was DRAW-BLIND: it built
`{ds}_ft_{base}_{arm}_seen.pt` with no `run_tag()`, so under `SUPERSIG_NH` /
`SUPERSIG_HOLDOUT_DRAW` every one of those scripts silently scored the
alphabetical-holdout spaces.  Exp 131's galaxy10 cells were run at the default
draw and are labelled as such, so no archived number is wrong; but the exp-125
downstream evaluations (80/100/104) MUST be run after this fix.  Same fix in
exp 76.  Pinned by `test_head_emb_honours_the_holdout_draw_tag`.

### Exp 135 — Corpus-adaptive normalisation on every cell (exp 133's analogue)

> **galaxy10 DRAWS DONE 2026-08-27 — the corpus-norm gain is FALSIFIED on
> draws** (d r2 -0.006+-0.051 dist, +0.001 np; 16 wins / 23 losses): class 9
> again.  Keep as opt-in, no claim.  **What survives is larger: frozen-head
> density-ratio pooling clears the gate on 45/45 draw cells (r1 0.40-0.57,
> every arm/base/class, no round-2 collapse)** where the head-training
> distance loop manages 0/5 (dino) to 5/5 (visreg, one arm).  First galaxy10
> discovery result that is base- and draw-independent.
> **ARCHIVED CELLS DONE 2026-08-27 (90 paired A/B) — SPLIT VERDICT.**  Holds for
> the DISTANCE pool (+0.020 r2 purity, 33/45 wins, 4 losses; galaxy10
> +0.07..+0.10, dtd +0.05; probe/eucl unchanged) and FAILS for the density-
> ratio pool (-0.004, 9 wins / 15 losses).  Falsifier clause 2 fires: it helps
> where r1 ~ 0 (3/4 cells), so on the ViT it is the normaliser re-centring the
> corpus, not "richer anchors compound".  Exp 133's mechanism reading is
> restated for transfer cells.  SECONDARY: frozen-head np pooling itself posts
> r1 purity 0.53-0.68 on galaxy10 (distance 0.01-0.52) and beats distance in
> 34/45 cells -- the exp-109 construction transfers; needs draws before quoting.
> galaxy10 draw cells pending.  `logs/SUMMARY_TABLES.md`, `logs/exp135/`.

> **CODE READY 2026-08-27.**  `135_corpus_norm_everywhere.py`, `--selftest`
> green + `tests/test_corpus_norm.py`.  Queued as Block K (evaluation-scale).

**Motivation.**  Exp 133 showed BN adaptation -- re-standardising frozen
features with discovery-corpus statistics, no labels -- is a real construction
(round-2 purity +0.04..+0.06 on C100).  It existed only on the CIFAR ResNet;
the transfer trunks are LayerNorm ViTs with no running statistics, so "BN
adaptation for everything" needs an analogue.

**Protocol.**  Feature-space discovery on the exp-70 banks with
`CorpusNorm(head) = BatchNorm1d(768, affine=False, momentum=None) -> head`,
statistics seeded from the seen train features, head frozen.  A `frozen`
(stats fixed; = freeze-both) vs B `corpus-norm` (stats adapt during the anchor
fine-tune; `bn_adapt=True`), both pool scorers, 2 rounds x 5 ep, arms
{supcon-ft, ss-ft, nplm-sup-ft}; archived unfrozen exp-70 post-probe as
context.  All 15 transfer cells + the galaxy10 draws.  Reports purity r1/r2,
post battery, test-set drift.

**Prediction.**  B >= A on round-2 purity where round-1 clears the gate, drift
10-30% of |z|, no probe cost; no change where round-1 purity ~0.

**Falsifier.**  B < A where discovery worked -> ResNet/BN peculiarity, does not
transfer.  B helps where purity ~0 -> the gain is the normaliser acting alone,
and exp 133's "richer anchors compound" reading must be restated.

### Exp 134a — Is the residual gain a construction, or just width?

> **CODE READY 2026-08-27.**  `134a_width_control.py` (comparison) +
> exp 70 `--emb-dim 200` (the control; artifacts now tagged `_e200` via
> `exp70.seed_sfx`, so it cannot overwrite the archived 100-D parents --
> pinned by `tests/test_exp134ac.py`).  Queued behind Block I.

**Motivation.**  Question (A) of exp 134: the concat is two full networks and
the flagship cars/VISReg +0.148 has never met a same-width non-residual
control; exp 85 already found a capacity effect on that cell.

**Protocol.**  supcon-ft fine-tuned with a 200-D head (same trunk, corpus,
epochs, recipe; no residual) on cars/visreg for seeds 0/1/2, plus galaxy10
dino/lejepa seed 0.  Compare parent(100) / concat(100+100, res-nplm) /
control(200) on probe, eucl, mahaT, paired by seed, exp-132 TIE guard.

**Prediction.**  The control recovers a minority of the gain; concat beats it
by more than the floor -> the construction is real.

**Falsifier.**  control >= concat on probe -> the gain was width; the residual
is a way of spending parameters and is dropped as a construction (the
child-alone detector result is half-width and survives).

### Exp 134c — Should the residual come AFTER discovery?

> **CODE READY 2026-08-27.**  `134c_residual_after_discovery.py`, `--selftest`
> green.  Queued behind 134a.

**Motivation.**  Question (C) of exp 134: the residual removes what the anchor
set explains; after discovery the anchor set is richer, so the child should
carry more of what is genuinely unexplained.

**Protocol.**  Per cell: load the exp-70 parent -> run exp 70's feature-space
"natural discovery" on its head (2 rounds) -> relabel the corpus LABEL-FREE
against the enlarged anchor set (seen keep labels; other images join only if
their nearest anchor is a discovered one, under that anchor's index) -> train
the exp-71 child (res / res-nplm) with r = z - anchor[y'] from the parent trunk
+ post-discovery head -> battery on parent-post / child / concat, beside the
archived exp-71 (pre-discovery) rows.  Cells: galaxy10 x 3 bases, cars/visreg,
dtd/dino (archived draw).  Reports discovery purity and the diagnostic
pseudo-label purity so the falsifier can be read off.

**Prediction.**  Child informativeness (probe, eucl) rises where round-1
purity cleared the 0.15 gate and is unchanged where it did not.

**Falsifier.**  Helps where purity ~0 -> the mechanism story is wrong.  Hurts
where discovery worked -> the child is fitting anchors planted on the novel
class and removing what the concat needed.

**Cost.**  One feature-space discovery + one exp-71-scale child ft per (cell,
objective).

### Exp 132 — A supervised linear probe (the metric we never ran)

> **DONE 2026-08-27 — FALSIFIER FIRES.**  ss-ft loses supervised top-1 to
> supcon-ft by 0.034-0.038 on galaxy10, 10/10 paired draws on dino+lejepa,
> every gap above the 0.017 floor (visreg archived draw -0.019).  The no-cost
> framing does not hold for ss-ft as a standalone space.  What DOES hold:
> supcon-ft->res-cat / ->resnplm-cat TIE supcon-ft on all three bases (0.663/
> 0.775/0.777 vs 0.666/0.781/0.776) -- the no-cost claim is a construction
> claim (parent + residual concat), not an objective claim.  C100 bank: the
> tau basin costs 0.15-0.31 top-1 (0.62 -> 0.41-0.51).  Tables in
> `logs/SUMMARY_TABLES.md`; JSON `logs/exp132/`.

> **CODE READY 2026-08-27.**  `--selftest` green (5 checks) +
> `tests/test_supervised_probe.py` (10).  NEEDS exp-70 arm banks.

**The problem it fixes.**  The paper wants to argue that discovery capability
costs nothing in representation quality.  The campaign cannot currently support
that, because **the `probe` column in every existing table is NOT a supervised
probe** — `exp29.linear_probe_novelty` is a HOLDOUT-VS-REST AUC
(`docs/METRICS.md:14`), i.e. a novelty metric.  Quoting it as evidence of
supervised quality would be wrong and is the first thing a referee catches.
The supervised evidence that exists is `acc` (nearest-centroid) and the
exp-62/63 closed-set numbers.

**Protocol.**  A plain `nn.Linear` trained on SEEN classes only, top-1 on the
seen-class test split, identical budget across arms, >= 3 seeds.  Features are
standardised: the arms differ in embedding SCALE by design (calibrated
objectives fix unit class width, softmax ones do not) and an unstandardised
probe would partly measure scale.  Two variants — the 768-D trunk (did the
fine-tune damage the backbone?) and the low-D head output (is the space we
claim about decodable?).

**Three traps the design avoids, all confirmed in the record:**
1. *Supervised vs SSL.*  Our workhorse uses labels and the LeJEPA-style arm
   does not, so beating it on a probe is unfair in our favour.  Frame as
   ss-ft vs supcon-ft (does SIGReg COST anything), with SSL arms for context.
2. *Published numbers.*  We have never compared to a published LeJEPA result
   and must not start: our LeJEPA backbone is a **community reproduction**
   (`OK-AI/lejepa-vitb16-pretrain-in1k`, card 72.0 IN-1k, explicitly below the
   official release, `40_dtd_bases.py:11-15`), `PAPER_DTD["lejepa"] = None`,
   and there is no ImageNet evaluation anywhere in the campaign.
3. *Full fine-tuning.*  Exp 62 aircraft closed-set top-1: plain CE-ft reaches
   **75.8-76.6** while ss-ft reaches **49.0-54.5**.  A broad "competitive at
   classification" claim dies on that table.  The defensible claim is about
   the frozen-trunk/head regime, where exp 63 shows supcon_sigreg heads
   beating CE heads (nearest-centroid 0.638 > CE's own 0.625 on LeJEPA-ft).

**A TIE guard, because this is where results get manufactured.**  The gaps the
paper would like to cite (supcon_sigreg vs supcon: 0.008 on scr-simclr, 0.016
on scr-visreg) sit UNDER the campaign's own noise floor (0.017 seed exp 52,
0.019 draw exp 118), single-seed.  The script declares a winner only when the
gap clears `max(floor, sd_a + sd_b)` and prints TIE otherwise.  Those two
comparisons are ties, and the paper must say so.

**Prediction.**  ss-ft ties supcon-ft on supervised top-1 (the no-cost claim)
while winning decisively on purity.  On `acc` the supervised arms already beat
the LeJEPA-style arm on all four DINO exp-70 cells (cars .330 vs .083, flowers
.940 vs .800, dtd .782 vs .635, galaxy10 .558 vs .408) — the probe should
agree, but that is context, not the headline.

**Falsifier.**  ss-ft loses supervised top-1 to supcon-ft by more than the
noise floor -> discovery capability DOES cost representation quality, and the
paper must say so and drop the no-cost framing.

**Demonstrated** on the exp-54 CIFAR-10 banks (evaluation-only, seconds): the
two supervised distance-NPLM arms tie each other (0.8964 vs 0.8865, gap 0.0099
< 0.017) and beat every instance-NPLM arm by 0.17-0.36.

**Cost.**  Minutes per cell given banks; piggybacks on the Block B runs.

### Exp 134 — What is the residual construction actually doing?

> **(B) DONE 2026-08-27 on 45 TRAINED pairs (all 15 transfer cells x 3) --
> RETRACTED, question closed.**  Ratios 0.50-6.3, only 6/45 one-sided; no
> combiner rescues eucl (standardise +0.000 mean, whiten hurts; best-of-4
> beats the parent's eucl in 9/45).  The geometry loss is the construction's.
> Probe gain robust (41/45).  Only the resnplm CHILD carries geometry
> (child-only eucl >= parent on 8/15).  Per-half standardisation is a safe
> default (+0.028 probe), not a repair.  (A) -> exp 134a, (C) -> exp 134c,
> both queued.  `logs/SUMMARY_TABLES.md`.

> **CODE READY 2026-08-27.**  `--selftest` green (4 checks) +
> `tests/test_residual_audit.py` (10).  NEEDS parent/child banks.
> (133 is taken by work on the other machine.)

Three questions that must be settled before the residual can carry a results
section.  The motivating pattern is suspiciously specific: the concat wins the
novelty probe in **30/30** cells but LOSES `eucl` in **14/15** and `mahaT` in
**11/15** against its own parent, and never improves purity.

**(A) Construction or width?**  The concat is two full networks — double the
parameters.  The flagship +0.148±0.004 (cars/VISReg) has never been compared to
a same-width NON-residual control.  Exp 85 already fired a related falsifier:
width-matched, "the single residual matches/beats the 3-way (+0.051 vs +0.048)"
on that very cell.  Arms: parent (1x) | residual concat (2x) | width-matched
non-residual control (2x).

**(B) Is the geometry loss a SCALE ARTEFACT?**  The halves are joined by a raw
`np.concatenate` with no normalisation, and they have no reason to share a
scale — the parent is a scale-free softmax space, the child carries a SIGReg
marginal pinning unit variance.  A linear probe is scale-robust (weights absorb
the mismatch); Euclidean distance is not.  Demonstrated, novelty living ONLY in
the child half:

    space                  probe    eucl
    parent alone          0.9493  0.4983
    child alone           1.0000  1.0000
    concat, child x0.05   0.9927  0.5143   <- probe sees it, distance does not
    concat, child x0.2    1.0000  0.7313
    concat, child x1      1.0000  1.0000
    standardised          1.0000  1.0000
    unitnorm              1.0000  1.0000
    whiten                1.0000  0.9660

That is exactly the archived pattern.  **If standardising recovers the geometry
on real halves, "the residual only helps the probe" is an artefact of the
combination rule and the paper's framing changes.**  First thing to compute is
the child/parent RMS-norm ratio — one number that predicts whether this matters.

**(C) Should the residual come AFTER discovery?**  Every residual is built on
the PRE-discovery parent (discovery then runs on the concat).  The reverse has
never been tried, and there is a mechanism: the residual removes what the
ANCHOR SET explains, and after discovery the anchor set is strictly richer, so
the child would carry more genuinely-unexplained variance.  Orders: archived |
parent->discovery->child->concat | both.  Prediction: helps where purity
cleared the gate, no-op where it did not.  Falsifier: it helps where purity was
~0, so the mechanism story is wrong.

> **(B) PARTIALLY RUN 2026-08-27** on real exp-54 CIFAR-10 spaces using the
> UNTRAINED linear residual (`logs/exp134/untrained_residual_cifar10.json`).
> **The ratio-predicts-sensitivity mechanism is CONFIRMED** — at ratio 0.945
> every combiner agrees to 3 decimals; at 0.156-0.194 the combiner moves the
> probe by up to 0.13.  **The artefact explanation is NOT confirmed and the
> strong form is retracted**: standardising does not rescue `eucl`, sometimes
> hurts it (0.7871 raw -> 0.7092), and whitening hurts it everywhere.  On this
> evidence the geometry trade-off is REAL and the residual stays probe-centric.
>
> Incidental, worth following up: the UNTRAINED linear residual already lifts
> the probe 0.7894 -> 0.9151 on nplm_sup_dist.  Some of the construction's
> value may not need the child trained at all.
>
> Still open, because the proxy differs from the archive in two ways: the 14/15
> eucl losses are TRAINED children on TRANSFER cells, and the trained child is
> predicted to be LARGER than its parent (SIGReg lam=5 drives unit per-dim
> variance, RMS = sqrt(d) = 1.6x-3.8x the parent here) rather than smaller.
> **First thing to run: the child/parent RMS ratio on a real trained pair.  If
> it is near 1, (B) is closed and only (A) and (C) matter.**

**Cost.**  (B) is evaluation-only given banks — minutes, do it first.  (A) and
(C) need training.

**Doc error fixed alongside:** PAPER_PLAN claimed "universal residual parent
12/12"; it is **10/12**, and the 12/12 that does hold is against the discovery
pipeline rather than the best known space.
