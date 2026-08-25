# Proposed tests to improve performance — exps 81–123

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

### Exp 122 — What IS the basin geometry? *(proposed)*

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

### Exp 123 — Are the four currencies actually independent? *(proposed)*

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
