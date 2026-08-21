# Proposed tests to improve performance — exps 81–104

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
| 99 | SparKer discovery reach `f95` | **done** — and mostly *unresolvable* on the current fraction grid (see exp 100) |
| 88 | classwise-realizable C100 | **BOTH CLAUSES FALSIFIED** — cent->anchor stall (~3.5) dim-independent at 100/128/200-D (optimization equilibrium, Q4 restated); unpredicted: supcon_sigreg control posts mahaT 0.545–0.558 at 100–128-D with probe 0.90–0.91, above the old ceiling |
| 91 | multi-seed the records | **done** — all citable, no ordering flips: aircraft 0.866±0.002, flowers 0.887±0.019, dtd 0.867±0.004 (seed-2 new best), galaxy10 0.971±0.004 |
| 97 | M / sigma-schedule systematics | **split** — sigma_ratio flat (annealed schedule robust); M INVERTS the prediction: high-ID wants FEWER kernels (supcon M=4 0.60 vs M=64 0.24) |
| 98 | SparKer-ft as the discovery ft objective | **FAILS beyond its falsifier** — worse than proto AND nplm on every column incl. its own statistic (post SparKer 0.06–0.12 vs 1.00); statistic-chasing destroys separation; with 96, contraindicates exp 95 as specified |
| 89, 83 | gate grid; variance-reduced NPLM | in flight (89 running, 83 queued) |
| 84, 85 | two-stage strict; iterated residuals | queued (85 quick smoke: 3-way +0.031, width control NEGATIVE) |
| 95, 100–104 | — | **not started** (95 contraindicated by 96+98 in the alternating form) |

Six of eleven predictions were falsified.  That is the intended hit rate: the
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

**Done** (see the status board above): 80, 81, 82, 86, 87, 88, 90, 91, 92,
92b, 93, 94, 96, 97, 98, 99.
**In flight** (2026-08-21): 89 (gate grid, running) -> 83 (variance-reduced
NPLM) -> 84 (two-stage strict) -> 85 (iterated residuals), chained on the
GPU.

Remaining, in the order we would run them:

1. **Exp 104** (interpretability panel) — evaluation-only, and it measures the
   property the program is built around but has never quantified: whether
   distance to a labelled anchor is a delta log-likelihood.  Script shipped.
2. **Exp 100** (dense fraction scan) — evaluation-only; 102 of 106 reach
   numbers are currently bounds rather than measurements.
3. **Exp 102** (residual child as standalone detector) — evaluation-only,
   follows directly from exp 80's surprise.
4. **Exp 103** (LID neighbourhood decomposition) — evaluation-only; third
   attempt at the LID mechanism, and worth capping if it also fails.
5. **Exp 101** (does frozen-space discovery generalize beyond aircraft?) — if
   exp 86 and 92b hold elsewhere they become the default recipe rather than a
   fine-grained fix, and the paper's framing changes again.
6. **Exp 95** (SparKer as a training loss) — formally cleared by exp 94, but
   CONTRAINDICATED by 96 (warm-start harmful) and 98 (SparKer-ft destroys
   the separation it optimizes); needs a class-structure-preserving redesign
   before any attempt.

Standing item resolved 2026-08-21: `tests/test_calibration.py` HAS been
executed on this machine — full suite green (103 passed) both at merge time
and after; the identities it pins are verified.
