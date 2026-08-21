# Proposed tests to improve performance — exps 81–98

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
while clamping preserves it).  Exp 82 below assumes those identities hold and
turns the calibration residual into a training-time diagnostic.

---

## Tier 1 — cheap, decisive, and targeting a known loss of performance

These three all attack the same thing: the label-free NPLM arm loses ~0.085
probe to supcon on C100 *on average* while its best seeds are competitive
(0.9349 archived vs 0.8548±0.042 over 5 seeds).  That is not a weak method, it
is a **high-variance** method, and variance is recoverable performance.  The
gradient analysis in `discovery_metrics_iclr.tex` App. A predicts where the
variance comes from and hands us two interventions.

### Exp 81 — Is the seed variance a property of the CRITIC, not the estimator?

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

**Motivation.**  The current guard is a hard clamp at `g_max = 30`, chosen to
preserve calibration (max-subtraction would destroy it — App. A §A.2).  A hard
clamp zeroes the gradient of exactly the pairs that carry the most signal about
miscalibration.  Two alternatives preserve the minimizer while bounding `s`:
(a) **use the distance parametrization** `g = -½||z-z'||² + b(x) + b(x')` with
learned bias, which is bounded above by `b(x)+b(x')` and is the form the critic
collapses to anyway once SIGReg is active; (b) **self-normalized importance
weighting**: track a running estimate of `E_ref[e^g]` and rescale, which is a
*known constant shift* and therefore correctable rather than gauge-destroying.

**Protocol.**  C100 32-D, 5 paired seeds, four arms: current bilinear+clamp;
bilinear+running-normalizer; distance+bias; distance+bias+running-normalizer.
Report probe mean±sd, calibration residual (exp 82), and the per-event power —
the last is essential, since the whole point is to keep calibration.

**Prediction.**  (a) and (b) both cut sd by ≥2× at equal-or-better mean probe;
per-event power is preserved (this is what separates a real fix from
accidentally reinventing a softmax).

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

1. **Exp 87** (LID on residual half) — DONE 2026-08-20, falsifier fired
   (flowers LID lives in the parent half).
2. **Exp 81 + 82** (critic variance + calibration residual) — DONE
   2026-08-20: 81 confirmed (sd splits 3.1x by critic; e^g spread is the
   predictor), 82 falsified (residual does not select seeds).
3. **Exp 92** (SparKer centres as the clustering) — the only proposal with a
   mechanism that predicts a fix for the cars failure; cheap, parents exist.
4. **Exp 94** (null validity) — cheap insurance, and a hard prerequisite for
   exp 95.  Run before, not after.
5. **Exp 86** (frozen-parent aircraft discovery) — DONE 2026-08-20: fixed,
   and more — freeze BOTH halves, discover anchors only (90-99% of the
   per-event gain at zero probe cost, purity no longer collapses).
6. **Exp 88** (realizable classwise C100) — RUNNING 2026-08-20 (early:
   cent->anchor stalls at 3.46 even at 100-D — the optimization-block
   falsifier branch).
7. **Exp 93, 96** (NP pool scorer, critic warm-start) — both cheap; 96 is the
   most direct evidence for the paper's §5 unification.
8. **Exp 83** (variance-reduced NPLM), **Exp 90** (score predictor — DONE
   2026-08-20, falsifier fired: regime rule stays empirical).
9. **Exp 84, 85** (two-stage, iterated residuals) — moderate cost, record-chasing.
10. **Exp 97, 98** (SparKer systematics, SparKer-ft) — after 92–94 report.
11. **Exp 95** (SparKer as a training loss) — the big swing; gated on 94.
12. **Exp 89, 91** (gate calibration, multi-seed) — 91 RUNNING 2026-08-20
    (seeds 1-2 of the four record cells).
