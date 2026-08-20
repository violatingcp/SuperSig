# Proposed tests to improve performance — exps 80+

Companion to [QUESTIONS.md](QUESTIONS.md) (whose "open items as of exp 58" list
this supersedes).  Every entry follows the standard protocol in
[METRICS.md](METRICS.md): `experiments/NN_<topic>.py` with `--quick`, npz to
`logs/expNN/`, paired seeds across cells, full metric row (probe AND geometry),
append to `logs/SUMMARY_TABLES.md`.

Each test states a **prediction** and a **falsifier**.  A test whose falsifier
is not stated is not a test; a run that cannot come out the other way is not
worth the GPU hours.  Cost is wall-clock on one GPU, rough.

Ordering is by expected value per unit cost, not by topic.

---

## Tier 1 — cheap, decisive, and targeting a known loss of performance

These three all attack the same thing: the label-free NPLM arm loses ~0.085
probe to supcon on C100 *on average* while its best seeds are competitive
(0.9349 archived vs 0.8548±0.042 over 5 seeds).  That is not a weak method, it
is a **high-variance** method, and variance is recoverable performance.  The
gradient analysis in `discovery_metrics_iclr.tex` App. A predicts where the
variance comes from and hands us two interventions.

### Exp 80 — Is the seed variance a property of the CRITIC, not the estimator?

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

### Exp 81 — Calibration residual as a free seed-selection and early-stop signal

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

**Cost.** ~5 runs, or free if folded into exp 80.
**Payoff.** An unsupervised model-selection criterion for the whole NPLM
family — and, if it works, it should be added to the standard battery as a
reported column.

### Exp 82 — Variance-reduced NPLM: bound the exponent instead of clamping it

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
Report probe mean±sd, calibration residual (exp 81), and the per-event power —
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

### Exp 83 — Two-stage recipe under the strict open-world protocol

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

### Exp 84 — Iterated residuals (does the residual trick stack with itself?)

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

### Exp 85 — Fix aircraft discovery by freezing the probe directions

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

### Exp 86 — LID on the residual half

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

### Exp 87 — Classwise SIGReg where the anchors are actually realizable

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

### Exp 88 — CIFAR-100 discovery rate unblocking

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

### Exp 89 — A predictor for *which* novelty score to use

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

### Exp 90 — Multi-seed the uncited records

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

1. **Exp 86** (LID on residual half) — evaluation only, hours.
2. **Exp 80 + 81** (critic variance + calibration residual) — one script, cheapest
   training test, biggest mechanistic payoff.
3. **Exp 85** (frozen-parent aircraft discovery) — parents exist, cheap, tests
   the paper's central claim.
4. **Exp 87** (realizable classwise C100) — most likely new record.
5. **Exp 82** (variance-reduced NPLM), **89** (score predictor).
6. **Exp 83, 84** (two-stage, iterated residuals) — moderate cost, record-chasing.
7. **Exp 88, 90** (gate calibration, multi-seed) — protocol debt, before writeup.
