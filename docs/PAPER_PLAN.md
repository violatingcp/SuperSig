# Paper plan — discovery-focused rewrite (decided 2026-08-25)

Working state of the restructure discussion.  The long-form technical report
(`discovery_metrics_iclr.tex`, 41pp / 25pp main text) stays as-is; this plans a
9-10pp ICLR submission version alongside it.

## Agreed emphasis (PH)

1. **Effectiveness of these spaces for DISCOVERING NEW CLASSES** is the thesis.
2. Demonstrate they beat other spaces *at discovery* specifically.
3. Argue the construction may also yield better representations generally.
4. **Cut the algorithm count** to the few that actually earn a place.

## Recommended shortlist (CC), from the accumulated evidence

"Best at discovery" splits four ways; since the thesis is *finding new
classes*, **pool purity is the binding constraint** (the purity gate).

| role | algorithm | evidence |
|---|---|---|
| discovery workhorse | **SupCon + strong SIGReg (lam=5)** | program's 3 highest purities: DTD 0.795/0.803/0.811; flowers 0.62-0.66; dominates the flowers pre-battery outright |
| few-class calibration | **supervised distance-NPLM** | galaxy10 purity 0.46-0.50, per-event 0.35->0.50 (supcon 0.02); most base-robust arm (spread 0.009 vs 0.076) |
| parent + baseline | **plain SupCon** | universal residual parent 12/12; best semantics; only universally safe discovery pipeline (12/12) |

**Constructions to keep (2):**
- residual decomposition + concat — 12/12, every dataset record; the res-nplm
  CHILD alone is the best detector measured anywhere (cars/visreg f95 0.049)
- frozen-space discovery with density-ratio (SparKer) pooling — broke the
  C100 geometry block (purity 0.121 -> 0.358, r2 0.418; **exp 127, BN
  actually frozen: 0.268 r1, round-2 rise not reproduced — quote 0.27,
  not 0.36**); made aircraft
  discovery strictly non-negative (record 0.8634 retained WITH per-event 0.523)

**Drop to a one-line ablation (4):** plain SimCLR, LeJEPA/sigreg-SSL,
NPLM-bilinear, NPLM-distance.  Never on the Pareto front for discovery
(purity 0.02-0.16); bilinear is high-variance and its performance lives in its
MIScalibration (exps 81/83).  10 objectives -> 3.

## Framing constraints (do not violate)

- **Do NOT argue SIGReg helps because it makes distance a log-likelihood.**
  Exp 105: forcing unit width is decorative.  Exp 120: seen-class fidelity
  ANTI-correlates with novelty calibration.  Defensible claim instead: SIGReg
  earns its place EMPIRICALLY (top purities) and STRUCTURALLY (it licenses the
  clustering -- Euclidean IS Mahalanobis under it).
- **Keep the tau basin out of the headline** — exps 120/121: unreachable by any
  label-free selection.  A landscape finding, not a recipe.
- **State the purity gate as a condition**, not a footnote: the method works
  where novelty is geometrically or density-ratio outlying; cars and C100
  (pre-109) show where it does not.
- Exp 123: the four-currency TAXONOMY is not validated by the correlation
  structure (within ~= between); the ~3-dimension count is.  Demote "four
  currencies" from a claim to an organizing device.

## Proposed 9pp outline (CC; not yet agreed)

1. The discovery problem: why novelty detection != classification (1)
2. Why standard spaces fail the purity gate (1.5)
3. The three objectives (2)
4. The two constructions (2)
5. Pipeline + records; better representations as a side effect (1.5)
6. When it does not work; limitations (1)
Everything else -> appendix (panel, SparKer protocol, reach/residual/per-dataset
tables, gradient appendix, tau section, LID mechanism).

## Provenance and holdout policy (decided 2026-08-25, PH)

Two structural decisions, plus the collision they create.

### 1. Switch the CIFAR cells to the leakage-free scratch lineage

Every CIFAR cell built with `pretrain=ds` inherits hub weights that already saw
the held-out class (`supersig/models.py:80-82`, self-flagged in code since
exp 09 but stated in NO doc and NOT in the tex dataset caption at
`discovery_metrics_iclr.tex:227`).  Discovery claims must not rest on it.
Clean lineage = exps 67/68 (`pretrain=None`, random init).

**Gap found:** exp 67 carried only ONE of the three shortlisted objectives.
Its `supsig` is the repulse/proto recipe, not SupCon+SIGReg; its `nplmcw` is
distance-NPLM + *classwise* SIGReg, not the plain arm.  Exp 50 `--scratch-base`
numbers are heads on a label-free scratch base, not end-to-end scratch.
**Fixed:** added `ssig` (= exp-70 `ss-ft`, ss_lam=5) and `nplmsd`
(= exp-70 `nplm-sup-ft`) to exp 67; verified objective-identical to the exp-70
step functions.  Exp 68 needs no change (`--bases` is generic).

Expect an honest NEGATIVE result here: exp 68's scratch C100 purities are
0.00-0.01, far under the 0.15 gate, and per-event power is dead at every
fraction (SparKer only fires at f=0.05).  Report the scratch cell as the
leakage-free control that shows C100 is genuinely hard, not as a showcase.

### 2. HOLDOUT POLICY (settled 2026-08-25, PH)

**Framing: REGIME SPLIT, adopted.**  Not main-vs-appendix by holdout count.
Single-holdout is the hard, low-rate regime where only calibrated per-event
scoring works; multi-class novelty is the higher-rate regime where pooling
constructions unlock the purity gate.  The rate floor is a RESULT, not an
omission.

Three operational rules:

1. **Every table is split by regime.**  Single-holdout and multi-holdout
   numbers never share a table.  They reach different conclusions (below), so
   pooling them is a category error.
2. **Run single-holdout for EVERYTHING** — all datasets, all cells.  This is
   the uniform backbone of the paper.
3. **Multi-holdout only for the label-rich datasets**, where holding out 10
   classes is a small fraction of the label set and the novel-class rate is
   high enough to matter: cars (196), flowers (102), aircraft (100),
   cifar100 (100), dtd (47).  NOT galaxy10 (10 classes — already nh=1).

Evidence the regimes genuinely differ (exps 89/109, the only direct
same-dataset comparisons):
- exp 109, frozen density-ratio pool on C100: h5/h10 clear the gate (best r1
  purity **0.358**, r2 0.418, vs the 0.121 distance-grid ceiling), but **h1
  stays at 0.03**.  CAVEAT (exp 128, 2026-08-26): calling this a "rate floor"
  overstates it.  purity <= min(1, b/q), so at q=0.05, b=0.01 the CEILING is
  0.20 — a 20x enrichment is available and we measure 3x.  h1 is
  signal-efficiency blocked, not information-theoretically blocked.  What does
  bind at h1 is BIC detectability: tightening q to raise purity leaves ~15
  novel points, too few to be a cluster.  State it that way in the paper.
- the quantile-strictness conclusion **inverts**: strictest q best at h5/h10,
  worst at h1.
- exp 89: "discovery hurts the probe" (-0.031..-0.038) is multi-holdout-only;
  at h1 the probe delta is ~0.
No experiment shows the *objective ranking* flipping between regimes.
Caveat: `118_holdout_audit.py` (holdout-DRAW stability, not count) has **never
been run** — no `logs/exp118*`.  Its reassurance is a prediction.

### 2b. Mechanism (implemented)

The rule `nh = 1 if ds == "galaxy10" else 10` was duplicated across **22
experiment files (23 sites)**, so neither regime was runnable on demand and a
single-holdout run would have **silently overwritten** every multi-holdout
artifact — including exp-70/71 fine-tune CHECKPOINTS, whose `_seen` suffix
means "trained excluding the holdouts" (contents differ, name does not), and
exp-80's resume cache, which would then have reused stale results as valid.

Now centralized in **`supersig/holdouts.py`**.  One env var switches the whole
battery and tags every artifact it writes:

    SUPERSIG_NH=1 python experiments/70_cars_ft_suite.py --dataset dtd ...

With `SUPERSIG_NH` unset, holdout sets and filenames are byte-identical to the
archive (`run_tag() == ""`), so existing results and resume logic are
untouched.  Enforced by `tests/test_holdouts.py` (25 tests), which also fails
if any experiment reintroduces the hardcoded rule.

## Naming hazard to fix first

`ss-ft` = **SupCon + lam=5 SIGReg** in exps 70+ (`70_cars_ft_suite.py:88`), but
`"ss"` in `docs/LOSSES.md` = **SimCLR + SIGReg** (unsupervised, exp 34).  Same
abbreviation, two different algorithms.  Rename before writing.

WORSE THAN LOGGED: exp-50's `supcon_sigreg` is supervised/cosine/softmax/sigreg
at the HybridContrastiveLoss **default lam=1**, while `ss-ft` is **lam=5**.  So
three distinct objectives share the "supcon+sigreg" family name at two different
regularizer strengths.  Verified numerically (aux term 0.381 vs 0.066 on the
same batch).  Do not pool exp-50 `supcon_sigreg` rows with exp-70 `ss-ft` rows.

## Open / next

- GPU, leakage-free shortlist (NEW, highest priority — unblocks decision 1):
  `python experiments/67_scratch_pretrain.py --arms supcon ssig nplmsd`
  then `python experiments/68_scratch_discovery.py --bases supcon,ssig,nplmsd`
  (2 x 200-epoch pretrains + one discovery pass, CIFAR-100, single holdout {4})
- GPU: `113 --save-embs` -> `121 --tau-archive` + `122` (one run, two analyses)
- Consider running `118_holdout_audit.py` — it is the only check on whether
  holdout CHOICE (not count) moves rankings, and it has never been run.
- PH to sign off on the regime-split resolution above; then draft the 9pp version
