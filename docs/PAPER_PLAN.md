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
  C100 geometry block (purity 0.121 -> 0.358, r2 0.418); made aircraft
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

## Naming hazard to fix first

`ss-ft` = **SupCon + lam=5 SIGReg** in exps 70+ (`70_cars_ft_suite.py:88`), but
`"ss"` in `docs/LOSSES.md` = **SimCLR + SIGReg** (unsupervised, exp 34).  Same
abbreviation, two different algorithms.  Rename before writing.

## Open / next

- GPU: `113 --save-embs` -> `121 --tau-archive` + `122` (one run, two analyses)
- PH to give reaction to the outline; then draft the 9pp version
