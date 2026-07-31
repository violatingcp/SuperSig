# SuperSig research questions — the systematic testing frame

The recurring questions this program keeps asking, their current status,
and the standard protocol for testing the next idea.  Detailed numbers:
`logs/SUMMARY_TABLES.md`.  Loss definitions: `docs/LOSSES.md`.  Metrics and
reporting: `docs/METRICS.md`.

## Q1. Probe vs calibrated geometry — can one space serve both?
The central tension.  Softmax-contrastive spaces (SupCon) maximize
linearly-readable structure but carry no absolute distance information
(per-event power 0.000); calibrated spaces (NPLM, supervised SIGReg) are
ready-to-use distance metrics at a probe cost.  Best single-space
compromise so far: `nplm_dist_sup_cw` on CIFAR-10 (probe 0.895 pre / 0.977
post, best geometry in program).  Status: OPEN in general — the answer so
far is "one space + discovery update", not "one loss".

## Q2. Which loss corner wins in which regime?
Regime variables: #seen classes vs dim, labels available, per-class data,
fine-grained vs separable.  Current rules of thumb:
- few classes, labels (C10): supervised distance NPLM + classwise SIGReg.
- many classes (C100, 32-D): label-free bilinear NPLM (probe within seed
  noise of SupCon, no labels, best geometry+gaussianity).
- fine-grained low-data frozen-features (aircraft): supcon_sigreg — the
  SIGReg marginal helps exactly there, opposite of CIFAR.
- 64-D C100 (exp 50/53 64d): bilinear arms REGRESS (0.935->0.858,
  0.925->0.825); supervised distance arms gain mildly.  High-D is not
  free for NPLM losses.

## Q3. Does NPLM calibration deliver absolute, usable distances?
Yes — verified analytically (34f), geometrically (exp 54 plots: tight
class Gaussians + radially displaced novelty), and operationally
(per-event power 0.20 with no anchors/discovery vs 0.000 for SupCon).

## Q4. Which Gaussian marginal: none / global / classwise?
Global SIGReg = default.  Classwise is strictly better IFF anchors are
realizable (dim >= n_classes, C10); it is harmful for label-free arms
when anchors are unreachable (C100 at 32-D AND 64-D — cent->anchor ~3.5).
lam is inert early (exp 52); tau is the knob that matters.

## Q5. Does discovery (cluster + update the space) help, and when?
- C10: spectacularly — the calibrated tail is exactly what pooling needs
  (purity 0.34); NPLM + proto-ft discovery is the best full pipeline on
  record (probe post 0.96-0.98, per-event 0.73-0.75 vs sup 0.59).
- C100: rate-blocked (500 holdout vs ~2500-event tail, purity ~0.003-0.01);
  conf mask (exp 56) is only a do-no-harm gate.  Untested fixes: stricter
  tau_quantile (0.995+), multi-class holdouts, larger holdout fraction.
- ft objective: proto/repulse >> NPLM-ft for the probe (0.966 vs 0.812 on
  nplm_sup_dist; exp 58 power grid pending at time of writing).

## Q6. Which dataset-level statistic when?
Statistic-geometry matching: parametric (per-event, Maha) needs calibrated
spaces; kernel stats (SparKer, MMD) need scale-matched kernels — annealed
median-heuristic sigma by default (exp 57), never fixed sigma across
heterogeneous geometries.  MMD is the dimension- and dataset-robust
statistic; Maha is dead on C100 regardless of space.

## Q7. Do conclusions transfer off CIFAR?
Partially.  Aircraft (frozen DINO, exp 51) inverts the CIFAR ordering
(supcon_sigreg dominates; label-free NPLM weak).  Fine-grained low-data
regimes need their own sweep; DTD/transfer suite (exps 37-49) tells the
same story for the older losses.

## Q8. Label efficiency — how far do augmentations alone go?
Plain LeJEPA and SimCLR are never on the Pareto front (all 3 datasets).
Label-free NPLM-bilinear is the interesting arm: near-SupCon probe on C100
without labels.  Unresolved: the 0.007 gap needs a 3-5 seed run.

## Standard protocol for testing a new loss / idea

1. Add the loss to `supersig/losses.py` (or a corner of
   HybridContrastiveLoss); calibration/sanity check a la `34f`.
2. Smoke test in an exp-50-style suite script with `--quick`.
3. Part A on cifar10 + cifar100 at 32-D, holdout 4, seed 0, 20 epochs,
   against the standard 8 arms (simclr / lejepa / ss / supcon /
   supcon_sigreg / nplm corners).  Report the full metric row — probe AND
   geometry.
4. Pre-discovery power batteries (annealed-sigma SparKer).
5. If geometry is promising: discovery pre/post (exp-55 pattern).
6. Knob scan only for knobs that matter (tau, not lam); paired seeds.
7. Gaps < 0.02 probe: multi-seed before claiming.
8. Aircraft (exp-51 harness) if the claim should generalize.
9. Append to SUMMARY_TABLES.md, follow docs/METRICS.md artifact naming,
   commit code+logs+npz+plots, update these docs if a verdict changed.

## Open items (as of exp 58)

- exp 58 power grid: NPLM-ft vs proto-ft (probe verdict already: proto).
- 64-D C100 classwise final arm + batteries; fold into SUMMARY_TABLES.
- 3-5 seed run of nplm_bilinear vs supcon on C100 (the 0.007 question).
- C100 discovery rate fixes: tau_quantile 0.995, multi-class holdouts.
- Annealed-sigma rerun of historical k1 SparKer rows where geometries
  differ from the supervised scale (exp-33 contrastive arms).
- Aircraft discovery protocol (never run).
