# SuperSig research questions — the systematic testing frame

The recurring questions this program keeps asking, their current status,
and the standard protocol for testing the next idea.  Detailed numbers:
`logs/SUMMARY_TABLES.md`.  Loss definitions: `docs/LOSSES.md`.  Metrics and
reporting: `docs/METRICS.md`.

## Q1. Probe vs calibrated geometry — can one space serve both?
The central tension.  Softmax-contrastive spaces (SupCon) maximize
linearly-readable structure but carry no absolute distance information
(per-event power 0.000); calibrated spaces (NPLM, supervised SIGReg) are
ready-to-use distance metrics at a probe cost.  Exp 59 largely closes this
on C10: `supcon+nplm` concat (0.954 probe + 0.768 eucl, discovery-free)
and `sup->res-nplm` (probe 0.984 post + record geometry + near-max post
power) each serve both currencies; the NPLM-bilinear half is a strict
upgrade over SimCLR as the feature half of any concat.  Remaining gap:
no SINGLE standalone loss does it yet.

## Q2. Which loss corner wins in which regime?
Regime variables: #seen classes vs dim, labels available, per-class data,
fine-grained vs separable.  Current rules of thumb:
- few classes, labels (C10): supervised distance NPLM + classwise SIGReg.
- many classes (C100, 32-D): label-free bilinear NPLM for GEOMETRY (eucl
  0.50 vs 0.40, mahaT 0.47 vs 0.39 across seeds) but NOT probe parity —
  exp 61 (5 paired seeds): supcon 0.940+-0.004 vs nplm_bilinear
  0.855+-0.042, paired diff -0.085 (t=-4.7); the exp-50 0.935 was a
  tail draw of a heavy-tailed seed distribution.
- fine-grained low-data frozen-features (aircraft): supcon_sigreg — the
  SIGReg marginal helps exactly there, opposite of CIFAR.
- 64-D C100 (exp 50/53 64d, complete): bilinear arms REGRESS (0.935->0.858
  losing probe AND geometry; 0.925->0.825); supervised distance arms and
  the softmax arms gain mildly (supcon_sigreg 0.913, simclr 0.885).
  High-D is not free for NPLM losses; 32-D stands as the C100 default.

## Q3. Does NPLM calibration deliver absolute, usable distances?
Yes — verified analytically (34f), geometrically (exp 54 plots: tight
class Gaussians + radially displaced novelty), and operationally
(per-event power 0.20 with no anchors/discovery vs 0.000 for SupCon).

## Q4. Which Gaussian marginal: none / global / classwise?
Global SIGReg = default.  Classwise is strictly better on C10 (dim >=
n_classes).  RESTATED by exp 88: on C100 the cent->anchor stall (~3.5)
is an OPTIMIZATION equilibrium, not a realizability limit — it persists
unchanged at 100/128/200-D across seeds, and cw mahaT declines with dim
(0.47 -> 0.33).  dim >= n_classes is neither the enabler nor the fix.
Meanwhile supcon_sigreg at 100-128-D posts mahaT 0.55 + probe 0.91 (3
seeds) — above the old C100 calibration ceiling from the softmax corner.
lam is inert early (exp 52); tau is the knob that matters.

## Q5. Does discovery (cluster + update the space) help, and when?
- C10: spectacularly — the calibrated tail is exactly what pooling needs
  (purity 0.34); NPLM + proto-ft discovery is the best full pipeline on
  record (probe post 0.96-0.98, per-event 0.73-0.75 vs sup 0.59).
- C100: rate-blocked (500 holdout vs ~2500-event tail, purity ~0.003-0.01);
  conf mask (exp 56) is only a do-no-harm gate.  Untested fixes: stricter
  tau_quantile (0.995+), multi-class holdouts, larger holdout fraction.
- ft objective (exp 58, complete): the probe-vs-calibration dissociation
  recurs at the fine-tune level.  proto/repulse wins the probe (0.97 vs
  0.81-0.84); NPLM+sigreg-ft wins EVERY power statistic (per-event 0.57 vs
  0.30-0.48 @0.02, SparKer 1.00/0.92, Maha 0.90, MMD 0.74), holds round-2
  pool purity (0.22-0.30 vs 0.03-0.11) and does not fragment anchors.
  Choose by consumer: linear readout -> proto; detection -> NPLM-ft.

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
Plain LeJEPA and SimCLR are never on the Pareto front (all 3+ datasets).
Label-free NPLM-bilinear buys calibrated geometry without labels, but NOT
probe parity: exp 61 settled the C100 question decisively for SupCon
(-0.085 paired, t=-4.7); nplm_bilinear's 10x-larger seed variance is an
open question of its own (what stabilizes the good seeds?).

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
10. If the change touched `supersig/` or the shared battery helpers
    (exps 28/29/30), run the unit tests first:
    `/home/pharris/venv/bin/python -m pytest tests/ -q` (~3 s).

## Open items (as of exp 79)

- exp-58 follow-up: a two-stage ft (NPLM-ft then a probe head, or
  proto-ft with an NPLM calibration term) to get 0.97 probe AND 0.57
  per-event in one space.
- C100 discovery rate fixes: tau_quantile 0.995, multi-class holdouts.
- Annealed-sigma rerun of historical k1 SparKer rows where geometries
  differ from the supervised scale (exp-33 contrastive arms).
- Aircraft low-purity discovery cells (0.32-0.53, probe-negative in
  exp 72): the LID pool scorer does NOT rescue them (exp 79 — LID
  separation is weak off-manifold); a different lever is needed.
- nplm_bilinear's 10x seed variance (exp 61): what stabilizes the good
  seeds?

Done since exp 58: the C100 multi-seed question (exp 61 settled it for
SupCon, -0.085 paired); aircraft discovery (exp 72, purity-gated);
residual-concat multi-seed validation (exp 75, 17/18 paired positive);
LID promoted into the standard battery after verification (exps 77-79).
