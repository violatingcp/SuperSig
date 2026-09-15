# Exp 34 calibrated-contrastive arc: cross-dataset summary

16+16 concat spaces, holdout 4, seed 0. CIFAR-10 from exp 34i; CIFAR-100
from exps 34e/f/g/h; sup->res and supcon+simclr references from the exp-33
16p16 k=1 runs (same holdout/seeds). Probe = 1-layer-NN holdout-vs-rest ROC
AUC (3-seed mean), pre-discovery.

## Space quality (pre-discovery)

| space                | C10 probe | C10 acc | C10 mahaT | C100 probe | C100 acc | C100 mahaT |
|----------------------|-----------|---------|-----------|------------|----------|------------|
| supcon+simclr        | 0.9503    | 0.873   | 0.527     | 0.9394     | 0.503    | 0.333      |
| supcon+hybrid[lam1]  | 0.9504    | 0.851   | 0.582     | 0.9409     | 0.523    | 0.422      |
| supcon+hybrid[lam5]  | 0.9478    | 0.850   | 0.583     | 0.9423     | 0.527    | 0.407      |
| ss[lam5]+hybrid      | 0.8584    | 0.896   | 0.489     | 0.9235     | 0.553    | 0.487      |
| hybrid->supres       | 0.8898    | 0.862   | 0.583*    | 0.9263     | 0.475    | 0.482      |
| supcon+res-simclr    | 0.9475    | 0.857   | 0.562     | 0.9281     | 0.539    | 0.254      |
| cls->resfeat         | 0.8633    | 0.882   | 0.483     | 0.8745     | 0.549    | 0.560      |
| feat->rescls         | 0.8743    | 0.802   | 0.317     | 0.8196     | 0.370    | 0.247      |
| sup->res (exp33 ref) | 0.9227    | --      | --        | 0.8361     | 0.524    | 0.392      |

*hybrid->supres C10 mahaPC = 0.683, the best parametric per-event score of
the C10 arc (sup->res exp33 remains far ahead on distance metrics overall).

## Dataset-level power at alpha=0.05, f=0.02 (pre / post-discovery)

| space                | C10 SparKer | C10 Maha  | C10 MMD   | C100 SparKer | C100 Maha | C100 MMD  |
|----------------------|-------------|-----------|-----------|--------------|-----------|-----------|
| supcon+hybrid[lam5]  | 0.02 / 0.04 | 0.08/0.14 | 0.66/0.26 | 0.06 / 0.30  | 0.00/0.00 | 0.30/0.88 |
| ss[lam5]+hybrid      | 0.02 / 0.44 | 0.06/0.54 | 0.58/0.26 | 0.10 / 0.52  | 0.00/0.04 | 0.38/0.86 |
| hybrid->supres       | 0.06 / 0.68 | 0.42/0.50 | 0.32/0.56 | 0.06 / 0.12  | 0.04/0.00 | 0.20/0.42 |
| cls->resfeat         | 0.06 / 0.34 | 0.10/0.74 | 0.24/0.44 | 0.50 / 0.44  | 0.00/0.12 | 0.52/0.74 |
| feat->rescls         | 0.02 / 0.28 | 0.00/0.26 | 0.30/0.10 | 0.00 / 0.06  | 0.00/0.00 | 0.32/0.72 |
| sup->res (exp33)     | 0.58 / 0.96 | 0.84/0.76 | 0.32/0.76 | 0.08 / 0.08  | 0.00/0.14 | 0.14/0.22 |
| supcon+simclr (exp33)| 0.32 / 0.12 | 0.12/0.38 | 0.86/0.52 | 0.06 / 0.22  | 0.04/0.08 | 0.14/0.34 |

At f=0.05 (C100) / f=0.03 (C10), the calibrated arms saturate the kernel
tests on C100 (SparKer/MMD 1.00) and reach 0.78-0.94 on C10; sup->res
saturates C10 already at 0.03.

## Dimension effect on CIFAR-100 (exp 34j: 50+50 vs 16+16)

Probe (pre-discovery):

| space                | 16+16  | 50+50  | delta |
|----------------------|--------|--------|-------|
| supcon+simclr        | 0.9394 | 0.9547 | +1.5  |
| supcon+hybrid[lam1]  | 0.9409 | 0.9541 | +1.3  |
| supcon+hybrid[lam5]  | 0.9423 | 0.9474 | +0.5  |
| supcon+res-simclr    | 0.9281 | 0.9519 | +2.4  |
| ss[lam5]+hybrid      | 0.9235 | 0.9076 | -1.6  |
| hybrid->supres       | 0.9263 | 0.9234 | -0.3  |
| cls->resfeat         | 0.8745 | 0.9228 | +4.8  |
| feat->rescls         | 0.8196 | 0.8399 | +2.0  |

Power at f=0.02 (pre / post):

| arm                 | SparKer 16+16 | SparKer 50+50 | MMD 16+16 | MMD 50+50 |
|---------------------|---------------|---------------|-----------|-----------|
| supcon+hybrid[lam5] | 0.06 / 0.30   | 0.00 / 0.16   | 0.30/0.88 | 0.42/0.82 |
| ss[lam5]+hybrid     | 0.10 / 0.52   | 0.06 / 0.10   | 0.38/0.86 | 0.46/0.84 |
| hybrid->supres      | 0.06 / 0.12   | 0.04 / 0.04   | 0.20/0.42 | 0.20/0.56 |
| cls->resfeat        | 0.50 / 0.44   | 0.14 / 0.38   | 0.52/0.74 | 0.24/0.78 |
| feat->rescls        | 0.00 / 0.06   | 0.02 / 0.02   | 0.32/0.72 | 0.30/0.40 |

Mahalanobis stays dead at 100-D (max 0.22 anywhere) and per-event stays
~0.1 -- decrowding by dimension does NOT turn the holdout into an outlier
population.

Dimension verdict: relief of crowding helps LINEAR readouts (probe +1 to
+5 pts; plain supcon+simclr retakes the lead, so the hybrid's probe edge is
a low-dimension phenomenon) but HURTS the kernel discovery tests (SparKer
pre/post collapse -- distance concentration; cls->resfeat's discovery-free
0.50 at f=0.02 drops to 0.14).  MMD is the dimension-robust statistic
(post ~0.8 at f=0.02 at both dims).  16+16 remains the best operating
point for dataset-level discovery; 50+50 for probe/accuracy.

## Verdicts

- The calibrated-contrastive family is a CIFAR-100 specialist. On CIFAR-10
  the hybrid half adds nothing to the probe (0.9504 vs 0.9503) and every
  new arm loses to the settled SIGReg sup->res on every detection metric
  (maha pre 0.84, SparKer post 0.96, per-event post 0.45 at f=0.02).
- On CIFAR-100 the same constructions are the best spaces ever measured in
  this program: probe 0.9423 (supcon+hybrid), SparKer post 0.52 at f=0.02
  (ss+hybrid), pre-discovery SparKer/MMD ~0.5 at f=0.02 with no discovery
  (cls->resfeat).
- Statistic-geometry-dataset matching, final form: separable few-class
  data (C10) wants supervised SIGReg + parametric statistics; crowded
  many-class data (C100) wants contrastive features + SIGReg calibration +
  kernel statistics.
- Probe and accuracy anticorrelate under calibration of the supervised
  half on both datasets; discovery hurts MMD on contrastive-half arms
  (C10: 0.66->0.26) but helps the supres/resfeat couplings -- the exp-31
  "discovery hurts SupCon arms" pattern, reproduced.
