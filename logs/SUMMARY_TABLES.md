# SuperSig exps 33-36: full results tables

Holdout 4, seed 0, 16+16 concat spaces (32-D total) unless noted.
probe = 1-layer-NN holdout-vs-rest ROC AUC (pre-discovery, 3-seed mean).
eucl / mahaT = per-event novelty AUC. Power columns = dataset-level power
at alpha=0.05, f=0.02, pre -> post-discovery; "mask" = post with the
exp-35/36b confidence-masked discovery (thresh 0.5). Annealed SparKer.
Sources: exp33 16p16 k1 (classic arms), exp34e-i (arc), exp35/36b (mask),
exp36 (hybrid residual).

## CIFAR-10

| space                 | probe  | acc    | eucl   | mahaT  | SparKer @0.02   | Maha @0.02      | MMD @0.02       |
|-----------------------|--------|--------|--------|--------|-----------------|-----------------|-----------------|
| sup->res-hybrid (36)  | 0.9381 | 0.9177 | 0.8159 | 0.7584 | 0.88/1.00 m0.98 | 0.94/0.84 m0.98 | 0.42/0.94 m0.90 |
| sup->res (36 rebuild) | 0.9298 | 0.9110 | 0.7953 | 0.7354 | 0.84/1.00 m0.98 | 0.80/0.76 m0.90 | 0.38/0.70 m0.84 |
| sup->res (exp33)      | 0.9227 | 0.9098 | 0.8102 | 0.7433 | 0.58/0.96       | 0.84/0.76       | 0.32/0.76       |
| sup                   | 0.9225 | 0.8838 | 0.7783 | 0.6451 | 0.60/0.98       | 0.74/0.70       | 0.44/0.82       |
| joint                 | 0.9030 | 0.8923 | 0.7518 | 0.5109 | 0.46/0.84       | 0.48/0.68       | 0.36/0.70       |
| ssl->supres           | 0.8904 | 0.8812 | 0.6799 | 0.5504 | 0.06/0.70       | 0.56/0.60       | 0.32/0.88       |
| supcon                | 0.9344 | 0.8727 | 0.7525 | 0.6533 | 0.94/0.52       | 0.18/0.36       | 0.78/0.40       |
| supcon+simclr (exp33) | 0.9490 | 0.8850 | 0.6451 | 0.5301 | 0.32/0.12       | 0.12/0.38       | 0.86/0.52       |
| supcon+hybrid[lam1]   | 0.9504 | 0.8507 | 0.4624 | 0.5821 | --              | --              | --              |
| supcon+simclr (34i)   | 0.9503 | 0.8730 | 0.5927 | 0.5273 | --              | --              | --              |
| supcon+hybrid[lam5]   | 0.9478 | 0.8499 | 0.4573 | 0.5834 | 0.02/0.04 m0.36 | 0.08/0.14 m0.22 | 0.66/0.26 m0.18 |
| supcon+res-simclr     | 0.9475 | 0.8567 | 0.5220 | 0.5618 | --              | --              | --              |
| ss[lam1]+hybrid       | 0.9097 | 0.8486 | 0.4772 | 0.5156 | --              | --              | --              |
| hybrid->supres        | 0.8898 | 0.8619 | 0.6461 | 0.5827 | 0.06/0.68       | 0.42/0.50       | 0.32/0.56       |
| feat->rescls          | 0.8743 | 0.8016 | 0.3665 | 0.3172 | 0.02/0.28       | 0.00/0.26       | 0.30/0.10       |
| cls->resfeat          | 0.8633 | 0.8823 | 0.6318 | 0.4834 | 0.06/0.34       | 0.10/0.74       | 0.24/0.44       |
| ss[lam5]+hybrid       | 0.8584 | 0.8958 | 0.5566 | 0.4892 | 0.02/0.44 m0.50 | 0.06/0.54 m0.34 | 0.26/0.12       |

CIFAR-10 champion stack: sup->res-hybrid + conf-masked discovery +
Mahalanobis (Maha pre 0.94, masked post 0.98 at f=0.02) -- Pareto-dominates
the settled classic recipe on every metric.

## CIFAR-100

| space                 | probe  | acc    | eucl   | mahaT  | SparKer @0.02   | Maha @0.02      | MMD @0.02       |
|-----------------------|--------|--------|--------|--------|-----------------|-----------------|-----------------|
| supcon+hybrid[lam5]   | 0.9423 | 0.5271 | 0.4605 | 0.4073 | 0.06/0.30 m0.28 | 0.00/0.00       | 0.30/0.88 m0.84 |
| supcon+hybrid[lam1]   | 0.9409 | 0.5230 | --     | 0.4222 | --              | --              | --              |
| supcon+simclr (r1)    | 0.9394 | 0.5030 | --     | 0.3329 | 0.06/0.22*      | 0.04/0.08*      | 0.14/0.34*      |
| supcon+res-simclr     | 0.9281 | 0.5386 | 0.3834 | 0.2541 | --              | --              | --              |
| hybrid->supres        | 0.9263 | 0.4752 | 0.5162 | 0.4820 | 0.06/0.12       | 0.04/0.00       | 0.20/0.42       |
| ss[lam5]+hybrid       | 0.9235 | 0.5533 | 0.5251 | 0.4867 | 0.10/0.52 m0.42 | 0.00/0.04       | 0.38/0.86 m0.82 |
| supcon (exp33)        | 0.9232 | 0.5629 | 0.4579 | 0.4500 | 0.04/0.16       | 0.04/0.14       | 0.14/0.26       |
| ssl->supres (exp33)   | 0.9178 | 0.4204 | 0.3558 | 0.3303 | 0.02/0.08       | 0.02/0.04       | 0.14/0.38       |
| sup (exp33)           | 0.9039 | 0.4911 | 0.5745 | 0.3416 | 0.12/0.12       | 0.04/0.16       | 0.08/0.26       |
| joint (exp33)         | 0.8985 | 0.4895 | 0.6184 | 0.3006 | 0.04/0.12       | 0.04/0.12       | 0.08/0.18       |
| cls->resfeat          | 0.8745 | 0.5490 | 0.5753 | 0.5599 | 0.50/0.44       | 0.00/0.12       | 0.52/0.74       |
| sup->res (exp33)      | 0.8361 | 0.5236 | 0.5201 | 0.3915 | 0.08/0.08       | 0.00/0.14       | 0.14/0.22       |
| feat->rescls          | 0.8196 | 0.3703 | 0.3186 | 0.2465 | 0.00/0.06       | 0.00/0.00       | 0.32/0.72       |
| sup->res-hybrid (36)  | (running) | -- | --     | --     | --              | --              | --              |

*exp33 supcon+simclr row's powers from the exp33 npz.
50+50 (100-D total, exp34j) probe: supcon+simclr 0.9547, supcon+hybrid
lam1 0.9541, res-simclr 0.9519, lam5 0.9474, ss lam1 0.9306, hybrid->supres
0.9234, cls->resfeat 0.9228, ss lam5 0.9076; kernel power @0.02 drops
(SparKer post ss+hybrid 0.10, cls->resfeat pre 0.14), MMD robust (~0.8).

CIFAR-100 menu: probe-best supcon+hybrid[lam5] (0.9423); post-detection
ss[lam5]+hybrid (SparKer 0.52, MMD 0.86); pre-detection cls->resfeat
(SparKer 0.50 / MMD 0.52 discovery-free); Maha dead at every dim/space.

## Program-level verdicts

- Statistic-geometry-dataset matching: separable few-class (C10) ->
  supervised SIGReg + residual + parametric stats; crowded many-class
  (C100) -> contrastive + SIGReg calibration + kernel stats.
- Hybrid (NT-Xent + SIGReg) beats either ingredient alone: as feature half
  on C100 (probe), as residual objective on C10 (everything).
- Conf-masked discovery (JEPAMatch M_i): free win above the pool-purity
  floor, mainly through Mahalanobis; useless below it. Asymmetric variance
  annealing: high-variance, off by default.
- Dimension: high-D helps linear readouts, hurts kernel tests; MMD is the
  dimension-robust statistic; C100 Maha-deadness is class similarity, not
  dimension.
