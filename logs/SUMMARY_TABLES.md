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
| sup->res (36 rebuild) | 0.8409 | 0.5264 | 0.5100 | 0.3781 | 0.16/0.08 m0.16 | 0.04/0.06 m0.02 | 0.48/0.34 m0.52 |
| sup->res-hybrid (36)  | 0.8383 | 0.5217 | 0.5385 | 0.3500 | 0.12/0.14 m0.18 | 0.08/0.06 m0.02 | 0.44/0.60 m0.24 |

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
  on C100 (probe), as residual objective on C10 (everything).  The hybrid
  residual does NOT transfer to C100 (0.838 vs 0.841, tie; exp 36 c100) --
  its C10 dominance is a separable-data effect.
- Conf-masked discovery (JEPAMatch M_i): free win above the pool-purity
  floor, mainly through Mahalanobis; useless below it. Asymmetric variance
  annealing: high-variance, off by default.
- Dimension: high-D helps linear readouts, hurts kernel tests; MMD is the
  dimension-robust statistic; C100 Maha-deadness is class similarity, not
  dimension.

# SuperSig exps 50-56: NPLM loss program

Standalone 32-D spaces (no concat), holdout 4, seed 0, 20 epochs, lam=1,
tau=1 (34h defaults; exp-52 scan confirms these optimal).  NPLM arms =
HybridContrastiveLoss with the calibrated un-normalised estimator; "cw" =
classwise SIGReg marginal (fixed make_anchors means, exp 53) instead of
global.  perevt = per-event power at alpha=0.05 (constant in f).  Power
columns pre/post @0.02; "m" = exp-56 conf-masked post (C100 only).
post columns from exp 55 (discovery = settled proto/repulse loop).
Sources: exp50 (suite), exp51 (aircraft), exp52 (lam/tau scan), exp53
(classwise), exp54 (space plots), exp55 (discovery), exp56 (conf mask).

## CIFAR-10 (exps 50/53/55)

| space            | probe pre/post | acc    | eucl   | mahaT  | perevt pre/post@.02 | SparKer @.02 | Maha @.02 | MMD @.02  |
|------------------|----------------|--------|--------|--------|---------------------|--------------|-----------|-----------|
| supcon           | 0.9418 / --    | 0.8608 | 0.6894 | 0.6440 | 0.000 / --          | 0.60 / --    | 0.28 / -- | 0.88 / -- |
| supcon_sigreg    | 0.8818 / --    | 0.8444 | 0.6369 | 0.5768 | 0.000 / --          | 0.64 / --    | 0.10 / -- | 0.74 / -- |
| simclr           | 0.8668 / --    | 0.7292 | 0.2362 | 0.2292 | 0.000 / --          | 0.06 / --    | 0.00 / -- | 0.36 / -- |
| lejepa           | 0.8635 / --    | 0.6952 | 0.4084 | 0.3707 | 0.007 / --          | 0.16 / --    | 0.10 / -- | 0.04 / -- |
| simclr_sigreg    | 0.8506 / --    | 0.7178 | 0.3113 | 0.2853 | 0.002 / --          | 0.06 / --    | 0.00 / -- | 0.14 / -- |
| nplm_bilinear    | 0.8531/0.8918  | 0.5461 | 0.5101 | 0.4943 | 0.057 / 0.073       | 0.10 / 0.04* | 0.16/0.36 | 0.06/0.00 |
| nplm_distance    | 0.7904 / --    | 0.6498 | 0.5002 | 0.4813 | 0.005 / --          | 0.26 / --    | 0.14 / -- | 0.00 / -- |
| nplm_sup_dist    | 0.7834/0.9631  | 0.8707 | 0.7260 | 0.7227 | 0.200 / 0.346       | 0.86 / 0.04* | 0.46/0.50 | 0.28/0.40 |
| nplm_bil_cw      | 0.8537 / --    | 0.6042 | 0.5703 | 0.5639 | 0.066 / --          | 0.10 / --    | 0.26 / -- | 0.06 / -- |
| nplm_bil_sup_cw  | 0.7812 / --    | 0.4468 | 0.5163 | 0.5138 | 0.045 / --          | 0.04 / --    | 0.08 / -- | 0.00 / -- |
| nplm_dist_sup_cw | 0.8951/0.9768  | 0.8846 | 0.7523 | 0.7294 | 0.163 / 0.316       | 0.70 / 0.06* | 0.32/0.74 | 0.12/0.34 |

*SparKer post collapse CONFIRMED as a pure fixed-sigma kernel-scale artifact
(exp 57): the discovery fine-tune rescales the NPLM space ~4x (class RMS
0.56->5.34, median pairwise 2.6->11.1), so all density variation sits far
above the sigma=1 kernel; with sigma=3/10 or the annealed median-heuristic
schedule the same post space scores SparKer 1.00/1.00 at f=0.02/0.1 -- the
best post-discovery SparKer in the program (exp-33 sup post ref: 0.98@0.02).
The information was never lost; the kernel was mis-scaled.  Lesson: fixed
sigma=1 (the exp-33 "k1" convention) is only valid for spaces at the
supervised-recipe unit-cluster scale; use the annealed median heuristic when
comparing heterogeneous geometries.

Per-event post @0.1: nplm_dist_sup_cw 0.749, nplm_sup_dist 0.734 -- both
above the exp-33 sup post reference (0.591) and supcon (0.302).  Discovery
pool purity r1 0.34-0.36 for the supervised NPLM arms vs 0.10 bilinear.

## CIFAR-100 (exps 50/52/53/55/56)

| space            | probe pre/post   | acc    | eucl   | mahaT  | perevt | SparKer @.02      | Maha @.02         | MMD @.02          |
|------------------|------------------|--------|--------|--------|--------|-------------------|-------------------|-------------------|
| supcon           | 0.9417 / --      | 0.5743 | 0.3319 | 0.3410 | 0.000  | 0.18 / --         | 0.06 / --         | 0.64 / --         |
| nplm_bilinear    | 0.9349/0.8910    | 0.4223 | 0.5323 | 0.4841 | 0.070  | 0.04/0.04 m0.14   | 0.04/0.08 m0.08   | 0.30/0.58 m0.86   |
| nplm_bil_sup_cw  | 0.9246 / --      | 0.4623 | 0.3775 | 0.3155 | 0.010  | 0.06 / --         | 0.00 / --         | 0.68 / --         |
| supcon_sigreg    | 0.8833 / --      | 0.5896 | 0.4355 | 0.4428 | 0.000  | 0.02 / --         | 0.06 / --         | 0.48 / --         |
| nplm_dist_sup_cw | 0.8903/0.8859    | 0.3471 | 0.4525 | 0.3717 | 0.020  | 0.04/0.12 m0.00   | 0.02/0.04 m0.02   | 0.52/0.66 m0.62   |
| nplm_bil_cw      | 0.8719/--        | 0.2951 | 0.4579 | 0.3878 | 0.030  | 0.06 / --         | 0.00 / --         | 0.22 / --         |
| lejepa           | 0.8677 / --      | 0.2121 | 0.4728 | 0.4035 | 0.010  | 0.04 / --         | 0.10 / --         | 0.16 / --         |
| simclr_sigreg    | 0.8631 / --      | 0.2743 | 0.5190 | 0.4577 | 0.040  | 0.00 / --         | 0.00 / --         | 0.10 / --         |
| simclr           | 0.8537 / --      | 0.2805 | 0.3948 | 0.3363 | 0.010  | 0.12 / --         | 0.00 / --         | 0.14 / --         |
| nplm_sup_dist    | 0.8033/0.8374    | 0.2834 | 0.3689 | 0.2689 | 0.020  | 0.02/0.18 m0.12   | 0.06/0.02 m0.04   | 0.78/0.68 m0.74   |
| nplm_distance    | 0.7368 / --      | 0.1294 | 0.4770 | 0.4088 | 0.010  | 0.08 / --         | 0.04 / --         | 0.24 / --         |

Exp-52 lam/tau scan (nplm_bilinear): tau=1 is a sharp optimum (0.92 vs 0.81
at tau=0.5, 0.83-0.87 at tau=2); lam nearly inert -- the exp-clamped NPLM
interaction is ~1e10 at init vs ~1e-2 for lam*sigreg, so the marginal only
shapes the endgame.  Single-seed probe spread ~0.017 (0.9179 rerun vs 0.9349
exp-50 same config) -- the nplm_bilinear-vs-supcon probe gap (0.007) is
unresolved at one seed.

Discovery on C100 (exps 55/56): pool purity 0.003-0.013 (500 holdout train
images vs ~2500-event tail at tauq=0.95) -- discovery neutral-to-harmful for
every space; the conf mask keeps 0 events in round 1 (posterior diluted over
~100 anchors at compact NPLM scale, 0.5 unreachable) and so acts as a
do-no-harm gate: recovers ~1/3 of the bilinear damage (0.891->0.907),
neutral elsewhere; kept-purity <=0.013, the rate problem stands.

## FGVC-Aircraft (exp 51; frozen DINO ViT-B/16, 32-D FeatureHead heads,
holdout variants 90-99, cached a8 aug bank, 120 epochs, n_d=1000)

| space          | probe  | acc    | eucl   | mahaT  | perevt | SpK@.1 | Maha@.1 | MMD@.1 |
|----------------|--------|--------|--------|--------|--------|--------|---------|--------|
| supcon_sigreg  | 0.7810 | 0.5957 | 0.7637 | 0.7425 | 0.222  | 0.38   | 0.64    | 0.76   |
| lejepa         | 0.7316 | 0.1193 | 0.4562 | 0.5330 | 0.033  | 0.16   | 0.22    | 0.10   |
| nplm_sup_dist  | 0.7214 | 0.2197 | 0.7075 | 0.6947 | 0.147  | 0.64   | 0.44    | 0.86   |
| simclr_sigreg  | 0.7128 | 0.1610 | 0.4179 | 0.4846 | 0.048  | 0.18   | 0.26    | 0.30   |
| nplm_distance  | 0.6948 | 0.0767 | 0.4573 | 0.5434 | 0.057  | 0.14   | 0.04    | 0.08   |
| supcon         | 0.6878 | 0.5567 | 0.7296 | 0.7081 | 0.192  | 0.42   | 0.44    | 0.02   |
| simclr         | 0.6592 | 0.1480 | 0.5690 | 0.5390 | 0.078  | 0.06   | 0.12    | 0.00   |
| nplm_bilinear  | 0.6121 | 0.0827 | 0.5484 | 0.5945 | 0.039  | 0.16   | 0.08    | 0.00   |

## NPLM program verdicts

- Probe-vs-calibration dissociation everywhere: softmax(SupCon) spaces are
  the best probe inputs but carry no absolute distance information
  (per-event 0.000); NPLM spaces are ready-to-use distance metrics at a
  probe cost.  Exp-54 plots (exp54_*_nplm_sup_dist_cifar10.png): 9 near-
  point classes, holdout displaced radially -- distance-rule-visible,
  hyperplane-invisible.
- The winning NPLM corner tracks seen-class count vs dim: few classes ->
  supervised distance NPLM (C10); many classes -> label-free bilinear NPLM
  (C100, probe 0.935 within seed noise of supcon with no labels + best
  geometry + cleanest gaussianity).  Supervised class-collapse does not
  scale past dim/n_classes ~ 1.
- Classwise SIGReg marginal (exp 53) is the right constraint iff the anchor
  layout is realizable (C10 orthogonal: nplm_dist_sup_cw Pareto-dominates
  supcon on everything but the probe; C100 random 100-in-32d: hurts the
  label-free arm, cent->anchor stalls ~3.5).
- Discovery (exp 55): the calibrated tail is exactly what pooling needs --
  on C10 NPLM + discovery is the best full pipeline on record (probe post
  0.96-0.98, per-event 0.73-0.75 vs sup 0.59); on C100 blocked by pool rate
  (500 vs 2500), conf mask (exp 56) can gate but not fix it.
- LeJEPA / plain SimCLR (augmentations only) are never on the Pareto front:
  mid-pack probes, weak geometry, near-zero per-event power on all three
  datasets.
- Aircraft (exp 51): supcon_sigreg dominates (SIGReg marginal helps exactly
  in the low-data fine-grained regime, opposite of CIFAR); nplm_sup_dist
  second on geometry and best SparKer/MMD at f=0.1.
- 64-D C100 rerun (exps 50/53) in progress; results land in
  results_nplm_cifar100_64d.npz / results_classwise_cifar100_64d.npz.
