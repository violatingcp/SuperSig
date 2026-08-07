# Transfer-suite master tables — aircraft (3 bases) + cars (exp 51/60/69/70)

Protocol: frozen trunk (features cached, a8 aug bank), 768->256->32
FeatureHead per arm, 120 epochs, holdout variants 90-99 (~330 signal test
events), n_d=1000 toys, fixed sigma=1 SparKer (all spaces here share the
compact head scale, so within-table comparison is fair).  Power columns at
f=0.1 (the only fraction with meaningful power at this pool size).
Sources: logs/exp51/results_nplm_aircraft_{dino,lejepa,visreg}.npz.

## DINO base

| arm | probe | acc | eucl | mahaT | perevt | SpK@.1 | Maha@.1 | MMD@.1 |
|---|---|---|---|---|---|---|---|---|
| supcon_sigreg | **0.781** | **0.596** | **0.764** | **0.743** | **0.222** | 0.38 | **0.64** | 0.76 |
| lejepa | 0.732 | 0.119 | 0.456 | 0.533 | 0.033 | 0.16 | 0.22 | 0.10 |
| nplm_sup_dist | 0.721 | 0.220 | 0.708 | 0.695 | 0.147 | **0.64** | 0.44 | **0.86** |
| simclr_sigreg | 0.713 | 0.161 | 0.418 | 0.485 | 0.048 | 0.18 | 0.26 | 0.30 |
| nplm_distance | 0.695 | 0.077 | 0.457 | 0.543 | 0.057 | 0.14 | 0.04 | 0.08 |
| supcon | 0.688 | 0.557 | 0.730 | 0.708 | 0.192 | 0.42 | 0.44 | 0.02 |
| simclr | 0.659 | 0.148 | 0.569 | 0.539 | 0.078 | 0.06 | 0.12 | 0.00 |
| nplm_bilinear | 0.612 | 0.083 | 0.548 | 0.595 | 0.039 | 0.16 | 0.08 | 0.00 |

## LeJEPA base

| arm | probe | acc | eucl | mahaT | perevt | SpK@.1 | Maha@.1 | MMD@.1 |
|---|---|---|---|---|---|---|---|---|
| supcon_sigreg | **0.724** | 0.404 | 0.673 | 0.627 | **0.201** | 0.52 | 0.12 | 0.24 |
| nplm_sup_dist | 0.722 | 0.178 | 0.655 | 0.651 | 0.120 | 0.28 | **0.50** | **0.72** |
| supcon | 0.706 | **0.483** | **0.704** | **0.688** | 0.171 | **0.64** | 0.16 | 0.10 |
| simclr_sigreg | 0.681 | 0.093 | 0.521 | 0.573 | 0.048 | 0.12 | 0.14 | 0.06 |
| nplm_bilinear | 0.681 | 0.066 | 0.571 | 0.611 | 0.084 | 0.18 | 0.18 | 0.00 |
| lejepa | 0.673 | 0.074 | 0.532 | 0.594 | 0.072 | 0.04 | 0.10 | 0.04 |
| nplm_distance | 0.659 | 0.053 | 0.498 | 0.561 | 0.063 | 0.12 | 0.04 | 0.00 |
| simclr | 0.642 | 0.096 | 0.610 | 0.589 | 0.084 | 0.44 | 0.04 | 0.00 |

## VISReg base

| arm | probe | acc | eucl | mahaT | perevt | SpK@.1 | Maha@.1 | MMD@.1 |
|---|---|---|---|---|---|---|---|---|
| supcon | **0.764** | **0.492** | **0.734** | **0.704** | **0.177** | **0.90** | 0.10 | 0.50 |
| supcon_sigreg | 0.737 | 0.428 | 0.703 | 0.639 | 0.138 | 0.36 | 0.22 | 0.78 |
| nplm_sup_dist | 0.731 | 0.186 | 0.631 | 0.617 | 0.087 | 0.68 | **0.40** | **0.78** |
| lejepa | 0.698 | 0.071 | 0.522 | 0.603 | 0.048 | 0.06 | 0.12 | 0.02 |
| simclr_sigreg | 0.694 | 0.082 | 0.505 | 0.565 | 0.045 | 0.10 | 0.08 | 0.08 |
| simclr | 0.668 | 0.088 | 0.595 | 0.585 | 0.099 | 0.22 | 0.06 | 0.04 |
| nplm_bilinear | 0.661 | 0.058 | 0.489 | 0.597 | 0.024 | 0.08 | 0.20 | 0.00 |
| nplm_distance | 0.648 | 0.056 | 0.504 | 0.573 | 0.072 | 0.10 | 0.04 | 0.00 |

## Stanford Cars, DINO base (196 classes, holdouts 186-195, n_d=2000)

| arm | probe | acc | eucl | mahaT | perevt | MMD@.1 |
|---|---|---|---|---|---|---|
| supcon_sigreg | **0.747** | **0.653** | **0.627** | **0.603** | 0.075 | **0.96** |
| supcon | 0.733 | 0.609 | 0.544 | 0.544 | 0.028 | 0.68 |
| simclr | 0.612 | 0.114 | 0.462 | 0.473 | 0.023 | 0.18 |
| lejepa | 0.596 | 0.096 | 0.477 | 0.490 | 0.038 | 0.12 |
| simclr_sigreg | 0.591 | 0.125 | 0.437 | 0.449 | 0.021 | 0.74 |
| nplm_bilinear | 0.570 | 0.058 | 0.476 | 0.488 | 0.033 | 0.02 |
| nplm_sup_dist | 0.541 | 0.110 | 0.531 | 0.493 | **0.106** | 0.66 |
| nplm_distance | 0.525 | 0.059 | 0.492 | 0.480 | 0.021 | 0.12 |

## Stanford Cars at 100-D (DINO; 32-D values in parens)

| arm | probe | acc | eucl | mahaT | perevt |
|---|---|---|---|---|---|
| supcon | 0.754 (0.733) | 0.584 | 0.477 (0.544) | 0.482 | 0.01 |
| supcon_sigreg | 0.745 (0.747) | 0.659 | 0.622 (0.627) | 0.565 | 0.08 |
| nplm_sup_dist | 0.637 (0.541) | 0.096 | 0.485 | 0.478 | 0.03 (0.106) |
| label-free arms | 0.49-0.62 | <=0.13 | ~0.47 | ~0.46 | <=0.04 |

100-D on cars: supcon takes the probe lead (+0.021, high-D readout
effect); supcon_sigreg flat but keeps acc/geometry/per-event/battery
crowns -- champion unchanged and dimension-indifferent.  nplm_sup_dist
gains the most probe (+0.096) but loses its per-event calibration lead
(0.106 -> 0.03): at 196 classes the extra dims go to probe-readable
directions, not calibrated ones.

## Cars discovery (exp 69; first transfer-suite discovery, feature-space
loop, 32-D supervised heads, annealed sigma; probe pre -> post)

| arm | probe pre/post | pool purity r1 | perevt post @.02 | Maha post @.05 |
|---|---|---|---|---|
| supcon | 0.7364 -> 0.7142 | 0.033 | 0.092 | 0.56 |
| supcon_sigreg | 0.7273 -> 0.6945 | 0.138 | 0.120 | 0.42 |
| nplm_sup_dist | 0.5868 -> 0.4936 | 0.119 | 0.138 | 0.12 |

Discovery is probe-NEGATIVE for every arm on cars despite favorable
holdout counts (~410 vs ~400-event tail): pool purity caps at 0.14
because novel fine-grained variants do not sit in the distance tail --
they sit among the seen classes (a novel car model looks like existing
car models).  BIC fragments the impure pool into 11-22 anchors and the
ft erodes the space (nplm_sup_dist worst, -0.09).  Dataset-level stats
improve modestly post (per-event 0.12-0.16, Maha 0.42-0.56 @0.05).
Extends the discovery lesson: pooling needs novelty that is GEOMETRICALLY
OUTLYING, which fine-grained novelty is not, calibrated space or no.
(BUT see exp 70 below: on end-to-end fine-tuned trunks discovery flips
probe-POSITIVE for all six arms.)

## Cars END-TO-END fine-tuning suite (exp 70; DINO ViT-B/16 trunk
trainable, exp-49 recipe, 20 ep, 100-D heads.  Holdouts 186-195 EXCLUDED
from the ft corpus — images and labels — so unlike the aircraft ft-trunk
rerun these novelty numbers are strictly open-world.  Discovery =
feature-space loop (exp-69 protocol) on each arm's own ft-trunk bank.)

Pre-discovery battery (powers at alpha=0.05, n_d=2000):

| arm | probe | acc | supAUC | eucl | mahaT | perevt | SpK@.1 | Maha@.1 | MMD@.1 |
|---|---|---|---|---|---|---|---|---|---|
| supcon-ft | **0.733** | **0.462** | **0.974** | **0.603** | **0.564** | **0.136** | 0.98 | **0.94** | **0.84** |
| ss-ft | 0.719 | 0.330 | 0.966 | 0.551 | 0.547 | 0.103 | **1.00** | 0.68 | 0.82 |
| simclr-ft | 0.624 | 0.108 | 0.776 | 0.469 | 0.476 | 0.023 | 0.30 | 0.18 | 0.20 |
| nplm-bil-ft | 0.591 | 0.063 | 0.760 | 0.506 | 0.496 | 0.026 | 0.12 | 0.04 | 0.02 |
| sigreg-ssl-ft | 0.584 | 0.083 | 0.803 | 0.483 | 0.496 | 0.033 | 0.12 | 0.16 | 0.08 |
| nplm-sup-ft | 0.549 | 0.105 | 0.916 | 0.555 | 0.547 | 0.061 | 0.94 | 0.18 | 0.28 |

Discovery, natural + injected post grid (probe 3-seed; post powers @f):

| arm | probe pre -> post | delta | purity r1 | mahaT pre -> post | perevt post@.05 | Maha post@.05 |
|---|---|---|---|---|---|---|
| supcon-ft | 0.7332 -> **0.7666** | +0.033 | 0.140 | 0.564 -> 0.546 | **0.202** | **0.68** |
| ss-ft | 0.7191 -> 0.7474 | +0.028 | 0.068 | 0.547 -> 0.541 | 0.195 | 0.44 |
| nplm-bil-ft | 0.5912 -> 0.7044 | +0.113 | 0.054 | 0.496 -> 0.507 | 0.108 | 0.24 |
| sigreg-ssl-ft | 0.5843 -> 0.6903 | +0.106 | 0.045 | 0.496 -> 0.517 | 0.042 | 0.20 |
| simclr-ft | 0.6238 -> 0.6839 | +0.060 | 0.035 | 0.476 -> 0.482 | 0.063 | 0.10 |
| nplm-sup-ft | 0.5490 -> 0.6001 | +0.051 | 0.068 | 0.547 -> 0.500 | 0.171 | 0.06 |

Verdicts.  (1) Discovery is probe-POSITIVE on ALL SIX end-to-end ft
spaces (+0.03 to +0.11) — the exact reversal of exp 69, where every
frozen-trunk arm lost probe.  Pool purity is STILL low (<= 0.14), so the
gain is not from pure pools: on a cars-specialized trunk the impure-pool
proto/repulse ft acts as beneficial refinement (the C100 classwise-lam5
mechanism), where on generic frozen features it eroded the space.
Fine-grained novelty remains non-outlying; what changed is that the
substrate can absorb the ft.  (Confound note: exp 69 heads were 32-D,
these are 100-D.)  (2) supcon-ft + discovery probe 0.767 is the NEW CARS
CHAMPION, beating every frozen-trunk space (best 0.754) under a stricter
protocol (holdouts excluded from ft).  (3) Pre-discovery, e2e ft roughly
matches the frozen 32-D champion on probe (0.733 vs 0.747) with lower
nearest-centroid acc (0.462 vs 0.653) but stronger dataset-level power
(SpK/Maha@.1 0.98/0.94 vs the frozen table's MMD-led 0.96) and the best
mahaT on cars to date (0.56).  (4) Supervised >> unsupervised pretraining
everywhere, and within families supcon > sigreg-supcon > nplm on cars;
nplm-sup-ft is again probe-worst / calibration-strong (SpK@.1 0.94,
per-event post 0.17) — the dissociation, unchanged.  (5) Post per-event
power of the supervised arms (0.17-0.20 @.05) is the best cars per-event
on record.

## Cars ft suite across bases (exp 70; probe pre -> post per base)

| arm | DINO | LeJEPA | VISReg |
|---|---|---|---|
| supcon-ft | 0.733 -> **0.767** | 0.699 -> 0.726 | 0.707 -> 0.759 |
| ss-ft | 0.719 -> 0.747 | **0.709** -> 0.699 | **0.741** -> 0.712 |
| simclr-ft | 0.624 -> 0.684 | 0.618 -> 0.614 | 0.610 -> 0.657 |
| nplm-bil-ft | 0.591 -> 0.704 | 0.611 -> 0.658 | 0.576 -> 0.624 |
| sigreg-ssl-ft | 0.584 -> 0.690 | 0.563 -> 0.617 | 0.574 -> 0.634 |
| nplm-sup-ft | 0.549 -> 0.600 | 0.582 -> 0.658 | 0.579 -> 0.549 |

End-to-end ft largely erases the frozen-base gap on cars (frozen probes:
DINO 0.75 vs LeJEPA/VISReg ~0.5-0.6 territory): VISReg ss-ft 0.741 is the
best cars PRE-probe on any base, VISReg supcon-ft has the best cars
geometry (eucl 0.715, mahaT 0.661, purity 0.206), echoing the exp-43/49
"fine-tuning inverts the base ranking" lesson.  Discovery stays broadly
positive (14/18 cells; the exceptions are ss-ft on LeJEPA/VISReg and
nplm-sup-ft/simclr-ft on one base each).  Best overall cars pipeline
remains DINO supcon-ft + discovery (0.767), with VISReg supcon-ft +
discovery (0.759) statistically adjacent.

## Flowers END-TO-END ft suite (exp 70, DINO; 102 classes, holdouts
92-101 excluded from ft, ~18 train imgs/class seen, n_d=1000)

| arm | probe pre -> post | acc | eucl | mahaT | perevt pre/post@.05 | purity r1 | Maha@.1 pre |
|---|---|---|---|---|---|---|---|
| ss-ft | **0.795** -> 0.788 | 0.940 | **0.933** | **0.883** | **0.524** / **0.710** | **0.657** | **1.00** |
| supcon-ft | 0.708 -> **0.814** | **0.944** | 0.911 | 0.873 | 0.369 / 0.413 | 0.624 | **1.00** |
| simclr-ft | 0.743 -> 0.754 | 0.839 | 0.671 | 0.673 | 0.071 / 0.052 | 0.098 | 0.82 |
| sigreg-ssl-ft | 0.732 -> 0.777 | 0.800 | 0.710 | 0.706 | 0.103 / 0.169 | 0.140 | 0.94 |
| nplm-bil-ft | 0.664 -> 0.808 | 0.531 | 0.667 | 0.646 | 0.083 / 0.179 | 0.164 | 0.58 |
| nplm-sup-ft | 0.625 -> 0.624 | 0.518 | 0.659 | 0.668 | 0.147 / 0.074 | 0.328 | 0.94 |

Flowers is the ANTI-CARS: coarse-separable holdouts make pool purity high
(supervised arms 0.33-0.66 in round 1 -- transfer-suite record) and
per-event power real pre-discovery (ss-ft 0.524).  Discovery gains:
supcon-ft +0.106 (post 0.814, flowers champion), nplm-bil-ft +0.145 (the
biggest single discovery gain in the transfer program), sigreg-ssl +0.05;
ss-ft is the one flat arm (its pre space is already the best-calibrated:
per-event post 0.71, Maha 0.88 @.05).  The supervised sigreg marginal
(ss-ft) dominates the flowers pre battery outright -- probe, eucl, mahaT,
per-event, purity -- the strongest single-space showing of ss in the
program.  nplm-sup-ft is flat/weak everywhere here: with only ~18
imgs/class, supervised distance-NPLM underfits.

## Flowers ft suite across bases (exp 70; probe pre -> post per base)

| arm | DINO | LeJEPA | VISReg |
|---|---|---|---|
| supcon-ft | 0.708 -> **0.814** | 0.678 -> 0.752 | 0.714 -> **0.810** |
| ss-ft | **0.795** -> 0.788 | **0.792** -> 0.613 (!) | **0.754** -> 0.743 |
| simclr-ft | 0.743 -> 0.754 | 0.725 -> 0.740 | 0.746 -> 0.728 |
| sigreg-ssl-ft | 0.732 -> 0.777 | 0.684 -> 0.719 | 0.738 -> 0.734 |
| nplm-bil-ft | 0.664 -> 0.808 | 0.768 -> 0.723 | 0.737 -> 0.737 |
| nplm-sup-ft | 0.625 -> 0.624 | 0.734 -> 0.706 | 0.631 -> 0.576 |

Base-invariant flowers structure: ss-ft always wins the pre-probe and the
supervised arms always pool at high purity (0.43-0.68 r1 on every base);
supcon-ft + discovery always wins post (0.752-0.814).  One pathology:
ss-ft discovery on LeJEPA collapses the probe (0.792 -> 0.613, worst
discovery outcome on record; its 6-anchor round-1 split at purity 0.605
fragments the strongest classes rather than the novelty) -- on DINO/VISReg
the same arm is merely flat, so treat ss-ft + discovery as unstable and
prefer supcon-ft when the discovery step is in the pipeline.  supcon-ft
geometry on VISReg is the best calibrated flowers space (eucl 0.949,
mahaT 0.919, acc 0.957).

## DTD END-TO-END ft suite (exp 70, DINO; 47 classes, holdouts 37-46
excluded from ft, 80 train imgs/class, n_d=1000)

| arm | probe pre -> post | acc | eucl | mahaT | perevt pre/post@.05 | purity r1 | Maha@.1 pre |
|---|---|---|---|---|---|---|---|
| simclr-ft | **0.808** -> 0.802 | 0.653 | 0.469 | 0.506 | 0.020 / 0.035 | 0.039 | 0.98 |
| sigreg-ssl-ft | 0.803 -> 0.802 | 0.635 | 0.543 | 0.577 | 0.048 / 0.110 | 0.026 | **1.00** |
| supcon-ft | 0.799 -> **0.811** | 0.759 | 0.704 | 0.646 | 0.040 / 0.165 | 0.129 | 0.76 |
| ss-ft | 0.782 -> 0.785 | **0.782** | **0.801** | **0.742** | **0.207** / **0.255** | **0.795** | 0.98 |
| nplm-bil-ft | 0.682 -> 0.786 | 0.366 | 0.465 | 0.479 | 0.020 / 0.068 | 0.169 | 0.14 |
| nplm-sup-ft | 0.586 -> 0.781 | 0.432 | 0.590 | 0.615 | 0.077 / 0.145 | 0.444 | 0.74 |

DTD sits between flowers and cars: textures make even unsupervised probes
strong (simclr-ft 0.808 pre-champion -- instance discrimination suits
texture novelty), while calibration currency again belongs to ss-ft
(purity 0.795 r1 -- program record; eucl 0.801, mahaT 0.742, per-event
0.21 -> 0.26).  Discovery is a RESCUE operation here: near-flat for the
already-strong arms but +0.10 for nplm-bil-ft and +0.19 for nplm-sup-ft
(0.586 -> 0.781, the largest discovery rescue on record) -- the impure-
pool ft recovers what the weak NPLM heads missed.  Post champion
supcon-ft 0.811.  Maha@.1 is near-saturated for all non-NPLM arms.

## DTD ft suite across bases (exp 70; probe pre -> post per base)

| arm | DINO | LeJEPA | VISReg |
|---|---|---|---|
| simclr-ft | **0.808** -> 0.802 | **0.817** -> 0.821 | **0.854** -> 0.848 |
| sigreg-ssl-ft | 0.803 -> 0.802 | 0.809 -> 0.826 | 0.808 -> 0.790 |
| supcon-ft | 0.799 -> **0.811** | 0.755 -> 0.790 | 0.750 -> 0.791 |
| ss-ft | 0.782 -> 0.785 | 0.787 -> 0.784 | 0.784 -> 0.785 |
| nplm-bil-ft | 0.682 -> 0.786 | 0.746 -> 0.745 | 0.778 -> 0.811 |
| nplm-sup-ft | 0.586 -> 0.781 | 0.581 -> 0.765 | 0.569 -> 0.707 |

DTD is base-invariantly the UNSUPERVISED dataset: simclr-ft wins the
pre-probe on all three bases, peaking at 0.854 on VISReg -- the best DTD
novelty probe on record (texture novelty is instance-discrimination
territory; labels add nothing to the probe here).  Supervised arms keep
the calibration crown: ss-ft round-1 purity 0.795/0.803/0.811
(DINO/LeJEPA/VISReg -- the three highest purities in the program) with
the best eucl/mahaT/acc everywhere.  The nplm-sup-ft discovery rescue
replicates on every base (+0.14 to +0.20).  Post champions: simclr-ft
(no-discovery) 0.848-0.854 for the probe; ss-ft for everything
calibrated.

## Galaxy10 END-TO-END ft suite (exp 70, DINO; 10 classes, holdout {9}
(1 class) excluded from ft, 10%/90% train/test split, n_d=2000)

| arm | probe pre -> post | acc | eucl | mahaT | perevt pre/post best | purity r1 | MMD@.05 pre |
|---|---|---|---|---|---|---|---|
| supcon-ft | **0.938** -> 0.939 | **0.610** | 0.666 | 0.786 | 0.021 / 0.230 | 0.070 | 0.94 |
| sigreg-ssl-ft | 0.914 -> 0.931 | 0.408 | 0.514 | 0.764 | 0.020 / 0.075 | 0.012 | 0.14 |
| nplm-sup-ft | 0.878 -> 0.912 | 0.414 | **0.779** | **0.824** | **0.348** / **0.502** | **0.459** | 0.88 |
| ss-ft | 0.872 -> 0.926 | 0.558 | 0.633 | 0.691 | 0.058 / 0.186 | 0.048 | **0.98** |
| simclr-ft | 0.868 -> **0.942** | 0.421 | 0.595 | 0.730 | 0.019 / 0.026 | 0.024 | 0.20 |
| nplm-bil-ft | 0.839 -> 0.928 | 0.319 | 0.664 | 0.712 | 0.072 / 0.019 | 0.140 | 0.40 |

Galaxy10 is the CIFAR-10 of the transfer suite (9 seen classes, coarse
morphological novelty) and the class-count rule holds exactly:
nplm-sup-ft takes the calibration crown back -- best eucl/mahaT, purity
0.459, per-event 0.35 pre / 0.50 post (vs supcon-ft 0.02 pre!) -- while
supcon-ft keeps the probe/acc crown.  Discovery is probe-positive for ALL
SIX arms (+0.00 to +0.09); post champion is simclr-ft 0.942 (+0.074).
MMD@.05 is near-saturated for the supervised arms (0.88-0.98 pre, 0.96-
1.00 post).  With 6/6 positive here, 6/6 on cars-ft, and 4/6-5/6 on
flowers/dtd, discovery on end-to-end ft spaces is now positive in 21/24
dataset x arm cells -- the frozen-trunk exp-69 negative was about the
substrate, not the datasets.

## Galaxy10 ft suite across bases (exp 70; probe pre -> post per base)

| arm | DINO | LeJEPA | VISReg |
|---|---|---|---|
| supcon-ft | **0.938** -> 0.939 | 0.919 -> 0.931 | **0.939** -> 0.944 |
| sigreg-ssl-ft | 0.914 -> 0.931 | **0.927** -> 0.940 | 0.931 -> 0.940 |
| simclr-ft | 0.868 -> **0.942** | 0.902 -> **0.946** | 0.933 -> 0.922 |
| nplm-sup-ft | 0.878 -> 0.912 | 0.899 -> 0.897 | 0.883 -> 0.933 |
| ss-ft | 0.872 -> 0.926 | 0.803 -> 0.871 | 0.865 -> **0.945** |
| nplm-bil-ft | 0.839 -> 0.928 | 0.893 -> 0.905 | 0.802 -> 0.904 |

Galaxy10 is base-robust and discovery-friendly (16/18 cells positive):
probes live in a tight 0.87-0.95 band and every base ends 0.93-0.95 post.
nplm-sup-ft holds the calibration crown on ALL THREE bases (eucl
0.77-0.79, mahaT 0.82-0.86, purity 0.46-0.50 -- remarkably stable), the
few-class regime where supervised distance-NPLM is the right corner.

## EXP-70 GRID VERDICTS (4 datasets x 3 bases x 6 arms, pre+post)

- **supcon-ft + discovery is probe-positive in 12/12 dataset x base
  cells** -- the only universally safe full pipeline in the program.
  Overall discovery is positive in 50/72 cells; catastrophic erosion is
  rare (worst: ss-ft flowers/LeJEPA -0.18).
- Pre-probe champion is REGIME-determined, base-invariant: fine-grained
  many-class (cars, flowers) -> ss-ft; texture (dtd) -> simclr-ft;
  few-class coarse (galaxy10) -> supcon-ft.  Calibration champion:
  ss-ft/supcon-ft on many-class data, nplm-sup-ft at few classes
  (galaxy10, 3/3 bases) -- the CIFAR class-count rule, verbatim.
- End-to-end ft largely erases the frozen-base ranking (VISReg holds
  several pre records: dtd simclr 0.854, cars ss 0.741) -- exp-43/49's
  inversion, now in the open-world battery.
- nplm-sup-ft's discovery rescue (+0.14 to +0.20) replicates on all
  three DTD bases; its failure mode is few-shot many-class data
  (flowers, ~18 imgs/class).
- Discovery helps most where the pre space is weakest (unsupervised arms
  +0.05..+0.15) and where purity is high; flowers/VISReg shows purity
  0.65+ can still leave strong arms flat (nothing left to learn).

## Aircraft NPLM residual/concat (exp 60, DINO, 16+16)

| arm | probe | acc | eucl | mahaT | perevt | MMD@.1 |
|---|---|---|---|---|---|---|
| supsig->res-nplm | 0.786 | 0.569 | 0.743 | 0.725 | 0.192 | 0.50 |
| supsig+nplm | 0.785 | 0.565 | 0.734 | 0.725 | 0.192 | 0.76 |
| supcon+nplm | 0.772 | 0.500 | 0.691 | 0.701 | 0.201 | 0.02 |
| nplmsup+nplm | 0.721 | 0.229 | 0.602 | 0.661 | 0.087 | 0.84 |

## End-to-end NPLM fine-tuning (exp 62; exp-49 recipe, trunk trainable,
closed-set test top-1 %, head 100-D / trunk 768-D probes)

| ft objective | DINO head/trunk | LeJEPA head/trunk | VISReg head/trunk |
|---|---|---|---|
| nplm-bil-ft (label-free) | 10.0 / 64.3 | 11.1 / 30.8 | 2.5 / 30.8 |
| nplm-dist-ft (label-free) | 6.6 / 65.3 | 5.6 / 32.9 | 5.6 / 31.2 |
| nplm-sup-ft (labeled) | 10.6 / **65.3** | 16.3 / **61.4** | 11.9 / **63.2** |
| refs: frozen / CE-ft | 64.9 / 65.5 | 34.3 / 75.8 | 37.9 / 76.6 |
| refs: SupCon-ft / ss-ft | 50.2 / 43.3 | 58.7 / 49.0 | 62.7 / 54.5 |

## Suite on the nplm-sup-ft trunks (exp 51 rerun; frozen-base values in
parens.  CAVEAT: the exp-62 ft saw images AND labels of all 100 variants
including holdouts, so the novelty columns here are optimistic vs the
strict open-world protocol; the acc gains are clean closed-set numbers.)

| arm | DINO-ft probe/acc/mahaT | LeJEPA-ft | VISReg-ft |
|---|---|---|---|
| supcon_sigreg | 0.758/0.591/0.751 (0.781/0.596/0.743) | **0.812/0.638/0.800** (0.724/0.404/0.627) | 0.786/0.656/0.768 (0.737/0.428/0.639) |
| supcon | 0.749/0.562/0.724 (0.688/0.557/0.708) | 0.699/0.630/0.709 (0.706/0.483/0.688) | 0.802/**0.665**/0.757 (0.764/0.492/0.704) |
| nplm_sup_dist | 0.721/0.227/0.688 (=) | 0.734/0.338/0.787 (0.722/0.178/0.651) | 0.742/0.305/**0.812** (0.731/0.186/0.617) |
| nplm_bilinear | 0.684/0.093/0.606 (0.612/...) | 0.753/0.231/0.656 (0.681/...) | 0.754/0.307/0.655 (0.661/...) |
| label-free others | ~= frozen | +0.05-0.08 probe | +0.06-0.07 probe |

## Supervised-CE head baseline (exp 63; 32-D head + linear classifier,
seen classes; cls-top1 = the classifier's own test accuracy)

| substrate | probe | acc(NC) | eucl | mahaT | perevt | cls-top1 |
|---|---|---|---|---|---|---|
| frozen DINO | 0.721 | 0.577 | 0.718 | 0.725 | 0.123 | 0.583 |
| DINO nplm-sup-ft | 0.735 | 0.589 | 0.726 | 0.728 | 0.156 | 0.573 |
| LeJEPA nplm-sup-ft | 0.772 | 0.543 | 0.771 | 0.785 | 0.204 | 0.625 |
| VISReg nplm-sup-ft | 0.769 | 0.528 | 0.776 | 0.767 | 0.258 | 0.602 |

CE is NOT the ceiling: on every substrate the supcon_sigreg head beats the
CE head on every column -- on LeJEPA-ft even supcon_sigreg's
nearest-centroid acc (0.638) exceeds CE's own trained-classifier top-1
(0.625).  Contrastive+SIGReg heads are better than plain supervision even
at pure classification in this regime.

## Reading

1. **DINO is the strongest base** on every currency (best probes, accs,
   geometry), consistent with the exp-44 transfer table.
2. **The supcon_sigreg advantage is DINO-specific**: on the weaker
   LeJEPA/VISReg trunks plain SupCon retakes probe/acc/eucl (VISReg:
   0.764/0.492/0.734 vs 0.737/0.428/0.703).  The SIGReg marginal helps
   most when the underlying features are already strong.
3. **nplm_sup_dist is the most base-robust arm**: probe 0.72-0.73 on all
   three trunks (spread 0.009 vs supcon's 0.076), consistently
   second-best geometry, and the best kernel-test space on every base
   (MMD@.1 0.86/0.72/0.78; Maha best on 2 of 3) -- the calibrated space
   survives trunk changes that reorder everything else.
4. **Label-free arms are uniformly weak on aircraft on every base**
   (probe <= 0.70, acc <= 0.16, per-event <= 0.10): fine-grained
   discrimination needs labels at these data sizes regardless of trunk.
5. **Per-event power is supervised-only** on aircraft (supcon family
   0.14-0.22, nplm_sup_dist 0.09-0.15, everything else ~0.05).
6. **Cars amplifies the aircraft regime** (196 classes, ~41 imgs/class):
   supcon_sigreg sweeps every column (probe 0.747, MMD 0.96); label-free
   arms collapse (probe <= 0.61, acc <= 0.13); the one NPLM bright spot is
   per-event calibration -- nplm_sup_dist 0.106, the best in the cars
   table, beating supcon_sigreg's 0.075 despite a far worse probe.
7. **Discovery on cars needs a task-specialized trunk** (exps 69 vs 70):
   probe-negative on every frozen-trunk space, probe-positive on every
   end-to-end fine-tuned space (all six arms, +0.03 to +0.11), with pool
   purity low in both.  The discovery ft is refinement when the substrate
   is aligned to the task and erosion when it is not -- same lesson as
   CIFAR-100 classwise-lam5, now in the fine-grained transfer regime.
   Best full cars pipeline: supcon end-to-end ft + discovery (probe
   0.767, per-event 0.20 / Maha 0.68 @f=0.05).
7. **Residual/concat constructions match but do not beat the standalone
   champion on aircraft** (exp 60: supsig->res-nplm 0.786/0.725 vs
   standalone supcon_sigreg 0.781/0.743); the NPLM residual adds nothing
   over the plain nplm_bil half (separable-data effect, third
   confirmation).  The nplm_bil half still lifts plain supcon
   (0.688 -> 0.772).
8. **The nplm-sup-ft trunks are far better suite substrates than their
   frozen selves** (exp 51 rerun): every arm on the LeJEPA/VISReg ft
   trunks improves (probe +0.05-0.09, acc +0.12-0.25), with new aircraft
   records — LeJEPA-ft + supcon_sigreg 0.812/0.638/0.800 and VISReg-ft
   nplm_sup_dist mahaT 0.812, VISReg-ft supcon acc 0.665.  DINO-ft ~=
   frozen DINO (already at ceiling).  Two-stage recipe emerges: NPLM-sup
   trunk ft, then a supcon_sigreg head.  Mind the holdout-contamination
   caveat above for the novelty columns.
9. **End-to-end NPLM ft (exp 62)**: nplm-sup-ft is the best
   contrastive-family trunk ft on every base -- matches CE-ft on DINO
   (65.3 vs 65.5), beats SupCon-ft on LeJEPA (61.4 vs 58.7), ties it on
   VISReg (63.2 vs 62.7); CE-ft keeps the crown on the sketching trunks
   (75.8/76.6).  The label-free NPLM fts are conservative: they preserve
   the strong DINO trunk (64-65 vs frozen 64.9, where SupCon-ft/ss-ft
   DESTROY it: 50.2/43.3) but cannot build up the weak trunks (30.8-32.9
   vs frozen 34.3/37.9).  NPLM heads are never linear-probe spaces
   (2.5-16.3%), consistent with the whole program.
