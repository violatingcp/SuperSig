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

Exp-67 from-scratch bases (C100, RANDOM init, 100-D, 200ep, holdout 4
excluded; checkpoints/scratch_{arm}_cifar100_100d.pt, resumable):
simclr probe 0.7914 / acc 0.2418 / eucl 0.4435 / mahaT 0.3853;
visreg 0.8031 / 0.1896 / 0.4993 / 0.3823;
nplm (bilinear, label-free) 0.7572 / 0.1814 / 0.4134 / 0.3610.
The ~0.06-0.13 probe drop vs hub-init counterparts quantifies the
hub-trunk contribution; these are the clean bases for caveat-free runs.

Supervised from-scratch bases (same protocol, 200ep):
supcon probe 0.9390 / acc 0.5451 / eucl 0.2334 / mahaT 0.2792 -- MATCHES
hub-init supcon (0.944): 200 supervised epochs fully erase the hub trunk.
supsig (recipe) 0.8144 / 0.3459 / 0.3402 / 0.3201 -- far below its
hub-init 0.9546: the supervised-SIGReg recipe leans heavily on the hub
trunk (or its 10-epoch schedule; chunked-Adam 200ep does not recover it).
nplmcw (classwise lam=5) 0.8987 / 0.4631 / **0.5252** / **0.4109** -- the
best all-round clean supervised base: probe within 0.04 of supcon with
2.3x its eucl; the classwise-calibrated geometry forms fully from scratch.

Full suite on the from-scratch bases (exp 50 --scratch-base, C100 100-D,
probe/acc; all caveat-free: no hub trunk, holdout never seen):

| arm            | scr-simclr    | scr-visreg    | scr-nplm      |
|----------------|---------------|---------------|---------------|
| supcon_sigreg  | **0.933**/0.457 | **0.926**/0.456 | 0.896/0.403 |
| supcon         | 0.925/0.462   | 0.910/0.463   | **0.925**/0.430 |
| nplm_sup_dist  | 0.851/0.119   | 0.854/0.136   | 0.818/0.159   |
| simclr_sigreg  | 0.827/0.242   | 0.814/0.212   | 0.750/0.187   |
| lejepa         | 0.817/0.202   | 0.791/0.189   | 0.682/0.137   |
| simclr         | 0.790/0.241   | 0.823/0.209   | 0.793/0.179   |
| nplm_distance  | 0.735/0.098   | 0.753/0.117   | 0.699/0.080   |
| nplm_bilinear  | 0.664/0.102   | 0.692/0.075   | 0.710/0.175   |

Suite on the scratch-nplmcw base (probe/acc; same protocol):
supcon_sigreg 0.9302/0.5248, supcon 0.9290/0.5326 (eucl 0.4618 -- best
supcon geometry on any base), simclr 0.8917/0.3360, lejepa 0.8539,
simclr_sigreg 0.8446, nplm_sup_dist 0.8390, nplm_bilinear 0.8344/0.2754
(mahaT 0.4667, best in table), nplm_distance 0.7541.  The calibrated
supervised base is the best SUBSTRATE of the four scratch bases: every
label-free head gains +0.05-0.17 probe over the label-free bases (simclr
0.79->0.89, nplm_bilinear 0.66-0.71->0.83), and supcon keeps far better
geometry on it.  Caveat: nplmcw pretraining used labels, so label-free
heads on it are no longer label-free pipelines.

Discovery on the supervised scratch bases (exp 68; probe pre/post,
MMD post @0.02/@0.05; annealed sigma): supcon 0.9376 -> 0.9287
(0.86/1.00), supsig 0.8433 -> 0.8211 (0.62/1.00), nplmcw 0.8990 ->
**0.9231** (0.50/1.00).  Clean-lineage confirmation of the program
pattern: only the classwise-NPLM base takes discovery positively (+0.024);
supcon/supsig degrade.  Per-event/Maha stay C100-dead; MMD post is strong
at f>=0.02 on all three.  Best clean full pipeline: scratch-nplmcw +
discovery = probe 0.923 with calibrated geometry.

Verdicts: (1) supcon_sigreg BEATS supcon on 2/3 clean bases (the hub-init
ordering reverses -- the SIGReg marginal earns its keep when the trunk is
honest).  (2) Supervised heads reach 0.90-0.93 on every scratch base,
within ~0.01-0.02 of the hub-init 100-D numbers: 20 epochs of supervised
loss training nearly erases the hub trunk's probe advantage.  (3) Scratch
bases do not rescue label-free NPLM (0.66-0.75).  Best caveat-free
pipeline: scratch-simclr base + supcon_sigreg head (0.9327).

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

### Exp 59: NPLM residual + concat spaces (C10, 16+16, annealed sigma)

| space          | probe pre/post | acc    | eucl   | mahaT  | perevt pre/post@.02 | SpK@.02   | Maha@.02  | MMD@.02   |
|----------------|----------------|--------|--------|--------|---------------------|-----------|-----------|-----------|
| sup->res-nplm  | 0.9340/0.9840  | 0.9117 | 0.8344 | 0.7680 | 0.275/0.511         | 0.88/1.00 | 0.70/0.90 | 0.34/0.90 |
| sup+nplm       | 0.9321/0.9791  | 0.8812 | 0.7910 | 0.7710 | 0.232/0.438         | 0.42/0.94 | 0.84/0.86 | 0.24/0.80 |
| supcon+nplm    | 0.9540/0.9195  | 0.8848 | 0.7681 | 0.7414 | 0.015/0.116         | 0.26/0.30 | 0.64/0.88 | 0.66/0.06 |
| nplmsup+nplm   | 0.8870/0.9180  | 0.8691 | 0.7051 | 0.7596 | 0.138/0.214         | 0.12/0.04 | 0.66/0.62 | 0.08/0.10 |
| nplmsup->res   | 0.8652/0.8956  | 0.7111 | 0.4316 | 0.5962 | 0.015/0.076         | 0.04/0.08 | 0.30/0.36 | 0.00/0.02 |

sup->res-nplm is the NEW OVERALL C10 CHAMPION: record probe post (0.9840),
record eucl (0.8344, beats exp-36 sup->res-hybrid 0.8159), and near-max
post power on every statistic -- the NPLM residual strictly improves on the
NT-Xent residual.  supcon+nplm takes the best PRE-discovery probe in the
program (0.9540) with real geometry (vs supcon+simclr's 0.645 eucl) -- the
label-free nplm_bilinear half is a strict upgrade over the SimCLR half --
but inherits SupCon's discovery fragility (0.954->0.920): use it
discovery-free.  Residual-on-NPLM-trunk fails (compact geometry leaves the
residual nothing); all-NPLM concat is mid-pack.

### Exp 59 on CIFAR-100 at 50+50 = 100-D, classwise-lam5 supervised half

| space          | probe pre/post | acc    | eucl   | mahaT  | SpK@.02   | MMD@.02   |
|----------------|----------------|--------|--------|--------|-----------|-----------|
| supcon+nplm    | 0.9431/**0.9467** | 0.5798 | 0.3281 | 0.3282 | 0.22/0.28 | 0.46/0.82 |
| sup->res-nplm  | 0.9294/0.9385  | 0.5890 | 0.5404 | 0.3846 | 0.16/0.12 | 0.26/0.24 |
| nplmsup+nplm   | 0.9206/0.9321  | 0.5286 | 0.4123 | 0.3760 | 0.00/0.28 | 0.42/**0.94** |
| sup+nplm       | 0.9191/0.9160  | 0.5290 | 0.5449 | 0.3294 | 0.42/0.06 | 0.22/0.44 |
| nplmsup->res   | 0.9111/0.9330  | 0.2021 | 0.4380 | 0.3517 | 0.04/0.12 | 0.12/0.44 |

The 100-D + cw-lam5 configuration redeems the C100 constructions: the NPLM
residual jumps 0.799 -> 0.929 (+0.13; the 32-D "residuals don't transfer"
verdict was partly dimensional), discovery is probe-POSITIVE for 4/5 arms
(supcon+nplm 0.9467 post = best C100 concat probe, second overall only to
sup@100d 0.9546), and nplmsup+nplm posts the C100 MMD record post (0.94
@0.02).  Maha/per-event stay weak (C100 pattern).  Backbones remain
hub-cifar100-pretrained (see exp 67 for from-scratch bases).

### Exp 59 on CIFAR-100 (16+16, annealed sigma)

| space          | probe pre/post | acc    | eucl   | mahaT  | SpK@.02   | Maha@.02  | MMD@.02   |
|----------------|----------------|--------|--------|--------|-----------|-----------|-----------|
| supcon+nplm    | 0.9417/0.9316  | 0.5532 | 0.4364 | 0.4146 | 0.26/0.56 | 0.00/0.08 | 0.48/**0.90** |
| sup+nplm       | 0.8958/0.8603  | 0.4959 | 0.5666 | 0.4166 | 0.10/0.06 | 0.04/0.08 | 0.30/0.46 |
| nplmsup+nplm   | 0.8957/0.8999  | 0.4135 | 0.5260 | 0.4892 | 0.16/0.14 | 0.02/0.00 | 0.62/0.62 |
| nplmsup->res   | 0.8448/0.8660  | 0.1513 | 0.5164 | 0.4167 | 0.04/0.22 | 0.04/0.00 | 0.12/0.52 |
| sup->res-nplm  | 0.7985/0.8001  | 0.5098 | 0.4899 | 0.3249 | 0.26/0.18 | 0.04/0.00 | 0.30/0.52 |

C100 verdicts: supcon+nplm ties standalone supcon's probe (0.9417) with
better geometry (mahaT 0.415 vs 0.341) and posts the best mid-fraction MMD
on record for C100 (0.90 post @0.02) -- the nplm-bilinear-as-feature-half
upgrade holds on both datasets.  The NPLM residual does NOT transfer
(0.799; the exp-36 "residual wins are a separable-data effect" verdict
extends to NPLM residuals).  Discovery on C100 stays probe-neutral
(rate-blocked) but lifts MMD broadly; Maha and per-event stay dead.

### Exp 58: discovery fine-tune A/B -- proto/repulse vs NPLM+sigreg
(C10, both supervised NPLM arms, annealed-sigma SparKer throughout.)

| config                  | probe post | perevt@.02 | SpK@.02 | Maha@.02 | MMD@.02 |
|-------------------------|-----------|------------|---------|----------|---------|
| nplm_sup_dist / proto   | 0.9661    | 0.299      | 0.52    | 0.40     | 0.34    |
| nplm_sup_dist / nplm    | 0.8117    | **0.572**  | **1.00**| **0.90** | **0.74**|
| nplm_dist_sup_cw / proto| 0.9676    | 0.483      | 0.80    | 0.62     | 0.42    |
| nplm_dist_sup_cw / nplm | 0.8364    | **0.575**  | **0.92**| 0.56     | **0.72**|

The probe-vs-calibration dissociation recurs at the FINE-TUNE level: the
proto/repulse update maximizes probe-readable structure (0.97) but
partially decalibrates the space; the NPLM+sigreg update keeps the space
calibrated -- every power statistic favors it, often 2x (per-event 0.31 at
f=0.001!), it holds round-2 pool purity (0.22-0.30 vs 0.03-0.11), does not
fragment discovered anchors (1 vs 5), and posts the best margin /
mean-anchor AUCs (0.95-0.96).  Choose the ft objective by the downstream
consumer: linear readout -> proto; dataset-level detection -> NPLM-ft.
These annealed-sigma proto rows supersede the exp-55 k1 SparKer artifacts.

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
shapes the endgame.

Exp-65 nplm_dist_sup_cw with LEARNED repulsed means (C100, paired vs the
exp-53 fixed-anchor row): probe 0.8890 (0.8864), acc 0.3861 (0.3471),
supAUC 0.932, MMD@0.02 0.68 (0.52) -- but eucl 0.3326 (0.4525), mahaT
0.3097 (0.3717), per-event 0.010 (0.020), cent->anchor 7.41 (3.28).
Verdict: learning the means under the recipe's Coulomb repulsion does NOT
fix realizability -- it overshoots (the repulsion equilibrium sits far
outside where classwise-sigreg alone can pull the embeddings, 7.4 vs 3.3),
trading calibrated geometry for modest acc/MMD gains.  Untried levers:
smaller rep_weight, centroid-initialized means, or a proto term to pull
embeddings to their means (which is exactly the supervised recipe).

Exp-64 pretrain -> full CE-ft (C100; pretrain 20ep on 99 seen classes,
then CE-ft 10ep on ALL 100 incl. holdout 4; 32-D):

| pretrain          | top1   | seen   | holdout recall |
|-------------------|--------|--------|----------------|
| nplm_dist_sup_cw  | 0.4914 | 0.4938 | 0.250          |
| nplm_bilinear     | 0.4741 | 0.4771 | 0.180          |
| simclr            | 0.4461 | 0.4498 | 0.080          |
| none (CE only)    | 0.4433 | 0.4477 | 0.010          |
| nplm_sup_dist     | 0.4385 | 0.4429 | 0.000          |
| simclr_sigreg     | 0.4374 | 0.4408 | 0.100          |
| lejepa            | 0.3991 | 0.4001 | 0.300          |

Verdicts: classwise-NPLM pretraining is the best substrate for later full
supervision on BOTH axes (+0.048 top1 over no-pretrain AND 25x the
holdout absorption); nplm_bilinear is the best label-free.  The marginal
choice flips new-class absorption completely: global-sigreg supervised
NPLM leaves ZERO room for the new class (recall 0.000 -- the collapse
geometry has nowhere to put it) while classwise reserves per-class
structure (0.250).  LeJEPA absorbs the new class best (0.30) but costs
-0.044 overall.

Exp-63 supervised-CE baseline (C100, 32-D, holdout 4): probe 0.9363,
acc 0.5356 (cls top-1 0.5204), eucl 0.3938, mahaT 0.3969, per-event 0.010.
Plain CE TIES supcon's probe (0.940+-0.004) with the same uncalibrated
geometry -- supcon's probe edge over simple supervision is negligible; the
whole value of the SIGReg/NPLM arms is the calibration axis.

Exp-61 multi-seed VERDICT (5 paired seeds, C100): the "nplm_bilinear
matches supcon's probe" claim from exp 50 does NOT survive.  supcon probe
0.9401 +- 0.0042 (seed-stable); nplm_bilinear 0.8548 +- 0.0422
(heavy-tailed: per-seed 0.789-0.889, with 0.918/0.935 from earlier seeds
as tail draws).  Paired diff -0.085 +- 0.018 sem, t=-4.73: supcon
decisively better on the probe.  The GEOMETRY advantage survives seeds:
nplm_bilinear eucl 0.502+-0.038 / mahaT 0.465+-0.028 vs supcon
0.404+-0.025 / 0.386+-0.013.  Corrected verdict: label-free NPLM buys
calibrated geometry, not probe parity; its seed variance (10x supcon's)
is itself a finding.

64-D C100 rerun (exps 50/53 64d, results_*_cifar100_64d.npz): probe
supcon 0.9389, supcon_sigreg 0.9129 (+0.03 vs 32d), simclr 0.8854 (+0.03),
nplm_bilinear 0.8584 (-0.077, eucl 0.35 vs 0.53 -- loses probe AND
geometry), nplm_sup_dist 0.8377 (+0.03, best MMD@0.02 0.66), lejepa 0.8265
(-0.04); classwise arms flat-to-worse (bil_cw 0.8677, bil_sup_cw 0.8253
(-0.10), dist_sup_cw 0.8605), cent->anchor still 3.4-3.7 -- 64-D does not
make 100 anchors realizable.  Verdict: high-D mildly helps the softmax
arms (linear-readout effect) but DEGRADES the NPLM/bilinear program --
the 32-D defaults stand for calibrated losses on C100.

100-D C100 study (exps 50/53/66 at --dim 100, holdout 4):

| arm              | probe  | acc    | eucl   | mahaT  | SpK@.02 | MMD@.02 | perevt |
|------------------|--------|--------|--------|--------|---------|---------|--------|
| sup (recipe)     | 0.9546 | 0.5545 | 0.3947 | 0.3927 | 0.04    | 0.52    | 0.010  |
| supcon           | 0.9443 | 0.5630 | 0.4034 | 0.3581 | 0.16    | 0.34    | 0.000  |
| supcon_sigreg    | 0.9364 | 0.5895 | 0.4764 | 0.4281 | 0.02    | 0.38    | 0.010  |
| nplm_dist_sup_cw | 0.8440 | 0.2591 | 0.5294 | 0.4627 | 0.18    | 0.50    | 0.030  |
| nplm_sup_dist    | 0.8271 | 0.2579 | 0.4103 | 0.3775 | 0.08    | 0.70    | 0.020  |

Verdicts: (1) the supervised-SIGReg recipe at 100-D posts the BEST C100
probe in the program (0.9546, vs 0.9039 at its native 16/32-D) -- high-D
linear-readout effect confirmed for the recipe itself.  (2) supcon_sigreg
+0.053 probe at 100-D (0.9364), now the best all-rounder (acc 0.590,
mahaT 0.428).  (3) ORTHOGONAL ANCHORS DO NOT FIX classwise-NPLM
realizability: cent->anchor is ~constant across dims (3.28/3.43/3.46 at
32/64/100-D) -- the stall is an equilibrium of pull strength (lam=1
classwise-sigreg vs the compact NPLM interaction), not dimensional
crowding.  Refutes the dimension hypothesis; the untested lever is lam or
anchor scale.  Still, 100-D gives classwise NPLM its best C100 geometry
(eucl 0.529, mahaT 0.463) at a probe cost (0.844 vs 0.886).

Classwise-lam scan + discovery (exp 53/55 rerun, C100 32-D,
nplm_dist_sup_cw): higher lam closes the anchor gap monotonically
(cent->anchor 3.28/2.68/2.11 at lam 1/5/20 -- confirming the
pull-strength-equilibrium mechanism after the dimension hypothesis died).
lam=5 is the sweet spot: acc 0.347 -> 0.522 (+0.17!), mahaT 0.392, probe
-0.015; lam=20 over-regularizes.  Discovery on the lam=5 arm: probe
0.8585 -> 0.8995 (+0.041) -- the FIRST C100 arm where discovery improves
the probe, despite pool purity still ~0.002 (the impure-pool ft now acts
as supervised refinement instead of noise, because classes are
separated); per-event post 0.13@0.02, MMD post 0.66@0.02 / 1.00@0.05.
Best C100 classwise-NPLM configuration on record: lam=5 + discovery.

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

## Stanford Cars end-to-end ft suite (exp 70; DINO trunk trainable,
exp-49 recipe, 100-D heads, holdouts 186-195 excluded from ft; discovery
= feature-space loop on each arm's own ft-trunk bank.  Full tables in
docs/AIRCRAFT_MASTER_TABLE.md.)

| arm | probe pre -> post | acc | eucl | mahaT | perevt pre/post@.05 | Maha@.1 pre |
|---|---|---|---|---|---|---|
| supcon-ft | 0.7332 -> **0.7666** | 0.462 | 0.603 | 0.564 | 0.136 / 0.202 | 0.94 |
| ss-ft | 0.7191 -> 0.7474 | 0.330 | 0.551 | 0.547 | 0.103 / 0.195 | 0.68 |
| nplm-bil-ft | 0.5912 -> 0.7044 | 0.063 | 0.506 | 0.496 | 0.026 / 0.108 | 0.04 |
| simclr-ft | 0.6238 -> 0.6839 | 0.108 | 0.469 | 0.476 | 0.023 / 0.063 | 0.18 |
| sigreg-ssl-ft | 0.5843 -> 0.6903 | 0.083 | 0.483 | 0.496 | 0.033 / 0.042 | 0.16 |
| nplm-sup-ft | 0.5490 -> 0.6001 | 0.105 | 0.555 | 0.547 | 0.061 / 0.171 | 0.18 |

Discovery probe-POSITIVE on all six e2e-ft spaces (reversing frozen-trunk
exp 69) with pool purity still <= 0.14: on a task-specialized trunk the
impure-pool ft refines instead of erodes.  supcon-ft + discovery (0.767)
is the cars probe champion under the strictest protocol on record.

Exp 70 was then run over the FULL GRID: {cars, flowers, dtd, galaxy10} x
{dino, lejepa, visreg} bases, same 6 arms + battery + discovery (24 h of
runs, 2026-08-06/07).  Grid verdicts (full tables in
docs/AIRCRAFT_MASTER_TABLE.md): supcon-ft + discovery probe-positive in
12/12 dataset x base cells (the universally safe pipeline; discovery
positive 50/72 cells overall).  Pre-probe champion by regime, base-
invariant: ss-ft on fine-grained many-class (cars 0.71-0.74, flowers
0.75-0.80), simclr-ft on texture (dtd 0.81-0.854), supcon-ft on few-class
coarse (galaxy10 0.92-0.94).  nplm-sup-ft: calibration crown at few
classes (galaxy10 all bases, purity 0.46-0.50, per-event 0.35-0.50) and a
+0.14..+0.20 discovery rescue on every DTD base, but underfits few-shot
many-class data (flowers).  ss-ft: record purities (dtd 0.80-0.81,
flowers 0.60-0.66) but discovery-unstable (flowers/LeJEPA 0.79 -> 0.61).
E2e ft erases the frozen-base ranking (VISReg takes several pre records).

Exp 71 (`71_residual_ft_grid.py`, 2026-08-07/08) ran the exp-49 residual
fine-tune on the exp-70 parents over the same 12-cell grid (supcon->res,
ss->res, supcon->res-nplm; 36 e2e fts).  Residuals beat the discovery
pipeline in 12/12 cells and set new records on cars 0.855 (visreg
res-nplm concat, +0.088 -- also best cars acc/eucl/mahaT/perevt at once,
first cars space to break the dissociation), flowers 0.885 (dino
res-nplm concat), galaxy10 0.975 (lejepa res concat, probe-only).  Only
dtd's simclr-ft pre record (0.854) survives.  Objective split: res-nplm
concat wins fine-grained cells, plain res wins texture/coarse; supcon-ft
is the universal parent (ss-ft parents fail fine-grained).  Full grid in
docs/AIRCRAFT_MASTER_TABLE.md.

Exp 72 (`72_residual_discovery.py`, 2026-08-09): discovery on each
cell's exp-71 winner.  Purity-gated and saturating: stacks where the
residual space separates novelty (flowers 0.885 -> 0.906 NEW RECORD,
dtd 0.847 -> 0.862 NEW RECORD -- simclr's last pre record finally
falls), flat/slightly negative elsewhere (cars 0.855 and galaxy10 0.975
stand from exp 71).  galaxy10/visreg residual + discovery: per-event
0.32-0.45 at ALL fractions (best low-f per-event on record).

Aircraft joined the grid (2026-08-10) with NEW open-world parents
(exp-70 protocol, holdouts 90-99 excluded): supcon->res-nplm concat wins
all three bases (0.816/0.848/0.863 dino/lejepa/visreg; visreg near-
Pareto, perevt 0.363), beating the caveated 0.812 prior best.  Exp-72
discovery on the aircraft winners is probe-NEGATIVE at moderate purity
(0.32-0.53) while buying geometry/per-event (visreg 0.30-0.50 all f):
purity is necessary but NOT sufficient -- fine-grained res-nplm probe
directions are what the discovery ft erodes.  Final probe records, all
residual pipelines: aircraft 0.863, cars 0.855, flowers 0.906, dtd
0.862, galaxy10 0.975.

Exp 73 (`73_cifar_residual_ft.py`, 2026-08-11) brought the exp-71
residual-ft recipe home to CIFAR: c10 32+32 supcon->res concat 0.9536
(ties the pre-discovery concat record); c100 100+100 supcon->res concat
0.9594 = NEW C100 PROBE RECORD (prev 0.9546; parent seed drew 0.9577 --
honest residual delta +0.017 over its own parent).  The plain-res
residual half is the best-calibrated supervised C100 space (eucl 0.650,
perevt 0.08).  Plain res beats res-nplm on CIFAR (many-class low-res:
same side of the objective split as dtd/galaxy10).  Also fixed
exp55.train_arm for plain arms.

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
