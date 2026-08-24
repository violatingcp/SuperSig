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

Exp 74 (`74_cifar_residual_discovery.py`, 2026-08-12): discovery on the
exp-73 concats (image-space loop, two-net concat backbone).  Same
purity-gate verdict as exp 72: at purity <=0.03 the probe is flat (c10
~0.953 both arms; c100 res flat / res-nplm -0.007) while calibration
improves a lot (c10 res-nplm eucl 0.712->0.794, Maha power 0.80@.02;
c100 res-nplm eucl 0.321->0.530, mahaT 0.302->0.455, MMD 0.84@.02).
Probe records stay pre-discovery (c100 0.9594) / exp-59 post (c10
0.984); discovery-on-residuals is a calibration repair, not a probe
booster, at low purity.

Exp 75 (`75_multiseed_residual.py` + seed-aware exps 70/71, 2026-08-16):
multi-seed validation of the residual concats (exp-61 protocol).  Fully
paired per seed: parent + res + res-nplm children, 3-probe-seed battery.

| cell | space | probe (mean+-sd) | paired delta | wins |
|---|---|---|---|---|
| c10 32d, 5 seeds | parent | 0.9395+-0.0020 | -- | -- |
| | res concat | 0.9552+-0.0023 | +0.0157+-0.0004 | 5/5 |
| | res-nplm concat | 0.9451+-0.0023 | +0.0057+-0.0014 | 5/5 |
| c100 100d, 5 seeds | parent | 0.9467+-0.0098 | -- | -- |
| | res concat | 0.9523+-0.0077 | +0.0056+-0.0038 | 4/5 |
| | res-nplm concat | 0.9552+-0.0064 | +0.0086+-0.0059 | 5/5 |
| cars/visreg, 3 seeds | supcon-ft parent | 0.686+-0.016 | -- | -- |
| | res-nplm concat | 0.833+-0.017 | +0.148+-0.004 | 3/3 |

Verdicts: (1) the residual effect is REAL everywhere -- 17/18 paired
comparisons positive; the transfer-scale effect (+0.148, sd 0.004 across
seeds!) is 20x the CIFAR-scale one.  (2) C100 objective ranking FLIPS
under seed averaging: res-nplm concat (0.9552, 5/5) beats plain res
(0.9523, 4/5) -- the exp-73 "plain res wins CIFAR" verdict now applies
only to c10; c100 rejoins the many-class fine-grained regime rule
(res-nplm).  (3) Records were fair draws, not flukes: c100 0.9594 ->
honest 0.955+-0.006 (best seeds 0.9617/0.9615); c10 res concat mean
0.9552 exceeds the old 0.9536; cars 0.855 -> honest 0.833+-0.017.
Geometry split persists: plain-res concat keeps the calibration crown on
c100 (eucl 0.607 vs res-nplm 0.339).  Results logs/exp75/*.npz.

Exps 76/77 (`76_interpretability.py`, `77_space_similarity.py`,
2026-08-16): class-centroid interpretability + inter-space similarity
batteries over the cached campaign spaces (full write-up
docs/SPACE_GEOMETRY.md).  Headlines: (1) cifar100 holdout beaver lands
mutually top-2 with otter unsupervised; supcon parent superclass
agree@1 0.70 (chance 0.04), dendro purity 0.59; the res-nplm concat
tightens semantics at near-parent purity while the plain-res residual
half scrambles them (concat restores).  (2) Supervision erases the
pretraining base (supcon-ft cross-base CKA 0.68-0.85 ~ cross-loss
same-base; simclr-ft 0.46-0.62 keeps base identity) -- the geometric
mechanism of the exp-70 base-invariant regime rules.  (3) Residual
children: near-rotations on coarse transfer cells (galaxy10 R^2 0.98),
~10% new variance on fine-grained (cars 0.91), near-total
reorganization on CIFAR (c100 plain-res R^2 0.32) -- but probe gains
anti-correlate with drift across regimes: new variance must encode the
holdout to pay.  (4) nplm-bilinear is a geometric outlier everywhere
(CKA ~0.2-0.4 vs all arms) yet linearly decodable FROM softmax spaces
(asym R^2 up to 0.97): NPLM reorganizes, does not add, information.
(5) TwoNN ID is a loss fingerprint (nplm-sup ~2-3 < ssl/sigreg 5-7 <
supcon/frozen 9-13).  (6) LID novelty (Levina-Bickel, seen-train refs):
weak on most cells (0.56-0.74) but 0.95-0.96 on flowers in EVERY base
-- record-territory unsupervised score, exp-78 verification candidate.
Results logs/exp76/interp_*.md, logs/exp77/results_*.npz.

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

Exp 78 (`78_lid_verification.py`, 2026-08-17): the flowers LID-novelty
result verified.  Holdout-rotation on the frozen trunk (10 random
10-class holdouts): flowers LID 0.879+-0.018 vs kNN-dist 0.692 (cars
0.545 -- dataset property, not artifact); LID beats kNN-dist by
+0.08-0.12 and eucl by +0.03-0.06 on all flowers cells (Spearman
0.55-0.92); bootstrap CIs +-0.006, worst holdout class 0.86, k=20-50
plateau.  HEADLINE: unsupervised LID 0.951-0.955 > supervised flowers
probe record 0.906 -- first unsupervised-beats-probe result of the
campaign.  Follow-up: add LID(k=20) to the standard battery.

Exp 79 (`79_lid_pool_discovery.py` + `discovery.lid_pool_scores`,
2026-08-17): LID as discovery pool scorer, A/B vs distance on the three
exp-72 flowers winners.  LID triples round-2 pool purity (dist collapses
0.18-0.33 as the ft inflates the space; scale-free LID holds 0.60-0.68)
-- strictly better pooling -- but post probes are unchanged (+0.000 /
-0.029 / +0.007; the 0.906 record ties).  Sharpened purity-gate lesson:
above the gate, purity is not the binding constraint; discovery-ft
dynamics are.  run_discovery gained pool_score={"dist","lid"} (default
unchanged).

Exp 87 (`87_lid_residual_half.py`, 2026-08-20; IMPROVEMENT_TESTS #87):
LID by half on the flowers/cars champions.  FALSIFIER FIRED on flowers:
the residual-half LID is uniformly WORSE than the parent half (dino
0.923 vs 0.951, lejepa 0.825 vs 0.893, visreg 0.733 vs 0.954); concat
~= parent everywhere.  The flowers LID signal lives in the PARENT
(class-structure) half -- novelty is a class-level dimensional anomaly,
not a within-class one, refining the exp-78 mechanism story.  On cars
the prediction half-holds: residual-half LID beats parent on dino
(0.677 vs 0.629) and visreg (0.752 vs 0.682 -- best cars LID on
record) but not lejepa (0.559 vs 0.648), and eucl 0.747 still matches
it: no regime flip, cars stays a distance-score dataset.  Pseudo-
rotation control (conservative; classes were labeled during ft) ranks
the halves identically.  Evaluation-only, no training.

Exp 90 (`90_score_predictor.py`, 2026-08-20; IMPROVEMENT_TESTS #90):
unsupervised predictor for WHICH novelty score (LID vs eucl) -- the
FALSIFIER FIRED.  Across 17 champion cells, none of the three label-free
tail diagnostics (pool dist-ratio, TwoNN ID gap, LID gap) tracks the
measured LID-eucl AUC gap: Spearman +0.07 / +0.09 / -0.16, best
sign-accuracy 12/17 vs an 11/17 always-say-LID base rate.  The
LID-vs-distance regime rule stays empirical; the "on-manifold is
measurable without labels" mechanism story does not survive contact with
these candidates.  Side-findings on the champion concats: LID wins the
gap in 12/17 cells, incl. cifar100 resnplm-cat +0.382 (LID 0.738 vs eucl
0.356 -- LID is the only usable novelty score in that space) and
galaxy10/lejepa +0.273; aircraft stays firmly a distance regime
(-0.11 to -0.20).  Evaluation-only, no training.

Exp 80 (`80_sparker_all_spaces.py`, 2026-08-20): SparKer pre-discovery
coverage closed -- 106 spaces that never had power batteries (aircraft
e2e cells were --skip-power; exps 71/73 never ran them), exp-70 protocol
exactly (annealed sigma, M=16, n_null=200, 50 toys), all from cached
banks.  Full grids logs/exp80/results_*.npz.  Verdicts:

- The res-nplm RESIDUAL CHILD is the best dataset-level detector in the
  fine-grained visreg/dino cells, beating its own supcon parent:
  aircraft/visreg 0.82@f=0.05 (parent 0.44), cars/visreg 0.96@0.05
  (child alone), cars/dino 0.80@0.05.  The calibrated residual half
  carries real SparKer power that the probe/eucl columns never showed.
- Concats preserve or improve the parent's power (aircraft/visreg
  res-cat 1.00@0.1, galaxy10/visreg resnplm-cat 1.00@0.05, 0.46@0.02)
  -- concatenation costs nothing at the dataset level either.
- Base rule: lejepa residual children are systematically weak (cars/
  lejepa res + resnplm 0.12-0.14@0.1 vs visreg 1.00) -- matches the
  exp-77 finding that lejepa parents leave the most residual drift.
- Flowers is SparKer-weak in every residual space (best 0.94@0.1,
  most <=0.4@0.1): the on-manifold novelty that LID detects per-event
  is nearly invisible to dataset-level kernel tests.  Score-regime and
  statistic-regime coincide.
- CIFAR: c10 supcon parent keeps the best curve (0.50@0.01, 0.98@0.02);
  resnplm-cat is the only construction close (0.34/0.92); plain
  res-cat and bare children are weak below f=0.1.  c100 all spaces flat
  to 0.02 (<=0.30) then saturate at 0.05 -- the injection clamp regime.

Exp 81 (`81_critic_variance.py`, 2026-08-20; IMPROVEMENT_TESTS #81):
critic-variance 2x2 CONFIRMED with a refinement.  C100 32-D, 5 paired
seeds, NPLM + global SIGReg, tau=1:

  arm        probe mean+-sd    s_fin   s_exp(e^g spread)
  dist-inst  0.7434+-0.0235    7.33    0.11
  dist-sup   0.8366+-0.0125    1.73    0.27
  bil-inst   0.8914+-0.0320    1.32    11.8
  bil-sup    0.8651+-0.0785    0.88    34.9

- sd splits by CRITIC 3.1x (dist 0.018, bil 0.055) vs 1.6x by positives
  -- the variance is a property of the bilinear critic, as App. A
  predicts.  One-line rule confirmed: never use the bilinear critic
  under NPLM when seed stability matters; dist-sup is the stable corner
  (+-0.0125).
- REFINEMENT: the raw critic spread s = sd(g) ANTI-correlates with the
  probe sd (-0.80) -- the loose proxy fails.  The e^g spread (the exact
  App.-A gradient-variance quantity) correlates +0.80 and separates the
  arms by two orders of magnitude (0.1-0.3 vs 12-35).  The theory holds
  only in its exponential form.
- Side numbers: bil-inst drew a strong mean this seed set (0.891+-0.032
  vs exp-61's 0.855+-0.042); geometry columns unchanged in ordering.

Exp 82 (same run; IMPROVEMENT_TESTS #82): calibration residual as a
label-free seed selector -- FALSIFIED operationally.  Spearman(|resid|,
probe): bil-inst +0.30, bil-sup -0.10 (prediction was < -0.6);
best-of-5-by-residual UNDERPERFORMS the random-seed mean on bil-inst
(0.839 vs 0.891, max 0.933).  The residual is structurally negative for
distance critics (e^g <= 1 caps E_ref[e^g] at 1 without a bias term --
motivates exp-83's distance+bias form) and far from 0 for every bilinear
seed; per-seed variation does not track probe.  The seed variance lives
in the SIGReg/interaction balance, not in calibration failure.

Exp 86 (`86_frozen_parent_discovery.py`, 2026-08-20; IMPROVEMENT_TESTS
#86): the aircraft discovery failure is FIXED -- by freezing MORE than
predicted.  Three variants per champion cell, exact exp-72 recipe/seed:

  cell    unfrozen dP/perevt  frz-parent dP/perevt  frz-BOTH dP/perevt/pur-r2
  dino    -0.037 / 0.330      -0.013 / 0.330        0.000 / 0.306 / 0.211
  lejepa  -0.016 / 0.297      -0.020 / 0.213        0.000 / 0.267 / 0.346
  visreg  -0.012 / 0.526      -0.013 / 0.498        0.000 / 0.523 / 0.176

- The stated prediction (freeze-parent holds probe, keeps gains) only
  holds on dino; on lejepa/visreg freeze-parent matches unfrozen's cost.
- The SHARPER verdict: freeze-both -- discover anchors in a completely
  frozen space -- keeps 90-99% of the per-event gain (0.523 vs 0.526 on
  the record cell) at exactly zero probe/geometry cost, and is the only
  variant whose round-2 pool purity does not collapse (0.176-0.346 vs
  0.023-0.065; the frozen space cannot inflate, so the distance
  threshold stays valid -- the exp-79 mechanism solved from the other
  side).  On fine-grained data the discovery ft update was almost pure
  cost: the per-event gain lives in the ANCHORS, not the space update.
- Operational rule: on fine-grained cells run discovery with the space
  frozen (anchors only).  Discovery is then strictly non-negative: the
  aircraft record 0.8634 is retained WITH per-event 0.523.  "Choose by
  consumer" collapses to one recipe there.

Exp 92 (`92_sparker_discovery.py`, 2026-08-20; IMPROVEMENT_TESTS #92):
SparKer density-ratio centres replace the discovery clustering (quantile
pool -> BIC k-means -> merge), A/B vs run_discovery at identical
recipe/seed:

  cell             variant   pur r1   pur r2   probe post   mahaT post
  cars/dino        dist/spk  0.161/0.214  0.059/0.212  0.8205/0.8291  0.5430/0.5423
  aircraft/visreg  dist/spk  0.525/0.591  0.065/0.620  0.8511/0.8397  0.7823/0.7472
  flowers/dino     dist/spk  0.609/0.618  0.240/0.610  0.9061/0.9015  0.8525/0.8386

- PARTIALLY CONFIRMED.  Cars purity rises (r1 0.161 -> 0.214, +33%
  rel.) as the mechanism predicts -- density ratio sees what distance
  misses -- but stays below the ~0.3 gate: a real improvement, not the
  full rescue.
- The UNPREDICTED headline: SparKer pooling makes round-2 purity
  immune to the space-inflation collapse in every cell (0.21-0.62 vs
  0.06-0.24) -- like LID (exp 79) but via density ratio, and it also
  proposes the anchors.  Round-2 pool sizes stay full (790 vs 321 on
  aircraft).
- Cost: the bigger, purer pools feed MORE discovery-ft pressure -- the
  aircraft probe cost worsens (-0.024 vs -0.012).  The obvious
  combination is exp-92 pooling + exp-86 frozen space (purity without
  the ft erosion); left as the 92+86 follow-up.
- Flowers probe 0.9015 vs the 0.9061 record tie: parity within seed
  noise, as predicted for the works-cell.

Exp 94 (`94_null_validity.py`, 2026-08-20; IMPROVEMENT_TESTS #94): null
validity under novelty-seeking fine-tunes -- FALSIFIER FIRED, and that
is the good outcome.  Realized FPR at nominal alpha=0.05 (null from
TEST-set background untouched by any ft; FPR on the corpus the
discovery ft optimized; Clopper-Pearson 68%):

  cell          frozen           ft-full          ft-split
  flowers/dino  0.000 [0,.018]   0.000 [0,.018]   0.000 [0,.018]
  cars/dino     0.040 [.021,.07] 0.040 [.021,.07] 0.060 [.037,.094]

The discovery ft cannot manufacture false positives at campaign sample
sizes: FPR is at or below nominal in every regime.  Flowers is
conservative at baseline (even the frozen champion was supcon-ft'd on
the train corpus, compressing its background); cars is the calibrated
witness (0.040 frozen) and stays nominal after ft-full.  Verdict: the
current protocol's nulls are VALID, split-disjointness is not required
at these sizes, and exp 95 (SparKer as a training loss) is formally
cleared -- with the caveat that exp-95's stronger, directly-adversarial
objective should still rerun this check on its own trained encoder.

Exp 92b (`92_sparker_discovery.py --variants *-frozen`, 2026-08-20): the
92+86 combination -- SparKer density-ratio pooling in a fully frozen
space -- DOMINATES.  Space bit-identical in every run (probe/mahaT
pre=post by construction); purity and anchor margin:

  cell             dist-frozen r1/r2   spk-frozen r1/r2   spk margin r1/r2
  cars/dino        0.161 / 0.068       0.214 / 0.215      0.782 / 0.757
  aircraft/visreg  0.525 / 0.176       0.591 / 0.608      0.893 / 0.895
  flowers/dino     0.609 / 0.414       0.618 / 0.620      0.962 / 0.962

- sparker-frozen beats distance-frozen on purity in all six round
  measurements and ties-or-beats the UNFROZEN sparker purity while
  eliminating the probe cost entirely (aircraft record 0.8634 intact).
- Mechanism sharpened: on cars the frozen-space distance r2 still
  collapses (0.068) -- so the round-2 failure there is ANCHOR
  ABSORPTION (novel points captured below the distance threshold by the
  discovered anchors), not space inflation; the density-ratio pool is
  indifferent to the anchors and immune to both failure modes.
- New default discovery recipe for fine-grained cells: freeze the
  space, pool AND propose anchors by SparKer density ratio, train
  anchors only.  Margin also improves on cars (0.78 vs 0.66).

Exp 93 (`93_np_pool_scorer.py` + `discovery.np_pool_scores`, 2026-08-21;
IMPROVEMENT_TESTS #93): dist vs LID vs NP-f pool scorer in the standard
loop -- the "np >= lid everywhere" prediction is FALSIFIED, but np owns
the regime that matters:

  cell            dist r1/r2     lid r1/r2      np r1/r2
  flowers/dino    0.609/0.240    0.678/0.679    0.618/0.620
  flowers/lejepa  0.439/0.328    0.576/0.598    0.528/0.492
  flowers/visreg  0.643/0.179    0.684/0.677    0.652/0.632
  cars/dino       0.161/0.059    0.133/0.166    0.214/0.152

- On flowers (LID's regime) lid > np in all three cells by 0.03-0.06:
  the NP lemma's asymptotic optimality loses to the closed-form ratio
  statistic at these pool sizes -- exactly the doc's stated falsifier.
- On cars (below the gate, LID's weak regime) np is the BEST r1 scorer
  (0.214 vs 0.161/0.133).  Its r2 slips under the ft'd space (0.152);
  the exp-92b frozen variant holds 0.215 -- pool with np, freeze the
  space.
- Probes unmoved everywhere (above-gate lesson of exp 79 stands).
- Operational rule: lid for on-manifold cells, np below the gate --
  and per exp-92b, run either in a FROZEN space.

Exp 99 (`99_discovery_reach.py`, 2026-08-21; from the paper branch):
f95 discovery reach -- every archived SparKer power curve inverted into
the signal fraction at which power crosses 0.95 (log-f interpolation,
honest flags for top-bracket/non-monotone/never).  575 series, 231
(40%) cross inside the measured grid; refreshed over the full current
archive incl. exp-80.  Headlines:
- Best reach anywhere: f95 = 0.019 -- a five-way tie among the c10
  post-discovery/hybrid spaces (exp 33/36/58/59) AND the plain exp-80
  c10 supcon pre space; resnplm-cat pre reaches 0.023.
- Best transfer reach: galaxy10/visreg resnplm-cat 0.046 and
  cars/visreg resnplm CHILD 0.049 -- the exp-80 finding again: the
  calibrated residual construction owns dataset-level sensitivity off
  CIFAR.
- 60% of series never cross 0.95 in-grid; the ">" rows quantify how far
  the weak spaces are from usable reach.

Exp 96 (`96_warmstart_sparker.py` + sparker mu_init, 2026-08-21;
IMPROVEMENT_TESTS #96): does the pairwise critic warm-start the
event-level test?  FALSIFIER FIRED, in the strong form: warm-starting
SparKer's centres at the trained seen-class centroids is neutral-to-
HARMFUL.  C10 32-D seed-exact spaces, f=0.02, 50-step budget power
(cold/warm): nplm_sup_dist 0.800/0.560, nplm_distance 0.120/0.080,
supcon 1.000/1.000 (saturated contrast).  Full-budget t_NP
trajectories are statistically indistinguishable.  Mechanism: the
class centroids sit exactly where the corpus/reference density ratio
is ~1, so warm centres must first migrate AWAY from their init, while
data-sample init occasionally seeds centres on novel events.  The
paper's SS5 unification claim needs its caveat stated plainly: the two
scales share the functional form and the loss, NOT transferable
learned content -- the representation's class structure is where
novelty ISN'T.

Exp 97 (`97_sparker_systematics.py`, 2026-08-21; IMPROVEMENT_TESTS #97):
M x sigma-schedule scan vs intrinsic dimension, cars/dino arms spanning
TwoNN ID 3/6/12, power@f=0.05 (100 nulls, 25 toys):

  arm (ID)         M4              M16             M64      (ratio 3/10/30)
  nplm-sup-ft (3)  .24/.28/.24     .32/.36/.32     .36/.32/.36
  sigreg-ssl (6)   .04/.04/.00     .16/.16/.12     .00/.00/.00
  supcon-ft (12)   .60/.52/.52     .44/.44/.40     .24/.20/.20

- sigma_ratio is FLAT everywhere (within toy noise): the annealed
  schedule is genuinely robust in the ratio knob -- the exp-57 lesson
  was about sigma0 matching and the anneal covers it.  That half of the
  falsifier fires, and it is worth stating.
- M is NOT flat, and the direction INVERTS the prediction: the high-ID
  supcon space wants FEWER kernels (M=4: 0.60 vs M=64: 0.24 -- more
  kernels overfit the null in high dimension, raising the threshold),
  while the low-ID nplm-sup space mildly prefers more (0.24 -> 0.36).
  The default M=16 leaves up to 0.16 power on the table on high-ID
  spaces.  Practical rule: match M INVERSELY to intrinsic dimension
  (high-ID -> M=4; low-ID -> M=16-64).
- sigreg-ssl-ft is killed outright by M=64 (0.00 at every ratio).

Exp 88 (`88_classwise_realizable.py`, 2026-08-20/21; IMPROVEMENT_TESTS
#88): realizable classwise C100 -- BOTH clauses of the prediction
falsified, and the control stole the show.  27 runs, dims 100/128/200 x
{cw, global, softmax} x 3 paired seeds:

  arm      dim  probe mean+-sd   mahaT  cent->anchor
  cw       100  0.8447+-0.0067   0.468   3.46
  cw       128  0.8456+-0.0183   0.422   3.47
  cw       200  0.8676+-0.0084   0.326   3.51
  global   100-200               0.37-0.39   --
  softmax  100  0.9125+-0.0135   0.545   --
  softmax  128  0.9044+-0.0097   0.558   --
  softmax  200  0.9063+-0.0035   0.506   --

- cent->anchor NEVER collapses: 3.46-3.52 at every dim, identical to
  the 32-D stall, across all 9 cw runs.  The anchors are unreachable
  for an OPTIMIZATION reason (the loss equilibrium parks centroids
  ~3.5 from their targets regardless of geometric feasibility) -- the
  "more interesting" falsifier branch.  Q4's realizability rule needs
  restating: dim >= n_classes is neither the enabler nor the fix.
- cw mahaT does not break the ceiling; it DECLINES with dim (0.468 ->
  0.326).  The archived 100-D row (0.463) was a fair draw, not a door.
- UNPREDICTED HEADLINE: the softmax control (supcon_sigreg) at
  100-128-D posts mahaT 0.545-0.558 (3 seeds, +-0.03) -- ABOVE the
  ~0.47-0.49 C100 calibration ceiling -- while holding probe 0.90-0.91.
  The best single-loss C100 both-currencies space on record comes from
  the CONTROL arm at high dim.

Exp 91 (`91_multiseed_records.sh` via seed-aware exps 70/71/72,
2026-08-20/21; IMPROVEMENT_TESTS #91): the four uncited records
multi-seeded (seeds 0=archived, 1, 2; paired protocol of exp-75/cars).
Citable records with uncertainties:

  cell                      construction              seeds            mean+-sd
  aircraft/visreg  supcon-ft->res-nplm concat  .8634/.8692/.8651  0.866+-0.002
  flowers/dino     resnplm-cat + discovery     .9061/.8936/.8598  0.887+-0.019
  dtd/visreg       res-cat + discovery         .8621/.8682/.8709  0.867+-0.004
  galaxy10/lejepa  supcon-ft->res concat       .9750/.9648/.9725  0.971+-0.004
  cars/visreg      (exp 75, 3 seeds)                              0.833+-0.017

- MILDER than the prediction (down 0.01-0.03): aircraft and dtd records
  were the LOW draws of their triples (seed 2 sets a new dtd best,
  0.8709), galaxy10 within 0.004.  Only flowers shifts down (-0.019,
  and its PRE-discovery concat spread is the campaign's widest:
  0.787-0.885 -- the discovery step compensates weak parents, +0.106 on
  the weakest seed vs +0.021 on the strongest).
- No ordering flipped.  All five records are now citable with
  uncertainties; master tables updated.

Exp 98 (`98_sparker_ft_discovery.py`, 2026-08-21; IMPROVEMENT_TESTS
#98): SparKer-ft as the discovery fine-tune objective -- FAILS, beyond
the stated falsifier.  Three fts fresh-paired on both exp-58 arms
(c10 32-D, identical loop/battery; spk = post power @ f=.01/.02/.03):

  arm:ft                    probe post  mahaT  perevt  spk@.02  pur r1/r2
  nplm_sup_dist:proto         0.9510    0.797   0.616    1.00   .334/.092
  nplm_sup_dist:nplm          0.8396    0.939   0.832    1.00   .334/.218
  nplm_sup_dist:sparker       0.7033    0.493   0.107    0.06   .334/.135
  nplm_dist_sup_cw:proto      0.9788    0.803   0.746    1.00   .424/.075
  nplm_dist_sup_cw:nplm       0.9017    0.923   0.779    1.00   .424/.248
  nplm_dist_sup_cw:sparker    0.6909    0.507   0.098    0.12   .424/.103

- SparKer-ft is worse than BOTH alternatives on every column INCLUDING
  the event-level statistic it optimizes (post SparKer power ~0.1 vs
  1.00).  The alternating scheme is statistic-chasing: pseudo-novel
  points climb the current f surface, class structure smears (only the
  global SIGReg guard remains), and the refitted kernel then finds less
  real separation.  Round-2 purity is mid (0.10-0.14) -- the stated
  falsifier question is moot next to the overall failure.
- Together with exp 96 (warm-start harmful) this bounds the SS5
  unification empirically: the NP loss transfers as a FORM across
  scales, but optimizing representations THROUGH the fitted kernel
  destroys the separation it measures.  Exp 95 (the same idea at full
  training scale) is hereby contraindicated in this alternating form --
  recommend not running it as specified, or redesigning with a
  class-structure-preserving term before any attempt.
- The proto/nplm split reproduces exp 58 exactly (probe vs power/purity
  dissociation) under the lighter battery -- the harness is sound.

Exp 103 (`103_lid_neighbourhood.py`, 2026-08-21; IMPROVEMENT_TESTS
#103): LID neighbourhood decomposition -- NEITHER the prediction nor
the falsifier; the mechanism is now measured.  AUC by component
(k=20 split at the modal reference class; parent/concat averaged):

  cell      full     same-class  mixed    frac-mixed(composition)
  flowers   0.87-0.96  0.85-0.95  0.52-0.62  0.83-0.94
  cars      0.60-0.72  0.60-0.73  0.54-0.65  0.59-0.70

- The prediction said the MIXED component carries the signal; observed:
  the SAME-CLASS component carries nearly all of it (flowers same
  0.94-0.95 vs full 0.95) while mixed sits at chance -- AND the bare
  neighbourhood composition (fraction of non-modal neighbours) is
  almost as good as LID itself (0.83-0.94).
- Mechanism, finally measured: a novel flower looks like a member of
  ONE seen class by distance, but (a) its radial ratio profile within
  that class's neighbours is wrong (off the class's local sheet), and
  (b) its neighbourhood is more mixed than a true member's.  The
  between-class distances carry nothing.  "Class-level anomaly" means
  anomalous AGAINST the modal class's local geometry.
- Cars: no component separates (all 0.54-0.73) -- consistent with
  0.545-0.75 full-LID; the regime rule stands, now with a mechanism on
  the side where LID works.
- Bonus practical finding: frac-mixed is a parameter-free, ratio-free
  novelty score at 0.94 on flowers -- worth a battery slot next sweep.

Exp 89 (`89_c100_rate_grid.py`, 2026-08-21; IMPROVEMENT_TESTS #89): the
C100 purity gate CANNOT be calibrated this way -- a sharpened falsifier.
Grid holdout size {1,5,10} x tau_quantile {.95,.99,.995} on fresh exp-73
resnplm concats (100-D, seed 0); purity r1/r2 and probe pre->post:

  size  q=.95        q=.99        q=.995       probe delta
  h1    .000/.004    .000/.002    .000/.004    ~0
  h5    .015/.094    .002/.059    .000/.105    -0.031
  h10   .036/.120    .002/.085    .000/.121    -0.034..-0.038

- Purity scales with holdout fraction but SATURATES far below the
  0.15-0.3 gate (max 0.121 at h10 = 10% novel corpus).
- Stricter quantiles HURT round-1 purity (h10: 0.036 -> 0.002): the
  extreme distance tail on C100 is owned by background outliers, not
  novelty.  C100 novelty is not geometrically outlying -- the same
  mechanism exp-92 identified on cars.
- Probes move DOWN wherever pools form (impure pseudo-labels poison the
  ft), so the gate's bindingness cannot even be tested by this grid.
- Conclusion: C100 discovery is not rate-blocked but GEOMETRY-blocked;
  the distance pool is the wrong instrument.  The exp-92 density-ratio
  pool (+ frozen space per 92b) is the remaining lever.

Exp 100 (`100_dense_reach.py`, 2026-08-21; IMPROVEMENT_TESTS #100):
dense f95 reach on the 17 champion spaces -- PARTIALLY CONFIRMED, and
the dataset ordering is now citable:

  citable reach (CP band):  cifar100 0.029 [.029,.035] < cifar10 0.038
  [.037,.039] < cars/visreg 0.063 [.059,.076] < galaxy10/visreg 0.067 <
  galaxy10/lejepa 0.071 < aircraft/visreg 0.072 [.068,.078] < dtd/visreg
  0.079 ~ cars/dino 0.079
  top-bracketed (*): dtd/dino 0.087, cars/lejepa 0.099
  never (> 0.1): all three flowers cells (max power 0.14-0.76),
  aircraft/dino+lejepa, dtd/lejepa, galaxy10/dino

- 8/17 cleanly bracketed vs the exp-99 baseline of 4/106 -- the dense
  grid converts half the champion table from bounds to measurements.
  The prediction ("every starred cell resolves within <=0.02") holds
  only partially: 7 cells genuinely need f > 0.1, so their reach IS the
  bound, not a grid artifact.
- Flowers confirmed >0.1 on all bases with SHALLOW curves (0.14-0.76 at
  f=0.1): per-event-detectable novelty (LID 0.95) that dataset-level
  kernel tests barely see -- the strongest score-vs-statistic regime
  split in the program.
- The visreg base owns transfer reach (best cell on all four transfer
  datasets); CIFAR concats dominate overall (0.029/0.038).

Exp 104 (`104_interpretability_panel.py`, 2026-08-21; Part C of
METRICS.md): the interpretability panel across every space -- is
distance to a labelled anchor a delta log-likelihood?  Selftest
validates all synthetic cases.  Reliable cells = CIFAR (450+/class);
the transfer cells are shrinkage-dominated (cond ~1e6-1e13 at 10-40
samples/class in 100-D) and should be read qualitatively only.

  c10:  supcon r_llr=.68 ece=.033 rms=.45 sep=7.7 | res r_llr=.81
        slope=4.3 rms=.79 sep=0.8 | resnplm rms=.13 | res-cat
        r_llr=.82 ece=.048 sep=5.5 | resnplm-cat r_llr=.67 ece=.035
        sep=10.6
  c100: everything r_llr .27-.64; supcon r_llr=.27 ece=.16; res child
        best r_llr=.64 but ece=.51

- NO space reaches slope~1: class-conditional widths are 0.13-0.79
  (all narrower than the unit target), so raw squared distances
  OVERSTATE delta-logL by 4-37x.  The SIGReg unit-width target is not
  actually achieved class-conditionally anywhere.
- The plain-res child on c10 is the most FAITHFUL space (slope 4.3,
  rms 0.79, r_llr 0.81) and the least discriminative (sep 0.85) -- the
  "perfectly faithful and useless" corner the panel was designed to
  expose.  The concats are the deployable compromise: c10 res-cat
  r_llr 0.82 + ece 0.048 + sep 5.5.
- Distance-as-LLR degrades with class count: c100 r_llr <= 0.64 and
  supcon c100 r_llr = 0.27 -- the proto-posterior reading is weakest
  exactly where mahaT is weakest.  Interpretability and class count
  trade off.

Exp 102 (`102_child_standalone.py`, 2026-08-21; IMPROVEMENT_TESTS
#102): is the residual child a standalone detector?  YES on the
visreg/dino fine-grained cells -- and cheaper than predicted:

  cell             probe(child/concat)  f95(child/concat)  perevt  agree1
  aircraft:visreg  0.858 / 0.863        0.063 / 0.072      0.411   0.53
  cars:visreg      0.836 / 0.855        0.049 / 0.063      0.256   0.44
  cars:dino        0.787 / 0.821        0.063 / 0.079      0.143   0.43
  aircraft:dino    0.811 / 0.816        0.085*/ >0.1       0.267   0.48
  dtd (all)        0.80-0.82            >0.1 / 0.079-0.087   --      --
  lejepa children  weak everywhere (cars:lejepa probe 0.63, f95 >0.1)

- On the visreg/dino fine-grained cells the child ALONE beats the
  concat's reach (cars:visreg f95 0.049 = best transfer reach measured
  anywhere) at a probe cost of only 0.005-0.034 -- NOT the "large probe
  cost" predicted -- and its semantics are legible (agree@1 0.43-0.53
  vs ~0.04 chance), NOT scrambled: the exp-76 scrambling result was
  about PLAIN-res residuals; the res-nplm children keep class structure.
- The falsifier fires on dtd: children standalone never cross 0.95
  while their concats bracket ~0.08 -- there the concat does the work.
  Same on flowers/galaxy10-dino (no standalone reach).
- Deployment rule: on fine-grained visreg/dino cells a detection
  pipeline can drop the parent (half the embedding, better reach,
  ~0.02 probe cost); elsewhere keep the concat.

Exp 101 (`101_frozen_generality.py`, 2026-08-21; IMPROVEMENT_TESTS
#101): frozen-space discovery generality -- PREDICTION CONFIRMED IN
FULL, including its coarse-cell clause.  Freeze-both discovery (zero
probe cost by construction) across the remaining 14 cells; per-event
from discovered anchors:

  fine-grained: cars/visreg 0.373, flowers/visreg 0.803, flowers/dino
    0.737, flowers/lejepa 0.472, cars/dino 0.228, dtd/lejepa 0.307 --
    strong retention; flowers frozen per-event is the best flowers
    dataset-level detection on record.
  coarse: galaxy10/dino 0.015, galaxy10/lejepa 0.023, cifar10 0.096
    (vs 0.73-0.75 unfrozen!), cifar100 0.030 -- the space update does
    real work where novelty is genuinely outlying; galaxy10/visreg is
    the coarse exception (0.348, its unfrozen 0.32-0.45 retained).

- The exp-86 recipe is now REGIME-SCOPED, not universal: freeze the
  space on fine-grained/on-manifold cells (all the probe-record cells
  keep their records WITH detection), fine-tune on coarse/outlying
  cells (CIFAR, galaxy10) where discovery ft earns its probe cost.
- "Choose by consumer" survives but flips axis: the choice is now by
  NOVELTY GEOMETRY (on-manifold -> frozen anchors; outlying -> ft),
  not by readout.

Exp 83 (`83_variance_reduced_nplm.py`, 2026-08-21; IMPROVEMENT_TESTS
#83): variance-reduced NPLM -- FALSIFIED, and the watched falsifier
fired too.  C100 32-D, 5 paired seeds (refs = exp-81 archived):

  arm             probe mean+-sd    perevt  |resid|
  bil+clamp(ref)  0.8914+-0.0320    0.042   1.77
  bil-norm        0.7808+-0.0401    0.024   0.11
  dist-bias       0.7761+-0.0249    0.006   0.12
  dist-bias-norm  0.8187+-0.0175    0.008   0.03

- No variant cuts sd >=2x at equal-or-better mean: bil-norm RAISES sd
  and drops the mean -0.11; dist-bias-norm halves sd (0.018) but costs
  -0.072 mean and kills per-event (0.008 vs 0.042) -- the watched
  falsifier (variance drops WITH the absolute scale).
- The deeper negative, paired with exp 82: the fixes genuinely ACHIEVE
  calibration (dist-bias-norm |E_ref[e^g]-1| = 0.03 vs the clamp arm's
  1.77) and it buys NOTHING -- neither probe, nor variance, nor
  per-event.  Calibration-in-the-residual-sense is neither the variance
  problem (exp 82) nor the performance lever (exp 83).  The bilinear
  clamp's "miscalibration" is where its probe performance LIVES.
- bil+clamp stands as the best label-free NPLM arm; the exp-81 rule
  (accept the variance or use dist-sup) is the final word on this line.

Exp 84 (`84_two_stage_strict.sh`, 2026-08-21; IMPROVEMENT_TESTS #84):
the two-stage recipe under the strict open-world protocol -- PREDICTION
CONFIRMED: the gains shrink slightly but SURVIVE; reading #8 was not
contamination.  Seen-only nplm-sup-ft trunks (exp-70 protocol) + exp-51
8-arm head suites; best rows per base:

  base    two-stage supcon_sigreg      caveated #8      exp-71 champ
  dino    0.787 / mahaT 0.746          --               0.816
  lejepa  0.796 / eucl .817 / mahaT .802   0.812/0.638/0.800   0.848
  visreg  0.805 / mahaT 0.773; nplm_sup_dist mahaT 0.814   (mahaT 0.812)  0.866

- The strict numbers land within 0.007-0.016 of the caveated ones
  (lejepa mahaT and visreg nplm_sup_dist mahaT fully survive at
  0.80-0.81): holdout exclusion barely moves the two-stage result, so
  reading #8 graduates from caveat to citable.
- As predicted, two-stage lands BETWEEN the exp-70 parents and the
  exp-71 residual champions on every base (0.787-0.805 vs champions
  0.816-0.866): a real recipe, but the residual construction remains
  strictly better on aircraft.
- The caveat can be struck from AIRCRAFT_MASTER_TABLE; cite the strict
  numbers above.

Exp 85 (`85_iterated_residuals.py`, 2026-08-21; IMPROVEMENT_TESTS #85):
iterated residuals -- SPLIT VERDICT along the campaign's regime line,
and TWO NEW RECORDS.  3 paired seeds, width-matched control (both
concats 300-D):

  cell           2way            3way [p;r1;r2]     wide [p;rW200]
  cars/visreg    0.833+-0.017    0.881+-0.008       0.884+-0.003
  flowers/dino   0.833+-0.040    0.910+-0.012       0.859+-0.004

- CARS: the falsifier fires -- the width-matched single residual
  matches/beats the 3-way (+0.051 vs +0.048).  There the residual gain
  is a CAPACITY effect.  Either construction sets a new cars record:
  0.884+-0.003 (was 0.855 single-seed / 0.833+-0.017 honest).
- FLOWERS: the prediction holds emphatically -- 3-way +0.077 vs wide
  +0.026, and the effect is LARGEST on the weakest parent seed (+0.123
  on s1): the second residual is also a seed-variance stabilizer
  (2way sd 0.040 -> 3way 0.012).  New flowers record 0.910+-0.012,
  beating the discovery-assisted 0.887+-0.019 WITHOUT discovery.
- The decomposition-vs-capacity split follows the SAME on-manifold vs
  off-manifold line as exps 78/92/101: on-manifold novelty (flowers)
  rewards a second decomposition step; off-manifold (cars) just wants
  residual capacity.
- Detection trade: the wide control keeps/raises per-event (0.35-0.41
  on flowers) while the 3-way trades a little of it for probe.

Exp 109 (`109_c100_density_pool.py`, 2026-08-22; IMPROVEMENT_TESTS #109):
density-ratio pooling on CIFAR-100 -- PREDICTION HOLDS, the geometry
block is BROKEN.  Exp-92b sparker-frozen recipe (freeze the exp-89
concat space, SparKer density-ratio pool + centre proposals, anchors
only) on the exp-89 cached spaces, seed 0:

  cfg          sparker r1   sparker r2   exp-89 distance r1
  h1:q0.95     0.030        0.034        0.000
  h1:q0.99     0.029        0.020        0.000
  h5:q0.95     0.148        0.169        0.015
  h5:q0.99     0.198        0.234        0.002
  h10:q0.95    0.232        0.265        0.036
  h10:q0.99    0.358        0.418        0.002

- Round-1 purity clears the 0.15 gate at h5 and h10 (best 0.358 at
  h10:q0.99, vs the distance grid's all-time ceiling 0.121), and round
  2 RISES (0.418) instead of collapsing -- exactly the exp-92b
  signature transplanted to the one fully-blocked dataset.
- The strictest quantile is now the BEST setting (q0.99 > q0.95 at
  h5/h10) -- opposite to the distance pool, where strict tails were
  pure background: the f-ratio tail is owned by novelty, the distance
  tail by outliers.  Exp 89's verdict is hereby SCOPED: C100 is
  geometry-blocked FOR DISTANCE POOLING only.
- h1 (single class, 500 images) stays at 0.03: with the class rate
  ~1% no pooling statistic reaches the gate -- a rate floor, not
  geometry.  Margin/mean-anchor at h10:q0.99: 0.708/0.771.
- Recipe addendum for the master line: coarse cells with MULTI-CLASS
  novelty now also take the frozen density-ratio pool; the classic
  unfrozen loop remains only for the single-holdout regime.

Exp 106 (`106_panel_control.py`, 2026-08-23; IMPROVEMENT_TESTS #106):
panel sample-size control -- PREDICTION HOLDS; the width headline
survives its own audit, the correlation columns do not.

(a) CIFAR subsampled to 10/20/40 samples/class (the transfer-cell
    regime): rms is STABLE within +-0.05 of full-n on every space
    (e.g. c10 resnplm 0.11@n10 vs 0.13@full; c100 res 0.65 vs 0.70) --
    the width measurement is first-moment-ish as predicted.  r_llr
    degrades sharply at n10 on the quantitative cells (c100 supcon
    0.28->0.04, resnplm-cat 0.38->0.23, c10 resnplm 0.82->0.59).
(b) PCA 16/32-D projections of the transfer heads (estimable
    covariance, same space): r_llr barely moves (examples 0.124->
    0.133-0.140, 0.446->0.438-0.456, 0.536->0.533) -- the low transfer
    r_llr is NOT a dimension artifact.  (rms under PCA is not
    comparable: top-PC variance concentration inflates it.)
(c) diag/iso reference bounds: r_llr rises to 0.73-0.91 where the
    full-covariance reference gives 0.12-0.45 -- much of the apparent
    correlation failure is covariance UNESTIMABILITY, bracketed
    between the full (pessimistic) and iso (optimistic) references.
    ece is invariant to cov_mode as it must be (consistency check).

- Verdict: exp 104's rms/width claim stands on ALL cells including the
  shrinkage-dominated transfer ones; its r_llr/ece columns on transfer
  cells should be cited only as [full, iso] reference bounds.

Exp 107 (`107_composition_vs_lid.py`, 2026-08-23; IMPROVEMENT_TESTS
#107): neighbourhood composition vs LID -- FALSIFIER DOES NOT FIRE,
but LID is NOT demoted.  17 cells (champion concats + CIFAR records):

  score    mean AUC   wins       notes
  lid20    0.694      3/17
  comp     0.680      2/17       ties lid within 0.02 on flowers
  knnd     0.679      5/17       aircraft's best (0.75-0.81)
  eucl     0.669      0/17
  ens      0.712      0/17       comp+knnd rank ensemble
  ens-lid  0.715      7/17       lid+eucl (exp-78) -- still the best slot

- On-manifold, as predicted: flowers comp within 0.016-0.033 of lid;
  dtd comp BEATS lid on 2/3 bases; cifar10 comp 0.865 vs lid 0.672
  (biggest composition win anywhere).
- The falsifier (comp trails >0.05 on flowers) does not fire.  But two
  results keep LID in the battery: (i) aircraft -- comp is
  ANTI-correlated (0.42-0.48; novel variants sit in high-purity
  same-manufacturer neighbourhoods, so counting says "member"), while
  lid stays >=0.52 and knnd carries the cell; (ii) the exp-78
  holdout-rotation control on frozen flowers keeps a real lid edge
  (0.875+-0.029 vs comp 0.818+-0.038; cars parity 0.56).
- Verdict: composition is a free extra battery slot and the best
  simple statistic on texture-like/coarse cells, but the radial-ratio
  information is doing real work exactly where the on-manifold story
  lives.  Battery: ADD comp, KEEP lid; ens-lid remains the default
  ensemble.

Exp 108 (`108_regime_predictor2.py`, 2026-08-23; IMPROVEMENT_TESTS
#108): on-manifold predictor, second attempt -- PREDICTION MET by the
mechanism feature.  17 cells, response = LID-eucl AUC gap, label-free
features from the exp-103 mechanism:

  feature    Spearman   best sign acc      (base rate 11/17;
  comp_gap   -0.16      11/17               exp-90 generic best 12/17)
  rr_disp    +0.37      15/17   <-- meets the pre-registered >=15/17
  ratio      -0.14      12/17

- The winning feature is exactly the mechanism quantity: the
  dispersion of the within-modal-class mean log radial ratio (what LID
  inverts).  The composition gap alone -- despite being exp-103's
  loudest AUC -- carries NO regime signal (base rate), consistent with
  exp 107: composition detects, radial ratios REGIME-select.
- Caveats stated up front: best-threshold accuracy is in-sample on the
  same 17 points as exp 90, and Spearman 0.37 says the relation is
  threshold-like, not monotone.  Fit for the paper as "the regime is
  label-free identifiable"; NOT fit as a plug-in predictor without an
  18th cell.
- The pre-committed stopping rule does not trigger; the line closes
  SUCCESSFULLY and no third attempt is licensed or needed.

Exp 111 (`111_child_deployment.py`, 2026-08-23; IMPROVEMENT_TESTS
#111): child-only deployment -- PREDICTION HOLDS; the split is
deployable.  Three qualifying cells (exp-102 clean-reach + legible):

  cell             perevt ch/cat   f95 ch/cat     top1-overlap par/ch
  cars:visreg      0.256 / 0.263   0.049 / 0.063  0.734 / 0.615
  aircraft:visreg  0.411 / 0.363   0.063 / 0.072  0.818 / 0.737
  cars:dino        0.143 / 0.148   0.063 / 0.079  0.852 / 0.689

- Detection: child-only matches the concat within 0.005-0.007 on the
  cars cells and STRICTLY BEATS it on aircraft:visreg (+0.048
  per-event), with better f95 reach on all three -- at HALF the
  embedding width (100 vs 200-D per event; the parent is only
  embedded for the ~5% flagged).
- Explanation (the falsifier readout): parent-only top-5 explanations
  overlap the concat's at 0.73-0.85 top-1 / 0.73-0.81 Jaccard, and the
  parent is ALWAYS closer to the concat than the child is (0.62-0.74)
  -- removing the parent from the SCORED space does not change what
  the analyst is told.  Falsifier does not fire.
- Protocol limit found: absolute superclass agreement is nearly
  unmeasurable under alphabetical holdouts (n_scorable 0-15 of 61-137
  flagged; cars holdouts 186-195 remove entire makes from the seen
  set).  The exp-102 agree1~0.44 numbers measure seen-centroid
  neighbour structure, not novel-event explanations.  A rotated
  holdout draw is the fix if absolute agreement is ever needed.
- Deployment recipe (citable): embed the res-nplm child for every
  event, flag at the 0.95 background quantile, embed the parent only
  for flagged events and explain by its nearest seen centroids.

Exp 110 (`110_c100_ceiling.py`, 2026-08-23; IMPROVEMENT_TESTS #110):
why the softmax control broke the C100 mahaT ceiling -- CAUSE
IDENTIFIED: a temperature x marginal INTERACTION, not dimension, not
width.  First finding: exp-88's "softmax" spec never set tau, so its
record arm accidentally ran at the HybridContrastiveLoss default
tau=1.0, not the canonical supcon temperature 0.1.  Factorial, C100,
3 paired seeds:

  arm (tau, sigreg)       mahaT @32d  @64d   @100d  @128d  @200d
  supcon (0.1, off)       0.356      0.342  0.344  0.363  0.368
  supcon_sigreg (0.1, on) 0.404      0.367  0.382  0.399  0.354
  supcon_t1 (1.0, off)    0.410      0.395  0.425  0.420  0.395
  supcon_sigreg_t1 (1.0,on) 0.539    0.557  0.562  0.572  0.492

- Only tau=1.0 AND the SIGReg marginal together break the 0.47-0.49
  ceiling; each alone stays under it.  The effect is fully present at
  32-D (0.539 +- 0.030) -- dimension is incidental (exp 88 only looked
  >=100-D); mild peak at 128-D, decline at 200-D.
- probe holds 0.90-0.92 (vs 0.94 at tau 0.1): the both-currencies
  record improves to mahaT 0.572+-0.031 @ 128-D, probe 0.910 -- the
  new best single-loss C100 space, now DESIGNED rather than
  accidental.  Also the only arms with nonzero per-event (0.04-0.06).
- The rms-tracking clause of the prediction is FALSIFIED informatively:
  the winner sits at rms~0.50 while supcon_t1 reaches rms~1.0 with low
  mahaT -- consistent with exps 105/83: unit width is neither necessary
  nor sufficient; what tau=1.0 buys under the marginal is a looser
  interaction that lets SIGReg shape the full covariance (slope 10-18,
  i.e. still anisotropic, yet mahaT-best).
- Nomenclature correction for exp-88/QUESTIONS: the C100
  ceiling-breaker is `supcon_sigreg @ tau=1.0`.

Exp 105 (`105_width_penalty.py`, 2026-08-23; IMPROVEMENT_TESTS #105):
direct class-conditional width penalty -- THE WATCHED FALSIFIER FIRES
(fifth confirmation of the meta-lesson), with two informative
exceptions.  Stage 1 scanned lambda_w {0, 0.1, 1, 10} at seed 0
(lambda=1 the operating point: rms~1 with no probe damage; lambda=10
reaches slope~1 but costs probe up to -0.04); stage 2 = 5 paired
seeds on lambda 0 vs 1, C10-32D + C100-100D x supcon/supcon_sigreg/
res-cat:

  the identity IS repairable: rms 0.37-0.65 -> 0.80-1.04 everywhere.
  what was promised does NOT follow: slope stops at 2.3-4.6 (the
  residual departure is SHAPE/anisotropy -- second falsifier also
  active); r_llr flat-to-down (c10 res-cat 0.766->0.695; c100
  supcon_sigreg 0.611->0.197); ece WORSENS nearly everywhere (c100
  supcon 0.138->0.312).
- Unlike exp 83, nothing collapses: probe moves <=0.004 on supcon/
  res-cat (supcon_sigreg pays 0.008-0.017) and per-event is
  unchanged-to-up.  The width term is SAFE at lambda=1 -- merely
  decorative for calibration.  "Repairing an identity is not the same
  as improving a space" now stands at five confirmations (82, 83, 96,
  98, 105); the distance-as-logL identity is satisfiable on demand
  and is not what makes a space work.  Publish as the significant
  negative it is.
- Exception 1: c10 res-cat improves as a SPACE (probe 0.9551->0.9593,
  sep 5.88->10.93, mahaT 0.537->0.601) -- the width term acts as a
  useful regularizer on the record concat construction.
- Exception 2 (headline): c100 supcon_sigreg + width jumps mahaT
  0.389->0.574+-0.031 with per-event 0->0.066 -- numerically the SAME
  ceiling break exp 110 traces to tau=1.0, reached by an independent
  intervention.  Both routes weaken/reshape the supcon interaction
  under the SIGReg marginal; the C100 ceiling break is a basin, not a
  knife-edge.  (c10 shows the opposite sign, 0.549->0.346: the basin
  is a C100/many-class phenomenon.)
- f95 was not run on these arms (spaces not all checkpointed; the
  verdict rests on probe/per-event/panel, which suffice for the
  stated falsifiers).

Exp 119 (`119_rrdisp_holdout_cell.py` + `119b_food101_banks.py`,
2026-08-24; IMPROVEMENT_TESTS #119): rr_disp out of sample -- FAILS AT
THE FROZEN THRESHOLD; per the pre-registered stopping rule the regime
line CLOSES.  Order of operations was enforced and is in git history:
threshold frozen on the original 17 cells (0.06052, sign +1, in-sample
14/17 under exp-119's self-consistent pipeline), predictions committed
(commit 2131998) BEFORE any scoring.

  held-out cell     rr_disp   predicted   actual     gap
  food101:dino      0.02597   distance    LID       +0.030   MISS
  food101:lejepa    0.03740   distance    distance  -0.018   OK
  food101:visreg    0.04198   distance    distance  -0.003   OK

- 2/3 correct; the bar (>=4/5, i.e. >=80%) is not met.  Verdict per
  exp 108's own stopping rule: rr_disp is downgraded to a DESCRIPTIVE
  correlation; the paper may say "the regime is label-free
  identifiable in-sample" and no more; no further attempts.
- Deviations from spec, stated: only 3 held-out cells were buildable
  (Food-101 x 3 bases on frozen ViT banks; CUB is not packaged, and a
  new base or ft-head spaces would not be evaluation-only), so the
  effective bar was 3/3.
- Context, not exculpation: every held-out gap is |gap| <= 0.03 --
  the held-out dataset happens to sit ON the regime boundary, where
  sign prediction is hardest and least consequential (either score
  works equally well there).  A future cell with a large true gap
  would be more diagnostic, but the pre-registered verdict stands.

Exp 116 (`116_sparker_m_matched.py`, 2026-08-24; IMPROVEMENT_TESTS
#116): SparKer M matched to intrinsic dimension, applied to the reach
table -- PARTIAL; the exp-97 rule does not transfer monotonically.
15 transfer champions, TwoNN ID -> M in {4, 16, 64}, vs exp-100 M=16:

  first-time crossings:  galaxy10:dino (>0.1 -> 0.075 at M=4),
                         dtd:lejepa   (>0.1 -> 0.079 at M=64)
  regressions at M=4:    cars:visreg 0.063 -> 0.079,
                         cars:lejepa 0.099* -> >0.1
  unchanged/noise:       aircraft (still mostly >0.1), flowers (all
                         >0.1 -- preserved), galaxy10:lejepa/visreg
                         identical; same-M reruns (cars:dino 0.079 ->
                         0.091*, dtd:dino 0.087* -> 0.072) put the
                         measurement noise at ~0.012-0.015 (n_null 100
                         vs 200), so differences below that are not
                         meaningful.

- The falsifier does NOT fire: the dataset ordering is preserved
  (flowers hardest everywhere, galaxy10/dtd easiest), so exp 100's
  sensitivity ranking stands.
- But the prediction only half-holds: matched M helps exactly at the
  extremes (one very-high-ID coarse cell, two very-low-ID texture
  cells) and HURTS the mid-ID fine-grained cars cells.  The exp-97
  bump-recovery rule is not a champion-space recipe.
- Recommendation: keep M=16 as the default reach protocol; cite
  matched-M as a sensitivity band, and note the two extra crossings
  raise the citable count from 8/17 to 10/17 under best-of-two.

Exp 118 (`118_holdout_audit.py`, 2026-08-24; IMPROVEMENT_TESTS #118):
holdout-selection audit -- PREDICTION HOLDS on the probe side, with a
variance refinement.  C100, 5 random 1-class draws x 3 seeds vs the
archived last-class holdout:

  draw       probe mean+-sd(seeds)   mahaT
  archived   0.9567+-0.0104          0.500
  random1-5  0.9302-0.9837           0.391-0.569

- The archived alphabetical draw sits WITHIN the random-draw
  distribution: the falsifier (headline probe confounded by
  alphabetical holdouts) does not fire.
- Refinement worth republishing: draw-to-draw spread (sd ~0.019 on
  probe, 0.18 range relative on mahaT) EXCEEDS the seed spread
  (0.001-0.010) -- WHICH class is held out matters more than the
  seed.  Any cross-holdout generalization claim needs draw-resampled
  intervals, not just seed-resampled ones (feeds the exp-117 MDE
  accounting).  Calibration (mahaT) figures are substantially
  draw-dependent and should be quoted with draw ranges.
- Scope limit: the script retrains per draw and supports CIFAR only;
  the transfer-cell superclass-agreement half of the original
  motivation (exp-111's scorable-class collapse) remains covered by
  exp 111's protocol note rather than by new random-draw numbers
  (retraining transfer heads per draw is not evaluation-only).
