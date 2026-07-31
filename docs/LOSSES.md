# SuperSig losses — reference for testing sessions

Canonical implementations: `supersig/losses.py` (loss functions) and
`supersig/train.py` (training loops).  Settled per-dataset configs:
`supersig/recipes.py` (`RECIPES`: cifar10 = 16-D default, sigreg_weight 20,
n_slices 256; cifar100 = 32-D, weight 1, n_slices 64; arch resnet20
CIFAR-pretrained for both; the NPLM program of exps 50+ standardized on
32-D for both datasets).  All losses operate on the raw backbone output
`z = backbone(x)`; only the softmax-contrastive losses L2-normalize
internally.

## Core primitives

- **`sigreg_loss(z, n_slices=64)`** — "SIGReg": sliced 1-D Wasserstein-2 to
  N(0, I).  Project z onto random unit directions, sort, compare against
  standard-normal quantiles, mean squared difference.  This is the repo's
  member of the VISReg/LeJEPA family (arXiv 2511.08544).
- **`classwise_sigreg_loss(z, y, means, n_slices, class_sigma=None)`** —
  SIGReg applied per class to `z - means[c]`: pulls class c toward
  N(means[c], I).  Skips classes with < `MIN_PER_CLASS=8` samples in the
  batch, so it REQUIRES a balanced loader
  (`cifar_two_view_balanced_loader`, 24/class/view); with the plain
  256-batch loader on CIFAR-100 it silently returns ~0.
- **`make_anchors(scale, emb_dim, n_classes)`** — fixed anchors, norm
  `scale` (= `pair_dist/sqrt(2)` in recipes).  Orthogonal basis vectors iff
  `emb_dim >= n_classes`, otherwise deterministic random unit directions.
  Realizability matters: 100 anchors in 32-D or 64-D are NOT reachable
  (exp 53/64d: centroids stall ~3.5 from anchors), 10 in 32-D are.
- **`supcon_loss(feats, labels, temp=0.1)`** — standard SupCon (Khosla
  2020) on L2-normalized features.  With instance ids as labels it is
  NT-Xent/SimCLR (temp 0.5).

## Named training objectives (`supersig/train.py`)

| loop | objective | labels | loader |
|---|---|---|---|
| `train_supervised` | CE | yes | plain |
| `train_sigreg_ssl` ("lejepa") | MSE invariance + lam*global SIGReg | no | two-view |
| `train_supcon` | SupCon, temp 0.1 | yes | two-view labeled |
| `train_simclr` | NT-Xent, temp 0.5 | no | two-view |
| `train_simclr_sigreg` ("ss" hybrid, exp 34e) | NT-Xent on normalized z + lam*SIGReg on raw z | no | two-view |
| `train_supcon_sigreg` (exp 34g) | SupCon + lam*SIGReg | yes | two-view labeled |
| `train_sigreg_hybrid` | proto CE (softmax of -0.5 d^2 to means) + repulsion + classwise SIGReg; means learnable in mode="repulse" | yes | balanced |
| `train_sigreg_residual_ssl` / `train_simclr_residual` / `train_supcon_sigreg_residual` | residual-half objectives on `z - means[y]` | mixed | balanced / two-view |

`train_sigreg_hybrid` is the supervised-SIGReg workhorse: it trains
`supervised_embedding` (recipes.py) and the discovery fine-tune
(`discovery.run_discovery`).

## HybridContrastiveLoss — the design cube (exps 34f-58)

`losses.HybridContrastiveLoss(positives, critic, estimator, marginal, tau,
lam, n_slices)`; objective `L = L_interaction(g) + lam * L_marginal(z)`.
Forward: `loss_fn(z, labels, means=None)` with z = `(2N, D)` raw embeddings
(two views stacked), labels = instance ids or class labels `cat([y, y])`.

| knob | values | meaning |
|---|---|---|
| positives | instance / supervised | what counts as a positive pair |
| critic | cosine / bilinear / distance | g = <z^,z^>/tau, <z,z>/tau, -0.5|z-z'|^2/tau |
| estimator | softmax / nplm | see below |
| marginal | none / sigreg / classwise_sigreg | Gaussian constraint |

**softmax estimator** = NT-Xent/SupCon: self-normalized per-row; recovers
PMI only up to an unknown per-row constant.

**nplm estimator** = un-normalized maximum likelihood
`L = E_ref[e^g - 1] - E_pos[g]`, minimizer g* = the ABSOLUTE log density
ratio log p(x,x')/p(x)p(x') with calibration `E_ref[e^{g*}] = 1`
(theory note: `plots/nplm_contrastive_note/contrastive_logprob_note.tex`;
calibration verified in `34f_nplm_calibration.py`, r > 0.9 vs analytic
PMI).  Reference set excludes positives and the diagonal.  Stabilized by
clamping the exponent at 30 (NOT row-max subtraction, which would destroy
the calibration).  Consequence (exp 52): at init the interaction is ~1e10
vs ~1e-2 for lam*sigreg, so **lam is inert for the first ~2 epochs** and
only shapes the endgame; **tau=1 is a sharp optimum** for bilinear on
CIFAR-100 (tau=0.5 and 2 both cost 0.06-0.10 probe).

**Caveat**: forward takes ONE label tensor driving both positives and the
classwise marginal.  For instance positives + classwise marginal, call
`loss_fn.interaction(z, inst_labels)` and add
`lam * classwise_sigreg_loss(z, cls_labels, means)` manually (pattern:
`experiments/53_nplm_classwise.py::train_nplm_classwise`).

## Named arms used across exps 50-58

| arm | positives / critic / estimator / marginal | labels | driver |
|---|---|---|---|
| simclr | instance / cosine / softmax / none | no | `train_simclr` |
| lejepa | MSE inv + global sigreg | no | `train_sigreg_ssl` |
| simclr_sigreg ("ss") | instance / cosine / softmax / sigreg, tau 0.5 | no | exp34h `train_hybrid` |
| supcon | supervised / cosine / softmax / none | yes | `train_supcon` |
| supcon_sigreg | supervised / cosine / softmax / sigreg, tau 0.1 | yes | exp34h |
| nplm_bilinear | instance / bilinear / nplm / sigreg, tau 1 | no | exp34h |
| nplm_distance | instance / distance / nplm / sigreg, tau 1 | no | exp34h |
| nplm_sup_dist | supervised / distance / nplm / sigreg, tau 1 | yes | exp34h |
| nplm_bil_cw | instance positives / bilinear / nplm + classwise sigreg | marginal only | exp53 |
| nplm_bil_sup_cw | supervised / bilinear / nplm + classwise sigreg | yes | exp53 |
| nplm_dist_sup_cw | supervised / distance / nplm + classwise sigreg | yes | exp53 |

Standard training: 20 epochs, Adam lr 1e-3, batch 256 two-view (balanced
24/class for classwise arms), lam=1, per-arm seed `args.seed + 20 + i`.

## Discovery fine-tune objectives

- **proto/repulse** (settled; `discovery.run_discovery`):
  `train_sigreg_hybrid(mode="repulse", disc="proto")` on seen labels +
  pseudo-labels, anchors co-learned.  NOTE: it rescales compact NPLM
  trunks ~4x (exp 57) — rescore SparKer with annealed sigma.
- **conf-masked** (exp 56 `run_discovery_conf`): exp-36b JEPAMatch mask —
  keep a pooled event iff proto-posterior at its assigned discovered
  anchor >= 0.5.  At compact NPLM scale with ~100 anchors the posterior is
  diluted and round 1 keeps nothing (do-no-harm gate).
- **NPLM-ft** (exp 58 `run_discovery_nplm_ft`): supervised-NPLM + global
  sigreg on single-view balanced pseudo-labeled batches; anchors
  recomputed as centroids each round.
