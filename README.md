# SuperSig

Learning structured embeddings on MNIST, CIFAR-10, and CIFAR-100 with **SIGReg**
(Sketched Isotropic Gaussian Regularization) and **supervised contrastive learning
(SupCon / supervised SimCLR)**, then evaluating them with frozen linear probes, ROC
curves, and corner plots of the latent space.

The unifying idea is a *distributional* prior on the embedding: SIGReg pushes the
learned features toward a (class-conditional) Gaussian — an isotropic-Gaussian /
negative-log-likelihood style regularizer — rather than relying only on a
discriminative loss. Every embedding is trained, then **frozen**, and a single linear
layer is trained on top with categorical cross-entropy.

## Method summary

| Embedding | Idea |
|-----------|------|
| Supervised baseline | CNN trained end-to-end with cross-entropy (reference) |
| SIGReg (SSL) | invariance between two augmented views + a global isotropic-Gaussian SIGReg term (no labels) |
| Class-conditional SIGReg | SIGReg applied per class, pulling each digit to `N(mean_c, I)` |
| &nbsp;&nbsp;· fixed anchors | class means fixed at orthogonal anchors |
| &nbsp;&nbsp;· learnable means | means trained, kept apart by a **hinge separation** term |
| &nbsp;&nbsp;· repulsive means | means trained, kept apart by an **inverse-square repulsion** + shrinkage |
| SupCon | supervised contrastive loss on two augmented views |

Two evaluation protocols:
- **Closed-set:** embedding on all digits → 10-way linear probe → one-vs-rest ROC.
- **Hold-out-4:** embedding trained *without* digit 4 → frozen → binary "4 vs rest"
  linear probe. Tests whether an unseen class still lands in its own latent region.

## Layout

```
supersig/          importable library
  config.py        paths, constants, device
  models.py        ConvBackbone, SupervisedCNN, CIFARBackbone, CIFARResNetBackbone
  losses.py        sigreg, class-conditional sigreg, separation/repulsion, supcon
  data.py          plain / two-view / hold-out / class-balanced loaders
                   (MNIST, CIFAR-10, CIFAR-100)
  train.py         training loops (incl. sigreg hybrids) + eval collectors
  plotting.py      ROC and corner-plot helpers
experiments/       runnable scripts (write figures to plots/)
  01_supervised_baseline.py
  02_sigreg_ssl.py
  03_sigreg_classwise.py   --mode fixed|learnmeans|repulse
  04_holdout4.py           --mode learnmeans|repulse|both
  05_supcon.py
  06_anchor_scan.py
  07_compare8d.py
  08_cifar_compare.py      CIFAR-10, 16-dim, from-scratch CNN
  09_cifar_resnet32.py     CIFAR-10, 32-dim, CIFAR-pretrained ResNet
  10_cifar_ablations.py    no-hinge SIGReg, no-augmentation SupCon
  11_cifar_repulse.py      repulsive floating means, 3-sigma seed
  12_cifar_hybrid.py       SIGReg + CE / SupCon discriminative term
  13_cifar_noce.py         head-free (proto) and CE-free (hinge) variants
  14_cifar100.py           CIFAR-100, --emb-dim 32|100|200
  15_gaussianity.py        per-class Gaussianity metric, CIFAR-10
  16_gaussianity100.py     Gaussianity of the CIFAR-100 table configs
  17_multi_holdout.py      hold out 2-3 classes at once
plots/             all generated figures
```

## Usage

```bash
pip install -r requirements.txt

# from the repo root; add --quick for a fast smoke test
python experiments/01_supervised_baseline.py
python experiments/02_sigreg_ssl.py
python experiments/03_sigreg_classwise.py --mode repulse
python experiments/04_holdout4.py --mode both
python experiments/05_supcon.py

# CIFAR series (GPU recommended; models are moved to CUDA when available)
python experiments/09_cifar_resnet32.py            # 32-dim, pretrained ResNet
python experiments/11_cifar_repulse.py             # repulsive means, 3-sigma seed
python experiments/13_cifar_noce.py                # proto / hinge variants
python experiments/14_cifar100.py --emb-dim 100    # CIFAR-100, wide latent
```

## Results (full runs, MNIST test set)

Closed-set, 10-way probe:

| Model | Probe acc | ROC micro-AUC |
|-------|-----------|---------------|
| Supervised CNN (end-to-end) | 0.990 | 0.9999 |
| SIGReg (SSL) | 0.961 | 0.9991 |
| Class SIGReg, fixed anchors | 0.979 | 0.9996 |
| Class SIGReg, learnable means | 0.976 | 0.9995 |
| Class SIGReg, repulsive means | 0.990 | 0.9999 |
| SupCon (supervised SimCLR) | 0.996 | 0.9999 |

Hold-out-4 detection (digit 4 unseen during embedding), 4-vs-rest AUC:

| Embedding | 4-vs-rest AUC |
|-----------|---------------|
| SupCon (supervised SimCLR) | 0.963 |
| Class SIGReg, learnable means | 0.953 |
| Class SIGReg, repulsive means | 0.887 |

Aggressive separation (repulsion) is best for closed-set accuracy but worse at
placing an *unseen* class in its own region — a closed-set vs open-set trade-off.
SupCon leads on both protocols here.

## CIFAR series (experiments 09–14)

Setup: CIFAR-pretrained ResNet-20 backbone (torch.hub
`chenyaofo/pytorch-cifar-models`) with a projection head to the latent, fine-tuned
end-to-end (10 SSL epochs + 5 probe epochs, seed 0).  Two protocols as before:
**inclusive** (all classes → k-way frozen linear probe → micro-AUC) and **holdout**
(one class removed from embedding training → frozen → binary vs-rest probe → AUC;
class 4 = "deer" on CIFAR-10, "beaver" on CIFAR-100).  Caveat: the pretrained
weights saw the held-out class during supervised pretraining (`--pretrain` can
select a label-disjoint init).

### CIFAR-10, 32-dim latent

| Embedding | Inclusive micro-AUC | Deer-holdout AUC |
|-----------|--------------------|------------------|
| SIGReg, hinge-separated means, scale-5 seed (09) | 0.9645 | 0.8407 |
| SIGReg, free means, scale-5 seed (10) | 0.9655 | 0.8324 |
| SIGReg, free means, 3σ seed (11) | 0.7689 | 0.7136 |
| SIGReg, repulsive means, 3σ seed (11) | 0.9822 | 0.8509 |
| SIGReg repulse + linear-head CE (12) | 0.9917 | 0.9235 |
| SIGReg repulse + SupCon term (12) | 0.9894 | 0.8855 |
| **SIGReg repulse + proto (13)** | **0.9920** | **0.9110** |
| SIGReg repulse + wrong-mean hinge (13) | 0.9794 | 0.8743 |
| SupCon, no augmentation (10) | 0.9904 | 0.9143 |
| SupCon, two-view augmentation (09) | 0.9920 | 0.9290 |

What the ablations established, in order:

1. **The hinge-separation term is droppable at this training length** (09 vs 10):
   from a scale-5 seed the means only drift to ~6σ apart in 10 epochs.  But
   collapse is a real attractor — with a tight 3σ seed and no geometry term the
   means shrink to 1.4σ and performance craters (11, "free").
2. **SupCon's edge is the loss, not the augmentations** (10): stripped to plain
   single views it loses almost nothing (0.9904/0.9143), still well ahead of plain
   SIGReg.  Classwise SIGReg never penalises a sample for sitting near a *wrong*
   mean.
3. **Inverse-square repulsion between learnable means, seeded 3σ apart,** lets the
   mean geometry adapt to the data (final spacing is non-uniform, min ≈ 8σ,
   mean ≈ 9.4σ) and beats every static geometry (11).
4. **Adding the model's own discriminative term closes the gap** (12–13): the
   Gaussian latent model classifies with logits −‖z−μ_c‖²/2, so cross-entropy on
   that posterior ("proto") adds zero parameters and ties augmented SupCon on the
   inclusive protocol *without any augmentation*.  A jointly-trained linear-head CE
   does marginally better on holdout; a purely geometric hinge captures only part
   of the gain (it stops supplying gradient once satisfied).

### CIFAR-100 (experiment 14) — latent width matters

| Embedding | Inclusive micro-AUC | Beaver-holdout AUC | Gaussianity |
|-----------|--------------------|--------------------|-------------|
| SIGReg+proto, 32-dim | 0.9670 | 0.6716 | — |
| SupCon (aug), 32-dim | 0.9825 | 0.7469 | — |
| SIGReg+proto, 100-dim, 3σ seed | 0.9822 | 0.9127 | — |
| SIGReg+proto, 100-dim, 5σ seed | 0.9833 | 0.9198 | 1.21 |
| SIGReg+proto, 100-dim, 5σ, repulsion ×3 | 0.9829 | 0.9339 | 1.26 |
| SIGReg+proto, 100-dim, 5σ, repulsion ×10 | 0.9821 | 0.9371 | 1.28 |
| **SIGReg+CE (linear head), 100-dim, 5σ** | **0.9828** | **0.9488** | **1.24** |
| SupCon (aug), 100-dim | 0.9853 | 0.9245 | 1.68 |
| SIGReg+proto, 200-dim | 0.9832 | 0.9306 | — |
| SupCon (aug), 200-dim | 0.9849 | 0.9414 | — |

The last column is the mean per-class Gaussianity ratio (experiment 16; 1 = as
Gaussian as a finite sample can look; test set has only 100 images/class so
ratios are compressed relative to the CIFAR-10 numbers — compare within the
column only).  Stronger repulsion monotonically costs Gaussianity (1.21 → 1.28);
the linear-head CE costs almost none (1.24) while gaining the most holdout AUC;
SupCon is the least Gaussian (1.68).  Backbones are checkpointed to
checkpoints/ by experiment 16 for metric reuse.

### Multi-class holdout (experiment 17)

Holding out a *set* of classes (100-dim, 5σ seed; combined = any unseen class vs
rest, plus per-class restricted AUCs from the same binary score):

| Held out | Method | Combined | Per-class |
|----------|--------|----------|-----------|
| beaver, rose | SIGReg+CE | **0.9078** | beaver 0.8646, rose 0.9511 |
| beaver, rose | SupCon (aug) | 0.8872 | beaver 0.8654, rose 0.9089 |
| beaver, rose, dolphin | SIGReg+CE | 0.8828 | beaver 0.8264, dolphin 0.9285, rose 0.8934 |
| beaver, rose, dolphin | SupCon (aug) | **0.8984** | beaver 0.8569, dolphin 0.9453, rose 0.8929 |

Scaling to larger unseen sets (k=10: every 10th class from index 4; k=20: every
5th — nested supersets, combined AUC):

| k held out | SIGReg+proto | SIGReg+CE | SupCon (aug) |
|-----------:|--------------|-----------|--------------|
| 1 (beaver) | 0.9198 | **0.9488** | 0.9245 |
| 2 | 0.8912 | **0.9078** | 0.8872 |
| 3 | 0.8715 | 0.8828 | **0.8984** |
| 10 | 0.7940 | 0.8152 | **0.8224** |
| 20 | 0.6938 | 0.7023 | **0.7423** |

Detection degrades steadily with k — each unseen class both crowds the vacant
latent space and makes the single binary probe's positive set more
heterogeneous.  SIGReg+CE wins for small unseen sets (k ≤ 2) but SupCon degrades
more gracefully and pulls ahead from k = 3, by 4 points at k = 20: SIGReg's
open-set advantage rests on explicit vacant structure around the class means,
which fills up as the unseen fraction of the label space grows.  SIGReg+proto
tracks CE in parallel ~1–3 points below it at every k — the crossover is a
property of the SIGReg framework, not of the CE head.

With an **augmentation layer in front** of the SIGReg embedding training
(`--augment`: the SupCon crop/flip/jitter stack on every training image;
probes/eval stay on plain images):

| k held out | proto plain → aug | CE plain → aug | SupCon (aug) |
|-----------:|-------------------|----------------|--------------|
| 1 | 0.9198 → **0.9359** | 0.9488 → 0.9251 | 0.9245 |
| 2 | 0.8912 → 0.8891 | 0.9078 → 0.8855 | 0.8872 |
| 3 | 0.8715 → 0.8589 | 0.8828 → 0.8707 | 0.8984 |
| 10 | 0.7940 → 0.8105 | 0.8152 → 0.8119 | 0.8224 |
| 20 | 0.6938 → 0.7093 | 0.7023 → **0.7234** | 0.7423 |

Augmentation helps SIGReg where the vacant-structure mechanism is weakest — at
large k both variants gain 1.5–2 points and close most of the gap to SupCon
(0.7234 vs 0.7423 at k = 20) — but *hurts* the CE hybrid at small k (−2.4 at
k = 1), where plain-image CE remains the best configuration in the study.
Within-class augmentation variance seems to act as a regularizer exactly when
many unseen classes crowd the latent, and as noise when one vacant region
suffices.  (Single-seed numbers; ±1 point differences are within noise.)

### Probe-free novelty from the Gaussian latent (experiment 18) — negative result

Hypothesis: the probed evaluation never uses SIGReg's structure, so grading it
"on its own exam" — novelty = low likelihood under every seen-class Gaussian,
no probe — should reveal the constraint's real value.  It does not:

| k | proto lik / typ | CE lik / typ | SupCon cos / typ | (best probed) |
|--:|-----------------|--------------|------------------|---------------|
| 1 | 0.38 / 0.62 | 0.54 / 0.45 | **0.69** / 0.50 | 0.9488 |
| 3 | 0.46 / 0.54 | 0.48 / 0.43 | **0.65** / 0.49 | 0.8984 |
| 10 | 0.56 / 0.44 | 0.50 / 0.49 | **0.66** / 0.51 | 0.8224 |
| 20 | 0.59 / 0.41 | 0.51 / 0.47 | **0.70** / 0.55 | 0.7423 |

Two failure modes, both instructive:
1. **Naive max-likelihood inverts** (0.38 at k=1): in 100 dims class members
   live on the √d ≈ 10σ typical shell while means sit 7–10σ apart, so novel
   points *between* clusters are closer to seen means than real members are —
   the Nalisnick-style "generative models assign higher likelihood to OOD"
   pathology, reproduced in this latent.
2. **The typicality correction (distance from the nearest shell) barely
   helps**: unseen classes do not land in vacant space — the backbone
   generalises them *onto the shells of related seen classes* (beaver embeds
   into otter's Gaussian).  Their displacement is directional, not radial, so
   an isotropic score is blind to it while a linear probe finds it easily —
   which is why the probed numbers are so much higher.

SupCon's angular nearest-centroid score is the only serviceable probe-free
signal (0.65–0.70 at every k; at k=20 it nearly matches its own probe).
Conclusion: the unit-covariance Gaussian prior does not by itself yield a
usable novelty density; capturing the directional displacement (e.g. empirical
per-class covariances / Mahalanobis scores) would be the next thing to try.

### Empirical Mahalanobis & the "true Mahalanobis space" test (experiment 19)

The design goal was a latent that *is* a Mahalanobis space — every class
N(μ_c, I), no empirical covariance needed.  Fitting the seen classes
empirically (Lee et al. 2018 style; tied and shrunken per-class covariances)
and diagnosing the actual within-class second moments:

| k | proto tied/per-class | CE tied/per-class | SupCon tied/per-class | (SupCon cosine) |
|--:|----------------------|-------------------|-----------------------|-----------------|
| 1 | 0.43 / 0.50 | 0.51 / 0.58 | 0.44 / 0.55 | 0.69 |
| 3 | 0.50 / 0.53 | 0.51 / 0.55 | 0.45 / 0.55 | 0.65 |
| 10 | 0.57 / 0.64 | 0.55 / 0.63 | 0.47 / 0.58 | 0.66 |
| 20 | 0.60 / **0.67** | 0.60 / **0.68** | 0.48 / 0.61 | 0.70 |

Findings:
1. **The latent is not a true Mahalanobis space.**  Pooled within-class
   covariance eigenvalues are ~0.001 / ~0.02 / 1–5 (min/median/max) against the
   1/1/1 ideal — the class clouds are radially collapsed pancakes, ~50× tighter
   than the unit target in the median direction, with a few stretched axes.
   Strikingly, SupCon (no Gaussian constraint at all) shows the *same*
   anisotropy: the sliced-Wasserstein term is too weak against the
   discriminative shrinkage pressure to control second moments in 100 dims
   with ~24 samples/class/batch.
2. Empirical Mahalanobis stays at chance for small k (the unseen class embeds
   *inside* the empirical distribution of a related seen class — separable by
   a labeled hyperplane, invisible to any density), but becomes the best
   probe-free score for SIGReg at large k (0.67–0.68 at k=20, within ~2 points
   of the probes), and there SIGReg beats SupCon's Mahalanobis.
3. Supervised probes dominate at small k because they are supervised — they
   see labeled examples of the unseen class; no density method gets that
   information.

To actually reach the no-empirical-estimate regime, the within-class second
moments need stronger enforcement than the current recipe provides: a larger
SIGReg weight relative to the discriminative term, more slices / larger
per-class batches, or explicit per-class whitening.

### Tuning by the eigenspectrum (experiment 20) — the trade-off is strictly bad

Sweeping the enforcement knobs (SIGReg weight w ∈ {1,5,20,100}, 64→256 slices,
24→48 samples/class) against the eigenspectrum target:

| w | eig median | k=1 probed (proto/CE) | k=1 mahal-pc | k=20 probed | k=20 mahal-pc |
|--:|-----------|------------------------|--------------|-------------|----------------|
| 1 | 0.014 | 0.9198 / 0.9488 | 0.50 / 0.58 | 0.694 / 0.702 | 0.67 / 0.68 |
| 10 | ~0.03 | 0.8561 / 0.8671 | 0.48 / 0.48 | 0.678 / 0.681 | 0.64 / 0.60 |
| 100 | ~0.03 | 0.6019 / 0.5777 | 0.51 / 0.46 | 0.603 / 0.585 | 0.52 / 0.52 |

The spectrum median saturates at ~0.03 (never approaching 1) regardless of
weight, while **both** metric families degrade monotonically — at w=100 the
probed AUC collapses ~35 points and even the Mahalanobis score falls to chance,
because a latent shaped mostly by the Gaussianization term stops encoding
class-discriminative structure at all.  The min eigenvalue stays pinned at
0.001 for every method *including SupCon*: 500 images of a CIFAR class do not
contain 100 independent directions of variability, so full-rank unit
within-class covariance at d=100 cannot be produced by any loss weight — the
class clouds are intrinsically low-rank.  w=1 dominates every measured cell.

Routes that could genuinely reach the self-calibrated ("no empirical
estimate") Mahalanobis space: size the latent to the classes' intrinsic
dimension (d ≈ 16–32, where full-rank unit covariance is attainable), or make
unit covariance structural (a per-class whitening layer) rather than penalised.  Per-class AUCs
(printed by experiment 17) span ~0.5–0.95 at k = 20: visually distinctive unseen
classes (cockroach, wardrobe, spider) stay easy; classes with in-distribution
lookalikes (fox, possum, cattle, tractor) approach chance.

100 classes need room: at 32 dims the means cannot be orthogonal, repulsion can
only push them to ~4σ minimum spacing, and both methods lose ~20 AUC points on
holdout.  At 100 dims the seeding is orthogonal again and SIGReg+proto returns to
near-parity with augmented SupCon.  Going further to 200 dims adds only ~2 more
holdout points for both methods — width saturates once it reaches the class count.
At 100 dims the minimum mean distance does *not* grow (semantically confusable
pairs stay ~4σ apart regardless; 200 dims stretches them to ~5.5σ); what width
mainly buys is average spacing and vacant directions for unseen classes.  Seeding
the means 5σ apart instead of 3σ (still 100-dim) preserves a wider final geometry
(min ~6.3σ) and adds ~0.7 holdout points — the repulsion never fully rescues pairs
that start close, so a generous seed helps when the dimension allows one.

Pushing further at 100-dim/5σ: stronger repulsion (×3/×10) widens the *average*
mean spacing to ~10σ and adds 1.4–1.7 holdout points, but leaves the hard-pair
minimum (~6σ) untouched and degrades the SIGReg/proto losses — mild gains,
saturating.  Swapping the proto term for a jointly-trained **linear-head CE**
(discarded after training) is the bigger win on the open-set task: 0.9488
beaver-holdout AUC, **beating augmented SupCon (0.9245)** with no augmentation,
at the cost of the head's extra parameters and a hair of inclusive AUC.  CIFAR-100 also requires
a class-balanced batch sampler (25 classes × 24 samples) so each batch carries
enough per-class samples for the sliced-Wasserstein statistic.

### Per-class Gaussianity metric (experiment 15)

`supersig/metrics.py` provides a **calibrated sliced-Wasserstein Gaussianity
ratio**: project a class's embeddings onto random unit directions, standardize
each 1-D projection (shape only — location/scale removed), average the squared
W2 distance to the standard-normal quantiles over directions, and divide by the
same statistic on true N(0, I) samples of identical (n, d).  Ratio 1 = as
Gaussian as a finite sample can look.  Validated on synthetic data (Gaussian
≈ 1, bimodal ≈ 49×, t₃ tails ≈ 9×, lognormal ≈ 47×); blind to CLT-Gaussianizing
products (e.g. uniform cube), which don't arise here.

On the trained CIFAR-10 latents (test set, per-class mean over 10 classes):
**SIGReg+proto 2.6×** vs **SupCon 25.5×** — SIGReg classes are near-Gaussian
(worst: cat 4.2×, mild positive skew), SupCon's are strongly non-Gaussian
(20–35×), confirming the embeddings differ in exactly the way the objectives
prescribe.  See `plots/gaussianity_cifar10.png`.

### Final recipe

**Classwise SIGReg + learnable means seeded 3σ apart + inverse-square repulsion
(+ mild shrinkage) + Gaussian-posterior cross-entropy (logits −‖z−μ_c‖²/2), with
latent width ≥ number of classes.**  Matches augmented SupCon on both closed-set
and unseen-class protocols while keeping an explicit generative latent model:
every class is a unit Gaussian at a known mean, and the classifier is the model's
own posterior.
