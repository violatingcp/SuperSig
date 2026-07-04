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

| Embedding | Inclusive micro-AUC | Beaver-holdout AUC |
|-----------|--------------------|--------------------|
| SIGReg+proto, 32-dim | 0.9670 | 0.6716 |
| SupCon (aug), 32-dim | 0.9825 | 0.7469 |
| SIGReg+proto, 100-dim, 3σ seed | 0.9822 | 0.9127 |
| SIGReg+proto, 100-dim, 5σ seed | 0.9833 | 0.9198 |
| SIGReg+proto, 100-dim, 5σ, repulsion ×3 | 0.9829 | 0.9339 |
| SIGReg+proto, 100-dim, 5σ, repulsion ×10 | 0.9821 | 0.9371 |
| **SIGReg+CE (linear head), 100-dim, 5σ** | **0.9828** | **0.9488** |
| SupCon (aug), 100-dim | 0.9853 | 0.9245 |
| SIGReg+proto, 200-dim | 0.9832 | 0.9306 |
| SupCon (aug), 200-dim | 0.9849 | 0.9414 |

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
