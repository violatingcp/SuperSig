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

### The series at a glance

| # | Study | Setup | Headline numbers (probed AUC unless noted) | Takeaway |
|--:|-------|-------|--------------------------------------------|----------|
| 09 | CIFAR-10 baseline | 32d, pretrained ResNet, deer holdout | SIGReg 0.965/0.841 · SupCon 0.992/0.929 (incl/holdout) | SupCon leads both protocols |
| 10 | Ablations | drop hinge; drop SupCon's augs | no-hinge ≈ same; SupCon-noaug 0.990/0.914 | hinge droppable; SupCon's edge is the **loss**, not augs |
| 11 | Repulsive means | 3σ seed + inverse-square repulsion | 0.982/0.851; free 3σ collapses (0.77/0.71) | adaptive geometry beats static; collapse is real |
| 12 | + discriminative term | linear-head CE / SupCon term | CE 0.992/0.924 · SupCon-term 0.989/0.886 | CE hybrid ties augmented SupCon, no augs |
| 13 | Head-free variants | Gaussian-posterior "proto"; wrong-mean hinge | proto 0.992/0.911 · hinge 0.979/0.874 | the model classifies itself; zero extra params |
| 14 | CIFAR-100 + width | 32d → 100d → 200d | holdout 0.67 → 0.91 → 0.93 (proto) | latent width ≥ n_classes is essential; saturates after |
| 14 | Seeds & repulsion | 5σ seed; rep ×3/×10; CE | 5σ +0.7; rep ×3 +1.4; **CE 0.9488 beats SupCon-aug 0.9245** | wide seed + CE = best CIFAR-100 holdout |
| 15/16 | Gaussianity metric | calibrated sliced-W2 ratio (1 = Gaussian) | CIFAR-10: SIGReg 2.6× vs SupCon 25.5×; CIFAR-100: 1.21–1.28 vs 1.68 | SIGReg classes really are Gaussian (in shape) |
| 17 | Multi-class holdout | k = 1→20 unseen classes | CE 0.949→0.702 · SupCon 0.925→0.742 (crossover at k≈3) | SIGReg wins few-unseen; SupCon degrades more gracefully |
| 17 | Augmentation layer | SupCon aug stack on SIGReg inputs | +1.5–2 pts at k≥10; −2.4 at k=1 (CE) | helps exactly where invariance matters |
| 18 | Probe-free novelty | score = model's own likelihood | naive **inverts** (0.38); typicality ~chance | Nalisnick OOD pathology reproduced; novel points sit *on* seen shells |
| 19 | Empirical Mahalanobis | fitted per-class covariances | chance at small k; 0.67–0.68 at k=20; **eig spectrum 0.001/0.02/1–5** | latent is not unit-Mahalanobis; class clouds are low-rank pancakes |
| 20 | Eigenspectrum tuning (100d) | SIGReg weight ×1→×100 | eig med stuck ~0.03; both metrics degrade monotonically | can't fix by loss weight; w=1 dominates |
| 20 | 16d CIFAR-100 | w=20, 256 slices | eig med ≈ 1 but detection ~chance | self-calibration and 100-class detection are incompatible |
| 20 | **16d CIFAR-10 (native regime)** | 10 classes, w=20 | probed 0.88; **probe-free 0.80–0.78, ≥ fitted Mahalanobis, beats own probe at k=3** | the "true Mahalanobis space" design realized |
| 20 | SupCon reference (16d) | same suite | probed 0.92–0.90; probe-free 0.81→0.71 | SupCon owns probes; SIGReg owns probe-free at k≥2 |
| 21 | Factorized two-stage | SSL trunk (no labels) + heads; leakage-free | frozen: bottleneck; SSL-SIGReg + fine-tune: probed 0.82–0.72, probe-free ~0.75 stable, best eigenspectra | augmentations factorize nuisance; no dead directions |

Final recipes: **probed / few unseen** → repulsive floating means (5σ seed) +
linear-head CE, width ≥ n_classes, w=1.  **Probe-free / calibrated** → width ≈
intrinsic class dim (~16), w=20, proto term; score novelty by distance to the
learned means, no probe, no fitting.  **Leakage-free** → SIGReg-SSL
pretraining, fine-tune with the trunk floating.

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
property of the SIGReg framework, not of the CE head.  Per-class AUCs span
~0.5–0.95 at k = 20: visually distinctive unseen classes (cockroach, wardrobe,
spider) stay easy; classes with in-distribution lookalikes (fox, possum,
cattle, tractor) approach chance.

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
unit covariance structural (a per-class whitening layer) rather than penalised.

### 16-dim latent: self-calibration achieved, detection lost (experiment 20)

At d=16 the eigenspectrum becomes genuinely tunable — the intrinsic-dimension
diagnosis was right:

| Config (d=16, 5-epoch sweep) | eig min / med / max |
|------------------------------|---------------------|
| w=1 (baseline) | 0.077 / 0.321 / 0.717 |
| w=5 | 0.146 / 0.633 / 1.106 |
| **w=20, 256 slices** | **0.216 / 0.872 / 1.708** |
| w=100, 256 slices | 0.213 / 0.882 / 1.832 |

The full suite at d=16, w=20 (proto & CE, k=1–20) shows per-cell spectra of
~0.4–0.5 / 0.9–1.1 / 1.5–2.4 — an approximately *true* unit-Mahalanobis space,
and as self-calibration predicts, the unit-covariance score matches or beats
the empirical Mahalanobis fit (e.g. 0.60 vs 0.51 at k=1 for CE).  But
detection collapses everywhere: probed AUC 0.25–0.59 (inverted at k=1),
probe-free ~0.45–0.57 — far below the 100-dim numbers (0.92–0.95 probed at
k=1).  100 classes in 16 dimensions with a 20:1 Gaussianisation:discrimination
ratio leaves too little separability for any detector.

**The structural conclusion of the whole series:** the "true Mahalanobis
space" and strong open-set detection are in direct tension when the class
count exceeds the feasible latent dimension.  Unit within-class covariance is
only fillable at d ≈ intrinsic class dimensionality (~16), while separating
100 classes (and giving unseen ones room) demands d ≥ 100.  The design as
originally envisioned is self-consistent only when n_classes ≲ d ≈ intrinsic
dim — e.g. CIFAR-10 at d=16, the regime the MNIST/CIFAR-10 studies happened to
live in.

### CIFAR-10 at 16-dim with tuned parameters — the design validated

In its native regime (10 classes, d=16, orthogonal 5σ anchors, w=20/256
slices) the trifecta holds (holdouts: deer; +truck; +airplane):

| k | method | probed | Mahal-pc | **unit-cov (no fit, no probe)** | eig med |
|--:|--------|--------|----------|--------------------------------|---------|
| 1 | proto | 0.8804 | 0.7816 | **0.8002** | 1.02 |
| 1 | CE | 0.8258 | 0.7244 | 0.7331 | 1.12 |
| 2 | proto | 0.7659 | 0.7329 | 0.7386 | 1.06 |
| 2 | CE | 0.7236 | 0.7266 | 0.6934 | 0.99 |
| 3 | proto | 0.7339 | 0.7411 | **0.7813** | 0.08* |
| 3 | CE | 0.7456 | 0.7423 | **0.7736** | 0.86 |

The eigenspectrum median sits at ~1 (the unit ideal; *one training run
slipped), the unit-covariance score **matches or beats the empirically fitted
Mahalanobis in every cell** — the space is self-calibrated, no empirical
estimate needed, exactly as designed — and at k=3 the probe-free model
density *outperforms the trained probe*.  Probed AUC costs only a few points
vs the untuned 32-dim recipe (0.88 vs 0.91 at k=1).  The residual weakness is
the min eigenvalue (~0.004): a few latent directions carry no within-class
variance even at d=16.

SupCon reference at the same width (two-view aug; probe-free score = nearest-
centroid cosine):

| k | | SIGReg proto (tuned) | SIGReg CE (tuned) | SupCon (aug) |
|--:|---|---------------------|-------------------|--------------|
| 1 | probed | 0.8804 | 0.8258 | **0.9210** |
| 1 | probe-free | 0.8002 | 0.7331 | **0.8127** |
| 2 | probed | 0.7659 | 0.7236 | **0.9040** |
| 2 | probe-free | **0.7386** | 0.6934 | 0.7239 |
| 3 | probed | 0.7339 | 0.7456 | **0.8982** |
| 3 | probe-free | **0.7813** | 0.7736 | 0.7060 |

SupCon's probed detection barely degrades with k (0.92 → 0.90 → 0.90) and
dominates the tuned SIGReg by up to 15 points — the w=20 Gaussianisation
deliberately trades probed performance for calibration.  On the probe-free
column the ordering flips with k: SupCon's cosine score falls (0.81 → 0.72 →
0.71) while SIGReg's self-calibrated density holds (0.80 → 0.74 → 0.78),
overtaking at k ≥ 2 and beating its own probe at k=3.  Each method is best at
the game it was designed for: contrastive + probe for supervised detection,
Gaussian latent + own likelihood when no probe (no labels for the unseen) is
available.

### Two-stage factorization (experiment 21)

Idea: the class labels are too coarse — end-to-end SIGReg crushes within-class
nuisance diversity (background, pose, color) into the class Gaussian.
Factorize: stage 1 learns an augmentation-invariant trunk with **no labels**
(SimCLR, or LeJEPA-style invariance + global SIGReg via `--stage1 sigreg`);
stage 2 trains the class-conditional SIGReg / SupCon latent on top (`--finetune`
leaves the trunk floating; default freezes it).  This also removes the
supervised-pretraining leakage caveat: the held-out class is never seen in any
form.

| k | | SIGReg probed / free | SupCon probed / free |
|--:|---|----------------------|----------------------|
| — | *frozen SimCLR trunk (20 ep)* | 0.75 / 0.58 · 0.77 / 0.62 · 0.71 / 0.56 | 0.81 / 0.65 · 0.75 / 0.62 · 0.80 / 0.64 |
| 1 | SSL-SIGReg init + fine-tune | 0.8225 / 0.7428 | 0.8937 / 0.7665 |
| 2 | SSL-SIGReg init + fine-tune | 0.6919 / **0.7477** | 0.8541 / 0.6971 |
| 3 | SSL-SIGReg init + fine-tune | 0.7242 / **0.7479** | 0.8653 / 0.7432 |

Findings: a *frozen* 20-epoch SSL trunk bottlenecks both methods (SimCLR needs
far longer to match fine-tuned features).  With the trunk **floating**
(fine-tuned from the SSL-SIGReg init) the leakage-free pipeline lands within
3–6 probed points of the supervised-pretrained end-to-end runs and matches
them probe-free; SIGReg's probe-free score is the most stable in the study
(~0.75 at every k, above its own probe for k ≥ 2), and the SSL-SIGReg
initialization yields the healthiest eigenspectra of the whole series
(min 0.02–0.03, median 0.84–0.97 — no dead directions).  The homogeneous
pipeline (global SIGReg pretraining → class-conditional SIGReg fine-tuning)
composes cleanly.

5× longer stage 1 (100 SSL epochs) does **not** close the probed gap — all
probed/probe-free changes are within single-seed noise (SIGReg probed
0.81/0.72/0.73; SupCon actually drifts down) — so the residual 3–6 points vs
the supervised-pretrained init are attributable to supervised feature shaping
(and its leakage), not SSL budget: the leakage-free numbers are the fair ones.
What longer SSL *does* buy is calibration: the SIGReg eigenspectrum floor
rises monotonically (0.017 → 0.07–0.11), putting every within-class direction
within one order of magnitude of unit variance for the first time in the
series.

### Dual-space concatenation (experiment 22) — best architecture in the series

Train BOTH spaces on the same input and concatenate: a 64-dim SSL space
(SIGReg-SSL trunk: augmentation invariance + global N(0,I), no labels) and a
16-dim supervised space (a copy of the trunk fine-tuned with tuned
SIGReg+proto or SupCon), evaluated as [sup ; ssl] (80-dim).  Leakage-free.

| k | method | probed sup / ssl / **concat** | free: sup / concat-Mahal |
|--:|--------|-------------------------------|--------------------------|
| 1 | sigreg | 0.8451 / 0.8261 / **0.9083** | 0.7352 / **0.7483** |
| 1 | supcon | 0.9111 / 0.8261 / **0.9390** | 0.7651 / 0.6854 |
| 2 | sigreg | 0.6431 / 0.7799 / **0.8113** | 0.6812 / **0.7084** |
| 2 | supcon | 0.8677 / 0.7799 / **0.8999** | 0.7109 / 0.6989 |
| 3 | sigreg | 0.7714 / 0.7800 / **0.8460** | 0.7333 / **0.7805** |
| 3 | supcon | 0.8768 / 0.7800 / **0.8953** | 0.7204 / 0.7461 |

Concatenation beats both parts probed in **every** cell (+2 to +15 points):
the two spaces have decorrelated failure modes — a novel class hiding inside a
seen-class Gaussian in the supervised space is still displaced in the SSL
space, and vice versa.  SupCon+concat (0.939 / 0.900 / 0.895) beats even the
leakage-*tainted* end-to-end numbers at every k — the best detection in the
series, with no asterisk.  Probe-free: raw concat center-distance is dragged
down by the SSL half (shell pathology), but concat **Mahalanobis** repairs it
and beats the sup-only free score in every SIGReg cell (0.75 / 0.71 / 0.78).
Caveat: probes are linear; some of the concat gain is simply more usable
dimensions — but the parts were probed at their own widths, so the synergy is
real.

**Equal-dimension variant (16 + 16 = 32d) and SSL-objective comparison**
(`--ssl-obj sigreg|simclr --ssl-dim 16`), concat probed / (synergy over best
part) / concat-Mahalanobis:

| k | branch | SIGReg-SSL trunk | SimCLR trunk |
|--:|--------|------------------|--------------|
| 1 | sigreg | 0.8976 (+6.7) / 0.70 | 0.9013 (+1.1) / 0.67 |
| 1 | supcon | 0.9205 (+4.7) / 0.69 | 0.9266 (+0.5) / 0.52 |
| 2 | sigreg | 0.8165 (+8.4) / 0.71 | 0.8287 (+2.5) / 0.66 |
| 2 | supcon | 0.8935 (+2.1) / 0.67 | 0.8937 (+3.6) / 0.57 |
| 3 | sigreg | 0.7732 (+6.2) / 0.74 | 0.8002 (+5.7) / 0.64 |
| 3 | supcon | 0.8755 (+0.0) / 0.76 | 0.8905 (+1.0) / 0.61 |

At equal (halved) width the concat still beats both parts almost everywhere —
the synergy is complementarity, not dimension count (the 32d concat is within
1–2 points of the 80d version except SupCon k=1).  The two SSL objectives play
different roles: **SimCLR** yields the stronger individual spaces (its init
lifts the supervised branch: 0.8899 vs 0.8303 at k=1) and slightly higher
absolute concat probed AUC in every cell, while **SIGReg-SSL** is the more
complementary partner (synergies up to +8.4 vs +1–3) and clearly better for
probe-free/Mahalanobis scoring (0.70–0.76 vs 0.52–0.67) — its globally
Gaussian space supports density corrections that SimCLR's does not.

### Discovered anchors (experiment 23) — open-world class discovery

After supervised training (16d tuned recipe), the FULL train set is embedded
as unlabeled data.  In the self-calibrated space Mahalanobis clustering is
plain k-means, and the model's own geometry supplies every decision: outlier
pool = points beyond the 0.95-quantile shell of every seen mean; cluster count
= BIC under unit-variance Gaussians; new anchors = the cluster centers.  The
pool is pseudo-labeled by cluster and training continues with the extended
anchor set.

| k held out | pool purity | k̂ | novelty AUC before → **after** | discovered-anchor AUC per class |
|--:|---|---|---|---|
| 1 | 0.48 | 4 | 0.7349 → **0.9322** | deer 0.926 |
| 2 | 0.61 | 4 | 0.7484 → **0.9217** | deer 0.945, truck 0.984 |
| 3 | 0.74 | 4 | 0.7330 → **0.9119** | airplane 0.938, deer 0.946, truck 0.956 |

**The strongest label-free result in the series.**  Discovery + fine-tuning
lifts probe-free novelty by ~18 points, beating the trained probes at every k
(e.g. 0.9322 vs 0.8804 at k=1), and each held-out class ends up owning a
discovered anchor that detects it at 0.93–0.98 AUC — classes nobody ever
labeled become first-class citizens of the Gaussian latent.  BIC over-clusters
(k̂=4 regardless of true k, since seen-class outliers add structure), but the
surplus anchors are harmless — the genuine classes claim their own.  Only
~40–50 % of the novel class sits beyond the initial shell threshold, yet
fine-tuning onto the discovered anchor pulls in the rest of the distribution.

**Same clustering on the SupCon space** (`--space supcon`): everything the
SIGReg space provides must be estimated — empirical tied-covariance whitening
for the Mahalanobis coordinates, empirical centroids for the means, and the
discovered clusters become new SupCon pseudo-classes:

| k | space | pool purity | before → after | per-class anchors |
|--:|-------|-------------|----------------|--------------------|
| 1 | SIGReg | 0.48 | 0.735 → **0.932** | deer 0.926 |
| 1 | SupCon | **0.001** | 0.269 → 0.787 | deer 0.913 |
| 2 | SIGReg | 0.61 | 0.748 → **0.922** | 0.945 / 0.984 |
| 2 | SupCon | 0.43 | 0.390 → 0.858 | 0.768 / 0.867 |
| 3 | SIGReg | 0.74 | 0.733 → **0.912** | 0.938 / 0.946 / 0.956 |
| 3 | SupCon | 0.21 | 0.300 → 0.849 | 0.844 / 0.805 / 0.693 |

SIGReg wins every metric at every k.  The SupCon pipeline fails at step 1: its
outlier pool is erratic (purity 0.001 at k=1 — the whitened tail is pure
seen-class noise, since whitening amplifies the near-dead covariance
directions) and its before-scores are inverted.  Fine-tuning partially rescues
it through an accidental "other-bucket" effect (outlier pseudo-classes attract
anything unlike the seen classes), but the discovered anchors are far less
faithful to the true novel classes.  Open-world discovery is where the
self-calibrated geometry is genuinely load-bearing, not merely elegant.

**CIFAR-100 (100-dim, w=1, `--dataset cifar100`)** — the uncalibrated regime.
Pool purity / after-margin / mean per-class discovered-anchor AUC:

| k | SIGReg space | SupCon space |
|--:|--------------|--------------|
| 1 | 0.001 / 0.49 / 0.69 | 0.001 / 0.69 / 0.73 |
| 3 | 0.011 / 0.49 / 0.42 | 0.011 / 0.62 / 0.59 |
| 10 | 0.123 / 0.67 / **0.78** | 0.041 / 0.71 / 0.71 |
| 20 | 0.289 / 0.67 / **0.83** | 0.097 / 0.67 / 0.72 |

At small k discovery fails in BOTH spaces (pool purity ~0: at 100 dims novel
points sit inside the seen shells, per exp 18, so the distance tail is noise) —
the CIFAR-10 small-k success required the self-calibrated space, which cannot
exist at 100 classes.  At large k the mechanism recovers: enough novel mass
reaches the tail, BIC tracks the true class count well (k̂ = 12 / 22 vs 10 /
20), and the SIGReg space yields clearly more faithful per-class anchors
(0.78 / 0.83 vs 0.71 / 0.72; 3× the pool purity), with a dozen k=20 classes
detected at 0.88–0.96 label-free — broadly better than the probed per-class
tail at the same k.

**Iterating the loop (experiment 24, `--rounds 2`)**: round 2 pools against
ALL current anchors with a recalibrated threshold, refreshes every pooled
point's pseudo-label to its nearest discovered anchor, and fine-tunes again.

| dataset / k | round 1 → 2: pool purity | margin AUC | mean anchor AUC |
|---|---|---|---|
| C10 k=1–3 | 0.47–0.74 → 0.06–0.31 | ~stable (0.91–0.94) | ~stable (0.89–0.95) |
| C100 k=10 | 0.095 → 0.138 | 0.605 → **0.694** | 0.661 → **0.755** |
| C100 k=20 | 0.268 → 0.296 | 0.646 → **0.713** | 0.805 → 0.768 |

Where round 1 already succeeded (CIFAR-10) the second round finds an
almost-empty, low-purity pool and changes nothing — iteration is safe.  Where
round 1 was partial (CIFAR-100) the loop is genuinely self-improving: +9
points margin and anchor AUC at k=10, +7 margin at k=20.  The k=20 per-class
dip (0.805 → 0.768) exposes the one design gap: each round appends k̂ new
anchors without merging, so 44 anchors fragment 20 classes — anchor
merge/pruning is the natural next refinement.

At a smaller width (`--emb-dim 32`, same CIFAR-100 protocol) discovery is as
good or better despite the anchor crowding: k=10 round 1 pool purity 0.142 vs
0.095 at 100d and mean anchor AUC 0.789 vs 0.661, with the best CIFAR-100
label-free margin of the series after round 2 (0.708); k=20 is a wash (margin
0.65 flat, anchors 0.73 → 0.78).  The better-conditioned covariance at 32d
outweighs the crowding penalty for discovery, at a third of the compute —
though neither width approaches the calibrated CIFAR-10 regime.

**Discovery in the concatenated space (experiment 25)**: dual 32-dim spaces
(frozen SIGReg-SSL trunk + supervised SIGReg branch), anchors =
[learned mean ; SSL centroid], pooling/BIC/k-means on the 64-dim joint
distance, rounds as exp 24.  Round-2 results vs the sup-only 32d loop:

| k | concat: purity / margin / anchors | sup-only 32d |
|--:|-----------------------------------|--------------|
| 1 | 0.008 / 0.675 / **0.799** | — (best prior C100 k=1: 0.73) |
| 3 | 0.035 / 0.519 / 0.733 | — |
| 10 | 0.121 / 0.661 / 0.781 | 0.212 / **0.708** / 0.740 |
| 20 | 0.174 / 0.598 / **0.805** | 0.289 / **0.646** / 0.779 |

The hoped-for small-k rescue does not happen: the concat pool is as empty as
the sup-only one (purity ~0.01 at k ≤ 3) — at 100 classes the SSL space also
embeds a lone novel class among its semantic neighbours, so the decorrelated
failure modes that powered the CIFAR-10 concat are much weaker here.  What the
concat does buy is anchor fidelity (best C100 per-class anchors at k=1 and
k=20: 0.799 / 0.805) at some cost in combined margin.  The single-novel-class-
at-scale problem remains open.



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
