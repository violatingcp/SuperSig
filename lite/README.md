# SuperSig-Lite: one discovery pipeline

A reduced distribution of the SuperSig campaign, focused on the single
strategy the full study (exps 146–153, three datasets, one currency) found
consistently best **and interpretable**:

> **SupCon parent → SIGReg-residual child → density-ratio discovery with the
> derived label-free cut → anchor-aware Euclidean detection**, with a
> frozen-trunk mean-Mahalanobis systematics monitor alongside.

Every component carries its justification:

| stage | choice | why |
|---|---|---|
| parent space | SupCon | best raw material: the strongest class geometry, and the parent whose residuals carry the most novelty |
| child space | `res`: NT-Xent + **SIGReg (λ=5)** on residuals r = z − μ_y | trained on exactly what the parent's anchors fail to explain; rescues draws where the parent's critic abstains; best pipeline on 2/3 Galaxy10 backbones |
| pool | Neyman–Pearson density ratio | purity 0.3–1.0 where the distance pool gives ~0; the scorer, not the cut, makes pools pure |
| cut | derived label-free rule, n_min=5, **skip-on-refuse** | no oracle constants; declines (and says so) when estimated novelty < clusterability — every observed failure is an abstention, not a wrong answer |
| adaptation | 2-round anchor+fine-tune loop (`train_sigreg_hybrid`, SIGReg weight 1) | discovery pays *only* through the anchors it plants |
| statistic | mean(d_seen − d_disc), the anchor-aware Euclidean, plus **anchor-seeded SparKer** (NP kernels initialised at the discovered anchors) | the trivial statistic wins once anchors sit on signal; SparKer@anchors is the robust second and never pathological; SIGReg's isotropy is what makes Euclidean ≈ Mahalanobis |
| monitor | mean-Mahalanobis on the **pretrained trunk** | diffuse/systematic shifts are invisible to the fine-tuned space but detected at f\*≈0.006–0.010 by the raw trunk |

**Where SIGReg acts** (it is not per-class everywhere — the placement is the
finding): the *child loss* applies a **global** marginal to the class-centred
residuals r = z − μ_y (one shared isotropic cloud); the *discovery loop's*
fine-tune applies **class-wise** SIGReg (each class isotropic about its
anchor, `classwise_sigreg_loss`); the optional `ssig` *parent* applies a
**global** marginal to the raw embedding. SparKer appears twice: as the
density-ratio **pool scorer** inside the loop, and as the anchor-seeded
**detection test**.

## Headline numbers (f\*(2σ): smallest injected fraction detected at 2σ median expected significance; lower is better)

| dataset | best frozen baseline | this pipeline | gain |
|---|---|---|---|
| Galaxy10 / LeJEPA | 0.023 (MMD) | **0.013** | 1.7× — equals the transductive GCD ceiling without seeing an unlabelled novel image |
| Galaxy10 / VISReg | 0.026 (MMD) | **0.014** | 1.7× |
| CIFAR-10 (scratch) | 0.017 (SparKer) | **0.013–0.015** | 1.3× |
| CIFAR-100 (scratch) | 0.018 (SparKer) | 0.008–0.014 on engaged draws | cut abstains on most draws at b=0.01, by design |

Known boundary (measured, not asserted): novelty that embeds *inside* the
seen manifold (fine-grained variants: aircraft, cars) and *diffuse* shifts
(a systematic smearing of seen classes) are not discoverable by this — or
any — clustering-based pipeline; the first is dead at the pool, the second
is caught by the trunk monitor instead. See the campaign summary
(`docs/sigma_summary.pdf` in the full repo) for the complete tables.

## Layout

- `supersig/` — the canonical library (vendored unchanged from the full repo;
  guarded there by the `tests/` suite: 350 tests).
- `lite/pipeline.py` — the four-call facade:
  `train_parent → train_child → discover → detect`.
- `lite/example_cifar10.py` — end-to-end on CIFAR-10 from random
  initialisation (`--quick` for a smoke run).

## Usage

```python
from lite.pipeline import train_parent, train_child, discover, detect

parent = train_parent("cifar10", holdout={4}, epochs=200)       # SupCon
child  = train_child(parent, epochs=20)                         # res child
post   = discover(child, rounds=2)          # np pool, derived cut, n_min=5
z      = detect(post, corpus)               # mean(d_seen - d_disc) -> Z
```

The `--parent ssig` flag switches to a SupCon+SIGReg parent — measured to be
the better choice only in the many-class pre-discovery regime (CIFAR-100
geometry); everywhere else the SupCon parent wins, with SIGReg doing its
work in the child loss, the loop regulariser, and the statistic's licence.

## Variance discipline

The held-out **draw** (which class is novel) dominates all other variance
(~8× the seed spread). Never quote a single-draw number without its spread;
the abstention rate across draws is part of the result, not noise.
