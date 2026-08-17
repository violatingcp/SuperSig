# Space geometry: interpretability (exp 76) & inter-space similarity (exp 77)

Two post-hoc batteries over the campaign's cached spaces.  Exp 76 asks
"what does each space think is near what" (class-centroid geometry);
exp 77 asks "how similar are the spaces to each other" (the
arXiv:2503.21073 LLM-geometry program on paired image embeddings).
Artifacts: `logs/exp76/interp_*.md` (nearest-class tables + metrics),
`logs/exp77/results_*.npz`, `plots/exp76_*`, `plots/exp77_*`.

## Exp 76 — centroid interpretability

Per space: top-5 nearest class-centroid tables (cosine), superclass-
ordered distance heatmap, average-linkage dendrogram; where a superclass
partition exists (cifar100 coarse labels, aircraft manufacturer, cars
make, galaxy10 morphology): NN superclass agreement@1/@5 vs chance,
dendrogram purity, centroid silhouette, within/between distance ratio.

Transfer-grid highlights:

- **Semantics track supervision, not the probe.**  galaxy10/dino:
  supcon-ft recovers the 4-group morphology partition perfectly
  (dendro purity 1.000, agree@1 0.9); label-free nplm-bil-ft has
  essentially no semantic structure (purity 0.49, silhouette -0.08)
  despite its calibration strengths.  The probe-vs-calibration
  dissociation is also a semantic-structure dissociation.
- **Holdout classes land in the right neighborhoods unsupervised.**
  galaxy10: the never-labeled holdout `edge-on bulge` sits 0.06 from
  `edge-on no bulge` (its true sibling), next-nearest `cigar smooth`
  (what an edge-on disk looks like).  The open-world novelty settles
  where it semantically belongs.
- **Residual concats inherit the parent's semantics** (galaxy10
  resnplm-cat purity 1.000 = parent) while adding the calibration half:
  concatenation costs nothing semantically.
- **Cars organizes by body style as much as by make** (frozen DINO
  make-agree@1 0.40 vs chance 0.04; errors like Audi convertible ->
  BMW convertible).  Fine-grained "mistakes" are visually coherent.

## Exp 77 — similarity between spaces

Metrics on aligned image samples (n=4000/cell): linear CKA, mutual
kNN@10, LLE-weight transfer, orthogonal Procrustes, ridge-map R^2
(directional), calibration transfer (eucl AUC through a train-fit ridge
map), TwoNN intrinsic dimension, Levina-Bickel LID novelty.

### 1. Supervision erases the pretraining base; self-supervision keeps it

Cross-base CKA of the same arm fine-tuned from different bases
(dino/lejepa/visreg pairs):

| dataset | supcon-ft (3 pairs) | simclr-ft (dino-visreg) |
|---|---|---|
| aircraft | 0.68 / 0.70 / 0.85 | (no simclr arm) |
| cars | 0.79 / 0.75 / 0.85 | 0.46 |
| dtd | 0.68 / 0.69 / 0.75 | 0.56 |
| flowers | 0.68 / 0.64 / 0.70 | 0.62 |
| galaxy10 | 0.51 / 0.72 / 0.74 | 0.58 |

supcon-ft across two *different bases* is as similar as two *different
supervised losses* on the same base (~0.74) -- the label signal pulls
every pretraining toward one shared supervised geometry.  This is the
geometric mechanism behind the base-invariant regime rules of exp 70.
lejepa<->visreg are consistently the closest pair; dino is the outlier
base.  On dtd, simclr-ft's *local* structure (mutual-kNN 0.28) agrees
across bases better than supcon-ft's (0.16): SimCLR's texture
neighborhoods are the transferable part, matching its dtd probe crown.

### 2. Residual children: rotation on coarse regimes, new content on fine

Parent vs res-nplm child (dino/visreg cells): galaxy10 ridge-R^2
0.97-0.98 (the child is nearly a linear re-map -- calibration is
re-weighting, not new information), cars/aircraft R^2 0.91-0.95 with
mutual-kNN only ~0.2 (locally reorganized, ~5-10% genuinely new
variance).  Within the transfer grid the concat gains track the
unexplained fraction: biggest where the child is least redundant.
lejepa-based children diverge far more (CKA 0.24-0.66) than dino/visreg
ones -- lejepa parents leave the most room for residual reorganization.

CIFAR refines the claim: there the children drift MUCH further (c100
supcon vs plain-res CKA 0.24, R^2 0.32; res-nplm 0.71/0.77 -- the
lightly-ft'd ViT trunks anchor transfer children, the fully-plastic
CIFAR resnets do not) yet the probe gains are the campaign's smallest
(+0.006-0.016 vs cars +0.148).  So unexplained variance is necessary
but not sufficient: what pays is whether the new variance encodes the
holdout (cars) rather than generic reorganization (CIFAR).  Drift is
capacity to help, not help itself.

### 3. nplm-bilinear is a geometric outlier everywhere

CKA ~0.2-0.36 and mutual-kNN ~0.03-0.06 against every other arm, LLE
ratio >1 (its local weights are invalid elsewhere) -- yet often highly
*decodable from* softmax spaces (asymmetric R^2, e.g. cars supcon-ft ->
nplm-sup-ft R^2 0.97 at CKA 0.36): the NPLM spaces are reorganizations
of information the softmax spaces contain, not new information.

### 4. Loss fingerprints in intrinsic dimension

TwoNN ID orders arms the same way in every cell: nplm-sup-ft most
compressed (ID ~2-3), SSL/sigreg arms mid (5-7), supcon/ss and frozen
highest (9-13).  The SIGReg Gaussian target and the NPLM collapse are
directly visible as dimension.

### 5. LID as a novelty score: weak in general, STRONG on flowers

Levina-Bickel LID (k=20, seen-class train refs) AUC by cell: aircraft/
galaxy10 ~0.56-0.66 (weak), cars up to 0.73, dtd 0.79-0.81, **flowers
0.95-0.96 in every base and most spaces** (best 0.957,
supcon-ft-resnplm-cat/visreg) -- far above the flowers eucl numbers and
in record territory for an unsupervised score.  LID is scale-free
(distance-ratio structure, not distance), so it detects what min-dist
misses when novel classes sit near some centroid.  Holdout LID > seen
LID in nearly every space (novelty lives in locally higher-dimensional
neighborhoods).  Worth promoting into the standard battery; the flowers
result deserves a dedicated verification (exp 78 candidate).

## Exp 78 — the flowers LID result verified (78_lid_verification.py)

All controls pass; the effect is real and is NOT rebranded distance:

- **Holdout rotation** (frozen trunk, 10 random 10-class holdouts):
  flowers LID 0.879+-0.018 (min 0.844) vs kNN-dist 0.692+-0.047 --
  every random holdout scores high; classes 92-101 are not special.
  Cars under the identical protocol: 0.545+-0.040 (the contrast is a
  dataset property, not a protocol artifact).
- **Distance controls**: LID beats the classic kNN-distance baseline by
  +0.08-0.12 (e.g. flowers/dino supcon-ft 0.951 vs 0.835) and eucl by
  +0.03-0.06 everywhere on flowers; Spearman(LID, kNN-dist) 0.55-0.92
  -- correlated but carrying real independent signal.  On cars the
  ordering flips (kNNd/eucl >= LID on visreg ft spaces): LID's edge is
  regime-specific, not universal.
- **Robustness**: bootstrap 95% CIs +-0.006; per-holdout-class AUC
  min 0.86-0.92 on flowers ft spaces (no single-class driver);
  k-sensitivity plateaus at k=20-50 (k=5 degrades to ~0.78 -- the
  ratio estimator needs enough neighbors).
- **HEADLINE**: flowers LID 0.951-0.955 (unsupervised, seen-class refs
  only, no novelty labels) EXCEEDS the supervised flowers probe record
  0.906 (exp 72).  The rank ensemble with eucl adds nothing (LID
  subsumes it).  New flowers novelty record, and the first time an
  unsupervised score beats the trained probe anywhere in the campaign.

Reading: holdout flowers sit ON the flower manifold (small distances,
mediocre eucl/kNNd) but with a distinct local dimensional signature --
exactly the failure mode of distance scores that a scale-free ratio
statistic catches.  Follow-up: promote LID(k=20) into the standard
battery alongside eucl/mahaT.

## Exp 76 on CIFAR-100 (100-D; beaver = open-world holdout)

The founding question ("are otters near beavers?") answered in the
supcon parent: otter -> seal(0.02) beaver*(0.08) bear(0.11); beaver* ->
otter(0.08) shrew(0.09) seal(0.10) porcupine(0.13) -- the never-labeled
holdout lands mutually top-2 with otter, purely from ft on the other 99
classes.  Big cats form their own clique (tiger-leopard 0.18,
leopard-lion 0.20).  Metrics (chance agree@1 = 0.040):

| space | agree@1 | agree@5 | purity | sil |
|---|---|---|---|---|
| supcon (parent) | 0.700 | 0.468 | 0.591 | 0.197 |
| supcon_sigreg | 0.690 | 0.460 | 0.603 | 0.199 |
| supcon-res-nplm (concat) | 0.670 | 0.452 | 0.564 | 0.188 |
| supcon-res (concat) | 0.640 | 0.406 | 0.514 | 0.162 |
| simclr | 0.570 | 0.366 | 0.457 | 0.118 |
| nplm_bilinear | 0.530 | 0.334 | 0.417 | 0.059 |
| supcon-res-nplm (residual) | 0.630 | 0.438 | 0.493 | 0.154 |
| supcon-res (residual) | 0.490 | 0.288 | 0.365 | 0.029 |
| lejepa | 0.450 | 0.342 | 0.427 | 0.046 |

- The record res-nplm concat *tightens* the semantic ring (beaver
  0.06-0.10) at near-parent purity: the best probe space is also
  semantically sound.
- The plain-res residual half scrambles semantics by design (beaver ->
  mushroom/crab/lobster, purity 0.365); the concat restores it while
  keeping the probe gain -- semantics ride in the parent half.
- nplm_bilinear keeps correct neighbor *ordering* on cifar100 (beaver:
  otter 0.04, seal 0.06) unlike on galaxy10, but with distances
  compressed to 0.03-0.07 (the calibrated-core signature); supervised
  softmax still carries the most hierarchy.
