# SuperSig metrics — reference and reporting template

Two layers: **Part A** (space quality on the holdout split) and **Part B**
(dataset-level statistical power, pre and post discovery).  Standard split:
one holdout class (CIFAR: class 4), seed 0, embeddings collected with
`train.collect_embeddings` on the plain-transform train and test sets.

## Part A — space metrics

All computed from frozen embeddings; anchors = seen-class centroids
(`exp28.class_centroids`) unless the arm has trained means.

| metric | source | what it measures | blind spot |
|---|---|---|---|
| probe | `exp29.linear_probe_novelty`, 3 seeds (1000+s), mean+-sd | AUC of a TRAINED linear holdout-vs-rest head | misses radial/non-linear novelty structure |
| acc | `exp29.evaluate_space` | nearest-anchor top-1 over seen classes | says nothing about novelty |
| supAUC | evaluate_space | macro OvR AUC of proto posterior | — |
| eucl | evaluate_space | novelty AUC of min anchor distance (raw) | needs calibrated distances |
| mahaT / mahaPC | evaluate_space | novelty AUC, tied / per-class Mahalanobis | per-class cov needs samples (shrink 0.1) |
| lid | evaluate_space (`exp29.lid_novelty`, Levina-Bickel k=20, seen-train refs) | novelty AUC of local intrinsic dimension; scale-free (distance ratios only), immune to ft space inflation | regime-specific: strong for on-manifold novelty (flowers 0.95, exp 78), weak off-manifold (cars/aircraft); needs k >= 20; exact-tie collapse capped at -1e-3 |
| gaussianity | `metrics.gaussianity_summary` + `exp28.print_gauss_table` | eig cond, class RMS, max corr, SW ratio, skew/kurt, separation | — |
| cent->anchor | exp53 | mean distance of class centroids to their target anchors | classwise-marginal realizability check |

**probe vs eucl/mahaT dissociate by design** — SupCon-family spaces win the
probe, calibrated (NPLM/SIGReg) spaces win the distance metrics.  Report
both; never conclude from one.  Single-seed probe spread is ~0.017
(exp 52), so gaps < ~0.02 need 3-5 seeds.

## Part B — power batteries

Toy protocol: N_D = 5000 events/toy (1000 on aircraft — test pool 3333),
signal fraction f injected from the holdout pool; power = fraction of 50
signal toys above the alpha=0.05 null quantile (200 null toys pre, 100
post).  Fractions: cifar10 `0.001,0.003,0.01,0.02,0.03,0.1`; cifar100
`0.001,0.003,0.01,0.02,0.05` (injection clamps at 500 — only 500 holdout
train images); aircraft `0.003,...,0.1`.

- **per-event** (`exp30.power_at_alpha`): threshold on min anchor distance
  (pre) or `d_seen - d_disc` (post).  Constant in f.  THE calibration
  metric: needs absolute distances; SupCon scores 0.000 here.
- **SparKer** (`exp31.run_test_battery`, M=16 kernels, 300 steps):
  semi-parametric kernel discrepancy vs a reference sample.
  **SIGMA RULE (exp 57)**: fixed sigma=1 ("k1") is calibrated for the
  supervised-recipe scale (unit class width) ONLY.  Discovery fine-tunes
  rescale compact trunks ~4x and sigma=1 goes blind at every fraction
  while the space is actually fine (annealed sigma: 1.00).  Default to the
  annealed median-heuristic schedule (omit sigma0 from sparker_kw) for any
  cross-geometry comparison.
- **Mahalanobis / MMD** (`exp32.make_stats_fns` + `battery`): parametric
  and kernel two-sample batteries.  Program-level: Maha is dead on
  CIFAR-100 at every dim/space (class similarity); MMD is the
  dimension-robust statistic.

## Part C — the interpretability panel (`104_interpretability_panel.py`)

Six numbers answering one question: **is distance to a labelled anchor a
delta log-likelihood?**  That is the property the whole program is built
around — it is what makes a distance *interpretable* (a statement about
relative likelihood) and what licenses the clustering and Neyman-Pearson
machinery downstream.  Every entry has ideal value 1 (or 0 for `ece`), so a
space can be read at a glance instead of triangulating Parts A and B.

If class c is `N(mu_c, I)` then `log p(z|c) = -1/2 ||z-mu_c||^2 + const` with
the same const for every c, so for two hypotheses the squared-distance
difference IS `2 * delta log L`, with no free scale.  The panel measures
departure from that identity.

| metric | ideal | what it measures | blind spot |
|---|---|---|---|
| `r_ll` | 1 | Pearson r between the proxy `-1/2||z-mu_c||^2` and the true class-conditional `log N(z; mu_c, Sigma_c)`, over all (point, class) pairs | insensitive to scale — see `slope` |
| `slope` | 1 | OLS slope of true log-density on the proxy.  Empirically `slope ~ 1/sigma^2`, so it is the width expressed in log-likelihood units | conflates width with anisotropy |
| `r_llr` | 1 | same correlation for pairwise *differences* (proxy LLR vs true LLR); per-class constants cancel, isolating the hypothesis-test quantity | — |
| `ece` | 0 | calibration error of `softmax_c(-1/2 d^2)` read as a class posterior; the operational cost of believing the distance | balanced-class assumption |
| `sw` | 1 | mean calibrated sliced-Wasserstein Gaussianity of the labelled components (`metrics.gaussianity_summary`) | CLT-Gaussianizing shapes |
| `rms` | 1 | mean per-dimension RMS of the labelled components = **the fitted sigma; how close the component width is to 1** | — |

Report `sep` (closest centroid pair / mean `rms`, i.e. class separation in
sigma) alongside: a space can be perfectly faithful and useless.
Interpretability and discrimination are separate axes and the panel is only
meaningful with both.

Reference density = per-class Gaussian with shrunk full covariance
(`shrink=0.1`, the campaign Mahalanobis default).  Validated on synthetic
spaces (`--selftest`): the ideal `N(mu, I)` returns
r 0.99 / slope 1.01 / ece 0.00 / sw 0.98 / rms 1.00; width 2x drives
slope -> 0.27 and rms -> 2.00; heavy (t_3) tails drive sw -> 13.7 while slope
stays ~1, confirming `sw` and `slope` are independent axes.

## Discovery metrics (`discovery.run_discovery` history)

Per round: `pool` (events past the 0.95 seen-distance quantile), `purity`
(fraction of pool that is truly novel — the make-or-break number: 0.34 on
CIFAR-10 vs 0.003-0.013 on CIFAR-100 where 500 holdout images sit in a
~2500-event tail), `kept-purity` (after conf mask), `k-hat` (BIC),
`n_anchors`, `margin` (AUC of d_seen - d_disc), `mean-anchor` (per-holdout
anchor AUC), and probe pre -> post.

## Reporting template

House style = `logs/SUMMARY_TABLES.md`.  For a new study add:

```markdown
## <dataset> (exp NN; <base model>, <dim>, holdout <h>, <epochs> ep)

| space | probe pre/post | acc | eucl | mahaT | lid | perevt | SpK@.02 | Maha@.02 | MMD@.02 |
|-------|----------------|-----|------|-------|-----|--------|---------|----------|---------|
| ...   | 0.xxxx/0.xxxx  | ... | ...  | ...   | ... | 0.xxx  | pre/post| pre/post | pre/post|

<3-6 bullet verdicts: who wins probe, who wins geometry/per-event, what
changed vs the previous table, caveats (seeds, sigma, clamping).>
```

Artifact conventions (a future session should follow these exactly):
- script: `experiments/NN_<topic>.py`, `--quick` smoke mode mandatory.
- log: `logs/expNN/expNN_<tag>.log` (nohup redirect).
- arrays: `logs/expNN/<what>_<dtag>.npz`, keys `{metric}_{arm}` and
  `{stat}_{arm}_{pre|post}`, always include `fractions` and `arms`.
- plots: `plots/expNN_{stat}_power_{dtag}.png`, `expNN_probe_{dtag}.png`.
- `dtag` = dataset, plus `_{dim}d` suffix when dim differs from the
  program default (32) so reruns never clobber prior artifacts.
- seeds: net i gets `torch.manual_seed(seed + 20 + i)`; probe repeats
  `1000 + s`; identical seeds across cells for paired scans.
- after a study: append to SUMMARY_TABLES.md, commit code + logs + npz +
  plots together (exclude in-progress logs), update docs/ if a
  program-level lesson changed.

## Known pitfalls checklist

1. SparKer fixed sigma across heterogeneous scales (exp 57) — use annealed.
2. Probe-only or geometry-only conclusions — always report both sides.
3. classwise_sigreg with unbalanced loaders — silently ~0 (MIN_PER_CLASS).
4. CIFAR-100 injection clamps at 500 events for f >= 0.01.
5. `--quick` numbers are pipeline checks, never results.
6. exp-50-family scripts recompute spaces from seeds; retrained spaces
   differ slightly (cuDNN nondeterminism, ~0.01 in AUCs).
7. Single-seed probe gaps < 0.02 are noise (exp 52).
