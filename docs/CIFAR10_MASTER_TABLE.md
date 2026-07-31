# CIFAR-10 master table — exps 33/34i/36 vs 50/53/55/58

All spaces 32-D total (concat arms 16+16), holdout 4 (deer), seed 0.
Columns: probe = holdout novelty linear-probe AUC pre/post-discovery; acc /
eucl / mahaT are pre-discovery (post not measured for them in any exp);
power columns = pre/post at alpha=0.05.  "--" = not measured.

Name mapping to the user-facing loss taxonomy:
- "supervised simclr" = SupCon.  "supervised sigreg" = the supervised
  SIGReg recipe (`sup`).  "nplm" arms all carry a global SIGReg marginal by
  construction; "+cw sigreg" = the classwise-SIGReg marginal (exp 53).
- lam5 hybrid variants exist on C10 only inside concat spaces (exp 34i);
  the standalone supcon_sigreg (exp 50) is lam=1.
- Post-discovery values: exp 33/34i/36 = their own discovery loops;
  NPLM arms = exp 55/58 (proto/repulse ft unless marked ft=nplm).

## Unsupervised (augmentation positives only)

| space | probe pre/post | acc | eucl | mahaT | SpK@.01 | Maha@.02 | MMD@.02 |
|---|---|---|---|---|---|---|---|
| simclr (50) | 0.867 / -- | 0.729 | 0.236 | 0.229 | 0.08/-- | 0.00/-- | 0.36/-- |
| simclr+sigreg "ss" (50) | 0.851 / -- | 0.718 | 0.311 | 0.285 | 0.02/-- | 0.00/-- | 0.14/-- |
| lejepa (50) | 0.864 / -- | 0.695 | 0.408 | 0.371 | 0.14/-- | 0.10/-- | 0.04/-- |
| nplm bilinear (50/55) | 0.853 / 0.892 | 0.546 | 0.510 | 0.494 | 0.04/0.02† | 0.16/0.36 | 0.06/0.00 |
| nplm distance (50) | 0.790 / -- | 0.650 | 0.500 | 0.481 | 0.10/-- | 0.14/-- | 0.00/-- |
| nplm bilinear +cw sigreg (53) | 0.854 / -- | 0.604 | 0.570 | 0.564 | 0.02/-- | 0.26/-- | 0.06/-- |

## Supervised

| space | probe pre/post | acc | eucl | mahaT | SpK@.01 | Maha@.02 | MMD@.02 |
|---|---|---|---|---|---|---|---|
| supcon (50, 20ep) | 0.942 / -- | 0.861 | 0.689 | 0.644 | 0.32/-- | 0.28/-- | 0.88/-- |
| supcon (33, 10ep) | 0.934 / 0.930 | 0.873 | 0.753 | 0.653 | 0.46/0.12† | 0.18/0.36 | 0.78/0.40 |
| supcon+sigreg lam1 (50) | 0.882 / -- | 0.844 | 0.637 | 0.577 | 0.30/-- | 0.10/-- | 0.74/-- |
| sup = supervised SIGReg (33) | 0.923 / 0.956 | 0.884 | 0.778 | 0.645 | 0.24/0.30 | 0.74/0.70 | 0.44/0.82 |
| joint (33) | 0.903 / 0.940 | 0.892 | 0.752 | 0.511 | 0.10/0.24 | 0.48/0.68 | 0.36/0.70 |
| sup nplm (50/55/58) | 0.783 / **0.963** | 0.871 | 0.726 | 0.723 | 0.26/0.06 | 0.46/0.40 | 0.28/0.34 |
| — same, ft=nplm (58) | 0.783 / 0.812 | " | " | " | 0.26/**0.32** | 0.46/**0.90** | 0.28/**0.74** |
| sup nplm bilinear +cw (53) | 0.781 / -- | 0.447 | 0.516 | 0.514 | 0.08/-- | 0.08/-- | 0.00/-- |
| sup nplm +cw sigreg (53/55/58) | 0.895 / **0.977** | 0.885 | 0.752 | 0.729 | 0.42/0.26 | 0.32/0.62 | 0.12/0.42 |
| — same, ft=nplm (58) | 0.895 / 0.836 | " | " | " | 0.42/**0.66** | 0.32/0.56 | 0.12/**0.72** |

## Combined (concatenated 16+16 spaces)

| space | probe pre/post | acc | eucl | mahaT | SpK@.01 | Maha@.02 | MMD@.02 |
|---|---|---|---|---|---|---|---|
| supcon+simclr (33) | 0.949 / 0.919 | 0.885 | 0.645 | 0.530 | 0.12/0.06† | 0.12/0.38 | 0.86/0.52 |
| supcon+hybrid[lam5] (34i) | 0.948 / -- | 0.850 | 0.457 | 0.583 | 0.02/0.08† | 0.08/0.14 | 0.66/0.26 |
| ss[lam5]+hybrid (34i) | 0.858 / -- | 0.896 | 0.557 | 0.489 | 0.00/0.10 | 0.06/0.54 | 0.58/0.26 |
| hybrid->supres (34i) | 0.890 / -- | 0.862 | 0.646 | 0.583 | 0.02/0.28 | 0.42/0.50 | 0.32/0.56 |
| sup->res (33) | 0.923 / 0.969 | 0.910 | 0.810 | 0.743 | 0.26/0.18 | 0.84/0.76 | 0.32/0.76 |
| ssl->supres (33) | 0.890 / 0.940 | 0.881 | 0.680 | 0.550 | 0.02/0.28 | 0.56/0.60 | 0.32/0.88 |
| sup->res (36) | 0.930 / -- | 0.911 | 0.795 | 0.735 | 0.34/0.22 | 0.80/0.76 | 0.38/0.70 |
| sup->res-hybrid (36) | 0.938 / -- | 0.918 | 0.816 | 0.758 | 0.28/0.38 | 0.94/0.84 | 0.42/0.94 |

† Fixed-sigma (k1) SparKer values on spaces whose scale differs from the
supervised-recipe geometry are suspect (exp 57): the nplm_bilinear post and
the contrastive-trunk post values (supcon 0.12, supcon+simclr 0.06,
supcon+hybrid 0.08) are likely artifacts of the sigma=1 kernel, not real
power loss.  The sup-NPLM post rows use exp-58 annealed-sigma values (their
exp-55 k1 counterparts were ~0.05 across the board).  exp-33/36 SIGReg-arm
values are scale-matched and trustworthy.

## Reading

1. **Best pre-discovery probe**: supcon+simclr concat (0.949) and
   supcon+hybrid[lam5] (0.948); best standalone: supcon 0.942.
2. **Best pre-discovery calibrated geometry**: sup->res-hybrid concat
   (eucl 0.816, mahaT 0.758, Maha@.02 0.94); best standalone: sup nplm
   +cw sigreg (0.752 / 0.729).
3. **Best post-discovery probe overall**: sup nplm +cw sigreg + proto ft
   (**0.977**), then sup->res (33) 0.969, sup nplm 0.963 — the NPLM
   standalone arms overtake every concat once discovery runs.
4. **Best post-discovery detection**: sup->res-hybrid (Maha 0.84, MMD 0.94
   @.02) among concats; sup nplm ft=nplm among standalones (Maha 0.90,
   MMD 0.74, SpK@.01 0.32-0.66, per-event 0.57@.02 — see exp 58).
5. **The ft-objective split (exp 58)**: proto ft -> probe (0.96-0.98);
   NPLM ft -> power (2x on most statistics).  No single update wins both.
6. **Unsupervised arms**: never competitive on C10 in either currency;
   the NPLM corners are the only ones with usable raw-distance geometry.
