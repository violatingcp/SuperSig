# CIFAR-100 master table — every probe + mahaT in the program

Holdout 4, seed 0 unless noted.  probe = holdout-novelty linear probe AUC
(pre-discovery; "post" = after the discovery update).  mahaT = tied-
covariance Mahalanobis novelty AUC (pre).  Sections ordered hub-init ->
fine-tuned constructions -> from-scratch lineages; rows sorted by probe.
Sources: exps 33/34/36 (curated), 50-68 npz archives.

## Program-wide probe leaderboard (top 10)

| # | space | probe | mahaT | lineage |
|---|---|---|---|---|
| 0 | supcon->res concat @100+100d (73) | **0.9594** | 0.440 | hub |
| 0b | supcon->res-nplm concat @100+100d (73) | 0.9574 | 0.312 | hub |
| 1 | sup recipe @100d (66) | 0.9546 | 0.393 | hub |
| 2 | supcon+simclr 50+50 (34j) | 0.9547* | -- | hub (*same value band) |
| 3 | supcon+hybrid[lam1] 50+50 (34j) | 0.9541 | -- | hub |
| 4 | supcon+nplm 50+50 post-discovery (59) | 0.9467 | 0.328 | hub |
| 5 | supcon @100d (50) | 0.9443 | 0.358 | hub |
| 6 | supcon+nplm 50+50 pre (59) | 0.9431 | 0.328 | hub |
| 7 | supcon+hybrid[lam5] 16+16 (34i) | 0.9423 | 0.407 | hub |
| 8 | supcon 32d (50; 5-seed 0.9401+-0.004) | 0.9417 | 0.341 | hub |
| 9 | supcon+nplm 16+16 (59) | 0.9417 | 0.415 | hub |
| 10 | supcon scratch 200ep (67) | 0.9390 | 0.279 | CLEAN |

Best clean-lineage rows: supcon scratch 0.9390; scratch-simclr +
supcon_sigreg 0.9327; scratch-nplmcw + discovery 0.9231 (only clean space
that IMPROVES under discovery, and with calibrated geometry).

## Hub-init, 32-D standalone (exps 50/53/61/63/65)

| space | probe | mahaT | note |
|---|---|---|---|
| supcon | 0.9417 | 0.341 | 5-seed 0.9401+-0.004 / mahaT 0.386 |
| supervised CE (63) | 0.9363 | 0.397 | ties supcon; per-event 0.01 |
| nplm_bilinear | 0.9349 | 0.484 | 5-seed 0.8548+-0.042 -- tail draw |
| nplm_bil_sup_cw lam1 | 0.9246 | 0.316 | |
| nplm_dist_sup_cw learned means (65) | 0.8890 | 0.310 | cent->anchor 7.4 |
| nplm_dist_sup_cw cw-lam1 | 0.8864 | 0.372 | |
| supcon_sigreg | 0.8833 | 0.443 | |
| nplm_bil_cw lam1 | 0.8719 | 0.388 | |
| nplm_dist_sup_cw cw-lam5 | 0.8705 | 0.392 | acc 0.522 (vs 0.347 lam1) |
| lejepa | 0.8677 | 0.404 | |
| simclr_sigreg | 0.8631 | 0.458 | |
| simclr | 0.8537 | 0.336 | |
| nplm_dist_sup_cw cw-lam20 | 0.8237 | 0.373 | |
| nplm_sup_dist | 0.8033 | 0.269 | |
| nplm_distance | 0.7368 | 0.409 | |

## Hub-init, 64-D standalone (exps 50/53 64d)

| space | probe | mahaT |
|---|---|---|
| supcon | 0.9389 | 0.342 |
| supcon_sigreg | 0.9129 | 0.462 |
| simclr | 0.8854 | 0.360 |
| nplm_bil_cw | 0.8677 | 0.380 |
| simclr_sigreg | 0.8666 | 0.406 |
| nplm_dist_sup_cw | 0.8605 | 0.297 |
| nplm_bilinear | 0.8584 | 0.339 |
| nplm_sup_dist | 0.8377 | 0.324 |
| lejepa | 0.8265 | 0.323 |
| nplm_bil_sup_cw | 0.8253 | 0.265 |
| nplm_distance | 0.7618 | 0.412 |

## Hub-init, 100-D standalone (exps 50/53/66 100d)

| space | probe | mahaT |
|---|---|---|
| sup recipe (66) | 0.9546 | 0.393 |
| supcon | 0.9443 | 0.358 |
| supcon_sigreg | 0.9364 | 0.428 |
| nplm_dist_sup_cw cw-lam1 | 0.8440 | 0.463 |
| nplm_sup_dist | 0.8271 | 0.378 |

## Hub-init 16+16 concats (exps 33/34/36, curated; probe / mahaT)

| space | probe | mahaT |
|---|---|---|
| supcon+hybrid[lam5] | 0.9423 | 0.407 |
| supcon+hybrid[lam1] | 0.9409 | 0.422 |
| supcon+simclr (r1) | 0.9394 | 0.333 |
| supcon+res-simclr | 0.9281 | 0.254 |
| hybrid->supres | 0.9263 | 0.482 |
| ss[lam5]+hybrid | 0.9235 | 0.487 |
| supcon (33) | 0.9232 | 0.450 |
| ssl->supres (33) | 0.9178 | 0.330 |
| sup (33) | 0.9039 | 0.342 |
| joint (33) | 0.8985 | 0.301 |
| cls->resfeat | 0.8745 | 0.560 |
| sup->res (36 rebuild) | 0.8409 | 0.378 |
| sup->res-hybrid (36) | 0.8383 | 0.350 |
| sup->res (33) | 0.8361 | 0.392 |
| feat->rescls | 0.8196 | 0.247 |

(34j 50+50 probes, no mahaT: supcon+simclr 0.9547, supcon+hybrid[lam1]
0.9541, res-simclr 0.9519, lam5 0.9474, ss[lam1] 0.9306, hybrid->supres
0.9234, cls->resfeat 0.9228, ss[lam5] 0.9076.)

## Hub NPLM concats/residuals (exp 59; probe pre/post, mahaT pre)

| space | 16+16 pre/post | mahaT | 50+50 cwlam5 pre/post | mahaT |
|---|---|---|---|---|
| supcon+nplm | 0.9417/0.9316 | 0.415 | 0.9431/**0.9467** | 0.328 |
| sup->res-nplm | 0.7985/0.8001 | 0.325 | 0.9294/0.9385 | 0.385 |
| nplmsup+nplm | 0.8957/0.8999 | 0.489 | 0.9206/0.9321 | 0.376 |
| sup+nplm | 0.8958/0.8603 | 0.417 | 0.9191/0.9160 | 0.329 |
| nplmsup->res | 0.8448/0.8660 | 0.417 | 0.9111/0.9330 | 0.352 |

## Hub 32-D + discovery (exps 55/56; probe pre -> post)

| space | proto ft | conf-mask |
|---|---|---|
| nplm_bilinear | 0.935 -> 0.891 | 0.935 -> 0.907 |
| nplm_dist_sup_cw lam1 | 0.890 -> 0.886 | 0.892 -> 0.875 |
| nplm_dist_sup_cw lam5 | 0.859 -> **0.900** | -- |
| nplm_sup_dist | 0.825 -> 0.837 | 0.812 -> 0.839 |

## Residual fine-tuning, exp-71 recipe (exp 73; 100-D parents, deepcopy
+ e2e residual ft, concat = 200-D; holdout 4)

| space | probe | acc | eucl | mahaT | perevt |
|---|---|---|---|---|---|
| supcon->res concat | **0.9594** | 0.539 | 0.575 | 0.440 | 0.020 |
| supcon (parent, this seed) | 0.9577 | 0.575 | 0.338 | 0.347 | 0.000 |
| supcon->res-nplm concat | 0.9574 | 0.580 | 0.296 | 0.312 | 0.000 |
| supcon_sigreg (parent) | 0.9455 | 0.593 | 0.388 | 0.381 | 0.000 |
| supcon_sigreg->res concat | 0.9417 | 0.578 | 0.568 | 0.479 | 0.040 |
| supcon->res residual alone | 0.8392 | 0.211 | 0.650 | 0.558 | **0.080** |

New C100 probe record 0.9594 (previous 0.9546, sup recipe).  Caveat:
this supcon parent seed (train_arm full-list index) drew 0.9577, above
the exp-50 archive's 0.9443 -- the +0.017 concat delta over its own
parent is the honest residual effect.  The plain-res residual half is
the best-calibrated supervised C100 space (eucl 0.650, per-event 0.08)
-- residual ft transfers to CIFAR exactly as on the transfer grid, with
plain res (not res-nplm) the better objective here (many-class,
low-res images: same side of the split as DTD/galaxy10).

## Discovery on the exp-73 concats (exp 74; image-space loop, two-net
concat backbone; probe / eucl / mahaT pre -> post, purity r1)

| space | probe | eucl | mahaT | purity | notes |
|---|---|---|---|---|---|
| supcon->res concat | 0.9566 -> 0.9567 | 0.585 -> 0.562 | 0.432 -> 0.483 | 0.009 | probe flat, mahaT +0.05 |
| supcon->res-nplm concat | 0.9562 -> 0.9497 | 0.321 -> **0.530** | 0.302 -> 0.455 | 0.000 | MMD 0.84@.02 post |

(Reproduction jitter: exp-73 archived 0.9594/0.9574 for these spaces;
the exp-74 retrain drew 0.9566/0.9562 -- cudnn nondeterminism at the
+-0.003 level.)  Same verdict as the transfer grid (exp 72): at
near-zero pool purity discovery cannot move the probe, but the ft
substantially repairs the concats' calibration (res-nplm eucl +0.21,
mahaT +0.15).  The probe record stays pre-discovery.

## From-scratch lineages (exps 67/68/50-scr; 100-D, no hub trunk)

Bases (exp 67): supcon 0.9390/0.279, nplmcw 0.8987/0.411, supsig
0.8144/0.320, visreg 0.8031/0.382, simclr 0.7914/0.385, nplm 0.7572/0.361.

Base + discovery (exp 68): supcon 0.9376 -> 0.9287; nplmcw 0.8990 ->
**0.9231**; supsig 0.8433 -> 0.8211.

Suites on the bases (probe / mahaT):

| arm | scr-simclr | scr-visreg | scr-nplm | scr-nplmcw |
|---|---|---|---|---|
| supcon_sigreg | 0.9327/0.361 | 0.9263/0.246 | 0.8960/0.262 | 0.9302/0.322 |
| supcon | 0.9251/0.347 | 0.9102/0.369 | 0.9251/0.347 | 0.9290/0.406 |
| nplm_sup_dist | 0.8513/0.308 | 0.8541/0.437 | 0.8176/0.358 | 0.8390/0.363 |
| simclr_sigreg | 0.8268/0.412 | 0.8138/0.420 | 0.7498/0.441 | 0.8446/0.292 |
| lejepa | 0.8168/0.378 | 0.7906/0.366 | 0.6820/0.316 | 0.8539/0.284 |
| simclr | 0.7896/0.412 | 0.8225/0.383 | 0.7928/0.368 | 0.8917/0.286 |
| nplm_distance | 0.7346/0.439 | 0.7529/0.443 | 0.6988/0.421 | 0.7541/0.322 |
| nplm_bilinear | 0.6636/0.421 | 0.6916/0.460 | 0.7103/0.401 | 0.8344/**0.467** |

## Reading

- Probe ceiling on C100 is ~0.95 and is owned by supervised softmax /
  sup-recipe spaces at 100-D; nothing calibrated reaches it pre-discovery.
- mahaT ceiling is ~0.47-0.49 (nplm_bilinear seeds, nplmsup+nplm 16+16,
  scr-nplmcw + nplm_bilinear head) -- Maha never gets truly strong on
  C100 (class similarity), so the calibration currency here is mostly
  MMD/per-event (see power tables in SUMMARY_TABLES).
- Discovery is probe-positive on C100 ONLY for classwise-lam5-family
  spaces (0.859->0.900 hub; 0.899->0.923 clean) and 100-D concats.
- Probe and mahaT are anti-correlated across the table (the program's
  central dissociation): no C100 space exceeds probe 0.94 with mahaT
  > 0.42; supcon+nplm concats come closest to having both.
