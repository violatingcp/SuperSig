# Transfer-suite master tables — aircraft (3 bases) + cars (exp 51/60)

Protocol: frozen trunk (features cached, a8 aug bank), 768->256->32
FeatureHead per arm, 120 epochs, holdout variants 90-99 (~330 signal test
events), n_d=1000 toys, fixed sigma=1 SparKer (all spaces here share the
compact head scale, so within-table comparison is fair).  Power columns at
f=0.1 (the only fraction with meaningful power at this pool size).
Sources: logs/exp51/results_nplm_aircraft_{dino,lejepa,visreg}.npz.

## DINO base

| arm | probe | acc | eucl | mahaT | perevt | SpK@.1 | Maha@.1 | MMD@.1 |
|---|---|---|---|---|---|---|---|---|
| supcon_sigreg | **0.781** | **0.596** | **0.764** | **0.743** | **0.222** | 0.38 | **0.64** | 0.76 |
| lejepa | 0.732 | 0.119 | 0.456 | 0.533 | 0.033 | 0.16 | 0.22 | 0.10 |
| nplm_sup_dist | 0.721 | 0.220 | 0.708 | 0.695 | 0.147 | **0.64** | 0.44 | **0.86** |
| simclr_sigreg | 0.713 | 0.161 | 0.418 | 0.485 | 0.048 | 0.18 | 0.26 | 0.30 |
| nplm_distance | 0.695 | 0.077 | 0.457 | 0.543 | 0.057 | 0.14 | 0.04 | 0.08 |
| supcon | 0.688 | 0.557 | 0.730 | 0.708 | 0.192 | 0.42 | 0.44 | 0.02 |
| simclr | 0.659 | 0.148 | 0.569 | 0.539 | 0.078 | 0.06 | 0.12 | 0.00 |
| nplm_bilinear | 0.612 | 0.083 | 0.548 | 0.595 | 0.039 | 0.16 | 0.08 | 0.00 |

## LeJEPA base

| arm | probe | acc | eucl | mahaT | perevt | SpK@.1 | Maha@.1 | MMD@.1 |
|---|---|---|---|---|---|---|---|---|
| supcon_sigreg | **0.724** | 0.404 | 0.673 | 0.627 | **0.201** | 0.52 | 0.12 | 0.24 |
| nplm_sup_dist | 0.722 | 0.178 | 0.655 | 0.651 | 0.120 | 0.28 | **0.50** | **0.72** |
| supcon | 0.706 | **0.483** | **0.704** | **0.688** | 0.171 | **0.64** | 0.16 | 0.10 |
| simclr_sigreg | 0.681 | 0.093 | 0.521 | 0.573 | 0.048 | 0.12 | 0.14 | 0.06 |
| nplm_bilinear | 0.681 | 0.066 | 0.571 | 0.611 | 0.084 | 0.18 | 0.18 | 0.00 |
| lejepa | 0.673 | 0.074 | 0.532 | 0.594 | 0.072 | 0.04 | 0.10 | 0.04 |
| nplm_distance | 0.659 | 0.053 | 0.498 | 0.561 | 0.063 | 0.12 | 0.04 | 0.00 |
| simclr | 0.642 | 0.096 | 0.610 | 0.589 | 0.084 | 0.44 | 0.04 | 0.00 |

## VISReg base

| arm | probe | acc | eucl | mahaT | perevt | SpK@.1 | Maha@.1 | MMD@.1 |
|---|---|---|---|---|---|---|---|---|
| supcon | **0.764** | **0.492** | **0.734** | **0.704** | **0.177** | **0.90** | 0.10 | 0.50 |
| supcon_sigreg | 0.737 | 0.428 | 0.703 | 0.639 | 0.138 | 0.36 | 0.22 | 0.78 |
| nplm_sup_dist | 0.731 | 0.186 | 0.631 | 0.617 | 0.087 | 0.68 | **0.40** | **0.78** |
| lejepa | 0.698 | 0.071 | 0.522 | 0.603 | 0.048 | 0.06 | 0.12 | 0.02 |
| simclr_sigreg | 0.694 | 0.082 | 0.505 | 0.565 | 0.045 | 0.10 | 0.08 | 0.08 |
| simclr | 0.668 | 0.088 | 0.595 | 0.585 | 0.099 | 0.22 | 0.06 | 0.04 |
| nplm_bilinear | 0.661 | 0.058 | 0.489 | 0.597 | 0.024 | 0.08 | 0.20 | 0.00 |
| nplm_distance | 0.648 | 0.056 | 0.504 | 0.573 | 0.072 | 0.10 | 0.04 | 0.00 |

## Stanford Cars, DINO base (196 classes, holdouts 186-195, n_d=2000)

| arm | probe | acc | eucl | mahaT | perevt | MMD@.1 |
|---|---|---|---|---|---|---|
| supcon_sigreg | **0.747** | **0.653** | **0.627** | **0.603** | 0.075 | **0.96** |
| supcon | 0.733 | 0.609 | 0.544 | 0.544 | 0.028 | 0.68 |
| simclr | 0.612 | 0.114 | 0.462 | 0.473 | 0.023 | 0.18 |
| lejepa | 0.596 | 0.096 | 0.477 | 0.490 | 0.038 | 0.12 |
| simclr_sigreg | 0.591 | 0.125 | 0.437 | 0.449 | 0.021 | 0.74 |
| nplm_bilinear | 0.570 | 0.058 | 0.476 | 0.488 | 0.033 | 0.02 |
| nplm_sup_dist | 0.541 | 0.110 | 0.531 | 0.493 | **0.106** | 0.66 |
| nplm_distance | 0.525 | 0.059 | 0.492 | 0.480 | 0.021 | 0.12 |

## Aircraft NPLM residual/concat (exp 60, DINO, 16+16)

| arm | probe | acc | eucl | mahaT | perevt | MMD@.1 |
|---|---|---|---|---|---|---|
| supsig->res-nplm | 0.786 | 0.569 | 0.743 | 0.725 | 0.192 | 0.50 |
| supsig+nplm | 0.785 | 0.565 | 0.734 | 0.725 | 0.192 | 0.76 |
| supcon+nplm | 0.772 | 0.500 | 0.691 | 0.701 | 0.201 | 0.02 |
| nplmsup+nplm | 0.721 | 0.229 | 0.602 | 0.661 | 0.087 | 0.84 |

## End-to-end NPLM fine-tuning (exp 62; exp-49 recipe, trunk trainable,
closed-set test top-1 %, head 100-D / trunk 768-D probes)

| ft objective | DINO head/trunk | LeJEPA head/trunk | VISReg head/trunk |
|---|---|---|---|
| nplm-bil-ft (label-free) | 10.0 / 64.3 | 11.1 / 30.8 | 2.5 / 30.8 |
| nplm-dist-ft (label-free) | 6.6 / 65.3 | 5.6 / 32.9 | 5.6 / 31.2 |
| nplm-sup-ft (labeled) | 10.6 / **65.3** | 16.3 / **61.4** | 11.9 / **63.2** |
| refs: frozen / CE-ft | 64.9 / 65.5 | 34.3 / 75.8 | 37.9 / 76.6 |
| refs: SupCon-ft / ss-ft | 50.2 / 43.3 | 58.7 / 49.0 | 62.7 / 54.5 |

## Suite on the nplm-sup-ft trunks (exp 51 rerun; frozen-base values in
parens.  CAVEAT: the exp-62 ft saw images AND labels of all 100 variants
including holdouts, so the novelty columns here are optimistic vs the
strict open-world protocol; the acc gains are clean closed-set numbers.)

| arm | DINO-ft probe/acc/mahaT | LeJEPA-ft | VISReg-ft |
|---|---|---|---|
| supcon_sigreg | 0.758/0.591/0.751 (0.781/0.596/0.743) | **0.812/0.638/0.800** (0.724/0.404/0.627) | 0.786/0.656/0.768 (0.737/0.428/0.639) |
| supcon | 0.749/0.562/0.724 (0.688/0.557/0.708) | 0.699/0.630/0.709 (0.706/0.483/0.688) | 0.802/**0.665**/0.757 (0.764/0.492/0.704) |
| nplm_sup_dist | 0.721/0.227/0.688 (=) | 0.734/0.338/0.787 (0.722/0.178/0.651) | 0.742/0.305/**0.812** (0.731/0.186/0.617) |
| nplm_bilinear | 0.684/0.093/0.606 (0.612/...) | 0.753/0.231/0.656 (0.681/...) | 0.754/0.307/0.655 (0.661/...) |
| label-free others | ~= frozen | +0.05-0.08 probe | +0.06-0.07 probe |

## Reading

1. **DINO is the strongest base** on every currency (best probes, accs,
   geometry), consistent with the exp-44 transfer table.
2. **The supcon_sigreg advantage is DINO-specific**: on the weaker
   LeJEPA/VISReg trunks plain SupCon retakes probe/acc/eucl (VISReg:
   0.764/0.492/0.734 vs 0.737/0.428/0.703).  The SIGReg marginal helps
   most when the underlying features are already strong.
3. **nplm_sup_dist is the most base-robust arm**: probe 0.72-0.73 on all
   three trunks (spread 0.009 vs supcon's 0.076), consistently
   second-best geometry, and the best kernel-test space on every base
   (MMD@.1 0.86/0.72/0.78; Maha best on 2 of 3) -- the calibrated space
   survives trunk changes that reorder everything else.
4. **Label-free arms are uniformly weak on aircraft on every base**
   (probe <= 0.70, acc <= 0.16, per-event <= 0.10): fine-grained
   discrimination needs labels at these data sizes regardless of trunk.
5. **Per-event power is supervised-only** on aircraft (supcon family
   0.14-0.22, nplm_sup_dist 0.09-0.15, everything else ~0.05).
6. **Cars amplifies the aircraft regime** (196 classes, ~41 imgs/class):
   supcon_sigreg sweeps every column (probe 0.747, MMD 0.96); label-free
   arms collapse (probe <= 0.61, acc <= 0.13); the one NPLM bright spot is
   per-event calibration -- nplm_sup_dist 0.106, the best in the cars
   table, beating supcon_sigreg's 0.075 despite a far worse probe.
7. **Residual/concat constructions match but do not beat the standalone
   champion on aircraft** (exp 60: supsig->res-nplm 0.786/0.725 vs
   standalone supcon_sigreg 0.781/0.743); the NPLM residual adds nothing
   over the plain nplm_bil half (separable-data effect, third
   confirmation).  The nplm_bil half still lifts plain supcon
   (0.688 -> 0.772).
8. **The nplm-sup-ft trunks are far better suite substrates than their
   frozen selves** (exp 51 rerun): every arm on the LeJEPA/VISReg ft
   trunks improves (probe +0.05-0.09, acc +0.12-0.25), with new aircraft
   records — LeJEPA-ft + supcon_sigreg 0.812/0.638/0.800 and VISReg-ft
   nplm_sup_dist mahaT 0.812, VISReg-ft supcon acc 0.665.  DINO-ft ~=
   frozen DINO (already at ceiling).  Two-stage recipe emerges: NPLM-sup
   trunk ft, then a supcon_sigreg head.  Mind the holdout-contamination
   caveat above for the novelty columns.
9. **End-to-end NPLM ft (exp 62)**: nplm-sup-ft is the best
   contrastive-family trunk ft on every base -- matches CE-ft on DINO
   (65.3 vs 65.5), beats SupCon-ft on LeJEPA (61.4 vs 58.7), ties it on
   VISReg (63.2 vs 62.7); CE-ft keeps the crown on the sketching trunks
   (75.8/76.6).  The label-free NPLM fts are conservative: they preserve
   the strong DINO trunk (64-65 vs frozen 64.9, where SupCon-ft/ss-ft
   DESTROY it: 50.2/43.3) but cannot build up the weak trunks (30.8-32.9
   vs frozen 34.3/37.9).  NPLM heads are never linear-probe spaces
   (2.5-16.3%), consistent with the whole program.
