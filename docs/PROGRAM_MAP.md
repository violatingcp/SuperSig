# SuperSig program map — tests, functions, architectures, losses, training

Code-level companion to the concept guides: [LOSSES.md](LOSSES.md) (loss
math + named arms), [METRICS.md](METRICS.md) (reporting conventions),
[QUESTIONS.md](QUESTIONS.md) (recurring research questions).  This file
answers "which function implements what, and which script runs which
test."  LaTeX summary of the exp-70/71 campaign:
`transfer_ft_summary.tex/.pdf`.

## 1. Neural-net architectures

| model | definition | used for |
|---|---|---|
| `CIFARResNetBackbone(dim, pretrain=ds)` | `supersig/models.py:69` — torch-hub resnet20 (cifar10/cifar100-pretrained, or `pretrain=None` = random init) + projection head to `dim` | all CIFAR experiments; scratch lineages (exp 67) use `pretrain=None` |
| `CIFARBackbone` / `ConvBackbone` / `SupervisedCNN` | `supersig/models.py` | legacy (pre-exp-25 history) |
| ViT-B/16 trunks | `experiments/40_dtd_bases.py::LOADERS` — {dino, lejepa, visreg}, frozen 768-D CLS features; `CACHE_TAG` names the feature caches | all transfer experiments |
| `FeatureHead(emb_dim)` | `experiments/37_dtd_vit.py:85` — 768 -> 256 -> emb MLP | heads on frozen features (exps 37-51, 69); the head half of `FineTuneModel` |
| `FineTuneModel(base, emb_dim, n_out=None)` | `experiments/43_dtd_finetune.py:110` — trunk + FeatureHead (or linear if `n_out`), everything trainable | end-to-end fine-tuning (exps 43/49/62/70/71) |

Dims: CIFAR-10 16-D (recipes) / 32-D (NPLM program); CIFAR-100 32/64/100-D;
transfer heads 32-D (frozen suite) or 100-D (end-to-end ft).

## 2. Loss functions (`supersig/losses.py`)

| function | line | what it is |
|---|---|---|
| `sigreg_loss(z, n_slices)` | 30 | SIGReg: sliced W2 distance to N(0, I) |
| `classwise_sigreg_loss(z, y, means, ...)` | 47 | SIGReg per class on z - means[c]; needs balanced loader (MIN_PER_CLASS=8) |
| `make_anchors(scale, emb_dim, n_classes)` | 71 | fixed class anchors; orthogonal iff emb_dim >= n_classes |
| `supcon_loss(feats, labels, temp)` | 306 | SupCon; = NT-Xent/SimCLR with instance ids + temp 0.5 |
| `HybridContrastiveLoss(positives, critic, estimator, marginal, tau, lam)` | 378 | the design cube; `_critic_matrix` (324) cosine/bilinear/distance, `_softmax_interaction` (345), `_nplm_interaction` (354) = E_ref[e^g - 1] - E_pos[g], exponent clamp 30 |
| `repulsion_loss` / `shrink_loss` / `separation_loss` | 103/120/97 | anchor-geometry terms (learned-means exp 65, hybrid recipes) |
| `mean_geometry` | 125 | anchor-spread diagnostics |
| `DualSuperVisReg` | 134 | legacy dual-space objective (exps 27-28) |

Named arm -> cube coordinates: see LOSSES.md section "Named arms".

## 3. Training strategies (loops and steps)

### Package loops (`supersig/train.py`) — CIFAR world, frozen-feature heads

| function | objective | labels |
|---|---|---|
| `train_supervised` | CE | yes |
| `train_supcon` / `train_simclr` | SupCon t=0.1 / NT-Xent t=0.5 | yes / no |
| `train_simclr_sigreg` / `train_supcon_sigreg` | + lam*SIGReg | no / yes |
| `train_sigreg_ssl` ("lejepa") | MSE invariance + SIGReg | no |
| `train_sigreg_hybrid` | proto-CE to anchors + repulsion + classwise SIGReg; the supervised-SIGReg workhorse AND the discovery fine-tune | yes |
| `train_sigreg_residual_ssl` / `train_simclr_residual` / `train_supcon_sigreg_residual` | residual halves on z - means[y] | mixed |
| `train_linear_probe` / `train_binary_probe` | evaluation probes | -- |
| `collect_embeddings` / `collect_probs` | inference helpers (input-agnostic; work on TensorDatasets) | -- |

### Experiment-level trainers

| function | where | role |
|---|---|---|
| `train_hybrid(head, loader, ep, spec, labeled, lam, n_slices)` | `34h_hybrid_nplm_cifar.py` | generic HybridContrastiveLoss trainer; drives all exp-50/51 hybrid arms |
| `train_nplm_classwise(...)` | `53_nplm_classwise.py` | NPLM interaction + manual classwise SIGReg (two-label pattern) |
| `train_arm(name, ds, cfg, args, ...)` | `55_nplm_discovery.py` | reproduces any exp-50/53 arm by original seeds |
| `ft_loop(model, loader, epochs, step_fn, args, tag)` | `49_aircraft_ssl_ft.py:74` | THE end-to-end ft loop (Adam trunk 1e-5 / head 1e-3, cosine, AMP fp16) |
| step factories: `simclr_step`, `sigreg_ssl_step` (exp 49); `make_nplm_step(positives, critic, lam, tau, n_slices)` (exp 62); `make_supcon_step(lam_sigreg)` (exp 70); `make_residual_step(cents, lam)` (exp 49); `make_res_nplm_step(cents, lam, n_slices)` (exp 71) | | plug into `ft_loop`; each returns `(main, aux)` losses per two-view batch |

## 4. Fine-tuning strategies

| strategy | recipe | scripts |
|---|---|---|
| **Frozen-head** | freeze trunk, train `FeatureHead` on cached feature banks (plain + a8 augmented replicas), 120 ep, batch 512 | exps 37-48, 51 (8-arm suite), 69 |
| **End-to-end ft** | `FineTuneModel` + `ft_loop`, 20 ep, batch 32x2 views; exp 70 excludes holdout classes from the corpus (strict open-world) | 43 (closed-set ce/ss/supcon), 49 (ssl+residual), 62 (NPLM), 70 (6-arm grid, `--dataset --base`) |
| **Residual ft** | deepcopy supervised parent, freeze parent-head seen centroids, `ft_loop` with residual step on r = z - cent_y; eval residual head + concat [parent; residual] | 49 (aircraft), 71 (12-cell grid: supcon->res, ss->res, supcon->res-nplm) |
| **Discovery ft** | `discovery.run_discovery(backbone, means, base_ds, ...)`: pool >0.95 dist quantile -> `bic_select` k-means -> `merge_anchors` -> `PseudoDataset` + `train_sigreg_hybrid` proto/repulse, 2 rounds x 5 ep.  Input-agnostic: base_ds may be a TensorDataset of features (exps 69/70) | 33-36, 55 (NPLM), 56 (conf-mask), 58 (NPLM-ft variant), 68 (scratch), 69/70 |
| **Trunk reuse** | extract feature banks from a ft trunk, rerun the frozen-head suite on them: `build_features_ft` / `--trunk-ckpt-arm` (exp 51), `trunk_banks` (exp 70) | 51-ft, 70, 71 |

Checkpoint conventions: `{ds}_ft_{base}_{arm}.pt` (closed-set, exps 43/49/62),
`{ds}_ft_{base}_{arm}_seen.pt` (holdouts excluded, exps 70/71),
`scratch_{arm}_{ds}_{dim}d.pt` (exp 67).  Feature banks:
`data/tf_feats_{ds}_{base}_vitb16[_ft70_{arm}].pt`.

## 5. The test suite (full metric battery)

| metric | function | notes |
|---|---|---|
| probe (holdout novelty AUC) | `29_residual_finetune.py::linear_probe_novelty` | 3 seeds `torch.manual_seed(1000+s)`, report mean +- sd |
| acc / supAUC / eucl / mahaT / mahaPC | `29_residual_finetune.py::evaluate_space(tr, tr_lab, te, te_lab, anchors, seen, holdouts)` | anchors = seen-class centroids (`28_concat_residual.py::class_centroids` — returns a CUDA tensor; `fill_means` pads to full n_classes) |
| gaussianity | `supersig/metrics.py::gaussianity_summary` (+ `sliced_gaussianity`, `classwise_gaussianity`); table via `28::print_gauss_table` | seen classes, test set |
| per-event power | `30_power_curves.py::power_at_alpha(bg_scores, sig_scores, alpha)` | score = min distance to anchors (pre) or d_seen - d_disc (post-discovery) |
| SparKer battery | `31_sparker_power.py::run_test_battery(bg, sg, R, fractions, n_d, n_null, n_sig, alpha, seed, sparker_kw)` | ALWAYS annealed sigma (omit sigma0) across heterogeneous geometries — exp-57 lesson |
| Maha / MMD batteries | `32_maha_mmd_power.py::make_stats_fns` + `battery` | toy-based two-sample power |
| discovery diagnostics | `run_discovery` history dicts: purity, n_anchors, margin AUC, mean-anchor AUC | per round |

Conventions: alpha 0.05; fractions c10 0.001-0.1, c100 0.001-0.05,
transfer 0.003-0.1 (post grids 0.003-0.05); n_d 1000 (aircraft/flowers/
dtd) or 2000 (cars/galaxy10); n_null 200 pre / 100 post, 50 signal toys;
holdouts = last 10 classes (galaxy10: last 1; CIFAR: class 4 / holdout
sets per recipes).

## 6. Experiment index (which script runs which test)

| exp | script | suite |
|---|---|---|
| 50 | `50_nplm_cifar10_suite.py` | 8-arm CIFAR suite (c10/c100, `--scratch-base`, dims) |
| 51 | `51_nplm_aircraft_suite.py` | 8-arm transfer suite (`--dataset`, `--base`, `--trunk-ckpt-arm`) |
| 52/53 | `52_nplm_bilinear_scan.py` / `53_nplm_classwise.py` | lam/tau scan; classwise-SIGReg arms |
| 54 | `54_plot_space.py` | latent/corner plots of any arm |
| 55/56/58 | `55_nplm_discovery.py` / `56_nplm_confmask.py` / `58_nplm_ft_discovery.py` | discovery on CIFAR NPLM spaces (proto / conf-mask / NPLM-ft) |
| 57 | `57_sparker_diagnosis.py` | SparKer kernel-scale diagnosis (annealed-sigma rule) |
| 59/60 | `59_nplm_residual_concat.py` / `60_nplm_residual_transfer.py` | CIFAR / aircraft residual+concat constructions |
| 61 | `61_multiseed_probe.py` | paired multi-seed supcon vs nplm_bilinear |
| 62 | `62_aircraft_nplm_ft.py` | aircraft e2e NPLM ft (closed-set) |
| 63/64 | `63_supervised_baseline.py` / `64_pretrain_then_ft.py` | CE baselines; pretrain -> full CE-ft |
| 65/66 | `65_nplm_cw_learned_means.py` / `66_sup100.py` | learned repulsed means; sup recipe at any dim |
| 67/68 | `67_scratch_pretrain.py` / `68_scratch_discovery.py` | from-scratch resnet20 lineages + discovery |
| 69 | `69_cars_discovery.py` | frozen-trunk cars discovery (probe-negative) |
| 70 | `70_cars_ft_suite.py` | THE e2e ft grid: 6 arms x {cars,flowers,dtd,galaxy10} x {dino,lejepa,visreg}, battery pre+post discovery |
| 71 | `71_residual_ft_grid.py` | residual ft on exp-70 parents, 12 cells (beats discovery 12/12) |

Result archives: `logs/exp{NN}/*.npz`; condensed tables
`logs/SUMMARY_TABLES.md`; master tables `docs/CIFAR10_MASTER_TABLE.md`,
`docs/CIFAR100_MASTER_TABLE.md`, `docs/AIRCRAFT_MASTER_TABLE.md`
(transfer, incl. the exp-70/71 grids).
