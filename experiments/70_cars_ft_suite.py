"""
Experiment 70: END-TO-END fine-tuning suite (cars / flowers / galaxy10 /
dtd) -- unsupervised and supervised pretraining for the simclr / sigreg /
nplm families -- with the full novelty battery pre AND post discovery.

Unlike exps 51/69 (frozen DINO trunk, head-only training), here the whole
ViT-B/16 fine-tunes on the target dataset (exp-49/62 recipe: trunk 1e-5 /
head 1e-3, Adam + cosine, AMP fp16, 20 epochs, batch 32x2 views, 100-D
head).  The fine-tuning corpus EXCLUDES the holdout classes (last 10;
galaxy10: last 1 of its 10) -- images and labels -- so holdout novelty
stays genuinely novel.  Six arms:

  unsupervised   simclr-ft      NT-Xent temp 0.5 (instance positives)
                 sigreg-ssl-ft  MSE invariance + global SIGReg (LeJEPA)
                 nplm-bil-ft    instance/bilinear NPLM + lam=1 SIGReg
  supervised     supcon-ft      SupCon temp 0.1
                 ss-ft          SupCon + lam=5 SIGReg (exp-43 champion)
                 nplm-sup-ft    supervised/distance NPLM + lam=1 SIGReg

Per arm: extract ft-trunk 768-d banks + 100-D head embeddings, report the
exp-51 battery (3-seed holdout probe, nearest-centroid acc, supAUC, eucl,
mahaT/PC, gaussianity, per-event/SparKer/Maha/MMD pre powers), then run the
settled discovery loop (exp-69 feature-space protocol: the head is the
discovery backbone over the arm's own ft-trunk features) -- natural
discovery probe/eucl/mahaT pre->post + the injected post-power grid
(annealed-sigma SparKer).

    python experiments/70_cars_ft_suite.py
    python experiments/70_cars_ft_suite.py --dataset flowers
    python experiments/70_cars_ft_suite.py --quick --arms supcon-ft
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import copy
import importlib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from supersig.config import DATA_DIR, DEVICE, REPO_DIR, plot_path
from supersig.losses import sigreg_loss, supcon_loss
from supersig.metrics import gaussianity_summary
from supersig.discovery import run_discovery
from supersig.train import collect_embeddings

exp28 = importlib.import_module("28_concat_residual")
exp29 = importlib.import_module("29_residual_finetune")
exp30 = importlib.import_module("30_power_curves")
exp31 = importlib.import_module("31_sparker_power")
exp32 = importlib.import_module("32_maha_mmd_power")
exp37 = importlib.import_module("37_dtd_vit")
exp43 = importlib.import_module("43_dtd_finetune")
exp44 = importlib.import_module("44_transfer_32d")
exp49 = importlib.import_module("49_aircraft_ssl_ft")
exp62 = importlib.import_module("62_aircraft_nplm_ft")

CKPT_DIR = os.path.join(REPO_DIR, "checkpoints")
DS = "cars"          # set from --dataset in main()
BASE = "dino"        # set from --base in main()
STATS = ["perevent", "sparker", "maha", "mmd"]
REP_WEIGHT = 20.0

# arm -> (labeled?, step factory)  -- step(model, v1, v2, y) -> (main, aux)
def make_supcon_step(lam_sigreg, n_slices):
    def step(model, v1, v2, y):
        x = torch.cat([v1, v2]).to(DEVICE, non_blocking=True)
        yy = torch.cat([y, y]).to(DEVICE)
        z = model(x).float()
        con = supcon_loss(F.normalize(z, dim=1), yy, temp=0.1)
        reg = (lam_sigreg * sigreg_loss(z, n_slices=n_slices) if lam_sigreg
               else torch.zeros((), device=DEVICE))
        return con, reg
    return step


def arm_specs(args):
    return {
        "simclr-ft": (False, exp49.simclr_step),
        "sigreg-ssl-ft": (False, exp49.sigreg_ssl_step),
        "nplm-bil-ft": (False, exp62.make_nplm_step(
            "instance", "bilinear", args.lam, args.tau, args.n_slices)),
        "supcon-ft": (True, make_supcon_step(0.0, args.n_slices)),
        "ss-ft": (True, make_supcon_step(5.0, args.n_slices)),
        "nplm-sup-ft": (True, exp62.make_nplm_step(
            "supervised", "distance", args.lam, args.tau, args.n_slices)),
    }


COLORS = {"simclr-ft": "#0072b2", "sigreg-ssl-ft": "#666666",
          "nplm-bil-ft": "#e51e1e", "supcon-ft": "#eda100",
          "ss-ft": "#008300", "nplm-sup-ft": "#8c2d9e"}


def eval_split(split, transform):
    """Plain-transform eval dataset; 'train' = the full labeled train pool
    (dtd: train+val, matching exp43.train_corpus)."""
    if DS == "dtd":
        if split == "train":
            return exp43.labeled_corpus(transform)
        from torchvision import datasets as tvd
        return tvd.DTD(DATA_DIR, split="test", download=True,
                       transform=transform)
    return exp44.make_split(DS, split, transform)


def corpus_labels(ds):
    """Per-index labels of a train corpus without decoding images
    (torchvision _labels / StanfordCars _samples / exp-44 parquet sets)."""
    if hasattr(ds, "_labels"):
        return np.asarray(ds._labels)
    if hasattr(ds, "_samples"):
        return np.asarray([s[1] for s in ds._samples])
    if hasattr(ds, "df") and hasattr(ds, "lab_col"):
        if hasattr(ds, "keep"):
            return np.asarray([int(ds.df.iloc[k][ds.lab_col])
                               for k in ds.keep])
        return ds.df[ds.lab_col].to_numpy().astype(int)
    if hasattr(ds, "datasets"):        # ConcatDataset
        return np.concatenate([corpus_labels(d) for d in ds.datasets])
    return np.asarray([ds[i][1] for i in range(len(ds))])


def seen_two_view_loader(corpus, seen_idx, args):
    ds = exp43.TwoViewLabeledImages(Subset(corpus, seen_idx))
    return DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                      num_workers=8, persistent_workers=True, drop_last=True,
                      pin_memory=True)


def seed_sfx(args):
    s = getattr(args, "seed", 0)
    return f"_s{s}" if s else ""


def trunk_banks(model, arm, args):
    """(train, test) 768-d plain-transform banks from the arm's ft trunk."""
    cache = os.path.join(DATA_DIR, f"tf_feats_{DS}_{BASE}_ft70_{arm}"
                         f"{seed_sfx(args)}"
                         f"{'_quick' if args.quick else ''}.pt")
    if os.path.exists(cache) and not args.refresh:
        return torch.load(cache)
    trunk = model.trunk.eval()
    plain = {}
    for split in ("train", "test"):
        d = eval_split(split, exp37.TF_EVAL)
        plain[split] = exp37.extract(trunk, d)
        print(f"  extracted ft70_{arm} {split}: "
              f"{tuple(plain[split][0].shape)}")
    torch.save(plain, cache)
    return plain


def main():
    global DS, BASE
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cars",
                    choices=["cars", "aircraft", "flowers", "galaxy10",
                             "dtd"])
    ap.add_argument("--base", default="dino",
                    choices=["dino", "lejepa", "visreg"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", nargs="+", default=list(COLORS),
                    choices=list(COLORS))
    ap.add_argument("--ft-epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr-backbone", type=float, default=1e-5)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--emb-dim", type=int, default=100)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--n-slices", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--pre-fractions", default="0.003,0.01,0.02,0.05,0.1")
    ap.add_argument("--post-fractions", default="0.003,0.01,0.02,0.05")
    ap.add_argument("--n-d", type=int, default=None,
                    help="toy size; default 2000 cars/galaxy10, else 1000")
    ap.add_argument("--kernels", type=int, default=16)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--skip-power", action="store_true")
    ap.add_argument("--skip-discovery", action="store_true")
    args = ap.parse_args()
    DS, BASE = args.dataset, args.base
    args.ft_epochs = args.ft_epochs or (1 if args.quick else 20)
    if args.n_d is None:
        args.n_d = 2000 if DS in ("cars", "galaxy10") else 1000

    N_CLS = 47 if DS == "dtd" else exp44.N_CLASSES[DS]
    n_hold = 1 if DS == "galaxy10" else 10
    holdouts = set(range(N_CLS - n_hold, N_CLS))
    seen = [c for c in range(N_CLS) if c not in holdouts]
    ft_ep_disc = 1 if args.quick else 5
    pre_fracs = [float(x) for x in args.pre_fractions.split(",")]
    post_fracs = [float(x) for x in args.post_fractions.split(",")]
    n_null_pre = 20 if args.quick else 200
    n_null_post = 20 if args.quick else 100
    n_sig_toys = 10 if args.quick else 50
    sparker_kw = dict(M=args.kernels, steps=args.steps)   # annealed sigma
    cfg = dict(n_classes=N_CLS, pair_dist=5.0, sigreg_weight=1.0,
               n_slices=args.n_slices,
               rep_weight=REP_WEIGHT * 45.0 / (N_CLS * (N_CLS - 1) / 2))
    specs = arm_specs(args)
    tag = f"{DS}_{BASE}_ft70"
    print(f"exp70 [{tag}] end-to-end ft suite, arms={args.arms}, "
          f"ft_epochs={args.ft_epochs}, emb={args.emb_dim}, "
          f"holdouts {min(holdouts)}-{max(holdouts)} EXCLUDED from ft")

    corpus = exp43.train_corpus(DS)
    ytr_all = corpus_labels(corpus)
    seen_idx_corpus = np.where(~np.isin(ytr_all, list(holdouts)))[0].tolist()
    print(f"  ft corpus: {len(seen_idx_corpus)}/{len(ytr_all)} train images "
          f"(holdout classes removed)")

    # ===== Phase A+B: fine-tune each arm, extract banks, pre metrics ========
    results, trains, tests, anchors_of = {}, {}, {}, {}
    heads, banks = {}, {}
    tr_lab = te_lab = None
    for i, arm in enumerate(args.arms):
        labeled, step = specs[arm]
        print(f"\n===== [{arm}] ({'supervised' if labeled else 'unsupervised'}"
              f") =====")
        torch.manual_seed(args.seed + 20 + i)
        np.random.seed(args.seed + 20 + i)
        model = exp43.FineTuneModel(BASE, args.emb_dim)
        ckpt = os.path.join(CKPT_DIR, f"{DS}_ft_{BASE}_{arm}_seen"
                            f"{seed_sfx(args)}"
                            f"{'_quick' if args.quick else ''}.pt")
        if os.path.exists(ckpt) and not args.refresh:
            print(f"  loading {ckpt}")
            model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        else:
            loader = seen_two_view_loader(corpus, seen_idx_corpus, args)
            exp49.ft_loop(model, loader, args.ft_epochs, step, args, arm)
            torch.save(model.state_dict(), ckpt)
            del loader
        plain = trunk_banks(model, arm, args)
        (Xtr, ytr), (Xte, yte) = plain["train"], plain["test"]
        tr_lab, te_lab = ytr.numpy(), yte.numpy()
        head = copy.deepcopy(model.head).float().to(DEVICE)
        heads[arm], banks[arm] = head, plain
        del model
        torch.cuda.empty_cache()

        tr = exp37.embed(head, Xtr.float()).numpy()
        te = exp37.embed(head, Xte.float()).numpy()
        trains[arm], tests[arm] = tr, te
        m = np.isin(tr_lab, seen)
        cents = exp28.class_centroids(tr[m], tr_lab[m], seen)
        anchors_of[arm] = cents.detach().float().to(DEVICE)
        r = exp29.evaluate_space(tr, tr_lab, te, te_lab, anchors_of[arm],
                                 seen, holdouts)
        aucs = []
        for s in range(3):
            torch.manual_seed(1000 + s)
            a, _, _ = exp29.linear_probe_novelty(tr, tr_lab, te, te_lab,
                                                 holdouts)
            aucs.append(a)
        pm, psd = float(np.mean(aucs)), float(np.std(aucs))
        g = gaussianity_summary(te, te_lab, seen, seed=args.seed)
        print(f"  [{arm:<14}] probe={pm:.4f}+-{psd:.4f} acc={r['acc']:.4f} "
              f"supAUC={r['sup_auc']:.4f} eucl={r['eucl']:.4f} "
              f"mahaT={r['maha_tied']:.4f} mahaPC={r['maha_pc']:.4f}")
        results[arm] = dict(probe=pm, probe_sd=psd, acc=r["acc"],
                            sup_auc=r["sup_auc"], eucl=r["eucl"],
                            mahaT=r["maha_tied"], mahaPC=r["maha_pc"],
                            gauss=g)

    print("\n===== PRE table =====")
    print(f"  {'arm':<16}{'probe':>16}{'acc':>8}{'supAUC':>8}{'eucl':>8}"
          f"{'mahaT':>8}{'mahaPC':>8}")
    for arm in args.arms:
        r = results[arm]
        print(f"  {arm:<16}{r['probe']:>9.4f}+-{r['probe_sd']:.4f}"
              f"{r['acc']:>8.4f}{r['sup_auc']:>8.4f}{r['eucl']:>8.4f}"
              f"{r['mahaT']:>8.4f}{r['mahaPC']:>8.4f}")
    print("\n===== gaussianity (seen classes, test set) =====")
    exp28.print_gauss_table({n: results[n]["gauss"] for n in args.arms})

    seen_idx = np.where(np.isin(tr_lab, seen))[0]
    sig_idx_all = np.where(np.isin(tr_lab, list(holdouts)))[0]

    # ===== pre-discovery power batteries ====================================
    pre_power = {s: {} for s in STATS}
    if not args.skip_power:
        print("\n===== PRE power batteries =====")
        for arm in args.arms:
            tr, te = trains[arm], tests[arm]
            bg_mask = np.isin(te_lab, seen)
            sig_mask = np.isin(te_lab, list(holdouts))
            d = torch.cdist(torch.as_tensor(te, dtype=torch.float32,
                                            device=DEVICE), anchors_of[arm])
            s = d.min(1).values.cpu().numpy()
            pe = exp30.power_at_alpha(s[bg_mask], s[sig_mask], args.alpha)
            pre_power["perevent"][arm] = [pe] * len(pre_fracs)
            print(f"  [{arm}] per-event pre power={pe:.3f}")
            R = torch.as_tensor(tr[np.isin(tr_lab, seen)][:20000],
                                dtype=torch.float32, device=DEVICE)
            bg = torch.as_tensor(te[bg_mask], dtype=torch.float32,
                                 device=DEVICE)
            sg = torch.as_tensor(te[sig_mask], dtype=torch.float32,
                                 device=DEVICE)
            print(f"  [{arm}] sparker (annealed)")
            pre_power["sparker"][arm], _ = exp31.run_test_battery(
                bg, sg, R, pre_fracs, args.n_d, n_null_pre, n_sig_toys,
                args.alpha, args.seed, sparker_kw, tag="pre-spk")
            maha_fn, mmd_fn, n_bg, n_sig = exp32.make_stats_fns(
                tr, tr_lab, te, te_lab, seen, holdouts, args.seed)
            print(f"  [{arm}] maha")
            pre_power["maha"][arm], _ = exp32.battery(
                maha_fn, n_bg, n_sig, pre_fracs, args.n_d, n_null_pre,
                n_sig_toys, args.alpha, args.seed, tag="pre-maha")
            print(f"  [{arm}] mmd")
            pre_power["mmd"][arm], _ = exp32.battery(
                mmd_fn, n_bg, n_sig, pre_fracs, args.n_d, n_null_pre,
                n_sig_toys, args.alpha, args.seed, tag="pre-mmd")

        for stat in STATS:
            print(f"\n===== EXP70 {stat.upper()} PRE POWER =====")
            print(f"  {'arm':<16}" + "".join(f"{f:>9}" for f in pre_fracs))
            for arm in args.arms:
                print(f"  {arm:<16}"
                      + "".join(f"{p:>9.3f}" for p in pre_power[stat][arm]))
            plt.figure(figsize=(8, 6.5))
            for arm in args.arms:
                plt.plot(pre_fracs, pre_power[stat][arm], "-o",
                         color=COLORS[arm], lw=2, ms=5, label=arm)
            plt.xscale("log")
            plt.axhline(args.alpha, color="gray", lw=1, ls=":")
            plt.xlabel("injected anomaly fraction")
            plt.ylabel(f"power at alpha={args.alpha}")
            plt.title(f"exp70 cars ft suite ({tag}): {stat} pre power")
            plt.grid(alpha=0.25, which="both")
            plt.legend(loc="upper left", fontsize=8, ncol=2)
            plt.tight_layout()
            plt.savefig(plot_path(f"exp70_{stat}_power_pre_{tag}.png"),
                        dpi=150)
            plt.close()
            print("  saved " + plot_path(f"exp70_{stat}_power_pre_{tag}.png"))

    # ===== discovery: natural (probe/eucl/mahaT post) + post-power grid =====
    post_results, hist = {}, {}
    post_power = {s: {n: [] for n in args.arms} for s in STATS}
    if not args.skip_discovery:
        for arm in args.arms:
            print(f"\n----- natural discovery: {arm} -----")
            (Xtr, ytr), (Xte, yte) = banks[arm]["train"], banks[arm]["test"]
            base_feats = TensorDataset(Xtr.float(), ytr)
            train_eval_loader = DataLoader(base_feats, batch_size=512,
                                           shuffle=False)
            test_loader = DataLoader(TensorDataset(Xte.float(), yte),
                                     batch_size=512, shuffle=False)
            m = np.isin(tr_lab, seen)
            cents = exp28.class_centroids(trains[arm][m], tr_lab[m], seen)
            means0 = exp28.fill_means(cents, seen, cfg).detach()
            bb = copy.deepcopy(heads[arm])
            _, hist[arm] = run_discovery(
                bb, means0.clone(), base_ds=base_feats,
                train_eval_loader=train_eval_loader, test_loader=test_loader,
                seen=seen, holdouts=holdouts, dataset_name=DS,
                rep_weight=cfg["rep_weight"],
                sigreg_weight=cfg["sigreg_weight"],
                n_slices=cfg["n_slices"], rounds=args.rounds,
                ft_epochs=ft_ep_disc, names=None, seed=args.seed)
            tr_post, _ = collect_embeddings(bb, train_eval_loader)
            te_post, _ = collect_embeddings(bb, test_loader)
            cents_p = exp28.class_centroids(tr_post[m], tr_lab[m], seen)
            anch_p = cents_p.detach().float().to(DEVICE)
            rp = exp29.evaluate_space(tr_post, tr_lab, te_post, te_lab,
                                      anch_p, seen, holdouts)
            aucs = []
            for s in range(3):
                torch.manual_seed(1000 + s)
                a, _, _ = exp29.linear_probe_novelty(tr_post, tr_lab, te_post,
                                                     te_lab, holdouts)
                aucs.append(a)
            post_results[arm] = dict(probe=float(np.mean(aucs)),
                                     probe_sd=float(np.std(aucs)),
                                     acc=rp["acc"], eucl=rp["eucl"],
                                     mahaT=rp["maha_tied"],
                                     mahaPC=rp["maha_pc"])
            print(f"  [{arm}] probe {results[arm]['probe']:.4f} -> "
                  f"{post_results[arm]['probe']:.4f}  eucl "
                  f"{results[arm]['eucl']:.4f} -> {rp['eucl']:.4f}  mahaT "
                  f"{results[arm]['mahaT']:.4f} -> {rp['maha_tied']:.4f}")
            del bb
            torch.cuda.empty_cache()

        if not args.skip_power:
            for i_f, f in enumerate(post_fracs):
                n_inj = int(round(f * len(seen_idx) / (1.0 - f)))
                rng = np.random.default_rng(args.seed * 1000 + i_f)
                inj = rng.choice(sig_idx_all,
                                 size=min(n_inj, len(sig_idx_all)),
                                 replace=False)
                sub_idx = np.concatenate([seen_idx, inj])
                print(f"\n===== POST grid, f={f} ({len(inj)} injected) =====")
                for arm in args.arms:
                    (Xtr, ytr), (Xte, yte) = (banks[arm]["train"],
                                              banks[arm]["test"])
                    sub = TensorDataset(Xtr[sub_idx].float(), ytr[sub_idx])
                    tel_loader = DataLoader(sub, batch_size=512,
                                            shuffle=False)
                    test_loader = DataLoader(TensorDataset(Xte.float(), yte),
                                             batch_size=512, shuffle=False)
                    train_eval_loader = DataLoader(
                        TensorDataset(Xtr.float(), ytr), batch_size=512,
                        shuffle=False)
                    m = np.isin(tr_lab, seen)
                    cents = exp28.class_centroids(trains[arm][m], tr_lab[m],
                                                  seen)
                    means0 = exp28.fill_means(cents, seen, cfg).detach()
                    bb = copy.deepcopy(heads[arm])
                    cur_means, _ = run_discovery(
                        bb, means0.clone(), base_ds=sub,
                        train_eval_loader=tel_loader, test_loader=test_loader,
                        seen=seen, holdouts=holdouts, dataset_name=DS,
                        rep_weight=cfg["rep_weight"],
                        sigreg_weight=cfg["sigreg_weight"],
                        n_slices=cfg["n_slices"], rounds=args.rounds,
                        ft_epochs=ft_ep_disc, names=None, seed=args.seed)
                    te_post, tel_post = collect_embeddings(bb, test_loader)
                    tr_post, trl_post = collect_embeddings(bb,
                                                           train_eval_loader)
                    zt = torch.as_tensor(te_post, dtype=torch.float32,
                                         device=DEVICE)
                    d_seen = torch.cdist(zt, cur_means[seen]).min(1).values
                    d_disc = (torch.cdist(zt,
                                          cur_means[N_CLS:]).min(1).values
                              if cur_means.size(0) > N_CLS else
                              torch.full_like(d_seen, float("inf")))
                    bg_mask = np.isin(tel_post, seen)
                    sig_mask = np.isin(tel_post, list(holdouts))
                    s = (d_seen - d_disc).cpu().numpy()
                    pe = exp30.power_at_alpha(s[bg_mask], s[sig_mask],
                                              args.alpha)
                    post_power["perevent"][arm].append(pe)
                    print(f"  [{arm}] per-event post f={f}: power={pe:.3f}")
                    R = torch.as_tensor(tr_post[np.isin(trl_post,
                                                        seen)][:20000],
                                        dtype=torch.float32, device=DEVICE)
                    bg = torch.as_tensor(te_post[bg_mask],
                                         dtype=torch.float32, device=DEVICE)
                    sg = torch.as_tensor(te_post[sig_mask],
                                         dtype=torch.float32, device=DEVICE)
                    print(f"  [{arm}] sparker (post, annealed)")
                    p, _ = exp31.run_test_battery(bg, sg, R, [f], args.n_d,
                                                  n_null_post, n_sig_toys,
                                                  args.alpha,
                                                  args.seed + i_f,
                                                  sparker_kw, tag="post-spk")
                    post_power["sparker"][arm].append(p[0])
                    maha_fn, mmd_fn, n_bg, n_sig = exp32.make_stats_fns(
                        tr_post, trl_post, te_post, tel_post, seen, holdouts,
                        args.seed + i_f)
                    print(f"  [{arm}] maha (post)")
                    p, _ = exp32.battery(maha_fn, n_bg, n_sig, [f], args.n_d,
                                         n_null_post, n_sig_toys, args.alpha,
                                         args.seed + i_f, tag="post-maha")
                    post_power["maha"][arm].append(p[0])
                    print(f"  [{arm}] mmd (post)")
                    p, _ = exp32.battery(mmd_fn, n_bg, n_sig, [f], args.n_d,
                                         n_null_post, n_sig_toys, args.alpha,
                                         args.seed + i_f, tag="post-mmd")
                    post_power["mmd"][arm].append(p[0])
                    del bb
                    torch.cuda.empty_cache()

            for stat in STATS:
                print(f"\n===== EXP70 {stat.upper()} POST POWER =====")
                print(f"  {'arm':<16}"
                      + "".join(f"{f:>9}" for f in post_fracs))
                for arm in args.arms:
                    print(f"  {arm:<16}"
                          + "".join(f"{p:>9.3f}"
                                    for p in post_power[stat][arm]))
                plt.figure(figsize=(8, 6.5))
                for arm in args.arms:
                    plt.plot(post_fracs, post_power[stat][arm], "-o",
                             color=COLORS[arm], lw=2, ms=5,
                             label=f"{arm} post")
                plt.xscale("log")
                plt.axhline(args.alpha, color="gray", lw=1, ls=":")
                plt.xlabel("injected anomaly fraction")
                plt.ylabel(f"power at alpha={args.alpha}")
                plt.title(f"exp70 cars ft suite ({tag}): {stat} post power")
                plt.grid(alpha=0.25, which="both")
                plt.legend(loc="upper left", fontsize=8, ncol=2)
                plt.tight_layout()
                plt.savefig(plot_path(f"exp70_{stat}_power_post_{tag}.png"),
                            dpi=150)
                plt.close()
                print("  saved "
                      + plot_path(f"exp70_{stat}_power_post_{tag}.png"))

    # ===== summary + npz ====================================================
    print(f"\n===== EXP70 SUMMARY [{tag}] =====")
    for arm in args.arms:
        r = results[arm]
        line = (f"  [{arm:<14}] probe={r['probe']:.4f}+-{r['probe_sd']:.4f} "
                f"acc={r['acc']:.4f} eucl={r['eucl']:.4f} "
                f"mahaT={r['mahaT']:.4f}")
        if arm in post_results:
            p = post_results[arm]
            line += (f"  || post: probe={p['probe']:.4f} "
                     f"eucl={p['eucl']:.4f} mahaT={p['mahaT']:.4f}")
        print(line)
        for h in hist.get(arm, []):
            print(f"          round {h['round']}: purity={h['purity']:.3f} "
                  f"anchors={h['n_anchors']}  margin={h['margin']:.4f}")

    xs = np.arange(len(args.arms))
    plt.figure(figsize=(9, 5.5))
    w = 0.38
    plt.bar(xs - w / 2, [results[n]["probe"] for n in args.arms], w,
            yerr=[results[n]["probe_sd"] for n in args.arms],
            color=[COLORS[n] for n in args.arms], capsize=3, label="pre")
    if post_results:
        plt.bar(xs + w / 2,
                [post_results.get(n, {}).get("probe", np.nan)
                 for n in args.arms], w,
                color=[COLORS[n] for n in args.arms], alpha=0.55,
                hatch="//", label="post-discovery")
    plt.xticks(xs, args.arms, rotation=15, ha="right")
    plt.ylabel("holdout probe ROC AUC")
    plt.title(f"exp70: cars end-to-end ft suite ({tag})")
    plt.legend()
    plt.grid(alpha=0.25, axis="y")
    plt.tight_layout()
    out = plot_path(f"exp70_probe_{tag}.png")
    plt.savefig(out, dpi=150); plt.close()
    print("saved", out)

    os.makedirs(os.path.join("logs", "exp70"), exist_ok=True)
    np.savez(
        os.path.join("logs", "exp70", f"results_{tag}.npz"),
        arms=np.array(args.arms), pre_fractions=np.array(pre_fracs),
        post_fractions=np.array(post_fracs),
        **{f"{k}_{n}": np.array(results[n][k]) for n in args.arms
           for k in ("probe", "probe_sd", "acc", "sup_auc", "eucl",
                     "mahaT", "mahaPC")},
        **{f"post_{k}_{n}": np.array(post_results[n][k])
           for n in post_results
           for k in ("probe", "probe_sd", "acc", "eucl", "mahaT", "mahaPC")},
        **{f"{s}_{n}_pre": np.array(pre_power[s][n]) for s in STATS
           for n in args.arms if n in pre_power[s]},
        **{f"{s}_{n}_post": np.array(post_power[s][n]) for s in STATS
           for n in args.arms if post_power[s][n]})
    print("Done.")


if __name__ == "__main__":
    main()
