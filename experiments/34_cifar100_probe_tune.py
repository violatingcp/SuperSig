"""
Experiment 34: tune the SIGReg element on CIFAR-100 against the holdout
1-layer-NN probe ROC AUC (the oracle linear-separability metric of exps 29/33).

Reference numbers to beat (exp 33 32-scan, holdout 4, pre-discovery probe):
  supcon 0.9232 / supcon+simclr 0.9368; best SIGReg arm ssl->supres 0.9178,
  plain sup 0.9039.

Stage 1 -- coordinate scan on the 32-D sup arm around the settled recipe
  (batch 25x24=600, sigreg_weight 1, 64 slices, 10 epochs): one knob at a
  time -- sigreg_weight {5, 20}, slices {256}, batch {50x24=1200, 99x24=2376},
  epochs {20, 30}.
Stage 2 -- combine the per-knob winners; add a step-compensated variant when
  the winning batch is larger (bigger balanced batches mean fewer optimizer
  steps per epoch).
Stage 3 -- retrain the exp-33 SIGReg arms (sup, sup->res, ssl->supres, joint)
  with the winning recipe and compare probes against same-run SupCon /
  SupCon+SimCLR baselines at settled defaults.

    python experiments/34_cifar100_probe_tune.py
    python experiments/34_cifar100_probe_tune.py --quick
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import copy
import importlib
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from supersig.config import DEVICE, plot_path
from supersig.data import (get_cifar_loaders, cifar_balanced_loader,
                           cifar_two_view_loader, cifar_two_view_balanced_loader)
from supersig.losses import make_anchors
from supersig.models import CIFARResNetBackbone
from supersig.metrics import gaussianity_summary
from supersig.recipes import recipe
from supersig.train import (train_sigreg_ssl, train_sigreg_hybrid,
                            train_sigreg_hybrid_aug, train_sigreg_residual_ssl,
                            train_supcon, train_simclr, collect_embeddings)

exp28 = importlib.import_module("28_concat_residual")
exp29 = importlib.import_module("29_residual_finetune")

REF = {"supcon": 0.9232, "supcon+simclr": 0.9368, "sup (exp33)": 0.9039,
       "ssl->supres (exp33)": 0.9178}


def probe_stat(tr, tr_lab, te, te_lab, holdouts, n_rep=3):
    """Mean/std probe AUC over n_rep probe-training seeds (frozen embeddings)."""
    aucs = []
    for s in range(n_rep):
        torch.manual_seed(1000 + s)
        a, _, _ = exp29.linear_probe_novelty(tr, tr_lab, te, te_lab, holdouts)
        aucs.append(a)
    return float(np.mean(aucs)), float(np.std(aucs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", type=int, default=4)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dim-single", type=int, default=32)
    ap.add_argument("--dim-half", type=int, default=16)
    args = ap.parse_args()
    ds = "cifar100"
    cfgS = recipe(ds, emb_dim=args.dim_single)
    cfgH = recipe(ds, emb_dim=args.dim_half)
    n_cls = cfgS["n_classes"]
    holdouts = {args.holdout}
    seen = [c for c in range(n_cls) if c not in holdouts]
    base_ep = 2 if args.quick else cfgS["ssl_epochs"]
    print(f"exp34 [{ds}] probe-AUC tuning, holdout={sorted(holdouts)}, "
          f"single={args.dim_single}d halves={args.dim_half}d")
    print("  references (exp33): " +
          ", ".join(f"{k}={v:.4f}" for k, v in REF.items()))

    train_loader, test_loader = get_cifar_loaders(quick=args.quick, dataset=ds)
    train_eval_loader = DataLoader(train_loader.dataset, batch_size=256,
                                   shuffle=False, num_workers=2)

    def train_sup_cfg(dim, c, seed, trunk=None, means_init=None):
        """Supervised SIGReg recipe with exposed batch/weight/slices/epochs."""
        torch.manual_seed(seed); np.random.seed(seed)
        cfg = cfgS if dim == args.dim_single else cfgH
        net = (copy.deepcopy(trunk) if trunk is not None else
               CIFARResNetBackbone(dim, arch=cfg["arch"],
                                   pretrain=ds).to(DEVICE))
        means = (means_init.clone() if means_init is not None else
                 make_anchors(cfg["pair_dist"] / math.sqrt(2.0), emb_dim=dim,
                              n_classes=n_cls).clone())
        loader = cifar_balanced_loader(ds, holdout=holdouts, quick=args.quick,
                                       classes_per_batch=c["cpb"],
                                       per_class=c["pc"])
        ep = 2 if args.quick else c["ep"]
        train_sigreg_hybrid(net, loader, ep, means, mode="repulse",
                            disc="proto", alpha=1.0,
                            rep_weight=cfg["rep_weight"],
                            sigreg_weight=c["w"], n_slices=c["s"])
        return net, means.detach()

    def space_embs(net, aug=None):
        tr, tr_lab = collect_embeddings(net, train_eval_loader)
        te, te_lab = collect_embeddings(net, test_loader)
        if aug is not None:
            tra, _ = collect_embeddings(aug, train_eval_loader)
            tea, _ = collect_embeddings(aug, test_loader)
            tr = np.concatenate([tr, tra], axis=1)
            te = np.concatenate([te, tea], axis=1)
        return tr, tr_lab, te, te_lab

    def eval_config(name, net, means, aug=None, anchors=None):
        tr, tr_lab, te, te_lab = space_embs(net, aug)
        anc = anchors if anchors is not None else means[seen]
        r = exp29.evaluate_space(tr, tr_lab, te, te_lab, anc, seen, holdouts)
        pm, psd = probe_stat(tr, tr_lab, te, te_lab, holdouts)
        g = gaussianity_summary(te, te_lab, seen, seed=args.seed)
        print(f"  [{name:<18}] probe={pm:.4f}+-{psd:.4f} acc={r['acc']:.4f} "
              f"supAUC={r['sup_auc']:.4f} eucl={r['eucl']:.4f}")
        return dict(probe=pm, probe_sd=psd, acc=r["acc"], sup_auc=r["sup_auc"],
                    eucl=r["eucl"], gauss=g, test=te, te_lab=te_lab)

    # ===== Stage 1: coordinate scan on the sup arm ==========================
    BASE = dict(cpb=25, pc=24, w=cfgS["sigreg_weight"], s=cfgS["n_slices"],
                ep=base_ep)
    SCAN = [("base", {}),
            ("w5", dict(w=5.0)), ("w20", dict(w=20.0)),
            ("s256", dict(s=256)),
            ("b1200", dict(cpb=50)), ("b2376", dict(cpb=99)),
            ("ep20", dict(ep=20)), ("ep30", dict(ep=30))]
    results, nets = {}, {}
    print("\n===== stage 1: coordinate scan (sup, "
          f"{args.dim_single}-D) =====")
    for name, delta in SCAN:
        c = dict(BASE); c.update(delta)
        print(f"\n----- {name}: batch={c['cpb']}x{c['pc']}="
              f"{c['cpb'] * c['pc']} w={c['w']} slices={c['s']} "
              f"ep={c['ep']} -----")
        net, means = train_sup_cfg(args.dim_single, c, args.seed)
        results[name] = eval_config(name, net, means)
        results[name]["cfg"] = c
        nets[name] = (net.cpu(), means)
        torch.cuda.empty_cache()

    # ===== Stage 2: combine per-knob winners ================================
    def best(names):
        pool = [n for n in names if n in results]
        return max(pool, key=lambda n: results[n]["probe"])
    comb = dict(BASE)
    comb["w"] = results[best(["base", "w5", "w20"])]["cfg"]["w"]
    comb["s"] = results[best(["base", "s256"])]["cfg"]["s"]
    comb["cpb"] = results[best(["base", "b1200", "b2376"])]["cfg"]["cpb"]
    comb["ep"] = results[best(["base", "ep20", "ep30"])]["cfg"]["ep"]
    stage2 = [("comb", comb)]
    if comb["cpb"] * comb["pc"] > 600:      # step-compensated long run
        c2 = dict(comb)
        c2["ep"] = min(40, round(comb["ep"] * comb["cpb"] * comb["pc"] / 600))
        stage2.append(("comb+steps", c2))
    seen_cfgs = {tuple(sorted(r["cfg"].items())) for r in results.values()}
    print("\n===== stage 2: combined winners =====")
    for name, c in stage2:
        if tuple(sorted(c.items())) in seen_cfgs:
            print(f"  {name}: duplicates a stage-1 config, skipped")
            continue
        seen_cfgs.add(tuple(sorted(c.items())))
        print(f"\n----- {name}: batch={c['cpb']}x{c['pc']}="
              f"{c['cpb'] * c['pc']} w={c['w']} slices={c['s']} "
              f"ep={c['ep']} -----")
        net, means = train_sup_cfg(args.dim_single, c, args.seed)
        results[name] = eval_config(name, net, means)
        results[name]["cfg"] = c
        nets[name] = (net.cpu(), means)
        torch.cuda.empty_cache()

    win = max(results, key=lambda n: results[n]["probe"])
    wcfg = results[win]["cfg"]
    print(f"\n===== scan summary (winner: {win}) =====")
    print(f"  {'config':<12}{'batch':>8}{'w':>6}{'slices':>8}{'ep':>5}"
          f"{'probe':>16}{'acc':>8}{'eucl':>8}")
    for name in results:
        c, r = results[name]["cfg"], results[name]
        print(f"  {name:<12}{c['cpb'] * c['pc']:>8}{c['w']:>6.0f}"
              f"{c['s']:>8}{c['ep']:>5}"
              f"{r['probe']:>9.4f}+-{r['probe_sd']:.4f}"
              f"{r['acc']:>8.4f}{r['eucl']:>8.4f}")

    print("\n===== gaussianity across scan configs (seen classes, test) =====")
    exp28.print_gauss_table({n: results[n]["gauss"] for n in results})

    order = list(results)
    plt.figure(figsize=(9, 5.5))
    xs = np.arange(len(order))
    plt.bar(xs, [results[n]["probe"] for n in order],
            yerr=[results[n]["probe_sd"] for n in order],
            color=["#d62728" if n == win else "#2a78d6" for n in order],
            capsize=3)
    for label, v, c in [("supcon+simclr (exp33)", REF["supcon+simclr"], "#e34948"),
                        ("supcon (exp33)", REF["supcon"], "#4a3aa7"),
                        ("sup default (exp33)", REF["sup (exp33)"], "gray")]:
        plt.axhline(v, color=c, ls="--", lw=1.2, label=label)
    plt.xticks(xs, order, rotation=30, ha="right")
    plt.ylim(0.75, 1.0)
    plt.ylabel("holdout probe ROC AUC (pre-discovery)")
    plt.title(f"exp34: CIFAR-100 sup-{args.dim_single}d SIGReg scan "
              f"(holdout {args.holdout})")
    plt.legend(fontsize=8); plt.grid(alpha=0.25, axis="y")
    plt.tight_layout()
    plt.savefig(plot_path("exp34_probe_scan_cifar100.png"), dpi=150)
    plt.close()
    print("saved", plot_path("exp34_probe_scan_cifar100.png"))

    # ===== Stage 3: winner recipe on all SIGReg arms vs baselines ===========
    print(f"\n===== stage 3: arms with winner recipe "
          f"(batch={wcfg['cpb']}x{wcfg['pc']} w={wcfg['w']} "
          f"s={wcfg['s']} ep={wcfg['ep']}) =====")
    ssl_ep = 2 if args.quick else 20
    res_ep = 2 if args.quick else 10
    sup_ep_dflt = 2 if args.quick else cfgS["ssl_epochs"]

    def cents_of(net):
        e, l = collect_embeddings(net, train_eval_loader)
        m = np.isin(l, seen)
        return exp28.class_centroids(e[m], l[m], seen)

    def backbone(dim):
        return CIFARResNetBackbone(dim, arch=cfgS["arch"],
                                   pretrain=ds).to(DEVICE)

    print("\n--- sup (winner, reused from scan) ---")
    supS = nets[win][0].to(DEVICE); means_supS = nets[win][1]
    print("\n--- sup-half (winner recipe) ---")
    supH, means_supH = train_sup_cfg(args.dim_half, wcfg, args.seed + 10)
    print("\n--- res-half (classwise residual post sup-half) ---")
    torch.manual_seed(args.seed + 11); np.random.seed(args.seed + 11)
    resH = copy.deepcopy(supH)
    train_sigreg_residual_ssl(
        resH, cifar_two_view_balanced_loader(ds, holdout=holdouts,
                                             quick=args.quick,
                                             classes_per_batch=wcfg["cpb"],
                                             per_class=wcfg["pc"]),
        res_ep, means_supH, n_slices=wcfg["s"], classwise=True)
    print("\n--- ssl-half (settled two-view SIGReg) ---")
    torch.manual_seed(args.seed + 12); np.random.seed(args.seed + 12)
    sslH = backbone(args.dim_half)
    train_sigreg_ssl(sslH, cifar_two_view_loader(quick=args.quick,
                                                 labeled=False,
                                                 holdout=holdouts,
                                                 dataset=ds), ssl_ep)
    sslH_cents = cents_of(sslH)
    print("\n--- supres-half (winner recipe post ssl trunk) ---")
    supresH, means_supresH = train_sup_cfg(
        args.dim_half, wcfg, args.seed + 13, trunk=sslH,
        means_init=exp28.fill_means(sslH_cents, seen, cfgH))
    print("\n--- joint (winner recipe, parallel sup x aug) ---")
    torch.manual_seed(args.seed + 4); np.random.seed(args.seed + 4)
    jointS = backbone(args.dim_single)
    means_jointS = make_anchors(cfgS["pair_dist"] / math.sqrt(2.0),
                                emb_dim=args.dim_single,
                                n_classes=n_cls).clone()
    train_sigreg_hybrid_aug(
        jointS, cifar_two_view_balanced_loader(ds, holdout=holdouts,
                                               quick=args.quick,
                                               classes_per_batch=wcfg["cpb"],
                                               per_class=wcfg["pc"]),
        2 if args.quick else wcfg["ep"], means_jointS,
        rep_weight=cfgS["rep_weight"], sigreg_weight=wcfg["w"],
        n_slices=wcfg["s"])
    means_jointS = means_jointS.detach()
    print("\n--- supcon (baseline, settled defaults) ---")
    torch.manual_seed(args.seed + 5); np.random.seed(args.seed + 5)
    supconS = backbone(args.dim_single)
    train_supcon(supconS, cifar_two_view_loader(quick=args.quick, labeled=True,
                                                holdout=holdouts, dataset=ds),
                 sup_ep_dflt)
    supconS_cents = cents_of(supconS)
    print("\n--- supcon-half + simclr-half (baseline, settled defaults) ---")
    torch.manual_seed(args.seed + 15); np.random.seed(args.seed + 15)
    supconH = backbone(args.dim_half)
    train_supcon(supconH, cifar_two_view_loader(quick=args.quick, labeled=True,
                                                holdout=holdouts, dataset=ds),
                 sup_ep_dflt)
    supconH_cents = cents_of(supconH)
    torch.manual_seed(args.seed + 16); np.random.seed(args.seed + 16)
    simclrH = backbone(args.dim_half)
    train_simclr(simclrH, cifar_two_view_loader(quick=args.quick,
                                                labeled=False,
                                                holdout=holdouts, dataset=ds),
                 ssl_ep)
    simclrH_cents = cents_of(simclrH)
    resH_cents = cents_of(resH)

    ARMS = {
        "sup->res*": (supH, means_supH, resH, resH_cents),
        "ssl->supres*": (supresH, means_supresH, sslH, sslH_cents),
        "joint*": (jointS, means_jointS, None, None),
        "sup*": (supS, means_supS, None, None),
        "supcon": (supconS, exp28.fill_means(supconS_cents, seen, cfgS),
                   None, None),
        "supcon+simclr": (supconH, exp28.fill_means(supconH_cents, seen, cfgH),
                          simclrH, simclrH_cents),
    }
    print("\n===== stage-3 table (*, winner recipe; baselines settled) =====")
    arm_results = {}
    for name, (net, means, aug, cents) in ARMS.items():
        anchors = (torch.cat([means[seen], cents], dim=1)
                   if aug is not None else means[seen])
        arm_results[name] = eval_config(name, net, means, aug=aug,
                                        anchors=anchors)

    print("\n===== gaussianity across stage-3 arms =====")
    exp28.print_gauss_table({n: arm_results[n]["gauss"] for n in arm_results})

    order3 = list(ARMS)
    plt.figure(figsize=(8.5, 5.5))
    xs = np.arange(len(order3))
    cols = ["#2a78d6" if "*" in n else "#4a3aa7" for n in order3]
    plt.bar(xs, [arm_results[n]["probe"] for n in order3],
            yerr=[arm_results[n]["probe_sd"] for n in order3],
            color=cols, capsize=3)
    plt.axhline(REF["supcon+simclr"], color="#e34948", ls="--", lw=1.2,
                label="supcon+simclr (exp33)")
    plt.axhline(REF["ssl->supres (exp33)"], color="#1baf7a", ls="--", lw=1.2,
                label="best SIGReg arm (exp33)")
    plt.xticks(xs, order3, rotation=20, ha="right")
    plt.ylim(0.75, 1.0)
    plt.ylabel("holdout probe ROC AUC (pre-discovery)")
    plt.title("exp34: tuned SIGReg arms (*) vs contrastive baselines, "
              "CIFAR-100")
    plt.legend(fontsize=8); plt.grid(alpha=0.25, axis="y")
    plt.tight_layout()
    plt.savefig(plot_path("exp34_probe_arms_cifar100.png"), dpi=150)
    plt.close()
    print("saved", plot_path("exp34_probe_arms_cifar100.png"))

    outdir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "logs", "exp34")
    os.makedirs(outdir, exist_ok=True)
    np.savez(os.path.join(outdir, "probe_scan.npz"),
             configs=np.array(list(results), dtype=object),
             probes=np.array([results[n]["probe"] for n in results]),
             probe_sds=np.array([results[n]["probe_sd"] for n in results]),
             arms=np.array(order3, dtype=object),
             arm_probes=np.array([arm_results[n]["probe"] for n in order3]),
             winner=np.array([win], dtype=object),
             allow_pickle=True)
    embs = {}
    for n in order3:
        key = n.replace("->", "_to_").replace("+", "_").replace("*", "_t")
        embs[f"{key}_test"] = arm_results[n]["test"]
    embs["te_lab"] = arm_results[order3[0]]["te_lab"]
    np.savez(os.path.join(outdir, "embeddings_stage3.npz"), **embs)
    print(f"saved {outdir}/probe_scan.npz, embeddings_stage3.npz")
    print(f"\nWINNER: {win}  batch={wcfg['cpb'] * wcfg['pc']} w={wcfg['w']} "
          f"slices={wcfg['s']} ep={wcfg['ep']}  "
          f"probe={results[win]['probe']:.4f}")


if __name__ == "__main__":
    main()
