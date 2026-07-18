"""
Experiment 34 round 2: follow the round-1 gradient. Round 1 showed the probe
rises with batch (2376 best) and falls with sigreg weight above 1 and with
longer supervised training. Round 2 therefore scans BELOW w=1 at the winning
batch, and tunes the two-view SSL-SIGReg trunk feeding the best arm
(ssl->supres): more SSL epochs and a lighter SSL sigreg lam.

Same-run round-1 references (holdout 4): sup* b2376 0.8968, ssl->supres*
0.9123, joint* 0.9078, supcon 0.9268, supcon+simclr 0.9394.

    python experiments/34b_cifar100_probe_round2.py
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
                           cifar_two_view_loader,
                           cifar_two_view_balanced_loader)
from supersig.losses import make_anchors
from supersig.models import CIFARResNetBackbone
from supersig.recipes import recipe
from supersig.train import (train_sigreg_ssl, train_sigreg_hybrid,
                            train_sigreg_hybrid_aug, collect_embeddings)

exp28 = importlib.import_module("28_concat_residual")
exp29 = importlib.import_module("29_residual_finetune")

REF = {"sup* (r1)": 0.8968, "ssl->supres* (r1)": 0.9123, "joint* (r1)": 0.9078,
       "supcon (r1)": 0.9268, "supcon+simclr (r1)": 0.9394}


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
    CPB, PC = 99, 24                      # round-1 winning batch (2376)
    sup_ep = 2 if args.quick else 10
    print(f"exp34b [{ds}] round 2, holdout={sorted(holdouts)}, "
          f"batch={CPB}x{PC}")
    print("  round-1 refs: " + ", ".join(f"{k}={v:.4f}"
                                         for k, v in REF.items()))

    train_loader, test_loader = get_cifar_loaders(quick=args.quick, dataset=ds)
    train_eval_loader = DataLoader(train_loader.dataset, batch_size=256,
                                   shuffle=False, num_workers=2)

    def probe_stat(tr, tr_lab, te, te_lab, n_rep=3):
        aucs = []
        for s in range(n_rep):
            torch.manual_seed(1000 + s)
            a, _, _ = exp29.linear_probe_novelty(tr, tr_lab, te, te_lab,
                                                 holdouts)
            aucs.append(a)
        return float(np.mean(aucs)), float(np.std(aucs))

    def space_embs(net, aug=None):
        tr, tr_lab = collect_embeddings(net, train_eval_loader)
        te, te_lab = collect_embeddings(net, test_loader)
        if aug is not None:
            tra, _ = collect_embeddings(aug, train_eval_loader)
            tea, _ = collect_embeddings(aug, test_loader)
            tr = np.concatenate([tr, tra], axis=1)
            te = np.concatenate([te, tea], axis=1)
        return tr, tr_lab, te, te_lab

    def eval_space(name, net, means, aug=None, anchors=None):
        tr, tr_lab, te, te_lab = space_embs(net, aug)
        anc = anchors if anchors is not None else means[seen]
        r = exp29.evaluate_space(tr, tr_lab, te, te_lab, anc, seen, holdouts)
        pm, psd = probe_stat(tr, tr_lab, te, te_lab)
        print(f"  [{name:<22}] probe={pm:.4f}+-{psd:.4f} acc={r['acc']:.4f} "
              f"supAUC={r['sup_auc']:.4f} eucl={r['eucl']:.4f}")
        return dict(probe=pm, probe_sd=psd, acc=r["acc"],
                    sup_auc=r["sup_auc"], eucl=r["eucl"])

    def train_sup(dim, w, ep, seed, trunk=None, means_init=None):
        torch.manual_seed(seed); np.random.seed(seed)
        cfg = cfgS if dim == args.dim_single else cfgH
        net = (copy.deepcopy(trunk) if trunk is not None else
               CIFARResNetBackbone(dim, arch=cfg["arch"],
                                   pretrain=ds).to(DEVICE))
        means = (means_init.clone() if means_init is not None else
                 make_anchors(cfg["pair_dist"] / math.sqrt(2.0), emb_dim=dim,
                              n_classes=n_cls).clone())
        loader = cifar_balanced_loader(ds, holdout=holdouts, quick=args.quick,
                                       classes_per_batch=CPB, per_class=PC)
        train_sigreg_hybrid(net, loader, 2 if args.quick else ep, means,
                            mode="repulse", disc="proto", alpha=1.0,
                            rep_weight=cfg["rep_weight"], sigreg_weight=w,
                            n_slices=cfg["n_slices"])
        return net, means.detach()

    def cents_of(net):
        e, l = collect_embeddings(net, train_eval_loader)
        m = np.isin(l, seen)
        return exp28.class_centroids(e[m], l[m], seen)

    results = {}

    # ----- A: sup weight scan below 1 ---------------------------------------
    print("\n===== A: sup sigreg_weight below 1 (batch 2376) =====")
    sup_nets = {}
    for w in (0.5, 0.2):
        name = f"sup w{w}"
        print(f"\n----- {name} -----")
        net, means = train_sup(args.dim_single, w, sup_ep, args.seed)
        results[name] = eval_space(name, net, means)
        sup_nets[w] = (net, means)
        torch.cuda.empty_cache()
    w_probe = {1.0: REF["sup* (r1)"]}
    w_probe.update({w: results[f"sup w{w}"]["probe"] for w in (0.5, 0.2)})
    w_best = max(w_probe, key=w_probe.get)
    print(f"\n  best sup weight: w={w_best} (probe {w_probe[w_best]:.4f})")

    # ----- B: SSL trunk variants feeding ssl->supres ------------------------
    print("\n===== B: SSL trunk variants -> supres arm =====")
    TRUNKS = [("ssl-40ep", dict(epochs=40, lam=1.0)),
              ("ssl-lam0.5", dict(epochs=20, lam=0.5))]
    for tname, tkw in TRUNKS:
        print(f"\n----- {tname} -----")
        torch.manual_seed(args.seed + 12); np.random.seed(args.seed + 12)
        ssl = CIFARResNetBackbone(args.dim_half, arch=cfgH["arch"],
                                  pretrain=ds).to(DEVICE)
        train_sigreg_ssl(ssl, cifar_two_view_loader(quick=args.quick,
                                                    labeled=False,
                                                    holdout=holdouts,
                                                    dataset=ds),
                         2 if args.quick else tkw["epochs"], lam=tkw["lam"])
        ssl_cents = cents_of(ssl)
        supres, means_supres = train_sup(
            args.dim_half, w_best, sup_ep, args.seed + 13, trunk=ssl,
            means_init=exp28.fill_means(ssl_cents, seen, cfgH))
        name = f"ssl->supres [{tname}]"
        anchors = torch.cat([means_supres[seen], ssl_cents], dim=1)
        results[name] = eval_space(name, supres, means_supres, aug=ssl,
                                   anchors=anchors)
        del ssl, supres
        torch.cuda.empty_cache()

    # ----- C: joint with the best weight ------------------------------------
    print(f"\n===== C: joint w={w_best} (batch 2376) =====")
    torch.manual_seed(args.seed + 4); np.random.seed(args.seed + 4)
    joint = CIFARResNetBackbone(args.dim_single, arch=cfgS["arch"],
                                pretrain=ds).to(DEVICE)
    means_joint = make_anchors(cfgS["pair_dist"] / math.sqrt(2.0),
                               emb_dim=args.dim_single, n_classes=n_cls).clone()
    train_sigreg_hybrid_aug(
        joint, cifar_two_view_balanced_loader(ds, holdout=holdouts,
                                              quick=args.quick,
                                              classes_per_batch=CPB,
                                              per_class=PC),
        sup_ep, means_joint, rep_weight=cfgS["rep_weight"],
        sigreg_weight=w_best, n_slices=cfgS["n_slices"])
    means_joint = means_joint.detach()
    results[f"joint w{w_best}"] = eval_space(f"joint w{w_best}", joint,
                                             means_joint)

    # ----- summary ----------------------------------------------------------
    print("\n===== round-2 summary =====")
    for k, v in REF.items():
        print(f"  {k:<26} probe={v:.4f}  (reference)")
    for name, r in results.items():
        print(f"  {name:<26} probe={r['probe']:.4f}+-{r['probe_sd']:.4f}  "
              f"acc={r['acc']:.4f}")

    order = list(results)
    plt.figure(figsize=(9, 5.5))
    xs = np.arange(len(order))
    plt.bar(xs, [results[n]["probe"] for n in order],
            yerr=[results[n]["probe_sd"] for n in order], color="#2a78d6",
            capsize=3)
    plt.axhline(REF["supcon+simclr (r1)"], color="#e34948", ls="--", lw=1.2,
                label="supcon+simclr (r1)")
    plt.axhline(REF["supcon (r1)"], color="#4a3aa7", ls="--", lw=1.2,
                label="supcon (r1)")
    plt.axhline(REF["ssl->supres* (r1)"], color="#1baf7a", ls="--", lw=1.2,
                label="ssl->supres* (r1)")
    plt.xticks(xs, order, rotation=25, ha="right")
    plt.ylim(0.75, 1.0)
    plt.ylabel("holdout probe ROC AUC (pre-discovery)")
    plt.title("exp34 round 2: sub-unity weights + SSL trunk tuning, CIFAR-100")
    plt.legend(fontsize=8); plt.grid(alpha=0.25, axis="y")
    plt.tight_layout()
    plt.savefig(plot_path("exp34_probe_round2_cifar100.png"), dpi=150)
    plt.close()
    print("saved", plot_path("exp34_probe_round2_cifar100.png"))

    outdir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "logs", "exp34")
    os.makedirs(outdir, exist_ok=True)
    np.savez(os.path.join(outdir, "probe_round2.npz"),
             names=np.array(order, dtype=object),
             probes=np.array([results[n]["probe"] for n in order]),
             probe_sds=np.array([results[n]["probe_sd"] for n in order]),
             allow_pickle=True)
    print(f"saved {outdir}/probe_round2.npz")


if __name__ == "__main__":
    main()
