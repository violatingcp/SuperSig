"""
Experiment 34e: structural hybrids of the contrastive and SIGReg families
(CIFAR-100, holdout 4, probe protocol of exps 34-34d).

New feature halves:
  hybrid16      : SimCLR NT-Xent + SIGReg-to-N(0,I) on one trunk (lam 1 / 5)
                  -- contrastive features in a calibrated Gaussian geometry.
  res-simclr16  : SimCLR trained on the residual z - mean_y of the SupCon
                  half (frozen SupCon centroids) -- the contrastive analog of
                  the exp-28 residual arm.

Evaluated 16+16 spaces:
  supcon+hybrid[lam]   : [supcon16 ; hybrid16]     (drop-in for supcon+simclr)
  hybrid->supres       : [supres16 ; hybrid16]     (supres warm-started FROM
                                                    the hybrid trunk, the
                                                    coupling that won exp 34d)
  supcon+res-simclr    : [supcon16 ; res-simclr16]

References (same holdout/seeds, exps 34-34c): supcon+simclr 0.9394,
supcon 0.9268, ssl->supres[40ep trunk] 0.9213.

    python experiments/34e_hybrid_contrastive.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import copy
import importlib
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from supersig.config import DEVICE, plot_path
from supersig.data import (get_cifar_loaders, cifar_balanced_loader,
                           cifar_two_view_loader)
from supersig.models import CIFARResNetBackbone
from supersig.metrics import gaussianity_summary
from supersig.recipes import recipe
from supersig.train import (train_sigreg_hybrid, train_supcon,
                            train_simclr_sigreg, train_simclr_residual,
                            collect_embeddings)

exp28 = importlib.import_module("28_concat_residual")
exp29 = importlib.import_module("29_residual_finetune")

REF = {"ssl->supres [40ep] (r2)": 0.9213, "supcon (r1)": 0.9268,
       "supcon+simclr (r1)": 0.9394}
CPB, PC = 99, 24


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", type=int, default=4)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dim-half", type=int, default=16)
    args = ap.parse_args()
    ds = "cifar100"
    cfgH = recipe(ds, emb_dim=args.dim_half)
    n_cls = cfgH["n_classes"]
    holdouts = {args.holdout}
    seen = [c for c in range(n_cls) if c not in holdouts]
    sup_ep = 2 if args.quick else 10
    con_ep = 2 if args.quick else 20
    print(f"exp34e [{ds}] hybrid contrastive halves, "
          f"holdout={sorted(holdouts)}")
    print("  refs: " + ", ".join(f"{k}={v:.4f}" for k, v in REF.items()))

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

    def cents_of(net):
        e, l = collect_embeddings(net, train_eval_loader)
        m = np.isin(l, seen)
        return exp28.class_centroids(e[m], l[m], seen)

    def backbone():
        return CIFARResNetBackbone(args.dim_half, arch=cfgH["arch"],
                                   pretrain=ds).to(DEVICE)

    # ----- networks ---------------------------------------------------------
    print("\n----- supcon16 (baseline half, settled defaults) -----")
    torch.manual_seed(args.seed + 15); np.random.seed(args.seed + 15)
    supcon = backbone()
    train_supcon(supcon, cifar_two_view_loader(quick=args.quick, labeled=True,
                                               holdout=holdouts, dataset=ds),
                 sup_ep)
    supcon_cents = cents_of(supcon)

    hybrids = {}
    for lam in (1.0, 5.0):
        print(f"\n----- hybrid16 simclr+sigreg (lam={lam}) -----")
        torch.manual_seed(args.seed + 16); np.random.seed(args.seed + 16)
        h = backbone()
        train_simclr_sigreg(h, cifar_two_view_loader(quick=args.quick,
                                                     labeled=False,
                                                     holdout=holdouts,
                                                     dataset=ds),
                            con_ep, lam=lam, n_slices=cfgH["n_slices"])
        hybrids[lam] = (h, cents_of(h))

    print("\n----- res-simclr16 (SimCLR on SupCon residuals) -----")
    torch.manual_seed(args.seed + 17); np.random.seed(args.seed + 17)
    res_simclr = copy.deepcopy(supcon)
    means_supcon = exp28.fill_means(supcon_cents, seen, cfgH).to(DEVICE)
    train_simclr_residual(res_simclr,
                          cifar_two_view_loader(quick=args.quick,
                                                labeled=True,
                                                holdout=holdouts, dataset=ds),
                          con_ep, means_supcon)
    res_simclr_cents = cents_of(res_simclr)

    print("\n----- supres16 warm-started from hybrid(lam=1) trunk -----")
    torch.manual_seed(args.seed + 13); np.random.seed(args.seed + 13)
    supres = copy.deepcopy(hybrids[1.0][0])
    means_supres = exp28.fill_means(hybrids[1.0][1], seen, cfgH).clone()
    train_sigreg_hybrid(supres,
                        cifar_balanced_loader(ds, holdout=holdouts,
                                              quick=args.quick,
                                              classes_per_batch=CPB,
                                              per_class=PC),
                        sup_ep, means_supres, mode="repulse", disc="proto",
                        alpha=1.0, rep_weight=cfgH["rep_weight"],
                        sigreg_weight=1.0, n_slices=cfgH["n_slices"])
    means_supres = means_supres.detach()

    # ----- evaluation -------------------------------------------------------
    ARMS = {
        "supcon+hybrid[lam1]": (supcon, supcon_cents,
                                hybrids[1.0][0], hybrids[1.0][1]),
        "supcon+hybrid[lam5]": (supcon, supcon_cents,
                                hybrids[5.0][0], hybrids[5.0][1]),
        "hybrid->supres": (supres, means_supres[seen],
                           hybrids[1.0][0], hybrids[1.0][1]),
        "supcon+res-simclr": (supcon, supcon_cents,
                              res_simclr, res_simclr_cents),
    }
    results = {}
    print("\n===== performance / novelty / probe table =====")
    for name, (a_net, a_anc, b_net, b_anc) in ARMS.items():
        tr, tr_lab = collect_embeddings(a_net, train_eval_loader)
        te, te_lab = collect_embeddings(a_net, test_loader)
        trb, _ = collect_embeddings(b_net, train_eval_loader)
        teb, _ = collect_embeddings(b_net, test_loader)
        tr = np.concatenate([tr, trb], axis=1)
        te = np.concatenate([te, teb], axis=1)
        anchors = torch.cat([torch.as_tensor(a_anc, device=DEVICE),
                             torch.as_tensor(b_anc, device=DEVICE)], dim=1)
        r = exp29.evaluate_space(tr, tr_lab, te, te_lab, anchors, seen,
                                 holdouts)
        pm, psd = probe_stat(tr, tr_lab, te, te_lab)
        g = gaussianity_summary(te, te_lab, seen, seed=args.seed)
        print(f"  [{name:<20}] probe={pm:.4f}+-{psd:.4f} acc={r['acc']:.4f} "
              f"supAUC={r['sup_auc']:.4f} eucl={r['eucl']:.4f} "
              f"mahaT={r['maha_tied']:.4f} mahaPC={r['maha_pc']:.4f}")
        results[name] = dict(probe=pm, probe_sd=psd, acc=r["acc"],
                             sup_auc=r["sup_auc"], eucl=r["eucl"],
                             mahaT=r["maha_tied"], mahaPC=r["maha_pc"],
                             gauss=g)

    print("\n===== gaussianity across arms (seen classes, test) =====")
    exp28.print_gauss_table({n: results[n]["gauss"] for n in results})

    print("\n===== summary =====")
    for k, v in REF.items():
        print(f"  {k:<26} probe={v:.4f}  (reference)")
    for name, r in results.items():
        print(f"  {name:<26} probe={r['probe']:.4f}+-{r['probe_sd']:.4f}  "
              f"acc={r['acc']:.4f}  mahaT={r['mahaT']:.4f}")

    order = list(results)
    plt.figure(figsize=(8.5, 5.5))
    xs = np.arange(len(order))
    plt.bar(xs, [results[n]["probe"] for n in order],
            yerr=[results[n]["probe_sd"] for n in order], color="#2a78d6",
            capsize=3)
    for (label, v), c in zip(REF.items(), ["#1baf7a", "#4a3aa7", "#e34948"]):
        plt.axhline(v, color=c, ls="--", lw=1.2, label=label)
    plt.xticks(xs, order, rotation=15, ha="right")
    plt.ylim(0.80, 0.97)
    plt.ylabel("holdout probe ROC AUC (pre-discovery)")
    plt.title("exp34e: hybrid contrastive halves, CIFAR-100")
    plt.legend(fontsize=8); plt.grid(alpha=0.25, axis="y")
    plt.tight_layout()
    plt.savefig(plot_path("exp34e_hybrid_probe_cifar100.png"), dpi=150)
    plt.close()
    print("saved", plot_path("exp34e_hybrid_probe_cifar100.png"))

    outdir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "logs", "exp34")
    os.makedirs(outdir, exist_ok=True)
    np.savez(os.path.join(outdir, "probe_hybrid.npz"),
             names=np.array(order, dtype=object),
             probes=np.array([results[n]["probe"] for n in order]),
             probe_sds=np.array([results[n]["probe_sd"] for n in order]),
             mahaT=np.array([results[n]["mahaT"] for n in order]),
             allow_pickle=True)
    print(f"saved {outdir}/probe_hybrid.npz")


if __name__ == "__main__":
    main()
