"""
Experiment 52: lam/tau scan for the label-free NPLM-bilinear space on
CIFAR-100.

Exp 50 found nplm_bilinear (instance / bilinear / nplm / sigreg, lam=1,
tau=1) within 0.007 probe AUC of SupCon on CIFAR-100 without labels, while
holding the best raw-distance novelty and gaussianity.  This scans the two
knobs -- lam (SIGReg marginal weight) and tau (critic temperature) -- on the
exp-50 protocol (CIFAR-100-pretrained ResNet-20 -> 32-D, holdout 4, 20
epochs), reporting the Part A metric suite plus per-event power per cell.
Every cell reuses the same seed, so comparisons are paired.  The heavy
sparker/maha/mmd batteries are skipped; rerun exp 50 on the winner for those.

    python experiments/52_nplm_bilinear_scan.py
    python experiments/52_nplm_bilinear_scan.py --quick --lams 1 --taus 0.5,1
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import importlib
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from supersig.config import DEVICE, plot_path
from supersig.data import get_cifar_loaders, cifar_two_view_loader
from supersig.models import CIFARResNetBackbone
from supersig.metrics import gaussianity_summary
from supersig.recipes import recipe
from supersig.train import collect_embeddings

exp28 = importlib.import_module("28_concat_residual")
exp29 = importlib.import_module("29_residual_finetune")
exp30 = importlib.import_module("30_power_curves")
exp34h = importlib.import_module("34h_hybrid_nplm_cifar")

# exp-50 CIFAR-100 references (32d, holdout 4, seed 0)
REF = {"supcon (exp50)": 0.9417, "nplm_bilinear lam1/tau1 (exp50)": 0.9349}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--holdout", type=int, default=4)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lams", default="0.5,1,2,5")
    ap.add_argument("--taus", default="0.5,1,2")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()
    ds = args.dataset

    cfg = recipe(ds, emb_dim=args.dim)
    n_cls = cfg["n_classes"]
    holdouts = {args.holdout}
    seen = [c for c in range(n_cls) if c not in holdouts]
    con_ep = args.epochs or (2 if args.quick else 20)
    lams = [float(x) for x in args.lams.split(",")]
    taus = [float(x) for x in args.taus.split(",")]
    print(f"exp52 [{ds}] nplm_bilinear scan, dim={args.dim}, epochs={con_ep}, "
          f"holdout={sorted(holdouts)}, lams={lams}, taus={taus}")

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

    results = {}
    for lam in lams:
        for tau in taus:
            key = (lam, tau)
            print(f"\n----- lam={lam} tau={tau} -----")
            torch.manual_seed(args.seed + 20); np.random.seed(args.seed + 20)
            net = CIFARResNetBackbone(args.dim, arch=cfg["arch"],
                                      pretrain=ds).to(DEVICE)
            loader = cifar_two_view_loader(quick=args.quick, labeled=False,
                                           holdout=holdouts, dataset=ds)
            loss_cfg = dict(positives="instance", critic="bilinear",
                            estimator="nplm", marginal="sigreg", tau=tau)
            exp34h.train_hybrid(net, loader, con_ep, loss_cfg, False,
                                lam=lam, n_slices=cfg["n_slices"])

            tr, tr_lab = collect_embeddings(net, train_eval_loader)
            te, te_lab = collect_embeddings(net, test_loader)
            m = np.isin(tr_lab, seen)
            cents = exp28.class_centroids(tr[m], tr_lab[m], seen)
            anchors = torch.as_tensor(cents, dtype=torch.float32,
                                      device=DEVICE)
            r = exp29.evaluate_space(tr, tr_lab, te, te_lab, anchors, seen,
                                     holdouts)
            pm, psd = probe_stat(tr, tr_lab, te, te_lab)
            g = gaussianity_summary(te, te_lab, seen, seed=args.seed)
            d = torch.cdist(torch.as_tensor(te, dtype=torch.float32,
                                            device=DEVICE), anchors)
            s = d.min(1).values.cpu().numpy()
            pe = exp30.power_at_alpha(s[np.isin(te_lab, seen)],
                                      s[np.isin(te_lab, list(holdouts))],
                                      args.alpha)
            print(f"  [lam={lam} tau={tau}] probe={pm:.4f}+-{psd:.4f} "
                  f"acc={r['acc']:.4f} eucl={r['eucl']:.4f} "
                  f"mahaT={r['maha_tied']:.4f} perevent={pe:.3f} "
                  f"SWratio={g['sw_ratio_mean']:.2f}")
            results[key] = dict(probe=pm, probe_sd=psd, acc=r["acc"],
                                sup_auc=r["sup_auc"], eucl=r["eucl"],
                                mahaT=r["maha_tied"], mahaPC=r["maha_pc"],
                                perevent=pe, sw_ratio=g["sw_ratio_mean"])
            del net
            torch.cuda.empty_cache()

    print("\n===== scan table =====")
    print(f"  {'lam':>6}{'tau':>6}{'probe':>16}{'acc':>8}{'eucl':>8}"
          f"{'mahaT':>8}{'perevt':>8}{'SWrat':>8}")
    for (lam, tau), r in results.items():
        print(f"  {lam:>6}{tau:>6}{r['probe']:>9.4f}+-{r['probe_sd']:.4f}"
              f"{r['acc']:>8.4f}{r['eucl']:>8.4f}{r['mahaT']:>8.4f}"
              f"{r['perevent']:>8.3f}{r['sw_ratio']:>8.2f}")
    for k, v in REF.items():
        print(f"  {k:<28} probe={v:.4f}  (reference)")

    for metric in ("probe", "mahaT"):
        grid = np.array([[results[(lam, tau)][metric] for tau in taus]
                         for lam in lams])
        plt.figure(figsize=(6.5, 5))
        plt.imshow(grid, cmap="viridis", aspect="auto")
        for i in range(len(lams)):
            for j in range(len(taus)):
                plt.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center",
                         color="white", fontsize=9)
        plt.xticks(range(len(taus)), taus); plt.yticks(range(len(lams)), lams)
        plt.xlabel("tau"); plt.ylabel("lam")
        plt.colorbar()
        plt.title(f"exp52 nplm_bilinear {ds}: {metric}")
        plt.tight_layout()
        out = plot_path(f"exp52_scan_{metric}_{ds}.png")
        plt.savefig(out, dpi=150); plt.close()
        print("saved", out)

    os.makedirs(os.path.join("logs", "exp52"), exist_ok=True)
    np.savez(os.path.join("logs", "exp52", f"scan_nplm_bilinear_{ds}.npz"),
             lams=np.array(lams), taus=np.array(taus),
             **{m: np.array([[results[(lam, tau)][m] for tau in taus]
                             for lam in lams])
                for m in ("probe", "probe_sd", "acc", "sup_auc", "eucl",
                          "mahaT", "mahaPC", "perevent", "sw_ratio")})
    print("Done.")


if __name__ == "__main__":
    main()
