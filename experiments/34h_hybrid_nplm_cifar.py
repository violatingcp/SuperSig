"""
Experiment 34h: HybridContrastiveLoss on CIFAR-100 (probe protocol of exp 34e).

Trains the configurable loss (supersig.losses.HybridContrastiveLoss) in several
corners of its design cube and evaluates each as a standalone feature space with
the holdout-novelty linear probe of exps 29/34.  The point is a like-for-like
comparison of the softmax (NT-Xent / SupCon) interaction against the calibrated
NPLM interaction, with SIGReg supplying the marginal in both.

Configs (edit CONFIGS or pass --configs):
  simclr_sigreg   instance / cosine   / softmax / sigreg   (reproduces exp-34e half)
  nplm_bilinear   instance / bilinear / nplm    / sigreg   (log-likelihood space)
  nplm_distance   instance / distance / nplm    / sigreg
  supcon_sigreg   supervised / cosine / softmax / sigreg
  nplm_sup_dist   supervised / distance / nplm  / sigreg

Smoke test (fast, tiny, just checks the pipeline runs end to end):
    python experiments/34h_hybrid_nplm_cifar.py --quick --configs nplm_bilinear

Full run (big GPU):
    python experiments/34h_hybrid_nplm_cifar.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import importlib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from supersig.config import DEVICE, plot_path
from supersig.data import get_cifar_loaders, cifar_two_view_loader
from supersig.models import CIFARResNetBackbone
from supersig.metrics import gaussianity_summary
from supersig.recipes import recipe
from supersig.losses import HybridContrastiveLoss
from supersig.train import collect_embeddings

exp28 = importlib.import_module("28_concat_residual")
exp29 = importlib.import_module("29_residual_finetune")

# name -> (loss kwargs, labeled?)  -- labeled drives supervised positives + loader
CONFIGS = {
    "simclr_sigreg": (dict(positives="instance",   critic="cosine",
                           estimator="softmax", marginal="sigreg", tau=0.5), False),
    "nplm_bilinear": (dict(positives="instance",   critic="bilinear",
                           estimator="nplm",    marginal="sigreg", tau=1.0), False),
    "nplm_distance": (dict(positives="instance",   critic="distance",
                           estimator="nplm",    marginal="sigreg", tau=1.0), False),
    "supcon_sigreg": (dict(positives="supervised", critic="cosine",
                           estimator="softmax", marginal="sigreg", tau=0.1), True),
    "nplm_sup_dist": (dict(positives="supervised", critic="distance",
                           estimator="nplm",    marginal="sigreg", tau=1.0), True),
}
REF = {"supcon+simclr (r1)": 0.9394, "supcon (r1)": 0.9268}


def train_hybrid(backbone, loader, epochs, loss_cfg, labeled, lam, n_slices,
                 lr=1e-3):
    """Generic contrastive loop driving HybridContrastiveLoss on raw embeddings."""
    loss_fn = HybridContrastiveLoss(lam=lam, n_slices=n_slices, **loss_cfg)
    opt = torch.optim.Adam(backbone.parameters(), lr=lr)
    backbone.train()
    for ep in range(epochs):
        int_run, marg_run, n = 0.0, 0.0, 0
        for batch in loader:
            if labeled:
                v1, v2, y = (t.to(DEVICE) for t in batch)
                labels = torch.cat([y, y])
            else:
                v1, v2 = batch[0].to(DEVICE), batch[1].to(DEVICE)
                inst = torch.arange(v1.size(0), device=DEVICE)
                labels = torch.cat([inst, inst])
            opt.zero_grad()
            z = backbone(torch.cat([v1, v2]))
            loss, parts = loss_fn(z, labels)
            loss.backward()
            opt.step()
            int_run += parts["interaction"].item() * v1.size(0)
            marg_run += parts["marginal"].item() * v1.size(0)
            n += v1.size(0)
        n = max(n, 1)
        print(f"  [{loss_cfg['estimator']}/{loss_cfg['critic']}] "
              f"epoch {ep+1}/{epochs}  interaction={int_run/n:.4f}  "
              f"marginal={marg_run/n:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--holdout", type=int, default=4)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS),
                    choices=list(CONFIGS))
    args = ap.parse_args()

    cfg = recipe(args.dataset, emb_dim=args.dim)
    n_cls = cfg["n_classes"]
    holdouts = {args.holdout}
    seen = [c for c in range(n_cls) if c not in holdouts]
    con_ep = args.epochs or (1 if args.quick else 20)
    print(f"exp34h [{args.dataset}] HybridContrastiveLoss, "
          f"holdout={sorted(holdouts)}, configs={args.configs}")

    _, test_loader = get_cifar_loaders(quick=args.quick, dataset=args.dataset)
    train_loader, _ = get_cifar_loaders(quick=args.quick, dataset=args.dataset)
    train_eval_loader = DataLoader(train_loader.dataset, batch_size=256,
                                   shuffle=False, num_workers=2)

    def probe_stat(tr, tr_lab, te, te_lab, n_rep=3):
        aucs = []
        for s in range(n_rep):
            torch.manual_seed(1000 + s)
            a, _, _ = exp29.linear_probe_novelty(tr, tr_lab, te, te_lab, holdouts)
            aucs.append(a)
        return float(np.mean(aucs)), float(np.std(aucs))

    results = {}
    for name in args.configs:
        loss_cfg, labeled = CONFIGS[name]
        print(f"\n----- {name}: {loss_cfg} (labeled={labeled}) -----")
        torch.manual_seed(args.seed); np.random.seed(args.seed)
        net = CIFARResNetBackbone(args.dim, arch=cfg["arch"],
                                  pretrain=args.dataset).to(DEVICE)
        loader = cifar_two_view_loader(quick=args.quick, labeled=labeled,
                                       holdout=holdouts, dataset=args.dataset)
        train_hybrid(net, loader, con_ep, loss_cfg, labeled,
                     lam=args.lam, n_slices=cfg["n_slices"])

        tr, tr_lab = collect_embeddings(net, train_eval_loader)
        te, te_lab = collect_embeddings(net, test_loader)
        m = np.isin(tr_lab, seen)
        cents = exp28.class_centroids(tr[m], tr_lab[m], seen)
        anchors = torch.as_tensor(cents, device=DEVICE)
        r = exp29.evaluate_space(tr, tr_lab, te, te_lab, anchors, seen, holdouts)
        pm, psd = probe_stat(tr, tr_lab, te, te_lab)
        g = gaussianity_summary(te, te_lab, seen, seed=args.seed)
        print(f"  [{name:<14}] probe={pm:.4f}+-{psd:.4f} acc={r['acc']:.4f} "
              f"supAUC={r['sup_auc']:.4f} eucl={r['eucl']:.4f} "
              f"mahaT={r['maha_tied']:.4f}")
        results[name] = dict(probe=pm, probe_sd=psd, acc=r["acc"],
                             sup_auc=r["sup_auc"], eucl=r["eucl"],
                             mahaT=r["maha_tied"], gauss=g)

    print("\n===== summary =====")
    for k, v in REF.items():
        print(f"  {k:<24} probe={v:.4f}  (reference)")
    for name, r in results.items():
        print(f"  {name:<24} probe={r['probe']:.4f}+-{r['probe_sd']:.4f}  "
              f"acc={r['acc']:.4f}  mahaT={r['mahaT']:.4f}")

    order = list(results)
    if order:
        plt.figure(figsize=(8.5, 5.5))
        xs = np.arange(len(order))
        plt.bar(xs, [results[n]["probe"] for n in order],
                yerr=[results[n]["probe_sd"] for n in order], color="#2a78d6",
                capsize=3)
        for (label, v), c in zip(REF.items(), ["#1baf7a", "#e34948"]):
            plt.axhline(v, color=c, ls="--", lw=1.2, label=label)
        plt.xticks(xs, order, rotation=15, ha="right")
        plt.ylabel("holdout probe ROC AUC (pre-discovery)")
        plt.title(f"exp34h: HybridContrastiveLoss, {args.dataset}")
        plt.legend(fontsize=8); plt.grid(alpha=0.25, axis="y")
        plt.tight_layout()
        out = plot_path(f"exp34h_hybrid_nplm_{args.dataset}.png")
        plt.savefig(out, dpi=150); plt.close()
        print("saved", out)


if __name__ == "__main__":
    main()
