"""
Experiment 65: nplm_dist_sup_cw with LEARNED, REPULSED means (CIFAR-100).

Exp 53's classwise-NPLM arm used fixed make_anchors means, which are
unrealizable for 100 classes at 32-D (centroids stall ~3.3 away; sibling
centroids merge, capping acc/maha).  This makes the means learnable and
adds the supervised-recipe mean-geometry regularizer (Coulomb repulsion +
shrink, rep_weight from the recipe), so the anchor layout can adapt:

  L = NPLM_interaction(z; supervised, distance)
      + lam * classwise_sigreg(z, y, means)
      + rep_weight * repulsion(means) + SHRINK * shrink(means)

Protocol otherwise identical to exp 53 (32-D, holdout 4, 20 epochs,
balanced two-view loader, same seed).  Reports the usual battery (Part A +
4 pre power batteries, fixed sigma=1 SparKer for like-for-like with the
exp-53 row) plus mean-geometry diagnostics.

    python experiments/65_nplm_cw_learned_means.py
    python experiments/65_nplm_cw_learned_means.py --quick
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import importlib
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")

from supersig.config import DEVICE, plot_path
from supersig.data import (get_cifar_loaders, cifar_two_view_balanced_loader)
from supersig.losses import (HybridContrastiveLoss, classwise_sigreg_loss,
                             make_anchors, repulsion_loss, shrink_loss,
                             mean_geometry)
from supersig.models import CIFARResNetBackbone
from supersig.metrics import gaussianity_summary
from supersig.recipes import recipe
from supersig.train import collect_embeddings
import supersig.train as T

exp28 = importlib.import_module("28_concat_residual")
exp29 = importlib.import_module("29_residual_finetune")
exp30 = importlib.import_module("30_power_curves")
exp31 = importlib.import_module("31_sparker_power")
exp32 = importlib.import_module("32_maha_mmd_power")

SHRINK = getattr(T, "SHRINK_WEIGHT", 1e-3)

# exp-53 fixed-anchor reference row (C100, same protocol)
REF = dict(probe=0.8864, acc=0.3471, eucl=0.4525, mahaT=0.3717,
           perevent=0.020)


def train_nplm_cw_learned(backbone, loader, epochs, means, rep_weight,
                          tau=1.0, lam=1.0, n_slices=64, lr=1e-3):
    loss_fn = HybridContrastiveLoss(positives="supervised", critic="distance",
                                    estimator="nplm", marginal="none",
                                    tau=tau)
    means.requires_grad_(True)
    opt = torch.optim.Adam(list(backbone.parameters()) + [means], lr=lr)
    backbone.train()
    for ep in range(epochs):
        int_run, cw_run, n = 0.0, 0.0, 0
        for v1, v2, y in loader:
            v1, v2, y = v1.to(DEVICE), v2.to(DEVICE), y.to(DEVICE)
            cls_lab = torch.cat([y, y])
            opt.zero_grad()
            z = backbone(torch.cat([v1, v2]))
            inter, _ = loss_fn(z, cls_lab)
            cw = classwise_sigreg_loss(z, cls_lab, means, n_slices=n_slices)
            aux = rep_weight * repulsion_loss(means) + SHRINK * shrink_loss(means)
            (inter + lam * cw + aux).backward()
            opt.step()
            int_run += inter.item() * v1.size(0)
            cw_run += cw.item() * v1.size(0)
            n += v1.size(0)
        n = max(n, 1)
        mn, mmean = mean_geometry(means.detach())
        print(f"  [nplm-cw-learned] epoch {ep+1}/{epochs}  "
              f"interaction={int_run/n:.4f}  classwise={cw_run/n:.4f}  "
              f"means min/mean dist={mn:.2f}/{mmean:.2f}")
    return means.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--holdout", type=int, default=4)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--fractions", default="0.001,0.003,0.01,0.02,0.05")
    ap.add_argument("--n-d", type=int, default=5000)
    ap.add_argument("--kernels", type=int, default=16)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--sparker-sigma", type=float, default=1.0)
    ap.add_argument("--skip-power", action="store_true")
    args = ap.parse_args()
    ds = args.dataset

    cfg = recipe(ds, emb_dim=args.dim)
    n_cls = cfg["n_classes"]
    holdouts = {args.holdout}
    seen = [c for c in range(n_cls) if c not in holdouts]
    con_ep = args.epochs or (2 if args.quick else 20)
    fractions = [float(x) for x in args.fractions.split(",")]
    n_null_pre = 20 if args.quick else 200
    n_sig_toys = 10 if args.quick else 50
    sparker_kw = dict(M=args.kernels, steps=args.steps)
    if args.sparker_sigma > 0:
        sparker_kw.update(sigma0=args.sparker_sigma, sigma_ratio=1.0,
                          n_checkpoints=1)
    name = "nplm_dist_sup_cw_learned"
    print(f"exp65 [{ds}] {name}, dim={args.dim}, epochs={con_ep}, "
          f"holdout={sorted(holdouts)}, rep_weight={cfg['rep_weight']:.4f}")

    train_loader, test_loader = get_cifar_loaders(quick=args.quick, dataset=ds)
    train_eval_loader = DataLoader(train_loader.dataset, batch_size=256,
                                   shuffle=False, num_workers=2)

    torch.manual_seed(args.seed + 22); np.random.seed(args.seed + 22)
    net = CIFARResNetBackbone(args.dim, arch=cfg["arch"],
                              pretrain=ds).to(DEVICE)
    means = make_anchors(cfg["pair_dist"] / math.sqrt(2.0), emb_dim=args.dim,
                         n_classes=n_cls).clone()
    loader = cifar_two_view_balanced_loader(ds, holdout=holdouts,
                                            quick=args.quick)
    means = train_nplm_cw_learned(net, loader, con_ep, means,
                                  rep_weight=cfg["rep_weight"], tau=args.tau,
                                  lam=args.lam, n_slices=cfg["n_slices"])

    tr, tr_lab = collect_embeddings(net, train_eval_loader)
    te, te_lab = collect_embeddings(net, test_loader)
    m = np.isin(tr_lab, seen)
    cents = exp28.class_centroids(tr[m], tr_lab[m], seen)
    anchors = torch.as_tensor(cents, dtype=torch.float32, device=DEVICE)
    d_anchor = float((anchors - means[seen]).norm(dim=1).mean())
    r = exp29.evaluate_space(tr, tr_lab, te, te_lab, anchors, seen, holdouts)
    aucs = []
    for s in range(3):
        torch.manual_seed(1000 + s)
        a, _, _ = exp29.linear_probe_novelty(tr, tr_lab, te, te_lab, holdouts)
        aucs.append(a)
    pm, psd = float(np.mean(aucs)), float(np.std(aucs))
    g = gaussianity_summary(te, te_lab, seen, seed=args.seed)
    d = torch.cdist(torch.as_tensor(te, dtype=torch.float32, device=DEVICE),
                    anchors)
    s_ = d.min(1).values.cpu().numpy()
    pe = exp30.power_at_alpha(s_[np.isin(te_lab, seen)],
                              s_[np.isin(te_lab, list(holdouts))], args.alpha)
    print(f"\n  [{name}] probe={pm:.4f}+-{psd:.4f} acc={r['acc']:.4f} "
          f"supAUC={r['sup_auc']:.4f} eucl={r['eucl']:.4f} "
          f"mahaT={r['maha_tied']:.4f} mahaPC={r['maha_pc']:.4f} "
          f"perevt={pe:.3f} cent->anchor={d_anchor:.2f}")
    print(f"  fixed-anchor ref (exp53): probe={REF['probe']} "
          f"acc={REF['acc']} eucl={REF['eucl']} mahaT={REF['mahaT']} "
          f"perevt={REF['perevent']}")
    print(f"  gauss: class RMS mean={g['class_rms_mean']:.3f} "
          f"SW ratio={g['sw_ratio_mean']:.2f} "
          f"sep={g['separation']:.2f}"
          if all(k in g for k in ("class_rms_mean", "sw_ratio_mean",
                                  "separation"))
          else f"  gauss keys: {sorted(g)}")
    exp28.print_gauss_table({name: g})

    pre_power = {}
    if not args.skip_power:
        print("\n===== PRE power batteries =====")
        bg_mask = np.isin(te_lab, seen)
        sig_mask = np.isin(te_lab, list(holdouts))
        pre_power["perevent"] = [pe] * len(fractions)
        R = torch.as_tensor(tr[np.isin(tr_lab, seen)][:20000],
                            dtype=torch.float32, device=DEVICE)
        bg = torch.as_tensor(te[bg_mask], dtype=torch.float32, device=DEVICE)
        sg = torch.as_tensor(te[sig_mask], dtype=torch.float32,
                             device=DEVICE)
        print(f"  [{name}] sparker")
        pre_power["sparker"], _ = exp31.run_test_battery(
            bg, sg, R, fractions, args.n_d, n_null_pre, n_sig_toys,
            args.alpha, args.seed, sparker_kw, tag="pre-spk")
        maha_fn, mmd_fn, n_bg, n_sig = exp32.make_stats_fns(
            tr, tr_lab, te, te_lab, seen, holdouts, args.seed)
        print(f"  [{name}] maha")
        pre_power["maha"], _ = exp32.battery(
            maha_fn, n_bg, n_sig, fractions, args.n_d, n_null_pre,
            n_sig_toys, args.alpha, args.seed, tag="pre-maha")
        print(f"  [{name}] mmd")
        pre_power["mmd"], _ = exp32.battery(
            mmd_fn, n_bg, n_sig, fractions, args.n_d, n_null_pre,
            n_sig_toys, args.alpha, args.seed, tag="pre-mmd")
        print(f"\n===== EXP65 PRE POWER =====")
        print(f"  {'stat':<10}" + "".join(f"{f:>9}" for f in fractions))
        for st in ("perevent", "sparker", "maha", "mmd"):
            print(f"  {st:<10}"
                  + "".join(f"{p:>9.3f}" for p in pre_power[st]))

    os.makedirs(os.path.join("logs", "exp65"), exist_ok=True)
    np.savez(os.path.join("logs", "exp65", f"learned_means_{ds}.npz"),
             fractions=np.array(fractions), probe=pm, probe_sd=psd,
             acc=r["acc"], sup_auc=r["sup_auc"], eucl=r["eucl"],
             mahaT=r["maha_tied"], mahaPC=r["maha_pc"], perevent=pe,
             d_anchor=d_anchor,
             **{f"{st}_pre": np.array(v) for st, v in pre_power.items()})
    print("Done.")


if __name__ == "__main__":
    main()
