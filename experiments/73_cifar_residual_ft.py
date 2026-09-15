"""
Experiment 73: exp-71 residual FINE-TUNING on CIFAR-10 / CIFAR-100.

Exp 59 trained fresh residual halves at split dims (16+16 / 50+50).
Here the transfer-grid recipe (exp 71) comes to CIFAR: train the
supervised parent (exp-50 arm, original seed), freeze its seen-class
centroids, DEEPCOPY the whole backbone and fine-tune it end-to-end on
the residual r = z - cent_y at FULL dim, then evaluate the residual
space and the concat [parent ; residual] (2*dim).

Constructions per dataset: supcon->res, supcon_sigreg->res,
supcon->res-nplm.  Residual objectives as in exps 49/59/71:
  res       NT-Xent (temp 0.5) on normalized r + lam=5 SIGReg(r)
  res-nplm  instance/bilinear NPLM on r + lam=1 SIGReg(r)
Backbones: hub-pretrained CIFARResNetBackbone; residual ft Adam 1e-3,
20 epochs (repo standard).  Battery: 3-seed holdout probe, acc, eucl,
mahaT, per-event.  Holdout class 4 (settled convention).

    python experiments/73_cifar_residual_ft.py --dataset cifar10 --dim 32
    python experiments/73_cifar_residual_ft.py --dataset cifar100 --dim 100
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import copy
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
from supersig.losses import (HybridContrastiveLoss, sigreg_loss,
                             supcon_loss)
from supersig.recipes import recipe
from supersig.train import collect_embeddings

exp28 = importlib.import_module("28_concat_residual")
exp29 = importlib.import_module("29_residual_finetune")
exp30 = importlib.import_module("30_power_curves")
exp55 = importlib.import_module("55_nplm_discovery")

RUNS = [("supcon", "res"), ("supcon_sigreg", "res"), ("supcon", "res-nplm")]


def make_res_step(cents, lam):
    def step(net, v1, v2, y):
        x = torch.cat([v1, v2]).to(DEVICE, non_blocking=True)
        yy = torch.cat([y, y]).to(DEVICE)
        r = net(x).float() - cents[yy]
        inst = torch.arange(v1.size(0), device=DEVICE)
        return (supcon_loss(F.normalize(r, dim=1),
                            torch.cat([inst, inst]), temp=0.5)
                + lam * sigreg_loss(r))
    return step


def make_res_nplm_step(cents, lam, n_slices):
    loss_fn = HybridContrastiveLoss(positives="instance", critic="bilinear",
                                    estimator="nplm", marginal="none",
                                    tau=1.0)
    def step(net, v1, v2, y):
        x = torch.cat([v1, v2]).to(DEVICE, non_blocking=True)
        yy = torch.cat([y, y]).to(DEVICE)
        r = net(x).float() - cents[yy]
        inst = torch.arange(v1.size(0), device=DEVICE)
        inter, _ = loss_fn(r, torch.cat([inst, inst]))
        return inter + lam * sigreg_loss(r, n_slices=n_slices)
    return step


def residual_ft(net, loader, epochs, step, tag):
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    net.train()
    for ep in range(epochs):
        run, n = 0.0, 0
        for v1, v2, y in loader:
            opt.zero_grad()
            loss = step(net, v1, v2, y)
            loss.backward()
            opt.step()
            run += loss.item() * v1.size(0)
            n += v1.size(0)
        if (ep + 1) % 5 == 0 or ep == 0 or ep == epochs - 1:
            print(f"  [{tag}] epoch {ep+1}/{epochs}  loss={run/n:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10",
                    choices=["cifar10", "cifar100"])
    ap.add_argument("--holdout", type=int, default=4)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()
    ds = args.dataset
    cfg = recipe(ds, emb_dim=args.dim)
    n_cls = cfg["n_classes"]
    holdouts = {args.holdout}
    seen = [c for c in range(n_cls) if c not in holdouts]
    con_ep = args.epochs or (2 if args.quick else 20)
    tag = f"{ds}_{args.dim}d"
    print(f"exp73 [{tag}] CIFAR residual ft (exp-71 recipe), runs={RUNS}, "
          f"epochs={con_ep}, holdout={sorted(holdouts)}")

    train_loader, test_loader = get_cifar_loaders(quick=args.quick,
                                                  dataset=ds)
    train_eval_loader = DataLoader(train_loader.dataset, batch_size=256,
                                   shuffle=False, num_workers=2)

    def battery(name, tr, te, tr_lab, te_lab):
        m = np.isin(tr_lab, seen)
        cents = exp28.class_centroids(tr[m], tr_lab[m], seen)
        anch = cents.detach().float().to(DEVICE)
        r = exp29.evaluate_space(tr, tr_lab, te, te_lab, anch, seen,
                                 holdouts)
        aucs = []
        for s in range(3):
            torch.manual_seed(1000 + s)
            a, _, _ = exp29.linear_probe_novelty(tr, tr_lab, te, te_lab,
                                                 holdouts)
            aucs.append(a)
        d = torch.cdist(torch.as_tensor(te, dtype=torch.float32,
                                        device=DEVICE), anch)
        s_ = d.min(1).values.cpu().numpy()
        bg = np.isin(te_lab, seen)
        sg = np.isin(te_lab, list(holdouts))
        pe = exp30.power_at_alpha(s_[bg], s_[sg], args.alpha)
        out = dict(probe=float(np.mean(aucs)), probe_sd=float(np.std(aucs)),
                   acc=r["acc"], eucl=r["eucl"], mahaT=r["maha_tied"],
                   lid=r["lid"], perevt=pe)
        print(f"  [{name:<28}] probe={out['probe']:.4f}+-{out['probe_sd']:.4f}"
              f" acc={out['acc']:.4f} eucl={out['eucl']:.4f} "
              f"mahaT={out['mahaT']:.4f} lid={out['lid']:.4f} "
              f"perevt={pe:.3f}")
        return out

    results, parent_cache = {}, {}
    for parent, obj in RUNS:
        key = f"{parent}->{obj}"
        print(f"\n===== [{tag}] {key} =====")
        if parent not in parent_cache:
            net = exp55.train_arm(parent, ds, cfg, args, con_ep, holdouts)
            ptr, tr_lab = collect_embeddings(net, train_eval_loader)
            pte, te_lab = collect_embeddings(net, test_loader)
            m = np.isin(tr_lab, seen)
            cents_full = torch.zeros(n_cls, args.dim, device=DEVICE)
            cents_full[torch.as_tensor(seen, device=DEVICE)] = \
                exp28.class_centroids(ptr[m], tr_lab[m],
                                      seen).detach().float().to(DEVICE)
            parent_cache[parent] = (net, ptr, pte, tr_lab, te_lab,
                                    cents_full)
            results[f"{parent} (parent)"] = battery(
                f"{parent} (parent)", ptr, pte, tr_lab, te_lab)
        net, ptr, pte, tr_lab, te_lab, cents_full = parent_cache[parent]

        torch.manual_seed(args.seed + 7); np.random.seed(args.seed + 7)
        child = copy.deepcopy(net)
        loader = cifar_two_view_loader(quick=args.quick, labeled=True,
                                       holdout=holdouts, dataset=ds)
        step = (make_res_step(cents_full, 5.0) if obj == "res"
                else make_res_nplm_step(cents_full, args.lam,
                                        cfg["n_slices"]))
        residual_ft(child, loader, con_ep, step, f"{parent}-{obj}")
        rtr, _ = collect_embeddings(child, train_eval_loader)
        rte, _ = collect_embeddings(child, test_loader)
        del child
        torch.cuda.empty_cache()

        results[f"{key} (residual)"] = battery(
            f"{key} (residual)", rtr, rte, tr_lab, te_lab)
        results[f"{key} (concat)"] = battery(
            f"{key} (concat)", np.concatenate([ptr, rtr], 1),
            np.concatenate([pte, rte], 1), tr_lab, te_lab)

    parent_cache.clear()
    torch.cuda.empty_cache()

    print(f"\n===== EXP73 SUMMARY [{tag}] =====")
    print(f"  {'space':<32}{'probe':>16}{'acc':>8}{'eucl':>8}{'mahaT':>8}"
          f"{'lid':>8}{'perevt':>8}")
    for k, r in results.items():
        print(f"  {k:<32}{r['probe']:>9.4f}+-{r['probe_sd']:.4f}"
              f"{r['acc']:>8.4f}{r['eucl']:>8.4f}{r['mahaT']:>8.4f}"
              f"{r['lid']:>8.4f}{r['perevt']:>8.3f}")

    labels = list(results)
    plt.figure(figsize=(1.0 * len(labels) + 3, 5.5))
    plt.bar(range(len(labels)), [results[k]["probe"] for k in labels],
            yerr=[results[k]["probe_sd"] for k in labels], capsize=3,
            color=["#eda100" if "parent" in k else
                   "#8c2d9e" if "nplm" in k else "#008300"
                   for k in labels])
    plt.xticks(range(len(labels)), labels, rotation=25, ha="right",
               fontsize=7)
    plt.ylabel("holdout probe ROC AUC")
    plt.title(f"exp73 CIFAR residual ft ({tag})")
    plt.grid(alpha=0.25, axis="y")
    plt.tight_layout()
    out = plot_path(f"exp73_probe_{tag}.png")
    plt.savefig(out, dpi=150); plt.close()
    print("saved", out)

    os.makedirs(os.path.join("logs", "exp73"), exist_ok=True)
    np.savez(os.path.join("logs", "exp73", f"results_{tag}.npz"),
             spaces=np.array(list(results)),
             **{f"{k.replace(' ', '_').replace('>', '')}__{f}":
                np.array(results[k][f]) for k in results
                for f in ("probe", "probe_sd", "acc", "eucl", "mahaT",
                          "lid", "perevt")})
    print("Done.")


if __name__ == "__main__":
    main()
