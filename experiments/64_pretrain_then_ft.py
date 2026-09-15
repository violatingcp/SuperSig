"""
Experiment 64: pretraining objective -> full supervised fine-tune (CIFAR-100,
holdout class included in the fine-tune).

Stage 1: pretrain the 32-D backbone on the 99 SEEN classes (holdout 4
excluded) with each objective -- simclr / lejepa / simclr_sigreg /
nplm_bilinear (label-free) / nplm_sup_dist (labelled) -- 20 epochs, suite
protocol.  Stage 2: full supervised CE fine-tune on ALL 100 classes
(backbone + Linear(32,100), everything trainable, balanced aug loader).
Question: which pretraining leaves the best substrate for supervised
learning, and which absorbs the never-before-seen class best?

Reported per arm: overall test top-1, seen-class top-1, holdout-class
recall.  Baseline "none" = CE straight from the hub-pretrained trunk.
NOTE: all backbones init from the cifar100-pretrained hub ResNet-20 (suite
convention), so "never saw the holdout" applies to the SSL/NPLM stage, not
the hub init.

    python experiments/64_pretrain_then_ft.py
    python experiments/64_pretrain_then_ft.py --quick --arms none nplm_bilinear
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import importlib
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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

exp34h = importlib.import_module("34h_hybrid_nplm_cifar")
exp50 = importlib.import_module("50_nplm_cifar10_suite")
exp53 = importlib.import_module("53_nplm_classwise")

ARMS = ["none", "simclr", "lejepa", "simclr_sigreg", "nplm_bilinear",
        "nplm_sup_dist", "nplm_dist_sup_cw"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--holdout", type=int, default=4)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--pre-epochs", type=int, default=None)
    ap.add_argument("--ft-epochs", type=int, default=None)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    ap.add_argument("--out-tag", default="")
    args = ap.parse_args()
    ds = args.dataset

    cfg = recipe(ds, emb_dim=args.dim)
    n_cls = cfg["n_classes"]
    holdouts = {args.holdout}
    seen = [c for c in range(n_cls) if c not in holdouts]
    pre_ep = args.pre_epochs or (2 if args.quick else 20)
    ft_ep = args.ft_epochs or (2 if args.quick else 10)
    print(f"exp64 [{ds}] pretrain->CE-ft, holdout={sorted(holdouts)} "
          f"(excluded from pretrain, included in ft), pre={pre_ep}ep "
          f"ft={ft_ep}ep, arms={args.arms}")

    _, test_loader = get_cifar_loaders(quick=args.quick, dataset=ds)
    ft_loader = cifar_balanced_loader(ds, holdout=None, quick=args.quick,
                                      augment=True)

    results = {}
    for i, name in enumerate(args.arms):
        print(f"\n===== arm: {name} =====")
        torch.manual_seed(args.seed + 20 + i)
        np.random.seed(args.seed + 20 + i)
        net = CIFARResNetBackbone(args.dim, arch=cfg["arch"],
                                  pretrain=ds).to(DEVICE)
        if name == "nplm_dist_sup_cw":
            print(f"--- stage 1: {name} pretrain (seen classes only) ---")
            loader = cifar_two_view_balanced_loader(ds, holdout=holdouts,
                                                    quick=args.quick)
            means = make_anchors(cfg["pair_dist"] / math.sqrt(2.0),
                                 emb_dim=args.dim,
                                 n_classes=n_cls).detach()
            exp53.train_nplm_classwise(net, loader, pre_ep, "supervised",
                                       "distance", means, tau=1.0,
                                       lam=args.lam,
                                       n_slices=cfg["n_slices"])
        elif name != "none":
            kind, spec, labeled = exp50.ARMS[name]
            loader = cifar_two_view_loader(quick=args.quick, labeled=labeled,
                                           holdout=holdouts, dataset=ds)
            print(f"--- stage 1: {name} pretrain (seen classes only) ---")
            if kind == "hybrid":
                exp34h.train_hybrid(net, loader, pre_ep, spec, labeled,
                                    lam=args.lam, n_slices=cfg["n_slices"])
            else:
                spec(net, loader, pre_ep)

        print(f"--- stage 2: CE fine-tune, all {n_cls} classes ---")
        cls_head = nn.Linear(args.dim, n_cls).to(DEVICE)
        opt = torch.optim.Adam(list(net.parameters())
                               + list(cls_head.parameters()), lr=1e-3)
        net.train()
        for ep in range(ft_ep):
            run, correct, n = 0.0, 0, 0
            for x, y in ft_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                opt.zero_grad()
                logits = cls_head(net(x))
                loss = F.cross_entropy(logits, y)
                loss.backward()
                opt.step()
                run += loss.item() * x.size(0)
                correct += int((logits.argmax(1) == y).sum())
                n += x.size(0)
            print(f"  [ce-ft] epoch {ep+1}/{ft_ep}  loss={run/n:.4f}  "
                  f"train-acc={correct/n:.4f}")

        net.eval()
        preds, labs = [], []
        with torch.no_grad():
            for x, y in test_loader:
                p = cls_head(net(x.to(DEVICE))).argmax(1).cpu()
                preds.append(p)
                labs.append(y)
        preds = torch.cat(preds).numpy()
        labs = torch.cat(labs).numpy()
        top1 = float((preds == labs).mean())
        m_seen = np.isin(labs, seen)
        m_hold = np.isin(labs, list(holdouts))
        seen_top1 = float((preds[m_seen] == labs[m_seen]).mean())
        hold_rec = float((preds[m_hold] == labs[m_hold]).mean())
        print(f"  [{name:<14}] top1={top1:.4f}  seen={seen_top1:.4f}  "
              f"holdout-recall={hold_rec:.4f}")
        results[name] = dict(top1=top1, seen_top1=seen_top1,
                             holdout_recall=hold_rec)
        del net, cls_head
        torch.cuda.empty_cache()

    print(f"\n===== EXP64 SUMMARY [{ds}] =====")
    print(f"  {'pretrain':<16}{'top1':>8}{'seen':>8}{'holdout':>9}")
    for name in args.arms:
        r = results[name]
        print(f"  {name:<16}{r['top1']:>8.4f}{r['seen_top1']:>8.4f}"
              f"{r['holdout_recall']:>9.4f}")

    xs = np.arange(len(args.arms))
    w = 0.35
    plt.figure(figsize=(8.5, 5))
    plt.bar(xs - w / 2, [results[n]["seen_top1"] for n in args.arms], w,
            label="seen top-1", color="#2a78d6")
    plt.bar(xs + w / 2, [results[n]["holdout_recall"] for n in args.arms],
            w, label="holdout recall", color="#d62728")
    plt.xticks(xs, args.arms, rotation=15, ha="right")
    plt.ylabel("test accuracy")
    plt.title(f"exp64: pretrain -> CE-ft ({ds}, holdout {args.holdout} "
              f"unseen in pretrain)")
    plt.legend(); plt.grid(alpha=0.25, axis="y")
    plt.tight_layout()
    out = plot_path(f"exp64_pretrain_ft_{ds}{args.out_tag}.png")
    plt.savefig(out, dpi=150); plt.close()
    print("saved", out)

    os.makedirs(os.path.join("logs", "exp64"), exist_ok=True)
    np.savez(os.path.join("logs", "exp64",
                          f"pretrain_ft_{ds}{args.out_tag}.npz"),
             arms=np.array(args.arms),
             **{f"{k}_{n}": np.array(results[n][k]) for n in args.arms
                for k in ("top1", "seen_top1", "holdout_recall")})
    print("Done.")


if __name__ == "__main__":
    main()
