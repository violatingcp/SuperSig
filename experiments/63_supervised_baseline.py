"""
Experiment 63: fully-supervised (CE) baseline spaces for the NPLM program.

The suite tables (exps 50/51) lack the plain supervised-classification
baseline.  This trains backbone + linear classifier with CE on the seen
classes and evaluates the penultimate embedding with the standard suite
protocol (Part A + per-event), plus the classifier's own test top-1.

  --dataset cifar100 : CIFARResNetBackbone(32) + Linear(32, 100), balanced
                       aug loader, 20 epochs (exp-50 protocol, holdout 4).
  --dataset aircraft : FeatureHead(32) + Linear(32, 100) on feature banks,
                       one run per substrate: frozen DINO and the three
                       exp-62 nplm-sup-ft trunks (120 epochs, holdouts
                       90-99).  CAVEAT: the ft trunks saw holdout images
                       and labels during exp-62 ft.

    python experiments/63_supervised_baseline.py --dataset cifar100
    python experiments/63_supervised_baseline.py --dataset aircraft
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import importlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")

from supersig.config import DEVICE, plot_path
from supersig.data import (get_cifar_loaders, cifar_balanced_loader)
from supersig.models import CIFARResNetBackbone
from supersig.metrics import gaussianity_summary
from supersig.recipes import recipe
from supersig.train import collect_embeddings

exp28 = importlib.import_module("28_concat_residual")
exp29 = importlib.import_module("29_residual_finetune")
exp30 = importlib.import_module("30_power_curves")
exp37 = importlib.import_module("37_dtd_vit")
exp40 = importlib.import_module("40_dtd_bases")
exp44 = importlib.import_module("44_transfer_32d")
exp51 = importlib.import_module("51_nplm_aircraft_suite")


def train_ce(backbone, cls_head, loader, epochs, feature_input, lr=1e-3):
    opt = torch.optim.Adam(list(backbone.parameters())
                           + list(cls_head.parameters()), lr=lr)
    backbone.train()
    for ep in range(epochs):
        run, correct, n = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logits = cls_head(backbone(x))
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            run += loss.item() * x.size(0)
            correct += int((logits.argmax(1) == y).sum())
            n += x.size(0)
        if (ep + 1) % 10 == 0 or ep == 0 or ep == epochs - 1:
            print(f"  [ce] epoch {ep+1}/{epochs}  loss={run/n:.4f}  "
                  f"train-acc={correct/n:.4f}")


def evaluate(name, tr, tr_lab, te, te_lab, seen, holdouts, cls_top1, seed,
             alpha=0.05):
    m = np.isin(tr_lab, seen)
    cents = exp28.class_centroids(tr[m], tr_lab[m], seen)
    anchors = torch.as_tensor(cents, dtype=torch.float32, device=DEVICE)
    r = exp29.evaluate_space(tr, tr_lab, te, te_lab, anchors, seen, holdouts)
    aucs = []
    for s in range(3):
        torch.manual_seed(1000 + s)
        a, _, _ = exp29.linear_probe_novelty(tr, tr_lab, te, te_lab, holdouts)
        aucs.append(a)
    pm, psd = float(np.mean(aucs)), float(np.std(aucs))
    d = torch.cdist(torch.as_tensor(te, dtype=torch.float32, device=DEVICE),
                    anchors)
    s_ = d.min(1).values.cpu().numpy()
    pe = exp30.power_at_alpha(s_[np.isin(te_lab, seen)],
                              s_[np.isin(te_lab, list(holdouts))], alpha)
    g = gaussianity_summary(te, te_lab, seen, seed=seed)
    print(f"  [{name:<22}] probe={pm:.4f}+-{psd:.4f} acc={r['acc']:.4f} "
          f"supAUC={r['sup_auc']:.4f} eucl={r['eucl']:.4f} "
          f"mahaT={r['maha_tied']:.4f} perevt={pe:.3f} "
          f"cls-top1={cls_top1:.4f}")
    return dict(probe=pm, probe_sd=psd, acc=r["acc"], sup_auc=r["sup_auc"],
                eucl=r["eucl"], mahaT=r["maha_tied"], mahaPC=r["maha_pc"],
                perevent=pe, cls_top1=cls_top1, gauss=g)


def run_cifar(args):
    ds = args.dataset
    cfg = recipe(ds, emb_dim=args.dim)
    n_cls = cfg["n_classes"]
    holdouts = {args.holdout}
    seen = [c for c in range(n_cls) if c not in holdouts]
    ep = args.epochs or (2 if args.quick else 20)
    torch.manual_seed(args.seed + 20); np.random.seed(args.seed + 20)
    net = CIFARResNetBackbone(args.dim, arch=cfg["arch"],
                              pretrain=ds).to(DEVICE)
    cls_head = nn.Linear(args.dim, n_cls).to(DEVICE)
    loader = cifar_balanced_loader(ds, holdout=holdouts, quick=args.quick,
                                   augment=True)
    print(f"===== training: supervised-CE [{ds}] =====")
    train_ce(net, cls_head, loader, ep, feature_input=False)

    train_loader, test_loader = get_cifar_loaders(quick=args.quick,
                                                  dataset=ds)
    train_eval_loader = DataLoader(train_loader.dataset, batch_size=256,
                                   shuffle=False, num_workers=2)
    tr, tr_lab = collect_embeddings(net, train_eval_loader)
    te, te_lab = collect_embeddings(net, test_loader)
    with torch.no_grad():
        logits = cls_head(torch.as_tensor(te, dtype=torch.float32,
                                          device=DEVICE))
        pred = logits.argmax(1).cpu().numpy()
    m = np.isin(te_lab, seen)
    top1 = float((pred[m] == te_lab[m]).mean())
    res = evaluate(f"ce_{ds}", tr, tr_lab, te, te_lab, seen, holdouts,
                   top1, args.seed)
    return {f"ce_{ds}": res}


def run_aircraft(args):
    N_CLS = exp44.N_CLASSES["aircraft"]
    holdouts = set(range(N_CLS - 10, N_CLS))
    seen = [c for c in range(N_CLS) if c not in holdouts]
    ep = args.epochs or (5 if args.quick else 120)
    results = {}
    for sub in args.substrates.split(","):
        print(f"\n===== training: supervised-CE [aircraft {sub}] =====")
        args.base = "dino" if sub in ("frozen", "dino-ft") else \
            sub.replace("-ft", "")
        args.trunk_ckpt_arm = None if sub == "frozen" else "nplm-sup-ft"
        plain, bank = (exp51.build_features_ft(args) if args.trunk_ckpt_arm
                       else exp44.build_features("aircraft", args.base,
                                                 args))
        (Xtr, ytr), (Xte, yte) = plain["train"], plain["test"]
        tr_lab, te_lab = ytr.numpy(), yte.numpy()
        seen_bank = exp51.filter_bank(
            bank, ~np.isin(bank["labels"].numpy(), list(holdouts)))
        torch.manual_seed(args.seed + 20); np.random.seed(args.seed + 20)
        head = exp37.FeatureHead(args.dim).to(DEVICE)
        cls_head = nn.Linear(args.dim, N_CLS).to(DEVICE)

        tv = exp37.TwoViewFeatures(seen_bank)
        loader = DataLoader(tv, batch_size=512, shuffle=True, drop_last=True)
        opt = torch.optim.Adam(list(head.parameters())
                               + list(cls_head.parameters()), lr=1e-3)
        head.train()
        for e in range(ep):
            run, correct, n = 0.0, 0, 0
            for f1, _, y in loader:
                f1, y = f1.to(DEVICE), y.to(DEVICE)
                opt.zero_grad()
                logits = cls_head(head(f1))
                loss = F.cross_entropy(logits, y)
                loss.backward()
                opt.step()
                run += loss.item() * f1.size(0)
                correct += int((logits.argmax(1) == y).sum())
                n += f1.size(0)
            if (e + 1) % 30 == 0 or e == 0 or e == ep - 1:
                print(f"  [ce/{sub}] epoch {e+1}/{ep}  loss={run/n:.4f}  "
                      f"train-acc={correct/n:.4f}")

        tr = exp37.embed(head, Xtr).numpy()
        te = exp37.embed(head, Xte).numpy()
        with torch.no_grad():
            pred = cls_head(torch.as_tensor(te, dtype=torch.float32,
                                            device=DEVICE)) \
                .argmax(1).cpu().numpy()
        m = np.isin(te_lab, seen)
        top1 = float((pred[m] == te_lab[m]).mean())
        results[f"ce_{sub}"] = evaluate(f"ce_{sub}", tr, tr_lab, te, te_lab,
                                        seen, holdouts, top1, args.seed)
        del head, cls_head
        torch.cuda.empty_cache()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar100",
                    choices=["cifar100", "cifar10", "aircraft"])
    ap.add_argument("--substrates", default="frozen,dino-ft,lejepa-ft,visreg-ft")
    ap.add_argument("--holdout", type=int, default=4)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--aug-reps", type=int, default=8)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    print(f"exp63 supervised-CE baseline [{args.dataset}]")

    results = (run_aircraft(args) if args.dataset == "aircraft"
               else run_cifar(args))

    print("\n===== EXP63 SUMMARY =====")
    for name, r in results.items():
        print(f"  [{name:<22}] probe={r['probe']:.4f} acc={r['acc']:.4f} "
              f"eucl={r['eucl']:.4f} mahaT={r['mahaT']:.4f} "
              f"perevt={r['perevent']:.3f} cls-top1={r['cls_top1']:.4f}")
    os.makedirs(os.path.join("logs", "exp63"), exist_ok=True)
    np.savez(os.path.join("logs", "exp63", f"ce_baseline_{args.dataset}.npz"),
             **{f"{k}_{n}": np.array(r[k]) for n, r in results.items()
                for k in ("probe", "probe_sd", "acc", "sup_auc", "eucl",
                          "mahaT", "mahaPC", "perevent", "cls_top1")})
    print("Done.")


if __name__ == "__main__":
    main()
