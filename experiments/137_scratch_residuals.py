"""
Experiment 137: residual / concat constructions on the from-scratch CIFAR
trunks -- the exp-73 recipe with leakage-free parents.

WHY.  The residual concat holds the CIFAR probe records (c10 0.955+-0.002,
c100 0.955+-0.006, exp 75) and is the paper's "no-cost" construction (exp
132), but every CIFAR residual was built on a hub-pretrained parent that saw
the held-out class.  Exp 136 fills the scratch grid for standalone spaces;
this fills it for the constructions.

PROTOCOL.  For each supervised scratch parent {supcon, ssig, nplmsd} (exp-67
checkpoint, 100-D, holdout-tagged), train the two exp-71/73 children --
res (NT-Xent + lam=5 SIGReg on r = z - cent_y) and res-nplm (bilinear NPLM +
lam=1 SIGReg on r) -- for 20 epochs from a deepcopy of the parent, then run
exp 73's battery on parent / residual / concat (probe x3, acc, eucl, mahaT,
lid, per-event) plus the exp-132 supervised top-1 on each.  Children are
checkpointed as checkpoints/scratch_{parent}_{obj}_{ds}_{dim}d{htag}.pt and
the banks are written next to exp 136's so exps 128/129/132/134 can consume
them.

PREDICTION.  The paired concat-over-parent probe gain reproduces on clean
trunks on both datasets (exp 75: +0.016 c10, +0.009 c100 hub), and the
concat ties the parent on supervised top-1 (exp 132's construction claim);
res-nplm is the two-currency child on c100 (many-class rule, exp 75) while
plain res wins probe on c10.

FALSIFIER.  The concat gain vanishes on clean trunks -> the CIFAR residual
records were riding backbone leakage.  Or the concat loses top-1 to the
parent by more than the 0.017 floor -> the no-cost construction claim fails
on CIFAR.

    python experiments/137_scratch_residuals.py --dataset cifar100
    python experiments/137_scratch_residuals.py --dataset cifar10 --holdout 8 --parents supcon ssig
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import copy
import importlib
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from supersig.config import DEVICE
from supersig.data import get_cifar_loaders, cifar_two_view_loader
from supersig.models import CIFARResNetBackbone
from supersig.recipes import recipe
from supersig.train import collect_embeddings

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.path.join(REPO, "checkpoints")
PARENTS = ["supcon", "ssig", "nplmsd"]
OBJS = ["res", "res-nplm"]


def htag(holdout):
    return "" if holdout == 4 else f"_h{holdout}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar100", choices=["cifar10", "cifar100"])
    ap.add_argument("--holdout", type=int, default=4)
    ap.add_argument("--parents", nargs="+", default=PARENTS, choices=PARENTS)
    ap.add_argument("--objs", nargs="+", default=OBJS, choices=OBJS)
    ap.add_argument("--dim", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--out", default="logs/exp137")
    args = ap.parse_args()

    exp28 = importlib.import_module("28_concat_residual")
    exp29 = importlib.import_module("29_residual_finetune")
    exp30 = importlib.import_module("30_power_curves")
    exp73 = importlib.import_module("73_cifar_residual_ft")
    exp132 = importlib.import_module("132_supervised_probe")

    ds, hold, tag = args.dataset, args.holdout, htag(args.holdout)
    cfg = recipe(ds, emb_dim=args.dim)
    n_cls = cfg["n_classes"]
    holdouts = {hold}
    seen = [c for c in range(n_cls) if c not in holdouts]
    con_ep = args.epochs or (2 if args.quick else 20)
    qs = "_quick" if args.quick else ""
    print(f"exp137 [{ds}{tag}] scratch residuals; parents={args.parents} objs={args.objs} "
          f"epochs={con_ep} holdout={hold}", flush=True)
    train_loader, test_loader = get_cifar_loaders(quick=args.quick, dataset=ds)
    tel = DataLoader(train_loader.dataset, batch_size=256, shuffle=False, num_workers=2)
    os.makedirs(os.path.join("logs", "exp136", "banks"), exist_ok=True)
    os.makedirs(args.out, exist_ok=True)

    def battery(name, tr, te, tr_lab, te_lab):
        m = np.isin(tr_lab, seen)
        anch = exp28.class_centroids(tr[m], tr_lab[m], seen).detach().float().to(DEVICE)
        r = exp29.evaluate_space(tr, tr_lab, te, te_lab, anch, seen, holdouts)
        aucs = []
        for s in range(3):
            torch.manual_seed(1000 + s)
            a, _, _ = exp29.linear_probe_novelty(tr, tr_lab, te, te_lab, holdouts)
            aucs.append(a)
        top1, top1_sd, _ = exp132.probe_multiseed(tr, tr_lab, te, te_lab, seen, seeds=3)
        d = torch.cdist(torch.as_tensor(te, dtype=torch.float32, device=DEVICE), anch)
        s_ = d.min(1).values.cpu().numpy()
        pe = exp30.power_at_alpha(s_[np.isin(te_lab, seen)],
                                  s_[np.isin(te_lab, list(holdouts))], args.alpha)
        out = dict(probe=float(np.mean(aucs)), probe_sd=float(np.std(aucs)),
                   top1=top1, top1_sd=top1_sd, acc=r["acc"], eucl=r["eucl"],
                   mahaT=r["maha_tied"], lid=r["lid"], perevt=pe)
        print(f"  [{name:<28}] probe={out['probe']:.4f}+-{out['probe_sd']:.4f} top1={top1:.4f}"
              f" acc={out['acc']:.4f} eucl={out['eucl']:.4f} mahaT={out['mahaT']:.4f} "
              f"lid={out['lid']:.4f} perevt={pe:.3f}", flush=True)
        return out

    results = {}
    for parent in args.parents:
        pck = os.path.join(CKPT_DIR, f"scratch_{parent}_{ds}_{args.dim}d{tag}.pt")
        if not os.path.exists(pck):
            print(f"  !! missing parent {pck}, skip"); continue
        net = CIFARResNetBackbone(args.dim, arch=cfg["arch"], pretrain=None).to(DEVICE)
        net.load_state_dict(torch.load(pck, map_location=DEVICE)["state_dict"])
        net.eval()
        ptr, tr_lab = collect_embeddings(net, tel)
        pte, te_lab = collect_embeddings(net, test_loader)
        m = np.isin(tr_lab, seen)
        cents_full = torch.zeros(n_cls, args.dim, device=DEVICE)
        cents_full[torch.as_tensor(seen, device=DEVICE)] = \
            exp28.class_centroids(ptr[m], tr_lab[m], seen).detach().float().to(DEVICE)
        results[f"{parent} (parent)"] = battery(f"{parent} (parent)", ptr, pte, tr_lab, te_lab)
        for obj in args.objs:
            key = f"{parent}->{obj}"
            print(f"\n===== [{ds}{tag}] {key} =====", flush=True)
            rck = os.path.join(CKPT_DIR, f"scratch_{parent}_{obj.replace('-', '')}_{ds}_"
                                         f"{args.dim}d{tag}{qs}.pt")
            child = copy.deepcopy(net)
            if os.path.exists(rck) and not args.refresh:
                print(f"  loading {rck}")
                child.load_state_dict(torch.load(rck, map_location=DEVICE))
            else:
                torch.manual_seed(args.seed + 7); np.random.seed(args.seed + 7)
                loader = cifar_two_view_loader(quick=args.quick, labeled=True,
                                               holdout=holdouts, dataset=ds)
                step = (exp73.make_res_step(cents_full, 5.0) if obj == "res"
                        else exp73.make_res_nplm_step(cents_full, args.lam, cfg["n_slices"]))
                exp73.residual_ft(child, loader, con_ep, step, f"{parent}-{obj}")
                torch.save(child.state_dict(), rck)
            child.eval()
            rtr, _ = collect_embeddings(child, tel)
            rte, _ = collect_embeddings(child, test_loader)
            del child; torch.cuda.empty_cache()
            ctr, cte = np.concatenate([ptr, rtr], 1), np.concatenate([pte, rte], 1)
            for nm, (a, b) in ((f"{key} (residual)", (rtr, rte)), (f"{key} (concat)", (ctr, cte))):
                results[nm] = battery(nm, a, b, tr_lab, te_lab)
                np.savez(os.path.join("logs", "exp136", "banks",
                                      f"embs_{nm.replace(' ', '_').replace('->', '-')}_{ds}{tag}.npz"),
                         tr=a, tr_lab=tr_lab, te=b, te_lab=te_lab)
        del net; torch.cuda.empty_cache()

    print(f"\n===== EXP137 SUMMARY [{ds}{tag}] (holdout {hold}) =====")
    print(f"  {'space':<30}{'probe':>16}{'top1':>8}{'acc':>8}{'eucl':>8}{'mahaT':>8}{'lid':>8}{'perevt':>8}")
    for k, r in results.items():
        print(f"  {k:<30}{r['probe']:>9.4f}+-{r['probe_sd']:.4f}{r['top1']:>8.4f}{r['acc']:>8.4f}"
              f"{r['eucl']:>8.4f}{r['mahaT']:>8.4f}{r['lid']:>8.4f}{r['perevt']:>8.3f}")
    for parent in args.parents:
        pk = f"{parent} (parent)"
        if pk not in results: continue
        for obj in args.objs:
            ck = f"{parent}->{obj} (concat)"
            if ck in results:
                print(f"  paired {ck} - parent: probe {results[ck]['probe']-results[pk]['probe']:+.4f}"
                      f"  top1 {results[ck]['top1']-results[pk]['top1']:+.4f}"
                      f"  eucl {results[ck]['eucl']-results[pk]['eucl']:+.4f}"
                      f"  mahaT {results[ck]['mahaT']-results[pk]['mahaT']:+.4f}")
    with open(os.path.join(args.out, f"residuals_{ds}{tag}.json"), "w") as fh:
        json.dump(results, fh, indent=1, default=float)
    print("EXP137 DONE.")


if __name__ == "__main__":
    main()
