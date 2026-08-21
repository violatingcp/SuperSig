"""
Experiment 89 (IMPROVEMENT_TESTS.md #89): CIFAR-100 discovery rate
unblocking — calibrate the purity gate.

C100 discovery is rate-blocked (500 holdout images in a ~2500-event tail,
purity 0.003-0.013).  Never tested: stricter tau_quantile, multi-class
holdouts.  Grid: holdout size {1, 5, 10 classes} x tau_quantile
{0.95, 0.99, 0.995} on the exp-73 res-nplm concat (100-D, seed 0; parent
+ child retrained per holdout set since holdouts are excluded from
training; size-1 = class 4 for comparability, sizes 5/10 = last-k).
Purity per round is the primary readout; probe pre/post secondary.

Prediction: purity scales with holdout fraction and quantile strictness;
the probe only moves once purity clears ~0.15-0.3.  Falsifier: purity
rises past 0.3 and the probe still does not move (the gate is not the
mechanism on C100).

    python experiments/89_c100_rate_grid.py
    python experiments/89_c100_rate_grid.py --quick --sizes 10 --quantiles 0.95
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import copy
import importlib
import numpy as np
import torch
from torch.utils.data import DataLoader

from supersig.config import DEVICE
from supersig.data import get_cifar_loaders, cifar_two_view_loader
from supersig.discovery import run_discovery
from supersig.recipes import recipe
from supersig.train import collect_embeddings

exp28 = importlib.import_module("28_concat_residual")
exp29 = importlib.import_module("29_residual_finetune")
exp55 = importlib.import_module("55_nplm_discovery")
exp73 = importlib.import_module("73_cifar_residual_ft")
exp74 = importlib.import_module("74_cifar_residual_discovery")

CKPT = "checkpoints"


def build_space(holdouts, cfg, args, con_ep):
    """exp-73 recipe at seed 0 for an arbitrary holdout set; ckpt-cached."""
    tag = f"exp89_c100_h{len(holdouts)}"
    pp = os.path.join(CKPT, f"{tag}_supcon.pt")
    cp = os.path.join(CKPT, f"{tag}_resnplm.pt")
    sargs = argparse.Namespace(**{**vars(args), "lam": 1.0, "tau": 1.0})
    if os.path.exists(pp) and not args.refresh:
        from supersig.models import CIFARResNetBackbone
        parent = CIFARResNetBackbone(args.dim, arch=cfg["arch"],
                                     pretrain="cifar100").to(DEVICE)
        parent.load_state_dict(torch.load(pp, map_location=DEVICE))
        print(f"  loaded {pp}")
    else:
        parent = exp55.train_arm("supcon", "cifar100", cfg, sargs, con_ep,
                                 holdouts)
        torch.save(parent.state_dict(), pp)
    if os.path.exists(cp) and not args.refresh:
        child = copy.deepcopy(parent)
        child.load_state_dict(torch.load(cp, map_location=DEVICE))
        print(f"  loaded {cp}")
    else:
        ev = DataLoader(get_cifar_loaders(quick=args.quick,
                                          dataset="cifar100")[0].dataset,
                        batch_size=256, shuffle=False, num_workers=2)
        tr, tr_lab = collect_embeddings(parent, ev)
        seen = [c for c in range(cfg["n_classes"]) if c not in holdouts]
        m = np.isin(tr_lab, seen)
        cents_full = exp28.fill_means(
            exp28.class_centroids(tr[m], tr_lab[m], seen), seen,
            dict(pair_dist=cfg["pair_dist"],
                 n_classes=cfg["n_classes"])).detach()
        torch.manual_seed(args.seed + 7); np.random.seed(args.seed + 7)
        child = copy.deepcopy(parent)
        loader = cifar_two_view_loader(quick=args.quick, labeled=True,
                                       holdout=holdouts, dataset="cifar100")
        step = exp73.make_res_nplm_step(cents_full, 1.0, cfg["n_slices"])
        exp73.residual_ft(child, loader, con_ep, step, "exp89-resnplm")
        torch.save(child.state_dict(), cp)
    return exp74.ConcatNets(parent, child).to(DEVICE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", type=int, default=[1, 5, 10])
    ap.add_argument("--quantiles", nargs="+", type=float,
                    default=[0.95, 0.99, 0.995])
    ap.add_argument("--dim", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    ds = "cifar100"
    cfg = recipe(ds, emb_dim=args.dim)
    n_cls = cfg["n_classes"]
    con_ep = args.epochs or (2 if args.quick else 20)
    ft_ep = 1 if args.quick else cfg["ft_epochs"]
    os.makedirs(CKPT, exist_ok=True)

    train_loader, test_loader = get_cifar_loaders(quick=args.quick,
                                                  dataset=ds)
    tel = DataLoader(train_loader.dataset, batch_size=256, shuffle=False,
                     num_workers=2)
    results = {}
    for size in args.sizes:
        holdouts = {4} if size == 1 else set(range(n_cls - size, n_cls))
        seen = [c for c in range(n_cls) if c not in holdouts]
        print(f"\n######## holdout size {size} "
              f"({sorted(holdouts)[:3]}{'...' if size > 3 else ''}) "
              f"########", flush=True)
        bb0 = build_space(holdouts, cfg, args, con_ep)
        tr0, tr_lab = collect_embeddings(bb0, tel)
        te0, te_lab = collect_embeddings(bb0, test_loader)
        torch.manual_seed(1000)
        a_pre, _, _ = exp29.linear_probe_novelty(tr0, tr_lab, te0, te_lab,
                                                 holdouts)
        m = np.isin(tr_lab, seen)
        means0 = exp28.fill_means(
            exp28.class_centroids(tr0[m], tr_lab[m], seen), seen,
            dict(pair_dist=cfg["pair_dist"], n_classes=n_cls)).detach()
        print(f"  pre probe={a_pre:.4f}", flush=True)

        for q in args.quantiles:
            bb = copy.deepcopy(bb0)
            _, hist = run_discovery(
                bb, means0.clone(), base_ds=train_loader.dataset,
                train_eval_loader=tel, test_loader=test_loader, seen=seen,
                holdouts=holdouts, dataset_name=ds,
                rep_weight=cfg["rep_weight"],
                sigreg_weight=cfg["sigreg_weight"],
                n_slices=cfg["n_slices"], rounds=2, ft_epochs=ft_ep,
                tau_quantile=q, names=None, seed=args.seed)
            trp, _ = collect_embeddings(bb, tel)
            tep, _ = collect_embeddings(bb, test_loader)
            torch.manual_seed(1000)
            a_post, _, _ = exp29.linear_probe_novelty(trp, tr_lab, tep,
                                                      te_lab, holdouts)
            results[f"h{size}:q{q}"] = dict(
                probe_pre=a_pre, probe_post=a_post,
                pool=[h["pool"] for h in hist],
                purity=[h["purity"] for h in hist],
                margin=[h["margin"] for h in hist])
            print(f"  [h{size} q={q}] purity " +
                  " ".join(f"r{h['round']}={h['purity']:.3f}"
                           f"(n={h['pool']})" for h in hist) +
                  f"  probe {a_pre:.4f}->{a_post:.4f}", flush=True)
            del bb
            torch.cuda.empty_cache()
        del bb0
        torch.cuda.empty_cache()

    print("\n===== EXP89 SUMMARY (C100 purity gate grid) =====")
    print(f"  {'cfg':<12}{'pur r1':>8}{'pur r2':>8}{'pool r1':>9}"
          f"{'probe pre->post':>18}")
    for k, r in results.items():
        pur = r["purity"] + [float("nan")] * (2 - len(r["purity"]))
        print(f"  {k:<12}{pur[0]:>8.3f}{pur[1]:>8.3f}{r['pool'][0]:>9}"
              f"{r['probe_pre']:>9.4f}->{r['probe_post']:.4f}")

    os.makedirs(os.path.join("logs", "exp89"), exist_ok=True)
    np.savez(os.path.join("logs", "exp89", "results.npz"),
             summary=np.array([repr(results)], dtype=object))
    print("EXP89 DONE.")


if __name__ == "__main__":
    main()
