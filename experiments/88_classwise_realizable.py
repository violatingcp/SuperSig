"""
Experiment 88 (IMPROVEMENT_TESTS.md #88): classwise SIGReg where the anchors
are actually realizable.

Q4 rule: classwise SIGReg is strictly better iff dim >= n_classes; at C100
32/64-D the 100 anchors stall at cent->anchor ~3.5.  The one 100-D run on
record (`nplm_dist_sup_cw` cw-lam1: probe 0.8440, mahaT 0.463 -- best 100-D
mahaT, near the C100 calibration ceiling ~0.47-0.49 vs 0.372 at 32-D) never
reported cent->anchor, so realizability was never verified, and the strict
case dim > n_classes is untested.

Grid: C100, dim {100, 128, 200}, 3 paired seeds, three arms per dim:
  cw      nplm_dist_sup_cw  supervised/distance/nplm + classwise SIGReg
                            (fixed make_anchors means)  -- the test arm
  global  nplm_dist_sup     same interaction + GLOBAL SIGReg  -- isolates
                            the classwise term
  softmax supcon_sigreg     supervised/cosine/softmax + global SIGReg

cent->anchor is reported at every dim (the actual independent variable).
Prediction: cent->anchor collapses toward ~0 once dim >= 100 and mahaT
rises with it, breaking the C100 calibration ceiling; the probe stays low
(calibration-side test).  Falsifiers: cent->anchor collapses but mahaT
does not move (realizability necessary but not sufficient), or
cent->anchor stays high at 100-D (an optimization, not geometric, block).

    python experiments/88_classwise_realizable.py
    python experiments/88_classwise_realizable.py --quick --dims 100 --seeds 1
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import importlib
import math
import numpy as np
import torch
from torch.utils.data import DataLoader

from supersig.config import DEVICE
from supersig.data import (get_cifar_loaders, cifar_two_view_loader,
                           cifar_two_view_balanced_loader)
from supersig.losses import make_anchors
from supersig.models import CIFARResNetBackbone
from supersig.recipes import recipe
from supersig.train import collect_embeddings

exp28 = importlib.import_module("28_concat_residual")
exp29 = importlib.import_module("29_residual_finetune")
exp30 = importlib.import_module("30_power_curves")
exp34h = importlib.import_module("34h_hybrid_nplm_cifar")
exp53 = importlib.import_module("53_nplm_classwise")

ARMS = ["cw", "global", "softmax"]
SPECS = {  # for the exp-34h train_hybrid arms
    "global": dict(positives="supervised", critic="distance",
                   estimator="nplm", marginal="sigreg"),
    "softmax": dict(positives="supervised", critic="cosine",
                    estimator="softmax", marginal="sigreg"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", nargs="+", type=int, default=[100, 128, 200])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    ds = "cifar100"
    holdouts = {args.holdout}
    con_ep = args.epochs or (2 if args.quick else 20)
    out_path = os.path.join("logs", "exp88", "results_c100.npz")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    done = {}
    if os.path.exists(out_path):
        d = np.load(out_path, allow_pickle=True)
        done = {k: d[k] for k in d.files}
    print(f"exp88 realizable classwise C100, dims={args.dims}, "
          f"{args.seeds} paired seeds, {con_ep} ep", flush=True)

    train_loader, test_loader = get_cifar_loaders(quick=args.quick,
                                                  dataset=ds)
    train_eval_loader = DataLoader(train_loader.dataset, batch_size=256,
                                   shuffle=False, num_workers=2)

    for dim in args.dims:
        cfg = recipe(ds, emb_dim=dim)
        seen = [c for c in range(cfg["n_classes"]) if c not in holdouts]
        means = make_anchors(cfg["pair_dist"] / math.sqrt(2.0),
                             emb_dim=dim, n_classes=cfg["n_classes"])
        for si in range(args.seeds):
            for arm in args.arms:
                key = f"{arm}_{dim}d_s{si}"
                if f"{key}_probe" in done:
                    print(f"  [{key}] cached, skipping", flush=True)
                    continue
                print(f"\n----- {key} -----", flush=True)
                torch.manual_seed(args.seed + 20 + si)
                np.random.seed(args.seed + 20 + si)
                net = CIFARResNetBackbone(dim, arch=cfg["arch"],
                                          pretrain=ds).to(DEVICE)
                if arm == "cw":
                    loader = cifar_two_view_balanced_loader(
                        ds, holdout=holdouts, quick=args.quick)
                    exp53.train_nplm_classwise(
                        net, loader, con_ep, "supervised", "distance",
                        means, tau=args.tau, lam=args.lam,
                        n_slices=cfg["n_slices"])
                else:
                    loader = cifar_two_view_loader(quick=args.quick,
                                                   labeled=True,
                                                   holdout=holdouts,
                                                   dataset=ds)
                    exp34h.train_hybrid(net, loader, con_ep, SPECS[arm],
                                        True, lam=args.lam,
                                        n_slices=cfg["n_slices"])

                tr, tr_lab = collect_embeddings(net, train_eval_loader)
                te, te_lab = collect_embeddings(net, test_loader)
                m = np.isin(tr_lab, seen)
                cents = exp28.class_centroids(tr[m], tr_lab[m], seen)
                anch = torch.as_tensor(cents, dtype=torch.float32,
                                       device=DEVICE)
                r = exp29.evaluate_space(tr, tr_lab, te, te_lab, anch,
                                         seen, holdouts)
                aucs = []
                for s in range(3):
                    torch.manual_seed(1000 + s)
                    a, _, _ = exp29.linear_probe_novelty(tr, tr_lab, te,
                                                         te_lab, holdouts)
                    aucs.append(a)
                d_te = torch.cdist(torch.as_tensor(
                    te, dtype=torch.float32, device=DEVICE), anch) \
                    .min(1).values.cpu().numpy()
                bg = np.isin(te_lab, seen)
                sg = np.isin(te_lab, list(holdouts))
                pe = exp30.power_at_alpha(d_te[bg], d_te[sg], args.alpha)
                d_anchor = float((anch - means[seen]).norm(dim=1).mean()) \
                    if arm == "cw" else float("nan")
                done[f"{key}_probe"] = np.float64(np.mean(aucs))
                done[f"{key}_acc"] = np.float64(r["acc"])
                done[f"{key}_eucl"] = np.float64(r["eucl"])
                done[f"{key}_mahaT"] = np.float64(r["maha_tied"])
                done[f"{key}_lid"] = np.float64(r["lid"])
                done[f"{key}_perevt"] = np.float64(pe)
                done[f"{key}_cent_anchor"] = np.float64(d_anchor)
                np.savez(out_path, **done)
                print(f"  [{key}] probe={np.mean(aucs):.4f} "
                      f"eucl={r['eucl']:.4f} mahaT={r['maha_tied']:.4f} "
                      f"lid={r['lid']:.4f} perevt={pe:.3f} "
                      f"cent->anchor={d_anchor:.2f}", flush=True)
                del net
                torch.cuda.empty_cache()

    print("\n===== EXP88 SUMMARY (C100; 32-D refs: cw mahaT 0.372, "
          "cent->anchor ~3.5) =====")
    print(f"  {'arm':<9}{'dim':>5}{'probe mean+-sd':>17}{'eucl':>7}"
          f"{'mahaT':>7}{'perevt':>7}{'cent->anchor':>13}")
    for arm in args.arms:
        for dim in args.dims:
            pr = [float(done[f"{arm}_{dim}d_s{si}_probe"])
                  for si in range(args.seeds)
                  if f"{arm}_{dim}d_s{si}_probe" in done]
            if not pr:
                continue
            g = lambda f: np.mean([float(done[f"{arm}_{dim}d_s{si}_{f}"])
                                   for si in range(args.seeds)
                                   if f"{arm}_{dim}d_s{si}_{f}" in done])
            print(f"  {arm:<9}{dim:>5}{np.mean(pr):>9.4f}+-{np.std(pr):.4f}"
                  f"{g('eucl'):>7.3f}{g('mahaT'):>7.3f}{g('perevt'):>7.3f}"
                  f"{g('cent_anchor'):>13.2f}")
    print("EXP88 DONE.")


if __name__ == "__main__":
    main()
