"""
Experiment 110 (IMPROVEMENT_TESTS.md #110): why did the softmax control
break the C100 mahaT ceiling?

Exp 88's unpredicted headline: plain supcon_sigreg at 100-128-D posts
mahaT 0.545-0.558 while holding probe 0.90-0.91, above the 0.47-0.49 band
that bounded every NPLM arm.  Nobody designed it and the cause is unknown:
the dimension, the SIGReg marginal, or their interaction.

Factorial: C100, dims {32, 64, 100, 128, 200} x {supcon (marginal OFF,
plain SupCon), supcon_sigreg (identical supervised/cosine/softmax
interaction + global SIGReg)} x 3 paired seeds -- 30 runs, full battery
plus the exp-104 panel per run (if the ceiling break is a width effect it
shows up as rms moving toward 1 with dimension).

Prediction: the marginal is necessary (plain supcon does not break the
ceiling) and the effect grows with dimension to ~128-D then saturates,
tracking rms.  Falsifier: plain supcon breaks it too -> the effect is
dimensional and the SIGReg marginal is incidental.

    python experiments/110_c100_ceiling.py
    python experiments/110_c100_ceiling.py --dims 100 --seeds 1 --quick
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import importlib
import numpy as np
import torch
from torch.utils.data import DataLoader

from supersig.config import DEVICE
from supersig.data import get_cifar_loaders, cifar_two_view_loader
from supersig.models import CIFARResNetBackbone
from supersig.recipes import recipe
from supersig.train import collect_embeddings, train_supcon

exp34h = importlib.import_module("34h_hybrid_nplm_cifar")
exp105 = importlib.import_module("105_width_penalty")

ARMS = ["supcon", "supcon_sigreg"]
SPEC_SS = dict(positives="supervised", critic="cosine",
               estimator="softmax", marginal="sigreg", tau=0.1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", nargs="+", type=int,
                    default=[32, 64, 100, 128, 200])
    ap.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    ds = "cifar100"
    holdouts = {args.holdout}
    con_ep = args.epochs or (2 if args.quick else 20)
    sfx = "_quick" if args.quick else ""
    out_path = os.path.join("logs", "exp110", f"results{sfx}.npz")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    done = {}
    if os.path.exists(out_path):
        d = np.load(out_path, allow_pickle=True)
        done = {k: d[k] for k in d.files}
    rng = np.random.default_rng(args.seed)

    train_loader, test_loader = get_cifar_loaders(quick=args.quick,
                                                  dataset=ds)
    tel = DataLoader(train_loader.dataset, batch_size=256, shuffle=False,
                     num_workers=2)

    for dim in args.dims:
        cfg = recipe(ds, emb_dim=dim)
        seen = [c for c in range(cfg["n_classes"]) if c not in holdouts]
        for si in range(args.seeds):
            for arm in args.arms:
                key = f"{arm}_{dim}d_s{si}"
                if f"{key}_probe" in done:
                    print(f"  [{key}] cached, skipping", flush=True)
                    continue
                print(f"\n----- {key} ({con_ep} ep) -----", flush=True)
                torch.manual_seed(args.seed + 20 + si)
                np.random.seed(args.seed + 20 + si)
                net = CIFARResNetBackbone(dim, arch=cfg["arch"],
                                          pretrain=ds).to(DEVICE)
                loader = cifar_two_view_loader(quick=args.quick,
                                               labeled=True,
                                               holdout=holdouts,
                                               dataset=ds)
                if arm == "supcon":
                    train_supcon(net, loader, con_ep)
                else:
                    exp34h.train_hybrid(net, loader, con_ep, SPEC_SS,
                                        True, lam=args.lam,
                                        n_slices=cfg["n_slices"])
                tr, tr_lab = collect_embeddings(net, tel)
                te, te_lab = collect_embeddings(net, test_loader)
                del net
                torch.cuda.empty_cache()
                r = exp105.battery(tr, tr_lab, te, te_lab, seen, holdouts,
                                   args.alpha, rng)
                for k, v in r.items():
                    done[f"{key}_{k}"] = np.float64(v)
                np.savez(out_path, **done)
                print(f"  [{key}] probe={r['probe']:.4f} "
                      f"mahaT={r['mahaT']:.4f} perevt={r['perevt']:.3f}  "
                      f"panel: rms={r['p_rms']:.2f} "
                      f"slope={r['p_slope']:.2f} "
                      f"r_llr={r['p_r_llr']:.2f}", flush=True)

    print("\n===== EXP110 SUMMARY (C100 ceiling factorial; exp-88 refs: "
          "softmax 100-128d mahaT 0.545-0.558) =====")
    print(f"  {'arm':<15}{'dim':>5}{'probe':>15}{'mahaT':>15}{'rms':>6}"
          f"{'slope':>7}{'perevt':>7}")
    for arm in args.arms:
        for dim in args.dims:
            pr = [float(done[f"{arm}_{dim}d_s{si}_probe"])
                  for si in range(args.seeds)
                  if f"{arm}_{dim}d_s{si}_probe" in done]
            if not pr:
                continue
            g = lambda f: [float(done[f"{arm}_{dim}d_s{si}_{f}"])
                           for si in range(args.seeds)
                           if f"{arm}_{dim}d_s{si}_{f}" in done]
            mh = g("mahaT")
            print(f"  {arm:<15}{dim:>5}"
                  f"{np.mean(pr):>9.4f}+-{np.std(pr):.4f}"
                  f"{np.mean(mh):>9.4f}+-{np.std(mh):.4f}"
                  f"{np.mean(g('p_rms')):>6.2f}"
                  f"{np.mean(g('p_slope')):>7.2f}"
                  f"{np.mean(g('perevt')):>7.3f}")
    print("EXP110 DONE.")


if __name__ == "__main__":
    main()
