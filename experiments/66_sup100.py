"""
Experiment 66: the supervised-SIGReg recipe ("sup") at arbitrary dim with
the exp-50 metric battery (companion to the 100-D CIFAR-100 study; the
other arms come from exp50/exp53 at --dim 100).

    python experiments/66_sup100.py --dataset cifar100 --dim 100
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

from supersig.config import DEVICE, plot_path
from supersig.data import get_cifar_loaders
from supersig.metrics import gaussianity_summary
from supersig.recipes import supervised_embedding, recipe
from supersig.train import collect_embeddings

exp28 = importlib.import_module("28_concat_residual")
exp29 = importlib.import_module("29_residual_finetune")
exp30 = importlib.import_module("30_power_curves")
exp31 = importlib.import_module("31_sparker_power")
exp32 = importlib.import_module("32_maha_mmd_power")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--holdout", type=int, default=4)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dim", type=int, default=100)
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
    fractions = [float(x) for x in args.fractions.split(",")]
    n_null_pre = 20 if args.quick else 200
    n_sig_toys = 10 if args.quick else 50
    sparker_kw = dict(M=args.kernels, steps=args.steps)
    if args.sparker_sigma > 0:
        sparker_kw.update(sigma0=args.sparker_sigma, sigma_ratio=1.0,
                          n_checkpoints=1)
    dtag = ds if args.dim == 32 else f"{ds}_{args.dim}d"
    print(f"exp66 [sup {dtag}] supervised-SIGReg recipe, "
          f"holdout={sorted(holdouts)}")

    net, means, _ = supervised_embedding(ds, holdouts=holdouts,
                                         quick=args.quick,
                                         seed=args.seed + 10,
                                         emb_dim=args.dim)
    means = means.detach()

    train_loader, test_loader = get_cifar_loaders(quick=args.quick, dataset=ds)
    train_eval_loader = DataLoader(train_loader.dataset, batch_size=256,
                                   shuffle=False, num_workers=2)
    tr, tr_lab = collect_embeddings(net, train_eval_loader)
    te, te_lab = collect_embeddings(net, test_loader)
    anchors = means[seen]
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
    print(f"\n  [sup{args.dim}] probe={pm:.4f}+-{psd:.4f} acc={r['acc']:.4f} "
          f"supAUC={r['sup_auc']:.4f} eucl={r['eucl']:.4f} "
          f"mahaT={r['maha_tied']:.4f} mahaPC={r['maha_pc']:.4f} "
          f"perevt={pe:.3f}")
    exp28.print_gauss_table({f"sup{args.dim}": g})

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
        print("  sparker")
        pre_power["sparker"], _ = exp31.run_test_battery(
            bg, sg, R, fractions, args.n_d, n_null_pre, n_sig_toys,
            args.alpha, args.seed, sparker_kw, tag="pre-spk")
        maha_fn, mmd_fn, n_bg, n_sig = exp32.make_stats_fns(
            tr, tr_lab, te, te_lab, seen, holdouts, args.seed)
        print("  maha")
        pre_power["maha"], _ = exp32.battery(
            maha_fn, n_bg, n_sig, fractions, args.n_d, n_null_pre,
            n_sig_toys, args.alpha, args.seed, tag="pre-maha")
        print("  mmd")
        pre_power["mmd"], _ = exp32.battery(
            mmd_fn, n_bg, n_sig, fractions, args.n_d, n_null_pre,
            n_sig_toys, args.alpha, args.seed, tag="pre-mmd")
        print(f"\n===== EXP66 PRE POWER ({dtag}) =====")
        print(f"  {'stat':<10}" + "".join(f"{f:>9}" for f in fractions))
        for st in ("perevent", "sparker", "maha", "mmd"):
            print(f"  {st:<10}"
                  + "".join(f"{p:>9.3f}" for p in pre_power[st]))

    os.makedirs(os.path.join("logs", "exp66"), exist_ok=True)
    np.savez(os.path.join("logs", "exp66", f"sup_{dtag}.npz"),
             fractions=np.array(fractions), probe=pm, probe_sd=psd,
             acc=r["acc"], sup_auc=r["sup_auc"], eucl=r["eucl"],
             mahaT=r["maha_tied"], mahaPC=r["maha_pc"], perevent=pe,
             **{f"{st}_pre": np.array(v) for st, v in pre_power.items()})
    print("Done.")


if __name__ == "__main__":
    main()
