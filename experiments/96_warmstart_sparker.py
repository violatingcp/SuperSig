"""
Experiment 96 (IMPROVEMENT_TESTS.md #96): does the pairwise critic
warm-start the event-level test?

Direct test of the paper's §5 unification: if the NPLM critic learned at
representation time and SparKer's f learned at test time are the same
object at two scales, the trained space's class structure should contain
what SparKer re-learns from scratch.  Warm start = init SparKer's centres
mu at the trained seen-class centroids ("the trained NPLM class anchors"
of the calibrated space; padded to M with data samples) vs the default
data-sample init.

Per space (CIFAR-10 32-D, holdout 4, exp-55 seed-exact reproductions of
nplm_sup_dist and nplm_distance, plus supcon as the non-calibrated
contrast): t_NP trajectories at matched steps (n_checkpoints=6, f=0.02
injection), and power at a REDUCED step budget (steps=50, n_null=50,
25 signal toys) cold vs warm.

Prediction: warm start converges in materially fewer steps at >= power in
the NPLM spaces; the supcon contrast shows a smaller (or no) warm-start
advantage.  Falsifier: no convergence difference anywhere -- the two
scales share a functional form but not learned content.

    python experiments/96_warmstart_sparker.py
    python experiments/96_warmstart_sparker.py --quick --arms nplm_sup_dist
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import importlib
import numpy as np
import torch
from torch.utils.data import DataLoader

from supersig.config import DEVICE
from supersig.data import get_cifar_loaders
from supersig.recipes import recipe
from supersig.sparker import aggregate_pvalues, median_pairwise, np_test_stats
from supersig.train import collect_embeddings

exp28 = importlib.import_module("28_concat_residual")
exp31 = importlib.import_module("31_sparker_power")
exp55 = importlib.import_module("55_nplm_discovery")

ARMS = ["nplm_sup_dist", "nplm_distance", "supcon"]


def power_at_budget(bg, sg, R, mu_init, n_d, frac, n_null, n_toys, steps,
                    alpha, seed):
    rng = np.random.default_rng(seed)
    kw = dict(M=16, steps=steps, mu_init=mu_init)
    sigma0 = median_pairwise(bg, seed=seed)
    null_ts = []
    for i in range(n_null):
        b, _ = exp31.toy_indices(rng, len(bg), len(sg), n_d, 0)
        null_ts.append(np_test_stats(bg[torch.as_tensor(b, device=DEVICE)],
                                     R, sigma0=sigma0, seed=seed + i, **kw))
    null_ts = np.array(null_ts)
    null_agg = np.array([aggregate_pvalues(null_ts[i],
                                           np.delete(null_ts, i, axis=0))
                         for i in range(n_null)])
    thr = np.quantile(null_agg, 1.0 - alpha)
    n_sig = int(round(frac * n_d))
    det = 0
    for j in range(n_toys):
        b, s = exp31.toy_indices(rng, len(bg), len(sg), n_d, n_sig)
        D = torch.cat([bg[torch.as_tensor(b, device=DEVICE)],
                       sg[torch.as_tensor(s, device=DEVICE)]])
        agg = aggregate_pvalues(
            np_test_stats(D, R, sigma0=sigma0, seed=seed + 7919 + j, **kw),
            null_ts)
        det += int(agg > thr)
    return det / n_toys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--holdout", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--frac", type=float, default=0.02)
    ap.add_argument("--n-d", type=int, default=5000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--budget-steps", type=int, default=50)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    ds = args.dataset
    cfg = recipe(ds, emb_dim=args.dim)
    holdouts = {args.holdout}
    seen = [c for c in range(cfg["n_classes"]) if c not in holdouts]
    con_ep = 2 if args.quick else 20
    n_null = 10 if args.quick else 50
    n_toys = 10 if args.quick else 25

    train_loader, test_loader = get_cifar_loaders(quick=args.quick,
                                                  dataset=ds)
    ev = DataLoader(train_loader.dataset, batch_size=256, shuffle=False,
                    num_workers=2)
    results = {}
    for name in args.arms:
        print(f"\n----- reproducing {name} (exp-55 seed-exact) -----",
              flush=True)
        sargs = argparse.Namespace(**vars(args), lam=1.0, tau=1.0)
        net = exp55.train_arm(name, ds, cfg, sargs, con_ep, holdouts)
        tr, tr_lab = collect_embeddings(net, ev)
        te, te_lab = collect_embeddings(net, test_loader)
        del net
        torch.cuda.empty_cache()
        m = np.isin(tr_lab, seen)
        cents = exp28.class_centroids(tr[m], tr_lab[m],
                                      seen).detach().float()
        R = torch.as_tensor(tr[m][:20000], dtype=torch.float32,
                            device=DEVICE)
        bg = torch.as_tensor(te[np.isin(te_lab, seen)], dtype=torch.float32,
                             device=DEVICE)
        sg = torch.as_tensor(te[np.isin(te_lab, list(holdouts))],
                             dtype=torch.float32, device=DEVICE)
        # warm centres: the 9 seen-class centroids + data samples to M=16
        g = torch.Generator().manual_seed(args.seed)
        pad = bg[torch.randperm(len(bg), generator=g)[: 16 - len(cents)]
                 .to(DEVICE)]
        warm = torch.cat([cents.to(DEVICE), pad])

        # trajectory at matched steps, one injected sample
        rng = np.random.default_rng(args.seed)
        n_sig = int(round(args.frac * args.n_d))
        b, s = exp31.toy_indices(rng, len(bg), len(sg), args.n_d, n_sig)
        D = torch.cat([bg[torch.as_tensor(b, device=DEVICE)],
                       sg[torch.as_tensor(s, device=DEVICE)]])
        sigma0 = median_pairwise(bg, seed=args.seed)
        traj = {}
        for tag, mu0 in (("cold", None), ("warm", warm)):
            traj[tag] = np_test_stats(D, R, M=16,
                                      steps=50 if args.quick else 300,
                                      sigma0=sigma0, n_checkpoints=6,
                                      seed=args.seed, mu_init=mu0)
        print(f"  [{name}] t_NP trajectory (6 sigma checkpoints):")
        for tag in ("cold", "warm"):
            print(f"    {tag}: " + " ".join(f"{t:8.1f}" for t in traj[tag]))

        # power at reduced budget
        pw = {}
        for tag, mu0 in (("cold", None), ("warm", warm)):
            pw[tag] = power_at_budget(bg, sg, R, mu0, args.n_d, args.frac,
                                      n_null, n_toys,
                                      args.budget_steps, args.alpha,
                                      args.seed)
            print(f"  [{name}] power@steps={args.budget_steps} "
                  f"f={args.frac} ({tag}) = {pw[tag]:.3f}", flush=True)
        results[name] = dict(traj_cold=traj["cold"], traj_warm=traj["warm"],
                             power_cold=pw["cold"], power_warm=pw["warm"])

    print("\n===== EXP96 SUMMARY (cold vs warm-start SparKer) =====")
    print(f"  {'arm':<15}{'t_NP final c/w':>18}{'power c/w':>14}")
    for name, r in results.items():
        print(f"  {name:<15}{r['traj_cold'][-1]:>8.1f}/"
              f"{r['traj_warm'][-1]:.1f}"
              f"{r['power_cold']:>8.3f}/{r['power_warm']:.3f}")

    os.makedirs(os.path.join("logs", "exp96"), exist_ok=True)
    np.savez(os.path.join("logs", "exp96", "results.npz"),
             summary=np.array([repr(results)], dtype=object))
    print("EXP96 DONE.")


if __name__ == "__main__":
    main()
