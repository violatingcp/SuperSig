"""
Experiment 57: why does discovery kill SparKer on the NPLM spaces (CIFAR-10)?

Exp-55 post-discovery SparKer power is ~alpha at EVERY fraction for all
three NPLM arms (even f=0.1, where every other post space in the program is
0.58-1.0), while per-event / Mahalanobis on the same spaces work well -- the
signal is there, the test is blind.  Two candidate mechanisms:
  (a) kernel scale: the proto/repulse fine-tune inflates the compact NPLM
      space to pair-dist-5 scale, mismatching the fixed sigma=1 kernel;
  (b) null variance: the fine-tune leaves class clusters hyper-tight, so
      background toys are near-delta spikes and per-spike Poisson count
      fluctuations blow up the null distribution of the fitted discrepancy.

Diagnostic: rebuild the exp-55 nplm_sup_dist post space (same seeds, natural
discovery), print its geometry pre vs post (per-class RMS, centroid scale,
median pairwise distance = the kernel's natural unit), then run the SparKer
battery at f=0.02/0.1 with sigma in {0.3, 1, 3, 10, median-annealed}.
If (a): some sigma recovers power.  If (b): no fixed sigma fully recovers.

    python experiments/57_sparker_diagnosis.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import copy
import importlib
import numpy as np
import torch
from torch.utils.data import DataLoader

from supersig.config import DATA_DIR, DEVICE
from supersig.data import get_cifar_loaders, _cifar_spec
from supersig.recipes import recipe
from supersig.discovery import run_discovery
from supersig.sparker import median_pairwise
from supersig.train import collect_embeddings

exp28 = importlib.import_module("28_concat_residual")
exp29 = importlib.import_module("29_residual_finetune")
exp31 = importlib.import_module("31_sparker_power")
exp55 = importlib.import_module("55_nplm_discovery")


def space_stats(tag, embs, labels, seen, holdouts):
    z = torch.as_tensor(embs, dtype=torch.float32, device=DEVICE)
    m = np.isin(labels, seen)
    cents = exp28.class_centroids(embs[m], labels[m], seen)
    rms = []
    for i, c in enumerate(seen):
        zc = z[torch.as_tensor(labels == c, device=DEVICE)] - cents[i]
        rms.append(float(zc.pow(2).sum(1).mean().sqrt()))
    cd = torch.cdist(cents, cents)
    iu = torch.triu_indices(len(seen), len(seen), offset=1)
    cd = cd[iu[0], iu[1]]
    bg = z[torch.as_tensor(m, device=DEVICE)]
    mp = float(median_pairwise(bg[:4000]))
    print(f"  [{tag}] class RMS mean={np.mean(rms):.3f} "
          f"[{np.min(rms):.3f},{np.max(rms):.3f}]  "
          f"centroid dist mean={cd.mean():.2f} min={cd.min():.2f}  "
          f"median pairwise (bg)={mp:.2f}  "
          f"RMS/median={np.mean(rms)/mp:.3f}")
    return mp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="nplm_sup_dist",
                    choices=exp55.ARMS)
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--holdout", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--fractions", default="0.02,0.1")
    ap.add_argument("--sigmas", default="0.3,1,3,10,0",
                    help="0 = annealed median-heuristic schedule")
    ap.add_argument("--n-d", type=int, default=5000)
    ap.add_argument("--kernels", type=int, default=16)
    ap.add_argument("--steps", type=int, default=300)
    args = ap.parse_args()
    ds = args.dataset

    cfg = recipe(ds, emb_dim=args.dim)
    n_cls = cfg["n_classes"]
    holdouts = {args.holdout}
    seen = [c for c in range(n_cls) if c not in holdouts]
    con_ep = args.epochs or (2 if args.quick else 20)
    ft_ep = 1 if args.quick else cfg["ft_epochs"]
    fractions = [float(x) for x in args.fractions.split(",")]
    sigmas = [float(x) for x in args.sigmas.split(",")]
    n_null = 20 if args.quick else 100
    n_sig_toys = 10 if args.quick else 50
    names = exp29.CIFAR_NAMES if ds == "cifar10" else None
    print(f"exp57 [{ds}] sparker diagnosis, arm={args.arm}, "
          f"sigmas={sigmas}, fractions={fractions}")

    train_loader, test_loader = get_cifar_loaders(quick=args.quick, dataset=ds)
    train_eval_loader = DataLoader(train_loader.dataset, batch_size=256,
                                   shuffle=False, num_workers=2)
    cls, plain, _ = _cifar_spec(ds)
    base = cls(DATA_DIR, train=True, download=True, transform=plain)

    print(f"\n===== training: {args.arm} =====")
    net = exp55.train_arm(args.arm, ds, cfg, args, con_ep, holdouts)
    tr, tr_lab = collect_embeddings(net, train_eval_loader)
    te, te_lab = collect_embeddings(net, test_loader)
    m = np.isin(tr_lab, seen)
    cents = exp28.class_centroids(tr[m], tr_lab[m], seen)
    means = exp28.fill_means(cents, seen, cfg).detach()

    print("\n===== natural discovery (exp-55 protocol) =====")
    bb = copy.deepcopy(net)
    run_discovery(bb, means.clone(), base_ds=base,
                  train_eval_loader=train_eval_loader,
                  test_loader=test_loader, seen=seen, holdouts=holdouts,
                  dataset_name=ds, rep_weight=cfg["rep_weight"],
                  sigreg_weight=cfg["sigreg_weight"],
                  n_slices=cfg["n_slices"], rounds=args.rounds,
                  ft_epochs=ft_ep, names=names, seed=args.seed)
    tr_post, trl_post = collect_embeddings(bb, train_eval_loader)
    te_post, tel_post = collect_embeddings(bb, test_loader)

    print("\n===== space geometry =====")
    space_stats("pre ", te, te_lab, seen, holdouts)
    mp_post = space_stats("post", te_post, tel_post, seen, holdouts)

    def pools(embs_tr, lab_tr, embs_te, lab_te):
        R = torch.as_tensor(embs_tr[np.isin(lab_tr, seen)][:20000],
                            dtype=torch.float32, device=DEVICE)
        bg = torch.as_tensor(embs_te[np.isin(lab_te, seen)],
                             dtype=torch.float32, device=DEVICE)
        sg = torch.as_tensor(embs_te[np.isin(lab_te, list(holdouts))],
                             dtype=torch.float32, device=DEVICE)
        return R, bg, sg

    results = {}
    for tag, (etr, ltr, ete, lte) in (("pre", (tr, tr_lab, te, te_lab)),
                                      ("post", (tr_post, trl_post, te_post,
                                                tel_post))):
        R, bg, sg = pools(etr, ltr, ete, lte)
        for sig in sigmas:
            kw = dict(M=args.kernels, steps=args.steps)
            if sig > 0:
                kw.update(sigma0=sig, sigma_ratio=1.0, n_checkpoints=1)
            label = f"{tag} sigma={'annealed' if sig == 0 else sig}"
            print(f"\n----- {label} -----")
            p, _ = exp31.run_test_battery(bg, sg, R, fractions, args.n_d,
                                          n_null, n_sig_toys, args.alpha,
                                          args.seed, kw, tag=label)
            results[label] = p
            print(f"  {label}: power={np.round(p, 3)}")

    print(f"\n===== EXP57 SUMMARY ({args.arm}, {ds}) =====")
    print(f"  {'setting':<22}" + "".join(f"{f:>9}" for f in fractions))
    for label, p in results.items():
        print(f"  {label:<22}" + "".join(f"{x:>9.3f}" for x in p))
    print(f"  post median pairwise distance = {mp_post:.2f} "
          f"(annealed heuristic centres sigma there)")
    print("Done.")


if __name__ == "__main__":
    main()
