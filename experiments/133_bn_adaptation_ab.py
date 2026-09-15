"""
Experiment 133: is BatchNorm adaptation a construction, or was it a leak?

WHY.  Exp 127 re-ran the frozen density-ratio pool (exp 109) with BatchNorm
actually frozen and the headline moved: h10 purity 0.358 -> 0.268, and the
round-2 rise (0.418) vanished (0.237).  Two things changed at once, though --
the BN fix, and a rebuilt exp-89 space (cuDNN nondeterminism, pitfall 6) -- so
exp 127 could not say how much of the archived gain was BN drift and how much
was the retrain.

What the drift WAS matters for the paper.  A frozen trunk with BN in train
mode is not "frozen": its running statistics keep adapting to the fine-tune
loader, which in discovery contains the pooled (novel) points.  That is
unsupervised test-time normalisation adaptation -- no labels, so legitimate --
but a different construction from "freeze the encoder, iterate the anchors",
and until 2026-08-26 it was happening silently and only on the CIFAR ResNet
trunk (the ViT transfer trunks are LayerNorm).

DESIGN.  A paired A/B on ONE checkpoint per holdout size, the exp-89 space
that exp 89/109/127 cached in checkpoints/exp89_c100_h{1,5,10}_*.pt:

    A  bn_adapt=False   frozen means frozen (the post-fix default)
    B  bn_adapt=True    weights frozen, BN statistics adapt (the old behaviour,
                        now an explicit opt-in: supersig.train.set_train_mode)

Same space, same seed, same loader order, same critic; the ONLY difference is
whether 21 BatchNorm2d layers update their running mean/var during the 5-epoch
anchor fine-tune.  Reported per (holdout size, quantile): purity r1/r2, margin
AUC, post probe, and the embedding drift of the test set relative to the
frozen space (max and mean |z_post - z_pre| against mean |z|), which is the
direct measurement of "did the space move".

PREDICTION.  B reproduces the archived exp-109 signature on the SAME space
that A does not: higher round-1 purity at h10 and a round-2 RISE, with test-set
drift of order the 1.29 exp 130 measured; A shows ~0 drift.  If so, BN
adaptation is a real construction ("corpus-adaptive normalisation helps the
density-ratio pool") and gets its own name and falsifier in the paper.

FALSIFIER.  B ~ A on purity (gap inside the seed noise, ~0.03) -> the archived
0.358 was retrain variance plus BN noise, not adaptation; drop the idea and
quote exp 127's numbers.  Or B < A -> adaptation is harmful and the old numbers
were lucky.

COST.  Evaluation-scale: the spaces are cached, so this is 3 sizes x 2
quantiles x 2 arms of the 2-round frozen loop (~exp 109's discovery half, x2).

    python experiments/133_bn_adaptation_ab.py --selftest
    python experiments/133_bn_adaptation_ab.py
    python experiments/133_bn_adaptation_ab.py --sizes 10 --quantiles 0.99 --quick
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
from supersig.data import get_cifar_loaders
from supersig.recipes import recipe
from supersig.train import collect_embeddings, set_train_mode

exp28 = importlib.import_module("28_concat_residual")
exp29 = importlib.import_module("29_residual_finetune")
exp89 = importlib.import_module("89_c100_rate_grid")
exp92 = importlib.import_module("92_sparker_discovery")

ARMS = {"frozen": False, "bn-adapt": True}


def drift(z_pre, z_post):
    d = np.linalg.norm(z_post - z_pre, axis=1)
    return dict(max=float(d.max()), mean=float(d.mean()),
                mean_abs_z=float(np.linalg.norm(z_pre, axis=1).mean()))


def _selftest():
    """The switch must do exactly one thing: keep BN in train mode on a frozen
    trunk when asked, and nothing else."""
    import torch.nn as nn
    net = nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4), nn.ReLU(),
                        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 2))
    for p in net.parameters():
        p.requires_grad_(False)
    bn = net[1]
    x = torch.randn(8, 3, 8, 8) * 5 + 3

    set_train_mode(net)                       # A: frozen -> eval
    assert not net.training
    m0 = bn.running_mean.clone(); net(x)
    assert torch.equal(bn.running_mean, m0), "eval mode must not touch BN stats"
    print("  bn_adapt=False on a frozen trunk: eval, stats untouched     OK")

    set_train_mode(net, bn_adapt=True)        # B: frozen but BN adapts
    assert net.training
    net(x)
    assert not torch.equal(bn.running_mean, m0), "bn_adapt must update BN stats"
    print("  bn_adapt=True  on a frozen trunk: train, stats move         OK")

    w0 = [p.clone() for p in net.parameters()]
    net(x).sum()                              # no grads anywhere
    assert all(torch.equal(a, b) for a, b in zip(w0, net.parameters()))
    print("  weights stay fixed under bn_adapt                          OK")

    for p in net.parameters():
        p.requires_grad_(True)
    set_train_mode(net, bn_adapt=True)
    assert net.training
    set_train_mode(net)
    assert net.training
    print("  unfrozen trunk: train either way (switch is a no-op)       OK")
    print("\nselftest OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--sizes", nargs="+", type=int, default=[1, 5, 10])
    ap.add_argument("--quantiles", nargs="+", type=float, default=[0.95, 0.99])
    ap.add_argument("--dim", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kernels", type=int, default=16)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--refresh", action="store_true",
                    help="rebuild the exp-89 spaces (DEFEATS the pairing "
                         "with exp 127 -- leave off)")
    ap.add_argument("--out", default="logs/exp133")
    args = ap.parse_args()
    if args.selftest:
        _selftest(); return

    ds = "cifar100"
    cfg = recipe(ds, emb_dim=args.dim)
    n_cls = cfg["n_classes"]
    con_ep = args.epochs or (2 if args.quick else 20)
    ft_ep = 1 if args.quick else cfg["ft_epochs"]
    steps = 50 if args.quick else args.steps
    train_loader, test_loader = get_cifar_loaders(quick=args.quick, dataset=ds)
    tel = DataLoader(train_loader.dataset, batch_size=256, shuffle=False,
                     num_workers=2)

    results = {}
    for size in args.sizes:
        holdouts = {4} if size == 1 else set(range(n_cls - size, n_cls))
        seen = [c for c in range(n_cls) if c not in holdouts]
        print(f"\n######## holdout size {size}: BN-adaptation A/B on the "
              f"cached exp-89 space ########", flush=True)
        bb0 = exp89.build_space(holdouts, cfg, args, con_ep)
        bb0.eval()
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
            for arm, adapt in ARMS.items():
                bb = copy.deepcopy(bb0)
                for p in bb.parameters():
                    p.requires_grad_(False)
                torch.manual_seed(args.seed)
                means_out, hist = exp92.sparker_discovery(
                    bb, means0.clone(), base_ds=train_loader.dataset,
                    train_eval_loader=tel, test_loader=test_loader,
                    seen=seen, holdouts=holdouts,
                    rep_weight=cfg["rep_weight"],
                    sigreg_weight=cfg["sigreg_weight"],
                    n_slices=cfg["n_slices"], rounds=2, ft_epochs=ft_ep,
                    tau_quantile=q, M=args.kernels, steps=steps,
                    seed=args.seed, bn_adapt=adapt)
                bb.eval()
                te1, _ = collect_embeddings(bb, test_loader)
                dr = drift(te0, te1)
                torch.manual_seed(1000)
                a_post, _, _ = exp29.linear_probe_novelty(
                    *collect_embeddings(bb, tel), te1, te_lab, holdouts)
                key = f"h{size}:q{q}:{arm}"
                results[key] = dict(
                    size=size, q=q, arm=arm, bn_adapt=adapt,
                    probe_pre=float(a_pre), probe_post=float(a_post),
                    pool=[h["pool"] for h in hist],
                    purity=[h["purity"] for h in hist],
                    margin=[h.get("margin", float("nan")) for h in hist],
                    drift=dr)
                print(f"  [h{size} q={q} {arm:9s}] purity " +
                      " ".join(f"r{h['round']}={h['purity']:.3f}"
                               f"(n={h['pool']})" for h in hist) +
                      f"  probe {a_pre:.3f}->{a_post:.3f}"
                      f"  drift max={dr['max']:.3f} mean={dr['mean']:.3f}"
                      f" (|z|={dr['mean_abs_z']:.2f})", flush=True)
                del bb
                torch.cuda.empty_cache()
        del bb0
        torch.cuda.empty_cache()

    print("\n===== EXP133 SUMMARY (paired on one checkpoint; gate 0.15) =====")
    print(f"  {'cfg':<10}{'frozen r1/r2':>16}{'bn-adapt r1/r2':>18}"
          f"{'d purity r1':>13}{'drift(adapt)':>14}")
    for size in args.sizes:
        for q in args.quantiles:
            A = results[f"h{size}:q{q}:frozen"]
            B = results[f"h{size}:q{q}:bn-adapt"]
            pa = A["purity"] + [float("nan")] * (2 - len(A["purity"]))
            pb = B["purity"] + [float("nan")] * (2 - len(B["purity"]))
            print(f"  h{size}:q{q:<5}{pa[0]:>8.3f}/{pa[1]:<7.3f}"
                  f"{pb[0]:>10.3f}/{pb[1]:<7.3f}{pb[0]-pa[0]:>+13.3f}"
                  f"{B['drift']['max']:>14.3f}")
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "bn_ab.json"), "w") as fh:
        json.dump(results, fh, indent=1, default=float)
    print(f"\nwrote {args.out}/bn_ab.json")
    print("EXP133 DONE.")


if __name__ == "__main__":
    main()
