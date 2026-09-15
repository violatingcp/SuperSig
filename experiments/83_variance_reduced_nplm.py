"""
Experiment 83 (IMPROVEMENT_TESTS.md #83): variance-reduced NPLM — bound the
exponent instead of clamping it.

Exp 81: the NPLM seed variance is the bilinear critic's unbounded e^g
(e^g spread 12-35 vs 0.1-0.3 for distance).  Two minimizer-preserving
fixes: (a) the distance parametrization g = -1/2||z-z'||^2/tau + b(x) +
b(x') with a learned bias head (bounded above by b+b', and the only way
a distance critic can reach E_ref[e^g] = 1 -- exps 81/82 showed the pure
distance critic is structurally under-calibrated); (b) self-normalized
importance weighting: an EMA c of log E_ref[e^g], with the interaction
computed on g - c (a known constant shift, correctable, unlike
max-subtraction).

Arms (instance positives, global SIGReg, tau=1, C100 32-D, 5 paired
seeds; the exp-81 bil-inst/dist-inst runs are the archived baselines):
  bil-norm        bilinear + running normalizer
  dist-bias       distance + learned bias
  dist-bias-norm  both

Prediction: sd cut >= 2x vs bil-inst's 0.0320 at equal-or-better mean
(0.8914); per-event power preserved.  Falsifier TO WATCH: variance drops
but per-event drops with it (the normalizer silently removed the
absolute scale = InfoNCE re-derived the hard way).

    python experiments/83_variance_reduced_nplm.py
    python experiments/83_variance_reduced_nplm.py --quick --seeds 1 --arms dist-bias
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import importlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from supersig.config import DEVICE
from supersig.data import get_cifar_loaders, cifar_two_view_loader
from supersig.losses import sigreg_loss
from supersig.models import CIFARResNetBackbone
from supersig.recipes import recipe
from supersig.train import collect_embeddings

exp28 = importlib.import_module("28_concat_residual")
exp29 = importlib.import_module("29_residual_finetune")
exp30 = importlib.import_module("30_power_curves")

ARMS = ["bil-norm", "dist-bias", "dist-bias-norm"]
REF_81 = {"bil-inst": (0.8914, 0.0320, 0.042),
          "dist-inst": (0.7434, 0.0235, 0.000)}


def train_arm(net, bias_head, loader, epochs, arm, lam, n_slices, tau=1.0,
              lr=1e-3, ema=0.99, clamp=30.0):
    params = list(net.parameters())
    if bias_head is not None:
        params += list(bias_head.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    net.train()
    c = torch.zeros((), device=DEVICE)          # running log E_ref[e^g]
    r_hist, s_hist = [], []
    for ep in range(epochs):
        r_run, s_run, n = 0.0, 0.0, 0
        for v1, v2, _ in loader:
            v1, v2 = v1.to(DEVICE), v2.to(DEVICE)
            opt.zero_grad()
            z = net(torch.cat([v1, v2])).float()
            b = v1.size(0)
            inst = torch.arange(b, device=DEVICE)
            labels = torch.cat([inst, inst])
            self_mask = torch.eye(2 * b, dtype=torch.bool, device=DEVICE)
            pos = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~self_mask
            if arm.startswith("bil"):
                g = (z @ z.t()) / tau
            else:
                bb = bias_head(z).squeeze(1)
                g = (-0.5 * torch.cdist(z, z).pow(2)) / tau \
                    + bb.unsqueeze(0) + bb.unsqueeze(1)
            g_eff = g - c.detach() if arm.endswith("norm") else g
            ref = (~pos) & (~self_mask)
            eg = torch.exp(g_eff[ref].clamp(max=clamp))
            inter = (eg - 1.0).mean() - g_eff[pos].mean()
            marg = sigreg_loss(z, n_slices=n_slices)
            (inter + lam * marg).backward()
            opt.step()
            with torch.no_grad():
                if arm.endswith("norm"):
                    lme = torch.logsumexp(
                        g[ref].detach().clamp(max=clamp), 0) \
                        - np.log(int(ref.sum()))
                    c = ema * c + (1 - ema) * lme
                r_run += float((eg.detach() - 1.0).mean()) * b
                s_run += float(eg.detach().std()) * b
                n += b
        n = max(n, 1)
        r_hist.append(r_run / n)
        s_hist.append(s_run / n)
        print(f"    [{arm}] epoch {ep+1}/{epochs}  resid={r_hist[-1]:+.4f}"
              f"  s_exp={s_hist[-1]:.3f}  c={float(c):+.3f}", flush=True)
    return r_hist, s_hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--holdout", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    ds = args.dataset
    cfg = recipe(ds, emb_dim=args.dim)
    holdouts = {args.holdout}
    seen = [c for c in range(cfg["n_classes"]) if c not in holdouts]
    con_ep = args.epochs or (2 if args.quick else 20)
    out_path = os.path.join("logs", "exp83", "results_c100_32d.npz")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    done = {}
    if os.path.exists(out_path):
        d = np.load(out_path, allow_pickle=True)
        done = {k: d[k] for k in d.files}
    print(f"exp83 [{ds} {args.dim}d] variance-reduced NPLM, "
          f"{args.seeds} paired seeds (exp-81 baselines archived)",
          flush=True)

    train_loader, test_loader = get_cifar_loaders(quick=args.quick,
                                                  dataset=ds)
    tel = DataLoader(train_loader.dataset, batch_size=256, shuffle=False,
                     num_workers=2)

    for si in range(args.seeds):
        for arm in args.arms:
            key = f"{arm}_s{si}"
            if f"{key}_probe" in done:
                print(f"  [{key}] cached, skipping", flush=True)
                continue
            print(f"\n----- {key} -----", flush=True)
            torch.manual_seed(args.seed + 20 + si)
            np.random.seed(args.seed + 20 + si)
            net = CIFARResNetBackbone(args.dim, arch=cfg["arch"],
                                      pretrain=ds).to(DEVICE)
            bias_head = (nn.Linear(args.dim, 1).to(DEVICE)
                         if "bias" in arm else None)
            loader = cifar_two_view_loader(quick=args.quick, labeled=True,
                                           holdout=holdouts, dataset=ds)
            r_hist, s_hist = train_arm(net, bias_head, loader, con_ep, arm,
                                       args.lam, cfg["n_slices"])

            tr, tr_lab = collect_embeddings(net, tel)
            te, te_lab = collect_embeddings(net, test_loader)
            m = np.isin(tr_lab, seen)
            anch = exp28.class_centroids(tr[m], tr_lab[m],
                                         seen).detach().float().to(DEVICE)
            r = exp29.evaluate_space(tr, tr_lab, te, te_lab, anch, seen,
                                     holdouts)
            aucs = []
            for s in range(3):
                torch.manual_seed(1000 + s)
                a, _, _ = exp29.linear_probe_novelty(tr, tr_lab, te,
                                                     te_lab, holdouts)
                aucs.append(a)
            d_te = torch.cdist(torch.as_tensor(te, dtype=torch.float32,
                                               device=DEVICE), anch) \
                .min(1).values.cpu().numpy()
            bg = np.isin(te_lab, seen)
            sg = np.isin(te_lab, list(holdouts))
            pe = exp30.power_at_alpha(d_te[bg], d_te[sg], args.alpha)
            for f, v in (("probe", np.mean(aucs)), ("acc", r["acc"]),
                         ("eucl", r["eucl"]), ("mahaT", r["maha_tied"]),
                         ("lid", r["lid"]), ("perevt", pe)):
                done[f"{key}_{f}"] = np.float64(v)
            done[f"{key}_resid_hist"] = np.array(r_hist)
            done[f"{key}_s_exp_hist"] = np.array(s_hist)
            np.savez(out_path, **done)
            print(f"  [{key}] probe={np.mean(aucs):.4f} eucl={r['eucl']:.4f}"
                  f" mahaT={r['maha_tied']:.4f} perevt={pe:.3f} "
                  f"resid_fin={r_hist[-1]:+.4f}", flush=True)
            del net, bias_head
            torch.cuda.empty_cache()

    print(f"\n===== EXP83 SUMMARY (refs exp-81: bil-inst "
          f"{REF_81['bil-inst'][0]:.4f}+-{REF_81['bil-inst'][1]:.4f} "
          f"pe={REF_81['bil-inst'][2]:.3f}; dist-inst "
          f"{REF_81['dist-inst'][0]:.4f}+-{REF_81['dist-inst'][1]:.4f}) "
          f"=====")
    print(f"  {'arm':<16}{'probe mean+-sd':>17}{'eucl':>7}{'mahaT':>7}"
          f"{'perevt':>8}{'|resid|':>9}")
    for arm in args.arms:
        pr = [float(done[f"{arm}_s{si}_probe"]) for si in range(args.seeds)
              if f"{arm}_s{si}_probe" in done]
        if not pr:
            continue
        g = lambda f: np.mean([float(done[f"{arm}_s{si}_{f}"])
                               for si in range(args.seeds)
                               if f"{arm}_s{si}_{f}" in done])
        rf = np.mean([abs(done[f"{arm}_s{si}_resid_hist"][-1])
                      for si in range(args.seeds)
                      if f"{arm}_s{si}_resid_hist" in done])
        print(f"  {arm:<16}{np.mean(pr):>9.4f}+-{np.std(pr):.4f}"
              f"{g('eucl'):>7.3f}{g('mahaT'):>7.3f}{g('perevt'):>8.3f}"
              f"{rf:>9.4f}")
    print("EXP83 DONE.")


if __name__ == "__main__":
    main()
