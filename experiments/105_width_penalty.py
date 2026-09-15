"""
Experiment 105 (IMPROVEMENT_TESTS.md #105): a DIRECT class-conditional
width penalty.

Exp 104: no space achieves unit class-conditional width (rms 0.13-0.79),
so distances overstate delta-logL by 4-37x.  SIGReg pins the aggregate
marginal; the classwise marginal never reaches its anchors (exp 88).  A
width penalty needs no anchors:

    L_w = lambda_w * ( E_batch ||z - mu_y||^2 / d  -  1 )^2

with mu_y an EMA class centroid (momentum 0.9, initialised at first
sight).  Smallest intervention that targets the stated ideal.

Arms (C10 32-D and C100 100-D, paired seeds): supcon (plain), supcon_sigreg
(hybrid softmax + global SIGReg), res-cat (supcon parent -> res child on
C10 / res-nplm child on C100, width applied to BOTH halves).  lambda_w in
{0 (baseline), 0.1, 1, 10}; stage 1 scans lambda at seed 0, stage 2
multiseeds baseline + chosen lambda.  Report the full exp-104 panel plus
probe / per-event.

Prediction: rms -> 1 and slope -> 1; r_llr and ECE improve; sep degrades
somewhat.  WATCHED FALSIFIER (the campaign's meta-lesson, stated up
front): rms reaches 1 and nothing else moves, or probe/per-event fall as
in exp 83 -> the distance-as-logL identity is decorative; publish as a
significant negative.  Second falsifier: rms -> 1 but slope does not
follow -> the departure is anisotropy/shape, not scale.

    python experiments/105_width_penalty.py --datasets cifar10 --lams 0,1 --quick
    python experiments/105_width_penalty.py --lams 0,0.1,1,10 --seeds 1
    python experiments/105_width_penalty.py --lams 0,<best> --seeds 5
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import copy
import importlib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from supersig.config import DEVICE
from supersig.data import get_cifar_loaders, cifar_two_view_loader
from supersig.losses import HybridContrastiveLoss, sigreg_loss, supcon_loss
from supersig.models import CIFARResNetBackbone
from supersig.recipes import recipe
from supersig.train import collect_embeddings

exp28 = importlib.import_module("28_concat_residual")
exp29 = importlib.import_module("29_residual_finetune")
exp30 = importlib.import_module("30_power_curves")
exp74 = importlib.import_module("74_cifar_residual_discovery")
exp104 = importlib.import_module("104_interpretability_panel")

CKPT = "checkpoints"
ARMS = ["supcon", "supcon_sigreg", "res-cat"]
PANEL_FIELDS = ("r_ll", "slope", "r_llr", "ece", "sw", "rms", "sep")


def train_width(net, loader, epochs, base_loss, lam_w, n_cls, dim, tag,
                mom=0.9, lr=1e-3):
    """Generic loop: base_loss(z, yy) plus the EMA-centroid width penalty."""
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    mu = torch.zeros(n_cls, dim, device=DEVICE)
    seen_cls = torch.zeros(n_cls, dtype=torch.bool, device=DEVICE)
    net.train()
    for ep in range(epochs):
        run_b, run_w, n = 0.0, 0.0, 0
        for v1, v2, y in loader:
            v1, v2, y = v1.to(DEVICE), v2.to(DEVICE), y.to(DEVICE)
            yy = torch.cat([y, y])
            opt.zero_grad()
            z = net(torch.cat([v1, v2])).float()
            loss = base_loss(z, yy)
            w = torch.zeros((), device=DEVICE)
            if lam_w > 0:
                ok = seen_cls[yy]
                if ok.any():
                    wid = (z[ok] - mu[yy[ok]]).pow(2).sum(1).mean() / dim
                    w = lam_w * (wid - 1.0) ** 2
            (loss + w).backward()
            opt.step()
            with torch.no_grad():
                zd = z.detach()
                for c in yy.unique():
                    bm = zd[yy == c].mean(0)
                    if seen_cls[c]:
                        mu[c] = mom * mu[c] + (1 - mom) * bm
                    else:
                        mu[c] = bm
                        seen_cls[c] = True
            run_b += loss.item() * v1.size(0)
            run_w += float(w) * v1.size(0)
            n += v1.size(0)
        if ep == 0 or ep == epochs - 1 or (ep + 1) % 5 == 0:
            print(f"  [{tag}] epoch {ep+1}/{epochs}  base={run_b/n:.4f}  "
                  f"width={run_w/n:.4f}", flush=True)


def base_losses(arm, cfg, lam, cents=None):
    """arm -> base_loss(z, yy) closure."""
    if arm == "supcon":
        return lambda z, yy: supcon_loss(F.normalize(z, dim=1), yy, temp=0.1)
    if arm == "supcon_sigreg":
        return lambda z, yy: (supcon_loss(F.normalize(z, dim=1), yy,
                                          temp=0.1)
                              + lam * sigreg_loss(z,
                                                  n_slices=cfg["n_slices"]))
    if arm == "res":                        # c10 child
        def f(z, yy):
            r = z - cents[yy]
            inst = torch.arange(len(yy) // 2, device=DEVICE)
            return (supcon_loss(F.normalize(r, dim=1),
                                torch.cat([inst, inst]), temp=0.5)
                    + 5.0 * sigreg_loss(r))
        return f
    if arm == "resnplm":                    # c100 child
        loss_fn = HybridContrastiveLoss(positives="instance",
                                        critic="bilinear", estimator="nplm",
                                        marginal="none", tau=1.0)
        def f(z, yy):
            r = z - cents[yy]
            inst = torch.arange(len(yy) // 2, device=DEVICE)
            inter, _ = loss_fn(r, torch.cat([inst, inst]))
            return inter + lam * sigreg_loss(r, n_slices=cfg["n_slices"])
        return f
    raise ValueError(arm)


def battery(tr, tr_lab, te, te_lab, seen, holdouts, alpha, rng):
    m = np.isin(tr_lab, seen)
    anch = exp28.class_centroids(tr[m], tr_lab[m],
                                 seen).detach().float().to(DEVICE)
    r = exp29.evaluate_space(tr, tr_lab, te, te_lab, anch, seen, holdouts)
    aucs = []
    for s in range(3):
        torch.manual_seed(1000 + s)
        a, _, _ = exp29.linear_probe_novelty(tr, tr_lab, te, te_lab,
                                             holdouts)
        aucs.append(a)
    d_te = torch.cdist(torch.as_tensor(te, dtype=torch.float32,
                                       device=DEVICE), anch) \
        .min(1).values.cpu().numpy()
    bg = np.isin(te_lab, seen)
    sg = np.isin(te_lab, list(holdouts))
    pe = exp30.power_at_alpha(d_te[bg], d_te[sg], alpha)
    z, y = tr[m], tr_lab[m]
    if len(z) > 8000:
        idx = rng.choice(len(z), 8000, replace=False)
        z, y = z[idx], y[idx]
    pan = exp104.panel(z, y, classes=np.asarray(seen))
    return dict(probe=float(np.mean(aucs)), eucl=r["eucl"],
                mahaT=r["maha_tied"], lid=r["lid"], perevt=float(pe),
                **{f"p_{k}": float(pan[k]) for k in PANEL_FIELDS})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["cifar10", "cifar100"])
    ap.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    ap.add_argument("--lams", default="0,0.1,1,10",
                    help="comma list of lambda_w (0 = baseline)")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    lams = [float(x) for x in args.lams.split(",")]
    con_ep = args.epochs or (2 if args.quick else 20)
    sfx = "_quick" if args.quick else ""
    out_path = os.path.join("logs", "exp105", f"results{sfx}.npz")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    os.makedirs(CKPT, exist_ok=True)
    done = {}
    if os.path.exists(out_path):
        d = np.load(out_path, allow_pickle=True)
        done = {k: d[k] for k in d.files}
    rng = np.random.default_rng(args.seed)
    holdouts = {args.holdout}

    def lw_tag(lw):
        return f"lw{lw:g}"

    for ds in args.datasets:
        dim = 32 if ds == "cifar10" else 100
        child_arm = "res" if ds == "cifar10" else "resnplm"
        cfg = recipe(ds, emb_dim=dim)
        n_cls = cfg["n_classes"]
        seen = [c for c in range(n_cls) if c not in holdouts]
        train_loader, test_loader = get_cifar_loaders(quick=args.quick,
                                                      dataset=ds)
        tel = DataLoader(train_loader.dataset, batch_size=256,
                         shuffle=False, num_workers=2)

        for si in range(args.seeds):
            for lw in lams:
                # ---- supcon parent (shared by res-cat), ckpt-cached
                ptag = f"exp105_{ds}_{dim}d_supcon_{lw_tag(lw)}_s{si}{sfx}"
                pck = os.path.join(CKPT, ptag + ".pt")
                parent = None

                def get_parent():
                    net = CIFARResNetBackbone(
                        dim, arch=cfg["arch"], pretrain=ds).to(DEVICE)
                    if os.path.exists(pck):
                        net.load_state_dict(torch.load(pck,
                                                       map_location=DEVICE))
                        return net
                    torch.manual_seed(args.seed + 20 + si)
                    np.random.seed(args.seed + 20 + si)
                    loader = cifar_two_view_loader(quick=args.quick,
                                                   labeled=True,
                                                   holdout=holdouts,
                                                   dataset=ds)
                    train_width(net, loader, con_ep,
                                base_losses("supcon", cfg, args.lam), lw,
                                n_cls, dim, ptag)
                    torch.save(net.state_dict(), pck)
                    return net

                for arm in args.arms:
                    key = f"{ds}_{arm}_{lw_tag(lw)}_s{si}"
                    if f"{key}_probe" in done:
                        print(f"  [{key}] cached, skipping", flush=True)
                        continue
                    print(f"\n----- {key} ({con_ep} ep) -----", flush=True)
                    if arm == "supcon":
                        net = get_parent()
                        tr, tr_lab = collect_embeddings(net, tel)
                        te, te_lab = collect_embeddings(net, test_loader)
                        del net
                    elif arm == "supcon_sigreg":
                        torch.manual_seed(args.seed + 20 + si)
                        np.random.seed(args.seed + 20 + si)
                        net = CIFARResNetBackbone(
                            dim, arch=cfg["arch"], pretrain=ds).to(DEVICE)
                        loader = cifar_two_view_loader(quick=args.quick,
                                                       labeled=True,
                                                       holdout=holdouts,
                                                       dataset=ds)
                        train_width(net, loader, con_ep,
                                    base_losses(arm, cfg, args.lam), lw,
                                    n_cls, dim, key)
                        tr, tr_lab = collect_embeddings(net, tel)
                        te, te_lab = collect_embeddings(net, test_loader)
                        del net
                    else:                               # res-cat
                        parent = get_parent()
                        ptr, tr_lab = collect_embeddings(parent, tel)
                        pte, te_lab = collect_embeddings(parent,
                                                         test_loader)
                        m = np.isin(tr_lab, seen)
                        cents = torch.zeros(n_cls, dim, device=DEVICE)
                        cents[torch.as_tensor(seen, device=DEVICE)] = \
                            exp28.class_centroids(
                                ptr[m], tr_lab[m],
                                seen).detach().float().to(DEVICE)
                        ctag = (f"exp105_{ds}_{dim}d_{child_arm}_"
                                f"{lw_tag(lw)}_s{si}{sfx}")
                        cck = os.path.join(CKPT, ctag + ".pt")
                        child = copy.deepcopy(parent)
                        if os.path.exists(cck):
                            child.load_state_dict(
                                torch.load(cck, map_location=DEVICE))
                        else:
                            torch.manual_seed(args.seed + 7 + si)
                            np.random.seed(args.seed + 7 + si)
                            loader = cifar_two_view_loader(
                                quick=args.quick, labeled=True,
                                holdout=holdouts, dataset=ds)
                            train_width(child, loader, con_ep,
                                        base_losses(child_arm, cfg,
                                                    args.lam, cents), lw,
                                        n_cls, dim, ctag)
                            torch.save(child.state_dict(), cck)
                        rtr, _ = collect_embeddings(child, tel)
                        rte, _ = collect_embeddings(child, test_loader)
                        tr = np.concatenate([ptr, rtr], 1)
                        te = np.concatenate([pte, rte], 1)
                        del parent, child
                    torch.cuda.empty_cache()
                    r = battery(tr, tr_lab, te, te_lab, seen, holdouts,
                                args.alpha, rng)
                    for k, v in r.items():
                        done[f"{key}_{k}"] = np.float64(v)
                    np.savez(out_path, **done)
                    print(f"  [{key}] probe={r['probe']:.4f} "
                          f"perevt={r['perevt']:.3f} "
                          f"mahaT={r['mahaT']:.4f}  panel: rms="
                          f"{r['p_rms']:.2f} slope={r['p_slope']:.2f} "
                          f"r_llr={r['p_r_llr']:.2f} ece={r['p_ece']:.3f} "
                          f"sep={r['p_sep']:.2f}", flush=True)

    print(f"\n===== EXP105 SUMMARY (width penalty; ideal rms=slope=1) =====")
    print(f"  {'key':<34}{'probe':>7}{'perevt':>7}{'rms':>6}{'slope':>7}"
          f"{'r_llr':>7}{'ece':>7}{'sep':>7}")
    keys = sorted({k.rsplit("_", 1)[0] for k in done
                   if k.endswith("_probe")})
    for key in keys:
        g = lambda f: float(done.get(f"{key}_{f}", np.nan))
        print(f"  {key:<34}{g('probe'):>7.4f}{g('perevt'):>7.3f}"
              f"{g('p_rms'):>6.2f}{g('p_slope'):>7.2f}{g('p_r_llr'):>7.2f}"
              f"{g('p_ece'):>7.3f}{g('p_sep'):>7.2f}")
    print("EXP105 DONE.")


if __name__ == "__main__":
    main()
