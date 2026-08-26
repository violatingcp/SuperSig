"""
Experiment 67: from-scratch label-free pretrainings (fresh base models).

Trains ResNet-20 backbones from RANDOM init (pretrain=None -- no hub
weights, removing the hub-trunk caveat that applies to every prior arm)
on CIFAR-100 seen classes (holdout 4 excluded), 100-D heads, long
schedule.  Label-free arms:

  simclr   NT-Xent (temp 0.5)
  visreg   MSE invariance + global SIGReg (the LeJEPA/VISReg objective)
  nplm     instance / bilinear / nplm + global SIGReg (lam=1, tau=1)

Supervised arms:

  supcon   SupCon (temp 0.1), no marginal
  supsig   the repulse/proto recipe (train_sigreg_hybrid)
  nplmcw   supervised / distance / nplm + CLASSWISE SIGReg (cw_lam=5)
  ssig     SupCon (temp 0.1) + global SIGReg, ss_lam=5   [= exp-70 "ss-ft"]
  nplmsd   supervised / distance / nplm + global SIGReg  [= exp-70 "nplm-sup-ft"]

WHY ssig/nplmsd EXIST.  Every CIFAR cell elsewhere in the campaign builds its
trunk with pretrain=ds, i.e. from hub weights that already saw the held-out
class (see supersig/models.py:80-82) -- backbone-level label leakage in exactly
the cells used to demonstrate discovery.  This script is the leakage-free
lineage, but it originally carried only ONE of the three objectives the paper
shortlists (supcon).  `ssig` and `nplmsd` mirror exp-70's ss-ft and nplm-sup-ft
step functions exactly, so the scratch and fine-tune lineages run the SAME
objective and the leakage caveat can be discharged for all three.

NAMING HAZARD.  `ssig` uses ss_lam=5, matching 70_cars_ft_suite.py:88.  The arm
exp-50 calls "supcon_sigreg" uses the HybridContrastiveLoss default lam=1.
Those are DIFFERENT objectives sharing a family name; do not pool their numbers.

Checkpoints saved every 25 epochs and at the end to
checkpoints/scratch_{arm}_{ds}_{dim}d.pt (state_dict; resumable via
--resume).  Ends with a light suite eval (probe/acc/eucl/mahaT) as a
sanity signal.

    python experiments/67_scratch_pretrain.py --arms simclr visreg nplm
    # the paper's leakage-free shortlist (then feed to exp 68):
    python experiments/67_scratch_pretrain.py --arms supcon ssig nplmsd
    python experiments/68_scratch_discovery.py --bases supcon,ssig,nplmsd
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import importlib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")

from supersig.config import DEVICE, REPO_DIR
from supersig.data import (get_cifar_loaders, cifar_two_view_loader,
                           cifar_two_view_balanced_loader,
                           cifar_balanced_loader)
import math
from supersig.losses import (HybridContrastiveLoss, sigreg_loss,
                             supcon_loss, classwise_sigreg_loss,
                             make_anchors)
from supersig.train import train_sigreg_hybrid
from supersig.models import CIFARResNetBackbone
from supersig.recipes import recipe
from supersig.train import collect_embeddings

exp28 = importlib.import_module("28_concat_residual")
exp29 = importlib.import_module("29_residual_finetune")

CKPT_DIR = os.path.join(REPO_DIR, "checkpoints")
ARMS = ["simclr", "visreg", "nplm", "supcon", "supsig",
        "nplmcw", "ssig", "nplmsd"]
LABELED = {"supcon", "nplmcw", "ssig", "nplmsd"}


def pretrain(arm, net, loader, epochs, ckpt, lam, tau, n_slices,
             start_ep=0, lr=1e-3, cw_means=None, cw_lam=5.0, ss_lam=5.0):
    nplm_fn = HybridContrastiveLoss(positives="instance", critic="bilinear",
                                    estimator="nplm", marginal="sigreg",
                                    tau=tau, lam=lam, n_slices=n_slices)
    cw_fn = HybridContrastiveLoss(positives="supervised", critic="distance",
                                  estimator="nplm", marginal="none", tau=tau)
    # `nplmsd` is the SCRATCH twin of exp-70's "nplm-sup-ft" (62.make_nplm_step
    # supervised/distance) and `ssig` of its "ss-ft" (70.make_supcon_step(5.0)).
    # Both mirror those steps exactly so the scratch and fine-tune lineages are
    # the same objective.  NOTE ss_lam=5, NOT the lam=1 that exp-50 calls
    # "supcon_sigreg" -- those are different losses sharing a family name.
    sd_fn = HybridContrastiveLoss(positives="supervised", critic="distance",
                                  estimator="nplm", marginal="none", tau=tau)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    for ep in range(start_ep, epochs):
        run_a, run_b, n = 0.0, 0.0, 0
        for batch in loader:
            if arm in LABELED:
                v1, v2, y = batch
                y = y.to(DEVICE)
            else:
                v1, v2 = batch
            v1, v2 = v1.to(DEVICE), v2.to(DEVICE)
            opt.zero_grad()
            if arm == "supcon":
                z = net(torch.cat([v1, v2]))
                a = supcon_loss(F.normalize(z, dim=1),
                                torch.cat([y, y]), temp=0.1)
                b = torch.zeros((), device=DEVICE)
            elif arm == "nplmcw":
                z = net(torch.cat([v1, v2]))
                cls = torch.cat([y, y])
                a, _ = cw_fn(z, cls)
                b = cw_lam * classwise_sigreg_loss(z, cls, cw_means,
                                                   n_slices=n_slices)
            elif arm == "ssig":
                z = net(torch.cat([v1, v2]))
                a = supcon_loss(F.normalize(z, dim=1),
                                torch.cat([y, y]), temp=0.1)
                b = ss_lam * sigreg_loss(z, n_slices=n_slices)
            elif arm == "nplmsd":
                z = net(torch.cat([v1, v2]))
                a, _ = sd_fn(z, torch.cat([y, y]))
                b = lam * sigreg_loss(z, n_slices=n_slices)
            elif arm == "simclr":
                z = net(torch.cat([v1, v2]))
                inst = torch.arange(v1.size(0), device=DEVICE)
                a = supcon_loss(F.normalize(z, dim=1),
                                torch.cat([inst, inst]), temp=0.5)
                b = torch.zeros((), device=DEVICE)
            elif arm == "visreg":
                z1, z2 = net(v1), net(v2)
                a = F.mse_loss(z1, z2)
                b = 0.5 * (sigreg_loss(z1, n_slices=n_slices)
                           + sigreg_loss(z2, n_slices=n_slices))
            else:  # nplm
                z = net(torch.cat([v1, v2]))
                inst = torch.arange(v1.size(0), device=DEVICE)
                total, parts = nplm_fn(z, torch.cat([inst, inst]))
                a, b = total, parts["marginal"]
            loss = a + b if arm in ("visreg", "nplmcw", "ssig",
                                    "nplmsd") else a
            loss.backward()
            opt.step()
            run_a += float(a) * v1.size(0)
            run_b += float(b) * v1.size(0)
            n += v1.size(0)
        n = max(n, 1)
        if (ep + 1) % 5 == 0 or ep == start_ep or ep == epochs - 1:
            print(f"  [{arm}] epoch {ep+1}/{epochs}  main={run_a/n:.4f}  "
                  f"aux={run_b/n:.4f}", flush=True)
        if (ep + 1) % 25 == 0 or ep == epochs - 1:
            torch.save({"state_dict": net.state_dict(), "epoch": ep + 1},
                       ckpt)
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--holdout", type=int, default=4)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dim", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    ap.add_argument("--cw-lam", type=float, default=5.0)
    ap.add_argument("--ss-lam", type=float, default=5.0,
                    help="SIGReg weight for the `ssig` arm; 5.0 matches exp-70 ss-ft")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    ds = args.dataset

    cfg = recipe(ds, emb_dim=args.dim)
    n_cls = cfg["n_classes"]
    holdouts = {args.holdout}
    seen = [c for c in range(n_cls) if c not in holdouts]
    epochs = args.epochs or (3 if args.quick else 200)
    print(f"exp67 [{ds}] FROM-SCRATCH pretraining, dim={args.dim}, "
          f"epochs={epochs}, holdout={sorted(holdouts)} excluded, "
          f"arms={args.arms}", flush=True)

    train_loader, test_loader = get_cifar_loaders(quick=args.quick,
                                                  dataset=ds)
    train_eval_loader = DataLoader(train_loader.dataset, batch_size=256,
                                   shuffle=False, num_workers=2)

    for i, arm in enumerate(args.arms):
        print(f"\n===== pretraining from scratch: {arm} =====", flush=True)
        torch.manual_seed(args.seed + 30 + i)
        np.random.seed(args.seed + 30 + i)
        net = CIFARResNetBackbone(args.dim, arch=cfg["arch"],
                                  pretrain=None).to(DEVICE)
        ckpt = os.path.join(CKPT_DIR,
                            f"scratch_{arm}_{ds}_{args.dim}d"
                            f"{'_quick' if args.quick else ''}.pt")
        start_ep = 0
        if args.resume and os.path.exists(ckpt):
            state = torch.load(ckpt, map_location=DEVICE)
            net.load_state_dict(state["state_dict"])
            start_ep = state["epoch"]
            print(f"  resuming {ckpt} at epoch {start_ep}")
        if arm == "supsig":
            means = make_anchors(cfg["pair_dist"] / math.sqrt(2.0),
                                 emb_dim=args.dim,
                                 n_classes=n_cls).clone()
            loader = cifar_balanced_loader(ds, holdout=holdouts,
                                           quick=args.quick)
            done = start_ep
            while done < epochs:
                chunk = min(50, epochs - done)
                train_sigreg_hybrid(net, loader, chunk, means,
                                    mode="repulse", disc="proto", alpha=1.0,
                                    rep_weight=cfg["rep_weight"],
                                    sigreg_weight=cfg["sigreg_weight"],
                                    n_slices=cfg["n_slices"])
                done += chunk
                torch.save({"state_dict": net.state_dict(),
                            "means": means.detach().cpu(), "epoch": done},
                           ckpt)
                print(f"  [supsig] checkpoint at epoch {done}", flush=True)
        else:
            cw_means = None
            if arm == "nplmcw":
                cw_means = make_anchors(cfg["pair_dist"] / math.sqrt(2.0),
                                        emb_dim=args.dim,
                                        n_classes=n_cls).detach()
                loader = cifar_two_view_balanced_loader(
                    ds, holdout=holdouts, quick=args.quick)
            elif arm in LABELED:
                loader = cifar_two_view_loader(quick=args.quick,
                                               labeled=True,
                                               holdout=holdouts, dataset=ds)
            else:
                loader = cifar_two_view_loader(quick=args.quick,
                                               labeled=False,
                                               holdout=holdouts, dataset=ds)
            pretrain(arm, net, loader, epochs, ckpt, lam=args.lam,
                     tau=args.tau, n_slices=cfg["n_slices"],
                     start_ep=start_ep, cw_means=cw_means,
                     cw_lam=args.cw_lam, ss_lam=args.ss_lam)
        print(f"  saved {ckpt}")

        tr, tr_lab = collect_embeddings(net, train_eval_loader)
        te, te_lab = collect_embeddings(net, test_loader)
        m = np.isin(tr_lab, seen)
        cents = exp28.class_centroids(tr[m], tr_lab[m], seen)
        anchors = torch.as_tensor(cents, dtype=torch.float32, device=DEVICE)
        r = exp29.evaluate_space(tr, tr_lab, te, te_lab, anchors, seen,
                                 holdouts)
        torch.manual_seed(1000)
        a, _, _ = exp29.linear_probe_novelty(tr, tr_lab, te, te_lab,
                                             holdouts)
        print(f"  [{arm:<8}] probe={a:.4f} acc={r['acc']:.4f} "
              f"eucl={r['eucl']:.4f} mahaT={r['maha_tied']:.4f}", flush=True)
        del net
        torch.cuda.empty_cache()
    print("Done.")


if __name__ == "__main__":
    main()
