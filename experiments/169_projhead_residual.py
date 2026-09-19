"""
Experiment 169: residual children for the exp-168 encoder arms, 10-D CIFAR-10.

For each parent arm the residual child = deepcopy(parent backbone), fine-tuned
with an INSTANCE contrastive term on the residual r = backbone(x) - cent_y plus
lam*SIGReg(r) -- the contrastive geometry MATCHES the parent arm:

  supconproj : cosine NT-Xent through a throwaway projection g (normalized)
  supconeucl : bilinear (raw inner-product) softmax, no normalize
  supcondist : distance (raw squared-distance) softmax, no normalize

The backbone is saved as scratch_{arm}_res_{ds}_{dim}d{htag}.pt so exp 162 loads
it as the "{arm}-res" arm and runs the forward-concat f*(2sigma) battery:

  python experiments/169_projhead_residual.py --arm supconeucl --holdout 4
  python experiments/162_cifar_concat_residual.py --dim 10 --holdout 4 \
      --pairs supconeucl:supconeucl-res --discover residual --out logs/exp169
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import copy
import importlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from supersig.config import DEVICE
from supersig.data import get_cifar_loaders, cifar_two_view_loader
from supersig.models import CIFARResNetBackbone
from supersig.recipes import recipe
from supersig.losses import supcon_loss, sigreg_loss, HybridContrastiveLoss
from supersig.train import collect_embeddings

exp28 = importlib.import_module("28_concat_residual")
CKPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "checkpoints")
HYBRID = {"supconeucl": "bilinear", "supcondist": "distance"}


def make_res_step(cents, arm, proj, tau, lam, n_slices):
    """residual contrastive term (instance positives) matching the parent arm."""
    hyb = None
    if arm in HYBRID:
        hyb = HybridContrastiveLoss(positives="instance", critic=HYBRID[arm],
                                    estimator="softmax", marginal="none",
                                    tau=tau, n_slices=n_slices)

    def step(backbone, v1, v2, y):
        x = torch.cat([v1, v2]).to(DEVICE, non_blocking=True)
        yy = torch.cat([y, y]).to(DEVICE)
        r = backbone(x).float() - cents[yy]                 # tested residual
        inst = torch.arange(v1.size(0), device=DEVICE)
        lab = torch.cat([inst, inst])
        if arm == "supconproj":
            crit = supcon_loss(F.normalize(proj(r), dim=1), lab, temp=0.5)
        elif arm in HYBRID:
            crit = hyb.interaction(r, lab)                  # raw-geometry softmax
        else:  # supconnoaug etc: standard cosine instance NT-Xent on residual
            crit = supcon_loss(F.normalize(r, dim=1), lab, temp=0.5)
        return crit + lam * sigreg_loss(r, n_slices=n_slices)
    return step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--holdout", type=int, default=4)
    ap.add_argument("--arm", default="supconproj",
                    choices=["supconproj", "supconeucl", "supcondist",
                             "supconnoaug"])
    ap.add_argument("--dim", type=int, default=10)
    ap.add_argument("--proj-dim", type=int, default=128)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=5.0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    ds = args.dataset
    cfg = recipe(ds, emb_dim=args.dim)
    n_cls = cfg["n_classes"]
    holdouts = {args.holdout}
    seen = [c for c in range(n_cls) if c not in holdouts]
    htag = "" if args.holdout == 4 else f"_h{args.holdout}"
    pck = os.path.join(CKPT, f"scratch_{args.arm}_{ds}_{args.dim}d{htag}.pt")
    rck = os.path.join(CKPT, f"scratch_{args.arm}_res_{ds}_{args.dim}d{htag}.pt")
    if not os.path.exists(pck):
        print(f"[wait] parent ckpt missing: {pck}", flush=True); sys.exit(2)
    if os.path.exists(rck) and not args.refresh:
        print(f"[done] child already exists: {rck}", flush=True); return

    parent = CIFARResNetBackbone(args.dim, arch=cfg["arch"],
                                 pretrain=None).to(DEVICE)
    st = torch.load(pck, map_location=DEVICE)
    parent.load_state_dict(st["state_dict"] if "state_dict" in st else st)
    parent.eval()
    train_loader, _ = get_cifar_loaders(dataset=ds)
    tel = DataLoader(train_loader.dataset, batch_size=256, shuffle=False,
                     num_workers=2)
    ptr, tr_lab = collect_embeddings(parent, tel)
    m = np.isin(tr_lab, seen)
    cents_full = torch.zeros(n_cls, args.dim, device=DEVICE)
    cents_full[torch.as_tensor(seen, device=DEVICE)] = \
        exp28.class_centroids(ptr[m], tr_lab[m], seen).detach().float().to(DEVICE)

    torch.manual_seed(args.seed + 7); np.random.seed(args.seed + 7)
    child = copy.deepcopy(parent).to(DEVICE)
    proj = (nn.Sequential(nn.Linear(args.dim, args.proj_dim), nn.ReLU(),
                          nn.Linear(args.proj_dim, args.proj_dim)).to(DEVICE)
            if args.arm == "supconproj" else None)
    step = make_res_step(cents_full, args.arm, proj, args.tau, args.lam,
                         cfg["n_slices"])
    params = list(child.parameters()) + (list(proj.parameters())
                                         if proj is not None else [])
    opt = torch.optim.Adam(params, lr=1e-3)
    loader = cifar_two_view_loader(labeled=True, holdout=holdouts, dataset=ds)
    child.train()
    print(f"exp169 [{ds}{htag}] {args.arm} residual child dim={args.dim} "
          f"tau={args.tau} lam={args.lam} epochs={args.epochs}", flush=True)
    for ep in range(args.epochs):
        run, n = 0.0, 0
        for v1, v2, y in loader:
            opt.zero_grad()
            loss = step(child, v1, v2, y)
            loss.backward(); opt.step()
            run += loss.item() * v1.size(0); n += v1.size(0)
        if (ep + 1) % 5 == 0 or ep == 0 or ep == args.epochs - 1:
            print(f"  [{args.arm}-res] epoch {ep+1}/{args.epochs} "
                  f"loss={run/max(n,1):.4f}", flush=True)
    child.eval()
    torch.save(child.state_dict(), rck)
    print(f"  saved {rck}", flush=True)


if __name__ == "__main__":
    main()
