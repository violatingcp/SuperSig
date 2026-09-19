"""
Experiment 168: AutoSciDACT-style SupCon with a PROJECTION HEAD, 10-D CIFAR-10.

Motivation (arxiv/blog autoscidact, 2026): the standard SimCLR/SupCon recipe
trains the contrastive (cosine/normalized) loss on a PROJECTION HEAD output
z = g(h), then DISCARDS g and uses the pre-projection encoder representation
h as the embedding for the downstream statistical test.  Our plain-SupCon arm
instead normalizes the SAME tensor it later tests on (no separate head), so
the tested embedding IS the direct output of the normalized loss.

This script isolates that one architectural choice.  The backbone still emits a
10-D embedding h (identical architecture to plain SupCon); a small MLP head
g: 10 -> proj_dim projects h to the space the SupCon loss normalizes.  After
training g is discarded; discovery + the six-test f*(2sigma) battery run on h.

  net = backbone (-> h, 10-D, TESTED) + proj (-> z, normalized SupCon loss)

The battery mirrors exp 162 (pre = frozen; post = s/sqrt(b) discovery on h),
so the numbers slot directly against the plain-SupCon 10-D cells.

    python experiments/168_projhead_supcon.py --holdout 4          # train + eval
    python experiments/168_projhead_supcon.py --holdout 4 --skip-train
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import copy
import importlib
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from supersig.config import DEVICE
from supersig.data import (get_cifar_loaders, cifar_two_view_loader,
                           _cifar_spec, DATA_DIR)
from supersig.models import CIFARResNetBackbone
from supersig.recipes import recipe
from supersig.losses import supcon_loss, sigreg_loss, HybridContrastiveLoss
from supersig.discovery import run_discovery
from supersig.train import collect_embeddings

exp28 = importlib.import_module("28_concat_residual")

# arm -> output-file stem (proj keeps its historical stem)
STEM = {"supconproj": "projhead_supcon", "supconeucl": "supconeucl",
        "supcondist": "supcondist", "supconnoaug": "supconnoaug"}
# the two normalization-free hybrid arms: SupCon softmax on a raw-geometry
# critic (bilinear = inner product; distance = squared distance) + SIGReg,
# which supplies the scale that F.normalize used to.
HYBRID = {"supconeucl": "bilinear", "supcondist": "distance"}
exp146 = importlib.import_module("146_min_frac_2sigma")
exp162 = importlib.import_module("162_cifar_concat_residual")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(REPO, "checkpoints")
OUT = os.path.join(REPO, "logs", "exp168")
TESTS = ["eucl", "eucl-disc", "maha", "mmd", "sparker", "sparker-anch"]


class ProjHeadNet(nn.Module):
    """Backbone -> h (tested, 10-D); proj MLP -> z (normalized SupCon loss)."""
    def __init__(self, emb_dim, arch, proj_dim=128):
        super().__init__()
        self.backbone = CIFARResNetBackbone(emb_dim, arch=arch, pretrain=None)
        self.proj = nn.Sequential(nn.Linear(emb_dim, proj_dim), nn.ReLU(),
                                  nn.Linear(proj_dim, proj_dim))

    def forward(self, x):            # the embedding used downstream (h)
        return self.backbone(x)

    def project(self, x):            # the space the SupCon loss lives in (z)
        return self.proj(self.backbone(x))


def train_projhead(net, loader, epochs, ckpt, temp, lr, seed):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    for ep in range(epochs):
        run, n = 0.0, 0
        for v1, v2, y in loader:
            v1, v2, y = v1.to(DEVICE), v2.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            z = net.project(torch.cat([v1, v2]))          # loss on projection
            a = supcon_loss(F.normalize(z, dim=1), torch.cat([y, y]),
                            temp=temp)
            a.backward(); opt.step()
            run += float(a) * v1.size(0); n += v1.size(0)
        n = max(n, 1)
        if (ep + 1) % 5 == 0 or ep == 0 or ep == epochs - 1:
            print(f"  [supconproj] epoch {ep+1}/{epochs}  loss={run/n:.4f}",
                  flush=True)
        if (ep + 1) % 25 == 0 or ep == epochs - 1:
            # save the BACKBONE only -> loads straight into CIFARResNetBackbone
            torch.save({"state_dict": net.backbone.state_dict(),
                        "epoch": ep + 1}, ckpt)
    return net


def train_supcon_noaug(net, loader, epochs, ckpt, temp, lr):
    """Supervised SupCon with NO augmentation: a single plain view per image,
    positives = same class within the batch (no augmented pair)."""
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    for ep in range(epochs):
        run, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            z = net(x)                                 # single plain view
            loss = supcon_loss(F.normalize(z, dim=1), y, temp=temp)
            loss.backward(); opt.step()
            run += float(loss) * x.size(0); n += x.size(0)
        n = max(n, 1)
        if (ep + 1) % 5 == 0 or ep == 0 or ep == epochs - 1:
            print(f"  [supconnoaug] epoch {ep+1}/{epochs}  loss={run/n:.4f}",
                  flush=True)
        if (ep + 1) % 25 == 0 or ep == epochs - 1:
            torch.save({"state_dict": net.state_dict(), "epoch": ep + 1}, ckpt)
    return net


def train_hybrid(net, loader, epochs, ckpt, critic, tau, lam, n_slices, lr):
    """SupCon softmax on a RAW-geometry critic (no F.normalize) + lam*SIGReg."""
    loss_fn = HybridContrastiveLoss(positives="supervised", critic=critic,
                                    estimator="softmax", marginal="sigreg",
                                    tau=tau, lam=lam, n_slices=n_slices)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    for ep in range(epochs):
        ra, rb, n = 0.0, 0.0, 0
        for v1, v2, y in loader:
            v1, v2, y = v1.to(DEVICE), v2.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            z = net(torch.cat([v1, v2]))               # RAW embedding (tested)
            total, parts = loss_fn(z, torch.cat([y, y]))
            total.backward(); opt.step()
            ra += float(parts["interaction"]) * v1.size(0)
            rb += float(parts["marginal"]) * v1.size(0); n += v1.size(0)
        n = max(n, 1)
        if (ep + 1) % 5 == 0 or ep == 0 or ep == epochs - 1:
            print(f"  [{critic}] epoch {ep+1}/{epochs}  inter={ra/n:.4f}  "
                  f"sigreg={rb/n:.4f}", flush=True)
        if (ep + 1) % 25 == 0 or ep == epochs - 1:
            torch.save({"state_dict": net.state_dict(), "epoch": ep + 1}, ckpt)
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--holdout", type=int, default=4)
    ap.add_argument("--arm", default="supconproj",
                    choices=["supconproj", "supconeucl", "supcondist",
                             "supconnoaug"])
    ap.add_argument("--dim", type=int, default=10)
    ap.add_argument("--proj-dim", type=int, default=128)
    ap.add_argument("--temp", type=float, default=0.1)
    ap.add_argument("--tau", type=float, default=1.0,
                    help="temperature for the raw-critic hybrid arms")
    ap.add_argument("--lam", type=float, default=5.0,
                    help="SIGReg weight for the hybrid arms (holds the scale)")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--ref-arm", default=None,
                    help="skip training; load scratch_{ref-arm}_... and run "
                         "the identical battery (method-matched control, "
                         "e.g. plain supcon). Output tagged ref_{arm}.")
    ap.add_argument("--fractions", default="0.006,0.01,0.02,0.03,0.05,0.1")
    ap.add_argument("--n-min", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--ft-epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-d", type=int, default=5000)
    ap.add_argument("--kernels", type=int, default=16)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    ds = args.dataset
    cfg = recipe(ds, emb_dim=args.dim)
    n_cls = cfg["n_classes"]
    holdouts = {args.holdout}
    seen = [c for c in range(n_cls) if c not in holdouts]
    fracs = [float(x) for x in args.fractions.split(",")]
    ft_ep = args.ft_epochs or cfg["ft_epochs"]
    htag = "" if args.holdout == 4 else f"_h{args.holdout}"
    os.makedirs(args.out, exist_ok=True)
    arm_tag = args.ref_arm if args.ref_arm else args.arm
    ckpt = os.path.join(CKPT, f"scratch_{arm_tag}_{ds}_{args.dim}d{htag}.pt")

    # ---- train (or reuse) the encoder ------------------------------------
    if args.ref_arm:
        print(f"[ref-arm] method-matched control, loading {ckpt}", flush=True)
    elif args.skip_train and os.path.exists(ckpt):
        print(f"[skip-train] using {ckpt}", flush=True)
    elif args.arm == "supconproj":
        torch.manual_seed(args.seed + 30); np.random.seed(args.seed + 30)
        net = ProjHeadNet(args.dim, cfg["arch"], args.proj_dim).to(DEVICE)
        loader = cifar_two_view_loader(labeled=True, holdout=holdouts,
                                       dataset=ds)
        print(f"exp168 [{ds}] proj-head SupCon dim={args.dim} "
              f"proj={args.proj_dim} temp={args.temp} epochs={args.epochs} "
              f"holdout={args.holdout}", flush=True)
        train_projhead(net, loader, args.epochs, ckpt, args.temp, args.lr,
                       args.seed)
        del net; torch.cuda.empty_cache()
        print(f"  saved {ckpt}", flush=True)
    elif args.arm == "supconnoaug":  # SupCon, NO augmentation (single plain view)
        torch.manual_seed(args.seed + 30); np.random.seed(args.seed + 30)
        net = CIFARResNetBackbone(args.dim, arch=cfg["arch"],
                                  pretrain=None).to(DEVICE)
        cls, plain, _ = _cifar_spec(ds)
        base = cls(DATA_DIR, train=True, download=True, transform=plain)
        s_idx = np.where(np.isin(np.array(base.targets), seen))[0].tolist()
        loader = DataLoader(Subset(base, s_idx), batch_size=256, shuffle=True,
                            num_workers=2, drop_last=True)
        print(f"exp168 [{ds}] supconnoaug (SupCon, NO aug, single view) "
              f"dim={args.dim} temp={args.temp} epochs={args.epochs} "
              f"holdout={args.holdout}", flush=True)
        train_supcon_noaug(net, loader, args.epochs, ckpt, args.temp, args.lr)
        del net; torch.cuda.empty_cache()
        print(f"  saved {ckpt}", flush=True)
    else:  # supconeucl / supcondist -- raw-critic SupCon + SIGReg, no normalize
        torch.manual_seed(args.seed + 30); np.random.seed(args.seed + 30)
        net = CIFARResNetBackbone(args.dim, arch=cfg["arch"],
                                  pretrain=None).to(DEVICE)
        loader = cifar_two_view_loader(labeled=True, holdout=holdouts,
                                       dataset=ds)
        print(f"exp168 [{ds}] {args.arm} ({HYBRID[args.arm]} critic + SIGReg, "
              f"no normalize) dim={args.dim} tau={args.tau} lam={args.lam} "
              f"epochs={args.epochs} holdout={args.holdout}", flush=True)
        train_hybrid(net, loader, args.epochs, ckpt, HYBRID[args.arm],
                     args.tau, args.lam, cfg["n_slices"], args.lr)
        del net; torch.cuda.empty_cache()
        print(f"  saved {ckpt}", flush=True)

    # ---- load backbone (h-space) for the f* battery -----------------------
    net = CIFARResNetBackbone(args.dim, arch=cfg["arch"], pretrain=None).to(DEVICE)
    st = torch.load(ckpt, map_location=DEVICE)
    net.load_state_dict(st["state_dict"] if "state_dict" in st else st)
    net.eval()

    train_loader, test_loader = get_cifar_loaders(dataset=ds)
    tel = DataLoader(train_loader.dataset, batch_size=256, shuffle=False,
                     num_workers=2)
    cls, plain, _ = _cifar_spec(ds)
    base_ds = cls(DATA_DIR, train=True, download=True, transform=plain)
    base_t = np.array(base_ds.targets)
    seen_idx = np.where(np.isin(base_t, seen))[0]
    sig_idx_all = np.where(np.isin(base_t, list(holdouts)))[0]

    R0_tr, tr_lab = collect_embeddings(net, tel)
    R0_te, te_lab = collect_embeddings(net, test_loader)
    m = np.isin(tr_lab, seen)
    means0 = exp28.fill_means(exp28.class_centroids(R0_tr[m], tr_lab[m], seen),
                              seen, cfg).detach()

    stem = f"ref_{args.ref_arm}" if args.ref_arm else STEM[args.arm]
    res_path = os.path.join(args.out,
                            f"{stem}_{ds}_h{args.holdout}_e{args.dim}.json")
    results = json.load(open(res_path)) if os.path.exists(res_path) else \
        dict(fractions=fracs, arm=(args.ref_arm or args.arm),
             proj_dim=args.proj_dim,
             temp=args.temp, pre={t: {} for t in TESTS},
             post={t: {} for t in TESTS}, cut={})

    def z_of(fn, bg_t, sig_t, f, seed):
        na, sa = exp146.toys_battery(fn, len(bg_t), len(sig_t), [f],
                                     args.n_d, 200, 50, seed, tag="")
        return exp146.z_curve(na, sa)[0]

    for i_f, f in enumerate(fracs):
        fk = str(f)
        if fk in results["post"]["eucl"]:
            print(f"[skip] f={f}", flush=True); continue
        n_inj = int(round(f * len(seen_idx) / (1 - f)))
        rng = np.random.default_rng(args.seed * 1000 + i_f)
        inj = rng.choice(sig_idx_all, size=min(n_inj, len(sig_idx_all)),
                         replace=False)
        sub = Subset(base_ds, np.concatenate([seen_idx, inj]).tolist())
        sub_loader = DataLoader(sub, batch_size=256, shuffle=False,
                                num_workers=2)
        bb = copy.deepcopy(net)
        print(f"\n== f={f} ({len(inj)} injected) discovery on h ==", flush=True)
        cur_means, hist = run_discovery(
            bb, means0.clone(), base_ds=sub, train_eval_loader=sub_loader,
            test_loader=test_loader, seen=seen, holdouts=holdouts,
            dataset_name=ds, rep_weight=cfg["rep_weight"],
            sigreg_weight=cfg["sigreg_weight"], n_slices=cfg["n_slices"],
            rounds=args.rounds, ft_epochs=ft_ep, names=None, seed=args.seed,
            pool_score="np", cut_rule="ssb", n_min=args.n_min,
            on_refuse="skip")
        c0 = hist[0].get("cut", {}) if hist else {}
        results["cut"][fk] = dict(ok=bool(c0.get("ok", True)),
                                  pur=float(hist[0]["purity"]) if hist
                                  else float("nan"))
        Rp_tr, _ = collect_embeddings(bb, tel)
        Rp_te, _ = collect_embeddings(bb, test_loader)
        del bb; torch.cuda.empty_cache()
        A = cur_means[n_cls:].detach().cpu().numpy()
        A = A if len(A) else None
        spaces = {"pre": (R0_tr, R0_te, None), "post": (Rp_tr, Rp_te, A)}
        for state, (Ctr, Cte, Aa) in spaces.items():
            _, det = exp162._prep(Ctr, tr_lab, Cte, te_lab, seen, holdouts,
                                  Aa, args)
            bg_t, sig_t, nb = det["bg_t"], det["sig_t"], det["nb"]
            fns = {
                "eucl": det["meanf"](det["s_eu"][:nb], det["s_eu"][nb:]),
                "eucl-disc": (det["meanf"](det["s_ed"][:nb], det["s_ed"][nb:])
                              if det["s_ed"] is not None else None),
                "maha": det["meanf"](det["s_mb"], det["s_ms"]),
                "mmd": det["mmd_fn"], "sparker": det["spk"](None),
                "sparker-anch": (det["spk"](torch.as_tensor(
                    Aa, dtype=torch.float32, device=DEVICE))
                    if (Aa is not None and len(Aa)) else None)}
            for tn, fn in fns.items():
                results[state][tn][fk] = (z_of(fn, bg_t, sig_t, f,
                                               args.seed + i_f)
                                          if fn is not None else None)
            print(f"  {state}: eucl={results[state]['eucl'].get(fk)} "
                  f"ed={results[state]['eucl-disc'].get(fk)} "
                  f"mmd={results[state]['mmd'].get(fk)} "
                  f"spk={results[state]['sparker'].get(fk)}", flush=True)
        json.dump(results, open(res_path, "w"), indent=1)
        print(f"[done] f={f} pur={results['cut'][fk]['pur']:.3f}", flush=True)
    print("\nexp168 done.", flush=True)


if __name__ == "__main__":
    main()
