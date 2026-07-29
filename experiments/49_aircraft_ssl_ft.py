"""
Aircraft: unsupervised fine-tuning (SimCLR / SIGReg-SSL) and residual
fine-tuning on top of the supervised-ft models, with concat evaluation.

Exp 43 on aircraft showed end-to-end fine-tuning inverts the base ranking
(VISReg CE-ft 76.6 vs DINO 65.5) and that even SupCon-ft lifts the
sketching trunks hugely.  Two follow-ups here, per base (dino, visreg):

  A. UNSUPERVISED fine-tuning -- no labels touch the backbone:
       simclr-ft     : NT-Xent (instance positives, temp 0.5), two views
       sigreg-ssl-ft : invariance MSE + global SIGReg to N(0,I), lam=1
                       (the VISReg/LeJEPA-family objective)
     -> labeled linear probes on head (100-d) and trunk (768-d).

  B. RESIDUAL fine-tuning from the exp-43 supervised checkpoints
     (aircraft_ft_{base}_{supcon,ss}.pt): freeze the parent's class
     centroids (head space, train pool), deepcopy the parent, fine-tune
     end-to-end with the hybrid residual objective -- NT-Xent + SIGReg
     lam=5 on z - centroid_y (exp 36 champion objective, both styles at
     once) -> probe parent head, residual head, and the CONCAT
     [parent ; residual] (200-d), plus the trunk concat (1536-d).

Same optimizer schedule as exp 43 (trunk 1e-5 / head 1e-3, Adam+cosine,
AMP, 20 epochs, batch 32x2 views).

    python experiments/49_aircraft_ssl_ft.py
    python experiments/49_aircraft_ssl_ft.py --bases visreg --arms residual
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import copy
import importlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from supersig.config import DEVICE, REPO_DIR, plot_path
from supersig.losses import sigreg_loss, supcon_loss

exp37 = importlib.import_module("37_dtd_vit")
exp43 = importlib.import_module("43_dtd_finetune")
exp40 = importlib.import_module("40_dtd_bases")

CKPT_DIR = os.path.join(REPO_DIR, "checkpoints")
DS = "aircraft"
NCLS = 100

REFS = {
    "dino":   {"raw probe (frozen) [44]": 64.9, "CE-ft [43]": 65.5,
               "SupCon-ft head / trunk [43]": 50.2,
               "ss-ft head / trunk [43]": 43.3},
    "visreg": {"raw probe (frozen) [44]": 37.9, "CE-ft [43]": 76.6,
               "SupCon-ft head / trunk [43]": 62.7,
               "ss-ft head / trunk [43]": 54.5},
    "lejepa": {"raw probe (frozen) [44]": 34.3, "CE-ft [43]": 75.8,
               "SupCon-ft head / trunk [43]": 58.7,
               "ss-ft head / trunk [43]": 49.0},
}


def two_view_loader(args):
    return DataLoader(
        exp43.TwoViewLabeledImages(exp43.train_corpus(DS)),
        batch_size=args.batch_size, shuffle=True, num_workers=8,
        persistent_workers=True, drop_last=True, pin_memory=True)


def ft_loop(model, loader, epochs, step_fn, args, tag):
    opt, sched = exp43.make_optim(model, epochs, args.lr_backbone,
                                  args.lr_head)
    scaler = torch.amp.GradScaler(enabled=DEVICE.type == "cuda")
    model.train()
    for ep in range(epochs):
        run_a, run_b, n = 0.0, 0.0, 0
        for v1, v2, y in loader:
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.float16,
                                enabled=DEVICE.type == "cuda"):
                a, b = step_fn(model, v1, v2, y)
                loss = a + b
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            run_a += a.item() * v1.size(0)
            run_b += float(b) * v1.size(0)
            n += v1.size(0)
        sched.step()
        if (ep + 1) % 5 == 0 or ep == 0 or ep == epochs - 1:
            print(f"  [{tag}] epoch {ep+1}/{epochs}  main={run_a/n:.4f}  "
                  f"aux={run_b/n:.4f}")


def simclr_step(model, v1, v2, y):
    x = torch.cat([v1, v2]).to(DEVICE, non_blocking=True)
    z = model(x)
    inst = torch.arange(v1.size(0), device=DEVICE)
    return supcon_loss(F.normalize(z.float(), dim=1),
                       torch.cat([inst, inst]), temp=0.5), \
        torch.zeros((), device=DEVICE)


def sigreg_ssl_step(model, v1, v2, y):
    z1 = model(v1.to(DEVICE, non_blocking=True))
    z2 = model(v2.to(DEVICE, non_blocking=True))
    inv = F.mse_loss(z1.float(), z2.float())
    reg = 0.5 * (sigreg_loss(z1.float()) + sigreg_loss(z2.float()))
    return inv, reg


def make_residual_step(cents, lam):
    def step(model, v1, v2, y):
        x = torch.cat([v1, v2]).to(DEVICE, non_blocking=True)
        yy = torch.cat([y, y]).to(DEVICE)
        r = model(x).float() - cents[yy]
        inst = torch.arange(v1.size(0), device=DEVICE)
        con = supcon_loss(F.normalize(r, dim=1), torch.cat([inst, inst]),
                          temp=0.5)
        return con, lam * sigreg_loss(r)
    return step


def probe(Z, y, Zt, yt, epochs):
    head = nn.Linear(Z.size(1), NCLS).to(DEVICE)
    loader = DataLoader(TensorDataset(Z, y), batch_size=256, shuffle=True)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    for _ in range(epochs):
        for z, yy in loader:
            z, yy = z.to(DEVICE), yy.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(head(z), yy).backward()
            opt.step()
    with torch.no_grad():
        pred = head(Zt.to(DEVICE)).argmax(1).cpu()
    return (pred == yt).float().mean().item()


def feats_of(model):
    tr = exp43.extract_both(model, DS, "train")
    te = exp43.extract_both(model, DS, "test")
    return tr, te   # each: (trunk, head, labels)


def run_base(base, args):
    print(f"\n######## {DS} on {exp40.BASE_LABELS[base]} ########")
    R = {}
    loader = two_view_loader(args)

    if "ssl" in args.arms:
        for arm, step in (("simclr-ft", simclr_step),
                          ("sigreg-ssl-ft", sigreg_ssl_step)):
            print(f"--- [{base}] {arm} (unsupervised) ---")
            torch.manual_seed(args.seed); np.random.seed(args.seed)
            model = exp43.FineTuneModel(base, args.emb_dim)
            ckpt = os.path.join(CKPT_DIR, f"{DS}_ft_{base}_{arm}"
                                f"{'_quick' if args.quick else ''}.pt")
            if os.path.exists(ckpt) and not args.refresh:
                print(f"  loading {ckpt}")
                model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
            else:
                ft_loop(model, loader, args.ft_epochs, step, args, arm)
                torch.save(model.state_dict(), ckpt)
            (Ttr, Htr, ytr), (Tte, Hte, yte) = feats_of(model)
            R[f"{arm} head probe"] = probe(Htr, ytr, Hte, yte,
                                           args.probe_epochs)
            R[f"{arm} trunk probe"] = probe(Ttr, ytr, Tte, yte,
                                            args.probe_epochs)
            del model
            torch.cuda.empty_cache()

    if "residual" in args.arms:
        for parent in ("supcon", "ss"):
            pck = os.path.join(CKPT_DIR, f"{DS}_ft_{base}_{parent}.pt")
            if not os.path.exists(pck):
                print(f"  !! missing parent checkpoint {pck}, skipping")
                continue
            print(f"--- [{base}] residual-ft on {parent}-ft parent ---")
            par = exp43.FineTuneModel(base, args.emb_dim)
            par.load_state_dict(torch.load(pck, map_location=DEVICE))
            (pTtr, pHtr, ytr), (pTte, pHte, yte) = feats_of(par)
            cents = torch.stack([pHtr[ytr == c].mean(0)
                                 for c in range(NCLS)]).to(DEVICE)
            torch.manual_seed(args.seed + 7); np.random.seed(args.seed + 7)
            res = copy.deepcopy(par)
            ckpt = os.path.join(CKPT_DIR, f"{DS}_ft_{base}_res_{parent}"
                                f"{'_quick' if args.quick else ''}.pt")
            if os.path.exists(ckpt) and not args.refresh:
                print(f"  loading {ckpt}")
                res.load_state_dict(torch.load(ckpt, map_location=DEVICE))
            else:
                ft_loop(res, loader, args.ft_epochs,
                        make_residual_step(cents, args.lam), args,
                        f"res-{parent}")
                torch.save(res.state_dict(), ckpt)
            (rTtr, rHtr, _), (rTte, rHte, _) = feats_of(res)
            R[f"{parent}-ft (parent head)"] = probe(pHtr, ytr, pHte, yte,
                                                    args.probe_epochs)
            R[f"res-{parent} head probe"] = probe(rHtr, ytr, rHte, yte,
                                                  args.probe_epochs)
            R[f"[{parent} ; res] head concat"] = probe(
                torch.cat([pHtr, rHtr], 1), ytr,
                torch.cat([pHte, rHte], 1), yte, args.probe_epochs)
            R[f"[{parent} ; res] trunk concat"] = probe(
                torch.cat([pTtr, rTtr], 1), ytr,
                torch.cat([pTte, rTte], 1), yte, args.probe_epochs)
            del par, res
            torch.cuda.empty_cache()

    for k, v in R.items():
        print(f"  [{base}] {k:<30} {100 * v:.1f}")
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bases", default="dino,visreg")
    ap.add_argument("--arms", default="ssl,residual")
    ap.add_argument("--ft-epochs", type=int, default=None)
    ap.add_argument("--probe-epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr-backbone", type=float, default=1e-5)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--emb-dim", type=int, default=100)
    ap.add_argument("--lam", type=float, default=5.0)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    args.ft_epochs = args.ft_epochs or (1 if args.quick else 20)
    args.probe_epochs = args.probe_epochs or (5 if args.quick else 50)
    args.arms = args.arms.split(",")
    args.dataset = DS   # for exp43.finetune compatibility if reused
    bases = args.bases.split(",")
    print(f"device={DEVICE}  {DS}  bases={bases}  arms={args.arms}  "
          f"ft_epochs={args.ft_epochs}")

    all_r = {b: run_base(b, args) for b in bases}

    print(f"\n===== AIRCRAFT SSL-FT + RESIDUAL-FT SUMMARY =====")
    methods = list(next(iter(all_r.values())))
    print(f"  {'method':<34}" + "".join(f"{b:>10}" for b in bases))
    for m in next(iter(REFS.values())):
        print(f"  {m:<34}"
              + "".join(f"{REFS[b][m]:>10.1f}" for b in bases))
    for m in methods:
        print(f"  {m:<34}"
              + "".join(f"{100 * all_r[b].get(m, float('nan')):>10.1f}"
                        for b in bases))

    x = np.arange(len(methods))
    w = 0.8 / len(bases)
    plt.figure(figsize=(10, 5))
    for i, b in enumerate(bases):
        plt.bar(x + i * w, [100 * all_r[b].get(m, float("nan"))
                            for m in methods], w,
                label=exp40.BASE_LABELS[b])
    plt.xticks(x + 0.4 - w / 2, methods, rotation=25, ha="right", fontsize=7)
    plt.ylabel("test top-1 (%)")
    plt.title("aircraft: unsupervised + residual fine-tuning")
    plt.legend(fontsize=7)
    plt.tight_layout()
    out = plot_path("aircraft_ssl_ft.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\n  saved {out}")
    print("Done.")


if __name__ == "__main__":
    main()
