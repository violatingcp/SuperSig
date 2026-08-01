"""
Experiment 62: aircraft end-to-end fine-tuning with the NPLM loss variants
(exp-49 protocol, all three ViT-B/16 bases).

Repeats exp 49's unsupervised-ft arm with the NPLM corners as the ft
objective, trunk trainable, per base in {dino, lejepa, visreg}:

  nplm-bil-ft   instance / bilinear / nplm  + lam*global SIGReg (label-free)
  nplm-dist-ft  instance / distance / nplm  + lam*global SIGReg (label-free)
  nplm-sup-ft   supervised / distance / nplm + lam*global SIGReg
                (labels DO touch the backbone -- the supervised corner)

Same recipe as exps 43/49: trunk lr 1e-5 / head lr 1e-3, Adam + cosine,
AMP fp16, 20 epochs, batch 32x2 views, emb 100-D head.  Eval = exp-49
closed-set protocol: labeled linear probe on the head (100-d) and trunk
(768-d) features -> test top-1, against the exp-43/49 reference numbers.
Checkpoints cached in checkpoints/ (exp-49 naming).

    python experiments/62_aircraft_nplm_ft.py
    python experiments/62_aircraft_nplm_ft.py --quick --bases dino --arms nplm-bil-ft
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import importlib
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from supersig.config import DEVICE, REPO_DIR, plot_path
from supersig.losses import HybridContrastiveLoss, sigreg_loss

exp40 = importlib.import_module("40_dtd_bases")
exp43 = importlib.import_module("43_dtd_finetune")
exp49 = importlib.import_module("49_aircraft_ssl_ft")

CKPT_DIR = os.path.join(REPO_DIR, "checkpoints")
DS = "aircraft"

# arm -> (positives, critic)
ARMS = {
    "nplm-bil-ft": ("instance", "bilinear"),
    "nplm-dist-ft": ("instance", "distance"),
    "nplm-sup-ft": ("supervised", "distance"),
}


def make_nplm_step(positives, critic, lam, tau, n_slices):
    loss_fn = HybridContrastiveLoss(positives=positives, critic=critic,
                                    estimator="nplm", marginal="none",
                                    tau=tau)
    def step(model, v1, v2, y):
        x = torch.cat([v1, v2]).to(DEVICE, non_blocking=True)
        z = model(x).float()
        if positives == "supervised":
            labels = torch.cat([y, y]).to(DEVICE)
        else:
            inst = torch.arange(v1.size(0), device=DEVICE)
            labels = torch.cat([inst, inst])
        inter, _ = loss_fn(z, labels)
        return inter, lam * sigreg_loss(z, n_slices=n_slices)
    return step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bases", default="dino,lejepa,visreg")
    ap.add_argument("--arms", nargs="+", default=list(ARMS),
                    choices=list(ARMS))
    ap.add_argument("--ft-epochs", type=int, default=None)
    ap.add_argument("--probe-epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr-backbone", type=float, default=1e-5)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--emb-dim", type=int, default=100)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--n-slices", type=int, default=64)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    args.ft_epochs = args.ft_epochs or (2 if args.quick else 20)
    args.probe_epochs = args.probe_epochs or (5 if args.quick else 50)
    bases = args.bases.split(",")
    print(f"exp62 [{DS}] NPLM end-to-end ft, bases={bases}, "
          f"arms={args.arms}, epochs={args.ft_epochs}, lam={args.lam}")

    all_R = {}
    for base in bases:
        print(f"\n######## {DS} on {exp40.BASE_LABELS[base]} ########")
        R = {}
        loader = exp49.two_view_loader(args)
        for arm in args.arms:
            positives, critic = ARMS[arm]
            print(f"--- [{base}] {arm} ({positives}/{critic}) ---")
            torch.manual_seed(args.seed); np.random.seed(args.seed)
            model = exp43.FineTuneModel(base, args.emb_dim)
            ckpt = os.path.join(CKPT_DIR, f"{DS}_ft_{base}_{arm}"
                                f"{'_quick' if args.quick else ''}.pt")
            if os.path.exists(ckpt) and not args.refresh:
                print(f"  loading {ckpt}")
                model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
            else:
                step = make_nplm_step(positives, critic, args.lam, args.tau,
                                      args.n_slices)
                exp49.ft_loop(model, loader, args.ft_epochs, step, args, arm)
                torch.save(model.state_dict(), ckpt)
            (Ttr, Htr, ytr), (Tte, Hte, yte) = exp49.feats_of(model)
            R[f"{arm} head probe"] = exp49.probe(Htr, ytr, Hte, yte,
                                                 args.probe_epochs)
            R[f"{arm} trunk probe"] = exp49.probe(Ttr, ytr, Tte, yte,
                                                  args.probe_epochs)
            del model
            torch.cuda.empty_cache()

        print(f"\n===== [{base}] references (exp 43/44/49) =====")
        for k, v in exp49.REFS[base].items():
            print(f"  [{base}] {k:<34} {v:.1f}")
        for k, v in R.items():
            print(f"  [{base}] {k:<34} {100 * v:.1f}")
        all_R[base] = R

    print("\n===== EXP62 SUMMARY (test top-1, %) =====")
    for base in bases:
        for k, v in all_R[base].items():
            print(f"  [{base}] {k:<34} {100 * v:.1f}")

    labels, vals, colors = [], [], []
    cmap = {"dino": "#2a78d6", "lejepa": "#8c2d9e", "visreg": "#d62728"}
    for base in bases:
        for arm in args.arms:
            labels.append(f"{base}\n{arm.replace('-ft','')}\nhead")
            vals.append(100 * all_R[base][f"{arm} head probe"])
            colors.append(cmap.get(base, "#666"))
    plt.figure(figsize=(1.1 * len(labels) + 2, 5))
    plt.bar(range(len(vals)), vals, color=colors)
    plt.xticks(range(len(vals)), labels, fontsize=7)
    plt.ylabel("test top-1 (%)")
    plt.title(f"exp62: NPLM end-to-end ft on {DS} (head probes)")
    plt.grid(alpha=0.25, axis="y")
    plt.tight_layout()
    out = plot_path(f"exp62_nplm_ft_{DS}.png")
    plt.savefig(out, dpi=150); plt.close()
    print("saved", out)

    os.makedirs(os.path.join("logs", "exp62"), exist_ok=True)
    np.savez(os.path.join("logs", "exp62", f"nplm_ft_{DS}.npz"),
             bases=np.array(bases), arms=np.array(args.arms),
             **{f"{base}__{k.replace(' ', '_')}": v
                for base in bases for k, v in all_R[base].items()})
    print("Done.")


if __name__ == "__main__":
    main()
