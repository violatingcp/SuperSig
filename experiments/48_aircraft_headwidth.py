"""
Head-width ablation on aircraft/DINO: is any of the 100-d gap an artifact
of the 256-unit hidden bottleneck?

Exp 44/47 heads are Linear(768->256)-ReLU-Linear(256->100).  The PCA rank
diagnostic says the aircraft signal needs ~400 linear directions, so the
final 100 dims should dominate the loss -- but the 256 hidden layer is a
second, milder bottleneck.  This ablation crosses:

  hidden width : 0 (direct Linear 768->100), 256 (current), 768 (wide)
  objective    : SupCon (aug), ss[lam5] (SupCon + SIGReg)

on frozen DINO ViT-B/16 aircraft features (cached from exp 44), same
seeds/epochs/probe as exps 44-47.  References: raw 768-d probe 64.9,
PCA-100 55.3, exp-44 heads (=hidden 256) SupCon 60.8 / ss 63.3.

    python experiments/48_aircraft_headwidth.py
    python experiments/48_aircraft_headwidth.py --dataset cars --base dino
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import importlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from supersig.config import DEVICE, plot_path
from supersig.train import train_supcon, train_supcon_sigreg

exp37 = importlib.import_module("37_dtd_vit")
exp44 = importlib.import_module("44_transfer_32d")


class WidthHead(nn.Module):
    def __init__(self, hidden, emb_dim):
        super().__init__()
        if hidden == 0:
            self.head = nn.Linear(exp37.FEAT_DIM, emb_dim)
        else:
            self.head = nn.Sequential(
                nn.Linear(exp37.FEAT_DIM, hidden), nn.ReLU(),
                nn.Linear(hidden, emb_dim))

    def forward(self, x):
        return self.head(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset", default="aircraft")
    ap.add_argument("--base", default="dino")
    ap.add_argument("--widths", default="0,256,768")
    ap.add_argument("--embed-epochs", type=int, default=None)
    ap.add_argument("--probe-epochs", type=int, default=None)
    ap.add_argument("--aug-reps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--emb-dim", type=int, default=100)
    ap.add_argument("--lam", type=float, default=5.0)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    args.embed_epochs = args.embed_epochs or (5 if args.quick else 120)
    args.probe_epochs = args.probe_epochs or (5 if args.quick else 50)
    args.aug_reps = args.aug_reps or (2 if args.quick else 8)
    widths = [int(w) for w in args.widths.split(",")]
    print(f"device={DEVICE}  {args.dataset}/{args.base}  emb_dim="
          f"{args.emb_dim}  hidden widths={widths}")

    plain, bank = exp44.build_features(args.dataset, args.base, args)
    (Xtv, ytv), (Xte, yte) = plain["train"], plain["test"]
    ncls = exp44.N_CLASSES[args.dataset]

    def probe(Z, Zt):
        import torch.nn.functional as F
        head = nn.Linear(Z.size(1), ncls).to(DEVICE)
        loader = DataLoader(TensorDataset(Z, ytv), batch_size=256,
                            shuffle=True)
        opt = torch.optim.Adam(head.parameters(), lr=1e-3)
        for _ in range(args.probe_epochs):
            for z, y in loader:
                z, y = z.to(DEVICE), y.to(DEVICE)
                opt.zero_grad()
                F.cross_entropy(head(z), y).backward()
                opt.step()
        with torch.no_grad():
            pred = head(Zt.to(DEVICE)).argmax(1).cpu()
        return (pred == yte).float().mean().item()

    two_view_lab = DataLoader(exp37.TwoViewFeatures(bank),
                              batch_size=args.batch_size, shuffle=True,
                              drop_last=True)

    results = {}
    for w in widths:
        tag = "linear" if w == 0 else f"hidden {w}"
        print(f"\n--- SupCon (aug), {tag} ---")
        torch.manual_seed(args.seed + 20); np.random.seed(args.seed + 20)
        h = WidthHead(w, args.emb_dim).to(DEVICE)
        train_supcon(h, two_view_lab, args.embed_epochs)
        results[("SupCon", w)] = probe(exp37.embed(h, Xtv),
                                       exp37.embed(h, Xte))
        print(f"\n--- ss[lam{args.lam:g}], {tag} ---")
        torch.manual_seed(args.seed + 30); np.random.seed(args.seed + 30)
        h = WidthHead(w, args.emb_dim).to(DEVICE)
        train_supcon_sigreg(h, two_view_lab, args.embed_epochs, temp=0.1,
                            lam=args.lam, n_slices=64)
        results[("ss[lam5]", w)] = probe(exp37.embed(h, Xtv),
                                        exp37.embed(h, Xte))

    print(f"\n===== {args.dataset}/{args.base} HEAD-WIDTH ABLATION "
          f"({args.emb_dim}-d output) =====")
    print(f"  raw 768d probe reference: 64.9   PCA-100: 55.3")
    print(f"  {'objective':<12}" + "".join(
        f"{('linear' if w == 0 else f'hid {w}'):>10}" for w in widths))
    for obj in ("SupCon", "ss[lam5]"):
        print(f"  {obj:<12}" + "".join(
            f"{100 * results[(obj, w)]:>10.1f}" for w in widths))

    x = np.arange(len(widths))
    plt.figure(figsize=(7, 4.5))
    for obj, c in (("SupCon", "#1f77b4"), ("ss[lam5]", "#d62728")):
        plt.plot(x, [100 * results[(obj, w)] for w in widths], "-o",
                 color=c, label=obj)
    plt.axhline(64.9, ls=":", color="gray", label="raw 768d probe")
    plt.xticks(x, ["linear" if w == 0 else str(w) for w in widths])
    plt.xlabel("hidden width")
    plt.ylabel("test top-1 (%)")
    plt.title(f"{args.dataset}/{args.base}: head width vs accuracy "
              f"({args.emb_dim}-d output)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    out = plot_path(f"headwidth_{args.dataset}_{args.base}.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\n  saved {out}")
    print("Done.")


if __name__ == "__main__":
    main()
