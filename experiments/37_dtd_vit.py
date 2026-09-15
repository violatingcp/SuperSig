"""
DTD transfer benchmark against the VISReg paper (arXiv 2606.02572).

The paper evaluates SSL-pretrained ViT backbones on DTD (47 texture classes)
with a frozen-feature linear probe at 224x224.  Reported DTD top-1:

    MoCoV3  ViT-B/16  73.7
    DINO    ViT-B/16  74.3      <-- the base used here
    iBOT    ViT-L/16  75.3
    VISReg  ViT-B/16  75.7
    VISReg  ViT-L/14  76.5

Here the top CIFAR-10 configurations from this repo (exp 09-13 leaderboard:
SIGReg repulse + proto / + linear-head CE, SupCon with and without two-view
augmentation) are trained as embedding heads on the SAME base -- frozen DINO
ViT-B/16 CLS features -- following the settled recipe scaled to DTD:
64-dim latent (>= 47 classes), 5-sigma orthogonal seed, inverse-square
repulsion with the class-count-scaled weight, w=1.  A frozen linear probe on
the learned embedding gives the paper-comparable top-1 test accuracy; a plain
linear probe on the raw 768-dim features anchors the comparison to the
paper's DINO row.

Caveat stated up front: the paper's methods are *unsupervised* pretraining +
supervised probe, while these embedding heads are *supervised* on DTD
(train+val labels -- the same labels the final probes see).  The comparison
isolates what the structured-embedding objectives add over a plain linear
probe on an identical frozen base with identical supervision.

Splits: torchvision DTD partition 1 (1880/1880/1880).  Embeddings and final
probes train on train+val; the raw-feature baseline is also reported in the
paper's protocol (train only, probe lr swept on val).

Outputs (plots/): dtd_vitb16_accuracy.png
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import datasets, transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from supersig.config import DEVICE, DATA_DIR, plot_path
from supersig.data import BalancedBatchSampler
from supersig.losses import make_anchors, mean_geometry
from supersig.train import (
    train_sigreg_hybrid, train_supcon, train_supcon_plain, REP_WEIGHT,
)

N_CLASSES = 47
FEAT_DIM = 768
EMB_DIM = 64          # default; --emb-dim overrides (e.g. 100)

PAPER_ROWS = [
    ("MoCoV3 ViT-B/16 (paper)", 73.7),
    ("DINO ViT-B/16 (paper)", 74.3),
    ("iBOT ViT-L/16 (paper)", 75.3),
    ("VISReg ViT-B/16 (paper)", 75.7),
    ("VISReg ViT-L/14 (paper)", 76.5),
]

IMAGENET_NORM = transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))
TF_EVAL = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(), IMAGENET_NORM,
])
# The repo's SupCon stack (RandomResizedCrop + flip + jitter) at ViT resolution.
TF_AUG = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.25, 1.0),
                                 interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.ToTensor(), IMAGENET_NORM,
])


class FeatureHead(nn.Module):
    """Projection head on frozen ViT features (mirrors CIFARResNetBackbone.head)."""

    def __init__(self, emb_dim=EMB_DIM):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(FEAT_DIM, 256), nn.ReLU(),
            nn.Linear(256, emb_dim),
        )

    def forward(self, x):
        return self.head(x)


# --------------------------------------------------------------------------- #
# Frozen DINO ViT-B/16 feature extraction (cached)                            #
# --------------------------------------------------------------------------- #
def load_dino():
    model = torch.hub.load("facebookresearch/dino:main", "dino_vitb16")
    model.eval().to(DEVICE)
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.no_grad()
def extract(model, ds, batch_size=64):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)
    feats, labels = [], []
    for x, y in loader:
        with torch.autocast("cuda", dtype=torch.float16,
                            enabled=DEVICE.type == "cuda"):
            f = model(x.to(DEVICE))
        feats.append(f.float().cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def build_plain_features(args):
    cache = os.path.join(DATA_DIR, "dtd_feats_dino_vitb16.pt")
    if os.path.exists(cache) and not args.refresh:
        print(f"  cached plain features: {cache}")
        return torch.load(cache)
    model = load_dino()
    out = {}
    for split in ("train", "val", "test"):
        ds = datasets.DTD(DATA_DIR, split=split, download=True, transform=TF_EVAL)
        out[split] = extract(model, ds)
        print(f"  extracted {split}: {tuple(out[split][0].shape)}")
    torch.save(out, cache)
    return out


def build_aug_bank(args):
    """A replicas of independently augmented train+val features, fp16 [A,N,768]."""
    cache = os.path.join(DATA_DIR, f"dtd_augfeats_dino_vitb16_a{args.aug_reps}.pt")
    if os.path.exists(cache) and not args.refresh:
        print(f"  cached aug bank: {cache}")
        return torch.load(cache)
    model = load_dino()
    tr = datasets.DTD(DATA_DIR, split="train", download=True, transform=TF_AUG)
    va = datasets.DTD(DATA_DIR, split="val", download=True, transform=TF_AUG)
    ds = torch.utils.data.ConcatDataset([tr, va])
    reps, labels = [], None
    for a in range(args.aug_reps):
        f, y = extract(model, ds)
        reps.append(f.half())
        labels = y
        print(f"  aug replica {a + 1}/{args.aug_reps}")
    bank = {"feats": torch.stack(reps), "labels": labels}
    torch.save(bank, cache)
    return bank


class TwoViewFeatures(Dataset):
    """Two independently augmented feature views of each image, plus label."""

    def __init__(self, bank):
        self.feats = bank["feats"]          # [A, N, D] fp16
        self.labels = bank["labels"]

    def __len__(self):
        return self.feats.size(1)

    def __getitem__(self, idx):
        a1, a2 = torch.randperm(self.feats.size(0))[:2]
        return (self.feats[a1, idx].float(), self.feats[a2, idx].float(),
                self.labels[idx])


# --------------------------------------------------------------------------- #
# Probes                                                                       #
# --------------------------------------------------------------------------- #
def probe_accuracy(Ztr, ytr, Ztest, ytest, epochs, lr=1e-3, batch_size=256):
    """Frozen-embedding linear probe (repo default: Adam, CE) -> test top-1."""
    head = nn.Linear(Ztr.size(1), N_CLASSES).to(DEVICE)
    loader = DataLoader(TensorDataset(Ztr, ytr), batch_size=batch_size,
                        shuffle=True)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    for _ in range(epochs):
        for z, y in loader:
            z, y = z.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(head(z), y).backward()
            opt.step()
    with torch.no_grad():
        pred = head(Ztest.to(DEVICE)).argmax(1).cpu()
    return (pred == ytest).float().mean().item()


def swept_probe(Ztr, ytr, Zval, yval, Ztest, ytest, epochs,
                lrs=(1e-4, 3e-4, 1e-3, 3e-3, 1e-2)):
    """Paper-style protocol: sweep probe lr on val, report test at the best."""
    best_lr, best_val = None, -1.0
    for lr in lrs:
        acc = probe_accuracy(Ztr, ytr, Zval, yval, epochs, lr=lr)
        print(f"    lr={lr:g}  val acc={acc:.4f}")
        if acc > best_val:
            best_lr, best_val = lr, acc
    test = probe_accuracy(Ztr, ytr, Ztest, ytest, epochs, lr=best_lr)
    return test, best_val, best_lr


@torch.no_grad()
def embed(head, X, batch_size=1024):
    head.eval()
    return torch.cat([head(X[i:i + batch_size].to(DEVICE)).cpu()
                      for i in range(0, X.size(0), batch_size)])


# --------------------------------------------------------------------------- #
# Method training (embedding heads on frozen features)                        #
# --------------------------------------------------------------------------- #
def balanced_loader(X, y, classes_per_batch=24, per_class=24):
    sampler = BalancedBatchSampler(y.tolist(), classes_per_batch, per_class)
    print(f"  balanced feature loader: {X.size(0)} samples, "
          f"{len(sampler)} batches of {sampler.n_classes}x{per_class}")
    return DataLoader(TensorDataset(X, y), batch_sampler=sampler)


def train_sigreg(X, y, disc, args):
    head = FeatureHead(args.emb_dim).to(DEVICE)
    means = make_anchors(args.pair_dist / math.sqrt(2.0), emb_dim=args.emb_dim,
                         n_classes=N_CLASSES).clone()
    d0, _ = mean_geometry(means)
    print(f"  seed anchors: pairwise distance={d0:.2f} sigma")
    rep_w = REP_WEIGHT * 45.0 / (N_CLASSES * (N_CLASSES - 1) / 2)
    loader = balanced_loader(X, y)
    train_sigreg_hybrid(head, loader, args.embed_epochs, means, mode="repulse",
                        disc=disc, alpha=1.0, rep_weight=rep_w,
                        sigreg_weight=1.0, n_slices=64)
    return head, means


def train_supcon_aug(bank, args):
    head = FeatureHead(args.emb_dim).to(DEVICE)
    loader = DataLoader(TwoViewFeatures(bank), batch_size=args.batch_size,
                        shuffle=True, drop_last=True)
    train_supcon(head, loader, args.embed_epochs)
    return head


def train_supcon_noaug(X, y, args):
    head = FeatureHead(args.emb_dim).to(DEVICE)
    loader = DataLoader(TensorDataset(X, y), batch_size=args.batch_size,
                        shuffle=True, drop_last=True)
    train_supcon_plain(head, loader, args.embed_epochs)
    return head


@torch.no_grad()
def proto_accuracy(head, means, Xtest, ytest):
    """The Gaussian model's own posterior: argmin ||z - mean_c||."""
    z = embed(head, Xtest).to(DEVICE)
    pred = torch.cdist(z, means).argmin(1).cpu()
    return (pred == ytest).float().mean().item()


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--embed-epochs", type=int, default=None)
    ap.add_argument("--probe-epochs", type=int, default=None)
    ap.add_argument("--aug-reps", type=int, default=None,
                    help="augmented feature replicas for the SupCon two-view bank")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--pair-dist", type=float, default=5.0)
    ap.add_argument("--emb-dim", type=int, default=EMB_DIM)
    ap.add_argument("--refresh", action="store_true",
                    help="rebuild the cached ViT feature banks")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    args.embed_epochs = args.embed_epochs or (5 if args.quick else 120)
    args.probe_epochs = args.probe_epochs or (5 if args.quick else 50)
    args.aug_reps = args.aug_reps or (2 if args.quick else 24)
    print(f"device={DEVICE}  base=DINO ViT-B/16 (frozen)  emb_dim={args.emb_dim}  "
          f"seed pair-dist={args.pair_dist} sigma  embed_epochs={args.embed_epochs}")

    print("\n--- features ---")
    plain = build_plain_features(args)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = plain["train"], plain["val"], plain["test"]
    Xtv, ytv = torch.cat([Xtr, Xva]), torch.cat([ytr, yva])
    bank = build_aug_bank(args)

    results = {}

    print("\n=== linear probe on raw DINO features (paper protocol: "
          "train only, lr swept on val) ===")
    test, val, lr = swept_probe(Xtr, ytr, Xva, yva, Xte, yte, args.probe_epochs)
    print(f"  best lr={lr:g}  val={val:.4f}  TEST={test:.4f}")
    results["Linear probe, raw DINO (train)"] = test

    print("\n=== linear probe on raw DINO features (train+val) ===")
    test = probe_accuracy(Xtv, ytv, Xte, yte, args.probe_epochs)
    print(f"  TEST={test:.4f}")
    results["Linear probe, raw DINO (train+val)"] = test

    print("\n=== SIGReg repulse + proto (top CIFAR-10 config, exp 13) ===")
    head, means = train_sigreg(Xtv, ytv, "proto", args)
    acc = probe_accuracy(embed(head, Xtv), ytv, embed(head, Xte), yte,
                         args.probe_epochs)
    pacc = proto_accuracy(head, means, Xte, yte)
    print(f"  probe TEST={acc:.4f}   proto-posterior TEST={pacc:.4f}")
    results["SIGReg repulse + proto"] = acc
    results["SIGReg + proto (own posterior)"] = pacc

    print("\n=== SIGReg repulse + linear-head CE (exp 12) ===")
    head, _ = train_sigreg(Xtv, ytv, "ce", args)
    acc = probe_accuracy(embed(head, Xtv), ytv, embed(head, Xte), yte,
                         args.probe_epochs)
    print(f"  probe TEST={acc:.4f}")
    results["SIGReg repulse + CE"] = acc

    print("\n=== SupCon, two-view augmentation (exp 09) ===")
    head = train_supcon_aug(bank, args)
    acc = probe_accuracy(embed(head, Xtv), ytv, embed(head, Xte), yte,
                         args.probe_epochs)
    print(f"  probe TEST={acc:.4f}")
    results["SupCon (aug)"] = acc

    print("\n=== SupCon, no augmentation (exp 10) ===")
    head = train_supcon_noaug(Xtv, ytv, args)
    acc = probe_accuracy(embed(head, Xtv), ytv, embed(head, Xte), yte,
                         args.probe_epochs)
    print(f"  probe TEST={acc:.4f}")
    results["SupCon (no aug)"] = acc

    print("\n===== DTD TOP-1 SUMMARY (base: frozen DINO ViT-B/16) =====")
    for name, acc in PAPER_ROWS:
        print(f"  {name:<42} {acc:.1f}")
    for name, acc in results.items():
        print(f"  {name:<42} {100 * acc:.1f}")

    names = [n for n, _ in PAPER_ROWS] + list(results)
    accs = [a for _, a in PAPER_ROWS] + [100 * a for a in results.values()]
    colors = ["#999999"] * len(PAPER_ROWS) + ["#1f77b4"] * len(results)
    plt.figure(figsize=(9, 5))
    ypos = np.arange(len(names))
    plt.barh(ypos, accs, color=colors)
    plt.yticks(ypos, names, fontsize=8)
    plt.xlabel("DTD test top-1 (%)")
    plt.xlim(min(accs) - 3, max(accs) + 1)
    plt.title("DTD: paper benchmarks vs SuperSig heads on frozen DINO ViT-B/16")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    tag = "" if args.emb_dim == EMB_DIM else f"_d{args.emb_dim}"
    plt.savefig(plot_path(f"dtd_vitb16_accuracy{tag}.png"), dpi=150)
    plt.close()
    print(f"\n  saved {plot_path(f'dtd_vitb16_accuracy{tag}.png')}")
    print("Done.")


if __name__ == "__main__":
    main()
