"""
DTD end-to-end fine-tuning of the ViT bases (unfreezing what exps 37-42 froze).

Every prior DTD study trained heads on FROZEN ViT-B/16 features.  Here the
whole model fine-tunes on DTD train+val: backbone at a low lr (1e-5) with
the head at 1e-3 (Adam + cosine, AMP, two-view augmentations from the
image-space stack).  Three arms per base:

  ce-ft     : trunk + linear(768->47), single-view CE -- the standard
              supervised fine-tune reference (direct test acc + trunk probe)
  ss-ft     : trunk + FeatureHead(emb_dim), SupCon+SIGReg lam=5 on two views
              (the exp 41 single-space champion, now end-to-end)
  supcon-ft : same architecture, plain SupCon -- the uncalibrated control

Evaluation matches exps 37-42: freeze after fine-tuning, extract plain-
transform features, Adam linear probe (train+val -> test top-1) on the head
embedding and on the 768-d trunk features (did fine-tuning improve the
base itself?).

Caveat: SupCon/SIGReg batch statistics run on batch_size*2 = 64 views here
(GPU memory bound), far smaller than the 512-feature batches of the frozen
studies -- the sliced-Wasserstein sketch is noisier per step.

    python experiments/43_dtd_finetune.py
    python experiments/43_dtd_finetune.py --bases visreg --arms ss --quick
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import importlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import datasets
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from supersig.config import DATA_DIR, DEVICE, REPO_DIR, plot_path
from supersig.losses import sigreg_loss, supcon_loss

exp37 = importlib.import_module("37_dtd_vit")
exp40 = importlib.import_module("40_dtd_bases")
exp44 = importlib.import_module("44_transfer_32d")

CKPT_DIR = os.path.join(REPO_DIR, "checkpoints")

# Frozen-base references at the same operating point (100-d heads).
FROZEN_ROWS = {
    "dtd": {
        "dino":   {"raw probe (frozen) [40]": 76.5,
                   "ss[lam5] (frozen) [41]": 77.1,
                   "SupCon aug (frozen) [40]": 75.5},
        "lejepa": {"raw probe (frozen) [40]": 76.2,
                   "ss[lam5] (frozen) [41]": 75.8,
                   "SupCon aug (frozen) [40]": 77.0},
        "visreg": {"raw probe (frozen) [40]": 77.9,
                   "ss[lam5] (frozen) [41]": 78.3,
                   "SupCon aug (frozen) [40]": 76.5},
    },
    "aircraft": {
        "dino":   {"raw probe (frozen) [44]": 64.9,
                   "ss[lam5] (frozen) [44]": 63.3,
                   "SupCon aug (frozen) [44]": 60.8},
        "lejepa": {"raw probe (frozen) [44]": 34.3,
                   "ss[lam5] (frozen) [44]": 38.4,
                   "SupCon aug (frozen) [44]": 48.0},
        "visreg": {"raw probe (frozen) [44]": 37.9,
                   "ss[lam5] (frozen) [44]": 37.8,
                   "SupCon aug (frozen) [44]": 52.4},
    },
}


def n_classes_of(ds):
    return exp37.N_CLASSES if ds == "dtd" else exp44.N_CLASSES[ds]


def train_corpus(ds, transform=None):
    """The labeled training pool (dtd: train+val; else exp 44's split)."""
    if ds == "dtd":
        return labeled_corpus(transform)
    return exp44.make_split(ds, "train", transform)


class TwoViewLabeledImages(Dataset):
    """Two independently augmented views + label (image space)."""

    def __init__(self, base, aug=exp37.TF_AUG):
        self.base, self.aug = base, aug

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, y = self.base[idx]
        return self.aug(img), self.aug(img), y


def labeled_corpus(transform=None):
    tr = datasets.DTD(DATA_DIR, split="train", download=True,
                      transform=transform)
    va = datasets.DTD(DATA_DIR, split="val", download=True,
                      transform=transform)
    return ConcatDataset([tr, va])


class FineTuneModel(nn.Module):
    """Pretrained ViT trunk + projection head, everything trainable."""

    def __init__(self, base, emb_dim, n_out=None):
        super().__init__()
        self.trunk = exp40.LOADERS[base]()
        self.head = (nn.Linear(exp37.FEAT_DIM, n_out) if n_out
                     else exp37.FeatureHead(emb_dim)).to(DEVICE)

    def forward(self, x):
        return self.head(self.trunk(x))


def make_optim(model, epochs, lr_backbone, lr_head):
    opt = torch.optim.Adam([
        {"params": model.trunk.parameters(), "lr": lr_backbone},
        {"params": model.head.parameters(), "lr": lr_head},
    ])
    return opt, torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)


def finetune(model, arm, args):
    """arm in {ce, ss, supcon}; returns the trained model."""
    if arm == "ce":
        loader = DataLoader(train_corpus(args.dataset, exp37.TF_AUG),
                            batch_size=args.batch_size * 2, shuffle=True,
                            num_workers=8, persistent_workers=True,
                            drop_last=True, pin_memory=True)
    else:
        loader = DataLoader(TwoViewLabeledImages(train_corpus(args.dataset)),
                            batch_size=args.batch_size, shuffle=True,
                            num_workers=8, persistent_workers=True,
                            drop_last=True, pin_memory=True)
    opt, sched = make_optim(model, args.ft_epochs, args.lr_backbone,
                            args.lr_head)
    scaler = torch.amp.GradScaler(enabled=DEVICE.type == "cuda")
    model.train()
    for ep in range(args.ft_epochs):
        run_a, run_b, n = 0.0, 0.0, 0
        for batch in loader:
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.float16,
                                enabled=DEVICE.type == "cuda"):
                if arm == "ce":
                    x, y = batch
                    x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE)
                    a = F.cross_entropy(model(x), y)
                    b = torch.zeros((), device=DEVICE)
                else:
                    v1, v2, y = batch
                    x = torch.cat([v1, v2]).to(DEVICE, non_blocking=True)
                    yy = torch.cat([y, y]).to(DEVICE)
                    z = model(x)
                    a = supcon_loss(F.normalize(z.float(), dim=1), yy,
                                    temp=0.1)
                    b = (sigreg_loss(z.float(), n_slices=64) if arm == "ss"
                         else torch.zeros((), device=DEVICE))
                loss = a + args.lam * b if arm == "ss" else a
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            bs = batch[0].size(0)
            run_a += a.item() * bs; run_b += float(b) * bs; n += bs
        sched.step()
        if (ep + 1) % 5 == 0 or ep == 0 or ep == args.ft_epochs - 1:
            print(f"  [{arm}-ft] epoch {ep+1}/{args.ft_epochs}  "
                  f"main={run_a/n:.4f}  sigreg={run_b/n:.4f}")
    return model


@torch.no_grad()
def extract_both(model, ds_name, split, batch_size=128):
    """(trunk 768-d, head emb) features on the plain transform."""
    if ds_name == "dtd":
        ds = datasets.DTD(DATA_DIR, split=split, download=True,
                          transform=exp37.TF_EVAL)
    else:
        ds = exp44.make_split(ds_name, split, exp37.TF_EVAL)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=8)
    model.eval()
    tr, hd, labels = [], [], []
    for x, y in loader:
        with torch.autocast("cuda", dtype=torch.float16,
                            enabled=DEVICE.type == "cuda"):
            f = model.trunk(x.to(DEVICE))
            h = model.head(f)
        tr.append(f.float().cpu()); hd.append(h.float().cpu())
        labels.append(y)
    return torch.cat(tr), torch.cat(hd), torch.cat(labels)


def run_base(base, args):
    print(f"\n################ base: {exp40.BASE_LABELS[base]} "
          f"(FINE-TUNED) ################")
    R = {}
    ncls = n_classes_of(args.dataset)
    for arm in args.arms:
        print(f"--- [{base}] {arm}-ft ---")
        torch.manual_seed(args.seed); np.random.seed(args.seed)
        model = FineTuneModel(base, args.emb_dim,
                              n_out=ncls if arm == "ce" else None)
        ckpt = os.path.join(CKPT_DIR, f"{args.dataset}_ft_{base}_{arm}"
                            f"{'_quick' if args.quick else ''}.pt")
        if args.dataset == "dtd" and not os.path.exists(ckpt):
            legacy = os.path.join(CKPT_DIR, f"dtd_ft_{base}_{arm}"
                                  f"{'_quick' if args.quick else ''}.pt")
            ckpt = legacy if os.path.exists(legacy) else ckpt
        if os.path.exists(ckpt) and not args.refresh:
            print(f"  loading {ckpt}")
            model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        else:
            finetune(model, arm, args)
            torch.save(model.state_dict(), ckpt)
        splits = (("train", "val", "test") if args.dataset == "dtd"
                  else ("train", "test"))
        feats = {s: extract_both(model, args.dataset, s) for s in splits}
        pool = [s for s in splits if s != "test"]
        Ttv = torch.cat([feats[s][0] for s in pool])
        Htv = torch.cat([feats[s][1] for s in pool])
        ytv = torch.cat([feats[s][2] for s in pool])
        Tte, Hte, yte = feats["test"]
        def probe(Z, Zt):
            import torch.nn.functional as F
            head = nn.Linear(Z.size(1), ncls).to(DEVICE)
            loader = DataLoader(
                torch.utils.data.TensorDataset(Z, ytv), batch_size=256,
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

        if arm == "ce":
            with torch.no_grad():
                acc = (Hte.argmax(1) == yte).float().mean().item()
            R["CE fine-tune (direct)"] = acc
        else:
            R[f"{'ss[lam5]' if arm == 'ss' else 'SupCon'}-ft (head probe)"] = \
                probe(Htv, Hte)
        R[f"{arm}-ft trunk probe (768d)"] = probe(Ttv, Tte)
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    for k, v in R.items():
        print(f"  [{base}] {k:<32} {100 * v:.1f}")
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset", default="dtd",
                    choices=["dtd", "aircraft", "cars", "flowers",
                             "galaxy10"])
    ap.add_argument("--bases", default="dino,lejepa,visreg")
    ap.add_argument("--arms", default="ce,ss,supcon")
    ap.add_argument("--ft-epochs", type=int, default=None)
    ap.add_argument("--probe-epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=32,
                    help="per-view batch; two-view arms forward 2x this")
    ap.add_argument("--lr-backbone", type=float, default=1e-5)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--emb-dim", type=int, default=100)
    ap.add_argument("--lam", type=float, default=5.0)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    args.ft_epochs = args.ft_epochs or (1 if args.quick else 20)
    args.probe_epochs = args.probe_epochs or (5 if args.quick else 50)
    args.arms = args.arms.split(",")
    bases = args.bases.split(",")
    print(f"device={DEVICE}  bases={bases}  arms={args.arms}  "
          f"emb_dim={args.emb_dim}  ft_epochs={args.ft_epochs}  "
          f"lr={args.lr_backbone:g}/{args.lr_head:g}")

    all_r = {b: run_base(b, args) for b in bases}

    refs = FROZEN_ROWS.get(args.dataset, {})
    print(f"\n===== {args.dataset.upper()} TOP-1, END-TO-END FINE-TUNED "
          f"(vs frozen refs) =====")
    methods = list(next(iter(all_r.values())))
    print(f"  {'method':<36}" + "".join(f"{b:>10}" for b in bases))
    for m in (next(iter(refs.values())) if refs else ()):
        print(f"  {m:<36}"
              + "".join(f"{refs[b][m]:>10.1f}" for b in bases))
    for m in methods:
        print(f"  {m:<36}"
              + "".join(f"{100 * all_r[b][m]:>10.1f}" for b in bases))

    x = np.arange(len(methods))
    w = 0.8 / len(bases)
    plt.figure(figsize=(10, 5))
    for i, b in enumerate(bases):
        plt.bar(x + i * w, [100 * all_r[b][m] for m in methods], w,
                label=exp40.BASE_LABELS[b])
    plt.xticks(x + 0.4 - w / 2, methods, rotation=15, ha="right", fontsize=8)
    plt.ylabel("DTD test top-1 (%)")
    plt.title(f"{args.dataset} end-to-end fine-tuning by base "
              f"(ViT-B/16 unfrozen)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    out = plot_path(f"{args.dataset}_finetune_accuracy.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\n  saved {out}")
    print("Done.")


if __name__ == "__main__":
    main()
