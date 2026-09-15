"""
DTD unsupervised-from-scratch: the fair-protocol version of exps 37-38.

Exps 37-38 trained supervised heads on frozen DINO features, so their wins
over the VISReg paper's numbers (arXiv 2606.02572) carried a supervision
asterisk.  This experiment removes it by following the paper's protocol
end-to-end at our scale: pretrain a from-scratch backbone with an
UNSUPERVISED objective (no labels touch the representation), freeze it, and
train only a linear probe on the DTD labels.

What cannot be matched is the pretraining corpus: VISReg pretrains ViT-B on
ImageNet-1K; here the unlabeled corpus is DTD train+val itself (3760
images), and the backbone is a from-scratch ResNet-18 (a ViT from scratch
collapses at this data scale -- that is precisely why the paper needs
ImageNet).  Absolute accuracy will therefore sit far below the paper's
transfer numbers; the fair, informative comparison is the RANKING of the
unsupervised objectives at identical arch / data / budget -- the same
methodological comparison the paper makes:

  sigreg-ssl : two-view invariance (MSE) + global SIGReg to N(0,I), lam=1
               -- this repo's member of the VISReg/LeJEPA family
  simclr     : NT-Xent, temp 0.5 -- the contrastive baseline
  hybrid     : NT-Xent + SIGReg on raw z, lam=5 -- exp 34e's feature half
  concat     : [simclr trunk ; sigreg-ssl trunk] probe (1024-d) -- exp 22
  supervised : from-scratch CE reference (the label ceiling at this scale)

All arms: identical ResNet-18 trunk (512-d probe features), projection head
to 128-d for the SSL losses, identical aug stack / epochs / optimizer (Adam
+ cosine, AMP).  Probes as in exps 37/38: Adam on frozen features,
train+val, DTD test top-1.  Trunks are checkpointed to checkpoints/.

    python experiments/39_dtd_scratch.py
    python experiments/39_dtd_scratch.py --quick
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
from torchvision import datasets, transforms
from torchvision.models import resnet18
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from supersig.config import DATA_DIR, DEVICE, REPO_DIR, plot_path
from supersig.losses import sigreg_loss, supcon_loss

exp37 = importlib.import_module("37_dtd_vit")

N_CLASSES = 47
PROJ_DIM = 128
CKPT_DIR = os.path.join(REPO_DIR, "checkpoints")

IMAGENET_NORM = transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))
TF_SSL = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.2, 1.0),
                                 interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(), IMAGENET_NORM,
])


class ScratchBackbone(nn.Module):
    """From-scratch ResNet-18 trunk (512-d) + projection head for SSL losses."""

    def __init__(self):
        super().__init__()
        net = resnet18(weights=None)
        net.fc = nn.Identity()
        self.trunk = net
        self.proj = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, PROJ_DIM),
        )

    def forward(self, x):
        return self.proj(self.trunk(x))


class TwoViewImages(Dataset):
    """Two independently augmented views of each unlabeled image."""

    def __init__(self, base, aug=TF_SSL):
        self.base, self.aug = base, aug

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, _ = self.base[idx]
        return self.aug(img), self.aug(img)


def ssl_corpus():
    """DTD train+val images, labels discarded (the unlabeled pretrain corpus)."""
    tr = datasets.DTD(DATA_DIR, split="train", download=True, transform=None)
    va = datasets.DTD(DATA_DIR, split="val", download=True, transform=None)
    return ConcatDataset([tr, va])


def make_optim(params, epochs, lr=1e-3):
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    return opt, sched


def train_ssl(model, loader, epochs, objective, lam=1.0, temp=0.5, tag="ssl"):
    """
    AMP two-view SSL loop mirroring the library trainers:
      objective="sigreg" : F.mse_loss(z1,z2) + lam * mean sigreg (train_sigreg_ssl)
      objective="simclr" : NT-Xent at `temp` (train_simclr)
      objective="hybrid" : NT-Xent on normalized z + lam * sigreg on raw z
                           (train_simclr_sigreg)
    """
    opt, sched = make_optim(model.parameters(), epochs)
    scaler = torch.amp.GradScaler(enabled=DEVICE.type == "cuda")
    model.train()
    for ep in range(epochs):
        run_a, run_b, n = 0.0, 0.0, 0
        for v1, v2 in loader:
            v1, v2 = v1.to(DEVICE, non_blocking=True), v2.to(DEVICE, non_blocking=True)
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.float16,
                                enabled=DEVICE.type == "cuda"):
                if objective == "sigreg":
                    z1, z2 = model(v1), model(v2)
                    a = F.mse_loss(z1, z2)
                    b = 0.5 * (sigreg_loss(z1.float()) + sigreg_loss(z2.float()))
                    loss = a + lam * b
                else:
                    z = model(torch.cat([v1, v2]))
                    inst = torch.arange(v1.size(0), device=DEVICE)
                    a = supcon_loss(F.normalize(z.float(), dim=1),
                                    torch.cat([inst, inst]), temp=temp)
                    b = (sigreg_loss(z.float()) if objective == "hybrid"
                         else torch.zeros((), device=DEVICE))
                    loss = a + (lam * b if objective == "hybrid" else 0.0)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            run_a += a.item() * v1.size(0)
            run_b += float(b) * v1.size(0)
            n += v1.size(0)
        sched.step()
        if (ep + 1) % 10 == 0 or ep == 0 or ep == epochs - 1:
            print(f"  [{tag}] epoch {ep+1}/{epochs}  main={run_a/n:.4f}  "
                  f"sigreg={run_b/n:.4f}")


def train_supervised_ref(model, epochs, batch_size):
    """From-scratch supervised CE on trunk+linear (same augs/optimizer)."""
    tr = datasets.DTD(DATA_DIR, split="train", download=True, transform=TF_SSL)
    va = datasets.DTD(DATA_DIR, split="val", download=True, transform=TF_SSL)
    loader = DataLoader(ConcatDataset([tr, va]), batch_size=batch_size,
                        shuffle=True, num_workers=8, persistent_workers=True,
                        drop_last=True, pin_memory=True)
    head = nn.Linear(512, N_CLASSES).to(DEVICE)
    opt, sched = make_optim(list(model.parameters()) + list(head.parameters()),
                            epochs)
    scaler = torch.amp.GradScaler(enabled=DEVICE.type == "cuda")
    model.train()
    for ep in range(epochs):
        run, correct, n = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.float16,
                                enabled=DEVICE.type == "cuda"):
                logits = head(model.trunk(x))
                loss = F.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            run += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += x.size(0)
        sched.step()
        if (ep + 1) % 10 == 0 or ep == 0 or ep == epochs - 1:
            print(f"  [supervised] epoch {ep+1}/{epochs}  loss={run/n:.4f}  "
                  f"acc={correct/n:.4f}")


@torch.no_grad()
def trunk_features(model, split, batch_size=128):
    """Frozen 512-d trunk features on the plain eval transform."""
    ds = datasets.DTD(DATA_DIR, split=split, download=True,
                      transform=exp37.TF_EVAL)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=8)
    model.eval()
    feats, labels = [], []
    for x, y in loader:
        with torch.autocast("cuda", dtype=torch.float16,
                            enabled=DEVICE.type == "cuda"):
            f = model.trunk(x.to(DEVICE))
        feats.append(f.float().cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ssl-epochs", type=int, default=None)
    ap.add_argument("--sup-epochs", type=int, default=None)
    ap.add_argument("--probe-epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lam-hybrid", type=float, default=5.0)
    ap.add_argument("--arms", default="sigreg,simclr,hybrid,supervised")
    ap.add_argument("--refresh", action="store_true",
                    help="retrain even if a trunk checkpoint exists")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    args.ssl_epochs = args.ssl_epochs or (2 if args.quick else 300)
    args.sup_epochs = args.sup_epochs or (2 if args.quick else 100)
    args.probe_epochs = args.probe_epochs or (5 if args.quick else 50)
    arms = args.arms.split(",")
    print(f"device={DEVICE}  backbone=ResNet-18 from scratch @224  "
          f"corpus=DTD train+val unlabeled ({args.ssl_epochs} ssl epochs)")

    two_view = DataLoader(TwoViewImages(ssl_corpus()),
                          batch_size=args.batch_size, shuffle=True,
                          num_workers=8, persistent_workers=True,
                          drop_last=True, pin_memory=True)

    specs = {
        "sigreg": dict(objective="sigreg", lam=1.0,
                       label="SIGReg-SSL (VISReg family)"),
        "simclr": dict(objective="simclr",
                       label="SimCLR (contrastive baseline)"),
        "hybrid": dict(objective="hybrid", lam=args.lam_hybrid,
                       label=f"SimCLR+SIGReg hybrid (lam={args.lam_hybrid:g})"),
    }

    feats = {}
    results = {}
    for arm in arms:
        if arm == "supervised":
            continue
        spec = specs[arm]
        ckpt = os.path.join(CKPT_DIR, f"dtd_scratch_{arm}"
                            f"{'_quick' if args.quick else ''}.pt")
        model = ScratchBackbone().to(DEVICE)
        if os.path.exists(ckpt) and not args.refresh:
            print(f"\n----- {arm}: loading {ckpt} -----")
            model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        else:
            print(f"\n----- {arm}: {spec['label']} -----")
            torch.manual_seed(args.seed); np.random.seed(args.seed)
            train_ssl(model, two_view, args.ssl_epochs,
                      objective=spec["objective"], lam=spec.get("lam", 1.0),
                      tag=arm)
            torch.save(model.state_dict(), ckpt)
        f = {s: trunk_features(model, s) for s in ("train", "val", "test")}
        feats[arm] = f
        Ztv = torch.cat([f["train"][0], f["val"][0]])
        ytv = torch.cat([f["train"][1], f["val"][1]])
        acc = exp37.probe_accuracy(Ztv, ytv, f["test"][0], f["test"][1],
                                   args.probe_epochs)
        results[spec["label"]] = acc
        print(f"  {arm} probe TEST={acc:.4f}")

    if "simclr" in feats and "sigreg" in feats:
        Ztv = torch.cat([torch.cat([feats[a]["train"][0], feats[a]["val"][0]])
                         for a in ("simclr", "sigreg")], dim=1)
        Zte = torch.cat([feats[a]["test"][0] for a in ("simclr", "sigreg")],
                        dim=1)
        ytv = torch.cat([feats["simclr"]["train"][1],
                         feats["simclr"]["val"][1]])
        acc = exp37.probe_accuracy(Ztv, ytv, Zte, feats["simclr"]["test"][1],
                                   args.probe_epochs)
        results["Concat [simclr ; sigreg-ssl] (exp 22)"] = acc
        print(f"  concat probe TEST={acc:.4f}")

    if "supervised" in arms:
        ckpt = os.path.join(CKPT_DIR, f"dtd_scratch_supervised"
                            f"{'_quick' if args.quick else ''}.pt")
        model = ScratchBackbone().to(DEVICE)
        if os.path.exists(ckpt) and not args.refresh:
            print(f"\n----- supervised: loading {ckpt} -----")
            model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        else:
            print("\n----- supervised CE reference (from scratch) -----")
            torch.manual_seed(args.seed); np.random.seed(args.seed)
            train_supervised_ref(model, args.sup_epochs, args.batch_size)
            torch.save(model.state_dict(), ckpt)
        f = {s: trunk_features(model, s) for s in ("train", "val", "test")}
        Ztv = torch.cat([f["train"][0], f["val"][0]])
        ytv = torch.cat([f["train"][1], f["val"][1]])
        acc = exp37.probe_accuracy(Ztv, ytv, f["test"][0], f["test"][1],
                                   args.probe_epochs)
        results["Supervised CE from scratch (ceiling)"] = acc
        print(f"  supervised probe TEST={acc:.4f}")

    print("\n===== DTD TOP-1: unsupervised from scratch on DTD "
          "(ResNet-18, no labels in the representation) =====")
    print("  reference (ImageNet-pretrained / paper):")
    for name, acc in exp37.PAPER_ROWS:
        print(f"    {name:<44} {acc:.1f}")
    print(f"    {'Linear probe, raw DINO (train+val) [37]':<44} 76.4")
    print("  from scratch on DTD only:")
    for name, acc in results.items():
        print(f"    {name:<44} {100 * acc:.1f}")

    names = list(results)
    accs = [100 * a for a in results.values()]
    plt.figure(figsize=(9, 4.5))
    ypos = np.arange(len(names))
    plt.barh(ypos, accs, color="#2a9d5c")
    plt.yticks(ypos, names, fontsize=8)
    plt.axvline(74.3, color="gray", ls=":", lw=1.2,
                label="DINO ViT-B/16 transfer (paper, 74.3)")
    plt.xlabel("DTD test top-1 (%)")
    plt.title("DTD from-scratch unsupervised pretraining (fair protocol, "
              "DTD-only corpus)")
    plt.gca().invert_yaxis()
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(plot_path("dtd_scratch_accuracy.png"), dpi=150)
    plt.close()
    print(f"\n  saved {plot_path('dtd_scratch_accuracy.png')}")
    print("Done.")


if __name__ == "__main__":
    main()
