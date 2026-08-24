"""
Frozen-base transfer suite at 32-d: Aircraft / Cars / Flowers102 / Galaxy10.

Extends the DTD program (exps 37-42) to the other transfer datasets of the
VISReg paper (arXiv 2606.02572, Tables 5-6), same machinery: frozen
ViT-B/16 bases (DINO / LeJEPA community repro / VISReg released weights),
cached CLS features, feature-space heads, Adam probes -> test top-1.

Paper reference rows (linear probe):
  Aircraft:  MoCoV3 57.9  DINO 63.6  VISReg-B 57.1
  Cars:      MoCoV3 67.5  DINO 73.9  VISReg-B 64.8
  Flowers:   MoCoV3 91.5  DINO 94.6  VISReg-B 90.4
  Galaxy10:  MoCoV3 73.1  DINO 72.8  VISReg-B 74.0   (10%/90% split)
On the fine-grained sets the paper itself has VISReg LOSING to DINO --
the DTD/Galaxy10 wins are its texture/OOD story.

Arms (32-d halves -- deliberately BELOW the class count of aircraft/cars/
flowers, the crowded regime; the SIGReg-anchor arms are omitted since
orthogonal anchors don't exist there): raw-feature probe, SupCon (aug),
ss[lam5] (SupCon+SIGReg), and the concats supcon+hybrid / ss+hybrid.

Splits: aircraft trainval/test; flowers train+val/test; cars train/test
(no val exists); galaxy10 stratified 10%/90% as in the paper.  Aug banks:
8 replicas (the larger corpora supply natural diversity).

    python experiments/44_transfer_32d.py
    python experiments/44_transfer_32d.py --datasets galaxy10 --bases visreg
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import importlib
import io
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import datasets
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from supersig.config import DATA_DIR, DEVICE, plot_path
from supersig.train import train_simclr_sigreg, train_supcon_sigreg

exp37 = importlib.import_module("37_dtd_vit")
exp38 = importlib.import_module("38_dtd_arc")
exp40 = importlib.import_module("40_dtd_bases")

PAPER = {  # dataset -> {method: acc}
    "aircraft": {"MoCoV3": 57.9, "DINO": 63.6, "VISReg-B": 57.1},
    "cars": {"MoCoV3": 67.5, "DINO": 73.9, "VISReg-B": 64.8},
    "flowers": {"MoCoV3": 91.5, "DINO": 94.6, "VISReg-B": 90.4},
    "galaxy10": {"MoCoV3": 73.1, "DINO": 72.8, "VISReg-B": 74.0},
}
N_CLASSES = {"aircraft": 100, "cars": 196, "flowers": 102, "galaxy10": 10,
             "food101": 101}


class Galaxy10(Dataset):
    """
    Galaxy10 DECals via HF parquet (matthieulel/galaxy10_decals; the astroNN
    h5 URL is dead).  ALL shards are pooled, then a stratified 10%/90%
    train/test split is drawn (the paper's protocol), seed 0.
    """

    def __init__(self, split, transform, seed=0):
        import pandas as pd
        from PIL import Image
        from huggingface_hub import hf_hub_download, list_repo_files
        files = sorted(f for f in list_repo_files(
            "matthieulel/galaxy10_decals", repo_type="dataset")
            if f.endswith(".parquet"))
        self.df = pd.concat(
            [pd.read_parquet(hf_hub_download("matthieulel/galaxy10_decals",
                                             f, repo_type="dataset"))
             for f in files], ignore_index=True)
        img_col = [c for c in self.df.columns if "image" in c.lower()][0]
        lab_col = [c for c in self.df.columns if "label" in c.lower()][0]
        self.img_col, self.lab_col = img_col, lab_col
        labels = self.df[lab_col].to_numpy()
        rng = np.random.default_rng(seed)
        train_idx = []
        for c in np.unique(labels):
            idx = np.where(labels == c)[0]
            rng.shuffle(idx)
            train_idx += idx[: max(1, int(round(0.1 * len(idx))))].tolist()
        train_idx = set(train_idx)
        self.keep = [i for i in range(len(labels))
                     if (i in train_idx) == (split == "train")]
        self.transform, self._Image = transform, Image

    def __len__(self):
        return len(self.keep)

    def __getitem__(self, i):
        row = self.df.iloc[self.keep[i]]
        rec = row[self.img_col]
        raw = rec["bytes"] if isinstance(rec, dict) else rec
        img = self._Image.open(io.BytesIO(raw)).convert("RGB")
        return self.transform(img) if self.transform else img, \
            int(row[self.lab_col])


class HFCars(Dataset):
    """Stanford Cars via HF parquet (torchvision's download URL is dead)."""

    def __init__(self, split, transform):
        import pandas as pd
        from PIL import Image
        from huggingface_hub import hf_hub_download, list_repo_files
        files = [f for f in list_repo_files("tanganke/stanford_cars",
                                            repo_type="dataset")
                 if f.endswith(".parquet") and split in f]
        assert files, f"no parquet for split {split}"
        dfs = [pd.read_parquet(hf_hub_download("tanganke/stanford_cars", f,
                                               repo_type="dataset"))
               for f in sorted(files)]
        self.df = pd.concat(dfs, ignore_index=True)
        img_col = [c for c in self.df.columns if "image" in c.lower()][0]
        lab_col = [c for c in self.df.columns if "label" in c.lower()][0]
        self.img_col, self.lab_col = img_col, lab_col
        self.transform, self._Image = transform, Image

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        rec = self.df.iloc[i][self.img_col]
        raw = rec["bytes"] if isinstance(rec, dict) else rec
        img = self._Image.open(io.BytesIO(raw)).convert("RGB")
        y = int(self.df.iloc[i][self.lab_col])
        return self.transform(img) if self.transform else img, y


def make_split(name, split, transform):
    """split in {'train','test'}; 'train' is the full training pool."""
    if name == "aircraft":
        if split == "train":
            return datasets.FGVCAircraft(DATA_DIR, split="trainval",
                                         download=True, transform=transform)
        return datasets.FGVCAircraft(DATA_DIR, split="test", download=True,
                                     transform=transform)
    if name == "flowers":
        if split == "train":
            return ConcatDataset([
                datasets.Flowers102(DATA_DIR, split=s, download=True,
                                    transform=transform)
                for s in ("train", "val")])
        return datasets.Flowers102(DATA_DIR, split="test", download=True,
                                   transform=transform)
    if name == "cars":
        try:
            return datasets.StanfordCars(DATA_DIR, split=split,
                                         download=True, transform=transform)
        except Exception:
            return HFCars(split, transform)
    if name == "galaxy10":
        return Galaxy10(split, transform)
    if name == "food101":
        return datasets.Food101(DATA_DIR, split=split, download=True,
                                transform=transform)
    raise ValueError(name)


def build_features(ds_name, base, args):
    tag = f"{ds_name}_{exp40.CACHE_TAG[base]}"
    plain_cache = os.path.join(DATA_DIR, f"tf_feats_{tag}.pt")
    aug_cache = os.path.join(DATA_DIR, f"tf_augfeats_{tag}_a{args.aug_reps}.pt")
    model = None
    if os.path.exists(plain_cache) and not args.refresh:
        plain = torch.load(plain_cache)
    else:
        model = exp40.LOADERS[base]()
        plain = {}
        for split in ("train", "test"):
            d = make_split(ds_name, split, exp37.TF_EVAL)
            plain[split] = exp37.extract(model, d)
            print(f"  extracted {ds_name}/{base} {split}: "
                  f"{tuple(plain[split][0].shape)}")
        torch.save(plain, plain_cache)
    if os.path.exists(aug_cache) and not args.refresh:
        bank = torch.load(aug_cache)
    else:
        model = model or exp40.LOADERS[base]()
        d = make_split(ds_name, "train", exp37.TF_AUG)
        reps, labels = [], None
        for a in range(args.aug_reps):
            f, labels = exp37.extract(model, d)
            reps.append(f.half())
            print(f"  {ds_name}/{base} aug replica {a + 1}/{args.aug_reps}")
        bank = {"feats": torch.stack(reps), "labels": labels}
        torch.save(bank, aug_cache)
    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return plain, bank


def run_cell(ds_name, base, args):
    print(f"\n######## {ds_name} on {exp40.BASE_LABELS[base]} ########")
    plain, bank = build_features(ds_name, base, args)
    (Xtv, ytv), (Xte, yte) = plain["train"], plain["test"]
    two_view_lab = DataLoader(exp37.TwoViewFeatures(bank),
                              batch_size=args.batch_size, shuffle=True,
                              drop_last=True)
    two_view_unlab = DataLoader(exp38.TwoViewUnlabeled(bank),
                                batch_size=args.batch_size, shuffle=True,
                                drop_last=True)
    R = {}

    def probe(Z, Zt, ncls):
        import torch.nn as nn
        import torch.nn.functional as F
        head = nn.Linear(Z.size(1), ncls).to(DEVICE)
        loader = DataLoader(torch.utils.data.TensorDataset(Z, ytv),
                            batch_size=256, shuffle=True)
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

    ncls = N_CLASSES[ds_name]
    R["raw probe (768d)"] = probe(Xtv, Xte, ncls)

    torch.manual_seed(args.seed + 20); np.random.seed(args.seed + 20)
    supcon = exp37.train_supcon_aug(bank, args)
    Zsc, Zsc_t = exp37.embed(supcon, Xtv), exp37.embed(supcon, Xte)
    R["SupCon (aug)"] = probe(Zsc, Zsc_t, ncls)

    torch.manual_seed(args.seed + 30); np.random.seed(args.seed + 30)
    ss = exp37.FeatureHead(args.emb_dim).to(DEVICE)
    train_supcon_sigreg(ss, two_view_lab, args.embed_epochs, temp=0.1,
                        lam=args.lam, n_slices=64)
    Zss, Zss_t = exp37.embed(ss, Xtv), exp37.embed(ss, Xte)
    R["ss[lam5]"] = probe(Zss, Zss_t, ncls)

    torch.manual_seed(args.seed + 21); np.random.seed(args.seed + 21)
    hyb = exp37.FeatureHead(args.emb_dim).to(DEVICE)
    train_simclr_sigreg(hyb, two_view_unlab, args.embed_epochs, lam=args.lam,
                        n_slices=64)
    Zh, Zh_t = exp37.embed(hyb, Xtv), exp37.embed(hyb, Xte)
    R["supcon+hybrid (concat)"] = probe(torch.cat([Zsc, Zh], 1),
                                        torch.cat([Zsc_t, Zh_t], 1), ncls)
    R["ss+hybrid (concat)"] = probe(torch.cat([Zss, Zh], 1),
                                    torch.cat([Zss_t, Zh_t], 1), ncls)
    for k, v in R.items():
        print(f"  [{ds_name}/{base}] {k:<26} {100 * v:.1f}")
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bases", default="dino,lejepa,visreg")
    ap.add_argument("--datasets", default="aircraft,cars,flowers,galaxy10")
    ap.add_argument("--embed-epochs", type=int, default=None)
    ap.add_argument("--probe-epochs", type=int, default=None)
    ap.add_argument("--aug-reps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--pair-dist", type=float, default=5.0)
    ap.add_argument("--emb-dim", type=int, default=32)
    ap.add_argument("--lam", type=float, default=5.0)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    args.embed_epochs = args.embed_epochs or (5 if args.quick else 120)
    args.probe_epochs = args.probe_epochs or (5 if args.quick else 50)
    args.aug_reps = args.aug_reps or (2 if args.quick else 8)
    bases = args.bases.split(",")
    ds_names = args.datasets.split(",")
    print(f"device={DEVICE}  emb_dim={args.emb_dim}  datasets={ds_names}  "
          f"bases={bases}")

    all_r = {d: {b: run_cell(d, b, args) for b in bases} for d in ds_names}

    print(f"\n===== TRANSFER TOP-1, {args.emb_dim}-d heads on frozen "
          f"ViT-B/16 =====")
    for d in ds_names:
        print(f"\n  --- {d} (paper: "
              + "  ".join(f"{k} {v}" for k, v in PAPER[d].items()) + ") ---")
        methods = list(next(iter(all_r[d].values())))
        print(f"  {'method':<28}" + "".join(f"{b:>10}" for b in bases))
        for m in methods:
            print(f"  {m:<28}"
                  + "".join(f"{100 * all_r[d][b][m]:>10.1f}" for b in bases))

    fig, axes = plt.subplots(1, len(ds_names), figsize=(4.2 * len(ds_names), 4.5),
                             squeeze=False)
    for ax, d in zip(axes[0], ds_names):
        methods = list(next(iter(all_r[d].values())))
        x = np.arange(len(methods))
        w = 0.8 / len(bases)
        for i, b in enumerate(bases):
            ax.bar(x + i * w, [100 * all_r[d][b][m] for m in methods], w,
                   label=exp40.BASE_LABELS[b])
        ax.axhline(PAPER[d]["DINO"], ls=":", lw=1, color="gray")
        ax.set_xticks(x + 0.4 - w / 2)
        ax.set_xticklabels(methods, rotation=35, ha="right", fontsize=6)
        ax.set_title(f"{d} (dotted = paper DINO)", fontsize=9)
    axes[0][0].set_ylabel("test top-1 (%)")
    axes[0][0].legend(fontsize=6)
    plt.tight_layout()
    out = plot_path(f"transfer32d_accuracy_d{args.emb_dim}.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\n  saved {out}")
    print("Done.")


if __name__ == "__main__":
    main()
