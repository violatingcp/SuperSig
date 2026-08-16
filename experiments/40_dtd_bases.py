"""
DTD across pretrained bases: DINO vs LeJEPA vs VISReg's own weights.

Exps 37/38 used DINO ViT-B/16 as the frozen base because the VISReg paper's
weights were assumed unreleased.  Both assumptions fell:

  - VISReg (arXiv 2606.02572) released its ImageNet-1K ViT-B/16 backbone
    (HF BooBooWu/visreg, visreg-vit-b-inet1k.pth, loads into timm
    vit_base_patch16_224) -- the paper's own DTD linear-probe row is 75.7,
    so our raw-feature probe on it doubles as a protocol validation.
  - LeJEPA (arXiv 2511.08544, the SIGReg source paper) has an ImageNet-1K
    ViT-B/16 community reproduction (HF OK-AI/lejepa-vitb16-pretrain-in1k,
    DINOv2-design ViT, no registers; weights remap 1:1 into timm).  NOT an
    official Balestriero/LeCun release; its card reports a 72.0 IN-1k
    online probe, so expect it a notch below the official paper's numbers.

Per base, same protocol as exps 37/38 at 100-dim heads (the current best
operating point): raw-feature linear probes (paper protocol + train+val),
SIGReg repulse+proto (+ own posterior), SIGReg repulse+CE, SupCon (aug),
and the two leader concats supcon+hybrid[lam5] / supcon+simclr (200-dim).
Features and 24-replica aug banks are cached per base in data/.

    python experiments/40_dtd_bases.py
    python experiments/40_dtd_bases.py --bases visreg,lejepa --quick
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import importlib
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from supersig.config import DATA_DIR, DEVICE, plot_path
from supersig.train import train_simclr, train_simclr_sigreg

exp37 = importlib.import_module("37_dtd_vit")
exp38 = importlib.import_module("38_dtd_arc")

PAPER_DTD = {"dino": 74.3, "visreg": 75.7, "lejepa": None}
BASE_LABELS = {
    "dino": "DINO ViT-B/16",
    "visreg": "VISReg ViT-B/16 (paper's weights)",
    "lejepa": "LeJEPA ViT-B/16 (community repro)",
}


def load_visreg():
    import timm
    from huggingface_hub import hf_hub_download
    # pin the 2026-04-08 "publish model" revision: the 2026-08-14 upstream
    # update rewrote this file with projection layers and (unverified)
    # possibly different trunk weights -- all campaign results used this one
    p = hf_hub_download(repo_id="BooBooWu/visreg",
                        filename="visreg-vit-b-inet1k.pth",
                        revision="0d5f4fd3e282ce2c9615805cc3e660e462e6bcb1",
                        local_dir=os.path.join(DATA_DIR, "..", "checkpoints",
                                               "visreg"))
    m = timm.create_model("vit_base_patch16_224", pretrained=False,
                          num_classes=0)
    m.load_state_dict(torch.load(p, map_location="cpu", weights_only=True))
    return m.eval().to(DEVICE)


def load_lejepa():
    import timm
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    d = snapshot_download("OK-AI/lejepa-vitb16-pretrain-in1k")
    sd = load_file(os.path.join(d, "model.safetensors"))
    new = {k[len("backbone."):]: v for k, v in sd.items()
           if k.startswith("backbone.") and "cva_module" not in k}
    m = timm.create_model("vit_base_patch16_224", pretrained=False,
                          num_classes=0)
    missing, unexpected = m.load_state_dict(new, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    return m.eval().to(DEVICE)


LOADERS = {"dino": exp37.load_dino, "visreg": load_visreg,
           "lejepa": load_lejepa}
# exp 37 cached the DINO banks under these historical names.
CACHE_TAG = {"dino": "dino_vitb16", "visreg": "visreg_vitb16",
             "lejepa": "lejepa_vitb16"}


def build_features(base, args):
    tag = CACHE_TAG[base]
    plain_cache = os.path.join(DATA_DIR, f"dtd_feats_{tag}.pt")
    aug_cache = os.path.join(DATA_DIR, f"dtd_augfeats_{tag}_a{args.aug_reps}.pt")
    model = None
    if os.path.exists(plain_cache) and not args.refresh:
        print(f"  cached plain features: {plain_cache}")
        plain = torch.load(plain_cache)
    else:
        model = LOADERS[base]()
        plain = {}
        for split in ("train", "val", "test"):
            ds = datasets.DTD(DATA_DIR, split=split, download=True,
                              transform=exp37.TF_EVAL)
            plain[split] = exp37.extract(model, ds)
            print(f"  extracted {base} {split}: {tuple(plain[split][0].shape)}")
        torch.save(plain, plain_cache)
    if os.path.exists(aug_cache) and not args.refresh:
        print(f"  cached aug bank: {aug_cache}")
        bank = torch.load(aug_cache)
    else:
        model = model or LOADERS[base]()
        tr = datasets.DTD(DATA_DIR, split="train", download=True,
                          transform=exp37.TF_AUG)
        va = datasets.DTD(DATA_DIR, split="val", download=True,
                          transform=exp37.TF_AUG)
        ds = torch.utils.data.ConcatDataset([tr, va])
        reps, labels = [], None
        for a in range(args.aug_reps):
            f, labels = exp37.extract(model, ds)
            reps.append(f.half())
            print(f"  {base} aug replica {a + 1}/{args.aug_reps}")
        bank = {"feats": torch.stack(reps), "labels": labels}
        torch.save(bank, aug_cache)
    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return plain, bank


def run_base(base, args):
    print(f"\n################ base: {BASE_LABELS[base]} ################")
    plain, bank = build_features(base, args)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = (plain["train"], plain["val"],
                                          plain["test"])
    Xtv, ytv = torch.cat([Xtr, Xva]), torch.cat([ytr, yva])
    R = {}

    def probe(Z, Zt):
        return exp37.probe_accuracy(Z, ytv, Zt, yte, args.probe_epochs)

    print(f"--- [{base}] raw probe, paper protocol (train only, swept) ---")
    test, val, lr = exp37.swept_probe(Xtr, ytr, Xva, yva, Xte, yte,
                                      args.probe_epochs)
    print(f"  best lr={lr:g}  val={val:.4f}  TEST={test:.4f}")
    R["raw probe (train, swept)"] = test
    R["raw probe (train+val)"] = probe(Xtv, Xte)
    print(f"  train+val TEST={R['raw probe (train+val)']:.4f}")

    print(f"--- [{base}] SIGReg repulse + proto ---")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    head, means = exp37.train_sigreg(Xtv, ytv, "proto", args)
    R["SIGReg+proto"] = probe(exp37.embed(head, Xtv), exp37.embed(head, Xte))
    R["SIGReg+proto (posterior)"] = exp37.proto_accuracy(head, means, Xte, yte)

    print(f"--- [{base}] SIGReg repulse + CE ---")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    head, _ = exp37.train_sigreg(Xtv, ytv, "ce", args)
    R["SIGReg+CE"] = probe(exp37.embed(head, Xtv), exp37.embed(head, Xte))

    print(f"--- [{base}] SupCon (aug) ---")
    torch.manual_seed(args.seed + 20); np.random.seed(args.seed + 20)
    supcon = exp37.train_supcon_aug(bank, args)
    Zsc, Zsc_t = exp37.embed(supcon, Xtv), exp37.embed(supcon, Xte)
    R["SupCon (aug)"] = probe(Zsc, Zsc_t)

    two_view_unlab = DataLoader(exp38.TwoViewUnlabeled(bank),
                                batch_size=args.batch_size, shuffle=True,
                                drop_last=True)
    print(f"--- [{base}] hybrid half (NT-Xent + SIGReg lam=5) ---")
    torch.manual_seed(args.seed + 21); np.random.seed(args.seed + 21)
    hyb = exp37.FeatureHead(args.emb_dim).to(DEVICE)
    train_simclr_sigreg(hyb, two_view_unlab, args.embed_epochs, lam=5.0,
                        n_slices=64)
    R["supcon+hybrid (concat)"] = probe(
        torch.cat([Zsc, exp37.embed(hyb, Xtv)], dim=1),
        torch.cat([Zsc_t, exp37.embed(hyb, Xte)], dim=1))

    print(f"--- [{base}] simclr half (plain NT-Xent) ---")
    torch.manual_seed(args.seed + 22); np.random.seed(args.seed + 22)
    sim = exp37.FeatureHead(args.emb_dim).to(DEVICE)
    train_simclr(sim, two_view_unlab, args.embed_epochs)
    R["supcon+simclr (concat)"] = probe(
        torch.cat([Zsc, exp37.embed(sim, Xtv)], dim=1),
        torch.cat([Zsc_t, exp37.embed(sim, Xte)], dim=1))

    for k, v in R.items():
        print(f"  [{base}] {k:<28} {100 * v:.1f}")
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bases", default="dino,lejepa,visreg")
    ap.add_argument("--embed-epochs", type=int, default=None)
    ap.add_argument("--probe-epochs", type=int, default=None)
    ap.add_argument("--aug-reps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--pair-dist", type=float, default=5.0)
    ap.add_argument("--emb-dim", type=int, default=100)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    args.embed_epochs = args.embed_epochs or (5 if args.quick else 120)
    args.probe_epochs = args.probe_epochs or (5 if args.quick else 50)
    args.aug_reps = args.aug_reps or (2 if args.quick else 24)
    bases = args.bases.split(",")
    print(f"device={DEVICE}  bases={bases}  emb_dim={args.emb_dim}")

    all_r = {b: run_base(b, args) for b in bases}

    print("\n===== DTD TOP-1 BY BASE (100-dim heads; paper row = the "
          "papers' own linear-probe numbers) =====")
    methods = list(next(iter(all_r.values())))
    hdr = f"  {'method':<30}" + "".join(f"{b:>10}" for b in bases)
    print(hdr)
    paper = [f"{PAPER_DTD[b]:>10.1f}" if PAPER_DTD[b] else f"{'--':>10}"
             for b in bases]
    print(f"  {'(paper linear probe)':<30}" + "".join(paper))
    for m in methods:
        print(f"  {m:<30}"
              + "".join(f"{100 * all_r[b][m]:>10.1f}" for b in bases))

    x = np.arange(len(methods))
    w = 0.8 / len(bases)
    plt.figure(figsize=(11, 5))
    for i, b in enumerate(bases):
        plt.bar(x + i * w, [100 * all_r[b][m] for m in methods], w,
                label=BASE_LABELS[b])
        if PAPER_DTD[b]:
            plt.axhline(PAPER_DTD[b], ls=":", lw=1,
                        color=plt.gca().patches[-1].get_facecolor())
    plt.xticks(x + 0.4 - w / 2, methods, rotation=20, ha="right", fontsize=8)
    plt.ylabel("DTD test top-1 (%)")
    plt.ylim(60, 82)
    plt.title("DTD by pretrained base (frozen ViT-B/16, 100-dim heads); "
              "dotted = paper's own linear-probe number")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(plot_path("dtd_bases_accuracy.png"), dpi=150)
    plt.close()
    print(f"\n  saved {plot_path('dtd_bases_accuracy.png')}")
    print("Done.")


if __name__ == "__main__":
    main()
