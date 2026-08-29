"""Extract DINO ViT-B/16 [CLS] features for CIFAR-10/100 at 224 px (the GCD
protocol's backbone), in the same bank format as data/tf_feats_{ds}_dino_vitb16.pt:
{"train": (X, y), "test": (X, y)}.  Evaluation-only; ~15 min per dataset."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse, importlib, torch
from torchvision import datasets
from supersig.config import DATA_DIR
exp37 = importlib.import_module("37_dtd_vit")

ap = argparse.ArgumentParser(); ap.add_argument("--datasets", default="cifar10,cifar100"); args = ap.parse_args()
model = exp37.load_dino().eval()
for ds in args.datasets.split(","):
    out = os.path.join(DATA_DIR, f"tf_feats_{ds}_dino_vitb16.pt")
    if os.path.exists(out):
        print("exists", out); continue
    cls = datasets.CIFAR10 if ds == "cifar10" else datasets.CIFAR100
    bank = {}
    for split, train in (("train", True), ("test", False)):
        d = cls(DATA_DIR, train=train, download=False, transform=exp37.TF_EVAL)
        X, y = exp37.extract(model, d)
        bank[split] = (X, y); print(ds, split, tuple(X.shape), flush=True)
    torch.save(bank, out); print("saved", out, flush=True)
