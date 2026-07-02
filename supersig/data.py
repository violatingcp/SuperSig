"""MNIST data loaders: plain, two-view (augmented), and hold-out variants."""
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .config import DATA_DIR, HOLDOUT

NORM = transforms.Normalize((0.1307,), (0.3081,))
TF_PLAIN = transforms.Compose([transforms.ToTensor(), NORM])
TF_AUG = transforms.Compose([
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(), NORM,
])


# --------------------------------------------------------------------------- #
# Plain loaders                                                               #
# --------------------------------------------------------------------------- #
def get_loaders(batch_size=128, quick=False):
    """Standard train/test loaders (all classes)."""
    train = datasets.MNIST(DATA_DIR, train=True, download=True, transform=TF_PLAIN)
    test = datasets.MNIST(DATA_DIR, train=False, download=True, transform=TF_PLAIN)
    if quick:
        train, test = Subset(train, range(4000)), Subset(test, range(2000))
    return (DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2),
            DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=2))


def build_holdout_loaders(batch_size=256, quick=False, holdout=HOLDOUT):
    """
    Return (emb_loader, probe_loader, test_loader):
        emb_loader   -- training images with `holdout` removed (for embedding training)
        probe_loader -- full training images (for the frozen linear probe)
        test_loader  -- full test images
    """
    train_full = datasets.MNIST(DATA_DIR, train=True, download=True, transform=TF_PLAIN)
    test = datasets.MNIST(DATA_DIR, train=False, download=True, transform=TF_PLAIN)

    targets = train_full.targets
    n_train = 8000 if quick else len(train_full)
    base_idx = list(range(n_train))
    emb_idx = [i for i in base_idx if int(targets[i]) != holdout]
    if quick:
        test = Subset(test, range(3000))

    emb_ds, probe_ds = Subset(train_full, emb_idx), Subset(train_full, base_idx)
    print(f"  embedding-train images (no {holdout}): {len(emb_ds)}   "
          f"probe-train images (all): {len(probe_ds)}")
    return (DataLoader(emb_ds, batch_size=batch_size, shuffle=True, num_workers=2),
            DataLoader(probe_ds, batch_size=batch_size, shuffle=True, num_workers=2),
            DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=2))


# --------------------------------------------------------------------------- #
# Two-view (augmented) datasets for self-supervised / contrastive training    #
# --------------------------------------------------------------------------- #
class TwoViewMNIST(torch.utils.data.Dataset):
    """Two independently augmented views of each image (no label)."""

    def __init__(self, base, aug=TF_AUG):
        self.base, self.aug = base, aug

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, _ = self.base[idx]
        return self.aug(img), self.aug(img)


class TwoViewLabeledMNIST(torch.utils.data.Dataset):
    """Two augmented views of each image plus its label (for SupCon)."""

    def __init__(self, base, aug=TF_AUG):
        self.base, self.aug = base, aug

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, y = self.base[idx]
        return self.aug(img), self.aug(img), y


def two_view_loader(batch_size=256, quick=False, labeled=False, holdout=None):
    """Augmented two-view loader; optionally drops `holdout` and/or returns labels."""
    raw = datasets.MNIST(DATA_DIR, train=True, download=True, transform=None)
    n = 8000 if quick else len(raw)
    idx = [i for i in range(n) if (holdout is None or int(raw.targets[i]) != holdout)]
    base = Subset(raw, idx)
    ds = TwoViewLabeledMNIST(base) if labeled else TwoViewMNIST(base)
    tag = "" if holdout is None else f" (no {holdout})"
    print(f"  two-view embedding-train images{tag}: {len(ds)}")
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2,
                      drop_last=labeled)
