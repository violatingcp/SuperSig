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


# =========================================================================== #
# CIFAR-10 / CIFAR-100 loaders (plain, two-view augmented, hold-out)          #
# =========================================================================== #
CIFAR_NORM = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
CIFAR100_NORM = transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
CIFAR_TF_PLAIN = transforms.Compose([transforms.ToTensor(), CIFAR_NORM])
CIFAR_TF_AUG = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.ToTensor(), CIFAR_NORM,
])


def _holdout_set(holdout):
    """Normalize a holdout spec (None, int, or iterable of ints) to a set."""
    if holdout is None:
        return set()
    if isinstance(holdout, int):
        return {holdout}
    return set(holdout)


def _cifar_spec(dataset):
    """(dataset class, plain transform, two-view aug transform) for a CIFAR variant."""
    if dataset == "cifar100":
        cls, norm = datasets.CIFAR100, CIFAR100_NORM
    else:
        cls, norm = datasets.CIFAR10, CIFAR_NORM
    plain = transforms.Compose([transforms.ToTensor(), norm])
    aug = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.ToTensor(), norm,
    ])
    return cls, plain, aug


class BalancedBatchSampler(torch.utils.data.Sampler):
    """
    Batches of `n_classes` randomly chosen classes x `n_per_class` samples each.

    Guarantees every class present in a batch has enough samples for the
    per-class SIGReg statistic (cf. losses.MIN_PER_CLASS), which random batches
    cannot provide when the number of classes is large (e.g. CIFAR-100).
    """

    def __init__(self, targets, n_classes=25, n_per_class=24):
        targets = torch.as_tensor(targets)
        self.classes = torch.unique(targets)
        self.idx_by_class = {int(c): torch.nonzero(targets == c).flatten()
                             for c in self.classes}
        self.n_classes = min(n_classes, len(self.classes))
        self.n_per_class = n_per_class
        self.n_batches = max(1, len(targets) // (self.n_classes * n_per_class))

    def __iter__(self):
        for _ in range(self.n_batches):
            cs = self.classes[torch.randperm(len(self.classes))[:self.n_classes]]
            batch = []
            for c in cs:
                idx = self.idx_by_class[int(c)]
                batch += idx[torch.randint(len(idx), (self.n_per_class,))].tolist()
            yield batch

    def __len__(self):
        return self.n_batches


def cifar_balanced_loader(dataset="cifar10", holdout=None, quick=False, limit=None,
                          classes_per_batch=25, per_class=24):
    """Plain-transform loader with class-balanced batches (optionally minus `holdout`)."""
    cls, plain, _ = _cifar_spec(dataset)
    ds = cls(DATA_DIR, train=True, download=True, transform=plain)
    targets = list(ds.targets)
    n = 8000 if quick else (limit or len(ds))
    hs = _holdout_set(holdout)
    idx = [i for i in range(n) if targets[i] not in hs]
    sub = Subset(ds, idx)
    sampler = BalancedBatchSampler([targets[i] for i in idx], classes_per_batch, per_class)
    tag = "" if holdout is None else f" (no {holdout})"
    print(f"  {dataset} balanced loader{tag}: {len(sub)} images, "
          f"{len(sampler)} batches of {sampler.n_classes}x{per_class}")
    return DataLoader(sub, batch_sampler=sampler, num_workers=2)


def get_cifar_loaders(batch_size=256, quick=False, limit=None, dataset="cifar10"):
    cls, plain, _ = _cifar_spec(dataset)
    train = cls(DATA_DIR, train=True, download=True, transform=plain)
    test = cls(DATA_DIR, train=False, download=True, transform=plain)
    if quick:
        train, test = Subset(train, range(4000)), Subset(test, range(2000))
    elif limit:
        train = Subset(train, range(limit))
    return (DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2),
            DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=2))


def build_cifar_holdout_loaders(batch_size=256, quick=False, holdout=HOLDOUT, limit=None,
                                dataset="cifar10"):
    cls, plain, _ = _cifar_spec(dataset)
    train_full = cls(DATA_DIR, train=True, download=True, transform=plain)
    test = cls(DATA_DIR, train=False, download=True, transform=plain)
    targets = list(train_full.targets)
    n_train = 8000 if quick else (limit or len(train_full))
    base_idx = list(range(n_train))
    hs = _holdout_set(holdout)
    emb_idx = [i for i in base_idx if targets[i] not in hs]
    if quick:
        test = Subset(test, range(3000))
    emb_ds, probe_ds = Subset(train_full, emb_idx), Subset(train_full, base_idx)
    print(f"  CIFAR embedding-train images (no {holdout}): {len(emb_ds)}   "
          f"probe-train images (all): {len(probe_ds)}")
    return (DataLoader(emb_ds, batch_size=batch_size, shuffle=True, num_workers=2),
            DataLoader(probe_ds, batch_size=batch_size, shuffle=True, num_workers=2),
            DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=2))


def cifar_two_view_loader(batch_size=256, quick=False, labeled=False, holdout=None, limit=None,
                          dataset="cifar10"):
    cls, _, aug = _cifar_spec(dataset)
    raw = cls(DATA_DIR, train=True, download=True, transform=None)
    n = 8000 if quick else (limit or len(raw))
    tgt = list(raw.targets)
    hs = _holdout_set(holdout)
    idx = [i for i in range(n) if tgt[i] not in hs]
    base = Subset(raw, idx)
    ds = TwoViewLabeledMNIST(base, aug) if labeled else TwoViewMNIST(base, aug)
    tag = "" if holdout is None else f" (no {holdout})"
    print(f"  CIFAR two-view embedding-train images{tag}: {len(ds)}")
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=labeled)
