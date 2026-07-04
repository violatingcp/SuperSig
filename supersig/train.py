"""Training loops and evaluation collectors for every experiment."""
import numpy as np
import torch
import torch.nn.functional as F

from .config import DEVICE, HOLDOUT, N_CLASSES
from .losses import (
    classwise_sigreg_loss, sigreg_loss, separation_loss,
    repulsion_loss, shrink_loss, mean_geometry, make_anchors, supcon_loss,
)

# Weights for the repulsive-means objective.
REP_WEIGHT = 20.0
SHRINK_WEIGHT = 0.02
BETA_SEP = 0.5          # weight of the hinge separation term (learnable means)


# --------------------------------------------------------------------------- #
# Supervised baseline                                                          #
# --------------------------------------------------------------------------- #
def train_supervised(model, loader, epochs, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for ep in range(epochs):
        tot, correct, run = 0, 0, 0.0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            run += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            tot += x.size(0)
        print(f"  [supervised] epoch {ep+1}/{epochs}  loss={run/tot:.4f}  acc={correct/tot:.4f}")


# --------------------------------------------------------------------------- #
# Self-supervised SIGReg (invariance + global isotropic-Gaussian)             #
# --------------------------------------------------------------------------- #
def train_sigreg_ssl(backbone, two_view_loader, epochs, lr=1e-3, lam=1.0):
    opt = torch.optim.Adam(backbone.parameters(), lr=lr)
    backbone.train()
    for ep in range(epochs):
        inv_run, reg_run, n = 0.0, 0.0, 0
        for v1, v2 in two_view_loader:
            v1, v2 = v1.to(DEVICE), v2.to(DEVICE)
            opt.zero_grad()
            z1, z2 = backbone(v1), backbone(v2)
            inv = F.mse_loss(z1, z2)
            reg = 0.5 * (sigreg_loss(z1) + sigreg_loss(z2))
            (inv + lam * reg).backward()
            opt.step()
            inv_run += inv.item() * v1.size(0)
            reg_run += reg.item() * v1.size(0)
            n += v1.size(0)
        print(f"  [sigreg-ssl] epoch {ep+1}/{epochs}  inv={inv_run/n:.4f}  sigreg={reg_run/n:.4f}")


# --------------------------------------------------------------------------- #
# Class-conditional SIGReg (fixed anchors / learnable means / repulsion)      #
# --------------------------------------------------------------------------- #
def train_sigreg_classwise(backbone, loader, epochs, means,
                           learn_means=False, mode="fixed", lr=1e-3):
    """
    mode="fixed"      : anchors frozen, no mean regularizer.
    mode="learnmeans" : means learnable, hinge separation term.
    mode="repulse"    : means learnable, inverse-square repulsion + shrinkage.
    """
    params = list(backbone.parameters())
    if learn_means:
        means.requires_grad_(True)
        params = params + [means]
    else:
        means.requires_grad_(False)
    opt = torch.optim.Adam(params, lr=lr)
    means0 = means.detach().clone()          # reference for drift diagnostic
    backbone.train()
    for ep in range(epochs):
        reg_run, aux_run, n = 0.0, 0.0, 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            z = backbone(x)
            reg = classwise_sigreg_loss(z, y, means)
            if mode == "learnmeans":
                aux = BETA_SEP * separation_loss(means)
            elif mode == "repulse":
                aux = REP_WEIGHT * repulsion_loss(means) + SHRINK_WEIGHT * shrink_loss(means)
            else:
                aux = torch.zeros((), device=DEVICE)
            loss = reg + aux
            if not loss.requires_grad:      # batch too small for any class: skip
                continue
            loss.backward()
            opt.step()
            reg_run += reg.item() * x.size(0)
            aux_run += float(aux) * x.size(0)
            n += x.size(0)
        dmin, dmean = mean_geometry(means.detach())
        drift = (means.detach() - means0).norm().item()
        print(f"  [sigreg-{mode}] epoch {ep+1}/{epochs}  sigreg={reg_run/n:.4f}  "
              f"aux={aux_run/n:.4f}  min_dist={dmin:.2f}  mean_dist={dmean:.2f}  drift={drift:.3f}")


# --------------------------------------------------------------------------- #
# Supervised contrastive (supervised SimCLR)                                  #
# --------------------------------------------------------------------------- #
def train_supcon(backbone, loader, epochs, temp=0.1, lr=1e-3):
    opt = torch.optim.Adam(backbone.parameters(), lr=lr)
    backbone.train()
    for ep in range(epochs):
        run, n = 0.0, 0
        for v1, v2, y in loader:
            v1, v2, y = v1.to(DEVICE), v2.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            z = F.normalize(backbone(torch.cat([v1, v2])), dim=1)
            loss = supcon_loss(z, torch.cat([y, y]), temp=temp)
            loss.backward()
            opt.step()
            run += loss.item() * v1.size(0)
            n += v1.size(0)
        print(f"  [supcon] epoch {ep+1}/{epochs}  loss={run/n:.4f}")


def train_sigreg_hybrid(backbone, loader, epochs, means, mode="repulse",
                        disc="supcon", alpha=1.0, temp=0.1, lr=1e-3, margin=3.0,
                        rep_weight=REP_WEIGHT):
    """
    Classwise SIGReg + mean-geometry regularizer + a discriminative term.

    disc="supcon" : SupCon on the L2-normalised embeddings (plain single views).
    disc="ce"     : cross-entropy through a jointly trained linear head, which is
                    discarded afterwards (the frozen probe is trained separately).
    disc="proto"  : cross-entropy of the Gaussian model's own posterior,
                    logits = -||z - mean_c||^2 / 2 (no extra parameters).
    disc="hinge"  : purely geometric, CE-free: relu(margin - ||z - mean_wrong||)^2
                    keeps every sample at least `margin` sigma from wrong means.
    Mean-geometry `mode` follows train_sigreg_classwise (means always learnable).
    """
    means.requires_grad_(True)
    params = list(backbone.parameters()) + [means]
    head = None
    if disc == "ce":
        head = torch.nn.Linear(means.size(1), means.size(0)).to(DEVICE)
        params += list(head.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    backbone.train()
    for ep in range(epochs):
        reg_run, disc_run, n = 0.0, 0.0, 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            z = backbone(x)
            reg = classwise_sigreg_loss(z, y, means)
            if mode == "learnmeans":
                aux = BETA_SEP * separation_loss(means)
            elif mode == "repulse":
                aux = rep_weight * repulsion_loss(means) + SHRINK_WEIGHT * shrink_loss(means)
            else:
                aux = torch.zeros((), device=DEVICE)
            if disc == "supcon":
                d = supcon_loss(F.normalize(z, dim=1), y, temp=temp)
            elif disc == "ce":
                d = F.cross_entropy(head(z), y)
            elif disc == "proto":
                d = F.cross_entropy(-0.5 * torch.cdist(z, means).pow(2), y)
            else:  # "hinge"
                dist = torch.cdist(z, means)
                dist = dist + F.one_hot(y, means.size(0)).float() * 1e6  # mask own class
                d = F.relu(margin - dist).pow(2).mean()
            (reg + aux + alpha * d).backward()
            opt.step()
            reg_run += reg.item() * x.size(0)
            disc_run += d.item() * x.size(0)
            n += x.size(0)
        dmin, dmean = mean_geometry(means.detach())
        print(f"  [sigreg+{disc}] epoch {ep+1}/{epochs}  sigreg={reg_run/n:.4f}  "
              f"{disc}={disc_run/n:.4f}  min_dist={dmin:.2f}  mean_dist={dmean:.2f}")


def train_supcon_plain(backbone, loader, epochs, temp=0.1, lr=1e-3):
    """SupCon on single un-augmented views: positives are same-class samples only."""
    opt = torch.optim.Adam(backbone.parameters(), lr=lr)
    backbone.train()
    for ep in range(epochs):
        run, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            z = F.normalize(backbone(x), dim=1)
            loss = supcon_loss(z, y, temp=temp)
            loss.backward()
            opt.step()
            run += loss.item() * x.size(0)
            n += x.size(0)
        print(f"  [supcon-plain] epoch {ep+1}/{epochs}  loss={run/n:.4f}")


# --------------------------------------------------------------------------- #
# Frozen-backbone linear probes                                               #
# --------------------------------------------------------------------------- #
def _freeze(backbone):
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()


def train_linear_probe(backbone, head, loader, epochs, lr=1e-3):
    """Freeze backbone, train a multi-class linear head with cross-entropy."""
    _freeze(backbone)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    for ep in range(epochs):
        tot, correct, run = 0, 0, 0.0
        head.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.no_grad():
                z = backbone(x)
            opt.zero_grad()
            logits = head(z)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            run += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            tot += x.size(0)
        print(f"  [linear probe] epoch {ep+1}/{epochs}  loss={run/tot:.4f}  acc={correct/tot:.4f}")


def train_binary_probe(backbone, head, loader, epochs, positive=HOLDOUT, lr=1e-3):
    """Freeze backbone, train a 2-way (positive vs rest) linear head."""
    _freeze(backbone)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    for ep in range(epochs):
        tot, correct, run = 0, 0, 0.0
        head.train()
        for x, y in loader:
            x = x.to(DEVICE)
            yb = (y == positive).long().to(DEVICE)
            with torch.no_grad():
                z = backbone(x)
            opt.zero_grad()
            logits = head(z)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()
            run += loss.item() * x.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            tot += x.size(0)
        print(f"  [binary probe] epoch {ep+1}/{epochs}  loss={run/tot:.4f}  acc={correct/tot:.4f}")


# --------------------------------------------------------------------------- #
# Evaluation collectors                                                        #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def collect_probs(forward_fn, loader):
    probs, labels = [], []
    for x, y in loader:
        p = F.softmax(forward_fn(x.to(DEVICE)), dim=1)
        probs.append(p.cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


@torch.no_grad()
def collect_binary_scores(backbone, head, loader, positive=HOLDOUT):
    backbone.eval(); head.eval()
    scores, labels = [], []
    for x, y in loader:
        p = F.softmax(head(backbone(x.to(DEVICE))), dim=1)[:, 1]
        scores.append(p.cpu().numpy())
        labels.append((y == positive).long().numpy())
    return np.concatenate(scores), np.concatenate(labels)


@torch.no_grad()
def collect_embeddings(backbone, loader):
    backbone.eval()
    embs, labels = [], []
    for x, y in loader:
        embs.append(backbone(x.to(DEVICE)).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(embs), np.concatenate(labels)
