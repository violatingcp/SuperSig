"""
SuperSig-Lite: the one pipeline, as four calls.

    parent = train_parent("cifar10", holdout={4}, epochs=200)
    child  = train_child(parent, epochs=20)
    post   = discover(child, rounds=2)
    report = detect(post)

Strategy (fixed; justifications in README.md):
  parent   SupCon on the seen classes (``--parent ssig`` adds a global
           SIGReg marginal at lambda=5 -- the many-class variant).
  child    ``res``: NT-Xent + global SIGReg (lambda=5) on the residuals
           r = z - mu_y.  Because r is class-centred, this enforces one
           shared isotropic marginal over the class-centred cloud.
  discover the settled loop with the density-ratio (SparKer-critic) pool,
           the derived label-free cut at n_min=5, skip-on-refuse, and the
           class-wise SIGReg-regularised fine-tune (2 rounds x 5 epochs).
  detect   the anchor-aware Euclidean statistic mean(d_seen - d_disc),
           plus anchor-seeded SparKer, calibrated on null toys; and the
           frozen-trunk mean-Mahalanobis systematics monitor.

Every function returns a plain dict so stages can be cached/inspected.
The heavy lifting lives unchanged in the vendored ``supersig`` package.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import copy

import numpy as np
import torch
import torch.nn.functional as F

from supersig.config import DEVICE
from supersig.data import (get_cifar_loaders, cifar_two_view_loader,
                           _cifar_spec, DATA_DIR)
from supersig.discovery import run_discovery
from supersig.losses import supcon_loss, sigreg_loss
from supersig.metrics import mahalanobis_novelty
from supersig.models import CIFARResNetBackbone
from supersig.recipes import recipe
from supersig.sparker import np_test_stats, median_pairwise
from supersig.train import (collect_embeddings, train_supcon,
                            train_supcon_sigreg)


def _centroids(tr, tr_lab, seen, emb_dim, n_classes):
    """Per-class embedding means for the seen classes (zeros elsewhere)."""
    cents = torch.zeros(n_classes, emb_dim, device=DEVICE)
    z = torch.as_tensor(tr, dtype=torch.float32, device=DEVICE)
    lab = torch.as_tensor(tr_lab, device=DEVICE)
    for c in seen:
        cents[c] = z[lab == c].mean(0)
    return cents


def train_parent(dataset="cifar10", holdout={4}, epochs=200, emb_dim=100,
                 parent="supcon", quick=False, seed=0):
    """SupCon (default) or SupCon+SIGReg ('ssig') on the seen classes only."""
    cfg = recipe(dataset, emb_dim=emb_dim)
    n_cls = cfg["n_classes"]
    seen = [c for c in range(n_cls) if c not in holdout]
    torch.manual_seed(seed)
    net = CIFARResNetBackbone(emb_dim, arch=cfg["arch"], pretrain=None).to(DEVICE)
    loader = cifar_two_view_loader(quick=quick, labeled=True,
                                   holdout=holdout, dataset=dataset)
    ep = 2 if quick else epochs
    if parent == "supcon":
        train_supcon(net, loader, ep)
    elif parent == "ssig":
        train_supcon_sigreg(net, loader, ep, lam=5.0)
    else:
        raise ValueError(parent)
    return dict(net=net, dataset=dataset, holdout=set(holdout), seen=seen,
                n_classes=n_cls, emb_dim=emb_dim, cfg=cfg, quick=quick,
                seed=seed, parent=parent)


def train_child(parent, epochs=20, lam=5.0):
    """The res child: NT-Xent + global SIGReg on r = z - mu_y (class-centred)."""
    net, ds = parent["net"], parent["dataset"]
    tel = torch.utils.data.DataLoader(
        get_cifar_loaders(quick=parent["quick"], dataset=ds)[0].dataset,
        batch_size=256, shuffle=False, num_workers=2)
    tr, tr_lab = collect_embeddings(net, tel)
    cents = _centroids(tr, tr_lab, parent["seen"], parent["emb_dim"],
                       parent["n_classes"])
    child = copy.deepcopy(net)
    loader = cifar_two_view_loader(quick=parent["quick"], labeled=True,
                                   holdout=parent["holdout"], dataset=ds)
    opt = torch.optim.Adam(child.parameters(), lr=1e-3)
    child.train()
    ep = 1 if parent["quick"] else epochs
    for e in range(ep):
        run, n = 0.0, 0
        for v1, v2, y in loader:
            x = torch.cat([v1, v2]).to(DEVICE, non_blocking=True)
            yy = torch.cat([y, y]).to(DEVICE)
            r = child(x).float() - cents[yy]
            inst = torch.arange(v1.size(0), device=DEVICE)
            loss = (supcon_loss(F.normalize(r, dim=1),
                                torch.cat([inst, inst]), temp=0.5)
                    + lam * sigreg_loss(r))
            opt.zero_grad(); loss.backward(); opt.step()
            run += loss.item() * v1.size(0); n += v1.size(0)
        print(f"  [child] epoch {e+1}/{ep}  loss={run/n:.4f}", flush=True)
    out = dict(parent)
    out["net"] = child
    out["parent_net"] = net
    return out


def discover(space, rounds=2, ft_epochs=5, n_min=5, seed=0):
    """The settled loop: np pool, derived cut, skip-on-refuse, SIGReg fine-tune."""
    ds, cfg = space["dataset"], space["cfg"]
    train_loader, test_loader = get_cifar_loaders(quick=space["quick"],
                                                  dataset=ds)
    tel = torch.utils.data.DataLoader(train_loader.dataset, batch_size=256,
                                      shuffle=False, num_workers=2)
    tr, tr_lab = collect_embeddings(space["net"], tel)
    means0 = _centroids(tr, tr_lab, space["seen"], space["emb_dim"],
                        space["n_classes"])
    cls, plain, _ = _cifar_spec(ds)
    base_ds = cls(DATA_DIR, train=True, download=True, transform=plain)
    bb = copy.deepcopy(space["net"])
    cur_means, hist = run_discovery(
        bb, means0.clone(), base_ds=base_ds, train_eval_loader=tel,
        test_loader=test_loader, seen=space["seen"],
        holdouts=space["holdout"], dataset_name=ds,
        rep_weight=cfg["rep_weight"], sigreg_weight=cfg["sigreg_weight"],
        n_slices=cfg["n_slices"], rounds=1 if space["quick"] else rounds,
        ft_epochs=1 if space["quick"] else ft_epochs, names=None, seed=seed,
        pool_score="np", cut_rule="legal", n_min=n_min, on_refuse="skip")
    out = dict(space)
    out.update(net=bb, means=cur_means, history=hist,
               declined=bool(hist and hist[-1].get("declined")))
    return out


def detect(post, alpha=0.05, sparker_kw=None):
    """Anchor-aware Euclidean + anchor-seeded SparKer on the corpus, plus the
    frozen-trunk Mahalanobis systematics monitor.  Returns scores + AUCs."""
    from sklearn.metrics import roc_auc_score
    _, test_loader = get_cifar_loaders(quick=post["quick"],
                                       dataset=post["dataset"])
    tel = torch.utils.data.DataLoader(
        get_cifar_loaders(quick=post["quick"],
                          dataset=post["dataset"])[0].dataset,
        batch_size=256, shuffle=False, num_workers=2)
    te, te_lab = collect_embeddings(post["net"], test_loader)
    tr, tr_lab = collect_embeddings(post["net"], tel)
    seen, n_cls = post["seen"], post["n_classes"]
    is_novel = np.isin(te_lab, list(post["holdout"]))
    zt = torch.as_tensor(te, dtype=torch.float32, device=DEVICE)
    d_seen = torch.cdist(zt, post["means"][seen]).min(1).values
    disc = post["means"][n_cls:]
    report = dict(n_discovered_anchors=int(disc.size(0)),
                  declined=post.get("declined", False))
    if disc.size(0):
        d_disc = torch.cdist(zt, disc).min(1).values
        s = (d_seen - d_disc).cpu().numpy()
        report["eucl_anchor_auc"] = float(roc_auc_score(is_novel, s))
        report["scores_eucl_anchor"] = s
        R = torch.as_tensor(tr[np.isin(tr_lab, seen)][:20000],
                            dtype=torch.float32, device=DEVICE)
        bg = zt[torch.as_tensor(~is_novel, device=DEVICE)]
        sigma0 = median_pairwise(bg, seed=0)
        kw = dict(M=16, steps=60 if post["quick"] else 300, sigma0=sigma0,
                  mu_init=disc.detach())
        kw.update(sparker_kw or {})
        report["sparker_anchor_t"] = [float(t) for t in
                                      np_test_stats(zt, R, **kw)]
    _, pc, _ = mahalanobis_novelty(tr, tr_lab, te, seen)
    report["monitor_maha_mean"] = float(np.mean(pc))
    report["per_event_power"] = (float(np.mean(
        report["scores_eucl_anchor"][is_novel] > np.quantile(
            report["scores_eucl_anchor"][~is_novel], 1 - alpha)))
        if disc.size(0) else None)
    return report
