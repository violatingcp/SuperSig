"""
Experiment 28: SSL SIGReg pretraining, concatenated spaces, and SSL on the
supervised residual (matching-pursuit inspired, arXiv:2506.03093).

Four 4-dim (or --emb-dim) spaces on the same holdout split, all evaluated with
the exp-27 anomaly suite (nearest-anchor accuracy on seen classes, novelty
AUC, gaussianity table, discovery-with-clustering, latent plots):

  sup    : the settled supervised embedding (classwise SIGReg + proto +
           repulsed means -- the exp-26 recipe; baseline).
  ssl    : plain unlabeled SIGReg SSL with augmentations (two views,
           invariance + global N(0,I); holdout excluded).  Anchors are the
           empirical seen-class centroids.
  concat : [sup ; ssl] (2 x emb_dim).  Anchors [learned mean ; ssl centroid];
           discovery clusters in the concat space and fine-tunes the SUP
           branch only (exp-25 recipe).
  res    : SSL on the supervised residual.  Starting from the trained sup
           backbone, train on augmented views with invariance + SIGReg applied
           to z - mean_y (means frozen): the class atom explains the class
           component, the residual is shaped into one augment-invariant
           N(0, I) -- one matching-pursuit step past the class atoms.

    python experiments/28_concat_residual.py                    # CIFAR-10
    python experiments/28_concat_residual.py --dataset mnist
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import copy
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.metrics import roc_auc_score

from supersig.config import DATA_DIR, DEVICE, plot_path
from supersig.data import (get_cifar_loaders, get_loaders, cifar_two_view_loader,
                           two_view_loader, cifar_balanced_loader,
                           mnist_balanced_loader, BalancedBatchSampler,
                           _cifar_spec, TF_PLAIN)
from supersig.losses import make_anchors
from supersig.models import CIFARResNetBackbone, ConvBackbone
from supersig.metrics import gaussianity_summary
from supersig.plotting import plot_latent_panels
from supersig.recipes import supervised_embedding, recipe
from supersig.discovery import run_discovery, PseudoDataset, bic_select
from supersig.train import (train_sigreg_ssl, train_sigreg_hybrid,
                            train_sigreg_residual_ssl, collect_embeddings)

CIFAR_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

GAUSS_ROWS = [
    ("eig min (class cov)", "eig_min", ".3f"),
    ("eig max (class cov)", "eig_max", ".3f"),
    ("eig cond worst", "eig_cond_max", ".1f"),
    ("class RMS min", "rms_min", ".3f"),
    ("class RMS mean", "rms_mean", ".3f"),
    ("class RMS max", "rms_max", ".3f"),
    ("max |corr| (worst class)", "corr_max", ".3f"),
    ("SW ratio mean", "sw_ratio_mean", ".2f"),
    ("SW ratio worst", "sw_ratio_max", ".2f"),
    ("|skew| mean", "skew_mean", ".3f"),
    ("|ex-kurt| mean", "kurt_mean", ".3f"),
    ("centroid dist min", "cdist_min", ".2f"),
    ("centroid dist mean", "cdist_mean", ".2f"),
    ("separation (min d/RMS)", "separation", ".2f"),
]


def print_gauss_table(spaces):
    print(f"  {'metric':<26}" + "".join(f"{n:>14}" for n in spaces))
    for label, key, fmt in GAUSS_ROWS:
        print(f"  {label:<26}"
              + "".join(f"{spaces[n][key]:>14{fmt}}" for n in spaces))


def anchor_eval(embs, lab, anchors, seen, holdouts):
    """Nearest-anchor accuracy on seen test classes + novelty AUC.

    anchors: (len(seen), D) matrix, row i = anchor of class seen[i]."""
    z = torch.as_tensor(embs, device=DEVICE)
    d = torch.cdist(z, torch.as_tensor(anchors, device=DEVICE))
    pred = np.array(seen)[d.argmin(1).cpu().numpy()]
    seen_mask = np.isin(lab, seen)
    acc = float((pred[seen_mask] == lab[seen_mask]).mean())
    is_unseen = np.isin(lab, list(holdouts)).astype(int)
    auc = float(roc_auc_score(is_unseen, d.min(1).values.cpu().numpy()))
    return acc, auc


def class_centroids(embs, lab, classes):
    Z = torch.as_tensor(embs, device=DEVICE)
    return torch.stack([Z[torch.as_tensor(lab == c, device=DEVICE)].mean(0)
                        for c in classes])


def fill_means(centroids, seen, cfg):
    """Full n_classes-row means matrix: empirical centroids for seen rows,
    fixed-anchor fill for holdout rows (parity with exp 27)."""
    means = make_anchors(cfg["pair_dist"] / math.sqrt(2.0),
                         emb_dim=centroids.size(1),
                         n_classes=cfg["n_classes"]).clone()
    for i, c in enumerate(seen):
        means[c] = centroids[i]
    return means


def run_concat_discovery(sup, trunk, means_sup, ssl_cents, *, base, dim,
                         train_eval_loader, test_loader, seen, holdouts, cfg,
                         rounds=2, ft_epochs=5, tau_quantile=0.95, names=None,
                         seed=0):
    """
    Discovery in the concatenated [sup ; ssl] space (exp-25 recipe): pool ->
    BIC k-means in concat -> pseudo-label -> fine-tune the SUP branch only ->
    refresh discovered ssl halves.  Returns history like run_discovery.
    """
    n_classes = cfg["n_classes"]
    ssl_tr, tr_lab = collect_embeddings(trunk, train_eval_loader)
    ssl_te, te_lab = collect_embeddings(trunk, test_loader)
    Zssl_tr = torch.as_tensor(ssl_tr, device=DEVICE)
    Zssl_te = torch.as_tensor(ssl_te, device=DEVICE)
    is_seen_lab = np.isin(tr_lab, seen)
    cur_means = means_sup.detach().clone()
    disc_ssl = None
    pooled = np.zeros(len(train_eval_loader.dataset), dtype=bool)
    history = []
    for r in range(1, rounds + 1):
        sup_tr, _ = collect_embeddings(sup, train_eval_loader)
        Zcat = torch.cat([torch.as_tensor(sup_tr, device=DEVICE), Zssl_tr], 1)
        seen_anchors = torch.cat([cur_means[seen], ssl_cents], dim=1)
        anchor_mat = seen_anchors if disc_ssl is None else torch.cat(
            [seen_anchors, torch.cat([cur_means[n_classes:], disc_ssl], 1)])
        dmin = torch.cdist(Zcat, anchor_mat).min(1).values
        tau = torch.quantile(dmin[torch.as_tensor(is_seen_lab, device=DEVICE)],
                             tau_quantile)
        pool = (dmin > tau).cpu().numpy()
        purity = (~is_seen_lab[pool]).mean() if pool.any() else float("nan")
        km = max(4, len(holdouts) + 2)
        khat, centers, _ = bic_select(Zcat[torch.as_tensor(pool, device=DEVICE)],
                                      kmax=km, seed=seed + r)
        cur_means = torch.cat([cur_means, centers[:, :dim].detach()], dim=0)
        disc_ssl = centers[:, dim:].detach() if disc_ssl is None else torch.cat(
            [disc_ssl, centers[:, dim:].detach()])
        pooled |= pool
        disc_anchors = torch.cat([cur_means[n_classes:], disc_ssl], dim=1)
        p_idx = np.where(pooled)[0]
        p_lab = n_classes + torch.cdist(
            Zcat[torch.as_tensor(pooled, device=DEVICE)],
            disc_anchors).argmin(1).cpu().numpy()
        lab_idx = np.where(is_seen_lab)[0]
        ft_idx = np.concatenate([lab_idx, p_idx])
        ft_lab = np.concatenate([tr_lab[lab_idx], p_lab])
        n_pb = len(seen) + disc_ssl.size(0)
        sampler = BalancedBatchSampler(list(ft_lab), n_classes=n_pb,
                                       n_per_class=24)
        ft_loader = DataLoader(PseudoDataset(base, ft_idx, ft_lab),
                               batch_sampler=sampler, num_workers=2)
        train_sigreg_hybrid(sup, ft_loader, ft_epochs, cur_means,
                            mode="repulse", disc="proto", alpha=1.0,
                            rep_weight=cfg["rep_weight"],
                            sigreg_weight=cfg["sigreg_weight"],
                            n_slices=cfg["n_slices"],
                            rep_exempt_from=n_classes)
        cur_means = cur_means.detach()
        sup_tr, _ = collect_embeddings(sup, train_eval_loader)
        Zcat = torch.cat([torch.as_tensor(sup_tr, device=DEVICE), Zssl_tr], 1)
        for j in range(disc_ssl.size(0)):
            m = np.zeros(len(tr_lab), dtype=bool)
            m[p_idx[p_lab == n_classes + j]] = True
            if m.any():
                disc_ssl[j] = Zssl_tr[torch.as_tensor(m, device=DEVICE)].mean(0)

        sup_te, _ = collect_embeddings(sup, test_loader)
        Zcat_te = torch.cat([torch.as_tensor(sup_te, device=DEVICE), Zssl_te], 1)
        seen_anchors = torch.cat([cur_means[seen], ssl_cents], dim=1)
        disc_anchors = torch.cat([cur_means[n_classes:], disc_ssl], dim=1)
        d_seen = torch.cdist(Zcat_te, seen_anchors).min(1).values
        d_each = torch.cdist(Zcat_te, disc_anchors)
        is_unseen = np.isin(te_lab, list(holdouts)).astype(int)
        margin = roc_auc_score(is_unseen,
                               (d_seen - d_each.min(1).values).cpu().numpy())
        per_class = {}
        for c in sorted(holdouts):
            counts = [int(((te_lab == c) & (d_each.argmin(1).cpu().numpy() == j)).sum())
                      for j in range(d_each.size(1))]
            j = int(np.argmax(counts))
            per_class[c] = roc_auc_score((te_lab == c).astype(int),
                                         (-d_each[:, j]).cpu().numpy())
        history.append(dict(round=r, pool=int(pool.sum()), purity=float(purity),
                            khat=khat, n_anchors=int(disc_ssl.size(0)),
                            margin=float(margin),
                            per_class={int(c): float(a) for c, a in per_class.items()},
                            mean_pc=float(np.mean(list(per_class.values())))))
        h = history[-1]
        pc = "  ".join(f"{(names[c] if names else c)}={a:.3f}"
                       for c, a in per_class.items()) if len(per_class) <= 5 else ""
        print(f"  round {r}: pool={h['pool']} purity={h['purity']:.3f} "
              f"k-hat={h['khat']} anchors={h['n_anchors']}  "
              f"margin={h['margin']:.4f}  mean-anchor={h['mean_pc']:.4f}  {pc}")
    return history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar10", "mnist"], default="cifar10")
    ap.add_argument("--holdout", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--emb-dim", type=int, default=4)
    ap.add_argument("--ssl-epochs", type=int, default=None)
    ap.add_argument("--res-epochs", type=int, default=None)
    ap.add_argument("--plots", action="store_true")
    args = ap.parse_args()
    ds = args.dataset
    cfg = recipe("cifar10", emb_dim=args.emb_dim)
    ssl_ep = args.ssl_epochs or (2 if args.quick else 20)
    res_ep = args.res_epochs or (2 if args.quick else 10)
    ft_ep = 1 if args.quick else cfg["ft_epochs"]
    holdouts = {args.holdout}
    seen = [c for c in range(cfg["n_classes"]) if c not in holdouts]
    names = [str(d) for d in range(10)] if ds == "mnist" else CIFAR_NAMES

    if ds == "mnist":
        train_loader, test_loader = get_loaders(batch_size=256, quick=args.quick)
        base = datasets.MNIST(DATA_DIR, train=True, download=True,
                              transform=TF_PLAIN)
        tv_loader = two_view_loader(quick=args.quick, labeled=False,
                                    holdout=args.holdout)
        tv_lab_loader = two_view_loader(quick=args.quick, labeled=True,
                                        holdout=args.holdout)
    else:
        train_loader, test_loader = get_cifar_loaders(quick=args.quick,
                                                      limit=args.limit)
        cls, plain, _ = _cifar_spec("cifar10")
        base = cls(DATA_DIR, train=True, download=True, transform=plain)
        tv_loader = cifar_two_view_loader(quick=args.quick, labeled=False,
                                          holdout=holdouts, limit=args.limit)
        tv_lab_loader = cifar_two_view_loader(quick=args.quick, labeled=True,
                                              holdout=holdouts, limit=args.limit)
    train_eval_loader = DataLoader(train_loader.dataset, batch_size=256,
                                   shuffle=False, num_workers=2)
    print(f"exp28 [{ds}] emb_dim={cfg['emb_dim']} holdout={sorted(holdouts)} "
          f"ssl_ep={ssl_ep} res_ep={res_ep}")

    # ----- sup: settled supervised embedding --------------------------------
    print("\n----- space: sup (supervised SIGReg, exp-26 recipe) -----")
    if ds == "cifar10":
        sup, means_sup, _ = supervised_embedding(
            "cifar10", holdouts=holdouts, quick=args.quick, limit=args.limit,
            seed=args.seed, emb_dim=cfg["emb_dim"])
    else:
        torch.manual_seed(args.seed); np.random.seed(args.seed)
        sup = ConvBackbone(cfg["emb_dim"]).to(DEVICE)
        means_sup = make_anchors(cfg["pair_dist"] / math.sqrt(2.0),
                                 emb_dim=cfg["emb_dim"],
                                 n_classes=cfg["n_classes"]).clone()
        train_sigreg_hybrid(sup, mnist_balanced_loader(holdout=holdouts,
                                                       quick=args.quick),
                            cfg["ssl_epochs"], means_sup, mode="repulse",
                            disc="proto", alpha=1.0,
                            rep_weight=cfg["rep_weight"],
                            sigreg_weight=cfg["sigreg_weight"],
                            n_slices=cfg["n_slices"])
    means_sup = means_sup.detach()

    # ----- ssl: unlabeled two-view SIGReg -----------------------------------
    print("\n----- space: ssl (unlabeled SIGReg + augmentations) -----")
    torch.manual_seed(args.seed + 1); np.random.seed(args.seed + 1)
    trunk = (ConvBackbone(cfg["emb_dim"]) if ds == "mnist" else
             CIFARResNetBackbone(cfg["emb_dim"], arch=cfg["arch"],
                                 pretrain="cifar10")).to(DEVICE)
    train_sigreg_ssl(trunk, tv_loader, ssl_ep)

    # ----- res: SSL on the supervised residual ------------------------------
    print("\n----- space: res (SSL on the supervised residual) -----")
    torch.manual_seed(args.seed + 2); np.random.seed(args.seed + 2)
    res = copy.deepcopy(sup)
    train_sigreg_residual_ssl(res, tv_lab_loader, res_ep, means_sup,
                              n_slices=cfg["n_slices"])

    # ----- assemble spaces and anchors --------------------------------------
    sup_te, te_lab = collect_embeddings(sup, test_loader)
    ssl_te, _ = collect_embeddings(trunk, test_loader)
    res_te, _ = collect_embeddings(res, test_loader)
    cat_te = np.concatenate([sup_te, ssl_te], axis=1)

    ssl_tr, tr_lab = collect_embeddings(trunk, train_eval_loader)
    tr_seen = np.isin(tr_lab, seen)
    ssl_cents = class_centroids(ssl_tr[tr_seen], tr_lab[tr_seen], seen)
    res_tr, _ = collect_embeddings(res, train_eval_loader)
    res_cents = class_centroids(res_tr[tr_seen], tr_lab[tr_seen], seen)

    anchors = {
        "sup": means_sup[seen],
        "ssl": ssl_cents,
        "concat": torch.cat([means_sup[seen], ssl_cents], dim=1),
        "res": res_cents,
    }
    tests = {"sup": sup_te, "ssl": ssl_te, "concat": cat_te, "res": res_te}

    print("\n----- pre-discovery metrics -----")
    pre = {}
    for name in ("sup", "ssl", "concat", "res"):
        acc, auc = anchor_eval(tests[name], te_lab, anchors[name], seen, holdouts)
        pre[name] = (acc, auc)
        print(f"  [{name:6s}] seen nearest-anchor acc={acc:.4f}  "
              f"novelty AUC={auc:.4f}")

    print("\n----- gaussianity (seen classes, test set) -----")
    gauss = {name: gaussianity_summary(tests[name], te_lab, seen, seed=args.seed)
             for name in tests}
    print_gauss_table(gauss)

    if args.plots:
        tag = f"{ds}_exp28_{cfg['emb_dim']}d"
        plot_latent_panels({n: (tests[n], te_lab) for n in tests}, holdouts,
                           names, plot_path(f"latent_{tag}.png"),
                           title=f"exp28 [{ds}]: sup / ssl / concat / res")

    # ----- discovery with clustering in each space --------------------------
    # run_discovery fine-tunes the backbone in place: give each loop a copy.
    hist = {}
    print("\n----- discovery: sup -----")
    _, hist["sup"] = run_discovery(
        copy.deepcopy(sup), means_sup.clone(), base_ds=base,
        train_eval_loader=train_eval_loader,
        test_loader=test_loader, seen=seen, holdouts=holdouts, dataset_name=ds,
        rep_weight=cfg["rep_weight"], sigreg_weight=cfg["sigreg_weight"],
        n_slices=cfg["n_slices"], rounds=args.rounds, ft_epochs=ft_ep,
        names=names, seed=args.seed)
    print("\n----- discovery: ssl (centroid anchors) -----")
    _, hist["ssl"] = run_discovery(
        copy.deepcopy(trunk), fill_means(ssl_cents, seen, cfg), base_ds=base,
        train_eval_loader=train_eval_loader, test_loader=test_loader,
        seen=seen, holdouts=holdouts, dataset_name=ds,
        rep_weight=cfg["rep_weight"], sigreg_weight=cfg["sigreg_weight"],
        n_slices=cfg["n_slices"], rounds=args.rounds, ft_epochs=ft_ep,
        names=names, seed=args.seed)
    print("\n----- discovery: concat (ft sup branch only) -----")
    hist["concat"] = run_concat_discovery(
        copy.deepcopy(sup), trunk, means_sup.clone(), ssl_cents, base=base,
        dim=cfg["emb_dim"], train_eval_loader=train_eval_loader,
        test_loader=test_loader, seen=seen, holdouts=holdouts, cfg=cfg,
        rounds=args.rounds, ft_epochs=ft_ep, names=names, seed=args.seed)
    print("\n----- discovery: res (centroid anchors) -----")
    _, hist["res"] = run_discovery(
        copy.deepcopy(res), fill_means(res_cents, seen, cfg), base_ds=base,
        train_eval_loader=train_eval_loader, test_loader=test_loader,
        seen=seen, holdouts=holdouts, dataset_name=ds,
        rep_weight=cfg["rep_weight"], sigreg_weight=cfg["sigreg_weight"],
        n_slices=cfg["n_slices"], rounds=args.rounds, ft_epochs=ft_ep,
        names=names, seed=args.seed)

    print(f"\n===== EXP28 SUMMARY [{ds}, {cfg['emb_dim']}d] =====")
    for name in ("sup", "ssl", "concat", "res"):
        acc, auc = pre[name]
        print(f"  [{name:6s}] pre: acc={acc:.4f} novelty-AUC={auc:.4f}")
        for h in hist[name]:
            print(f"          round {h['round']}: purity={h['purity']:.3f} "
                  f"anchors={h['n_anchors']}  margin={h['margin']:.4f}  "
                  f"mean-anchor={h['mean_pc']:.4f}")
    print("Done.")


if __name__ == "__main__":
    main()
