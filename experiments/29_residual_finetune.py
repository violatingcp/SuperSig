"""
Experiment 29: ordering the supervised x augmentation combination (CIFAR-10).

Six arms, exp-25 style: every arm that has both a supervised and an
augmentation-trained part is evaluated in the CONCATENATED space (discovery
fine-tunes the supervised branch only, the aug branch stays frozen):

  sup->res      : supervised SIGReg, then classwise residual-SSL on
                  augmentations (exp-28 res-cw); eval on [sup ; res]
  ssl->supres   : SIGReg SSL on augmentations, then supervised SIGReg seeded
                  from its centroids (exp-28 supres); eval on [supres ; ssl]
  joint         : ONE network, supervised SIGReg hybrid + augmentation
                  invariance in parallel (train_sigreg_hybrid_aug)
  sup           : straight supervised SIGReg (exp-26 recipe; baseline)
  supcon        : supervised SimCLR (SupCon, two augmented views)
  supcon+simclr : SupCon and unsupervised SimCLR concatenated

Per arm: performance (seen nearest-anchor accuracy, macro one-vs-rest AUC of
the proto posterior over seen classes), novelty AUC by Euclidean distance and
by Mahalanobis distance (tied and per-class covariances,
supersig.metrics.mahalanobis_novelty), the gaussianity table, and the settled
discovery clustering (pool -> BIC k-means -> anchors -> fine-tune rounds).

    python experiments/29_residual_finetune.py
    python experiments/29_residual_finetune.py --quick --rounds 1
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import copy
import importlib
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from supersig.config import DATA_DIR, DEVICE, plot_path
from supersig.data import (get_cifar_loaders, cifar_balanced_loader,
                           cifar_two_view_loader, cifar_two_view_balanced_loader,
                           _cifar_spec)
from supersig.losses import make_anchors
from supersig.models import CIFARResNetBackbone
from supersig.metrics import gaussianity_summary, mahalanobis_novelty
from supersig.plotting import plot_latent_panels, plot_corner
from supersig.recipes import supervised_embedding, recipe
from supersig.discovery import run_discovery
from supersig.train import (train_sigreg_ssl, train_sigreg_hybrid,
                            train_sigreg_hybrid_aug, train_sigreg_residual_ssl,
                            train_supcon, train_simclr, collect_embeddings)

exp28 = importlib.import_module("28_concat_residual")

CIFAR_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]


def evaluate_space(tr_embs, tr_lab, te_embs, te_lab, anchors, seen, holdouts):
    """Performance + novelty metrics for one evaluation space.

    anchors: (len(seen), D), row i = anchor of class seen[i]."""
    z = torch.as_tensor(te_embs, device=DEVICE)
    d = torch.cdist(z, torch.as_tensor(anchors, device=DEVICE))
    pred = np.array(seen)[d.argmin(1).cpu().numpy()]
    seen_mask = np.isin(te_lab, seen)
    acc = float((pred[seen_mask] == te_lab[seen_mask]).mean())

    # macro one-vs-rest AUC of the proto posterior, seen test samples only
    post = torch.softmax(-0.5 * d.pow(2), dim=1).cpu().numpy()[seen_mask]
    lab_seen = te_lab[seen_mask]
    aucs = [roc_auc_score((lab_seen == c).astype(int), post[:, i])
            for i, c in enumerate(seen)]
    sup_auc = float(np.mean(aucs))

    is_unseen = np.isin(te_lab, list(holdouts)).astype(int)
    eucl_scores = d.min(1).values.cpu().numpy()
    eucl_auc = float(roc_auc_score(is_unseen, eucl_scores))
    tied, perclass, eigs = mahalanobis_novelty(tr_embs, tr_lab, te_embs, seen)
    maha_tied = float(roc_auc_score(is_unseen, tied))
    maha_pc = float(roc_auc_score(is_unseen, perclass))
    return dict(acc=acc, sup_auc=sup_auc, eucl=eucl_auc,
                maha_tied=maha_tied, maha_pc=maha_pc, eigs=eigs,
                scores={"eucl": eucl_scores, "maha_pc": perclass,
                        "is_unseen": is_unseen})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--emb-dim", type=int, default=16)
    ap.add_argument("--res-pooled", action="store_true",
                    help="pooled residual SIGReg (default: classwise)")
    ap.add_argument("--plots", action="store_true")
    args = ap.parse_args()
    ds = "cifar10"
    cfg = recipe(ds, emb_dim=args.emb_dim)
    ssl_ep = 2 if args.quick else 20
    sup_ep = cfg["ssl_epochs"]
    res_ep = 2 if args.quick else 10
    ft_ep = 1 if args.quick else cfg["ft_epochs"]
    holdouts = {args.holdout}
    seen = [c for c in range(cfg["n_classes"]) if c not in holdouts]
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    train_loader, test_loader = get_cifar_loaders(quick=args.quick,
                                                  limit=args.limit)
    train_eval_loader = DataLoader(train_loader.dataset, batch_size=256,
                                   shuffle=False, num_workers=2)
    cls, plain, _ = _cifar_spec(ds)
    base = cls(DATA_DIR, train=True, download=True, transform=plain)
    print(f"exp29 [{ds}] emb_dim={cfg['emb_dim']} holdout={sorted(holdouts)} "
          f"residual={'pooled' if args.res_pooled else 'classwise'}")

    def cents_of(net):
        e, l = collect_embeddings(net, train_eval_loader)
        m = np.isin(l, seen)
        return exp28.class_centroids(e[m], l[m], seen)

    # ----- networks ---------------------------------------------------------
    print("\n===== training: sup (supervised SIGReg) =====")
    sup, means_sup, _ = supervised_embedding(ds, holdouts=holdouts,
                                             quick=args.quick, limit=args.limit,
                                             seed=args.seed,
                                             emb_dim=cfg["emb_dim"])
    means_sup = means_sup.detach()

    print("\n===== training: res (residual-SSL on augmentations, post sup) =====")
    torch.manual_seed(args.seed + 1); np.random.seed(args.seed + 1)
    res = copy.deepcopy(sup)
    res_loader = (cifar_two_view_loader(quick=args.quick, labeled=True,
                                        holdout=holdouts, limit=args.limit)
                  if args.res_pooled else
                  cifar_two_view_balanced_loader(ds, holdout=holdouts,
                                                 quick=args.quick,
                                                 limit=args.limit))
    train_sigreg_residual_ssl(res, res_loader, res_ep, means_sup,
                              n_slices=cfg["n_slices"],
                              classwise=not args.res_pooled)

    print("\n===== training: ssl (SIGReg on augmentations) =====")
    torch.manual_seed(args.seed + 2); np.random.seed(args.seed + 2)
    trunk = CIFARResNetBackbone(cfg["emb_dim"], arch=cfg["arch"],
                                pretrain=ds).to(DEVICE)
    train_sigreg_ssl(trunk, cifar_two_view_loader(quick=args.quick,
                                                  labeled=False,
                                                  holdout=holdouts,
                                                  limit=args.limit), ssl_ep)
    ssl_cents = cents_of(trunk)

    print("\n===== training: supres (supervised SIGReg post ssl) =====")
    torch.manual_seed(args.seed + 3); np.random.seed(args.seed + 3)
    supres = copy.deepcopy(trunk)
    means_supres = exp28.fill_means(ssl_cents, seen, cfg).clone()
    train_sigreg_hybrid(supres, cifar_balanced_loader(ds, holdout=holdouts,
                                                      quick=args.quick,
                                                      limit=args.limit),
                        sup_ep, means_supres, mode="repulse", disc="proto",
                        alpha=1.0, rep_weight=cfg["rep_weight"],
                        sigreg_weight=cfg["sigreg_weight"],
                        n_slices=cfg["n_slices"])
    means_supres = means_supres.detach()

    print("\n===== training: joint (supervised SIGReg + augmentations in parallel) =====")
    torch.manual_seed(args.seed + 4); np.random.seed(args.seed + 4)
    joint = CIFARResNetBackbone(cfg["emb_dim"], arch=cfg["arch"],
                                pretrain=ds).to(DEVICE)
    means_joint = make_anchors(cfg["pair_dist"] / math.sqrt(2.0),
                               emb_dim=cfg["emb_dim"],
                               n_classes=cfg["n_classes"]).clone()
    train_sigreg_hybrid_aug(joint,
                            cifar_two_view_balanced_loader(ds, holdout=holdouts,
                                                           quick=args.quick,
                                                           limit=args.limit),
                            sup_ep, means_joint,
                            rep_weight=cfg["rep_weight"],
                            sigreg_weight=cfg["sigreg_weight"],
                            n_slices=cfg["n_slices"])
    means_joint = means_joint.detach()

    print("\n===== training: supcon (supervised SimCLR) =====")
    torch.manual_seed(args.seed + 5); np.random.seed(args.seed + 5)
    supcon = CIFARResNetBackbone(cfg["emb_dim"], arch=cfg["arch"],
                                 pretrain=ds).to(DEVICE)
    train_supcon(supcon, cifar_two_view_loader(quick=args.quick, labeled=True,
                                               holdout=holdouts,
                                               limit=args.limit), sup_ep)
    supcon_cents = cents_of(supcon)

    print("\n===== training: simclr (unsupervised SimCLR on augmentations) =====")
    torch.manual_seed(args.seed + 6); np.random.seed(args.seed + 6)
    simclr = CIFARResNetBackbone(cfg["emb_dim"], arch=cfg["arch"],
                                 pretrain=ds).to(DEVICE)
    train_simclr(simclr, cifar_two_view_loader(quick=args.quick, labeled=False,
                                               holdout=holdouts,
                                               limit=args.limit), ssl_ep)
    simclr_cents = cents_of(simclr)
    res_cents = cents_of(res)

    # ----- evaluation spaces ------------------------------------------------
    # concat arms: (sup part net, means/anchors) + (frozen aug part net, cents)
    def embs(net, loader):
        return collect_embeddings(net, loader)

    sup_tr, tr_lab = embs(sup, train_eval_loader)
    sup_te, te_lab = embs(sup, test_loader)
    res_tr, _ = embs(res, train_eval_loader); res_te, _ = embs(res, test_loader)
    ssl_tr, _ = embs(trunk, train_eval_loader); ssl_te, _ = embs(trunk, test_loader)
    supres_tr, _ = embs(supres, train_eval_loader)
    supres_te, _ = embs(supres, test_loader)
    joint_tr, _ = embs(joint, train_eval_loader)
    joint_te, _ = embs(joint, test_loader)
    supcon_tr, _ = embs(supcon, train_eval_loader)
    supcon_te, _ = embs(supcon, test_loader)
    simclr_tr, _ = embs(simclr, train_eval_loader)
    simclr_te, _ = embs(simclr, test_loader)

    cat = lambda a, b: np.concatenate([a, b], axis=1)
    spaces = {
        "sup->res": (cat(sup_tr, res_tr), cat(sup_te, res_te),
                     torch.cat([means_sup[seen], res_cents], dim=1)),
        "ssl->supres": (cat(supres_tr, ssl_tr), cat(supres_te, ssl_te),
                        torch.cat([means_supres[seen], ssl_cents], dim=1)),
        "joint": (joint_tr, joint_te, means_joint[seen]),
        "sup": (sup_tr, sup_te, means_sup[seen]),
        "supcon": (supcon_tr, supcon_te, supcon_cents),
        "supcon+simclr": (cat(supcon_tr, simclr_tr), cat(supcon_te, simclr_te),
                          torch.cat([supcon_cents, simclr_cents], dim=1)),
    }

    print("\n===== performance / novelty table =====")
    print(f"  {'space':<16}{'acc':>8}{'supAUC':>8}{'eucl':>8}"
          f"{'mahaT':>8}{'mahaPC':>8}{'tied eig min/med/max':>24}")
    perf = {}
    for name, (tr, te, anc) in spaces.items():
        r = evaluate_space(tr, tr_lab, te, te_lab, anc, seen, holdouts)
        perf[name] = r
        e = r["eigs"]
        print(f"  {name:<16}{r['acc']:>8.4f}{r['sup_auc']:>8.4f}"
              f"{r['eucl']:>8.4f}{r['maha_tied']:>8.4f}{r['maha_pc']:>8.4f}"
              f"   {e[0]:>6.3f}/{e[1]:>6.3f}/{e[2]:>6.3f}")

    print("\n===== gaussianity (seen classes, test set) =====")
    gauss = {n: gaussianity_summary(spaces[n][1], te_lab, seen, seed=args.seed)
             for n in spaces}
    exp28.print_gauss_table(gauss)

    if args.plots:
        dim = cfg["emb_dim"]
        plot_latent_panels({n: (spaces[n][1], te_lab) for n in spaces},
                           holdouts, CIFAR_NAMES,
                           plot_path(f"latent_cifar10_exp29_{dim}d.png"),
                           title="exp29 [cifar10]: " + " / ".join(spaces))

        # all novelty ROC curves on one plot (validated reference palette)
        colors = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7",
                  "#e34948"]
        plt.figure(figsize=(7.5, 7))
        for (name, r), c in zip(perf.items(), colors):
            s = r["scores"]
            fpr, tpr, _ = roc_curve(s["is_unseen"], s["maha_pc"])
            plt.plot(fpr, tpr, color=c, lw=2,
                     label=f"{name} maha-PC ({r['maha_pc']:.3f})")
            fpr, tpr, _ = roc_curve(s["is_unseen"], s["eucl"])
            plt.plot(fpr, tpr, color=c, lw=1.2, ls="--", alpha=0.7,
                     label=f"{name} eucl ({r['eucl']:.3f})")
        plt.plot([0, 1], [0, 1], color="gray", lw=1, ls=":")
        plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
        plt.title(f"exp29 novelty ROC (holdout {sorted(holdouts)}, "
                  f"{dim}d): solid = Mahalanobis per-class, dashed = Euclidean")
        plt.legend(loc="lower right", fontsize=8)
        plt.grid(alpha=0.25); plt.tight_layout()
        plt.savefig(plot_path(f"exp29_novelty_roc_{dim}d.png"), dpi=150)
        plt.close()
        print(f"  saved {plot_path(f'exp29_novelty_roc_{dim}d.png')}")

        # corner plots: 16d spaces -> first 6 dims; concat -> 3 from each half
        for name, (tr, te, anc) in spaces.items():
            if te.shape[1] > dim:
                sl = np.concatenate([te[:, :3], te[:, dim:dim + 3]], axis=1)
                note = "dims 0-2 (sup) + 0-2 (aug)"
            else:
                sl = te[:, :6]
                note = f"first 6 of {te.shape[1]} dims"
            tag = name.replace("->", "_").replace("+", "_")
            plot_corner(sl, te_lab,
                        plot_path(f"corner_cifar10_exp29_{dim}d_{tag}.png"),
                        title=f"exp29 {name} ({note})")

        # embeddings archive so plots can be regenerated without retraining
        os.makedirs(os.path.join("logs", "exp29"), exist_ok=True)
        np.savez_compressed(
            os.path.join("logs", "exp29", f"embs_{dim}d.npz"),
            te_lab=te_lab,
            **{f"{n}_te": spaces[n][1].astype(np.float16) for n in spaces},
            **{f"{n}_anchors": spaces[n][2].detach().cpu().numpy()
               for n in spaces})
        print(f"  saved logs/exp29/embs_{dim}d.npz")

    # ----- discovery clustering --------------------------------------------
    disc_kw = dict(base_ds=base, train_eval_loader=train_eval_loader,
                   test_loader=test_loader, seen=seen, holdouts=holdouts,
                   dataset_name=ds, rep_weight=cfg["rep_weight"],
                   sigreg_weight=cfg["sigreg_weight"], n_slices=cfg["n_slices"],
                   rounds=args.rounds, ft_epochs=ft_ep, names=CIFAR_NAMES,
                   seed=args.seed)
    hist = {}
    print("\n----- discovery: sup->res (ft sup branch) -----")
    hist["sup->res"] = exp28.run_concat_discovery(
        copy.deepcopy(sup), res, means_sup.clone(), res_cents, base=base,
        dim=cfg["emb_dim"], train_eval_loader=train_eval_loader,
        test_loader=test_loader, seen=seen, holdouts=holdouts, cfg=cfg,
        rounds=args.rounds, ft_epochs=ft_ep, names=CIFAR_NAMES, seed=args.seed)
    print("\n----- discovery: ssl->supres (ft supres branch) -----")
    hist["ssl->supres"] = exp28.run_concat_discovery(
        copy.deepcopy(supres), trunk, means_supres.clone(), ssl_cents,
        base=base, dim=cfg["emb_dim"], train_eval_loader=train_eval_loader,
        test_loader=test_loader, seen=seen, holdouts=holdouts, cfg=cfg,
        rounds=args.rounds, ft_epochs=ft_ep, names=CIFAR_NAMES, seed=args.seed)
    print("\n----- discovery: joint -----")
    _, hist["joint"] = run_discovery(copy.deepcopy(joint), means_joint.clone(),
                                     **disc_kw)
    print("\n----- discovery: sup -----")
    _, hist["sup"] = run_discovery(copy.deepcopy(sup), means_sup.clone(),
                                   **disc_kw)
    print("\n----- discovery: supcon -----")
    _, hist["supcon"] = run_discovery(
        copy.deepcopy(supcon), exp28.fill_means(supcon_cents, seen, cfg),
        **disc_kw)
    print("\n----- discovery: supcon+simclr (ft supcon branch) -----")
    hist["supcon+simclr"] = exp28.run_concat_discovery(
        copy.deepcopy(supcon), simclr,
        exp28.fill_means(supcon_cents, seen, cfg), simclr_cents, base=base,
        dim=cfg["emb_dim"], train_eval_loader=train_eval_loader,
        test_loader=test_loader, seen=seen, holdouts=holdouts, cfg=cfg,
        rounds=args.rounds, ft_epochs=ft_ep, names=CIFAR_NAMES, seed=args.seed)

    print(f"\n===== EXP29 SUMMARY [cifar10, {cfg['emb_dim']}d] =====")
    for name in spaces:
        r = perf[name]
        print(f"  [{name:<14}] acc={r['acc']:.4f} supAUC={r['sup_auc']:.4f} "
              f"eucl={r['eucl']:.4f} mahaT={r['maha_tied']:.4f} "
              f"mahaPC={r['maha_pc']:.4f}")
        for h in hist[name]:
            print(f"          round {h['round']}: purity={h['purity']:.3f} "
                  f"anchors={h['n_anchors']}  margin={h['margin']:.4f}  "
                  f"mean-anchor={h['mean_pc']:.4f}")
    print("Done.")


if __name__ == "__main__":
    main()
