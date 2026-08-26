"""
Experiment 77: similarity metrics BETWEEN the learned spaces (the
LLM-geometry program of arXiv:2503.21073 ported to our image embeddings,
where every space embeds the same images so all points are paired).

Pairwise, on n aligned sampled train images per cell:
  cka        linear CKA (global Gram-matrix agreement, dim-agnostic)
  knn        mutual kNN@k Jaccard overlap (local, Platonic-style)
  lle        LLE-weight transfer: reconstruction error in B using A's
             local weights, normalized by uniform-weight baseline (<1 =
             A's local geometry is valid in B)
  proc       orthogonal Procrustes residual ||XR-Y||^2/||Y||^2 (equal dim)
  r2         ridge map A->B fraction of variance explained (directional)
  cal        calibration transfer: eucl novelty AUC in B using A's test
             points pushed through the train-fit ridge map (vs native B)

Per space: TwoNN global intrinsic dimension; Levina-Bickel LID (k=20) of
test points against seen-class train references, mean seen vs holdout and
the AUC of LID as a novelty score (a new battery candidate).

Modes:
  transfer cell   --dataset cars --base visreg     (cached banks + heads)
  cross-base      --dataset cars --cross-base      (same arms across bases)
  cifar           --dataset cifar100 --dim 100     (exp-76 ckpts required)

    python experiments/77_space_similarity.py --dataset cars --base visreg
    python experiments/77_space_similarity.py --dataset cars --cross-base
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from supersig.holdouts import holdout_set, n_holdout, run_tag
import argparse
import importlib
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import orthogonal_procrustes

from supersig.config import DATA_DIR, DEVICE, REPO_DIR, plot_path
from supersig.data import get_cifar_loaders
from supersig.models import CIFARResNetBackbone
from supersig.recipes import recipe
from supersig.train import collect_embeddings

exp37 = importlib.import_module("37_dtd_vit")
exp43 = importlib.import_module("43_dtd_finetune")
exp44 = importlib.import_module("44_transfer_32d")

CKPT_DIR = os.path.join(REPO_DIR, "checkpoints")
ARMS_70 = ["simclr-ft", "sigreg-ssl-ft", "nplm-bil-ft",
           "supcon-ft", "ss-ft", "nplm-sup-ft"]
CHILDREN_71 = [("supcon-ft", "res"), ("ss-ft", "res"),
               ("supcon-ft", "resnplm")]


# ---------------------------------------------------------------- loading

def head_emb(DS, BASE, arm, emb_dim, split):
    bp = os.path.join(DATA_DIR, f"tf_feats_{DS}_{BASE}_ft70_{arm}.pt")
    cp = os.path.join(CKPT_DIR, f"{DS}_ft_{BASE}_{arm}_seen.pt")
    if not (os.path.exists(bp) and os.path.exists(cp)):
        return None
    X, y = torch.load(bp)[split]
    mod = exp43.FineTuneModel(BASE, emb_dim)
    mod.load_state_dict(torch.load(cp, map_location=DEVICE))
    H = exp37.embed(mod.head.float(), X.float()).numpy()
    del mod
    return H, y.numpy()

def transfer_cell(args, DS, BASE, arms=None, prefix=""):
    """dict name -> (Xtr, ytr, Xte, yte)."""
    out = {}
    fz = os.path.join(DATA_DIR, f"tf_feats_{DS}_{BASE}_vitb16.pt")
    if os.path.exists(fz) and not arms:
        b = torch.load(fz)
        out[f"{prefix}frozen"] = (b["train"][0].float().numpy(),
                                  b["train"][1].numpy(),
                                  b["test"][0].float().numpy(),
                                  b["test"][1].numpy())
    for arm in (arms or ARMS_70):
        r_tr = head_emb(DS, BASE, arm, args.emb_dim, "train")
        r_te = head_emb(DS, BASE, arm, args.emb_dim, "test")
        if r_tr and r_te:
            out[f"{prefix}{arm}"] = (*r_tr, *r_te)
    if not arms:
        for parent, obj in CHILDREN_71:
            r_tr = head_emb(DS, BASE, f"{parent}_{obj}", args.emb_dim,
                            "train")
            r_te = head_emb(DS, BASE, f"{parent}_{obj}", args.emb_dim,
                            "test")
            pk = f"{prefix}{parent}"
            if r_tr and r_te and pk in out:
                out[f"{prefix}{parent}-{obj}"] = (*r_tr, *r_te)
                pXtr, _, pXte, _ = out[pk]
                out[f"{prefix}{parent}-{obj}-cat"] = (
                    np.concatenate([pXtr, r_tr[0]], 1), r_tr[1],
                    np.concatenate([pXte, r_te[0]], 1), r_te[1])
    return out

def cifar_cell(args, ds):
    cfg = recipe(ds, emb_dim=args.dim)
    train_loader, test_loader = get_cifar_loaders(quick=args.quick,
                                                  dataset=ds)
    ev = DataLoader(train_loader.dataset, batch_size=256, shuffle=False,
                    num_workers=2)
    out = {}
    names = ["supcon", "res", "resnplm"] + list(args.arms)
    for name in names:
        ck = os.path.join(CKPT_DIR, f"exp76_{ds}_{args.dim}d_{name}.pt")
        if not os.path.exists(ck):
            print(f"  !! missing {ck}, skipping {name}")
            continue
        net = CIFARResNetBackbone(args.dim, arch=cfg["arch"],
                                  pretrain=ds).to(DEVICE)
        net.load_state_dict(torch.load(ck, map_location=DEVICE))
        Xtr, ytr = collect_embeddings(net, ev)
        Xte, yte = collect_embeddings(net, test_loader)
        del net
        torch.cuda.empty_cache()
        out[name] = (Xtr, ytr, Xte, yte)
    for obj in ("res", "resnplm"):
        if obj in out and "supcon" in out:
            p, c = out["supcon"], out[obj]
            out[f"{obj}-cat"] = (np.concatenate([p[0], c[0]], 1), p[1],
                                 np.concatenate([p[2], c[2]], 1), p[3])
    return out


# ---------------------------------------------------------------- metrics

def cka(X, Y):
    Xc = X - X.mean(0); Yc = Y - Y.mean(0)
    num = np.linalg.norm(Xc.T @ Yc) ** 2
    den = np.linalg.norm(Xc.T @ Xc) * np.linalg.norm(Yc.T @ Yc)
    return float(num / max(den, 1e-30))

def knn_idx(X, k):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    S = Xn @ Xn.T
    np.fill_diagonal(S, -np.inf)
    return np.argsort(-S, axis=1)[:, :k]

def mutual_knn(iA, iB):
    ja = [len(set(a) & set(b)) / len(set(a) | set(b))
          for a, b in zip(iA, iB)]
    return float(np.mean(ja))

def lle_transfer(X, Y, iA, reg=1e-3):
    """Mean ||y - Wy_nn|| with A's LLE weights / uniform-weight baseline."""
    errs, base = [], []
    for i in range(len(X)):
        nn = iA[i]
        Z = X[nn] - X[i]
        G = Z @ Z.T
        G += np.eye(len(nn)) * reg * max(np.trace(G), 1e-12)
        w = np.linalg.solve(G, np.ones(len(nn)))
        w /= w.sum()
        errs.append(np.linalg.norm(Y[i] - w @ Y[nn]))
        base.append(np.linalg.norm(Y[i] - Y[nn].mean(0)))
    return float(np.mean(errs) / max(np.mean(base), 1e-12))

def procrustes_resid(X, Y):
    if X.shape[1] != Y.shape[1]:
        return float("nan")
    Xc = (X - X.mean(0)); Yc = (Y - Y.mean(0))
    Xc /= np.linalg.norm(Xc); Yc /= np.linalg.norm(Yc)
    R, s = orthogonal_procrustes(Xc, Yc)
    return float(np.linalg.norm(Xc @ R * s - Yc) ** 2)

def ridge_map(X, Y, lam=1e-3):
    Xc = X - X.mean(0); Yc = Y - Y.mean(0)
    G = Xc.T @ Xc
    W = np.linalg.solve(G + lam * np.trace(G) / G.shape[0]
                        * np.eye(G.shape[0]), Xc.T @ Yc)
    return W, X.mean(0), Y.mean(0)

def ridge_r2(X, Y, W=None, mx=None, my=None):
    if W is None:
        W, mx, my = ridge_map(X, Y)
    pred = (X - mx) @ W + my
    ss = ((Y - pred) ** 2).sum()
    tot = ((Y - Y.mean(0)) ** 2).sum()
    return float(1.0 - ss / max(tot, 1e-30))

def sqdist(A, B):
    """Pairwise squared euclidean distances without the n x m x d blowup."""
    D = (A * A).sum(1)[:, None] + (B * B).sum(1)[None] - 2.0 * A @ B.T
    return np.maximum(D, 0.0)

def eucl_auc(Xtr, ytr, Xte, yte, seen, holdouts):
    from sklearn.metrics import roc_auc_score
    cents = np.stack([Xtr[ytr == c].mean(0) for c in seen])
    d = np.sqrt(sqdist(Xte, cents)).min(1)
    m = np.isin(yte, list(holdouts)) | np.isin(yte, seen)
    return float(roc_auc_score(np.isin(yte[m], list(holdouts)), d[m]))

def twonn_id(X, rng):
    sub = X[rng.choice(len(X), min(2000, len(X)), replace=False)]
    D = np.sqrt(sqdist(sub, sub))
    np.fill_diagonal(D, np.inf)
    P = np.sort(D, axis=1)
    mu = P[:, 1] / np.maximum(P[:, 0], 1e-12)
    mu = mu[mu > 1]
    return float(len(mu) / np.log(mu).sum())

def lid_scores(Xref, Xq, k=20):
    """Levina-Bickel MLE LID of each query vs reference points."""
    D = np.sqrt(sqdist(Xq, Xref))
    P = np.sort(D, axis=1)[:, :k]
    rk = np.maximum(P[:, -1:], 1e-12)
    m = np.mean(np.log(np.maximum(P[:, :-1], 1e-12) / rk), axis=1)
    # collapsed spaces produce exact distance ties (m -> 0-): cap the LID
    return -1.0 / np.minimum(m, -1e-3)


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cars",
                    choices=["cifar10", "cifar100", "cars", "aircraft",
                             "flowers", "galaxy10", "dtd"])
    ap.add_argument("--base", default="dino",
                    choices=["dino", "lejepa", "visreg"])
    ap.add_argument("--cross-base", action="store_true",
                    help="compare supcon-ft/ss-ft/simclr-ft across bases")
    ap.add_argument("--dim", type=int, default=100)
    ap.add_argument("--emb-dim", type=int, default=100)
    ap.add_argument("--holdout", type=int, default=4)
    ap.add_argument("--arms", nargs="*",
                    default=["simclr", "lejepa", "supcon_sigreg",
                             "nplm_bilinear"])
    ap.add_argument("--n-points", type=int, default=4000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    ds = args.dataset
    cifar = ds.startswith("cifar")
    rng = np.random.default_rng(0)

    if cifar:
        tag = f"{ds}_{args.dim}d" + run_tag()
        spaces = cifar_cell(args, ds)
        n_cls = recipe(ds, emb_dim=args.dim)["n_classes"]
        holdouts = {args.holdout}
    elif args.cross_base:
        tag = f"{ds}_crossbase" + run_tag()
        spaces = {}
        for b in ("dino", "lejepa", "visreg"):
            spaces.update(transfer_cell(
                args, ds, b, arms=["supcon-ft", "ss-ft", "simclr-ft"],
                prefix=f"{b}:"))
        n_cls = 47 if ds == "dtd" else exp44.N_CLASSES[ds]
        nh = n_holdout(ds)
        holdouts = holdout_set(ds, n_cls)
    else:
        tag = f"{ds}_{args.base}" + run_tag()
        spaces = transfer_cell(args, ds, args.base)
        n_cls = 47 if ds == "dtd" else exp44.N_CLASSES[ds]
        nh = n_holdout(ds)
        holdouts = holdout_set(ds, n_cls)
    seen = [c for c in range(n_cls) if c not in holdouts]
    names = list(spaces)
    print(f"exp77 [{tag}] {len(names)} spaces: {names}")
    if len(names) < 2:
        print("nothing to compare"); return

    ytr = spaces[names[0]][1]
    idx = rng.choice(len(ytr), min(args.n_points, len(ytr)),
                     replace=False)
    sub = {k: np.asarray(v[0], dtype=np.float64)[idx]
           for k, v in spaces.items()}
    knn = {k: knn_idx(v, args.k) for k, v in sub.items()}

    # per-space: intrinsic dim + LID novelty
    print(f"\n  {'space':<26}{'TwoNN-ID':>9}{'LID seen':>9}"
          f"{'LID hold':>9}{'LID-AUC':>9}")
    per = {}
    for k, (Xtr_, ytr_, Xte_, yte_) in spaces.items():
        Xtr_ = np.asarray(Xtr_, dtype=np.float64)
        Xte_ = np.asarray(Xte_, dtype=np.float64)
        ref = Xtr_[np.isin(ytr_, seen)]
        ref = ref[rng.choice(len(ref), min(4000, len(ref)),
                             replace=False)]
        qi = rng.choice(len(yte_), min(3000, len(yte_)), replace=False)
        lid = lid_scores(ref, Xte_[qi], k=20)
        hm = np.isin(yte_[qi], list(holdouts))
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(hm, lid)) if hm.any() else float("nan")
        per[k] = dict(twonn=twonn_id(sub[k], rng),
                      lid_seen=float(lid[~hm].mean()),
                      lid_hold=float(lid[hm].mean()) if hm.any() else
                      float("nan"), lid_auc=auc)
        print(f"  {k:<26}{per[k]['twonn']:>9.2f}{per[k]['lid_seen']:>9.2f}"
              f"{per[k]['lid_hold']:>9.2f}{auc:>9.3f}")

    n = len(names)
    M = {m: np.full((n, n), np.nan) for m in
         ("cka", "knn", "lle", "proc", "r2", "cal")}
    for i, a in enumerate(names):
        M["cka"][i, i] = M["knn"][i, i] = M["r2"][i, i] = 1.0
        for j, b in enumerate(names):
            if j <= i:
                continue
            A, B = sub[a], sub[b]
            M["cka"][i, j] = M["cka"][j, i] = cka(A, B)
            M["knn"][i, j] = M["knn"][j, i] = mutual_knn(knn[a], knn[b])
            M["lle"][i, j] = lle_transfer(A, B, knn[a])
            M["lle"][j, i] = lle_transfer(B, A, knn[b])
            M["proc"][i, j] = M["proc"][j, i] = procrustes_resid(A, B)
            M["r2"][i, j] = ridge_r2(A, B)
            M["r2"][j, i] = ridge_r2(B, A)
            # calibration transfer a->b and b->a on full banks
            for (src, dst, ii, jj) in ((a, b, i, j), (b, a, j, i)):
                Xs, ys, Xs_te, _ = spaces[src]
                Xd, yd, Xd_te, yd_te = spaces[dst]
                W, mx, my = ridge_map(
                    np.asarray(Xs, np.float64)[idx],
                    np.asarray(Xd, np.float64)[idx])
                mapped_te = (np.asarray(Xs_te, np.float64) - mx) @ W + my
                M["cal"][ii, jj] = eucl_auc(
                    np.asarray(Xd, np.float64), yd, mapped_te, yd_te,
                    seen, holdouts)
        print(f"  pairwise row done: {a}")

    print(f"\n===== EXP77 SUMMARY [{tag}] (n={len(idx)}, k={args.k}) =====")
    hdr = "  " + f"{'A vs B':<40}" + "".join(
        f"{m:>8}" for m in ("cka", "knn", "lleA>B", "proc", "r2A>B",
                            "calA>B"))
    print(hdr)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if j <= i:
                continue
            print(f"  {a + ' vs ' + b:<40}"
                  f"{M['cka'][i, j]:>8.3f}{M['knn'][i, j]:>8.3f}"
                  f"{M['lle'][i, j]:>8.3f}{M['proc'][i, j]:>8.3f}"
                  f"{M['r2'][i, j]:>8.3f}{M['cal'][i, j]:>8.3f}")

    for m in ("cka", "knn"):
        plt.figure(figsize=(0.55 * n + 3, 0.55 * n + 2.5))
        plt.imshow(M[m], cmap="magma", vmin=0, vmax=1)
        plt.colorbar(label=m)
        plt.xticks(range(n), names, rotation=35, ha="right", fontsize=6)
        plt.yticks(range(n), names, fontsize=6)
        plt.title(f"exp77 {m} ({tag})")
        plt.tight_layout()
        plt.savefig(plot_path(f"exp77_{m}_{tag}.png"), dpi=150)
        plt.close()

    os.makedirs(os.path.join("logs", "exp77"), exist_ok=True)
    np.savez(os.path.join("logs", "exp77", f"results_{tag}.npz"),
             names=np.array(names), idx=idx,
             **{m: M[m] for m in M},
             **{f"per__{k.replace(' ', '_')}__{f}": per[k][f]
                for k in per for f in per[k]})
    print("Done.")


if __name__ == "__main__":
    main()
