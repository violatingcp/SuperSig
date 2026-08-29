"""
Experiment 144: our constructions under the Generalized Category Discovery
(GCD) benchmark protocol, so they can be compared with GCD / UNO+ / SimGCD /
RLCD on the numbers those papers report.

PROTOCOL (Vaze et al. 2022; SimGCD Table 1; RLCD arXiv:2506.02334).
  * backbone   DINO ViT-B/16 [CLS] features, 768-D
  * split      known ("Old") classes = the first N class indices; 50% of the
               known-class TRAIN images are labelled (D^l); every other train
               image -- the rest of the known classes and all novel classes --
               is unlabelled (D^u).  N: cifar10 5/10, cifar100 80/100,
               cars 98/196, aircraft 50/100.
  * metric     clustering accuracy on D^u: one Hungarian assignment between the
               K predicted clusters and the K ground-truth classes (K assumed
               known), reported as All / Old / New (v2 of the GCD metric).
  * reference  (All/Old/New, DINO ViT-B/16):
               k-means on raw DINO  cifar10 83.6/85.7/82.5  cifar100 52.0/52.2/50.8  cars 12.8/10.6/13.8  aircraft 16.0/14.4/16.8
               GCD                  91.5/97.9/88.2          73.0/76.2/66.5          39.0/57.6/29.9       45.0/41.1/46.9
               UNO+                 68.6/98.3/53.8          69.5/80.6/47.2          35.5/70.5/18.6       40.3/56.4/32.2
               SimGCD               97.1/95.1/98.1          80.1/81.2/77.8          53.8/71.9/45.0       54.2/59.1/51.8
               RLCD                 97.4/96.4/97.9          83.4/84.2/81.9          64.9/79.3/58.0       60.6/62.2/59.8
               (GCD/UNO+/SimGCD/RLCD fine-tune the last ViT block on D^l + D^u
               for 200 epochs; the k-means row is the frozen trunk.)

WHAT WE RUN, on the frozen trunk (no fine-tune -- the rows are directly
comparable with the k-means row and are a LOWER bound for our fine-tuned
variants):
  kmeans        k-means with K on D^u                      (replicates the reference row)
  ss-kmeans     semi-supervised k-means: K centroids, the N known ones fixed to
                the D^l class centroids at init, labelled points pinned         (GCD's clustering step)
  np-anchors    OUR construction: seen centroids from D^l; density-ratio pool on
                D^u vs D^l (NP critic, exp 92/135); label-free cut (exp 129) or
                the 0.95 quantile; BIC k-means on the pool with kmax = K-N;
                assign D^u to the nearest anchor (seen + discovered).  The
                number of clusters is whatever BIC found, so ACC is computed
                with a rectangular Hungarian assignment (missing clusters score
                zero) -- this row is honest about not knowing K.
  np-seeded     the same anchors used to SEED semi-supervised k-means with K
                (pad with k-means++ on the pool if BIC found fewer) -- the
                K-known variant, directly comparable with ss-kmeans.
  and the same four on a 100-D FeatureHead trained on D^l only with SupCon
  (`head-supcon`) and SupCon+SIGReg lam=5 (`head-ssig`, the discovery workhorse).

Three split seeds; mean +- sd.  Output: logs/exp144/gcd_{ds}.json and a table.

    python experiments/144_gcd_benchmark.py --selftest
    python experiments/144_gcd_benchmark.py --dataset cars
    python experiments/144_gcd_benchmark.py --dataset cifar10 --seeds 0
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import importlib
import json

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, TensorDataset

from supersig import poolcut
from supersig.config import DATA_DIR, DEVICE
from supersig.discovery import bic_select, np_pool_scores
from supersig.losses import sigreg_loss, supcon_loss

KNOWN = {"cifar10": 5, "cifar100": 80, "cars": 98, "aircraft": 50}
TOTAL = {"cifar10": 10, "cifar100": 100, "cars": 196, "aircraft": 100}
REF = {  # All / Old / New, DINO ViT-B/16, from the papers
    "cifar10": {"k-means (paper)": (83.6, 85.7, 82.5), "GCD": (91.5, 97.9, 88.2), "UNO+": (68.6, 98.3, 53.8),
                "SimGCD": (97.1, 95.1, 98.1), "RLCD": (97.4, 96.4, 97.9)},
    "cifar100": {"k-means (paper)": (52.0, 52.2, 50.8), "GCD": (73.0, 76.2, 66.5), "UNO+": (69.5, 80.6, 47.2),
                 "SimGCD": (80.1, 81.2, 77.8), "RLCD": (83.4, 84.2, 81.9)},
    "cars": {"k-means (paper)": (12.8, 10.6, 13.8), "GCD": (39.0, 57.6, 29.9), "UNO+": (35.5, 70.5, 18.6),
             "SimGCD": (53.8, 71.9, 45.0), "RLCD": (64.9, 79.3, 58.0)},
    "aircraft": {"k-means (paper)": (16.0, 14.4, 16.8), "GCD": (45.0, 41.1, 46.9), "UNO+": (40.3, 56.4, 32.2),
                 "SimGCD": (54.2, 59.1, 51.8), "RLCD": (60.6, 62.2, 59.8)},
}


# ------------------------------------------------------------------ metric
def cluster_acc(y_true, y_pred, n_known, K):
    """GCD v2 ACC: one Hungarian over all K classes on D^u; Old/New are the
    subsets under that single assignment.  Rectangular if fewer clusters."""
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    n_pred = int(y_pred.max()) + 1
    W = np.zeros((max(n_pred, K), K), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        W[p, t] += 1
    r, c = linear_sum_assignment(-W)
    m = dict(zip(r, c))
    hit = np.array([m.get(p, -1) == t for t, p in zip(y_true, y_pred)])
    old = y_true < n_known
    return dict(all=float(hit.mean()) * 100, old=float(hit[old].mean()) * 100 if old.any() else float("nan"),
                new=float(hit[~old].mean()) * 100 if (~old).any() else float("nan"), n_clusters=n_pred)


# ------------------------------------------------------------------ split
def gcd_split(y, n_known, frac=0.5, seed=0):
    rng = np.random.default_rng(seed)
    lab = np.zeros(len(y), dtype=bool)
    for c in range(n_known):
        idx = np.where(y == c)[0]
        lab[rng.choice(idx, size=int(round(frac * len(idx))), replace=False)] = True
    return lab                              # True = labelled (in D^l)


# ------------------------------------------------------------- clustering
def kmeans_pred(Z, K, seed):
    return KMeans(n_clusters=K, n_init=10, random_state=seed).fit_predict(Z)


def ss_kmeans(Z, y, lab, n_known, K, init_extra=None, iters=50, seed=0):
    """Semi-supervised k-means (GCD): labelled points pinned to their class
    centroid; K centroids, the first n_known initialised from D^l; the rest
    from `init_extra` (our anchors) padded by k-means++ on the unlabelled set."""
    Zt = torch.as_tensor(Z, dtype=torch.float32, device=DEVICE)
    yt = torch.as_tensor(y, device=DEVICE); labt = torch.as_tensor(lab, device=DEVICE)
    C = torch.zeros(K, Z.shape[1], device=DEVICE)
    for c in range(n_known):
        C[c] = Zt[labt & (yt == c)].mean(0)
    extra = [] if init_extra is None else [torch.as_tensor(e, dtype=torch.float32, device=DEVICE) for e in init_extra]
    extra = extra[:K - n_known]
    g = torch.Generator().manual_seed(seed)
    U = Zt[~labt]
    while len(extra) < K - n_known:                      # k-means++ padding
        cur = torch.cat([C[:n_known]] + ([torch.stack(extra)] if extra else []))
        d2 = torch.cdist(U, cur).min(1).values ** 2
        extra.append(U[int(torch.multinomial((d2 / d2.sum()).cpu(), 1, generator=g))].clone())
    C[n_known:] = torch.stack(extra)
    for _ in range(iters):
        a = torch.cdist(Zt, C).argmin(1)
        a[labt] = yt[labt]                                  # pin labelled points
        for k in range(K):
            if (a == k).any():
                C[k] = Zt[a == k].mean(0)
    a = torch.cdist(Zt, C).argmin(1); a[labt] = yt[labt]
    return a.cpu().numpy(), C


def np_anchors(Z, y, lab, n_known, K, seed, cut="legal"):
    """Our zero-training construction: density-ratio pool of the corpus vs the
    labelled reference, cut, BIC k-means -> discovered anchors."""
    Zt = torch.as_tensor(Z, dtype=torch.float32, device=DEVICE)
    is_ref = np.asarray(lab)
    f = np_pool_scores(Zt, is_ref, seed=seed).cpu().numpy()
    if cut == "legal":
        mask, info = poolcut.legal_pool(f, is_ref, n_min=10)
        kmax = max(2, min(K - n_known, info.get("kmax", K - n_known)))
    else:
        tau = np.quantile(f[is_ref], 0.95); mask = f > tau; info = dict(ok=True, reason="quantile")
        kmax = K - n_known
    mask = mask & ~is_ref                                   # never pool labelled points
    pool_is_novel = (y[mask] >= n_known)
    purity = float(pool_is_novel.mean()) if mask.any() else 0.0
    khat, centers, _ = bic_select(Zt[torch.as_tensor(mask, device=DEVICE)], kmax=int(kmax), seed=seed)
    return centers.detach().cpu().numpy(), dict(purity=purity, pool=int(mask.sum()), khat=int(khat),
                                                 ok=bool(info.get("ok", True)), q=float(mask.mean()))


def assign_to_anchors(Z, y, lab, n_known, anchors):
    Zt = torch.as_tensor(Z, dtype=torch.float32, device=DEVICE)
    labt = torch.as_tensor(lab, device=DEVICE); yt = torch.as_tensor(y, device=DEVICE)
    seen = torch.stack([Zt[labt & (yt == c)].mean(0) for c in range(n_known)])
    A = torch.cat([seen, torch.as_tensor(anchors, dtype=torch.float32, device=DEVICE)])
    return torch.cdist(Zt, A).argmin(1).cpu().numpy()


# ------------------------------------------------------------------ heads
class FeatureHead(torch.nn.Module):
    def __init__(self, emb_dim=100, feat_dim=768):
        super().__init__()
        self.head = torch.nn.Sequential(torch.nn.Linear(feat_dim, 256), torch.nn.ReLU(), torch.nn.Linear(256, emb_dim))

    def forward(self, x):
        return self.head(x)


def train_head(Xl, yl, lam, epochs=30, seed=0, emb_dim=100):
    """SupCon (lam=0) or SupCon+SIGReg (lam=5) head on the LABELLED features
    only -- the exp-70 objectives on a frozen trunk."""
    torch.manual_seed(seed)
    head = FeatureHead(emb_dim, Xl.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(torch.as_tensor(Xl, dtype=torch.float32), torch.as_tensor(yl)),
                        batch_size=512, shuffle=True, drop_last=True)
    for _ in range(epochs):
        for x, yy in loader:
            z = head(x.to(DEVICE))
            loss = supcon_loss(F.normalize(z, dim=1), yy.to(DEVICE), temp=0.1)
            if lam:
                loss = loss + lam * sigreg_loss(z)
            opt.zero_grad(); loss.backward(); opt.step()
    head.eval()
    with torch.no_grad():
        return head


@torch.no_grad()
def embed(head, X):
    return torch.cat([head(torch.as_tensor(X[i:i + 4096], dtype=torch.float32, device=DEVICE)).cpu()
                      for i in range(0, len(X), 4096)]).numpy()


# ------------------------------------------------------------------ driver
def run_space(name, Z, y, lab, n_known, K, seed, out):
    u = ~lab
    rows = {}
    p = kmeans_pred(Z[u], K, seed); rows["kmeans"] = cluster_acc(y[u], p, n_known, K)
    p, _ = ss_kmeans(Z, y, lab, n_known, K, seed=seed); rows["ss-kmeans"] = cluster_acc(y[u], p[u], n_known, K)
    anchors, info = np_anchors(Z, y, lab, n_known, K, seed)
    p = assign_to_anchors(Z, y, lab, n_known, anchors); rows["np-anchors"] = cluster_acc(y[u], p[u], n_known, K)
    rows["np-anchors"].update(info)
    p, _ = ss_kmeans(Z, y, lab, n_known, K, init_extra=list(anchors), seed=seed)
    rows["np-seeded"] = cluster_acc(y[u], p[u], n_known, K)
    for m, r in rows.items():
        print(f"    [{name:12s} {m:11s}] All {r['all']:5.1f}  Old {r['old']:5.1f}  New {r['new']:5.1f}"
              + (f"   (pool purity {r['purity']:.3f}, k_hat {r['khat']}, ok={r['ok']})" if m == "np-anchors" else ""),
              flush=True)
    out[name] = rows


def _selftest():
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(6), 50); Z = rng.normal(0, 1, (300, 8)) + 6 * rng.normal(0, 1, (6, 8))[y]
    lab = gcd_split(y, 3, 0.5, 0)
    assert lab.sum() == 75 and not lab[y >= 3].any()
    print("  split: 50% of the first 3 classes labelled, novel never labelled   OK")
    perm = rng.permutation(6); r = cluster_acc(y, perm[y], 3, 6)
    assert r["all"] == 100.0 and r["old"] == 100.0 and r["new"] == 100.0
    print("  ACC invariant to cluster relabelling (Hungarian)                   OK")
    r = cluster_acc(y, np.minimum(y, 3), 3, 6)          # 4 clusters for 6 classes
    assert r["n_clusters"] == 4 and r["old"] == 100.0 and 0 < r["new"] < 100
    print("  rectangular assignment: missing clusters score zero                OK")
    p, _ = ss_kmeans(Z, y, lab, 3, 6, seed=0)
    r = cluster_acc(y[~lab], p[~lab], 3, 6); assert r["all"] > 95, r
    print(f"  ss-kmeans recovers a separable synthetic: All {r['all']:.1f}        OK")
    print("\nselftest OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dataset", default="cars", choices=list(KNOWN))
    ap.add_argument("--base", default="dino")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--head-epochs", type=int, default=30)
    ap.add_argument("--skip-heads", action="store_true")
    ap.add_argument("--out", default="logs/exp144")
    args = ap.parse_args()
    if args.selftest:
        _selftest(); return
    ds = args.dataset; n_known, K = KNOWN[ds], TOTAL[ds]
    bank = torch.load(os.path.join(DATA_DIR, f"tf_feats_{ds}_{args.base}_vitb16.pt"))
    X, y = bank["train"]; X = X.float().numpy(); y = y.numpy()
    assert y.max() + 1 == K, (y.max(), K)
    print(f"exp144 [{ds}/{args.base}] GCD protocol: known {n_known}/{K}, train {len(y)}", flush=True)
    results = {}
    for seed in [int(s) for s in args.seeds.split(",")]:
        lab = gcd_split(y, n_known, 0.5, seed)
        print(f"\n  seed {seed}: |D^l|={int(lab.sum())} |D^u|={int((~lab).sum())}", flush=True)
        out = {}
        run_space("raw-dino", X, y, lab, n_known, K, seed, out)
        if not args.skip_heads:
            for name, lam in (("head-supcon", 0.0), ("head-ssig", 5.0)):
                head = train_head(X[lab], y[lab], lam, epochs=args.head_epochs, seed=seed)
                run_space(name, embed(head, X), y, lab, n_known, K, seed, out)
        results[seed] = out
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"gcd_{ds}_{args.base}.json"), "w") as fh:
        json.dump(dict(dataset=ds, base=args.base, n_known=n_known, K=K, results=results, reference=REF[ds]),
                  fh, indent=1, default=float)
    print(f"\n===== EXP144 [{ds}] clustering ACC on D^u, All / Old / New (mean+-sd over {len(results)} seeds) =====")
    for name in list(next(iter(results.values())).keys()):
        for m in ("kmeans", "ss-kmeans", "np-anchors", "np-seeded"):
            v = np.array([[results[s][name][m][k] for k in ("all", "old", "new")] for s in results])
            print(f"  {name:12s} {m:11s} " + "  ".join(f"{v[:, i].mean():5.1f}+-{v[:, i].std():4.1f}" for i in range(3)))
    for k, v in REF[ds].items():
        print(f"  {'reference':12s} {k:16s} {v[0]:5.1f}      {v[1]:5.1f}      {v[2]:5.1f}")
    print("EXP144 DONE.")


if __name__ == "__main__":
    main()
