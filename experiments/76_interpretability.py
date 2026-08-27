"""
Experiment 76: interpretability of the learned spaces via class-centroid
geometry ("are otters near beavers?").

For every space of a cell we compute train-bank class centroids and report
  (a) top-5 nearest class-centroid tables (cosine), holdouts starred,
  (b) a superclass-ordered cosine-distance heatmap,
  (c) an average-linkage dendrogram over centroids,
  (d) where a superclass partition exists, summary numbers:
      NN superclass agreement@1/@5 (vs chance), dendrogram purity,
      centroid silhouette, within/between superclass distance ratio.

Superclass ground truth: cifar100 coarse labels (pickle meta), cifar10
animal/vehicle, aircraft manufacturer (paired annotation levels),
cars make (first token of the class name, when names resolve),
galaxy10 coarse morphology; flowers/dtd have none (tables + dendrogram
only).

Transfer mode (cars/aircraft/flowers/galaxy10/dtd x dino/lejepa/visreg)
reads the cached exp-70/71 trunk banks + saved heads -- no training.
CIFAR mode reproduces the exp-73 trio (supcon parent -> res / res-nplm
children + concats) and optional exp-50 arms, saving reusable ckpts
checkpoints/exp76_{ds}_{dim}d_{name}.pt.

    python experiments/76_interpretability.py --dataset cars --base visreg
    python experiments/76_interpretability.py --dataset cifar100 --dim 100 \
        --arms simclr lejepa supcon_sigreg nplm_bilinear
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from supersig.holdouts import holdout_set, n_holdout, run_tag
import argparse
import copy
import importlib
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy

from supersig.config import DATA_DIR, DEVICE, REPO_DIR, plot_path
from supersig.data import get_cifar_loaders, cifar_two_view_loader
from supersig.models import CIFARResNetBackbone
from supersig.recipes import recipe
from supersig.train import collect_embeddings

exp37 = importlib.import_module("37_dtd_vit")
exp43 = importlib.import_module("43_dtd_finetune")
exp44 = importlib.import_module("44_transfer_32d")
exp55 = importlib.import_module("55_nplm_discovery")
exp73 = importlib.import_module("73_cifar_residual_ft")

CKPT_DIR = os.path.join(REPO_DIR, "checkpoints")

ARMS_70 = ["simclr-ft", "sigreg-ssl-ft", "nplm-bil-ft",
           "supcon-ft", "ss-ft", "nplm-sup-ft"]
CHILDREN_71 = [("supcon-ft", "res"), ("ss-ft", "res"),
               ("supcon-ft", "resnplm")]
GALAXY10_NAMES = ["disturbed", "merging", "round smooth",
                  "in-between smooth", "cigar smooth", "barred spiral",
                  "unbarred tight spiral", "unbarred loose spiral",
                  "edge-on no bulge", "edge-on bulge"]
GALAXY10_SUP = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3]  # disturbed/smooth/spiral/edge
CIFAR10_SUP = [1, 0, 1, 1, 1, 1, 1, 1, 0, 0]   # 0 vehicle, 1 animal


def class_names_and_sup(ds):
    """(names, sup) with sup a per-class superclass id list or None."""
    if ds == "cifar100":
        with open(os.path.join(DATA_DIR, "cifar-100-python", "meta"),
                  "rb") as f:
            m = pickle.load(f, encoding="latin1")
        with open(os.path.join(DATA_DIR, "cifar-100-python", "train"),
                  "rb") as f:
            d = pickle.load(f, encoding="latin1")
        f2c = {}
        for fl, cl in zip(d["fine_labels"], d["coarse_labels"]):
            f2c[fl] = cl
        return m["fine_label_names"], [f2c[i] for i in range(100)]
    if ds == "cifar10":
        return ["airplane", "automobile", "bird", "cat", "deer", "dog",
                "frog", "horse", "ship", "truck"], CIFAR10_SUP
    if ds == "galaxy10":
        return GALAXY10_NAMES, GALAXY10_SUP
    if ds == "aircraft":
        from torchvision import datasets
        dv = datasets.FGVCAircraft(DATA_DIR, split="trainval",
                                   annotation_level="variant")
        dm = datasets.FGVCAircraft(DATA_DIR, split="trainval",
                                   annotation_level="manufacturer")
        v2m = {}
        for yv, ym in zip(dv._labels, dm._labels):
            v2m.setdefault(yv, ym)
        names = list(dv.classes)
        return names, [v2m.get(i, -1) for i in range(len(names))]
    if ds == "cars":
        try:
            import re
            from huggingface_hub import hf_hub_download
            p = hf_hub_download("tanganke/stanford_cars", "README.md",
                                repo_type="dataset")
            pairs = re.findall(r"'(\d+)': (.+)", open(p).read())
            names = [n.strip() for _, n in
                     sorted(pairs, key=lambda x: int(x[0]))]
            assert len(names) == 196, len(names)
            makes = [n.split()[0] for n in names]
            uniq = sorted(set(makes))
            return names, [uniq.index(mk) for mk in makes]
        except Exception as e:
            print(f"  (cars names unavailable: {e})")
            return [f"class{i}" for i in range(196)], None
    if ds == "dtd":
        from torchvision import datasets
        try:
            return list(datasets.DTD(DATA_DIR, split="train").classes), None
        except Exception:
            return [f"class{i}" for i in range(47)], None
    if ds == "flowers":
        return [f"class{i}" for i in range(102)], None
    raise ValueError(ds)


def centroid_dist(X, y, n_cls):
    """L2-normalized class centroids + cosine distance matrix."""
    C = np.zeros((n_cls, X.shape[1]), dtype=np.float64)
    ok = np.zeros(n_cls, dtype=bool)
    for c in range(n_cls):
        m = y == c
        if m.sum():
            C[c] = X[m].mean(0)
            ok[c] = True
    Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    D = 1.0 - Cn @ Cn.T
    np.fill_diagonal(D, 0.0)
    return Cn, D, ok


def dendrogram_purity(Z, sup):
    """Expected purity of the smallest cluster containing a same-superclass
    pair (average over all such pairs)."""
    n = len(sup)
    members = {i: [i] for i in range(n)}
    sup = np.asarray(sup)
    tot, acc = 0, 0.0
    done = set()
    for k, (a, b, _, _) in enumerate(Z):
        merged = members[int(a)] + members[int(b)]
        members[n + k] = merged
        marr = sup[merged]
        for i in members[int(a)]:
            for j in members[int(b)]:
                if sup[i] == sup[j] and (i, j) not in done:
                    acc += (marr == sup[i]).mean()
                    tot += 1
    return acc / max(tot, 1)


def sup_metrics(D, Cn, sup):
    sup = np.asarray(sup)
    n = len(sup)
    order = np.argsort(D + np.eye(n) * 9.9, axis=1)
    same = sup[order] == sup[:, None]
    a1, a5 = same[:, 0].mean(), same[:, :5].mean()
    _, cnt = np.unique(sup, return_counts=True)
    chance = ((cnt * (cnt - 1)).sum()) / (n * (n - 1))
    within = np.mean([D[i, j] for i in range(n) for j in range(n)
                      if i != j and sup[i] == sup[j]])
    between = np.mean([D[i, j] for i in range(n) for j in range(n)
                       if sup[i] != sup[j]])
    Z = hierarchy.linkage(Cn, method="average", metric="cosine")
    dp = dendrogram_purity(Z, sup)
    try:
        from sklearn.metrics import silhouette_score
        keep = np.isin(sup, np.unique(sup)[cnt >= 2])
        sil = float(silhouette_score(Cn[keep], sup[keep], metric="cosine"))
    except Exception:
        sil = float("nan")
    return dict(agree1=float(a1), agree5=float(a5), chance=float(chance),
                wb_ratio=float(within / between), dendro_purity=float(dp),
                silhouette=sil)


def report(space, X, y, names, sup, holdouts, tag, md, mets, plots=False):
    n_cls = len(names)
    Cn, D, ok = centroid_dist(X, y, n_cls)
    star = lambda c: names[c] + ("*" if c in holdouts else "")
    md.append(f"\n## {space}  [{tag}]\n")
    m = None
    if sup is not None:
        m = sup_metrics(D, Cn, sup)
        md.append(f"agree@1={m['agree1']:.3f} agree@5={m['agree5']:.3f} "
                  f"(chance {m['chance']:.3f})  "
                  f"dendro-purity={m['dendro_purity']:.3f}  "
                  f"silhouette={m['silhouette']:.3f}  "
                  f"within/between={m['wb_ratio']:.3f}\n")
        print(f"  [{space:<28}] agree@1={m['agree1']:.3f} "
              f"agree@5={m['agree5']:.3f} (chance {m['chance']:.3f}) "
              f"purity={m['dendro_purity']:.3f} sil={m['silhouette']:.3f} "
              f"w/b={m['wb_ratio']:.3f}")
    else:
        print(f"  [{space:<28}] tables only (no superclass partition)")
    md.append("\n| class | 5 nearest class centroids (cosine dist) |")
    md.append("|---|---|")
    for c in range(n_cls):
        if not ok[c]:
            continue
        nn = np.argsort(D[c] + (np.arange(n_cls) == c) * 9.9)[:5]
        row = " ".join(f"{star(j)}({D[c, j]:.2f})" for j in nn)
        md.append(f"| {star(c)} | {row} |")
    mets[space] = m or {}
    if plots:
        sp = space.replace(" ", "_").replace(">", "")
        so = (np.lexsort((np.arange(n_cls), np.asarray(sup)))
              if sup is not None else np.arange(n_cls))
        plt.figure(figsize=(8, 7))
        plt.imshow(D[np.ix_(so, so)], cmap="viridis")
        plt.colorbar(label="cosine distance")
        if sup is not None:
            edges = np.where(np.diff(np.asarray(sup)[so]))[0] + 0.5
            for e in edges:
                plt.axhline(e, color="w", lw=0.4)
                plt.axvline(e, color="w", lw=0.4)
        plt.title(f"exp76 centroid distances ({space}, {tag})\n"
                  f"{'superclass-ordered' if sup is not None else ''}")
        plt.tight_layout()
        plt.savefig(plot_path(f"exp76_heat_{tag}_{sp}.png"), dpi=150)
        plt.close()
        Z = hierarchy.linkage(Cn[ok], method="average", metric="cosine")
        idx = np.where(ok)[0]
        plt.figure(figsize=(max(7, 0.11 * ok.sum()), 6))
        hierarchy.dendrogram(Z, labels=[star(c) for c in idx],
                             leaf_font_size=3 if n_cls > 60 else 7,
                             color_threshold=0.5 * Z[:, 2].max())
        plt.title(f"exp76 centroid dendrogram ({space}, {tag})")
        plt.tight_layout()
        plt.savefig(plot_path(f"exp76_dendro_{tag}_{sp}.png"), dpi=150)
        plt.close()


def transfer_spaces(args, DS, BASE):
    """Yield (space_name, Xtr, ytr, plots) from cached banks + heads."""
    def bank(path):
        b = torch.load(path)
        (X, y) = b["train"]
        return X.float(), y.numpy()

    fz = os.path.join(DATA_DIR, f"tf_feats_{DS}_{BASE}_vitb16.pt")
    if os.path.exists(fz):
        X, y = bank(fz)
        yield f"frozen-{BASE} (768d)", X.numpy(), y, True
    emb = {}
    for arm in ARMS_70:
        bp = os.path.join(DATA_DIR, f"tf_feats_{DS}_{BASE}_ft70_{arm}{run_tag()}.pt")
        cp = os.path.join(CKPT_DIR, f"{DS}_ft_{BASE}_{arm}_seen{run_tag()}.pt")
        if not (os.path.exists(bp) and os.path.exists(cp)):
            continue
        X, y = bank(bp)
        mod = exp43.FineTuneModel(BASE, args.emb_dim)
        mod.load_state_dict(torch.load(cp, map_location=DEVICE))
        H = exp37.embed(mod.head.float(), X).numpy()
        emb[arm] = (H, y)
        del mod
        yield arm, H, y, arm == "supcon-ft"
    for parent, obj in CHILDREN_71:
        bp = os.path.join(DATA_DIR,
                          f"tf_feats_{DS}_{BASE}_ft70_{parent}_{obj}{run_tag()}.pt")
        cp = os.path.join(CKPT_DIR,
                          f"{DS}_ft_{BASE}_{parent}_{obj}_seen{run_tag()}.pt")
        if not (os.path.exists(bp) and os.path.exists(cp)
                and parent in emb):
            continue
        X, y = bank(bp)
        mod = exp43.FineTuneModel(BASE, args.emb_dim)
        mod.load_state_dict(torch.load(cp, map_location=DEVICE))
        H = exp37.embed(mod.head.float(), X).numpy()
        del mod
        yield f"{parent}-{obj} (residual)", H, y, False
        pH, _ = emb[parent]
        yield (f"{parent}-{obj} (concat)", np.concatenate([pH, H], 1), y,
               obj == "resnplm")


def cifar_spaces(args, ds):
    """Yield (space_name, Xtr, ytr, plots); trains/loads exp76 ckpts."""
    cfg = recipe(ds, emb_dim=args.dim)
    holdouts = {args.holdout}
    con_ep = args.epochs or (2 if args.quick else 20)
    train_loader, test_loader = get_cifar_loaders(quick=args.quick,
                                                  dataset=ds)
    ev = DataLoader(train_loader.dataset, batch_size=256, shuffle=False,
                    num_workers=2)

    def get(name, build):
        ck = os.path.join(CKPT_DIR,
                          f"exp76_{ds}_{args.dim}d_{name}"
                          f"{'_quick' if args.quick else ''}.pt")
        net = CIFARResNetBackbone(args.dim, arch=cfg["arch"],
                                  pretrain=ds).to(DEVICE)
        if os.path.exists(ck) and not args.refresh:
            net.load_state_dict(torch.load(ck, map_location=DEVICE))
        else:
            net = build()
            torch.save(net.state_dict(), ck)
        return net

    sargs = argparse.Namespace(seed=0, quick=args.quick, lam=args.lam,
                               tau=args.tau, dim=args.dim)
    parent = get("supcon", lambda: exp55.train_arm("supcon", ds, cfg,
                                                   sargs, con_ep, holdouts))
    ptr, ytr = collect_embeddings(parent, ev)
    yield "supcon (parent)", ptr, ytr, True

    seen = [c for c in range(cfg["n_classes"]) if c not in holdouts]
    exp28 = importlib.import_module("28_concat_residual")
    m = np.isin(ytr, seen)
    cents = torch.zeros(cfg["n_classes"], args.dim, device=DEVICE)
    cents[torch.as_tensor(seen, device=DEVICE)] = exp28.class_centroids(
        ptr[m], ytr[m], seen).detach().float().to(DEVICE)
    for obj in ("res", "res-nplm"):
        def build(obj=obj):
            torch.manual_seed(7); np.random.seed(7)
            child = copy.deepcopy(parent)
            loader = cifar_two_view_loader(quick=args.quick, labeled=True,
                                           holdout=holdouts, dataset=ds)
            step = (exp73.make_res_step(cents, 5.0) if obj == "res"
                    else exp73.make_res_nplm_step(cents, args.lam,
                                                  cfg["n_slices"]))
            exp73.residual_ft(child, loader, con_ep, step, f"76-{obj}")
            return child
        child = get(obj.replace("-", ""), build)
        rtr, _ = collect_embeddings(child, ev)
        del child
        torch.cuda.empty_cache()
        yield f"supcon-{obj} (residual)", rtr, ytr, False
        yield (f"supcon-{obj} (concat)", np.concatenate([ptr, rtr], 1),
               ytr, True)
    for arm in args.arms:
        net = get(arm, lambda a=arm: exp55.train_arm(a, ds, cfg, sargs,
                                                     con_ep, holdouts))
        X, _ = collect_embeddings(net, ev)
        del net
        torch.cuda.empty_cache()
        yield arm, X, ytr, arm in ("simclr", "nplm_bilinear")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar100",
                    choices=["cifar10", "cifar100", "cars", "aircraft",
                             "flowers", "galaxy10", "dtd"])
    ap.add_argument("--base", default="dino",
                    choices=["dino", "lejepa", "visreg"])
    ap.add_argument("--dim", type=int, default=100)
    ap.add_argument("--emb-dim", type=int, default=100)
    ap.add_argument("--holdout", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--arms", nargs="+", default=[])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--all-plots", action="store_true")
    args = ap.parse_args()
    ds = args.dataset
    cifar = ds.startswith("cifar")
    tag = (f"{ds}_{args.dim}d" if cifar else f"{ds}_{args.base}") + run_tag()
    print(f"exp76 [{tag}] class-centroid interpretability")

    names, sup = class_names_and_sup(ds)
    if sup is not None and min(sup) < 0:
        sup = None
    if cifar:
        n_hold, n_cls = None, len(names)
        holdouts = {args.holdout}
    else:
        n_cls = 47 if ds == "dtd" else exp44.N_CLASSES[ds]
        nh = n_holdout(ds)
        holdouts = holdout_set(ds, n_cls)

    md = [f"# exp76 interpretability [{tag}]",
          f"holdout classes starred; superclass partition: "
          f"{'yes' if sup is not None else 'none'}"]
    mets = {}
    spaces = (cifar_spaces(args, ds) if cifar
              else transfer_spaces(args, ds, args.base))
    for space, X, y, plots in spaces:
        report(space, np.asarray(X, dtype=np.float64), np.asarray(y),
               names, sup, holdouts, tag, md, mets,
               plots=plots or args.all_plots)

    os.makedirs(os.path.join("logs", "exp76"), exist_ok=True)
    out_md = os.path.join("logs", "exp76", f"interp_{tag}.md")
    with open(out_md, "w") as f:
        f.write("\n".join(md) + "\n")
    print("wrote", out_md)
    if mets and any(mets.values()):
        np.savez(os.path.join("logs", "exp76", f"metrics_{tag}.npz"),
                 spaces=np.array([k for k, v in mets.items() if v]),
                 **{f"{k.replace(' ', '_')}__{f}": v[f]
                    for k, v in mets.items() if v for f in v})
        print(f"\n===== EXP76 SUMMARY [{tag}] =====")
        print(f"  {'space':<30}{'agree@1':>9}{'agree@5':>9}{'chance':>8}"
              f"{'purity':>8}{'sil':>8}{'w/b':>8}")
        for k, v in mets.items():
            if v:
                print(f"  {k:<30}{v['agree1']:>9.3f}{v['agree5']:>9.3f}"
                      f"{v['chance']:>8.3f}{v['dendro_purity']:>8.3f}"
                      f"{v['silhouette']:>8.3f}{v['wb_ratio']:>8.3f}")
    print("Done.")


if __name__ == "__main__":
    main()
