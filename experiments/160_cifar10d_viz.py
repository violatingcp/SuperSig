"""
Experiment 160: a 10-DIMENSIONAL from-scratch CIFAR-10 latent for
visualization -- the CIFAR twin of exp 157 (Galaxy10).

CIFAR-10 has 10 classes; this consumes the dim-10 scratch heads trained by
exp 67 (parents, --dim 10) and exp 137 (residual children, --dim 10) -- both
name their checkpoints `scratch_*_cifar10_10d*.pt`, so the 100-D archive is
untouched.  It embeds the train/test images to 10-D (PRE), runs the settled
discovery loop with the s/sqrt(b) cut (np pool, cut ssb, n_min 5), embeds
again (POST), and saves per arm an npz with pre/post embeddings, labels,
discovered anchors and seen centroids -- the payload exp 158/159 plot.

    python experiments/160_cifar10d_viz.py --holdout 4 --arms supcon supcon-res
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import copy
import importlib
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from supersig.config import DEVICE
from supersig.data import get_cifar_loaders, _cifar_spec, DATA_DIR
from supersig.discovery import run_discovery
from supersig.models import CIFARResNetBackbone
from supersig.recipes import recipe
from supersig.train import collect_embeddings

exp28 = importlib.import_module("28_concat_residual")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(REPO, "checkpoints")
OUT = os.path.join(REPO, "logs", "exp160")
CIFAR10_NAMES = {0: "airplane", 1: "automobile", 2: "bird", 3: "cat",
                 4: "deer", 5: "dog", 6: "frog", 7: "horse", 8: "ship",
                 9: "truck"}
ARMS = ["supcon", "ssig", "nplmsd", "nplmcw", "supcon-res",
        "supcon-resnplm", "ssig-res", "nplmsd-res", "nplmcw-res"]


def load_net(ds, arm, holdout, dim, cfg):
    fname = f"scratch_{arm.replace('-', '_')}_{ds}_{dim}d" \
            f"{'' if holdout == 4 else f'_h{holdout}'}.pt"
    p = os.path.join(CKPT, fname)
    if not os.path.exists(p):
        print(f"[miss] {arm}: {fname}", flush=True)
        return None
    net = CIFARResNetBackbone(dim, arch=cfg["arch"], pretrain=None).to(DEVICE)
    ck = torch.load(p, map_location=DEVICE)
    net.load_state_dict(ck["state_dict"] if isinstance(ck, dict)
                        and "state_dict" in ck else ck)
    net.eval()
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--holdout", type=int, default=4)
    ap.add_argument("--dim", type=int, default=10)
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--n-min", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--ft-epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    ds = args.dataset
    cfg = recipe(ds, emb_dim=args.dim)
    n_cls = cfg["n_classes"]
    holdouts = {args.holdout}
    seen = [c for c in range(n_cls) if c not in holdouts]
    ft_ep = args.ft_epochs or cfg["ft_epochs"]
    os.makedirs(args.out, exist_ok=True)
    print(f"exp160 {ds} holdout {args.holdout} dim {args.dim}, s/sqrt(b)",
          flush=True)

    train_loader, test_loader = get_cifar_loaders(dataset=ds)
    tel = DataLoader(train_loader.dataset, batch_size=256, shuffle=False,
                     num_workers=2)
    cls, plain, _ = _cifar_spec(ds)
    base_ds = cls(DATA_DIR, train=True, download=True, transform=plain)

    summary = {}
    for arm in args.arms:
        net = load_net(ds, arm, args.holdout, args.dim, cfg)
        if net is None:
            continue
        tr0, tr_lab = collect_embeddings(net, tel)
        te0, te_lab = collect_embeddings(net, test_loader)
        m = np.isin(tr_lab, seen)
        cents = exp28.class_centroids(tr0[m], tr_lab[m], seen)
        means0 = exp28.fill_means(cents, seen, cfg).detach()

        bb = copy.deepcopy(net)
        cur_means, hist = run_discovery(
            bb, means0.clone(), base_ds=base_ds, train_eval_loader=tel,
            test_loader=test_loader, seen=seen, holdouts=holdouts,
            dataset_name=ds, rep_weight=cfg["rep_weight"],
            sigreg_weight=cfg["sigreg_weight"], n_slices=cfg["n_slices"],
            rounds=args.rounds, ft_epochs=ft_ep, names=None, seed=args.seed,
            pool_score="np", cut_rule="ssb", n_min=args.n_min,
            on_refuse="skip")
        tr_post, trl_post = collect_embeddings(bb, tel)
        te_post, tel_post = collect_embeddings(bb, test_loader)
        del bb
        torch.cuda.empty_cache()
        anchors = cur_means[n_cls:].detach().cpu().numpy()
        mp = np.isin(trl_post, seen)
        sc = exp28.class_centroids(tr_post[mp], trl_post[mp], seen)
        sc = np.asarray(sc.detach().cpu() if hasattr(sc, "detach") else sc)
        pur1 = float(hist[0]["purity"]) if hist else float("nan")
        engaged = bool(hist and hist[0].get("cut", {}).get("ok", False))

        np.savez(os.path.join(args.out, f"viz_{ds}_h{args.holdout}_"
                              f"e{args.dim}_{arm}.npz"),
                 tr=tr0, te=te0, tr_lab=tr_lab, te_lab=te_lab,
                 tr_post=tr_post, te_post=te_post, seen_centroids=sc,
                 anchors=anchors, seen=np.asarray(seen),
                 holdout=np.asarray(sorted(holdouts)))
        summary[arm] = dict(engaged=engaged, purity1=pur1,
                            n_anchors=int(anchors.shape[0]))
        print(f"[{arm}] engaged={engaged} purity={pur1:.3f} "
              f"anchors={anchors.shape[0]} dim={tr0.shape[1]}", flush=True)

    json.dump(summary, open(os.path.join(
        args.out, f"summary_{ds}_h{args.holdout}_e{args.dim}.json"), "w"),
        indent=1)
    print(f"\nwrote {args.out}/viz_* ({len(summary)} arms)", flush=True)


if __name__ == "__main__":
    main()
