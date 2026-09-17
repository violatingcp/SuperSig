"""
Experiment 165: the exp-28 alternate procedure -- discovery IN the concat
[supcon ; residual] space, fine-tuning ONLY the supervised (supcon) branch,
with the residual a frozen channel that helps CLUSTER the novelty -- on the
10-D CIFAR-10 h4 spaces, in the f*(2sigma) currency.

Contrast with exp 162 (which discovers in ONE space then concatenates): here
exp28.run_concat_discovery pools + BIC-clusters in the joint [sup ; ssl]
space and moves only the sup branch, refreshing the residual anchor-halves
from the pooled points.  The idea (the user's): supcon+discovery sharpens the
supervised space, and the frozen residual adds clustering signal.

Per injected fraction f: inject f held-out train images, run
run_concat_discovery (exp28 quantile cut), then score the post concat
[sup*(x) ; residual(x)] with the six tests (eucl, eucl-disc, maha, mmd,
sparker, sparker-anch) through the exp-146 toys.

    python experiments/165_exp28_concat_discovery.py --dim 10 --holdout 4
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
from supersig.models import CIFARResNetBackbone
from supersig.recipes import recipe
from supersig.train import collect_embeddings

exp28 = importlib.import_module("28_concat_residual")
exp146 = importlib.import_module("146_min_frac_2sigma")
exp162 = importlib.import_module("162_cifar_concat_residual")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(REPO, "checkpoints")
OUT = os.path.join(REPO, "logs", "exp165")
TESTS = ["eucl", "eucl-disc", "maha", "mmd", "sparker", "sparker-anch"]


def load_net(ds, arm, holdout, dim, cfg):
    fn = f"scratch_{arm.replace('-', '_')}_{ds}_{dim}d" \
         f"{'' if holdout == 4 else f'_h{holdout}'}.pt"
    net = CIFARResNetBackbone(dim, arch=cfg["arch"], pretrain=None).to(DEVICE)
    ck = torch.load(os.path.join(CKPT, fn), map_location=DEVICE)
    net.load_state_dict(ck["state_dict"] if isinstance(ck, dict)
                        and "state_dict" in ck else ck)
    net.eval()
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--dim", type=int, default=10)
    ap.add_argument("--holdout", type=int, default=4)
    ap.add_argument("--pairs", default="supcon:supcon-res")
    ap.add_argument("--fractions", default="0.006,0.01,0.02,0.03,0.05,0.1")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--ft-epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-d", type=int, default=5000)
    ap.add_argument("--kernels", type=int, default=16)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    ds = args.dataset
    cfg = recipe(ds, emb_dim=args.dim)
    n_cls = cfg["n_classes"]
    holdouts = {args.holdout}
    seen = [c for c in range(n_cls) if c not in holdouts]
    fracs = [float(x) for x in args.fractions.split(",")]
    pairs = [tuple(p.split(":")) for p in args.pairs.split(",")]
    ft_ep = args.ft_epochs or cfg["ft_epochs"]
    os.makedirs(args.out, exist_ok=True)

    train_loader, test_loader = get_cifar_loaders(dataset=ds)
    tel = DataLoader(train_loader.dataset, batch_size=256, shuffle=False,
                     num_workers=2)
    cls, plain, _ = _cifar_spec(ds)
    base_ds = cls(DATA_DIR, train=True, download=True, transform=plain)
    base_t = np.array(base_ds.targets)
    seen_idx = np.where(np.isin(base_t, seen))[0]
    sig_idx_all = np.where(np.isin(base_t, list(holdouts)))[0]
    res_path = os.path.join(args.out,
                            f"exp28cat_{ds}_h{args.holdout}_e{args.dim}.json")
    results = json.load(open(res_path)) if os.path.exists(res_path) else {}

    def z_of(fn, bg_t, sig_t, f, seed):
        na, sa = exp146.toys_battery(fn, len(bg_t), len(sig_t), [f],
                                     args.n_d, 200, 50, seed, tag="")
        return exp146.z_curve(na, sa)[0]

    for sup_arm, res_arm in pairs:
        sup0 = load_net(ds, sup_arm, args.holdout, args.dim, cfg)
        trunk = load_net(ds, res_arm, args.holdout, args.dim, cfg)  # frozen ssl
        S0_tr, tr_lab = collect_embeddings(sup0, tel)
        R_tr, _ = collect_embeddings(trunk, tel)          # frozen residual
        R_te, te_lab = collect_embeddings(trunk, test_loader)
        m = np.isin(tr_lab, seen)
        means_sup = exp28.fill_means(exp28.class_centroids(S0_tr[m], tr_lab[m],
                                                           seen), seen,
                                     cfg).detach()
        ssl_cents = torch.as_tensor(
            exp28.class_centroids(R_tr[m], tr_lab[m], seen),
            dtype=torch.float32, device=DEVICE)
        key = f"{sup_arm}+{res_arm}"
        entry = results.get(key, dict(fractions=fracs, cut={},
                                      concat={t: {} for t in TESTS}))

        for i_f, f in enumerate(fracs):
            fk = str(f)
            if fk in entry["concat"]["eucl"]:
                print(f"[skip] {key} f={f}", flush=True); continue
            n_inj = int(round(f * len(seen_idx) / (1 - f)))
            rng = np.random.default_rng(args.seed * 1000 + i_f)
            inj = rng.choice(sig_idx_all, size=min(n_inj, len(sig_idx_all)),
                             replace=False)
            sub_ids = np.concatenate([seen_idx, inj]).tolist()
            sub = Subset(base_ds, sub_ids)
            sub_loader = DataLoader(sub, batch_size=256, shuffle=False,
                                    num_workers=2)
            sup = copy.deepcopy(sup0)
            print(f"\n== {key} f={f} ({len(inj)} injected) exp28 concat "
                  f"discovery ==", flush=True)
            hist, out = exp28.run_concat_discovery(
                sup, trunk, means_sup.clone(), ssl_cents, base=sub,
                dim=args.dim, train_eval_loader=sub_loader,
                test_loader=test_loader, seen=seen, holdouts=holdouts,
                cfg=cfg, rounds=args.rounds, ft_epochs=ft_ep, seed=args.seed)
            cur_means, disc_ssl = out["cur_means"], out["disc_ssl"]
            entry["cut"][fk] = dict(pur=float(hist[0]["purity"]) if hist
                                    else float("nan"),
                                    n_anchors=int(disc_ssl.size(0)))
            # post concat embeddings [sup*(x) ; residual(x)]
            Sp_tr, trl2 = collect_embeddings(sup, tel)
            Sp_te, tel2 = collect_embeddings(sup, test_loader)
            del sup; torch.cuda.empty_cache()
            C_tr = np.concatenate([Sp_tr, R_tr], 1)
            C_te = np.concatenate([Sp_te, R_te], 1)
            # concat discovered anchors [sup-anchor ; residual-anchor]
            A = (torch.cat([cur_means[n_cls:], disc_ssl], 1).cpu().numpy()
                 if disc_ssl.size(0) else None)
            _, det = exp162._prep(C_tr, tr_lab, C_te, te_lab, seen, holdouts,
                                  A, args)
            bg_t, sig_t, nb = det["bg_t"], det["sig_t"], det["nb"]
            fns = {
                "eucl": det["meanf"](det["s_eu"][:nb], det["s_eu"][nb:]),
                "eucl-disc": (det["meanf"](det["s_ed"][:nb], det["s_ed"][nb:])
                              if det["s_ed"] is not None else None),
                "maha": det["meanf"](det["s_mb"], det["s_ms"]),
                "mmd": det["mmd_fn"], "sparker": det["spk"](None),
                "sparker-anch": (det["spk"](torch.as_tensor(
                    A, dtype=torch.float32, device=DEVICE))
                    if (A is not None and len(A)) else None)}
            for tn, fn in fns.items():
                entry["concat"][tn][fk] = (z_of(fn, bg_t, sig_t, f,
                                                args.seed + i_f)
                                           if fn is not None else None)
            results[key] = entry
            json.dump(results, open(res_path, "w"), indent=1)
            print(f"[done] {key} f={f} pur={entry['cut'][fk]['pur']:.3f} "
                  f"anchors={entry['cut'][fk]['n_anchors']} "
                  f"ed={entry['concat']['eucl-disc'][fk]} "
                  f"mmd={entry['concat']['mmd'][fk]}", flush=True)
    print("\nexp165 done.", flush=True)


if __name__ == "__main__":
    main()
