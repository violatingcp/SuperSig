"""
Experiment 153: the on-manifold corner of the failure 2x2 -- aircraft and
cars in the sigma currency.

An unseen aircraft variant or car model embeds INSIDE the seen-class
manifold (the campaign's zero-training pools sat at the base rate there).
This experiment puts that failure in the same f*(2sigma) currency as the
Galaxy10/CIFAR successes: pretrained ViT-B/16 trunk (identity head, the
zero-training construction -- these datasets have no single-holdout
fine-tuned parents), single holdout, draws 0-4, Maha / MMD / SparKer toys.

Caveats stated in the output: the held-out test pools are tiny (aircraft
~33 images/class), so high-fraction toys resample them heavily, and the
background pools are smaller than N_D (bootstrap draws) -- both push
significance UP, so a censored f* here is conservative evidence of failure.

    python experiments/153_onmanifold_battery.py --dataset aircraft
    python experiments/153_onmanifold_battery.py --quick --dataset cars --bases dino --draws 0
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import importlib
import json
import numpy as np
import torch

from supersig.config import DEVICE
from supersig.metrics import mahalanobis_novelty
from supersig.sparker import (np_test_stats, median_pairwise, krr_term,
                              mmd2_multi_stats)
from supersig.holdouts import holdout_set

exp28 = importlib.import_module("28_concat_residual")
exp29 = importlib.import_module("29_residual_finetune")
exp30 = importlib.import_module("30_power_curves")
exp146 = importlib.import_module("146_min_frac_2sigma")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data")
OUT = os.path.join(REPO, "logs", "exp153")
N_CLS = {"aircraft": 100, "cars": 196}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="aircraft",
                    choices=["aircraft", "cars"])
    ap.add_argument("--bases", default="dino,lejepa,visreg")
    ap.add_argument("--draws", default="0,1,2,3,4")
    ap.add_argument("--fractions", default="0.006,0.01,0.02,0.03,0.05,0.1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-d", type=int, default=5000)
    ap.add_argument("--kernels", type=int, default=16)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    ds, n_cls = args.dataset, N_CLS[args.dataset]
    fracs = [float(x) for x in args.fractions.split(",")]
    n_null = 20 if args.quick else 200
    n_sig_toys = 10 if args.quick else 50
    steps = 60 if args.quick else args.steps
    os.makedirs(args.out, exist_ok=True)

    for base in args.bases.split(","):
        bp = os.path.join(DATA, f"tf_feats_{ds}_{base}_vitb16.pt")
        if not os.path.exists(bp):
            print(f"[miss] {bp}")
            continue
        b = torch.load(bp, map_location="cpu")
        (Xtr, ytr), (Xte, yte) = b["train"], b["test"]
        tr, te = Xtr.numpy(), Xte.numpy()
        trl, tel = ytr.numpy(), yte.numpy()
        res_path = os.path.join(args.out, f"minfrac_{ds}_{base}.json")
        results = (json.load(open(res_path)) if os.path.exists(res_path)
                   else {})
        for draw in [int(x) for x in args.draws.split(",")]:
            key = f"pretrained_d{draw}"
            if key in results and not args.quick:
                print(f"[skip] {ds}/{base} {key}", flush=True)
                continue
            holdouts = holdout_set(ds, n_cls, nh=1, draw=draw)
            seen = [c for c in range(n_cls) if c not in holdouts]
            bgm = np.isin(tel, seen)
            sgm = np.isin(tel, list(holdouts))
            m = np.isin(trl, seen)
            anch = torch.as_tensor(
                exp28.class_centroids(tr[m], trl[m], seen),
                dtype=torch.float32, device=DEVICE)
            ev = exp29.evaluate_space(tr, trl, te, tel, anch, seen, holdouts)
            torch.manual_seed(1000)
            probe, _, _ = exp29.linear_probe_novelty(tr, trl, te, tel,
                                                     holdouts)
            d = torch.cdist(torch.as_tensor(te, dtype=torch.float32,
                                            device=DEVICE), anch)
            s_ = d.min(1).values.cpu().numpy()
            pe = exp30.power_at_alpha(s_[bgm], s_[sgm], args.alpha)
            entry = dict(fractions=fracs, holdout=sorted(holdouts),
                         n_sig_pool=int(sgm.sum()), n_bg_pool=int(bgm.sum()),
                         metrics=dict(probe=float(probe), eucl=ev["eucl"],
                                      mahaT=ev["maha_tied"], perevt=pe))
            print(f"[{ds}/{base} d{draw}] holdout {sorted(holdouts)} "
                  f"sig={int(sgm.sum())} probe={probe:.3f} "
                  f"eucl={ev['eucl']:.3f} mahaT={ev['maha_tied']:.3f} "
                  f"perev={pe:.2f}", flush=True)

            R = torch.as_tensor(tr[m][:20000], dtype=torch.float32,
                                device=DEVICE)
            bg_t = torch.as_tensor(te[bgm], dtype=torch.float32,
                                   device=DEVICE)
            sig_t = torch.as_tensor(te[sgm], dtype=torch.float32,
                                    device=DEVICE)
            _, pc, _ = mahalanobis_novelty(tr, trl, te, seen)
            s_bg, s_sig = pc[bgm], pc[sgm]

            def maha_fn(bi, si, seed):
                s = (np.concatenate([s_bg[bi], s_sig[si]]) if len(si)
                     else s_bg[bi])
                return [float(s.mean())]

            g = np.random.default_rng(args.seed)
            R_pool = tr[m]
            R_mmd = torch.as_tensor(
                R_pool[g.choice(len(R_pool), size=min(5000, len(R_pool)),
                                replace=False)],
                dtype=torch.float32, device=DEVICE)
            med = median_pairwise(bg_t, seed=args.seed)
            sigmas = [0.5 * med, med, 2.0 * med]
            krr = krr_term(R_mmd, sigmas)

            def mmd_fn(bi, si, seed):
                D = (torch.cat([bg_t[torch.as_tensor(bi, device=DEVICE)],
                                sig_t[torch.as_tensor(si, device=DEVICE)]])
                     if len(si) else bg_t[torch.as_tensor(bi, device=DEVICE)])
                return mmd2_multi_stats(D, R_mmd, sigmas, krr)

            sigma0 = median_pairwise(bg_t, seed=args.seed)

            def spk_fn(bi, si, seed):
                D = (torch.cat([bg_t[torch.as_tensor(bi, device=DEVICE)],
                                sig_t[torch.as_tensor(si, device=DEVICE)]])
                     if len(si) else bg_t[torch.as_tensor(bi, device=DEVICE)])
                return np_test_stats(D, R, M=args.kernels, steps=steps,
                                     sigma0=sigma0, seed=seed)

            for name, fn in (("maha", maha_fn), ("mmd", mmd_fn),
                             ("sparker", spk_fn)):
                null_agg, sig_agg = exp146.toys_battery(
                    fn, len(bg_t), len(sig_t), fracs, args.n_d, n_null,
                    n_sig_toys, args.seed, tag=f"{base}-d{draw}-{name}")
                zs = exp146.z_curve(null_agg, sig_agg)
                fs, fss = exp146.f_star(fracs, zs)
                entry[name] = dict(z=zs, f2sigma=fs, f2sigma_str=fss)
                print(f"  [{key}] {name}: Z={np.round(zs, 2).tolist()} "
                      f"f*={fss}", flush=True)
            results[key] = entry
            json.dump(results, open(res_path, "w"), indent=1)

        print(f"\n== {ds}/{base}: on-manifold single-holdout battery ==")
        for key, r in sorted(results.items()):
            mt = r["metrics"]
            print(f"  {key:<16} hold={r['holdout']} sig={r['n_sig_pool']:>4} "
                  f"probe={mt['probe']:.3f} eucl={mt['eucl']:.3f} "
                  f"perev={mt['perevt']:.2f} | "
                  f"maha={r['maha']['f2sigma_str']:>8} "
                  f"mmd={r['mmd']['f2sigma_str']:>8} "
                  f"spk={r['sparker']['f2sigma_str']:>8}")


if __name__ == "__main__":
    main()
