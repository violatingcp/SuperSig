"""
Experiment 146c: add the Euclidean dataset-level f*(2sigma) to the
pre-discovery battery (exp 146), for the sigma_pre tables.

exp 146 archived Maha / MMD / SparKer f* but only the Euclidean *AUC* (a
decodability metric), never a Euclidean *f\* toy*.  The Euclidean
dataset-level test is the frozen analogue of eucl-disc: per test point,
the minimum Euclidean distance to the seen-class centroids (fit on the
labelled train split); the toy statistic is the mean of that over the
sample.  Cheap (no kernels), so it merges into the existing exp-146 JSONs
under the top-level key 'eucl' without re-running Maha/MMD/SparKer.

    python experiments/146c_eucl_ftest.py --dataset cifar10 --holdouts 4,7,8,9
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import importlib
import json
import numpy as np
import torch

from supersig.config import DEVICE

e146 = importlib.import_module("146_min_frac_2sigma")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BANKS = os.path.join(REPO, "logs", "exp136", "banks")
OUT = os.path.join(REPO, "logs", "exp146")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--holdouts", default="4,7,8,9")
    ap.add_argument("--fractions",
                    default="0.001,0.003,0.006,0.01,0.02,0.03,0.05,0.1")
    ap.add_argument("--n-d", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    ds = args.dataset
    n_cls = 100 if ds == "cifar100" else 10
    fracs = [float(x) for x in args.fractions.split(",")]

    for h in [int(x) for x in args.holdouts.split(",")]:
        seen = [c for c in range(n_cls) if c != h]
        htag = "" if h == 4 else f"_h{h}"
        jpath = os.path.join(OUT, f"minfrac_{ds}_h{h}.json")
        if not os.path.exists(jpath):
            print(f"[miss] {jpath}"); continue
        J = json.load(open(jpath))
        for label, tag, _, _ in e146.space_list(ds, h):
            if label not in J:
                continue
            bank = os.path.join(BANKS, f"embs_{tag}_{ds}{htag}.npz")
            if not os.path.exists(bank):
                continue
            d = np.load(bank)
            tr, trl, te, tel = d["tr"], d["tr_lab"], d["te"], d["te_lab"]
            m = np.isin(trl, seen)
            # seen-class centroids from the labelled train split
            cents = np.stack([tr[m][trl[m] == c].mean(0) for c in seen])
            C = torch.as_tensor(cents, dtype=torch.float32, device=DEVICE)
            T = torch.as_tensor(te, dtype=torch.float32, device=DEVICE)
            s = torch.cdist(T, C).min(1).values.cpu().numpy()  # min-dist
            bg = np.isin(tel, seen); sg = tel == h
            s_bg, s_sig = s[bg], s[sg]

            def eucl_fn(bi, si, seed):
                v = (np.concatenate([s_bg[bi], s_sig[si]]) if len(si)
                     else s_bg[bi])
                return [float(v.mean())]

            null_agg, sig_agg = e146.toys_battery(
                eucl_fn, int(bg.sum()), int(sg.sum()), fracs, args.n_d,
                200, 50, args.seed, tag=f"{label}-eucl")
            zs = e146.z_curve(null_agg, sig_agg)
            fstar, fstr = e146.f_star(fracs, zs)
            J[label]["eucl"] = dict(z=zs, f2sigma=fstar, f2sigma_str=fstr)
            print(f"  [{ds} h{h}] {label}: eucl f*={fstr}", flush=True)
        json.dump(J, open(jpath, "w"), indent=1)
        print(f"merged eucl f* into {jpath}", flush=True)


if __name__ == "__main__":
    main()
