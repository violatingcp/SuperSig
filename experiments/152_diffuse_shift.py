"""
Experiment 152: diffuse-shift detection on Galaxy10 -- the interpretability
failure corner.  Instead of a novel class, the injected "novelty" is a small
subset of SEEN-class images pushed through one systematic smearing (the same
transformation for every image: PSF-style Gaussian blur by default, or
additive noise / brightness scaling).  Two versions:

  --mode one    the smeared subset is drawn from a single label
  --mode all    the smeared subset is drawn stratified across all seen labels

Everything else is the exp-150 battery: R = labelled train-split embeddings,
bg pool = test seen-class embeddings (smear-source rows excluded), sig pool =
the smeared images embedded through the SAME trunk+head as the space; Maha /
MMD / SparKer toys -> Z(f) and f*(2sigma).  The archived default spaces are
used (holdout class 9 excluded everywhere, seen = 0..8).

Smearing is applied to the normalised eval tensors; Gaussian blur commutes
exactly with the per-channel affine normalisation, so this equals smearing
the raw image.  Diagnostics per cell: ROC-AUC of min-anchor distance for
smeared-vs-background (is the shift even outlying?) and a smeared-vs-clean
linear probe AUC (is it decodable?).

    python experiments/152_diffuse_shift.py --base lejepa --mode one
    python experiments/152_diffuse_shift.py --quick --base dino --mode all --arms supcon-ft --fractions 0.1
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import importlib
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import GaussianBlur

from supersig.config import DEVICE
from supersig.metrics import mahalanobis_novelty
from supersig.sparker import (np_test_stats, median_pairwise, krr_term,
                              mmd2_multi_stats)

exp37 = importlib.import_module("37_dtd_vit")
exp40 = importlib.import_module("40_dtd_bases")
exp43 = importlib.import_module("43_dtd_finetune")
exp44 = importlib.import_module("44_transfer_32d")
exp146 = importlib.import_module("146_min_frac_2sigma")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA, CKPT = os.path.join(REPO, "data"), os.path.join(REPO, "checkpoints")
OUT = os.path.join(REPO, "logs", "exp152")
DS = "galaxy10"
HOLDOUT = 9                      # the archived default draw; excluded entirely
SEEN = list(range(9))


def smear_batch(x, kind, sev):
    if kind == "blur":
        k = int(2 * round(3 * sev) + 1)
        return GaussianBlur(kernel_size=k, sigma=sev)(x)
    if kind == "noise":
        g = torch.Generator(device="cpu").manual_seed(0)
        return x + sev * torch.randn(x.shape, generator=g)
    if kind == "brightness":
        return x * sev
    raise ValueError(kind)


def load_space(base, arm):
    """(trunk, head, bank) for one archived default space."""
    if arm == "pretrained":
        bp = os.path.join(DATA, f"tf_feats_{DS}_{base}_vitb16.pt")
        if not os.path.exists(bp):
            return None
        trunk = exp40.LOADERS[base]().to(DEVICE).eval()
        return trunk, nn.Identity().to(DEVICE), torch.load(bp,
                                                           map_location="cpu")
    ck = os.path.join(CKPT, f"{DS}_ft_{base}_{arm}_seen.pt")
    bp = os.path.join(DATA, f"tf_feats_{DS}_{base}_ft70_{arm}.pt")
    if not (os.path.exists(ck) and os.path.exists(bp)):
        return None
    sd = torch.load(ck, map_location=DEVICE)
    emb_dim = next(v.shape[0] for k, v in reversed(list(sd.items()))
                   if k.startswith("head") and v.dim() == 2)
    mod = exp43.FineTuneModel(base, emb_dim)
    mod.load_state_dict(sd)
    trunk = mod.trunk.to(DEVICE).eval()
    head = mod.head.float().to(DEVICE)
    return trunk, head, torch.load(bp, map_location="cpu")


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="lejepa",
                    choices=["dino", "lejepa", "visreg"])
    ap.add_argument("--arms",
                    default="pretrained,supcon-ft,ss-ft,nplm-sup-ft")
    ap.add_argument("--mode", default="one", choices=["one", "all"])
    ap.add_argument("--label", type=int, default=2,
                    help="source label for --mode one")
    ap.add_argument("--n-smear", type=int, default=1000)
    ap.add_argument("--smear", default="blur",
                    choices=["blur", "noise", "brightness"])
    ap.add_argument("--severity", type=float, default=2.0,
                    help="blur: sigma px; noise: sd in normalised units; "
                         "brightness: scale factor")
    ap.add_argument("--fractions", default="0.006,0.01,0.02,0.03,0.05,0.1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-d", type=int, default=5000)
    ap.add_argument("--kernels", type=int, default=16)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    fracs = [float(x) for x in args.fractions.split(",")]
    n_null = 20 if args.quick else 200
    n_sig_toys = 10 if args.quick else 50
    steps = 60 if args.quick else args.steps
    os.makedirs(args.out, exist_ok=True)
    tag = (f"{args.mode}" + (f"_c{args.label}" if args.mode == "one" else "")
           + f"_{args.smear}{args.severity:g}")
    res_path = os.path.join(args.out, f"diffuse_{DS}_{args.base}_{tag}.json")
    results = json.load(open(res_path)) if os.path.exists(res_path) else {}

    test_ds = exp44.Galaxy10("test", exp37.TF_EVAL)
    te_lab_img = np.array([test_ds.df[test_ds.lab_col].iloc[i]
                           for i in test_ds.keep])
    rng = np.random.default_rng(args.seed)
    if args.mode == "one":
        cand = np.where(te_lab_img == args.label)[0]
    else:
        cand = np.concatenate([
            rng.permutation(np.where(te_lab_img == c)[0])
            [: max(1, args.n_smear // len(SEEN))] for c in SEEN])
    n_s = min(args.n_smear, len(cand))
    smear_idx = np.sort(rng.choice(cand, size=n_s, replace=False))
    print(f"[{args.base} {tag}] smearing {n_s} test images "
          f"({'label ' + str(args.label) if args.mode == 'one' else 'all seen labels'}); "
          f"smear={args.smear}@{args.severity}", flush=True)
    smear_loader = DataLoader(Subset(test_ds, smear_idx.tolist()),
                              batch_size=32, shuffle=False, num_workers=4)

    for arm in args.arms.split(","):
        if arm in results and not args.quick:
            print(f"[skip] {arm}", flush=True)
            continue
        got = load_space(args.base, arm)
        if got is None:
            print(f"[miss] {arm}", flush=True)
            continue
        trunk, head, bank = got
        (Xtr, ytr), (Xte, yte) = bank["train"], bank["test"]
        tr_lab, te_lab = ytr.numpy(), yte.numpy()
        assert np.array_equal(te_lab, te_lab_img), "bank/test order mismatch"
        Htr = exp37.embed(head, Xtr.float()).numpy()
        Hte = exp37.embed(head, Xte.float()).numpy()

        feats = []
        with torch.no_grad():
            for x, _ in smear_loader:
                x = smear_batch(x, args.smear, args.severity)
                with torch.autocast("cuda", dtype=torch.float16,
                                    enabled=DEVICE.type == "cuda"):
                    f = trunk(x.to(DEVICE))
                feats.append(f.float().cpu())
        Hs = exp37.embed(head, torch.cat(feats)).numpy()

        keep_bg = np.ones(len(Hte), dtype=bool)
        keep_bg[smear_idx] = False
        keep_bg &= np.isin(te_lab, SEEN)
        m = np.isin(tr_lab, SEEN)
        bg, sig = Hte[keep_bg], Hs
        clean_src = Hte[smear_idx]           # the same images, unsmeared

        # diagnostics ------------------------------------------------------
        exp28 = importlib.import_module("28_concat_residual")
        anch = torch.as_tensor(exp28.class_centroids(Htr[m], tr_lab[m], SEEN),
                               dtype=torch.float32, device=DEVICE)
        def mind(A):
            return torch.cdist(torch.as_tensor(A, dtype=torch.float32,
                                               device=DEVICE),
                               anch).min(1).values.cpu().numpy()
        d_bg, d_sig = mind(bg), mind(sig)
        auc_out = roc_auc_score(np.r_[np.zeros(len(d_bg)),
                                      np.ones(len(d_sig))],
                                np.r_[d_bg, d_sig])
        X = np.r_[clean_src, sig]
        yflag = np.r_[np.zeros(len(clean_src)), np.ones(len(sig))]
        sh = rng.permutation(len(X))
        ntr = len(X) // 2
        clf = LogisticRegression(max_iter=2000).fit(X[sh[:ntr]],
                                                    yflag[sh[:ntr]])
        auc_probe = roc_auc_score(yflag[sh[ntr:]],
                                  clf.predict_proba(X[sh[ntr:]])[:, 1])
        shift = float(np.mean(d_sig) - np.mean(d_bg))
        print(f"  [{arm}] outlier AUC={auc_out:.3f}  smear-probe "
              f"AUC={auc_probe:.3f}  mean-dist shift={shift:+.3f}",
              flush=True)

        entry = dict(fractions=fracs, n_smear=int(n_s), mode=args.mode,
                     label=(args.label if args.mode == "one" else None),
                     smear=args.smear, severity=args.severity,
                     metrics=dict(auc_outlier=float(auc_out),
                                  auc_smear_probe=float(auc_probe),
                                  dist_shift=shift))
        R = torch.as_tensor(Htr[m][:20000], dtype=torch.float32,
                            device=DEVICE)
        bg_t = torch.as_tensor(bg, dtype=torch.float32, device=DEVICE)
        sig_t = torch.as_tensor(sig, dtype=torch.float32, device=DEVICE)

        _, pc, _ = mahalanobis_novelty(Htr, tr_lab,
                                       np.concatenate([bg, sig]), SEEN)
        s_bg, s_sig = pc[:len(bg)], pc[len(bg):]

        def maha_fn(bg_idx, sig_idx, seed):
            s = (np.concatenate([s_bg[bg_idx], s_sig[sig_idx]])
                 if len(sig_idx) else s_bg[bg_idx])
            return [float(s.mean())]

        g = np.random.default_rng(args.seed)
        R_pool = Htr[m]
        R_mmd = torch.as_tensor(
            R_pool[g.choice(len(R_pool), size=min(5000, len(R_pool)),
                            replace=False)],
            dtype=torch.float32, device=DEVICE)
        med = median_pairwise(bg_t, seed=args.seed)
        sigmas = [0.5 * med, med, 2.0 * med]
        krr = krr_term(R_mmd, sigmas)

        def mmd_fn(bg_idx, sig_idx, seed):
            D = (torch.cat([bg_t[torch.as_tensor(bg_idx, device=DEVICE)],
                            sig_t[torch.as_tensor(sig_idx, device=DEVICE)]])
                 if len(sig_idx) else
                 bg_t[torch.as_tensor(bg_idx, device=DEVICE)])
            return mmd2_multi_stats(D, R_mmd, sigmas, krr)

        sigma0 = median_pairwise(bg_t, seed=args.seed)

        def spk_fn(bg_idx, sig_idx, seed):
            D = (torch.cat([bg_t[torch.as_tensor(bg_idx, device=DEVICE)],
                            sig_t[torch.as_tensor(sig_idx, device=DEVICE)]])
                 if len(sig_idx) else
                 bg_t[torch.as_tensor(bg_idx, device=DEVICE)])
            return np_test_stats(D, R, M=args.kernels, steps=steps,
                                 sigma0=sigma0, seed=seed)

        for name, fn in (("maha", maha_fn), ("mmd", mmd_fn),
                         ("sparker", spk_fn)):
            null_agg, sig_agg = exp146.toys_battery(
                fn, len(bg_t), len(sig_t), fracs, args.n_d, n_null,
                n_sig_toys, args.seed, tag=f"{arm}-{name}")
            zs = exp146.z_curve(null_agg, sig_agg)
            fs, fs_str = exp146.f_star(fracs, zs)
            entry[name] = dict(z=zs, f2sigma=fs, f2sigma_str=fs_str)
            print(f"  [{arm}] {name}: Z={np.round(zs, 2).tolist()} "
                  f"f*={fs_str}", flush=True)
        results[arm] = entry
        json.dump(results, open(res_path, "w"), indent=1)
        del trunk, head
        torch.cuda.empty_cache()

    print(f"\n== {DS}/{args.base} diffuse shift ({tag}) ==")
    print(f"{'arm':<14}{'outAUC':>8}{'prbAUC':>8}{'shift':>8}"
          f"{'f* maha':>9}{'f* mmd':>9}{'f* spk':>9}")
    for arm, r in results.items():
        mt = r["metrics"]
        print(f"{arm:<14}{mt['auc_outlier']:>8.3f}{mt['auc_smear_probe']:>8.3f}"
              f"{mt['dist_shift']:>8.3f}{r['maha']['f2sigma_str']:>9}"
              f"{r['mmd']['f2sigma_str']:>9}{r['sparker']['f2sigma_str']:>9}")


if __name__ == "__main__":
    main()
