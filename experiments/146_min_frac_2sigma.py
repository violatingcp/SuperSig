"""
Experiment 146: minimum injected signal fraction for 2-sigma dataset-level
discovery significance, on the leakage-free CIFAR banks (exp 136/137), with
and without the residual construction.  PRE-discovery spaces only.

Protocol mirrors exp 136's pre_power battery exactly (R = first 20k seen
train embeddings, bg/sig = test split, N_D=5000, 200 null toys, 50 signal
toys per fraction, annealed-sigma SparKer M=16/300 steps, exp-32 Mahalanobis
and MMD statistics) but archives the PER-TOY aggregate scores, so the power
threshold is not baked in.  From those toys we report, per (space, test):

  Z(f)  = one-sided Gaussian significance of the MEDIAN signal-toy aggregate
          against the 200-toy null (add-one empirical tail; 200 nulls resolve
          up to 2.65 sigma), i.e. the median expected significance.
  f*(2sigma) = smallest f with Z(f) >= 2, log-interpolated between grid
          fractions; "<f_min" if already >= 2 at the smallest fraction,
          ">f_max" if never reached.

Decodability/geometry columns (probe, eucl, mahaT, per-event) are read from
the exp 136/137 master JSONs, not recomputed.

    python experiments/146_min_frac_2sigma.py --dataset cifar10
    python experiments/146_min_frac_2sigma.py --quick --spaces supcon
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
from supersig.sparker import (np_test_stats, aggregate_pvalues,
                              median_pairwise, krr_term, mmd2_multi_stats)

exp31 = importlib.import_module("31_sparker_power")

BANKS = os.path.join("logs", "exp136", "banks")
OUT = os.path.join("logs", "exp146")

# (row label, bank tag, json file, json key) -- bank file is
# embs_<tag>_<ds>.npz; json None means the label is not in a master file.
def space_list(ds, holdout=4):
    htag = "" if holdout == 4 else f"_h{holdout}"
    rows = []
    m136 = os.path.join("logs", "exp136", f"master_{ds}{htag}.json")
    m137 = os.path.join("logs", "exp137", f"residuals_{ds}{htag}.json")
    for parent in ("supcon", "ssig", "nplmsd", "nplmcw"):
        rows.append((f"{parent} (parent)", parent, m136, parent))
        for child in ("res", "res-nplm"):
            for use in ("residual", "concat"):
                rows.append((f"{parent}->{child} ({use})",
                             f"{parent}-{child}_({use})",
                             m137, f"{parent}->{child} ({use})"))
    return rows


def json_metrics(path, key):
    if not os.path.exists(path):
        return {}
    j = json.load(open(path))
    r = j.get(key, {})
    if "pre" in r:                      # exp136 master layout
        r = r["pre"]
    return {k: r.get(k) for k in ("probe", "eucl", "mahaT", "perevt")}


def toys_battery(stats_fn, n_bg, n_sig_pool, fractions, N_D, n_null,
                 n_sig_toys, seed, tag=""):
    """exp31/exp32 battery, but returning per-toy aggregates."""
    rng = np.random.default_rng(seed)
    null_ts = []
    for i in range(n_null):
        bg, sg = exp31.toy_indices(rng, n_bg, n_sig_pool, N_D, 0)
        null_ts.append(stats_fn(bg, sg, seed + i))
    null_ts = np.array(null_ts)
    null_agg = np.array([aggregate_pvalues(null_ts[i],
                                           np.delete(null_ts, i, axis=0))
                         for i in range(n_null)])
    sig_agg = np.full((len(fractions), n_sig_toys), np.nan)
    for fi, f in enumerate(fractions):
        n_s = int(round(f * N_D))
        for j in range(n_sig_toys):
            bg, sg = exp31.toy_indices(rng, n_bg, n_sig_pool, N_D, n_s)
            sig_agg[fi, j] = aggregate_pvalues(
                stats_fn(bg, sg, seed + 7919 + j), null_ts)
        print(f"    {tag} f={f}: med agg={np.median(sig_agg[fi]):.3f} "
              f"(null med {np.median(null_agg):.3f})", flush=True)
    return null_agg, sig_agg


def z_curve(null_agg, sig_agg):
    """Median expected significance per fraction (add-one empirical tail)."""
    from scipy.stats import norm
    zs = []
    for row in sig_agg:
        med = np.median(row)
        n = len(null_agg)
        p = (1.0 + (null_agg >= med).sum()) / (n + 1.0)
        p = min(p, n / (n + 1.0))     # keep Z finite when the stat anti-correlates
        zs.append(float(norm.ppf(1.0 - p)))
    return zs


def f_star(fractions, zs, target=2.0):
    """Smallest f with Z >= target, log-interpolated; string for the table."""
    zs = np.asarray(zs, dtype=float)
    if zs[0] >= target:
        return float(fractions[0]), f"$<{fractions[0]}$"
    for i in range(1, len(fractions)):
        if zs[i] >= target and np.isfinite(zs[i - 1]):
            lf0, lf1 = np.log(fractions[i - 1]), np.log(fractions[i])
            t = (target - zs[i - 1]) / (zs[i] - zs[i - 1])
            f = float(np.exp(lf0 + t * (lf1 - lf0)))
            return f, f"{f:.3f}"
    return np.inf, f"$>{fractions[-1]}$"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10",
                    choices=["cifar10", "cifar100"])
    ap.add_argument("--holdout", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-d", type=int, default=5000)
    ap.add_argument("--kernels", type=int, default=16)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--fractions",
                    default="0.001,0.003,0.006,0.01,0.02,0.03,0.05,0.1")
    ap.add_argument("--spaces", default=None,
                    help="comma subset of bank tags (default: all)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    ds = args.dataset
    n_classes = 100 if ds == "cifar100" else 10
    seen = [c for c in range(n_classes) if c != args.holdout]
    fracs = [float(x) for x in args.fractions.split(",")]
    n_null = 20 if args.quick else 200
    n_sig_toys = 10 if args.quick else 50
    steps = 60 if args.quick else args.steps
    os.makedirs(args.out, exist_ok=True)
    res_path = os.path.join(args.out, f"minfrac_{ds}_h{args.holdout}.json")
    results = json.load(open(res_path)) if os.path.exists(res_path) else {}

    htag = "" if args.holdout == 4 else f"_h{args.holdout}"
    for label, tag, jpath, jkey in space_list(ds, args.holdout):
        if args.spaces and tag not in args.spaces.split(","):
            continue
        if label in results and not args.quick:
            print(f"[skip] {label}: already in {res_path}", flush=True)
            continue
        bank = os.path.join(BANKS, f"embs_{tag}_{ds}{htag}.npz")
        if not os.path.exists(bank):
            print(f"[miss] {label}: no bank {bank}", flush=True)
            continue
        d = np.load(bank)
        tr, trl, te, tel = d["tr"], d["tr_lab"], d["te"], d["te_lab"]
        bg_mask = np.isin(tel, seen)
        sig_mask = tel == args.holdout
        R = torch.as_tensor(tr[np.isin(trl, seen)][:20000],
                            dtype=torch.float32, device=DEVICE)
        bg_t = torch.as_tensor(te[bg_mask], dtype=torch.float32, device=DEVICE)
        sig_t = torch.as_tensor(te[sig_mask], dtype=torch.float32,
                                device=DEVICE)
        print(f"[{label}] dim={tr.shape[1]} bg={len(bg_t)} sig={len(sig_t)}",
              flush=True)

        # ---- SparKer (annealed median-heuristic sigma, exp-136 kwargs) ----
        sigma0 = median_pairwise(bg_t, seed=args.seed)

        def spk_fn(bg_idx, sig_idx, seed):
            D = (torch.cat([bg_t[torch.as_tensor(bg_idx, device=DEVICE)],
                            sig_t[torch.as_tensor(sig_idx, device=DEVICE)]])
                 if len(sig_idx) else
                 bg_t[torch.as_tensor(bg_idx, device=DEVICE)])
            return np_test_stats(D, R, M=args.kernels, steps=steps,
                                 sigma0=sigma0, seed=seed)

        # ---- Mahalanobis (exp-32: per-event min-Maha mean, single scale) --
        _, pc, _ = mahalanobis_novelty(tr, trl, te, seen)
        s_bg, s_sig = pc[bg_mask], pc[sig_mask]

        def maha_fn(bg_idx, sig_idx, seed):
            s = (np.concatenate([s_bg[bg_idx], s_sig[sig_idx]])
                 if len(sig_idx) else s_bg[bg_idx])
            return [float(s.mean())]

        # ---- MMD (exp-32: 3 scales around the median heuristic) -----------
        g = np.random.default_rng(args.seed)
        R_pool = tr[np.isin(trl, seen)]
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

        entry = dict(fractions=fracs, n_null=n_null, n_sig_toys=n_sig_toys,
                     metrics=json_metrics(jpath, jkey))
        for name, fn in (("maha", maha_fn), ("mmd", mmd_fn),
                         ("sparker", spk_fn)):
            null_agg, sig_agg = toys_battery(
                fn, len(bg_t), len(sig_t), fracs, args.n_d, n_null,
                n_sig_toys, args.seed, tag=f"{tag}-{name}")
            zs = z_curve(null_agg, sig_agg)
            fs, fs_str = f_star(fracs, zs)
            entry[name] = dict(z=zs, f2sigma=fs, f2sigma_str=fs_str)
            np.savez(os.path.join(args.out,
                                  f"toys_{tag}_{ds}_h{args.holdout}_{name}.npz"),
                     null_agg=null_agg, sig_agg=sig_agg,
                     fractions=np.array(fracs))
            print(f"  {label} {name}: Z={np.round(zs, 2).tolist()} "
                  f"f*(2sig)={fs_str}", flush=True)
        results[label] = entry
        json.dump(results, open(res_path, "w"), indent=1)
        print(f"[done] {label} -> {res_path}", flush=True)

    # ---- summary table -----------------------------------------------------
    print(f"\n== {ds} h{args.holdout}: min injected fraction for 2-sigma "
          f"(median expected) ==")
    print(f"{'space':<28}{'probe':>7}{'eucl':>7}{'mahaT':>7}{'per-ev':>7}"
          f"{'f* maha':>9}{'f* mmd':>9}{'f* spk':>9}")
    for label, tag, jpath, jkey in space_list(ds, args.holdout):
        r = results.get(label)
        if not r:
            continue
        m = r["metrics"]
        fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else "  --"
        print(f"{label:<28}"
              f"{fmt(m.get('probe')):>7}{fmt(m.get('eucl')):>7}"
              f"{fmt(m.get('mahaT')):>7}{fmt(m.get('perevt')):>7}"
              f"{r['maha']['f2sigma_str']:>9}{r['mmd']['f2sigma_str']:>9}"
              f"{r['sparker']['f2sigma_str']:>9}")


if __name__ == "__main__":
    main()
