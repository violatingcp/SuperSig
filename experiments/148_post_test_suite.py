"""
Experiment 148: the full dataset-level test suite on POST-discovery spaces,
including SparKer seeded at the discovered anchors -- supersedes exp 147's
SparKer-only battery.

Per (arm, fraction f): inject f of the held-out class into the corpus
(exp-68 protocol), run the settled discovery loop on a copy of the scratch
checkpoint, then on the UPDATED space run six dataset-level tests through
the exp-146 toy machinery (200 null / 50 signal toys, N_D=5000) and record
the median expected significance Z at that f:

  eucl         mean over the sample of min Euclidean distance to the seen
               post-space class centroids (single scale)
  eucl-disc    mean of d_seen - d_disc: the anchor-aware Euclidean statistic;
               uses the DISCOVERED anchors as matched filters
  maha         exp-32 mean per-event min-Mahalanobis to the seen-class
               Gaussians, fitted on the post-space reference
  mmd          exp-32 unbiased MMD^2 at 3 bandwidths
  sparker      exp-31 annealed-sigma NP kernel ensemble, data-init (M=16)
  sparker-anch the same NP test with kernels seeded AT the discovered
               anchors (mu_init = the anchor rows discovery appended);
               falls back to data-init if discovery planted no anchors

Each fraction is scored in its own fine-tuned space.  f*(2sigma) per test is
log-interpolated across fractions at the end.  Arm names as in exp 147
("supcon", "supcon-res", "supcon-resnplm", ...).

    python experiments/148_post_test_suite.py --dataset cifar10
    python experiments/148_post_test_suite.py --quick --arms supcon --fractions 0.1
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

from supersig.config import DATA_DIR, DEVICE
from supersig.data import get_cifar_loaders, _cifar_spec
from supersig.models import CIFARResNetBackbone
from supersig.recipes import recipe
from supersig.discovery import run_discovery
from supersig.train import collect_embeddings
from supersig.metrics import mahalanobis_novelty
from supersig.sparker import (np_test_stats, median_pairwise, krr_term,
                              mmd2_multi_stats)

exp28 = importlib.import_module("28_concat_residual")
exp146 = importlib.import_module("146_min_frac_2sigma")

OUT = os.path.join("logs", "exp148")
TESTS = ["eucl", "eucl-disc", "maha", "mmd", "sparker", "sparker-anch"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10",
                    choices=["cifar10", "cifar100"])
    ap.add_argument("--holdout", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dim", type=int, default=100)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--n-d", type=int, default=5000)
    ap.add_argument("--kernels", type=int, default=16)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--fractions", default="0.006,0.01,0.02,0.03,0.05,0.1")
    ap.add_argument("--arms", default="supcon,ssig,nplmsd,nplmcw")
    ap.add_argument("--pool", default="dist", choices=["dist", "np"],
                    help="pool scorer for the discovery loop: distance to "
                         "anchors (campaign default) or the NP density ratio")
    ap.add_argument("--cut", default="quantile", choices=["quantile", "legal"],
                    help="pool cut: 95th percentile of seen scores (campaign "
                         "default) or the paper's label-free rule (needs --pool np)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    ptag = ("" if args.pool == "dist" else f"_{args.pool}") + \
           ("" if args.cut == "quantile" else f"_{args.cut}")

    ds = args.dataset
    cfg = recipe(ds, emb_dim=args.dim)
    n_cls = cfg["n_classes"]
    holdouts = {args.holdout}
    seen = [c for c in range(n_cls) if c not in holdouts]
    fracs = [float(x) for x in args.fractions.split(",")]
    arms = args.arms.split(",")
    ft_ep = 1 if args.quick else cfg["ft_epochs"]
    n_null = 20 if args.quick else 200
    n_sig_toys = 10 if args.quick else 50
    steps = 60 if args.quick else args.steps
    os.makedirs(args.out, exist_ok=True)
    res_path = os.path.join(args.out, f"suite_{ds}_h{args.holdout}{ptag}.json")
    results = json.load(open(res_path)) if os.path.exists(res_path) else {}

    train_loader, test_loader = get_cifar_loaders(quick=args.quick, dataset=ds)
    cls, plain, _ = _cifar_spec(ds)
    base_ds = cls(DATA_DIR, train=True, download=True, transform=plain)
    base_targets = np.array(base_ds.targets)
    n_base = 8000 if args.quick else len(base_ds)
    seen_idx = np.where(np.isin(base_targets[:n_base], seen))[0]
    sig_idx_all = np.where(np.isin(base_targets[:n_base],
                                   list(holdouts)))[0]
    train_eval_loader = DataLoader(train_loader.dataset, batch_size=256,
                                   shuffle=False, num_workers=2)

    def load_base(name):
        net = CIFARResNetBackbone(args.dim, arch=cfg["arch"],
                                  pretrain=None).to(DEVICE)
        htag = "" if args.holdout == 4 else f"_h{args.holdout}"
        fname = f"scratch_{name.replace('-', '_')}_{ds}_{args.dim}d{htag}.pt"
        ck = torch.load(os.path.join("checkpoints", fname),
                        map_location=DEVICE)
        if isinstance(ck, dict) and "state_dict" in ck:
            net.load_state_dict(ck["state_dict"])
        else:
            net.load_state_dict(ck)
            ck = {}
        return net, ck

    for name in arms:
        entry = results.get(name, dict(fractions=fracs, purity1={}, cut={},
                                       z={t: {} for t in TESTS}))
        net, ck = load_base(name)
        tr, tr_lab = collect_embeddings(net, train_eval_loader)
        if name == "supsig" and "means" in ck:
            means0 = ck["means"].to(DEVICE).detach()
        else:
            m = np.isin(tr_lab, seen)
            cents = exp28.class_centroids(tr[m], tr_lab[m], seen)
            means0 = exp28.fill_means(cents, seen, cfg).detach()

        for i_f, f in enumerate(fracs):
            fk = str(f)
            if all(fk in entry["z"].get(t, {}) for t in TESTS):
                print(f"[skip] {name} f={f}: archived", flush=True)
                continue
            n_inj = int(round(f * len(seen_idx) / (1.0 - f)))
            rng = np.random.default_rng(args.seed * 1000 + i_f)
            inj = rng.choice(sig_idx_all, size=min(n_inj, len(sig_idx_all)),
                             replace=False)
            sub = Subset(base_ds, np.concatenate([seen_idx, inj]).tolist())
            tel_loader = DataLoader(sub, batch_size=256, shuffle=False,
                                    num_workers=2)
            print(f"\n===== {name} f={f} ({len(inj)} injected, pool={args.pool}, cut={args.cut}) =====",
                  flush=True)
            bb = copy.deepcopy(net)
            cur_means, hist = run_discovery(
                bb, means0.clone(), base_ds=sub,
                train_eval_loader=tel_loader, test_loader=test_loader,
                seen=seen, holdouts=holdouts, dataset_name=ds,
                rep_weight=cfg["rep_weight"],
                sigreg_weight=cfg["sigreg_weight"],
                n_slices=cfg["n_slices"], rounds=args.rounds,
                ft_epochs=ft_ep, names=None, seed=args.seed,
                pool_score=args.pool, cut_rule=args.cut,
                on_refuse="skip")
            pur1 = float(hist[0]["purity"]) if hist else float("nan")
            c0 = hist[0].get("cut", {}) if hist else {}
            entry.setdefault("cut", {})[fk] = dict(
                ok=bool(c0.get("ok", True)), q=float(c0.get("q", float("nan"))),
                reason=str(c0.get("reason", "")))
            te_post, tel_post = collect_embeddings(bb, test_loader)
            tr_post, trl_post = collect_embeddings(bb, train_eval_loader)
            del bb
            torch.cuda.empty_cache()
            bg_mask = np.isin(tel_post, seen)
            sig_mask = np.isin(tel_post, list(holdouts))
            R_t = torch.as_tensor(tr_post[np.isin(trl_post, seen)][:20000],
                                  dtype=torch.float32, device=DEVICE)
            bg_t = torch.as_tensor(te_post[bg_mask], dtype=torch.float32,
                                   device=DEVICE)
            sig_t = torch.as_tensor(te_post[sig_mask], dtype=torch.float32,
                                    device=DEVICE)
            anchors = cur_means[n_cls:].detach()
            np.savez(os.path.join(
                args.out, f"anchors_{name}_{ds}_h{args.holdout}{ptag}_f{f}.npz"),
                anchors=anchors.cpu().numpy(), purity1=pur1)

            # per-event score pools on the post space -----------------------
            mpost = np.isin(trl_post, seen)
            anch_seen = torch.as_tensor(
                exp28.class_centroids(tr_post[mpost], trl_post[mpost], seen),
                dtype=torch.float32, device=DEVICE)
            zt = torch.cat([bg_t, sig_t])
            d_seen = torch.cdist(zt, anch_seen).min(1).values
            d_disc = (torch.cdist(zt, anchors).min(1).values
                      if len(anchors) else torch.zeros_like(d_seen))
            s_eucl = d_seen.cpu().numpy()
            s_ed = ((d_seen - d_disc).cpu().numpy() if len(anchors)
                    else None)
            nb = len(bg_t)
            _, pc, _ = mahalanobis_novelty(tr_post, trl_post, te_post, seen)
            s_maha_bg, s_maha_sig = pc[bg_mask], pc[sig_mask]

            def mean_fn(s_bg, s_sig):
                def fn(bg_idx, sig_idx, seed):
                    s = (np.concatenate([s_bg[bg_idx], s_sig[sig_idx]])
                         if len(sig_idx) else s_bg[bg_idx])
                    return [float(s.mean())]
                return fn

            g = np.random.default_rng(args.seed)
            R_pool = tr_post[np.isin(trl_post, seen)]
            R_mmd = torch.as_tensor(
                R_pool[g.choice(len(R_pool),
                                size=min(5000, len(R_pool)),
                                replace=False)],
                dtype=torch.float32, device=DEVICE)
            med = median_pairwise(bg_t, seed=args.seed)
            sigmas = [0.5 * med, med, 2.0 * med]
            krr = krr_term(R_mmd, sigmas)

            def mmd_fn(bg_idx, sig_idx, seed):
                D = (torch.cat([bg_t[torch.as_tensor(bg_idx, device=DEVICE)],
                                sig_t[torch.as_tensor(sig_idx,
                                                      device=DEVICE)]])
                     if len(sig_idx) else
                     bg_t[torch.as_tensor(bg_idx, device=DEVICE)])
                return mmd2_multi_stats(D, R_mmd, sigmas, krr)

            sigma0 = median_pairwise(bg_t, seed=args.seed)

            def spk_fn_factory(mu_init):
                def fn(bg_idx, sig_idx, seed):
                    D = (torch.cat([bg_t[torch.as_tensor(bg_idx,
                                                         device=DEVICE)],
                                    sig_t[torch.as_tensor(sig_idx,
                                                          device=DEVICE)]])
                         if len(sig_idx) else
                         bg_t[torch.as_tensor(bg_idx, device=DEVICE)])
                    return np_test_stats(D, R_t, M=args.kernels, steps=steps,
                                         sigma0=sigma0, seed=seed,
                                         mu_init=mu_init)
                return fn

            suites = [
                ("eucl", mean_fn(s_eucl[:nb], s_eucl[nb:])),
                ("eucl-disc", mean_fn(s_ed[:nb], s_ed[nb:])
                 if s_ed is not None else None),
                ("maha", mean_fn(s_maha_bg, s_maha_sig)),
                ("mmd", mmd_fn),
                ("sparker", spk_fn_factory(None)),
                ("sparker-anch", spk_fn_factory(anchors)
                 if len(anchors) else None),
            ]
            for tname, fn in suites:
                if fk in entry["z"].get(tname, {}):
                    continue
                if fn is None:
                    entry["z"].setdefault(tname, {})[fk] = None
                    print(f"  [{name}] {tname} f={f}: no discovered anchors",
                          flush=True)
                    continue
                null_agg, sig_agg = exp146.toys_battery(
                    fn, len(bg_t), len(sig_t), [f], args.n_d, n_null,
                    n_sig_toys, args.seed + i_f, tag=f"{name}-{tname}")
                z = exp146.z_curve(null_agg, sig_agg)[0]
                entry["z"].setdefault(tname, {})[fk] = z
                np.savez(os.path.join(
                    args.out,
                    f"toys_{name}_{ds}_h{args.holdout}{ptag}_f{f}_{tname}.npz"),
                    null_agg=null_agg, sig_agg=sig_agg, fraction=f)
                print(f"  [{name}] {tname} f={f}: Z={z:.2f}", flush=True)
            entry["purity1"][fk] = pur1
            results[name] = entry
            json.dump(results, open(res_path, "w"), indent=1)
        del net
        torch.cuda.empty_cache()

    # ---- summary ----------------------------------------------------------
    print(f"\n== {ds} h{args.holdout} pool={args.pool} cut={args.cut}: POST-discovery suite, "
          f"f*(2sigma) ==")
    hdr = "".join(f"{t:>13}" for t in TESTS)
    print(f"{'arm':<18}{hdr}")
    for name in arms:
        e = results.get(name)
        if not e:
            continue
        cells = []
        for t in TESTS:
            zf = [(float(k), v) for k, v in e["z"].get(t, {}).items()
                  if v is not None]
            if not zf:
                cells.append("--")
                continue
            zf.sort()
            _, fs = exp146.f_star([k for k, _ in zf], [v for _, v in zf])
            cells.append(fs)
        print(f"{name:<18}" + "".join(f"{c:>13}" for c in cells))


if __name__ == "__main__":
    main()
