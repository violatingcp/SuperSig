"""
Experiment 154: discovery mode + residuals on the ON-MANIFOLD corner --
aircraft and cars in the sigma currency.

Exp 153 measured the frozen pretrained trunk only.  This experiment runs the
settled discovery loop (density-ratio pool, derived label-free cut at
n_min=5, skip-on-refuse) on the exp-70 fine-tuned heads AND the residual
children, head-only over the cached trunk banks (exp-151 protocol), plus a
frozen PRE-discovery battery per arm so the residual comparison exists
pre and post.

Holdout convention: the exp-70 archived split (last 10 classes held out of
100/196, untagged artifacts).  This is the MULTI-holdout regime -- injected
"signal" is a 10-class mixture -- and must not be pooled with the exp-153
single-holdout draws.  Injection = exp-151: n = f/(1-f)*|seen train| held-out
TRAIN-bank images (clamped at the pool size; actual count archived).

    python experiments/154_onmanifold_post_suite.py --dataset aircraft --base dino
    python experiments/154_onmanifold_post_suite.py --quick --dataset cars --base visreg --arms supcon-ft --fractions 0.1
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import copy
import importlib
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from supersig.config import DEVICE
from supersig.discovery import run_discovery
from supersig.train import collect_embeddings
from supersig.metrics import mahalanobis_novelty
from supersig.sparker import (np_test_stats, median_pairwise, krr_term,
                              mmd2_multi_stats)
from supersig.holdouts import holdout_set

exp28 = importlib.import_module("28_concat_residual")
exp29 = importlib.import_module("29_residual_finetune")
exp30 = importlib.import_module("30_power_curves")
exp37 = importlib.import_module("37_dtd_vit")
exp43 = importlib.import_module("43_dtd_finetune")
exp146 = importlib.import_module("146_min_frac_2sigma")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA, CKPT = os.path.join(REPO, "data"), os.path.join(REPO, "checkpoints")
OUT = os.path.join(REPO, "logs", "exp154")
N_CLS_MAP = {"aircraft": 100, "cars": 196}
TESTS = ["eucl", "eucl-disc", "maha", "mmd", "sparker", "sparker-anch"]


def load_arm(ds, base, arm):
    """(head module, Xtr, ytr, Xte, yte) for one exp-70 arm, or None."""
    bp = os.path.join(DATA, f"tf_feats_{ds}_{base}_ft70_{arm}.pt")
    ck = os.path.join(CKPT, f"{ds}_ft_{base}_{arm}_seen.pt")
    if not (os.path.exists(bp) and os.path.exists(ck)):
        return None
    sd = torch.load(ck, map_location=DEVICE)
    emb_dim = next(v.shape[0] for k, v in reversed(list(sd.items()))
                   if k.startswith("head") and v.dim() == 2)
    mod = exp43.FineTuneModel(base, emb_dim)
    mod.load_state_dict(sd)
    head = copy.deepcopy(mod.head).float().to(DEVICE)
    del mod
    torch.cuda.empty_cache()
    b = torch.load(bp, map_location="cpu")
    (Xtr, ytr), (Xte, yte) = b["train"], b["test"]
    return head, Xtr.float(), ytr, Xte.float(), yte


def toy_fns(tr, trl, te, tel, seen, holdouts, seed, kernels, steps):
    """(maha_fn, mmd_fn, spk_factory, bg_t, sig_t) on a fixed embedding."""
    bg_mask = np.isin(tel, seen)
    sig_mask = np.isin(tel, list(holdouts))
    R_t = torch.as_tensor(tr[np.isin(trl, seen)][:20000],
                          dtype=torch.float32, device=DEVICE)
    bg_t = torch.as_tensor(te[bg_mask], dtype=torch.float32, device=DEVICE)
    sig_t = torch.as_tensor(te[sig_mask], dtype=torch.float32, device=DEVICE)
    _, pc, _ = mahalanobis_novelty(tr, trl, te, seen)
    s_bg, s_sig = pc[bg_mask], pc[sig_mask]

    def maha_fn(bi, si, seed_):
        s = np.concatenate([s_bg[bi], s_sig[si]]) if len(si) else s_bg[bi]
        return [float(s.mean())]

    g = np.random.default_rng(seed)
    R_pool = tr[np.isin(trl, seen)]
    R_mmd = torch.as_tensor(
        R_pool[g.choice(len(R_pool), size=min(5000, len(R_pool)),
                        replace=False)],
        dtype=torch.float32, device=DEVICE)
    med = median_pairwise(bg_t, seed=seed)
    sigmas = [0.5 * med, med, 2.0 * med]
    krr = krr_term(R_mmd, sigmas)

    def mmd_fn(bi, si, seed_):
        D = (torch.cat([bg_t[torch.as_tensor(bi, device=DEVICE)],
                        sig_t[torch.as_tensor(si, device=DEVICE)]])
             if len(si) else bg_t[torch.as_tensor(bi, device=DEVICE)])
        return mmd2_multi_stats(D, R_mmd, sigmas, krr)

    sigma0 = median_pairwise(bg_t, seed=seed)

    def spk_factory(mu_init):
        def fn(bi, si, seed_):
            D = (torch.cat([bg_t[torch.as_tensor(bi, device=DEVICE)],
                            sig_t[torch.as_tensor(si, device=DEVICE)]])
                 if len(si) else bg_t[torch.as_tensor(bi, device=DEVICE)])
            return np_test_stats(D, R_t, M=kernels, steps=steps,
                                 sigma0=sigma0, seed=seed_, mu_init=mu_init)
        return fn

    return maha_fn, mmd_fn, spk_factory, bg_t, sig_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="aircraft",
                    choices=["aircraft", "cars"])
    ap.add_argument("--base", default="dino",
                    choices=["dino", "lejepa", "visreg"])
    ap.add_argument("--arms", default="supcon-ft,ss-ft,supcon-ft_res,"
                                      "supcon-ft_resnplm,ss-ft_res")
    ap.add_argument("--fractions", default="0.01,0.02,0.03,0.05,0.1")
    ap.add_argument("--pool", default="np", choices=["dist", "np"])
    ap.add_argument("--cut", default="legal", choices=["quantile", "legal"])
    ap.add_argument("--n-min", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--ft-epochs", type=int, default=None)
    ap.add_argument("--n-slices", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-d", type=int, default=5000)
    ap.add_argument("--kernels", type=int, default=16)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--skip-pre", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    ds, n_cls = args.dataset, N_CLS_MAP[args.dataset]
    fracs = [float(x) for x in args.fractions.split(",")]
    n_null = 20 if args.quick else 200
    n_sig_toys = 10 if args.quick else 50
    steps = 60 if args.quick else args.steps
    ft_ep = args.ft_epochs or (1 if args.quick else 5)
    ptag = f"_{args.pool}_{args.cut}_nmin{args.n_min}"
    cfg = dict(n_classes=n_cls, pair_dist=5.0)
    rep_weight = 20.0 * 45.0 / (n_cls * (n_cls - 1) / 2)   # exp 70
    os.makedirs(args.out, exist_ok=True)

    holdouts = holdout_set(ds, n_cls)          # exp-70 default: last 10
    seen = [c for c in range(n_cls) if c not in holdouts]
    res_path = os.path.join(args.out, f"suite_{ds}_{args.base}{ptag}.json")
    results = json.load(open(res_path)) if os.path.exists(res_path) else {}

    for arm in args.arms.split(","):
        got = load_arm(ds, args.base, arm)
        if got is None:
            print(f"[miss] {ds}/{args.base} {arm}", flush=True)
            continue
        head, Xtr, ytr, Xte, yte = got
        tr_lab, te_lab = ytr.numpy(), yte.numpy()
        tel_full = DataLoader(TensorDataset(Xtr, ytr), batch_size=512,
                              shuffle=False)
        test_loader = DataLoader(TensorDataset(Xte, yte), batch_size=512,
                                 shuffle=False)
        tr0, _ = collect_embeddings(head, tel_full)
        m = np.isin(tr_lab, seen)
        entry = results.get(arm, dict(fractions=fracs,
                                      holdout=sorted(holdouts), purity1={},
                                      cut={}, n_inj={},
                                      z={t: {} for t in TESTS}))
        entry.setdefault("z", {t: {} for t in TESTS})

        # ---------- PRE: the frozen battery on this arm's space ----------
        if not args.skip_pre and ("pre" not in entry or args.quick):
            te0, _ = collect_embeddings(head, test_loader)
            cents = exp28.class_centroids(tr0[m], tr_lab[m], seen)
            anch = torch.as_tensor(cents, dtype=torch.float32, device=DEVICE)
            ev = exp29.evaluate_space(tr0, tr_lab, te0, te_lab, anch, seen,
                                      holdouts)
            torch.manual_seed(1000)
            probe, _, _ = exp29.linear_probe_novelty(tr0, tr_lab, te0,
                                                     te_lab, holdouts)
            d = torch.cdist(torch.as_tensor(te0, dtype=torch.float32,
                                            device=DEVICE), anch)
            s_ = d.min(1).values.cpu().numpy()
            bgm = np.isin(te_lab, seen)
            sgm = np.isin(te_lab, list(holdouts))
            pe = exp30.power_at_alpha(s_[bgm], s_[sgm], args.alpha)
            pre = dict(metrics=dict(probe=float(probe), eucl=ev["eucl"],
                                    mahaT=ev["maha_tied"], perevt=pe))
            print(f"[{ds}/{args.base} {arm}] PRE probe={probe:.3f} "
                  f"eucl={ev['eucl']:.3f} mahaT={ev['maha_tied']:.3f} "
                  f"perev={pe:.2f}", flush=True)
            maha_fn, mmd_fn, spk_factory, bg_t, sig_t = toy_fns(
                tr0, tr_lab, te0, te_lab, seen, holdouts, args.seed,
                args.kernels, steps)
            for name, fn in (("maha", maha_fn), ("mmd", mmd_fn),
                             ("sparker", spk_factory(None))):
                null_agg, sig_agg = exp146.toys_battery(
                    fn, len(bg_t), len(sig_t), fracs, args.n_d, n_null,
                    n_sig_toys, args.seed, tag=f"{arm}-pre-{name}")
                zs = exp146.z_curve(null_agg, sig_agg)
                fs, fss = exp146.f_star(fracs, zs)
                pre[name] = dict(z=zs, f2sigma=fs, f2sigma_str=fss)
                print(f"  [{arm}] PRE {name}: Z={np.round(zs, 2).tolist()} "
                      f"f*={fss}", flush=True)
            entry["pre"] = pre
            results[arm] = entry
            json.dump(results, open(res_path, "w"), indent=1)

        # ---------- POST: discovery per injected fraction ----------
        cents = exp28.class_centroids(tr0[m], tr_lab[m], seen)
        means0 = exp28.fill_means(cents, seen, cfg).detach()
        seen_idx = np.where(m)[0]
        sig_idx_all = np.where(np.isin(tr_lab, list(holdouts)))[0]
        for i_f, f in enumerate(fracs):
            fk = str(f)
            if all(fk in entry["z"].get(t, {}) for t in TESTS):
                print(f"[skip] {ds}/{args.base} {arm} f={f}", flush=True)
                continue
            n_inj = int(round(f * len(seen_idx) / (1.0 - f)))
            rng = np.random.default_rng(args.seed * 1000 + i_f)
            inj = rng.choice(sig_idx_all, size=min(n_inj, len(sig_idx_all)),
                             replace=False)
            sub_idx = np.concatenate([seen_idx, inj])
            sub = TensorDataset(Xtr[sub_idx], ytr[sub_idx])
            tel_loader = DataLoader(sub, batch_size=512, shuffle=False)
            print(f"\n===== {ds}/{args.base} {arm} f={f} "
                  f"({len(inj)}/{n_inj} injected, pool={args.pool}, "
                  f"cut={args.cut}, n_min={args.n_min}) =====", flush=True)
            bb = copy.deepcopy(head)
            cur_means, hist = run_discovery(
                bb, means0.clone(), base_ds=sub, train_eval_loader=tel_loader,
                test_loader=test_loader, seen=seen, holdouts=holdouts,
                dataset_name=ds, rep_weight=rep_weight, sigreg_weight=1.0,
                n_slices=args.n_slices, rounds=args.rounds, ft_epochs=ft_ep,
                names=None, seed=args.seed, pool_score=args.pool,
                cut_rule=args.cut, n_min=args.n_min, on_refuse="skip")
            pur1 = float(hist[0]["purity"]) if hist else float("nan")
            c0 = hist[0].get("cut", {}) if hist else {}
            entry["cut"][fk] = dict(ok=bool(c0.get("ok", True)),
                                    q=float(c0.get("q", float("nan"))),
                                    reason=str(c0.get("reason", "")))
            entry["n_inj"][fk] = int(len(inj))
            te_post, tel_post = collect_embeddings(bb, test_loader)
            tr_post, trl_post = collect_embeddings(bb, tel_full)
            del bb
            torch.cuda.empty_cache()
            anchors = cur_means[n_cls:].detach()
            mpost = np.isin(trl_post, seen)
            anch_seen = torch.as_tensor(
                exp28.class_centroids(tr_post[mpost], trl_post[mpost], seen),
                dtype=torch.float32, device=DEVICE)
            maha_fn, mmd_fn, spk_factory, bg_t, sig_t = toy_fns(
                tr_post, trl_post, te_post, tel_post, seen, holdouts,
                args.seed, args.kernels, steps)
            zt = torch.cat([bg_t, sig_t])
            d_seen = torch.cdist(zt, anch_seen).min(1).values
            d_disc = (torch.cdist(zt, anchors).min(1).values
                      if len(anchors) else torch.zeros_like(d_seen))
            s_eucl = d_seen.cpu().numpy()
            s_ed = (d_seen - d_disc).cpu().numpy() if len(anchors) else None
            nb = len(bg_t)

            def mean_fn(s_bg, s_sig):
                def fn(bi, si, seed_):
                    s = (np.concatenate([s_bg[bi], s_sig[si]]) if len(si)
                         else s_bg[bi])
                    return [float(s.mean())]
                return fn

            suites = [
                ("eucl", mean_fn(s_eucl[:nb], s_eucl[nb:])),
                ("eucl-disc", mean_fn(s_ed[:nb], s_ed[nb:])
                 if s_ed is not None else None),
                ("maha", maha_fn),
                ("mmd", mmd_fn),
                ("sparker", spk_factory(None)),
                ("sparker-anch", spk_factory(anchors)
                 if len(anchors) else None),
            ]
            for tname, fn in suites:
                if fk in entry["z"].get(tname, {}):
                    continue
                if fn is None:
                    entry["z"].setdefault(tname, {})[fk] = None
                    print(f"  [{arm}] {tname} f={f}: no anchors", flush=True)
                    continue
                null_agg, sig_agg = exp146.toys_battery(
                    fn, len(bg_t), len(sig_t), [f], args.n_d, n_null,
                    n_sig_toys, args.seed + i_f, tag=f"{arm}-{tname}")
                z = exp146.z_curve(null_agg, sig_agg)[0]
                entry["z"].setdefault(tname, {})[fk] = z
                print(f"  [{arm}] {tname} f={f}: Z={z:.2f}", flush=True)
            entry["purity1"][fk] = pur1
            results[arm] = entry
            json.dump(results, open(res_path, "w"), indent=1)
            print(f"[done] {ds}/{args.base} {arm} f={f}: purity={pur1:.3f} "
                  f"cut={'ENG' if entry['cut'][fk]['ok'] else 'declined'}",
                  flush=True)
        del head
        torch.cuda.empty_cache()

    print(f"\n== {ds}/{args.base} on-manifold post suite done "
          f"(pool={args.pool}, cut={args.cut}, n_min={args.n_min}) ==")


if __name__ == "__main__":
    main()
