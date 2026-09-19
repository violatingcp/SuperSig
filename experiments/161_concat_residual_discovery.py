"""
Experiment 161: f*(2sigma) on the CONCAT [supervised parent || residual]
space, with discovery run on the RESIDUAL child only.

Motivation (2026-09): discovery cannot fine-tune a two-encoder concat
coherently, but it CAN fine-tune the single residual child.  So: run the
settled s/sqrt(b) discovery loop on the residual child alone, then evaluate
the f* battery on the concatenation of the FROZEN supervised parent
embedding (strong labelled class geometry) with the discovery-refined
residual embedding.  This keeps the parent's separability and adds the
discovery gain, without ever fine-tuning the concat.

Per (parent, child, fraction f): inject f into the train corpus (exp-151
protocol), discover on the residual child head over its trunk bank, then
build
    concat_pre  = [ parent(frozen) || residual(frozen child) ]
    concat_post = [ parent(frozen) || residual(post-discovery child) ]
and score both with eucl / eucl-disc / maha / mmd / sparker / sparker-anch
through the exp-146 toy machinery.  The discovered anchor in concat is
[ mean parent-embedding of the pooled cluster members || residual anchor ].

    python experiments/161_concat_residual_discovery.py --base lejepa --draws 0 \
        --pairs supcon-ft:supcon-ft_res
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
exp37 = importlib.import_module("37_dtd_vit")
exp43 = importlib.import_module("43_dtd_finetune")
exp146 = importlib.import_module("146_min_frac_2sigma")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA, CKPT = os.path.join(REPO, "data"), os.path.join(REPO, "checkpoints")
OUT = os.path.join(REPO, "logs", "exp161")
DS, N_CLS = "galaxy10", 10
TESTS = ["eucl", "eucl-disc", "maha", "mmd", "sparker", "sparker-anch"]


def load_head(base, arm, draw):
    tag = f"_h1_d{draw}"
    bp = os.path.join(DATA, f"tf_feats_{DS}_{base}_ft70_{arm}{tag}.pt")
    ck = os.path.join(CKPT, f"{DS}_ft_{base}_{arm}_seen{tag}.pt")
    if not (os.path.exists(bp) and os.path.exists(ck)):
        return None
    sd = torch.load(ck, map_location=DEVICE)
    ed = next(v.shape[0] for k, v in reversed(list(sd.items()))
              if k.startswith("head") and v.dim() == 2)
    mod = exp43.FineTuneModel(base, ed)
    mod.load_state_dict(sd)
    head = copy.deepcopy(mod.head).float().to(DEVICE)
    del mod
    torch.cuda.empty_cache()
    b = torch.load(bp, map_location="cpu")
    (Xtr, ytr), (Xte, yte) = b["train"], b["test"]
    return head, Xtr.float(), ytr, Xte.float(), yte


def toy_fns(tr, trl, te, tel, seen, holdouts, seed, kernels, steps):
    """maha/mmd/sparker factories on a fixed concat embedding."""
    bgm = np.isin(tel, seen); sgm = np.isin(tel, list(holdouts))
    R_t = torch.as_tensor(tr[np.isin(trl, seen)][:20000],
                          dtype=torch.float32, device=DEVICE)
    bg_t = torch.as_tensor(te[bgm], dtype=torch.float32, device=DEVICE)
    sig_t = torch.as_tensor(te[sgm], dtype=torch.float32, device=DEVICE)
    _, pc, _ = mahalanobis_novelty(tr, trl, te, seen)
    s_bg, s_sig = pc[bgm], pc[sgm]

    def maha_fn(bi, si, sd):
        s = np.concatenate([s_bg[bi], s_sig[si]]) if len(si) else s_bg[bi]
        return [float(s.mean())]

    g = np.random.default_rng(seed)
    Rp = tr[np.isin(trl, seen)]
    R_mmd = torch.as_tensor(Rp[g.choice(len(Rp), size=min(5000, len(Rp)),
                                        replace=False)],
                            dtype=torch.float32, device=DEVICE)
    med = median_pairwise(bg_t, seed=seed); sig3 = [0.5 * med, med, 2 * med]
    krr = krr_term(R_mmd, sig3)

    def mmd_fn(bi, si, sd):
        D = (torch.cat([bg_t[torch.as_tensor(bi, device=DEVICE)],
                        sig_t[torch.as_tensor(si, device=DEVICE)]])
             if len(si) else bg_t[torch.as_tensor(bi, device=DEVICE)])
        return mmd2_multi_stats(D, R_mmd, sig3, krr)

    sigma0 = median_pairwise(bg_t, seed=seed)

    def spk_factory(mu):
        def fn(bi, si, sd):
            D = (torch.cat([bg_t[torch.as_tensor(bi, device=DEVICE)],
                            sig_t[torch.as_tensor(si, device=DEVICE)]])
                 if len(si) else bg_t[torch.as_tensor(bi, device=DEVICE)])
            return np_test_stats(D, R_t, M=kernels, steps=steps,
                                 sigma0=sigma0, seed=sd, mu_init=mu)
        return fn
    return maha_fn, mmd_fn, spk_factory, bg_t, sig_t, bgm, sgm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="lejepa")
    ap.add_argument("--draws", default="0")
    ap.add_argument("--pairs", default="supcon-ft:supcon-ft_res,"
                                       "ss-ft:ss-ft_res")
    ap.add_argument("--fractions", default="0.01,0.02,0.03,0.05,0.1")
    ap.add_argument("--n-min", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--ft-epochs", type=int, default=5)
    ap.add_argument("--n-slices", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-d", type=int, default=5000)
    ap.add_argument("--kernels", type=int, default=16)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    fracs = [float(x) for x in args.fractions.split(",")]
    pairs = [tuple(p.split(":")) for p in args.pairs.split(",")]
    rep_weight = 20.0 * 45.0 / (N_CLS * (N_CLS - 1) / 2)
    cfg = dict(n_classes=N_CLS, pair_dist=5.0)
    os.makedirs(args.out, exist_ok=True)

    for draw in [int(x) for x in args.draws.split(",")]:
        holdouts = holdout_set(DS, N_CLS, nh=1, draw=draw)
        seen = [c for c in range(N_CLS) if c not in holdouts]
        res_path = os.path.join(args.out,
                                f"concatres_{DS}_{args.base}_d{draw}.json")
        results = json.load(open(res_path)) if os.path.exists(res_path) else {}
        for parent, child in pairs:
            gp, gc = load_head(args.base, parent, draw), \
                load_head(args.base, child, draw)
            if gp is None or gc is None:
                print(f"[miss] {parent}/{child} d{draw}", flush=True)
                continue
            pH, Xtr_p, ytr, Xte_p, yte = gp
            cH, Xtr_c, _, Xte_c, _ = gc
            tr_lab, te_lab = ytr.numpy(), yte.numpy()
            # frozen parent embeddings (identical pre and post)
            telp = DataLoader(TensorDataset(Xtr_p, ytr), batch_size=512)
            tep = DataLoader(TensorDataset(Xte_p, yte), batch_size=512)
            P_tr, _ = collect_embeddings(pH, telp)
            P_te, _ = collect_embeddings(pH, tep)
            # residual child trunk bank + loaders (discovery runs on this head)
            telc = DataLoader(TensorDataset(Xtr_c, ytr), batch_size=512)
            tec = DataLoader(TensorDataset(Xte_c, yte), batch_size=512)
            R0_tr, _ = collect_embeddings(cH, telc)
            m = np.isin(tr_lab, seen)
            cents = exp28.class_centroids(R0_tr[m], tr_lab[m], seen)
            means0 = exp28.fill_means(cents, seen, cfg).detach()
            seen_idx = np.where(m)[0]
            sig_idx_all = np.where(np.isin(tr_lab, list(holdouts)))[0]
            key = f"{parent}+{child}"
            entry = results.get(key, dict(fractions=fracs,
                                          pre={t: {} for t in TESTS},
                                          post={t: {} for t in TESTS},
                                          cut={}))

            for i_f, f in enumerate(fracs):
                fk = str(f)
                if fk in entry["post"]["eucl"]:
                    print(f"[skip] {key} f={f}", flush=True)
                    continue
                n_inj = int(round(f * len(seen_idx) / (1 - f)))
                rng = np.random.default_rng(args.seed * 1000 + i_f)
                inj = rng.choice(sig_idx_all,
                                 size=min(n_inj, len(sig_idx_all)),
                                 replace=False)
                sub_idx = np.concatenate([seen_idx, inj])
                sub = TensorDataset(Xtr_c[sub_idx], ytr[sub_idx])
                sub_loader = DataLoader(sub, batch_size=512, shuffle=False)
                bb = copy.deepcopy(cH)
                cur_means, hist = run_discovery(
                    bb, means0.clone(), base_ds=sub,
                    train_eval_loader=sub_loader, test_loader=tec, seen=seen,
                    holdouts=holdouts, dataset_name=DS, rep_weight=rep_weight,
                    sigreg_weight=1.0, n_slices=args.n_slices,
                    rounds=args.rounds, ft_epochs=args.ft_epochs, names=None,
                    seed=args.seed, pool_score="np", cut_rule="ssb",
                    n_min=args.n_min, on_refuse="skip")
                c0 = hist[0].get("cut", {}) if hist else {}
                entry["cut"][fk] = dict(ok=bool(c0.get("ok", True)),
                                        pur=float(hist[0]["purity"])
                                        if hist else float("nan"))
                Rpost_tr, trl2 = collect_embeddings(bb, telc)
                Rpost_te, tel2 = collect_embeddings(bb, tec)
                del bb; torch.cuda.empty_cache()
                anch_res = cur_means[N_CLS:].detach().cpu().numpy()

                # build concat spaces (parent frozen; residual pre / post)
                C_pre_tr = np.concatenate([P_tr, R0_tr], 1)
                C_pre_te = np.concatenate([P_te,
                                           collect_embeddings(cH, tec)[0]], 1)
                C_post_tr = np.concatenate([P_tr, Rpost_tr], 1)
                C_post_te = np.concatenate([P_te, Rpost_te], 1)

                # discovered anchor in concat: [parent centroid of cluster || res anchor]
                anch_concat = None
                if len(anch_res):
                    Rp_t = torch.as_tensor(Rpost_tr, dtype=torch.float32,
                                           device=DEVICE)
                    Ar = torch.as_tensor(anch_res, dtype=torch.float32,
                                         device=DEVICE)
                    assign = torch.cdist(Rp_t, Ar).argmin(1).cpu().numpy()
                    rows = []
                    for k in range(len(anch_res)):
                        mk = assign == k
                        php = (P_tr[mk].mean(0) if mk.any()
                               else P_tr.mean(0))
                        rows.append(np.concatenate([php, anch_res[k]]))
                    anch_concat = np.asarray(rows, np.float32)

                for state, (Ctr, Cte) in (("pre", (C_pre_tr, C_pre_te)),
                                          ("post", (C_post_tr, C_post_te))):
                    A = anch_concat if state == "post" else None
                    maha_fn, mmd_fn, spk, bg_t, sig_t, bgm, sgm = toy_fns(
                        Ctr, tr_lab, Cte, te_lab, seen, holdouts, args.seed,
                        args.kernels, 60 if False else args.steps)
                    ms = np.isin(tr_lab, seen)
                    ancs = torch.as_tensor(
                        exp28.class_centroids(Ctr[ms], tr_lab[ms], seen),
                        dtype=torch.float32, device=DEVICE)
                    zt = torch.cat([bg_t, sig_t])
                    d_seen = torch.cdist(zt, ancs).min(1).values
                    if A is not None and len(A):
                        At = torch.as_tensor(A, dtype=torch.float32,
                                             device=DEVICE)
                        d_disc = torch.cdist(zt, At).min(1).values
                        s_ed = (d_seen - d_disc).cpu().numpy()
                    else:
                        s_ed = None
                    s_eu = d_seen.cpu().numpy(); nb = len(bg_t)

                    def mean_fn(a, b):
                        def fn(bi, si, sd):
                            s = (np.concatenate([a[bi], b[si]]) if len(si)
                                 else a[bi])
                            return [float(s.mean())]
                        return fn
                    suites = [("eucl", mean_fn(s_eu[:nb], s_eu[nb:])),
                              ("eucl-disc", mean_fn(s_ed[:nb], s_ed[nb:])
                               if s_ed is not None else None),
                              ("maha", maha_fn), ("mmd", mmd_fn),
                              ("sparker", spk(None)),
                              ("sparker-anch",
                               spk(torch.as_tensor(A, dtype=torch.float32,
                                                   device=DEVICE))
                               if (A is not None and len(A)) else None)]
                    for tn, fn in suites:
                        if fn is None:
                            entry[state][tn][fk] = None
                            continue
                        na, sa = exp146.toys_battery(
                            fn, len(bg_t), len(sig_t), [f], args.n_d, 200, 50,
                            args.seed + i_f, tag=f"{key}-{state}-{tn}")
                        entry[state][tn][fk] = exp146.z_curve(na, sa)[0]
                    print(f"  [{key}] f={f} {state}: "
                          f"eucl-disc Z={entry[state]['eucl-disc'].get(fk)} "
                          f"mmd Z={entry[state]['mmd'].get(fk)}", flush=True)
                results[key] = entry
                json.dump(results, open(res_path, "w"), indent=1)
                print(f"[done] {key} f={f} pur={entry['cut'][fk]['pur']:.3f}",
                      flush=True)
    print("\nexp161 done.", flush=True)


if __name__ == "__main__":
    main()
