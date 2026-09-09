"""
Experiment 155: discovery mode + residuals on the DIFFUSE corner --
the exp-152 smeared-subset study with the settled discovery loop.

Physical story: a fraction f of the (unlabelled) corpus was hit by one
systematic smearing (Gaussian blur sigma=2 by default).  Implementation:
n = round(f * |seen train|) seen-class TRAIN rows are REPLACED by their
smeared versions (features recomputed through the arm's own trunk), carrying
pseudo-label 10 = "unlabelled anomalous"; the labelled seen reference is the
untouched remainder.  The settled loop (density-ratio pool, derived cut at
n_min=5, skip-on-refuse) then runs head-only over this corpus, and the
updated head is scored with the six exp-148 tests where the signal pool is
the exp-152 smeared TEST subset (same rng, so pools match exp 152) and the
background excludes the smear-source rows.

  --mode one   smeared subset drawn from a single label (default 2)
  --mode all   stratified across all seen labels

Spaces: the archived exp-70 default galaxy10 arms (holdout class 9 excluded
everywhere, seen = 0..8), now INCLUDING the residual children.

    python experiments/155_diffuse_post_suite.py --base lejepa --mode one
    python experiments/155_diffuse_post_suite.py --quick --base dino --mode all --arms supcon-ft --fractions 0.1
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import copy
import importlib
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset

from supersig.config import DEVICE
from supersig.discovery import run_discovery
from supersig.train import collect_embeddings
from supersig.metrics import mahalanobis_novelty
from supersig.sparker import (np_test_stats, median_pairwise, krr_term,
                              mmd2_multi_stats)

exp28 = importlib.import_module("28_concat_residual")
exp37 = importlib.import_module("37_dtd_vit")
exp44 = importlib.import_module("44_transfer_32d")
exp146 = importlib.import_module("146_min_frac_2sigma")
exp152 = importlib.import_module("152_diffuse_shift")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "logs", "exp155")
DS, N_CLS = "galaxy10", 10
SEEN = list(range(9))            # archived default: class 9 excluded
ANOM_LAB = 10                    # pseudo-label for smeared corpus rows
TESTS = ["eucl", "eucl-disc", "maha", "mmd", "sparker", "sparker-anch"]


def smear_feats(trunk, ds_img, idx, kind, sev):
    """Trunk features of images at `idx` after the systematic smear."""
    loader = DataLoader(Subset(ds_img, [int(i) for i in idx]),
                        batch_size=32, shuffle=False, num_workers=4)
    feats = []
    with torch.no_grad():
        for x, _ in loader:
            x = exp152.smear_batch(x, kind, sev)
            feats.append(trunk(x.to(DEVICE)).float().cpu())
    return torch.cat(feats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="lejepa",
                    choices=["dino", "lejepa", "visreg"])
    ap.add_argument("--arms", default="supcon-ft,ss-ft,supcon-ft_res,"
                                      "supcon-ft_resnplm,ss-ft_res")
    ap.add_argument("--mode", default="one", choices=["one", "all"])
    ap.add_argument("--label", type=int, default=2)
    ap.add_argument("--n-smear", type=int, default=1000)
    ap.add_argument("--smear", default="blur",
                    choices=["blur", "noise", "brightness"])
    ap.add_argument("--severity", type=float, default=2.0)
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
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    fracs = [float(x) for x in args.fractions.split(",")]
    n_null = 20 if args.quick else 200
    n_sig_toys = 10 if args.quick else 50
    steps = 60 if args.quick else args.steps
    ft_ep = args.ft_epochs or (1 if args.quick else 5)
    tag = (f"{args.mode}" + (f"_c{args.label}" if args.mode == "one" else "")
           + f"_{args.smear}{args.severity:g}")
    ptag = f"_{args.pool}_{args.cut}_nmin{args.n_min}"
    cfg = dict(n_classes=N_CLS, pair_dist=5.0)
    rep_weight = 20.0 * 45.0 / (N_CLS * (N_CLS - 1) / 2)   # exp 70
    os.makedirs(args.out, exist_ok=True)
    res_path = os.path.join(args.out,
                            f"diffsuite_{DS}_{args.base}_{tag}{ptag}.json")
    results = json.load(open(res_path)) if os.path.exists(res_path) else {}

    # image datasets (eval transform; smear commutes with normalisation)
    train_img = exp44.Galaxy10("train", exp37.TF_EVAL)
    test_img = exp44.Galaxy10("test", exp37.TF_EVAL)
    tr_lab_img = np.array([train_img.df[train_img.lab_col].iloc[i]
                           for i in train_img.keep])
    te_lab_img = np.array([test_img.df[test_img.lab_col].iloc[i]
                           for i in test_img.keep])

    # test-side smear pool: exp-152 selection, same rng
    rng = np.random.default_rng(args.seed)
    if args.mode == "one":
        cand_te = np.where(te_lab_img == args.label)[0]
    else:
        cand_te = np.concatenate([
            rng.permutation(np.where(te_lab_img == c)[0])
            [: max(1, args.n_smear // len(SEEN))] for c in SEEN])
    n_s = min(args.n_smear, len(cand_te))
    smear_idx_te = np.sort(rng.choice(cand_te, size=n_s, replace=False))

    # train-side smear candidates (sources for corpus replacement)
    if args.mode == "one":
        cand_tr = np.where(tr_lab_img == args.label)[0]
    else:
        cand_tr = np.where(np.isin(tr_lab_img, SEEN))[0]
    print(f"[{args.base} {tag}] test smear pool {n_s}; "
          f"train candidates {len(cand_tr)}", flush=True)

    for arm in args.arms.split(","):
        got = exp152.load_space(args.base, arm)
        if got is None:
            print(f"[miss] {arm}", flush=True)
            continue
        trunk, head, bank = got
        trunk = trunk.to(DEVICE).eval()
        (Xtr, ytr), (Xte, yte) = bank["train"], bank["test"]
        Xtr, Xte = Xtr.float(), Xte.float()
        tr_lab, te_lab = ytr.numpy(), yte.numpy()
        assert np.array_equal(tr_lab, tr_lab_img), "train bank order mismatch"
        assert np.array_equal(te_lab, te_lab_img), "test bank order mismatch"

        # smeared trunk feats through THIS arm's trunk (once per arm)
        F_tr_sm = smear_feats(trunk, train_img, cand_tr, args.smear,
                              args.severity)
        F_te_sm = smear_feats(trunk, test_img, smear_idx_te, args.smear,
                              args.severity)
        trunk.cpu()
        del trunk
        torch.cuda.empty_cache()
        head = head.float().to(DEVICE)

        tel_full = DataLoader(TensorDataset(Xtr, ytr), batch_size=512,
                              shuffle=False)
        test_loader = DataLoader(TensorDataset(Xte, yte), batch_size=512,
                                 shuffle=False)
        tr0, _ = collect_embeddings(head, tel_full)
        m = np.isin(tr_lab, SEEN)
        cents = exp28.class_centroids(tr0[m], tr_lab[m], SEEN)
        means0 = exp28.fill_means(cents, SEEN, cfg).detach()
        seen_idx = np.where(m)[0]
        pos_of = {int(g): p for p, g in enumerate(seen_idx)}
        cand_pos = np.array([pos_of[int(i)] for i in cand_tr
                             if int(i) in pos_of])
        entry = results.get(arm, dict(fractions=fracs, mode=args.mode,
                                      label=(args.label if args.mode == "one"
                                             else None), smear=args.smear,
                                      severity=args.severity,
                                      n_smear_test=int(n_s), purity1={},
                                      cut={}, n_rep={}, auc_out={},
                                      z={t: {} for t in TESTS}))
        entry.setdefault("z", {t: {} for t in TESTS})

        bg_keep = np.isin(te_lab, SEEN)
        bg_keep[smear_idx_te] = False

        for i_f, f in enumerate(fracs):
            fk = str(f)
            if all(fk in entry["z"].get(t, {}) for t in TESTS):
                print(f"[skip] {args.base} {arm} f={f}", flush=True)
                continue
            n_rep = int(round(f * len(seen_idx)))
            rng_f = np.random.default_rng(args.seed * 1000 + i_f)
            rep = rng_f.choice(cand_pos, size=min(n_rep, len(cand_pos)),
                               replace=False)
            Xcorp = Xtr[seen_idx].clone()
            ycorp = torch.as_tensor(tr_lab[seen_idx]).clone()
            src_cand_order = {int(c): j for j, c in enumerate(cand_tr)}
            for p in rep:
                gi = int(seen_idx[p])
                Xcorp[p] = F_tr_sm[src_cand_order[gi]]
                ycorp[p] = ANOM_LAB
            sub = TensorDataset(Xcorp, ycorp)
            tel_loader = DataLoader(sub, batch_size=512, shuffle=False)
            print(f"\n===== {args.base} {arm} {tag} f={f} "
                  f"({len(rep)}/{n_rep} replaced, pool={args.pool}, "
                  f"cut={args.cut}, n_min={args.n_min}) =====", flush=True)
            bb = copy.deepcopy(head)
            cur_means, hist = run_discovery(
                bb, means0.clone(), base_ds=sub, train_eval_loader=tel_loader,
                test_loader=test_loader, seen=SEEN, holdouts={ANOM_LAB},
                dataset_name=DS, rep_weight=rep_weight, sigreg_weight=1.0,
                n_slices=args.n_slices, rounds=args.rounds, ft_epochs=ft_ep,
                names=None, seed=args.seed, pool_score=args.pool,
                cut_rule=args.cut, n_min=args.n_min, on_refuse="skip")
            pur1 = float(hist[0]["purity"]) if hist else float("nan")
            c0 = hist[0].get("cut", {}) if hist else {}
            entry["cut"][fk] = dict(ok=bool(c0.get("ok", True)),
                                    q=float(c0.get("q", float("nan"))),
                                    reason=str(c0.get("reason", "")))
            entry["n_rep"][fk] = int(len(rep))

            te_post, tel_post = collect_embeddings(bb, test_loader)
            tr_post, trl_post = collect_embeddings(bb, tel_full)
            sm_post = exp37.embed(bb, F_te_sm).numpy()
            del bb
            torch.cuda.empty_cache()
            anchors = cur_means[N_CLS:].detach()
            mpost = np.isin(trl_post, SEEN)
            anch_seen = torch.as_tensor(
                exp28.class_centroids(tr_post[mpost], trl_post[mpost], SEEN),
                dtype=torch.float32, device=DEVICE)
            bg = te_post[bg_keep]
            TE = np.concatenate([bg, sm_post])
            nb = len(bg)
            bg_t = torch.as_tensor(bg, dtype=torch.float32, device=DEVICE)
            sig_t = torch.as_tensor(sm_post, dtype=torch.float32,
                                    device=DEVICE)
            zt = torch.cat([bg_t, sig_t])
            d_seen = torch.cdist(zt, anch_seen).min(1).values
            d_disc = (torch.cdist(zt, anchors).min(1).values
                      if len(anchors) else torch.zeros_like(d_seen))
            s_eucl = d_seen.cpu().numpy()
            s_ed = (d_seen - d_disc).cpu().numpy() if len(anchors) else None
            lab01 = np.r_[np.zeros(nb), np.ones(len(sm_post))]
            from sklearn.metrics import roc_auc_score
            entry["auc_out"][fk] = float(roc_auc_score(lab01, s_eucl))
            _, pc, _ = mahalanobis_novelty(tr_post, trl_post, TE, SEEN)
            R_t = torch.as_tensor(tr_post[mpost][:20000],
                                  dtype=torch.float32, device=DEVICE)
            g = np.random.default_rng(args.seed)
            R_pool = tr_post[mpost]
            R_mmd = torch.as_tensor(
                R_pool[g.choice(len(R_pool), size=min(5000, len(R_pool)),
                                replace=False)],
                dtype=torch.float32, device=DEVICE)
            med = median_pairwise(bg_t, seed=args.seed)
            sigmas = [0.5 * med, med, 2.0 * med]
            krr = krr_term(R_mmd, sigmas)

            def mean_fn(s_bg, s_sig):
                def fn(bi, si, seed_):
                    s = (np.concatenate([s_bg[bi], s_sig[si]]) if len(si)
                         else s_bg[bi])
                    return [float(s.mean())]
                return fn

            def mmd_fn(bi, si, seed_):
                D = (torch.cat([bg_t[torch.as_tensor(bi, device=DEVICE)],
                                sig_t[torch.as_tensor(si, device=DEVICE)]])
                     if len(si) else bg_t[torch.as_tensor(bi, device=DEVICE)])
                return mmd2_multi_stats(D, R_mmd, sigmas, krr)

            sigma0 = median_pairwise(bg_t, seed=args.seed)

            def spk_factory(mu_init):
                def fn(bi, si, seed_):
                    D = (torch.cat([bg_t[torch.as_tensor(bi, device=DEVICE)],
                                    sig_t[torch.as_tensor(si,
                                                          device=DEVICE)]])
                         if len(si) else
                         bg_t[torch.as_tensor(bi, device=DEVICE)])
                    return np_test_stats(D, R_t, M=args.kernels, steps=steps,
                                         sigma0=sigma0, seed=seed_,
                                         mu_init=mu_init)
                return fn

            suites = [
                ("eucl", mean_fn(s_eucl[:nb], s_eucl[nb:])),
                ("eucl-disc", mean_fn(s_ed[:nb], s_ed[nb:])
                 if s_ed is not None else None),
                ("maha", mean_fn(pc[:nb], pc[nb:])),
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
                    fn, nb, len(sm_post), [f], args.n_d, n_null,
                    n_sig_toys, args.seed + i_f, tag=f"{arm}-{tag}-{tname}")
                z = exp146.z_curve(null_agg, sig_agg)[0]
                entry["z"].setdefault(tname, {})[fk] = z
                print(f"  [{arm}] {tname} f={f}: Z={z:.2f}", flush=True)
            entry["purity1"][fk] = pur1
            results[arm] = entry
            json.dump(results, open(res_path, "w"), indent=1)
            print(f"[done] {args.base} {arm} {tag} f={f}: "
                  f"purity={pur1:.3f} aucOut={entry['auc_out'][fk]:.3f} "
                  f"cut={'ENG' if entry['cut'][fk]['ok'] else 'declined'}",
                  flush=True)
        del head
        torch.cuda.empty_cache()

    print(f"\n== {DS}/{args.base} diffuse post suite ({tag}) done ==")


if __name__ == "__main__":
    main()
