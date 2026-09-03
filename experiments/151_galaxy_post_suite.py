"""
Experiment 151: the post-discovery test suite (exp 148) on Galaxy10 --
discovery mode on the fine-tuned transfer heads, across draws.

Per (backbone, arm, draw, fraction f): inject n = f/(1-f) * |seen train|
held-out TRAIN-bank images into the corpus (exp-70 POST-grid protocol, same
rng; galaxy train holdout pools are small, so the draw clamps at the pool
size and the actual count is archived), run the settled discovery loop on a
copy of the checkpoint HEAD over the cached 768-D trunk bank (exp-70: the
trunk does not move in discovery), then score the updated head space with
the six exp-148 tests (eucl, anchor-aware eucl, maha, mmd, SparKer,
SparKer@anchors) through the exp-146 toy machinery.

Defaults follow the campaign's settled choices: density-ratio pool, the
paper's derived cut at n_min=5 (exp-148 sweep), skip-on-refuse.  Loop
constants are exp 70's: rounds=2, ft_epochs=5, rep_weight=20 (10 classes),
sigreg_weight=1, n_slices=64.

    python experiments/151_galaxy_post_suite.py --base lejepa
    python experiments/151_galaxy_post_suite.py --quick --base dino --arms supcon-ft --draws 3 --fractions 0.1
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
OUT = os.path.join(REPO, "logs", "exp151")
DS, N_CLS = "galaxy10", 10
TESTS = ["eucl", "eucl-disc", "maha", "mmd", "sparker", "sparker-anch"]


def load_arm(base, arm, draw):
    """(head module, Xtr, ytr, Xte, yte) for one (arm, draw), or None."""
    tag = f"_h1_d{draw}"
    bp = os.path.join(DATA, f"tf_feats_{DS}_{base}_ft70_{arm}{tag}.pt")
    ck = os.path.join(CKPT, f"{DS}_ft_{base}_{arm}_seen{tag}.pt")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="lejepa",
                    choices=["dino", "lejepa", "visreg"])
    ap.add_argument("--arms", default="supcon-ft,ss-ft,nplm-sup-ft")
    ap.add_argument("--draws", default="0,3,5,7,8")
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
    ptag = f"_{args.pool}_{args.cut}_nmin{args.n_min}"
    cfg = dict(n_classes=N_CLS, pair_dist=5.0)
    rep_weight = 20.0 * 45.0 / (N_CLS * (N_CLS - 1) / 2)   # exp 70
    os.makedirs(args.out, exist_ok=True)

    for draw in [int(x) for x in args.draws.split(",")]:
        holdouts = holdout_set(DS, N_CLS, nh=1, draw=draw)
        seen = [c for c in range(N_CLS) if c not in holdouts]
        res_path = os.path.join(
            args.out, f"suite_{DS}_{args.base}_d{draw}{ptag}.json")
        results = (json.load(open(res_path)) if os.path.exists(res_path)
                   else {})
        for arm in args.arms.split(","):
            got = load_arm(args.base, arm, draw)
            if got is None:
                print(f"[miss] {args.base} d{draw} {arm}", flush=True)
                continue
            head, Xtr, ytr, Xte, yte = got
            tr_lab, te_lab = ytr.numpy(), yte.numpy()
            tel_full = DataLoader(TensorDataset(Xtr, ytr), batch_size=512,
                                  shuffle=False)
            test_loader = DataLoader(TensorDataset(Xte, yte), batch_size=512,
                                     shuffle=False)
            tr0, _ = collect_embeddings(head, tel_full)
            m = np.isin(tr_lab, seen)
            cents = exp28.class_centroids(tr0[m], tr_lab[m], seen)
            means0 = exp28.fill_means(cents, seen, cfg).detach()
            seen_idx = np.where(m)[0]
            sig_idx_all = np.where(np.isin(tr_lab, list(holdouts)))[0]
            entry = results.get(arm, dict(fractions=fracs, purity1={},
                                          cut={}, n_inj={},
                                          z={t: {} for t in TESTS}))
            for i_f, f in enumerate(fracs):
                fk = str(f)
                if all(fk in entry["z"].get(t, {}) for t in TESTS):
                    print(f"[skip] {args.base} d{draw} {arm} f={f}",
                          flush=True)
                    continue
                n_inj = int(round(f * len(seen_idx) / (1.0 - f)))
                rng = np.random.default_rng(args.seed * 1000 + i_f)
                inj = rng.choice(sig_idx_all,
                                 size=min(n_inj, len(sig_idx_all)),
                                 replace=False)
                sub_idx = np.concatenate([seen_idx, inj])
                sub = TensorDataset(Xtr[sub_idx], ytr[sub_idx])
                tel_loader = DataLoader(sub, batch_size=512, shuffle=False)
                print(f"\n===== {args.base} d{draw} {arm} f={f} "
                      f"({len(inj)}/{n_inj} injected, pool={args.pool}, "
                      f"cut={args.cut}, n_min={args.n_min}) =====",
                      flush=True)
                bb = copy.deepcopy(head)
                cur_means, hist = run_discovery(
                    bb, means0.clone(), base_ds=sub,
                    train_eval_loader=tel_loader, test_loader=test_loader,
                    seen=seen, holdouts=holdouts, dataset_name=DS,
                    rep_weight=rep_weight, sigreg_weight=1.0,
                    n_slices=args.n_slices, rounds=args.rounds,
                    ft_epochs=ft_ep, names=None, seed=args.seed,
                    pool_score=args.pool, cut_rule=args.cut,
                    n_min=args.n_min, on_refuse="skip")
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
                bg_mask = np.isin(tel_post, seen)
                sig_mask = np.isin(tel_post, list(holdouts))
                R_t = torch.as_tensor(
                    tr_post[np.isin(trl_post, seen)][:20000],
                    dtype=torch.float32, device=DEVICE)
                bg_t = torch.as_tensor(te_post[bg_mask],
                                       dtype=torch.float32, device=DEVICE)
                sig_t = torch.as_tensor(te_post[sig_mask],
                                        dtype=torch.float32, device=DEVICE)
                anchors = cur_means[N_CLS:].detach()

                mpost = np.isin(trl_post, seen)
                anch_seen = torch.as_tensor(
                    exp28.class_centroids(tr_post[mpost], trl_post[mpost],
                                          seen),
                    dtype=torch.float32, device=DEVICE)
                zt = torch.cat([bg_t, sig_t])
                d_seen = torch.cdist(zt, anch_seen).min(1).values
                d_disc = (torch.cdist(zt, anchors).min(1).values
                          if len(anchors) else torch.zeros_like(d_seen))
                s_eucl = d_seen.cpu().numpy()
                s_ed = ((d_seen - d_disc).cpu().numpy() if len(anchors)
                        else None)
                nb = len(bg_t)
                _, pc, _ = mahalanobis_novelty(tr_post, trl_post, te_post,
                                               seen)
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
                    D = (torch.cat([bg_t[torch.as_tensor(bg_idx,
                                                         device=DEVICE)],
                                    sig_t[torch.as_tensor(sig_idx,
                                                          device=DEVICE)]])
                         if len(sig_idx) else
                         bg_t[torch.as_tensor(bg_idx, device=DEVICE)])
                    return mmd2_multi_stats(D, R_mmd, sigmas, krr)

                sigma0 = median_pairwise(bg_t, seed=args.seed)

                def spk_factory(mu_init):
                    def fn(bg_idx, sig_idx, seed):
                        D = (torch.cat([bg_t[torch.as_tensor(bg_idx,
                                                             device=DEVICE)],
                                        sig_t[torch.as_tensor(
                                            sig_idx, device=DEVICE)]])
                             if len(sig_idx) else
                             bg_t[torch.as_tensor(bg_idx, device=DEVICE)])
                        return np_test_stats(D, R_t, M=args.kernels,
                                             steps=steps, sigma0=sigma0,
                                             seed=seed, mu_init=mu_init)
                    return fn

                suites = [
                    ("eucl", mean_fn(s_eucl[:nb], s_eucl[nb:])),
                    ("eucl-disc", mean_fn(s_ed[:nb], s_ed[nb:])
                     if s_ed is not None else None),
                    ("maha", mean_fn(s_maha_bg, s_maha_sig)),
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
                        print(f"  [{arm}] {tname} f={f}: no anchors",
                              flush=True)
                        continue
                    null_agg, sig_agg = exp146.toys_battery(
                        fn, len(bg_t), len(sig_t), [f], args.n_d, n_null,
                        n_sig_toys, args.seed + i_f,
                        tag=f"{arm}-{tname}")
                    z = exp146.z_curve(null_agg, sig_agg)[0]
                    entry["z"].setdefault(tname, {})[fk] = z
                    print(f"  [{arm}] {tname} f={f}: Z={z:.2f}", flush=True)
                entry["purity1"][fk] = pur1
                results[arm] = entry
                json.dump(results, open(res_path, "w"), indent=1)
                print(f"[done] {args.base} d{draw} {arm} f={f}: "
                      f"purity={pur1:.3f} "
                      f"cut={'ENG' if entry['cut'][fk]['ok'] else 'declined'}",
                      flush=True)
            del head
            torch.cuda.empty_cache()

    print(f"\n== {DS}/{args.base} post-discovery suite done "
          f"(pool={args.pool}, cut={args.cut}, n_min={args.n_min}) ==")


if __name__ == "__main__":
    main()
