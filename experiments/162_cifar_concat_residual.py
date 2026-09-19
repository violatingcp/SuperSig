"""
Experiment 162: f*(2sigma) on CIFAR, discovery on the RESIDUAL child, scored
on the residual-only space AND the concat [frozen parent || residual] --
side by side, pre and post discovery (CIFAR twin of exp 161).

Per (parent, child, fraction f): inject f held-out train images into the
seen corpus (exp-148 protocol), run the s/sqrt(b) discovery loop on the
residual child net, then score BOTH spaces with the six tests through the
exp-146 toys:
    resid  = child(x)                 (the residual space; == exp148 arm)
    concat = [ parent(x) || child(x) ] (frozen parent + refined residual)
`pre` uses the frozen child (no anchors -> eucl/maha/mmd/sparker only);
`post` uses the discovery-refined child + discovered anchors (all six).

    python experiments/162_cifar_concat_residual.py --dim 10 --holdout 4 \
        --pairs supcon:supcon-res
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
from supersig.discovery import run_discovery
from supersig.models import CIFARResNetBackbone
from supersig.recipes import recipe
from supersig.metrics import mahalanobis_novelty
from supersig.sparker import (np_test_stats, median_pairwise, krr_term,
                              mmd2_multi_stats)
from supersig.train import collect_embeddings

exp28 = importlib.import_module("28_concat_residual")
exp146 = importlib.import_module("146_min_frac_2sigma")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(REPO, "checkpoints")
OUT = os.path.join(REPO, "logs", "exp162")
TESTS = ["eucl", "eucl-disc", "maha", "mmd", "sparker", "sparker-anch"]


def load_net(ds, arm, holdout, dim, cfg):
    fn = f"scratch_{arm.replace('-', '_')}_{ds}_{dim}d" \
         f"{'' if holdout == 4 else f'_h{holdout}'}.pt"
    p = os.path.join(CKPT, fn)
    if not os.path.exists(p):
        print(f"[miss] {arm}: {fn}", flush=True); return None
    net = CIFARResNetBackbone(dim, arch=cfg["arch"], pretrain=None).to(DEVICE)
    ck = torch.load(p, map_location=DEVICE)
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
    ap.add_argument("--discover", default="residual", choices=["residual", "parent"])
    ap.add_argument("--fractions", default="0.006,0.01,0.02,0.03,0.05,0.1")
    ap.add_argument("--n-min", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--ft-epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-d", type=int, default=5000)
    ap.add_argument("--kernels", type=int, default=16)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    ds = args.dataset; cfg = recipe(ds, emb_dim=args.dim)
    n_cls = cfg["n_classes"]; holdouts = {args.holdout}
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
    dtag = "" if args.discover == "residual" else "_discparent"
    res_path = os.path.join(args.out,
                            f"concatres_{ds}_h{args.holdout}_e{args.dim}{dtag}.json")
    results = json.load(open(res_path)) if os.path.exists(res_path) else {}

    def z_of(fn, bg_t, sig_t, f, seed):
        na, sa = exp146.toys_battery(fn, len(bg_t), len(sig_t), [f],
                                     args.n_d, 200, 50, seed, tag="")
        return exp146.z_curve(na, sa)[0]

    for parent, child in pairs:
        pn = load_net(ds, parent, args.holdout, args.dim, cfg)
        cn = load_net(ds, child, args.holdout, args.dim, cfg)
        if pn is None or cn is None:
            continue
        dn, fn_net = (pn, cn) if args.discover == "parent" else (cn, pn)
        F_tr, tr_lab = collect_embeddings(fn_net, tel)      # frozen half
        F_te, te_lab = collect_embeddings(fn_net, test_loader)
        R0_tr, _ = collect_embeddings(dn, tel)              # discovery half (pre)
        R0_te, _ = collect_embeddings(dn, test_loader)
        m = np.isin(tr_lab, seen)
        cents = exp28.class_centroids(R0_tr[m], tr_lab[m], seen)
        means0 = exp28.fill_means(cents, seen, cfg).detach()
        def cat(dtr, dte):   # parent-first concat order
            if args.discover == "parent":
                return (np.concatenate([dtr, F_tr], 1),
                        np.concatenate([dte, F_te], 1))
            return (np.concatenate([F_tr, dtr], 1),
                    np.concatenate([F_te, dte], 1))
        key = f"{parent}+{child}"
        entry = results.get(key, dict(fractions=fracs,
                                      resid={"pre": {t: {} for t in TESTS},
                                             "post": {t: {} for t in TESTS}},
                                      concat={"pre": {t: {} for t in TESTS},
                                              "post": {t: {} for t in TESTS}},
                                      cut={}))

        for i_f, f in enumerate(fracs):
            fk = str(f)
            if fk in entry["concat"]["post"]["eucl"]:
                print(f"[skip] {key} f={f}", flush=True); continue
            n_inj = int(round(f * len(seen_idx) / (1 - f)))
            rng = np.random.default_rng(args.seed * 1000 + i_f)
            inj = rng.choice(sig_idx_all, size=min(n_inj, len(sig_idx_all)),
                             replace=False)
            sub = Subset(base_ds, np.concatenate([seen_idx, inj]).tolist())
            sub_loader = DataLoader(sub, batch_size=256, shuffle=False,
                                    num_workers=2)
            bb = copy.deepcopy(dn)
            cur_means, hist = run_discovery(
                bb, means0.clone(), base_ds=sub, train_eval_loader=sub_loader,
                test_loader=test_loader, seen=seen, holdouts=holdouts,
                dataset_name=ds, rep_weight=cfg["rep_weight"],
                sigreg_weight=cfg["sigreg_weight"], n_slices=cfg["n_slices"],
                rounds=args.rounds, ft_epochs=ft_ep, names=None,
                seed=args.seed, pool_score="np", cut_rule="ssb",
                n_min=args.n_min, on_refuse="skip")
            c0 = hist[0].get("cut", {}) if hist else {}
            entry["cut"][fk] = dict(ok=bool(c0.get("ok", True)),
                                    pur=float(hist[0]["purity"]) if hist
                                    else float("nan"))
            Rp_tr, trl2 = collect_embeddings(bb, tel)
            Rp_te, tel2 = collect_embeddings(bb, test_loader)
            del bb; torch.cuda.empty_cache()
            A_res = cur_means[n_cls:].detach().cpu().numpy()
            A_res_concat = A_concat_post = None
            if len(A_res):
                asg = torch.cdist(torch.as_tensor(Rp_tr, dtype=torch.float32,
                                                  device=DEVICE),
                                  torch.as_tensor(A_res, dtype=torch.float32,
                                                  device=DEVICE)).argmin(1).cpu().numpy()
                A_res_concat = A_res
                fcent = np.asarray([F_tr[asg == k].mean(0) if (asg == k).any()
                                    else F_tr.mean(0) for k in range(len(A_res))])
                if args.discover == "parent":
                    A_concat_post = np.concatenate([A_res, fcent], 1).astype(np.float32)
                else:
                    A_concat_post = np.concatenate([fcent, A_res], 1).astype(np.float32)
            cpre_tr, cpre_te = cat(R0_tr, R0_te)
            cpost_tr, cpost_te = cat(Rp_tr, Rp_te)
            spaces = {
                "resid": {"pre": (R0_tr, R0_te, None),
                          "post": (Rp_tr, Rp_te, A_res_concat)},
                "concat": {"pre": (cpre_tr, cpre_te, None),
                           "post": (cpost_tr, cpost_te, A_concat_post)},
            }
            for sp, st in spaces.items():
                for state, (Ctr, Cte, A) in st.items():
                    _, det = _prep(Ctr, tr_lab, Cte, te_lab, seen, holdouts,
                                   A, args)
                    bg_t, sig_t = det["bg_t"], det["sig_t"]
                    nb = det["nb"]
                    fns = {
                        "eucl": det["meanf"](det["s_eu"][:nb], det["s_eu"][nb:]),
                        "eucl-disc": (det["meanf"](det["s_ed"][:nb],
                                                   det["s_ed"][nb:])
                                      if det["s_ed"] is not None else None),
                        "maha": det["meanf"](det["s_mb"], det["s_ms"]),
                        "mmd": det["mmd_fn"], "sparker": det["spk"](None),
                        "sparker-anch": (det["spk"](torch.as_tensor(
                            A, dtype=torch.float32, device=DEVICE))
                            if (A is not None and len(A)) else None)}
                    for tn, fn in fns.items():
                        if fn is None:
                            entry[sp][state][tn][fk] = None; continue
                        entry[sp][state][tn][fk] = z_of(fn, bg_t, sig_t, f,
                                                        args.seed + i_f)
                    print(f"  [{key}] f={f} {sp}/{state}: "
                          f"ed={entry[sp][state]['eucl-disc'].get(fk)} "
                          f"mmd={entry[sp][state]['mmd'].get(fk)} "
                          f"spk={entry[sp][state]['sparker'].get(fk)}",
                          flush=True)
            results[key] = entry
            json.dump(results, open(res_path, "w"), indent=1)
            print(f"[done] {key} f={f} pur={entry['cut'][fk]['pur']:.3f}",
                  flush=True)
    print("\nexp162 done.", flush=True)


def _prep(Ctr, tr_lab, Cte, te_lab, seen, holdouts, A, args):
    bgm = np.isin(te_lab, seen); sgm = np.isin(te_lab, list(holdouts))
    R_t = torch.as_tensor(Ctr[np.isin(tr_lab, seen)][:20000],
                          dtype=torch.float32, device=DEVICE)
    bg_t = torch.as_tensor(Cte[bgm], dtype=torch.float32, device=DEVICE)
    sig_t = torch.as_tensor(Cte[sgm], dtype=torch.float32, device=DEVICE)
    ms = np.isin(tr_lab, seen)
    ancs = torch.as_tensor(exp28.class_centroids(Ctr[ms], tr_lab[ms], seen),
                           dtype=torch.float32, device=DEVICE)
    zt = torch.cat([bg_t, sig_t]); nb = len(bg_t)
    d_seen = torch.cdist(zt, ancs).min(1).values
    s_ed = None
    if A is not None and len(A):
        At = torch.as_tensor(A, dtype=torch.float32, device=DEVICE)
        s_ed = (d_seen - torch.cdist(zt, At).min(1).values).cpu().numpy()
    s_eu = d_seen.cpu().numpy()
    _, pc, _ = mahalanobis_novelty(Ctr, tr_lab, Cte, seen)
    g = np.random.default_rng(args.seed); Rp = Ctr[np.isin(tr_lab, seen)]
    R_mmd = torch.as_tensor(Rp[g.choice(len(Rp), size=min(5000, len(Rp)),
                                        replace=False)],
                            dtype=torch.float32, device=DEVICE)
    med = median_pairwise(bg_t, seed=args.seed); s3 = [0.5*med, med, 2*med]
    krr = krr_term(R_mmd, s3); sig0 = med

    def meanf(a, b):
        def fn(bi, si, sd):
            s = np.concatenate([a[bi], b[si]]) if len(si) else a[bi]
            return [float(s.mean())]
        return fn

    def mmd_fn(bi, si, sd):
        D = (torch.cat([bg_t[torch.as_tensor(bi, device=DEVICE)],
                        sig_t[torch.as_tensor(si, device=DEVICE)]])
             if len(si) else bg_t[torch.as_tensor(bi, device=DEVICE)])
        return mmd2_multi_stats(D, R_mmd, s3, krr)

    def spk(mu):
        def fn(bi, si, sd):
            D = (torch.cat([bg_t[torch.as_tensor(bi, device=DEVICE)],
                            sig_t[torch.as_tensor(si, device=DEVICE)]])
                 if len(si) else bg_t[torch.as_tensor(bi, device=DEVICE)])
            return np_test_stats(D, R_t, M=args.kernels, steps=args.steps,
                                 sigma0=sig0, seed=sd, mu_init=mu)
        return fn
    return None, dict(bg_t=bg_t, sig_t=sig_t, nb=nb, s_eu=s_eu, s_ed=s_ed,
                      s_mb=pc[bgm], s_ms=pc[sgm], meanf=meanf, mmd_fn=mmd_fn,
                      spk=spk)


if __name__ == "__main__":
    main()
