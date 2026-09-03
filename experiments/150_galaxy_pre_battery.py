"""
Experiment 150: the sigma-currency pre-discovery battery (exp 146) on
Galaxy10 -- every fine-tuned space and every residual construction, across
the five single-holdout draws.

Spaces per (backbone, draw), from the exp-70/71 archives (head applied to
the per-arm trunk feature bank, exp-71 convention):

  fine-tuned arms : supcon-ft, ss-ft, nplm-sup-ft, simclr-ft,
                    sigreg-ssl-ft, nplm-bil-ft
  transductive    : gcd-ft, gcd-sigreg-ft (trained with the unlabelled
                    corpus, novel images included -- reported but flagged)
  residuals       : supcon-ft_res, supcon-ft_resnplm, ss-ft_res as
                    (residual) and (concat) -- the constructions that were
                    trained across draws (exp 126)

Per space: probe / eucl / mahaT / per-event on the frozen embedding, then
the three dataset-level tests (Maha, MMD, SparKer) through the exp-146 toy
machinery (200 null / 50 signal toys, N_D=5000, median expected
significance), giving f*(2sigma) per test.  Reference R = the labelled
train-split seen-class embeddings; bg/sig pools = the corpus split.

    python experiments/150_galaxy_pre_battery.py --base dino
    python experiments/150_galaxy_pre_battery.py --quick --base dino --arms supcon-ft --draws 3
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
from supersig.sparker import median_pairwise, krr_term, mmd2_multi_stats, np_test_stats
from supersig.holdouts import holdout_set

exp28 = importlib.import_module("28_concat_residual")
exp29 = importlib.import_module("29_residual_finetune")
exp30 = importlib.import_module("30_power_curves")
exp37 = importlib.import_module("37_dtd_vit")
exp43 = importlib.import_module("43_dtd_finetune")
exp146 = importlib.import_module("146_min_frac_2sigma")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA, CKPT = os.path.join(REPO, "data"), os.path.join(REPO, "checkpoints")
OUT = os.path.join(REPO, "logs", "exp150")
DS, N_CLS = "galaxy10", 10
FT_ARMS = ["supcon-ft", "ss-ft", "nplm-sup-ft", "simclr-ft",
           "sigreg-ssl-ft", "nplm-bil-ft"]
GCD_ARMS = ["gcd-ft", "gcd-sigreg-ft"]
RES_PARENTS = {"supcon-ft": ["res", "resnplm"], "ss-ft": ["res"]}


def head_embs(base, arm, draw, emb_dim_cache={}):
    """Load bank + checkpoint head for one arm; return (Htr, ytr, Hte, yte)."""
    tag = f"_h1_d{draw}"
    bp = os.path.join(DATA, f"tf_feats_{DS}_{base}_ft70_{arm}{tag}.pt")
    ck = os.path.join(CKPT, f"{DS}_ft_{base}_{arm}_seen{tag}.pt")
    if not (os.path.exists(bp) and os.path.exists(ck)):
        return None
    sd = torch.load(ck, map_location=DEVICE)
    emb_dim = sd["head.net.4.weight"].shape[0] if "head.net.4.weight" in sd \
        else next(v.shape[0] for k, v in reversed(list(sd.items()))
                  if k.startswith("head") and v.dim() == 2)
    mod = exp43.FineTuneModel(base, emb_dim)
    mod.load_state_dict(sd)
    head = mod.head.float()
    b = torch.load(bp, map_location="cpu")
    (Xtr, ytr), (Xte, yte) = b["train"], b["test"]
    Htr = exp37.embed(head, Xtr.float()).numpy()
    Hte = exp37.embed(head, Xte.float()).numpy()
    del mod
    torch.cuda.empty_cache()
    return Htr, ytr.numpy(), Hte, yte.numpy()


def space_list(base, draw):
    """[(label, callable -> (Htr, ytr, Hte, yte) or None)]"""
    out = []
    for arm in FT_ARMS + GCD_ARMS:
        out.append((arm, lambda a=arm: head_embs(base, a, draw)))
    for parent, objs in RES_PARENTS.items():
        for obj in objs:
            def build(p=parent, o=obj, use="residual"):
                pe = head_embs(base, p, draw)
                ce = head_embs(base, f"{p}_{o}", draw)
                if pe is None or ce is None:
                    return None
                if use == "residual":
                    return ce
                return (np.concatenate([pe[0], ce[0]], 1), pe[1],
                        np.concatenate([pe[2], ce[2]], 1), pe[3])
            out.append((f"{parent}->{obj} (residual)",
                        lambda p=parent, o=obj: build(p, o, "residual")))
            out.append((f"{parent}->{obj} (concat)",
                        lambda p=parent, o=obj: build(p, o, "concat")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="dino",
                    choices=["dino", "lejepa", "visreg"])
    ap.add_argument("--draws", default="0,3,5,7,8")
    ap.add_argument("--arms", default=None,
                    help="comma subset of space labels (default all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-d", type=int, default=5000)
    ap.add_argument("--kernels", type=int, default=16)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--fractions", default="0.006,0.01,0.02,0.03,0.05,0.1")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    fracs = [float(x) for x in args.fractions.split(",")]
    n_null = 20 if args.quick else 200
    n_sig_toys = 10 if args.quick else 50
    steps = 60 if args.quick else args.steps
    os.makedirs(args.out, exist_ok=True)

    for draw in [int(x) for x in args.draws.split(",")]:
        holdouts = holdout_set(DS, N_CLS, nh=1, draw=draw)
        seen = [c for c in range(N_CLS) if c not in holdouts]
        res_path = os.path.join(args.out,
                                f"minfrac_{DS}_{args.base}_d{draw}.json")
        results = (json.load(open(res_path)) if os.path.exists(res_path)
                   else {})
        for label, loader in space_list(args.base, draw):
            if args.arms and label not in args.arms.split(","):
                continue
            if label in results and not args.quick:
                print(f"[skip] {args.base} d{draw} {label}", flush=True)
                continue
            got = loader()
            if got is None:
                print(f"[miss] {args.base} d{draw} {label}: no bank/ckpt",
                      flush=True)
                continue
            Htr, ytr, Hte, yte = got
            bg_mask = np.isin(yte, seen)
            sig_mask = np.isin(yte, list(holdouts))
            m = np.isin(ytr, seen)
            anch = torch.as_tensor(
                exp28.class_centroids(Htr[m], ytr[m], seen),
                dtype=torch.float32, device=DEVICE)
            ev = exp29.evaluate_space(Htr, ytr, Hte, yte, anch, seen,
                                      holdouts)
            aucs = []
            for s_ in range(3):
                torch.manual_seed(1000 + s_)
                a, _, _ = exp29.linear_probe_novelty(Htr, ytr, Hte, yte,
                                                     holdouts)
                aucs.append(a)
            d = torch.cdist(torch.as_tensor(Hte, dtype=torch.float32,
                                            device=DEVICE), anch)
            s_ev = d.min(1).values.cpu().numpy()
            pe = exp30.power_at_alpha(s_ev[bg_mask], s_ev[sig_mask],
                                      args.alpha)
            entry = dict(fractions=fracs, n_null=n_null,
                         n_sig_toys=n_sig_toys, holdout=sorted(holdouts),
                         metrics=dict(probe=float(np.mean(aucs)),
                                      eucl=ev["eucl"],
                                      mahaT=ev["maha_tied"], perevt=pe))
            print(f"[{args.base} d{draw} {label}] dim={Htr.shape[1]} "
                  f"bg={int(bg_mask.sum())} sig={int(sig_mask.sum())} "
                  f"probe={entry['metrics']['probe']:.3f} "
                  f"mahaT={ev['maha_tied']:.3f} perev={pe:.2f}", flush=True)

            R = torch.as_tensor(Htr[m][:20000], dtype=torch.float32,
                                device=DEVICE)
            bg_t = torch.as_tensor(Hte[bg_mask], dtype=torch.float32,
                                   device=DEVICE)
            sig_t = torch.as_tensor(Hte[sig_mask], dtype=torch.float32,
                                    device=DEVICE)

            _, pc, _ = mahalanobis_novelty(Htr, ytr, Hte, seen)
            s_bg, s_sig = pc[bg_mask], pc[sig_mask]

            def maha_fn(bg_idx, sig_idx, seed):
                s = (np.concatenate([s_bg[bg_idx], s_sig[sig_idx]])
                     if len(sig_idx) else s_bg[bg_idx])
                return [float(s.mean())]

            g = np.random.default_rng(args.seed)
            R_pool = Htr[m]
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

            def spk_fn(bg_idx, sig_idx, seed):
                D = (torch.cat([bg_t[torch.as_tensor(bg_idx, device=DEVICE)],
                                sig_t[torch.as_tensor(sig_idx,
                                                      device=DEVICE)]])
                     if len(sig_idx) else
                     bg_t[torch.as_tensor(bg_idx, device=DEVICE)])
                return np_test_stats(D, R, M=args.kernels, steps=steps,
                                     sigma0=sigma0, seed=seed)

            for name, fn in (("maha", maha_fn), ("mmd", mmd_fn),
                             ("sparker", spk_fn)):
                null_agg, sig_agg = exp146.toys_battery(
                    fn, len(bg_t), len(sig_t), fracs, args.n_d, n_null,
                    n_sig_toys, args.seed, tag=f"{label}-{name}")
                zs = exp146.z_curve(null_agg, sig_agg)
                fs, fs_str = exp146.f_star(fracs, zs)
                entry[name] = dict(z=zs, f2sigma=fs, f2sigma_str=fs_str)
                print(f"  [{label}] {name}: Z={np.round(zs, 2).tolist()} "
                      f"f*={fs_str}", flush=True)
            results[label] = entry
            json.dump(results, open(res_path, "w"), indent=1)

        print(f"\n== {DS}/{args.base} draw {draw} "
              f"(holdout {sorted(holdouts)}) ==")
        print(f"{'space':<30}{'probe':>7}{'mahaT':>7}{'per-ev':>7}"
              f"{'f* maha':>9}{'f* mmd':>9}{'f* spk':>9}")
        for label, _ in space_list(args.base, draw):
            r = results.get(label)
            if not r:
                continue
            mtr = r["metrics"]
            print(f"{label:<30}{mtr['probe']:>7.3f}{mtr['mahaT']:>7.3f}"
                  f"{mtr['perevt']:>7.2f}{r['maha']['f2sigma_str']:>9}"
                  f"{r['mmd']['f2sigma_str']:>9}"
                  f"{r['sparker']['f2sigma_str']:>9}")


if __name__ == "__main__":
    main()
