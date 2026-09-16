"""
Experiment 163: Galaxy10 twin of exp 162 -- f*(2sigma) with discovery on the
RESIDUAL child, scored on the residual-only space AND the concat
[frozen parent || residual], side by side, pre and post s/sqrt(b) discovery.

Head-on-bank (like exp 151/161), so light on the GPU -- runs alongside the
CIFAR concat sweep.  Uses the dim-tagged (_e{dim}) heads/banks trained by
exp 70/71 with --emb-dim; the discovery loop fine-tunes only the residual
child head over its trunk bank.

    python experiments/163_galaxy_concat_residual.py --base lejepa --draw 0 \
        --emb-dim 10 --pairs supcon-ft:supcon-ft_res,ss-ft:ss-ft_res
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
from supersig.holdouts import holdout_set

exp28 = importlib.import_module("28_concat_residual")
exp43 = importlib.import_module("43_dtd_finetune")
exp146 = importlib.import_module("146_min_frac_2sigma")
exp162 = importlib.import_module("162_cifar_concat_residual")   # reuse _prep

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA, CKPT = os.path.join(REPO, "data"), os.path.join(REPO, "checkpoints")
OUT = os.path.join(REPO, "logs", "exp163")
DS, N_CLS = "galaxy10", 10
TESTS = ["eucl", "eucl-disc", "maha", "mmd", "sparker", "sparker-anch"]


def load_head(base, arm, draw, dim):
    tag = f"_h1_d{draw}" + ("" if dim == 100 else f"_e{dim}")
    bp = os.path.join(DATA, f"tf_feats_{DS}_{base}_ft70_{arm}{tag}.pt")
    ck = os.path.join(CKPT, f"{DS}_ft_{base}_{arm}_seen{tag}.pt")
    if not (os.path.exists(bp) and os.path.exists(ck)):
        print(f"[miss] {arm}: {os.path.basename(bp)}", flush=True); return None
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="lejepa")
    ap.add_argument("--draw", type=int, default=0)
    ap.add_argument("--emb-dim", type=int, default=10)
    ap.add_argument("--pairs", default="supcon-ft:supcon-ft_res,"
                                       "ss-ft:ss-ft_res")
    ap.add_argument("--fractions", default="0.006,0.01,0.02,0.03,0.05,0.1")
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
    holdouts = holdout_set(DS, N_CLS, nh=1, draw=args.draw)
    seen = [c for c in range(N_CLS) if c not in holdouts]
    os.makedirs(args.out, exist_ok=True)
    res_path = os.path.join(args.out, f"concatres_{DS}_{args.base}_"
                            f"d{args.draw}_e{args.emb_dim}.json")
    results = json.load(open(res_path)) if os.path.exists(res_path) else {}

    def z_of(fn, bg_t, sig_t, f, seed):
        na, sa = exp146.toys_battery(fn, len(bg_t), len(sig_t), [f],
                                     args.n_d, 200, 50, seed, tag="")
        return exp146.z_curve(na, sa)[0]

    for parent, child in pairs:
        gp = load_head(args.base, parent, args.draw, args.emb_dim)
        gc = load_head(args.base, child, args.draw, args.emb_dim)
        if gp is None or gc is None:
            continue
        pH, Xtr_p, ytr, Xte_p, yte = gp
        cH, Xtr_c, _, Xte_c, _ = gc
        tr_lab, te_lab = ytr.numpy(), yte.numpy()
        P_tr, _ = collect_embeddings(pH, DataLoader(TensorDataset(Xtr_p, ytr),
                                                    batch_size=512))
        P_te, _ = collect_embeddings(pH, DataLoader(TensorDataset(Xte_p, yte),
                                                    batch_size=512))
        telc = DataLoader(TensorDataset(Xtr_c, ytr), batch_size=512)
        tec = DataLoader(TensorDataset(Xte_c, yte), batch_size=512)
        R0_tr, _ = collect_embeddings(cH, telc)
        R0_te, _ = collect_embeddings(cH, tec)
        m = np.isin(tr_lab, seen)
        cents = exp28.class_centroids(R0_tr[m], tr_lab[m], seen)
        means0 = exp28.fill_means(cents, seen, cfg).detach()
        seen_idx = np.where(m)[0]
        sig_idx_all = np.where(np.isin(tr_lab, list(holdouts)))[0]
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
            sub_idx = np.concatenate([seen_idx, inj])
            sub = TensorDataset(Xtr_c[sub_idx], ytr[sub_idx])
            sub_loader = DataLoader(sub, batch_size=512, shuffle=False)
            bb = copy.deepcopy(cH)
            cur_means, hist = run_discovery(
                bb, means0.clone(), base_ds=sub, train_eval_loader=sub_loader,
                test_loader=tec, seen=seen, holdouts=holdouts,
                dataset_name=DS, rep_weight=rep_weight, sigreg_weight=1.0,
                n_slices=args.n_slices, rounds=args.rounds,
                ft_epochs=args.ft_epochs, names=None, seed=args.seed,
                pool_score="np", cut_rule="ssb", n_min=args.n_min,
                on_refuse="skip")
            c0 = hist[0].get("cut", {}) if hist else {}
            entry["cut"][fk] = dict(ok=bool(c0.get("ok", True)),
                                    pur=float(hist[0]["purity"]) if hist
                                    else float("nan"))
            Rp_tr, trl2 = collect_embeddings(bb, telc)
            Rp_te, tel2 = collect_embeddings(bb, tec)
            del bb; torch.cuda.empty_cache()
            A_res = cur_means[N_CLS:].detach().cpu().numpy()
            A_concat = None
            if len(A_res):
                asg = torch.cdist(torch.as_tensor(Rp_tr, dtype=torch.float32,
                                                  device=DEVICE),
                                  torch.as_tensor(A_res, dtype=torch.float32,
                                                  device=DEVICE)
                                  ).argmin(1).cpu().numpy()
                A_concat = np.asarray(
                    [np.concatenate([P_tr[asg == k].mean(0) if (asg == k).any()
                                     else P_tr.mean(0), A_res[k]])
                     for k in range(len(A_res))], np.float32)

            spaces = {
                "resid": {"pre": (R0_tr, R0_te, None),
                          "post": (Rp_tr, Rp_te, A_res)},
                "concat": {"pre": (np.concatenate([P_tr, R0_tr], 1),
                                   np.concatenate([P_te, R0_te], 1), None),
                           "post": (np.concatenate([P_tr, Rp_tr], 1),
                                    np.concatenate([P_te, Rp_te], 1),
                                    A_concat)}}
            for sp, st in spaces.items():
                for state, (Ctr, Cte, A) in st.items():
                    _, det = exp162._prep(Ctr, tr_lab, Cte, te_lab, seen,
                                          holdouts, A, args)
                    bg_t, sig_t, nb = det["bg_t"], det["sig_t"], det["nb"]
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
                        entry[sp][state][tn][fk] = (
                            z_of(fn, bg_t, sig_t, f, args.seed + i_f)
                            if fn is not None else None)
                    print(f"  [{key}] f={f} {sp}/{state}: "
                          f"ed={entry[sp][state]['eucl-disc'].get(fk)} "
                          f"spk={entry[sp][state]['sparker'].get(fk)}",
                          flush=True)
            results[key] = entry
            json.dump(results, open(res_path, "w"), indent=1)
            print(f"[done] {key} f={f} pur={entry['cut'][fk]['pur']:.3f}",
                  flush=True)
    print("\nexp163 done.", flush=True)


if __name__ == "__main__":
    main()
