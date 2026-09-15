"""
Experiment 147: post-discovery SparKer significance at each injected
fraction, leakage-free CIFAR scratch arms -- the discovery-mode counterpart
of exp 146.

Per (arm, fraction f): inject n = f/(1-f) * |seen train| held-out train
images into the corpus (exp-68 protocol, same rng), run the settled
discovery loop (quantile cut, distance pool, 2 rounds x ft_epochs) on a copy
of the scratch checkpoint, then on the UPDATED space run the SparKer toy
battery (R = post-space seen train embeddings [:20000], bg/sig = post-space
test split, N_D=5000, 200 null / 50 signal toys at that same f) and record
the median expected significance Z as in exp 146.  f*(2sigma) is then
log-interpolated across fractions, each fraction scored in its own
fine-tuned space.

Discovery is defined for single-encoder spaces; the residual concats have no
discovery run (paper: "discovery on the scratch residual concats has not
been run"), so the arms here are the four supervised parents.

    python experiments/147_post_discovery_2sigma.py --dataset cifar10
    python experiments/147_post_discovery_2sigma.py --quick --arms supcon --fractions 0.01,0.1
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
from supersig.sparker import np_test_stats, aggregate_pvalues, median_pairwise

exp28 = importlib.import_module("28_concat_residual")
exp31 = importlib.import_module("31_sparker_power")
exp146 = importlib.import_module("146_min_frac_2sigma")

OUT = os.path.join("logs", "exp147")


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
    ap.add_argument("--fractions",
                    default="0.001,0.003,0.006,0.01,0.02,0.03,0.05,0.1")
    ap.add_argument("--arms", default="supcon,ssig,nplmsd,nplmcw")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

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
    res_path = os.path.join(args.out, f"post2sig_{ds}_h{args.holdout}.json")
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
        # "supcon" -> scratch_supcon_...; "supcon-res" / "supcon-resnplm"
        # -> the exp-137 residual-child checkpoints (raw state dicts).
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
        entry = results.get(name, dict(fractions=fracs, z=[], purity1=[],
                                       done=[]))
        net, ck = load_base(name)
        tr, tr_lab = collect_embeddings(net, train_eval_loader)
        if name == "supsig" and "means" in ck:
            means0 = ck["means"].to(DEVICE).detach()
        else:
            m = np.isin(tr_lab, seen)
            cents = exp28.class_centroids(tr[m], tr_lab[m], seen)
            means0 = exp28.fill_means(cents, seen, cfg).detach()

        for i_f, f in enumerate(fracs):
            if f in entry.get("done", []):
                print(f"[skip] {name} f={f}: already archived", flush=True)
                continue
            n_inj = int(round(f * len(seen_idx) / (1.0 - f)))
            rng = np.random.default_rng(args.seed * 1000 + i_f)
            inj = rng.choice(sig_idx_all, size=min(n_inj, len(sig_idx_all)),
                             replace=False)
            sub = Subset(base_ds, np.concatenate([seen_idx, inj]).tolist())
            tel_loader = DataLoader(sub, batch_size=256, shuffle=False,
                                    num_workers=2)
            print(f"\n===== {name} f={f} ({len(inj)} injected) =====",
                  flush=True)
            bb = copy.deepcopy(net)
            _, hist = run_discovery(
                bb, means0.clone(), base_ds=sub,
                train_eval_loader=tel_loader, test_loader=test_loader,
                seen=seen, holdouts=holdouts, dataset_name=ds,
                rep_weight=cfg["rep_weight"],
                sigreg_weight=cfg["sigreg_weight"],
                n_slices=cfg["n_slices"], rounds=args.rounds,
                ft_epochs=ft_ep, names=None, seed=args.seed)
            pur1 = float(hist[0]["purity"]) if hist else float("nan")
            te_post, tel_post = collect_embeddings(bb, test_loader)
            tr_post, trl_post = collect_embeddings(bb, train_eval_loader)
            bg_mask = np.isin(tel_post, seen)
            sig_mask = np.isin(tel_post, list(holdouts))
            R = torch.as_tensor(tr_post[np.isin(trl_post, seen)][:20000],
                                dtype=torch.float32, device=DEVICE)
            bg_t = torch.as_tensor(te_post[bg_mask], dtype=torch.float32,
                                   device=DEVICE)
            sig_t = torch.as_tensor(te_post[sig_mask], dtype=torch.float32,
                                    device=DEVICE)
            del bb
            torch.cuda.empty_cache()
            sigma0 = median_pairwise(bg_t, seed=args.seed)

            def spk_fn(bg_idx, sig_idx, seed):
                D = (torch.cat([bg_t[torch.as_tensor(bg_idx, device=DEVICE)],
                                sig_t[torch.as_tensor(sig_idx,
                                                      device=DEVICE)]])
                     if len(sig_idx) else
                     bg_t[torch.as_tensor(bg_idx, device=DEVICE)])
                return np_test_stats(D, R, M=args.kernels, steps=steps,
                                     sigma0=sigma0, seed=seed)

            null_agg, sig_agg = exp146.toys_battery(
                spk_fn, len(bg_t), len(sig_t), [f], args.n_d, n_null,
                n_sig_toys, args.seed + i_f, tag=f"{name}-post-spk")
            z = exp146.z_curve(null_agg, sig_agg)[0]
            entry.setdefault("z", []).append(z)
            entry.setdefault("purity1", []).append(pur1)
            entry.setdefault("done", []).append(f)
            np.savez(os.path.join(
                args.out, f"toys_{name}_{ds}_h{args.holdout}_f{f}.npz"),
                null_agg=null_agg, sig_agg=sig_agg, fraction=f, purity1=pur1)
            results[name] = entry
            json.dump(results, open(res_path, "w"), indent=1)
            print(f"  [{name}] post f={f}: r1 purity={pur1:.3f} "
                  f"Z={z:.2f}", flush=True)
        del net
        torch.cuda.empty_cache()

    # ---- summary ----------------------------------------------------------
    print(f"\n== {ds} h{args.holdout}: POST-discovery SparKer, median "
          f"expected significance ==")
    for name in arms:
        e = results.get(name)
        if not e or not e.get("done"):
            continue
        order = np.argsort(e["done"])
        fs = [e["done"][i] for i in order]
        zs = [e["z"][i] for i in order]
        f2, f2s = exp146.f_star(fs, zs)
        print(f"  {name:<10} Z@{fs}={np.round(zs, 2).tolist()} "
              f"f*(2sig)={f2s}")


if __name__ == "__main__":
    main()
