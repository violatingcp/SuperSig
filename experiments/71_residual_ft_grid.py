"""
Experiment 71: residual fine-tuning on the exp-70 end-to-end checkpoints
(cars / flowers / dtd / galaxy10 x dino / lejepa / visreg).

Exp-49 recipe generalized: from a supervised exp-70 parent (trained on
seen classes only), freeze the parent's seen-class centroids in head
space, deepcopy the parent, and fine-tune end-to-end on the seen two-view
corpus with a residual objective on r = z - cent_y:

  res       NT-Xent (temp 0.5, instance positives) + lam=5 SIGReg on r
            (the exp-36/49 champion residual objective)
  res-nplm  instance/bilinear NPLM + lam=1 SIGReg on r (exp-59 corner)

Default constructions per cell: supcon-ft->res, ss-ft->res,
supcon-ft->res-nplm.  Evaluation (exp-70 novelty battery, no toys):
3-seed holdout probe, nearest-centroid acc, eucl, mahaT and per-event
power for the parent head (reference), the residual head, and the concat
[parent ; residual] (200-D).  Holdouts as in exp 70 (last 10; galaxy10
last 1).  Checkpoints {ds}_ft_{base}_{parent}_res[nplm]_seen.pt.

    python experiments/71_residual_ft_grid.py --dataset cars --base dino
    python experiments/71_residual_ft_grid.py --dataset dtd --base visreg --quick
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import copy
import importlib
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from supersig.config import DEVICE, REPO_DIR, plot_path
from supersig.losses import HybridContrastiveLoss, sigreg_loss, supcon_loss

exp28 = importlib.import_module("28_concat_residual")
exp29 = importlib.import_module("29_residual_finetune")
exp30 = importlib.import_module("30_power_curves")
exp37 = importlib.import_module("37_dtd_vit")
exp43 = importlib.import_module("43_dtd_finetune")
exp44 = importlib.import_module("44_transfer_32d")
exp49 = importlib.import_module("49_aircraft_ssl_ft")
exp70 = importlib.import_module("70_cars_ft_suite")

CKPT_DIR = os.path.join(REPO_DIR, "checkpoints")

# (parent arm, objective)
RUNS = [("supcon-ft", "res"), ("ss-ft", "res"), ("supcon-ft", "res-nplm")]


def make_res_nplm_step(cents, lam, n_slices):
    loss_fn = HybridContrastiveLoss(positives="instance", critic="bilinear",
                                    estimator="nplm", marginal="none",
                                    tau=1.0)
    def step(model, v1, v2, y):
        x = torch.cat([v1, v2]).to(DEVICE, non_blocking=True)
        yy = torch.cat([y, y]).to(DEVICE)
        r = model(x).float() - cents[yy]
        inst = torch.arange(v1.size(0), device=DEVICE)
        inter, _ = loss_fn(r, torch.cat([inst, inst]))
        return inter, lam * sigreg_loss(r, n_slices=n_slices)
    return step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cars",
                    choices=["cars", "flowers", "galaxy10", "dtd"])
    ap.add_argument("--base", default="dino",
                    choices=["dino", "lejepa", "visreg"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ft-epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr-backbone", type=float, default=1e-5)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--emb-dim", type=int, default=100)
    ap.add_argument("--n-slices", type=int, default=64)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    exp70.DS, exp70.BASE = args.dataset, args.base
    DS, BASE = args.dataset, args.base
    args.ft_epochs = args.ft_epochs or (1 if args.quick else 20)

    N_CLS = 47 if DS == "dtd" else exp44.N_CLASSES[DS]
    n_hold = 1 if DS == "galaxy10" else 10
    holdouts = set(range(N_CLS - n_hold, N_CLS))
    seen = [c for c in range(N_CLS) if c not in holdouts]
    tag = f"{DS}_{BASE}_ft71"
    print(f"exp71 [{tag}] residual ft on exp-70 parents, runs={RUNS}, "
          f"epochs={args.ft_epochs}")

    corpus = exp43.train_corpus(DS)
    ytr_all = exp70.corpus_labels(corpus)
    seen_idx_corpus = np.where(~np.isin(ytr_all,
                                        list(holdouts)))[0].tolist()

    def battery(name, tr, te, tr_lab, te_lab):
        m = np.isin(tr_lab, seen)
        cents = exp28.class_centroids(tr[m], tr_lab[m], seen)
        anch = cents.detach().float().to(DEVICE)
        r = exp29.evaluate_space(tr, tr_lab, te, te_lab, anch, seen,
                                 holdouts)
        aucs = []
        for s in range(3):
            torch.manual_seed(1000 + s)
            a, _, _ = exp29.linear_probe_novelty(tr, tr_lab, te, te_lab,
                                                 holdouts)
            aucs.append(a)
        d = torch.cdist(torch.as_tensor(te, dtype=torch.float32,
                                        device=DEVICE), anch)
        s_ = d.min(1).values.cpu().numpy()
        bg = np.isin(te_lab, seen)
        sg = np.isin(te_lab, list(holdouts))
        pe = exp30.power_at_alpha(s_[bg], s_[sg], args.alpha)
        out = dict(probe=float(np.mean(aucs)), probe_sd=float(np.std(aucs)),
                   acc=r["acc"], eucl=r["eucl"], mahaT=r["maha_tied"],
                   perevt=pe)
        print(f"  [{name:<28}] probe={out['probe']:.4f}+-{out['probe_sd']:.4f}"
              f" acc={out['acc']:.4f} eucl={out['eucl']:.4f} "
              f"mahaT={out['mahaT']:.4f} perevt={pe:.3f}")
        return out

    results = {}
    parent_cache = {}
    for parent, obj in RUNS:
        key = f"{parent}->{obj}"
        print(f"\n===== [{tag}] {key} =====")
        pck = os.path.join(CKPT_DIR, f"{DS}_ft_{BASE}_{parent}_seen"
                           f"{'_quick' if args.quick else ''}.pt")
        if not os.path.exists(pck):
            print(f"  !! missing parent {pck}, skipping")
            continue
        if parent not in parent_cache:
            par = exp43.FineTuneModel(BASE, args.emb_dim)
            par.load_state_dict(torch.load(pck, map_location=DEVICE))
            pb = exp70.trunk_banks(par, parent, args)
            (pXtr, pytr), (pXte, pyte) = pb["train"], pb["test"]
            ph = par.head.float()
            pHtr = exp37.embed(ph, pXtr.float()).numpy()
            pHte = exp37.embed(ph, pXte.float()).numpy()
            tr_lab, te_lab = pytr.numpy(), pyte.numpy()
            m = np.isin(tr_lab, seen)
            cents_full = torch.zeros(N_CLS, args.emb_dim, device=DEVICE)
            cents_full[torch.as_tensor(seen, device=DEVICE)] = \
                exp28.class_centroids(pHtr[m], tr_lab[m],
                                      seen).detach().float().to(DEVICE)
            parent_cache[parent] = (par, pHtr, pHte, tr_lab, te_lab,
                                    cents_full)
            results[f"{parent} (parent)"] = battery(
                f"{parent} (parent)", pHtr, pHte, tr_lab, te_lab)
        par, pHtr, pHte, tr_lab, te_lab, cents_full = parent_cache[parent]

        rck = os.path.join(CKPT_DIR, f"{DS}_ft_{BASE}_{parent}_"
                           f"{obj.replace('-', '')}_seen"
                           f"{'_quick' if args.quick else ''}.pt")
        torch.manual_seed(args.seed + 7); np.random.seed(args.seed + 7)
        child = copy.deepcopy(par)
        if os.path.exists(rck) and not args.refresh:
            print(f"  loading {rck}")
            child.load_state_dict(torch.load(rck, map_location=DEVICE))
        else:
            step = (exp49.make_residual_step(cents_full, 5.0)
                    if obj == "res"
                    else make_res_nplm_step(cents_full, 1.0, args.n_slices))
            loader = exp70.seen_two_view_loader(corpus, seen_idx_corpus,
                                                args)
            exp49.ft_loop(child, loader, args.ft_epochs, step, args,
                          f"{parent}-{obj}")
            torch.save(child.state_dict(), rck)
            del loader
        rb = exp70.trunk_banks(child, f"{parent}_{obj.replace('-', '')}",
                               args)
        ch = child.head.float()
        rHtr = exp37.embed(ch, rb["train"][0].float()).numpy()
        rHte = exp37.embed(ch, rb["test"][0].float()).numpy()
        del child
        torch.cuda.empty_cache()

        results[f"{key} (residual)"] = battery(
            f"{key} (residual)", rHtr, rHte, tr_lab, te_lab)
        results[f"{key} (concat)"] = battery(
            f"{key} (concat)", np.concatenate([pHtr, rHtr], 1),
            np.concatenate([pHte, rHte], 1), tr_lab, te_lab)

    parent_cache.clear()
    torch.cuda.empty_cache()

    print(f"\n===== EXP71 SUMMARY [{tag}] =====")
    print(f"  {'space':<30}{'probe':>16}{'acc':>8}{'eucl':>8}{'mahaT':>8}"
          f"{'perevt':>8}")
    for k, r in results.items():
        print(f"  {k:<30}{r['probe']:>9.4f}+-{r['probe_sd']:.4f}"
              f"{r['acc']:>8.4f}{r['eucl']:>8.4f}{r['mahaT']:>8.4f}"
              f"{r['perevt']:>8.3f}")

    labels = list(results)
    plt.figure(figsize=(1.0 * len(labels) + 3, 5.5))
    plt.bar(range(len(labels)), [results[k]["probe"] for k in labels],
            yerr=[results[k]["probe_sd"] for k in labels], capsize=3,
            color=["#eda100" if "parent" in k else
                   "#8c2d9e" if "nplm" in k else "#008300"
                   for k in labels])
    plt.xticks(range(len(labels)), labels, rotation=25, ha="right",
               fontsize=7)
    plt.ylabel("holdout probe ROC AUC")
    plt.title(f"exp71 residual ft ({tag})")
    plt.grid(alpha=0.25, axis="y")
    plt.tight_layout()
    out = plot_path(f"exp71_probe_{tag}.png")
    plt.savefig(out, dpi=150); plt.close()
    print("saved", out)

    os.makedirs(os.path.join("logs", "exp71"), exist_ok=True)
    np.savez(os.path.join("logs", "exp71", f"results_{tag}.npz"),
             spaces=np.array(list(results)),
             **{f"{k.replace(' ', '_').replace('>', '')}__{f}":
                np.array(results[k][f]) for k in results
                for f in ("probe", "probe_sd", "acc", "eucl", "mahaT",
                          "perevt")})
    print("Done.")


if __name__ == "__main__":
    main()
