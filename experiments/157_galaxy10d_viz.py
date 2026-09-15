"""
Experiment 157: a 10-DIMENSIONAL LeJEPA/Galaxy10 latent for visualization.

Galaxy10 has 10 classes; this trains the head to a 10-D embedding (vs the
campaign's 100-D) so the latent can be visualized directly, for ONE holdout
draw, ALL arms, with s/sqrt(b) discovery mode.  The dim-10 spaces are trained
by exp 70 (parents) and exp 71 (residual children) with --emb-dim 10, which
tags every artifact `_e10` (seed_sfx) so the 100-D archive is untouched.

This script consumes those dim-10 heads: it embeds train/test to 10-D
(PRE), runs the settled discovery loop with the s/sqrt(b) cut (pool_score
np, cut_rule ssb, n_min 5) on the natural train corpus, embeds again (POST),
and saves per arm an npz with everything a plot needs:

    tr, te            (N,10) pre-discovery embeddings
    tr_lab, te_lab    integer class labels (holdout class flagged by `holdout`)
    tr_post, te_post  (N,10) post-discovery embeddings
    seen_centroids    (n_seen,10) seen-class means (post)
    anchors           (k,10) discovered anchors (post), or empty
    f2sigma, engaged  the s/sqrt(b) detection summary for this arm

    python experiments/157_galaxy10d_viz.py --base lejepa --draw 0
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
exp37 = importlib.import_module("37_dtd_vit")
exp43 = importlib.import_module("43_dtd_finetune")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA, CKPT = os.path.join(REPO, "data"), os.path.join(REPO, "checkpoints")
OUT = os.path.join(REPO, "logs", "exp157")
DS, N_CLS = "galaxy10", 10
ARMS = ["simclr-ft", "sigreg-ssl-ft", "nplm-bil-ft", "supcon-ft", "ss-ft",
        "nplm-sup-ft", "supcon-cw-ft", "supcon-ft_res", "supcon-ft_resnplm",
        "ss-ft_res"]


def load_arm(base, arm, draw, emb_dim):
    """(head, Xtr, ytr, Xte, yte) for a dim-tagged arm, or None."""
    tag = f"_h1_d{draw}_e{emb_dim}"
    bp = os.path.join(DATA, f"tf_feats_{DS}_{base}_ft70_{arm}{tag}.pt")
    ck = os.path.join(CKPT, f"{DS}_ft_{base}_{arm}_seen{tag}.pt")
    if not (os.path.exists(bp) and os.path.exists(ck)):
        print(f"[miss] {arm}: {os.path.basename(bp)} / "
              f"{os.path.basename(ck)}", flush=True)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="lejepa",
                    choices=["dino", "lejepa", "visreg"])
    ap.add_argument("--draw", type=int, default=0)
    ap.add_argument("--emb-dim", type=int, default=10)
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--n-min", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--ft-epochs", type=int, default=5)
    ap.add_argument("--n-slices", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    holdouts = holdout_set(DS, N_CLS, nh=1, draw=args.draw)
    seen = [c for c in range(N_CLS) if c not in holdouts]
    rep_weight = 20.0 * 45.0 / (N_CLS * (N_CLS - 1) / 2)   # exp 70
    cfg = dict(n_classes=N_CLS, pair_dist=5.0)
    os.makedirs(args.out, exist_ok=True)
    print(f"exp157 {DS}/{args.base} draw {args.draw} (holdout {sorted(holdouts)}), "
          f"emb_dim={args.emb_dim}, s/sqrt(b) discovery", flush=True)

    summary = {}
    for arm in args.arms:
        got = load_arm(args.base, arm, args.draw, args.emb_dim)
        if got is None:
            continue
        head, Xtr, ytr, Xte, yte = got
        tr_lab, te_lab = ytr.numpy(), yte.numpy()
        tel = DataLoader(TensorDataset(Xtr, ytr), batch_size=512,
                         shuffle=False)
        test_loader = DataLoader(TensorDataset(Xte, yte), batch_size=512,
                                 shuffle=False)
        tr0, _ = collect_embeddings(head, tel)
        te0, _ = collect_embeddings(head, test_loader)
        m = np.isin(tr_lab, seen)
        cents = exp28.class_centroids(tr0[m], tr_lab[m], seen)
        means0 = exp28.fill_means(cents, seen, cfg).detach()

        # s/sqrt(b) discovery on the natural train corpus (holdout present)
        bb = copy.deepcopy(head)
        cur_means, hist = run_discovery(
            bb, means0.clone(), base_ds=TensorDataset(Xtr, ytr),
            train_eval_loader=tel, test_loader=test_loader, seen=seen,
            holdouts=holdouts, dataset_name=DS, rep_weight=rep_weight,
            sigreg_weight=1.0, n_slices=args.n_slices, rounds=args.rounds,
            ft_epochs=args.ft_epochs, names=None, seed=args.seed,
            pool_score="np", cut_rule="ssb", n_min=args.n_min,
            on_refuse="skip")
        tr_post, trl_post = collect_embeddings(bb, tel)
        te_post, tel_post = collect_embeddings(bb, test_loader)
        del bb
        torch.cuda.empty_cache()
        anchors = cur_means[N_CLS:].detach().cpu().numpy()
        mp = np.isin(trl_post, seen)
        seen_cents = exp28.class_centroids(tr_post[mp], trl_post[mp], seen)
        seen_cents = np.asarray(seen_cents.detach().cpu()
                                if hasattr(seen_cents, "detach")
                                else seen_cents)
        pur1 = float(hist[0]["purity"]) if hist else float("nan")
        engaged = bool(hist and hist[0].get("cut", {}).get("ok", False))

        np.savez(os.path.join(args.out,
                              f"viz_{DS}_{args.base}_d{args.draw}_"
                              f"e{args.emb_dim}_{arm}.npz"),
                 tr=tr0, te=te0, tr_lab=tr_lab, te_lab=te_lab,
                 tr_post=tr_post, te_post=te_post,
                 seen_centroids=seen_cents,
                 anchors=anchors, seen=np.asarray(seen),
                 holdout=np.asarray(sorted(holdouts)))
        summary[arm] = dict(engaged=engaged, purity1=pur1,
                            n_anchors=int(anchors.shape[0]),
                            n_novel_test=int(np.isin(te_lab,
                                                     list(holdouts)).sum()))
        print(f"[{arm}] engaged={engaged} purity={pur1:.3f} "
              f"anchors={anchors.shape[0]} emb_dim={tr0.shape[1]}", flush=True)

    json.dump(summary, open(os.path.join(
        args.out, f"summary_{DS}_{args.base}_d{args.draw}_"
        f"e{args.emb_dim}.json"), "w"), indent=1)
    print(f"\nwrote {args.out}/viz_* + summary "
          f"({len(summary)} arms)", flush=True)


if __name__ == "__main__":
    main()
