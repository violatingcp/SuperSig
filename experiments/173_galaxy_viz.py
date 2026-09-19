"""
Experiment 173: galaxy10 latent export for the exp-170 normalization arms, so
the CIFAR exp-160/plot can be reproduced on galaxy.  Head-only over the RAW
frozen lejepa trunk (matching exp-170 default): per arm, train the 10-D head,
embed the test set (PRE), run the s/sqrt(b) discovery loop on the full train
pool (natural ~10% novel), embed again (POST), and save an npz with the same
payload shape exp-160 writes (tr/te/*_lab/*_post/anchors/seen_centroids).

    python experiments/173_galaxy_viz.py --arms supcon supconproj supconeucl supcondist
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse, copy, importlib
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from supersig.config import DEVICE
from supersig.discovery import run_discovery
from supersig.train import collect_embeddings
from supersig.holdouts import holdout_set

exp28 = importlib.import_module("28_concat_residual")
exp170 = importlib.import_module("170_galaxy_normfree")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data")
OUT = os.path.join(REPO, "logs", "exp173")
DS, N_CLS = "galaxy10", 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+",
                    default=["supcon", "supconproj", "supconeucl", "supcondist"])
    ap.add_argument("--draw", type=int, default=0)
    ap.add_argument("--dim", type=int, default=10)
    ap.add_argument("--base", default="lejepa")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=5.0)
    ap.add_argument("--n-slices", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--ft-epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    holdouts = holdout_set(DS, N_CLS, nh=1, draw=args.draw)
    seen = [c for c in range(N_CLS) if c not in holdouts]
    rep_weight = 20.0 * 45.0 / (N_CLS * (N_CLS - 1) / 2)
    cfg = dict(n_classes=N_CLS, pair_dist=5.0)

    raw = torch.load(os.path.join(DATA, f"tf_feats_{DS}_{args.base}_vitb16.pt"),
                     map_location="cpu")
    Xtr, ytr = raw["train"]; Xte, yte = raw["test"]
    Xtr, Xte = Xtr.float(), Xte.float()
    tr_lab, te_lab = ytr.numpy(), yte.numpy()
    aug = torch.load(os.path.join(DATA,
                     f"tf_augfeats_{DS}_{args.base}_vitb16_a8.pt"),
                     map_location="cpu")
    augf, auglab = aug["feats"], aug["labels"].numpy()
    seen_pos = np.where(np.isin(auglab, seen))[0]
    aug_seen = augf[:, seen_pos, :]
    y_seen = torch.as_tensor(auglab[seen_pos], dtype=torch.long)
    tel = DataLoader(TensorDataset(Xtr, ytr), batch_size=512)
    tec = DataLoader(TensorDataset(Xte, yte), batch_size=512)

    for arm in args.arms:
        print(f"\n===== {arm} =====", flush=True)
        net = exp170.train_head(arm, aug_seen, y_seen, args.dim, args.epochs,
                                args.tau, args.lam, args.n_slices, args.seed)
        tr0, _ = collect_embeddings(net, tel)
        te0, _ = collect_embeddings(net, tec)
        m = np.isin(tr_lab, seen)
        means0 = exp28.fill_means(exp28.class_centroids(tr0[m], tr_lab[m], seen),
                                  seen, cfg).detach()
        bb = copy.deepcopy(net)
        cur_means, hist = run_discovery(
            bb, means0.clone(), base_ds=TensorDataset(Xtr, ytr),
            train_eval_loader=tel, test_loader=tec, seen=seen,
            holdouts=holdouts, dataset_name=DS, rep_weight=rep_weight,
            sigreg_weight=1.0, n_slices=args.n_slices, rounds=args.rounds,
            ft_epochs=args.ft_epochs, names=None, seed=args.seed,
            pool_score="np", cut_rule="ssb", n_min=5, on_refuse="skip")
        tr_post, _ = collect_embeddings(bb, tel)
        te_post, _ = collect_embeddings(bb, tec)
        del bb
        anchors = cur_means[N_CLS:].detach().cpu().numpy()
        mp = np.isin(tr_lab, seen)
        sc = np.asarray(exp28.class_centroids(tr_post[mp], tr_lab[mp], seen))
        pur = float(hist[0]["purity"]) if hist else float("nan")
        np.savez(os.path.join(args.out,
                 f"viz_{DS}_d{args.draw}_e{args.dim}_{arm}.npz"),
                 tr=tr0, te=te0, tr_lab=tr_lab, te_lab=te_lab,
                 tr_post=tr_post, te_post=te_post, seen_centroids=sc,
                 anchors=anchors, seen=np.asarray(seen),
                 holdout=np.asarray(sorted(holdouts)))
        print(f"  [{arm}] purity={pur:.3f} anchors={len(anchors)} saved",
              flush=True)
    print("\nexp173 done.", flush=True)


if __name__ == "__main__":
    main()
