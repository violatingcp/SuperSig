"""
Experiment 170: galaxy10 test of the exp-168 normalization conclusions.

Trains four 10-D heads over the SAME frozen raw lejepa ViT trunk (neutral
common backbone -- clean isolation of the head geometry), one holdout draw,
then runs the identical exp-168 f*(2sigma) battery (pre frozen + post s/sqrt(b)
discovery; six tests) so the CIFAR conclusions can be checked on galaxy:

  supcon      plain cosine SupCon (F.normalize, temp 0.1)          [= plain]
  supconproj  cosine SupCon through a throwaway projection g        [= proj-head]
  supconeucl  bilinear (raw inner-product) softmax + SIGReg, no norm
  supcondist  distance (raw squared-distance) softmax + SIGReg, no norm

Head-on-bank + CPU-friendly (run with CUDA_VISIBLE_DEVICES="") so it does not
disturb the CIFAR GPU campaign.  Raw banks:
  data/tf_feats_galaxy10_lejepa_vitb16.pt         (train/test 768-d features)
  data/tf_augfeats_galaxy10_lejepa_vitb16_a8.pt   (8 augmented views of train)

    python experiments/170_galaxy_normfree.py --arm supconeucl --draw 0
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse, copy, importlib, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from supersig.config import DEVICE
from supersig.losses import (supcon_loss, sigreg_loss, HybridContrastiveLoss)
from supersig.discovery import run_discovery
from supersig.train import collect_embeddings
from supersig.holdouts import holdout_set

exp28 = importlib.import_module("28_concat_residual")
exp146 = importlib.import_module("146_min_frac_2sigma")
exp162 = importlib.import_module("162_cifar_concat_residual")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data")
OUT = os.path.join(REPO, "logs", "exp170")
DS, N_CLS = "galaxy10", 10
TESTS = ["eucl", "eucl-disc", "maha", "mmd", "sparker", "sparker-anch"]
HYBRID = {"supconeucl": "bilinear", "supcondist": "distance"}


class Head(nn.Module):
    def __init__(self, dim, feat=768, proj=False, proj_dim=128):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(feat, 256), nn.ReLU(),
                                  nn.Linear(256, dim))
        self.proj = (nn.Sequential(nn.Linear(dim, proj_dim), nn.ReLU(),
                                   nn.Linear(proj_dim, proj_dim))
                     if proj else None)

    def forward(self, x):            # tested embedding h
        return self.head(x)

    def project(self, x):
        return self.proj(self.head(x))


def train_head(arm, aug_seen, y_seen, dim, epochs, tau, lam, n_slices, seed):
    """aug_seen: (A, Ns, 768) augmented views of the SEEN train images."""
    torch.manual_seed(seed); np.random.seed(seed)
    net = Head(dim, proj=(arm == "supconproj")).to(DEVICE)
    hyb = (HybridContrastiveLoss(positives="supervised", critic=HYBRID[arm],
                                 estimator="softmax", marginal="none",
                                 tau=tau, n_slices=n_slices)
           if arm in HYBRID else None)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    A, Ns, _ = aug_seen.shape
    y_seen = y_seen.to(DEVICE)
    bs = 256
    net.train()
    for ep in range(epochs):
        perm = torch.randperm(Ns)
        run = 0.0; nb = 0
        for i in range(0, Ns, bs):
            idx = perm[i:i + bs]
            a, b = np.random.choice(A, 2, replace=False)
            v1 = aug_seen[a, idx].float().to(DEVICE)
            v2 = aug_seen[b, idx].float().to(DEVICE)
            yy = torch.cat([y_seen[idx], y_seen[idx]])
            opt.zero_grad()
            if arm == "supcon":
                z = net(torch.cat([v1, v2]))
                loss = supcon_loss(F.normalize(z, dim=1), yy, temp=tau_cos())
            elif arm == "supconproj":
                z = net.project(torch.cat([v1, v2]))
                loss = supcon_loss(F.normalize(z, dim=1), yy, temp=tau_cos())
            else:
                z = net(torch.cat([v1, v2]))
                loss = hyb.interaction(z, yy) + lam * sigreg_loss(z, n_slices=n_slices)
            loss.backward(); opt.step()
            run += float(loss) * len(idx); nb += len(idx)
        if (ep + 1) % 10 == 0 or ep == 0 or ep == epochs - 1:
            print(f"  [{arm}] epoch {ep+1}/{epochs} loss={run/max(nb,1):.4f}",
                  flush=True)
    net.eval()
    return net


def tau_cos():
    return 0.1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="supcon",
                    choices=["supcon", "supconproj", "supconeucl", "supcondist"])
    ap.add_argument("--draw", type=int, default=0)
    ap.add_argument("--dim", type=int, default=10)
    ap.add_argument("--base", default="lejepa")
    ap.add_argument("--trunk-ft", action="store_true",
                    help="load the exp-70 trunk-fine-tuned bank+head for this "
                         "arm instead of training a head over the frozen trunk")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=5.0)
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
    os.makedirs(args.out, exist_ok=True)

    holdouts = holdout_set(DS, N_CLS, nh=1, draw=args.draw)
    seen = [c for c in range(N_CLS) if c not in holdouts]
    fracs = [float(x) for x in args.fractions.split(",")]
    rep_weight = 20.0 * 45.0 / (N_CLS * (N_CLS - 1) / 2)
    cfg = dict(n_classes=N_CLS, pair_dist=5.0)

    if args.trunk_ft:
        # TRUNK FINE-TUNE: load the exp-70 fine-tuned bank + head for this arm
        # (trunk reshaped end-to-end, not head-only over the frozen raw trunk).
        exp163 = importlib.import_module("163_galaxy_concat_residual")
        FT_ARM = {"supcon": "supcon-ft", "supconeucl": "supconeucl-ft",
                  "supcondist": "supcondist-ft", "supconproj": "supconproj-ft"}
        loaded = exp163.load_head(args.base, FT_ARM[args.arm], args.draw, args.dim)
        if loaded is None:
            print(f"[miss] ft bank/head for {FT_ARM[args.arm]}", flush=True)
            sys.exit(2)
        net, Xtr, ytr, Xte, yte = loaded
        Xtr, Xte = Xtr.float(), Xte.float()
        tr_lab, te_lab = ytr.numpy(), yte.numpy()
        print(f"exp170 [{DS}] TRUNK-FT arm={args.arm} draw={args.draw} "
              f"holdout={sorted(holdouts)} dim={args.dim}", flush=True)
    else:
        raw = torch.load(os.path.join(DATA, f"tf_feats_{DS}_{args.base}_vitb16.pt"),
                         map_location="cpu")
        Xtr, ytr = raw["train"]; Xte, yte = raw["test"]
        Xtr, Xte = Xtr.float(), Xte.float()
        tr_lab, te_lab = ytr.numpy(), yte.numpy()
        aug = torch.load(os.path.join(DATA,
                         f"tf_augfeats_{DS}_{args.base}_vitb16_a8.pt"),
                         map_location="cpu")
        augf, auglab = aug["feats"], aug["labels"].numpy()
        seen_pos = np.where(np.isin(auglab, seen))[0]     # seen cols in aug
        aug_seen = augf[:, seen_pos, :]                   # (8, Ns, 768)
        y_seen = torch.as_tensor(auglab[seen_pos], dtype=torch.long)
        print(f"exp170 [{DS}] arm={args.arm} draw={args.draw} "
              f"holdout={sorted(holdouts)} dim={args.dim} epochs={args.epochs}",
              flush=True)
        net = train_head(args.arm, aug_seen, y_seen, args.dim, args.epochs,
                         args.tau, args.lam, args.n_slices, args.seed)

    tel = DataLoader(TensorDataset(Xtr, ytr), batch_size=512)
    tec = DataLoader(TensorDataset(Xte, yte), batch_size=512)
    R0_tr, _ = collect_embeddings(net, tel)
    R0_te, _ = collect_embeddings(net, tec)
    m = np.isin(tr_lab, seen)
    means0 = exp28.fill_means(exp28.class_centroids(R0_tr[m], tr_lab[m], seen),
                              seen, cfg).detach()
    seen_idx = np.where(m)[0]
    sig_idx = np.where(np.isin(tr_lab, list(holdouts)))[0]

    sfx = "_ft" if args.trunk_ft else ""
    res_path = os.path.join(args.out, f"{args.arm}{sfx}_{DS}_d{args.draw}_e{args.dim}.json")
    results = json.load(open(res_path)) if os.path.exists(res_path) else \
        dict(fractions=fracs, arm=args.arm, holdout=sorted(holdouts),
             pre={t: {} for t in TESTS}, post={t: {} for t in TESTS}, cut={})

    def z_of(fn, bg_t, sig_t, f, seed):
        na, sa = exp146.toys_battery(fn, len(bg_t), len(sig_t), [f],
                                     args.n_d, 200, 50, seed, tag="")
        return exp146.z_curve(na, sa)[0]

    for i_f, f in enumerate(fracs):
        fk = str(f)
        if fk in results["post"]["eucl"]:
            print(f"[skip] f={f}", flush=True); continue
        n_inj = int(round(f * len(seen_idx) / (1 - f)))
        rng = np.random.default_rng(args.seed * 1000 + i_f)
        inj = rng.choice(sig_idx, size=min(n_inj, len(sig_idx)), replace=False)
        sub_idx = np.concatenate([seen_idx, inj])
        sub = TensorDataset(Xtr[sub_idx], ytr[sub_idx])
        sub_loader = DataLoader(sub, batch_size=512, shuffle=False)
        bb = copy.deepcopy(net)
        print(f"\n== f={f} ({len(inj)} injected) discovery ==", flush=True)
        cur_means, hist = run_discovery(
            bb, means0.clone(), base_ds=sub, train_eval_loader=sub_loader,
            test_loader=tec, seen=seen, holdouts=holdouts, dataset_name=DS,
            rep_weight=rep_weight, sigreg_weight=1.0, n_slices=args.n_slices,
            rounds=args.rounds, ft_epochs=args.ft_epochs, names=None,
            seed=args.seed, pool_score="np", cut_rule="ssb", n_min=args.n_min,
            on_refuse="skip")
        results["cut"][fk] = dict(pur=float(hist[0]["purity"]) if hist else float("nan"))
        Rp_tr, _ = collect_embeddings(bb, tel)
        Rp_te, _ = collect_embeddings(bb, tec)
        del bb
        A = cur_means[N_CLS:].detach().cpu().numpy(); A = A if len(A) else None
        for state, (Ctr, Cte, Aa) in {"pre": (R0_tr, R0_te, None),
                                      "post": (Rp_tr, Rp_te, A)}.items():
            _, det = exp162._prep(Ctr, tr_lab, Cte, te_lab, seen, holdouts, Aa, args)
            bg_t, sig_t, nb = det["bg_t"], det["sig_t"], det["nb"]
            fns = {
                "eucl": det["meanf"](det["s_eu"][:nb], det["s_eu"][nb:]),
                "eucl-disc": (det["meanf"](det["s_ed"][:nb], det["s_ed"][nb:])
                              if det["s_ed"] is not None else None),
                "maha": det["meanf"](det["s_mb"], det["s_ms"]),
                "mmd": det["mmd_fn"], "sparker": det["spk"](None),
                "sparker-anch": (det["spk"](torch.as_tensor(
                    Aa, dtype=torch.float32, device=DEVICE))
                    if (Aa is not None and len(Aa)) else None)}
            for tn, fn in fns.items():
                results[state][tn][fk] = (z_of(fn, bg_t, sig_t, f, args.seed + i_f)
                                          if fn is not None else None)
            print(f"  {state}: eucl={results[state]['eucl'].get(fk)} "
                  f"maha={results[state]['maha'].get(fk)} "
                  f"mmd={results[state]['mmd'].get(fk)} "
                  f"spk={results[state]['sparker'].get(fk)}", flush=True)
        json.dump(results, open(res_path, "w"), indent=1)
        print(f"[done] f={f} pur={results['cut'][fk]['pur']:.3f}", flush=True)
    print("\nexp170 done.", flush=True)


if __name__ == "__main__":
    main()
