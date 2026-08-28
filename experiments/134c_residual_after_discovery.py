"""
Experiment 134c: should the residual be built AFTER discovery rather than before?

Question (C) of exp 134.  Every residual in the campaign is trained against
the PRE-discovery parent's seen-class centroids, and discovery is then run on
the concat.  The residual removes the variance the ANCHOR SET explains; after
discovery the anchor set is strictly richer (it holds the discovered anchors
too), so the same construction removes more, and the child should carry more
of what is genuinely unexplained.

ORDERS.  archived  parent -> child -> concat            (exp 71; loaded, not re-run)
         new       parent -> discovery -> child -> concat

The NEW order, per cell and objective:
  1. load the exp-70 parent (trunk + head), extract its 768-D banks;
  2. run the settled discovery loop on the parent HEAD in feature space
     (exactly exp 70's "natural discovery": pool -> BIC k-means -> proto ft of
     the head, 2 rounds) -> a post-discovery head and an ENLARGED anchor set
     (seen centroids + k discovered);
  3. relabel the corpus label-free against that anchor set: seen images keep
     their labels; every other corpus image whose nearest anchor is a
     DISCOVERED one joins the child corpus under that anchor's index (holdout
     labels are never read);
  4. train the child (exp-71 objective, res or res-nplm) on that corpus with
     r = z - anchor[y'] against the enlarged set, starting from the parent
     trunk + the post-discovery head;
  5. evaluate parent-post, child, concat with exp 71's battery and put the
     archived exp-71 rows beside them.

PREDICTION.  The new order raises child informativeness (probe, eucl) where
discovery WORKED (round-1 purity above the 0.15 gate) and changes nothing
where it did not.  A free improvement in the cells the paper cares about, a
no-op elsewhere.

FALSIFIER.  It helps where purity was ~0 -> the mechanism story (richer anchor
set -> more explained variance removed) is wrong and whatever helps is
something else (e.g. simply more fine-tuning).  Or it HURTS where discovery
worked -> the child is fitting anchors planted on the novel class and removing
exactly the structure the concat needed.

COST.  Per (cell, objective): one feature-space discovery (minutes) + one
exp-71-scale child fine-tune (13-40 min on the ViT).  Honours SUPERSIG_NH /
SUPERSIG_HOLDOUT_DRAW; artifacts tagged `_postdisc`.

    python experiments/134c_residual_after_discovery.py --selftest
    python experiments/134c_residual_after_discovery.py --dataset galaxy10 --base dino
    python experiments/134c_residual_after_discovery.py --dataset cars --base visreg --objs res-nplm
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from supersig.holdouts import holdout_set, n_holdout, run_tag
import argparse
import copy
import importlib
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

from supersig.config import DEVICE
from supersig.discovery import run_discovery
from supersig.train import collect_embeddings

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.path.join(REPO, "checkpoints")


class Relabeled(Dataset):
    """Two augmented views of corpus[idx[i]] with an EXTERNAL label.  The
    corpus's own label is never read, so pseudo-labelled novel-candidate
    images can join the child corpus without touching holdout labels."""

    def __init__(self, corpus, idx, labels, aug):
        assert len(idx) == len(labels)
        self.corpus, self.idx, self.labels, self.aug = corpus, idx, labels, aug

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        img, _ = self.corpus[int(self.idx[i])]
        return self.aug(img), self.aug(img), int(self.labels[i])


def relabel_against_anchors(H, is_seen_lab, tr_lab, anchors, n_cls):
    """Label-free corpus relabelling against the enlarged anchor set.

    Seen-labelled rows keep their label.  Every other row is assigned to its
    nearest anchor; it joins the child corpus ONLY if that anchor is a
    discovered one (index >= n_cls).  Returns (idx, labels, n_pseudo)."""
    Z = torch.as_tensor(np.asarray(H, np.float32), device=DEVICE)
    A = torch.as_tensor(anchors, dtype=torch.float32, device=DEVICE)
    near = torch.cdist(Z, A).argmin(1).cpu().numpy()
    idx, lab = [], []
    for i in range(len(H)):
        if is_seen_lab[i]:
            idx.append(i); lab.append(int(tr_lab[i]))
        elif near[i] >= n_cls:
            idx.append(i); lab.append(int(near[i]))
    return np.asarray(idx), np.asarray(lab), int(sum(l >= n_cls for l in lab))


def _selftest():
    rng = np.random.default_rng(0)
    n_cls, k, d = 5, 2, 8
    A = rng.normal(0, 4, (n_cls + k, d))
    y = np.repeat(np.arange(n_cls + k), 30)          # true ids, last k "novel"
    H = A[y] + rng.normal(0, 0.3, (len(y), d))
    tr_lab = np.where(y < n_cls, y, 99)                # holdout label is opaque
    is_seen = y < n_cls
    idx, lab, n_ps = relabel_against_anchors(H, is_seen, tr_lab, A, n_cls)
    assert (lab[is_seen[idx]] == tr_lab[idx][is_seen[idx]]).all()
    assert n_ps == 2 * 30 and set(lab[~is_seen[idx]]) == {n_cls, n_cls + 1}
    print(f"  seen rows keep labels; {n_ps} novel rows join under discovered "
          f"anchors {sorted(set(lab[~is_seen[idx]]))}          OK")
    # a novel row whose nearest anchor is a SEEN one must be left out
    H2 = H.copy(); H2[~is_seen] = A[0] + 0.01
    idx2, lab2, n2 = relabel_against_anchors(H2, is_seen, tr_lab, A, n_cls)
    assert n2 == 0 and len(idx2) == is_seen.sum()
    print("  novel rows nearest a SEEN anchor are excluded (no pseudo-label)  OK")

    class Toy(Dataset):
        def __len__(self): return 4
        def __getitem__(self, i): return torch.full((3,), float(i)), 1000 + i
    rl = Relabeled(Toy(), np.array([3, 1]), np.array([7, 9]), aug=lambda t: t * 2)
    v1, v2, lab = rl[0]
    assert float(v1[0]) == 6.0 and float(v2[0]) == 6.0 and lab == 7 and len(rl) == 2
    print("  Relabeled returns two views + the external label, never the "
          "corpus label  OK")
    print("\nselftest OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dataset", default="galaxy10",
                    choices=["cars", "aircraft", "flowers", "galaxy10", "dtd"])
    ap.add_argument("--base", default="dino", choices=["dino", "lejepa", "visreg"])
    ap.add_argument("--parent", default="supcon-ft")
    ap.add_argument("--objs", default="res,res-nplm")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ft-epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr-backbone", type=float, default=1e-5)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--emb-dim", type=int, default=100)
    ap.add_argument("--n-slices", type=int, default=64)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--out", default="logs/exp134")
    args = ap.parse_args()
    if args.selftest:
        _selftest(); return
    args.ft_epochs = args.ft_epochs or (1 if args.quick else 20)

    exp28 = importlib.import_module("28_concat_residual")
    exp29 = importlib.import_module("29_residual_finetune")
    exp30 = importlib.import_module("30_power_curves")
    exp37 = importlib.import_module("37_dtd_vit")
    exp43 = importlib.import_module("43_dtd_finetune")
    exp44 = importlib.import_module("44_transfer_32d")
    exp49 = importlib.import_module("49_aircraft_ssl_ft")
    exp70 = importlib.import_module("70_cars_ft_suite")
    exp71 = importlib.import_module("71_residual_ft_grid")
    DS, BASE = args.dataset, args.base
    exp70.DS, exp70.BASE = DS, BASE
    exp71.DS, exp71.BASE = DS, BASE
    N_CLS = 47 if DS == "dtd" else exp44.N_CLASSES[DS]
    holdouts = holdout_set(DS, N_CLS)
    seen = [c for c in range(N_CLS) if c not in holdouts]
    tag = f"{DS}_{BASE}_ft134c{run_tag()}{exp70.seed_sfx(args)}"
    REP = exp70.REP_WEIGHT * 45.0 / (N_CLS * (N_CLS - 1) / 2)
    cfg = dict(n_classes=N_CLS, pair_dist=5.0, sigreg_weight=1.0,
               n_slices=args.n_slices, rep_weight=REP)
    print(f"exp134c [{tag}] residual AFTER discovery; parent={args.parent} "
          f"objs={args.objs} holdouts={sorted(holdouts)}")

    # ---- 1. parent + banks ------------------------------------------------
    pck = os.path.join(CKPT_DIR, f"{DS}_ft_{BASE}_{args.parent}_seen"
                       f"{run_tag()}{exp70.seed_sfx(args)}"
                       f"{'_quick' if args.quick else ''}.pt")
    assert os.path.exists(pck), f"missing parent {pck} (run exp 70 first)"
    par = exp43.FineTuneModel(BASE, args.emb_dim)
    par.load_state_dict(torch.load(pck, map_location=DEVICE))
    pb = exp70.trunk_banks(par, args.parent, args)
    (Xtr, ytr), (Xte, yte) = pb["train"], pb["test"]
    tr_lab, te_lab = ytr.numpy(), yte.numpy()
    is_seen = np.isin(tr_lab, seen)
    ph = par.head.float()
    pHtr = exp37.embed(ph, Xtr.float()).numpy()
    pHte = exp37.embed(ph, Xte.float()).numpy()

    def battery(name, tr, te):
        m = np.isin(tr_lab, seen)
        anch = exp28.class_centroids(tr[m], tr_lab[m], seen).detach().float().to(DEVICE)
        r = exp29.evaluate_space(tr, tr_lab, te, te_lab, anch, seen, holdouts)
        aucs = []
        for s in range(3):
            torch.manual_seed(1000 + s)
            a, _, _ = exp29.linear_probe_novelty(tr, tr_lab, te, te_lab, holdouts)
            aucs.append(a)
        d = torch.cdist(torch.as_tensor(te, dtype=torch.float32, device=DEVICE), anch)
        s_ = d.min(1).values.cpu().numpy()
        pe = exp30.power_at_alpha(s_[np.isin(te_lab, seen)],
                                  s_[np.isin(te_lab, list(holdouts))], args.alpha)
        out = dict(probe=float(np.mean(aucs)), probe_sd=float(np.std(aucs)),
                   acc=r["acc"], eucl=r["eucl"], mahaT=r["maha_tied"],
                   lid=r["lid"], perevt=pe)
        print(f"  [{name:<34}] probe={out['probe']:.4f}+-{out['probe_sd']:.4f}"
              f" acc={out['acc']:.4f} eucl={out['eucl']:.4f} "
              f"mahaT={out['mahaT']:.4f} lid={out['lid']:.4f} perevt={pe:.3f}",
              flush=True)
        return out

    results = {"parent (pre)": battery("parent (pre)", pHtr, pHte)}

    # ---- 2. discovery on the parent head (feature space, exp-70 recipe) ----
    print(f"\n----- discovery on the parent head -----", flush=True)
    base_feats = TensorDataset(Xtr.float(), ytr)
    tel = DataLoader(base_feats, batch_size=512, shuffle=False)
    tsl = DataLoader(TensorDataset(Xte.float(), yte), batch_size=512, shuffle=False)
    cents = exp28.class_centroids(pHtr[is_seen], tr_lab[is_seen], seen)
    means0 = exp28.fill_means(cents, seen, cfg).detach()
    head_post = copy.deepcopy(par.head).float()
    torch.manual_seed(args.seed)
    means_ext, hist = run_discovery(
        head_post, means0.clone(), base_ds=base_feats, train_eval_loader=tel,
        test_loader=tsl, seen=seen, holdouts=holdouts, dataset_name=DS,
        rep_weight=cfg["rep_weight"], sigreg_weight=cfg["sigreg_weight"],
        n_slices=cfg["n_slices"], rounds=args.rounds,
        ft_epochs=1 if args.quick else 5, seed=args.seed)
    for h in hist:
        print(f"  round {h['round']}: pool={h['pool']} purity={h['purity']:.3f} "
              f"anchors={h['n_anchors']} margin={h['margin']:.4f}")
    k_disc = int(means_ext.size(0) - N_CLS)
    qHtr, _ = collect_embeddings(head_post, tel)
    qHte, _ = collect_embeddings(head_post, tsl)
    results["parent (post-discovery)"] = battery("parent (post-discovery)", qHtr, qHte)

    # ---- 3. label-free relabelling against the enlarged anchor set --------
    idx, lab, n_ps = relabel_against_anchors(qHtr, is_seen, tr_lab,
                                             means_ext.detach().cpu().numpy(), N_CLS)
    n_true_novel = int(np.isin(tr_lab[idx][lab >= N_CLS], list(holdouts)).sum())
    print(f"  child corpus: {int(is_seen.sum())} seen + {n_ps} pseudo-labelled "
          f"under {k_disc} discovered anchors "
          f"(pseudo-label purity {n_true_novel / max(n_ps, 1):.3f}; "
          f"diagnostic only)", flush=True)

    # ---- 4. child against the enlarged anchor set -------------------------
    corpus = exp43.train_corpus(DS)
    cents_ext = means_ext.detach().float().to(DEVICE)
    child_rows = {}
    for obj in args.objs.split(","):
        key = f"{args.parent}->{obj} (post-discovery)"
        print(f"\n===== [{tag}] {key} =====", flush=True)
        rck = os.path.join(CKPT_DIR, f"{DS}_ft_{BASE}_{args.parent}_"
                           f"{obj.replace('-', '')}_postdisc_seen"
                           f"{run_tag()}{exp70.seed_sfx(args)}"
                           f"{'_quick' if args.quick else ''}.pt")
        child = copy.deepcopy(par)
        child.head = copy.deepcopy(head_post).to(DEVICE)
        if os.path.exists(rck) and not args.refresh:
            print(f"  loading {rck}")
            child.load_state_dict(torch.load(rck, map_location=DEVICE))
        else:
            torch.manual_seed(args.seed + 7); np.random.seed(args.seed + 7)
            step = (exp49.make_residual_step(cents_ext, 5.0) if obj == "res"
                    else exp71.make_res_nplm_step(cents_ext, 1.0, args.n_slices))
            loader = DataLoader(Relabeled(corpus, idx, lab, exp37.TF_AUG),
                                batch_size=args.batch_size, shuffle=True,
                                num_workers=8, persistent_workers=True,
                                drop_last=True, pin_memory=True)
            exp49.ft_loop(child, loader, args.ft_epochs, step, args,
                          f"{args.parent}-{obj}-postdisc")
            torch.save(child.state_dict(), rck)
            del loader
        rb = exp70.trunk_banks(child, f"{args.parent}_{obj.replace('-', '')}_postdisc", args)
        ch = child.head.float()
        rHtr = exp37.embed(ch, rb["train"][0].float()).numpy()
        rHte = exp37.embed(ch, rb["test"][0].float()).numpy()
        del child; torch.cuda.empty_cache()
        results[f"{key} residual"] = battery(f"{key} residual", rHtr, rHte)
        results[f"{key} concat"] = battery(
            f"{key} concat", np.concatenate([qHtr, rHtr], 1),
            np.concatenate([qHte, rHte], 1))

    # ---- 5. archived (pre-discovery) exp-71 rows for the same cell ---------
    arch = {}
    f71 = os.path.join("logs", "exp71", f"results_{DS}_{BASE}_ft71{run_tag()}"
                       f"{exp70.seed_sfx(args)}.npz")
    if os.path.exists(f71):
        d = np.load(f71, allow_pickle=True)
        for sp in d["spaces"]:
            sp = str(sp)
            if sp.startswith(args.parent):
                key71 = sp.replace(" ", "_").replace("->", "-")   # npz key form
                arch[sp] = {m: float(d[f"{key71}__{m}"]) for m in
                            ("probe", "eucl", "mahaT", "perevt")
                            if f"{key71}__{m}" in d.files}
    print(f"\n===== EXP134c SUMMARY [{tag}]  (discovery r1 purity "
          f"{hist[0]['purity']:.3f}, gate 0.15) =====")
    print(f"  {'space':<44}{'probe':>8}{'eucl':>8}{'mahaT':>8}{'perevt':>8}")
    for k, r in results.items():
        print(f"  {k:<44}{r['probe']:>8.4f}{r['eucl']:>8.4f}{r['mahaT']:>8.4f}{r['perevt']:>8.3f}")
    for k, r in arch.items():
        print(f"  {'[archived exp71] ' + k:<44}{r.get('probe', float('nan')):>8.4f}"
              f"{r.get('eucl', float('nan')):>8.4f}{r.get('mahaT', float('nan')):>8.4f}"
              f"{r.get('perevt', float('nan')):>8.3f}")
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"postdisc_{tag}.json"), "w") as fh:
        json.dump(dict(results=results, archived_exp71=arch,
                       discovery=[{k: v for k, v in h.items() if k != "per_class"}
                                  for h in hist],
                       n_pseudo=n_ps, k_discovered=k_disc,
                       pseudo_purity=n_true_novel / max(n_ps, 1)),
                  fh, indent=1, default=float)
    print("EXP134c DONE.")


if __name__ == "__main__":
    main()
