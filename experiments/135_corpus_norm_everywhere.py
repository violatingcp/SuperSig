"""
Experiment 135: corpus-adaptive normalisation on EVERY cell -- the ViT
analogue of exp 133's BatchNorm adaptation.

WHY.  Exp 133 showed that what the "frozen" CIFAR loop was silently doing --
re-standardising a frozen trunk's features with statistics of the discovery
corpus (novel points included), no labels -- is a real construction: it
restores the round-2 purity rise (h10 0.237 -> 0.299) on the same checkpoint.
It only existed on the CIFAR ResNet because the transfer trunks are LayerNorm
ViTs with no running statistics.  This makes the same construction available
everywhere and runs the same paired A/B on every transfer cell and draw.

THE ANALOGUE.  Feature-space discovery (exp 70's "natural discovery") trains
the projection HEAD on cached 768-D trunk features.  We put a running-stats
normaliser on the head's input:

    CorpusNorm(head) = BatchNorm1d(768, affine=False, momentum=None) -> head

initialised from the SEEN-labelled train features, head weights frozen.  Then:

    A  frozen        normaliser in eval(): fixed seen statistics.  This is
                     "freeze the encoder, iterate only the anchors" (exp 86
                     freeze-both / exp 109 frozen), on the transfer trunks.
    B  corpus-norm   normaliser adapting (train mode) during the anchor
                     fine-tune: statistics absorb the pooled corpus each
                     round.  Same switch as exp 133: bn_adapt=True.
    C  unfrozen      the archived exp-70 natural discovery (head trains),
                     read from results_{ds}_{base}_ft70{tag}.npz for context.

Both pool scorers (distance = campaign default; np = density ratio, the
exp-109/133 construction) x 2 rounds x 5 head-ft epochs, seen-class arms
{supcon-ft, ss-ft, nplm-sup-ft}.  Per arm: purity r1/r2, margin, post probe /
eucl / mahaT / per-event, and the test-set drift |z_post - z_pre| / |z| --
the direct "did the space move" measure.  Evaluation-scale (banks cached).

PREDICTION.  B >= A on round-2 purity where round-1 purity clears the gate
(galaxy10 lejepa/visreg easy draws, dtd, flowers), with drift of order 10-30%
of |z| and no probe cost; ~no change where round-1 purity is ~0 (cars,
aircraft at multi-holdout; hard galaxy10 draws).  Same shape as exp 133.

FALSIFIER.  B < A where discovery worked -> the construction was a ResNet/BN
peculiarity and does not transfer.  Or B helps where round-1 purity ~0 -> the
gain is not "richer anchors compound" but something the normaliser does on
its own, and exp 133's reading needs restating.

    python experiments/135_corpus_norm_everywhere.py --selftest
    python experiments/135_corpus_norm_everywhere.py --cells galaxy10:dino
    SUPERSIG_NH=1 SUPERSIG_HOLDOUT_DRAW=3 python experiments/135_corpus_norm_everywhere.py --cells galaxy10:lejepa
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
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from supersig.config import DEVICE
from supersig.discovery import run_discovery
from supersig.train import collect_embeddings, backbone_is_frozen

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.path.join(REPO, "checkpoints")
DATA_DIR = os.path.join(REPO, "data")
ARMS = ["supcon-ft", "ss-ft", "nplm-sup-ft"]
VARIANTS = {"frozen": False, "corpus-norm": True}
FEAT_DIM = 768


class CorpusNorm(nn.Module):
    """Running-statistics normaliser in front of a (frozen) head.

    affine=False so it has NO parameters: backbone_is_frozen() stays True and
    set_train_mode(bn_adapt=...) decides whether the statistics adapt -- the
    exact switch exp 133 used on the ResNet's BatchNorm2d layers.
    momentum=None: cumulative average over everything it has seen, so the
    corpus statistics accumulate rather than chase the last batch."""

    def __init__(self, head, dim=FEAT_DIM):
        super().__init__()
        self.norm = nn.BatchNorm1d(dim, affine=False, momentum=None)
        self.head = head

    def forward(self, x):
        return self.head(self.norm(x))

    @torch.no_grad()
    def init_stats(self, X, bs=2048):
        """Seed the running statistics from the SEEN train features."""
        dev = self.norm.running_mean.device        # follow the module
        self.norm.reset_running_stats()
        self.norm.train()
        for i in range(0, len(X), bs):
            self.norm(X[i:i + bs].to(dev))
        self.norm.eval()
        return self


def drift(z_pre, z_post):
    d = np.linalg.norm(z_post - z_pre, axis=1)
    nz = np.linalg.norm(z_pre, axis=1).mean()
    return dict(mean=float(d.mean()), max=float(d.max()), mean_abs_z=float(nz),
                rel=float(d.mean() / nz) if nz > 0 else float("nan"))


def _selftest():
    torch.manual_seed(0)
    head = nn.Linear(8, 3)
    for p in head.parameters():
        p.requires_grad_(False)
    X = torch.randn(64, 8) * 3 + 1
    m = CorpusNorm(head, dim=8).to(DEVICE).init_stats(X)
    assert backbone_is_frozen(m), "CorpusNorm must add no parameters"
    print("  CorpusNorm(head) has no trainable parameters -> counts as frozen  OK")
    mu0 = m.norm.running_mean.clone()
    from supersig.train import set_train_mode
    set_train_mode(m)                            # A
    assert not m.training
    m((torch.randn(32, 8) * 9 - 4).to(DEVICE))
    assert torch.equal(m.norm.running_mean, mu0)
    print("  bn_adapt=False: stats fixed at the seen initialisation        OK")
    set_train_mode(m, bn_adapt=True)             # B
    assert m.training
    m((torch.randn(32, 8) * 9 - 4).to(DEVICE))
    assert not torch.equal(m.norm.running_mean, mu0)
    print("  bn_adapt=True : stats absorb the corpus batch                 OK")
    w = head.weight.clone()
    m.eval(); m(X.to(DEVICE))
    assert torch.equal(w, head.weight)
    print("  head weights never move                                       OK")
    # eval() output uses running stats: identical inputs -> identical outputs
    m.eval(); a = m(X[:5].to(DEVICE)); b = m(X[:5].to(DEVICE))
    assert torch.allclose(a, b)
    print("  eval() is deterministic (running stats, not batch stats)      OK")
    print("\nselftest OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--cells", default="galaxy10:dino")
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--scorers", default="dist,np")
    ap.add_argument("--emb-dim", type=int, default=100)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--ft-epochs", type=int, default=5)
    ap.add_argument("--n-slices", type=int, default=64)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="logs/exp135")
    args = ap.parse_args()
    if args.selftest:
        _selftest(); return

    exp28 = importlib.import_module("28_concat_residual")
    exp29 = importlib.import_module("29_residual_finetune")
    exp30 = importlib.import_module("30_power_curves")
    exp37 = importlib.import_module("37_dtd_vit")
    exp43 = importlib.import_module("43_dtd_finetune")
    exp44 = importlib.import_module("44_transfer_32d")
    exp70 = importlib.import_module("70_cars_ft_suite")
    os.makedirs(args.out, exist_ok=True)
    tag = run_tag()
    results = {}

    for cell in args.cells.split(","):
        ds, base = cell.split(":")
        n_cls = 47 if ds == "dtd" else exp44.N_CLASSES[ds]
        holdouts = holdout_set(ds, n_cls)
        seen = [c for c in range(n_cls) if c not in holdouts]
        cfg = dict(n_classes=n_cls, pair_dist=5.0, sigreg_weight=1.0,
                   n_slices=args.n_slices,
                   rep_weight=exp70.REP_WEIGHT * 45.0 / (n_cls * (n_cls - 1) / 2))
        f70 = os.path.join("logs", "exp70", f"results_{ds}_{base}_ft70{tag}.npz")
        arch = np.load(f70, allow_pickle=True) if os.path.exists(f70) else None
        print(f"\n######## {cell}{tag}: holdouts={sorted(holdouts)[:4]}"
              f"{'...' if len(holdouts) > 4 else ''} ########", flush=True)
        for arm in args.arms:
            ck = os.path.join(CKPT_DIR, f"{ds}_ft_{base}_{arm}_seen{tag}.pt")
            bp = os.path.join(DATA_DIR, f"tf_feats_{ds}_{base}_ft70_{arm}{tag}.pt")
            if not (os.path.exists(ck) and os.path.exists(bp)):
                print(f"  [{arm}] missing {'ckpt' if not os.path.exists(ck) else 'bank'}, skip")
                continue
            mod = exp43.FineTuneModel(base, args.emb_dim)
            mod.load_state_dict(torch.load(ck, map_location=DEVICE))
            head0 = copy.deepcopy(mod.head).float().to(DEVICE)
            del mod
            for p in head0.parameters():
                p.requires_grad_(False)
            b = torch.load(bp)
            (Xtr, ytr), (Xte, yte) = b["train"], b["test"]
            Xtr, Xte = Xtr.float(), Xte.float()
            tr_lab, te_lab = ytr.numpy(), yte.numpy()
            m_seen = np.isin(tr_lab, seen)
            base_feats = TensorDataset(Xtr, ytr)
            tel = DataLoader(base_feats, batch_size=512, shuffle=False)
            tsl = DataLoader(TensorDataset(Xte, yte), batch_size=512, shuffle=False)

            def battery(net):
                tr, _ = collect_embeddings(net, tel)
                te, _ = collect_embeddings(net, tsl)
                anch = exp28.class_centroids(tr[m_seen], tr_lab[m_seen], seen) \
                    .detach().float().to(DEVICE)
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
                return dict(probe=float(np.mean(aucs)), probe_sd=float(np.std(aucs)),
                            acc=r["acc"], eucl=r["eucl"], mahaT=r["maha_tied"],
                            perevt=pe), te

            net0 = CorpusNorm(copy.deepcopy(head0)).to(DEVICE).init_stats(Xtr[m_seen])
            pre, te0 = battery(net0)
            print(f"\n  --- {arm}: pre probe={pre['probe']:.4f} eucl={pre['eucl']:.4f} "
                  f"mahaT={pre['mahaT']:.4f} perevt={pre['perevt']:.3f}"
                  + (f"   [archived unfrozen post probe "
                     f"{float(arch[f'post_probe_{arm}']):.4f}]"
                     if arch is not None and f"post_probe_{arm}" in arch.files else ""),
                  flush=True)
            cents = exp28.class_centroids(
                collect_embeddings(net0, tel)[0][m_seen], tr_lab[m_seen], seen)
            means0 = exp28.fill_means(cents, seen, cfg).detach()

            for scorer in args.scorers.split(","):
                for var, adapt in VARIANTS.items():
                    net = copy.deepcopy(net0).to(DEVICE)
                    torch.manual_seed(args.seed); np.random.seed(args.seed)
                    _, hist = run_discovery(
                        net, means0.clone(), base_ds=base_feats,
                        train_eval_loader=tel, test_loader=tsl, seen=seen,
                        holdouts=holdouts, dataset_name=ds,
                        rep_weight=cfg["rep_weight"],
                        sigreg_weight=cfg["sigreg_weight"],
                        n_slices=cfg["n_slices"], rounds=args.rounds,
                        ft_epochs=1 if args.quick else args.ft_epochs,
                        seed=args.seed, pool_score=scorer, bn_adapt=adapt)
                    post, te1 = battery(net)
                    dr = drift(te0, te1)
                    key = f"{cell}|{arm}|{scorer}|{var}"
                    results[key] = dict(cell=cell, arm=arm, scorer=scorer,
                                        variant=var, bn_adapt=adapt, pre=pre,
                                        post=post, drift=dr,
                                        purity=[h["purity"] for h in hist],
                                        pool=[h["pool"] for h in hist],
                                        margin=[h["margin"] for h in hist])
                    print(f"    [{scorer:4s} {var:11s}] purity "
                          + " ".join(f"r{h['round']}={h['purity']:.3f}" for h in hist)
                          + f"  probe {pre['probe']:.3f}->{post['probe']:.3f}"
                          f"  eucl {pre['eucl']:.3f}->{post['eucl']:.3f}"
                          f"  perevt {pre['perevt']:.3f}->{post['perevt']:.3f}"
                          f"  drift {dr['rel']:.3f}", flush=True)
                    del net; torch.cuda.empty_cache()

    print(f"\n===== EXP135 SUMMARY{tag} (paired frozen vs corpus-norm; gate 0.15) =====")
    print(f"  {'cell|arm|scorer':<40}{'r1':>7}{'r2 frz':>8}{'r2 cn':>8}{'d r2':>8}"
          f"{'probe frz/cn':>16}{'drift':>7}")
    for key, r in results.items():
        if r["variant"] != "frozen":
            continue
        cn = results.get(key.replace("|frozen", "|corpus-norm"))
        if not cn:
            continue
        p = r["purity"] + [float("nan")] * (2 - len(r["purity"]))
        q = cn["purity"] + [float("nan")] * (2 - len(cn["purity"]))
        print(f"  {key.replace('|frozen', ''):<40}{p[0]:>7.3f}{p[1]:>8.3f}{q[1]:>8.3f}"
              f"{q[1] - p[1]:>+8.3f}{r['post']['probe']:>8.3f}/{cn['post']['probe']:<7.3f}"
              f"{cn['drift']['rel']:>7.3f}")
    src = args.cells.replace(":", "-").replace(",", "_")
    # The seed goes in the filename or a multi-seed sweep silently overwrites
    # itself (exp 139).  seed 0 keeps the archived name, so nothing already
    # written is orphaned.
    stag = "" if args.seed == 0 else f"_s{args.seed}"
    out_path = os.path.join(args.out, f"corpus_norm_{src}{tag}{stag}.json")
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=1, default=float)
    print(f"\nwrote {out_path}\nEXP135 DONE.")


if __name__ == "__main__":
    main()
