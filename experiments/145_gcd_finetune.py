"""
Experiment 145: the fine-tuned tier of the GCD benchmark (companion to exp 144).

Exp 144 evaluates our constructions on the FROZEN DINO trunk, which is the
k-means row of the GCD papers; GCD / UNO+ / SimGCD / RLCD all fine-tune the
ViT.  This script fine-tunes our exp-70 objectives on the GCD split --
labelled images only, i.e. 50% of the first-N classes -- and re-runs exp 144's
clustering rows on (a) the fine-tuned trunk's 768-D [CLS] features and (b)
the 100-D head space.  Same ACC, same D^u, same three seeds if asked.

The comparison is still asymmetric, in our disfavour: the published methods
train on D^l AND D^u (semi-supervised, 200 epochs, last block); we train on
D^l alone for 20 epochs (the exp-70 recipe, full trunk at lr 1e-5).  Our
discovery loop is what consumes D^u, and exp 144 showed it does not help a
K-way partition metric, so this is the honest reading: "what our objectives
buy on the trunk, versus their objectives plus unlabelled data".

    python experiments/145_gcd_finetune.py --dataset aircraft --arms supcon-ft --quick
    python experiments/145_gcd_finetune.py --dataset cars --arms supcon-ft ss-ft nplm-sup-ft --seeds 0
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import importlib
import json

import numpy as np
import torch

from supersig.config import DATA_DIR, DEVICE

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.path.join(REPO, "checkpoints")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="aircraft", choices=["cars", "aircraft"])
    ap.add_argument("--base", default="dino")
    ap.add_argument("--arms", nargs="+", default=["supcon-ft", "ss-ft", "nplm-sup-ft"])
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--ft-epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr-backbone", type=float, default=1e-5)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--emb-dim", type=int, default=100)
    ap.add_argument("--n-slices", type=int, default=64)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--out", default="logs/exp144")
    args = ap.parse_args()
    args.ft_epochs = args.ft_epochs or (1 if args.quick else 20)

    exp37 = importlib.import_module("37_dtd_vit")
    exp43 = importlib.import_module("43_dtd_finetune")
    exp44 = importlib.import_module("44_transfer_32d")
    exp49 = importlib.import_module("49_aircraft_ssl_ft")
    exp70 = importlib.import_module("70_cars_ft_suite")
    exp144 = importlib.import_module("144_gcd_benchmark")
    ds, base = args.dataset, args.base
    exp70.DS, exp70.BASE = ds, base
    n_known, K = exp144.KNOWN[ds], exp144.TOTAL[ds]
    qs = "_quick" if args.quick else ""

    corpus = exp43.train_corpus(ds)
    y_all = exp70.corpus_labels(corpus)
    eval_ds = exp44.make_split(ds, "train", exp37.TF_EVAL)
    specs = exp70.arm_specs(args)
    print(f"exp145 [{ds}/{base}] GCD fine-tune tier: known {n_known}/{K}, arms={args.arms}, "
          f"epochs={args.ft_epochs}", flush=True)

    results = {}
    for seed in [int(s) for s in args.seeds.split(",")]:
        lab = exp144.gcd_split(y_all, n_known, 0.5, seed)
        lab_idx = np.where(lab)[0].tolist()
        print(f"\n  seed {seed}: |D^l|={len(lab_idx)} |D^u|={int((~lab).sum())}", flush=True)
        out = {}
        for arm in args.arms:
            labeled, step = specs[arm]
            assert labeled, "GCD tier uses the supervised arms only"
            ck = os.path.join(CKPT_DIR, f"{ds}_ft_{base}_{arm}_gcd_s{seed}{qs}.pt")
            torch.manual_seed(seed + 20); np.random.seed(seed + 20)
            model = exp43.FineTuneModel(base, args.emb_dim)
            if os.path.exists(ck) and not args.refresh:
                print(f"  [{arm}] loading {ck}")
                model.load_state_dict(torch.load(ck, map_location=DEVICE))
            else:
                loader = exp70.seen_two_view_loader(corpus, lab_idx, args)
                exp49.ft_loop(model, loader, args.ft_epochs, step, args, f"{arm}-gcd")
                torch.save(model.state_dict(), ck)
                del loader
            model.eval()
            Xtr, ytr = exp37.extract(model.trunk.eval(), eval_ds)
            ytr = ytr.numpy(); assert (ytr == y_all).all(), "corpus/eval label order mismatch"
            H = exp37.embed(model.head.float(), Xtr.float()).numpy()
            del model; torch.cuda.empty_cache()
            exp144.run_space(f"{arm} trunk", Xtr.float().numpy(), ytr, lab, n_known, K, seed, out)
            exp144.run_space(f"{arm} head", H, ytr, lab, n_known, K, seed, out)
        results[seed] = out

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"gcd_ft_{ds}_{base}{qs}.json"), "w") as fh:
        json.dump(dict(dataset=ds, base=base, n_known=n_known, K=K, results=results,
                       reference=exp144.REF[ds], epochs=args.ft_epochs), fh, indent=1, default=float)
    print(f"\n===== EXP145 [{ds}] fine-tuned tier, ACC on D^u (All/Old/New, mean+-sd over {len(results)} seeds) =====")
    for name in list(next(iter(results.values())).keys()):
        for m in ("kmeans", "ss-kmeans", "np-anchors", "np-seeded"):
            v = np.array([[results[s][name][m][k] for k in ("all", "old", "new")] for s in results])
            print(f"  {name:20s} {m:11s} " + "  ".join(f"{v[:, i].mean():5.1f}+-{v[:, i].std():4.1f}" for i in range(3)))
    for k, v in exp144.REF[ds].items():
        print(f"  {'reference':20s} {k:16s} {v[0]:5.1f}      {v[1]:5.1f}      {v[2]:5.1f}")
    print("EXP145 DONE.")


if __name__ == "__main__":
    main()
