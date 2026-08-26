"""
Experiment 102 (IMPROVEMENT_TESTS.md #102): is the residual child a
standalone detector?

Exp 80's surprise: the res-nplm child out-detects its own parent at the
dataset level (aircraft/visreg 0.82@0.05 vs 0.44).  Every deployment uses
the child inside a concat.  Full battery on the residual halves ALONE
across the 12 exp-71 champion cells: probe, eucl/mahaT/lid, per-event,
dense-grid SparKer f95 (exp-100 fractions), plus the exp-76 semantic
metrics (superclass agreement/purity where a partition exists) -- exp 76
showed the residual half scrambles semantics, so a child-only detector
may be undiagnosable even when it fires.

Prediction: child-alone matches/beats the concat on dataset-level power
at a large probe cost and is semantically illegible; the deliverable is
the trade curve.  Falsifier: child-alone power does not survive being
scored on its own (the concat was doing the work).

    python experiments/102_child_standalone.py
    python experiments/102_child_standalone.py --cells cars:visreg --quick
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from supersig.holdouts import n_holdout, run_tag
import argparse
import importlib
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from supersig.config import DEVICE

exp28 = importlib.import_module("28_concat_residual")
exp29 = importlib.import_module("29_residual_finetune")
exp30 = importlib.import_module("30_power_curves")
exp31 = importlib.import_module("31_sparker_power")
exp44 = importlib.import_module("44_transfer_32d")
exp72 = importlib.import_module("72_residual_discovery")
exp76 = importlib.import_module("76_interpretability")
exp77 = importlib.import_module("77_space_similarity")
exp99 = importlib.import_module("99_discovery_reach")

CELLS = [f"{d}:{b}" for d in ("aircraft", "cars", "flowers", "dtd",
                              "galaxy10") for b in ("dino", "lejepa",
                                                    "visreg")]
FRACS = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=",".join(CELLS))
    ap.add_argument("--emb-dim", type=int, default=100)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    n_null = 20 if args.quick else 200
    n_toys = 10 if args.quick else 50
    sparker_kw = dict(M=16, steps=50 if args.quick else 300)
    out_path = os.path.join("logs", "exp102", f"results{run_tag()}.npz")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    done = {}
    if os.path.exists(out_path) and not args.refresh:
        d = np.load(out_path, allow_pickle=True)
        done = {k: d[k] for k in d.files}

    rows = {}
    for cell in args.cells.split(","):
        ds, base = cell.split(":")
        parent, obj, kind = exp72.WINNERS[(ds, base)]
        key = cell.replace(":", "_")
        n_cls = 47 if ds == "dtd" else exp44.N_CLASSES[ds]
        nh = n_holdout(ds)
        holdouts = set(range(n_cls - nh, n_cls))
        seen = [c for c in range(n_cls) if c not in holdouts]
        n_d = 2000 if ds in ("cars", "galaxy10") else 1000
        c_tr = exp77.head_emb(ds, base, f"{parent}_{obj}", args.emb_dim,
                              "train")
        c_te = exp77.head_emb(ds, base, f"{parent}_{obj}", args.emb_dim,
                              "test")
        if not (c_tr and c_te):
            print(f"  !! [{cell}] missing child banks, skipping")
            continue
        (Xtr, ytr), (Xte, yte) = c_tr, c_te
        print(f"\n######## [{cell}] child {parent}->{obj} standalone "
              f"########", flush=True)

        m = np.isin(ytr, seen)
        anch = exp28.class_centroids(Xtr[m], ytr[m],
                                     seen).detach().float().to(DEVICE)
        r = exp29.evaluate_space(Xtr, ytr, Xte, yte, anch, seen, holdouts)
        aucs = []
        for s in range(3):
            torch.manual_seed(1000 + s)
            a, _, _ = exp29.linear_probe_novelty(Xtr, ytr, Xte, yte,
                                                 holdouts)
            aucs.append(a)
        d_te = torch.cdist(torch.as_tensor(Xte, dtype=torch.float32,
                                           device=DEVICE), anch) \
            .min(1).values.cpu().numpy()
        bgm = np.isin(yte, seen)
        sgm = np.isin(yte, list(holdouts))
        pe = exp30.power_at_alpha(d_te[bgm], d_te[sgm], args.alpha)

        # semantic metrics (where a superclass partition exists)
        sem = {}
        try:
            names, sup = exp76.class_names_and_sup(ds)
            if sup is not None:
                Cn, D, ok = exp76.centroid_dist(
                    np.asarray(Xtr, np.float64), ytr, n_cls)
                keep = np.where(ok)[0]
                sem = exp76.sup_metrics(D[np.ix_(keep, keep)], Cn[keep],
                                        [sup[i] for i in keep])
        except Exception as e:
            print(f"  (semantics skipped: {e})")

        # dense SparKer reach
        skey = f"sparker_{key}"
        if skey in done:
            powers = list(done[skey])
        else:
            R = torch.as_tensor(np.asarray(Xtr, np.float32)[m][:20000],
                                device=DEVICE)
            bg = torch.as_tensor(np.asarray(Xte, np.float32)[bgm],
                                 device=DEVICE)
            sg = torch.as_tensor(np.asarray(Xte, np.float32)[sgm],
                                 device=DEVICE)
            powers, _ = exp31.run_test_battery(bg, sg, R, FRACS, n_d,
                                               n_null, n_toys, args.alpha,
                                               args.seed, dict(sparker_kw),
                                               tag=key)
            done[skey] = np.array(powers)
            done[f"fractions_{key}"] = np.array(FRACS)
            np.savez(out_path, **done)
        f95v, flag = exp99.f95(FRACS, powers)
        rows[cell] = dict(probe=float(np.mean(aucs)), eucl=r["eucl"],
                          mahaT=r["maha_tied"], lid=r["lid"], perevt=pe,
                          f95=f95v, flag=flag,
                          agree1=sem.get("agree1", float("nan")),
                          purity=sem.get("dendro_purity", float("nan")))
        print(f"  [{cell}] probe={np.mean(aucs):.4f} eucl={r['eucl']:.4f}"
              f" mahaT={r['maha_tied']:.4f} perevt={pe:.3f} "
              f"f95={exp99.fmt(f95v, flag, max(FRACS))} "
              f"agree1={rows[cell]['agree1']:.3f} "
              f"sem-purity={rows[cell]['purity']:.3f}", flush=True)

    print("\n===== EXP102 SUMMARY (residual child standalone) =====")
    print(f"  {'cell':<18}{'probe':>8}{'mahaT':>8}{'perevt':>8}{'f95':>9}"
          f"{'agree1':>8}{'sem-pur':>9}")
    for cell, b in rows.items():
        print(f"  {cell:<18}{b['probe']:>8.4f}{b['mahaT']:>8.3f}"
              f"{b['perevt']:>8.3f}"
              f"{exp99.fmt(b['f95'], b['flag'], 0.1):>9}"
              f"{b['agree1']:>8.3f}{b['purity']:>9.3f}")
    np.savez(out_path, **done,
             summary=np.array([repr(rows)], dtype=object))
    print("EXP102 DONE.")


if __name__ == "__main__":
    main()
