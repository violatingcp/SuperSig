"""
Experiment 111 (IMPROVEMENT_TESTS.md #111): child-only deployment study.

Exp 102: the res-nplm child ALONE beats its concat's reach on the
fine-grained VISReg/DINO cells (cars:visreg f95 0.049, aircraft:visreg
0.063, cars:dino 0.063 -- the three qualifying cells) at probe cost
0.005-0.034 and stays semantically legible.  The deployable configuration
nobody has evaluated end to end: DETECT with the child, keep the parent
only for EXPLANATION.

Pipeline per qualifying cell:
  detect   child-only min-centroid distance; flag test events beyond the
           0.95 background quantile; report per-event power, flagged-set
           recall/precision, and the child/concat f95 from the exp-102/100
           archives
  explain  each flagged TRUE-NOVEL event gets a top-5 nearest-seen-class-
           centroid explanation, scored in three spaces: parent-only (the
           deployment), child-only, concat (the baseline that pays double
           width); agree@1/agree@5 = does the top-1/any top-5 centroid
           share the true class's superclass
  cost     embedding width of what must be computed per event

Prediction: equal or better detection at half the width, explanation
within 0.05 agree@1 of the concat.  Falsifier: explanations degrade
materially without the parent in the scored space.

    python experiments/111_child_deployment.py
    python experiments/111_child_deployment.py --cells cars:visreg
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import importlib
import numpy as np

exp30 = importlib.import_module("30_power_curves")
exp44 = importlib.import_module("44_transfer_32d")
exp72 = importlib.import_module("72_residual_discovery")
exp76 = importlib.import_module("76_interpretability")
exp77 = importlib.import_module("77_space_similarity")
exp99 = importlib.import_module("99_discovery_reach")

CELLS = ["cars:visreg", "aircraft:visreg", "cars:dino"]


def explain(Xtr, ytr, Xq, seen, sup, y_true, names, k=5, show=0):
    """Top-k nearest seen-class centroids; agree@1/agree@k vs superclass.

    Superclass agreement is only DEFINED for events whose true superclass
    is represented among the seen classes (holdouts can remove a whole
    make/manufacturer); returns (a1, ak, top, n_scorable).
    """
    cents = np.stack([Xtr[ytr == c].mean(0) for c in seen])
    D = np.sqrt(exp77.sqdist(np.asarray(Xq, np.float64), cents))
    order = np.argsort(D, axis=1)[:, :k]
    top = np.asarray(seen)[order]                     # (n, k) class ids
    seen_sups = {sup[c] for c in seen}
    ok = np.array([sup[y_true[i]] in seen_sups for i in range(len(Xq))])
    a1 = np.array([sup[top[i, 0]] == sup[y_true[i]]
                   for i in range(len(Xq))], dtype=float)
    ak = np.array([any(sup[c] == sup[y_true[i]] for c in top[i])
                   for i in range(len(Xq))], dtype=float)
    a1m = float(a1[ok].mean()) if ok.any() else float("nan")
    akm = float(ak[ok].mean()) if ok.any() else float("nan")
    if show:
        for i in range(min(show, len(Xq))):
            print(f"      true {names[y_true[i]]:<28} -> " +
                  ", ".join(names[c] for c in top[i][:3]))
    return a1m, akm, top, int(ok.sum())


def overlap(topA, topB):
    """(top-1 match rate, mean top-5 Jaccard) between two explanation sets."""
    m1 = float((topA[:, 0] == topB[:, 0]).mean())
    j = [len(set(a) & set(b)) / len(set(a) | set(b))
         for a, b in zip(topA, topB)]
    return m1, float(np.mean(j))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=",".join(CELLS))
    ap.add_argument("--emb-dim", type=int, default=100)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--q", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show", type=int, default=5)
    args = ap.parse_args()

    # archived reach numbers for the detection column
    f95s = {}
    for tag, path in (("child", "logs/exp102/results.npz"),
                      ("concat", "logs/exp100/results.npz")):
        try:
            d = np.load(path, allow_pickle=True)
            for k in d.files:
                if k.startswith("sparker_"):
                    key = k[len("sparker_"):]
                    fr = list(d[f"fractions_{key}"])
                    v, fl = exp99.f95(fr, list(d[k]))
                    f95s[f"{tag}:{key}"] = exp99.fmt(v, fl, max(fr))
        except Exception as e:
            print(f"  ({tag} archive unavailable: {e})")

    rows = {}
    for cell in args.cells.split(","):
        ds, base = cell.split(":")
        parent, obj, kind = exp72.WINNERS[(ds, base)]
        key = cell.replace(":", "_")
        n_cls = 47 if ds == "dtd" else exp44.N_CLASSES[ds]
        holdouts = set(range(n_cls - 10, n_cls))
        seen = [c for c in range(n_cls) if c not in holdouts]
        p_tr = exp77.head_emb(ds, base, parent, args.emb_dim, "train")
        p_te = exp77.head_emb(ds, base, parent, args.emb_dim, "test")
        c_tr = exp77.head_emb(ds, base, f"{parent}_{obj}", args.emb_dim,
                              "train")
        c_te = exp77.head_emb(ds, base, f"{parent}_{obj}", args.emb_dim,
                              "test")
        if not (p_tr and p_te and c_tr and c_te):
            print(f"  !! [{cell}] missing banks, skipping")
            continue
        (Xp, yp), (Xpt, ypt) = p_tr, p_te
        (Xc, yc), (Xct, yct) = c_tr, c_te
        names, sup = exp76.class_names_and_sup(ds)
        if sup is None:
            print(f"  !! [{cell}] no superclass partition, skipping")
            continue
        print(f"\n######## [{cell}] child-only deployment "
              f"({parent}->{obj}) ########", flush=True)

        # ---- detect in the CHILD space only
        m = np.isin(yc, seen)
        cents_c = np.stack([Xc[yc == c].mean(0) for c in seen])
        d_tr = np.sqrt(exp77.sqdist(np.asarray(Xc, np.float64)[m],
                                    cents_c)).min(1)
        d_te = np.sqrt(exp77.sqdist(np.asarray(Xct, np.float64),
                                    cents_c)).min(1)
        bgm = np.isin(yct, seen)
        sgm = np.isin(yct, list(holdouts))
        pe = exp30.power_at_alpha(d_te[bgm], d_te[sgm], args.alpha)
        tau = np.quantile(d_te[bgm], args.q)
        flag = d_te > tau
        recall = float(flag[sgm].mean())
        precision = float(sgm[flag].mean()) if flag.any() else float("nan")

        # concat detection baseline at the same working point
        Xcat = np.concatenate([Xp, Xc], 1)
        Xcatt = np.concatenate([Xpt, Xct], 1)
        cents_cat = np.stack([Xcat[yc == c].mean(0) for c in seen])
        d_te_cat = np.sqrt(exp77.sqdist(np.asarray(Xcatt, np.float64),
                                        cents_cat)).min(1)
        pe_cat = exp30.power_at_alpha(d_te_cat[bgm], d_te_cat[sgm],
                                      args.alpha)
        tau_c = np.quantile(d_te_cat[bgm], args.q)
        flag_c = d_te_cat > tau_c
        rec_c = float(flag_c[sgm].mean())
        prc_c = float(sgm[flag_c].mean()) if flag_c.any() else float("nan")

        # ---- explain the flagged TRUE-NOVEL events
        fn = flag & sgm
        y_fn = yct[fn]
        print(f"  child flags {int(flag.sum())} events: recall={recall:.3f}"
              f" precision={precision:.3f} (concat: {rec_c:.3f}/"
              f"{prc_c:.3f});  {int(fn.sum())} true-novel to explain",
              flush=True)
        a1p, a5p, top_p, n_sc = explain(Xp, yp, Xpt[fn], seen, sup, y_fn,
                                        names, show=args.show)
        a1c, a5c, top_c, _ = explain(Xc, yc, Xct[fn], seen, sup, y_fn,
                                     names)
        a1cat, a5cat, top_cat, _ = explain(Xcat, yc, Xcatt[fn], seen, sup,
                                           y_fn, names)
        m1_p, j5_p = overlap(top_p, top_cat)     # parent vs concat baseline
        m1_c, j5_c = overlap(top_c, top_cat)     # child vs concat

        rows[cell] = dict(perevt=pe, perevt_cat=pe_cat, recall=recall,
                          precision=precision, recall_cat=rec_c,
                          precision_cat=prc_c, n_flagged=int(flag.sum()),
                          n_scorable=n_sc,
                          a1_parent=a1p, a5_parent=a5p, a1_child=a1c,
                          a5_child=a5c, a1_cat=a1cat, a5_cat=a5cat,
                          m1_parent=m1_p, j5_parent=j5_p,
                          m1_child=m1_c, j5_child=j5_c,
                          width_deploy=Xc.shape[1],
                          width_cat=Xcat.shape[1],
                          f95_child=f95s.get(f"child:{key}", "--"),
                          f95_concat=f95s.get(f"concat:{key}", "--"))
        r = rows[cell]
        print(f"  detect: perevt child={pe:.3f} concat={pe_cat:.3f}  "
              f"f95 child={r['f95_child']} concat={r['f95_concat']}")
        print(f"  explain agree@1/@5 (n_scorable={n_sc}): "
              f"parent={a1p:.3f}/{a5p:.3f}  child={a1c:.3f}/{a5c:.3f}  "
              f"concat={a1cat:.3f}/{a5cat:.3f}")
        print(f"  explain overlap-with-concat (top1 / top5-Jaccard): "
              f"parent={m1_p:.3f}/{j5_p:.3f}  child={m1_c:.3f}/{j5_c:.3f}",
              flush=True)

    print("\n===== EXP111 SUMMARY (detect with child, explain with parent) "
          "=====")
    print(f"  {'cell':<16}{'pe ch/cat':>12}{'a@1 par/cat':>13}"
          f"{'ovl1 par/ch':>13}{'j5 par/ch':>12}{'width':>9}")
    for cell, r in rows.items():
        print(f"  {cell:<16}{r['perevt']:>6.3f}/{r['perevt_cat']:.3f}"
              f"{r['a1_parent']:>7.3f}/{r['a1_cat']:.3f}"
              f"{r['m1_parent']:>7.3f}/{r['m1_child']:.3f}"
              f"{r['j5_parent']:>6.3f}/{r['j5_child']:.3f}"
              f"{r['width_deploy']:>5}v{r['width_cat']}")

    os.makedirs(os.path.join("logs", "exp111"), exist_ok=True)
    np.savez(os.path.join("logs", "exp111", "results.npz"),
             summary=np.array([repr(rows)], dtype=object))
    print("EXP111 DONE.")


if __name__ == "__main__":
    main()
