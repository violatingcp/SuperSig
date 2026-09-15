"""
Experiment 128: where should the discovery pool cut actually sit?

WHY.  `tau_quantile=0.95` has been the pool cut since exp 23 and was never
derived.  Exp 112's constants audit lists it as "swept exp 89 (distance only)";
exp 109 tried only {0.95, 0.99} for the density scorer.  Three values, on a grid
that skips the entire 0.95-0.99 interval where the h10 optimum evidently lives.

THE ALGEBRA (this is the point of the script).  Write

    b    = base rate of novel points in the train-eval bank
    q    = pool size / N                         (the cut, = 1 - tau_quantile)
    e_s  = fraction of NOVEL points retained     (signal efficiency)
    e_b  = fraction of BACKGROUND points retained

then

    q      = e_s*b + e_b*(1-b)
    purity = e_s*b / q
    E      = purity / b = e_s / q                (enrichment == lift at q)

and because e_s <= 1,

    purity <= min(1, b/q)                        (THE CEILING)

At q=0.05, b=0.01 the ceiling is 0.20 -- a 20x enrichment is available -- while
the measured density-ratio purity at h1 is 0.030, i.e. E=3.0 and e_s=0.15.  So
the h1 failure is NOT a rate floor: it is a SIGNAL-EFFICIENCY floor, and the
cut is a free parameter sitting far from its own ceiling.  (This corrects the
"base rate >= 4%" rule of thumb, which silently assumed E~3.5 is a property of
the problem rather than of one arbitrary cut.)

THE SECOND CONSTRAINT.  Tightening q raises purity only while enough novel
points survive to be a cluster at all:

    n_novel = purity * q * N   >=   n_min        (BIC detectability)

At h10/q0.99 that is ~179 points (fine).  At h1/q0.99 it is ~15 -- BIC will
never call that a component.  That is why tightening helped at h10
(0.232 -> 0.358) and did nothing at h1 (0.030 -> 0.029).  So the objective is
two-sided:

    maximize   purity(q)      subject to   n_novel(q) >= n_min

WHAT THIS SCRIPT DOES.  Given embeddings + labels for a cell, it computes each
pool scorer (dist / lid / np), sweeps q densely, and reports the full operating
curve against the ceiling: purity, n_novel, e_s, e_b, enrichment, headroom
(purity / ceiling), whether the purity gate is cleared, and -- optionally, the
expensive part -- whether BIC on the pooled points actually recovers a
majority-novel cluster.

It also reports each scorer's CUT-FREE quality (novel-vs-seen AUC of the score
itself).  Purity at one cut conflates scorer quality with cut choice; the AUC
separates them, and the table above says the deficit is mostly scorer quality.

Evaluation-only: trains nothing, needs no GPU beyond the NP critic fit.

    python experiments/128_pool_cut_optimization.py --selftest
    python experiments/128_pool_cut_optimization.py --embs logs/exp113/embs/cifar100_on.npz
    python experiments/128_pool_cut_optimization.py --embs ... --holdouts 90-99 --bic
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from supersig.holdouts import n_holdout, run_tag
import argparse
import glob
import json

import numpy as np
import torch

GATE = 0.15          # the purity gate the campaign reports against
N_MIN = 100          # novel points needed for BIC to plausibly find a component


# --------------------------------------------------------------- the algebra

def operating_point(scores, is_novel, q):
    """Metrics for the cut that keeps the top-`q` fraction of ALL points.

    Note the campaign's tau is a quantile over SEEN-labelled points, which is
    the open-world-legal version; `q` here is the resulting pool fraction.  Both
    are reported so the two conventions can be compared.
    """
    n = len(scores)
    k = max(1, int(round(q * n)))
    idx = np.argsort(-scores)[:k]
    sel = np.zeros(n, dtype=bool)
    sel[idx] = True

    b = float(is_novel.mean())
    n_nov = int(is_novel[sel].sum())
    purity = n_nov / k
    e_s = n_nov / max(int(is_novel.sum()), 1)
    e_b = int((~is_novel)[sel].sum()) / max(int((~is_novel).sum()), 1)
    ceiling = min(1.0, b / q) if q > 0 else 1.0
    return dict(q=float(q), pool=k, purity=float(purity), n_novel=n_nov,
                eps_s=float(e_s), eps_b=float(e_b),
                enrichment=float(purity / b) if b > 0 else float("nan"),
                ceiling=float(ceiling),
                headroom=float(purity / ceiling) if ceiling > 0 else float("nan"),
                gate=bool(purity >= GATE), detectable=bool(n_nov >= N_MIN),
                usable=bool(purity >= GATE and n_nov >= N_MIN))


def tau_quantile_for(scores, is_seen_lab, tau_q):
    """The campaign's actual rule: threshold at the tau_q quantile of the score
    over SEEN-LABELLED points.  Returns the resulting pool mask."""
    tau = np.quantile(scores[is_seen_lab], tau_q)
    return scores > tau


def curve(scores, is_novel, qs):
    return [operating_point(scores, is_novel, q) for q in qs]


def auc(scores, is_novel):
    """Cut-free scorer quality: novel-vs-seen AUC of the raw score."""
    from sklearn.metrics import roc_auc_score
    if is_novel.sum() == 0 or is_novel.sum() == len(is_novel):
        return float("nan")
    return float(roc_auc_score(is_novel.astype(int), scores))


def best_q(rows, n_min=N_MIN):
    """Highest-purity cut that still leaves a detectable cluster."""
    ok = [r for r in rows if r["n_novel"] >= n_min]
    return max(ok, key=lambda r: r["purity"]) if ok else None


# ------------------------------------------------------------- BIC recovery

def bic_recovers(z, sel, is_novel, kmax, seed=0):
    """Does BIC-selected k-means on the pool produce a majority-novel cluster?

    This is the question the purity gate is a proxy for.  Returns the purity of
    the most-novel cluster and its size.
    """
    from supersig.discovery import bic_select
    Z = torch.as_tensor(z[sel], dtype=torch.float32)
    if len(Z) < 4:
        return dict(khat=0, best_cluster_purity=0.0, best_cluster_n=0)
    khat, centers, _ = bic_select(Z, kmax=kmax, seed=seed)
    a = torch.cdist(Z, centers).argmin(1).numpy()
    nov = is_novel[sel]
    best_p, best_n = 0.0, 0
    for j in range(centers.shape[0]):
        m = a == j
        if m.sum() == 0:
            continue
        p = float(nov[m].mean())
        if p > best_p:
            best_p, best_n = p, int(m.sum())
    return dict(khat=int(khat), best_cluster_purity=best_p,
                best_cluster_n=best_n)


# ------------------------------------------------------------------ scorers

def all_scores(z, is_seen_lab, seed=0, which=("dist", "lid", "np")):
    """The three pool scorers, on a shared embedding bank."""
    from supersig.discovery import lid_pool_scores, np_pool_scores
    zt = torch.as_tensor(z, dtype=torch.float32)
    out = {}
    if "dist" in which:
        # distance to the nearest SEEN-class centroid (the anchors a round-1
        # loop would use before anything has been discovered)
        lab = np.where(is_seen_lab)[0]
        cent = torch.as_tensor(
            np.stack([z[lab].mean(0)]) if len(lab) else np.zeros((1, z.shape[1])),
            dtype=torch.float32)
        out["dist"] = torch.cdist(zt, cent).min(1).values.numpy()
    if "lid" in which:
        out["lid"] = lid_pool_scores(zt, is_seen_lab, seed=seed).cpu().numpy()
    if "np" in which:
        out["np"] = np_pool_scores(zt, is_seen_lab, seed=seed).cpu().numpy()
    return out


def seen_centroid_scores(z, tr_lab, seen):
    """Distance to the nearest per-class seen centroid -- the real round-1
    `dist` scorer (one centroid per seen class, not a single global mean)."""
    cents = np.stack([z[tr_lab == c].mean(0) for c in seen
                      if (tr_lab == c).sum() > 0])
    zt = torch.as_tensor(z, dtype=torch.float32)
    ct = torch.as_tensor(cents, dtype=torch.float32)
    return torch.cdist(zt, ct).min(1).values.numpy()


# ----------------------------------------------------------------- selftest

def _selftest():
    """The algebra must hold identically, and the curve must behave correctly
    at the two extremes: a perfect scorer saturates the ceiling, a random
    scorer sits at enrichment 1."""
    rng = np.random.default_rng(0)
    N, b = 20000, 0.01
    n_nov = int(N * b)
    is_novel = np.zeros(N, dtype=bool)
    is_novel[rng.choice(N, n_nov, replace=False)] = True

    qs = np.array([0.002, 0.005, 0.01, 0.02, 0.05, 0.1])

    # 1. PERFECT scorer: novel always ranked above background.
    s_perfect = is_novel.astype(float) + 1e-6 * rng.random(N)
    for r in curve(s_perfect, is_novel, qs):
        assert abs(r["purity"] - r["ceiling"]) < 1e-6, r
        assert r["headroom"] > 0.999, r
    print("  perfect scorer saturates the ceiling at every q          OK")

    # 2. RANDOM scorer: enrichment ~ 1 IN EXPECTATION.  A single draw is a
    #    small-count experiment (at b=0.01, q=0.02 the pool holds ~4 novel
    #    points, so a 0 is ordinary), hence average over many scorers.
    hi_novel = np.zeros(N, dtype=bool)
    hi_novel[rng.choice(N, int(N * 0.05), replace=False)] = True
    es = []
    for t in range(20):
        r = operating_point(rng.random(N), hi_novel, 0.05)
        es.append(r["enrichment"])
    m = float(np.mean(es))
    assert 0.85 < m < 1.15, (m, es)
    assert abs(np.mean([operating_point(rng.random(N), hi_novel, 0.10)["purity"]
                        for _ in range(20)]) - 0.05) < 0.01
    print(f"  random scorer sits at enrichment ~1 (mean {m:.3f}, n=20)   OK")

    # 3. The identity E = e_s / q must hold exactly, for any scorer.
    s_mid = is_novel * rng.normal(1.2, 1.0, N) + rng.normal(0, 1, N)
    for r in curve(s_mid, is_novel, qs):
        lhs = r["enrichment"]
        rhs = r["eps_s"] / r["q"]
        assert abs(lhs - rhs) < 1e-6 * max(1, lhs), (lhs, rhs, r)
    print("  identity  E == eps_s / q  holds exactly                  OK")

    # 4. The ceiling is never exceeded, by any scorer at any cut.
    for s in (s_perfect, rng.random(N), s_mid):
        for r in curve(s, is_novel, qs):
            assert r["purity"] <= r["ceiling"] + 1e-9, r
    print("  purity <= min(1, b/q) always                             OK")

    # 5. The published arithmetic reproduces: q=0.05, b=0.01 -> ceiling 0.20.
    r = operating_point(s_perfect, is_novel, 0.05)
    assert abs(r["ceiling"] - 0.20) < 1e-9, r
    print(f"  q=0.05, b=0.01 -> ceiling {r['ceiling']:.2f} (the 20x claim)   OK")

    # 6. n_novel bookkeeping: purity * pool == n_novel.
    for r in curve(s_mid, is_novel, qs):
        assert abs(r["purity"] * r["pool"] - r["n_novel"]) < 1e-6, r
    print("  n_novel == purity * pool                                 OK")

    # 7. best_q respects the detectability floor.
    rows = curve(s_perfect, is_novel, np.array([0.001, 0.005, 0.01, 0.05]))
    bq = best_q(rows, n_min=100)
    assert bq is not None and bq["n_novel"] >= 100, bq
    # with a perfect scorer and 200 novel points, the tightest usable cut is
    # the one whose pool is ~n_min
    assert bq["purity"] >= 0.99, bq
    print(f"  best_q honours n_min (picked q={bq['q']}, "
          f"n_novel={bq['n_novel']})       OK")

    # 8. A scorer that is ANTI-selective (the C100 distance case) reports
    #    enrichment < 1 rather than silently looking fine.
    s_anti = -s_mid
    rr = operating_point(s_anti, is_novel, 0.05)
    assert rr["enrichment"] < 1.0, rr
    print(f"  anti-selective scorer flagged (E={rr['enrichment']:.2f} < 1)     OK")

    print("\nselftest OK")


# --------------------------------------------------------------------- main

def _parse_holdouts(spec):
    if not spec:
        return None
    out = set()
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out |= set(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--embs", default="",
                    help="npz with tr/tr_lab (e.g. from 113 --save-embs)")
    ap.add_argument("--glob", default="",
                    help="glob over several embedding npz files")
    ap.add_argument("--holdouts", default="",
                    help="e.g. '99' or '90-99'; default = last n_holdout(ds)")
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--scorers", default="dist,lid,np")
    ap.add_argument("--bic", action="store_true",
                    help="also run BIC k-means per cut (slower)")
    ap.add_argument("--n-min", type=int, default=N_MIN)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="logs/exp128")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    files = ([args.embs] if args.embs else []) + sorted(glob.glob(args.glob))
    if not files:
        ap.error("need --embs or --glob (or --selftest)")

    # dense grid: the campaign only ever tried 0.95/0.99/0.995
    qs = np.unique(np.concatenate([
        np.geomspace(0.001, 0.02, 12), np.linspace(0.02, 0.20, 19)]))

    os.makedirs(args.out, exist_ok=True)
    results = {}
    for fn in files:
        d = np.load(fn, allow_pickle=True)
        z = np.asarray(d["tr"], dtype=np.float64)
        lab = np.asarray(d["tr_lab"])
        n_cls = int(lab.max()) + 1
        hold = _parse_holdouts(args.holdouts)
        if hold is None:
            k = n_holdout(args.dataset)
            hold = set(range(n_cls - k, n_cls))
        is_novel = np.isin(lab, list(hold))
        is_seen_lab = ~is_novel
        seen = [c for c in range(n_cls) if c not in hold]
        b = float(is_novel.mean())
        key = os.path.basename(fn).replace(".npz", "")
        print(f"\n######## {key}: N={len(z)}, base rate b={b:.4f} "
              f"({int(is_novel.sum())} novel), holdouts={sorted(hold)} ########")

        which = tuple(args.scorers.split(","))
        sc = all_scores(z, is_seen_lab, seed=args.seed, which=which)
        if "dist" in which:                      # per-class centroids, not global
            sc["dist"] = seen_centroid_scores(z, lab, seen)

        for name, s in sc.items():
            a = auc(s, is_novel)
            rows = curve(s, is_novel, qs)
            for r in rows:
                r["scorer"] = name
                if args.bic:
                    k = max(1, int(round(r["q"] * len(s))))
                    sel = np.zeros(len(s), dtype=bool)
                    sel[np.argsort(-s)[:k]] = True
                    r.update(bic_recovers(z, sel, is_novel,
                                          kmax=max(4, len(hold) + 2),
                                          seed=args.seed))
            # the campaign's actual rule, for reference
            camp = {}
            for tq in (0.95, 0.99, 0.995):
                m = tau_quantile_for(s, is_seen_lab, tq)
                camp[tq] = dict(pool=int(m.sum()),
                                purity=float(is_novel[m].mean()) if m.any() else 0.0,
                                n_novel=int(is_novel[m].sum()))
            bq = best_q(rows, args.n_min)

            print(f"\n  --- scorer '{name}'  (cut-free novel-vs-seen AUC = {a:.4f}) ---")
            print(f"  {'q':>7s}{'pool':>8s}{'purity':>9s}{'ceil':>8s}"
                  f"{'head':>7s}{'E':>7s}{'eps_s':>8s}{'n_nov':>8s}  flags")
            for r in rows:
                flags = ("GATE " if r["gate"] else "     ") + \
                        ("DET" if r["detectable"] else "   ")
                print(f"  {r['q']:>7.4f}{r['pool']:>8d}{r['purity']:>9.4f}"
                      f"{r['ceiling']:>8.3f}{r['headroom']:>7.2f}"
                      f"{r['enrichment']:>7.2f}{r['eps_s']:>8.3f}"
                      f"{r['n_novel']:>8d}  {flags}")
            print(f"  campaign cuts: " + "  ".join(
                f"tau_q={t}: pool={c['pool']} purity={c['purity']:.4f} "
                f"n_nov={c['n_novel']}" for t, c in camp.items()))
            if bq:
                print(f"  BEST usable cut: q={bq['q']:.4f} purity={bq['purity']:.4f} "
                      f"n_novel={bq['n_novel']} (headroom {bq['headroom']:.2f})")
            else:
                print(f"  NO usable cut: no q leaves >= {args.n_min} novel points")
            results[f"{key}|{name}"] = dict(auc=a, base_rate=b, rows=rows,
                                            campaign=camp, best=bq)

    with open(os.path.join(args.out, f"cutscan{run_tag()}.json"), "w") as fh:
        json.dump(results, fh, indent=1, default=float)
    print(f"\nwrote {args.out}/cutscan{run_tag()}.json")


if __name__ == "__main__":
    main()
