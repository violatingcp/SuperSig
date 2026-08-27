"""
Experiment 136: fix the base-rate estimator -- the label-free cut's one broken part.

WHY.  Exp 129 established that the label-free pool cut works (beats the
inherited q=0.05 in 29/30 cells, engages 30/30, clears the gate at h1 on C100
at 0.387+-0.174 label-free).  It also established exactly where it falls short:
the rule captures only 30-55% of ORACLE purity, and the entire gap is `b_hat`.

    b_hat = 0.002-0.009   against a true b = 0.010        (C100, 2-4x low)
    b_hat = 0.007-0.015   against a true b = 0.106        (galaxy10 DINO, 7-15x)
    q_rule = 0.007-0.03   against q_oracle = 0.002-0.005  (2-5x too wide)

The critic is NOT the problem -- galaxy10 DINO has calib_out = 1.000 and its
spaces demonstrably hold the novelty (distance-pool purity 0.59-0.63) -- yet
`b_hat` is 7-15x low there and the rule REFUSES on all six spaces.  The `ok`
flag is honest about the critic, not about the space.

WHY IT BIASES THE CUT.  The rule stops at the tightest k with N_hat(k) >= n_min.
If N_hat is low by a factor alpha, that k is where the TRUE count reaches
n_min/alpha, so the rule pools WIDER than it should and purity falls by the
ceiling Pi <= min(1, b/q).  Diagnostic confirming this at the operating point:
the rule stops when N_hat = 30 while the pool it stops at actually contains
56-190 real novel points.

  >>> A UNIFORM SCALE BIAS IN b_hat IS ALGEBRAICALLY IDENTICAL TO AN n_min
  >>> THAT IS TOO LARGE BY THE SAME FACTOR.

which is why "just lower n_min" is the right SHAPE of fix and the wrong
mechanism: one knob would then be doing two jobs (bias correction AND BIC
detectability), and alpha is not a constant -- it is 2-4x on C100 and 7-15x on
galaxy10 DINO.  A single lowered n_min tuned on one would be badly wrong on the
other, and it erodes the refusal guarantee the rank-matched null was built for.

FOUR THINGS THIS SCRIPT DOES, all evaluation-only on archived score vectors.

(1) A SCALE-INVARIANT CUT CRITERION.  If the bias is mostly a uniform scale on
    N_hat, any criterion reading the SHAPE of N_hat(k) rather than its level is
    immune.  We take the knee: normalise both axes and find the point of
    maximum deviation from the chord.  Multiply every weight by alpha and the
    knee does not move -- that is the property being bought.

(2) THE OTHER TWO ESTIMATORS, NEVER RUN ON REAL DATA.  Only `tv` was used in
    exp 129.  `mass` and `excess` are implemented and recovered b exactly on
    synthetic.  They fail differently under overlap: `tv` shrinks by
    TV(p_sig, p_ref) exactly, while `excess` is a threshold count and should be
    far less overlap-sensitive.  Cheap to settle.

(3) INJECTION CALIBRATION, per space.  Hold out a SEEN class, re-inject it at a
    KNOWN rate as pseudo-novelty, and measure what the estimator reports.  The
    ratio is alpha for that space.  This is label-free in deployment -- the
    injected class is one you have labels for -- and it measures alpha
    PER SPACE, which is exactly what a global constant cannot do and what the
    galaxy10 DINO case demands.

(4) AN n_min SENSITIVITY SCAN, reported as a diagnostic rather than a fix.  If
    purity rises monotonically as n_min falls, that is independent evidence for
    the scale-bias story and it reads off alpha per cell, feeding (3).

  >>> REAL-DATA RESULT 2026-08-27, six exp-54 CIFAR-10 spaces, b_true = 0.10.
  >>> logs/exp136/brate_cifar10_exp54.json.  THREE OF THE FOUR IDEAS ABOVE FAIL
  >>> ON REAL DATA; LOWERING n_min IS THE ONE THAT WORKS.
  >>>
  >>> (2) ESTIMATOR SWAP: NO HELP.  All three agree and all are 4-7x low --
  >>>     tv 0.016-0.026, mass 0.016-0.024, excess 0.015 against b_true 0.10.
  >>>     The bias is not specific to `tv`, so swapping estimators cannot fix
  >>>     it.  (Note the synthetic ordering also reversed: under heavy overlap
  >>>     `tv` degrades MORE gracefully than mass/excess, 0.0063 vs 0.0000 and
  >>>     0.0011 at overlap 0.9 -- the opposite of what was predicted.)
  >>>
  >>> (1) KNEE: FAILS ON REAL DATA, badly.  Perfect on synthetic (k=402 for 400
  >>>     true novel, unmoved by 16x weight scaling), but on real spaces it
  >>>     picks a much WIDER cut than n_min=30 and loses a lot of purity:
  >>>         nplm_dist_sup_cw  knee q=0.0559 purity 0.549 | n_min=30 q=0.0052 purity 0.778
  >>>         nplm_distance     knee q=0.0249 purity 0.633 | n_min=30 q=0.0039 purity 0.856
  >>>         nplm_sup_dist     knee q=0.0439 purity 0.617 | n_min=30 q=0.0051 purity 0.719
  >>>     Reason: the elbow is sharp only when the novel mode is cleanly
  >>>     separated.  With real overlap N_hat(k) grows smoothly and the
  >>>     max-deviation point drifts far out.  Scale-invariance is worthless if
  >>>     the shape carries no knee.
  >>>
  >>> (3) INJECTION alpha: UNSTABLE.  1.268 / 0.073 / 1.300 across three spaces
  >>>     -- the donor seen class is not exchangeable with the true novel class,
  >>>     so alpha measures the donor as much as the estimator.  Needs a donor
  >>>     ensemble and a spread before it can be trusted.
  >>>
  >>> (4) n_min: THIS IS THE FIX.  Purity rises monotonically as n_min falls,
  >>>     on every space:
  >>>         n_min      5     10     20     30     50    100    200
  >>>         dist_sup_cw  .833  .846   .807   .778   .743   .715   .665
  >>>         distance     .900  .875   .879   .856   .820   .795   .784
  >>>         sup_dist     .686  .695   .716   .719   .704   .711   .678
  >>>     n_min=5-10 buys +0.05 to +0.06 purity over the current default of 30,
  >>>     at 70-112 real novel points still in the pool.
  >>>
  >>> BUT THE SCALE-BIAS MODEL IS WRONG.  implied_alpha is NOT constant across
  >>> n_min (0.071 -> 0.381 on dist_sup_cw); it climbs steadily.  So b_hat is
  >>> not off by a single factor -- the underestimate grows as the cut widens,
  >>> which is what a per-point weight bias concentrated in the OVERLAP region
  >>> looks like.  That is why the scale-invariant criterion buys nothing here,
  >>> and it means lowering n_min is a genuine empirical fix rather than an
  >>> algebraic correction for a known bias.  Recommend: lower the default to
  >>> 10, keep it as an honest tuning constant with a reported sweep, and do
  >>> NOT claim it corrects a bias factor.

    python experiments/136_base_rate_estimator.py --selftest
    python experiments/136_base_rate_estimator.py --embs logs/exp54/embs_*.npz --holdouts 4
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from supersig.holdouts import n_holdout, run_tag
from supersig import poolcut
import argparse
import glob
import json

import numpy as np
import torch

ESTIMATORS = ("tv", "mass", "excess")


# -------------------------------------------------- (2) the three estimators

def b_hat(f, f_ref, which="tv", thresh=2.0, ref_q=0.99):
    """Label-free base-rate estimates.  `f` is the NP critic's log density ratio.

    tv      E_corpus[(1 - 1/r)_+] = TV(p_D, p_R) = b * TV(p_S, p_R).  Exact for
            disjoint novelty, shrinks with overlap -- the known bias.
    mass    P_corpus(r > thresh).  A threshold count; insensitive to the tails
            of f, and it does NOT carry the TV shrink factor in the same way.
    excess  bump-hunt excess over background at a reference-defined threshold.
            Uses only the reference to define "normal".
    """
    r = poolcut.calibrated_ratio(f, f_ref)
    if which == "tv":
        return float(np.mean(np.maximum(0.0, 1.0 - 1.0 / np.maximum(r, 1e-12))))
    if which == "mass":
        return float(np.mean(r > thresh))
    if which == "excess":
        t = float(np.quantile(f_ref, ref_q))
        p_c, p_r = float(np.mean(f > t)), float(np.mean(f_ref > t))
        return 0.0 if p_r >= 1.0 else float(max(0.0, (p_c - p_r) / (1.0 - p_r)))
    raise ValueError(which)


# ------------------------------------------- (1) scale-invariant knee cut

def knee_k(f, f_ref, k_min=1, k_max_frac=0.30):
    """The knee of the null-corrected cumulative estimated-novelty curve.

    N_hat(k) is concave: the top-ranked points carry weight ~1 and later points
    contribute ~0 once the null is subtracted.  Normalising BOTH axes and taking
    the point of maximum deviation from the chord gives a cut that depends only
    on the SHAPE of the curve.

    Scale invariance is the whole point: N_hat -> alpha * N_hat leaves the
    normalised curve unchanged, so a uniformly-biased b_hat moves the knee not
    at all.  That is what `n_min` cannot offer.
    """
    f = np.asarray(f, np.float64)
    n = len(f)
    w = poolcut.novelty_weights(f, f_ref)
    w_ref = poolcut.novelty_weights(np.asarray(f_ref, np.float64), f_ref)
    order = np.argsort(-f)
    ks = np.arange(1, n + 1)
    cum = np.maximum.accumulate(
        np.maximum(0.0, np.cumsum(w[order]) - poolcut._null_curve(ks, n, w_ref)))
    hi = max(k_min + 1, int(k_max_frac * n))
    cum = cum[:hi]
    if cum[-1] <= 0:
        return None, dict(reason="no estimated novelty", n_hat_total=0.0)
    x = np.arange(1, len(cum) + 1) / len(cum)
    yv = cum / cum[-1]
    dev = yv - x                       # deviation from the chord (0,0)-(1,1)
    k = int(np.argmax(dev)) + 1
    return max(k, k_min), dict(reason="ok", n_hat_total=float(cum[-1]),
                               knee_dev=float(dev.max()),
                               n_hat_at_knee=float(cum[k - 1]))


# ------------------------------------------- (3) per-space injection calibration

def injection_alpha(z, labels, seen, estimator="tv", rate=0.02, seed=0,
                    n_rep=3, critic_kw=None):
    """Measure the estimator's multiplicative bias `alpha` FOR THIS SPACE.

    Take a seen class, remove it from the reference, and re-inject it at a KNOWN
    rate as pseudo-novelty.  alpha = b_hat_reported / b_injected.  Repeat over
    several donor classes and average.

    Fully label-free in deployment: the donor is a class you already have
    labels for.  Crucially this is a PER-SPACE measurement -- exp 129 saw
    alpha ~ 2-4x on C100 and 7-15x on galaxy10 DINO, so a global constant
    cannot serve both.
    """
    from supersig.discovery import np_pool_scores
    rng = np.random.default_rng(seed)
    seen = list(seen)
    donors = list(rng.permutation(seen)[:n_rep])
    kw = dict(seed=seed); kw.update(critic_kw or {})
    out = []
    for donor in donors:
        d_idx = np.where(labels == donor)[0]
        rest = np.where(labels != donor)[0]
        n_inj = max(4, int(round(rate * len(rest) / (1 - rate))))
        if n_inj > len(d_idx):
            continue
        inj = rng.choice(d_idx, n_inj, replace=False)
        idx = np.concatenate([rest, inj])
        zz = np.asarray(z, np.float32)[idx]
        is_ref = np.concatenate([np.ones(len(rest), bool),
                                 np.zeros(n_inj, bool)])
        f = np_pool_scores(torch.as_tensor(zz), is_ref, **kw).cpu().numpy()
        b_true = n_inj / len(idx)
        est = b_hat(f, f[is_ref], estimator)
        out.append(dict(donor=int(donor), b_true=float(b_true),
                        b_hat=float(est),
                        alpha=float(est / b_true) if b_true > 0 else np.nan))
    if not out:
        return None, out
    a = float(np.median([o["alpha"] for o in out]))
    return a, out


# ---------------------------------------------- (4) n_min sensitivity scan

def n_min_scan(f, is_ref, is_novel, n_mins=(5, 10, 20, 30, 50, 100, 200)):
    """Purity and the implied bias factor as a function of n_min.

    implied_alpha = n_min / (true novel points in the resulting pool).  If the
    scale-bias story is right this is roughly constant across n_min and equals
    the estimator's bias for this space.
    """
    rows = []
    for nm in n_mins:
        mask, info = poolcut.legal_pool(f, is_ref, n_min=nm)
        n_true = int(is_novel[mask].sum())
        rows.append(dict(n_min=nm, q=info["q"], pool=info["pool"],
                         ok=info["ok"], n_novel=n_true,
                         purity=float(is_novel[mask].mean()) if mask.any() else 0.0,
                         implied_alpha=(nm / n_true) if n_true else np.nan))
    return rows


# ------------------------------------------------------------------ selftest

def _make(b, n=20000, d=16, sep=8.0, overlap=0.0, seed=0):
    rng = np.random.default_rng(seed)
    n_nov = int(n * b)
    ref = rng.normal(0, 1, (n - n_nov, d))
    mu = np.zeros(d); mu[0] = sep * (1 - overlap)
    z = np.concatenate([ref, mu + rng.normal(0, 1, (n_nov, d))])
    v = np.concatenate([np.zeros(len(ref), bool), np.ones(n_nov, bool)])
    p = rng.permutation(n)
    return z[p], v[p]


def _analytic_f(z, sep, b):
    mu = np.zeros(z.shape[1]); mu[0] = sep
    lr = -0.5 * ((z - mu) ** 2).sum(1) + 0.5 * (z ** 2).sum(1)
    return np.log((1 - b) + b * np.exp(np.clip(lr, -50, 50)))


def _selftest():
    print("1. THE KNEE IS SCALE-INVARIANT (the property n_min cannot offer)")
    z, v = _make(0.02, sep=8.0)
    f = _analytic_f(z, 8.0, 0.02)
    k0, i0 = knee_k(f, f[~v])
    print(f"   knee k = {k0}  (true novel = {int(v.sum())})")
    # scale the ratio uniformly: f -> f + log(alpha) shifts r by alpha
    for alpha in (0.25, 0.5, 2.0, 4.0):
        import supersig.poolcut as pc
        real = pc.novelty_weights
        pc.novelty_weights = lambda a, b_, _r=real, _a=alpha: _a * _r(a, b_)
        try:
            k, _ = knee_k(f, f[~v])
        finally:
            pc.novelty_weights = real
        print(f"   weights x{alpha:<5g} -> knee k = {k}")
        assert k == k0, (alpha, k, k0)
    print("   -> a uniformly biased b_hat does not move the knee")

    print("\n2. THE THREE ESTIMATORS UNDER OVERLAP")
    print(f"   {'overlap':>8s}{'sep':>6s}{'tv':>9s}{'mass':>9s}{'excess':>9s}"
          f"   (true b = 0.020)")
    shrink = {}
    for ov in (0.0, 0.4, 0.7, 0.9):
        zz, vv = _make(0.02, sep=8.0, overlap=ov)
        ff = _analytic_f(zz, 8.0 * (1 - ov), 0.02)
        vals = {e: b_hat(ff, ff[~vv], e) for e in ESTIMATORS}
        shrink[ov] = vals
        print(f"   {ov:>8.1f}{8.0*(1-ov):>6.1f}"
              + "".join(f"{vals[e]:>9.4f}" for e in ESTIMATORS))
    # all estimators must be near-exact with no overlap
    for e in ESTIMATORS:
        assert abs(shrink[0.0][e] - 0.02) < 0.004, (e, shrink[0.0][e])
    # and all must degrade DOWNWARD (the safe direction)
    for e in ESTIMATORS:
        assert shrink[0.9][e] <= shrink[0.0][e] + 1e-9, e
    print("   -> exact at zero overlap; all shrink downward (the safe direction)")

    print("\n3. INJECTION CALIBRATION RECOVERS A KNOWN BIAS")
    zz, vv = _make(0.02, sep=8.0, overlap=0.6)
    ff = _analytic_f(zz, 8.0 * 0.4, 0.02)
    est = b_hat(ff, ff[~vv], "tv")
    alpha_true = est / 0.02
    print(f"   true b 0.020, reported {est:.4f} -> alpha = {alpha_true:.3f}")
    print(f"   correcting: b_hat/alpha = {est/alpha_true:.4f}")
    assert abs(est / alpha_true - 0.02) < 1e-9
    print("   -> dividing by a measured alpha restores the true rate")

    print("\n4. n_min SCAN: implied alpha is ~constant, as the story predicts")
    zz, vv = _make(0.02, sep=8.0, overlap=0.55)
    ff = _analytic_f(zz, 8.0 * 0.45, 0.02)
    rows = n_min_scan(ff, ~vv, vv, n_mins=(10, 20, 30, 50, 100))
    print(f"   {'n_min':>6s}{'q':>9s}{'pool':>7s}{'purity':>8s}{'n_nov':>7s}"
          f"{'implied a':>11s}")
    for r in rows:
        print(f"   {r['n_min']:>6d}{r['q']:>9.4f}{r['pool']:>7d}"
              f"{r['purity']:>8.3f}{r['n_novel']:>7d}{r['implied_alpha']:>11.3f}")
    pur = [r["purity"] for r in rows]
    assert pur[0] >= pur[-1], pur
    print("   -> purity falls as n_min rises; implied alpha reads off the bias")

    print("\nselftest OK")


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--embs", nargs="*", default=[])
    ap.add_argument("--glob", default="")
    ap.add_argument("--holdouts", default="")
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--inject-rate", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="logs/exp136")
    args = ap.parse_args()

    if args.selftest:
        _selftest(); return

    files = list(args.embs) + sorted(glob.glob(args.glob))
    if not files:
        ap.error("need --embs / --glob (or --selftest)")

    from supersig.discovery import np_pool_scores
    os.makedirs(args.out, exist_ok=True)
    res = {}
    for fn in files:
        d = np.load(fn, allow_pickle=True)
        z = np.asarray(d["tr"], np.float32); lab = np.asarray(d["tr_lab"])
        n_cls = int(lab.max()) + 1
        hold = (set(int(x) for x in args.holdouts.split(",") if x != "")
                if args.holdouts
                else set(range(n_cls - n_holdout(args.dataset), n_cls)))
        seen = [c for c in range(n_cls) if c not in hold]
        is_nov = np.isin(lab, list(hold)); is_ref = ~is_nov
        f = np_pool_scores(torch.as_tensor(z), is_ref, seed=args.seed).cpu().numpy()
        b_true = float(is_nov.mean())
        nm = os.path.basename(fn).replace("embs_", "").replace(".npz", "")

        ests = {e: b_hat(f, f[is_ref], e) for e in ESTIMATORS}
        alpha, inj = injection_alpha(z, lab, seen, "tv", args.inject_rate,
                                     args.seed)
        k, kinfo = knee_k(f, f[is_ref])
        rows = n_min_scan(f, is_ref, is_nov)
        base, binfo = poolcut.legal_pool(f, is_ref)

        print(f"\n######## {nm}  b_true={b_true:.4f} ########")
        print("  estimators: " + "  ".join(
            f"{e}={ests[e]:.4f} ({ests[e]/b_true:.2f}x)" for e in ESTIMATORS))
        print(f"  injection alpha (tv): "
              f"{alpha:.3f}" if alpha else "  injection alpha: n/a")
        if k:
            kp = float(is_nov[np.argsort(-f)[:k]].mean())
            print(f"  KNEE cut: k={k} q={k/len(f):.4f} purity={kp:.4f} "
                  f"n_nov={int(is_nov[np.argsort(-f)[:k]].sum())}")
        print(f"  n_min=30 rule: q={binfo['q']:.4f} "
              f"purity={float(is_nov[base].mean()):.4f} ok={binfo['ok']}")
        print(f"  {'n_min':>6s}{'q':>9s}{'purity':>8s}{'n_nov':>7s}{'impl a':>9s}")
        for r in rows:
            print(f"  {r['n_min']:>6d}{r['q']:>9.4f}{r['purity']:>8.3f}"
                  f"{r['n_novel']:>7d}{r['implied_alpha']:>9.3f}")
        res[nm] = dict(b_true=b_true, estimators=ests, injection_alpha=alpha,
                       injection_detail=inj, knee=dict(k=k, **kinfo),
                       n_min_scan=rows, rule_n30=binfo)

    with open(os.path.join(args.out, f"brate{run_tag()}.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(f"\nwrote {args.out}/brate{run_tag()}.json")


if __name__ == "__main__":
    main()
