"""
Experiment 129: a LABEL-FREE rule for the discovery pool cut, and its oracle gap.

WHY THIS IS NOT JUST "SWEEP IT".  Exp 128 shows the pool cut `q` matters a lot
and that `tau_quantile=0.95` (inherited from exp 23, never derived) is far from
optimal.  But `purity` is computed from the novel labels, so choosing `q` by
sweeping purity is ORACLE TUNING.  The campaign has already been careful about
exactly this for `tau`: exp 115 (the dissociation survives per-arm tuning that
is restricted to seen-class accuracy) and exps 120/121 (the tau basin is
unreachable by label-free OR transductive selection).  It would be incoherent
to establish that for tau and then quietly oracle-tune q.

So exp 128 is a DIAGNOSTIC (how much headroom is there, is the scorer or the cut
the bottleneck) and this script is the DELIVERABLE: one global, label-free rule,
reported honestly against the oracle it is trying to approximate.

THE RULE.  From exp 128, purity = e_s*b/q and purity <= min(1, b/q), so the
right cut scales with the base rate:

    q* = clip( b_hat / P_TARGET , Q_MIN, Q_MAX )      P_TARGET = 0.30
    abort if b_hat * N < N_MIN                        (no detectable cluster)

P_TARGET = 0.30 gives margin over the 0.15 gate while keeping the pool large
enough to cluster.  `b` is unknown in the open world -- but the NP critic we
already fit estimates it, with no labels.

ESTIMATING b WITHOUT LABELS.  The critic returns f = log(p_corpus / p_ref),
self-calibrated so E_ref[e^f] = 1.  Under contamination

    p_corpus = (1 - b) p_ref + b p_sig     =>     r := e^f = (1-b) + b p_sig/p_ref

three estimators fall out, implemented here and left for the data to choose:

  tv        b_hat = E_ref[max(0, r - 1)].  This is the total variation distance
            TV(p_corpus, p_ref) = b * TV(p_sig, p_ref), so it is EXACT when the
            novel class is disjoint from the reference and UNDER-estimates as
            they overlap.  The bias direction is known and safe: it shrinks q.
  mass      b_hat = P_corpus(r > 2), the corpus fraction at least 2x over-dense.
            Cruder, but insensitive to the tails of f.
  excess    the classic bump-hunt excess over background:
            b_hat = (P_c(f>t) - P_r(f>t)) / (1 - P_r(f>t)) at t = the 0.99
            reference quantile.  Uses only the reference to define "normal".

All three see only the corpus and the seen-labelled reference -- never a
holdout label -- so the rule is open-world legal.

WHAT IS REPORTED.  Per cell and scorer: b_true vs each b_hat; the oracle-best
usable cut and its purity; the rule's cut and its purity; the ORACLE GAP
between them; and the incumbent q=0.05 baseline, because the practical question
is whether the rule beats the inherited constant, not whether it matches an
oracle it cannot see.

Either outcome is a result.  A small gap gives the paper a principled, adaptive
cut.  A large gap puts q alongside tau as a knob whose optimum is invisible to
legal selection -- which is the campaign's recurring and honest finding, not a
failure.

Evaluation-only.

    python experiments/129_legal_pool_cut.py --selftest
    python experiments/129_legal_pool_cut.py --embs logs/exp113/embs/cifar100_on.npz --holdouts 99
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from supersig.holdouts import n_holdout, run_tag
import argparse
import glob
import importlib.util
import json

import numpy as np
import torch

_spec = importlib.util.spec_from_file_location(
    "exp128", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "128_pool_cut_optimization.py"))
exp128 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp128)

P_TARGET = 0.30
Q_MIN, Q_MAX = 0.002, 0.20
N_MIN = exp128.N_MIN


# ------------------------------------------------- label-free base-rate estimators

def _ratio(f, f_ref):
    """exp(f), renormalised so that E_ref[r] == 1.

    The NP minimiser satisfies E_ref[e^f] = 1, but the FITTED critic often does
    not: `f` is clamped to +-20 and a clamped value passes zero gradient, so a
    reference point pegged at the clamp contributes e^20 to the loss with no
    corrective signal.  Measured on a synthetic 8-sigma-separated mode,
    E_ref[e^f] came out 60 to 1.1e6 instead of 1, and got WORSE with more steps.
    Renormalising restores the identity the estimators below assume, using only
    reference points (no labels).  It is a no-op on a well-calibrated critic.
    """
    r = np.exp(np.clip(f, -20, 20))
    r_ref = np.exp(np.clip(f_ref, -20, 20))
    z = float(np.mean(r_ref))
    return r / z if z > 0 else r


def calibration_error(f_ref):
    """E_ref[e^f]; 1.0 for a converged critic.  Report it -- a value far from 1
    means the critic did not converge and every downstream number is suspect."""
    return float(np.mean(np.exp(np.clip(f_ref, -20, 20))))


def b_hat_tv(f_corpus, f_ref):
    """TV / excess-mass estimator, evaluated on the CORPUS.

        TV = int max(0, p_c - p_r) = E_corpus[ max(0, 1 - 1/r) ]

    and TV(p_corpus, p_ref) = b * TV(p_sig, p_ref), so this is exact when the
    novel class is disjoint from the reference and UNDER-estimates as they
    overlap -- a safe bias direction, since it shrinks q.

    NOTE the equivalent reference-side form E_ref[max(0, r-1)] is algebraically
    identical but USELESS in Monte Carlo: when novelty is disjoint the excess
    lives where p_ref ~ 0, so no reference sample ever lands there and the
    estimate collapses to 0.  Measured: 1.2e-5 against a true b of 0.01.
    Always integrate the excess against the sample that COVERS it.
    """
    r = _ratio(f_corpus, f_ref)
    return float(np.mean(np.maximum(0.0, 1.0 - 1.0 / np.maximum(r, 1e-12))))


def b_hat_mass(f_corpus, f_ref, thresh=2.0):
    """Corpus fraction at least `thresh`x over-dense relative to the reference."""
    r = _ratio(f_corpus, f_ref)
    return float(np.mean(r > thresh))


def b_hat_excess(f_corpus, f_ref, ref_q=0.99):
    """Bump-hunt excess over background at a reference-defined threshold."""
    t = float(np.quantile(f_ref, ref_q))
    p_c = float(np.mean(f_corpus > t))
    p_r = float(np.mean(f_ref > t))
    if p_r >= 1.0:
        return 0.0
    return float(max(0.0, (p_c - p_r) / (1.0 - p_r)))


ESTIMATORS = {"tv": b_hat_tv, "mass": b_hat_mass, "excess": b_hat_excess}


def estimate_b(scores, is_seen_lab, which="tv"):
    """`scores` must be the NP critic's f (log density ratio).  The reference is
    the seen-labelled subset -- the same split the pool threshold already uses,
    so no new information is consumed."""
    f_ref = scores[is_seen_lab]
    return ESTIMATORS[which](scores, f_ref)


# ------------------------------------------------------------------- the rule

def novelty_weights(f_corpus, f_ref):
    """Per-point estimated 'novel-ness', label-free.

    Under p_corpus = (1-b) p_ref + b p_sig the posterior that a corpus point is
    novel is  1 - p_ref/p_corpus = 1 - 1/r.  Summing it over any subset is a
    label-free estimate of HOW MANY novel points that subset contains -- which
    is exactly the quantity the BIC-detectability constraint needs and which
    purity cannot supply without labels.
    """
    r = _ratio(f_corpus, f_ref)
    return np.maximum(0.0, 1.0 - 1.0 / np.maximum(r, 1e-12))


def rule_q(b_hat, n_total, p_target=P_TARGET, q_min=Q_MIN, q_max=Q_MAX,
           n_min=N_MIN):
    """RULE A (purity-target): q* = clip(b_hat / P_TARGET).

    Hits P_TARGET by construction when the scorer is good, which is also its
    weakness: it stops there instead of tightening further.  Kept as a
    baseline for RULE B.
    """
    if b_hat <= 0:
        return q_max, False, "b_hat<=0"
    q = float(np.clip(b_hat / p_target, q_min, q_max))
    if b_hat * n_total < n_min:
        return q, False, f"b_hat*N={b_hat*n_total:.0f} < n_min={n_min}"
    return q, True, "ok"


def rule_q_detect(f_corpus, f_ref, qs, n_min=N_MIN, p_target=P_TARGET,
                  q_min=Q_MIN, q_max=Q_MAX):
    """RULE B (detectability-limited): the TIGHTEST cut that still holds
    `n_min` ESTIMATED novel points.

    Purity = e_s*b/q rises monotonically as q shrinks, so the optimum is at the
    tight end and the binding constraint is not a purity target but whether a
    cluster survives.  `novelty_weights` estimates the survivor count without
    labels, so this is open-world legal.

    Returns (q, ok, reason, diagnostics).
    """
    w = novelty_weights(f_corpus, f_ref)
    n = len(f_corpus)
    order = np.argsort(-f_corpus)
    cw = np.cumsum(w[order])                       # est. novel count vs pool size
    total = float(cw[-1])
    if total < n_min:
        return q_max, False, f"total est novelty {total:.0f} < n_min={n_min}", \
            dict(n_hat_total=total)
    best = None
    for q in sorted(qs):
        k = max(1, int(round(q * n)))
        n_hat = float(cw[min(k, n) - 1])
        if n_hat >= n_min:
            best = (float(np.clip(q, q_min, q_max)), n_hat, n_hat / k)
            break
    if best is None:
        return q_max, False, "no q holds n_min", dict(n_hat_total=total)
    q, n_hat, p_hat = best
    ok = p_hat >= 0.5 * p_target                   # sanity floor on est. purity
    return q, ok, "ok" if ok else f"est purity {p_hat:.3f} low", \
        dict(n_hat=n_hat, purity_hat=p_hat, n_hat_total=total)


# ------------------------------------------------------------------ evaluation

def evaluate(scores, is_novel, is_seen_lab, qs, p_target=P_TARGET,
             n_min=N_MIN, est="tv"):
    """Oracle-best cut vs the rule's cut vs the inherited q=0.05."""
    n = len(scores)
    b_true = float(is_novel.mean())
    b_est = estimate_b(scores, is_seen_lab, est)

    rows = exp128.curve(scores, is_novel, qs)
    oracle = exp128.best_q(rows, n_min)

    q_r, ok, why = rule_q(b_est, n, p_target, n_min=n_min)
    r_rule = exp128.operating_point(scores, is_novel, q_r)
    q_d, ok_d, why_d, diag = rule_q_detect(scores, scores[is_seen_lab], qs,
                                           n_min, p_target)
    r_det = exp128.operating_point(scores, is_novel, q_d)
    r_base = exp128.operating_point(scores, is_novel, 0.05)

    return dict(
        b_true=b_true, b_hat=b_est,
        b_rel_err=float((b_est - b_true) / b_true) if b_true > 0 else float("nan"),
        q_oracle=oracle["q"] if oracle else None,
        purity_oracle=oracle["purity"] if oracle else None,
        q_rule=q_r, rule_ok=ok, rule_reason=why,
        purity_rule=r_rule["purity"], n_novel_rule=r_rule["n_novel"],
        q_base=0.05, purity_base=r_base["purity"], n_novel_base=r_base["n_novel"],
        q_detect=q_d, detect_ok=ok_d, detect_reason=why_d,
        purity_detect=r_det["purity"], n_novel_detect=r_det["n_novel"],
        n_hat=diag.get("n_hat"), purity_hat=diag.get("purity_hat"),
        oracle_gap=(oracle["purity"] - r_rule["purity"]) if oracle else None,
        oracle_gap_detect=(oracle["purity"] - r_det["purity"]) if oracle else None,
        beats_incumbent=bool(r_rule["purity"] > r_base["purity"]),
        detect_beats_incumbent=bool(r_det["purity"] > r_base["purity"]),
    )


# ------------------------------------------------------------------- selftest

def _make(b, n=20000, d=16, sep=6.0, overlap=0.0, seed=0):
    """Corpus with a known novel fraction `b`.  `overlap` slides the novel mode
    back toward the reference, which is what makes the estimators degrade."""
    rng = np.random.default_rng(seed)
    n_nov = int(n * b)
    ref = rng.normal(0, 1, (n - n_nov, d))
    mu = np.zeros(d)
    mu[0] = sep * (1.0 - overlap)
    nov = mu + rng.normal(0, 1, (n_nov, d))
    z = np.concatenate([ref, nov])
    is_novel = np.concatenate([np.zeros(len(ref), bool), np.ones(n_nov, bool)])
    p = rng.permutation(n)
    return z[p], is_novel[p]


def analytic_f(z, mu, b):
    """Exact f = log(p_corpus/p_ref) for the synthetic mixture, so the ESTIMATOR
    ALGEBRA can be tested independently of whether the critic converged."""
    d2_ref = (z ** 2).sum(1)
    d2_sig = ((z - mu) ** 2).sum(1)
    lr = -0.5 * d2_sig + 0.5 * d2_ref                 # log(p_sig/p_ref)
    return np.log((1 - b) + b * np.exp(np.clip(lr, -50, 50)))


def _selftest():
    print("1. ESTIMATOR ALGEBRA on an EXACT critic (tests the math, not the fit)")
    print(f"   {'b_true':>8s}{'tv':>9s}{'mass':>9s}{'excess':>9s}")
    for b in (0.01, 0.02, 0.05, 0.10):
        z, is_novel = _make(b, sep=8.0)
        mu = np.zeros(z.shape[1]); mu[0] = 8.0
        f = analytic_f(z, mu, b)
        est = {k: estimate_b(f, ~is_novel, k) for k in ESTIMATORS}
        print(f"   {b:>8.3f}{est['tv']:>9.4f}{est['mass']:>9.4f}"
              f"{est['excess']:>9.4f}")
        for k, v in est.items():
            assert 0.5 * b <= v <= 2.0 * b, (b, k, v)

    print("\n2. the rule's q tracks the base rate (q* ~ b/0.3)")
    for b in (0.01, 0.05, 0.10):
        q, ok, why = rule_q(b, 20000)
        assert abs(q - float(np.clip(b / P_TARGET, Q_MIN, Q_MAX))) < 1e-9
        print(f"   b={b:.3f} -> q*={q:.4f}  ok={ok}")

    print("\n3. the rule ABORTS when too few novel points to cluster")
    _, ok, why = rule_q(0.001, 20000)
    assert not ok
    print(f"   b_hat=0.001, N=20000 -> ok={ok} ({why})")
    _, ok, _ = rule_q(0.05, 20000)
    assert ok
    print(f"   b_hat=0.050, N=20000 -> ok={ok}")

    print("\n4. END-TO-END with the exact critic: two rules vs oracle vs q=0.05")
    qs = np.unique(np.concatenate([np.geomspace(0.002, 0.02, 10),
                                   np.linspace(0.02, 0.20, 19)]))
    print(f"   {'b':>6s}{'b_hat':>8s}{'q_orc':>8s}{'P_orc':>8s}"
          f"{'qA':>8s}{'P_A':>7s}{'qB':>8s}{'P_B':>7s}{'n_hat':>7s}"
          f"{'P@.05':>7s}")
    for b in (0.01, 0.02, 0.05, 0.10):
        z, is_novel = _make(b, sep=8.0)
        mu = np.zeros(z.shape[1]); mu[0] = 8.0
        f = analytic_f(z, mu, b)
        r = evaluate(f, is_novel, ~is_novel, qs)
        print(f"   {r['b_true']:>6.3f}{r['b_hat']:>8.4f}"
              f"{(r['q_oracle'] or 0):>8.4f}{(r['purity_oracle'] or 0):>8.4f}"
              f"{r['q_rule']:>8.4f}{r['purity_rule']:>7.3f}"
              f"{r['q_detect']:>8.4f}{r['purity_detect']:>7.3f}"
              f"{(r['n_hat'] or 0):>7.0f}{r['purity_base']:>7.3f}")
        # Rule B (detectability-limited) must dominate Rule A (purity-target):
        # purity rises as q shrinks, and B tightens until detectability binds.
        assert r["purity_detect"] >= r["purity_rule"] - 1e-9, r
        # and B must keep a usable cluster
        assert r["n_novel_detect"] >= 0.5 * N_MIN, r
    print("   (Rule B dominates Rule A everywhere: purity rises as q shrinks,")
    print("    so target a DETECTABILITY floor, not a purity level.)")

    print("\n5. estimators never OVER-state b, and degrade under heavy overlap")
    vals = []
    for ov in (0.0, 0.3, 0.6, 0.85, 0.95):
        z, is_novel = _make(0.05, sep=8.0, overlap=ov)
        mu = np.zeros(z.shape[1]); mu[0] = 8.0 * (1 - ov)
        bh = estimate_b(analytic_f(z, mu, 0.05), ~is_novel, "tv")
        vals.append(bh)
        print(f"   overlap={ov:.2f} (sep {8.0*(1-ov):.1f}) -> b_hat={bh:.4f}"
              f"   (true 0.0500)")
    # never materially over-states: the bias direction must be SAFE (shrinks q)
    assert max(vals) <= 0.05 * 1.05, vals
    # and once the modes genuinely overlap it degrades downward
    assert vals[-1] < 0.5 * vals[0], vals
    print("   bias is one-sided (downward), which is the safe direction:")
    print("   under-stating b tightens q, it does not inflate the pool.")

    print("\n6. renormalisation repairs a MIScalibrated critic")
    z, is_novel = _make(0.05, sep=8.0)
    mu = np.zeros(z.shape[1]); mu[0] = 8.0
    f = analytic_f(z, mu, 0.05)
    good = estimate_b(f, ~is_novel, "tv")
    bad = estimate_b(f + 3.0, ~is_novel, "tv")   # a constant offset = miscalibration
    print(f"   b_hat {good:.4f} (calibrated) vs {bad:.4f} (offset by +3 nats)")
    print(f"   E_ref[e^f]: {calibration_error(f[~is_novel]):.3f} -> "
          f"{calibration_error(f[~is_novel] + 3.0):.3f}")
    assert abs(good - bad) < 1e-6, (good, bad)

    print("\n7. CRITIC HEALTH WARNING (characterisation, not a pass/fail)")
    from supersig.discovery import np_pool_scores
    from sklearn.metrics import roc_auc_score
    z, is_novel = _make(0.01, n=6000, sep=8.0)
    ff = np_pool_scores(torch.as_tensor(z, dtype=torch.float32), ~is_novel,
                        seed=0).cpu().numpy()
    ce = calibration_error(ff[~is_novel])
    a = roc_auc_score(is_novel, ff)
    print(f"   fitted critic on a trivially-separated 1% mode:")
    print(f"     AUC {a:.4f} (a converged critic should be ~1.00)")
    print(f"     E_ref[e^f] {ce:.1f} (should be 1.00)")
    print("   -> the fitted critic is the bottleneck, not the cut."
          "  See exp 130.")

    print("\nselftest OK")


# ----------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--embs", default="")
    ap.add_argument("--glob", default="")
    ap.add_argument("--holdouts", default="")
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--estimator", default="tv", choices=list(ESTIMATORS))
    ap.add_argument("--p-target", type=float, default=P_TARGET)
    ap.add_argument("--n-min", type=int, default=N_MIN)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="logs/exp129")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    files = ([args.embs] if args.embs else []) + sorted(glob.glob(args.glob))
    if not files:
        ap.error("need --embs or --glob (or --selftest)")

    from supersig.discovery import np_pool_scores
    qs = np.unique(np.concatenate([np.geomspace(0.002, 0.02, 12),
                                   np.linspace(0.02, 0.20, 19)]))
    os.makedirs(args.out, exist_ok=True)
    out = {}
    print(f"{'cell':28s}{'b_true':>8s}{'b_hat':>8s}{'q_orc':>8s}{'P_orc':>8s}"
          f"{'q_rule':>8s}{'P_rule':>8s}{'P@.05':>8s}{'gap':>8s}  rule")
    for fn in files:
        d = np.load(fn, allow_pickle=True)
        z = np.asarray(d["tr"], dtype=np.float64)
        lab = np.asarray(d["tr_lab"])
        n_cls = int(lab.max()) + 1
        hold = exp128._parse_holdouts(args.holdouts)
        if hold is None:
            hold = set(range(n_cls - n_holdout(args.dataset), n_cls))
        is_novel = np.isin(lab, list(hold))
        f = np_pool_scores(torch.as_tensor(z, dtype=torch.float32),
                           ~is_novel, seed=args.seed).cpu().numpy()
        r = evaluate(f, is_novel, ~is_novel, qs, args.p_target,
                     args.n_min, args.estimator)
        key = os.path.basename(fn).replace(".npz", "")
        out[key] = r
        print(f"{key[:27]:28s}{r['b_true']:>8.4f}{r['b_hat']:>8.4f}"
              f"{(r['q_oracle'] or 0):>8.4f}{(r['purity_oracle'] or 0):>8.4f}"
              f"{r['q_rule']:>8.4f}{r['purity_rule']:>8.4f}"
              f"{r['purity_base']:>8.4f}{(r['oracle_gap'] or 0):>8.4f}"
              f"  {'OK' if r['rule_ok'] else r['rule_reason']}")

    with open(os.path.join(args.out, f"legal_cut{run_tag()}.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    gaps = [v["oracle_gap"] for v in out.values() if v["oracle_gap"] is not None]
    wins = sum(v["beats_incumbent"] for v in out.values())
    if gaps:
        print(f"\nmean oracle gap {np.mean(gaps):+.4f}; "
              f"rule beats the inherited q=0.05 in {wins}/{len(out)} cells")
    print(f"wrote {args.out}/legal_cut{run_tag()}.json")


if __name__ == "__main__":
    main()
