"""Label-free selection of the discovery pool cut and the BIC search range.

Replaces two constants that the campaign inherited without deriving:

    tau_quantile = 0.95                     (exp 23, never justified)
    kmax = max(4, len(holdouts) + 2)        (discovery.py -- uses the NUMBER OF
                                             NOVEL CLASSES, i.e. oracle info)

THE ALGEBRA (exp 128).  With b the novel base rate, q the pool fraction, and
e_s / e_b the signal / background efficiencies of the cut,

    q      = e_s*b + e_b*(1-b)
    purity = e_s*b / q
    E      = purity/b = e_s/q               (enrichment == lift at q)
    purity <= min(1, b/q)                   (THE CEILING)

Purity rises monotonically as q shrinks, so there is no interior optimum in
purity and targeting a purity LEVEL is the wrong objective.  What actually
stops you is whether a CLUSTER survives.

THE RULE (exp 129, "Rule B").  The NP critic gives f = log(p_corpus/p_ref).
Under contamination p_corpus = (1-b) p_ref + b p_sig, the posterior that a
corpus point is novel is 1 - (1-b)/r with r = e^f; the b-free positive part

    w(x) = [1 - 1/r(x)]_+

sums over any subset to an estimate of how many NOVEL points it contains --
label-free, which is exactly what purity cannot be.  Then

    q* = the tightest cut whose estimated novel count still clears n_min
    k_max = clip( floor( sum(w) / n_min ), 2, 64 )

Both use only the corpus and the seen-labelled reference, never a holdout
label, so the whole selection is open-world legal.  This matters: the campaign
was careful to show the tau basin is unreachable by legal tuning (exps
115/120/121), and it would be incoherent to then oracle-tune the pool cut.

CONSTANTS, and how they were set (exp 129 sweep on real exp-54 CIFAR-10
embeddings with the novel class subsampled to realistic rates,
logs/exp129/n_min_sweep_real.json):

  N_MIN = 10    purity falls monotonically as n_min grows.  Lowered from 30
                after exp 138 measured the sweep on six real trained spaces:
                n_min      5     10     20     30     50    100    200
                purity  .833  .846   .807   .778   .743   .715   .665  (one space)
                n_min=10 buys +0.05-0.06 purity over 30 while leaving 107-112
                real novel points pooled.  THIS IS AN HONEST TUNING CONSTANT,
                not a bias correction: exp 138 showed the underestimate in
                b_hat is NOT a uniform scale (implied alpha climbs 0.071->0.381
                across the sweep), so lowering n_min does not "cancel" a known
                factor -- it is simply the operating point that measures best.
                Report the sweep alongside any purity number.
  Q_MIN = 5e-4  0.002 was BINDING at every n_min <= 75.  Purity still improves
                down to q ~ 0.001 and then plateaus, while recall collapses to
                well under 1%, so there is no gain in going tighter.
  Q_MAX = 0.20  a wide-pool fallback; reaching it means the rule did not engage.

KNOWN LIMITATION (exp 129, real data).  sum(w) estimates TV(p_corpus, p_ref)
= b * TV(p_sig, p_ref), so it is exact only when the novel class is disjoint
from the reference and biased DOWNWARD as they overlap.  On real embeddings
that bias is 4x on a well-separated space (AUC 0.766) and 100-300x on a weak
one (AUC 0.53-0.71).  The bias direction is safe -- it tightens the cut -- but
on weak spaces the estimate collapses, q saturates at Q_MAX, and `ok` is False.
REPORT THAT FLAG: False means "this space cannot support discovery", which is
a result, not a crash.
"""
import numpy as np

N_MIN = 10
Q_MIN, Q_MAX = 5e-4, 0.20
K_FLOOR, K_CAP = 2, 64


def calibrated_ratio(f, f_ref):
    """exp(f) renormalised so E_ref[r] == 1.

    The NP minimiser satisfies E_ref[e^f] = 1, but a FITTED critic need not --
    exp 130 measured in-sample 0.997-1.019 (fine) but out-of-sample up to 2.7
    on real spaces, i.e. mild overfitting of the reference subsample.
    Renormalising restores the identity these estimators assume, using
    reference points only.  A no-op on a perfectly calibrated critic.
    """
    r = np.exp(np.clip(f, -20, 20))
    z = float(np.mean(np.exp(np.clip(f_ref, -20, 20))))
    return r / z if z > 0 else r


def novelty_weights(f, f_ref):
    """Per-point estimated novel-ness in [0, 1); sums to an estimated count."""
    r = calibrated_ratio(f, f_ref)
    return np.maximum(0.0, 1.0 - 1.0 / np.maximum(r, 1e-12))


def estimated_novel_count(f, f_ref, subtract_null=True):
    """Estimated novel count in the corpus, with the null level subtracted.

    WHY SUBTRACTION IS NEEDED.  w = [1 - 1/r]_+ is a positive part, so ANY
    spread in a fitted f produces w > 0 even when there is no novelty at all.
    Measured: pure-noise scores (sd 0.3, no novel class) gave sum(w) = 665 out
    of 8000 points -- 8% phantom novelty, enough for the rule to engage on
    nothing.

    Under the null the corpus and the reference are the same distribution, so
    the per-point mean of w on REFERENCE points estimates that phantom level.
    Subtracting N_D * mean_ref(w) cancels it and leaves ~0 under the null,
    while a real excess survives.  Label-free: reference points only.
    """
    w = novelty_weights(f, f_ref)
    total = float(np.sum(w))
    if not subtract_null:
        return total
    return float(max(0.0, total - _null_expectation(len(w), len(w),
                                                    novelty_weights(
                                                        np.asarray(f_ref),
                                                        f_ref))))


def _null_curve(ks, n, w_ref):
    """Expected sum of `w` over the top-k of the corpus UNDER THE NULL.

    Under H0 corpus and reference are the same distribution, so the top
    q = k/n of the corpus is distributed like the top q of the reference:

        null(k) = k * mean( top-q fraction of w_ref )

    Rank-matched, so it cancels exactly when there is no novelty, while a real
    excess (which sits above the reference's own tail) survives.
    """
    w_sorted = np.sort(np.asarray(w_ref, dtype=np.float64))[::-1]
    m_ref = len(w_sorted)
    if m_ref == 0:
        return np.zeros(len(ks), dtype=np.float64)
    csum = np.cumsum(w_sorted)
    m = np.clip(np.round(ks / n * m_ref).astype(int), 1, m_ref)
    return ks * (csum[m - 1] / m)


def _null_expectation(k, n, w_ref):
    return float(_null_curve(np.array([k]), n, w_ref)[0])


def legal_pool(scores, is_seen_lab, n_min=N_MIN, q_min=Q_MIN, q_max=Q_MAX):
    """The label-free pool: tightest cut holding >= n_min ESTIMATED novel points.

    Returns (mask, info).  `info['ok']` is False when the estimated novelty
    cannot support a detectable cluster at any cut -- pool at q_max and say so.
    """
    scores = np.asarray(scores, dtype=np.float64)
    is_seen_lab = np.asarray(is_seen_lab, dtype=bool)
    n = len(scores)
    f_ref = scores[is_seen_lab]
    w = novelty_weights(scores, f_ref)
    w_ref = novelty_weights(f_ref, f_ref)
    order = np.argsort(-scores)
    cw_raw = np.cumsum(w[order])
    # RANK-MATCHED null: under H0 the top-k of the corpus look like the top
    # q = k/n fraction of the reference, so subtract that instead of a flat
    # mean.  A flat mean does NOT cancel -- the top-ranked null points carry
    # far above-average w, and their tail survives subtraction (measured: 405
    # phantom novel points remained out of 8000 pure-noise scores).
    ks = np.arange(1, n + 1)
    null = _null_curve(ks, n, w_ref)
    cw = np.maximum(0.0, cw_raw - null)
    total = float(cw[-1]) if n else 0.0
    null_rate = float(np.mean(w_ref))

    k_max_pts = max(1, int(round(q_max * n)))
    k_min_pts = max(1, int(round(q_min * n)))
    if total < n_min:
        k = k_max_pts
        ok, why = False, f"estimated novelty {total:.1f} < n_min={n_min}"
    else:
        hit = np.searchsorted(cw, n_min) + 1          # tightest k with cw >= n_min
        k = int(np.clip(hit, k_min_pts, k_max_pts))
        ok, why = True, "ok"
    mask = np.zeros(n, dtype=bool)
    mask[order[:k]] = True
    return mask, dict(ok=ok, reason=why, q=k / n, pool=k,
                      n_hat_pool=float(cw[min(k, n) - 1]) if n else 0.0,
                      n_hat_total=total,
                      null_rate=null_rate,
                      kmax=label_free_kmax(
                          np.maximum(0.0, w - null_rate), n_min))


def label_free_kmax(w, n_min=N_MIN, k_floor=K_FLOOR, k_cap=K_CAP):
    """BIC search range from estimated novel mass -- never from len(holdouts).

    NOTE (exp 129): at SINGLE holdout this is inert.  BIC returns khat=1 in
    essentially every measured row at every kmax from 2 to 63, which with one
    novel class is the correct answer.  The oracle leak it replaces therefore
    only has teeth in the multi-holdout regime.
    """
    return int(np.clip(int(np.floor(float(np.sum(w)) / max(n_min, 1))),
                       k_floor, k_cap))
