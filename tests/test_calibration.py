"""Regression tests for the property that separates the two estimators of PMI:
ABSOLUTE calibration of the recovered log-ratio.

Every per-event power number in the campaign rests on one claim -- that the NPLM
estimator recovers log p(x,x')/p(x)p(x') on an absolute scale, so a threshold
fitted on a reference population transfers to data -- and on one implementation
detail, that `_nplm_interaction` is stabilised by CLAMPING the exponent rather
than by subtracting a per-row max.  Neither was covered: the existing
`test_nplm_clamp_prevents_overflow` asserts the clamp returns a finite number,
which a max-subtracting implementation would also do while silently destroying
the calibration.  These tests close that gap.

The organising fact (derived in docs/discovery_metrics_iclr.tex App. A) is that
a per-anchor shift g -> g + c is an EXACT flat direction of the softmax
estimator -- zero gradient and zero curvature, so the constant is not merely
unidentified but invisible to gradient descent -- and a strictly convex
direction of the NPLM estimator, whose unique minimum sits at the calibration
condition E_ref[e^g] = 1.  Tests below check both halves of that statement, plus
the population minimiser itself.

All tests are exact or convex-deterministic and run on CPU in well under a
second; there is no training and no dataset.
"""
import pytest
import torch

from supersig.config import DEVICE
from supersig.losses import _nplm_interaction, _softmax_interaction

DT = torch.float64            # tight tolerances; the identities below are exact


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def _toy_joint(K=4, seed=0):
    """A strictly positive joint p(a,b) over K x K categories, and its marginals.

    Discrete categories make the population PMI table available in closed form,
    which is what lets these tests assert exact values instead of trends.
    """
    g = torch.Generator().manual_seed(seed)
    P = torch.rand(K, K, generator=g, dtype=DT) + 0.05
    P = (P / P.sum()).to(DEVICE)
    return P, P.sum(1, keepdim=True), P.sum(0, keepdim=True)   # P, p(a), p(b)


def _population_nplm(gtab, P, pa, pb):
    """E_ref[e^g - 1] - E_pos[g] with expectations taken EXACTLY.

    Reference = the product of marginals p(a)p(b); positives = the joint P.
    Using exact expectations rather than samples removes Monte-Carlo noise, so
    the recovered table can be compared to the analytic PMI at 1e-3.
    """
    return ((pa * pb) * (torch.exp(gtab) - 1.0)).sum() - (P * gtab).sum()


def _fit_table(loss_fn, K=4, steps=3000, lr=0.05):
    """Minimise a convex loss over a free K x K critic table."""
    g = torch.zeros(K, K, dtype=DT, device=DEVICE, requires_grad=True)
    opt = torch.optim.Adam([g], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss_fn(g).backward()
        opt.step()
    return g.detach()


def _pair_setup(n=8, seed=0):
    """A symmetric critic matrix with a disjoint positive pairing (0-1, 2-3, ...)."""
    g = torch.Generator().manual_seed(seed)
    m = 0.5 * torch.randn(n, n, generator=g, dtype=DT)
    m = (m + m.t())                                   # symmetric, |entries| ~ O(1)
    m = m.to(DEVICE)
    self_mask = torch.eye(n, dtype=torch.bool, device=DEVICE)
    pos = torch.zeros(n, n, dtype=torch.bool, device=DEVICE)
    lo = torch.arange(0, n, 2, device=DEVICE)
    pos[lo, lo + 1] = True
    pos[lo + 1, lo] = True
    return m, pos, self_mask


# --------------------------------------------------------------------------- #
# 1. the NPLM population minimiser IS the calibrated PMI                       #
# --------------------------------------------------------------------------- #
def test_nplm_population_minimiser_is_calibrated_pmi():
    """The scientific claim, on a toy where PMI is known in closed form.

    Minimising E_ref[e^g - 1] - E_pos[g] over a free table must return
    g* = log P(a,b)/p(a)p(b) exactly -- not up to a constant -- and that
    minimiser must satisfy the calibration condition E_ref[e^{g*}] = 1.
    """
    P, pa, pb = _toy_joint(seed=0)
    ghat = _fit_table(lambda g: _population_nplm(g, P, pa, pb))
    pmi = torch.log(P / (pa * pb))

    assert torch.allclose(ghat, pmi, atol=2e-3)
    assert ((pa * pb) * torch.exp(ghat)).sum().item() == pytest.approx(1.0,
                                                                      abs=2e-3)


def test_calibration_condition_holds_exactly_for_true_pmi():
    """E_ref[e^{PMI}] = sum_ab p(a)p(b) * P(a,b)/(p(a)p(b)) = sum_ab P = 1.

    An algebraic identity, asserted at machine precision so that any future
    change to what the estimator targets trips this first.
    """
    P, pa, pb = _toy_joint(seed=3)
    pmi = torch.log(P / (pa * pb))
    assert ((pa * pb) * torch.exp(pmi)).sum().item() == pytest.approx(1.0,
                                                                     abs=1e-12)


# --------------------------------------------------------------------------- #
# 2. the gauge: softmax is invariant, NPLM is not                              #
# --------------------------------------------------------------------------- #
def test_softmax_interaction_is_exactly_invariant_to_per_anchor_shift():
    """The flat direction.  Adding c_i to row i leaves NT-Xent/SupCon unchanged
    to machine precision -- which is why the softmax estimator recovers PMI only
    up to a per-row constant, and why max-subtraction is a legal stabiliser
    there."""
    g, pos, self_mask = _pair_setup(seed=0)
    c = torch.randn(g.size(0), 1, dtype=DT, device=DEVICE)
    base = _softmax_interaction(g, pos, self_mask)
    shifted = _softmax_interaction(g + c, pos, self_mask)
    assert torch.allclose(base, shifted, atol=1e-10)


def test_nplm_interaction_is_not_invariant_to_shift():
    """The same shift must MOVE the NPLM loss; if it ever stops doing so, the
    absolute normalisation has been lost and per-event power will silently go
    to zero."""
    g, pos, self_mask = _pair_setup(seed=1)
    base = _nplm_interaction(g, pos, self_mask)
    # Shifts are deliberately large and asymmetric: L is convex with its
    # minimum at c* = -log E_ref[e^g] (typically a few tenths below 0), so a
    # small symmetric pair like -+0.5 can straddle c* and return almost to
    # L(0) -- a near-miss that would make this test flaky rather than wrong.
    for c in (-1.5, 1.5):
        shifted = _nplm_interaction(g + c, pos, self_mask)
        assert abs((shifted - base).item()) > 1e-2


def test_nplm_shift_gradient_equals_calibration_residual():
    """App. A: dL/dc at c=0 is exactly E_ref[e^g] - 1, the calibration residual.

    This is the identity that makes the summed critic gradient a free training
    diagnostic, so it is worth pinning.  (Exp 82 falsified the stronger use of
    it as a label-free SEED SELECTOR -- per-seed calibration does not track
    per-seed probe -- but the identity itself is what this test asserts.)
    """
    g, pos, self_mask = _pair_setup(seed=2)
    c = torch.zeros((), dtype=DT, device=DEVICE, requires_grad=True)
    _nplm_interaction(g + c, pos, self_mask).backward()

    ref = (~pos) & (~self_mask)
    expected = torch.exp(g[ref]).mean() - 1.0
    assert c.grad.item() == pytest.approx(expected.item(), rel=1e-8)


def test_nplm_shift_direction_is_convex_with_minimum_at_calibration():
    """The other half of App. A: the direction softmax cannot see is strictly
    convex for NPLM, and its minimiser c* = -log E_ref[e^g] is exactly the shift
    that restores E_ref[e^{g+c}] = 1."""
    g, pos, self_mask = _pair_setup(seed=4)
    ref = (~pos) & (~self_mask)
    c_star = -torch.log(torch.exp(g[ref]).mean())

    at = _nplm_interaction(g + c_star, pos, self_mask).item()
    for delta in (-0.25, 0.25):
        assert _nplm_interaction(g + c_star + delta, pos, self_mask).item() > at

    shifted_residual = torch.exp((g + c_star)[ref]).mean().item()
    assert shifted_residual == pytest.approx(1.0, rel=1e-8)


# --------------------------------------------------------------------------- #
# 3. clamping preserves the minimiser; max-subtraction destroys it             #
# --------------------------------------------------------------------------- #
def test_max_subtraction_destroys_absolute_normalisation():
    """Why docs/LOSSES.md forbids the softmax stabiliser here.

    Subtracting a per-row max is free in a log-ratio (it cancels) but shifts the
    NPLM minimiser bodily, breaking E_ref[e^g] = 1.  Asserted on the analytic
    PMI table so the failure is exact rather than optimisation-dependent.
    """
    P, pa, pb = _toy_joint(seed=1)
    ref, pmi = pa * pb, torch.log(P / (pa * pb))

    assert (ref * torch.exp(pmi)).sum().item() == pytest.approx(1.0, abs=1e-12)

    # Any nonzero per-row shift breaks it; on this toy the damage is large
    # (the sum lands around 0.6-0.7), but the assertion only needs "not 1".
    mangled = pmi - pmi.max(dim=1, keepdim=True).values
    assert abs((ref * torch.exp(mangled)).sum().item() - 1.0) > 1e-6


def test_clamping_leaves_the_critic_untouched_in_the_normal_range():
    """The legal stabiliser is a no-op wherever the critic is not pathological,
    so it cannot move g* -- the complement of the test above."""
    P, pa, pb = _toy_joint(seed=2)
    pmi = torch.log(P / (pa * pb))
    assert pmi.abs().max().item() < 30.0            # toy is in the normal range
    assert torch.allclose(pmi.clamp(max=30.0), pmi)


def test_nplm_interaction_matches_closed_form_when_clamp_inactive():
    """Implementation check: with every entry below the clamp, the returned
    value is exactly E_ref[e^g - 1] - E_pos[g]."""
    g, pos, self_mask = _pair_setup(seed=5)
    assert g.abs().max().item() < 30.0

    ref = (~pos) & (~self_mask)
    expected = (torch.exp(g[ref]) - 1.0).mean() - g[pos].mean()
    got = _nplm_interaction(g, pos, self_mask)
    assert got.item() == pytest.approx(expected.item(), rel=1e-10)
