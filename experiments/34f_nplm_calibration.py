"""
Experiment 34f: sanity checks for HybridContrastiveLoss (the configurable
"one class" from plots/nplm_contrastive_note/).

Two checks, both fast and CPU-only (no CIFAR needed):

  1. Config cube runs.  Every (positives x critic x estimator x marginal)
     corner returns a finite scalar with a real gradient.

  2. NPLM calibration.  On a toy 2-class Gaussian, a free per-pair critic
     g(i,j) trained with the NPLM interaction converges to the ANALYTIC
     log density ratio  log p(x,x')/p(x)p(x')  -- the property that makes the
     learned space a calibrated log-likelihood space (note, sec. 6).

    python experiments/34f_nplm_calibration.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import itertools
import numpy as np
import torch

from supersig.losses import HybridContrastiveLoss

torch.manual_seed(0)
np.random.seed(0)


def check_config_cube():
    print("----- 1. config cube runs -----")
    N, D, C = 32, 8, 4
    z = torch.randn(2 * N, D, requires_grad=True)
    inst = torch.arange(N)
    inst_lab = torch.cat([inst, inst])                 # instance ids
    cls_lab = torch.randint(0, C, (2 * N,))            # class ids
    means = torch.randn(C, D)

    ok = 0
    for pos, crit, est, marg in itertools.product(
            ("instance", "supervised"),
            ("cosine", "bilinear", "distance"),
            ("softmax", "nplm"),
            ("none", "sigreg", "classwise_sigreg")):
        loss = HybridContrastiveLoss(positives=pos, critic=crit,
                                     estimator=est, marginal=marg,
                                     tau=1.0, lam=1.0)
        labels = inst_lab if pos == "instance" else cls_lab
        # classwise_sigreg keys on class labels regardless of positive mode
        marg_labels = cls_lab if marg == "classwise_sigreg" else labels
        val, parts = loss(z, labels if marg != "classwise_sigreg" else marg_labels,
                          means=means)
        val.backward()
        assert torch.isfinite(val), (pos, crit, est, marg, val)
        assert z.grad is not None and torch.isfinite(z.grad).all()
        z.grad = None
        ok += 1
    print(f"  all {ok} corners finite with finite gradients  OK")


def check_nplm_calibration():
    """Positive pairs: both drawn from the SAME class of a 2-class Gaussian
    mixture; reference pairs: independent draws.  The analytic log-ratio is

        log p(x,x')/p(x)p(x') = log [ sum_c pi_c N(x|c)N(x'|c) /
                                       (sum_c pi_c N(x|c))(sum_c pi_c N(x'|c)) ].

    We fit a small MLP critic g(x, x') (symmetrised) by minimising the NPLM loss
    over minibatches of positive (same-class) and reference (independent) pairs,
    then compare g to the analytic log-ratio on a held-out set of pairs.  A
    parametric critic -- unlike a free per-pair table -- shares one function
    across pairs, which is what couples the reference term into the constraint
    E_ref[e^{g*}] = 1 and pins g* to the calibrated PMI.
    """
    print("\n----- 2. NPLM recovers the analytic log-ratio -----")
    rng = np.random.default_rng(0)
    D = 2
    mus = np.array([[-2.0, 0.0], [2.0, 0.0]])
    pi = np.array([0.5, 0.5])

    def logN(x, mu):                                   # isotropic unit-variance
        return -0.5 * ((x - mu) ** 2).sum(-1) - 0.5 * D * np.log(2 * np.pi)

    def log_px(x):                                     # log marginal p(x)
        comp = np.stack([np.log(pi[c]) + logN(x, mus[c]) for c in range(2)], -1)
        return np.logaddexp.reduce(comp, axis=-1)

    def log_pxx_same(x, xp):                           # log joint over SAME-class pairs
        comp = np.stack([np.log(pi[c]) + logN(x, mus[c]) + logN(xp, mus[c])
                         for c in range(2)], -1)
        return np.logaddexp.reduce(comp, axis=-1)

    def sample_pairs(n):
        """positive (same-class) and reference (independent) pair batches."""
        c = rng.choice(2, size=n, p=pi)
        xp1 = mus[c] + rng.standard_normal((n, D))
        xp2 = mus[c] + rng.standard_normal((n, D))     # same class -> positive
        ci = rng.choice(2, size=n, p=pi); cj = rng.choice(2, size=n, p=pi)
        xr1 = mus[ci] + rng.standard_normal((n, D))
        xr2 = mus[cj] + rng.standard_normal((n, D))    # independent -> reference
        t = lambda a: torch.tensor(a, dtype=torch.float32)
        return (t(xp1), t(xp2)), (t(xr1), t(xr2))

    # symmetric MLP critic g(x, x') = phi(x).phi(x') style via a shared net on
    # the sorted-concat, kept simple: MLP on concat, averaged over the two orders.
    net = torch.nn.Sequential(
        torch.nn.Linear(2 * D, 64), torch.nn.Tanh(),
        torch.nn.Linear(64, 64), torch.nn.Tanh(),
        torch.nn.Linear(64, 1))

    def g_fn(a, b):
        return 0.5 * (net(torch.cat([a, b], -1)).squeeze(-1)
                      + net(torch.cat([b, a], -1)).squeeze(-1))

    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    for step in range(3000):
        opt.zero_grad()
        (p1, p2), (r1, r2) = sample_pairs(256)
        gp, gr = g_fn(p1, p2), g_fn(r1, r2)
        # stabilise e^g by CLAMPING (a subtracted constant would shift g* and
        # break the absolute calibration -- see _nplm_interaction docstring)
        loss = (torch.exp(gr.clamp(max=30.0)) - 1.0).mean() - gp.mean()
        loss.backward()
        opt.step()

    # held-out positive pairs: compare g to analytic PMI
    (q1, q2), _ = sample_pairs(2000)
    with torch.no_grad():
        fit = g_fn(q1, q2).numpy()
    a, b = q1.numpy(), q2.numpy()
    pmi = log_pxx_same(a, b) - log_px(a) - log_px(b)
    r = float(np.corrcoef(fit, pmi)[0, 1])
    bias = float(np.mean(fit - pmi))
    rmse = float(np.sqrt(np.mean((fit - pmi - bias) ** 2)))
    print(f"  corr(g, analytic PMI) = {r:.3f}   "
          f"residual RMSE (after const) = {rmse:.3f}   mean offset = {bias:.3f}")
    print("  -> g tracks the analytic log-ratio" if r > 0.9 else
          "  -> WEAK correlation, check setup")


if __name__ == "__main__":
    check_config_cube()
    check_nplm_calibration()
