"""
Experiment 104: the interpretability panel -- is embedded distance a
log-likelihood, are the labelled components Gaussian, and is their width sigma=1?

WHY.  The campaign reports many metrics and it has become hard to see whether
the space is actually doing the one thing the program is built around: making
    distance to a labelled anchor  ==  a delta log-likelihood in a hypothesis test.
That property is what makes a space interpretable (a distance is a statement
about relative likelihood) AND is what licenses the downstream clustering and
Neyman-Pearson machinery.  This script collapses it into ONE panel of six
numbers per space, every one of which has ideal value 1 (or 0 for ECE), so a
reader can see alignment at a glance instead of triangulating ten tables.

THE IDEAL.  If class c is N(mu_c, I) in the latent, then
    log p(z | c) = -1/2 ||z - mu_c||^2 + const,           (const the same for all c)
so for two hypotheses c, c'
    Delta log L = -1/2 ( ||z-mu_c||^2 - ||z-mu_c'||^2 ),
i.e. the SQUARED-DISTANCE DIFFERENCE IS EXACTLY 2 * delta log-likelihood, with
no free scale.  Everything below measures departure from that identity.

THE PANEL (ideal in brackets)
  r_ll     [1] Pearson r between the distance proxy -1/2||z-mu_c||^2 and the true
               class-conditional log-density log N(z; mu_c, Sigma_c), over ALL
               (point, class) pairs -- the hypothesis-test geometry.
  slope    [1] OLS slope of true log-density on the proxy.  This is the sharp
               one: r can be high while a unit of 1/2 d^2 is NOT a unit of
               log-likelihood.  slope < 1 = distances over-stated,
               slope > 1 = under-stated.
  r_llr    [1] the same correlation for PAIRWISE differences (proxy LLR vs true
               LLR).  Per-class constants cancel here, so this isolates the
               hypothesis-test quantity from the per-class normalisations.
  ece      [0] expected calibration error of softmax_c(-1/2 d^2) read as a class
               posterior.  If distance is a true log-likelihood and the classes
               are balanced this softmax IS the Bayes posterior, so ECE is the
               operational cost of believing the distance.
  sw       [1] mean calibrated sliced-Wasserstein Gaussianity ratio of the
               labelled components (metrics.gaussianity_summary); 1 = as
               Gaussian as a finite sample of a true Gaussian.
  rms      [1] mean per-dimension RMS of the labelled components = the fitted
               sigma.  This is "how close is the component width to sigma = 1".
  Plus, for discrimination: sep = (closest centroid pair) / (mean rms), i.e. how
  many sigma apart the two nearest classes sit.  Interpretability without this
  is a space that is faithful and useless.

Reference density: per-class Gaussian with shrunk full covariance
(Sigma_c = (1-a) S_c + a * tr(S_c)/d * I, a = `--shrink`) so the "truth" is
estimable at the per-class sample sizes here.  With a -> 1 the reference becomes
isotropic and the panel degenerates to measuring the width alone; a = 0.1 is the
campaign default used elsewhere for Mahalanobis.

    python experiments/104_interpretability_panel.py --selftest
    python experiments/104_interpretability_panel.py --cells cars:dino,flowers:dino
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import numpy as np
import torch

from supersig.config import DEVICE
from supersig.metrics import gaussianity_summary


def _class_stats(z, y, classes, shrink=0.1):
    mus, covs = [], []
    d = z.shape[1]
    for c in classes:
        zc = z[y == c]
        mu = zc.mean(0)
        X = zc - mu
        S = X.T @ X / max(len(zc) - 1, 1)
        S = (1 - shrink) * S + shrink * (np.trace(S) / d) * np.eye(d)
        mus.append(mu); covs.append(S)
    return np.stack(mus), np.stack(covs)


def _log_gauss(z, mu, S):
    """log N(z; mu, S) for one class, vectorised over points."""
    d = z.shape[1]
    L = np.linalg.cholesky(S)
    diff = z - mu
    sol = np.linalg.solve(L, diff.T)                       # (d, n)
    quad = (sol ** 2).sum(0)
    logdet = 2.0 * np.log(np.diag(L)).sum()
    return -0.5 * (quad + logdet + d * np.log(2 * np.pi))


def _ece(prob, y_idx, n_bins=15):
    conf = prob.max(1)
    pred = prob.argmax(1)
    acc = (pred == y_idx).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def panel(z, y, classes=None, shrink=0.1, sw=True, seed=0):
    """The six-number interpretability panel for one space.

    z: (n, d) embeddings.  y: (n,) integer labels.  Uses only labelled points
    of `classes` (default: all present), which is the anchor set a hypothesis
    test would compare against.
    """
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y)
    classes = np.unique(y) if classes is None else np.asarray(classes)
    keep = np.isin(y, classes)
    z, y = z[keep], y[keep]
    mus, covs = _class_stats(z, y, classes, shrink)

    # (n, C) proxy and reference log-densities
    proxy = -0.5 * ((z[:, None, :] - mus[None]) ** 2).sum(-1)
    true = np.stack([_log_gauss(z, mus[i], covs[i]) for i in range(len(classes))], 1)

    p, t = proxy.ravel(), true.ravel()
    r_ll = float(np.corrcoef(p, t)[0, 1])
    slope = float(np.polyfit(p, t, 1)[0])

    # pairwise LLRs: per-class constants cancel
    iu = np.triu_indices(len(classes), 1)
    dp = (proxy[:, iu[0]] - proxy[:, iu[1]]).ravel()
    dt = (true[:, iu[0]] - true[:, iu[1]]).ravel()
    r_llr = float(np.corrcoef(dp, dt)[0, 1])

    # distance posterior calibration
    m = proxy - proxy.max(1, keepdims=True)
    prob = np.exp(m); prob /= prob.sum(1, keepdims=True)
    y_idx = np.searchsorted(classes, y)
    ece = _ece(prob, y_idx)

    out = dict(r_ll=r_ll, slope=slope, r_llr=r_llr, ece=ece)
    if sw:
        g = gaussianity_summary(z.astype(np.float32), y, list(classes),
                                n_null=8, seed=seed)
        out.update(sw=g["sw_ratio_mean"], rms=g["rms_mean"],
                   cond=g["eig_cond_max"], sep=g["separation"])
    return out


def _selftest():
    """The panel must return the ideal values on a space that IS the ideal, and
    must move in the right direction under each named departure."""
    rng = np.random.default_rng(0)
    C, d, n = 6, 16, 400
    mus = 6.0 * np.eye(C, d)
    y = np.repeat(np.arange(C), n)

    def make(scale=1.0, aniso=1.0, heavy=False):
        zs = []
        for c in range(C):
            e = rng.standard_normal((n, d))
            if heavy:
                e = rng.standard_t(3, size=(n, d)) / np.sqrt(3.0)
            e[:, 0] *= aniso
            zs.append(mus[c] + scale * e)
        return np.concatenate(zs)

    print(f"{'case':22s}{'r_ll':>8s}{'slope':>8s}{'r_llr':>8s}{'ece':>8s}"
          f"{'sw':>7s}{'rms':>7s}{'sep':>7s}")
    for name, kw in [("ideal N(mu,I)", {}), ("width 2x", dict(scale=2.0)),
                     ("width 0.5x", dict(scale=0.5)),
                     ("anisotropic 4x", dict(aniso=4.0)),
                     ("heavy tails t3", dict(heavy=True))]:
        z = make(**kw)
        p = panel(z, y)
        print(f"{name:22s}{p['r_ll']:8.3f}{p['slope']:8.3f}{p['r_llr']:8.3f}"
              f"{p['ece']:8.3f}{p['sw']:7.2f}{p['rms']:7.2f}{p['sep']:7.2f}")
    print("\nexpected: ideal -> r~1, slope~1, ece~0, sw~1, rms~1;"
          " width 2x -> slope~0.25 (1/scale^2), rms~2;"
          " width 0.5x -> slope~4, rms~0.5; anisotropy -> slope<1 and cond high;"
          " heavy tails -> sw >> 1 with slope still ~1.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--shrink", type=float, default=0.1)
    ap.add_argument("--cells", default="")
    ap.add_argument("--out", default="logs/exp104")
    args = ap.parse_args()

    if args.selftest:
        _selftest(); return

    # Space loading mirrors exp 77/80 (cached banks + heads); left to the
    # caller's environment because it needs the feature caches.
    import importlib
    exp77 = importlib.import_module("77_space_similarity")
    os.makedirs(args.out, exist_ok=True)
    rows = {}
    for cell in args.cells.split(","):
        ds, base = cell.split(":")
        for name, (z_tr, y_tr, _, _) in exp77.load_cell_spaces(ds, base).items():
            rows[f"{ds}_{base}_{name}"] = panel(z_tr, y_tr, shrink=args.shrink)
            print(cell, name, rows[f"{ds}_{base}_{name}"])
    np.savez(os.path.join(args.out, "panel.npz"),
             **{k: np.array(list(v.values())) for k, v in rows.items()},
             fields=np.array(list(next(iter(rows.values())).keys())))


if __name__ == "__main__":
    main()
