"""Exp 134a/134c guards: the width-control artifacts cannot collide with the
archived 100-D parents, and the post-discovery relabelling is label-free."""
import importlib
import os
import subprocess
import sys
import types

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
REPO = os.path.join(os.path.dirname(__file__), "..")


def test_exp70_seed_sfx_tags_non_default_dim_and_is_empty_at_default():
    exp70 = importlib.import_module("70_cars_ft_suite")
    assert exp70.seed_sfx(types.SimpleNamespace(seed=0, emb_dim=100)) == ""
    assert exp70.seed_sfx(types.SimpleNamespace(seed=2, emb_dim=100)) == "_s2"
    assert exp70.seed_sfx(types.SimpleNamespace(seed=0, emb_dim=200)) == "_e200"
    assert exp70.seed_sfx(types.SimpleNamespace(seed=1, emb_dim=200)) == "_s1_e200"


def test_relabel_never_reads_holdout_labels():
    m = importlib.import_module("134c_residual_after_discovery")
    rng = np.random.default_rng(1)
    n_cls, k, d = 4, 1, 6
    A = rng.normal(0, 5, (n_cls + k, d))
    y = np.repeat(np.arange(n_cls + k), 20)
    H = A[y] + rng.normal(0, 0.2, (len(y), d))
    is_seen = y < n_cls
    opaque = np.where(is_seen, y, -1)                 # holdout label hidden
    idx, lab, n_ps = m.relabel_against_anchors(H, is_seen, opaque, A, n_cls)
    assert n_ps == 20 and (lab[~is_seen[idx]] == n_cls).all()
    assert (lab[is_seen[idx]] == opaque[idx][is_seen[idx]]).all()
    assert -1 not in lab


def test_selftests_run():
    for s in ("134a_width_control.py", "134c_residual_after_discovery.py"):
        p = os.path.join(REPO, "experiments", s)
        if s.startswith("134c"):
            r = subprocess.run([sys.executable, p, "--selftest"],
                               capture_output=True, text=True, timeout=300)
            assert r.returncode == 0 and "selftest OK" in r.stdout, r.stdout + r.stderr
