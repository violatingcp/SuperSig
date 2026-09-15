"""Shared setup for the SuperSig unit tests.

Puts the repo root and experiments/ on sys.path (experiment modules have
numeric names, so tests import them via importlib.import_module).  All
tests run on small synthetic data — no dataset downloads, no checkpoints,
seconds per file even on CPU.

    /home/pharris/venv/bin/python -m pytest tests/ -q
"""
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))


@pytest.fixture(autouse=True)
def _seed_everything():
    torch.manual_seed(0)
    np.random.seed(0)


def make_clusters(centers, n_per, dim, noise=1.0, seed=0):
    """(embs, labels) — one unit-ish Gaussian blob per row of `centers`."""
    g = np.random.default_rng(seed)
    embs, labs = [], []
    for c, mu in enumerate(centers):
        pad = np.zeros(dim)
        pad[: len(mu)] = np.asarray(mu, dtype=np.float64)
        embs.append(pad + noise * g.standard_normal((n_per, dim)))
        labs.append(np.full(n_per, c))
    return (np.concatenate(embs).astype(np.float32),
            np.concatenate(labs).astype(np.int64))
