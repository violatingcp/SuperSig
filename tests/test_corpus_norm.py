"""Exp 135's CorpusNorm: parameter-free, so the frozen/bn_adapt switch of exp
133 governs it exactly; statistics move only under bn_adapt=True."""
import importlib
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
from supersig.train import backbone_is_frozen, set_train_mode  # noqa: E402

m135 = importlib.import_module("135_corpus_norm_everywhere")


def _net():
    head = nn.Linear(6, 2)
    for p in head.parameters():
        p.requires_grad_(False)
    X = torch.randn(40, 6) * 2 + 3
    return m135.CorpusNorm(head, dim=6).init_stats(X), X


def test_corpus_norm_is_parameter_free_and_frozen():
    net, _ = _net()
    assert backbone_is_frozen(net)
    assert sum(p.numel() for p in net.norm.parameters()) == 0


def test_init_stats_match_seen_features():
    net, X = _net()
    assert torch.allclose(net.norm.running_mean, X.mean(0), atol=1e-4)


def test_stats_fixed_when_frozen_and_move_when_adapting():
    net, X = _net()
    mu = net.norm.running_mean.clone()
    set_train_mode(net)
    net(torch.randn(16, 6) * 7 - 5)
    assert torch.equal(net.norm.running_mean, mu)
    set_train_mode(net, bn_adapt=True)
    net(torch.randn(16, 6) * 7 - 5)
    assert not torch.equal(net.norm.running_mean, mu)


def test_head_weights_never_change():
    net, X = _net()
    w = net.head.weight.clone()
    set_train_mode(net, bn_adapt=True)
    net(X)
    assert torch.equal(net.head.weight, w)
