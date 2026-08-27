"""The explicit BN-adaptation opt-in (exp 133) and the draw-tagged loader
(exp 77) -- both pinned so the 2026-08-26 freeze fix cannot silently regress
and the 2026-08-27 draw-blind loader fix cannot come back."""
import importlib
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
from supersig.train import set_train_mode, train_sigreg_hybrid  # noqa: E402


def _frozen_bn_net():
    net = nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4), nn.ReLU(),
                        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 2))
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def test_default_keeps_frozen_trunk_in_eval():
    net = _frozen_bn_net()
    set_train_mode(net)
    assert not net.training
    m0 = net[1].running_mean.clone()
    net(torch.randn(8, 3, 8, 8) * 5 + 3)
    assert torch.equal(net[1].running_mean, m0)


def test_bn_adapt_keeps_frozen_trunk_in_train_and_moves_stats():
    net = _frozen_bn_net()
    set_train_mode(net, bn_adapt=True)
    assert net.training
    m0 = net[1].running_mean.clone()
    net(torch.randn(8, 3, 8, 8) * 5 + 3)
    assert not torch.equal(net[1].running_mean, m0)


def test_bn_adapt_is_a_no_op_on_an_unfrozen_trunk():
    net = _frozen_bn_net()
    for p in net.parameters():
        p.requires_grad_(True)
    set_train_mode(net, bn_adapt=False)
    assert net.training
    set_train_mode(net, bn_adapt=True)
    assert net.training


def test_train_sigreg_hybrid_accepts_bn_adapt_and_defaults_off():
    import inspect
    sig = inspect.signature(train_sigreg_hybrid)
    assert "bn_adapt" in sig.parameters
    assert sig.parameters["bn_adapt"].default is False
    from supersig.discovery import run_discovery
    assert inspect.signature(run_discovery).parameters["bn_adapt"].default is False
    exp92 = importlib.import_module("92_sparker_discovery")
    assert inspect.signature(exp92.sparker_discovery).parameters["bn_adapt"] \
        .default is False


def test_head_emb_honours_the_holdout_draw_tag(monkeypatch):
    """exp77.head_emb must ask for the _h1_dN artifacts when the env is set.
    Until 2026-08-27 it ignored the tag, so every consumer (exps 80/100/102/
    103/111/131) scored the alphabetical-holdout spaces under any draw."""
    exp77 = importlib.import_module("77_space_similarity")
    asked = []

    def fake_exists(p):
        asked.append(os.path.basename(p))
        return False                         # -> head_emb returns None early
    monkeypatch.setattr(exp77.os.path, "exists", fake_exists)

    monkeypatch.delenv("SUPERSIG_NH", raising=False)
    monkeypatch.delenv("SUPERSIG_HOLDOUT_DRAW", raising=False)
    assert exp77.head_emb("galaxy10", "dino", "supcon-ft", 100, "train") is None
    assert "tf_feats_galaxy10_dino_ft70_supcon-ft.pt" in asked

    asked.clear()
    monkeypatch.setenv("SUPERSIG_NH", "1")
    monkeypatch.setenv("SUPERSIG_HOLDOUT_DRAW", "3")
    assert exp77.head_emb("galaxy10", "dino", "supcon-ft", 100, "train") is None
    assert any(a.endswith("_h1_d3.pt") for a in asked), asked
    assert all("_h1_d3" in a for a in asked), asked
