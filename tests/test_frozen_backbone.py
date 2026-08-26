"""A frozen backbone must produce bit-identical embeddings.

THE BUG THIS PINS.  `requires_grad_(False)` stops the weights updating but does
NOT stop BatchNorm running statistics from updating -- `.train()` mode keeps
accumulating them, and `collect_embeddings` (which calls `.eval()`) then reads
the drifted statistics.  So a "frozen" backbone with BatchNorm silently gave
DIFFERENT embeddings on every discovery round.

This is not hypothetical.  The CIFAR ResNet-20 trunk has 21 BatchNorm2d layers
with track_running_stats=True, so the frozen-space discovery runs (exps 86,
92b, 109) were really "weights frozen, BN still adapting" on CIFAR.  The
transfer trunk is a ViT (LayerNorm, no running stats), which is why the frozen
transfer results were exact while the CIFAR one was not.

The frozen recipe's whole claim -- that purity climbs across rounds while the
representation stays put, so the gain lives in the anchors -- depends on the
representation actually staying put.
"""
import copy

import pytest
import torch
import torch.nn as nn

from supersig.train import backbone_is_frozen, set_train_mode


class BNNet(nn.Module):
    """Minimal stand-in for the CIFAR trunk: has BatchNorm with running stats."""

    def __init__(self, d_in=8, d_out=4):
        super().__init__()
        self.fc1 = nn.Linear(d_in, 16)
        self.bn = nn.BatchNorm1d(16)
        self.fc2 = nn.Linear(16, d_out)

    def forward(self, x):
        return self.fc2(torch.relu(self.bn(self.fc1(x))))


class LNNet(nn.Module):
    """Stand-in for the transfer trunk: LayerNorm, no running statistics."""

    def __init__(self, d_in=8, d_out=4):
        super().__init__()
        self.fc1 = nn.Linear(d_in, 16)
        self.ln = nn.LayerNorm(16)
        self.fc2 = nn.Linear(16, d_out)

    def forward(self, x):
        return self.fc2(torch.relu(self.ln(self.fc1(x))))


def _freeze(m):
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def _embed(m, x):
    """What collect_embeddings does: eval mode, no grad."""
    m.eval()
    with torch.no_grad():
        return m(x).clone()


def _simulate_round(m, batches):
    """One discovery round's worth of forward passes in the training mode the
    trainer would select."""
    set_train_mode(m)
    with torch.no_grad():
        for b in batches:
            m(b)


@pytest.fixture
def data():
    g = torch.Generator().manual_seed(0)
    probe = torch.randn(32, 8, generator=g)
    # deliberately shifted/scaled batches, so BN stats would move a lot
    batches = [3.0 + 2.0 * torch.randn(16, 8, generator=g) for _ in range(5)]
    return probe, batches


def test_frozen_backbone_is_detected():
    assert backbone_is_frozen(_freeze(BNNet()))
    assert not backbone_is_frozen(BNNet())


def test_set_train_mode_puts_frozen_backbone_in_eval():
    m = _freeze(BNNet())
    set_train_mode(m)
    assert not m.training, "frozen backbone must be in eval mode"


def test_set_train_mode_leaves_unfrozen_backbone_in_train():
    m = BNNet()
    set_train_mode(m)
    assert m.training, "unfrozen training must be unaffected"


def test_frozen_bn_backbone_gives_bit_identical_embeddings(data):
    """THE regression test: the frozen recipe's core claim."""
    probe, batches = data
    m = _freeze(BNNet())
    before = _embed(m, probe)
    for _ in range(3):                       # three discovery rounds
        _simulate_round(m, batches)
    after = _embed(m, probe)
    assert torch.equal(before, after), (
        "frozen backbone embeddings drifted; max delta "
        f"{(before - after).abs().max().item():.3e}")


def test_the_old_behaviour_really_did_drift(data):
    """Guards the test itself: if plain .train() did not drift, this test
    would pass vacuously and prove nothing."""
    probe, batches = data
    m = _freeze(BNNet())
    before = _embed(m, probe)
    for _ in range(3):
        m.train()                            # the pre-fix call
        with torch.no_grad():
            for b in batches:
                m(b)
    after = _embed(m, probe)
    assert not torch.equal(before, after), "BN stats did not move; test is vacuous"
    assert (before - after).abs().max().item() > 1e-3


def test_running_stats_are_untouched_when_frozen(data):
    probe, batches = data
    m = _freeze(BNNet())
    mean0 = m.bn.running_mean.clone()
    var0 = m.bn.running_var.clone()
    n0 = m.bn.num_batches_tracked.clone()
    for _ in range(3):
        _simulate_round(m, batches)
    assert torch.equal(m.bn.running_mean, mean0)
    assert torch.equal(m.bn.running_var, var0)
    assert torch.equal(m.bn.num_batches_tracked, n0)


def test_layernorm_trunk_was_already_safe(data):
    """The transfer path had no bug -- it must also keep working."""
    probe, batches = data
    m = _freeze(LNNet())
    before = _embed(m, probe)
    for _ in range(3):
        _simulate_round(m, batches)
    assert torch.equal(before, _embed(m, probe))


def test_weights_are_unchanged_when_frozen(data):
    probe, batches = data
    m = _freeze(BNNet())
    ref = copy.deepcopy(m.state_dict())
    for _ in range(3):
        _simulate_round(m, batches)
    for k, v in m.state_dict().items():
        assert torch.equal(v, ref[k]), f"{k} changed under freeze"


def test_real_cifar_trunk_has_batchnorm_and_transfer_does_not():
    """Documents WHY this matters, and fails loudly if a trunk swaps norm type
    (which would silently change whether freezing is exact)."""
    from supersig.models import CIFARResNetBackbone
    m = CIFARResNetBackbone(32, arch="resnet20", pretrain=None)
    bns = [mod for mod in m.modules() if isinstance(mod, nn.BatchNorm2d)]
    assert len(bns) > 0, "CIFAR trunk unexpectedly has no BatchNorm"
    assert all(b.track_running_stats for b in bns)
    # and the fix covers it
    _freeze(m)
    set_train_mode(m)
    assert not m.training
