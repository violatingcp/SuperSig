"""Tests for supersig.holdouts -- the single/multi holdout regime switch.

The load-bearing property is BACKWARD COMPATIBILITY: with $SUPERSIG_NH unset,
every holdout set and every artifact filename must be byte-identical to what
the campaign produced before the switch existed.  Roughly 600 archived power
series and a directory of expensive fine-tune checkpoints depend on it.

The second property is NON-COLLISION: a single-holdout run must never write to
a path a multi-holdout run wrote, because the two regimes reach different
conclusions (exps 89/109) and must not be pooled or silently overwritten.
"""
import contextlib
import os
import re
import glob

import pytest

from supersig.holdouts import (n_holdout, holdout_set, seen_classes, run_tag,
                               regime, describe, ENV_VAR, DRAW_VAR)

# (dataset, n_classes, the holdout set the campaign used before this module)
CAMPAIGN = [
    ("dtd", 47, set(range(37, 47))),
    ("flowers", 102, set(range(92, 102))),
    ("cars", 196, set(range(186, 196))),
    ("aircraft", 100, set(range(90, 100))),
    ("cifar100", 100, set(range(90, 100))),
    ("galaxy10", 10, {9}),
]


@contextlib.contextmanager
def env(value):
    old = os.environ.get(ENV_VAR)
    if value is None:
        os.environ.pop(ENV_VAR, None)
    else:
        os.environ[ENV_VAR] = str(value)
    try:
        yield
    finally:
        os.environ.pop(ENV_VAR, None)
        if old is not None:
            os.environ[ENV_VAR] = old


@pytest.mark.parametrize("ds,n_cls,expected", CAMPAIGN)
def test_default_reproduces_campaign_holdouts(ds, n_cls, expected):
    with env(None):
        assert holdout_set(ds, n_cls) == expected


def test_default_run_tag_is_empty_so_filenames_are_unchanged():
    with env(None):
        assert run_tag() == ""


@pytest.mark.parametrize("ds,n_cls,_", CAMPAIGN)
def test_env_switch_makes_every_dataset_single_holdout(ds, n_cls, _):
    with env(1):
        assert holdout_set(ds, n_cls) == {n_cls - 1}
        assert regime(ds, n_cls) == "single"
        assert run_tag() == "_h1"


def test_seen_and_holdout_partition_the_classes():
    for ds, n_cls, _ in CAMPAIGN:
        for nh in (None, 1, 5):
            with env(nh):
                h = holdout_set(ds, n_cls)
                s = seen_classes(ds, n_cls)
                assert set(s) | h == set(range(n_cls))
                assert not (set(s) & h)
                assert len(s) + len(h) == n_cls


def test_explicit_argument_overrides_environment():
    with env(1):
        assert holdout_set("dtd", 47, nh=10) == set(range(37, 47))
        assert run_tag(10) == "_h10"


def test_tags_are_distinct_across_regimes():
    """The non-collision guarantee: no two regimes share an artifact name."""
    tags = []
    for v in (None, 1, 5, 10):
        with env(v):
            tags.append(run_tag())
    assert len(set(tags)) == len(tags), tags


def test_never_holds_out_every_class():
    with env(50):
        h = holdout_set("galaxy10", 10)
        assert len(h) == 9 and seen_classes("galaxy10", 10) == [0]


@pytest.mark.parametrize("bad", ["0", "-3", "abc", "1.5"])
def test_invalid_env_raises_rather_than_defaulting_silently(bad):
    with env(bad):
        with pytest.raises(ValueError):
            n_holdout("dtd")


def test_regime_labels_match_the_paper_split():
    with env(None):
        # galaxy10 is the one cell that was already single-holdout
        assert regime("galaxy10", 10) == "single"
        assert regime("dtd", 47) == "multi"
    with env(5):
        assert regime("cifar100", 100) == "multi"


def test_describe_mentions_the_regime():
    with env(1):
        d = describe("dtd", 47)
        assert "single" in d and "_h1" in d


def test_no_experiment_still_hardcodes_the_holdout_rule():
    """The rule lived in 23 places; it must live in exactly one now."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    offenders = []
    pat = re.compile(r'1 if (ds|DS) == "galaxy10" else 10')
    for fn in glob.glob(os.path.join(root, "experiments", "*.py")):
        if pat.search(open(fn).read()):
            offenders.append(os.path.basename(fn))
    assert not offenders, f"hardcoded holdout rule still in: {offenders}"


def test_experiments_using_the_helper_also_import_it():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bad = []
    for fn in glob.glob(os.path.join(root, "experiments", "*.py")):
        s = open(fn).read()
        uses = re.search(r"\b(run_tag|n_holdout)\(", s)
        if uses and "from supersig.holdouts import" not in s:
            bad.append(os.path.basename(fn))
    assert not bad, f"use helper without importing it: {bad}"


# ---------------------------------------------------------------- draws


@contextlib.contextmanager
def draw_env(value):
    old = os.environ.get(DRAW_VAR)
    if value is None:
        os.environ.pop(DRAW_VAR, None)
    else:
        os.environ[DRAW_VAR] = str(value)
    try:
        yield
    finally:
        os.environ.pop(DRAW_VAR, None)
        if old is not None:
            os.environ[DRAW_VAR] = old


def test_draw_is_reproducible_across_processes():
    """The bug this pins: hash() randomizes strings per process, so a draw
    seeded with hash() silently picks a different class on every run."""
    import subprocess, sys, os as _os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code = ("import sys; sys.path.insert(0, %r);"
            "from supersig.holdouts import holdout_set;"
            "print(sorted(holdout_set('cifar100', 100, nh=1, draw=3)))" % root)
    outs = set()
    for hs in ("0", "1", "12345"):
        e = dict(_os.environ, PYTHONHASHSEED=hs)
        outs.add(subprocess.run([sys.executable, "-c", code], env=e,
                                capture_output=True, text=True).stdout.strip())
    assert len(outs) == 1, f"draw not reproducible across PYTHONHASHSEED: {outs}"


def test_draws_differ_from_each_other_and_from_default():
    with env(1):
        default = holdout_set("cifar100", 100)
        drawn = [frozenset(holdout_set("cifar100", 100, draw=d)) for d in range(8)]
        assert len(set(drawn)) > 1, "all draws identical"
        assert any(set(x) != default for x in drawn)


def test_draw_respects_holdout_count_and_bounds():
    for nh in (1, 5, 10):
        with env(nh):
            for d in range(5):
                h = holdout_set("cifar100", 100, draw=d)
                assert len(h) == nh
                assert all(0 <= c < 100 for c in h)


def test_draw_tag_is_distinct_so_draws_cannot_overwrite():
    tags = set()
    with env(1):
        tags.add(run_tag())
        for d in range(5):
            tags.add(run_tag(draw=d))
    assert len(tags) == 6, tags


def test_draw_env_var_matches_explicit_argument():
    with env(1):
        explicit = holdout_set("dtd", 47, draw=2)
        with draw_env(2):
            assert holdout_set("dtd", 47) == explicit


def test_default_unaffected_by_draw_var_being_unset():
    with env(None), draw_env(None):
        assert holdout_set("dtd", 47) == set(range(37, 47))
        assert run_tag() == ""


def test_draw_is_paired_across_arms_within_a_dataset():
    """Draw d must give the same holdout for every arm/base, or comparisons
    stop being paired.  It depends only on (ds, n_cls, k, d)."""
    with env(1):
        a = holdout_set("flowers", 102, draw=4)
        b = holdout_set("flowers", 102, draw=4)
        assert a == b


@pytest.mark.parametrize("bad", ["abc", "1.5"])
def test_invalid_draw_env_raises(bad):
    with env(1), draw_env(bad):
        with pytest.raises(ValueError):
            holdout_set("dtd", 47)
