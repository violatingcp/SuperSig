"""Exp 136 guards: holdout tags agree across 67/68/136 and are empty at the
archived holdout, so draws cannot overwrite the archived scratch lineage."""
import importlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
REPO = os.path.join(os.path.dirname(__file__), "..")


def test_holdout_tag_is_empty_at_archived_holdout_and_set_elsewhere():
    m = importlib.import_module("136_scratch_master")
    assert m.htag(4) == ""
    assert m.htag(8) == "_h8" and m.htag(43) == "_h43"


def test_67_and_68_use_the_same_tag_rule():
    for f in ("67_scratch_pretrain.py", "68_scratch_discovery.py"):
        src = open(os.path.join(REPO, "experiments", f)).read()
        assert 'htag = "" if args.holdout == 4 else f"_h{args.holdout}"' in src, f


def test_exp68_saves_purity_and_post_geometry():
    src = open(os.path.join(REPO, "experiments", "68_scratch_discovery.py")).read()
    assert 'f"purity_{n}"' in src and 'f"post_{k}_{n}"' in src
