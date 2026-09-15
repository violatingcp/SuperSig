"""Exp 137 / exp 59 artifact tags: draws and seeds never overwrite archives."""
import importlib, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
REPO = os.path.join(os.path.dirname(__file__), "..")


def test_137_holdout_tag_matches_67():
    m = importlib.import_module("137_scratch_residuals")
    assert m.htag(4) == "" and m.htag(43) == "_h43"


def test_59_seed_tag_present():
    src = open(os.path.join(REPO, "experiments", "59_nplm_residual_concat.py")).read()
    assert 'dtag = ds + (f"_s{args.seed}" if args.seed else "")' in src
