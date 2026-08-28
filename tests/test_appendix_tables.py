"""Exp 141 appendix tables: every emitter builds and its rows match its colspec."""
import importlib, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))


def test_appendix_tables_selftest():
    m = importlib.import_module("141_appendix_tables")
    assert m.selftest() == 0


def test_grid_tables_selftest():
    m = importlib.import_module("142_grid_tables")
    assert m.selftest() == 0
