"""Exp 144 GCD-protocol harness: split, Hungarian ACC and ss-k-means selftest."""
import importlib, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))


def test_gcd_selftest_runs():
    m = importlib.import_module("144_gcd_benchmark")
    m._selftest()


def test_known_split_convention():
    m = importlib.import_module("144_gcd_benchmark")
    assert m.KNOWN == {"cifar10": 5, "cifar100": 80, "cars": 98, "aircraft": 50}
    assert m.TOTAL == {"cifar10": 10, "cifar100": 100, "cars": 196, "aircraft": 100}
