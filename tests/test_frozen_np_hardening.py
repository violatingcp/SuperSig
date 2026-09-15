"""Tests for the exp-139 hardening statistics.

Two things this pins, both of which produced a wrong number during development:

1. The variance decomposition must run WITHIN (cell, arm) strata. Grouping by
   draw alone makes the "within-draw" replicates different arms and bases, so
   the number reported as seed variance is arm/base variance -- larger, and
   answering a different question.
2. With one seed per cell the seed term is UNMEASURED and must be reported as
   such, never silently substituted.
"""
import importlib.util
import os

import numpy as np
import pytest

_P = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  "experiments", "139_frozen_np_hardening.py")
_s = importlib.util.spec_from_file_location("exp139", _P)
exp139 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(exp139)


def _rows(seeds, draws=(0, 3, 5), cells=("galaxy10:dino", "galaxy10:visreg"),
          arms=("supcon-ft", "ss-ft"), base=0.5, draw_sd=0.15, seed_sd=0.02,
          arm_offset=0.10, rng=None):
    rng = rng or np.random.default_rng(0)
    dcent = {d: base + rng.normal(0, draw_sd) for d in draws}
    out = []
    for ci, c in enumerate(cells):
        for ai, a in enumerate(arms):
            for d in draws:
                for s in seeds:
                    out.append(dict(
                        cell=c, arm=a, draw=d, seed=s,
                        r1=dcent[d] + ai * arm_offset + rng.normal(0, seed_sd),
                        r2=dcent[d] + ai * arm_offset + rng.normal(0, seed_sd),
                        margin=0.8, file="x"))
    return out


def test_selftest_runs():
    exp139._selftest()


def test_seed_term_is_reported_unmeasured_with_one_seed():
    res = exp139.analyse(_rows(seeds=(0,)))
    v = res["r1_variance"]
    assert v["seed_term_measured"] is False
    assert v["sd_within_seed"] is None
    assert v["max_seeds_per_cell"] == 1


def test_seed_term_is_measured_with_several_seeds():
    res = exp139.analyse(_rows(seeds=(0, 1, 2)))
    v = res["r1_variance"]
    assert v["seed_term_measured"] is True
    assert v["max_seeds_per_cell"] == 3
    assert v["sd_within_seed"] > 0


def test_stratification_keeps_arm_offset_out_of_the_seed_term():
    """THE regression test. A large constant arm offset must not inflate the
    seed term, because the decomposition runs inside each (cell, arm)."""
    res = exp139.analyse(_rows(seeds=(0, 1, 2), seed_sd=0.01, arm_offset=0.30))
    v = res["r1_variance"]
    assert v["sd_within_seed"] < 0.05, v["sd_within_seed"]


def test_draw_dominance_is_detected():
    big_draw = exp139.analyse(_rows(seeds=(0, 1, 2), draw_sd=0.20, seed_sd=0.01))
    big_seed = exp139.analyse(_rows(seeds=(0, 1, 2), draw_sd=0.01, seed_sd=0.20))
    assert big_draw["r1_variance"]["draw_dominates"] is True
    assert big_seed["r1_variance"]["draw_dominates"] is False


def test_gate_report_and_clopper_pearson():
    g = exp139.gate_report([0.4] * 45)
    assert g["k"] == 45 and g["n"] == 45
    assert 0.90 < g["lo"] < 0.94          # 45/45 is NOT 1.0
    g2 = exp139.gate_report([0.4] * 44 + [0.05])
    assert g2["k"] == 44 and g2["lo"] < g["lo"]


def test_variance_decomposition_handles_archived_none_draw():
    """Archived files carry no _dN suffix, so their draw key is None; mixing
    them with integer draws must not raise."""
    v = exp139.variance_decomposition({None: [0.5, 0.52], 0: [0.44], 3: [0.55]})
    assert v is not None and v["n_draws"] == 3 and v["n_cells"] == 4


def test_variance_decomposition_returns_none_when_empty():
    assert exp139.variance_decomposition({}) is None
    assert exp139.variance_decomposition({0: []}) is None


def test_plan_is_the_full_grid_and_pins_the_frozen_np_arm():
    lines = exp139.plan(["galaxy10:dino"], (0, 3), (0, 1, 2))
    assert len(lines) == 6
    for ln in lines:
        assert "--scorers np" in ln            # arm A of exp 135
        assert "SUPERSIG_NH=1" in ln           # single-holdout regime
        assert "SUPERSIG_HOLDOUT_DRAW=" in ln


def test_paired_delta_only_uses_shared_cells():
    a = {"x": 0.5, "y": 0.6, "z": 0.7}
    b = {"x": 0.4, "y": 0.5}
    p = exp139.paired_delta(a, b)
    assert p["n"] == 2 and p["wins"] == 2
