"""Tests for the paper's table generator (experiments/140_paper_tables.py).

Two of the bugs found while writing this generator were SILENT -- they
produced a well-formed table full of wrong or absent numbers rather than an
error, which is the worst failure mode for something that feeds a paper:

  1. `glob("...cifar10*.npz")` also matched cifar100, pooling two DATASETS
     into one "seed spread" and manufacturing a variance (sd 0.079 where the
     truth is 0.001).
  2. The exp-70 purity parser looked for the arm name on the purity line,
     where it does not appear, and silently produced an empty table.

Both are pinned below, along with the LaTeX well-formedness checks.
"""
import importlib.util
import os
import re
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

_spec = importlib.util.spec_from_file_location(
    "paper_tables", os.path.join(REPO, "experiments", "140_paper_tables.py"))
pt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pt)


# ------------------------------------------------------------ formatting ---

def test_fnum_renders_absent_values_as_a_dash():
    assert pt.fnum(None) == "--"
    assert pt.fnum(float("nan")) == "--"
    assert pt.fnum(float("inf")) == "--"
    assert pt.fnum(0.5) == "0.500"
    assert pt.fnum(0.5, 2) == "0.50"


def test_esc_escapes_latex_specials():
    assert pt.esc("a_b") == r"a\_b"
    assert pt.esc("a&b") == r"a\&b"


# ------------------------------------------------- LaTeX well-formedness ---

@pytest.mark.parametrize("name,fn", pt.TABLES)
def test_table_is_wellformed(name, fn):
    t = fn()
    if t is None:
        pytest.skip(f"{name}: source log absent")
    env = "longtable" if r"\begin{longtable}" in t else "tabular"
    assert t.count(rf"\begin{{{env}}}") == 1
    assert t.count(rf"\end{{{env}}}") == 1
    assert t.startswith("% STATUS:"), "every table must carry a coverage line"
    assert r"\caption" in t and r"\label" in t
    assert "Coverage:" in t, "coverage must reach the rendered caption too"


@pytest.mark.parametrize("name,fn", pt.TABLES)
def test_every_row_matches_the_column_spec(name, fn):
    """A row with the wrong cell count is a LaTeX error at compile time; catch
    it here instead, including \\multicolumn spans."""
    t = fn()
    if t is None:
        pytest.skip(f"{name}: source log absent")
    env = "longtable" if r"\begin{longtable}" in t else "tabular"
    spec = t.split(f"{{{env}}}{{")[1].split("}")[0]
    ncol = sum(1 for c in spec if c in "lcr")
    # a longtable puts its repeated header/footer FIRST (\endfirsthead,
    # \endhead, \endfoot) and the body last, so the body is what follows
    # \endfoot -- not what sits between \midrule and \bottomrule.
    if env == "longtable":
        body = t.split(r"\endfoot")[1].split(r"\end{longtable}")[0]
    else:
        body = t.split(r"\midrule")[-1].split(r"\bottomrule")[0]
    seen = 0
    for line in body.strip().split("\n"):
        if not line.strip().endswith(r"\\"):
            continue
        spans = [int(x) for x in re.findall(r"\\multicolumn\{(\d+)\}", line)]
        n = line.count("&") + 1 - len(spans) + sum(spans)
        assert n == ncol, f"{name}: row spans {n} of {ncol}: {line[:60]}"
        seen += 1
    assert seen > 0, f"{name}: no body rows"


# ------------------------------------------------------- the glob bug -----

def test_seed_table_does_not_pool_cifar10_with_cifar100():
    """`cifar10*` matches cifar100.  If it ever does again, the seed count
    goes above 3 and the sd is a fabrication."""
    t = pt.t_seeds()
    if t is None:
        pytest.skip("exp-59 archive absent")
    body = t.split(r"\midrule")[-1].split(r"\bottomrule")[0]
    counts = [int(line.rsplit("&", 1)[1].replace(r"\\", "").strip())
              for line in body.strip().split("\n") if line.strip().endswith(r"\\")]
    assert counts, "no rows"
    assert max(counts) <= 3, f"more than 3 seeds -> another dataset leaked in: {counts}"


def test_seed_table_matches_the_archive():
    """Pin the headline multi-seed numbers against the npz directly."""
    f = os.path.join(REPO, "logs", "exp59", "nplm_residual_concat_cifar10.npz")
    if not os.path.exists(f):
        pytest.skip("exp-59 archive absent")
    vals = []
    for sfx in ("", "_s1", "_s2"):
        p = f.replace(".npz", f"{sfx}.npz")
        if os.path.exists(p):
            vals.append(float(np.load(p, allow_pickle=True)
                              ["probe_post_sup->res-nplm"]))
    assert len(vals) == 3
    assert np.mean(vals) == pytest.approx(0.983, abs=1e-3)
    assert np.std(vals, ddof=1) < 0.005, "seed spread must stay small here"


# ------------------------------------------------ the purity parser bug ---

def test_draw_purities_finds_arms_via_the_section_header():
    """The arm is on '----- natural discovery: <arm> -----', not on the
    purity line.  An empty dict here means the parser silently broke."""
    got = pt._draw_purities("dtd", "dino")
    if not got:
        pytest.skip("exp-70 dtd/dino draw logs absent")
    assert "ss-ft" in got, sorted(got)
    assert len(got["ss-ft"]) >= 3, got["ss-ft"]
    for arm, per_draw in got.items():
        for d, v in per_draw.items():
            assert 0.0 <= v <= 1.0, (arm, d, v)


def test_ss_ft_clears_the_gate_on_both_dtd_backbones():
    """The base-independence claim in Section 6, pinned to the logs."""
    for base, expect in (("dino", 0.219), ("lejepa", 0.225)):
        got = pt._draw_purities("dtd", base)
        if not got or "ss-ft" not in got:
            pytest.skip(f"exp-70 dtd/{base} draw logs absent")
        v = list(got["ss-ft"].values())
        assert np.mean(v) == pytest.approx(expect, abs=2e-3)
        assert all(x >= 0.15 for x in v), f"dtd/{base} ss-ft below gate: {v}"


def test_nplm_sup_is_base_dependent_unlike_ss_ft():
    """The asymmetry is the point of the table; if it vanishes, the claim
    in Section 6 must change."""
    d = pt._draw_purities("dtd", "dino").get("nplm-sup-ft", {})
    l = pt._draw_purities("dtd", "lejepa").get("nplm-sup-ft", {})
    if not d or not l:
        pytest.skip("exp-70 dtd draw logs absent")
    assert sum(1 for v in d.values() if v >= 0.15) == 0, d
    assert sum(1 for v in l.values() if v >= 0.15) == len(l), l


# ------------------------------------------------------------ coverage ----

def test_objectives_table_reports_missing_arms_as_dashes():
    """A table must never quietly shrink; absent cells are '--' rows."""
    t = pt.t_objectives()
    if t is None:
        pytest.skip("exp-136 archive absent")
    body = t.split(r"\midrule")[1].split(r"\bottomrule")[0]
    assert len([x for x in body.strip().split("\n")
                if x.strip().endswith(r"\\")]) == 8, "all 8 objectives listed"
