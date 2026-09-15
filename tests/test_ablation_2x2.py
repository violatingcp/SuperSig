"""Tests for the discovery x construction 2x2 (supersig/ablation2x2.py, exp 134d).

The interaction is a difference of DIFFERENCES, which is exactly the kind of
quantity a sign slip hides in: every one of A/B/C/D can look sane while the
contrast is wrong.  So the algebra is pinned as an identity over random
inputs, not on one worked example.

The last group pins the paper's actual claim against the JSONs on disk, so
that a re-run which moves the dtd/dino interaction fails here rather than
silently changing a headline number.
"""
import glob
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from supersig.ablation2x2 import (ALPHA, METRICS, archived_rows, cell_2x2,
                                  cell_name, npz_key, r1_purity)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG134 = os.path.join(REPO, "logs", "exp134")


def make_cell(A, B, D, purity=0.3, parent="p", obj="res", kind="residual",
              metric="perevt"):
    return {"results": {
        "parent (pre)": {metric: A},
        "parent (post-discovery)": {metric: B},
        f"{parent}->{obj} (post-discovery) {kind}": {metric: D}},
        "discovery": [{"purity": purity}]}


def one(A, B, C, D, **kw):
    metric = kw.get("metric", "perevt")
    j = make_cell(A, B, D, **kw)
    pre = {f"{kw.get('parent', 'p')}->{kw.get('obj', 'res')} "
           f"({kw.get('kind', 'residual')})": {metric: C}}
    rows = cell_2x2(j, pre, parent=kw.get("parent", "p"))
    return [r for r in rows if r["metric"] == metric][0]


# --------------------------------------------------------------- algebra ---

@pytest.mark.parametrize("seed", range(20))
def test_total_decomposes_exactly(seed):
    """total must equal disc + ctor + interaction for ANY A/B/C/D."""
    rng = np.random.default_rng(seed)
    A, B, C, D = rng.uniform(0, 1, 4)
    r = one(A, B, C, D)
    assert r["total"] == pytest.approx(
        r["main_disc"] + r["main_ctor"] + r["interaction"], abs=1e-12)


@pytest.mark.parametrize("seed", range(20))
def test_interaction_is_difference_of_differences(seed):
    rng = np.random.default_rng(seed)
    A, B, C, D = rng.uniform(-1, 1, 4)
    r = one(A, B, C, D)
    assert r["interaction"] == pytest.approx((D - C) - (B - A), abs=1e-12)
    assert r["main_disc"] == pytest.approx(B - A, abs=1e-12)
    assert r["main_ctor"] == pytest.approx(C - A, abs=1e-12)
    assert r["total"] == pytest.approx(D - A, abs=1e-12)


def test_additive_stack_has_zero_interaction():
    """If the construction adds the same amount with and without discovery,
    the interaction is exactly 0 -- this is the null the claim is against."""
    A, B, gain = 0.10, 0.15, 0.12
    r = one(A, B, A + gain, B + gain)
    assert r["interaction"] == pytest.approx(0.0, abs=1e-12)
    assert r["total"] == pytest.approx(0.17, abs=1e-12)


def test_interaction_ignores_parent_shift():
    """Adding a constant to BOTH parent cells leaves the interaction alone
    (it is a contrast, not a level) -- while the main effect of the
    construction, which is measured against A, does move."""
    base = one(0.04, 0.068, 0.0225, 0.253)
    for shift in (0.1, -0.05, 1.0):
        r = one(0.04 + shift, 0.068 + shift, 0.0225, 0.253)
        assert r["interaction"] == pytest.approx(base["interaction"], abs=1e-12)
        assert r["main_disc"] == pytest.approx(base["main_disc"], abs=1e-12)
        assert r["main_ctor"] == pytest.approx(base["main_ctor"] - shift, abs=1e-12)


def test_superadditive_and_subadditive_signs():
    assert one(0.04, 0.068, 0.0225, 0.253)["interaction"] > 0     # dtd/dino res
    assert one(0.04, 0.068, 0.113, 0.087)["interaction"] < 0      # dtd/dino res-nplm


# ------------------------------------------------------- missing handling ---

def test_missing_pre_child_is_retained_not_dropped():
    j = make_cell(0.04, 0.068, 0.253)
    rows = cell_2x2(j, {"p->res (residual)": {}}, parent="p")
    assert len(rows) == len(METRICS), "every metric must still yield a row"
    r = [x for x in rows if x["metric"] == "perevt"][0]
    assert r["missing"] is True
    assert "interaction" not in r and "total" not in r
    assert r["A"] == 0.04 and r["D"] == 0.253 and r["C"] is None


def test_missing_parent_rows_yield_no_output():
    j = {"results": {"p->res (post-discovery) residual": {"perevt": 0.2}},
         "discovery": []}
    assert cell_2x2(j, {"p->res (residual)": {"perevt": 0.1}}, parent="p") == []


def test_parent_post_row_is_not_treated_as_a_construction():
    j = make_cell(0.04, 0.068, 0.253)
    rows = cell_2x2(j, {"p->res (residual)": {"perevt": 0.0225}}, parent="p")
    assert {r["obj"] for r in rows} == {"res"}
    assert all(r["kind"] == "residual" for r in rows)


def test_both_kinds_and_objs_are_parsed():
    j = {"results": {
        "parent (pre)": {"perevt": 0.04},
        "parent (post-discovery)": {"perevt": 0.068},
        "p->res (post-discovery) residual": {"perevt": 0.253},
        "p->res (post-discovery) concat": {"perevt": 0.235},
        "p->res-nplm (post-discovery) residual": {"perevt": 0.087},
        "p->res-nplm (post-discovery) concat": {"perevt": 0.130}},
        "discovery": [{"purity": 0.129}]}
    rows = [r for r in cell_2x2(j, {}, parent="p") if r["metric"] == "perevt"]
    assert {(r["obj"], r["kind"]) for r in rows} == {
        ("res", "residual"), ("res", "concat"),
        ("res-nplm", "residual"), ("res-nplm", "concat")}
    assert all(r["missing"] for r in rows)      # no pre rows supplied


# ------------------------------------------------------------ npz naming ---

@pytest.mark.parametrize("space,key", [
    ("supcon-ft (parent)", "supcon-ft_(parent)"),
    ("supcon-ft->res (residual)", "supcon-ft-res_(residual)"),
    ("supcon-ft->res (concat)", "supcon-ft-res_(concat)"),
    ("supcon-ft->res-nplm (residual)", "supcon-ft-res-nplm_(residual)"),
    ("ss-ft->res (concat)", "ss-ft-res_(concat)"),
])
def test_npz_key_matches_exp71_naming(space, key):
    assert npz_key(space) == key


def test_npz_keys_exist_in_a_real_exp71_archive():
    """Guard the mapping against the actual archive, not just its spec."""
    f = os.path.join(REPO, "logs", "exp71", "results_dtd_dino_ft71.npz")
    if not os.path.exists(f):
        pytest.skip("exp-71 dtd/dino archive absent")
    d = np.load(f, allow_pickle=True)
    for sp in d["spaces"]:
        assert f"{npz_key(str(sp))}__perevt" in d.files, sp


def test_archived_rows_missing_file_is_empty_not_error():
    rows, path = archived_rows("nosuchds", "nobase")
    assert rows == {} and path.endswith(".npz")


def test_archived_rows_filters_to_the_named_parent():
    f = os.path.join(REPO, "logs", "exp71", "results_dtd_dino_ft71.npz")
    if not os.path.exists(f):
        pytest.skip("exp-71 dtd/dino archive absent")
    rows, _ = archived_rows("dtd", "dino", parent="supcon-ft")
    assert rows and all(k.startswith("supcon-ft") for k in rows)
    assert not any(k.startswith("ss-ft") for k in rows)


# ---------------------------------------------------------- the repair -----

@pytest.mark.parametrize("ds,base", [("galaxy10", "dino"), ("galaxy10", "lejepa")])
def test_galaxy10_pre_children_are_recoverable_from_the_npz(ds, base):
    """The 134c JSONs hold `{}` for these children because 134c snapshotted
    the exp-71 npz before the galaxy10 h1 draw sweep re-wrote it.  The data
    is on disk.  If this regresses, the 2x2 silently loses two cells."""
    f = os.path.join(REPO, "logs", "exp71", f"results_{ds}_{base}_ft71.npz")
    if not os.path.exists(f):
        pytest.skip(f"{f} absent")
    rows, _ = archived_rows(ds, base)
    kids = {k: v for k, v in rows.items() if "parent" not in k}
    assert len(kids) == 4, sorted(kids)
    for k, v in kids.items():
        assert "perevt" in v and "eucl" in v, (k, v)

    stale = json.load(open(os.path.join(
        LOG134, f"postdisc_{ds}_{base}_ft134c.json"))).get("archived_exp71", {})
    assert any(not v for v in stale.values()), \
        "JSON no longer stale -- update this test's premise"


# ------------------------------------------- claims pinned to the archive ---

def _real(cell, obj, kind, metric="perevt"):
    p = os.path.join(LOG134, f"postdisc_{cell}_ft134c.json")
    if not os.path.exists(p):
        pytest.skip(f"{p} absent")
    ds, base = cell.rsplit("_", 1)
    pre, _ = archived_rows(ds, base)
    rows = cell_2x2(json.load(open(p)), pre)
    m = [r for r in rows if r["obj"] == obj and r["kind"] == kind
         and r["metric"] == metric]
    assert m, (cell, obj, kind, metric)
    return m[0]


def test_dtd_dino_res_is_the_superadditive_cell():
    """The paper's capability claim, pinned.  Parent at/below nominal,
    construction alone HURTS, composed clears 4x nominal."""
    r = _real("dtd_dino", "res", "residual")
    assert r["A"] == pytest.approx(0.040, abs=1e-3)
    assert r["C"] == pytest.approx(0.0225, abs=1e-3)
    assert r["D"] == pytest.approx(0.253, abs=1e-3)
    assert r["A"] <= ALPHA, "parent must be at or below nominal for a crossing"
    assert r["main_ctor"] < 0, "construction alone must hurt here"
    assert r["D"] > 4 * ALPHA
    assert r["interaction"] == pytest.approx(0.2025, abs=1e-3)


def test_galaxy10_dino_striking_gain_is_a_construction_main_effect():
    """0.021 -> 0.299 is NOT composition: C is already 0.281."""
    r = _real("galaxy10_dino", "res-nplm", "residual")
    assert r["A"] == pytest.approx(0.0214, abs=1e-3)
    assert r["C"] == pytest.approx(0.281, abs=1e-2)
    assert r["main_ctor"] > 0.25, "the residual does this alone"
    assert abs(r["interaction"]) < 0.02, "and composition adds ~nothing"


def test_galaxy10_lejepa_striking_gain_is_a_discovery_main_effect():
    """0.007 -> 0.206 is discovery, in the cell whose r2 purity is 0.000 --
    the mechanism falsifier.  A large interaction here would mean the
    'richer correct anchor set' story is wrong."""
    r = _real("galaxy10_lejepa", "res-nplm", "concat")
    assert r["main_disc"] > 0.18
    assert abs(r["interaction"]) < 0.02
    j = json.load(open(os.path.join(LOG134, "postdisc_galaxy10_lejepa_ft134c.json")))
    assert j["discovery"][-1]["purity"] == pytest.approx(0.0, abs=1e-9)


def test_only_dtd_dino_res_rows_are_interaction_driven_crossings():
    """Across every cell, a per-event threshold crossing that is actually
    driven by the INTERACTION occurs only for dtd/dino `res`."""
    driven = []
    for p in sorted(glob.glob(os.path.join(LOG134, "postdisc_*_ft134c.json"))):
        cell = cell_name(p)
        ds, base = cell.rsplit("_", 1)
        pre, _ = archived_rows(ds, base)
        for r in cell_2x2(json.load(open(p)), pre):
            if r["metric"] != "perevt" or r["missing"]:
                continue
            if r["A"] <= ALPHA * 1.5 and r["D"] > 4 * ALPHA and r["interaction"] > 0.10:
                driven.append((cell, r["obj"]))
    if not driven:
        pytest.skip("134c archive absent")
    assert {c for c, _ in driven} == {"dtd_dino"}, driven
    assert {o for _, o in driven} == {"res"}, driven


def test_r1_purity_reads_round_one_not_the_last_round():
    j = {"discovery": [{"purity": 0.129}, {"purity": 0.318}]}
    assert r1_purity(j) == pytest.approx(0.129)
    assert np.isnan(r1_purity({"discovery": []}))


def test_cell_name_strips_the_wrapper():
    assert cell_name("logs/exp134/postdisc_dtd_dino_ft134c.json") == "dtd_dino"
    assert cell_name("postdisc_galaxy10_lejepa_ft134c.json") == "galaxy10_lejepa"
