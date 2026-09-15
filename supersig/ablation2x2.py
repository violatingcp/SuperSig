"""The discovery x construction 2x2 ablation.

Exp 134c measured the composed order (parent -> discovery -> child) and put
the archived exp-71 pre-discovery rows beside it, but it never formed the
INTERACTION, and the interaction is the claim: for per-event power the two
ingredients do nothing alone and something large together.

The 2x2, per (cell, construction, metric):

                     no discovery        + discovery
    parent           A  (exp71 parent)   B  (134c "parent (post-discovery)")
    + construction   C  (exp71 child)    D  (134c child)

    main_disc  = B - A      what discovery buys the parent
    main_ctor  = C - A      what the construction buys pre-discovery
    interaction= (D - C) - (B - A)
               = D - C - B + A
    total      = D - A

SUPERADDITIVE means interaction > 0: the composition beats the sum of its
parts.  On dtd/dino `res` per-event, A=0.040 B=0.068 C=0.023 D=0.253, so
main_disc=+0.028, main_ctor=-0.018, interaction=+0.203 -- neither ingredient
works alone and the pair takes per-event power from nominal (alpha=0.05) to
6x nominal.

WHY THE ARCHIVED ROWS CAN BE EMPTY.  Exp 134c snapshots the exp-71 npz at run
time.  Where exp 71 was re-run afterwards (the galaxy10 h1 draw sweep landed
after 134c), the JSON holds `{}` for the children while the npz on disk now
has them.  `archived_rows` re-reads the npz, so a stale snapshot is repaired
by re-aggregating -- no GPU work.  Cells whose npz genuinely lacks the child
are reported as MISSING rather than silently dropped.
"""
import json
import os

import numpy as np

METRICS = ("perevt", "eucl", "probe", "mahaT")

# per-event power is a power at a nominal false-positive rate; at or below
# alpha it means NO ability to flag an individual novel sample.
ALPHA = 0.05


def npz_key(space):
    """exp-71 space name -> its npz key stem.  'a->b (c)' -> 'a-b_(c)'."""
    return space.replace(" ", "_").replace("->", "-")


def archived_rows(ds, base, parent="supcon-ft", tag="", logdir="logs/exp71"):
    """Pre-discovery (exp-71) rows for one cell, re-read from the npz.

    Returns (rows, path).  `rows` is {} when the npz is absent."""
    path = os.path.join(logdir, f"results_{ds}_{base}_ft71{tag}.npz")
    if not os.path.exists(path):
        return {}, path
    d = np.load(path, allow_pickle=True)
    rows = {}
    for sp in d["spaces"]:
        sp = str(sp)
        if not sp.startswith(parent):
            continue
        k = npz_key(sp)
        rows[sp] = {m: float(d[f"{k}__{m}"]) for m in METRICS
                    if f"{k}__{m}" in d.files}
    return rows, path


def _post_key(parent, obj, kind):
    return f"{parent}->{obj} (post-discovery) {kind}"


def _pre_key(parent, obj, kind):
    return f"{parent}->{obj} ({kind})"


def cell_2x2(post_json, pre_rows, parent="supcon-ft"):
    """Form the 2x2 for every (construction, kind, metric) in one cell.

    `post_json` is a loaded 134c JSON; `pre_rows` comes from archived_rows.
    Returns a list of dicts, one per (obj, kind, metric), each carrying
    A/B/C/D and the three contrasts.  Rows whose pre-discovery child is
    missing are returned with `missing=True` and no contrasts, so a caller
    can report them instead of dropping them."""
    res = post_json["results"]
    out = []
    if "parent (pre)" not in res or "parent (post-discovery)" not in res:
        return out
    for key in res:
        if "(post-discovery)" not in key or key == "parent (post-discovery)":
            continue
        stem, kind = key.rsplit(" ", 1)          # '... (post-discovery)', 'residual'
        obj = stem.split("->")[1].split(" ")[0]
        pre_child = pre_rows.get(_pre_key(parent, obj, kind), {})
        for m in METRICS:
            A = res["parent (pre)"].get(m)
            B = res["parent (post-discovery)"].get(m)
            D = res[key].get(m)
            C = pre_child.get(m)
            row = dict(obj=obj, kind=kind, metric=m, A=A, B=B, C=C, D=D)
            if C is None or A is None or B is None or D is None:
                row["missing"] = True
            else:
                row.update(missing=False, main_disc=B - A, main_ctor=C - A,
                           interaction=(D - C) - (B - A), total=D - A)
            out.append(row)
    return out


def load_cell(path):
    with open(path) as fh:
        return json.load(fh)


def cell_name(path):
    return os.path.basename(path).replace("postdisc_", "").replace("_ft134c.json", "")


def r1_purity(post_json):
    h = post_json.get("discovery") or []
    return float(h[0]["purity"]) if h else float("nan")
