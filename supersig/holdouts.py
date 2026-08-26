"""Central holdout-set selection and run tagging.

WHY THIS EXISTS.  The pair

    nh = 1 if ds == "galaxy10" else 10
    holdouts = set(range(n_cls - nh, n_cls))

was duplicated across 22 experiment files (23 sites).  That made the campaign's
holdout policy un-changeable in practice, and the paper needs BOTH regimes:

  * single-holdout  -- the hard, low-rate regime.  One novel class at a rate of
    ~1/n_cls.  Exps 89/109: on CIFAR-100 no pooling statistic reaches the purity
    gate here (0.03) -- a RATE floor, not a geometry failure.  Only calibrated
    per-event scoring works.  This is the regime we now run for EVERY dataset.
  * multi-holdout   -- the higher-rate regime, reserved for label-rich datasets.
    Exp 109: the frozen density-ratio pool clears the gate at h5/h10 (purity
    0.358, round 2 0.418) and the quantile-strictness conclusion INVERTS
    relative to h1.

Because the two regimes reach different conclusions they must never be pooled
in a table, and -- the reason for `run_tag()` below -- their artifacts must
never share a filename.

THE OVERWRITE HAZARD.  Every output path in the campaign is keyed on
dataset/base/arm/seed and encodes nothing about the holdout count.  Most
dangerous are the fine-tune checkpoints, e.g. exp 70

    checkpoints/{ds}_ft_{base}_{arm}_seen.pt

where `_seen` means "trained with the holdouts excluded" -- the CONTENTS differ
between nh=1 and nh=10 while the name does not.  Running the single-holdout
battery without tagging would destroy the existing multi-holdout trunks, and
exp 80's resume logic would then silently reuse stale results as valid.

USAGE.  One environment variable switches the whole battery and tags every
artifact it writes:

    SUPERSIG_NH=1 python experiments/70_cars_ft_suite.py --dataset dtd ...

With SUPERSIG_NH unset the campaign default is reproduced EXACTLY -- same
holdout sets, same filenames (run_tag() == "") -- so archived results and
resume logic keep working untouched.

Note galaxy10 is nh=1 by default already, so it is unaffected by the switch;
its artifacts keep their untagged names in a single-holdout run only if
SUPERSIG_NH is unset.  When set, everything is tagged uniformly.
"""
import os

# Campaign default: last 10 classes, except galaxy10 which has only 10 classes
# total and so holds out 1.
DEFAULT_NH = 10
PER_DATASET_NH = {"galaxy10": 1}

ENV_VAR = "SUPERSIG_NH"
DRAW_VAR = "SUPERSIG_HOLDOUT_DRAW"


def _env_nh():
    v = os.environ.get(ENV_VAR, "").strip()
    if not v:
        return None
    try:
        n = int(v)
    except ValueError:
        raise ValueError(f"{ENV_VAR}={v!r} is not an integer")
    if n < 1:
        raise ValueError(f"{ENV_VAR}={n} must be >= 1")
    return n


def n_holdout(ds, nh=None):
    """Number of held-out classes for `ds`.

    Precedence: explicit `nh` argument > $SUPERSIG_NH > per-dataset default.
    """
    if nh is not None:
        return int(nh)
    env = _env_nh()
    if env is not None:
        return env
    return PER_DATASET_NH.get(ds, DEFAULT_NH)


def _env_draw():
    v = os.environ.get(DRAW_VAR, "").strip()
    if not v:
        return None
    try:
        return int(v)
    except ValueError:
        raise ValueError(f"{DRAW_VAR}={v!r} is not an integer")


def holdout_set(ds, n_cls, nh=None, draw=None):
    """The held-out class ids.

    Default (draw=None): the LAST `n_holdout(ds)` classes -- the campaign rule,
    reproduced exactly.  Clamped to n_cls - 1 so at least one class is seen.

    With a `draw` index (or $SUPERSIG_HOLDOUT_DRAW), the holdouts are instead a
    RANDOM subset chosen by a generator seeded on (ds, n_cls, k, draw).

    WHY DRAWS EXIST.  Exp 118 found that draw-to-draw spread exceeds seed
    spread -- WHICH class is held out matters more than the training seed
    (probe sd ~0.019; mahaT 0.391-0.569 across draws vs 0.001-0.010 across
    seeds).  That is tolerable when 10 classes are held out and the draw
    averages over them, but under nh=1 the entire result rests on ONE class.
    Single-holdout numbers therefore need an interval over draws, not just over
    seeds.  Seeding on the dataset identity keeps draw d comparable across
    arms and bases within a dataset, so comparisons stay paired.
    """
    k = min(n_holdout(ds, nh), int(n_cls) - 1)
    d = draw if draw is not None else _env_draw()
    if d is None:
        return set(range(int(n_cls) - k, int(n_cls)))
    import hashlib as _hl
    import numpy as _np
    # NOT hash(): Python randomizes string hashing per process (PYTHONHASHSEED),
    # which would silently give draw d a DIFFERENT holdout class on every run.
    key = f"{ds}|{int(n_cls)}|{int(k)}|{int(d)}".encode()
    seed = int.from_bytes(_hl.blake2b(key, digest_size=4).digest(), "big")
    rng = _np.random.default_rng(seed)
    return set(int(c) for c in rng.choice(int(n_cls), size=k, replace=False))


def seen_classes(ds, n_cls, nh=None, draw=None):
    h = holdout_set(ds, n_cls, nh, draw)
    return [c for c in range(int(n_cls)) if c not in h]


def run_tag(nh=None, draw=None):
    """Filename suffix distinguishing a non-default holdout run.

    Empty when the campaign default is in force, so existing artifact names and
    resume logic are byte-identical.  `"_h1"` under SUPERSIG_NH=1, `"_h1_d3"`
    with draw 3, `"_d3"` for a default-nh run of draw 3.
    """
    n = nh if nh is not None else _env_nh()
    d = draw if draw is not None else _env_draw()
    return ("" if n is None else f"_h{int(n)}") + ("" if d is None else f"_d{int(d)}")


def regime(ds, n_cls=None, nh=None):
    """'single' or 'multi' -- which table this run's numbers belong in."""
    return "single" if n_holdout(ds, nh) == 1 else "multi"


def describe(ds, n_cls, nh=None, draw=None):
    h = sorted(holdout_set(ds, n_cls, nh, draw))
    return (f"{ds}: {len(h)} holdout(s) {h[0]}..{h[-1]} of {n_cls} "
            f"[{regime(ds, n_cls, nh)}-holdout regime]"
            + (f", tag '{run_tag(nh, draw)}'" if run_tag(nh, draw) else ""))


def _selftest():
    import contextlib

    @contextlib.contextmanager
    def env(v):
        old = os.environ.get(ENV_VAR)
        if v is None:
            os.environ.pop(ENV_VAR, None)
        else:
            os.environ[ENV_VAR] = str(v)
        try:
            yield
        finally:
            os.environ.pop(ENV_VAR, None)
            if old is not None:
                os.environ[ENV_VAR] = old

    # 1. Unset reproduces the campaign default EXACTLY, including empty tags.
    with env(None):
        assert holdout_set("dtd", 47) == set(range(37, 47)), holdout_set("dtd", 47)
        assert holdout_set("flowers", 102) == set(range(92, 102))
        assert holdout_set("galaxy10", 10) == {9}
        assert holdout_set("cifar100", 100) == set(range(90, 100))
        assert run_tag() == ""
        assert regime("dtd", 47) == "multi"
        assert regime("galaxy10", 10) == "single"

    # 2. The switch makes every dataset single-holdout and tags the artifacts.
    with env(1):
        for ds, n in [("dtd", 47), ("flowers", 102), ("cars", 196),
                      ("aircraft", 100), ("galaxy10", 10), ("cifar100", 100)]:
            assert holdout_set(ds, n) == {n - 1}, (ds, holdout_set(ds, n))
            assert regime(ds, n) == "single"
        assert run_tag() == "_h1"

    # 3. Intermediate values work and tag distinctly (exp-89/109 grid).
    with env(5):
        assert holdout_set("cifar100", 100) == set(range(95, 100))
        assert run_tag() == "_h5" and regime("cifar100", 100) == "multi"

    # 4. Explicit argument beats the environment.
    with env(1):
        assert holdout_set("dtd", 47, nh=10) == set(range(37, 47))
        assert run_tag(10) == "_h10"

    # 5. Clamping: never hold out every class.
    with env(50):
        assert holdout_set("galaxy10", 10) == set(range(1, 10))

    # 6. Bad input is rejected loudly rather than silently defaulting.
    for bad in ("0", "-3", "abc"):
        with env(bad):
            try:
                n_holdout("dtd")
            except ValueError:
                pass
            else:
                raise AssertionError(f"{bad!r} should have raised")

    # 7. Tag uniqueness -- the whole point is no two regimes share a filename.
    tags = set()
    for v in (None, 1, 5, 10):
        with env(v):
            tags.add(run_tag())
    assert len(tags) == 4, tags

    print("supersig.holdouts selftest OK")
    with env(None):
        print(" default:", describe("dtd", 47))
    with env(1):
        print(" SUPERSIG_NH=1:", describe("dtd", 47))


if __name__ == "__main__":
    _selftest()
