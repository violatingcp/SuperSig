"""Guard against the count-without-set holdout bug (2026-08-26).

The holdout centralization originally routed only the holdout COUNT through
supersig.holdouts (n_holdout) while experiments kept building the SET inline
as set(range(n_cls - nh, n_cls)) -- so SUPERSIG_HOLDOUT_DRAW tagged every
artifact but never changed which class was held out, and 5 "draws" produced
byte-identical results.  These tests pin the two invariants that failure
violated.
"""
import glob
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from supersig.holdouts import holdout_set  # noqa: E402

EXP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "experiments")
INLINE = re.compile(r"holdouts?\s*=\s*set\(range\(\w+\s*-\s*\w+,\s*\w+\)\)")


def test_no_experiment_builds_holdout_set_inline():
    """Any file that imports n_holdout must build the SET via holdout_set;
    an inline set(range(...)) silently ignores SUPERSIG_HOLDOUT_DRAW."""
    offenders = []
    for fn in glob.glob(os.path.join(EXP, "*.py")):
        src = open(fn).read()
        if "n_holdout(" in src and INLINE.search(src):
            offenders.append(os.path.basename(fn))
    assert not offenders, (
        f"inline holdout-set construction (draw-blind) in: {offenders}")


def test_draw_env_changes_the_selected_classes():
    """SUPERSIG_HOLDOUT_DRAW must change WHICH classes are held out, in a
    fresh interpreter (the env is read at call time)."""
    code = ("from supersig.holdouts import holdout_set;"
            "print(sorted(holdout_set('galaxy10', 10)))")
    def run(env_draw):
        env = dict(os.environ)
        env.pop("SUPERSIG_HOLDOUT_DRAW", None)
        env.pop("SUPERSIG_NH", None)
        if env_draw is not None:
            env["SUPERSIG_HOLDOUT_DRAW"] = str(env_draw)
        return subprocess.run([sys.executable, "-c", code], env=env,
                              capture_output=True, text=True).stdout.strip()
    default = run(None)
    draws = {run(d) for d in range(5)}
    assert default == "[9]"                       # campaign default preserved
    # seeded draws may collide on a 10-class dataset; they must still VARY
    assert len(draws) >= 2, f"draws do not vary: {draws}"
    assert draws != {default}, "draws never leave the default class"


def test_default_reproduces_campaign_rule():
    for ds, n_cls, want in (("galaxy10", 10, {9}),
                            ("dtd", 47, set(range(37, 47))),
                            ("cars", 196, set(range(186, 196)))):
        assert holdout_set(ds, n_cls) == want
