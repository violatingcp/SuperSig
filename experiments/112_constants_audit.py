"""
Experiment 112 (IMPROVEMENT_TESTS.md #112): the inherited-constants audit.

Exp 110 found the campaign's best C100 space by accident: `tau` was never
set, so the arm ran at the loss-class default 1.0 rather than the inherited
SupCon 0.1.  That is a symptom -- the campaign swept the DISCRETE design cube
exhaustively and never swept the CONTINUOUS constants at all.  Before deciding
what to re-run we should know what was never chosen.

This is static analysis: it extracts every numeric default from the library
and experiment surface, cross-references each against the campaign log to see
whether any experiment ever varied it, and classifies provenance.  No GPU, no
data, no training.

Deliverable: logs/exp112/constants.md -- a table of (constant, value, where it
lives, ever swept?, by which exp), plus a shortlist of NEVER-SWEPT constants
that sit under load-bearing claims.

Prediction: at least three more never-swept constants sit under headline claims.

    python experiments/112_constants_audit.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import ast
import json
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Constants we already know the provenance of, with the claim they sit under.
# "swept_by" is filled from the log scan; "claim" is curated.
KNOWN = {
    "tau":            dict(claim="every softmax-arm comparison (probe/calibration dissociation)"),
    "lam":            dict(claim="marginal strength; declared inert for NPLM at init (exp 52)"),
    "n_slices":       dict(claim="SIGReg Monte-Carlo budget"),
    "clamp":          dict(claim="NPLM stabiliser; App. A argues max-subtraction is fatal"),
    "M":              dict(claim="SparKer kernel count; exp 97 inverted the default"),
    "sigma_ratio":    dict(claim="SparKer anneal; exp 97 found it flat"),
    "k":              dict(claim="LID neighbours; swept in exp 78"),
    "shrink":         dict(claim="Mahalanobis / panel reference covariance"),
    "tau_quantile":   dict(claim="discovery pool threshold; swept exp 89 (distance only)"),
    "kmax":           dict(claim="BIC cluster count; fragmentation is documented behaviour"),
    "merge_dist":     dict(claim="anchor merge; absorbs BIC fragmentation"),
    "temp":           dict(claim="SupCon temperature (the exp-110 constant)"),
    "steps":          dict(claim="SparKer fit budget"),
    "lr":             dict(claim="optimiser"),
    "epochs":         dict(claim="training length"),
}

SKIP_NAMES = {"self", "seed", "device", "dataset", "quick", "verbose", "out",
              "n_null", "n_sig", "alpha"}


def numeric_defaults(path):
    """(function, arg, default) for every numeric-defaulted kwarg in a file."""
    try:
        tree = ast.parse(open(path).read())
    except SyntaxError:
        return []
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        args = node.args
        defaults = list(args.defaults)
        named = args.args[len(args.args) - len(defaults):] if defaults else []
        pairs = list(zip(named, defaults))
        pairs += [(a, d) for a, d in zip(args.kwonlyargs, args.kw_defaults) if d]
        for a, d in pairs:
            if a.arg in SKIP_NAMES:
                continue
            if isinstance(d, ast.Constant) and isinstance(d.value, (int, float)) \
                    and not isinstance(d.value, bool):
                out.append((node.name, a.arg, d.value))
    return out


def scan_log_for_sweeps(names):
    """Which experiment entries in SUMMARY_TABLES mention varying each name."""
    log = os.path.join(ROOT, "logs", "SUMMARY_TABLES.md")
    if not os.path.exists(log):
        return {n: [] for n in names}
    text = open(log).read()
    # entries look like "Exp NN (`script`, date...):"
    entries = re.split(r"\nExp (\d+[a-z]?) \(", text)
    chunks = [(entries[i], entries[i + 1]) for i in range(1, len(entries) - 1, 2)]
    hits = {n: [] for n in names}
    for exp, body in chunks:
        low = body.lower()
        for n in names:
            # a sweep mentions the name AND a scan/grid/factorial word nearby
            if re.search(rf"\b{re.escape(n)}\b", low) and re.search(
                    r"scan|grid|factorial|sweep|vs |x \{|\bab\b|paired", low):
                hits[n].append(exp)
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="logs/exp112")
    args = ap.parse_args()

    rows = []
    for sub in ("supersig", "experiments"):
        d = os.path.join(ROOT, sub)
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".py"):
                continue
            for func, arg, val in numeric_defaults(os.path.join(d, fn)):
                rows.append(dict(file=f"{sub}/{fn}", func=func, name=arg,
                                 value=val))

    names = sorted({r["name"] for r in rows})
    swept = scan_log_for_sweeps(names)

    # collapse to one row per constant name, listing where it appears
    byname = {}
    for r in rows:
        b = byname.setdefault(r["name"], dict(values=set(), sites=[]))
        b["values"].add(r["value"])
        b["sites"].append(f"{r['file']}::{r['func']}")

    os.makedirs(args.out, exist_ok=True)
    md = ["# Exp 112 — inherited-constants audit", "",
          "Auto-generated by `experiments/112_constants_audit.py`.  "
          "`swept by` is a keyword scan of `logs/SUMMARY_TABLES.md` and is a "
          "LOWER bound: it finds experiments that discuss varying the "
          "constant, not every run that happened to change it.", "",
          "| constant | value(s) | sites | swept by | sits under |",
          "|---|---|---|---|---|"]
    never = []
    for n in sorted(byname, key=lambda x: (not KNOWN.get(x), x)):
        b = byname[n]
        vals = ",".join(str(v) for v in sorted(b["values"])[:6])
        s = swept.get(n, [])
        claim = KNOWN.get(n, {}).get("claim", "")
        md.append(f"| `{n}` | {vals} | {len(b['sites'])} | "
                  f"{', '.join('exp ' + e for e in s) if s else '**never**'} | "
                  f"{claim} |")
        if not s and claim:
            never.append((n, vals, claim))

    md += ["", "## Never-swept constants sitting under a named claim", ""]
    if never:
        for n, v, c in never:
            md.append(f"- **`{n}`** (= {v}) — {c}")
    else:
        md.append("- (none)")
    md += ["", f"Total distinct numeric constants: {len(byname)}; "
               f"never-swept-with-a-claim: {len(never)}.",
           "", "Prediction was: at least three more never-swept constants sit "
               "under headline claims.  "
               f"Observed: {len(never)}."]

    path = os.path.join(args.out, "constants.md")
    open(path, "w").write("\n".join(md) + "\n")
    json.dump({n: dict(values=sorted(b["values"]), n_sites=len(b["sites"]),
                       swept_by=swept.get(n, []))
               for n, b in byname.items()},
              open(os.path.join(args.out, "constants.json"), "w"), indent=1)
    print("\n".join(md[-8:]))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
