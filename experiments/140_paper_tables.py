"""Experiment 140: generate the paper's result tables from the logs.

Nothing here computes anything new.  It reads the archived result files and
emits LaTeX into `docs/tables/`, so that every number in the paper has one
source and a re-run regenerates the tables instead of requiring a human to
re-transcribe them.

    python experiments/140_paper_tables.py            # write docs/tables/*.tex
    python experiments/140_paper_tables.py --stdout   # print instead
    python experiments/140_paper_tables.py --selftest

DESIGN NOTE.  Every table carries its own completeness marker.  The campaign
is deliberately being written up while incomplete, so a table that is missing
cells must SAY it is missing them rather than quietly showing fewer rows: each
emitter returns a `status` line that names what is present and what is absent,
and that line is printed into the LaTeX as a comment AND rendered in the
caption.  A blank in a table body is always `--`, never an omitted row.
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from supersig.ablation2x2 import archived_rows, cell_2x2, cell_name

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(REPO, "logs")
OUT = os.path.join(REPO, "docs", "tables")

# paper-facing names for the code's arm keys (docs/LOSSES.md is canonical)
PRETTY = {
    "gcd-ft": "GCD", "gcd-sigreg-ft": "GCD+SIGReg",
    "simclr": "SimCLR", "visreg": "VISReg/LeJEPA", "nplm": "NPLM (unsup.)",
    "supcon": "SupCon", "supsig": "SupCon+SIGReg (repulse)",
    "nplmcw": "NPLM-dist.+classwise SIGReg", "ssig": r"\textbf{SupCon+SIGReg ($\lambda{=}5$)}",
    "nplmsd": "NPLM-dist. (sup.)",
    "simclr-ft": "SimCLR", "sigreg-ssl-ft": "SimCLR+SIGReg",
    "nplm-bil-ft": "NPLM-bilinear", "supcon-ft": "SupCon",
    "ss-ft": r"\textbf{SupCon+SIGReg ($\lambda{=}5$)}",
    "nplm-sup-ft": "NPLM-dist. (sup.)",
}


CELL = {"cars_visreg": "cars / VISReg", "dtd_dino": "DTD / DINO",
        "galaxy10_dino": "galaxy10 / DINO",
        "galaxy10_lejepa": "galaxy10 / LeJEPA",
        "galaxy10_visreg": "galaxy10 / VISReg"}
CTOR = {"res": "SupCon", "res-nplm": "NPLM"}
# how a construction reads in the residuals table (appendix)
CHILD = {"res (residual)": "residual child (SupCon), bare",
         "res (concat)": "residual child (SupCon), concat",
         "res-nplm (residual)": "residual child (NPLM), bare",
         "res-nplm (concat)": "residual child (NPLM), concat"}
KIND = {"residual": "bare", "concat": "concat"}


def esc(s):
    # `->` is not a text-mode glyph in OT1 (it prints as `-¿`); typeset it.
    return (str(s).replace("_", r"\_").replace("&", r"\&")
            .replace("->", r"$\to$"))


def fnum(x, p=3, dash="--"):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return dash
    return f"{x:.{p}f}"


def _wrap(body, caption, label, status, colspec, header, long=False,
          wide=False, size=None):
    """`long=True` emits a longtable, which breaks across pages -- required
    for any table with more rows than fit on one (a plain `table` silently
    overflows the page instead of breaking).  `wide=True` rotates the table
    into a landscape page (pdflscape) and drops to \scriptsize with a tight
    column gap -- for the grids that overflow the 5.5in text width.  `size`
    overrides the font size command (e.g. r"\footnotesize")."""
    sz = size or (r"\footnotesize" if wide else r"\small")
    setup = [sz] + ([r"\setlength{\tabcolsep}{3pt}"] if wide else [])
    pre = [r"\begin{landscape}"] if wide else []
    post = [r"\end{landscape}"] if wide else []
    if long:
        return "\n".join([
            f"% STATUS: {status}"] + pre + [
            r"\begin{center}"] + setup + [
            rf"\begin{{longtable}}{{{colspec}}}",
            rf"\caption{{{caption} \emph{{Coverage:}} {status}}}"
            rf"\label{{{label}}}\\",
            r"\toprule", header, r"\midrule", r"\endfirsthead",
            r"\toprule", header, r"\midrule", r"\endhead",
            r"\bottomrule", r"\endfoot",
            body, r"\end{longtable}", r"\end{center}"] + post + [""])
    return "\n".join([
        f"% STATUS: {status}"] + pre + [
        r"\begin{table}[t]" if not wide else r"\begin{table}[p]", r"\centering"] + setup + [
        rf"\begin{{tabular}}{{{colspec}}}", r"\toprule",
        header, r"\midrule", body, r"\bottomrule", r"\end{tabular}",
        rf"\caption{{{caption} \emph{{Coverage:}} {status}}}",
        rf"\label{{{label}}}", r"\end{table}"] + post + [""])


# --------------------------------------------------------------- table 1 ---
def t_objectives():
    """Leakage-free from-scratch CIFAR-100 battery, all 8 objectives."""
    f = os.path.join(LOGS, "exp136", "master_cifar100.json")
    if not os.path.exists(f):
        return None
    m = json.load(open(f))
    order = ["simclr", "visreg", "nplm", "supcon", "supsig", "nplmcw",
             "ssig", "nplmsd"]
    rows, n = [], 0
    for a in order:
        if a not in m:
            rows.append(f"{esc(a)} & " + " & ".join(["--"] * 7) + r" \\")
            continue
        d = m[a].get("pre", m[a])
        n += 1
        sup = "y" if a in ("supcon", "supsig", "nplmcw", "ssig", "nplmsd") else "n"
        rows.append(" & ".join([
            PRETTY.get(a, esc(a)), sup,
            fnum(d.get("probe")), fnum(d.get("top1")), fnum(d.get("acc")),
            fnum(d.get("eucl")), fnum(d.get("mahaT")),
            fnum(d.get("perevt"), 2)]) + r" \\")
    status = (f"{n}/8 objectives present; single holdout (class 4), "
              r"one seed per cell, probe averaged over 3 probe restarts.")
    return _wrap(
        "\n".join(rows),
        r"\textbf{Objectives on the leakage-free CIFAR-100 lineage} "
        r"(random init; the encoder never saw the held-out class). "
        r"\emph{Decodable} columns: linear-probe AUC for the held-out class "
        r"(trained \emph{with} its labels), top-1 accuracy on the seen classes, "
        r"and nearest-anchor accuracy on the seen classes. "
        r"\emph{Discoverable} columns: novelty-ranking AUC by Euclidean and "
        r"by tied-Mahalanobis distance, and per-event detection power at a "
        r"$5\%$ false-positive rate --- the last is $\approx0.05$ for a space "
        r"carrying no usable signal, so $0.00$ means the space cannot flag an "
        r"individual novel image at all. Adding SIGReg is a probe tie and a "
        r"top-1 loss, and wins every discoverability column.",
        "tab:objectives", status,
        "llcccccc",
        r"& & \multicolumn{3}{c}{decodable} & "
        r"\multicolumn{3}{c}{discoverable} \\"
        "\n" r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}" "\n"
        r"objective & sup. & probe & top-1 & anchor & Eucl. & Maha. & "
        r"per-event \\")


# --------------------------------------------------------------- table 2 ---
# The arm is NOT on the purity line; it comes from the section header that
# precedes it ("----- natural discovery: ss-ft -----"), so the log has to be
# walked in order rather than pattern-matched line by line.
_ARM = re.compile(r"^-+\s*natural discovery:\s*(\S+)\s*-+\s*$")
_PUR = re.compile(r"^\s*round 1: pool=(\d+) purity=([\d.]+)")
_POSTF = re.compile(r"^===== POST grid, f=([\d.]+)")
_ARMPOST = re.compile(r"^\s+\[(\S+)\] per-event post f=")

# TWO discovery passes live in every exp-70 log / npz, and they answer
# different questions:
#   natural pass  -- the pool is drawn from the full train bank, so the WHOLE
#                    held-out class is present (keys post_probe_{arm},
#                    post_mahaT_{arm}; "----- natural discovery: arm -----").
#   POST grid     -- discovery re-run from a small INJECTED sample of the
#                    held-out class at fraction f (keys postf_{metric}_{arm}
#                    indexed by post_fractions; "===== POST grid, f=... =====").
# The paper's regime is the second (a small new sample), at F_PAPER.  The
# first is reported only as a clearly-labelled "whole class present" column.
F_PAPER = 0.02
DRAWS70 = {"galaxy10": (0, 3, 5, 7, 8), "dtd": (0, 1, 3, 4, 5)}


def _log70(ds, base, d):
    """exp-70 draw log; galaxy10 logs are named `g10`."""
    for p in (os.path.join(LOGS, f"exp70_{ds}_{base}_h1_d{d}.log"),
              os.path.join(LOGS, f"exp70_g10_{base}_h1_d{d}.log")):
        if os.path.exists(p):
            return p
    return None


def _draw_purities(ds, base, draws=None):
    """NATURAL pass (whole class present): round-1 pool purity per arm per
    draw, read from the exp-70 logs (the npz files carry no purity field)."""
    out = {}
    for d in draws or DRAWS70.get(ds, (0, 1, 3, 4, 5)):
        p = _log70(ds, base, d)
        if p is None:
            continue
        arm = None
        for line in open(p, errors="ignore"):
            if line.startswith("====="):
                arm = None                      # the POST grid ends the pass
            m = _ARM.match(line.strip())
            if m:
                arm = m.group(1)
                continue
            m = _PUR.match(line)
            if m and arm is not None:
                # first round-1 line after the header is that arm's
                out.setdefault(arm, {}).setdefault(d, float(m.group(2)))
    return out


def _draw_purities_f(ds, base, f=F_PAPER, draws=None):
    """INJECTED-sample pass: round-1 pool purity per arm per draw from the
    POST grid at fraction f.  In that block the round lines PRECEDE the
    `[arm] per-event post f=` line that names the arm, so the round-1 value
    is buffered until the arm line arrives (same walk as exp 125)."""
    out = {}
    for d in draws or DRAWS70.get(ds, (0, 1, 3, 4, 5)):
        p = _log70(ds, base, d)
        if p is None:
            continue
        cur, r1 = None, None
        for line in open(p, errors="ignore"):
            m = _POSTF.match(line)
            if m:
                cur, r1 = float(m.group(1)), None
                continue
            if cur is None:
                continue
            m = _PUR.match(line)
            if m and r1 is None:
                r1 = float(m.group(2))
                continue
            m = _ARMPOST.match(line)
            if m:
                if abs(cur - f) < 1e-9 and r1 is not None:
                    out.setdefault(m.group(1), {}).setdefault(d, r1)
                r1 = None
    return out


def _postf70(ds, base, metric, arm, f=F_PAPER, draws=None):
    """INJECTED-sample post geometry: per-draw `postf_{metric}_{arm}` at
    fraction f from the exp-70 npz.  Draws whose npz predates the key
    (runs before 2026-08-29) are skipped, so an empty dict renders as `--`
    -- never as the natural-pass value."""
    out = {}
    for d in draws or DRAWS70.get(ds, (0, 1, 3, 4, 5)):
        p = os.path.join(LOGS, "exp70", f"results_{ds}_{base}_ft70_h1_d{d}.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p, allow_pickle=True)
        key = f"postf_{metric}_{arm}"
        if key not in z.files or "post_fractions" not in z.files:
            continue
        fr = np.asarray(z["post_fractions"], dtype=float)
        i = int(np.argmin(np.abs(fr - f)))
        v = np.asarray(z[key], dtype=float).ravel()
        if abs(fr[i] - f) < 1e-9 and i < v.size and np.isfinite(v[i]):
            out[d] = float(v[i])
    return out


def _msd(v, p=3):
    v = [x for x in v if x is not None and np.isfinite(x)]
    if not v:
        return "--"
    if len(v) == 1:
        return fnum(v[0], p)
    return f"{np.mean(v):.{p}f}$\\pm${np.std(v, ddof=1):.{p}f}"


def t_draws():
    """Single-holdout purity across holdout draws, both DTD bases: discovery
    from a 2% injected sample (paper regime) with mean / sd / worst draw, and
    the whole-class natural pass as a reference column."""
    cells = [("dtd", "dino"), ("dtd", "lejepa")]
    data = {c: _draw_purities(*c) for c in cells}
    dataf = {c: _draw_purities_f(*c, F_PAPER, DRAWS70[c[0]]) for c in cells}
    arms = ["simclr-ft", "sigreg-ssl-ft", "nplm-bil-ft", "supcon-ft",
            "ss-ft", "nplm-sup-ft"]
    rows, have = [], 0
    for a in arms:
        cs = []
        for c in cells:
            v = list(data[c].get(a, {}).values())
            vf = list(dataf[c].get(a, {}).values())
            if vf:
                have += 1
                cs += [fnum(np.mean(vf)), fnum(np.std(vf, ddof=1), 3),
                       fnum(min(vf))]
            else:
                cs += ["--", "--", "--"]
            cs.append(fnum(np.mean(v)) if v else "--")
        rows.append(" & ".join([PRETTY.get(a, esc(a))] + cs) + r" \\")
    ndraw = max([len(v) for c in cells for v in dataf[c].values()] or [0])
    status = (f"DTD, {ndraw} holdout draws per base, single holdout; "
              f"{have}/{len(arms) * len(cells)} (arm, base) pairs present. "
              r"Purity read from the run logs (POST grid at "
              rf"$f={F_PAPER}$; `whole' = the natural pass).")
    return _wrap(
        "\n".join(rows),
        r"\textbf{Pool purity is stable across holdout draws, and only one "
        r"objective is stable across \emph{backbones}.} Round-1 pool purity "
        r"of discovery from a 2\% injected sample of the held-out class: "
        r"mean $\pm$ sd over draws, with the worst draw. The `whole' column "
        r"is the natural pass, in which the whole held-out class is present "
        r"in the unlabelled bank --- a different, easier regime, shown for "
        r"reference only. SupCon+SIGReg holds its purity on both backbones; "
        r"supervised distance-NPLM matches it on LeJEPA and loses a factor "
        r"of four on DINO.",
        "tab:draws", status,
        "lcccccccc",
        r"& \multicolumn{4}{c}{DTD / DINO} & \multicolumn{4}{c}{DTD / LeJEPA} \\"
        "\n" r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}" "\n"
        r"objective & mean & sd & min & whole & mean & sd & min & whole \\",
        size=r"\footnotesize")


# --------------------------------------------------------------- table 3 ---
def t_hardening():
    """Frozen-NP headline: the seed/draw variance split.

    NOTE: this reads ONLY the `*_variance` dicts.  The archived `*_gate`
    dicts are byte-identical between r1 and r2 (exp 139 computed the
    distribution summary once), so their min/median are not round-specific
    and must not be tabulated per round.
    """
    f = os.path.join(LOGS, "exp139", "hardening.json")
    if not os.path.exists(f):
        return None
    a = json.load(open(f))["analysis"]
    rows, n = [], None
    for rnd in ("r1", "r2"):
        v = a.get(f"{rnd}_variance", {})
        if not v:
            continue
        n = a.get(f"{rnd}_gate", {}).get("n", n)
        ss, sd = v.get("sd_within_seed"), v.get("sd_between_draw")
        ratio = fnum(sd / ss, 1) + r"$\times$" if ss and sd else "--"
        rows.append(" & ".join([
            rnd, str(v.get("n_strata", "--")), fnum(v.get("grand_mean")),
            fnum(ss), fnum(sd), ratio]) + r" \\")
    if not rows:
        return None
    status = (f"galaxy10, 3 backbones $\\times$ 3 objectives $\\times$ 5 draws "
              f"$\\times$ 3 seeds ($+1$ archived) $= {n}$ cells; variance "
              f"stratified within (cell, objective). Round-level distribution "
              f"summaries are not reported: the archive stores one summary for "
              f"both rounds.")
    return _wrap(
        "\n".join(rows),
        r"\textbf{The variance that matters is the holdout draw, not the "
        r"seed.} One-way decomposition of frozen-anchor round-1 purity, "
        r"stratified within (cell, objective) so the seed term is measured "
        r"at fixed draw. Seed-to-seed spread is an eighth of draw-to-draw "
        r"spread in both rounds, so the single-seed purities elsewhere in "
        r"this paper carry a caveat about which class was held out and not "
        r"about which seed was run.",
        "tab:hardening", status,
        "lccccc",
        r"round & strata & mean $\Pi$ & sd$_{\text{seed}}$ & "
        r"sd$_{\text{draw}}$ & ratio \\")


# --------------------------------------------------------------- table 4 ---
def t_2x2():
    """The construction x discovery interaction, per-event power."""
    paths = sorted(glob.glob(os.path.join(LOGS, "exp134",
                                          "postdisc_*_ft134c.json")))
    if not paths:
        return None
    rows, n_missing = [], 0
    for p in paths:
        name = cell_name(p)
        ds, base = name.rsplit("_", 1)
        pre, _ = archived_rows(ds, base)
        for r in cell_2x2(json.load(open(p)), pre):
            if r["metric"] != "perevt":
                continue
            if r["missing"]:
                n_missing += 1
                rows.append(" & ".join(
                    [CELL.get(name, esc(name)),
                     CTOR.get(r["obj"], esc(r["obj"])),
                     KIND.get(r["kind"], r["kind"])] + ["--"] * 7) + r" \\")
                continue
            rows.append(" & ".join([
                CELL.get(name, esc(name)),
                CTOR.get(r["obj"], esc(r["obj"])),
                KIND.get(r["kind"], r["kind"]), fnum(r["A"]), fnum(r["B"]), fnum(r["C"]), fnum(r["D"]),
                f"{r['main_disc']:+.3f}", f"{r['main_ctor']:+.3f}",
                rf"$\mathbf{{{r['interaction']:+.3f}}}$"
                if r["interaction"] > 0.10 else f"{r['interaction']:+.3f}",
            ]) + r" \\")
    done = len(rows) - n_missing
    status = (f"{len(paths)} cells $\\times$ 2 constructions $\\times$ 2 "
              f"variants; {done} complete, {n_missing} lacking a "
              r"pre-discovery child. One seed per cell.")
    return _wrap(
        "\n".join(rows),
        r"\textbf{Discovery $\times$ construction, per-event power at "
        r"$\alpha{=}0.05$.} Each row is one $2\times2$: $A$ is the parent "
        r"space alone, $B$ the parent after the discovery loop, $C$ the "
        r"construction built on the parent \emph{without} discovery, and $D$ "
        r"both together. The two main effects are $B-A$ (discovery) and "
        r"$C-A$ (construction). The residual child is trained under the loss in "
        r"the second column and then either used alone (\emph{bare}) or "
        r"concatenated with its parent (\emph{concat}). The interaction "
        r"$\Delta_{\text{int}}=(D-C)-(B-A)$ is zero if the two simply add, "
        r"and positive only if they do more together than apart. Bold marks "
        r"$\Delta_{\text{int}}>0.10$. Note that the two largest \emph{totals} "
        r"in the table (both galaxy10) have near-zero interaction: they are "
        r"main effects, not composition.",
        "tab:2x2", status,
        "lllccccccc",
        r"cell & child loss & used as & $A$ & $B$ & $C$ & $D$ & "
        r"$B{-}A$ & $C{-}A$ & $\Delta_{\text{int}}$ \\", size=r"\footnotesize")


# --------------------------------------------------------------- table 5 ---
def t_baserate():
    """Label-free base-rate estimation on clean CIFAR-100 banks."""
    f = os.path.join(LOGS, "exp138", "brate_cifar100_clean.json")
    if not os.path.exists(f):
        return None
    d = json.load(open(f))
    rows, n_ok, n_close = [], 0, 0
    for nm in sorted(d):
        r = d[nm]
        bt = r.get("b_true") or float("nan")
        est = r.get("estimators", {})
        rule = r.get("rule_default") or r.get("rule_n30") or {}
        ok = bool(rule.get("ok"))
        n_ok += ok
        tv = est.get("tv")
        ratio = (tv / bt) if (tv is not None and bt) else None
        if ratio is not None and 0.2 <= ratio <= 5.0:
            n_close += 1
        rows.append(" & ".join([
            esc(nm),
            fnum(ratio, 2) if ratio is not None else "--",
            fnum((est.get("mass") / bt) if est.get("mass") is not None else None, 2),
            "yes" if ok else "no",
            fnum(rule.get("q"), 4) if ok else "--",
        ]) + r" \\")
    status = (f"{len(rows)} spaces, clean CIFAR-100 banks, $b_{{\\text{{true}}}}"
              f"{{=}}0.01$; the rule engages on {n_ok}, and the best estimator "
              f"is within $5\\times$ on {n_close}.")
    return _wrap(
        "\n".join(rows),
        r"\textbf{The label-free cut is objective-specific, not universal.} "
        r"Ratio of estimated to true novel base rate at $b{=}0.01$; the rule "
        r"engages only where an NPLM critic sits on a SIGReg marginal, and "
        r"even then the resulting purity is only $0.046$--$0.101$ (against a "
        r"base rate of $0.01$). Reported as a "
        r"limit of the rule, not of the representations.",
        "tab:baserate", status,
        "lcccc",
        r"space & $\hat b/b$ (tv) & $\hat b/b$ (mass) & rule fires & $q^\star$ \\")


# ------------------------------------------------------- appendix tables ---
def t_residuals():
    """Every construction on the from-scratch CIFAR-100 trunks (exp 137)."""
    f = os.path.join(LOGS, "exp137", "residuals_cifar100.json")
    if not os.path.exists(f):
        return None
    d = json.load(open(f))
    parents = [k for k in d if k.endswith("(parent)")]
    rows = []
    for par in parents:
        p = d[par]
        stem = par.replace(" (parent)", "")
        rows.append(r"\multicolumn{6}{l}{\emph{parent: }" +
                    PRETTY.get(stem, esc(stem)) + r"} \\")
        rows.append(" & ".join(["\\quad (parent)", fnum(p.get("probe")),
                                "--", fnum(p.get("eucl")), "--",
                                fnum(p.get("perevt"), 2)]) + r" \\")
        for k in [x for x in d if x.startswith(stem + "->")]:
            c = d[k]
            rows.append(" & ".join([
                "\\quad " + CHILD.get(k.split("->", 1)[1],
                                      esc(k.split("->", 1)[1])),
                fnum(c.get("probe")), f"{c.get('probe', 0) - p.get('probe', 0):+.3f}",
                fnum(c.get("eucl")), f"{c.get('eucl', 0) - p.get('eucl', 0):+.3f}",
                fnum(c.get("perevt"), 2)]) + r" \\")
    status = (f"{len(parents)} parents $\\times$ 4 constructions on the "
              r"from-scratch CIFAR-100 trunks; one seed per cell.")
    return _wrap(
        "\n".join(rows),
        r"\textbf{All constructions on the leakage-free CIFAR-100 trunks.} "
        r"$\Delta$ columns are against that parent. The bare residual "
        r"generally loses probe accuracy and the concatenation recovers it; "
        r"the detection gain is large and nearly free.",
        "tab:residuals", status, "lccccc",
        r"construction & probe & $\Delta$ & Eucl.\ AUC & $\Delta$ & "
        r"per-event \\")


def t_seeds():
    """The multi-seed CIFAR-10 champions (exp 59): seeds 0/1/2."""
    # NOT glob("...cifar10*.npz") -- that also matches cifar100, which would
    # pool two DATASETS into one "seed spread" and manufacture a variance.
    # Seeds are the explicit suffixes only.
    stem = os.path.join(LOGS, "exp59", "nplm_residual_concat_cifar10")
    fs = [p for p in (f"{stem}.npz", f"{stem}_s1.npz", f"{stem}_s2.npz")
          if os.path.exists(p)]
    if not fs:
        return None
    # exp 59 uses a DIFFERENT npz schema from exp 71: the arm list is `arms`,
    # and keys are "<metric>_<arm>" (with "probe_post_<arm>" for post), not
    # exp 71's "<arm>__<metric>".  Guessing the exp-71 form here silently
    # yields an empty table, so the schema is pinned by a test.
    per = {}
    for f in fs:
        d = np.load(f, allow_pickle=True)
        if "arms" not in d.files:
            continue
        for a in d["arms"]:
            a = str(a)
            for label, key in (("probe_pre", f"probe_{a}"),
                               ("probe_post", f"probe_post_{a}"),
                               ("eucl", f"eucl_{a}"),
                               ("mahaT", f"mahaT_{a}")):
                if key in d.files:
                    per.setdefault(a, {}).setdefault(label, []).append(
                        float(d[key]))
    rows = []
    for sp in sorted(per):
        v = per[sp]
        cells = []
        for m in ("probe_pre", "probe_post", "eucl", "mahaT"):
            a = v.get(m, [])
            cells.append(f"{np.mean(a):.3f}$\\pm${np.std(a, ddof=1):.3f}"
                         if len(a) > 1 else (fnum(a[0]) if a else "--"))
        rows.append(" & ".join([esc(sp)] + cells + [str(len(v.get("eucl", [])))])
                    + r" \\")
    ns = max((len(v.get("eucl", [])) for v in per.values()), default=0)
    status = (f"CIFAR-10, {len(rows)} spaces, up to {ns} seeds; "
              r"one of the few multi-seed cells in the campaign.")
    return _wrap("\n".join(rows) or r"-- & -- & -- & -- & -- & -- \\",
                 r"\textbf{Multi-seed CIFAR-10 champions.} Mean $\pm$ sd over "
                 r"seeds. Seed spread here is small, consistent with "
                 r"Table~\ref{tab:hardening}.",
                 "tab:seeds", status, "lccccc",
                 r"space & probe (pre) & probe (post) & Eucl.\ AUC & Maha. & "
        r"seeds \\")


_AGG = re.compile(r"^\|\s*([a-z0-9\-]+)\s*\|(.+)\|\s*$")


def _agg_purity(ds, base):
    """(mean, sd) of round-1 pool purity per arm, from the exp-125 aggregate
    markdown.  Column 10 of that table is `purity r1` as `m+-s`."""
    p = os.path.join(LOGS, f"exp125_{ds}_{base}_agg.md")
    if not os.path.exists(p):
        return {}, 0
    out, ndraw = {}, 0
    for line in open(p, errors="ignore"):
        m = re.search(r"(\d+) draws", line)
        if m and not ndraw:
            ndraw = int(m.group(1))
        m = _AGG.match(line.strip())
        if not m:
            continue
        arm, rest = m.group(1), [c.strip() for c in m.group(2).split("|")]
        if len(rest) < 10 or "+-" not in rest[8]:
            continue
        try:
            mu, sd = rest[8].split("+-")
            out[arm] = (float(mu), float(sd))
        except ValueError:
            continue
    return out, ndraw


def t_galaxy():
    """galaxy10: the same dataset under two discovery procedures.

    galaxy10 is the only dataset in the study that NO backbone has seen, so
    it carries the most weight -- and it is also where the choice of
    procedure matters most.  The fine-tuning loop's distance pool (paper
    regime: discovery from a 2% injected sample; the natural pass with the
    whole class present is shown alongside, labelled) and the frozen-anchor
    density-ratio pool disagree by an order of magnitude on the same
    draws."""
    bases = ["dino", "lejepa", "visreg"]
    inj = {b: _draw_purities_f("galaxy10", b) for b in bases}
    nat = {b: _draw_purities("galaxy10", b) for b in bases}
    f = os.path.join(LOGS, "exp139", "hardening.json")
    frz = json.load(open(f))["analysis"]["per_cell"] if os.path.exists(f) else {}
    arms = ["simclr-ft", "sigreg-ssl-ft", "nplm-bil-ft", "supcon-ft",
            "ss-ft", "nplm-sup-ft"]
    rows, n_inj, n_frz = [], 0, 0
    for a in arms:
        cells = []
        for b in bases:
            v = list(inj[b].get(a, {}).values())
            n_inj += bool(v)
            cells.append(_msd(v))
        for b in bases:
            cells.append(_msd(list(nat[b].get(a, {}).values())))
        for b in bases:
            c = (frz.get(f"galaxy10:{b}") or {}).get(a)
            if c:
                n_frz += 1
                cells.append(f"{c['mean']:.3f}$\\pm${c['sd']:.3f}")
            else:
                cells.append("--")
        rows.append(" & ".join([PRETTY.get(a, esc(a))] + cells) + r" \\")
    nd = max([len(v) for b in bases for v in inj[b].values()] or [0])
    status = (f"galaxy10, {len(bases)} backbones; fine-tuning loop over "
              f"{nd} draws ({n_inj}/{len(arms) * len(bases)} cells present, "
              f"injected and whole-class passes from the same runs), "
              f"frozen-anchor pool over 5 draws $\\times$ 3 seeds "
              f"({n_frz}/{len(arms) * len(bases)} present --- that arm of the "
              r"study covers the three supervised objectives only).")
    return _wrap(
        "\n".join(rows),
        r"\textbf{galaxy10, the dataset no backbone has seen, under two "
        r"discovery procedures.} Round-1 pool purity, mean $\pm$ sd over "
        r"draws. Left: the fine-tuning loop's distance pool, discovery from "
        r"a 2\% injected sample of the held-out class (the paper's regime). "
        r"Middle: the same loop's natural pass with the whole held-out class "
        r"present in the bank --- a different, easier regime, for reference. "
        r"Right: the frozen-anchor density-ratio pool, also with the whole "
        r"class present in the bank (its injected-sample counterpart is not "
        r"yet measured), so the like-for-like comparison is middle vs right: "
        r"same dataset, same draws, a factor of three to five in purity from "
        r"the procedure alone. The left block shows how much harder the "
        r"paper's own regime is for the loop.",
        "tab:galaxy", status, "lccccccccc",
        r"& \multicolumn{3}{c}{loop, 2\% injected} & "
        r"\multicolumn{3}{c}{loop, whole class present} & "
        r"\multicolumn{3}{c}{frozen-anchor pool} \\"
        "\n" r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}" "\n"
        r"objective & DINO & LeJEPA & VISReg & DINO & LeJEPA & VISReg & "
        r"DINO & LeJEPA & VISReg \\", wide=True)


def t_inventory():
    """Every numbered study, with whether its results are on disk."""
    exps = sorted(glob.glob(os.path.join(REPO, "experiments", "*.py")))
    rows, n_done = [], 0
    for p in exps:
        b = os.path.basename(p)
        m = re.match(r"(\d+)([a-z]?)_(.+)\.py$", b)
        if not m:
            continue
        num, sfx, name = m.groups()
        pats = [os.path.join(LOGS, f"exp{num}{sfx}"),
                os.path.join(LOGS, f"exp{num}")]
        got = any(os.path.isdir(x) and os.listdir(x) for x in pats) or \
            bool(glob.glob(os.path.join(LOGS, f"exp{num}{sfx}*")))
        n_done += got
        rows.append(" & ".join([
            f"{num}{sfx}", esc(name.replace("_", " ")),
            r"\checkmark" if got else "--"]) + r" \\")
    status = (f"{len(rows)} numbered studies, {n_done} with results on disk, "
              f"{len(rows) - n_done} not yet run or run only on the GPU host.")
    return _wrap("\n".join(rows),
                 r"\textbf{Inventory of the full study suite.} A checkmark "
                 r"means result files for that study are present in this "
                 r"repository; it is not a claim that the study is complete. "
                 r"Studies without results are listed so the reader can see "
                 r"the shape of what is outstanding.",
                 "tab:inventory", status, "llc",
                 r"\# & study & results \\", long=True)


# ------------------------------------------------------- GCD dose table ---
GCD_DOSES = (0.05, 0.2, 0.5, 1.0)


def _gcd_files(ds, base, F, d):
    """(npz, log) for the exp-146/146b GCD arms at dose F (1.0 = exp 146)."""
    sfx = "_gcd" if F >= 1.0 else f"_u{F:g}_gcd"
    npz = os.path.join(LOGS, "exp70", f"results_{ds}_{base}_ft70_h1_d{d}{sfx}.npz")
    log = (os.path.join(LOGS, f"exp146_{ds}_{base}_h1_d{d}.log") if F >= 1.0
           else os.path.join(LOGS, f"exp146b_{ds}_{base}_h1_d{d}_u{F:g}.log"))
    return npz, log


def _purity_f_log(path, f=F_PAPER):
    """Round-1 pool purity per arm from one log's POST grid at fraction f."""
    out, cur, r1 = {}, None, None
    if not os.path.exists(path):
        return out
    for line in open(path, errors="ignore"):
        m = _POSTF.match(line)
        if m:
            cur = float(m.group(1)); r1 = None
            continue
        if cur is None:
            continue
        m = _PUR.match(line)
        if m and r1 is None:
            r1 = float(m.group(2))
        m = _ARMPOST.match(line)
        if m:
            if abs(cur - f) < 1e-9 and r1 is not None:
                out[m.group(1)] = r1
            r1 = None
    return out


def _dose_rows(ds, base, arms_ref=("supcon-ft", "ss-ft")):
    """Rows (label, arm, probe, mahaT, per-event post@f, purity@f) per dose."""
    rows = []
    draws = DRAWS70[ds]

    def collect(npz_of, log_of, arm):
        pr, ma, pe, pu = [], [], [], []
        for d in draws:
            npz, log = npz_of(d), log_of(d)
            if not os.path.exists(npz):
                continue
            z = np.load(npz, allow_pickle=True)
            if f"probe_{arm}" not in z.files:
                continue
            pr.append(float(z[f"probe_{arm}"])); ma.append(float(z[f"mahaT_{arm}"]))
            fr = np.asarray(z["post_fractions"], dtype=float)
            i = int(np.argmin(np.abs(fr - F_PAPER)))
            v = np.asarray(z[f"perevent_{arm}_post"], dtype=float).ravel()
            if i < v.size:
                pe.append(float(v[i]))
            u = _purity_f_log(log).get(arm)
            if u is not None:
                pu.append(u)
        return pr, ma, pe, pu

    for arm in arms_ref:
        r = collect(lambda d: os.path.join(LOGS, "exp70", f"results_{ds}_{base}_ft70_h1_d{d}.npz"),
                    lambda d: _log70(ds, base, d) or "", arm)
        if r[0]:
            rows.append(("0\\%", arm) + r)
    for F in GCD_DOSES:
        for arm in ("gcd-ft", "gcd-sigreg-ft"):
            r = collect(lambda d, F=F: _gcd_files(ds, base, F, d)[0],
                        lambda d, F=F: _gcd_files(ds, base, F, d)[1], arm)
            if r[0]:
                rows.append((f"{100 * F:g}\\%", arm) + r)
    return rows


def t_gcd_dose():
    """Exp 146/146b: the GCD representation loss (SimCLR over the whole
    corpus incl. the unlabelled novel images + SupCon on the labelled) as a
    function of how much of the novel class the corpus contains."""
    body, n = [], 0
    for ds, base in (("galaxy10", "lejepa"), ("dtd", "dino")):
        rows = _dose_rows(ds, base)
        if not rows:
            continue
        body.append(r"\multicolumn{6}{l}{\emph{" + esc(f"{ds} / {base}") + r"}} \\")
        for dose, arm, pr, ma, pe, pu in rows:
            n += 1
            body.append(" & ".join(["\\quad " + dose, PRETTY.get(arm, esc(arm)),
                                    _msd(pr), _msd(ma), _msd(pe, 2), _msd(pu)]) + r" \\")
    if not body:
        return None
    status = (f"{n} (dose, objective) rows; single holdout, 5 draws each, one seed; "
              r"`0\%' rows are the archived labelled-only arms on the same draws; the GCD "
              r"arms train on the whole corpus with the held-out images unlabelled, a random "
              r"fraction of them kept per dose; per-event power and purity are the POST grid "
              rf"at $f={F_PAPER}$ (discovery from a 2\% injected sample).")
    return _wrap(
        "\n".join(body),
        r"\textbf{Seeing the unlabelled novel images during representation "
        r"learning is a dose, not a switch.} GCD's representation loss "
        r"(SimCLR over every image, held-out ones included with masked labels, "
        r"plus SupCon on the labelled ones, $0.65{:}0.35$; `+SIGReg' adds our "
        r"marginal at $\lambda{=}5$) scored on our battery as a function of the "
        r"fraction of the held-out class present in the training corpus. The "
        r"gain in calibration, per-event power and pool purity is monotone in "
        r"the dose and absent at the contamination levels this paper is about "
        r"($\le 5\%$); on DTD the plain GCD loss never leaves SupCon's "
        r"neighbourhood below $100\%$ and it is the SIGReg marginal that "
        r"carries the effect.",
        "tab:gcd_dose", status, "llcccc",
        r"novel class & objective & probe & mahaT & per-event & purity \\",
        size=r"\footnotesize")



TABLES = [("objectives", t_objectives), ("draws", t_draws),
          ("hardening", t_hardening), ("2x2", t_2x2),
          ("baserate", t_baserate),
          # appendix
          ("galaxy", t_galaxy), ("residuals", t_residuals),
          ("seeds", t_seeds), ("gcd_dose", t_gcd_dose)]


def selftest():
    ok = True
    assert esc("a_b") == r"a\_b"
    assert fnum(None) == "--" and fnum(float("nan")) == "--"
    assert fnum(0.5) == "0.500" and fnum(0.5, 2) == "0.50"
    print("  [ok] formatting helpers")
    for name, fn in TABLES:
        try:
            t = fn()
        except Exception as e:                       # noqa: BLE001
            print(f"  [FAIL] {name}: {type(e).__name__}: {e}")
            ok = False
            continue
        if t is None:
            print(f"  [skip] {name}: source log absent")
            continue
        env = "longtable" if r"\begin{longtable}" in t else "tabular"
        assert t.count(rf"\begin{{{env}}}") == 1 and t.count(rf"\end{{{env}}}") == 1
        assert r"\caption" in t and r"\label" in t and t.startswith("% STATUS:")
        ncol = t.split(f"{{{env}}}{{")[1].split("}")[0]
        ncol = sum(1 for c in ncol if c in "lcr")
        for line in t.split(r"\midrule")[1].split(r"\bottomrule")[0].strip().split("\n"):
            if not line.strip().endswith(r"\\"):
                continue
            # a \multicolumn{k} cell stands for k columns, not one
            spans = [int(x) for x in re.findall(r"\\multicolumn\{(\d+)\}", line)]
            n = line.count("&") + 1 - len(spans) + sum(spans)
            assert n == ncol, f"{name}: row spans {n} cols, colspec has {ncol}: {line[:70]}"
        print(f"  [ok] {name}: {t.count(chr(92) + chr(92))} rows, {ncol} columns")
    print("SELFTEST PASS" if ok else "SELFTEST FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stdout", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())
    os.makedirs(args.out, exist_ok=True)
    for name, fn in TABLES:
        t = fn()
        if t is None:
            print(f"  [skip] {name}: source absent")
            continue
        if args.stdout:
            print(t)
        else:
            p = os.path.join(args.out, f"{name}.tex")
            open(p, "w").write(t)
            print(f"  wrote {os.path.relpath(p, REPO)}  "
                  f"({t.splitlines()[0][10:80]})")


if __name__ == "__main__":
    main()
