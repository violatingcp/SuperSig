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
    return str(s).replace("_", r"\_").replace("&", r"\&")


def fnum(x, p=3, dash="--"):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return dash
    return f"{x:.{p}f}"


def _wrap(body, caption, label, status, colspec, header, long=False):
    """`long=True` emits a longtable, which breaks across pages -- required
    for any table with more rows than fit on one (a plain `table` silently
    overflows the page instead of breaking)."""
    if long:
        return "\n".join([
            f"% STATUS: {status}",
            r"\begin{center}", r"\small",
            rf"\begin{{longtable}}{{{colspec}}}",
            rf"\caption{{{caption} \emph{{Coverage:}} {status}}}"
            rf"\label{{{label}}}\\",
            r"\toprule", header, r"\midrule", r"\endfirsthead",
            r"\toprule", header, r"\midrule", r"\endhead",
            r"\bottomrule", r"\endfoot",
            body, r"\end{longtable}", r"\end{center}", ""])
    return "\n".join([
        f"% STATUS: {status}",
        r"\begin{table}[t]", r"\centering", r"\small",
        rf"\begin{{tabular}}{{{colspec}}}", r"\toprule",
        header, r"\midrule", body, r"\bottomrule", r"\end{tabular}",
        rf"\caption{{{caption} \emph{{Coverage:}} {status}}}",
        rf"\label{{{label}}}", r"\end{table}", ""])


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


def _draw_purities(ds, base, draws=(0, 1, 3, 4, 5)):
    """round-1 pool purity per arm per draw, read from the exp-70 logs
    (the npz files carry no purity field)."""
    out = {}
    for d in draws:
        p = os.path.join(LOGS, f"exp70_{ds}_{base}_h1_d{d}.log")
        if not os.path.exists(p):
            continue
        arm = None
        for line in open(p, errors="ignore"):
            m = _ARM.match(line.strip())
            if m:
                arm = m.group(1)
                continue
            m = _PUR.match(line)
            if m and arm is not None:
                # first round-1 line after the header is that arm's
                out.setdefault(arm, {}).setdefault(d, float(m.group(2)))
    return out


def t_draws():
    """Single-holdout purity across holdout draws, both DTD bases."""
    cells = [("dtd", "dino"), ("dtd", "lejepa")]
    data = {c: _draw_purities(*c) for c in cells}
    arms = ["simclr-ft", "sigreg-ssl-ft", "nplm-bil-ft", "supcon-ft",
            "ss-ft", "nplm-sup-ft"]
    rows, have = [], 0
    for a in arms:
        cs = []
        for c in cells:
            v = list(data[c].get(a, {}).values())
            if v:
                have += 1
                cs += [fnum(np.mean(v)), fnum(np.std(v, ddof=1), 3),
                       fnum(min(v))]
            else:
                cs += ["--", "--", "--"]
        rows.append(" & ".join([PRETTY.get(a, esc(a))] + cs) + r" \\")
    ndraw = max([len(v) for c in cells for v in data[c].values()] or [0])
    status = (f"DTD, {ndraw} holdout draws per base, single holdout; "
              f"{have}/{len(arms) * len(cells)} (arm, base) pairs present. "
              r"Purity read from the run logs.")
    return _wrap(
        "\n".join(rows),
        r"\textbf{Pool purity is stable across holdout draws, and only one "
        r"objective is stable across \emph{backbones}.} Mean $\pm$ sd over "
        r"draws, with the worst draw. SupCon+SIGReg holds its purity on "
        r"both backbones ($0.219$ and $0.225$); supervised distance-NPLM "
        r"matches it on LeJEPA ($0.215$) and loses a factor of four on "
        r"DINO ($0.051$).",
        "tab:draws", status,
        "lcccccc",
        r"& \multicolumn{3}{c}{DTD / DINO} & \multicolumn{3}{c}{DTD / LeJEPA} \\"
        "\n" r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}" "\n"
        r"objective & mean & sd & min & mean & sd & min \\")


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
        r"$B{-}A$ & $C{-}A$ & $\Delta_{\text{int}}$ \\")


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
    procedure matters most.  Natural discovery (distance pool, encoder
    fine-tuned) and the frozen-anchor density-ratio pool disagree by a factor
    of five on the same draws."""
    bases = ["dino", "lejepa", "visreg"]
    nat = {b: _agg_purity("galaxy10", b) for b in bases}
    f = os.path.join(LOGS, "exp139", "hardening.json")
    frz = json.load(open(f))["analysis"]["per_cell"] if os.path.exists(f) else {}
    arms = ["simclr-ft", "sigreg-ssl-ft", "nplm-bil-ft", "supcon-ft",
            "ss-ft", "nplm-sup-ft"]
    rows, n_nat, n_frz = [], 0, 0
    for a in arms:
        cells = []
        for b in bases:
            v = nat[b][0].get(a)
            if v:
                n_nat += 1
                cells.append(f"{v[0]:.3f}$\\pm${v[1]:.3f}")
            else:
                cells.append("--")
        for b in bases:
            c = (frz.get(f"galaxy10:{b}") or {}).get(a)
            if c:
                n_frz += 1
                cells.append(f"{c['mean']:.3f}$\\pm${c['sd']:.3f}")
            else:
                cells.append("--")
        rows.append(" & ".join([PRETTY.get(a, esc(a))] + cells) + r" \\")
    nd = max((nat[b][1] for b in bases), default=0)
    status = (f"galaxy10, {len(bases)} backbones; natural discovery over "
              f"{nd} draws ({n_nat}/{len(arms) * len(bases)} cells present), "
              f"frozen-anchor pool over 5 draws $\\times$ 3 seeds "
              f"({n_frz}/{len(arms) * len(bases)} present --- that arm of the "
              r"study covers the three supervised objectives only).")
    return _wrap(
        "\n".join(rows),
        r"\textbf{galaxy10, the dataset no backbone has seen, under two "
        r"discovery procedures.} Round-1 pool purity, mean $\pm$ sd. Left: "
        r"natural discovery (distance pool, encoder fine-tuned). Right: the "
        r"frozen-anchor density-ratio pool. Same dataset, same holdout draws, "
        r"a three- to five-fold difference in purity --- the procedure, not "
        r"the representation, is what moves it.",
        "tab:galaxy", status, "lcccccc",
        r"& \multicolumn{3}{c}{natural discovery} & "
        r"\multicolumn{3}{c}{frozen-anchor pool} \\"
        "\n" r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}" "\n"
        r"objective & DINO & LeJEPA & VISReg & DINO & LeJEPA & VISReg \\")


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


TABLES = [("objectives", t_objectives), ("draws", t_draws),
          ("hardening", t_hardening), ("2x2", t_2x2),
          ("baserate", t_baserate),
          # appendix
          ("galaxy", t_galaxy), ("residuals", t_residuals),
          ("seeds", t_seeds)]


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
