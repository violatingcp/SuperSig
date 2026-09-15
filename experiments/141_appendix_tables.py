"""Experiment 141: the paper's DETAILED appendix tables, generated from the logs.

Companion to 140_paper_tables.py (whose helpers it reuses); emits into
docs/tables/app_*.tex.  Same contract: nothing computed here, every table
carries a coverage line, blanks are `--`, longtables where rows exceed a page.

    python experiments/141_appendix_tables.py            # write docs/tables/app_*.tex
    python experiments/141_appendix_tables.py --selftest
"""
import argparse
import glob
import importlib
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
t140 = importlib.import_module("140_paper_tables")
REPO, LOGS, OUT = t140.REPO, t140.LOGS, t140.OUT
esc, fnum, _wrap, PRETTY = t140.esc, t140.fnum, t140._wrap, t140.PRETTY

ARMS_SCRATCH = ["supcon", "ssig", "nplmcw", "supsig", "nplmsd", "simclr", "visreg", "nplm"]
ARMS_70 = ["supcon-ft", "ss-ft", "nplm-sup-ft", "simclr-ft", "sigreg-ssl-ft", "nplm-bil-ft"]
G10_DRAWS = (0, 3, 5, 7, 8)
DTD_DRAWS = (0, 1, 3, 4, 5)


def msd(v, p=3):
    v = [x for x in v if x is not None and np.isfinite(x)]
    if not v:
        return "--"
    if len(v) == 1:
        return fnum(v[0], p)
    return f"{np.mean(v):.{p}f}$\\pm${np.std(v):.{p}f}"


# ------------------------------------------------------- scratch CIFAR ---
def _master(ds):
    f = os.path.join(LOGS, "exp136", f"master_{ds}.json")
    return json.load(open(f)) if os.path.exists(f) else None


_H = {"cifar10": 4, "cifar100": 4}      # the master JSONs are the holdout-4 cells


def d68_injected(ds, tag="", f=None):
    """Exp-68 INJECTED-sample pass, read from disk at table time (so Block S5
    fills it in place without re-running exp 136): {arm: {"probe","eucl",
    "mahaT","purity"}} at fraction f.  Values come from postf_* npz keys and
    the POST-grid log lines; anything absent is simply missing -> `--`."""
    f = t140.F_PAPER if f is None else f
    out = {}
    zp = os.path.join(LOGS, "exp68", f"scratch_discovery_{ds}{tag}.npz")
    z = np.load(zp, allow_pickle=True) if os.path.exists(zp) else None
    if z is not None and "fractions" in z.files:
        fr = np.asarray(z["fractions"], dtype=float); i = int(np.argmin(np.abs(fr - f)))
        if abs(fr[i] - f) < 1e-9:
            for k in z.files:
                m = re.match(r"postf_(probe|eucl|mahaT|mahaPC)_(.+)$", k)
                if m:
                    v = np.asarray(z[k], dtype=float).ravel()
                    if i < v.size and np.isfinite(v[i]):
                        out.setdefault(m.group(2), {})[m.group(1)] = float(v[i])
    h = _H.get(ds, 4) if not tag else int(tag[2:])
    for fn in (os.path.join(LOGS, f"exp68_{ds}_h{h}.log"),
               os.path.join(LOGS, "exp68", f"exp68_scratch_discovery_{ds}{tag}.log")):
        if os.path.exists(fn):
            break
    else:
        return out
    cur, r1 = None, None
    for line in open(fn, errors="ignore"):
        m = t140._POSTF.match(line)
        if m:
            cur, r1 = float(m.group(1)), None; continue
        if cur is None:
            continue
        m = t140._PUR.match(line)
        if m and r1 is None:
            r1 = float(m.group(2)); continue
        m = t140._ARMPOST.match(line)
        if m:
            if abs(cur - f) < 1e-9 and r1 is not None:
                out.setdefault(m.group(1), {})["purity"] = r1
            r1 = None
    return out


def t_scratch_pre(ds):
    m = _master(ds)
    if m is None:
        return None
    rows, n = [], 0
    for a in ARMS_SCRATCH:
        if a not in m:
            rows.append(f"{PRETTY.get(a, esc(a))} & " + " & ".join(["--"] * 10) + r" \\"); continue
        p = m[a]["pre"]; n += 1
        rows.append(" & ".join([PRETTY.get(a, esc(a)), fnum(p["probe"]), fnum(p["top1"]),
                                fnum(p["acc"]), fnum(p["sup_auc"]), fnum(p["eucl"]),
                                fnum(p["mahaT"]), fnum(p["mahaPC"]), fnum(p["lid"]),
                                fnum(p["perevt"], 2), fnum(p.get("probe_sd"), 3)]) + r" \\")
    tag = "CIFAR-10" if ds == "cifar10" else "CIFAR-100"
    b = "0.10" if ds == "cifar10" else "0.01"
    status = (f"{n}/8 objectives; {tag} from random init, 100-D, holdout class 4 "
              f"(single holdout, $b={b}$), seed 0; probe and top-1 over 3 probe restarts.")
    return _wrap("\n".join(rows),
                 rf"\textbf{{{tag} leakage-free lineage: the frozen-space battery.}} "
                 r"probe = holdout-vs-rest linear AUC; top-1 = supervised linear probe on "
                 r"seen classes; acc = nearest-centroid; supAUC = macro OvR AUC of the "
                 r"proto posterior; eucl / mahaT / mahaPC = novelty AUC of min-centroid "
                 r"distance, tied and per-class Mahalanobis; lid = local-intrinsic-dimension "
                 r"AUC; per-ev = per-event power at $\alpha{=}0.05$.",
                 f"tab:app_{ds}_pre", status, "lcccccccccc",
                 r"objective & probe & top-1 & acc & supAUC & eucl & mahaT & mahaPC & lid & per-ev & probe sd \\", wide=True)


def t_scratch_power(ds):
    m = _master(ds)
    if m is None:
        return None
    fr = None
    rows, n = [], 0
    for a in ARMS_SCRATCH:
        pw = m.get(a, {}).get("pre_power")
        if not pw:
            rows.append(f"{PRETTY.get(a, esc(a))} & " + " & ".join(["--"] * 9) + r" \\"); continue
        fr = pw["fractions"]; n += 1
        pick = [f for f in (0.01, 0.02, 0.05, 0.1) if f in fr][:3]
        cells = []
        for stat in ("sparker", "maha", "mmd"):
            cells += [fnum(pw[stat][fr.index(f)], 2) for f in pick]
        rows.append(" & ".join([PRETTY.get(a, esc(a))] + cells) + r" \\")
    pick = [f for f in (0.01, 0.02, 0.05, 0.1) if fr and f in fr][:3] if fr else [0.01, 0.02, 0.05]
    tag = "CIFAR-10" if ds == "cifar10" else "CIFAR-100"
    status = (f"{n}/8 objectives; annealed-$\\sigma$ SparKer, parametric Mahalanobis and "
              r"MMD two-sample power at $\alpha{=}0.05$, $N_D{=}5000$, 200 null / 50 signal toys; "
              f"fractions {', '.join(str(f) for f in pick)}.")
    head = " & ".join([f"${f}$" for f in pick])
    return _wrap("\n".join(rows),
                 rf"\textbf{{{tag} leakage-free lineage: dataset-level detection power}} on the "
                 r"frozen space, before any discovery. Signal fraction $f$ of held-out images "
                 r"injected into a seen-class test sample.",
                 f"tab:app_{ds}_power", status, "lccccccccc",
                 r"& \multicolumn{3}{c}{SparKer} & \multicolumn{3}{c}{Mahalanobis} & \multicolumn{3}{c}{MMD} \\"
                 "\n" r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}" "\n"
                 f"objective & {head} & {head} & {head} " r"\\")


def t_scratch_pools(ds):
    """Loop columns are the paper regime (discovery from a 2% injected
    sample, exp-68 POST grid); the natural pass (whole class present) is the
    labelled pair on the right."""
    m = _master(ds)
    if m is None:
        return None
    inj = d68_injected(ds)
    rows, n, n_inj = [], 0, 0
    for a in ARMS_SCRATCH:
        if a not in m:
            rows.append(f"{PRETTY.get(a, esc(a))} & " + " & ".join(["--"] * 11) + r" \\"); continue
        r = m[a]; z = r["frozen"]; n += 1
        l10, l30 = r.get("legal_cut", {}), r.get("legal_cut_n30", {})
        d68 = r.get("discovery68") or {}
        ij = inj.get(a, {})
        n_inj += "probe" in ij
        L = lambda l: (fnum(l.get("purity")) if l.get("ok") else "ref.") if l else "--"
        rows.append(" & ".join([
            PRETTY.get(a, esc(a)),
            fnum(z["dist|frozen"]["purity"][0]), fnum(z["np|frozen"]["purity"][0]),
            fnum(z["np|frozen"]["purity"][1]), fnum(z["np|bn-adapt"]["purity"][1]),
            L(l10), L(l30),
            fnum(ij.get("purity")), fnum(ij.get("probe")), fnum(ij.get("mahaT")),
            fnum((d68.get("purity") or [None])[0]) if d68 else "--",
            (f"{fnum(d68['probe_pre'])}$\\to${fnum(d68['probe_post'])}" if d68 else "--")]) + r" \\")
    tag = "CIFAR-10" if ds == "cifar10" else "CIFAR-100"
    status = (f"{n}/8 objectives; frozen pools = trunk frozen (BN in eval unless "
              r"`BN adapt'), $2\times2$-epoch anchor rounds; legal cut at $n_{\min}=10$ and $30$ "
              r"(`ref.' = the rule declined); loop = exp-68 fine-tuning loop, distance pool, "
              rf"$2\times5$ epochs. Injected-sample probe/mahaT present for {n_inj}/{n} objectives "
              r"(`--' = npz predates the per-fraction keys).")
    return _wrap("\n".join(rows),
                 rf"\textbf{{{tag} leakage-free lineage: pools and discovery.}} Round-1 purity "
                 r"of the frozen distance and density-ratio (np) pools, the np pool's round 2 "
                 r"with BN frozen and adapting, the label-free cut, then the fine-tuning loop "
                 r"for \emph{discovery from a 2\% injected sample} of the held-out class (the "
                 r"paper's regime): round-1 purity, post probe and post tied Mahalanobis. The "
                 r"last two columns are the loop's natural pass, in which the whole held-out "
                 r"class is present in the unlabelled bank --- a different, easier regime, "
                 r"shown for reference only.",
                 f"tab:app_{ds}_pools", status, "lccccccccccc",
                 r"& \multicolumn{4}{c}{frozen pools} & \multicolumn{2}{c}{legal cut} & "
                 r"\multicolumn{3}{c}{loop, 2\% injected} & \multicolumn{2}{c}{loop, whole class} \\"
                 "\n" r"\cmidrule(lr){2-5}\cmidrule(lr){6-7}\cmidrule(lr){8-10}\cmidrule(lr){11-12}" "\n"
                 r"objective & dist r1 & np r1 & np r2 & np+BN r2 & $n_{\min}{=}10$ & $n_{\min}{=}30$ & "
                 r"purity & probe & mahaT & purity & probe pre$\to$post \\", wide=True)


def t_residuals_ds(ds):
    f = os.path.join(LOGS, "exp137", f"residuals_{ds}.json")
    if not os.path.exists(f):
        return None
    d = json.load(open(f))
    parents = [k for k in d if k.endswith("(parent)")]
    rows = []
    for par in parents:
        p = d[par]; stem = par.replace(" (parent)", "")
        rows.append(r"\multicolumn{8}{l}{\emph{parent: }" + PRETTY.get(stem, esc(stem)) + r"} \\")
        rows.append(" & ".join(["\\quad (parent)", fnum(p["probe"]), fnum(p["top1"]), fnum(p["acc"]),
                                fnum(p["eucl"]), fnum(p["mahaT"]), fnum(p["lid"]), fnum(p["perevt"], 2)]) + r" \\")
        for k in [x for x in d if x.startswith(stem + "->")]:
            c = d[k]
            rows.append(" & ".join(["\\quad " + esc(k.split("->", 1)[1]), fnum(c["probe"]), fnum(c["top1"]),
                                    fnum(c["acc"]), fnum(c["eucl"]), fnum(c["mahaT"]), fnum(c["lid"]),
                                    fnum(c["perevt"], 2)]) + r" \\")
    tag = "CIFAR-10" if ds == "cifar10" else "CIFAR-100"
    status = f"{len(parents)} parents $\\times$ 4 constructions on the from-scratch {tag} trunks; seed 0, holdout 4."
    return _wrap("\n".join(rows),
                 rf"\textbf{{{tag} leakage-free lineage: every residual construction}}, with the "
                 r"supervised top-1 that the main text's no-cost claim rests on.",
                 f"tab:app_{ds}_residuals", status, "lccccccc",
                 r"space & probe & top-1 & acc & eucl & mahaT & lid & per-ev \\")


# ------------------------------------------------- transfer draw tables ---
def _log70(ds, base, d):
    for p in (os.path.join(LOGS, f"exp70_{ds}_{base}_h1_d{d}.log"),
              os.path.join(LOGS, f"exp70_g10_{base}_h1_d{d}.log")):
        if os.path.exists(p):
            return p
    return None


def _purities70(ds, base, draws):
    out = {}
    for d in draws:
        p = _log70(ds, base, d)
        if not p:
            continue
        arm = None
        for line in open(p, errors="ignore"):
            if line.startswith(("-----", "=====")):
                arm = None
            m = t140._ARM.match(line.strip())
            if m:
                arm = m.group(1); continue
            m = t140._PUR.match(line)
            if m and arm:
                out.setdefault(arm, {}).setdefault(d, float(m.group(2)))
    return out


def t_transfer_draws(ds, draws):
    """Post-discovery columns are the paper regime (discovery from a 2%
    injected sample: postf_* npz keys, POST-grid purity); the natural pass
    (whole class present) is the labelled pair on the right."""
    bases = ["dino", "lejepa", "visreg"]
    F = t140.F_PAPER
    rows, have, n_postf = [], 0, 0
    for base in bases:
        files = {d: os.path.join(LOGS, "exp70", f"results_{ds}_{base}_ft70_h1_d{d}.npz") for d in draws}
        files = {d: f for d, f in files.items() if os.path.exists(f)}
        if not files:
            continue
        pur = _purities70(ds, base, draws)
        purf = t140._draw_purities_f(ds, base, F, draws)
        Z = {d: np.load(f, allow_pickle=True) for d, f in files.items()}
        rows.append(r"\multicolumn{12}{l}{\emph{" + esc(f"{ds} / {base}") + f", {len(files)} draws" + r"}} \\")
        for a in ARMS_70:
            g = lambda k: [float(Z[d][k]) for d in Z if k in Z[d].files]
            pv = [x for x in (pur.get(a, {}).get(d) for d in Z) if x is not None]
            pf = [x for x in (purf.get(a, {}).get(d) for d in Z) if x is not None]
            probe_f = list(t140._postf70(ds, base, "probe", a, F, draws).values())
            mahaT_f = list(t140._postf70(ds, base, "mahaT", a, F, draws).values())
            n_postf += bool(probe_f)
            have += 1
            rows.append(" & ".join([
                "\\quad " + PRETTY.get(a, esc(a)), msd(g(f"probe_{a}")), msd(g(f"acc_{a}")),
                msd(g(f"eucl_{a}")), msd(g(f"mahaT_{a}")),
                msd([float(np.asarray(Z[d][f"perevent_{a}_pre"]).ravel()[0]) for d in Z]),
                msd(probe_f), msd(mahaT_f), msd(pf),
                fnum(min(pf)) if pf else "--",
                msd(g(f"post_probe_{a}")), msd(pv)]) + r" \\")
    if not rows:
        return None
    status = (f"{ds}, single holdout, draws {list(draws)} (distinct held-out classes), "
              f"{have} (arm, base) rows; mean $\\pm$ sd across draws, one seed per draw; "
              f"injected-sample probe/mahaT present for {n_postf}/{have} rows "
              r"(`--' = npz predates the per-fraction keys); purity read from the run logs.")
    return _wrap("\n".join(rows),
                 rf"\textbf{{{esc(ds)}: the single-holdout battery across holdout draws, all "
                 r"three backbones.} Pre-discovery probe, geometry and per-event power, then the "
                 r"fine-tuning loop's post-discovery probe, tied Mahalanobis, round-1 pool purity (mean and "
                 r"worst draw) for \emph{discovery from a 2\% injected sample} of the "
                 r"held-out class (the paper's regime). The last two columns are the natural pass, "
                 r"in which the whole held-out class is present in the unlabelled bank --- a "
                 r"different, easier regime, shown for reference only.",
                 f"tab:app_{ds}_draws", status, "lccccccccccc",
                 r"& \multicolumn{5}{c}{pre-discovery} & \multicolumn{4}{c}{post, 2\% injected} & "
                 r"\multicolumn{2}{c}{post, whole class} \\"
                 "\n" r"\cmidrule(lr){2-6}\cmidrule(lr){7-10}\cmidrule(lr){11-12}" "\n"
                 r"objective & probe & acc & eucl & mahaT & per-ev & probe & mahaT & purity & min & probe & purity \\",
                 long=True, wide=True)


def t_galaxy_residual_draws():
    spaces = ["supcon-ft_(parent)", "supcon-ft-res_(residual)", "supcon-ft-res_(concat)",
              "supcon-ft-res-nplm_(residual)", "supcon-ft-res-nplm_(concat)",
              "ss-ft_(parent)", "ss-ft-res_(residual)", "ss-ft-res_(concat)"]
    rows, nb = [], 0
    for base in ["dino", "lejepa", "visreg"]:
        fs = [os.path.join(LOGS, "exp71", f"results_galaxy10_{base}_ft71_h1_d{d}.npz") for d in G10_DRAWS]
        fs = [f for f in fs if os.path.exists(f)]
        if not fs:
            continue
        nb += 1
        Z = [np.load(f, allow_pickle=True) for f in fs]
        rows.append(r"\multicolumn{7}{l}{\emph{galaxy10 / " + base + f", {len(fs)} draws" + r"}} \\")
        for sp in spaces:
            g = lambda m: [float(z[f"{sp}__{m}"]) for z in Z if f"{sp}__{m}" in z.files]
            par = sp.split("-res")[0].replace("_(parent)", "") + "_(parent)"
            dp = ([float(z[f"{sp}__probe"]) - float(z[f"{par}__probe"]) for z in Z]
                  if "concat" in sp and all(f"{par}__probe" in z.files for z in Z) else [])
            rows.append(" & ".join([
                "\\quad " + esc(sp.replace("_", " ")), msd(g("probe")), msd(g("acc")), msd(g("eucl")),
                msd(g("mahaT")), msd(g("perevt")),
                (f"{np.mean(dp):+.3f} ({sum(1 for x in dp if x > 0)}/{len(dp)})" if dp else "--")]) + r" \\")
    if not rows:
        return None
    status = (f"galaxy10, {nb} backbones, single holdout, draws {list(G10_DRAWS)}; mean $\\pm$ sd "
              r"across draws; last column = paired concat$-$parent probe gain and wins/draws.")
    return _wrap("\n".join(rows),
                 r"\textbf{galaxy10: residual constructions across holdout draws.} The paired "
                 r"probe gain of the concatenation over its own parent is the one galaxy10 "
                 r"effect that survives draw variance.",
                 "tab:app_galaxy10_residual_draws", status, "lcccccc",
                 r"space & probe & acc & eucl & mahaT & per-ev & concat$-$parent (wins) \\",
                 long=True, wide=True)


# ------------------------------------------------- construction audits ---
def t_width_control():
    fs = sorted(glob.glob(os.path.join(LOGS, "exp134", "width_control_*.json")))
    if not fs:
        return None
    rows = []
    for f in fs:
        cell = re.search(r"width_control_(.+?)\.json", os.path.basename(f)).group(1)
        d = json.load(open(f))
        for m in ("probe", "eucl", "mahaT"):
            r = d["rows"].get(m, {}); v = d["verdicts"].get(m, {})
            rows.append(" & ".join([
                esc(cell) if m == "probe" else "", m,
                msd(list(r.get("parent", {}).values())), msd(list(r.get("control", {}).values())),
                msd(list(r.get("concat", {}).values())),
                (f"{v['gap']:+.3f}" if v else "--"), esc(v.get("winner", "--")) if v else "--"]) + r" \\")
    status = (f"{len(fs)} cells; control = the parent objective with a 200-D head (no residual); "
              r"concat = 100+100 residual concat; TIE unless the gap clears $\max(0.017, \text{sd}_a+\text{sd}_b)$.")
    return _wrap("\n".join(rows),
                 r"\textbf{Width-matched control for the residual concat (exp 134a).} A "
                 r"same-width non-residual head recovers little of the flagship probe gain and "
                 r"loses geometry on cars/VISReg; on galaxy10 the probe gain is a tie on DINO.",
                 "tab:app_width", status, "llccccc",
                 r"cell & metric & parent (100) & control (200) & concat (100+100) & concat$-$control & verdict \\", size=r"\footnotesize")


def t_postdisc():
    fs = sorted(glob.glob(os.path.join(LOGS, "exp134", "postdisc_*.json")))
    if not fs:
        return None
    rows = []
    for f in fs:
        d = json.load(open(f)); cell = re.search(r"postdisc_(.+?)_ft134c", os.path.basename(f)).group(1)
        pur = d["discovery"][0]["purity"] if d.get("discovery") else float("nan")
        rows.append(r"\multicolumn{9}{l}{\emph{" + esc(cell) + f"}}: round-1 purity {pur:.3f}, "
                    f"{d.get('n_pseudo', '?')} pseudo-labelled at purity {d.get('pseudo_purity', float('nan')):.2f}" + r"} \\")
        res, arch = d["results"], d.get("archived_exp71", {})
        for k in [x for x in res if "concat" in x]:
            new = res[k]; ak = k.replace(" (post-discovery) concat", " (concat)")
            old = arch.get(ak, {})
            rows.append(" & ".join([
                "\\quad " + esc(k.replace(" (post-discovery) concat", "")),
                fnum(old.get("probe")), fnum(new["probe"]), fnum(old.get("eucl")), fnum(new["eucl"]),
                fnum(old.get("mahaT")), fnum(new["mahaT"]), fnum(old.get("perevt"), 2), fnum(new["perevt"], 2)]) + r" \\")
    status = f"{len(fs)} cells, archived holdout draw, seed 0; parent = SupCon fine-tune; archived = exp-71 pre-discovery order."
    return _wrap("\n".join(rows),
                 r"\textbf{Residual built after discovery vs.\ before (exp 134c).} The "
                 r"pseudo-labelled child corpus is 100\% novel in every cell, including where "
                 r"the pool purity is very low.",
                 "tab:app_postdisc", status, "lcccccccc",
                 r"concat & probe (pre) & probe (post) & eucl (pre) & eucl (post) & mahaT (pre) & mahaT (post) & per-ev (pre) & per-ev (post) \\",
                 long=True, wide=True)


def t_seeds_c100():
    stem = os.path.join(LOGS, "exp59", "nplm_residual_concat_cifar100")
    cfgs = [("100-D (50+50), classwise-$\\lambda$5", [f"{stem}_100d_cwlam5.npz", f"{stem}_s1_100d_cwlam5.npz", f"{stem}_s2_100d_cwlam5.npz"]),
            ("32-D (16+16)", [f"{stem}.npz", f"{stem}_s1.npz", f"{stem}_s2.npz"])]
    rows_all, nf = [], 0
    for title, cand in cfgs:
        fs = [p for p in cand if os.path.exists(p)]
        if not fs:
            continue
        nf += len(fs)
        rows_all.append(r"\multicolumn{8}{l}{\emph{" + title + f", {len(fs)} seeds" + r"}} \\")
        rows_all += _seed_rows(fs)
    if not rows_all:
        return None
    status = f"CIFAR-100, two exp-59 configurations, {nf} result files; hub-pretrained trunk (ablation lineage)."
    return _wrap("\n".join(rows_all),
                 r"\textbf{Multi-seed CIFAR-100 champions (hub lineage).} Power columns at $f{=}0.02$, post-discovery.",
                 "tab:app_seeds_c100", status, "lccccccc",
                 r"space & probe pre & probe post & eucl & mahaT & SpK post & MMD post & $n$ \\", wide=True)


def _seed_rows(fs):
    per = {}
    for f in fs:
        d = np.load(f, allow_pickle=True)
        fr = [float(x) for x in d["fractions"]]; i = fr.index(0.02) if 0.02 in fr else -1
        for a in d["arms"]:
            a = str(a)
            for lab, key, idx in (("probe_pre", f"probe_{a}", None), ("probe_post", f"probe_post_{a}", None),
                                  ("eucl", f"eucl_{a}", None), ("mahaT", f"mahaT_{a}", None),
                                  ("mmd_post", f"mmd_{a}_post", i), ("spk_post", f"sparker_{a}_post", i)):
                if key in d.files:
                    v = np.asarray(d[key]).ravel()
                    per.setdefault(a, {}).setdefault(lab, []).append(float(v[idx] if idx is not None and v.size > 1 else v[0]))
    return ["\\quad " + " & ".join([esc(sp)] + [msd(per[sp].get(m, [])) for m in ("probe_pre", "probe_post", "eucl", "mahaT", "spk_post", "mmd_post")]
                             + [str(len(per[sp].get("eucl", [])))]) + r" \\" for sp in sorted(per)]


def t_top1_galaxy():
    fs = sorted(glob.glob(os.path.join(LOGS, "exp132", "probe_cells_galaxy10-*.json")))
    if not fs:
        return None
    arms = ["supcon-ft", "ss-ft", "nplm-sup-ft", "simclr-ft", "sigreg-ssl-ft", "nplm-bil-ft",
            "supcon-ft-res-cat", "supcon-ft-resnplm-cat", "ss-ft-res-cat"]
    per = {}
    for f in fs:
        base = re.search(r"galaxy10-(\w+?)(_h1_d\d+)?\.json", os.path.basename(f)).group(1)
        draw = "draw" if "_h1_d" in f else "archived"
        d = json.load(open(f))
        for a in arms:
            v = d.get(f"galaxy10:{base}|{a}", {}).get("top1")
            if v is not None:
                per.setdefault((base, a), {}).setdefault(draw, []).append(v)
    rows = []
    for base in ("dino", "lejepa", "visreg"):
        rows.append(r"\multicolumn{4}{l}{\emph{galaxy10 / " + base + r"}} \\")
        for a in arms:
            v = per.get((base, a), {})
            rows.append(" & ".join(["\\quad " + esc(a), msd(v.get("archived", [])), msd(v.get("draw", [])),
                                    str(len(v.get("draw", [])))]) + r" \\")
    status = f"{len(fs)} cell files; 3 probe restarts per cell; draws are the exp-125 single-holdout draws."
    return _wrap("\n".join(rows),
                 r"\textbf{Supervised top-1 (exp 132) on galaxy10: archived draw and the draw interval.} "
                 r"SupCon+SIGReg costs $0.035$ against SupCon on every draw; the residual concats tie the parent.",
                 "tab:app_top1_galaxy", status, "lccc",
                 r"space & top-1 (archived) & top-1 (draws) & $n$ \\", long=True, size=r"\footnotesize")


def t_frozen_np_datasets():
    fs = sorted(glob.glob(os.path.join(LOGS, "exp135", "corpus_norm_*_frozen*.json")))
    if not fs:
        return None
    per = {}
    for f in fs:
        m = re.search(r"corpus_norm_(\w+?)-(\w+?)_frozen(_h1_d\d+)?\.json", os.path.basename(f))
        if not m:
            continue
        ds, base, single = m.group(1), m.group(2), bool(m.group(3))
        d = json.load(open(f))
        for k, v in d.items():
            if v["variant"] != "frozen":
                continue
            per.setdefault((ds, base, "single" if single else "multi", v["scorer"]), []).append(v["purity"])
    rows = []
    for ds in ("galaxy10", "dtd", "flowers", "cars", "aircraft"):
        for base in ("dino", "lejepa", "visreg"):
            cells = []
            for reg in ("multi", "single"):
                for sc in ("dist", "np"):
                    v = per.get((ds, base, reg, sc), [])
                    r1 = [x[0] for x in v]; r2 = [x[1] for x in v if len(x) > 1]
                    cells += [msd(r1), msd(r2), fnum(min(r1)) if r1 else "--"]
            if any(c != "--" for c in cells):
                rows.append(" & ".join([esc(f"{ds}/{base}")] + cells) + r" \\")
    if not rows:
        return None
    n = len(fs)
    status = (f"{n} cell files (multi-holdout default draw, and single-holdout draws 0--4); pretrained "
              r"ViT-B/16 features, identity head, anchors = seen centroids, no fine-tune.")
    return _wrap("\n".join(rows),
                 r"\textbf{Frozen-backbone pooling with no training at all}, on the pretrained backbone "
                 r"features: distance vs density-ratio pool, multi- vs single-holdout regime; "
                 r"round-1 and round-2 purity with the worst draw.",
                 "tab:app_frozen_np", status, "lcccccccccccc",
                 r"& \multicolumn{6}{c}{multi-holdout} & \multicolumn{6}{c}{single holdout (draws)} \\"
                 "\n" r"\cmidrule(lr){2-7}\cmidrule(lr){8-13}" "\n"
                 r"& \multicolumn{3}{c}{dist} & \multicolumn{3}{c}{np} & \multicolumn{3}{c}{dist} & \multicolumn{3}{c}{np} \\"
                 "\n" r"cell & r1 & r2 & min & r1 & r2 & min & r1 & r2 & min & r1 & r2 & min \\",
                 long=True, wide=True)


def t_hub_residual_seeds():
    rows = []
    for ds, tag in (("cifar10", "32d"), ("cifar100", "100d")):
        f = os.path.join(LOGS, "exp75", f"results_{ds}_{tag}.npz")
        if not os.path.exists(f):
            continue
        d = np.load(f, allow_pickle=True)
        rows.append(r"\multicolumn{5}{l}{\emph{" + esc(f"{ds}, {tag}, {int(d['n_seeds'])} paired seeds") + r"}} \\")
        for sp in ("parent", "res_concat", "res-nplm_concat"):
            g = lambda m: list(np.asarray(d[f"{sp}__{m}"], float))
            rows.append(" & ".join(["\\quad " + esc(sp), msd(g("probe")), msd(g("acc")), msd(g("eucl")), msd(g("mahaT"))]) + r" \\")
    if not rows:
        return None
    return _wrap("\n".join(rows),
                 r"\textbf{Hub-lineage residual concats, 5 paired seeds (exp 75; ablation lineage).}",
                 "tab:app_hub_residual_seeds", "exp 75, both CIFAR datasets, 5 seeds; hub-pretrained trunk (saw the held-out class).",
                 "lcccc", r"space & probe & acc & eucl & mahaT \\")


def t_zero_training():
    """Main-text table: single-holdout purity of the zero-training construction
    (pretrained trunk + np pool) per dataset x backbone, vs the fine-tuned
    references where they exist."""
    fs = sorted(glob.glob(os.path.join(LOGS, "exp135", "corpus_norm_*_frozen_h1_d*.json")))
    if not fs:
        return None
    per = {}
    for f in fs:
        m = re.search(r"corpus_norm_(\w+?)-(\w+?)_frozen_h1_d\d+\.json", os.path.basename(f))
        if not m:
            continue
        d = json.load(open(f))
        for k, v in d.items():
            if v["variant"] == "frozen" and v["scorer"] == "np":
                per.setdefault((m.group(1), m.group(2)), []).append(v["purity"][0])
    # fine-tuned references: exp-70 loop best arm (mean over draws), from the
    # POST-grid logs -- discovery from a 2% injected sample, the paper regime
    ref = {}
    for ds, draws in (("galaxy10", (0, 3, 5, 7, 8)), ("dtd", (0, 1, 3, 4, 5))):
        for base in ("dino", "lejepa", "visreg"):
            pur = t140._draw_purities_f(ds, base, t140.F_PAPER, draws)
            best = max(((np.mean(list(v.values())), a) for a, v in pur.items() if v), default=None)
            if best:
                ref[(ds, base)] = best
    rate = {"galaxy10": "1/10", "dtd": "1/47", "flowers": "1/102", "aircraft": "1/100", "cars": "1/196"}
    rows = []
    for ds in ("galaxy10", "dtd", "flowers", "aircraft", "cars"):
        for i, base in enumerate(("dino", "lejepa", "visreg")):
            v = per.get((ds, base), [])
            r = ref.get((ds, base))
            rows.append(" & ".join([
                (esc(ds) if i == 0 else ""), (rate[ds] if i == 0 else ""), base,
                msd(v), fnum(min(v)) if v else "--",
                (f"{r[0]:.3f} ({esc(r[1])})" if r else "--")]) + r" \\")
    status = (f"{len(fs)} cells; pretrained ViT-B/16 features, identity head, anchors = seen-class centroids, "
              r"density-ratio pool, anchors-only iteration, round 1; five single-holdout draws per (dataset, "
              r"backbone). Reference = the best objective of the exp-70 fine-tuning loop on the same "
              r"draws, discovery from a 2\% injected sample (paper regime, not the whole-class natural pass).")
    return _wrap("\n".join(rows),
                 r"\textbf{The regime boundary on one construction that needs no training.} Round-1 pool "
                 r"purity of the pretrained trunk pooled by density ratio, single holdout, mean $\pm$ sd over "
                 r"five held-out classes, with the worst draw. galaxy10 and DTD sit far above the best "
                 r"fine-tuned objective on every draw; flowers is marginal; aircraft and cars are dead although "
                 r"aircraft's rate equals flowers'.",
                 "tab:zero_training", status, "lllccc",
                 r"dataset & rate & backbone & purity & min & fine-tuned loop, best objective \\")


def t_gcd_benchmark():
    """Main-text comparison under the GCD protocol (exps 144/145)."""
    rows, nfiles = [], 0
    DS = [("cifar10", "CIFAR-10"), ("cifar100", "CIFAR-100"), ("cars", "Cars"), ("aircraft", "Aircraft")]
    def cellstr(v):
        return f"{v[0]:.1f}/{v[1]:.1f}/{v[2]:.1f}"
    def mean_rows(res, space, method):
        v = np.array([[res[s][space][method][k] for k in ("all", "old", "new")] for s in res
                      if space in res[s] and method in res[s][space]])
        return cellstr(v.mean(0)) if len(v) else "--"
    data = {}
    for ds, _ in DS:
        f = os.path.join(LOGS, "exp144", f"gcd_{ds}_dino.json")
        if os.path.exists(f):
            data[ds] = json.load(open(f)); nfiles += 1
        f2 = os.path.join(LOGS, "exp144", f"gcd_ft_{ds}_dino.json")
        if os.path.exists(f2):
            data[ds + "_ft"] = json.load(open(f2)); nfiles += 1
    if not data:
        return None
    def line(label, key, space, method):
        cells = []
        for ds, _ in DS:
            d = data.get(ds + key)
            cells.append(mean_rows(d["results"], space, method) if d else "--")
        rows.append(" & ".join([label] + cells) + r" \\")
    rows.append(r"\multicolumn{5}{l}{\emph{frozen DINO trunk (ours, exp 144)}} \\")
    line(r"\quad $k$-means, raw features", "", "raw-dino", "kmeans")
    line(r"\quad ss-$k$-means, raw features", "", "raw-dino", "ss-kmeans")
    line(r"\quad np-anchors, raw features ($K$ free)", "", "raw-dino", "np-anchors")
    line(r"\quad ss-$k$-means, SupCon head", "", "head-supcon", "ss-kmeans")
    line(r"\quad ss-$k$-means, SupCon+SIGReg head", "", "head-ssig", "ss-kmeans")
    rows.append(r"\multicolumn{5}{l}{\emph{fine-tuned trunk, labelled data only (ours, exp 145)}} \\")
    for arm, lab in (("supcon-ft", "SupCon"), ("ss-ft", "SupCon+SIGReg"), ("nplm-sup-ft", "NPLM-dist.")):
        line(rf"\quad ss-$k$-means, {lab} trunk", "_ft", f"{arm} trunk", "ss-kmeans")
    rows.append(r"\multicolumn{5}{l}{\emph{published, fine-tuned on labelled + unlabelled}} \\")
    ref = {ds: data[ds]["reference"] for ds, _ in DS if ds in data}
    for name in ("k-means (paper)", "GCD", "UNO+", "SimGCD", "RLCD"):
        rows.append(" & ".join([r"\quad " + esc(name)] + [cellstr(ref[ds][name]) if ds in ref else "--" for ds, _ in DS]) + r" \\")
    status = (f"{nfiles} result files; GCD protocol (first-$N$ known classes, 50\% labelled, Hungarian ACC on the "
              r"unlabelled train set, $K$ known), DINO ViT-B/16, 3 split seeds (means shown).")
    return _wrap("\n".join(rows),
                 r"\textbf{Under the Generalized Category Discovery protocol.} Clustering accuracy All/Old/New. "
                 r"Our frozen-trunk $k$-means row reproduces the published one; a head trained on labelled data "
                 r"alone doubles it on the fine-grained sets and collapses novel structure on CIFAR-10.",
                 "tab:gcd", status, "lcccc",
                 r"method & CIFAR-10 & CIFAR-100 & Cars & Aircraft \\", wide=True)


def t_scratch_draws(ds):
    """Exp 136 Tier 3: the scratch shortlist across holdout draws (the archived
    holdout plus the random draws), mean +- sd; loop purity in both regimes."""
    files = sorted(f for f in glob.glob(os.path.join(LOGS, "exp136", f"master_{ds}*.json"))
                   if re.fullmatch(rf".*master_{ds}(_h\d+)?\.json", f))
    if len(files) < 2:
        return None
    per = {}
    for f in files:
        m = re.search(r"_h(\d+)\.json$", f); tag = f"_h{m.group(1)}" if m else ""
        d = json.load(open(f)); inj = d68_injected(ds, tag)
        for a, r in d.items():
            q = per.setdefault(a, {k: [] for k in ("probe", "top1", "mahaT", "perevt", "frz", "pur", "purf", "probe_post")})
            q["probe"].append(r["pre"]["probe"]); q["top1"].append(r["pre"]["top1"])
            q["mahaT"].append(r["pre"]["mahaT"]); q["perevt"].append(r["pre"]["perevt"])
            q["frz"].append(r["frozen"]["np|frozen"]["purity"][0])
            d68 = r.get("discovery68") or {}
            if d68.get("purity"):
                q["pur"].append(d68["purity"][0])
            if d68.get("probe_post") is not None:
                q["probe_post"].append(d68["probe_post"])
            if a in inj and "purity" in inj[a]:
                q["purf"].append(inj[a]["purity"])
    rows = []
    for a in ("supcon", "ssig", "nplmsd", "nplmcw", "supsig", "simclr", "visreg", "nplm"):
        if a not in per or len(per[a]["probe"]) < 2:
            continue
        q = per[a]
        rows.append(" & ".join([PRETTY.get(a, esc(a)), str(len(q["probe"])), msd(q["probe"]), msd(q["top1"]),
                                msd(q["mahaT"]), msd(q["perevt"]), msd(q["frz"]), msd(q["purf"]),
                                msd(q["pur"]), msd(q["probe_post"])]) + r" \\")
    if not rows:
        return None
    hs = [int((re.search(r"_h(\d+)\.json$", f) or [None, 4])[1]) for f in files]
    status = (f"{ds} from scratch, single holdout, holdouts {sorted(hs)} (the archived class plus random draws), "
              r"one seed per draw; objectives with fewer than two draws omitted; injected-sample purity from the "
              r"POST-grid logs (`--' = run predates it).")
    return _wrap("\n".join(rows),
                 rf"\textbf{{{esc(ds)} from scratch, across holdout draws.}} Pre-discovery probe, top-1, tied "
                 r"Mahalanobis and per-event power; frozen density-ratio pool purity; the fine-tuning loop's "
                 r"round-1 purity for discovery from a 2\% injected sample and, for reference, with the whole "
                 r"held-out class present in the bank, with the whole-class post probe.",
                 f"tab:app_{ds}_scratch_draws", status, "lccccccccc",
                 r"& & \multicolumn{4}{c}{pre-discovery} & frozen & \multicolumn{3}{c}{loop} \\"
                 "\n" r"\cmidrule(lr){3-6}\cmidrule(lr){7-7}\cmidrule(lr){8-10}" "\n"
                 r"objective & draws & probe & top-1 & mahaT & per-ev & purity & purity 2\% & purity whole & probe post whole \\",
                 wide=True, size=r"\footnotesize")


def t_residuals_draws(ds):
    """Exp 137 across holdout draws: every residual construction on the
    scratch trunks, mean +- sd over the archived holdout plus the draws."""
    files = sorted(f for f in glob.glob(os.path.join(LOGS, "exp137", f"residuals_{ds}*.json"))
                   if re.fullmatch(rf".*residuals_{ds}(_h\d+)?\.json", f))
    if len(files) < 2:
        return None
    per = {}
    for f in files:
        for k, v in json.load(open(f)).items():
            q = per.setdefault(k, {m: [] for m in ("probe", "top1", "eucl", "mahaT", "perevt")})
            for m in q:
                if v.get(m) is not None:
                    q[m].append(v[m])
    rows = []
    for par in [k for k in per if k.endswith("(parent)")]:
        stem = par.replace(" (parent)", "")
        rows.append(r"\multicolumn{7}{l}{\emph{parent: }" + PRETTY.get(stem, esc(stem)) + r"} \\")
        for k in [par] + [x for x in per if x.startswith(stem + "->")]:
            q = per[k]
            lab = "(parent)" if k == par else esc(k.split("->", 1)[1])
            rows.append(" & ".join(["\\quad " + lab, str(len(q["probe"])), msd(q["probe"]), msd(q["top1"]),
                                    msd(q["eucl"]), msd(q["mahaT"]), msd(q["perevt"], 2)]) + r" \\")
    hs = sorted(int((re.search(r"_h(\d+)\.json$", f) or [None, 4])[1]) for f in files)
    tag = "CIFAR-10" if ds == "cifar10" else "CIFAR-100"
    status = f"from-scratch {tag} trunks, single holdout, holdouts {hs}; seed 0 per holdout; mean $\\pm$ sd over holdouts."
    return _wrap("\n".join(rows),
                 rf"\textbf{{{tag} leakage-free lineage: the residual constructions across holdout draws.}} "
                 r"Mean $\pm$ sd over the archived holdout and the random draws; the single-holdout "
                 r"table above is the archived class alone.",
                 f"tab:app_{ds}_residual_draws", status, "lcccccc",
                 r"space & draws & probe & top-1 & eucl & mahaT & per-ev \\", size=r"\footnotesize")


TABLES = [
    ("app_cifar10_pre", lambda: t_scratch_pre("cifar10")), ("app_cifar10_power", lambda: t_scratch_power("cifar10")),
    ("app_cifar10_pools", lambda: t_scratch_pools("cifar10")), ("app_cifar10_residuals", lambda: t_residuals_ds("cifar10")),
    ("app_cifar100_pre", lambda: t_scratch_pre("cifar100")), ("app_cifar100_power", lambda: t_scratch_power("cifar100")),
    ("app_cifar100_pools", lambda: t_scratch_pools("cifar100")), ("app_cifar100_residuals", lambda: t_residuals_ds("cifar100")),
    ("app_galaxy10_draws", lambda: t_transfer_draws("galaxy10", G10_DRAWS)),
    ("app_galaxy10_residual_draws", t_galaxy_residual_draws),
    ("app_dtd_draws", lambda: t_transfer_draws("dtd", DTD_DRAWS)),
    ("app_width", t_width_control), ("app_postdisc", t_postdisc),
    ("app_seeds_c100", t_seeds_c100), ("app_top1_galaxy", t_top1_galaxy),
    ("app_frozen_np", t_frozen_np_datasets), ("app_hub_residual_seeds", t_hub_residual_seeds),
    ("zero_training", t_zero_training),
    ("app_cifar10_scratch_draws", lambda: t_scratch_draws("cifar10")),
    ("app_cifar10_residual_draws", lambda: t_residuals_draws("cifar10")),
    ("app_cifar100_residual_draws", lambda: t_residuals_draws("cifar100")),
    ("app_cifar100_scratch_draws", lambda: t_scratch_draws("cifar100"))]


def selftest():
    ok = True
    for name, fn in TABLES:
        try:
            t = fn()
        except Exception as e:                       # noqa: BLE001
            print(f"  [FAIL] {name}: {type(e).__name__}: {e}"); ok = False; continue
        if t is None:
            print(f"  [skip] {name}: source absent"); continue
        env = "longtable" if r"\begin{longtable}" in t else "tabular"
        ncol = t.split(f"{{{env}}}{{")[1].split("}")[0]
        ncol = sum(1 for c in ncol if c in "lcr")
        body = t.split(r"\midrule")[-1].split(r"\bottomrule")[0] if env == "tabular" else t.split(r"\endfoot")[1].split(r"\end{longtable}")[0]
        for line in body.strip().split("\n"):
            if not line.strip().endswith(r"\\"):
                continue
            spans = [int(x) for x in re.findall(r"\\multicolumn\{(\d+)\}", line)]
            n = line.count("&") + 1 - len(spans) + sum(spans)
            if n != ncol:
                print(f"  [FAIL] {name}: row spans {n} cols, colspec has {ncol}: {line[:80]}"); ok = False; break
        else:
            print(f"  [ok] {name}: {t.count(chr(92) + chr(92))} rows, {ncol} columns")
    print("SELFTEST PASS" if ok else "SELFTEST FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())
    os.makedirs(args.out, exist_ok=True)
    for name, fn in TABLES:
        t = fn()
        if t is None:
            print(f"  [skip] {name}"); continue
        p = os.path.join(args.out, f"{name}.tex"); open(p, "w").write(t)
        print(f"  wrote {os.path.relpath(p, REPO)}  ({t.splitlines()[0][10:90]})")


if __name__ == "__main__":
    main()
