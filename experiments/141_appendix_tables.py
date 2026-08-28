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
    m = _master(ds)
    if m is None:
        return None
    rows, n = [], 0
    for a in ARMS_SCRATCH:
        if a not in m:
            rows.append(f"{PRETTY.get(a, esc(a))} & " + " & ".join(["--"] * 10) + r" \\"); continue
        r = m[a]; z = r["frozen"]; n += 1
        l10, l30 = r.get("legal_cut", {}), r.get("legal_cut_n30", {})
        d68 = r.get("discovery68") or {}
        L = lambda l: (fnum(l.get("purity")) if l.get("ok") else "ref.") if l else "--"
        rows.append(" & ".join([
            PRETTY.get(a, esc(a)),
            fnum(z["dist|frozen"]["purity"][0]), fnum(z["np|frozen"]["purity"][0]),
            fnum(z["np|frozen"]["purity"][1]), fnum(z["np|bn-adapt"]["purity"][1]),
            L(l10), L(l30),
            fnum((d68.get("purity") or [None])[0]) if d68 else "--",
            (f"{fnum(d68['probe_pre'])}$\\to${fnum(d68['probe_post'])}" if d68 else "--"),
            fnum(d68.get("post", {}).get("eucl")) if d68 else "--",
            fnum(d68.get("post", {}).get("maha_tied")) if d68 else "--"]) + r" \\")
    tag = "CIFAR-10" if ds == "cifar10" else "CIFAR-100"
    status = (f"{n}/8 objectives; frozen pools = trunk frozen (BN in eval unless "
              r"`BN adapt'), $2\times2$-epoch anchor rounds; legal cut at $n_{\min}=10$ and $30$ "
              r"(`ref.' = the rule declined); loop = exp-68 fine-tuning loop, distance pool, "
              r"$2\times5$ epochs. Gate $=0.15$.")
    return _wrap("\n".join(rows),
                 rf"\textbf{{{tag} leakage-free lineage: pools and discovery.}} Round-1 purity "
                 r"of the frozen distance and density-ratio (np) pools, the np pool's round 2 "
                 r"with BN frozen and adapting, the label-free cut, and the fine-tuning loop's "
                 r"purity, probe and post geometry.",
                 f"tab:app_{ds}_pools", status, "lcccccccccc",
                 r"objective & dist r1 & np r1 & np r2 & np+BN r2 & legal$_{10}$ & legal$_{30}$ & "
                 r"loop r1 & loop probe & post eucl & post mahaT \\", wide=True)


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
    bases = ["dino", "lejepa", "visreg"]
    rows, have = [], 0
    for base in bases:
        files = {d: os.path.join(LOGS, "exp70", f"results_{ds}_{base}_ft70_h1_d{d}.npz") for d in draws}
        files = {d: f for d, f in files.items() if os.path.exists(f)}
        if not files:
            continue
        pur = _purities70(ds, base, draws)
        Z = {d: np.load(f, allow_pickle=True) for d, f in files.items()}
        rows.append(r"\multicolumn{9}{l}{\emph{" + esc(f"{ds} / {base}") + f", {len(files)} draws" + r"}} \\")
        for a in ARMS_70:
            g = lambda k: [float(Z[d][k]) for d in Z if k in Z[d].files]
            pv = [pur.get(a, {}).get(d) for d in Z]
            pv = [x for x in pv if x is not None]
            have += 1
            rows.append(" & ".join([
                "\\quad " + PRETTY.get(a, esc(a)), msd(g(f"probe_{a}")), msd(g(f"post_probe_{a}")),
                msd(g(f"acc_{a}")), msd(g(f"eucl_{a}")), msd(g(f"mahaT_{a}")),
                msd([float(np.asarray(Z[d][f"perevent_{a}_pre"]).ravel()[0]) for d in Z]),
                msd(pv), f"{sum(1 for x in pv if x >= 0.15)}/{len(pv)}" if pv else "--"]) + r" \\")
    if not rows:
        return None
    status = (f"{ds}, single holdout, draws {list(draws)} (distinct held-out classes), "
              f"{have} (arm, base) rows; mean $\\pm$ sd across draws, one seed per draw; "
              r"purity read from the run logs; gate $=0.15$.")
    return _wrap("\n".join(rows),
                 rf"\textbf{{{esc(ds)}: the single-holdout battery across holdout draws, all "
                 r"three backbones.} Pre-discovery probe, geometry and per-event power; "
                 r"round-1 pool purity of the fine-tuning loop and the number of draws "
                 r"clearing the gate.",
                 f"tab:app_{ds}_draws", status, "lcccccccc",
                 r"objective & probe & probe post & acc & eucl & mahaT & per-ev & purity r1 & $\ge$gate \\",
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
                 r"the pool purity is far below the gate.",
                 "tab:app_postdisc", status, "lcccccccc",
                 r"concat & probe (pre) & probe (post) & eucl (pre) & eucl (post) & mahaT (pre) & mahaT (post) & per-ev (pre) & per-ev (post) \\",
                 long=True, wide=True)


def t_seeds_c100():
    stem = os.path.join(LOGS, "exp59", "nplm_residual_concat_cifar100")
    fs = [p for p in (f"{stem}_100d_cwlam5.npz", f"{stem}_s1_100d_cwlam5.npz", f"{stem}_s2_100d_cwlam5.npz")
          if os.path.exists(p)]
    if not fs:
        return None
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
    rows = [" & ".join([esc(sp)] + [msd(per[sp].get(m, [])) for m in ("probe_pre", "probe_post", "eucl", "mahaT", "spk_post", "mmd_post")]
                       + [str(len(per[sp].get("eucl", [])))]) + r" \\" for sp in sorted(per)]
    status = f"CIFAR-100, 100-D (50+50) classwise-$\\lambda$5 configuration, {len(fs)} seeds; hub-pretrained trunk (ablation lineage)."
    return _wrap("\n".join(rows),
                 r"\textbf{Multi-seed CIFAR-100 champions (hub lineage).} Power columns at $f{=}0.02$, post-discovery.",
                 "tab:app_seeds_c100", status, "lccccccc",
                 r"space & probe pre & probe post & eucl & mahaT & SpK post & MMD post & $n$ \\", wide=True)


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
                    cells += [msd(r1), msd(r2), f"{sum(1 for x in r1 if x >= 0.15)}/{len(r1)}" if r1 else "--"]
            if any(c != "--" for c in cells):
                rows.append(" & ".join([esc(f"{ds}/{base}")] + cells) + r" \\")
    if not rows:
        return None
    n = len(fs)
    status = (f"{n} cell files (multi-holdout default draw, and single-holdout draws 0--4); pretrained "
              r"ViT-B/16 features, identity head, anchors = seen centroids, no fine-tune; gate $=0.15$.")
    return _wrap("\n".join(rows),
                 r"\textbf{Frozen-trunk pooling with no training at all}, on the pretrained backbone "
                 r"features: distance vs density-ratio pool, multi- vs single-holdout regime.",
                 "tab:app_frozen_np", status, "lcccccccccccc",
                 r"& \multicolumn{6}{c}{multi-holdout} & \multicolumn{6}{c}{single holdout (draws)} \\"
                 "\n" r"\cmidrule(lr){2-7}\cmidrule(lr){8-13}" "\n"
                 r"& \multicolumn{3}{c}{dist} & \multicolumn{3}{c}{np} & \multicolumn{3}{c}{dist} & \multicolumn{3}{c}{np} \\"
                 "\n" r"cell & r1 & r2 & $\ge$g & r1 & r2 & $\ge$g & r1 & r2 & $\ge$g & r1 & r2 & $\ge$g \\",
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
]


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
