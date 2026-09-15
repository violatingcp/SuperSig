"""Experiment 142: the 2x2 summary grids for the paper appendix.

One row per (objective, construction) with construction in {plain, +res
concat, +res-nplm concat}; columns = metric x {pre-discovery, post-discovery}.
Two tables per dataset: A = probe / eucl / mahaT / per-event; B = SparKer /
Mahalanobis / MMD power at f = 0.02, alpha = 0.05.  A cell that was never
measured is `--`, so the grid doubles as the coverage map.

Datasets: cifar10 / cifar100 leakage-free (exps 136/68/137), cifar10 /
cifar100 hub-pretrained (exps 50/53/55/58/59/73/74/80; ablation lineage),
galaxy10 per backbone over the exp-125 draws (exps 70/71/72/80), and the
after-discovery residual (exp 134c) as its own rows where it exists.

    python experiments/142_grid_tables.py            # write docs/tables/grid_*.tex
    python experiments/142_grid_tables.py --selftest
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
t141 = importlib.import_module("141_appendix_tables")
REPO, LOGS, OUT = t140.REPO, t140.LOGS, t140.OUT
esc, fnum, _wrap = t140.esc, t140.fnum, t140._wrap

F = 0.02
A_COLS = ["probe", "eucl", "mahaT", "perevt"]
B_COLS = ["sparker", "maha", "mmd"]
HEAD_A = (r"space & \multicolumn{2}{c}{probe} & \multicolumn{2}{c}{eucl} & \multicolumn{2}{c}{mahaT} & "
          r"\multicolumn{2}{c}{per-event} \\" "\n" r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}"
          "\n" r"& pre & post & pre & post & pre & post & pre & post \\")
HEAD_B = (r"space & \multicolumn{2}{c}{SparKer} & \multicolumn{2}{c}{Mahalanobis} & \multicolumn{2}{c}{MMD} \\"
          "\n" r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}" "\n"
          r"& pre & post & pre & post & pre & post \\")

# ----------------------------------------------------------------- helpers
def at(d, key, f=F, fr_default=(0.003, 0.01, 0.02, 0.05)):
    """scalar, or the value at fraction f of a power curve; None if absent.
    Fraction axes differ by archive: `fractions` (exps 50-59, 74), or
    `pre_fractions` / `post_fractions` (exp 70), or none at all (exp 72,
    which used exp 70's post grid)."""
    if d is None or key not in d.files:
        return None
    v = np.asarray(d[key], dtype=float).ravel()
    if v.size == 1:
        return float(v[0])
    if "fractions" in d.files:
        fr = [float(x) for x in d["fractions"]]
    elif "post_fractions" in d.files and ("_post" in key or key.startswith(("post_", "postf_"))):
        fr = [float(x) for x in d["post_fractions"]]
    elif "pre_fractions" in d.files:
        fr = [float(x) for x in d["pre_fractions"]]
    else:
        fr = list(fr_default)
    if f in fr and len(v) == len(fr):
        return float(v[fr.index(f)])
    if f in fr and len(v) < len(fr) and fr.index(f) < len(v):
        return float(v[fr.index(f)])
    return None


def pair(d, key, i):
    """i-th element of a stored (pre, post) pair; None if absent."""
    if d is None or key not in d.files:
        return None
    v = np.asarray(d[key], dtype=float).ravel()
    return float(v[i]) if v.size > i else None


def cell(x, p=3):
    if x is None:
        return "--"
    if isinstance(x, (list, tuple, np.ndarray)):
        x = [y for y in x if y is not None and np.isfinite(y)]
        if not x:
            return "--"
        return fnum(x[0], p) if len(x) == 1 else f"{np.mean(x):.{p}f}$\\pm${np.std(x):.{p}f}"
    return fnum(x, p)


def row(label, m):
    """m: dict metric -> (pre, post)."""
    a = [label] + [cell(m.get(k, (None, None))[i]) for k in A_COLS for i in (0, 1)]
    b = [label] + [cell(m.get(k, (None, None))[i], 2) for k in B_COLS for i in (0, 1)]
    return " & ".join(a) + r" \\", " & ".join(b) + r" \\"


def group(title, ncolA=9, ncolB=7):
    return (rf"\multicolumn{{{ncolA}}}{{l}}{{\emph{{{title}}}}} \\",
            rf"\multicolumn{{{ncolB}}}{{l}}{{\emph{{{title}}}}} \\")


def emit(name, rowsA, rowsB, caption, status):
    A = _wrap("\n".join(rowsA), caption + " (A) Probe and per-event geometry.", f"tab:grid_{name}_a",
              status, "lcccccccc", HEAD_A, long=True, wide=True)
    B = _wrap("\n".join(rowsB), caption + f" (B) Dataset-level power at $f={F}$, $\\alpha=0.05$.",
              f"tab:grid_{name}_b", status, "lcccccc", HEAD_B, long=True)
    return A, B


def npz(path):
    return np.load(path, allow_pickle=True) if os.path.exists(path) else None


# ------------------------------------------------------- scratch lineage
PRETTY = {"simclr": "SimCLR", "visreg": "VISReg/LeJEPA", "nplm": "NPLM (unsup.)", "supcon": "SupCon",
          "supsig": "SupCon+SIGReg (repulse)", "nplmcw": "NPLM-dist.+cw SIGReg",
          "ssig": r"SupCon+SIGReg ($\lambda$5)", "nplmsd": "NPLM-dist. (sup.)"}


def grid_scratch(ds):
    fm = os.path.join(LOGS, "exp136", f"master_{ds}.json")
    if not os.path.exists(fm):
        return None
    m = json.load(open(fm))
    r137 = json.load(open(os.path.join(LOGS, "exp137", f"residuals_{ds}.json"))) \
        if os.path.exists(os.path.join(LOGS, "exp137", f"residuals_{ds}.json")) else {}
    RA, RB, n, n_inj = [], [], 0, 0
    for a in ["supcon", "ssig", "nplmcw", "supsig", "nplmsd", "simclr", "visreg", "nplm"]:
        gA, gB = group(PRETTY[a]); RA.append(gA); RB.append(gB)
        r = m.get(a)
        if r:
            n += 1
            p, pw, d68 = r["pre"], r.get("pre_power", {}), r.get("discovery68") or {}
            fr = pw.get("fractions", [])
            pre_pow = lambda s: pw[s][fr.index(F)] if fr and F in fr and s in pw else None
            pp = d68.get("post_power", {}); fr68 = d68.get("fractions", [])
            post_pow = lambda s: pp[s][fr68.index(F)] if fr68 and F in fr68 and s in pp else None
            # post probe/eucl/mahaT from the INJECTED-sample pass (exp-68 postf_*
            # at F, read from the npz on disk), never the whole-class natural pass
            ij = t141.d68_injected(ds).get(a, {}); n_inj += "probe" in ij
            mm = {"probe": (p["probe"], ij.get("probe")),
                  "eucl": (p["eucl"], ij.get("eucl")),
                  "mahaT": (p["mahaT"], ij.get("mahaT")),
                  "perevt": (p["perevt"], post_pow("perevent")),
                  "sparker": (pre_pow("sparker"), post_pow("sparker")),
                  "maha": (pre_pow("maha"), post_pow("maha")),
                  "mmd": (pre_pow("mmd"), post_pow("mmd"))}
        else:
            mm = {}
        a1, b1 = row(r"\quad plain", mm); RA.append(a1); RB.append(b1)
        for obj, lab in (("res", r"\quad +res concat"), ("res-nplm", r"\quad +res-nplm concat")):
            k = f"{a}->{obj} (concat)"
            c = r137.get(k)
            mm = {"probe": (c["probe"], None), "eucl": (c["eucl"], None), "mahaT": (c["mahaT"], None),
                  "perevt": (c["perevt"], None)} if c else {}
            a1, b1 = row(lab, mm); RA.append(a1); RB.append(b1)
    tag = "CIFAR-10" if ds == "cifar10" else "CIFAR-100"
    b = "0.10" if ds == "cifar10" else "0.01"
    status = (f"{n}/8 objectives, from random init, 100-D, holdout 4 (single holdout, $b={b}$), seed 0. "
              f"`post' = the exp-68 fine-tuning loop (distance pool, 2 rounds), discovery from a {100 * F:.0f}\\% "
              f"injected sample; per-fraction post geometry present for {n_inj}/{n} objectives (`--' otherwise; "
              r"the whole-class natural pass is in the pools table). Residual concats: parents "
              r"SupCon / SupCon+SIGReg / NPLM-dist.\ only (exp 137); discovery on scratch concats not run.")
    return emit(f"{ds}_scratch", RA, RB,
                rf"\textbf{{{tag}, leakage-free lineage: every (objective, construction) $\times$ (pre, post).}}",
                status)


# ------------------------------------------------------------- hub lineage
HUB_ARMS = [("supcon", "SupCon"), ("supcon_sigreg", r"SupCon+SIGReg ($\lambda$1)"),
            ("nplm_sup_dist", "NPLM-dist. (sup.)"), ("nplm_dist_sup_cw", "NPLM-dist.+cw SIGReg"),
            ("nplm_bilinear", "NPLM-bilinear"), ("nplm_bil_cw", "NPLM-bil.+cw SIGReg"),
            ("nplm_bil_sup_cw", "NPLM-bil.(sup.)+cw SIGReg"), ("nplm_distance", "NPLM-dist. (unsup.)"),
            ("simclr", "SimCLR"), ("simclr_sigreg", "SimCLR+SIGReg"), ("lejepa", "LeJEPA")]


def grid_hub(ds):
    d50 = npz(os.path.join(LOGS, "exp50", f"results_nplm_{ds}.npz"))
    d53 = npz(os.path.join(LOGS, "exp53", f"results_classwise_{ds}.npz"))
    d55 = npz(os.path.join(LOGS, "exp55", f"discovery_nplm_{ds}.npz"))
    d58 = npz(os.path.join(LOGS, "exp58", f"nplm_ft_discovery_{ds}.npz"))
    d59 = npz(os.path.join(LOGS, "exp59", f"nplm_residual_concat_{ds}.npz"))
    dim = "32d" if ds == "cifar10" else "100d"
    d73 = npz(os.path.join(LOGS, "exp73", f"results_{ds}_{dim}.npz"))
    d74 = npz(os.path.join(LOGS, "exp74", f"results_{ds}_{dim}.npz"))
    d80 = npz(os.path.join(LOGS, "exp80", f"results_{ds}.npz"))
    if d50 is None:
        return None
    RA, RB, n = [], [], 0
    gA, gB = group("Standalone objectives, 32-D (exps 50/53; post = exp 55 proto fine-tune)"); RA.append(gA); RB.append(gB)
    for a, lab in HUB_ARMS:
        src = d50 if f"probe_{a}" in d50.files else (d53 if d53 is not None and f"probe_{a}" in d53.files else None)
        if src is None:
            continue
        n += 1
        mm = {"probe": (at(src, f"probe_{a}"), pair(d55, f"probe_{a}", 1)),
              "eucl": (at(src, f"eucl_{a}"), None), "mahaT": (at(src, f"mahaT_{a}"), None),
              "perevt": (at(src, f"perevent_{a}_pre"), at(d55, f"perevent_{a}_post")),
              "sparker": (at(src, f"sparker_{a}_pre"), at(d55, f"sparker_{a}_post")),
              "maha": (at(src, f"maha_{a}_pre"), at(d55, f"maha_{a}_post")),
              "mmd": (at(src, f"mmd_{a}_pre"), at(d55, f"mmd_{a}_post"))}
        a1, b1 = row(r"\quad " + lab, mm); RA.append(a1); RB.append(b1)
        if d58 is not None and f"probe_{a}_nplm" in d58.files:
            mm = {"probe": (None, pair(d58, f"probe_{a}_nplm", 1) if np.asarray(d58[f"probe_{a}_nplm"]).size > 1 else at(d58, f"probe_{a}_nplm")),
                  "perevt": (None, at(d58, f"perevent_{a}_nplm_post")),
                  "sparker": (None, at(d58, f"sparker_{a}_nplm_post")), "maha": (None, at(d58, f"maha_{a}_nplm_post")),
                  "mmd": (None, at(d58, f"mmd_{a}_nplm_post"))}
            a1, b1 = row(r"\quad\quad same, NPLM fine-tune (exp 58)", mm); RA.append(a1); RB.append(b1)
    if d59 is not None:
        gA, gB = group("Concatenated / residual spaces, 16+16 (exp 59; post = discovery on the concat)"); RA.append(gA); RB.append(gB)
        for a in [str(x) for x in d59["arms"]]:
            mm = {"probe": (at(d59, f"probe_{a}"), at(d59, f"probe_post_{a}")),
                  "eucl": (at(d59, f"eucl_{a}"), None), "mahaT": (at(d59, f"mahaT_{a}"), None),
                  "perevt": (at(d59, f"perevent_{a}_pre"), at(d59, f"perevent_{a}_post")),
                  "sparker": (at(d59, f"sparker_{a}_pre"), at(d59, f"sparker_{a}_post")),
                  "maha": (at(d59, f"maha_{a}_pre"), at(d59, f"maha_{a}_post")),
                  "mmd": (at(d59, f"mmd_{a}_pre"), at(d59, f"mmd_{a}_post"))}
            a1, b1 = row(r"\quad " + esc(a), mm); RA.append(a1); RB.append(b1)
    if d73 is not None:
        gA, gB = group(f"Residual fine-tune, {dim.replace('d', '')}+{dim.replace('d', '')} (exps 73/74/80)"); RA.append(gA); RB.append(gB)
        spaces = [("supcon_(parent)", r"\quad SupCon parent", "supcon", None),
                  ("supcon-res_(concat)", r"\quad\quad +res concat", "res-cat", "res"),
                  ("supcon-res-nplm_(concat)", r"\quad\quad +res-nplm concat", "resnplm-cat", "res-nplm"),
                  ("supcon-res_(residual)", r"\quad\quad res child alone", "res", None),
                  ("supcon-res-nplm_(residual)", r"\quad\quad res-nplm child alone", "resnplm", None),
                  ("supcon_sigreg_(parent)", r"\quad SupCon+SIGReg ($\lambda$1) parent", None, None),
                  ("supcon_sigreg-res_(concat)", r"\quad\quad +res concat", None, None)]
        for sp, lab, k80, k74 in spaces:
            if f"{sp}__probe" not in d73.files:
                continue
            mm = {"probe": (at(d73, f"{sp}__probe"), at(d74, f"{k74}_probe_post") if k74 else None),
                  "eucl": (at(d73, f"{sp}__eucl"), at(d74, f"{k74}_eucl_post") if k74 else None),
                  "mahaT": (at(d73, f"{sp}__mahaT"), at(d74, f"{k74}_maha_post") if k74 else None),
                  "perevt": (at(d73, f"{sp}__perevt"), at(d74, f"{k74}_post_perevent") if k74 else None),
                  "sparker": (at(d80, f"sparker_{k80}_pre") if k80 else None, at(d74, f"{k74}_post_sparker") if k74 else None),
                  "maha": (None, at(d74, f"{k74}_post_maha") if k74 else None),
                  "mmd": (None, at(d74, f"{k74}_post_mmd") if k74 else None)}
            a1, b1 = row(lab, mm); RA.append(a1); RB.append(b1)
    tag = "CIFAR-10" if ds == "cifar10" else "CIFAR-100"
    status = (f"{n} standalone arms; hub-pretrained trunk (the encoder saw the held-out class -- ablation lineage); "
              r"holdout 4, seed 0; post geometry for standalone arms was never measured (exp 55 recorded probe and power only).")
    return emit(f"{ds}_hub", RA, RB,
                rf"\textbf{{{tag}, hub-pretrained lineage (ablation): every (space, construction) $\times$ (pre, post).}}",
                status)


# ---------------------------------------------------------------- galaxy10
G10_ARMS = [("supcon-ft", "SupCon"), ("ss-ft", r"SupCon+SIGReg ($\lambda$5)"), ("nplm-sup-ft", "NPLM-dist. (sup.)"),
            ("simclr-ft", "SimCLR"), ("sigreg-ssl-ft", "SimCLR+SIGReg"), ("nplm-bil-ft", "NPLM-bilinear")]
G10_DRAWS = (0, 3, 5, 7, 8)


def grid_galaxy(base):
    Z70 = [z for z in (npz(os.path.join(LOGS, "exp70", f"results_galaxy10_{base}_ft70_h1_d{d}.npz")) for d in G10_DRAWS) if z is not None]
    if not Z70:
        return None
    Z71 = [z for z in (npz(os.path.join(LOGS, "exp71", f"results_galaxy10_{base}_ft71_h1_d{d}.npz")) for d in G10_DRAWS) if z is not None]
    Z72 = [z for z in (npz(os.path.join(LOGS, "exp72", f"residual_discovery_h1_d{d}_galaxy10_{base}.npz")) for d in G10_DRAWS) if z is not None]
    d80 = npz(os.path.join(LOGS, "exp80", f"results_galaxy10_{base}.npz"))
    key = f"galaxy10_{base}"
    gl = lambda Z, k: [at(z, k) for z in Z if at(z, k) is not None]
    RA, RB = [], []
    for a, lab in G10_ARMS:
        gA, gB = group(lab); RA.append(gA); RB.append(gB)
        # post geometry from the INJECTED-sample pass (postf_* at F, the paper
        # regime), never the natural whole-class pass (post_probe_* etc.);
        # npz files predating the postf_* keys render as `--`.
        mm = {"probe": (gl(Z70, f"probe_{a}"), gl(Z70, f"postf_probe_{a}")),
              "eucl": (gl(Z70, f"eucl_{a}"), gl(Z70, f"postf_eucl_{a}")),
              "mahaT": (gl(Z70, f"mahaT_{a}"), gl(Z70, f"postf_mahaT_{a}")),
              "perevt": ([float(np.asarray(z[f"perevent_{a}_pre"]).ravel()[0]) for z in Z70], gl(Z70, f"perevent_{a}_post")),
              "sparker": (gl(Z70, f"sparker_{a}_pre"), gl(Z70, f"sparker_{a}_post")),
              "maha": (gl(Z70, f"maha_{a}_pre"), gl(Z70, f"maha_{a}_post")),
              "mmd": (gl(Z70, f"mmd_{a}_pre"), gl(Z70, f"mmd_{a}_post"))}
        a1, b1 = row(r"\quad plain", mm); RA.append(a1); RB.append(b1)
        if a in ("supcon-ft", "ss-ft"):
            for obj, lab2, k80 in (("res", r"\quad +res concat", f"{a}-res-cat"), ("res-nplm", r"\quad +res-nplm concat", f"{a}-resnplm-cat")):
                sp = f"{a}-{obj}_(concat)"
                if not Z71 or f"{sp}__probe" not in Z71[0].files:
                    a1, b1 = row(lab2, {}); RA.append(a1); RB.append(b1); continue
                # exp 72 ran discovery on the WINNER concat only: supcon-ft->res for every galaxy10 base
                post = Z72 if (a == "supcon-ft" and obj == "res") else []
                mm = {"probe": (gl(Z71, f"{sp}__probe"), gl(post, f"{key}_probe_post")),
                      "eucl": (gl(Z71, f"{sp}__eucl"), gl(post, f"{key}_eucl_post")),
                      "mahaT": (gl(Z71, f"{sp}__mahaT"), gl(post, f"{key}_maha_post")),
                      "perevt": (gl(Z71, f"{sp}__perevt"), gl(post, f"{key}_post_perevent")),
                      "sparker": ([at(d80, f"sparker_{k80}_pre")] if d80 is not None and at(d80, f"sparker_{k80}_pre") is not None else None,
                                  gl(post, f"{key}_post_sparker")),
                      "maha": (None, gl(post, f"{key}_post_maha")),
                      "mmd": (None, gl(post, f"{key}_post_mmd"))}
                a1, b1 = row(lab2, mm); RA.append(a1); RB.append(b1)
    n_pf = sum(1 for z in Z70 if any(k.startswith("postf_") for k in z.files))
    status = (f"galaxy10 / {base}, single holdout, {len(Z70)} draws (classes 2,3,4,5,6), mean$\\pm$sd across draws, "
              f"one seed per draw; plain-arm post columns = discovery from a {100 * F:.0f}\\% injected sample "
              f"(per-fraction post geometry present in {n_pf}/{len(Z70)} draws, `--' otherwise; the whole-class "
              rf"natural pass is in Table~\ref{{tab:app_galaxy10_draws}}); residual concats from exp 71 ({len(Z71)} draws), post-discovery on the "
              f"SupCon+res concat only (exp 72, {len(Z72)} draws); concat SparKer pre from the archived draw (exp 80).")
    return emit(f"galaxy10_{base}", RA, RB,
                rf"\textbf{{galaxy10 / {base}: every (objective, construction) $\times$ (pre, post), across holdout draws.}}",
                status)


TABLES = [("cifar10_scratch", lambda: grid_scratch("cifar10")), ("cifar100_scratch", lambda: grid_scratch("cifar100")),
          ("cifar10_hub", lambda: grid_hub("cifar10")), ("cifar100_hub", lambda: grid_hub("cifar100")),
          ("galaxy10_dino", lambda: grid_galaxy("dino")), ("galaxy10_lejepa", lambda: grid_galaxy("lejepa")),
          ("galaxy10_visreg", lambda: grid_galaxy("visreg"))]


def _check(name, t):
    ncol = t.split("{longtable}{")[1].split("}")[0]; ncol = sum(1 for c in ncol if c in "lcr")
    body = t.split(r"\endfoot")[1].split(r"\end{longtable}")[0]
    for line in body.strip().split("\n"):
        if not line.strip().endswith(r"\\"):
            continue
        spans = [int(x) for x in re.findall(r"\\multicolumn\{(\d+)\}", line)]
        n = line.count("&") + 1 - len(spans) + sum(spans)
        assert n == ncol, f"{name}: {n} vs {ncol}: {line[:80]}"
    return ncol


def selftest():
    ok = True
    for name, fn in TABLES:
        try:
            r = fn()
        except Exception as e:                       # noqa: BLE001
            print(f"  [FAIL] {name}: {type(e).__name__}: {e}"); ok = False; continue
        if r is None:
            print(f"  [skip] {name}"); continue
        for suf, t in zip("ab", r):
            try:
                nc = _check(name + suf, t); print(f"  [ok] grid_{name}_{suf}: {t.count(chr(92)+chr(92))} rows, {nc} cols")
            except AssertionError as e:
                print(f"  [FAIL] {e}"); ok = False
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
        r = fn()
        if r is None:
            print(f"  [skip] {name}"); continue
        for suf, t in zip("ab", r):
            p = os.path.join(args.out, f"grid_{name}_{suf}.tex"); open(p, "w").write(t)
            print(f"  wrote {os.path.relpath(p, REPO)}")


if __name__ == "__main__":
    main()
