"""
Experiment 149b: Galaxy10 sigma-currency tables (exps 150/151) + the
cross-dataset best-algorithm summary.  Companion to exp 149.

Emits, per backbone:
  sigma_galaxy_pre_<base>.tex   every (space, draw) point: f*(2sigma) for
                                Maha / MMD / SparKer at each of the 5 draws
  sigma_galaxy_post_<base>.tex  every (arm, test, draw) point of the
                                discovery-mode suite (np pool, derived cut,
                                n_min=5), with declined-at-crossing daggers
and one decision table:
  sigma_best.tex                best frozen and best discovery pipeline per
                                (dataset, backbone), by median f* over draws

    python experiments/149b_galaxy_tables.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glob
import importlib
import json
import numpy as np

e146 = importlib.import_module("146_min_frac_2sigma")
e149 = importlib.import_module("149_sigma_tables")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "docs", "tables")
BASES = ["lejepa", "visreg", "dino"]
DRAWS = [0, 3, 5, 7, 8]
PRE_TESTS = ["maha", "mmd", "sparker"]
PRE_HEAD = {"maha": "Maha.", "mmd": "MMD", "sparker": "SparKer"}
POST_TESTS = e149.POST_TESTS
POST_HEAD = e149.POST_HEAD
ARMS = ["supcon-ft", "supcon-ft_res", "supcon-ft_resnplm",
        "ss-ft", "ss-ft_res", "nplm-sup-ft"]
PRETTY_ARM = {"supcon-ft": "SupCon", "ss-ft": r"SupCon+SIGReg ($\lambda{=}5$)",
              "nplm-sup-ft": "NPLM-dist.\\ (sup.)",
              "supcon-ft_res": r"SupCon $\to$ res (residual)",
              "supcon-ft_resnplm": r"SupCon $\to$ res-nplm (residual)",
              "ss-ft_res": r"SupCon+SIGReg $\to$ res (residual)"}
PRETTY_SPACE = {
    "supcon-cw-ft": r"SupCon+classwise SIGReg ($\lambda{=}5$)",
    "supcon-ft": "SupCon", "ss-ft": r"SupCon+SIGReg ($\lambda{=}5$)",
    "nplm-sup-ft": "NPLM-dist.\\ (sup.)", "simclr-ft": "SimCLR",
    "sigreg-ssl-ft": "SimCLR+SIGReg", "nplm-bil-ft": "NPLM-bilinear",
    "gcd-ft": r"GCD (transductive)", "gcd-sigreg-ft": r"GCD+SIGReg (transd.)",
}


def pretty_space(lab):
    if lab in PRETTY_SPACE:
        return PRETTY_SPACE[lab]
    parent, rest = lab.split("->")
    child, use = rest.split(" (")
    child = {"res": "res", "resnplm": "res-nplm"}.get(child, child)
    return (rf"\quad {PRETTY_SPACE[parent].split(' ')[0]}"
            rf" $\to$ {child} ({use}")


def fx(v):
    if v is None:
        return "--"
    if not np.isfinite(v):
        return "$>$.1"
    return f"{v:.3f}".lstrip("0")


def pre_data(base):
    out = {}
    for d in DRAWS:
        p = os.path.join(REPO, "logs", "exp150",
                         f"minfrac_galaxy10_{base}_d{d}.json")
        if os.path.exists(p):
            out[d] = json.load(open(p))
    return out


def post_data(base):
    out = {}
    for d in DRAWS:
        p = os.path.join(REPO, "logs", "exp151",
                         f"suite_galaxy10_{base}_d{d}_np_legal_nmin5.json")
        if os.path.exists(p):
            out[d] = json.load(open(p))
    return out


# ------------------------------------------------------------ galaxy pre
def t_galaxy_pre(base):
    data = pre_data(base)
    if not data:
        return None
    labels, n = [], 0
    for d in data.values():
        for lab in d:
            if lab not in labels:
                labels.append(lab)
    holdmap = {d: data[d][next(iter(data[d]))]["holdout"][0] for d in data}
    rows = []
    for lab in labels:
        n += 1
        probe = np.mean([data[d][lab]["metrics"]["probe"] for d in data
                         if lab in data[d]])
        pe = np.mean([data[d][lab]["metrics"]["perevt"] for d in data
                      if lab in data[d]])
        cells = []
        for t in PRE_TESTS:
            for d in DRAWS:
                e = data.get(d, {}).get(lab)
                cells.append(fx(e[t]["f2sigma"]) if e else "--")
        if lab.startswith("gcd-ft") or lab == "supcon-ft->res (residual)":
            rows.append(r"\addlinespace")
        rows.append(" & ".join([pretty_space(lab), f"{probe:.3f}",
                                f"{pe:.2f}"] + cells) + r" \\")
    hdr_draws = " & ".join(f"d{d}\\,(c{holdmap.get(d, '?')})" for d in DRAWS)
    head = (r"space & probe & per-ev. & "
            + " & ".join(rf"\multicolumn{{5}}{{c}}{{{PRE_HEAD[t]}}}"
                         for t in PRE_TESTS) + r" \\"
            + "\n" + r"\cmidrule(lr){4-8}\cmidrule(lr){9-13}\cmidrule(lr){14-18}"
            + "\n" + r"& & & " + " & ".join([hdr_draws] * 3).replace(
                hdr_draws, hdr_draws, 3) + r" \\")
    # build the per-test draw header properly
    head = (r"space & probe & per-ev. & "
            + " & ".join(rf"\multicolumn{{5}}{{c}}{{{PRE_HEAD[t]}}}"
                         for t in PRE_TESTS) + r" \\"
            + "\n" + r"\cmidrule(lr){4-8}\cmidrule(lr){9-13}\cmidrule(lr){14-18}"
            + "\n" + r"& & & " + " & ".join(
                " & ".join(f"c{holdmap.get(d, '?')}" for d in DRAWS)
                for _ in PRE_TESTS) + r" \\")
    status = (f"Galaxy10 / {base.upper() if base != 'lejepa' else 'LeJEPA'}, "
              f"{n} spaces $\\times$ 5 single-holdout draws (columns named by the "
              r"held-out class); exp-150 battery, 200 null / 50 signal toys, "
              r"$N_D{=}5000$, fractions $0.006$--$0.1$.  GCD arms trained on the "
              r"unlabelled corpus with the novel images present (transductive).")
    cap = (rf"\textbf{{Galaxy10 / {base}: pre-discovery $f^\star(2\sigma)$, every "
           r"(space, draw) point.} `$>$.1' = the test never reached $2\sigma$ by "
           r"$f{=}0.1$ on that draw.")
    return e149.wrap("\n".join(rows), cap, f"tab:sigma_galaxy_pre_{base}",
                     status, "lcc" + "c" * 15, head, size="footnotesize")


# ------------------------------------------------------------ galaxy post
def post_fstar_draw(e, t):
    zf = sorted((float(k), v) for k, v in e["z"].get(t, {}).items()
                if v is not None)
    if len(zf) < 2:
        return None, False
    f, _ = e146.f_star([k for k, _ in zf], [v for _, v in zf])
    hit = next((k for k, v in zf if v >= 2.0), None)
    declined = (hit is not None
                and not e.get("cut", {}).get(str(hit), {}).get("ok", True))
    return f, declined


def t_galaxy_post(base):
    data = post_data(base)
    if not data:
        return None
    rows, n = [], 0
    for arm in ARMS:
        purs, engs = [], []
        for d in DRAWS:
            e = data.get(d, {}).get(arm)
            if not e:
                continue
            ok = [float(k) for k in e["cut"] if e["cut"][k]["ok"]]
            engs.append(f"d{d}: {min(ok):g}" if ok else f"d{d}: never")
            purs += [e["purity1"][k] for k in e["cut"] if e["cut"][k]["ok"]
                     and np.isfinite(e["purity1"][k])]
        pur_txt = (f"median engaged purity {np.median(purs):.2f}"
                   if purs else "never engaged")
        rows.append(rf"\multicolumn{{8}}{{l}}{{\emph{{{PRETTY_ARM[arm]}}} --- "
                    rf"engaged from {', '.join(engs)}; {pur_txt}}} \\")
        for t in POST_TESTS:
            cells, vals = [], []
            for d in DRAWS:
                e = data.get(d, {}).get(arm)
                if not e:
                    cells.append("--")
                    continue
                f, dec = post_fstar_draw(e, t)
                n += f is not None
                cells.append(fx(f) + (r"$^\dagger$" if dec and f is not None
                                      else ""))
                if f is not None:
                    vals.append(f if np.isfinite(f) else 0.15)
            med = fx(np.median(vals)) if vals else "--"
            rows.append(" & ".join([rf"\quad {POST_HEAD[t]}"] + cells
                                   + [med]) + r" \\")
        rows.append(r"\addlinespace")
    rows = rows[:-1]
    head = (r"test & " + " & ".join(f"d{d}" for d in DRAWS) + r" & median \\")
    status = (f"Galaxy10 / {base}, discovery mode: density-ratio pool, derived cut, "
              r"$n_{\min}{=}5$, skip-on-refuse; head-only loop (2 rounds $\times$ 5 "
              r"epochs) over the cached trunk banks; injection from the train bank "
              r"(clamped at the holdout pool size); 200 null / 50 signal toys; "
              f"{n} (test, draw) points.")
    cap = (rf"\textbf{{Galaxy10 / {base}: post-discovery $f^\star(2\sigma)$, every "
           r"(arm, test, draw) point.} $\dagger$ = the derived cut had declined at "
           r"the crossing fraction (the space there is the frozen one); `$>$.1' = "
           r"never reached $2\sigma$.  \emph{Eucl.\,(anch.)} and "
           r"\emph{SparKer\,(anch.)} consume the discovered anchors.")
    return e149.wrap("\n".join(rows), cap, f"tab:sigma_galaxy_post_{base}",
                     status, "lcccccc c", head, size="footnotesize")


# ------------------------------------------------------------ best summary
def cifar_pre_best():
    data = {}
    for h in (4, 7, 8, 9):
        p = os.path.join(REPO, "logs", "exp146", f"minfrac_cifar10_h{h}.json")
        if os.path.exists(p):
            data[h] = json.load(open(p))
    best = (np.inf, None, None)
    for lab in data[4]:
        for t in PRE_TESTS:
            vals = []
            for h in data:
                e = data[h].get(lab) or data[h].get(
                    lab.replace("(parent)", "(no residual)"))
                if e:
                    v = e[t]["f2sigma"]
                    vals.append(v if np.isfinite(v) else 0.15)
            med = np.median(vals)
            if med < best[0]:
                best = (med, lab, t)
    return best


def cifar_post_best(ds="cifar10", holdouts=(4, 7, 8, 9)):
    best = (np.inf, None, None)
    for arm in ("supcon", "ssig", "nplmsd", "nplmcw",
                "supcon-res", "supcon-resnplm", "ssig-res", "ssig-resnplm",
                "nplmsd-res", "nplmsd-resnplm", "nplmcw-res",
                "nplmcw-resnplm"):
        for t in POST_TESTS:
            vals = []
            for h in holdouts:
                p = os.path.join(REPO, "logs", "exp148",
                                 f"suite_{ds}_h{h}_np_legal.json")
                if not os.path.exists(p):
                    continue
                e = json.load(open(p)).get(arm)
                if not e:
                    continue
                f, dec = post_fstar_draw(e, t)
                if f is not None and not dec:     # declined = frozen number
                    vals.append(f if np.isfinite(f) else 0.15)
            if vals:
                med = np.median(vals)
                if med < best[0]:
                    best = (med, arm, t)
    return best


def galaxy_pre_best(base, transductive=False):
    data = pre_data(base)
    best = (np.inf, None, None)
    for lab in data[DRAWS[0]]:
        if lab.startswith("gcd") != transductive:
            continue
        for t in PRE_TESTS:
            vals = [data[d][lab][t]["f2sigma"] for d in data if lab in data[d]]
            vals = [v if np.isfinite(v) else 0.15 for v in vals]
            med = np.median(vals)
            if med < best[0]:
                best = (med, lab, t)
    return best


def galaxy_post_best(base):
    data = post_data(base)
    best = (np.inf, None, None)
    for arm in ARMS:
        for t in POST_TESTS:
            vals = []
            for d in data:
                e = data[d].get(arm)
                if not e:
                    continue
                f, _ = post_fstar_draw(e, t)
                if f is not None:
                    vals.append(f if np.isfinite(f) else 0.15)
            if vals:
                med = np.median(vals)
                if med < best[0]:
                    best = (med, arm, t)
    return best


def short(lab):
    lab = (lab.replace("-ft_resnplm", "@RN@").replace("-ft_res", "@R@")
           .replace("-resnplm", "@RN@").replace("-res", "@R@"))
    lab = (lab.replace("@RN@", r"$\to$res-nplm").replace("@R@", r"$\to$res"))
    return (lab.replace(" (parent)", "").replace("(residual)", "(resid.)")
            .replace("->", r"$\to$").replace("nplm-sup-ft", "NPLM-dist.")
            .replace("supcon-ft", "SupCon").replace("ss-ft", "SupCon+SIGReg")
            .replace("supcon", "SupCon").replace("ssig", "SupCon+SIGReg")
            .replace("nplmsd", "NPLM-dist.").replace("nplmcw", "NPLM-cw"))


def cifar100_pre_best():
    data = {}
    for h in (4, 43, 48, 57):
        p = os.path.join(REPO, "logs", "exp146", f"minfrac_cifar100_h{h}.json")
        if os.path.exists(p):
            data[h] = json.load(open(p))
    best = (np.inf, None, None)
    for lab in data[4]:
        for t in PRE_TESTS:
            vals = []
            for h in data:
                e = data[h].get(lab)
                if e:
                    v = e[t]["f2sigma"]
                    vals.append(v if np.isfinite(v) else 0.15)
            if vals:
                med = np.median(vals)
                if med < best[0]:
                    best = (med, lab, t)
    return best


def t_best():
    rows = []
    cm, cl, ct = cifar100_pre_best()
    dm100, dl100, dt100 = cifar_post_best("cifar100", (4, 43, 48, 57))
    rows.append(" & ".join([
        "CIFAR-100 (scratch, 4 draws)",
        rf"{short(cl)} / {PRE_HEAD[ct]}", f"{cm:.3f}",
        rf"{short(dl100)} / {POST_HEAD[dt100]}$^{{\rm e}}$", f"{dm100:.3f}",
        rf"{cm/dm100:.1f}$\times$" if dm100 < cm else "frozen wins"])
        + r" \\"[:3])
    pm, pl, pt = cifar_pre_best()
    qm, ql, qt = cifar_post_best()
    rows.append(" & ".join([
        "CIFAR-10 (scratch, 4 draws)",
        rf"{short(pl)} / {PRE_HEAD[pt]}", f"{pm:.3f}",
        rf"{short(ql)} / {POST_HEAD[qt]}", f"{qm:.3f}",
        rf"{pm/qm:.1f}$\times$" if qm < pm else "frozen wins"]) + r" \\")
    for base in BASES:
        pm, pl, pt = galaxy_pre_best(base)
        qm, ql, qt = galaxy_post_best(base)
        rows.append(" & ".join([
            f"Galaxy10 / {base}",
            rf"{short(pl)} / {PRE_HEAD[pt]}", f"{pm:.3f}",
            rf"{short(ql)} / {POST_HEAD[qt]}", f"{qm:.3f}",
            rf"{pm/qm:.1f}$\times$" if qm < pm else "frozen wins"]) + r" \\")
    rows.append(r"\addlinespace")
    for base in BASES:
        gm, gl, gt = galaxy_pre_best(base, transductive=True)
        rows.append(" & ".join([
            rf"\quad transductive ref.\ ({base})",
            rf"{short(gl)} / {PRE_HEAD[gt]}", f"{gm:.3f}", "--", "--",
            "sees the class"]) + r" \\")
    head = (r"dataset / backbone & \multicolumn{2}{c}{best frozen (space / test, "
            r"median $f^\star$)} & \multicolumn{2}{c}{best discovery (arm / test)} "
            r"& gain \\")
    status = (r"Medians over draws (censored cells imputed at $0.15$); CIFAR-10 "
              r"discovery = np pool + derived cut ($n_{\min}{=}10$), Galaxy10 "
              r"discovery = np pool + derived cut ($n_{\min}{=}5$), both "
              r"skip-on-refuse; frozen candidates exclude the transductive GCD "
              r"arms (shown separately).  Declined-cut crossings are excluded "
              r"from the discovery medians; $^{\rm e}$ = CIFAR-100 medians "
              r"cover the engaged draws only (the cut abstains on 33/48 "
              r"cells at $b{=}0.01$).")
    cap = (r"\textbf{The chosen algorithm, per dataset: best frozen space vs.\ "
           r"best discovery pipeline in the same currency.}  Discovery wins "
           r"everywhere, always through an anchor-aware test; the transductive "
           r"GCD rows are the ceiling given the whole novel class unlabelled.")
    return e149.wrap("\n".join(rows), cap, "tab:sigma_best", status,
                     "lccccc", head, size="footnotesize")


# ------------------------------------------------------------ diffuse shift
DIFF_ARMS = ["pretrained", "supcon-ft", "ss-ft", "nplm-sup-ft"]
DIFF_PRETTY = {"pretrained": "pretrained trunk", "supcon-ft": "SupCon",
               "ss-ft": r"SupCon+SIGReg ($\lambda{=}5$)",
               "nplm-sup-ft": "NPLM-dist.\\ (sup.)"}


def t_diffuse():
    rows, n = [], 0
    for base in BASES:
        data = {}
        for m, tg in (("one", "one_c2_blur2"), ("all", "all_blur2")):
            pth = os.path.join(REPO, "logs", "exp152",
                               f"diffuse_galaxy10_{base}_{tg}.json")
            if os.path.exists(pth):
                data[m] = json.load(open(pth))
        if not data:
            continue
        rows.append(rf"\multicolumn{{11}}{{l}}{{\emph{{{base}}}}} \\")
        for arm in DIFF_ARMS:
            cells = []
            for m in ("one", "all"):
                e = data.get(m, {}).get(arm)
                if not e:
                    cells += ["--"] * 5
                    continue
                n += 1
                mt = e["metrics"]
                cells += [f"{mt['auc_outlier']:.2f}",
                          f"{mt['auc_smear_probe']:.2f}",
                          fx(e["maha"]["f2sigma"]),
                          fx(e["mmd"]["f2sigma"]),
                          fx(e["sparker"]["f2sigma"])]
            rows.append(" & ".join([rf"\quad {DIFF_PRETTY[arm]}"] + cells)
                        + r" \\")
        rows.append(r"\addlinespace")
    rows = rows[:-1]
    head = (r"space & \multicolumn{5}{c}{one label (class 2)} & "
            r"\multicolumn{5}{c}{all seen labels} \\"
            "\n" r"\cmidrule(lr){2-6}\cmidrule(lr){7-11}"
            "\n" r"& out.\ AUC & probe & Maha. & MMD & SparKer & "
            r"out.\ AUC & probe & Maha. & MMD & SparKer \\")
    status = (f"Galaxy10, {n} (backbone, space, mode) cells; smear = Gaussian "
              r"blur $\sigma{=}2$\,px applied identically to $\le$1000 "
              r"seen-class test images (source rows removed from the "
              r"background pool); exp-146 toy battery.  \emph{out.\ AUC} = "
              r"min-anchor-distance ROC of smeared vs.\ background; "
              r"\emph{probe} = smeared-vs-clean linear AUC.")
    cap = (r"\textbf{Diffuse-shift detection: a systematic smear of seen-class "
           r"images instead of a novel class.}  The smear stays almost "
           r"perfectly \emph{decodable} (probe $0.86$--$1.00$) but the "
           r"fine-tuned discovery spaces barely see it as outlying; "
           r"mean-shift Mahalanobis becomes the only reliable test there, "
           r"while the raw pretrained trunk detects it at "
           r"$f^\star{=}0.006$--$0.010$ everywhere --- the systematics "
           r"monitor.  SparKer prefers the semi-clustered one-label smear in "
           r"every engaged comparison: the kernel tests' class-novelty "
           r"sharpness is a dependence on clusteredness.")
    return e149.wrap("\n".join(rows), cap, "tab:sigma_diffuse", status,
                     "l" + "c" * 10, head, size="footnotesize")


def t_diffuse_vs_novel():
    """Same label, two anomaly types: held-out class 2 vs smeared class 2."""
    pn = os.path.join(REPO, "logs", "exp152", "pretrained_novel_c2.json")
    pretrained_novel = json.load(open(pn)) if os.path.exists(pn) else {}
    rows, n = [], 0
    for base in BASES:
        d0 = os.path.join(REPO, "logs", "exp150",
                          f"minfrac_galaxy10_{base}_d0.json")
        sm = os.path.join(REPO, "logs", "exp152",
                          f"diffuse_galaxy10_{base}_one_c2_blur2.json")
        if not (os.path.exists(d0) and os.path.exists(sm)):
            continue
        novel, smear = json.load(open(d0)), json.load(open(sm))
        rows.append(rf"\multicolumn{{8}}{{l}}{{\emph{{{base}}}}} \\")
        for arm in DIFF_ARMS:
            ne = (pretrained_novel.get(base) if arm == "pretrained"
                  else novel.get(arm))
            se = smear.get(arm)
            cells = []
            for e in (ne, se):
                for t in PRE_TESTS:
                    cells.append(fx(e[t]["f2sigma"]) if e else "--")
            ratio = "--"
            if ne and se:
                bn = min(ne[t]["f2sigma"] for t in PRE_TESTS)
                bs = min(se[t]["f2sigma"] for t in PRE_TESTS)
                if np.isfinite(bn):
                    ratio = (f"{bs/bn:.1f}$\\times$"[:20] if np.isfinite(bs)
                             else "$>$" + f"{0.1/bn:.0f}$\\times$"[:20])
                n += 1
            rows.append(" & ".join([rf"\quad {DIFF_PRETTY[arm]}"] + cells
                                   + [ratio]) + r" \\"[:3])
        rows.append(r"\addlinespace")
    rows = rows[:-1]
    head = (r"space & \multicolumn{3}{c}{novel class 2 (held out)} & "
            r"\multicolumn{3}{c}{smeared class 2 (seen, blur $\sigma{=}2$)} "
            r"& cost \\" + "\n"
            r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}" + "\n"
            r"& Maha. & MMD & SparKer & Maha. & MMD & SparKer & \\")
    status = (f"Galaxy10, {n} (backbone, space) pairs, both anomalies built from "
              r"CLASS 2: left = class 2 held out of training and injected as a "
              r"novel class (exp-150 draw d0); right = class 2 seen in training, "
              r"a subset smeared and injected (exp 152).  The pretrained-trunk "
              r"row uses the identical space on both sides; the fine-tuned rows "
              r"differ in which class the fine-tune excluded (2 vs.\ the "
              r"archived 9).  cost = ratio of best-test $f^\star$, "
              r"smeared over novel.")
    cap = (r"\textbf{Same label, two anomaly types: class novelty vs.\ diffuse "
           r"smear.}  The reduction is per test family, not blanket: the "
           r"clustering-sensitive tests lose heavily on the smear (MMD censors "
           r"on 8 of 9 fine-tuned rows; SparKer costs $1.5$--$2.4\times$ where "
           r"both engage), while the mean-shift Mahalanobis often detects the "
           r"smear \emph{cheaper} than novelty (costs down to $0.3\times$) "
           r"--- but a mean-shift detection carries no class structure for an "
           r"anchor to mean.  The raw pretrained trunk sees the smear "
           r"$3$--$5\times$ better than it sees the same class as novelty "
           r"(cost $0.2$--$0.3\times$): the systematics monitor.")
    return e149.wrap("\n".join(rows), cap, "tab:sigma_diffuse_vs_novel",
                     status, "l" + "c" * 7, head, size="footnotesize")


def t_onmanifold():
    """Aircraft/cars: the on-manifold corner in the sigma currency."""
    rows, n = [], 0
    for ds in ("aircraft", "cars"):
        rows.append(rf"\multicolumn{{7}}{{l}}{{\emph{{{ds}}}}}" + r" \\")
        for base in BASES + []:
            pth = os.path.join(REPO, "logs", "exp153",
                               f"minfrac_{ds}_{base}.json")
            if not os.path.exists(pth):
                continue
            r = json.load(open(pth))
            ents = [r[k] for k in sorted(r)]
            n += len(ents)
            probe = np.mean([e["metrics"]["probe"] for e in ents])
            eucl = np.mean([e["metrics"]["eucl"] for e in ents])
            pe = np.mean([e["metrics"]["perevt"] for e in ents])
            cells = []
            for t in PRE_TESTS:
                vals = [e[t]["f2sigma"] for e in ents]
                cens = sum(1 for v in vals if not np.isfinite(v))
                med = np.median([v if np.isfinite(v) else 0.15 for v in vals])
                body = f"{med:.3f}" if cens < len(vals) / 2 else "$>$.1"
                cells.append(body + (rf"$^{{{cens}}}$" if cens else ""))
            rows.append(" & ".join([rf"\quad {base}", f"{probe:.3f}",
                                    f"{eucl:.3f}", f"{pe:.2f}"] + cells)
                        + r" \\"[:3])
        rows.append(r"\addlinespace")
    rows = rows[:-1]
    head = (r"dataset / backbone & probe & eucl AUC & per-ev. & "
            r"Maha. & MMD & SparKer" + r" \\")
    status = (f"Pretrained ViT-B/16 trunk (identity head), single holdout, 5 "
              f"draws per (dataset, backbone), {n} cells; medians over draws, "
              r"superscript = censored draws.  Held-out pools are tiny "
              r"(aircraft $\sim$33 test images/class), so toys bootstrap "
              r"them --- censoring here is conservative evidence.")
    cap = (r"\textbf{The on-manifold corner: fine-grained novelty inside the "
           r"seen manifold (aircraft variants, car models).}  The discovery "
           r"machinery is blind --- min-anchor-distance ranks the novel class "
           r"\emph{below} chance (eucl AUC $\sim$$0.47$ on cars), per-event "
           r"power is dead, pools starve at the natural rate --- yet the "
           r"novel class is still a dense clump, and SparKer's local kernels "
           r"detect the injected mass at $f^\star{=}0.012$--$0.021$ on cars, "
           r"every draw and backbone, while mean-shift Mahalanobis censors: "
           r"the exact mirror of the diffuse case.  Detection survives; "
           r"nothing pool-able or anchor-able comes with it.")
    return e149.wrap("\n".join(rows), cap, "tab:sigma_onmanifold", status,
                     "lccc ccc", head, size="footnotesize")


def main():
    tables = [("sigma_best", t_best), ("sigma_diffuse", t_diffuse),
              ("sigma_diffuse_vs_novel", t_diffuse_vs_novel),
              ("sigma_onmanifold", t_onmanifold)]
    for b in BASES:
        tables.append((f"sigma_galaxy_pre_{b}", lambda b=b: t_galaxy_pre(b)))
        tables.append((f"sigma_galaxy_post_{b}", lambda b=b: t_galaxy_post(b)))
    os.makedirs(OUT, exist_ok=True)
    for name, fn in tables:
        t = fn()
        if t is None:
            print(f"  [skip] {name}")
            continue
        open(os.path.join(OUT, f"{name}.tex"), "w").write(t)
        print(f"  wrote docs/tables/{name}.tex")


if __name__ == "__main__":
    main()
