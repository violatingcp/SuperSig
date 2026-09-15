"""
Experiment 149c: the discovery-cut comparison tables (2026-09) --
tight (legal, tv), s/sqrt(b), and max(tv,BBE) side by side, in the
f*(2sigma) currency, across CIFAR-10 / CIFAR-100 / Galaxy.

THE QUESTION.  The settled pipeline's pool cut declines whenever the
label-free estimated novel count falls below n_min -- an HONEST abstention,
but exp 138 showed the base-rate estimate `b_hat` (tv-sum) is biased LOW on
overlapping/low-rate spaces, so the cut over-abstains.  Two alternatives:

  s/sqrt(b)   (exp148 --cut ssb)  the significance-optimal cut: scan the
              density-ratio threshold, maximise the Punzi FOM
              s/(1 + sqrt(b)); never abstains.
  max(tv,bbe) (exp148 --cut legal --b-est max_tv_bbe)  keep the tight cut but
              feed it a less-biased b_hat = max(tv-sum, BBE best-bin MPE),
              which lowers the effective n_min per cell; abstains only when
              BOTH estimators agree novelty is absent.

Per (dataset, cut) we report, over all (arm, draw) cells: engagement rate,
anchor-aware rate (the winning test is eucl-disc or sparker-anch -- i.e. the
detection is carried by the DISCOVERED anchors, not a blind two-sample
test), best f*, and median f*.  A per-arm median table per dataset follows.

    python experiments/149c_cut_comparison.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib
import json
import numpy as np

e146 = importlib.import_module("146_min_frac_2sigma")
e149 = importlib.import_module("149_sigma_tables")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "docs", "tables")
POST = ["eucl", "eucl-disc", "maha", "mmd", "sparker", "sparker-anch"]
ANCHOR = {"eucl-disc", "sparker-anch"}

# (dataset label, loader-key, draws, cut-tag map)
CIFAR_ARMS = ["supcon", "ssig", "nplmsd", "nplmcw", "supcon-res",
              "supcon-resnplm", "ssig-res", "ssig-resnplm", "nplmsd-res",
              "nplmsd-resnplm", "nplmcw-res", "nplmcw-resnplm"]
C10_EXTRA = ["supconcw", "supsig", "supsig-res", "supsig-resnplm"]
GAL_ARMS = ["supcon-ft", "ss-ft", "nplm-sup-ft", "supcon-ft_res",
            "supcon-ft_resnplm", "ss-ft_res", "supcon-cw-ft"]
PRETTY = {
    "supcon": "SupCon", "ssig": r"SupCon+SIGReg", "nplmsd": "NPLM-dist.",
    "nplmcw": "NPLM+cw-SIGReg", "supconcw": "SupCon+cw-SIGReg",
    "supsig": "SupCon+SIGReg (rep.)",
    "supcon-ft": "SupCon", "ss-ft": r"SupCon+SIGReg",
    "nplm-sup-ft": "NPLM-dist.", "supcon-cw-ft": "SupCon+cw-SIGReg",
}


def pretty(arm):
    for stem in ("-resnplm", "-res", "_resnplm", "_res"):
        if arm.endswith(stem):
            base = arm[: -len(stem)]
            child = "res-nplm" if "resnplm" in stem else "res"
            bp = PRETTY.get(base, base.replace("_", r"\_"))
            return rf"\quad {bp.split(' ')[0]} $\to$ {child}"
    return PRETTY.get(arm, arm.replace("_", r"\_"))


def cut_cell(e):
    """Best f* over the six tests for one arm/draw, respecting decline."""
    fr = e["fractions"]
    best = (np.inf, None)
    for t in POST:
        zs = [e["z"].get(t, {}).get(str(f)) for f in fr]
        fv, _ = e146.f_star([k for k in fr if e["z"].get(t, {}).get(str(k))
                             is not None],
                            [z for z in zs if z is not None]) \
            if any(z is not None for z in zs) else (None, "")
        hit = next((f for f, z in zip(fr, zs)
                    if z is not None and z >= 2.0), None)
        dec = hit is not None and not e.get("cut", {}).get(
            str(hit), {}).get("ok", True)
        if fv is not None and np.isfinite(fv) and fv < best[0] and not dec:
            best = (fv, t)
    return best


def load_cifar(ds, draws, tag):
    out = {}
    for h in draws:
        p = os.path.join(REPO, "logs", "exp148",
                         f"suite_{ds}_h{h}_np{tag}.json")
        if os.path.exists(p):
            out[h] = json.load(open(p))
    return out


def load_galaxy(base, draws, tag):
    out = {}
    for d in draws:
        p = os.path.join(REPO, "logs", "exp151",
                         f"suite_galaxy10_{base}_d{d}_np{tag}.json")
        if os.path.exists(p):
            out[d] = json.load(open(p))
    return out


def _complete(e):
    """True iff every test has a value at every fraction (cell fully run)."""
    return all(str(f) in e["z"].get(t, {})
               for t in e["z"] for f in e.get("fractions", []))


def agg(suite, arms, draws, valid=None):
    fs, aw, eng, tot = [], 0, 0, 0
    for h in draws:
        if h not in suite:
            continue
        for a in arms:
            if a not in suite[h]:
                continue
            if valid is not None and (h, a) not in valid:
                continue
            tot += 1
            f, t = cut_cell(suite[h][a])
            if f is not None and np.isfinite(f):
                fs.append(f); eng += 1; aw += int(t in ANCHOR)
    best = min(fs) if fs else np.inf
    med = np.median(fs) if fs else np.inf
    return eng, tot, aw, best, med


def common_cells(suites, arms, draws):
    """(draw, arm) pairs that are COMPLETE in every cut's suite -- so the
    three cuts are compared on an identical cell set."""
    valid = set()
    for h in draws:
        for a in arms:
            if all(h in s and a in s[h] and _complete(s[h][a])
                   for s in suites.values()):
                valid.add((h, a))
    return valid


def fx(v):
    return f"{v:.3f}" if np.isfinite(v) else "$>$.1"


CUTS = [(r"tight", "the settled tight cut (label-free, tv-sum $b$, declines "
                   "below $n_{\\min}$)", "_legal"),
        (r"$s/\sqrt{b}$", "the significance-optimal Punzi cut (never abstains)",
         "_ssb"),
        (r"$\max($tv,BBE$)$", "the tight cut with the less-biased "
                             "$\\hat b=\\max($tv,\\,BBE$)$", "_legal_maxbbe")]
GAL_TAGS = {"_legal": "_legal_nmin5", "_ssb": "_ssb_nmin5",
            "_legal_maxbbe": "_legal_nmin5_maxbbe"}


def t_cut_summary():
    """One row per (dataset, cut): engage / anchor-aware / best / median."""
    rows = []
    blocks = [
        ("CIFAR-10", lambda tag: load_cifar("cifar10", [4, 7, 8, 9], tag),
         CIFAR_ARMS + C10_EXTRA, [4, 7, 8, 9]),
        ("CIFAR-100", lambda tag: load_cifar("cifar100", [4, 43, 48, 57], tag),
         CIFAR_ARMS, [4, 43, 48, 57]),
    ]
    for base in ("lejepa", "visreg", "dino"):
        blocks.append((f"Galaxy/{base}",
                       lambda tag, b=base: load_galaxy(b, [0, 3, 5, 7, 8],
                                                       GAL_TAGS[tag]),
                       GAL_ARMS, [0, 3, 5, 7, 8]))
    n = 0
    for label, loader, arms, draws in blocks:
        suites = {tag: loader(tag) for _, _, tag in CUTS}
        valid = common_cells(suites, arms, draws)
        first = True
        for cname, _, tag in CUTS:
            suite = suites[tag]
            if not suite:
                continue
            eng, tot, aw, best, med = agg(suite, arms, draws, valid)
            n += 1
            ds_cell = rf"\emph{{{label}}}" if first else ""
            first = False
            rows.append(" & ".join([
                ds_cell, cname, f"{eng}/{tot}",
                f"{aw}/{tot}", fx(best), fx(med)]) + r" \\")
        rows.append(r"\addlinespace")
    rows = rows[:-1]
    head = (r"dataset & cut & engage & anchor-aware & best $f^\star$ "
            r"& median $f^\star$ \\")
    status = (f"{n} (dataset, cut) rows; each aggregates all (arm, draw) "
              r"cells (CIFAR: 16/12 arms $\times$ 4 draws; Galaxy: 7 arms "
              r"$\times$ 5 draws).  \emph{engage} = cells with a finite "
              r"$f^\star$; \emph{anchor-aware} = of those, how many are won "
              r"by an anchor-using test (eucl-disc / sparker-anch) rather "
              r"than a blind two-sample test; best/median over engaged "
              r"cells.  np density-ratio pool, $n_{\min}{=}5$ throughout.")
    cap = (r"\textbf{The discovery-cut comparison.}  The tight cut is "
           r"tightest \emph{where it fires} but abstains often (esp.\ on the "
           r"residual children and the low-rate CIFAR-100 regime); "
           r"$s/\sqrt{b}$ and $\max($tv,BBE$)$ both engage nearly "
           r"everywhere and are statistically interchangeable, with "
           r"$s/\sqrt{b}$ marginally ahead (more anchor-aware wins, equal or "
           r"lower median $f^\star$) while being the simpler rule.  On the "
           r"clean-pool datasets (CIFAR-10, Galaxy) essentially every "
           r"detection is anchor-aware; only CIFAR-100's $b{=}0.01$ regime "
           r"splits into $\sim$half anchor-aware, half blind.")
    return e149.wrap("\n".join(rows), cap, "tab:cut_summary", status,
                     "llcccc", head, size="small")


def t_cut_perarm(ds_label, loader, arms, draws, tabname):
    """Per-arm median f* for the three cuts on one dataset."""
    rows, n = [], 0
    suites = {tag: loader(tag) for _, _, tag in CUTS}
    for a in arms:
        cells = []
        any_row = False
        for _, _, tag in CUTS:
            s = suites[tag]
            v = []
            for h in draws:
                if h not in s or a not in s[h]:
                    continue
                f, _ = cut_cell(s[h][a])
                if f is not None and np.isfinite(f):
                    v.append(f)
            if v:
                any_row = True
                cells.append(rf"{np.median(v):.3f}\,[{min(v):.3f}--"
                             rf"{max(v):.3f}]")
            else:
                cells.append(r"$>$.1")
        if any_row:
            n += 1
            rows.append(" & ".join([pretty(a)] + cells) + r" \\")
    if not rows:
        return None
    head = (r"method & tight & $s/\sqrt{b}$ & $\max($tv,BBE$)$ \\")
    status = (f"{ds_label}: per-method best-$f^\\star$ median\\,[min--max] "
              f"over {len(draws)} draws, three cuts; `$>$.1' = the method "
              r"never reached $2\sigma$ by $f{=}0.1$ under that cut "
              r"(the tight cut's blanks are its abstentions).")
    cap = (rf"\textbf{{{ds_label}: per-method $f^\star$ under the three "
           r"cuts.}  The tight cut's $>$.1 rows are arms it declines on every "
           r"draw --- overwhelmingly the residual children --- which "
           r"$s/\sqrt{b}$ and $\max($tv,BBE$)$ recover.")
    return e149.wrap("\n".join(rows), cap, f"tab:{tabname}", status,
                     "lccc", head, size="footnotesize")


def main():
    os.makedirs(OUT, exist_ok=True)
    tables = [("cut_summary", t_cut_summary),
              ("cut_perarm_cifar10",
               lambda: t_cut_perarm("CIFAR-10",
                                    lambda tag: load_cifar("cifar10",
                                                           [4, 7, 8, 9], tag),
                                    CIFAR_ARMS + C10_EXTRA, [4, 7, 8, 9],
                                    "cut_perarm_cifar10")),
              ("cut_perarm_cifar100",
               lambda: t_cut_perarm("CIFAR-100",
                                    lambda tag: load_cifar("cifar100",
                                                           [4, 43, 48, 57],
                                                           tag),
                                    CIFAR_ARMS, [4, 43, 48, 57],
                                    "cut_perarm_cifar100")),
              ("cut_perarm_galaxy_lejepa",
               lambda: t_cut_perarm("Galaxy/LeJEPA",
                                    lambda tag: load_galaxy("lejepa",
                                                            [0, 3, 5, 7, 8],
                                                            GAL_TAGS[tag]),
                                    GAL_ARMS, [0, 3, 5, 7, 8],
                                    "cut_perarm_galaxy_lejepa"))]
    for name, fn in tables:
        t = fn()
        if t is None:
            print(f"  [skip] {name}")
            continue
        open(os.path.join(OUT, f"{name}.tex"), "w").write(t)
        print(f"  wrote docs/tables/{name}.tex")


if __name__ == "__main__":
    main()
