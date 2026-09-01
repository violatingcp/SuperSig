"""
Experiment 149: LaTeX tables for the sigma-currency battery (exps 146/148).

Reads logs/exp146/minfrac_<ds>_h<H>.json (pre-discovery, 3 tests x 20 spaces
x holdouts) and logs/exp148/suite_<ds>_h<H>[_np][_legal].json (post-discovery
suite, 6 tests) and writes docs/tables/sigma_*.tex in the house style (each
table carries a machine-written coverage line).  f* is the smallest injected
fraction at which the median expected significance reaches 2 sigma; a cell
whose curve never reaches 2 sigma on the grid prints as ">0.1", and a
post-discovery cell where the label-free cut DECLINED at every fraction below
the crossing prints its f* with a dagger.

    python experiments/149_sigma_tables.py [--dataset cifar10] [--holdouts 4,7,8,9]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import importlib
import json
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "docs", "tables")
e146 = importlib.import_module("146_min_frac_2sigma")

PRETTY = {
    "supcon": "SupCon", "ssig": r"SupCon+SIGReg ($\lambda{=}5$)",
    "nplmsd": "NPLM-dist.\\ (sup.)", "nplmcw": "NPLM-dist.+classwise SIGReg",
}
CHILD = {"res": "res", "res-nplm": "res-nplm", "resnplm": "res-nplm"}
POST_TESTS = ["eucl", "eucl-disc", "maha", "mmd", "sparker", "sparker-anch"]
POST_HEAD = {"eucl": "Eucl.", "eucl-disc": "Eucl.\\,(anch.)", "maha": "Maha.",
             "mmd": "MMD", "sparker": "SparKer", "sparker-anch": "SparKer\\,(anch.)"}
VARIANTS = [("", "distance pool, 95th-pct.\\ cut"),
            ("_legal", "distance pool, derived cut"),
            ("_np", "density-ratio pool, 95th-pct.\\ cut"),
            ("_np_legal", "density-ratio pool, derived cut")]


def pretty_space(label):
    """'supcon->res-nplm (residual)' -> 'SupCon $\\to$ res-nplm (residual)'."""
    if "(parent)" in label:
        return PRETTY[label.split(" ")[0]]
    parent, rest = label.split("->")
    child, use = rest.split(" (")
    return rf"\quad $\to$ {CHILD.get(child, child)} ({use}"


def pretty_arm(arm):
    if "-" not in arm:
        return PRETTY[arm]
    parent, child = arm.split("-", 1)
    return rf"\quad $\to$ {CHILD.get(child, child)} (residual)"


def fnum(v):
    if v is None:
        return "--"
    if np.isinf(v):
        return "$>0.1$"
    return f"{v:.3f}"


def wrap(rows, caption, label, status, colspec, header, size="small"):
    return "\n".join([
        f"% STATUS: {status}",
        r"\begin{table}[t]", r"\centering", f"\\{size}",
        f"\\begin{{tabular}}{{{colspec}}}", r"\toprule", header, r"\midrule",
        rows, r"\bottomrule", r"\end{tabular}",
        f"\\caption{{{caption} \\emph{{Coverage:}} {status}}}",
        f"\\label{{{label}}}", r"\end{table}", ""])


# --------------------------------------------------------------------- pre
def t_pre(ds, holdouts):
    data = {}
    for h in holdouts:
        p = os.path.join(REPO, "logs", "exp146", f"minfrac_{ds}_h{h}.json")
        if os.path.exists(p):
            data[h] = json.load(open(p))
    if not data:
        return None
    labels = [l for l, _, _, _ in e146.space_list(ds, 4)]
    rows, n_full = [], 0
    for lab in labels:
        ents = [data[h].get(lab) or data[h].get(lab.replace("(parent)", "(no residual)"))
                for h in data]
        ents = [e for e in ents if e]
        if not ents:
            continue
        if len(ents) == len(data):
            n_full += 1
        probe = np.mean([e["metrics"]["probe"] for e in ents if e["metrics"].get("probe")])
        perev = np.mean([e["metrics"]["perevt"] for e in ents if e["metrics"].get("perevt") is not None])
        cells = []
        for t in ("maha", "mmd", "sparker"):
            fs = [e[t]["f2sigma"] for e in ents]
            fin = [f for f in fs if np.isfinite(f)]
            med = np.median([f if np.isfinite(f) else 0.15 for f in fs])
            cens = len(fs) - len(fin)
            body = f"{med:.3f}" if cens < len(fs) / 2 else ">0.1"
            cells.append(f"${body}^{{{cens}}}$" if cens else
                         (f"${body}$" if body.startswith(">") else body))
        spk = [e["sparker"]["f2sigma"] for e in ents]
        spk_rng = (f"{min(spk):.3f}--{max(spk):.3f}" if all(np.isfinite(spk))
                   else "--")
        rows.append(" & ".join([pretty_space(lab), f"{probe:.3f}", f"{perev:.2f}"]
                               + cells + [spk_rng]) + r" \\")
        if "(parent)" in lab and lab != labels[0]:
            rows[-1] = r"\addlinespace" + "\n" + rows[-1]
    hs = ", ".join(str(h) for h in data)
    status = (f"{ds.upper().replace('CIFAR', 'CIFAR-')}, leakage-free lineage, "
              f"single holdout; draws {{{hs}}}; {n_full}/{len(labels)} spaces on every draw; "
              r"200 null / 50 signal toys per fraction, $N_D{=}5000$; median over draws, "
              r"superscript = draws where the test never reached $2\sigma$ by $f{=}0.1$.")
    cap = (r"\textbf{Pre-discovery: minimum injected fraction for a $2\sigma$ "
           r"dataset-level detection}, with and without the residual construction. "
           r"$f^\star$ is the smallest fraction of the held-out class injected into a "
           r"$5000$-point corpus at which the median expected significance of the "
           r"test reaches $2\sigma$; lower is better. The probe (oracle) and per-event "
           r"power are shown for reference; the last column is SparKer's range across draws.")
    head = (r"space & probe & per-ev. & \multicolumn{3}{c}{$f^\star(2\sigma)$, median over draws} & SparKer range \\"
            "\n" r"\cmidrule(lr){4-6}" "\n"
            r"& & & Maha. & MMD & SparKer & \\")
    return wrap("\n".join(rows), cap, f"tab:sigma_pre_{ds}", status, "lccccc c", head)


# -------------------------------------------------------------------- post
def load_suite(ds, h, tag):
    p = os.path.join(REPO, "logs", "exp148", f"suite_{ds}_h{h}{tag}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def post_fstar(e, t):
    zf = sorted((float(k), v) for k, v in e["z"].get(t, {}).items() if v is not None)
    if len(zf) < 2:
        return None, False
    f, _ = e146.f_star([k for k, _ in zf], [v for _, v in zf])
    # the crossing is attributed to the first grid fraction whose Z >= 2; if
    # the cut had DECLINED at that fraction the space there is the frozen one
    hit = next((k for k, v in zf if v >= 2.0), None)
    cut = e.get("cut", {})
    declined = (hit is not None and not cut.get(str(hit), {}).get("ok", True))
    return f, declined


def t_post(ds, h, tag, desc):
    r = load_suite(ds, h, tag)
    if not r:
        return None
    pre = {}
    p146 = os.path.join(REPO, "logs", "exp146", f"minfrac_{ds}_h{h}.json")
    if os.path.exists(p146):
        j = json.load(open(p146))
        for lab, e in j.items():
            arm = lab.split(" ")[0]
            if "(parent)" in lab or "(no residual)" in lab:
                pre[arm] = e["sparker"]["f2sigma"]
            elif "(residual)" in lab:
                parent, child = arm.split("->")
                pre[f"{parent}-{child.replace('-', '')}"] = e["sparker"]["f2sigma"]
    rows, n = [], 0
    parents = [a for a in r if "-" not in a]
    arms = []
    for p_ in parents:                      # each parent, then its children
        arms.append(p_)
        arms += [a for a in r if a.startswith(p_ + "-")]
    arms += [a for a in r if a not in arms]  # orphans (children without parent)
    last_parent = None
    for arm in arms:
        e = r[arm]
        n += 1
        cells, best = [], (np.inf, None)
        for t in POST_TESTS:
            f, dec = post_fstar(e, t)
            if f is not None and np.isfinite(f) and f < best[0]:
                best = (f, t)
            cells.append(fnum(f) + (r"$^\dagger$" if dec and f is not None else ""))
        cells = [(rf"\textbf{{{c}}}" if POST_TESTS[i] == best[1] else c)
                 for i, c in enumerate(cells)]
        pur = e["purity1"].get("0.03")
        n_done = len(e["z"].get("sparker", {}))
        partial = n_done < len(e.get("fractions", []))
        parent = arm.split("-")[0]
        if parent != last_parent and "-" not in arm and rows:
            rows.append(r"\addlinespace")
        last_parent = parent
        if partial:                       # arm still running: never show a curve
            cells = [f"({n_done}/{len(e['fractions'])})"] + ["--"] * (len(POST_TESTS) - 1)
        rows.append(" & ".join([pretty_arm(arm), fnum(pre.get(arm)),
                                f"{pur:.2f}" if pur is not None and np.isfinite(pur) else "--"]
                               + cells) + r" \\")
    status = (f"{ds.upper().replace('CIFAR', 'CIFAR-')} holdout {h}, {desc}; {n} arms; "
              r"each fraction scored in its own fine-tuned space (2 rounds); 200 null / 50 "
              r"signal toys, $N_D{=}5000$; purity = round-1 pool purity at $f{=}0.03$; "
              r"$\dagger$ = the label-free cut had declined at the crossing fraction "
              r"(the space is then the frozen one at that fraction); purity `--' = declined at $f{=}0.03$; "
              r"an `($n$/$m$)' row is an arm still running.")
    cap = (rf"\textbf{{Post-discovery $f^\star(2\sigma)$ on the {desc}.}} The frozen-space "
           r"SparKer $f^\star$ is repeated for reference; bold marks the best post-discovery "
           r"test per arm. \emph{Eucl.\,(anch.)} is the mean of $d_{\rm seen}-d_{\rm disc}$ "
           r"and \emph{SparKer\,(anch.)} seeds the kernels at the discovered anchors; both "
           r"consume the anchors discovery planted, the other four do not.")
    head = (r"arm & frozen SparKer & purity & " + " & ".join(POST_HEAD[t] for t in POST_TESTS) + r" \\")
    tagname = tag.strip("_").replace("_", "-") or "quantile"
    return wrap("\n".join(rows), cap, f"tab:sigma_post_{ds}_h{h}_{tagname}", status,
                "lcc" + "c" * len(POST_TESTS), head, size="footnotesize")


# ----------------------------------------------------------------- summary
SUMMARY_VARIANTS = [("", "dist/95th"), ("_legal", "dist/derived"),
                    ("_np_legal", "np/derived")]


def t_summary(ds, holdouts):
    """Per arm x draw: frozen SparKer f* vs the best post test per loop variant."""
    rows, n_cells, n_win = [], 0, 0
    parents = ["supcon", "ssig", "nplmsd", "nplmcw"]
    any_data = False
    for arm in parents:
        block = []
        for h in holdouts:
            p146 = os.path.join(REPO, "logs", "exp146", f"minfrac_{ds}_h{h}.json")
            frozen = None
            if os.path.exists(p146):
                j = json.load(open(p146))
                e = j.get(f"{arm} (parent)") or j.get(f"{arm} (no residual)")
                frozen = e["sparker"]["f2sigma"] if e else None
            cells = []
            for tag, _ in SUMMARY_VARIANTS:
                r = load_suite(ds, h, tag)
                e = r.get(arm) if r else None
                if not e or len(e["z"].get("sparker", {})) < len(e.get("fractions", [])):
                    cells.append("--")
                    continue
                any_data = True
                best = (np.inf, None, False)
                for t in POST_TESTS:
                    f, dec = post_fstar(e, t)
                    if f is not None and np.isfinite(f) and f < best[0]:
                        best = (f, t, dec)
                if best[1] is None:
                    cells.append("$>0.1$")
                    continue
                n_cells += 1
                # a crossing reached while the cut had declined is the FROZEN
                # space's number, not the loop's: never counted as a win
                win = (frozen is not None and np.isfinite(frozen)
                       and best[0] < frozen and not best[2])
                n_win += int(win)
                txt = f"{best[0]:.3f} ({POST_HEAD[best[1]].replace(chr(92)+',', ' ')})"
                txt += r"$^\dagger$" if best[2] else ""
                cells.append(rf"\textbf{{{txt}}}" if win else txt)
            block.append(" & ".join([PRETTY[arm] if h == holdouts[0] else "",
                                     f"h{h}", fnum(frozen)] + cells) + r" \\")
        rows += block + [r"\addlinespace"]
    if not any_data:
        return None
    rows = rows[:-1]
    status = (f"{ds.upper().replace('CIFAR', 'CIFAR-')}, four parents x draws "
              f"{{{', '.join(str(h) for h in holdouts)}}}; {n_cells} complete (arm, draw, variant) "
              f"cells, the loop beats the frozen space in {n_win}; $\\dagger$ = the derived cut had "
              r"declined at the crossing fraction, so that number is the frozen space's and is not counted "
              r"as a win; `--' = variant not run or still running.")
    cap = (r"\textbf{Does the discovery loop beat the frozen space?} Frozen-space SparKer "
           r"$f^\star(2\sigma)$ against the best post-discovery test (named in parentheses) under "
           r"three loop variants: the campaign's distance pool with the 95th-percentile cut, the "
           r"distance pool with the derived cut, and the density-ratio pool with the derived cut "
           r"(the paper's construction). Bold = the loop wins on that draw.")
    head = (r"arm & draw & frozen SparKer & " + " & ".join(d for _, d in SUMMARY_VARIANTS) + r" \\")
    return wrap("\n".join(rows), cap, f"tab:sigma_summary_{ds}", status, "llcccc", head,
                size="footnotesize")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--holdouts", default="4,7,8,9")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    hs = [int(x) for x in args.holdouts.split(",")]
    os.makedirs(args.out, exist_ok=True)
    tables = [(f"sigma_pre_{args.dataset}", lambda: t_pre(args.dataset, hs)),
              (f"sigma_summary_{args.dataset}", lambda: t_summary(args.dataset, hs))]
    for h in hs:
        for tag, desc in VARIANTS:
            name = f"sigma_post_{args.dataset}_h{h}{tag}"
            tables.append((name, (lambda h=h, tag=tag, desc=desc:
                                  t_post(args.dataset, h, tag, desc))))
    for name, fn in tables:
        t = fn()
        if t is None:
            print(f"  [skip] {name}: source absent")
            continue
        p = os.path.join(args.out, f"{name}.tex")
        open(p, "w").write(t)
        print(f"  wrote {os.path.relpath(p, REPO)}")


if __name__ == "__main__":
    main()
