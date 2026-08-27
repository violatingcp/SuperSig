"""
Experiment 132: a SUPERVISED linear probe -- the metric the campaign never ran.

WHY THIS EXISTS.  The paper wants to claim that discovery capability is not
bought at the cost of representation quality.  The campaign cannot currently
support that claim, for a reason that is easy to miss:

    THE `probe` COLUMN IN EVERY EXISTING TABLE IS NOT A SUPERVISED PROBE.

`exp29.linear_probe_novelty` trains a linear HOLDOUT-VS-REST head and reports
its AUC (docs/METRICS.md:14).  It is a novelty-detection metric.  Quoting it as
evidence of supervised representation quality would be simply wrong, and it is
the first thing a referee would catch.

The supervised evidence that does exist is `acc` (nearest-centroid top-1) and
the exp-62/63 closed-set numbers.  This script adds the missing piece: a
standard linear probe trained on the SEEN classes and evaluated by top-1, run
identically across arms so the comparison is controlled.

WHAT IT CAN AND CANNOT SETTLE.  Three traps this design is built to avoid:

  1. SUPERVISED vs SSL.  Our workhorse uses labels; the LeJEPA-style arm does
     not.  Beating it on a probe is unfair in our favour.  The comparison is
     therefore framed as "does adding SIGReg to a supervised objective COST
     anything", i.e. ss-ft vs supcon-ft, with the SSL arms reported for context
     only -- never as a headline win.
  2. PUBLISHED NUMBERS.  We never compare to published LeJEPA results, and must
     not start here: our LeJEPA backbone is a community reproduction
     (OK-AI/lejepa-vitb16-pretrain-in1k, card reports 72.0 IN-1k, explicitly
     below the official release) and there is no ImageNet evaluation anywhere
     in the campaign.
  3. FULL FINE-TUNING.  Exp 62 on aircraft: plain CE fine-tuning reaches 75.8
     -76.6 closed-set top-1 while ss-ft reaches 49.0-54.5.  Any broad "our
     objectives are competitive at classification" claim dies on that table.
     The defensible claim is narrower and is about the FROZEN-trunk / head
     regime, where exp 63 already shows supcon_sigreg heads beating CE heads.

TWO PROBES, because they answer different questions:
  trunk  (768-D backbone features)  did the fine-tune DAMAGE the backbone?
  emb    (the low-D head output)     is the SPACE WE CLAIM ABOUT decodable?
The second is the one the paper's claim is about; the first guards against
"we kept the embedding decodable by wrecking the trunk".

SEEDS ARE MANDATORY HERE.  The gaps at stake are small -- the
supcon_sigreg-vs-supcon differences the paper would like to cite are 0.008 and
0.016 -- while the campaign's own noise floor is 0.017 (seed, exp 52) and 0.019
(holdout draw, exp 118).  This script therefore refuses to declare a winner
unless the gap clears the measured spread, and prints TIE otherwise.

Evaluation-only given cached banks: fits a linear head, trains nothing else.

    python experiments/132_supervised_probe.py --selftest
    python experiments/132_supervised_probe.py --embs logs/exp54/embs_*_cifar10.npz
    python experiments/132_supervised_probe.py --cells cars:dino --seeds 5
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from supersig.holdouts import n_holdout, run_tag
import argparse
import glob
import json

import numpy as np
import torch
import torch.nn.functional as F

from supersig.config import DEVICE

# The campaign's measured noise floors; a gap below these is a TIE.
SEED_SPREAD = 0.017      # exp 52
DRAW_SPREAD = 0.019      # exp 118


def linear_probe_supervised(tr, tr_lab, te, te_lab, seen, epochs=40,
                            lr=1e-2, wd=0.0, bs=512, seed=0,
                            standardize=True):
    """Top-1 of a linear classifier trained on SEEN classes only.

    Deliberately plain: one `nn.Linear`, Adam, cross-entropy, fixed budget --
    the point is a controlled comparison ACROSS ARMS, not the best achievable
    number for any one arm.  Standardisation matters: these arms differ in
    embedding SCALE by design (the calibrated objectives fix unit class width,
    the softmax ones do not), and an unstandardised probe would partly measure
    scale rather than linear separability.
    """
    seen = np.asarray(sorted(seen))
    mtr, mte = np.isin(tr_lab, seen), np.isin(te_lab, seen)
    Xtr = np.ascontiguousarray(tr[mtr], dtype=np.float32)
    Xte = np.ascontiguousarray(te[mte], dtype=np.float32)
    if standardize:
        mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    ytr = np.searchsorted(seen, tr_lab[mtr])
    yte = np.searchsorted(seen, te_lab[mte])

    g = torch.Generator(device="cpu").manual_seed(seed)
    torch.manual_seed(seed)
    Xtr_t = torch.as_tensor(Xtr, device=DEVICE)
    ytr_t = torch.as_tensor(ytr, dtype=torch.long, device=DEVICE)
    Xte_t = torch.as_tensor(Xte, device=DEVICE)

    head = torch.nn.Linear(Xtr_t.size(1), len(seen)).to(DEVICE)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=wd)
    for _ in range(epochs):
        perm = torch.randperm(len(Xtr_t), generator=g).to(DEVICE)
        for i in range(0, len(Xtr_t), bs):
            idx = perm[i:i + bs]
            loss = F.cross_entropy(head(Xtr_t[idx]), ytr_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred = head(Xte_t).argmax(1).cpu().numpy()
    return float((pred == yte).mean())


def probe_multiseed(tr, tr_lab, te, te_lab, seen, seeds=3, **kw):
    vals = [linear_probe_supervised(tr, tr_lab, te, te_lab, seen, seed=s, **kw)
            for s in range(seeds)]
    return float(np.mean(vals)), float(np.std(vals)), vals


def verdict(a, b, sd_a, sd_b, floor=SEED_SPREAD):
    """Declare a winner only if the gap clears BOTH the measured spread and the
    campaign's noise floor.  Otherwise TIE.

    This is the guard that matters: the differences the paper would like to
    cite (0.008-0.016) sit under the floor (0.017-0.019), so an unguarded
    comparison would manufacture a result.
    """
    gap = a - b
    thresh = max(floor, sd_a + sd_b)
    if abs(gap) <= thresh:
        return "TIE", gap, thresh
    return ("A" if gap > 0 else "B"), gap, thresh


def _selftest():
    rng = np.random.default_rng(0)
    C, d, n = 10, 16, 400
    mus = rng.normal(0, 3.0, (C, d))
    y = np.repeat(np.arange(C), n)

    def bank(scale=1.0, noise=1.0, rot=False):
        X = np.concatenate([mus[c] + noise * rng.normal(0, 1, (n, d))
                            for c in range(C)]) * scale
        if rot:
            Q, _ = np.linalg.qr(rng.normal(0, 1, (d, d)))
            X = X @ Q
        return X.astype(np.float32)

    seen = list(range(C - 1))          # hold one class out, as the campaign does

    print("1. a clean space probes high; a noisy one probes low")
    hi = bank(noise=0.6)
    lo = bank(noise=6.0)
    a, sa, _ = probe_multiseed(hi, y, hi, y, seen, seeds=2, epochs=15)
    b, sb, _ = probe_multiseed(lo, y, lo, y, seen, seeds=2, epochs=15)
    print(f"   clean {a:.4f}+-{sa:.4f}   noisy {b:.4f}+-{sb:.4f}")
    assert a > b + 0.1, (a, b)

    print("\n2. the probe is INVARIANT to the embedding SCALE")
    print("   (arms differ in scale by design; an unstandardised probe would")
    print("    partly measure scale rather than separability)")
    base = bank(noise=1.0)
    for s in (0.1, 1.0, 10.0):
        v, _, _ = probe_multiseed(base * s, y, base * s, y, seen, seeds=1,
                                  epochs=15)
        print(f"   scale x{s:<5} -> {v:.4f}")
    v1, _, _ = probe_multiseed(base * 0.1, y, base * 0.1, y, seen, seeds=1,
                               epochs=15)
    v2, _, _ = probe_multiseed(base * 10.0, y, base * 10.0, y, seen, seeds=1,
                               epochs=15)
    assert abs(v1 - v2) < 0.05, (v1, v2)

    print("\n3. it is invariant to rotation (linear separability, not axes)")
    # NB one bank, used for BOTH train and test -- calling bank(rot=True) twice
    # would draw a different rotation for each and measure nothing.
    rotated = bank(rot=True)
    r1, _, _ = probe_multiseed(base, y, base, y, seen, seeds=1, epochs=15)
    r2, _, _ = probe_multiseed(rotated, y, rotated, y, seen, seeds=1, epochs=15)
    print(f"   unrotated {r1:.4f}   rotated {r2:.4f}")
    assert abs(r1 - r2) < 0.05, (r1, r2)

    print("\n4. it uses SEEN classes only -- the holdout never appears")
    assert C - 1 not in seen
    X = bank()
    v, _, _ = probe_multiseed(X, y, X, y, seen, seeds=1, epochs=10)
    print(f"   {len(seen)} seen classes, holdout {C-1} excluded -> {v:.4f}")

    print("\n5. the TIE guard refuses sub-noise gaps")
    for gap in (0.008, 0.016, 0.030):
        w, g, t = verdict(0.90 + gap, 0.90, 0.002, 0.002)
        print(f"   gap {gap:+.3f} vs floor {t:.3f} -> {w}")
    assert verdict(0.908, 0.900, 0.002, 0.002)[0] == "TIE"
    assert verdict(0.916, 0.900, 0.002, 0.002)[0] == "TIE"
    assert verdict(0.930, 0.900, 0.002, 0.002)[0] == "A"

    print("\nselftest OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--embs", nargs="*", default=[],
                    help="npz banks with tr/tr_lab/te/te_lab")
    ap.add_argument("--glob", default="")
    ap.add_argument("--holdouts", default="")
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--no-standardize", action="store_true")
    ap.add_argument("--baseline", default="",
                    help="arm name to compare everything against (e.g. supcon)")
    ap.add_argument("--out", default="logs/exp132")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    files = list(args.embs) + sorted(glob.glob(args.glob))
    if not files:
        ap.error("need --embs / --glob (or --selftest)")

    os.makedirs(args.out, exist_ok=True)
    rows = {}
    print(f"{'arm':28s}{'top1':>9s}{'sd':>8s}{'n_seen':>8s}{'dim':>6s}")
    print("-" * 60)
    for fn in files:
        d = np.load(fn, allow_pickle=True)
        need = {"tr", "tr_lab", "te", "te_lab"}
        if not need.issubset(set(d.files)):
            print(f"  !! {os.path.basename(fn)}: missing {need - set(d.files)}")
            continue
        tr, tr_lab = np.asarray(d["tr"]), np.asarray(d["tr_lab"])
        te, te_lab = np.asarray(d["te"]), np.asarray(d["te_lab"])
        n_cls = int(max(tr_lab.max(), te_lab.max())) + 1
        hold = (set(int(x) for x in args.holdouts.split(",") if x != "")
                if args.holdouts
                else set(range(n_cls - n_holdout(args.dataset), n_cls)))
        seen = [c for c in range(n_cls) if c not in hold]
        m, sd, vals = probe_multiseed(tr, tr_lab, te, te_lab, seen,
                                      seeds=args.seeds, epochs=args.epochs,
                                      lr=args.lr,
                                      standardize=not args.no_standardize)
        arm = os.path.basename(fn).replace("embs_", "").replace(".npz", "")
        rows[arm] = dict(top1=m, sd=sd, vals=vals, n_seen=len(seen),
                         dim=int(tr.shape[1]), holdouts=sorted(hold))
        print(f"{arm[:27]:28s}{m:>9.4f}{sd:>8.4f}{len(seen):>8d}"
              f"{tr.shape[1]:>6d}")

    if args.baseline and args.baseline in rows:
        base = rows[args.baseline]
        print(f"\nvs baseline '{args.baseline}' "
              f"(TIE unless the gap clears max(noise floor, sd_a+sd_b)):")
        for arm, r in rows.items():
            if arm == args.baseline:
                continue
            w, gap, thr = verdict(r["top1"], base["top1"], r["sd"], base["sd"])
            tag = {"A": arm, "B": args.baseline, "TIE": "TIE"}[w]
            print(f"  {arm[:30]:32s} {gap:+.4f}  (thresh {thr:.4f})  -> {tag}")
            rows[arm]["vs_baseline"] = dict(gap=gap, thresh=thr, winner=w)

    with open(os.path.join(args.out, f"probe{run_tag()}.json"), "w") as fh:
        json.dump(rows, fh, indent=1, default=float)
    print(f"\nwrote {args.out}/probe{run_tag()}.json")
    print("\nREMINDER: report this as a NO-COST argument (does adding SIGReg to "
          "a\nsupervised objective cost supervised accuracy?), not as a win "
          "over SSL\nbaselines -- our objectives use labels and theirs do not.")


if __name__ == "__main__":
    main()
