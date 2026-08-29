"""
Experiment 136: the from-scratch CIFAR master battery -- every metric on every
leakage-free space, for cifar10 and cifar100, at any holdout.

WHY.  Every CIFAR cell in the campaign except exps 67/68 was built on hub
weights that saw the held-out class (pitfall 9).  The scratch lineage carried
only a light sanity battery (probe/acc/eucl/mahaT) and discovery on three
supervised bases, on cifar100, at holdout {4}.  This script fills the grid:

  datasets   cifar10, cifar100
  arms       the eight exp-67 objectives (simclr, visreg, nplm, supcon,
             supsig, nplmcw, ssig, nplmsd)
  holdouts   the archived {4} plus random draws (passed as --holdout N;
             exps 67/68 tag their artifacts _h{N} away from the default)

For each (dataset, holdout, arm) it loads the exp-67 checkpoint, extracts the
train/test banks (saved to logs/exp136/banks/ for exps 128/129/132/134), and
computes:

  PRE (frozen space)   probe (holdout-vs-rest AUC, 3 seeds); supervised top-1
                       (exp-132 linear probe, 3 seeds); acc / supAUC / eucl /
                       mahaT / mahaPC / lid; per-event power; SparKer / Maha /
                       MMD power at the CIFAR fractions (annealed sigma,
                       exp-31/32 protocol, N_D=5000)
  FROZEN POOLS         round-1/2 purity with the trunk frozen: distance pool
                       and density-ratio (np) pool, each with BN frozen (A)
                       and BN adapting (B, exp 133); legal-cut round-1
                       purity + ok flag (exps 129/131)
  DISCOVERY (exp 68)   merged from logs/exp68/scratch_discovery_{ds}{tag}.npz
                       when present: probe pre/post, purity r1/r2, post
                       acc/eucl/mahaT/lid, post per-event/SparKer/Maha/MMD

Output: logs/exp136/master_{ds}{tag}.json and one printed row per arm.
Aggregate across holdouts with experiments/136_aggregate.py.

    python experiments/136_scratch_master.py --selftest
    python experiments/136_scratch_master.py --dataset cifar100
    python experiments/136_scratch_master.py --dataset cifar10 --holdout 8 --skip-power
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import copy
import importlib
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from supersig import poolcut
from supersig.config import DEVICE
from supersig.discovery import run_discovery, np_pool_scores
from supersig.models import CIFARResNetBackbone
from supersig.recipes import recipe
from supersig.train import collect_embeddings

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.path.join(REPO, "checkpoints")
ARMS = ["simclr", "visreg", "nplm", "supcon", "supsig", "nplmcw", "ssig", "nplmsd"]
LABELED = {"supcon", "supsig", "nplmcw", "ssig", "nplmsd"}
STATS = ["perevent", "sparker", "maha", "mmd"]
FRACS = {"cifar10": [0.001, 0.003, 0.01, 0.02, 0.03, 0.1],
         "cifar100": [0.001, 0.003, 0.01, 0.02, 0.05]}


def htag(holdout):
    return "" if holdout == 4 else f"_h{holdout}"


def exp68_purity_inj(ds, holdout):
    """{arm: {f: round-1 purity}} from the exp-68 log's POST grid (discovery
    from an injected sample).  Round lines precede the `[arm] per-event post
    f=` line that names the arm, so round 1 is buffered until it arrives."""
    import re
    out, cur, r1 = {}, None, None
    for fn in (os.path.join("logs", f"exp68_{ds}_h{holdout}.log"),
               os.path.join("logs", "exp68", f"exp68_scratch_discovery_{ds}{htag(holdout)}.log")):
        if os.path.exists(fn):
            break
    else:
        return out
    for line in open(fn, errors="ignore"):
        m = re.match(r"===== POST grid, f=([\d.]+)", line)
        if m:
            cur, r1 = float(m.group(1)), None; continue
        if cur is None:
            continue
        m = re.match(r"\s+round 1: pool=\d+ purity=([\d.]+)", line)
        if m and r1 is None:
            r1 = float(m.group(1)); continue
        m = re.match(r"\s+\[(\S+)\] per-event post f=", line)
        if m:
            if r1 is not None:
                out.setdefault(m.group(1), {})[cur] = r1
            r1 = None
    return out


def _selftest():
    """The artifact tags must agree across 67/68/136 and be empty at the
    archived holdout so nothing existing is renamed."""
    assert htag(4) == "" and htag(8) == "_h8"
    print("  holdout tag: '' at 4, '_h8' at 8                       OK")
    src67 = open(os.path.join(REPO, "experiments", "67_scratch_pretrain.py")).read()
    src68 = open(os.path.join(REPO, "experiments", "68_scratch_discovery.py")).read()
    assert 'f"_h{args.holdout}"' in src67 and 'f"_h{args.holdout}"' in src68
    print("  exps 67/68 carry the same _h tag                       OK")
    assert set(ARMS) == set(["simclr", "visreg", "nplm", "supcon", "supsig",
                             "nplmcw", "ssig", "nplmsd"])
    print("  eight arms = the exp-67 cube                            OK")
    print("\nselftest OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dataset", default="cifar100", choices=["cifar10", "cifar100"])
    ap.add_argument("--holdout", type=int, default=4)
    ap.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    ap.add_argument("--dim", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--n-d", type=int, default=5000)
    ap.add_argument("--kernels", type=int, default=16)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--skip-power", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--legal-only", action="store_true",
                    help="recompute only the legal-cut entries (both n_min) from "
                         "the saved banks into an existing master JSON")
    ap.add_argument("--out", default="logs/exp136")
    args = ap.parse_args()
    if args.selftest:
        _selftest(); return

    exp28 = importlib.import_module("28_concat_residual")
    exp29 = importlib.import_module("29_residual_finetune")
    exp30 = importlib.import_module("30_power_curves")
    exp31 = importlib.import_module("31_sparker_power")
    exp32 = importlib.import_module("32_maha_mmd_power")
    exp132 = importlib.import_module("132_supervised_probe")
    from supersig.data import get_cifar_loaders

    ds, hold = args.dataset, args.holdout
    tag = htag(hold)
    cfg = recipe(ds, emb_dim=args.dim)
    n_cls = cfg["n_classes"]
    holdouts = {hold}
    seen = [c for c in range(n_cls) if c not in holdouts]
    fracs = FRACS[ds]
    n_null = 20 if args.quick else 200
    n_sig_toys = 10 if args.quick else 50
    steps = 50 if args.quick else args.steps
    sparker_kw = dict(M=args.kernels, steps=steps)
    os.makedirs(os.path.join(args.out, "banks"), exist_ok=True)
    out_path = os.path.join(args.out, f"master_{ds}{tag}.json")
    results = json.load(open(out_path)) if (os.path.exists(out_path) and not args.refresh) else {}

    train_loader, test_loader = get_cifar_loaders(quick=args.quick, dataset=ds)
    tel = DataLoader(train_loader.dataset, batch_size=256, shuffle=False, num_workers=2)
    f68 = os.path.join("logs", "exp68", f"scratch_discovery_{ds}{tag}.npz")
    d68 = np.load(f68, allow_pickle=True) if os.path.exists(f68) else None
    print(f"exp136 [{ds}{tag}] scratch master battery; holdout {hold}; "
          f"arms={args.arms}; exp68 {'merged' if d68 is not None else 'absent'}",
          flush=True)

    if args.legal_only:
        for arm in args.arms:
            bp = os.path.join(args.out, "banks", f"embs_{arm}_{ds}{tag}.npz")
            if arm not in results or not os.path.exists(bp):
                print(f"  [{arm}] no cached result/bank, skip"); continue
            b = np.load(bp)
            tr, tr_lab = b["tr"], b["tr_lab"]
            m = np.isin(tr_lab, seen)
            f, cal = np_pool_scores(torch.as_tensor(tr, dtype=torch.float32, device=DEVICE),
                                    m, seed=args.seed, return_calib=True)
            f = f.cpu().numpy()
            for nm, key in ((10, "legal_cut"), (30, "legal_cut_n30")):
                mask, info = poolcut.legal_pool(f, m, n_min=nm)
                results[arm][key] = dict(
                    n_min=nm, ok=bool(info["ok"]), reason=info.get("reason", ""),
                    q=float(mask.mean()), pool=int(mask.sum()),
                    purity=float((~m)[mask].mean()) if mask.any() else 0.0,
                    n_novel=int((~m)[mask].sum()), calib_in=cal["calib_in"],
                    calib_out=cal["calib_out"])
                print(f"  [{arm}] legal n_min={nm}: ok={info['ok']} purity="
                      f"{results[arm][key]['purity']:.3f} n_novel={results[arm][key]['n_novel']}")
        with open(out_path, "w") as fh:
            json.dump(results, fh, indent=1, default=float)
        print(f"wrote {out_path}"); return

    for arm in args.arms:
        if arm in results and not args.refresh:
            print(f"  [{arm}] cached"); continue
        ck = os.path.join(CKPT_DIR, f"scratch_{arm}_{ds}_{args.dim}d{tag}.pt")
        if not os.path.exists(ck):
            print(f"  [{arm}] missing {ck}, skip"); continue
        print(f"\n===== [{ds}{tag}] {arm} =====", flush=True)
        net = CIFARResNetBackbone(args.dim, arch=cfg["arch"], pretrain=None).to(DEVICE)
        state = torch.load(ck, map_location=DEVICE)
        net.load_state_dict(state["state_dict"])
        net.eval()
        tr, tr_lab = collect_embeddings(net, tel)
        te, te_lab = collect_embeddings(net, test_loader)
        np.savez(os.path.join(args.out, "banks", f"embs_{arm}_{ds}{tag}.npz"),
                 tr=tr, tr_lab=tr_lab, te=te, te_lab=te_lab)
        m = np.isin(tr_lab, seen)
        r = {"arm": arm, "labeled": arm in LABELED, "holdout": hold}

        # ---- PRE battery ---------------------------------------------------
        cents = exp28.class_centroids(tr[m], tr_lab[m], seen)
        anch = torch.as_tensor(cents, dtype=torch.float32, device=DEVICE)
        ev = exp29.evaluate_space(tr, tr_lab, te, te_lab, anch, seen, holdouts)
        aucs = []
        for s in range(3):
            torch.manual_seed(1000 + s)
            a, _, _ = exp29.linear_probe_novelty(tr, tr_lab, te, te_lab, holdouts)
            aucs.append(a)
        top1, top1_sd, _ = exp132.probe_multiseed(tr, tr_lab, te, te_lab, seen, seeds=3)
        d = torch.cdist(torch.as_tensor(te, dtype=torch.float32, device=DEVICE), anch)
        s_ = d.min(1).values.cpu().numpy()
        bg_mask, sig_mask = np.isin(te_lab, seen), np.isin(te_lab, list(holdouts))
        pe = exp30.power_at_alpha(s_[bg_mask], s_[sig_mask], args.alpha)
        r["pre"] = dict(probe=float(np.mean(aucs)), probe_sd=float(np.std(aucs)),
                        top1=top1, top1_sd=top1_sd, acc=ev["acc"], sup_auc=ev["sup_auc"],
                        eucl=ev["eucl"], mahaT=ev["maha_tied"], mahaPC=ev["maha_pc"],
                        lid=ev["lid"], perevt=pe)
        print(f"  pre: probe={r['pre']['probe']:.4f} top1={top1:.4f} acc={ev['acc']:.4f} "
              f"eucl={ev['eucl']:.4f} mahaT={ev['maha_tied']:.4f} lid={ev['lid']:.4f} "
              f"perevt={pe:.3f}", flush=True)
        if not args.skip_power:
            R = torch.as_tensor(tr[m][:20000], dtype=torch.float32, device=DEVICE)
            bg = torch.as_tensor(te[bg_mask], dtype=torch.float32, device=DEVICE)
            sg = torch.as_tensor(te[sig_mask], dtype=torch.float32, device=DEVICE)
            spk, _ = exp31.run_test_battery(bg, sg, R, fracs, args.n_d, n_null,
                                            n_sig_toys, args.alpha, args.seed,
                                            dict(sparker_kw), tag=f"{arm}-spk")
            maha_fn, mmd_fn, n_bg, n_sig = exp32.make_stats_fns(
                tr, tr_lab, te, te_lab, seen, holdouts, args.seed)
            maha, _ = exp32.battery(maha_fn, n_bg, n_sig, fracs, args.n_d, n_null,
                                    n_sig_toys, args.alpha, args.seed, tag=f"{arm}-maha")
            mmd, _ = exp32.battery(mmd_fn, n_bg, n_sig, fracs, args.n_d, n_null,
                                   n_sig_toys, args.alpha, args.seed, tag=f"{arm}-mmd")
            r["pre_power"] = dict(fractions=fracs, sparker=list(map(float, spk)),
                                  maha=list(map(float, maha)), mmd=list(map(float, mmd)))
            print(f"  pre power @{fracs}: spk {np.round(spk, 2).tolist()} "
                  f"maha {np.round(maha, 2).tolist()} mmd {np.round(mmd, 2).tolist()}",
                  flush=True)

        # ---- FROZEN pools: dist / np x BN frozen / adapting ---------------
        means0 = exp28.fill_means(cents, seen, cfg).detach()
        if arm == "supsig" and "means" in state:
            means0 = state["means"].to(DEVICE).detach()
        base_feats_ds = train_loader.dataset
        r["frozen"] = {}
        for scorer in ("dist", "np"):
            for var, adapt in (("frozen", False), ("bn-adapt", True)):
                bb = copy.deepcopy(net)
                for p in bb.parameters():
                    p.requires_grad_(False)
                torch.manual_seed(args.seed)
                _, hist = run_discovery(
                    bb, means0.clone(), base_ds=base_feats_ds, train_eval_loader=tel,
                    test_loader=test_loader, seen=seen, holdouts=holdouts,
                    dataset_name=ds, rep_weight=cfg["rep_weight"],
                    sigreg_weight=cfg["sigreg_weight"], n_slices=cfg["n_slices"],
                    rounds=args.rounds, ft_epochs=1 if args.quick else 2,   # anchors only: converges fast
                    seed=args.seed, pool_score=scorer, bn_adapt=adapt)
                r["frozen"][f"{scorer}|{var}"] = dict(
                    purity=[h["purity"] for h in hist], pool=[h["pool"] for h in hist],
                    margin=[h["margin"] for h in hist])
                print(f"  frozen pool [{scorer:4s} {var:8s}] purity "
                      + " ".join(f"r{h['round']}={h['purity']:.3f}" for h in hist), flush=True)
                del bb; torch.cuda.empty_cache()
        # legal cut, round 1, from the np scores
        f, cal = np_pool_scores(torch.as_tensor(tr, dtype=torch.float32, device=DEVICE),
                                m, seed=args.seed, return_calib=True)
        f = f.cpu().numpy()
        # Both operating points are recorded: n_min=30 (exps 129/131/135 and
        # Tier 1 as first run) and n_min=10 (poolcut default since exp 138).
        # `legal_cut` is the n_min=10 entry; `legal_cut_n30` the old one.
        for nm, key in ((10, "legal_cut"), (30, "legal_cut_n30")):
            mask, info = poolcut.legal_pool(f, m, n_min=nm)
            r[key] = dict(n_min=nm, ok=bool(info["ok"]), reason=info.get("reason", ""),
                          q=float(mask.mean()), pool=int(mask.sum()),
                          purity=float((~m)[mask].mean()) if mask.any() else 0.0,
                          n_novel=int((~m)[mask].sum()), calib_in=cal["calib_in"],
                          calib_out=cal["calib_out"])
            print(f"  legal cut n_min={nm}: ok={info['ok']} q={mask.mean():.4f} purity="
                  f"{r[key]['purity']:.3f} n_novel={r[key]['n_novel']} "
                  f"calib_out={cal['calib_out']:.2f}", flush=True)

        # ---- exp 68 merge --------------------------------------------------
        if d68 is not None and f"probe_{arm}" in d68.files:
            pp = d68[f"probe_{arm}"]
            r["discovery68"] = dict(
                probe_pre=float(pp[0]), probe_post=float(pp[1]),
                purity=[float(x) for x in d68[f"purity_{arm}"]] if f"purity_{arm}" in d68.files else None,
                post={k: float(d68[f"post_{k}_{arm}"]) for k in ("acc", "eucl", "maha_tied", "maha_pc", "lid")
                      if f"post_{k}_{arm}" in d68.files},
                post_power={s: [float(x) for x in d68[f"{s}_{arm}_post"]] for s in STATS
                            if f"{s}_{arm}_post" in d68.files},
                fractions=[float(x) for x in d68["fractions"]],
                # injected-sample pass (exp 68 >= 2026-08-29): per-fraction post
                # geometry from the npz and POST-grid round-1 purity from the log.
                # Absent keys stay absent -- the tables print `--`, never the
                # whole-class natural-pass value.
                postf={k: [float(x) for x in d68[f"postf_{k}_{arm}"]]
                       for k in ("probe", "eucl", "mahaT", "mahaPC")
                       if f"postf_{k}_{arm}" in d68.files},
                purity_inj=exp68_purity_inj(ds, hold).get(arm, {}))
        results[arm] = r
        with open(out_path, "w") as fh:
            json.dump(results, fh, indent=1, default=float)
        del net; torch.cuda.empty_cache()

    print(f"\n===== EXP136 MASTER [{ds}{tag}] (holdout {hold}) =====")
    print(f"  {'arm':<8}{'probe':>7}{'top1':>7}{'acc':>7}{'eucl':>7}{'mahaT':>7}{'lid':>7}"
          f"{'perevt':>8}{'spk@.02':>8}{'mmd@.02':>8}{'frz-np r1':>10}{'legal':>7}"
          f"{'68 post':>9}{'68 pur1':>8}")
    for arm in ARMS:
        r = results.get(arm)
        if not r: continue
        p = r["pre"]; fr = r.get("pre_power"); fz = r["frozen"].get("np|frozen", {})
        i02 = fracs.index(0.02) if 0.02 in fracs else -1
        spk = f"{fr['sparker'][i02]:.2f}" if fr else "--"
        mmd = f"{fr['mmd'][i02]:.2f}" if fr else "--"
        d68r = r.get("discovery68")
        print(f"  {arm:<8}{p['probe']:>7.3f}{p['top1']:>7.3f}{p['acc']:>7.3f}{p['eucl']:>7.3f}"
              f"{p['mahaT']:>7.3f}{p['lid']:>7.3f}{p['perevt']:>8.3f}{spk:>8}{mmd:>8}"
              f"{(fz.get('purity') or [float('nan')])[0]:>10.3f}"
              f"{r['legal_cut']['purity'] if r['legal_cut']['ok'] else float('nan'):>7.3f}"
              f"{(d68r['probe_post'] if d68r else float('nan')):>9.3f}"
              f"{((d68r['purity'] or [float('nan')])[0] if d68r else float('nan')):>8.3f}")
    print(f"wrote {out_path}\nEXP136 DONE.")


if __name__ == "__main__":
    main()
