"""
Experiment 72: discovery phase on the BEST residual spaces (exp-71
winners), one per dataset x base cell.

The exp-71 champion construction per cell (res-nplm concats on
fine-grained cells, plain res concats on texture/coarse, one pure
residual) is rebuilt from the cached ft-trunk banks, and the settled
feature-space discovery loop (exp-69/70 protocol) runs on it.  For a
concat space the discovery backbone is a two-head wrapper: parent head
on the parent-trunk bank, residual head on the child-trunk bank,
outputs concatenated (200-D); run_discovery fine-tunes both heads.
Reports probe/eucl/mahaT pre -> post, pool purity, and the injected
post-power grid (per-event / SparKer annealed / Maha / MMD).

    python experiments/72_residual_discovery.py
    python experiments/72_residual_discovery.py --cells cars:visreg --quick
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from supersig.holdouts import n_holdout, run_tag
import argparse
import copy
import importlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from supersig.config import DEVICE, REPO_DIR, plot_path
from supersig.discovery import run_discovery
from supersig.train import collect_embeddings

exp28 = importlib.import_module("28_concat_residual")
exp29 = importlib.import_module("29_residual_finetune")
exp30 = importlib.import_module("30_power_curves")
exp31 = importlib.import_module("31_sparker_power")
exp32 = importlib.import_module("32_maha_mmd_power")
exp43 = importlib.import_module("43_dtd_finetune")
exp44 = importlib.import_module("44_transfer_32d")
exp70 = importlib.import_module("70_cars_ft_suite")

CKPT_DIR = os.path.join(REPO_DIR, "checkpoints")
STATS = ["perevent", "sparker", "maha", "mmd"]

# (ds, base) -> (parent, objective, kind)   [exp-71 winners]
WINNERS = {
    ("aircraft", "dino"): ("supcon-ft", "resnplm", "concat"),
    ("aircraft", "lejepa"): ("supcon-ft", "resnplm", "concat"),
    ("aircraft", "visreg"): ("supcon-ft", "resnplm", "concat"),
    ("cars", "dino"): ("supcon-ft", "resnplm", "concat"),
    ("cars", "lejepa"): ("supcon-ft", "resnplm", "concat"),
    ("cars", "visreg"): ("supcon-ft", "resnplm", "concat"),
    ("flowers", "dino"): ("supcon-ft", "resnplm", "concat"),
    ("flowers", "lejepa"): ("ss-ft", "res", "concat"),
    ("flowers", "visreg"): ("supcon-ft", "resnplm", "concat"),
    ("dtd", "dino"): ("supcon-ft", "res", "concat"),
    ("dtd", "lejepa"): ("ss-ft", "res", "concat"),
    ("dtd", "visreg"): ("supcon-ft", "res", "concat"),
    ("galaxy10", "dino"): ("supcon-ft", "res", "concat"),
    ("galaxy10", "lejepa"): ("supcon-ft", "res", "concat"),
    ("galaxy10", "visreg"): ("supcon-ft", "res", "residual"),
}
REP_WEIGHT = 20.0


class ConcatHeads(nn.Module):
    """Parent head on x[:, :768], residual head on x[:, 768:], concat out."""

    def __init__(self, parent_head, child_head):
        super().__init__()
        self.p, self.c = parent_head, child_head

    def forward(self, x):
        return torch.cat([self.p(x[:, :768]), self.c(x[:, 768:])], dim=1)


def load_cell(ds, base, parent, obj, args):
    """Returns (backbone, Xtr, ytr, Xte, yte) for the winner space.
    args.ckpt_sfx (e.g. "_s1") selects seed-suffixed exp-70/71 ckpts."""
    exp70.DS, exp70.BASE = ds, base
    sfx = getattr(args, "ckpt_sfx", "")
    par = exp43.FineTuneModel(base, args.emb_dim)
    par.load_state_dict(torch.load(
        os.path.join(CKPT_DIR, f"{ds}_ft_{base}_{parent}_seen{run_tag()}{sfx}.pt"),
        map_location=DEVICE))
    child = exp43.FineTuneModel(base, args.emb_dim)
    child.load_state_dict(torch.load(
        os.path.join(CKPT_DIR, f"{ds}_ft_{base}_{parent}_{obj}_seen{run_tag()}{sfx}.pt"),
        map_location=DEVICE))
    bank_args = argparse.Namespace(quick=False, refresh=args.refresh,
                                   seed=int(sfx[2:]) if sfx else 0)
    pb = exp70.trunk_banks(par, parent, bank_args)
    cb = exp70.trunk_banks(child, f"{parent}_{obj}", bank_args)
    (pXtr, ytr), (pXte, yte) = pb["train"], pb["test"]
    (cXtr, _), (cXte, _) = cb["train"], cb["test"]
    ph = copy.deepcopy(par.head).float().to(DEVICE)
    ch = copy.deepcopy(child.head).float().to(DEVICE)
    del par, child
    torch.cuda.empty_cache()
    kind = WINNERS[(ds, base)][2]
    if kind == "residual":
        return ch, cXtr.float(), ytr, cXte.float(), yte
    bb = ConcatHeads(ph, ch).to(DEVICE)
    return (bb, torch.cat([pXtr, cXtr], 1).float(), ytr,
            torch.cat([pXte, cXte], 1).float(), yte)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=None,
                    help="comma list ds:base; default = all 12 winners")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--emb-dim", type=int, default=100)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--n-slices", type=int, default=64)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--fractions", default="0.003,0.01,0.02,0.05")
    ap.add_argument("--kernels", type=int, default=16)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--skip-power", action="store_true")
    ap.add_argument("--ckpt-sfx", default="",
                    help="seed suffix of the exp-70/71 ckpts, e.g. _s1")
    args = ap.parse_args()
    cells = ([tuple(c.split(":")) for c in args.cells.split(",")]
             if args.cells else list(WINNERS))
    ft_ep = 1 if args.quick else 5
    fractions = [float(x) for x in args.fractions.split(",")]
    n_null = 20 if args.quick else 100
    n_sig = 10 if args.quick else 50
    sparker_kw = dict(M=args.kernels, steps=args.steps)   # annealed

    all_out = {}
    for ds, base in cells:
        parent, obj, kind = WINNERS[(ds, base)]
        key = f"{ds}_{base}"
        space = f"{parent}->{obj} {kind}"
        N_CLS = 47 if ds == "dtd" else exp44.N_CLASSES[ds]
        n_hold = n_holdout(ds)
        holdouts = set(range(N_CLS - n_hold, N_CLS))
        seen = [c for c in range(N_CLS) if c not in holdouts]
        n_d = 2000 if ds in ("cars", "galaxy10") else 1000
        cfg = dict(n_classes=N_CLS, pair_dist=5.0, sigreg_weight=1.0,
                   n_slices=args.n_slices,
                   rep_weight=REP_WEIGHT * 45.0 / (N_CLS * (N_CLS - 1) / 2))
        print(f"\n######## [{key}] discovery on {space} ########")
        bb0, Xtr, ytr, Xte, yte = load_cell(ds, base, parent, obj, args)
        tr_lab, te_lab = ytr.numpy(), yte.numpy()
        base_feats = TensorDataset(Xtr, ytr)
        tr_loader = DataLoader(base_feats, batch_size=512, shuffle=False)
        te_loader = DataLoader(TensorDataset(Xte, yte), batch_size=512,
                               shuffle=False)
        seen_idx = np.where(np.isin(tr_lab, seen))[0]
        sig_idx = np.where(np.isin(tr_lab, list(holdouts)))[0]

        def space_scores(tr, te):
            m = np.isin(tr_lab, seen)
            cents = exp28.class_centroids(tr[m], tr_lab[m], seen)
            anch = cents.detach().float().to(DEVICE)
            r = exp29.evaluate_space(tr, tr_lab, te, te_lab, anch, seen,
                                     holdouts)
            aucs = []
            for s in range(3):
                torch.manual_seed(1000 + s)
                a, _, _ = exp29.linear_probe_novelty(tr, tr_lab, te, te_lab,
                                                     holdouts)
                aucs.append(a)
            return (float(np.mean(aucs)), float(np.std(aucs)), r["eucl"],
                    r["maha_tied"], r["lid"])

        tr0, _ = collect_embeddings(bb0, tr_loader)
        te0, _ = collect_embeddings(bb0, te_loader)
        pr0, sd0, eu0, ma0, li0 = space_scores(tr0, te0)
        m = np.isin(tr_lab, seen)
        means0 = exp28.fill_means(
            exp28.class_centroids(tr0[m], tr_lab[m], seen), seen,
            cfg).detach()

        bb = copy.deepcopy(bb0)
        _, hist = run_discovery(
            bb, means0.clone(), base_ds=base_feats,
            train_eval_loader=tr_loader, test_loader=te_loader, seen=seen,
            holdouts=holdouts, dataset_name=ds,
            rep_weight=cfg["rep_weight"], sigreg_weight=cfg["sigreg_weight"],
            n_slices=cfg["n_slices"], rounds=args.rounds, ft_epochs=ft_ep,
            names=None, seed=args.seed)
        trp, _ = collect_embeddings(bb, tr_loader)
        tep, _ = collect_embeddings(bb, te_loader)
        pr1, sd1, eu1, ma1, li1 = space_scores(trp, tep)
        print(f"  [{key}] {space}: probe {pr0:.4f}+-{sd0:.4f} -> "
              f"{pr1:.4f}+-{sd1:.4f}  eucl {eu0:.4f} -> {eu1:.4f}  "
              f"mahaT {ma0:.4f} -> {ma1:.4f}  lid {li0:.4f} -> {li1:.4f}")
        for h in hist:
            print(f"          round {h['round']}: purity={h['purity']:.3f} "
                  f"anchors={h['n_anchors']}  margin={h['margin']:.4f}")
        out = dict(space=space, probe_pre=pr0, probe_post=pr1,
                   eucl_pre=eu0, eucl_post=eu1, maha_pre=ma0, maha_post=ma1,
                   lid_pre=li0, lid_post=li1,
                   purity=[h["purity"] for h in hist])
        del bb
        torch.cuda.empty_cache()

        if not args.skip_power:
            post_power = {s: [] for s in STATS}
            for i_f, f in enumerate(fractions):
                n_inj = int(round(f * len(seen_idx) / (1.0 - f)))
                rng = np.random.default_rng(args.seed * 1000 + i_f)
                inj = rng.choice(sig_idx, size=min(n_inj, len(sig_idx)),
                                 replace=False)
                sub_idx = np.concatenate([seen_idx, inj])
                sub = TensorDataset(Xtr[sub_idx], ytr[sub_idx])
                sub_loader = DataLoader(sub, batch_size=512, shuffle=False)
                bb = copy.deepcopy(bb0)
                cur_means, _ = run_discovery(
                    bb, means0.clone(), base_ds=sub,
                    train_eval_loader=sub_loader, test_loader=te_loader,
                    seen=seen, holdouts=holdouts, dataset_name=ds,
                    rep_weight=cfg["rep_weight"],
                    sigreg_weight=cfg["sigreg_weight"],
                    n_slices=cfg["n_slices"], rounds=args.rounds,
                    ft_epochs=ft_ep, names=None, seed=args.seed)
                tep2, tel2 = collect_embeddings(bb, te_loader)
                trp2, trl2 = collect_embeddings(bb, tr_loader)
                zt = torch.as_tensor(tep2, dtype=torch.float32,
                                     device=DEVICE)
                d_seen = torch.cdist(zt, cur_means[seen]).min(1).values
                d_disc = (torch.cdist(zt, cur_means[N_CLS:]).min(1).values
                          if cur_means.size(0) > N_CLS else
                          torch.full_like(d_seen, float("inf")))
                bg_m = np.isin(tel2, seen)
                sg_m = np.isin(tel2, list(holdouts))
                s_ = (d_seen - d_disc).cpu().numpy()
                post_power["perevent"].append(
                    exp30.power_at_alpha(s_[bg_m], s_[sg_m], args.alpha))
                R = torch.as_tensor(trp2[np.isin(trl2, seen)][:20000],
                                    dtype=torch.float32, device=DEVICE)
                bg = torch.as_tensor(tep2[bg_m], dtype=torch.float32,
                                     device=DEVICE)
                sg = torch.as_tensor(tep2[sg_m], dtype=torch.float32,
                                     device=DEVICE)
                p, _ = exp31.run_test_battery(bg, sg, R, [f], n_d, n_null,
                                              n_sig, args.alpha,
                                              args.seed + i_f, sparker_kw,
                                              tag="post-spk")
                post_power["sparker"].append(p[0])
                maha_fn, mmd_fn, n_bg, n_sg = exp32.make_stats_fns(
                    trp2, trl2, tep2, tel2, seen, holdouts, args.seed + i_f)
                p, _ = exp32.battery(maha_fn, n_bg, n_sg, [f], n_d, n_null,
                                     n_sig, args.alpha, args.seed + i_f,
                                     tag="post-maha")
                post_power["maha"].append(p[0])
                p, _ = exp32.battery(mmd_fn, n_bg, n_sg, [f], n_d, n_null,
                                     n_sig, args.alpha, args.seed + i_f,
                                     tag="post-mmd")
                post_power["mmd"].append(p[0])
                print(f"  [{key}] post f={f}: " + "  ".join(
                    f"{s}={post_power[s][-1]:.3f}" for s in STATS))
                del bb
                torch.cuda.empty_cache()
            out["post_power"] = post_power
        all_out[key] = out
        del bb0
        torch.cuda.empty_cache()

    print("\n===== EXP72 SUMMARY (discovery on exp-71 winners) =====")
    print(f"  {'cell':<18}{'space':<26}{'probe pre->post':>20}"
          f"{'mahaT pre->post':>18}{'purity r1':>10}")
    for key, o in all_out.items():
        print(f"  {key:<18}{o['space']:<26}"
              f"{o['probe_pre']:>9.4f}->{o['probe_post']:.4f}"
              f"{o['maha_pre']:>9.4f}->{o['maha_post']:.4f}"
              f"{o['purity'][0]:>10.3f}")

    keys = list(all_out)
    xs = np.arange(len(keys))
    plt.figure(figsize=(1.1 * len(keys) + 3, 5.5))
    w = 0.38
    plt.bar(xs - w / 2, [all_out[k]["probe_pre"] for k in keys], w,
            label="pre", color="#eda100")
    plt.bar(xs + w / 2, [all_out[k]["probe_post"] for k in keys], w,
            label="post-discovery", color="#008300")
    plt.xticks(xs, keys, rotation=25, ha="right", fontsize=7)
    plt.ylabel("holdout probe ROC AUC")
    plt.title("exp72: discovery on the exp-71 residual winners")
    plt.legend()
    plt.grid(alpha=0.25, axis="y")
    plt.tight_layout()
    out = plot_path(f"exp72_residual_discovery{run_tag()}{args.ckpt_sfx}.png")
    plt.savefig(out, dpi=150); plt.close()
    print("saved", out)

    os.makedirs(os.path.join("logs", "exp72"), exist_ok=True)
    ftag = (("_" + "_".join(k for k in keys)) if args.cells else "") \
        + args.ckpt_sfx
    np.savez(os.path.join("logs", "exp72", f"residual_discovery{run_tag()}{ftag}.npz"),
             cells=np.array(keys),
             **{f"{k}_{f}": np.array(all_out[k][f]) for k in keys
                for f in ("probe_pre", "probe_post", "eucl_pre", "eucl_post",
                          "maha_pre", "maha_post", "lid_pre", "lid_post",
                          "purity")},
             **{f"{k}_post_{s}": np.array(all_out[k]["post_power"][s])
                for k in keys for s in STATS
                if "post_power" in all_out[k]})
    print("Done.")


if __name__ == "__main__":
    main()
