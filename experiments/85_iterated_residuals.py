"""
Experiment 85 (IMPROVEMENT_TESTS.md #85): iterated residuals — does the
residual trick stack with itself?

Exp 77: the res-nplm child leaves 5-10% genuinely new variance on
fine-grained data — the condition under which a SECOND residual should
still have something to extract.  On the cars/visreg and flowers/dino
champions (the two biggest residual gains), per seed s in {0,1,2}:

  r2 child   deepcopy(child1); residual-nplm ft on r2 = z - cent_y(r1)
             (child1's seen centroids), seed s+14; eval the 3-way concat
             [parent; r1; r2] (300-D).
  width ctrl parent + ONE residual of double width (200-D head, trunk
             warm-started from the parent, centroids zero-padded), same
             epochs — so any 3-way gain is attributable to the
             DECOMPOSITION, not parameter count (both concats are 300-D).

Prediction: diminishing but positive (+0.01-0.03 on fine-grained).
Falsifier: the double-width single residual matches the 3-way concat —
residual ft is a capacity effect, not a decomposition effect (a
significant reinterpretation of the paper's SS5).

    python experiments/85_iterated_residuals.py
    python experiments/85_iterated_residuals.py --cells flowers:dino --seeds 1 --quick
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import copy
import importlib
import numpy as np
import torch

from supersig.config import DEVICE, REPO_DIR

exp28 = importlib.import_module("28_concat_residual")
exp29 = importlib.import_module("29_residual_finetune")
exp30 = importlib.import_module("30_power_curves")
exp37 = importlib.import_module("37_dtd_vit")
exp43 = importlib.import_module("43_dtd_finetune")
exp44 = importlib.import_module("44_transfer_32d")
exp49 = importlib.import_module("49_aircraft_ssl_ft")
exp70 = importlib.import_module("70_cars_ft_suite")
exp71 = importlib.import_module("71_residual_ft_grid")

CKPT_DIR = os.path.join(REPO_DIR, "checkpoints")
CELLS = ["cars:visreg", "flowers:dino"]


def head_embs(model, banks):
    h = model.head.float()
    return (exp37.embed(h, banks["train"][0].float()).numpy(),
            exp37.embed(h, banks["test"][0].float()).numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=",".join(CELLS))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--emb-dim", type=int, default=100)
    ap.add_argument("--wide-dim", type=int, default=200)
    ap.add_argument("--ft-epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr-backbone", type=float, default=1e-5)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--n-slices", type=int, default=64)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    args.ft_epochs = args.ft_epochs or (1 if args.quick else 20)

    results = {}
    for cell in args.cells.split(","):
        DS, BASE = cell.split(":")
        exp70.DS, exp70.BASE = DS, BASE
        exp71.DS, exp71.BASE = DS, BASE
        N_CLS = exp44.N_CLASSES[DS]
        holdouts = set(range(N_CLS - 10, N_CLS))
        seen = [c for c in range(N_CLS) if c not in holdouts]
        corpus = exp43.train_corpus(DS)
        ytr_all = exp70.corpus_labels(corpus)
        seen_idx_corpus = np.where(~np.isin(ytr_all,
                                            list(holdouts)))[0].tolist()

        def battery(tr, te, tr_lab, te_lab):
            m = np.isin(tr_lab, seen)
            anch = exp28.class_centroids(tr[m], tr_lab[m],
                                         seen).detach().float().to(DEVICE)
            r = exp29.evaluate_space(tr, tr_lab, te, te_lab, anch, seen,
                                     holdouts)
            aucs = []
            for s in range(3):
                torch.manual_seed(1000 + s)
                a, _, _ = exp29.linear_probe_novelty(tr, tr_lab, te,
                                                     te_lab, holdouts)
                aucs.append(a)
            d = torch.cdist(torch.as_tensor(te, dtype=torch.float32,
                                            device=DEVICE), anch)
            s_ = d.min(1).values.cpu().numpy()
            bg = np.isin(te_lab, seen)
            sg = np.isin(te_lab, list(holdouts))
            pe = exp30.power_at_alpha(s_[bg], s_[sg], args.alpha)
            return dict(probe=float(np.mean(aucs)), eucl=r["eucl"],
                        mahaT=r["maha_tied"], lid=r["lid"], perevt=pe)

        for s in range(args.seeds):
            sargs = argparse.Namespace(**vars(args), seed=s)
            sfx = exp70.seed_sfx(sargs)
            key = f"{cell}_s{s}"
            print(f"\n######## [{key}] iterated residual ########",
                  flush=True)
            par = exp43.FineTuneModel(BASE, args.emb_dim)
            par.load_state_dict(torch.load(
                os.path.join(CKPT_DIR,
                             f"{DS}_ft_{BASE}_supcon-ft_seen{sfx}.pt"),
                map_location=DEVICE))
            ch1 = exp43.FineTuneModel(BASE, args.emb_dim)
            ch1.load_state_dict(torch.load(
                os.path.join(CKPT_DIR, f"{DS}_ft_{BASE}_supcon-ft_"
                                       f"resnplm_seen{sfx}.pt"),
                map_location=DEVICE))
            pb = exp70.trunk_banks(par, "supcon-ft", sargs)
            rb1 = exp70.trunk_banks(ch1, "supcon-ft_resnplm", sargs)
            tr_lab = pb["train"][1].numpy()
            te_lab = pb["test"][1].numpy()
            pHtr, pHte = head_embs(par, pb)
            r1tr, r1te = head_embs(ch1, rb1)
            m = np.isin(tr_lab, seen)

            # --- second residual: child2 from child1's centroids
            c2ck = os.path.join(CKPT_DIR, f"{DS}_ft_{BASE}_supcon-ft_"
                                          f"resnplm2_seen{sfx}.pt")
            cents1 = torch.zeros(N_CLS, args.emb_dim, device=DEVICE)
            cents1[torch.as_tensor(seen, device=DEVICE)] = \
                exp28.class_centroids(r1tr[m], tr_lab[m],
                                      seen).detach().float().to(DEVICE)
            torch.manual_seed(s + 14); np.random.seed(s + 14)
            ch2 = copy.deepcopy(ch1)
            if os.path.exists(c2ck) and not args.refresh:
                print(f"  loading {c2ck}")
                ch2.load_state_dict(torch.load(c2ck, map_location=DEVICE))
            else:
                step = exp71.make_res_nplm_step(cents1, 1.0, args.n_slices)
                loader = exp70.seen_two_view_loader(corpus,
                                                    seen_idx_corpus, sargs)
                exp49.ft_loop(ch2, loader, args.ft_epochs, step, sargs,
                              f"resnplm2{sfx}")
                torch.save(ch2.state_dict(), c2ck)
                del loader
            rb2 = exp70.trunk_banks(ch2, "supcon-ft_resnplm2", sargs)
            r2tr, r2te = head_embs(ch2, rb2)
            del ch2

            # --- width control: one 200-D residual from the parent
            wck = os.path.join(CKPT_DIR, f"{DS}_ft_{BASE}_supcon-ft_"
                                         f"resnplmW{args.wide_dim}"
                                         f"_seen{sfx}.pt")
            torch.manual_seed(s + 14); np.random.seed(s + 14)
            chw = exp43.FineTuneModel(BASE, args.wide_dim)
            chw.trunk.load_state_dict(par.trunk.state_dict())
            if os.path.exists(wck) and not args.refresh:
                print(f"  loading {wck}")
                chw.load_state_dict(torch.load(wck, map_location=DEVICE))
            else:
                centsP = torch.zeros(N_CLS, args.emb_dim, device=DEVICE)
                centsP[torch.as_tensor(seen, device=DEVICE)] = \
                    exp28.class_centroids(pHtr[m], tr_lab[m],
                                          seen).detach().float().to(DEVICE)
                centsW = torch.zeros(N_CLS, args.wide_dim, device=DEVICE)
                centsW[:, :args.emb_dim] = centsP
                step = exp71.make_res_nplm_step(centsW, 1.0, args.n_slices)
                loader = exp70.seen_two_view_loader(corpus,
                                                    seen_idx_corpus, sargs)
                exp49.ft_loop(chw, loader, args.ft_epochs, step, sargs,
                              f"resnplmW{sfx}")
                torch.save(chw.state_dict(), wck)
                del loader
            rbw = exp70.trunk_banks(chw, f"supcon-ft_resnplmW"
                                         f"{args.wide_dim}", sargs)
            rwtr, rwte = head_embs(chw, rbw)
            del chw, par, ch1
            torch.cuda.empty_cache()

            spaces = {
                "2way [p;r1]": (np.concatenate([pHtr, r1tr], 1),
                                np.concatenate([pHte, r1te], 1)),
                "3way [p;r1;r2]": (np.concatenate([pHtr, r1tr, r2tr], 1),
                                   np.concatenate([pHte, r1te, r2te], 1)),
                "wide [p;rW200]": (np.concatenate([pHtr, rwtr], 1),
                                   np.concatenate([pHte, rwte], 1)),
            }
            for name, (tr, te) in spaces.items():
                b = battery(tr, te, tr_lab, te_lab)
                results[f"{key}:{name}"] = b
                print(f"  [{key} {name:<15}] probe={b['probe']:.4f} "
                      f"eucl={b['eucl']:.4f} mahaT={b['mahaT']:.4f} "
                      f"lid={b['lid']:.4f} perevt={b['perevt']:.3f}",
                      flush=True)

    print("\n===== EXP85 SUMMARY (per-seed; 2way vs 3way vs width ctrl) ====")
    print(f"  {'cell:space':<36}{'probe':>8}{'eucl':>7}{'mahaT':>7}"
          f"{'perevt':>8}")
    for k, b in results.items():
        print(f"  {k:<36}{b['probe']:>8.4f}{b['eucl']:>7.3f}"
              f"{b['mahaT']:>7.3f}{b['perevt']:>8.3f}")
    for cell in args.cells.split(","):
        for sp in ("2way [p;r1]", "3way [p;r1;r2]", "wide [p;rW200]"):
            pr = [results[f"{cell}_s{s}:{sp}"]["probe"]
                  for s in range(args.seeds)
                  if f"{cell}_s{s}:{sp}" in results]
            if pr:
                print(f"  {cell} {sp}: probe {np.mean(pr):.4f}"
                      f"+-{np.std(pr):.4f}")

    os.makedirs(os.path.join("logs", "exp85"), exist_ok=True)
    np.savez(os.path.join("logs", "exp85", "results.npz"),
             summary=np.array([repr(results)], dtype=object))
    print("EXP85 DONE.")


if __name__ == "__main__":
    main()
