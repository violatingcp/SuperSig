"""
Exp 119 support: extract frozen ViT-B/16 CLS feature banks for Food-101
(the held-out dataset) on the three campaign bases, in the exp-44 plain-bank
format `data/tf_feats_food101_{base}_vitb16.pt`.

    python experiments/119b_food101_banks.py
    python experiments/119b_food101_banks.py --bases dino
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import importlib
import torch

from supersig.config import DATA_DIR, DEVICE

exp37 = importlib.import_module("37_dtd_vit")
exp40 = importlib.import_module("40_dtd_bases")
exp44 = importlib.import_module("44_transfer_32d")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bases", nargs="+",
                    default=["dino", "lejepa", "visreg"])
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    for base in args.bases:
        cache = os.path.join(DATA_DIR,
                             f"tf_feats_food101_{exp40.CACHE_TAG[base]}.pt")
        if os.path.exists(cache) and not args.refresh:
            print(f"  {cache} exists, skipping")
            continue
        model = exp40.LOADERS[base]()
        plain = {}
        for split in ("train", "test"):
            d = exp44.make_split("food101", split, exp37.TF_EVAL)
            plain[split] = exp37.extract(model, d)
            print(f"  extracted food101/{base} {split}: "
                  f"{tuple(plain[split][0].shape)}", flush=True)
        torch.save(plain, cache)
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    print("FOOD101 BANKS DONE.")


if __name__ == "__main__":
    main()
