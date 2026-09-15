"""End-to-end SuperSig-Lite on CIFAR-10 from random initialisation.

    python lite/example_cifar10.py --quick     # ~3 min smoke run
    python lite/example_cifar10.py             # full (200-epoch parent)
"""
import argparse
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lite.pipeline import train_parent, train_child, discover, detect


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10",
                    choices=["cifar10", "cifar100"])
    ap.add_argument("--holdout", type=int, default=4)
    ap.add_argument("--parent", default="supcon", choices=["supcon", "ssig"])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    print("== 1/4 parent (SupCon on seen classes)")
    parent = train_parent(args.dataset, holdout={args.holdout},
                          epochs=args.epochs, parent=args.parent,
                          quick=args.quick)
    print("== 2/4 child (res: NT-Xent + SIGReg on residuals)")
    child = train_child(parent)
    print("== 3/4 discovery (np pool, derived cut, skip-on-refuse)")
    post = discover(child)
    if post["declined"]:
        print("   the label-free cut DECLINED: estimated novelty below "
              "clusterability -- the method abstains (this is an output).")
    print("== 4/4 detection")
    rep = detect(post)
    print(f"   discovered anchors : {rep['n_discovered_anchors']}")
    if rep.get("eucl_anchor_auc") is not None:
        print(f"   eucl-anchor AUC    : {rep['eucl_anchor_auc']:.3f}")
        print(f"   per-event power    : {rep['per_event_power']:.3f} "
              f"(alpha=0.05)")
        print(f"   SparKer@anchors t  : {rep['sparker_anchor_t']}")
    print(f"   monitor (mean Maha): {rep['monitor_maha_mean']:.3f}")


if __name__ == "__main__":
    main()
