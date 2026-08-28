#!/bin/bash
# Block M: (1) exp 137 scratch residuals per cell as its parents land; (2) exp 59 seeds 1,2 on cifar10.
cd /home/pharris/sigreg/SuperSig
P=/home/pharris/venv/bin/python; L=logs/chainT9M.log
unset SUPERSIG_NH SUPERSIG_HOLDOUT_DRAW
ok() { [ $1 -eq 0 ] && echo "$2_OK" >> $L || echo "$2_FAIL" >> $L; }
# cifar100 h4: parents exist now
$P experiments/137_scratch_residuals.py --dataset cifar100 > logs/exp137_cifar100_h4.log 2>&1; ok $? M137_cifar100-h4
# exp 59 champions, seeds 1 and 2 (cifar10, 16+16)
for S in 1 2; do
  $P experiments/59_nplm_residual_concat.py --dataset cifar10 --seed $S > logs/exp59_cifar10_s$S.log 2>&1; ok $? M59_cifar10-s$S
done
# cifar10 h4 after Block L tier 2
until grep -q TIER2_DONE logs/chainT9L.log 2>/dev/null; do sleep 600; done
$P experiments/137_scratch_residuals.py --dataset cifar10 > logs/exp137_cifar10_h4.log 2>&1; ok $? M137_cifar10-h4
# draws after Block L finishes
until grep -q BLOCK_L_DONE logs/chainT9L.log 2>/dev/null; do sleep 600; done
for spec in "cifar10 8" "cifar10 9" "cifar10 7" "cifar100 43" "cifar100 57" "cifar100 48"; do set -- $spec
  $P experiments/137_scratch_residuals.py --dataset $1 --holdout $2 > logs/exp137_$1_h$2.log 2>&1; ok $? M137_$1-h$2
done
echo BLOCK_M_DONE >> $L
