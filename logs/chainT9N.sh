#!/bin/bash
# Block N: seeds 1,2 on the exp-59 CIFAR-100 champions (100-D cw-lam5 config first, then 16+16). Waits for Block M's cifar10 seeds.
cd /home/pharris/sigreg/SuperSig
P=/home/pharris/venv/bin/python; L=logs/chainT9N.log
unset SUPERSIG_NH SUPERSIG_HOLDOUT_DRAW
ok() { [ $1 -eq 0 ] && echo "$2_OK" >> $L || echo "$2_FAIL" >> $L; }
until grep -q "M59_cifar10-s2" logs/chainT9M.log 2>/dev/null; do sleep 600; done
for S in 1 2; do
  $P experiments/59_nplm_residual_concat.py --dataset cifar100 --dim-half 50 --cw-lam 5 --seed $S > logs/exp59_cifar100_100d_cwlam5_s$S.log 2>&1; ok $? N59_cifar100-100d-s$S
done
for S in 1 2; do
  $P experiments/59_nplm_residual_concat.py --dataset cifar100 --seed $S > logs/exp59_cifar100_s$S.log 2>&1; ok $? N59_cifar100-16p16-s$S
done
echo BLOCK_N_DONE >> $L
