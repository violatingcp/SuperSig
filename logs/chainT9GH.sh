#!/bin/bash
# Block G (exp 132 supervised probe) then Block H (exp 133 BN-adaptation A/B).
cd /home/pharris/sigreg/SuperSig
P=/home/pharris/venv/bin/python; L=logs/chainT9GH.log
ok() { [ $1 -eq 0 ] && echo "$2_OK" >> $L || echo "$2_FAIL" >> $L; }
unset SUPERSIG_NH SUPERSIG_HOLDOUT_DRAW
$P experiments/132_supervised_probe.py --glob 'logs/exp54/embs_*_cifar10.npz' --dataset cifar10 --holdouts 4 --seeds 3 > logs/exp132_c10bank.log 2>&1; ok $? G132_C10BANK
$P experiments/132_supervised_probe.py --glob 'logs/exp113/embs/cifar100_tau*_s*.npz' --dataset cifar100 --holdouts 99 --seeds 3 > logs/exp132_c100bank.log 2>&1; ok $? G132_C100BANK
for B in dino lejepa visreg; do
  $P experiments/132_supervised_probe.py --cells galaxy10:$B --baseline "galaxy10:$B|supcon-ft" --seeds 3 > logs/exp132_galaxy10_$B.log 2>&1; ok $? G132_G10-$B
done
for B in dino lejepa; do for D in 0 3 5 7 8; do
  SUPERSIG_NH=1 SUPERSIG_HOLDOUT_DRAW=$D $P experiments/132_supervised_probe.py --cells galaxy10:$B --baseline "galaxy10:$B|supcon-ft" --seeds 3 > logs/exp132_galaxy10_${B}_h1_d$D.log 2>&1; ok $? G132_G10-$B-D$D
done; done
echo BLOCK_G_DONE >> $L
$P experiments/133_bn_adaptation_ab.py > logs/exp133_bn_ab.log 2>&1; ok $? H133
echo BLOCK_H_DONE >> $L
