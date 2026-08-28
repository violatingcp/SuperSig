#!/bin/bash
# Block O: exp 139 seed grid (seeds 1,2 only -- seed 0 exists and keeps the archived filename) + exp 138 on the clean cifar100 banks.
cd /home/pharris/sigreg/SuperSig
P=/home/pharris/venv/bin/python; L=logs/chainT9O.log
ok() { [ $1 -eq 0 ] && echo "$2_OK" >> $L || echo "$2_FAIL" >> $L; }
unset SUPERSIG_NH SUPERSIG_HOLDOUT_DRAW
$P experiments/138_base_rate_estimator.py --glob 'logs/exp136/banks/embs_*_cifar100.npz' --holdouts 4 --dataset cifar100 > logs/exp138_cifar100_clean.log 2>&1; ok $? O138_cifar100-clean
for B in dino lejepa visreg; do for D in 0 3 5 7 8; do for S in 1 2; do
  SUPERSIG_NH=1 SUPERSIG_HOLDOUT_DRAW=$D $P experiments/135_corpus_norm_everywhere.py --cells galaxy10:$B --scorers np --seed $S --out logs/exp135 > logs/exp139_g10_${B}_d${D}_s$S.log 2>&1; ok $? O139_${B}-D${D}-S$S
done; done; done
unset SUPERSIG_NH SUPERSIG_HOLDOUT_DRAW
$P experiments/139_frozen_np_hardening.py --aggregate > logs/exp139_aggregate.log 2>&1; ok $? O139_AGG
echo BLOCK_O_DONE >> $L
