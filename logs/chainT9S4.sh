#!/bin/bash
# Block S4: eval-only exp-70 re-pass on the archived six-arm draw checkpoints so the
# injected-fraction post metrics (postf_*) exist for the archived arms.
cd /home/pharris/sigreg/SuperSig; P=/home/pharris/venv/bin/python; L=logs/chainT9S.log
ok() { [ $1 -eq 0 ] && echo "$2_OK" >> $L || echo "$2_FAIL" >> $L; }
for B in dino lejepa visreg; do for D in 0 3 5 7 8; do
  SUPERSIG_NH=1 SUPERSIG_HOLDOUT_DRAW=$D $P experiments/70_cars_ft_suite.py --dataset galaxy10 --base $B > logs/exp70_galaxy10_${B}_h1_d${D}_repass.log 2>&1; ok $? S4_galaxy10-$B-D$D
done; done
for D in 0 1 3 4 5; do
  SUPERSIG_NH=1 SUPERSIG_HOLDOUT_DRAW=$D $P experiments/70_cars_ft_suite.py --dataset dtd --base dino > logs/exp70_dtd_dino_h1_d${D}_repass.log 2>&1; ok $? S4_dtd-dino-D$D
done
echo BLOCK_S4_DONE >> $L
