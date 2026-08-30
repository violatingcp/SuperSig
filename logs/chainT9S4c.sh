#!/bin/bash
# Block S4c: eval-only exp-70 re-pass on the archived dtd/visreg draw checkpoints (postf_* keys).
cd /home/pharris/sigreg/SuperSig; P=/home/pharris/venv/bin/python; L=logs/chainT9S.log
ok() { [ $1 -eq 0 ] && echo "$2_OK" >> $L || echo "$2_FAIL" >> $L; }
for D in 0 1 3 4 5; do
  SUPERSIG_NH=1 SUPERSIG_HOLDOUT_DRAW=$D $P experiments/70_cars_ft_suite.py --dataset dtd --base visreg > logs/exp70_dtd_visreg_h1_d${D}_repass.log 2>&1; ok $? S4c_dtd-visreg-D$D
done
echo BLOCK_S4C_DONE >> $L
