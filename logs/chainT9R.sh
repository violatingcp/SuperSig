#!/bin/bash
# Block R: exp 145 fine-tuned GCD tier -- aircraft then cars, 3 supervised arms, seed 0; seeds 1,2 afterwards.
cd /home/pharris/sigreg/SuperSig; P=/home/pharris/venv/bin/python; L=logs/chainT9R.log
ok() { [ $1 -eq 0 ] && echo "$2_OK" >> $L || echo "$2_FAIL" >> $L; }
until grep -q BLOCK_Q_DONE logs/chainT9Q.log 2>/dev/null; do sleep 120; done
for D in aircraft cars; do
  $P experiments/145_gcd_finetune.py --dataset $D --seeds 0 > logs/exp145_${D}_s0.log 2>&1; ok $? R145_${D}-s0
done
for D in aircraft cars; do
  $P experiments/145_gcd_finetune.py --dataset $D --seeds 0,1,2 > logs/exp145_${D}_s012.log 2>&1; ok $? R145_${D}-s012
done
echo BLOCK_R_DONE >> $L
