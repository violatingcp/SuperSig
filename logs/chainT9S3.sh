#!/bin/bash
# Block S3: exp 146b -- dose-response of the GCD arms in the fraction of the novel class
# present (unlabelled) in the training corpus.  Waits for chainT9S2 to finish.
cd /home/pharris/sigreg/SuperSig; P=/home/pharris/venv/bin/python; L=logs/chainT9S.log
while pgrep -f "chainT9S[2].sh" >/dev/null; do sleep 300; done
ok() { [ $1 -eq 0 ] && echo "$2_OK" >> $L || echo "$2_FAIL" >> $L; }
ARMS="gcd-ft gcd-sigreg-ft"
for F in 0.05 0.2 0.5; do
  for D in 0 3 5 7 8; do
    SUPERSIG_NH=1 SUPERSIG_HOLDOUT_DRAW=$D $P experiments/70_cars_ft_suite.py --dataset galaxy10 --base lejepa --arms $ARMS --gcd-unl-frac $F > logs/exp146b_galaxy10_lejepa_h1_d${D}_u$F.log 2>&1; ok $? S146b_galaxy10-lejepa-D$D-u$F
  done
  for D in 0 1 3 4 5; do
    SUPERSIG_NH=1 SUPERSIG_HOLDOUT_DRAW=$D $P experiments/70_cars_ft_suite.py --dataset dtd --base dino --arms $ARMS --gcd-unl-frac $F > logs/exp146b_dtd_dino_h1_d${D}_u$F.log 2>&1; ok $? S146b_dtd-dino-D$D-u$F
  done
done
echo BLOCK_S3_DONE >> $L
