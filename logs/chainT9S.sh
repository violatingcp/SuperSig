#!/bin/bash
# Block S: exp 146 -- GCD's loss as exp-70 arms (gcd-ft, gcd-sigreg-ft) on our battery.
cd /home/pharris/sigreg/SuperSig; P=/home/pharris/venv/bin/python; L=logs/chainT9S.log
ok() { [ $1 -eq 0 ] && echo "$2_OK" >> $L || echo "$2_FAIL" >> $L; }
ARMS="gcd-ft gcd-sigreg-ft"
for B in dino lejepa visreg; do for D in 0 3 5 7 8; do
  SUPERSIG_NH=1 SUPERSIG_HOLDOUT_DRAW=$D $P experiments/70_cars_ft_suite.py --dataset galaxy10 --base $B --arms $ARMS > logs/exp146_galaxy10_${B}_h1_d$D.log 2>&1; ok $? S146_galaxy10-$B-D$D
done; done
echo S_GALAXY10_DONE >> $L
for D in 0 1 3 4 5; do
  SUPERSIG_NH=1 SUPERSIG_HOLDOUT_DRAW=$D $P experiments/70_cars_ft_suite.py --dataset dtd --base dino --arms $ARMS > logs/exp146_dtd_dino_h1_d$D.log 2>&1; ok $? S146_dtd-dino-D$D
done
unset SUPERSIG_NH SUPERSIG_HOLDOUT_DRAW
for DS in cars aircraft flowers; do
  $P experiments/70_cars_ft_suite.py --dataset $DS --base dino --arms $ARMS > logs/exp146_${DS}_dino.log 2>&1; ok $? S146_${DS}-dino
done
echo BLOCK_S_DONE >> $L
