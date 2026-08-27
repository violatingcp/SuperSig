#!/bin/bash
# Residual battery on the exp-125 galaxy10 draws: exp 71 (res / res-nplm ft) + exp 72 (discovery on residual winners).
# dino + lejepa parents exist; visreg draws wait for their exp-70 parent to land.
cd /home/pharris/sigreg/SuperSig
P=/home/pharris/venv/bin/python; L=logs/chainT9R.log
run() { # base draw
  export SUPERSIG_NH=1 SUPERSIG_HOLDOUT_DRAW=$2
  $P experiments/71_residual_ft_grid.py --dataset galaxy10 --base $1 > logs/exp71_galaxy10_$1_h1_d$2.log 2>&1 \
     && echo "R71_$1-D$2_OK" >> $L || echo "R71_$1-D$2_FAIL" >> $L
  $P experiments/72_residual_discovery.py --cells galaxy10:$1 > logs/exp72_galaxy10_$1_h1_d$2.log 2>&1 \
     && echo "R72_$1-D$2_OK" >> $L || echo "R72_$1-D$2_FAIL" >> $L
}
for B in dino lejepa; do for D in 0 3 5 7 8; do run $B $D; done; done
for D in 0 3 5 7 8; do
  until [ -f logs/exp70/results_galaxy10_visreg_ft70_h1_d$D.npz ]; do sleep 300; done
  run visreg $D
done
echo RESIDUAL_G10_DONE >> $L
