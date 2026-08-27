#!/bin/bash
# Tier 9 Block B remainder: galaxy10 lejepa d5/d7/d8, galaxy10 visreg x5, dtd x3 bases x5.
# galaxy10 draws {0,3,5,7,8} -> distinct holdout classes {2,6,5,4,3}
# dtd draws {0,1,3,4,5} -> distinct holdout classes {8,46,38,15,16} (draw 2 == draw 0)
cd /home/pharris/sigreg/SuperSig
PY=/home/pharris/venv/bin/python
run() { # ds base draw
  local tag="exp70_${1}_${2}_h1_d${3}"
  SUPERSIG_NH=1 SUPERSIG_HOLDOUT_DRAW=$3 $PY experiments/70_cars_ft_suite.py --dataset $1 --base $2 > logs/$tag.log 2>&1 \
    && echo "T9_${1}-${2}-D${3}_OK" >> logs/chainT9B4.log || echo "T9_${1}-${2}-D${3}_FAIL" >> logs/chainT9B4.log
}
for D in 5 7 8; do run galaxy10 lejepa $D; done
for D in 0 3 5 7 8; do run galaxy10 visreg $D; done
echo BLOCK_B_G10_DONE >> logs/chainT9B4.log
for B in dino lejepa visreg; do for D in 0 1 3 4 5; do run dtd $B $D; done; done
echo BLOCK_B_DTD_DONE >> logs/chainT9B4.log
