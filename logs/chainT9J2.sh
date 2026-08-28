#!/bin/bash
cd /home/pharris/sigreg/SuperSig; P=/home/pharris/venv/bin/python; L=logs/chainT9J.log
unset SUPERSIG_NH SUPERSIG_HOLDOUT_DRAW
for C in galaxy10:visreg cars:visreg dtd:dino; do D=${C%%:*}; B=${C##*:}
  $P experiments/134c_residual_after_discovery.py --dataset $D --base $B > logs/exp134c_${D}_$B.log 2>&1 && echo "J134C_${D}-${B}_OK" >> $L || echo "J134C_${D}-${B}_FAIL" >> $L
done; echo BLOCK_J2_DONE >> $L
