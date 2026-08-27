#!/bin/bash
# Block I: exp 134 part (B) -- combiner scan on every TRAINED parent/child pair.
cd /home/pharris/sigreg/SuperSig
P=/home/pharris/venv/bin/python; L=logs/chainT9I.log
unset SUPERSIG_NH SUPERSIG_HOLDOUT_DRAW
CELLS=""; for D in cars flowers dtd aircraft galaxy10; do for B in dino lejepa visreg; do CELLS="$CELLS,$D:$B"; done; done
$P experiments/134_residual_audit.py --cells ${CELLS#,} > logs/exp134_archived_cells.log 2>&1 && echo I134_ARCHIVED_OK >> $L || echo I134_ARCHIVED_FAIL >> $L
# galaxy10 draw pairs: wait for the residual chain (exp 71) to finish
until grep -q RESIDUAL_G10_DONE logs/chainT9R.log 2>/dev/null; do sleep 300; done
for B in dino lejepa visreg; do for D in 0 3 5 7 8; do
  SUPERSIG_NH=1 SUPERSIG_HOLDOUT_DRAW=$D $P experiments/134_residual_audit.py --cells galaxy10:$B > logs/exp134_galaxy10_${B}_h1_d$D.log 2>&1 && echo "I134_G10-$B-D${D}_OK" >> $L || echo "I134_G10-$B-D${D}_FAIL" >> $L
done; done
echo BLOCK_I_DONE >> $L
