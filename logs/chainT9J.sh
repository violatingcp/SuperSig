#!/bin/bash
# Block J: exp 134a (width control) then exp 134c (residual after discovery). Waits for Block I.
cd /home/pharris/sigreg/SuperSig
P=/home/pharris/venv/bin/python; L=logs/chainT9J.log
unset SUPERSIG_NH SUPERSIG_HOLDOUT_DRAW
until grep -q BLOCK_I_DONE logs/chainT9I.log 2>/dev/null; do sleep 300; done
for S in 0 1 2; do
  $P experiments/70_cars_ft_suite.py --dataset cars --base visreg --emb-dim 200 --arms supcon-ft --skip-discovery --seed $S > logs/exp134a_cars_visreg_e200_s$S.log 2>&1 && echo "J134A_CARS-S${S}_OK" >> $L || echo "J134A_CARS-S${S}_FAIL" >> $L
done
for B in dino lejepa; do
  $P experiments/70_cars_ft_suite.py --dataset galaxy10 --base $B --emb-dim 200 --arms supcon-ft --skip-discovery > logs/exp134a_galaxy10_${B}_e200.log 2>&1 && echo "J134A_G10-${B}_OK" >> $L || echo "J134A_G10-${B}_FAIL" >> $L
done
$P experiments/134a_width_control.py --dataset cars --base visreg > logs/exp134a_compare_cars_visreg.log 2>&1
for B in dino lejepa; do $P experiments/134a_width_control.py --dataset galaxy10 --base $B --concat supcon-ft-res > logs/exp134a_compare_galaxy10_$B.log 2>&1; done
echo BLOCK_J_134A_DONE >> $L
for C in galaxy10:dino galaxy10:lejepa galaxy10:visreg cars:visreg dtd:dino; do
  D=${C%%:*}; B=${C##*:}
  $P experiments/134c_residual_after_discovery.py --dataset $D --base $B > logs/exp134c_${D}_$B.log 2>&1 && echo "J134C_${D}-${B}_OK" >> $L || echo "J134C_${D}-${B}_FAIL" >> $L
done
echo BLOCK_J_DONE >> $L
