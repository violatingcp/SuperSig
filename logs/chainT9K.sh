#!/bin/bash
# Block K: exp 135 corpus-adaptive normalisation A/B on every transfer cell + galaxy10 draws.
cd /home/pharris/sigreg/SuperSig
P=/home/pharris/venv/bin/python; L=logs/chainT9K.log
unset SUPERSIG_NH SUPERSIG_HOLDOUT_DRAW
for D in galaxy10 dtd flowers cars aircraft; do for B in dino lejepa visreg; do
  $P experiments/135_corpus_norm_everywhere.py --cells $D:$B > logs/exp135_${D}_$B.log 2>&1 && echo "K135_${D}-${B}_OK" >> $L || echo "K135_${D}-${B}_FAIL" >> $L
done; done
echo BLOCK_K_ARCHIVED_DONE >> $L
for B in dino lejepa visreg; do for D in 0 3 5 7 8; do
  until [ -f logs/exp70/results_galaxy10_${B}_ft70_h1_d$D.npz ]; do sleep 300; done
  SUPERSIG_NH=1 SUPERSIG_HOLDOUT_DRAW=$D $P experiments/135_corpus_norm_everywhere.py --cells galaxy10:$B > logs/exp135_galaxy10_${B}_h1_d$D.log 2>&1 && echo "K135_G10-${B}-D${D}_OK" >> $L || echo "K135_G10-${B}-D${D}_FAIL" >> $L
done; done
echo BLOCK_K_DONE >> $L
