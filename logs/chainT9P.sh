#!/bin/bash
# Block P: frozen-np (pretrained trunk, identity head) on flowers/cars/aircraft (+galaxy10/dtd for comparison),
# multi-holdout default and single-holdout draws 0-4.
cd /home/pharris/sigreg/SuperSig
P=/home/pharris/venv/bin/python; L=logs/chainT9P.log
ok() { [ $1 -eq 0 ] && echo "$2_OK" >> $L || echo "$2_FAIL" >> $L; }
for D in flowers cars aircraft galaxy10 dtd; do for B in dino lejepa visreg; do
  unset SUPERSIG_NH SUPERSIG_HOLDOUT_DRAW
  $P experiments/135_corpus_norm_everywhere.py --cells $D:$B --arms frozen > logs/exp135f_${D}_${B}.log 2>&1; ok $? P135F_${D}-${B}-multi
  for d in 0 1 2 3 4; do
    SUPERSIG_NH=1 SUPERSIG_HOLDOUT_DRAW=$d $P experiments/135_corpus_norm_everywhere.py --cells $D:$B --arms frozen > logs/exp135f_${D}_${B}_h1_d$d.log 2>&1; ok $? P135F_${D}-${B}-D$d
  done
done; done
echo BLOCK_P_DONE >> $L
