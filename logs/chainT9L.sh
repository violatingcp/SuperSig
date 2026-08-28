#!/bin/bash
# Block L: from-scratch CIFAR master grid (exps 67/68/136).
#   Tier 1: cifar100 holdout 4 (8 ckpts exist) -> 68 (all 8) -> 136
#   Tier 2: cifar10  holdout 4: 67 x 8 arms -> 68 -> 136
#   Tier 3: draws for the shortlist {supcon, ssig, nplmsd, nplmcw}:
#           cifar10 holdouts {8,9,7} (draws 0,1,2), cifar100 {43,57,48} (draws 0,1,2)
cd /home/pharris/sigreg/SuperSig
P=/home/pharris/venv/bin/python; L=logs/chainT9L.log
unset SUPERSIG_NH SUPERSIG_HOLDOUT_DRAW
ok() { [ $1 -eq 0 ] && echo "$2_OK" >> $L || echo "$2_FAIL" >> $L; }
cell() { # ds holdout arms...
  local ds=$1 h=$2; shift 2
  $P experiments/68_scratch_discovery.py --dataset $ds --holdout $h --bases $(echo "$@" | tr ' ' ',') > logs/exp68_${ds}_h$h.log 2>&1; ok $? L68_${ds}-h$h
  $P experiments/136_scratch_master.py --dataset $ds --holdout $h --arms "$@" > logs/exp136_${ds}_h$h.log 2>&1; ok $? L136_${ds}-h$h
}
ALL="simclr visreg nplm supcon supsig nplmcw ssig nplmsd"; SHORT="supcon ssig nplmsd nplmcw"
# Tier 1
cell cifar100 4 $ALL
echo TIER1_DONE >> $L
# Tier 2
$P experiments/67_scratch_pretrain.py --dataset cifar10 --holdout 4 --arms $ALL --resume > logs/exp67_cifar10_h4.log 2>&1; ok $? L67_cifar10-h4
cell cifar10 4 $ALL
echo TIER2_DONE >> $L
# Tier 3 (interleave datasets so both get intervals early)
for i in 0 1 2; do
  for spec in "cifar10 8 9 7" "cifar100 43 57 48"; do set -- $spec; ds=$1; h=$(echo $spec | cut -d' ' -f$((i+2)))
    $P experiments/67_scratch_pretrain.py --dataset $ds --holdout $h --arms $SHORT --resume > logs/exp67_${ds}_h$h.log 2>&1; ok $? L67_${ds}-h$h
    cell $ds $h $SHORT
  done
done
echo BLOCK_L_DONE >> $L
