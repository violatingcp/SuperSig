#!/bin/bash
# Exp 127 (re-run 109 with BN actually frozen) + Block E (128/129 on the exp-113 embedding bank).
cd /home/pharris/sigreg/SuperSig
P=/home/pharris/venv/bin/python
L=logs/chainT9EF.log
$P experiments/109_c100_density_pool.py --refresh > logs/exp127_rerun109.log 2>&1 && echo EXP127_OK >> $L || echo EXP127_FAIL >> $L
$P experiments/128_pool_cut_optimization.py --glob "logs/exp113/embs/cifar100_tau*_s*.npz" --holdouts 99 --bic > logs/exp128_c100bank.log 2>&1 && echo EXP128_OK >> $L || echo EXP128_FAIL >> $L
$P experiments/129_legal_pool_cut.py --glob "logs/exp113/embs/cifar100_tau*_s*.npz" --holdouts 99 > logs/exp129_c100bank.log 2>&1 && echo EXP129_OK >> $L || echo EXP129_FAIL >> $L
echo BLOCK_EF_DONE >> $L
