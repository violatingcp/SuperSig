#!/bin/bash
# Block Q: exp 144 GCD-protocol benchmark on aircraft, cifar10, cifar100 (cars launched separately first).
cd /home/pharris/sigreg/SuperSig; P=/home/pharris/venv/bin/python; L=logs/chainT9Q.log
ok() { [ $1 -eq 0 ] && echo "$2_OK" >> $L || echo "$2_FAIL" >> $L; }
while pgrep -f "144_gcd_benchmark.py --dataset cars" >/dev/null; do sleep 30; done; echo Q144_cars_DONE >> $L
$P experiments/144_gcd_benchmark.py --dataset aircraft > logs/exp144_aircraft.log 2>&1; ok $? Q144_aircraft
$P experiments/144_gcd_benchmark.py --dataset cifar10 > logs/exp144_cifar10.log 2>&1; ok $? Q144_cifar10
until [ -f data/tf_feats_cifar100_dino_vitb16.pt ]; do sleep 60; done
$P experiments/144_gcd_benchmark.py --dataset cifar100 > logs/exp144_cifar100.log 2>&1; ok $? Q144_cifar100
echo BLOCK_Q_DONE >> $L
