#!/bin/bash
# Experiment 91 (IMPROVEMENT_TESTS.md #91): multi-seed the uncited records.
#
# Seeds 1-2 (seed 0 = the archived records) of the four record cells,
# paired protocol of exp-75/cars: seed-aware exp-70 parent ft ->
# seed-aware exp-71 residual ft -> (where the record includes it) the
# exp-72 discovery step on the seeded checkpoints (--ckpt-sfx).
# Checkpoints/banks/npz all carry the _s{seed} suffix (clobber bug fixed
# in the exp-70/71 tag this session).  Cells:
#   aircraft/visreg supcon-ft->res-nplm concat   (0.863, pre-discovery)
#   flowers/dino    supcon-ft->res-nplm + disc   (0.885 -> 0.906)
#   dtd/visreg      supcon-ft->res     + disc    (0.847 -> 0.862)
#   galaxy10/lejepa supcon-ft->res concat        (0.975, pre-discovery)
# cars/visreg (0.855) was already multi-seeded in exp 75.
#
#   nohup bash experiments/91_multiseed_records.sh > logs/exp91_run.log 2>&1 &
set -e
PY=/home/pharris/venv/bin/python
cd "$(dirname "$0")/.."

for s in 1 2; do
  echo "########## EXP91 SEED $s ##########"
  echo "----- exp91 aircraft:visreg seed $s -----"
  $PY experiments/70_cars_ft_suite.py --dataset aircraft --base visreg \
      --arms supcon-ft --seed $s --skip-power --skip-discovery
  $PY experiments/71_residual_ft_grid.py --dataset aircraft --base visreg \
      --seed $s --runs supcon-ft:res-nplm
  echo "----- exp91 flowers:dino seed $s -----"
  $PY experiments/70_cars_ft_suite.py --dataset flowers --base dino \
      --arms supcon-ft --seed $s --skip-power --skip-discovery
  $PY experiments/71_residual_ft_grid.py --dataset flowers --base dino \
      --seed $s --runs supcon-ft:res-nplm
  $PY experiments/72_residual_discovery.py --cells flowers:dino \
      --seed $s --ckpt-sfx _s$s --skip-power
  echo "----- exp91 dtd:visreg seed $s -----"
  $PY experiments/70_cars_ft_suite.py --dataset dtd --base visreg \
      --arms supcon-ft --seed $s --skip-power --skip-discovery
  $PY experiments/71_residual_ft_grid.py --dataset dtd --base visreg \
      --seed $s --runs supcon-ft:res
  $PY experiments/72_residual_discovery.py --cells dtd:visreg \
      --seed $s --ckpt-sfx _s$s --skip-power
  echo "----- exp91 galaxy10:lejepa seed $s -----"
  $PY experiments/70_cars_ft_suite.py --dataset galaxy10 --base lejepa \
      --arms supcon-ft --seed $s --skip-power --skip-discovery
  $PY experiments/71_residual_ft_grid.py --dataset galaxy10 --base lejepa \
      --seed $s --runs supcon-ft:res
done
echo "EXP91 DONE."
