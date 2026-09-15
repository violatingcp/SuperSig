#!/bin/bash
# Experiment 84 (IMPROVEMENT_TESTS.md #84): the two-stage recipe under the
# strict open-world protocol.
#
# AIRCRAFT_MASTER_TABLE reading #8 (LeJEPA-ft + supcon_sigreg 0.812 etc.)
# used exp-62 trunks fine-tuned on ALL 100 variants incl. the holdouts --
# optimistic and uncitable.  Stage 1 here retrains the NPLM-sup trunk with
# holdouts 90-99 EXCLUDED (exp-70 protocol, arm nplm-sup-ft, seed 0);
# stage 2 runs the exp-51 8-arm head suite on the resulting banks
# (--trunk-ckpt-arm nplm-sup-ft_seen resolves to the _seen ckpt and names
# the banks distinctly).  Compare vs the exp-71 champion 0.866+-0.002.
#
# Prediction: gains shrink but survive on LeJEPA/VISReg; two-stage lands
# between the exp-70 parents and the exp-71 champions.  Falsifier: gains
# vanish -> reading #8 was contamination and is struck from the table.
#
#   nohup bash experiments/84_two_stage_strict.sh > logs/exp84_run.log 2>&1 &
set -e
PY=/home/pharris/venv/bin/python
cd "$(dirname "$0")/.."

for base in dino lejepa visreg; do
  echo "----- exp84 stage1 aircraft:$base nplm-sup-ft (seen-only) -----"
  $PY experiments/70_cars_ft_suite.py --dataset aircraft --base $base \
      --arms nplm-sup-ft --skip-power --skip-discovery
  echo "----- exp84 stage2 aircraft:$base 8-arm head suite -----"
  $PY experiments/51_nplm_aircraft_suite.py --dataset aircraft --base $base \
      --trunk-ckpt-arm nplm-sup-ft_seen --skip-power
done
echo "EXP84 DONE."
