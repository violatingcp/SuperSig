#!/bin/bash
# Block S5: eval-only exp-68 re-pass on every archived scratch-CIFAR cell lacking the
# injected-fraction post metrics (postf_*).  Waits for Block L to finish.
cd /home/pharris/sigreg/SuperSig; P=/home/pharris/venv/bin/python; L=logs/chainT9S.log
while pgrep -f "chainT9[L].sh" >/dev/null; do sleep 600; done
ok() { [ $1 -eq 0 ] && echo "$2_OK" >> $L || echo "$2_FAIL" >> $L; }
for f in $($P - <<'PY'
import numpy as np, glob, os
for f in sorted(glob.glob('logs/exp68/scratch_discovery_cifar*.npz')):
    if 'archived' in f: continue
    z=np.load(f,allow_pickle=True)
    if any(k.startswith('postf') for k in z): continue
    b=os.path.basename(f)[len('scratch_discovery_'):-4]
    ds,_,h=b.partition('_h'); print(f"{ds}:{h or 4}:{','.join(map(str,z['bases']))}")
PY
); do
  ds=${f%%:*}; r=${f#*:}; h=${r%%:*}; bases=${r#*:}
  $P experiments/68_scratch_discovery.py --dataset $ds --holdout $h --bases $bases > logs/exp68_${ds}_h${h}_repass.log 2>&1; ok $? S5_${ds}-h$h
done
echo BLOCK_S5_DONE >> $L
