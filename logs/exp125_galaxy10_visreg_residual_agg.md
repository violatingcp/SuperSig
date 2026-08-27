## galaxy10 residual spaces (exp 125/71; visreg, 100(+100)d, holdout 1 [single-holdout], 5 draws d[0, 3, 5, 7, 8], 20 ep)

| space | probe | acc | eucl | mahaT | perevt | archived d{9} probe |
|---|---|---|---|---|---|---|
| supcon-ft (parent) | 0.906+-0.043 | 0.775+-0.019 | 0.700+-0.076 | 0.686+-0.074 | 0.095+-0.054 | 0.939 |
| supcon-ft-res (residual) | 0.930+-0.036 | 0.691+-0.030 | 0.528+-0.068 | 0.606+-0.052 | 0.056+-0.055 | 0.964 |
| supcon-ft-res (concat) | 0.947+-0.025 | 0.769+-0.020 | 0.647+-0.065 | 0.658+-0.061 | 0.087+-0.054 | 0.964 |
| supcon-ft-res-nplm (residual) | 0.890+-0.053 | 0.774+-0.017 | 0.685+-0.109 | 0.709+-0.109 | 0.114+-0.101 | 0.936 |
| supcon-ft-res-nplm (concat) | 0.923+-0.036 | 0.781+-0.019 | 0.709+-0.097 | 0.711+-0.106 | 0.116+-0.088 | 0.946 |
| ss-ft (parent) | 0.823+-0.108 | 0.744+-0.017 | 0.615+-0.118 | 0.581+-0.136 | 0.090+-0.096 | 0.865 |
| ss-ft-res (residual) | 0.867+-0.054 | 0.645+-0.023 | 0.498+-0.082 | 0.554+-0.102 | 0.048+-0.032 | 0.916 |
| ss-ft-res (concat) | 0.888+-0.052 | 0.740+-0.018 | 0.576+-0.095 | 0.586+-0.136 | 0.086+-0.062 | 0.920 |

PAIRED per-draw deltas (concat - its parent), draws [0, 3, 5, 7, 8]
  supcon-ft-res_(concat)       probe   +0.041+-0.029  wins 5/5  +0.022 +0.042 +0.031 +0.097 +0.013
  supcon-ft-res_(concat)       eucl    -0.053+-0.044  wins 1/5  -0.099 +0.024 -0.051 -0.092 -0.046
  supcon-ft-res_(concat)       mahaT   -0.028+-0.037  wins 2/5  -0.041 -0.018 +0.008 -0.093 +0.004
  supcon-ft-res_(concat)       perevt  -0.008+-0.058  wins 2/5  -0.013 +0.054 -0.004 -0.113 +0.034
  supcon-ft-res-nplm_(concat)  probe   +0.017+-0.009  wins 5/5  +0.014 +0.013 +0.018 +0.033 +0.006
  supcon-ft-res-nplm_(concat)  eucl    +0.009+-0.023  wins 2/5  -0.000 -0.011 -0.016 +0.032 +0.042
  supcon-ft-res-nplm_(concat)  mahaT   +0.025+-0.036  wins 4/5  +0.025 +0.006 -0.032 +0.053 +0.071
  supcon-ft-res-nplm_(concat)  perevt  +0.021+-0.037  wins 2/5  -0.012 -0.002 -0.013 +0.063 +0.069
  ss-ft-res_(concat)           probe   +0.065+-0.066  wins 5/5  +0.019 +0.048 +0.045 +0.195 +0.018
  ss-ft-res_(concat)           eucl    -0.039+-0.067  wins 2/5  -0.047 +0.023 +0.042 -0.070 -0.143
  ss-ft-res_(concat)           mahaT   +0.005+-0.048  wins 3/5  -0.072 +0.012 +0.057 -0.021 +0.050
  ss-ft-res_(concat)           perevt  -0.004+-0.037  wins 3/5  +0.019 +0.024 +0.016 -0.076 -0.004

exp 72 discovery on the winner concat (5 draws):
  probe_pre        0.930+-0.036   per-draw 0.961 0.926 0.863 0.941 0.960
  probe_post       0.922+-0.037   per-draw 0.960 0.905 0.862 0.925 0.958
  eucl_pre         0.528+-0.068   per-draw 0.458 0.641 0.462 0.526 0.554
  eucl_post        0.584+-0.090   per-draw 0.472 0.623 0.483 0.653 0.691
  maha_pre         0.606+-0.052   per-draw 0.626 0.616 0.542 0.558 0.686
  maha_post        0.602+-0.046   per-draw 0.598 0.605 0.540 0.582 0.682
  purity_r1        0.098+-0.099   per-draw 0.000 0.231 0.060 0.000 0.202
  purity_r2        0.061+-0.063   per-draw 0.026 0.036 0.048 0.011 0.186
  perevt_post@.05  0.072+-0.071   per-draw 0.025 0.114 0.031 0.000 0.192
  spk_post@.05     0.616+-0.333   per-draw 0.940 0.240 0.180 0.860 0.860
