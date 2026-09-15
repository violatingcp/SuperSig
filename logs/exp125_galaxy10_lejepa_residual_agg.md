## galaxy10 residual spaces (exp 125/71; lejepa, 100(+100)d, holdout 1 [single-holdout], 5 draws d[0, 3, 5, 7, 8], 20 ep)

| space | probe | acc | eucl | mahaT | perevt | archived d{9} probe |
|---|---|---|---|---|---|---|
| supcon-ft (parent) | 0.892+-0.056 | 0.753+-0.018 | 0.591+-0.080 | 0.711+-0.115 | 0.047+-0.027 | 0.919 |
| supcon-ft-res (residual) | 0.910+-0.045 | 0.638+-0.018 | 0.623+-0.058 | 0.643+-0.090 | 0.114+-0.057 | 0.960 |
| supcon-ft-res (concat) | 0.945+-0.033 | 0.668+-0.016 | 0.630+-0.059 | 0.709+-0.079 | 0.124+-0.059 | 0.975 |
| supcon-ft-res-nplm (residual) | 0.887+-0.049 | 0.702+-0.025 | 0.596+-0.119 | 0.598+-0.105 | 0.083+-0.066 | 0.916 |
| supcon-ft-res-nplm (concat) | 0.931+-0.045 | 0.760+-0.022 | 0.644+-0.080 | 0.699+-0.089 | 0.086+-0.061 | 0.941 |
| ss-ft (parent) | 0.762+-0.181 | 0.696+-0.030 | 0.633+-0.148 | 0.665+-0.163 | 0.134+-0.119 | 0.803 |
| ss-ft-res (residual) | 0.854+-0.075 | 0.611+-0.010 | 0.521+-0.123 | 0.589+-0.132 | 0.061+-0.060 | 0.854 |
| ss-ft-res (concat) | 0.855+-0.094 | 0.706+-0.020 | 0.627+-0.068 | 0.692+-0.118 | 0.093+-0.047 | 0.879 |

PAIRED per-draw deltas (concat - its parent), draws [0, 3, 5, 7, 8]
  supcon-ft-res_(concat)       probe   +0.053+-0.030  wins 5/5  +0.019 +0.045 +0.069 +0.103 +0.029
  supcon-ft-res_(concat)       eucl    +0.039+-0.064  wins 4/5  -0.062 +0.127 +0.084 +0.025 +0.020
  supcon-ft-res_(concat)       mahaT   -0.002+-0.101  wins 3/5  -0.182 -0.004 +0.011 +0.035 +0.130
  supcon-ft-res_(concat)       perevt  +0.077+-0.037  wins 5/5  +0.099 +0.101 +0.047 +0.116 +0.020
  supcon-ft-res-nplm_(concat)  probe   +0.039+-0.025  wins 5/5  +0.018 +0.034 +0.030 +0.088 +0.025
  supcon-ft-res-nplm_(concat)  eucl    +0.053+-0.067  wins 4/5  -0.070 +0.102 +0.037 +0.116 +0.080
  supcon-ft-res-nplm_(concat)  mahaT   -0.013+-0.038  wins 3/5  -0.085 +0.006 +0.009 +0.022 -0.014
  supcon-ft-res-nplm_(concat)  perevt  +0.039+-0.040  wins 5/5  +0.001 +0.067 +0.020 +0.103 +0.004
  ss-ft-res_(concat)           probe   +0.093+-0.093  wins 5/5  +0.006 +0.078 +0.046 +0.273 +0.062
  ss-ft-res_(concat)           eucl    -0.006+-0.105  wins 2/5  -0.153 +0.157 +0.057 -0.053 -0.038
  ss-ft-res_(concat)           mahaT   +0.027+-0.055  wins 3/5  -0.055 +0.091 +0.033 -0.013 +0.080
  ss-ft-res_(concat)           perevt  -0.042+-0.119  wins 3/5  -0.273 +0.069 +0.003 +0.003 -0.011

exp 72 discovery on the winner concat (5 draws):
  probe_pre        0.945+-0.033   per-draw 0.982 0.929 0.894 0.943 0.978
  probe_post       0.936+-0.034   per-draw 0.986 0.923 0.887 0.924 0.963
  eucl_pre         0.630+-0.059   per-draw 0.648 0.711 0.561 0.667 0.564
  eucl_post        0.628+-0.084   per-draw 0.659 0.701 0.470 0.690 0.617
  maha_pre         0.709+-0.079   per-draw 0.730 0.653 0.586 0.787 0.789
  maha_post        0.743+-0.085   per-draw 0.859 0.677 0.619 0.796 0.763
  purity_r1        0.161+-0.121   per-draw 0.339 0.273 0.060 0.044 0.092
  purity_r2        0.023+-0.020   per-draw 0.038 0.012 0.000 0.054 0.013
  perevt_post@.05  0.117+-0.062   per-draw 0.147 0.203 0.038 0.143 0.056
  spk_post@.05     0.604+-0.259   per-draw 1.000 0.560 0.220 0.740 0.500
