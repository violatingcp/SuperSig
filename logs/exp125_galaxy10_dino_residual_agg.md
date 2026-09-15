## galaxy10 residual spaces (exp 125/71; dino, 100(+100)d, holdout 1 [single-holdout], 5 draws d[0, 3, 5, 7, 8], 20 ep)

| space | probe | acc | eucl | mahaT | perevt | archived d{9} probe |
|---|---|---|---|---|---|---|
| supcon-ft (parent) | 0.879+-0.026 | 0.616+-0.013 | 0.526+-0.145 | 0.500+-0.135 | 0.063+-0.092 | 0.938 |
| supcon-ft-res (residual) | 0.891+-0.025 | 0.526+-0.011 | 0.478+-0.107 | 0.467+-0.139 | 0.055+-0.048 | 0.952 |
| supcon-ft-res (concat) | 0.903+-0.027 | 0.594+-0.010 | 0.497+-0.122 | 0.488+-0.138 | 0.059+-0.078 | 0.955 |
| supcon-ft-res-nplm (residual) | 0.862+-0.027 | 0.598+-0.011 | 0.553+-0.115 | 0.504+-0.111 | 0.067+-0.066 | 0.925 |
| supcon-ft-res-nplm (concat) | 0.890+-0.023 | 0.619+-0.012 | 0.538+-0.140 | 0.494+-0.134 | 0.062+-0.088 | 0.947 |
| ss-ft (parent) | 0.804+-0.064 | 0.588+-0.015 | 0.499+-0.104 | 0.482+-0.114 | 0.040+-0.038 | 0.872 |
| ss-ft-res (residual) | 0.852+-0.033 | 0.522+-0.014 | 0.493+-0.084 | 0.487+-0.101 | 0.056+-0.042 | 0.923 |
| ss-ft-res (concat) | 0.869+-0.026 | 0.568+-0.013 | 0.494+-0.089 | 0.486+-0.116 | 0.052+-0.040 | 0.930 |

PAIRED per-draw deltas (concat - its parent), draws [0, 3, 5, 7, 8]
  supcon-ft-res_(concat)       probe   +0.023+-0.007  wins 5/5  +0.013 +0.030 +0.018 +0.033 +0.023
  supcon-ft-res_(concat)       eucl    -0.030+-0.043  wins 1/5  +0.043 -0.016 -0.048 -0.088 -0.039
  supcon-ft-res_(concat)       mahaT   -0.012+-0.020  wins 3/5  -0.045 +0.001 +0.001 -0.024 +0.008
  supcon-ft-res_(concat)       perevt  -0.004+-0.016  wins 2/5  +0.011 +0.004 -0.035 +0.000 +0.000
  supcon-ft-res-nplm_(concat)  probe   +0.011+-0.008  wins 5/5  +0.007 +0.025 +0.010 +0.000 +0.010
  supcon-ft-res-nplm_(concat)  eucl    +0.012+-0.021  wins 3/5  +0.014 -0.005 -0.016 +0.044 +0.021
  supcon-ft-res-nplm_(concat)  mahaT   -0.005+-0.015  wins 2/5  -0.024 -0.014 -0.011 +0.015 +0.008
  supcon-ft-res-nplm_(concat)  perevt  -0.001+-0.004  wins 2/5  +0.000 +0.001 -0.010 +0.003 +0.000
  ss-ft-res_(concat)           probe   +0.065+-0.047  wins 5/5  +0.020 +0.056 +0.033 +0.154 +0.062
  ss-ft-res_(concat)           eucl    -0.005+-0.036  wins 3/5  +0.044 -0.008 +0.005 -0.067 +0.001
  ss-ft-res_(concat)           mahaT   +0.004+-0.014  wins 3/5  +0.024 +0.010 +0.004 -0.002 -0.018
  ss-ft-res_(concat)           perevt  +0.012+-0.014  wins 5/5  +0.038 +0.012 +0.005 +0.003 +0.001

exp 72 discovery on the winner concat (5 draws):
  probe_pre        0.903+-0.027   per-draw 0.930 0.891 0.859 0.932 0.902
  probe_post       0.886+-0.024   per-draw 0.923 0.868 0.854 0.902 0.881
  eucl_pre         0.497+-0.122   per-draw 0.394 0.587 0.636 0.552 0.315
  eucl_post        0.506+-0.135   per-draw 0.372 0.615 0.645 0.583 0.315
  maha_pre         0.488+-0.138   per-draw 0.347 0.582 0.655 0.554 0.302
  maha_post        0.470+-0.146   per-draw 0.345 0.609 0.636 0.497 0.262
  purity_r1        0.109+-0.116   per-draw 0.073 0.167 0.307 0.000 0.000
  purity_r2        0.106+-0.126   per-draw 0.000 0.245 0.275 0.011 0.000
  perevt_post@.05  0.065+-0.074   per-draw 0.007 0.087 0.202 0.010 0.020
  spk_post@.05     0.204+-0.129   per-draw 0.320 0.340 0.060 0.260 0.040
