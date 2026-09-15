/home/pharris/venv/lib64/python3.9/site-packages/networkx/utils/backends.py:135: RuntimeWarning: networkx backend defined more than once: nx-loopback
  backends.update(_get_backends("networkx.backends"))
## dtd (exp 125; dino, 100d, holdout 1 [single-holdout], 5 draws d[0, 1, 3, 4, 5], 20 ep)

| arm | probe | probe post | acc | eucl | mahaT | mahaT post | perevt | perevt post@.02 | purity r1 | purity r2 | SpK@.05 | SpK@.05 post | MMD@.05 | MMD@.05 post |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| simclr-ft | 0.940+-0.039 | 0.906+-0.040 | 0.631+-0.004 | 0.550+-0.100 | 0.564+-0.111 | 0.554+-0.130 | 0.040+-0.080 | 0.140+-0.093 | 0.005+-0.010 | 0.003+-0.006 | 0.844+-0.194 | 0.924+-0.098 | 0.720+-0.173 | 0.832+-0.111 |
| sigreg-ssl-ft | 0.890+-0.070 | 0.834+-0.076 | 0.624+-0.010 | 0.627+-0.103 | 0.606+-0.090 | 0.591+-0.072 | 0.050+-0.042 | 0.195+-0.105 | 0.003+-0.002 | 0.000+-0.000 | 0.860+-0.180 | 0.932+-0.075 | 0.828+-0.175 | 0.900+-0.061 |
| nplm-bil-ft | 0.728+-0.113 | 0.879+-0.040 | 0.340+-0.073 | 0.513+-0.075 | 0.502+-0.081 | 0.607+-0.116 | 0.020+-0.019 | 0.105+-0.118 | 0.004+-0.005 | 0.000+-0.000 | 0.768+-0.208 | 0.980+-0.025 | 0.440+-0.307 | 0.716+-0.242 |
| supcon-ft | 0.883+-0.040 | 0.892+-0.070 | 0.739+-0.004 | 0.732+-0.045 | 0.628+-0.077 | 0.688+-0.057 | 0.020+-0.010 | 0.165+-0.131 | 0.028+-0.012 | 0.023+-0.011 | 0.856+-0.130 | 0.888+-0.121 | 0.840+-0.108 | 0.732+-0.122 |
| ss-ft | 0.787+-0.036 | 0.702+-0.056 | 0.762+-0.005 | 0.792+-0.019 | 0.732+-0.054 | 0.687+-0.053 | 0.230+-0.053 | 0.420+-0.095 | 0.219+-0.014 | 0.027+-0.021 | 0.800+-0.119 | 0.888+-0.116 | 0.920+-0.070 | 0.860+-0.079 |
| nplm-sup-ft | 0.749+-0.102 | 0.861+-0.044 | 0.445+-0.018 | 0.632+-0.066 | 0.635+-0.079 | 0.651+-0.103 | 0.055+-0.043 | 0.140+-0.133 | 0.051+-0.027 | 0.001+-0.002 | 0.724+-0.293 | 0.920+-0.067 | 0.736+-0.075 | 0.700+-0.342 |

per-draw spread (probe / mahaT / purity r1), draws [0, 1, 3, 4, 5]
  simclr-ft      probe 0.866-0.985  mahaT 0.415-0.698  purity 0.000-0.026   archived multi-holdout (10 cls) [DIFFERENT REGIME]: probe 0.808 mahaT 0.506
  sigreg-ssl-ft  probe 0.775-0.984  mahaT 0.476-0.718  purity 0.000-0.005   archived multi-holdout (10 cls) [DIFFERENT REGIME]: probe 0.803 mahaT 0.577
  nplm-bil-ft    probe 0.551-0.897  mahaT 0.407-0.650  purity 0.000-0.011   archived multi-holdout (10 cls) [DIFFERENT REGIME]: probe 0.681 mahaT 0.479
  supcon-ft      probe 0.809-0.919  mahaT 0.543-0.723  purity 0.011-0.042   archived multi-holdout (10 cls) [DIFFERENT REGIME]: probe 0.799 mahaT 0.646
  ss-ft          probe 0.732-0.847  mahaT 0.646-0.802  purity 0.193-0.233   archived multi-holdout (10 cls) [DIFFERENT REGIME]: probe 0.782 mahaT 0.742
  nplm-sup-ft    probe 0.586-0.859  mahaT 0.550-0.776  purity 0.021-0.102   archived multi-holdout (10 cls) [DIFFERENT REGIME]: probe 0.586 mahaT 0.615
