# exp76 interpretability [cifar10_32d]
holdout classes starred; superclass partition: yes

## supcon (parent)  [cifar10_32d]

agree@1=0.900 agree@5=0.780 (chance 0.533)  dendro-purity=0.904  silhouette=0.311  within/between=0.662


| class | 5 nearest class centroids (cosine dist) |
|---|---|
| airplane | deer*(0.39) bird(0.42) ship(0.45) truck(0.51) cat(0.53) |
| automobile | truck(0.45) ship(0.50) airplane(0.55) deer*(0.62) cat(0.63) |
| bird | deer*(0.13) cat(0.41) airplane(0.42) dog(0.43) frog(0.45) |
| cat | deer*(0.18) dog(0.27) bird(0.41) frog(0.42) horse(0.47) |
| deer* | bird(0.13) cat(0.18) dog(0.19) horse(0.24) frog(0.31) |
| dog | deer*(0.19) cat(0.27) bird(0.43) horse(0.44) frog(0.51) |
| frog | deer*(0.31) cat(0.42) bird(0.45) dog(0.51) airplane(0.62) |
| horse | deer*(0.24) dog(0.44) cat(0.47) bird(0.48) airplane(0.56) |
| ship | airplane(0.45) automobile(0.50) truck(0.51) deer*(0.59) bird(0.61) |
| truck | automobile(0.45) airplane(0.51) ship(0.51) deer*(0.56) cat(0.57) |

## supcon-res (residual)  [cifar10_32d]

agree@1=0.800 agree@5=0.760 (chance 0.533)  dendro-purity=0.904  silhouette=0.524  within/between=0.437


| class | 5 nearest class centroids (cosine dist) |
|---|---|
| airplane | ship(0.05) truck(0.22) bird(0.24) cat(0.29) automobile(0.30) |
| automobile | truck(0.17) ship(0.20) airplane(0.30) cat(0.44) bird(0.45) |
| bird | cat(0.02) dog(0.05) deer*(0.09) horse(0.15) frog(0.15) |
| cat | bird(0.02) dog(0.04) deer*(0.07) frog(0.10) horse(0.18) |
| deer* | cat(0.07) frog(0.08) bird(0.09) dog(0.10) horse(0.14) |
| dog | cat(0.04) bird(0.05) deer*(0.10) horse(0.15) frog(0.18) |
| frog | deer*(0.08) cat(0.10) bird(0.15) dog(0.18) horse(0.38) |
| horse | deer*(0.14) bird(0.15) dog(0.15) cat(0.18) airplane(0.34) |
| ship | airplane(0.05) truck(0.15) automobile(0.20) bird(0.35) cat(0.37) |
| truck | ship(0.15) automobile(0.17) airplane(0.22) cat(0.42) bird(0.43) |

## supcon-res (concat)  [cifar10_32d]

agree@1=0.800 agree@5=0.780 (chance 0.533)  dendro-purity=0.904  silhouette=0.363  within/between=0.612


| class | 5 nearest class centroids (cosine dist) |
|---|---|
| airplane | ship(0.27) bird(0.35) truck(0.37) deer*(0.40) cat(0.42) |
| automobile | truck(0.33) ship(0.37) airplane(0.44) cat(0.55) deer*(0.58) |
| bird | deer*(0.19) cat(0.26) dog(0.27) frog(0.33) airplane(0.35) |
| cat | dog(0.17) deer*(0.19) bird(0.26) frog(0.29) horse(0.36) |
| deer* | dog(0.18) cat(0.19) bird(0.19) frog(0.23) horse(0.27) |
| dog | cat(0.17) deer*(0.18) bird(0.27) horse(0.33) frog(0.37) |
| frog | deer*(0.23) cat(0.29) bird(0.33) dog(0.37) horse(0.54) |
| horse | deer*(0.27) dog(0.33) cat(0.36) bird(0.36) airplane(0.47) |
| ship | airplane(0.27) truck(0.36) automobile(0.37) bird(0.51) deer*(0.53) |
| truck | automobile(0.33) ship(0.36) airplane(0.37) cat(0.51) deer*(0.52) |

## supcon-res-nplm (residual)  [cifar10_32d]

agree@1=0.800 agree@5=0.760 (chance 0.533)  dendro-purity=0.904  silhouette=0.655  within/between=0.285


| class | 5 nearest class centroids (cosine dist) |
|---|---|
| airplane | ship(0.00) truck(0.01) automobile(0.02) cat(0.02) bird(0.02) |
| automobile | truck(0.00) ship(0.01) airplane(0.02) cat(0.05) bird(0.06) |
| bird | deer*(0.00) cat(0.00) dog(0.00) horse(0.01) frog(0.02) |
| cat | bird(0.00) dog(0.00) deer*(0.00) horse(0.01) frog(0.02) |
| deer* | bird(0.00) dog(0.00) cat(0.00) horse(0.01) frog(0.02) |
| dog | bird(0.00) cat(0.00) deer*(0.00) horse(0.01) frog(0.02) |
| frog | deer*(0.02) cat(0.02) bird(0.02) dog(0.02) horse(0.05) |
| horse | dog(0.01) bird(0.01) cat(0.01) deer*(0.01) airplane(0.03) |
| ship | airplane(0.00) truck(0.01) automobile(0.01) cat(0.04) bird(0.04) |
| truck | automobile(0.00) ship(0.01) airplane(0.01) cat(0.05) horse(0.06) |

## supcon-res-nplm (concat)  [cifar10_32d]

agree@1=0.900 agree@5=0.800 (chance 0.533)  dendro-purity=0.904  silhouette=0.328  within/between=0.637


| class | 5 nearest class centroids (cosine dist) |
|---|---|
| airplane | deer*(0.18) bird(0.24) ship(0.24) cat(0.28) dog(0.29) |
| automobile | truck(0.29) ship(0.31) airplane(0.32) deer*(0.36) cat(0.39) |
| bird | deer*(0.13) dog(0.23) cat(0.23) airplane(0.24) frog(0.27) |
| cat | deer*(0.12) dog(0.14) bird(0.23) frog(0.25) airplane(0.28) |
| deer* | dog(0.10) cat(0.12) bird(0.13) airplane(0.18) horse(0.20) |
| dog | deer*(0.10) cat(0.14) bird(0.23) horse(0.26) airplane(0.29) |
| frog | deer*(0.20) cat(0.25) bird(0.27) dog(0.29) airplane(0.36) |
| horse | deer*(0.20) dog(0.26) cat(0.29) bird(0.30) airplane(0.34) |
| ship | airplane(0.24) deer*(0.30) automobile(0.31) truck(0.32) cat(0.37) |
| truck | automobile(0.29) airplane(0.30) ship(0.32) deer*(0.35) cat(0.36) |
