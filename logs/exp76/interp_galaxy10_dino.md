# exp76 interpretability [galaxy10_dino]
holdout classes starred; superclass partition: yes

## frozen-dino (768d)  [galaxy10_dino]

agree@1=0.700 agree@5=0.280 (chance 0.178)  dendro-purity=0.783  silhouette=0.248  within/between=0.346


| class | 5 nearest class centroids (cosine dist) |
|---|---|
| disturbed | unbarred loose spiral(0.01) merging(0.01) in-between smooth(0.01) cigar smooth(0.02) barred spiral(0.02) |
| merging | disturbed(0.01) in-between smooth(0.01) unbarred loose spiral(0.01) round smooth(0.02) cigar smooth(0.02) |
| round smooth | in-between smooth(0.01) merging(0.02) disturbed(0.02) unbarred tight spiral(0.04) unbarred loose spiral(0.04) |
| in-between smooth | round smooth(0.01) merging(0.01) disturbed(0.01) cigar smooth(0.02) unbarred loose spiral(0.02) |
| cigar smooth | disturbed(0.02) in-between smooth(0.02) merging(0.02) edge-on bulge*(0.02) unbarred loose spiral(0.03) |
| barred spiral | unbarred loose spiral(0.01) unbarred tight spiral(0.01) disturbed(0.02) merging(0.02) in-between smooth(0.04) |
| unbarred tight spiral | barred spiral(0.01) unbarred loose spiral(0.01) disturbed(0.02) merging(0.02) round smooth(0.04) |
| unbarred loose spiral | barred spiral(0.01) disturbed(0.01) unbarred tight spiral(0.01) merging(0.01) in-between smooth(0.02) |
| edge-on no bulge | edge-on bulge*(0.03) cigar smooth(0.05) merging(0.07) disturbed(0.07) unbarred loose spiral(0.08) |
| edge-on bulge* | cigar smooth(0.02) edge-on no bulge(0.03) merging(0.04) disturbed(0.04) in-between smooth(0.05) |

## simclr-ft  [galaxy10_dino]

agree@1=0.600 agree@5=0.280 (chance 0.178)  dendro-purity=0.713  silhouette=0.333  within/between=0.502


| class | 5 nearest class centroids (cosine dist) |
|---|---|
| disturbed | unbarred loose spiral(0.45) merging(0.86) cigar smooth(0.93) unbarred tight spiral(0.95) round smooth(0.96) |
| merging | round smooth(0.53) cigar smooth(0.80) edge-on bulge*(0.80) in-between smooth(0.80) barred spiral(0.82) |
| round smooth | in-between smooth(0.42) merging(0.53) disturbed(0.96) edge-on bulge*(1.04) cigar smooth(1.10) |
| in-between smooth | round smooth(0.42) cigar smooth(0.61) merging(0.80) edge-on bulge*(0.97) disturbed(1.15) |
| cigar smooth | edge-on bulge*(0.53) in-between smooth(0.61) merging(0.80) disturbed(0.93) edge-on no bulge(0.95) |
| barred spiral | unbarred tight spiral(0.15) unbarred loose spiral(0.50) merging(0.82) disturbed(1.06) edge-on no bulge(1.16) |
| unbarred tight spiral | barred spiral(0.15) unbarred loose spiral(0.52) disturbed(0.95) merging(1.02) edge-on no bulge(1.06) |
| unbarred loose spiral | disturbed(0.45) barred spiral(0.50) unbarred tight spiral(0.52) merging(1.12) edge-on no bulge(1.16) |
| edge-on no bulge | edge-on bulge*(0.21) cigar smooth(0.95) disturbed(1.06) unbarred tight spiral(1.06) merging(1.07) |
| edge-on bulge* | edge-on no bulge(0.21) cigar smooth(0.53) merging(0.80) in-between smooth(0.97) round smooth(1.04) |

## sigreg-ssl-ft  [galaxy10_dino]

agree@1=0.700 agree@5=0.260 (chance 0.178)  dendro-purity=0.775  silhouette=0.290  within/between=0.446


| class | 5 nearest class centroids (cosine dist) |
|---|---|
| disturbed | unbarred loose spiral(0.40) cigar smooth(0.93) unbarred tight spiral(0.94) in-between smooth(1.02) edge-on no bulge(1.10) |
| merging | round smooth(0.24) in-between smooth(0.36) edge-on bulge*(0.62) cigar smooth(0.74) edge-on no bulge(0.97) |
| round smooth | in-between smooth(0.18) merging(0.24) cigar smooth(1.01) edge-on bulge*(1.11) disturbed(1.29) |
| in-between smooth | round smooth(0.18) merging(0.36) cigar smooth(0.52) edge-on bulge*(0.85) disturbed(1.02) |
| cigar smooth | edge-on bulge*(0.21) edge-on no bulge(0.48) in-between smooth(0.52) merging(0.74) disturbed(0.93) |
| barred spiral | unbarred tight spiral(0.14) unbarred loose spiral(0.37) disturbed(1.16) edge-on no bulge(1.40) edge-on bulge*(1.43) |
| unbarred tight spiral | barred spiral(0.14) unbarred loose spiral(0.26) disturbed(0.94) edge-on no bulge(1.45) round smooth(1.46) |
| unbarred loose spiral | unbarred tight spiral(0.26) barred spiral(0.37) disturbed(0.40) edge-on no bulge(1.26) cigar smooth(1.49) |
| edge-on no bulge | edge-on bulge*(0.17) cigar smooth(0.48) merging(0.97) disturbed(1.10) in-between smooth(1.22) |
| edge-on bulge* | edge-on no bulge(0.17) cigar smooth(0.21) merging(0.62) in-between smooth(0.85) round smooth(1.11) |

## nplm-bil-ft  [galaxy10_dino]

agree@1=0.200 agree@5=0.280 (chance 0.178)  dendro-purity=0.490  silhouette=-0.079  within/between=0.537


| class | 5 nearest class centroids (cosine dist) |
|---|---|
| disturbed | unbarred loose spiral(0.00) barred spiral(0.01) cigar smooth(0.01) in-between smooth(0.02) merging(0.02) |
| merging | in-between smooth(0.00) unbarred tight spiral(0.01) round smooth(0.01) disturbed(0.02) cigar smooth(0.02) |
| round smooth | merging(0.01) in-between smooth(0.01) unbarred tight spiral(0.02) cigar smooth(0.05) disturbed(0.05) |
| in-between smooth | merging(0.00) unbarred tight spiral(0.01) round smooth(0.01) disturbed(0.02) cigar smooth(0.02) |
| cigar smooth | disturbed(0.01) unbarred loose spiral(0.02) edge-on bulge*(0.02) merging(0.02) in-between smooth(0.02) |
| barred spiral | unbarred loose spiral(0.01) disturbed(0.01) unbarred tight spiral(0.01) merging(0.02) cigar smooth(0.02) |
| unbarred tight spiral | merging(0.01) in-between smooth(0.01) barred spiral(0.01) round smooth(0.02) disturbed(0.02) |
| unbarred loose spiral | disturbed(0.00) barred spiral(0.01) cigar smooth(0.02) unbarred tight spiral(0.02) in-between smooth(0.02) |
| edge-on no bulge | edge-on bulge*(0.03) cigar smooth(0.05) disturbed(0.07) merging(0.08) in-between smooth(0.08) |
| edge-on bulge* | cigar smooth(0.02) edge-on no bulge(0.03) merging(0.03) disturbed(0.03) in-between smooth(0.04) |

## supcon-ft  [galaxy10_dino]

agree@1=0.900 agree@5=0.300 (chance 0.178)  dendro-purity=1.000  silhouette=0.492  within/between=0.308


| class | 5 nearest class centroids (cosine dist) |
|---|---|
| disturbed | unbarred loose spiral(0.06) merging(0.09) barred spiral(0.11) in-between smooth(0.17) cigar smooth(0.18) |
| merging | disturbed(0.09) unbarred loose spiral(0.19) in-between smooth(0.19) barred spiral(0.20) round smooth(0.24) |
| round smooth | in-between smooth(0.08) merging(0.24) disturbed(0.24) cigar smooth(0.33) unbarred loose spiral(0.38) |
| in-between smooth | round smooth(0.08) cigar smooth(0.13) disturbed(0.17) merging(0.19) unbarred loose spiral(0.34) |
| cigar smooth | in-between smooth(0.13) disturbed(0.18) edge-on bulge*(0.19) merging(0.25) unbarred loose spiral(0.33) |
| barred spiral | unbarred loose spiral(0.03) unbarred tight spiral(0.10) disturbed(0.11) merging(0.20) cigar smooth(0.34) |
| unbarred tight spiral | unbarred loose spiral(0.07) barred spiral(0.10) disturbed(0.20) merging(0.37) round smooth(0.42) |
| unbarred loose spiral | barred spiral(0.03) disturbed(0.06) unbarred tight spiral(0.07) merging(0.19) cigar smooth(0.33) |
| edge-on no bulge | edge-on bulge*(0.06) cigar smooth(0.40) disturbed(0.47) unbarred loose spiral(0.51) unbarred tight spiral(0.53) |
| edge-on bulge* | edge-on no bulge(0.06) cigar smooth(0.19) disturbed(0.27) merging(0.32) unbarred loose spiral(0.37) |

## ss-ft  [galaxy10_dino]

agree@1=0.900 agree@5=0.320 (chance 0.178)  dendro-purity=0.825  silhouette=0.561  within/between=0.257


| class | 5 nearest class centroids (cosine dist) |
|---|---|
| disturbed | merging(0.24) unbarred loose spiral(0.30) barred spiral(0.46) in-between smooth(0.64) unbarred tight spiral(0.73) |
| merging | disturbed(0.24) in-between smooth(0.35) round smooth(0.49) cigar smooth(0.54) unbarred loose spiral(0.72) |
| round smooth | in-between smooth(0.10) merging(0.49) cigar smooth(0.74) disturbed(0.79) edge-on bulge*(1.16) |
| in-between smooth | round smooth(0.10) merging(0.35) cigar smooth(0.42) disturbed(0.64) edge-on bulge*(0.98) |
| cigar smooth | edge-on bulge*(0.32) in-between smooth(0.42) merging(0.54) round smooth(0.74) disturbed(0.76) |
| barred spiral | unbarred loose spiral(0.07) unbarred tight spiral(0.22) disturbed(0.46) merging(0.76) cigar smooth(1.26) |
| unbarred tight spiral | barred spiral(0.22) unbarred loose spiral(0.23) disturbed(0.73) merging(1.15) round smooth(1.35) |
| unbarred loose spiral | barred spiral(0.07) unbarred tight spiral(0.23) disturbed(0.30) merging(0.72) cigar smooth(1.27) |
| edge-on no bulge | edge-on bulge*(0.18) cigar smooth(0.78) merging(1.39) in-between smooth(1.40) round smooth(1.41) |
| edge-on bulge* | edge-on no bulge(0.18) cigar smooth(0.32) merging(0.90) in-between smooth(0.98) disturbed(1.12) |

## nplm-sup-ft  [galaxy10_dino]

agree@1=0.700 agree@5=0.320 (chance 0.178)  dendro-purity=0.825  silhouette=0.471  within/between=0.127


| class | 5 nearest class centroids (cosine dist) |
|---|---|
| disturbed | cigar smooth(0.25) merging(0.26) in-between smooth(0.49) round smooth(0.55) edge-on bulge*(0.56) |
| merging | in-between smooth(0.06) round smooth(0.09) cigar smooth(0.23) disturbed(0.26) edge-on bulge*(0.90) |
| round smooth | in-between smooth(0.00) merging(0.09) cigar smooth(0.50) disturbed(0.55) edge-on bulge*(1.27) |
| in-between smooth | round smooth(0.00) merging(0.06) cigar smooth(0.44) disturbed(0.49) edge-on bulge*(1.19) |
| cigar smooth | merging(0.23) disturbed(0.25) edge-on bulge*(0.30) in-between smooth(0.44) round smooth(0.50) |
| barred spiral | unbarred tight spiral(0.00) unbarred loose spiral(0.01) edge-on no bulge(1.13) edge-on bulge*(1.38) disturbed(1.61) |
| unbarred tight spiral | barred spiral(0.00) unbarred loose spiral(0.01) edge-on no bulge(1.15) edge-on bulge*(1.40) disturbed(1.64) |
| unbarred loose spiral | barred spiral(0.01) unbarred tight spiral(0.01) edge-on no bulge(1.02) edge-on bulge*(1.27) disturbed(1.54) |
| edge-on no bulge | edge-on bulge*(0.03) cigar smooth(0.50) disturbed(0.73) unbarred loose spiral(1.02) barred spiral(1.13) |
| edge-on bulge* | edge-on no bulge(0.03) cigar smooth(0.30) disturbed(0.56) merging(0.90) in-between smooth(1.19) |

## supcon-ft-res (residual)  [galaxy10_dino]

agree@1=0.800 agree@5=0.280 (chance 0.178)  dendro-purity=0.825  silhouette=0.330  within/between=0.398


| class | 5 nearest class centroids (cosine dist) |
|---|---|
| disturbed | unbarred loose spiral(0.06) merging(0.12) in-between smooth(0.13) barred spiral(0.13) cigar smooth(0.17) |
| merging | disturbed(0.12) in-between smooth(0.13) round smooth(0.15) barred spiral(0.18) cigar smooth(0.19) |
| round smooth | in-between smooth(0.06) merging(0.15) disturbed(0.18) unbarred loose spiral(0.26) barred spiral(0.26) |
| in-between smooth | round smooth(0.06) merging(0.13) disturbed(0.13) cigar smooth(0.15) barred spiral(0.24) |
| cigar smooth | edge-on bulge*(0.09) in-between smooth(0.15) disturbed(0.17) merging(0.19) edge-on no bulge(0.25) |
| barred spiral | unbarred loose spiral(0.06) unbarred tight spiral(0.10) disturbed(0.13) merging(0.18) in-between smooth(0.24) |
| unbarred tight spiral | unbarred loose spiral(0.09) barred spiral(0.10) disturbed(0.20) round smooth(0.29) in-between smooth(0.35) |
| unbarred loose spiral | barred spiral(0.06) disturbed(0.06) unbarred tight spiral(0.09) merging(0.19) in-between smooth(0.25) |
| edge-on no bulge | edge-on bulge*(0.07) cigar smooth(0.25) disturbed(0.41) unbarred loose spiral(0.45) merging(0.45) |
| edge-on bulge* | edge-on no bulge(0.07) cigar smooth(0.09) merging(0.25) disturbed(0.28) in-between smooth(0.32) |

## supcon-ft-res (concat)  [galaxy10_dino]

agree@1=0.800 agree@5=0.320 (chance 0.178)  dendro-purity=0.900  silhouette=0.423  within/between=0.346


| class | 5 nearest class centroids (cosine dist) |
|---|---|
| disturbed | unbarred loose spiral(0.06) merging(0.11) barred spiral(0.13) in-between smooth(0.15) cigar smooth(0.18) |
| merging | disturbed(0.11) in-between smooth(0.16) round smooth(0.19) barred spiral(0.19) unbarred loose spiral(0.20) |
| round smooth | in-between smooth(0.07) merging(0.19) disturbed(0.21) cigar smooth(0.31) unbarred loose spiral(0.32) |
| in-between smooth | round smooth(0.07) cigar smooth(0.14) disturbed(0.15) merging(0.16) unbarred loose spiral(0.30) |
| cigar smooth | edge-on bulge*(0.14) in-between smooth(0.14) disturbed(0.18) merging(0.22) round smooth(0.31) |
| barred spiral | unbarred loose spiral(0.05) unbarred tight spiral(0.10) disturbed(0.13) merging(0.19) in-between smooth(0.30) |
| unbarred tight spiral | unbarred loose spiral(0.08) barred spiral(0.10) disturbed(0.20) merging(0.36) round smooth(0.36) |
| unbarred loose spiral | barred spiral(0.05) disturbed(0.06) unbarred tight spiral(0.08) merging(0.20) in-between smooth(0.30) |
| edge-on no bulge | edge-on bulge*(0.07) cigar smooth(0.33) disturbed(0.45) unbarred loose spiral(0.48) merging(0.52) |
| edge-on bulge* | edge-on no bulge(0.07) cigar smooth(0.14) disturbed(0.27) merging(0.28) unbarred loose spiral(0.36) |

## ss-ft-res (residual)  [galaxy10_dino]

agree@1=0.700 agree@5=0.300 (chance 0.178)  dendro-purity=0.736  silhouette=0.306  within/between=0.442


| class | 5 nearest class centroids (cosine dist) |
|---|---|
| disturbed | unbarred loose spiral(0.29) merging(0.41) barred spiral(0.56) in-between smooth(0.59) unbarred tight spiral(0.65) |
| merging | in-between smooth(0.38) round smooth(0.40) disturbed(0.41) cigar smooth(0.61) barred spiral(0.65) |
| round smooth | in-between smooth(0.14) merging(0.40) disturbed(0.68) cigar smooth(0.98) unbarred tight spiral(1.02) |
| in-between smooth | round smooth(0.14) merging(0.38) disturbed(0.59) cigar smooth(0.67) edge-on bulge*(0.92) |
| cigar smooth | edge-on bulge*(0.22) merging(0.61) edge-on no bulge(0.66) in-between smooth(0.67) disturbed(0.79) |
| barred spiral | unbarred loose spiral(0.22) unbarred tight spiral(0.34) disturbed(0.56) merging(0.65) in-between smooth(0.94) |
| unbarred tight spiral | unbarred loose spiral(0.33) barred spiral(0.34) disturbed(0.65) round smooth(1.02) merging(1.05) |
| unbarred loose spiral | barred spiral(0.22) disturbed(0.29) unbarred tight spiral(0.33) merging(0.73) in-between smooth(1.01) |
| edge-on no bulge | edge-on bulge*(0.22) cigar smooth(0.66) merging(1.04) disturbed(1.18) round smooth(1.31) |
| edge-on bulge* | edge-on no bulge(0.22) cigar smooth(0.22) merging(0.66) in-between smooth(0.92) disturbed(1.00) |

## ss-ft-res (concat)  [galaxy10_dino]

agree@1=0.800 agree@5=0.300 (chance 0.178)  dendro-purity=0.825  silhouette=0.423  within/between=0.348


| class | 5 nearest class centroids (cosine dist) |
|---|---|
| disturbed | unbarred loose spiral(0.30) merging(0.35) barred spiral(0.52) in-between smooth(0.62) unbarred tight spiral(0.69) |
| merging | disturbed(0.35) in-between smooth(0.37) round smooth(0.45) cigar smooth(0.58) barred spiral(0.70) |
| round smooth | in-between smooth(0.12) merging(0.45) disturbed(0.73) cigar smooth(0.87) edge-on bulge*(1.10) |
| in-between smooth | round smooth(0.12) merging(0.37) cigar smooth(0.55) disturbed(0.62) edge-on bulge*(0.95) |
| cigar smooth | edge-on bulge*(0.26) in-between smooth(0.55) merging(0.58) edge-on no bulge(0.72) disturbed(0.78) |
| barred spiral | unbarred loose spiral(0.14) unbarred tight spiral(0.27) disturbed(0.52) merging(0.70) in-between smooth(1.12) |
| unbarred tight spiral | barred spiral(0.27) unbarred loose spiral(0.28) disturbed(0.69) merging(1.10) round smooth(1.20) |
| unbarred loose spiral | barred spiral(0.14) unbarred tight spiral(0.28) disturbed(0.30) merging(0.73) in-between smooth(1.14) |
| edge-on no bulge | edge-on bulge*(0.21) cigar smooth(0.72) merging(1.22) disturbed(1.31) round smooth(1.37) |
| edge-on bulge* | edge-on no bulge(0.21) cigar smooth(0.26) merging(0.77) in-between smooth(0.95) disturbed(1.05) |

## supcon-ft-resnplm (residual)  [galaxy10_dino]

agree@1=0.900 agree@5=0.320 (chance 0.178)  dendro-purity=0.900  silhouette=0.483  within/between=0.182


| class | 5 nearest class centroids (cosine dist) |
|---|---|
| disturbed | unbarred loose spiral(0.02) merging(0.02) barred spiral(0.03) in-between smooth(0.05) cigar smooth(0.06) |
| merging | disturbed(0.02) in-between smooth(0.04) round smooth(0.05) unbarred loose spiral(0.06) cigar smooth(0.07) |
| round smooth | in-between smooth(0.02) merging(0.05) disturbed(0.07) cigar smooth(0.10) unbarred loose spiral(0.13) |
| in-between smooth | round smooth(0.02) merging(0.04) disturbed(0.05) cigar smooth(0.05) unbarred loose spiral(0.10) |
| cigar smooth | in-between smooth(0.05) disturbed(0.06) merging(0.07) round smooth(0.10) edge-on bulge*(0.12) |
| barred spiral | unbarred loose spiral(0.01) unbarred tight spiral(0.02) disturbed(0.03) merging(0.07) in-between smooth(0.12) |
| unbarred tight spiral | unbarred loose spiral(0.02) barred spiral(0.02) disturbed(0.06) merging(0.11) in-between smooth(0.15) |
| unbarred loose spiral | barred spiral(0.01) unbarred tight spiral(0.02) disturbed(0.02) merging(0.06) in-between smooth(0.10) |
| edge-on no bulge | edge-on bulge*(0.06) cigar smooth(0.31) disturbed(0.42) unbarred loose spiral(0.46) merging(0.46) |
| edge-on bulge* | edge-on no bulge(0.06) cigar smooth(0.12) disturbed(0.20) merging(0.22) unbarred loose spiral(0.25) |

## supcon-ft-resnplm (concat)  [galaxy10_dino]

agree@1=0.900 agree@5=0.320 (chance 0.178)  dendro-purity=1.000  silhouette=0.490  within/between=0.261


| class | 5 nearest class centroids (cosine dist) |
|---|---|
| disturbed | unbarred loose spiral(0.04) merging(0.06) barred spiral(0.08) in-between smooth(0.10) cigar smooth(0.11) |
| merging | disturbed(0.06) in-between smooth(0.12) unbarred loose spiral(0.13) barred spiral(0.14) round smooth(0.14) |
| round smooth | in-between smooth(0.05) merging(0.14) disturbed(0.15) cigar smooth(0.21) unbarred loose spiral(0.25) |
| in-between smooth | round smooth(0.05) cigar smooth(0.09) disturbed(0.10) merging(0.12) unbarred loose spiral(0.22) |
| cigar smooth | in-between smooth(0.09) disturbed(0.11) edge-on bulge*(0.15) merging(0.16) round smooth(0.21) |
| barred spiral | unbarred loose spiral(0.02) unbarred tight spiral(0.07) disturbed(0.08) merging(0.14) in-between smooth(0.24) |
| unbarred tight spiral | unbarred loose spiral(0.05) barred spiral(0.07) disturbed(0.13) merging(0.25) round smooth(0.30) |
| unbarred loose spiral | barred spiral(0.02) disturbed(0.04) unbarred tight spiral(0.05) merging(0.13) in-between smooth(0.22) |
| edge-on no bulge | edge-on bulge*(0.06) cigar smooth(0.35) disturbed(0.45) unbarred loose spiral(0.48) unbarred tight spiral(0.50) |
| edge-on bulge* | edge-on no bulge(0.06) cigar smooth(0.15) disturbed(0.23) merging(0.27) unbarred loose spiral(0.31) |
