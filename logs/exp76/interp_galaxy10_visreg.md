# exp76 interpretability [galaxy10_visreg]
holdout classes starred; superclass partition: yes

## frozen-visreg (768d)  [galaxy10_visreg]

agree@1=0.600 agree@5=0.300 (chance 0.178)  dendro-purity=0.509  silhouette=0.069  within/between=0.598


| class | 5 nearest class centroids (cosine dist) |
|---|---|
| disturbed | merging(0.00) unbarred loose spiral(0.00) barred spiral(0.01) unbarred tight spiral(0.01) in-between smooth(0.01) |
| merging | disturbed(0.00) in-between smooth(0.00) unbarred tight spiral(0.00) cigar smooth(0.01) unbarred loose spiral(0.01) |
| round smooth | in-between smooth(0.01) merging(0.01) unbarred tight spiral(0.02) disturbed(0.02) cigar smooth(0.03) |
| in-between smooth | merging(0.00) disturbed(0.01) round smooth(0.01) cigar smooth(0.01) unbarred tight spiral(0.01) |
| cigar smooth | edge-on bulge*(0.00) disturbed(0.01) unbarred loose spiral(0.01) merging(0.01) in-between smooth(0.01) |
| barred spiral | unbarred loose spiral(0.00) disturbed(0.01) unbarred tight spiral(0.01) cigar smooth(0.01) merging(0.01) |
| unbarred tight spiral | merging(0.00) disturbed(0.01) barred spiral(0.01) unbarred loose spiral(0.01) in-between smooth(0.01) |
| unbarred loose spiral | barred spiral(0.00) disturbed(0.00) unbarred tight spiral(0.01) cigar smooth(0.01) merging(0.01) |
| edge-on no bulge | edge-on bulge*(0.01) cigar smooth(0.02) merging(0.02) disturbed(0.02) unbarred tight spiral(0.03) |
| edge-on bulge* | cigar smooth(0.00) edge-on no bulge(0.01) disturbed(0.01) merging(0.01) unbarred loose spiral(0.02) |

## simclr-ft  [galaxy10_visreg]

agree@1=0.400 agree@5=0.300 (chance 0.178)  dendro-purity=0.629  silhouette=0.190  within/between=0.682


| class | 5 nearest class centroids (cosine dist) |
|---|---|
| disturbed | unbarred loose spiral(0.50) merging(0.89) cigar smooth(0.92) in-between smooth(1.04) edge-on bulge*(1.05) |
| merging | edge-on bulge*(0.73) round smooth(0.84) cigar smooth(0.86) disturbed(0.89) in-between smooth(0.96) |
| round smooth | merging(0.84) in-between smooth(0.86) edge-on no bulge(1.08) edge-on bulge*(1.14) cigar smooth(1.15) |
| in-between smooth | cigar smooth(0.68) round smooth(0.86) merging(0.96) edge-on bulge*(1.02) disturbed(1.04) |
| cigar smooth | edge-on bulge*(0.41) in-between smooth(0.68) merging(0.86) disturbed(0.92) edge-on no bulge(0.95) |
| barred spiral | unbarred tight spiral(0.31) unbarred loose spiral(0.80) edge-on no bulge(1.16) merging(1.17) edge-on bulge*(1.19) |
| unbarred tight spiral | barred spiral(0.31) unbarred loose spiral(0.75) edge-on no bulge(1.01) edge-on bulge*(1.15) round smooth(1.15) |
| unbarred loose spiral | disturbed(0.50) unbarred tight spiral(0.75) barred spiral(0.80) edge-on bulge*(1.17) edge-on no bulge(1.17) |
| edge-on no bulge | edge-on bulge*(0.60) cigar smooth(0.95) unbarred tight spiral(1.01) round smooth(1.08) disturbed(1.11) |
| edge-on bulge* | cigar smooth(0.41) edge-on no bulge(0.60) merging(0.73) in-between smooth(1.02) disturbed(1.05) |

## sigreg-ssl-ft  [galaxy10_visreg]

agree@1=0.600 agree@5=0.260 (chance 0.178)  dendro-purity=0.713  silhouette=0.331  within/between=0.449


| class | 5 nearest class centroids (cosine dist) |
|---|---|
| disturbed | unbarred loose spiral(0.35) cigar smooth(0.97) unbarred tight spiral(1.02) barred spiral(1.15) edge-on no bulge(1.16) |
| merging | round smooth(0.52) in-between smooth(0.73) edge-on bulge*(0.81) cigar smooth(0.86) edge-on no bulge(0.88) |
| round smooth | in-between smooth(0.31) merging(0.52) cigar smooth(1.09) edge-on bulge*(1.16) edge-on no bulge(1.31) |
| in-between smooth | round smooth(0.31) cigar smooth(0.69) merging(0.73) edge-on bulge*(0.96) edge-on no bulge(1.30) |
| cigar smooth | edge-on bulge*(0.21) edge-on no bulge(0.53) in-between smooth(0.69) merging(0.86) disturbed(0.97) |
| barred spiral | unbarred tight spiral(0.05) unbarred loose spiral(0.48) disturbed(1.15) edge-on no bulge(1.31) merging(1.40) |
| unbarred tight spiral | barred spiral(0.05) unbarred loose spiral(0.38) disturbed(1.02) edge-on no bulge(1.31) merging(1.39) |
| unbarred loose spiral | disturbed(0.35) unbarred tight spiral(0.38) barred spiral(0.48) edge-on no bulge(1.22) cigar smooth(1.50) |
| edge-on no bulge | edge-on bulge*(0.15) cigar smooth(0.53) merging(0.88) disturbed(1.16) unbarred loose spiral(1.22) |
| edge-on bulge* | edge-on no bulge(0.15) cigar smooth(0.21) merging(0.81) in-between smooth(0.96) round smooth(1.16) |

## nplm-bil-ft  [galaxy10_visreg]

agree@1=0.600 agree@5=0.320 (chance 0.178)  dendro-purity=0.721  silhouette=0.260  within/between=0.348


| class | 5 nearest class centroids (cosine dist) |
|---|---|
| disturbed | in-between smooth(0.18) merging(0.32) round smooth(0.36) cigar smooth(0.36) unbarred loose spiral(0.74) |
| merging | round smooth(0.07) in-between smooth(0.23) disturbed(0.32) unbarred tight spiral(0.68) unbarred loose spiral(0.82) |
| round smooth | merging(0.07) in-between smooth(0.10) disturbed(0.36) unbarred tight spiral(0.93) cigar smooth(1.00) |
| in-between smooth | round smooth(0.10) disturbed(0.18) merging(0.23) cigar smooth(0.57) unbarred loose spiral(1.25) |
| cigar smooth | disturbed(0.36) edge-on bulge*(0.38) in-between smooth(0.57) edge-on no bulge(0.85) round smooth(1.00) |
| barred spiral | unbarred loose spiral(0.14) unbarred tight spiral(0.26) edge-on no bulge(0.49) edge-on bulge*(0.72) merging(1.10) |
| unbarred tight spiral | barred spiral(0.26) unbarred loose spiral(0.40) merging(0.68) round smooth(0.93) edge-on no bulge(1.06) |
| unbarred loose spiral | barred spiral(0.14) unbarred tight spiral(0.40) edge-on no bulge(0.67) edge-on bulge*(0.73) disturbed(0.74) |
| edge-on no bulge | edge-on bulge*(0.15) barred spiral(0.49) unbarred loose spiral(0.67) cigar smooth(0.85) unbarred tight spiral(1.06) |
| edge-on bulge* | edge-on no bulge(0.15) cigar smooth(0.38) barred spiral(0.72) unbarred loose spiral(0.73) disturbed(1.03) |

## supcon-ft  [galaxy10_visreg]

agree@1=0.700 agree@5=0.280 (chance 0.178)  dendro-purity=0.659  silhouette=0.253  within/between=0.610


| class | 5 nearest class centroids (cosine dist) |
|---|---|
| disturbed | unbarred loose spiral(0.11) in-between smooth(0.29) cigar smooth(0.33) barred spiral(0.34) merging(0.43) |
| merging | disturbed(0.43) edge-on bulge*(0.55) barred spiral(0.63) cigar smooth(0.63) unbarred loose spiral(0.63) |
| round smooth | disturbed(0.44) unbarred tight spiral(0.46) unbarred loose spiral(0.54) in-between smooth(0.63) merging(0.64) |
| in-between smooth | cigar smooth(0.26) disturbed(0.29) unbarred loose spiral(0.59) barred spiral(0.61) round smooth(0.63) |
| cigar smooth | in-between smooth(0.26) edge-on bulge*(0.30) disturbed(0.33) edge-on no bulge(0.49) unbarred loose spiral(0.55) |
| barred spiral | unbarred loose spiral(0.20) disturbed(0.34) unbarred tight spiral(0.46) in-between smooth(0.61) merging(0.63) |
| unbarred tight spiral | unbarred loose spiral(0.25) disturbed(0.45) round smooth(0.46) barred spiral(0.46) edge-on bulge*(0.65) |
| unbarred loose spiral | disturbed(0.11) barred spiral(0.20) unbarred tight spiral(0.25) edge-on bulge*(0.52) round smooth(0.54) |
| edge-on no bulge | edge-on bulge*(0.05) cigar smooth(0.49) merging(0.72) unbarred loose spiral(0.74) disturbed(0.79) |
| edge-on bulge* | edge-on no bulge(0.05) cigar smooth(0.30) disturbed(0.51) unbarred loose spiral(0.52) merging(0.55) |

## ss-ft  [galaxy10_visreg]

agree@1=0.700 agree@5=0.280 (chance 0.178)  dendro-purity=0.762  silhouette=0.302  within/between=0.588


| class | 5 nearest class centroids (cosine dist) |
|---|---|
| disturbed | unbarred loose spiral(0.44) in-between smooth(0.61) cigar smooth(0.68) merging(0.73) barred spiral(0.82) |
| merging | disturbed(0.73) edge-on bulge*(0.93) in-between smooth(0.94) cigar smooth(1.02) barred spiral(1.06) |
| round smooth | unbarred tight spiral(1.00) edge-on no bulge(1.02) in-between smooth(1.05) merging(1.20) edge-on bulge*(1.29) |
| in-between smooth | cigar smooth(0.42) disturbed(0.61) merging(0.94) edge-on bulge*(1.03) round smooth(1.05) |
| cigar smooth | edge-on bulge*(0.30) in-between smooth(0.42) edge-on no bulge(0.60) disturbed(0.68) merging(1.02) |
| barred spiral | unbarred loose spiral(0.40) unbarred tight spiral(0.72) disturbed(0.82) merging(1.06) edge-on bulge*(1.10) |
| unbarred tight spiral | unbarred loose spiral(0.47) barred spiral(0.72) round smooth(1.00) disturbed(1.17) edge-on no bulge(1.20) |
| unbarred loose spiral | barred spiral(0.40) disturbed(0.44) unbarred tight spiral(0.47) cigar smooth(1.15) merging(1.19) |
| edge-on no bulge | edge-on bulge*(0.11) cigar smooth(0.60) round smooth(1.02) merging(1.10) barred spiral(1.13) |
| edge-on bulge* | edge-on no bulge(0.11) cigar smooth(0.30) merging(0.93) in-between smooth(1.03) barred spiral(1.10) |

## nplm-sup-ft  [galaxy10_visreg]

agree@1=0.700 agree@5=0.280 (chance 0.178)  dendro-purity=0.775  silhouette=0.430  within/between=0.316


| class | 5 nearest class centroids (cosine dist) |
|---|---|
| disturbed | unbarred loose spiral(0.19) barred spiral(0.21) unbarred tight spiral(0.27) edge-on no bulge(1.09) edge-on bulge*(1.18) |
| merging | cigar smooth(0.13) edge-on bulge*(0.24) edge-on no bulge(0.34) in-between smooth(1.15) round smooth(1.23) |
| round smooth | in-between smooth(0.00) cigar smooth(0.82) unbarred tight spiral(1.16) merging(1.23) disturbed(1.39) |
| in-between smooth | round smooth(0.00) cigar smooth(0.73) merging(1.15) unbarred tight spiral(1.25) disturbed(1.45) |
| cigar smooth | merging(0.13) edge-on bulge*(0.49) edge-on no bulge(0.61) in-between smooth(0.73) round smooth(0.82) |
| barred spiral | unbarred loose spiral(0.00) unbarred tight spiral(0.06) disturbed(0.21) edge-on no bulge(1.08) edge-on bulge*(1.21) |
| unbarred tight spiral | barred spiral(0.06) unbarred loose spiral(0.08) disturbed(0.27) round smooth(1.16) in-between smooth(1.25) |
| unbarred loose spiral | barred spiral(0.00) unbarred tight spiral(0.08) disturbed(0.19) edge-on no bulge(1.02) edge-on bulge*(1.15) |
| edge-on no bulge | edge-on bulge*(0.01) merging(0.34) cigar smooth(0.61) unbarred loose spiral(1.02) barred spiral(1.08) |
| edge-on bulge* | edge-on no bulge(0.01) merging(0.24) cigar smooth(0.49) unbarred loose spiral(1.15) disturbed(1.18) |

## supcon-ft-res (residual)  [galaxy10_visreg]

agree@1=0.800 agree@5=0.280 (chance 0.178)  dendro-purity=0.701  silhouette=0.360  within/between=0.493


| class | 5 nearest class centroids (cosine dist) |
|---|---|
| disturbed | unbarred loose spiral(0.11) barred spiral(0.23) unbarred tight spiral(0.27) merging(0.27) cigar smooth(0.27) |
| merging | disturbed(0.27) edge-on bulge*(0.48) cigar smooth(0.49) unbarred loose spiral(0.50) in-between smooth(0.51) |
| round smooth | unbarred tight spiral(0.36) disturbed(0.36) unbarred loose spiral(0.45) in-between smooth(0.49) merging(0.57) |
| in-between smooth | cigar smooth(0.22) disturbed(0.27) unbarred loose spiral(0.48) round smooth(0.49) barred spiral(0.49) |
| cigar smooth | in-between smooth(0.22) disturbed(0.27) edge-on bulge*(0.29) unbarred loose spiral(0.45) merging(0.49) |
| barred spiral | unbarred loose spiral(0.09) unbarred tight spiral(0.22) disturbed(0.23) in-between smooth(0.49) cigar smooth(0.51) |
| unbarred tight spiral | unbarred loose spiral(0.12) barred spiral(0.22) disturbed(0.27) round smooth(0.36) cigar smooth(0.54) |
| unbarred loose spiral | barred spiral(0.09) disturbed(0.11) unbarred tight spiral(0.12) cigar smooth(0.45) round smooth(0.45) |
| edge-on no bulge | edge-on bulge*(0.07) cigar smooth(0.53) merging(0.71) disturbed(0.71) unbarred loose spiral(0.73) |
| edge-on bulge* | edge-on no bulge(0.07) cigar smooth(0.29) disturbed(0.47) merging(0.48) unbarred loose spiral(0.55) |

## supcon-ft-res (concat)  [galaxy10_visreg]

agree@1=0.700 agree@5=0.280 (chance 0.178)  dendro-purity=0.659  silhouette=0.298  within/between=0.563


| class | 5 nearest class centroids (cosine dist) |
|---|---|
| disturbed | unbarred loose spiral(0.11) in-between smooth(0.28) barred spiral(0.29) cigar smooth(0.30) merging(0.36) |
| merging | disturbed(0.36) edge-on bulge*(0.52) cigar smooth(0.56) unbarred loose spiral(0.57) barred spiral(0.58) |
| round smooth | disturbed(0.41) unbarred tight spiral(0.41) unbarred loose spiral(0.50) in-between smooth(0.56) merging(0.61) |
| in-between smooth | cigar smooth(0.25) disturbed(0.28) unbarred loose spiral(0.54) barred spiral(0.56) round smooth(0.56) |
| cigar smooth | in-between smooth(0.25) edge-on bulge*(0.30) disturbed(0.30) unbarred loose spiral(0.51) edge-on no bulge(0.51) |
| barred spiral | unbarred loose spiral(0.16) disturbed(0.29) unbarred tight spiral(0.34) in-between smooth(0.56) merging(0.58) |
| unbarred tight spiral | unbarred loose spiral(0.20) barred spiral(0.34) disturbed(0.37) round smooth(0.41) edge-on bulge*(0.63) |
| unbarred loose spiral | disturbed(0.11) barred spiral(0.16) unbarred tight spiral(0.20) round smooth(0.50) cigar smooth(0.51) |
| edge-on no bulge | edge-on bulge*(0.06) cigar smooth(0.51) merging(0.72) unbarred loose spiral(0.74) disturbed(0.76) |
| edge-on bulge* | edge-on no bulge(0.06) cigar smooth(0.30) disturbed(0.49) merging(0.52) unbarred loose spiral(0.53) |

## ss-ft-res (residual)  [galaxy10_visreg]

agree@1=0.700 agree@5=0.280 (chance 0.178)  dendro-purity=0.815  silhouette=0.320  within/between=0.557


| class | 5 nearest class centroids (cosine dist) |
|---|---|
| disturbed | unbarred loose spiral(0.61) merging(0.74) barred spiral(0.91) in-between smooth(1.00) cigar smooth(1.04) |
| merging | disturbed(0.74) in-between smooth(0.77) cigar smooth(0.87) edge-on bulge*(0.94) round smooth(1.11) |
| round smooth | in-between smooth(0.79) unbarred tight spiral(1.03) edge-on no bulge(1.08) merging(1.11) edge-on bulge*(1.22) |
| in-between smooth | cigar smooth(0.67) merging(0.77) round smooth(0.79) edge-on bulge*(0.95) disturbed(1.00) |
| cigar smooth | edge-on bulge*(0.20) in-between smooth(0.67) edge-on no bulge(0.72) merging(0.87) disturbed(1.04) |
| barred spiral | unbarred loose spiral(0.32) unbarred tight spiral(0.58) disturbed(0.91) edge-on no bulge(1.24) merging(1.26) |
| unbarred tight spiral | unbarred loose spiral(0.52) barred spiral(0.58) round smooth(1.03) edge-on no bulge(1.09) disturbed(1.11) |
| unbarred loose spiral | barred spiral(0.32) unbarred tight spiral(0.52) disturbed(0.61) edge-on no bulge(1.18) cigar smooth(1.29) |
| edge-on no bulge | edge-on bulge*(0.27) cigar smooth(0.72) round smooth(1.08) unbarred tight spiral(1.09) merging(1.17) |
| edge-on bulge* | cigar smooth(0.20) edge-on no bulge(0.27) merging(0.94) in-between smooth(0.95) disturbed(1.21) |

## ss-ft-res (concat)  [galaxy10_visreg]

agree@1=0.800 agree@5=0.280 (chance 0.178)  dendro-purity=0.774  silhouette=0.308  within/between=0.580


| class | 5 nearest class centroids (cosine dist) |
|---|---|
| disturbed | unbarred loose spiral(0.51) merging(0.73) in-between smooth(0.75) cigar smooth(0.83) barred spiral(0.85) |
| merging | disturbed(0.73) in-between smooth(0.89) edge-on bulge*(0.93) cigar smooth(0.96) barred spiral(1.13) |
| round smooth | in-between smooth(0.97) unbarred tight spiral(1.01) edge-on no bulge(1.05) merging(1.17) edge-on bulge*(1.26) |
| in-between smooth | cigar smooth(0.51) disturbed(0.75) merging(0.89) round smooth(0.97) edge-on bulge*(1.00) |
| cigar smooth | edge-on bulge*(0.26) in-between smooth(0.51) edge-on no bulge(0.66) disturbed(0.83) merging(0.96) |
| barred spiral | unbarred loose spiral(0.38) unbarred tight spiral(0.67) disturbed(0.85) merging(1.13) edge-on no bulge(1.17) |
| unbarred tight spiral | unbarred loose spiral(0.49) barred spiral(0.67) round smooth(1.01) disturbed(1.15) edge-on no bulge(1.15) |
| unbarred loose spiral | barred spiral(0.38) unbarred tight spiral(0.49) disturbed(0.51) cigar smooth(1.21) merging(1.22) |
| edge-on no bulge | edge-on bulge*(0.18) cigar smooth(0.66) round smooth(1.05) merging(1.13) unbarred tight spiral(1.15) |
| edge-on bulge* | edge-on no bulge(0.18) cigar smooth(0.26) merging(0.93) in-between smooth(1.00) disturbed(1.16) |

## supcon-ft-resnplm (residual)  [galaxy10_visreg]

agree@1=0.700 agree@5=0.260 (chance 0.178)  dendro-purity=0.659  silhouette=0.289  within/between=0.537


| class | 5 nearest class centroids (cosine dist) |
|---|---|
| disturbed | unbarred loose spiral(0.07) barred spiral(0.22) cigar smooth(0.24) in-between smooth(0.25) unbarred tight spiral(0.27) |
| merging | disturbed(0.43) cigar smooth(0.52) edge-on bulge*(0.55) unbarred loose spiral(0.58) in-between smooth(0.60) |
| round smooth | disturbed(0.36) unbarred tight spiral(0.42) unbarred loose spiral(0.47) in-between smooth(0.57) merging(0.60) |
| in-between smooth | cigar smooth(0.18) disturbed(0.25) unbarred loose spiral(0.45) barred spiral(0.52) round smooth(0.57) |
| cigar smooth | in-between smooth(0.18) disturbed(0.24) edge-on bulge*(0.36) unbarred loose spiral(0.41) barred spiral(0.52) |
| barred spiral | unbarred loose spiral(0.12) disturbed(0.22) unbarred tight spiral(0.35) cigar smooth(0.52) in-between smooth(0.52) |
| unbarred tight spiral | unbarred loose spiral(0.15) disturbed(0.27) barred spiral(0.35) round smooth(0.42) cigar smooth(0.56) |
| unbarred loose spiral | disturbed(0.07) barred spiral(0.12) unbarred tight spiral(0.15) cigar smooth(0.41) in-between smooth(0.45) |
| edge-on no bulge | edge-on bulge*(0.04) cigar smooth(0.55) merging(0.71) unbarred loose spiral(0.79) unbarred tight spiral(0.80) |
| edge-on bulge* | edge-on no bulge(0.04) cigar smooth(0.36) merging(0.55) disturbed(0.56) unbarred loose spiral(0.57) |

## supcon-ft-resnplm (concat)  [galaxy10_visreg]

agree@1=0.700 agree@5=0.260 (chance 0.178)  dendro-purity=0.659  silhouette=0.276  within/between=0.577


| class | 5 nearest class centroids (cosine dist) |
|---|---|
| disturbed | unbarred loose spiral(0.10) in-between smooth(0.27) barred spiral(0.29) cigar smooth(0.29) unbarred tight spiral(0.36) |
| merging | disturbed(0.43) edge-on bulge*(0.55) cigar smooth(0.58) unbarred loose spiral(0.61) round smooth(0.62) |
| round smooth | disturbed(0.40) unbarred tight spiral(0.44) unbarred loose spiral(0.51) in-between smooth(0.60) merging(0.62) |
| in-between smooth | cigar smooth(0.23) disturbed(0.27) unbarred loose spiral(0.52) barred spiral(0.56) round smooth(0.60) |
| cigar smooth | in-between smooth(0.23) disturbed(0.29) edge-on bulge*(0.33) unbarred loose spiral(0.49) edge-on no bulge(0.52) |
| barred spiral | unbarred loose spiral(0.17) disturbed(0.29) unbarred tight spiral(0.40) in-between smooth(0.56) cigar smooth(0.61) |
| unbarred tight spiral | unbarred loose spiral(0.21) disturbed(0.36) barred spiral(0.40) round smooth(0.44) cigar smooth(0.64) |
| unbarred loose spiral | disturbed(0.10) barred spiral(0.17) unbarred tight spiral(0.21) cigar smooth(0.49) round smooth(0.51) |
| edge-on no bulge | edge-on bulge*(0.05) cigar smooth(0.52) merging(0.72) unbarred loose spiral(0.76) unbarred tight spiral(0.79) |
| edge-on bulge* | edge-on no bulge(0.05) cigar smooth(0.33) disturbed(0.54) unbarred loose spiral(0.55) merging(0.55) |
