# exp76 interpretability [galaxy10_lejepa]
holdout classes starred; superclass partition: yes

## frozen-lejepa (768d)  [galaxy10_lejepa]

agree@1=0.800 agree@5=0.300 (chance 0.178)  dendro-purity=0.733  silhouette=0.278  within/between=0.416


| class | 5 nearest class centroids (cosine dist) |
|---|---|
| disturbed | merging(0.01) unbarred loose spiral(0.01) cigar smooth(0.01) unbarred tight spiral(0.01) in-between smooth(0.01) |
| merging | disturbed(0.01) in-between smooth(0.01) cigar smooth(0.01) unbarred loose spiral(0.01) unbarred tight spiral(0.01) |
| round smooth | in-between smooth(0.01) merging(0.01) disturbed(0.02) cigar smooth(0.03) unbarred tight spiral(0.03) |
| in-between smooth | round smooth(0.01) merging(0.01) disturbed(0.01) cigar smooth(0.01) edge-on bulge*(0.02) |
| cigar smooth | edge-on bulge*(0.01) disturbed(0.01) merging(0.01) in-between smooth(0.01) unbarred loose spiral(0.02) |
| barred spiral | unbarred loose spiral(0.00) unbarred tight spiral(0.01) disturbed(0.01) merging(0.02) cigar smooth(0.02) |
| unbarred tight spiral | barred spiral(0.01) unbarred loose spiral(0.01) disturbed(0.01) merging(0.01) cigar smooth(0.02) |
| unbarred loose spiral | barred spiral(0.00) unbarred tight spiral(0.01) disturbed(0.01) merging(0.01) cigar smooth(0.02) |
| edge-on no bulge | edge-on bulge*(0.01) cigar smooth(0.03) disturbed(0.03) unbarred loose spiral(0.03) merging(0.03) |
| edge-on bulge* | cigar smooth(0.01) merging(0.01) edge-on no bulge(0.01) disturbed(0.02) unbarred loose spiral(0.02) |

## simclr-ft  [galaxy10_lejepa]

agree@1=0.500 agree@5=0.300 (chance 0.178)  dendro-purity=0.775  silhouette=0.195  within/between=0.697


| class | 5 nearest class centroids (cosine dist) |
|---|---|
| disturbed | unbarred loose spiral(0.54) merging(0.87) in-between smooth(1.00) edge-on no bulge(1.03) edge-on bulge*(1.03) |
| merging | round smooth(0.72) edge-on bulge*(0.79) disturbed(0.87) in-between smooth(0.93) cigar smooth(0.96) |
| round smooth | merging(0.72) in-between smooth(0.95) edge-on bulge*(1.09) unbarred tight spiral(1.11) cigar smooth(1.12) |
| in-between smooth | cigar smooth(0.75) merging(0.93) round smooth(0.95) disturbed(1.00) edge-on bulge*(1.15) |
| cigar smooth | edge-on bulge*(0.71) in-between smooth(0.75) edge-on no bulge(0.89) merging(0.96) disturbed(1.05) |
| barred spiral | unbarred tight spiral(0.41) unbarred loose spiral(0.82) merging(1.08) edge-on no bulge(1.16) edge-on bulge*(1.21) |
| unbarred tight spiral | barred spiral(0.41) unbarred loose spiral(0.70) edge-on no bulge(1.01) edge-on bulge*(1.04) disturbed(1.08) |
| unbarred loose spiral | disturbed(0.54) unbarred tight spiral(0.70) barred spiral(0.82) edge-on no bulge(1.10) edge-on bulge*(1.11) |
| edge-on no bulge | edge-on bulge*(0.53) cigar smooth(0.89) unbarred tight spiral(1.01) disturbed(1.03) unbarred loose spiral(1.10) |
| edge-on bulge* | edge-on no bulge(0.53) cigar smooth(0.71) merging(0.79) disturbed(1.03) unbarred tight spiral(1.04) |

## sigreg-ssl-ft  [galaxy10_lejepa]

agree@1=0.600 agree@5=0.260 (chance 0.178)  dendro-purity=0.726  silhouette=0.313  within/between=0.479


| class | 5 nearest class centroids (cosine dist) |
|---|---|
| disturbed | unbarred loose spiral(0.45) cigar smooth(0.76) edge-on no bulge(0.93) in-between smooth(1.07) edge-on bulge*(1.19) |
| merging | round smooth(0.39) edge-on bulge*(0.59) edge-on no bulge(0.81) cigar smooth(0.83) in-between smooth(0.92) |
| round smooth | merging(0.39) in-between smooth(0.41) cigar smooth(1.11) edge-on bulge*(1.18) disturbed(1.29) |
| in-between smooth | round smooth(0.41) cigar smooth(0.61) merging(0.92) edge-on bulge*(0.97) disturbed(1.07) |
| cigar smooth | edge-on bulge*(0.26) edge-on no bulge(0.48) in-between smooth(0.61) disturbed(0.76) merging(0.83) |
| barred spiral | unbarred tight spiral(0.07) unbarred loose spiral(0.54) edge-on no bulge(1.26) edge-on bulge*(1.32) merging(1.35) |
| unbarred tight spiral | barred spiral(0.07) unbarred loose spiral(0.35) disturbed(1.20) edge-on no bulge(1.32) merging(1.37) |
| unbarred loose spiral | unbarred tight spiral(0.35) disturbed(0.45) barred spiral(0.54) edge-on no bulge(1.19) cigar smooth(1.47) |
| edge-on no bulge | edge-on bulge*(0.15) cigar smooth(0.48) merging(0.81) disturbed(0.93) unbarred loose spiral(1.19) |
| edge-on bulge* | edge-on no bulge(0.15) cigar smooth(0.26) merging(0.59) in-between smooth(0.97) round smooth(1.18) |

## nplm-bil-ft  [galaxy10_lejepa]

agree@1=0.700 agree@5=0.280 (chance 0.178)  dendro-purity=0.690  silhouette=0.227  within/between=0.600


| class | 5 nearest class centroids (cosine dist) |
|---|---|
| disturbed | edge-on no bulge(0.70) unbarred loose spiral(0.73) merging(0.76) edge-on bulge*(0.78) cigar smooth(0.79) |
| merging | edge-on bulge*(0.65) in-between smooth(0.69) disturbed(0.76) cigar smooth(0.76) edge-on no bulge(0.91) |
| round smooth | in-between smooth(0.81) merging(0.99) disturbed(1.08) unbarred tight spiral(1.10) edge-on no bulge(1.17) |
| in-between smooth | cigar smooth(0.52) merging(0.69) round smooth(0.81) edge-on bulge*(0.90) disturbed(0.90) |
| cigar smooth | edge-on bulge*(0.42) in-between smooth(0.52) edge-on no bulge(0.61) merging(0.76) disturbed(0.79) |
| barred spiral | unbarred tight spiral(0.44) unbarred loose spiral(0.54) merging(1.21) edge-on bulge*(1.26) edge-on no bulge(1.35) |
| unbarred tight spiral | barred spiral(0.44) unbarred loose spiral(0.58) round smooth(1.10) edge-on no bulge(1.18) edge-on bulge*(1.30) |
| unbarred loose spiral | barred spiral(0.54) unbarred tight spiral(0.58) disturbed(0.73) edge-on no bulge(1.19) merging(1.22) |
| edge-on no bulge | edge-on bulge*(0.39) cigar smooth(0.61) disturbed(0.70) merging(0.91) in-between smooth(1.13) |
| edge-on bulge* | edge-on no bulge(0.39) cigar smooth(0.42) merging(0.65) disturbed(0.78) in-between smooth(0.90) |

## supcon-ft  [galaxy10_lejepa]

agree@1=0.500 agree@5=0.260 (chance 0.178)  dendro-purity=0.659  silhouette=0.241  within/between=0.658


| class | 5 nearest class centroids (cosine dist) |
|---|---|
| disturbed | unbarred loose spiral(0.17) in-between smooth(0.28) cigar smooth(0.34) merging(0.40) barred spiral(0.40) |
| merging | disturbed(0.40) edge-on bulge*(0.60) in-between smooth(0.62) round smooth(0.64) cigar smooth(0.65) |
| round smooth | disturbed(0.42) unbarred tight spiral(0.48) unbarred loose spiral(0.61) in-between smooth(0.64) merging(0.64) |
| in-between smooth | disturbed(0.28) cigar smooth(0.31) unbarred loose spiral(0.58) merging(0.62) barred spiral(0.63) |
| cigar smooth | edge-on bulge*(0.26) in-between smooth(0.31) disturbed(0.34) edge-on no bulge(0.44) unbarred loose spiral(0.55) |
| barred spiral | unbarred loose spiral(0.33) disturbed(0.40) unbarred tight spiral(0.48) in-between smooth(0.63) merging(0.68) |
| unbarred tight spiral | unbarred loose spiral(0.26) disturbed(0.46) round smooth(0.48) barred spiral(0.48) edge-on bulge*(0.57) |
| unbarred loose spiral | disturbed(0.17) unbarred tight spiral(0.26) barred spiral(0.33) edge-on bulge*(0.45) cigar smooth(0.55) |
| edge-on no bulge | edge-on bulge*(0.06) cigar smooth(0.44) unbarred loose spiral(0.63) unbarred tight spiral(0.66) disturbed(0.69) |
| edge-on bulge* | edge-on no bulge(0.06) cigar smooth(0.26) disturbed(0.43) unbarred loose spiral(0.45) unbarred tight spiral(0.57) |

## ss-ft  [galaxy10_lejepa]

agree@1=0.700 agree@5=0.300 (chance 0.178)  dendro-purity=0.603  silhouette=0.289  within/between=0.560


| class | 5 nearest class centroids (cosine dist) |
|---|---|
| disturbed | unbarred loose spiral(0.34) barred spiral(0.53) merging(0.67) in-between smooth(0.67) cigar smooth(0.84) |
| merging | disturbed(0.67) edge-on bulge*(0.73) barred spiral(0.90) in-between smooth(0.99) cigar smooth(1.02) |
| round smooth | unbarred tight spiral(0.96) edge-on no bulge(1.08) in-between smooth(1.11) merging(1.22) cigar smooth(1.40) |
| in-between smooth | cigar smooth(0.42) disturbed(0.67) edge-on bulge*(0.99) merging(0.99) round smooth(1.11) |
| cigar smooth | edge-on bulge*(0.25) in-between smooth(0.42) edge-on no bulge(0.49) disturbed(0.84) merging(1.02) |
| barred spiral | unbarred loose spiral(0.26) disturbed(0.53) unbarred tight spiral(0.61) merging(0.90) edge-on bulge*(1.15) |
| unbarred tight spiral | unbarred loose spiral(0.39) barred spiral(0.61) round smooth(0.96) disturbed(1.09) edge-on no bulge(1.20) |
| unbarred loose spiral | barred spiral(0.26) disturbed(0.34) unbarred tight spiral(0.39) merging(1.06) edge-on bulge*(1.22) |
| edge-on no bulge | edge-on bulge*(0.17) cigar smooth(0.49) round smooth(1.08) merging(1.11) unbarred tight spiral(1.20) |
| edge-on bulge* | edge-on no bulge(0.17) cigar smooth(0.25) merging(0.73) disturbed(0.98) in-between smooth(0.99) |

## nplm-sup-ft  [galaxy10_lejepa]

agree@1=0.700 agree@5=0.300 (chance 0.178)  dendro-purity=0.782  silhouette=0.493  within/between=0.277


| class | 5 nearest class centroids (cosine dist) |
|---|---|
| disturbed | edge-on bulge*(0.28) edge-on no bulge(0.30) merging(0.43) cigar smooth(0.59) barred spiral(0.77) |
| merging | cigar smooth(0.08) edge-on bulge*(0.37) disturbed(0.43) edge-on no bulge(0.48) in-between smooth(0.66) |
| round smooth | in-between smooth(0.13) unbarred tight spiral(1.13) merging(1.14) cigar smooth(1.16) unbarred loose spiral(1.55) |
| in-between smooth | round smooth(0.13) merging(0.66) cigar smooth(0.68) edge-on bulge*(1.42) disturbed(1.46) |
| cigar smooth | merging(0.08) edge-on bulge*(0.28) edge-on no bulge(0.38) disturbed(0.59) in-between smooth(0.68) |
| barred spiral | unbarred loose spiral(0.00) unbarred tight spiral(0.14) disturbed(0.77) edge-on no bulge(0.90) edge-on bulge*(1.02) |
| unbarred tight spiral | unbarred loose spiral(0.11) barred spiral(0.14) round smooth(1.13) disturbed(1.22) edge-on no bulge(1.37) |
| unbarred loose spiral | barred spiral(0.00) unbarred tight spiral(0.11) disturbed(0.81) edge-on no bulge(0.96) edge-on bulge*(1.07) |
| edge-on no bulge | edge-on bulge*(0.01) disturbed(0.30) cigar smooth(0.38) merging(0.48) barred spiral(0.90) |
| edge-on bulge* | edge-on no bulge(0.01) cigar smooth(0.28) disturbed(0.28) merging(0.37) barred spiral(1.02) |

## supcon-ft-res (residual)  [galaxy10_lejepa]

agree@1=0.500 agree@5=0.280 (chance 0.178)  dendro-purity=0.671  silhouette=0.178  within/between=0.742


| class | 5 nearest class centroids (cosine dist) |
|---|---|
| disturbed | unbarred loose spiral(0.43) edge-on no bulge(0.71) cigar smooth(0.80) in-between smooth(0.82) edge-on bulge*(0.86) |
| merging | edge-on bulge*(0.81) cigar smooth(0.83) disturbed(0.95) unbarred loose spiral(1.00) edge-on no bulge(1.00) |
| round smooth | unbarred tight spiral(0.86) disturbed(0.92) in-between smooth(0.94) edge-on no bulge(0.96) cigar smooth(1.03) |
| in-between smooth | cigar smooth(0.68) disturbed(0.82) edge-on bulge*(0.93) round smooth(0.94) edge-on no bulge(1.05) |
| cigar smooth | edge-on bulge*(0.22) edge-on no bulge(0.49) in-between smooth(0.68) disturbed(0.80) merging(0.83) |
| barred spiral | unbarred loose spiral(0.58) unbarred tight spiral(0.79) disturbed(0.98) edge-on no bulge(1.04) in-between smooth(1.06) |
| unbarred tight spiral | unbarred loose spiral(0.54) barred spiral(0.79) round smooth(0.86) edge-on no bulge(0.94) disturbed(1.02) |
| unbarred loose spiral | disturbed(0.43) unbarred tight spiral(0.54) barred spiral(0.58) merging(1.00) edge-on no bulge(1.01) |
| edge-on no bulge | edge-on bulge*(0.18) cigar smooth(0.49) disturbed(0.71) unbarred tight spiral(0.94) round smooth(0.96) |
| edge-on bulge* | edge-on no bulge(0.18) cigar smooth(0.22) merging(0.81) disturbed(0.86) in-between smooth(0.93) |

## supcon-ft-res (concat)  [galaxy10_lejepa]

agree@1=0.500 agree@5=0.260 (chance 0.178)  dendro-purity=0.621  silhouette=0.186  within/between=0.733


| class | 5 nearest class centroids (cosine dist) |
|---|---|
| disturbed | unbarred loose spiral(0.38) edge-on no bulge(0.71) in-between smooth(0.72) cigar smooth(0.74) round smooth(0.77) |
| merging | edge-on bulge*(0.78) cigar smooth(0.81) disturbed(0.83) round smooth(0.92) unbarred loose spiral(0.93) |
| round smooth | unbarred tight spiral(0.77) disturbed(0.77) in-between smooth(0.86) merging(0.92) unbarred loose spiral(0.92) |
| in-between smooth | cigar smooth(0.64) disturbed(0.72) round smooth(0.86) edge-on bulge*(0.92) barred spiral(0.98) |
| cigar smooth | edge-on bulge*(0.22) edge-on no bulge(0.49) in-between smooth(0.64) disturbed(0.74) merging(0.81) |
| barred spiral | unbarred loose spiral(0.54) unbarred tight spiral(0.74) disturbed(0.87) in-between smooth(0.98) merging(1.02) |
| unbarred tight spiral | unbarred loose spiral(0.49) barred spiral(0.74) round smooth(0.77) edge-on no bulge(0.89) disturbed(0.92) |
| unbarred loose spiral | disturbed(0.38) unbarred tight spiral(0.49) barred spiral(0.54) round smooth(0.92) merging(0.93) |
| edge-on no bulge | edge-on bulge*(0.17) cigar smooth(0.49) disturbed(0.71) unbarred tight spiral(0.89) round smooth(0.93) |
| edge-on bulge* | edge-on no bulge(0.17) cigar smooth(0.22) merging(0.78) disturbed(0.80) in-between smooth(0.92) |

## ss-ft-res (residual)  [galaxy10_lejepa]

agree@1=0.800 agree@5=0.280 (chance 0.178)  dendro-purity=0.774  silhouette=0.295  within/between=0.587


| class | 5 nearest class centroids (cosine dist) |
|---|---|
| disturbed | unbarred loose spiral(0.55) merging(0.61) in-between smooth(0.87) cigar smooth(0.95) barred spiral(1.00) |
| merging | disturbed(0.61) in-between smooth(0.67) unbarred loose spiral(0.97) barred spiral(0.97) edge-on bulge*(1.01) |
| round smooth | in-between smooth(0.92) unbarred tight spiral(0.98) merging(1.04) edge-on no bulge(1.13) disturbed(1.23) |
| in-between smooth | cigar smooth(0.61) merging(0.67) disturbed(0.87) round smooth(0.92) edge-on bulge*(1.01) |
| cigar smooth | edge-on bulge*(0.25) edge-on no bulge(0.51) in-between smooth(0.61) disturbed(0.95) merging(1.03) |
| barred spiral | unbarred loose spiral(0.42) unbarred tight spiral(0.67) merging(0.97) disturbed(1.00) edge-on bulge*(1.36) |
| unbarred tight spiral | unbarred loose spiral(0.55) barred spiral(0.67) round smooth(0.98) edge-on no bulge(1.13) disturbed(1.20) |
| unbarred loose spiral | barred spiral(0.42) unbarred tight spiral(0.55) disturbed(0.55) merging(0.97) edge-on no bulge(1.27) |
| edge-on no bulge | edge-on bulge*(0.19) cigar smooth(0.51) disturbed(1.11) round smooth(1.13) unbarred tight spiral(1.13) |
| edge-on bulge* | edge-on no bulge(0.19) cigar smooth(0.25) merging(1.01) in-between smooth(1.01) disturbed(1.08) |

## ss-ft-res (concat)  [galaxy10_lejepa]

agree@1=0.700 agree@5=0.280 (chance 0.178)  dendro-purity=0.742  silhouette=0.299  within/between=0.578


| class | 5 nearest class centroids (cosine dist) |
|---|---|
| disturbed | unbarred loose spiral(0.44) merging(0.65) in-between smooth(0.74) barred spiral(0.76) cigar smooth(0.88) |
| merging | disturbed(0.65) edge-on bulge*(0.86) in-between smooth(0.91) barred spiral(0.93) unbarred loose spiral(1.02) |
| round smooth | unbarred tight spiral(0.96) in-between smooth(1.05) edge-on no bulge(1.10) merging(1.16) edge-on bulge*(1.32) |
| in-between smooth | cigar smooth(0.49) disturbed(0.74) merging(0.91) edge-on bulge*(1.00) round smooth(1.05) |
| cigar smooth | edge-on bulge*(0.27) in-between smooth(0.49) edge-on no bulge(0.50) disturbed(0.88) merging(1.02) |
| barred spiral | unbarred loose spiral(0.35) unbarred tight spiral(0.64) disturbed(0.76) merging(0.93) edge-on bulge*(1.27) |
| unbarred tight spiral | unbarred loose spiral(0.47) barred spiral(0.64) round smooth(0.96) disturbed(1.14) edge-on no bulge(1.17) |
| unbarred loose spiral | barred spiral(0.35) disturbed(0.44) unbarred tight spiral(0.47) merging(1.02) edge-on bulge*(1.25) |
| edge-on no bulge | edge-on bulge*(0.19) cigar smooth(0.50) round smooth(1.10) merging(1.15) unbarred tight spiral(1.17) |
| edge-on bulge* | edge-on no bulge(0.19) cigar smooth(0.27) merging(0.86) in-between smooth(1.00) disturbed(1.03) |

## supcon-ft-resnplm (residual)  [galaxy10_lejepa]

agree@1=0.500 agree@5=0.280 (chance 0.178)  dendro-purity=0.590  silhouette=0.257  within/between=0.442


| class | 5 nearest class centroids (cosine dist) |
|---|---|
| disturbed | unbarred loose spiral(0.11) merging(0.12) in-between smooth(0.13) cigar smooth(0.23) round smooth(0.27) |
| merging | in-between smooth(0.09) disturbed(0.12) cigar smooth(0.21) round smooth(0.25) unbarred loose spiral(0.29) |
| round smooth | in-between smooth(0.13) merging(0.25) disturbed(0.27) unbarred tight spiral(0.28) unbarred loose spiral(0.38) |
| in-between smooth | merging(0.09) round smooth(0.13) disturbed(0.13) cigar smooth(0.18) unbarred loose spiral(0.36) |
| cigar smooth | edge-on bulge*(0.11) in-between smooth(0.18) merging(0.21) disturbed(0.23) edge-on no bulge(0.35) |
| barred spiral | unbarred loose spiral(0.18) disturbed(0.32) unbarred tight spiral(0.34) merging(0.44) in-between smooth(0.50) |
| unbarred tight spiral | unbarred loose spiral(0.18) round smooth(0.28) barred spiral(0.34) disturbed(0.34) merging(0.38) |
| unbarred loose spiral | disturbed(0.11) barred spiral(0.18) unbarred tight spiral(0.18) merging(0.29) in-between smooth(0.36) |
| edge-on no bulge | edge-on bulge*(0.09) cigar smooth(0.35) merging(0.65) disturbed(0.68) unbarred loose spiral(0.74) |
| edge-on bulge* | edge-on no bulge(0.09) cigar smooth(0.11) merging(0.34) disturbed(0.39) in-between smooth(0.44) |

## supcon-ft-resnplm (concat)  [galaxy10_lejepa]

agree@1=0.500 agree@5=0.280 (chance 0.178)  dendro-purity=0.646  silhouette=0.247  within/between=0.585


| class | 5 nearest class centroids (cosine dist) |
|---|---|
| disturbed | unbarred loose spiral(0.14) in-between smooth(0.21) merging(0.28) cigar smooth(0.29) barred spiral(0.37) |
| merging | disturbed(0.28) in-between smooth(0.40) cigar smooth(0.47) edge-on bulge*(0.49) unbarred loose spiral(0.50) |
| round smooth | disturbed(0.37) unbarred tight spiral(0.41) in-between smooth(0.44) merging(0.51) unbarred loose spiral(0.52) |
| in-between smooth | disturbed(0.21) cigar smooth(0.25) merging(0.40) round smooth(0.44) unbarred loose spiral(0.47) |
| cigar smooth | edge-on bulge*(0.19) in-between smooth(0.25) disturbed(0.29) edge-on no bulge(0.40) merging(0.47) |
| barred spiral | unbarred loose spiral(0.26) disturbed(0.37) unbarred tight spiral(0.41) in-between smooth(0.57) merging(0.58) |
| unbarred tight spiral | unbarred loose spiral(0.22) disturbed(0.40) round smooth(0.41) barred spiral(0.41) in-between smooth(0.58) |
| unbarred loose spiral | disturbed(0.14) unbarred tight spiral(0.22) barred spiral(0.26) in-between smooth(0.47) edge-on bulge*(0.50) |
| edge-on no bulge | edge-on bulge*(0.08) cigar smooth(0.40) unbarred loose spiral(0.68) disturbed(0.68) unbarred tight spiral(0.72) |
| edge-on bulge* | edge-on no bulge(0.08) cigar smooth(0.19) disturbed(0.41) merging(0.49) unbarred loose spiral(0.50) |
