HGB on feature subsets (1M train / 1M test clusters, event-level split)

| features | AUC all | AUC L1 | AUC L2 | AUC L3 | AUC L4 |
|---|---|---|---|---|---|
| sizeY | 0.566 | 0.584 | 0.585 | 0.550 | 0.516 |
| sizeY + |z| + r (z-shape vs origin) | 0.656 | 0.669 | 0.658 | 0.634 | 0.618 |
| sizeX | 0.680 | 0.662 | 0.694 | 0.707 | 0.663 |
| sizeX + localX + r + layer (rphi-shape vs origin) | 0.729 | 0.744 | 0.736 | 0.718 | 0.670 |
| shape only: size,sizeX,sizeY,charge,qPerPix,sigX,sigY,layer | 0.749 | 0.740 | 0.762 | 0.761 | 0.715 |
| position only: |z|, r, localX, localY, layer | 0.571 | 0.601 | 0.561 | 0.532 | 0.520 |
| sizeX,sizeY,|z|,r,localX,layer | 0.778 | 0.793 | 0.793 | 0.771 | 0.714 |
| all non-angle (as s3) | 0.796 | 0.815 | 0.809 | 0.785 | 0.736 |
