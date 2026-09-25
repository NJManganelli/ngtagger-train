Per-layer test AUC (event-level 50/50 split)

| model | n_iter | fit_s | AUC L1 | AUC L2 | AUC L3 | AUC L4 | AUC Lall |
|---|---|---|---|---|---|---|---|
| HGB non-angle | 400.0000 | 38.3000 | 0.8180 | 0.8116 | 0.7875 | 0.7363 | 0.7982 |
| HGB + angles (SYNTHETIC) | 400.0000 | 49.7000 | 0.9750 | 0.9658 | 0.9637 | 0.9661 | 0.9685 |
| HGB + angles + origin-compat (SYNTHETIC) | 400.0000 | 46.4000 | 0.9809 | 0.9735 | 0.9710 | 0.9718 | 0.9751 |


AUC of unlinked vs each linked sub-population (same model, test set)

|  | all linked | linked pT>=2 | linked pT<0.3 | linked secondary vr>1cm | linked shared frac<0.9 |
|---|---|---|---|---|---|
| HGB non-angle | 0.7982 | 0.8584 | 0.7474 | 0.6556 | 0.5696 |
| HGB + angles (SYNTHETIC) | 0.9685 | 0.9757 | 0.9602 | 0.9414 | 0.9153 |
| HGB + angles + origin-compat (SYNTHETIC) | 0.9751 | 0.9830 | 0.9654 | 0.9470 | 0.9406 |


Working points (test set)

|  | linked kept (1-FPR) at 50% unlinked rejected | linked kept (1-FPR) at 80% unlinked rejected | linked kept (1-FPR) at 90% unlinked rejected | unlinked rejected at 1% linked loss | unlinked rejected at 5% linked loss | unlinked rejected at 1% loss of pT>=2 linked | unlinked rejected at 5% loss of pT>=2 linked |
|---|---|---|---|---|---|---|---|
| HGB non-angle | 0.8798 | 0.6452 | 0.4610 | 0.0882 | 0.2892 | 0.1289 | 0.4000 |
| HGB + angles (SYNTHETIC) | 0.9941 | 0.9606 | 0.9110 | 0.5889 | 0.8323 | 0.6152 | 0.8699 |
| HGB + angles + origin-compat (SYNTHETIC) | 0.9968 | 0.9749 | 0.9368 | 0.6708 | 0.8780 | 0.7403 | 0.9198 |


Permutation importance (mean ROC-AUC drop, 300k test clusters, 3 repeats)

|  | HGB non-angle | HGB + angles (SYNTHETIC) | HGB + angles + origin-compat (SYNTHETIC) |
|---|---|---|---|
| absZ | 0.0647 | 0.0172 | 0.0012 |
| charge | 0.0210 | 0.0030 | 0.0022 |
| dphiNorm |  |  | 0.0457 |
| globalR | 0.1222 | 0.0869 | 0.0129 |
| hasAlpha |  | 0.0002 | 0.0001 |
| hasBeta |  | 0.0067 | 0.0037 |
| layer | 0.0441 | 0.0672 | 0.0190 |
| localCotAlpha |  | 0.1886 | 0.1000 |
| localCotBeta |  | 0.1154 | 0.0733 |
| localX | 0.0109 | 0.0218 | 0.0000 |
| localY | 0.0001 | 0.0001 | -0.0000 |
| qPerPix | 0.0183 | 0.0030 | 0.0019 |
| sigAlpha |  | 0.0011 | 0.0007 |
| sigBeta |  | 0.0110 | 0.0024 |
| sigX | 0.0159 | 0.0031 | 0.0028 |
| sigY | 0.0043 | 0.0094 | 0.0063 |
| size | 0.0088 | 0.0020 | 0.0010 |
| sizeX | 0.1278 | 0.0709 | 0.0365 |
| sizeY | 0.1002 | 0.0578 | 0.0428 |
| z0imp |  |  | 0.0655 |
