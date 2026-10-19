Research results

Average recall is the unweighted mean of the 20 class recalls. Bold marks only the best number in each column within each evaluation table.

Fold 1: 38283 held-out parcels

| Method | Test mIoU | Test OA | Test average recall |
|---|---:|---:|---:|
| Eight-model ensemble: remove only RoPE (post-hoc) | **54.98** | 94.96 | 61.48 |
| Validation-selected ensemble (9 members) | 54.93 | **95.03** | 61.21 |
| Eight-model control: six + baseline checkpoints 27 / 78 | 54.16 | 94.97 | 59.63 |
| Six-model ensemble: previous five + CNN → TGT | 54.24 | 94.97 | 60.06 |
| Five-model ensemble: four neural models + XGBoost | 53.72 | 94.88 | 59.58 |
| Four-model ensemble: baseline + capacity + TGT + dropout | 52.62 | 94.80 | 58.02 |
| Three-model ensemble: baseline + capacity + TGT | 52.28 | 94.76 | 57.80 |
| XGBoost: frozen embeddings + spectral summaries | 52.23 | 93.51 | 61.62 |
| Date-dropout fine-tuning | 51.67 | 94.42 | 58.78 |
| Baseline checkpoint ensemble (epochs 27 / 38 / 78) | 51.61 | 94.67 | 57.47 |
| Baseline + frequency-based logit adjustment (tau=0.1) | 51.25 | 93.65 | **61.82** |
| Baseline + best single-class bias | 51.36 | 94.30 | 58.34 |
| Baseline + greedy class biases | 51.68 | 94.28 | 58.65 |
| Baseline + extended class-bias search (validation OA constrained) | 51.58 | 94.15 | 59.36 |
| Original PSE+TAE baseline (cached evaluation) | 51.31 | 94.24 | 58.35 |
| CNN → TGT correction (snapshot epoch 4) | 51.65 | 94.37 | 58.66 |
| Baseline-size PSE+TAE distilled from scratch (epoch 86) | 53.35 | 94.66 | 60.70 |
| Matched baseline-size CE-only control (epoch 80) | 51.68 | 94.43 | 58.47 |
| Earlier variants: no established improvement | | | |
| CNN spatial correction | 51.28 | 94.34 | 58.41 |
| Patch baseline, validation-best epoch 80 | 49.94 | 94.17 | 57.57 |
| Patch baseline, ensemble checkpoint 38 | 50.48 | 94.38 | 56.72 |
| TGT (epoch 38 ensemble snapshot) | 49.44 | 93.98 | 56.53 |
| Capacity control (epoch 67 ensemble snapshot) | 50.16 | 94.10 | 58.01 |
| Embedding moments (mean / std / skew / kurtosis) | 46.16 | 93.98 | 51.16 |
| Frozen encoder + date kurtosis | 51.47 | 94.31 | 59.41 |
| Scaled embedding kurtosis, gradients on | 47.98 | 94.17 | - |
| Parcel mean / std in the pixel encoder | 50.11 | - | - |
| Parcel mean / std / kurtosis in the pixel encoder | 50.05 | - | - |
| Inverse-square-root class-weighted loss | 49.07 | 92.14 | - |
| Small per-pixel temporal attention, from scratch | 48.11 | 93.73 | 55.82 |
| Small temporal + spatial attention, from scratch | 49.20 | 94.03 | 55.02 |
| Full-width embedding-only correction | 51.56 | 94.34 | 58.97 |
| Full-width TGT correction | 51.24 | 94.24 | 58.41 |
| Full-width TGT + spatial attention correction | 51.46 | 94.36 | 58.37 |
| Per-pixel temporal-distribution correction | 51.15 | 94.35 | 57.67 |
| Small patch-pixel temporal correction | 51.78 | 94.39 | 58.72 |
| Small temporal + spatial correction, no coordinates | 51.80 | 94.40 | 58.70 |
| Small temporal + spatial correction, with coordinates | 51.69 | 94.36 | 58.58 |
| Small replacement classifier (best new variant by validation) | 50.73 | 94.00 | 59.08 |

All values are percentages. One fold and one seed; these comparisons are exploratory.
Earlier variants include ties and inconclusive gains. Missing values are unmeasured, not zero.
Library additions and mixture weights are chosen on validation; test scores do not select the ensemble.

Five-fold results: mean and sample standard deviation

| Method | Test mIoU | Test OA | Test average recall |
|---|---:|---:|---:|
| PSE+TAE reproduction | **50.96** ± 0.60 | 94.22 ± 0.09 | **58.50** ± 0.94 |
| Earlier alternatives | | | |
| L-TAE reproduction | 50.67 ± 0.95 | **94.33** ± 0.04 | - |
| Scaled kurtosis, stopped gradient | 50.79 ± 0.75 | 94.17 ± 0.21 | - |
| XGBoost 60/20/20: band mean / std | 41.40 ± 0.29 | 93.04 ± 0.11 | 44.72 ± 0.32 |
| XGBoost 60/20/20: band mean / std + kurtosis | 40.87 ± 0.37 | 93.13 ± 0.12 | 44.04 ± 0.39 |
