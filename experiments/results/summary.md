# Results

Primary metric per benchmark (mean ± SD over the seed groups present, n = runs): toxicity_detection: `auprc_toxic`, claudette_tos: `weighted_auprc_8`, cti_vsp: `mad` (lower is better)

| arm variant | toxicity_detection | claudette_tos | cti_vsp |
|---|---|---|---|
| plain | 0.536 ± 0.000 (n=1) | 0.601 ± 0.000 (n=1) | 1.669 ± 0.000 (n=1) |
| api/deepseek | 0.342 ± 0.000 (n=1) | 0.470 ± 0.000 (n=1) | 0.785 ± 0.000 (n=1) |
| api/llama | 0.255 ± 0.000 (n=1) | 0.575 ± 0.000 (n=1) | 1.617 ± 0.000 (n=1) |
| api/gpt | 0.745 ± 0.000 (n=1) | 0.746 ± 0.000 (n=1) | 1.849 ± 0.000 (n=1) |
| icl/k5 | 0.714 ± 0.019 (n=5) | 0.612 ± 0.033 (n=5) | 1.510 ± 0.131 (n=5) |
| gold/k5 | 0.861 ± 0.011 (n=5) | 0.738 ± 0.019 (n=5) | 0.656 ± 0.025 (n=5) |
| bb/deepseek/k5/th0_7 | 0.717 ± 0.025 (n=5) | 0.673 ± 0.026 (n=5) | 1.022 ± 0.106 (n=5) |
| bb/llama/k5/th0_7 | 0.681 ± 0.046 (n=5) | 0.668 ± 0.027 (n=5) | 1.376 ± 0.091 (n=5) |
| fg/deepseek/k5/th0_7-0 | 0.672 ± 0.067 (n=5) | 0.704 ± 0.011 (n=5) | 1.029 ± 0.110 (n=5) |
| fg/llama/k5/th0_7-0 | 0.748 ± 0.022 (n=5) | 0.669 ± 0.018 (n=5) | 1.302 ± 0.062 (n=5) |
| hybrid/deepseek/r350-50/k5/th0_7-0 | 0.710 ± 0.053 (n=5) | 0.691 ± 0.029 (n=5) | 0.967 ± 0.045 (n=5) |
| hybrid/deepseek/r300-100/k5/th0_7-0 | 0.713 ± 0.036 (n=5) | 0.685 ± 0.026 (n=5) | 0.958 ± 0.070 (n=5) |
| hybrid/llama/r350-50/k5/th0_7-0 | 0.703 ± 0.050 (n=5) | 0.664 ± 0.019 (n=5) | 1.415 ± 0.051 (n=5) |
| hybrid/llama/r300-100/k5/th0_7-0 | 0.731 ± 0.018 (n=5) | 0.660 ± 0.026 (n=5) | 1.344 ± 0.119 (n=5) |
| icl/k10 | 0.728 ± 0.050 (n=5) | 0.611 ± 0.043 (n=5) | 1.268 ± 0.090 (n=5) |
| gold/k10 | 0.866 ± 0.010 (n=5) | 0.748 ± 0.025 (n=5) | 0.623 ± 0.016 (n=5) |
| bb/deepseek/k10/th0_7 | 0.706 ± 0.016 (n=4) | 0.685 ± 0.034 (n=5) | 0.939 ± 0.039 (n=5) |
| bb/llama/k10/th0_7 | 0.696 ± 0.042 (n=4) | 0.660 ± 0.024 (n=5) | 1.395 ± 0.075 (n=5) |
| fg/deepseek/k10/th0_7-0 | 0.722 ± 0.041 (n=5) | 0.694 ± 0.031 (n=5) | 0.986 ± 0.039 (n=4) |
| fg/llama/k10/th0_7-0 | 0.710 ± 0.020 (n=5) | 0.651 ± 0.012 (n=5) | 1.252 ± 0.144 (n=4) |
| hybrid/deepseek/r345-50/k10/th0_7-0 | 0.704 ± 0.021 (n=4) | 0.669 ± 0.019 (n=5) | 0.990 ± 0.027 (n=5) |
| hybrid/deepseek/r295-100/k10/th0_7-0 | 0.712 ± 0.029 (n=4) | 0.690 ± 0.029 (n=5) | 0.981 ± 0.101 (n=5) |
| hybrid/llama/r345-50/k10/th0_7-0 | 0.691 ± 0.044 (n=4) | 0.676 ± 0.018 (n=5) | 1.367 ± 0.077 (n=4) |
| hybrid/llama/r295-100/k10/th0_7-0 | 0.731 ± 0.012 (n=4) | 0.653 ± 0.009 (n=5) | 1.410 ± 0.128 (n=4) |

302 run(s) with bench done; per-run values in runs.csv.

## Generation acceptance

Per arm variant x benchmark: candidate counts (mean over the n runs) and rates in % (mean ± SD over the runs) of the checked candidates (accepted + rejected; candidates discarded after the target was reached are not counted). Rejection reasons: activation = target feature not active (checked first), ROUGE-L = too similar to the pool (ROUGE-L > rouge_threshold).

### Black-box (bb)

| arm variant | benchmark | n | accepted | rejected | acceptance % | rejection % |
|---|---|---|---|---|---|---|
| bb/deepseek/k5/th0_7 | toxicity_detection | 5 | 400 | 213.8 | 66.0 ± 8.8 | 34.0 ± 8.8 |
| bb/llama/k5/th0_7 | toxicity_detection | 5 | 400 | 15.6 | 96.3 ± 0.9 | 3.7 ± 0.9 |
| bb/deepseek/k10/th0_7 | toxicity_detection | 4 | 395 | 262.8 | 61.7 ± 11.4 | 38.3 ± 11.4 |
| bb/llama/k10/th0_7 | toxicity_detection | 4 | 395 | 22.2 | 94.7 ± 3.1 | 5.3 ± 3.1 |
| bb/deepseek/k5/th0_7 | claudette_tos | 5 | 400 | 913 | 31.2 ± 5.3 | 68.8 ± 5.3 |
| bb/llama/k5/th0_7 | claudette_tos | 5 | 400 | 52.4 | 88.6 ± 4.9 | 11.4 ± 4.9 |
| bb/deepseek/k10/th0_7 | claudette_tos | 5 | 395 | 730.2 | 35.2 ± 2.1 | 64.8 ± 2.1 |
| bb/llama/k10/th0_7 | claudette_tos | 5 | 395 | 41.6 | 90.6 ± 3.5 | 9.4 ± 3.5 |
| bb/deepseek/k5/th0_7 | cti_vsp | 5 | 400 | 37.6 | 91.4 ± 1.6 | 8.6 ± 1.6 |
| bb/llama/k5/th0_7 | cti_vsp | 5 | 400 | 8 | 98.1 ± 1.3 | 1.9 ± 1.3 |
| bb/deepseek/k10/th0_7 | cti_vsp | 5 | 395 | 25.4 | 94.0 ± 2.7 | 6.0 ± 2.7 |
| bb/llama/k10/th0_7 | cti_vsp | 5 | 395 | 9 | 97.8 ± 1.5 | 2.2 ± 1.5 |

### Feature-guided (fg)

| arm variant | benchmark | n | accepted | rej. ROUGE-L | rej. activation | rejected | acceptance % | ROUGE-L % | activation % | rejection % |
|---|---|---|---|---|---|---|---|---|---|---|
| fg/deepseek/k5/th0_7-0 | toxicity_detection | 5 | 400 | 1.6 | 286.4 | 288 | 58.2 ± 1.0 | 0.2 ± 0.2 | 41.6 ± 1.0 | 41.8 ± 1.0 |
| fg/llama/k5/th0_7-0 | toxicity_detection | 5 | 400 | 0 | 435.4 | 435.4 | 47.9 ± 1.5 | 0.0 ± 0.0 | 52.1 ± 1.5 | 52.1 ± 1.5 |
| fg/deepseek/k10/th0_7-0 | toxicity_detection | 5 | 395 | 3.2 | 296.4 | 299.6 | 56.9 ± 1.9 | 0.5 ± 0.2 | 42.6 ± 1.7 | 43.1 ± 1.9 |
| fg/llama/k10/th0_7-0 | toxicity_detection | 5 | 395 | 0.2 | 446.6 | 446.8 | 47.2 ± 3.9 | 0.0 ± 0.1 | 52.8 ± 3.9 | 52.8 ± 3.9 |
| fg/deepseek/k5/th0_7-0 | claudette_tos | 5 | 400 | 8 | 238.6 | 246.6 | 61.9 ± 0.9 | 1.2 ± 0.5 | 36.9 ± 0.5 | 38.1 ± 0.9 |
| fg/llama/k5/th0_7-0 | claudette_tos | 5 | 400 | 7.6 | 548 | 555.6 | 41.9 ± 0.8 | 0.8 ± 0.3 | 57.3 ± 0.7 | 58.1 ± 0.8 |
| fg/deepseek/k10/th0_7-0 | claudette_tos | 5 | 395 | 9 | 253.6 | 262.6 | 60.2 ± 2.7 | 1.4 ± 0.6 | 38.5 ± 2.5 | 39.8 ± 2.7 |
| fg/llama/k10/th0_7-0 | claudette_tos | 5 | 395 | 6.4 | 518.4 | 524.8 | 43.0 ± 1.1 | 0.7 ± 0.3 | 56.3 ± 0.8 | 57.0 ± 1.1 |
| fg/deepseek/k5/th0_7-0 | cti_vsp | 5 | 400 | 8.8 | 339.8 | 348.6 | 53.5 ± 2.0 | 1.1 ± 1.1 | 45.4 ± 1.0 | 46.5 ± 2.0 |
| fg/llama/k5/th0_7-0 | cti_vsp | 5 | 400 | 5 | 433.4 | 438.4 | 47.7 ± 1.1 | 0.6 ± 0.2 | 51.7 ± 0.9 | 52.3 ± 1.1 |
| fg/deepseek/k10/th0_7-0 | cti_vsp | 4 | 395 | 7.8 | 342.5 | 350.2 | 53.1 ± 3.0 | 1.0 ± 0.5 | 45.8 ± 2.6 | 46.9 ± 3.0 |
| fg/llama/k10/th0_7-0 | cti_vsp | 4 | 395 | 3.8 | 460 | 463.8 | 46.0 ± 1.2 | 0.4 ± 0.2 | 53.5 ± 1.1 | 54.0 ± 1.2 |

### Hybrid

| arm variant | benchmark | phase | n | accepted | rej. ROUGE-L | rej. activation | rejected | acceptance % | ROUGE-L % | activation % | rejection % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid/deepseek/r350-50/k5/th0_7-0 | toxicity_detection | bb | 5 | 350 | 159.2 | 0 | 159.2 | 69.3 ± 7.0 | 30.7 ± 7.0 | 0.0 ± 0.0 | 30.7 ± 7.0 |
| hybrid/deepseek/r350-50/k5/th0_7-0 | toxicity_detection | fg | 5 | 50 | 0 | 106.6 | 106.6 | 32.1 ± 2.3 | 0.0 ± 0.0 | 67.9 ± 2.3 | 67.9 ± 2.3 |
| hybrid/deepseek/r350-50/k5/th0_7-0 | toxicity_detection | total | 5 | 400 | 159.2 | 106.6 | 265.8 | 60.3 ± 4.1 | 23.6 ± 6.0 | 16.1 ± 2.3 | 39.7 ± 4.1 |
| hybrid/deepseek/r300-100/k5/th0_7-0 | toxicity_detection | bb | 5 | 300 | 130.6 | 0 | 130.6 | 70.2 ± 6.9 | 29.8 ± 6.9 | 0.0 ± 0.0 | 29.8 ± 6.9 |
| hybrid/deepseek/r300-100/k5/th0_7-0 | toxicity_detection | fg | 5 | 100 | 0.4 | 214.4 | 214.8 | 31.8 ± 0.9 | 0.1 ± 0.2 | 68.1 ± 0.8 | 68.2 ± 0.9 |
| hybrid/deepseek/r300-100/k5/th0_7-0 | toxicity_detection | total | 5 | 400 | 131 | 214.4 | 345.4 | 53.8 ± 3.5 | 17.4 ± 4.5 | 28.8 ± 1.2 | 46.2 ± 3.5 |
| hybrid/llama/r350-50/k5/th0_7-0 | toxicity_detection | bb | 5 | 350 | 12.6 | 0 | 12.6 | 96.5 ± 1.4 | 3.5 ± 1.4 | 0.0 ± 0.0 | 3.5 ± 1.4 |
| hybrid/llama/r350-50/k5/th0_7-0 | toxicity_detection | fg | 5 | 50 | 0.2 | 208.6 | 208.8 | 19.7 ± 2.9 | 0.1 ± 0.2 | 80.3 ± 2.9 | 80.3 ± 2.9 |
| hybrid/llama/r350-50/k5/th0_7-0 | toxicity_detection | total | 5 | 400 | 12.8 | 208.6 | 221.4 | 64.5 ± 3.4 | 2.1 ± 0.9 | 33.4 ± 4.2 | 35.5 ± 3.4 |
| hybrid/llama/r300-100/k5/th0_7-0 | toxicity_detection | bb | 5 | 300 | 14.2 | 0 | 14.2 | 95.5 ± 2.6 | 4.5 ± 2.6 | 0.0 ± 0.0 | 4.5 ± 2.6 |
| hybrid/llama/r300-100/k5/th0_7-0 | toxicity_detection | fg | 5 | 100 | 0.6 | 263.8 | 264.4 | 27.5 ± 1.0 | 0.2 ± 0.1 | 72.4 ± 0.9 | 72.5 ± 1.0 |
| hybrid/llama/r300-100/k5/th0_7-0 | toxicity_detection | total | 5 | 400 | 14.8 | 263.8 | 278.6 | 59.0 ± 1.6 | 2.2 ± 1.2 | 38.9 ± 1.0 | 41.0 ± 1.6 |
| hybrid/deepseek/r345-50/k10/th0_7-0 | toxicity_detection | bb | 4 | 345 | 228.5 | 0 | 228.5 | 62.0 ± 12.1 | 38.0 ± 12.1 | 0.0 ± 0.0 | 38.0 ± 12.1 |
| hybrid/deepseek/r345-50/k10/th0_7-0 | toxicity_detection | fg | 4 | 50 | 0 | 115.2 | 115.2 | 30.4 ± 2.9 | 0.0 ± 0.0 | 69.6 ± 2.9 | 69.6 ± 2.9 |
| hybrid/deepseek/r345-50/k10/th0_7-0 | toxicity_detection | total | 4 | 395 | 228.5 | 115.2 | 343.8 | 54.3 ± 7.7 | 29.7 ± 11.2 | 16.0 ± 3.7 | 45.7 ± 7.7 |
| hybrid/deepseek/r295-100/k10/th0_7-0 | toxicity_detection | bb | 4 | 295 | 163.2 | 0 | 163.2 | 65.7 ± 10.4 | 34.3 ± 10.4 | 0.0 ± 0.0 | 34.3 ± 10.4 |
| hybrid/deepseek/r295-100/k10/th0_7-0 | toxicity_detection | fg | 4 | 100 | 0 | 201 | 201 | 33.3 ± 1.8 | 0.0 ± 0.0 | 66.7 ± 1.8 | 66.7 ± 1.8 |
| hybrid/deepseek/r295-100/k10/th0_7-0 | toxicity_detection | total | 4 | 395 | 163.2 | 201 | 364.2 | 52.3 ± 4.2 | 21.0 ± 8.1 | 26.7 ± 4.1 | 47.7 ± 4.2 |
| hybrid/llama/r345-50/k10/th0_7-0 | toxicity_detection | bb | 4 | 345 | 17.2 | 0 | 17.2 | 95.3 ± 3.2 | 4.7 ± 3.2 | 0.0 ± 0.0 | 4.7 ± 3.2 |
| hybrid/llama/r345-50/k10/th0_7-0 | toxicity_detection | fg | 4 | 50 | 0 | 202 | 202 | 20.0 ± 1.7 | 0.0 ± 0.0 | 80.0 ± 1.7 | 80.0 ± 1.7 |
| hybrid/llama/r345-50/k10/th0_7-0 | toxicity_detection | total | 4 | 395 | 17.2 | 202 | 219.2 | 64.4 ± 2.0 | 2.8 ± 2.0 | 32.8 ± 2.7 | 35.6 ± 2.0 |
| hybrid/llama/r295-100/k10/th0_7-0 | toxicity_detection | bb | 4 | 295 | 11.8 | 0 | 11.8 | 96.2 ± 2.6 | 3.8 ± 2.6 | 0.0 ± 0.0 | 3.8 ± 2.6 |
| hybrid/llama/r295-100/k10/th0_7-0 | toxicity_detection | fg | 4 | 100 | 0.2 | 285.2 | 285.5 | 26.0 ± 1.6 | 0.1 ± 0.1 | 73.9 ± 1.5 | 74.0 ± 1.6 |
| hybrid/llama/r295-100/k10/th0_7-0 | toxicity_detection | total | 4 | 395 | 12 | 285.2 | 297.2 | 57.1 ± 2.6 | 1.7 ± 1.2 | 41.2 ± 1.5 | 42.9 ± 2.6 |
| hybrid/deepseek/r350-50/k5/th0_7-0 | claudette_tos | bb | 5 | 350 | 739.2 | 0 | 739.2 | 32.9 ± 5.6 | 67.1 ± 5.6 | 0.0 ± 0.0 | 67.1 ± 5.6 |
| hybrid/deepseek/r350-50/k5/th0_7-0 | claudette_tos | fg | 5 | 50 | 1 | 132.4 | 133.4 | 27.6 ± 3.7 | 0.5 ± 0.4 | 71.8 ± 3.7 | 72.4 ± 3.7 |
| hybrid/deepseek/r350-50/k5/th0_7-0 | claudette_tos | total | 5 | 400 | 740.2 | 132.4 | 872.6 | 32.0 ± 5.0 | 57.5 ± 5.8 | 10.5 ± 1.6 | 68.0 ± 5.0 |
| hybrid/deepseek/r300-100/k5/th0_7-0 | claudette_tos | bb | 5 | 300 | 532.4 | 0 | 532.4 | 36.5 ± 4.6 | 63.5 ± 4.6 | 0.0 ± 0.0 | 63.5 ± 4.6 |
| hybrid/deepseek/r300-100/k5/th0_7-0 | claudette_tos | fg | 5 | 100 | 4 | 169.4 | 173.4 | 36.6 ± 0.4 | 1.5 ± 0.8 | 62.0 ± 0.5 | 63.4 ± 0.4 |
| hybrid/deepseek/r300-100/k5/th0_7-0 | claudette_tos | total | 5 | 400 | 536.4 | 169.4 | 705.8 | 36.4 ± 3.4 | 48.1 ± 5.0 | 15.4 ± 1.6 | 63.6 ± 3.4 |
| hybrid/llama/r350-50/k5/th0_7-0 | claudette_tos | bb | 5 | 350 | 41.8 | 0 | 41.8 | 89.4 ± 3.2 | 10.6 ± 3.2 | 0.0 ± 0.0 | 10.6 ± 3.2 |
| hybrid/llama/r350-50/k5/th0_7-0 | claudette_tos | fg | 5 | 50 | 1.8 | 238.6 | 240.4 | 17.2 ± 0.6 | 0.6 ± 0.5 | 82.1 ± 0.9 | 82.8 ± 0.6 |
| hybrid/llama/r350-50/k5/th0_7-0 | claudette_tos | total | 5 | 400 | 43.6 | 238.6 | 282.2 | 58.7 ± 1.5 | 6.4 ± 2.1 | 35.0 ± 1.4 | 41.3 ± 1.5 |
| hybrid/llama/r300-100/k5/th0_7-0 | claudette_tos | bb | 5 | 300 | 27.6 | 0 | 27.6 | 91.6 ± 2.1 | 8.4 ± 2.1 | 0.0 ± 0.0 | 8.4 ± 2.1 |
| hybrid/llama/r300-100/k5/th0_7-0 | claudette_tos | fg | 5 | 100 | 2.6 | 287.4 | 290 | 25.7 ± 1.5 | 0.7 ± 0.5 | 73.6 ± 1.2 | 74.3 ± 1.5 |
| hybrid/llama/r300-100/k5/th0_7-0 | claudette_tos | total | 5 | 400 | 30.2 | 287.4 | 317.6 | 55.8 ± 1.7 | 4.2 ± 1.0 | 40.0 ± 1.9 | 44.2 ± 1.7 |
| hybrid/deepseek/r345-50/k10/th0_7-0 | claudette_tos | bb | 5 | 345 | 572 | 0 | 572 | 38.1 ± 4.4 | 61.9 ± 4.4 | 0.0 ± 0.0 | 61.9 ± 4.4 |
| hybrid/deepseek/r345-50/k10/th0_7-0 | claudette_tos | fg | 5 | 50 | 2 | 133 | 135 | 27.5 ± 3.9 | 1.1 ± 0.5 | 71.4 ± 4.0 | 72.5 ± 3.9 |
| hybrid/deepseek/r345-50/k10/th0_7-0 | claudette_tos | total | 5 | 395 | 574 | 133 | 707 | 36.1 ± 3.4 | 51.7 ± 5.4 | 12.2 ± 2.9 | 63.9 ± 3.4 |
| hybrid/deepseek/r295-100/k10/th0_7-0 | claudette_tos | bb | 5 | 295 | 457.4 | 0 | 457.4 | 39.6 ± 4.3 | 60.4 ± 4.3 | 0.0 ± 0.0 | 60.4 ± 4.3 |
| hybrid/deepseek/r295-100/k10/th0_7-0 | claudette_tos | fg | 5 | 100 | 3.6 | 169.8 | 173.4 | 36.7 ± 2.0 | 1.3 ± 1.1 | 62.1 ± 1.1 | 63.3 ± 2.0 |
| hybrid/deepseek/r295-100/k10/th0_7-0 | claudette_tos | total | 5 | 395 | 461 | 169.8 | 630.8 | 38.6 ± 2.6 | 44.7 ± 4.8 | 16.7 ± 2.2 | 61.4 ± 2.6 |
| hybrid/llama/r345-50/k10/th0_7-0 | claudette_tos | bb | 5 | 345 | 31.2 | 0 | 31.2 | 91.8 ± 3.9 | 8.2 ± 3.9 | 0.0 ± 0.0 | 8.2 ± 3.9 |
| hybrid/llama/r345-50/k10/th0_7-0 | claudette_tos | fg | 5 | 50 | 1.4 | 235.4 | 236.8 | 17.4 ± 0.2 | 0.5 ± 0.3 | 82.1 ± 0.5 | 82.6 ± 0.2 |
| hybrid/llama/r345-50/k10/th0_7-0 | claudette_tos | total | 5 | 395 | 32.6 | 235.4 | 268 | 59.6 ± 1.7 | 4.9 ± 2.2 | 35.5 ± 0.6 | 40.4 ± 1.7 |
| hybrid/llama/r295-100/k10/th0_7-0 | claudette_tos | bb | 5 | 295 | 28.8 | 0 | 28.8 | 91.2 ± 3.1 | 8.8 ± 3.1 | 0.0 ± 0.0 | 8.8 ± 3.1 |
| hybrid/llama/r295-100/k10/th0_7-0 | claudette_tos | fg | 5 | 100 | 3.2 | 288.4 | 291.6 | 25.6 ± 1.0 | 0.8 ± 0.3 | 73.6 ± 0.9 | 74.4 ± 1.0 |
| hybrid/llama/r295-100/k10/th0_7-0 | claudette_tos | total | 5 | 395 | 32 | 288.4 | 320.4 | 55.2 ± 1.3 | 4.5 ± 1.4 | 40.3 ± 1.4 | 44.8 ± 1.3 |
| hybrid/deepseek/r350-50/k5/th0_7-0 | cti_vsp | bb | 5 | 350 | 33.6 | 0 | 33.6 | 91.3 ± 2.7 | 8.7 ± 2.7 | 0.0 ± 0.0 | 8.7 ± 2.7 |
| hybrid/deepseek/r350-50/k5/th0_7-0 | cti_vsp | fg | 5 | 50 | 0.8 | 167.6 | 168.4 | 22.9 ± 1.2 | 0.4 ± 0.4 | 76.7 ± 1.2 | 77.1 ± 1.2 |
| hybrid/deepseek/r350-50/k5/th0_7-0 | cti_vsp | total | 5 | 400 | 34.4 | 167.6 | 202 | 66.5 ± 1.4 | 5.7 ± 1.8 | 27.8 ± 1.5 | 33.5 ± 1.4 |
| hybrid/deepseek/r300-100/k5/th0_7-0 | cti_vsp | bb | 5 | 300 | 27.6 | 0 | 27.6 | 91.6 ± 2.1 | 8.4 ± 2.1 | 0.0 ± 0.0 | 8.4 ± 2.1 |
| hybrid/deepseek/r300-100/k5/th0_7-0 | cti_vsp | fg | 5 | 100 | 1.8 | 194.6 | 196.4 | 33.9 ± 2.3 | 0.6 ± 0.6 | 65.5 ± 2.1 | 66.1 ± 2.3 |
| hybrid/deepseek/r300-100/k5/th0_7-0 | cti_vsp | total | 5 | 400 | 29.4 | 194.6 | 224 | 64.1 ± 1.5 | 4.7 ± 1.3 | 31.1 ± 2.4 | 35.9 ± 1.5 |
| hybrid/llama/r350-50/k5/th0_7-0 | cti_vsp | bb | 5 | 350 | 7.4 | 0 | 7.4 | 97.9 ± 1.2 | 2.1 ± 1.2 | 0.0 ± 0.0 | 2.1 ± 1.2 |
| hybrid/llama/r350-50/k5/th0_7-0 | cti_vsp | fg | 5 | 50 | 0.6 | 205.4 | 206 | 19.7 ± 2.1 | 0.2 ± 0.3 | 80.1 ± 1.9 | 80.3 ± 2.1 |
| hybrid/llama/r350-50/k5/th0_7-0 | cti_vsp | total | 5 | 400 | 8 | 205.4 | 213.4 | 65.3 ± 2.5 | 1.3 ± 0.6 | 33.4 ± 3.0 | 34.7 ± 2.5 |
| hybrid/llama/r300-100/k5/th0_7-0 | cti_vsp | bb | 5 | 300 | 3.4 | 0 | 3.4 | 98.9 ± 0.8 | 1.1 ± 0.8 | 0.0 ± 0.0 | 1.1 ± 0.8 |
| hybrid/llama/r300-100/k5/th0_7-0 | cti_vsp | fg | 5 | 100 | 1 | 244 | 245 | 29.0 ± 1.3 | 0.3 ± 0.3 | 70.7 ± 1.2 | 71.0 ± 1.3 |
| hybrid/llama/r300-100/k5/th0_7-0 | cti_vsp | total | 5 | 400 | 4.4 | 244 | 248.4 | 61.7 ± 1.5 | 0.7 ± 0.5 | 37.6 ± 1.4 | 38.3 ± 1.5 |
| hybrid/deepseek/r345-50/k10/th0_7-0 | cti_vsp | bb | 5 | 345 | 18 | 0 | 18 | 95.1 ± 1.6 | 4.9 ± 1.6 | 0.0 ± 0.0 | 4.9 ± 1.6 |
| hybrid/deepseek/r345-50/k10/th0_7-0 | cti_vsp | fg | 5 | 50 | 0.6 | 168.4 | 169 | 22.9 ± 1.7 | 0.3 ± 0.3 | 76.8 ± 1.8 | 77.1 ± 1.7 |
| hybrid/deepseek/r345-50/k10/th0_7-0 | cti_vsp | total | 5 | 395 | 18.6 | 168.4 | 187 | 67.9 ± 2.1 | 3.2 ± 1.1 | 28.9 ± 2.0 | 32.1 ± 2.1 |
| hybrid/deepseek/r295-100/k10/th0_7-0 | cti_vsp | bb | 5 | 295 | 16.6 | 0 | 16.6 | 94.7 ± 1.9 | 5.3 ± 1.9 | 0.0 ± 0.0 | 5.3 ± 1.9 |
| hybrid/deepseek/r295-100/k10/th0_7-0 | cti_vsp | fg | 5 | 100 | 1.2 | 199.6 | 200.8 | 33.3 ± 1.8 | 0.4 ± 0.4 | 66.3 ± 2.0 | 66.7 ± 1.8 |
| hybrid/deepseek/r295-100/k10/th0_7-0 | cti_vsp | total | 5 | 395 | 17.8 | 199.6 | 217.4 | 64.5 ± 2.0 | 2.9 ± 1.1 | 32.6 ± 1.8 | 35.5 ± 2.0 |
| hybrid/llama/r345-50/k10/th0_7-0 | cti_vsp | bb | 4 | 345 | 5.2 | 0 | 5.2 | 98.5 ± 1.5 | 1.5 ± 1.5 | 0.0 ± 0.0 | 1.5 ± 1.5 |
| hybrid/llama/r345-50/k10/th0_7-0 | cti_vsp | fg | 4 | 50 | 0.2 | 205.5 | 205.8 | 19.6 ± 1.2 | 0.1 ± 0.2 | 80.3 ± 1.1 | 80.4 ± 1.2 |
| hybrid/llama/r345-50/k10/th0_7-0 | cti_vsp | total | 4 | 395 | 5.5 | 205.5 | 211 | 65.2 ± 2.0 | 0.9 ± 0.8 | 33.9 ± 1.5 | 34.8 ± 2.0 |
| hybrid/llama/r295-100/k10/th0_7-0 | cti_vsp | bb | 4 | 295 | 2.2 | 0 | 2.2 | 99.2 ± 0.6 | 0.8 ± 0.6 | 0.0 ± 0.0 | 0.8 ± 0.6 |
| hybrid/llama/r295-100/k10/th0_7-0 | cti_vsp | fg | 4 | 100 | 1 | 234.2 | 235.2 | 29.9 ± 1.4 | 0.3 ± 0.2 | 69.8 ± 1.3 | 70.1 ± 1.4 |
| hybrid/llama/r295-100/k10/th0_7-0 | cti_vsp | total | 4 | 395 | 3.2 | 234.2 | 237.5 | 62.5 ± 1.5 | 0.5 ± 0.3 | 37.0 ± 1.6 | 37.5 ± 1.5 |
