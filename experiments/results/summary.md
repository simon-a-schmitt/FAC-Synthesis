# Results

Primary metric per benchmark (mean ± SD over the seed groups present, n = runs): toxicity_detection: `auprc_toxic`, claudette_tos: `weighted_auprc_8`, cti_vsp: `macro_f1`

| arm variant | toxicity_detection | claudette_tos | cti_vsp |
|---|---|---|---|
| plain | 0.536 ± 0.000 (n=1) | 0.601 ± 0.000 (n=1) | 0.476 ± 0.000 (n=1) |
| api/deepseek | 0.342 ± 0.000 (n=1) | 0.470 ± 0.000 (n=1) | 0.730 ± 0.000 (n=1) |
| api/llama | 0.255 ± 0.000 (n=1) | 0.575 ± 0.000 (n=1) | 0.474 ± 0.000 (n=1) |
| api/gpt | 0.745 ± 0.000 (n=1) | 0.746 ± 0.000 (n=1) | 0.512 ± 0.000 (n=1) |
| icl/k5 | 0.714 ± 0.019 (n=5) | 0.612 ± 0.033 (n=5) | 0.588 ± 0.035 (n=5) |
| gold/k5 | 0.866 ± 0.000 (n=1) |  |  |
| bb/deepseek/k5 | 0.674 ± 0.000 (n=1) |  |  |
| bb/llama/k5 | 0.748 ± 0.000 (n=1) |  |  |
| fg/deepseek/k5 | 0.670 ± 0.000 (n=1) |  |  |
| fg/llama/k5 | 0.730 ± 0.000 (n=1) |  |  |
| hybrid/deepseek/r350-50/k5 | 0.641 ± 0.000 (n=1) |  |  |
| hybrid/deepseek/r300-100/k5 | 0.726 ± 0.000 (n=1) |  |  |
| hybrid/llama/r350-50/k5 | 0.674 ± 0.000 (n=1) |  |  |
| hybrid/llama/r300-100/k5 | 0.678 ± 0.000 (n=1) |  |  |
| icl/k10 | 0.728 ± 0.050 (n=5) | 0.611 ± 0.043 (n=5) | 0.651 ± 0.026 (n=5) |

51 run(s) with bench done; per-run values in runs.csv.
