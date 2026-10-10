# Labeling benchmark

Metric: mean ± SD over the seed groups present (n). valid = mean valid_example_rate. Sources: lbench_exact / lbench_lenient = free generation parsed with the labeling / benchmark parser; experiments = outputs of the experiments benchmark (constrained for toxicity/claudette). Invalid predictions count as FN of the gold class, never as FP.

## toxicity_detection (f1_toxic)

| variant | lbench_exact | lbench_lenient | experiments | valid lbench_exact | valid lbench_lenient | valid experiments |
|---|---|---|---|---|---|---|
| plain |  |  | 0.504 (n=1) |  |  | 1.000 |
| icl/k5 |  |  | 0.646 ± 0.044 (n=5) |  |  | 1.000 |
| icl/k10 |  |  | 0.612 ± 0.089 (n=5) |  |  | 1.000 |
| gold/k5 |  |  | 0.764 ± 0.013 (n=5) |  |  | 1.000 |
| gold/k10 |  |  | 0.769 ± 0.006 (n=5) |  |  | 1.000 |
| bb/deepseek/k5/th0_7 |  |  | 0.513 ± 0.036 (n=5) |  |  | 1.000 |
| bb/deepseek/k10/th0_7 |  |  | 0.506 ± 0.051 (n=5) |  |  | 1.000 |
| bb/llama/k5/th0_7 |  |  | 0.546 ± 0.080 (n=5) |  |  | 1.000 |
| bb/llama/k10/th0_7 |  |  | 0.558 ± 0.044 (n=5) |  |  | 1.000 |
| fg/deepseek/k5/th0_7-0 |  |  | 0.324 ± 0.201 (n=5) |  |  | 1.000 |
| fg/deepseek/k10/th0_7-0 |  |  | 0.563 ± 0.115 (n=5) |  |  | 1.000 |
| fg/llama/k5/th0_7-0 |  |  | 0.576 ± 0.099 (n=5) |  |  | 1.000 |
| fg/llama/k10/th0_7-0 |  |  | 0.601 ± 0.037 (n=5) |  |  | 1.000 |
| hybrid/deepseek/r350-50/k5/th0_7-0 |  |  | 0.485 ± 0.036 (n=5) |  |  | 1.000 |
| hybrid/deepseek/r300-100/k5/th0_7-0 |  |  | 0.459 ± 0.132 (n=5) |  |  | 1.000 |
| hybrid/deepseek/r345-50/k10/th0_7-0 |  |  | 0.492 ± 0.049 (n=4) |  |  | 1.000 |
| hybrid/deepseek/r295-100/k10/th0_7-0 |  |  | 0.462 ± 0.044 (n=4) |  |  | 1.000 |
| hybrid/llama/r350-50/k5/th0_7-0 |  |  | 0.567 ± 0.070 (n=5) |  |  | 1.000 |
| hybrid/llama/r300-100/k5/th0_7-0 |  |  | 0.601 ± 0.045 (n=5) |  |  | 1.000 |
| hybrid/llama/r345-50/k10/th0_7-0 |  |  | 0.577 ± 0.065 (n=4) |  |  | 1.000 |
| hybrid/llama/r295-100/k10/th0_7-0 |  |  | 0.599 ± 0.041 (n=4) |  |  | 1.000 |

## claudette_tos (weighted_f1)

| variant | lbench_exact | lbench_lenient | experiments | valid lbench_exact | valid lbench_lenient | valid experiments |
|---|---|---|---|---|---|---|
| plain |  |  | 0.387 (n=1) |  |  | 1.000 |
| icl/k5 |  |  | 0.391 ± 0.019 (n=5) |  |  | 1.000 |
| icl/k10 |  |  | 0.398 ± 0.030 (n=5) |  |  | 1.000 |
| gold/k5 |  |  | 0.618 ± 0.043 (n=5) |  |  | 1.000 |
| gold/k10 |  |  | 0.625 ± 0.034 (n=5) |  |  | 1.000 |
| bb/deepseek/k5/th0_7 |  |  | 0.548 ± 0.013 (n=5) |  |  | 1.000 |
| bb/deepseek/k10/th0_7 |  |  | 0.576 ± 0.027 (n=5) |  |  | 1.000 |
| bb/llama/k5/th0_7 |  |  | 0.460 ± 0.025 (n=5) |  |  | 1.000 |
| bb/llama/k10/th0_7 |  |  | 0.452 ± 0.034 (n=5) |  |  | 1.000 |
| fg/deepseek/k5/th0_7-0 |  |  | 0.549 ± 0.021 (n=5) |  |  | 1.000 |
| fg/deepseek/k10/th0_7-0 |  |  | 0.553 ± 0.022 (n=5) |  |  | 1.000 |
| fg/llama/k5/th0_7-0 |  |  | 0.449 ± 0.021 (n=5) |  |  | 1.000 |
| fg/llama/k10/th0_7-0 |  |  | 0.457 ± 0.008 (n=5) |  |  | 1.000 |
| hybrid/deepseek/r350-50/k5/th0_7-0 |  |  | 0.552 ± 0.014 (n=5) |  |  | 1.000 |
| hybrid/deepseek/r300-100/k5/th0_7-0 |  |  | 0.546 ± 0.014 (n=5) |  |  | 1.000 |
| hybrid/deepseek/r345-50/k10/th0_7-0 |  |  | 0.553 ± 0.041 (n=5) |  |  | 1.000 |
| hybrid/deepseek/r295-100/k10/th0_7-0 |  |  | 0.555 ± 0.036 (n=5) |  |  | 1.000 |
| hybrid/llama/r350-50/k5/th0_7-0 |  |  | 0.459 ± 0.030 (n=5) |  |  | 1.000 |
| hybrid/llama/r300-100/k5/th0_7-0 |  |  | 0.454 ± 0.025 (n=5) |  |  | 1.000 |
| hybrid/llama/r345-50/k10/th0_7-0 |  |  | 0.454 ± 0.013 (n=5) |  |  | 1.000 |
| hybrid/llama/r295-100/k10/th0_7-0 |  |  | 0.457 ± 0.024 (n=5) |  |  | 1.000 |

## cti_vsp (macro_f1)

| variant | lbench_exact | lbench_lenient | experiments | valid lbench_exact | valid lbench_lenient | valid experiments |
|---|---|---|---|---|---|---|
| plain |  |  | 0.479 (n=1) |  |  | 0.990 |
| icl/k5 |  |  | 0.589 ± 0.036 (n=5) |  |  | 0.995 |
| icl/k10 |  |  | 0.652 ± 0.027 (n=5) |  |  | 0.997 |
| gold/k5 |  |  | 0.754 ± 0.004 (n=5) |  |  | 1.000 |
| gold/k10 |  |  | 0.759 ± 0.009 (n=5) |  |  | 1.000 |
| bb/deepseek/k5/th0_7 |  |  | 0.651 ± 0.012 (n=5) |  |  | 1.000 |
| bb/deepseek/k10/th0_7 |  |  | 0.671 ± 0.003 (n=5) |  |  | 1.000 |
| bb/llama/k5/th0_7 |  |  | 0.503 ± 0.008 (n=5) |  |  | 0.999 |
| bb/llama/k10/th0_7 |  |  | 0.511 ± 0.012 (n=5) |  |  | 1.000 |
| fg/deepseek/k5/th0_7-0 |  |  | 0.651 ± 0.018 (n=5) |  |  | 1.000 |
| fg/deepseek/k10/th0_7-0 |  |  | 0.662 ± 0.004 (n=4) |  |  | 0.998 |
| fg/llama/k5/th0_7-0 |  |  | 0.508 ± 0.006 (n=5) |  |  | 1.000 |
| fg/llama/k10/th0_7-0 |  |  | 0.521 ± 0.011 (n=4) |  |  | 1.000 |
| hybrid/deepseek/r350-50/k5/th0_7-0 |  |  | 0.661 ± 0.010 (n=5) |  |  | 1.000 |
| hybrid/deepseek/r300-100/k5/th0_7-0 |  |  | 0.658 ± 0.013 (n=5) |  |  | 1.000 |
| hybrid/deepseek/r345-50/k10/th0_7-0 |  |  | 0.663 ± 0.009 (n=5) |  |  | 1.000 |
| hybrid/deepseek/r295-100/k10/th0_7-0 |  |  | 0.664 ± 0.017 (n=5) |  |  | 1.000 |
| hybrid/llama/r350-50/k5/th0_7-0 |  |  | 0.494 ± 0.009 (n=5) |  |  | 1.000 |
| hybrid/llama/r300-100/k5/th0_7-0 |  |  | 0.505 ± 0.008 (n=5) |  |  | 1.000 |
| hybrid/llama/r345-50/k10/th0_7-0 |  |  | 0.501 ± 0.004 (n=5) |  |  | 1.000 |
| hybrid/llama/r295-100/k10/th0_7-0 |  |  | 0.509 ± 0.012 (n=5) |  |  | 1.000 |
