# Cost projection (OpenRouter, USD)

All synthetic runs of experiments.yaml (bb / fg / hybrid x setups x seed groups), per benchmark, arm, setup and phase. Rates per example from the billed cost of logged calls (usage.cost); source: current = runs of the current generation config, legacy = older runs of the benchmark (e.g. T=1.0), scaled = toxicity_detection rate x the benchmark's measured cost ratio.

| bench | arm | setup | phase | runs | prompt tok (M) | completion tok (M) | cost USD | rate source |
|---|---|---|---|---|---|---|---|---|
| claudette_tos | bb | deepseek | bb | 10 | 7.37 | 0.44 | 0.31 | legacy |
| claudette_tos | bb | deepseek | label | 10 | 0.96 | 0.14 | 0.11 | legacy |
| claudette_tos | bb | llama | bb | 10 | 3.37 | 0.17 | 0.07 | legacy |
| claudette_tos | bb | llama | label | 10 | 1.28 | 0.05 | 0.22 | scaled x3.87 (label) |
| claudette_tos | fg | deepseek | fg | 10 | 8.04 | 0.30 | 0.46 | legacy |
| claudette_tos | fg | deepseek | label | 10 | 0.96 | 0.14 | 0.11 | legacy |
| claudette_tos | fg | llama | fg | 10 | 11.09 | 0.37 | 0.24 | legacy |
| claudette_tos | fg | llama | label | 10 | 1.28 | 0.05 | 0.22 | scaled x3.87 (label) |
| claudette_tos | hybrid | deepseek | hybrid_bb | 20 | 13.37 | 0.77 | 1.07 | legacy |
| claudette_tos | hybrid | deepseek | hybrid_fg | 20 | 5.88 | 0.26 | 0.37 | legacy |
| claudette_tos | hybrid | deepseek | label | 20 | 1.92 | 0.28 | 0.21 | legacy |
| claudette_tos | hybrid | llama | hybrid_bb | 20 | 2.33 | 0.07 | 0.31 | scaled x0.47 (bb) |
| claudette_tos | hybrid | llama | hybrid_fg | 20 | 3.09 | 0.08 | 0.42 | scaled x0.47 (bb) |
| claudette_tos | hybrid | llama | label | 20 | 2.56 | 0.09 | 0.44 | scaled x3.87 (label) |
| cti_vsp | bb | deepseek | bb | 10 | 2.82 | 0.30 | 0.33 | legacy |
| cti_vsp | bb | deepseek | label | 10 | 0.14 | 0.01 | 0.01 | scaled x0.46 (bb) |
| cti_vsp | bb | llama | bb | 10 | 2.68 | 0.22 | 0.06 | legacy |
| cti_vsp | bb | llama | label | 10 | 0.15 | 0.01 | 0.03 | scaled x0.46 (bb) |
| cti_vsp | fg | deepseek | fg | 10 | 7.30 | 0.54 | 0.37 | legacy |
| cti_vsp | fg | deepseek | label | 10 | 0.14 | 0.01 | 0.01 | scaled x0.46 (bb) |
| cti_vsp | fg | llama | fg | 10 | 8.09 | 0.55 | 0.18 | legacy |
| cti_vsp | fg | llama | label | 10 | 0.15 | 0.01 | 0.03 | scaled x0.46 (bb) |
| cti_vsp | hybrid | deepseek | hybrid_bb | 20 | 4.41 | 0.42 | 0.32 | legacy |
| cti_vsp | hybrid | deepseek | hybrid_fg | 20 | 4.97 | 0.39 | 0.43 | legacy |
| cti_vsp | hybrid | deepseek | label | 20 | 0.28 | 0.01 | 0.03 | scaled x0.46 (bb) |
| cti_vsp | hybrid | llama | hybrid_bb | 20 | 4.30 | 0.36 | 0.10 | legacy |
| cti_vsp | hybrid | llama | hybrid_fg | 20 | 5.12 | 0.35 | 0.12 | legacy |
| cti_vsp | hybrid | llama | label | 20 | 0.30 | 0.01 | 0.05 | scaled x0.46 (bb) |
| toxicity_detection | bb | deepseek | bb | 10 | 4.89 | 0.17 | 0.56 | current |
| toxicity_detection | bb | deepseek | label | 10 | 0.31 | 0.02 | 0.03 | current |
| toxicity_detection | bb | llama | bb | 10 | 2.99 | 0.09 | 0.19 | current |
| toxicity_detection | bb | llama | label | 10 | 0.33 | 0.01 | 0.06 | current |
| toxicity_detection | fg | deepseek | fg | 10 | 6.83 | 0.21 | 0.79 | current |
| toxicity_detection | fg | deepseek | label | 10 | 0.31 | 0.02 | 0.03 | current |
| toxicity_detection | fg | llama | fg | 10 | 8.56 | 0.26 | 1.16 | current |
| toxicity_detection | fg | llama | label | 10 | 0.33 | 0.01 | 0.06 | current |
| toxicity_detection | hybrid | deepseek | hybrid_bb | 20 | 7.58 | 0.27 | 0.79 | current |
| toxicity_detection | hybrid | deepseek | hybrid_fg | 20 | 4.80 | 0.14 | 0.54 | current |
| toxicity_detection | hybrid | deepseek | label | 20 | 0.62 | 0.03 | 0.05 | current |
| toxicity_detection | hybrid | llama | hybrid_bb | 20 | 4.94 | 0.14 | 0.65 | current |
| toxicity_detection | hybrid | llama | hybrid_fg | 20 | 6.55 | 0.18 | 0.89 | current |
| toxicity_detection | hybrid | llama | label | 20 | 0.66 | 0.02 | 0.11 | current |

| bench | cost USD |
|---|---|
| claudette_tos | 4.54 |
| cti_vsp | 2.07 |
| toxicity_detection | 5.91 |
| **total** | **12.53** |

Measured rates (USD per 1000 examples):

| bench | phase | model | USD/1k | runs | source |
|---|---|---|---|---|---|
| claudette_tos | bb | deepseek/deepseek-v4-flash-0731 | 0.077 | 5 | legacy |
| claudette_tos | bb | meta-llama/llama-3.1-8b-instruct | 0.019 | 2 | legacy |
| claudette_tos | fg | deepseek/deepseek-v4-flash-0731 | 0.116 | 1 | legacy |
| claudette_tos | fg | meta-llama/llama-3.1-8b-instruct | 0.060 | 3 | legacy |
| claudette_tos | hybrid_bb | deepseek/deepseek-v4-flash-0731 | 0.166 | 3 | legacy |
| claudette_tos | hybrid_fg | deepseek/deepseek-v4-flash-0731 | 0.246 | 3 | legacy |
| claudette_tos | label | deepseek/deepseek-v4-flash-0731 | 0.027 | 1 | legacy |
| cti_vsp | bb | deepseek/deepseek-v4-flash-0731 | 0.082 | 3 | legacy |
| cti_vsp | bb | meta-llama/llama-3.1-8b-instruct | 0.016 | 2 | legacy |
| cti_vsp | fg | deepseek/deepseek-v4-flash-0731 | 0.094 | 1 | legacy |
| cti_vsp | fg | meta-llama/llama-3.1-8b-instruct | 0.046 | 1 | legacy |
| cti_vsp | hybrid_bb | deepseek/deepseek-v4-flash-0731 | 0.050 | 2 | legacy |
| cti_vsp | hybrid_bb | meta-llama/llama-3.1-8b-instruct | 0.016 | 2 | legacy |
| cti_vsp | hybrid_fg | deepseek/deepseek-v4-flash-0731 | 0.288 | 2 | legacy |
| cti_vsp | hybrid_fg | meta-llama/llama-3.1-8b-instruct | 0.078 | 2 | legacy |
| toxicity_detection | bb | deepseek/deepseek-v4-flash | 0.047 | 2 | legacy |
| toxicity_detection | bb | deepseek/deepseek-v4-flash-0731 | 0.141 | 1 | current |
| toxicity_detection | bb | meta-llama/llama-3.1-8b-instruct | 0.047 | 1 | current |
| toxicity_detection | fg | deepseek/deepseek-v4-flash-0731 | 0.200 | 1 | current |
| toxicity_detection | fg | meta-llama/llama-3.1-8b-instruct | 0.292 | 1 | current |
| toxicity_detection | hybrid_bb | deepseek/deepseek-v4-flash-0731 | 0.123 | 2 | current |
| toxicity_detection | hybrid_bb | meta-llama/llama-3.1-8b-instruct | 0.101 | 2 | current |
| toxicity_detection | hybrid_fg | deepseek/deepseek-v4-flash-0731 | 0.362 | 2 | current |
| toxicity_detection | hybrid_fg | meta-llama/llama-3.1-8b-instruct | 0.591 | 2 | current |
| toxicity_detection | label | deepseek/deepseek-v4-flash-0731 | 0.007 | 4 | current |
| toxicity_detection | label | openai/gpt-4o-mini-2024-07-18 | 0.014 | 5 | current |
