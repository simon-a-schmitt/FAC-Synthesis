# Cost projection (OpenRouter, USD)

All synthetic runs of experiments.yaml (bb / fg / hybrid x setups x seed groups), per benchmark, arm, setup and phase. Rates per example from the billed cost of logged calls (usage.cost); source: current = runs of the current generation config, legacy = older runs of the benchmark (e.g. T=1.0), scaled = toxicity_detection rate x the benchmark's measured cost ratio.

| bench | arm | setup | phase | runs | prompt tok (M) | completion tok (M) | cost USD | rate source |
|---|---|---|---|---|---|---|---|---|
| claudette_tos | bb | deepseek | bb | 10 | 30.18 | 1.80 | 4.17 | current |
| claudette_tos | bb | deepseek | label | 10 | 0.87 | 0.14 | 0.10 | current |
| claudette_tos | bb | llama | bb | 10 | 4.71 | 0.22 | 0.13 | current |
| claudette_tos | bb | llama | label | 10 | 0.84 | 0.13 | 0.20 | current |
| claudette_tos | fg | deepseek | fg | 10 | 7.95 | 0.28 | 1.23 | current |
| claudette_tos | fg | deepseek | label | 10 | 0.87 | 0.14 | 0.10 | current |
| claudette_tos | fg | llama | fg | 10 | 10.44 | 0.35 | 0.31 | current |
| claudette_tos | fg | llama | label | 10 | 0.84 | 0.13 | 0.20 | current |
| claudette_tos | hybrid | deepseek | hybrid_bb | 20 | 39.70 | 2.38 | 5.86 | current |
| claudette_tos | hybrid | deepseek | hybrid_fg | 20 | 6.00 | 0.22 | 0.83 | current |
| claudette_tos | hybrid | deepseek | label | 20 | 1.74 | 0.28 | 0.20 | current |
| claudette_tos | hybrid | llama | hybrid_bb | 20 | 8.73 | 0.38 | 0.41 | current |
| claudette_tos | hybrid | llama | hybrid_fg | 20 | 8.20 | 0.25 | 0.33 | current |
| claudette_tos | hybrid | llama | label | 20 | 1.69 | 0.25 | 0.41 | current |
| cti_vsp | bb | deepseek | bb | 10 | 2.86 | 0.27 | 0.51 | current |
| cti_vsp | bb | deepseek | label | 10 | 0.99 | 0.11 | 0.10 | current |
| cti_vsp | bb | llama | bb | 10 | 2.75 | 0.20 | 0.29 | current |
| cti_vsp | bb | llama | label | 10 | 0.95 | 0.10 | 0.20 | current |
| cti_vsp | fg | deepseek | fg | 10 | 8.34 | 0.60 | 1.48 | current |
| cti_vsp | fg | deepseek | label | 10 | 0.99 | 0.11 | 0.10 | current |
| cti_vsp | fg | llama | fg | 10 | 8.20 | 0.50 | 0.53 | current |
| cti_vsp | fg | llama | label | 10 | 0.95 | 0.10 | 0.20 | current |
| cti_vsp | hybrid | deepseek | hybrid_bb | 20 | 4.64 | 0.42 | 0.82 | current |
| cti_vsp | hybrid | deepseek | hybrid_fg | 20 | 5.44 | 0.39 | 1.01 | current |
| cti_vsp | hybrid | deepseek | label | 20 | 1.97 | 0.22 | 0.20 | current |
| cti_vsp | hybrid | llama | hybrid_bb | 20 | 4.46 | 0.31 | 0.23 | current |
| cti_vsp | hybrid | llama | hybrid_fg | 20 | 6.15 | 0.37 | 0.41 | current |
| cti_vsp | hybrid | llama | label | 20 | 1.90 | 0.21 | 0.41 | current |
| toxicity_detection | bb | deepseek | bb | 10 | 4.28 | 0.17 | 0.56 | current |
| toxicity_detection | bb | deepseek | label | 10 | 0.31 | 0.02 | 0.03 | current |
| toxicity_detection | bb | llama | bb | 10 | 2.83 | 0.09 | 0.18 | current |
| toxicity_detection | bb | llama | label | 10 | 0.33 | 0.01 | 0.06 | current |
| toxicity_detection | fg | deepseek | fg | 10 | 6.43 | 0.20 | 0.93 | current |
| toxicity_detection | fg | deepseek | label | 10 | 0.31 | 0.02 | 0.03 | current |
| toxicity_detection | fg | llama | fg | 10 | 8.30 | 0.28 | 0.72 | current |
| toxicity_detection | fg | llama | label | 10 | 0.33 | 0.01 | 0.06 | current |
| toxicity_detection | hybrid | deepseek | hybrid_bb | 20 | 6.56 | 0.26 | 0.87 | current |
| toxicity_detection | hybrid | deepseek | hybrid_fg | 20 | 4.54 | 0.14 | 0.64 | current |
| toxicity_detection | hybrid | deepseek | label | 20 | 0.63 | 0.03 | 0.06 | current |
| toxicity_detection | hybrid | llama | hybrid_bb | 20 | 4.66 | 0.14 | 0.39 | current |
| toxicity_detection | hybrid | llama | hybrid_fg | 20 | 6.10 | 0.20 | 0.52 | current |
| toxicity_detection | hybrid | llama | label | 20 | 0.65 | 0.02 | 0.11 | current |

| bench | cost USD |
|---|---|
| claudette_tos | 14.47 |
| cti_vsp | 6.50 |
| toxicity_detection | 5.15 |
| **total** | **26.13** |

Measured rates (USD per 1000 examples):

| bench | phase | model | USD/1k | runs | source |
|---|---|---|---|---|---|
| claudette_tos | bb | deepseek/deepseek-v4-flash-0731 | 1.049 | 1 | current |
| claudette_tos | bb | meta-llama/llama-3.1-8b-instruct | 0.033 | 1 | current |
| claudette_tos | fg | deepseek/deepseek-v4-flash-0731 | 0.310 | 1 | current |
| claudette_tos | fg | meta-llama/llama-3.1-8b-instruct | 0.077 | 1 | current |
| claudette_tos | hybrid_bb | deepseek/deepseek-v4-flash-0731 | 0.908 | 2 | current |
| claudette_tos | hybrid_bb | meta-llama/llama-3.1-8b-instruct | 0.063 | 2 | current |
| claudette_tos | hybrid_fg | deepseek/deepseek-v4-flash-0731 | 0.551 | 2 | current |
| claudette_tos | hybrid_fg | meta-llama/llama-3.1-8b-instruct | 0.220 | 2 | current |
| claudette_tos | label | deepseek/deepseek-v4-flash-0731 | 0.025 | 4 | current |
| claudette_tos | label | openai/gpt-4o-mini-2024-07-18 | 0.051 | 4 | current |
| cti_vsp | bb | deepseek/deepseek-v4-flash-0731 | 0.129 | 1 | current |
| cti_vsp | bb | meta-llama/llama-3.1-8b-instruct | 0.073 | 1 | current |
| cti_vsp | fg | deepseek/deepseek-v4-flash-0731 | 0.373 | 1 | current |
| cti_vsp | fg | meta-llama/llama-3.1-8b-instruct | 0.132 | 1 | current |
| cti_vsp | hybrid_bb | deepseek/deepseek-v4-flash-0731 | 0.128 | 2 | current |
| cti_vsp | hybrid_bb | meta-llama/llama-3.1-8b-instruct | 0.036 | 2 | current |
| cti_vsp | hybrid_fg | deepseek/deepseek-v4-flash-0731 | 0.675 | 2 | current |
| cti_vsp | hybrid_fg | meta-llama/llama-3.1-8b-instruct | 0.275 | 2 | current |
| cti_vsp | label | deepseek/deepseek-v4-flash-0731 | 0.025 | 4 | current |
| cti_vsp | label | openai/gpt-4o-mini-2024-07-18 | 0.051 | 4 | current |
| toxicity_detection | bb | deepseek/deepseek-v4-flash | 0.047 | 2 | legacy |
| toxicity_detection | bb | deepseek/deepseek-v4-flash-0731 | 0.142 | 5 | current |
| toxicity_detection | bb | meta-llama/llama-3.1-8b-instruct | 0.044 | 5 | current |
| toxicity_detection | fg | deepseek/deepseek-v4-flash-0731 | 0.235 | 5 | current |
| toxicity_detection | fg | meta-llama/llama-3.1-8b-instruct | 0.180 | 5 | current |
| toxicity_detection | hybrid_bb | deepseek/deepseek-v4-flash-0731 | 0.135 | 10 | current |
| toxicity_detection | hybrid_bb | meta-llama/llama-3.1-8b-instruct | 0.060 | 10 | current |
| toxicity_detection | hybrid_fg | deepseek/deepseek-v4-flash-0731 | 0.429 | 10 | current |
| toxicity_detection | hybrid_fg | meta-llama/llama-3.1-8b-instruct | 0.348 | 10 | current |
| toxicity_detection | label | deepseek/deepseek-v4-flash-0731 | 0.007 | 20 | current |
| toxicity_detection | label | openai/gpt-4o-mini-2024-07-18 | 0.014 | 20 | current |
