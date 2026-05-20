CaseHold benchmark runner

Files:
- `run_casehold_benchmark.py` - CLI runner for plain and ICL evaluation modes.
- `data_loader.py` - TSV loader tolerant to multiple CaseHold formats.
- `prompt_builder.py` - Plain and ICL prompt composition helpers.
- `model_wrapper.py` - Lightweight wrapper around `UnifiedGenerator` for local models.
- `evaluator.py` - Simple evaluator & label extraction utilities.

Quick smoke-run (example):

```bash
python benchmark_play_ground/run_casehold_benchmark.py \
  --model-path /pfs/work9/workspace/scratch/ka_ukhtw-master_thesis_experiment/models/llama-3.1-8b \
  --data-tsv benchmarks/casehold/casehold.tsv \
  --mode plain \
  --max-prompts 5 \
  --device cpu \
  --output-jsonl tmp_casehold_results.jsonl
```

Notes:
- The runner uses the project's `UnifiedGenerator` (in `sae_pretrain/generator_uni.py`) to load the model locally.
- It defaults to deterministic generation (`do_sample=False, temperature=0.0`) and extracts A-E letters from model text.
