#!/bin/bash
#SBATCH --job-name=lbench                    # Labeling-Benchmark (run_labeling_benchmark.py) über OpenRouter, kein GPU
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8gb
#SBATCH --time=01:00:00
#SBATCH --output=experiments/logs/%x_%j.out  # relativ zum Submit-Verzeichnis (= Repo-Root)
#SBATCH --error=experiments/logs/%x_%j.err

# ---------------------------------------------------------------------------
# Labeling-Benchmark für alle api_models aus experiments.yaml (deepseek, llama, gpt) auf einem
# Benchmark, die ersten <n> Test-Beispiele (Testlauf).
#
# Aufruf (aus dem Repo-Root FAC-Synthesis/):
#   sbatch benchmark_play_ground/labeling_benchmark_job.sh [<benchmark> [<n> [<output-dir>]]]
#   Defaults: cti_vsp, 200, benchmark_play_ground/labeling_benchmark/test_run
#
# Der Testlauf schreibt in ein eigenes Verzeichnis, damit seine Summaries nicht in die
# Gesamt-Summary von labeling_benchmark/ einfließen.
# ---------------------------------------------------------------------------

set -euo pipefail
BENCHMARK="${1:-cti_vsp}"
N_PROMPTS="${2:-200}"
OUTPUT_DIR="${3:-benchmark_play_ground/labeling_benchmark/test_run}"

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
[[ -f "$REPO/benchmark_play_ground/run_labeling_benchmark.py" ]] \
    || { echo "FEHLER: $REPO ist nicht der Repo-Root (sbatch aus FAC-Synthesis/ aufrufen)." >&2; exit 1; }
cd "$REPO"

source "${CLUSTER_ENV:-$REPO/experiments/config/cluster.env}"
set +u; source "$START_LLAMA_SH" >/dev/null; set -u

MAX_CONCURRENT="$(python -c "import yaml; print(yaml.safe_load(open('experiments/config/experiments.yaml'))['api']['benchmark']['max_concurrent_requests'])")"

echo "Benchmark: $BENCHMARK | max prompts: $N_PROMPTS | output: $OUTPUT_DIR | concurrency: $MAX_CONCURRENT"
python -u benchmark_play_ground/run_labeling_benchmark.py \
    --benchmarks "$BENCHMARK" \
    --max-prompts "$N_PROMPTS" \
    --max-concurrent-requests "$MAX_CONCURRENT" \
    --output-dir "$OUTPUT_DIR"
