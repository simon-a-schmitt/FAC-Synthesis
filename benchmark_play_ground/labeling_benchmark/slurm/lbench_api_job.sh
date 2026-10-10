#!/bin/bash
#SBATCH --job-name=lbench_api                # Labeling-Benchmark eines api-Runs über OpenRouter (kein GPU)
#SBATCH --partition=cpu                      # = CPU_PARTITION in experiments/config/cluster.env (lb_submit.py übergibt sie)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8gb
#SBATCH --time=02:00:00
#SBATCH --output=benchmark_play_ground/labeling_benchmark/logs/%x_%j.out  # relativ zum Submit-Verzeichnis (= Repo-Root)
#SBATCH --error=benchmark_play_ground/labeling_benchmark/logs/%x_%j.err

# ---------------------------------------------------------------------------
# api-Runs ({bench}__api__{model}): derselbe Ablauf wie lbench_job.sh, nur mit CPU-Ressourcen.
#
# Aufruf (aus dem Repo-Root FAC-Synthesis/; normalerweise über lb_submit.py):
#   sbatch benchmark_play_ground/labeling_benchmark/slurm/lbench_api_job.sh <bench>__api__<model>
# ---------------------------------------------------------------------------

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
[[ "${1:-}" == *__api__* ]] || { echo "FEHLER: lbench_api_job.sh ist nur für api-Runs ({bench}__api__{model})." >&2; exit 1; }
exec bash "$REPO/benchmark_play_ground/labeling_benchmark/slurm/lbench_job.sh" "$@"
