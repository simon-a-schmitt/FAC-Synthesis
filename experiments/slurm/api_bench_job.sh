#!/bin/bash
#SBATCH --job-name=apib                      # Benchmark eines api-Runs über OpenRouter (kein GPU)
#SBATCH --partition=cpu                      # = API_BENCH_PARTITION in cluster.env (#SBATCH kann keine Variablen lesen)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=02:00:00
#SBATCH --output=experiments/logs/%x_%j.out  # relativ zum Submit-Verzeichnis (= Repo-Root)
#SBATCH --error=experiments/logs/%x_%j.err

# ---------------------------------------------------------------------------
# api-Runs ({bench}__api__{model}): derselbe Ablauf wie experiments/slurm/ft_bench_job.sh
# (Done-Logik über stages.py, Fingerprint, bench.done, run_meta, Trunkierungs-Check) - nur die
# Ressourcen im Header unterscheiden sich. Dieses Skript führt ft_bench_job.sh deshalb direkt aus.
#
# Aufruf (aus dem Repo-Root FAC-Synthesis/; normalerweise über experiments/submit.py):
#   sbatch experiments/slurm/api_bench_job.sh <bench>__api__<model>
# ---------------------------------------------------------------------------

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
[[ "$1" == *__api__* ]] || { echo "FEHLER: api_bench_job.sh ist nur für api-Runs ({bench}__api__{model})." >&2; exit 1; }
exec bash "$REPO/experiments/slurm/ft_bench_job.sh" "$@"
