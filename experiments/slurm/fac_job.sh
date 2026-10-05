#!/bin/bash
#SBATCH --job-name=fac                       # Feature coverage der Trainingsdaten von FT-Runs
#SBATCH --partition=gpu_a100_short           # = GPU_PARTITION in cluster.env (#SBATCH kann keine Variablen lesen)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64gb
#SBATCH --time=00:30:00
#SBATCH --output=experiments/logs/%x_%j.out  # relativ zum Submit-Verzeichnis (= Repo-Root)
#SBATCH --error=experiments/logs/%x_%j.err

# ---------------------------------------------------------------------------
# FAC (experiments/run_fac.py) für einen oder mehrere gold/bb/fg/hybrid-Runs; Ergebnis pro Run in
# experiments/runs/<run_id>/fac.json. Llama + SAE werden einmal pro Benchmark geladen.
#
# Aufruf (aus dem Repo-Root FAC-Synthesis/):
#   sbatch experiments/slurm/fac_job.sh <run_id> [<run_id> ...]
#   sbatch --dependency=afterok:<jobs> experiments/slurm/fac_job.sh ...   # nach label_build
# ---------------------------------------------------------------------------

set -euo pipefail
export PYTHONUNBUFFERED=1
die() { echo "FEHLER: $*" >&2; exit 1; }
[[ $# -ge 1 ]] || die "Aufruf: sbatch experiments/slurm/fac_job.sh <run_id> [<run_id> ...]"

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
[[ -f "$REPO/experiments/runs.py" ]] || die "$REPO ist nicht der Repo-Root (sbatch aus FAC-Synthesis/ aufrufen)."
cd "$REPO"
source "${CLUSTER_ENV:-$REPO/experiments/config/cluster.env}"
set +u; source "$START_LLAMA_SH" >/dev/null; set -u

echo "Job ID: ${SLURM_JOB_ID:-local} | Node: $(hostname) | Runs: $*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || die "Keine GPU sichtbar."
python experiments/run_fac.py "$@"
echo "[INFO] Job finished successfully. Total: $(( SECONDS / 60 )) min $(( SECONDS % 60 )) s"
