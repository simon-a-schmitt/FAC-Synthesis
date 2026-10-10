#!/bin/bash
#SBATCH --job-name=lbench                    # Labeling-Benchmark eines Runs (freie Generierung, lokal)
#SBATCH --partition=gpu_a100_short           # = GPU_PARTITION in experiments/config/cluster.env (lb_submit.py übergibt sie)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64gb
#SBATCH --time=00:30:00
#SBATCH --output=benchmark_play_ground/labeling_benchmark/logs/%x_%j.out  # relativ zum Submit-Verzeichnis (= Repo-Root)
#SBATCH --error=benchmark_play_ground/labeling_benchmark/logs/%x_%j.err

# ---------------------------------------------------------------------------
# Ein Run des Labeling-Benchmarks (benchmark_play_ground/labeling_benchmark/lb_runs.py), gesteuert nur
# über die run_id. Nutzt die Modelle aus experiments/runs/<run_id>/ nur lesend; Ausgaben unter
# benchmark_play_ground/labeling_benchmark/runs/<run_id>/. Done-/Stale-/Resume-Logik: run_lbench.py.
#
# Aufruf (aus dem Repo-Root FAC-Synthesis/; normalerweise über lb_submit.py):
#   sbatch benchmark_play_ground/labeling_benchmark/slurm/lbench_job.sh <run_id>
# ---------------------------------------------------------------------------

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

die() { echo "FEHLER: $*" >&2; exit 1; }
[[ $# -eq 1 ]] || die "Aufruf: sbatch benchmark_play_ground/labeling_benchmark/slurm/lbench_job.sh <run_id>"
RUN_ID_ARG="$1"

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
LB="$REPO/benchmark_play_ground/labeling_benchmark"
[[ -f "$LB/run_lbench.py" ]] || die "$REPO ist nicht der Repo-Root (sbatch aus FAC-Synthesis/ aufrufen)."
cd "$REPO"

source "${CLUSTER_ENV:-$REPO/experiments/config/cluster.env}"
set +u; source "$START_LLAMA_SH" >/dev/null; set -u
export HF_HUB_OFFLINE=1

echo "====== JOB ======"
echo "Job ID:  ${SLURM_JOB_ID:-<kein SLURM>}  Partition: ${SLURM_JOB_PARTITION:-}  Node: $(hostname)"
echo "Run:     $RUN_ID_ARG"
echo "Git:     $(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"
if [[ "$RUN_ID_ARG" != *__api__* ]]; then
    nvidia-smi || echo "WARNUNG: nvidia-smi fehlgeschlagen."
fi
echo ""

python "$LB/run_lbench.py" "$RUN_ID_ARG"
echo "[INFO] Job finished. Total: $(( SECONDS / 60 )) min $(( SECONDS % 60 )) s"
