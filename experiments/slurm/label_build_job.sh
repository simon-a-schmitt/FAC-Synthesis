#!/bin/bash
#SBATCH --job-name=lb                        # Labeling + LLaMA-Factory-Dataset eines Runs
#SBATCH --partition=cpu                      # = CPU_PARTITION in cluster.env (#SBATCH kann keine Variablen lesen)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8gb
#SBATCH --time=00:30:00
#SBATCH --output=experiments/logs/%x_%j.out  # relativ zum Submit-Verzeichnis (= Repo-Root)
#SBATCH --error=experiments/logs/%x_%j.err

# ---------------------------------------------------------------------------
# Stage label_build eines bb/fg/hybrid-Runs, gesteuert nur über die run_id. Alle Pfade/Parameter
# kommen aus `experiments/runs.py show <run_id> --format shell`, alles Clusterspezifische aus
# experiments/config/cluster.env (Override: $CLUSTER_ENV).
#
# Aufruf (aus dem Repo-Root FAC-Synthesis/):
#   sbatch experiments/slurm/label_build_job.sh <run_id>
#   sbatch --time=01:00:00 --partition=<CPU_PARTITION> experiments/slurm/label_build_job.sh <run_id>
#
# Ablauf:
#   Erledigt (-> [SKIP]): experiments/stages.py label-build-done (Dataset da, Gate grün).
#   Vorbedingung: Generierung abgeschlossen (experiments/stages.py gen-done), sonst
#                 Abbruch ohne Labeling.
#   Labeling:     data_synthesis/labeling/run_labeling.py (fortsetzbar), bis zu $MAX_LABEL_PASSES
#                 Durchläufe, solange Labels fehlen; danach Vollständigkeit (stages.py
#                 labels: Zeilen == n_total, Majority-Fallback <= 2 %), sonst Exit != 0.
#   Dataset:      experiments/prepare_lf_dataset.py <run_id> (Gate).
#   Meta:         experiments/run_meta.py, stage label_build.
# ---------------------------------------------------------------------------

set -euo pipefail
export PYTHONUNBUFFERED=1

MAX_LABEL_PASSES=3

die() { echo "FEHLER: $*" >&2; exit 1; }

[[ $# -eq 1 ]] || die "Aufruf: sbatch experiments/slurm/label_build_job.sh <run_id>"
RUN_ID_ARG="$1"

# Unter SLURM liegt das Skript in einer Spool-Kopie; das Repo ist das Submit-Verzeichnis.
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
[[ -f "$REPO/experiments/runs.py" ]] || die "$REPO ist nicht der Repo-Root (sbatch aus FAC-Synthesis/ aufrufen)."
cd "$REPO"

# ---- Cluster + Run auflösen ---------------------------------------------------------
CLUSTER_ENV="${CLUSTER_ENV:-$REPO/experiments/config/cluster.env}"
[[ -f "$CLUSTER_ENV" ]] || die "Cluster-Konfiguration $CLUSTER_ENV fehlt."
source "$CLUSTER_ENV"

set +u; source "$START_LLAMA_SH" >/dev/null; set -u

RUN_SHELL="$(python experiments/runs.py show "$RUN_ID_ARG" --format shell)" \
    || die "runs.py konnte run_id '$RUN_ID_ARG' nicht auflösen."
eval "$RUN_SHELL"

[[ " ${STAGES[*]} " == *" label_build "* ]] \
    || die "Run $RUN_ID hat keine Stage 'label_build' (stages: ${STAGES[*]})."

echo "====== JOB ======"
echo "Job ID:     ${SLURM_JOB_ID:-<kein SLURM>}"
echo "Partition:  ${SLURM_JOB_PARTITION:-}"
echo "Node:       $(hostname)"
echo "Cluster:    $CLUSTER_ENV"
echo "Run:        $RUN_ID ($ARM, stages: ${STAGES[*]})"
echo "Input:      $GEN_ACCEPTED_JSON"
echo "Labels:     $LABEL_TSV ($LABEL_MODEL, $LABEL_MAX_CONCURRENT parallel)"
echo "Dataset:    $LF_DATASET_JSON"
echo ""

if python experiments/stages.py label-build-done "$RUN_ID"; then
    echo "[SKIP] label_build: Dataset vorhanden, Gate grün."
    exit 0
fi

python experiments/stages.py gen-done "$RUN_ID" \
    || die "Generierung von $RUN_ID nicht abgeschlossen (siehe oben) - kein Labeling."

started_at="$(date -Iseconds)"; t0=$SECONDS
mkdir -p "$RUN_DIR"

# ---- Labeling --------------------------------------------------------------------------
# run_labeling.py überspringt bereits gelabelte Texte; ein weiterer Durchlauf holt nur nach, was
# fehlt (z.B. nach einem Abbruch, die TSV wird erst am Ende eines Durchlaufs geschrieben).
for (( pass = 1; pass <= MAX_LABEL_PASSES; pass++ )); do
    if python experiments/stages.py labels "$RUN_ID" >/dev/null; then
        break
    fi
    echo "[INFO] Labeling-Durchlauf $pass/$MAX_LABEL_PASSES ..."
    python data_synthesis/labeling/run_labeling.py \
        --source "$LABEL_SOURCE" \
        --domain "$BENCH" \
        --input-json "$GEN_ACCEPTED_JSON" \
        --model "$LABEL_MODEL" \
        --max-concurrent-requests "$LABEL_MAX_CONCURRENT" \
        --env-file "$REPO/.env" \
        || echo "WARNUNG: run_labeling.py (Durchlauf $pass) mit Exit $? beendet."
done
python experiments/stages.py labels "$RUN_ID" \
    || die "Labels von $RUN_ID unvollständig nach $MAX_LABEL_PASSES Durchläufen (siehe oben; Fehler: $LABEL_FAILED_JSON)."

# ---- LLaMA-Factory-Dataset (Gate) ---------------------------------------------------------
python experiments/prepare_lf_dataset.py "$RUN_ID"

wall=$(( SECONDS - t0 ))
echo "[TIME] label_build: $(( wall / 60 )) min $(( wall % 60 )) s"
n_fallback="$(python experiments/stages.py fallbacks "$RUN_ID")"
python experiments/run_meta.py "$RUN_ID" --stage label_build \
    --started-at "$started_at" --finished-at "$(date -Iseconds)" --wall-seconds "$wall" \
    --extra-json "{\"n_majority_fallback\": $n_fallback}"

echo "[INFO] Job finished successfully. Total: $(( SECONDS / 60 )) min $(( SECONDS % 60 )) s"
