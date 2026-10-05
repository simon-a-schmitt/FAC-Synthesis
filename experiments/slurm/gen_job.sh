#!/bin/bash
#SBATCH --job-name=gen                       # Generierung eines Packs von bb/fg/hybrid-Runs
#SBATCH --partition=gpu_a100_il              # = GPU_LONG_PARTITION in cluster.env (#SBATCH kann keine Variablen lesen)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120gb
#SBATCH --time=08:00:00
#SBATCH --output=experiments/logs/%x_%j.out  # relativ zum Submit-Verzeichnis (= Repo-Root)
#SBATCH --error=experiments/logs/%x_%j.err

# ---------------------------------------------------------------------------
# Stage gen für einen oder mehrere Runs ("Pack") in EINEM Job, gesteuert nur über run_ids. Alle
# Pfade/Parameter kommen aus `experiments/runs.py show <run_id> --format shell`, alles
# Clusterspezifische aus experiments/config/cluster.env (Override: $CLUSTER_ENV).
#
# Aufruf (aus dem Repo-Root FAC-Synthesis/); alle run_ids eines Aufrufs brauchen dieselbe
# GEN_RESOURCE (bb: cpu, fg/hybrid: gpu):
#   gpu (Header-Default, GPU_LONG_PARTITION; höchstens GEN_PACK_SIZE Runs pro Job):
#     sbatch experiments/slurm/gen_job.sh <fg/hybrid run_id> [<run_id> ...]
#   gpu, kurze Runs (< 30 min, z.B. toxicity-fg) auf GPU_PARTITION (A100 40GB -> höchstens 2 Runs):
#     sbatch --partition=gpu_a100_short --time=00:30:00 experiments/slurm/gen_job.sh <run_id> [<run_id>]
#   cpu (CPU_PARTITION, ohne GPU):
#     sbatch --partition=cpu --gres=none experiments/slurm/gen_job.sh <bb run_id> [<run_id> ...]
#   sbatch --time=... überschreibt das Zeitlimit; ein Timeout ist unkritisch (s.u. resume).
#
# Pro run_id (jeweils eigener Hintergrundprozess, Log experiments/logs/gen/<run_id>_<jobid>.log):
#   done     Generierung abgeschlossen (label_build_checks.py gen-done)    -> [SKIP]
#   resume   GEN_CHECKPOINT_JSON existiert                                 -> mit --resume fortsetzen
#   fresh    noch keine Outputs                                            -> neu starten
#   partial  Outputs ohne Checkpoint, nicht abgeschlossen                  -> Fehler für diesen Run
#            (die übrigen Runs laufen weiter)
#   Nach Exit 0 muss gen-done bestehen (ein Lauf, der am Call-Limit stoppt, endet mit Exit 0 und
#   behält seinen Checkpoint); dann run_meta.py stage gen (wall_seconds des Prozesses, pack).
# Am Ende: Übersicht je Run; Exit != 0, wenn ein Run fehlschlug (Liste der run_ids).
# Auf GPU wird nvidia-smi alle 60 s nach experiments/logs/gen/gpu_<jobid>.csv geloggt.
# ---------------------------------------------------------------------------

set -euo pipefail
export PYTHONUNBUFFERED=1

die() { echo "FEHLER: $*" >&2; exit 1; }

[[ $# -ge 1 ]] || die "Aufruf: sbatch experiments/slurm/gen_job.sh <run_id> [<run_id> ...]"
RUN_IDS=("$@")

# Unter SLURM liegt das Skript in einer Spool-Kopie; das Repo ist das Submit-Verzeichnis.
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
[[ -f "$REPO/experiments/runs.py" ]] || die "$REPO ist nicht der Repo-Root (sbatch aus FAC-Synthesis/ aufrufen)."
cd "$REPO"

CLUSTER_ENV="${CLUSTER_ENV:-$REPO/experiments/config/cluster.env}"
[[ -f "$CLUSTER_ENV" ]] || die "Cluster-Konfiguration $CLUSTER_ENV fehlt."
source "$CLUSTER_ENV"

set +u; source "$START_LLAMA_SH" >/dev/null; set -u

JOB_ID="${SLURM_JOB_ID:-local}"
LOG_DIR="$REPO/experiments/logs/gen"
mkdir -p "$LOG_DIR"

# ---- Runs auflösen, Ressource prüfen ----------------------------------------------------
declare -A RUN_SHELL
RESOURCE=""
for run_id in "${RUN_IDS[@]}"; do
    [[ -z "${RUN_SHELL[$run_id]:-}" ]] || die "run_id $run_id doppelt angegeben."
    RUN_SHELL[$run_id]="$(python experiments/runs.py show "$run_id" --format shell)" \
        || die "runs.py konnte run_id '$run_id' nicht auflösen."
    res="$(eval "${RUN_SHELL[$run_id]}"; [[ " ${STAGES[*]} " == *" gen "* ]] && echo "$GEN_RESOURCE")" \
        || die "Run $run_id hat keine Stage 'gen'."
    [[ -z "$RESOURCE" || "$res" == "$RESOURCE" ]] \
        || die "Gemischte GEN_RESOURCE im Pack ($RESOURCE vs. $res bei $run_id) - getrennt einreichen."
    RESOURCE="$res"
done

# gpu: GPU_LONG_PARTITION, für kurze Runs (< 30 min) auch GPU_PARTITION.
case "$RESOURCE" in
    cpu) ALLOWED_PARTITIONS=("$CPU_PARTITION") ;;
    gpu) ALLOWED_PARTITIONS=("$GPU_LONG_PARTITION" "$GPU_PARTITION") ;;
    *)   die "Unbekannte GEN_RESOURCE '$RESOURCE'." ;;
esac
if [[ -n "${SLURM_JOB_PARTITION:-}" && " ${ALLOWED_PARTITIONS[*]} " != *" $SLURM_JOB_PARTITION "* ]]; then
    die "GEN_RESOURCE=$RESOURCE gehört auf Partition ${ALLOWED_PARTITIONS[*]}, Job läuft auf $SLURM_JOB_PARTITION" \
        "(cpu: sbatch --partition=$CPU_PARTITION --gres=none ...)."
fi
if [[ "$RESOURCE" == "gpu" ]] && (( ${#RUN_IDS[@]} > GEN_PACK_SIZE )); then
    echo "WARNUNG: ${#RUN_IDS[@]} Runs > GEN_PACK_SIZE=$GEN_PACK_SIZE - GPU-Speicher könnte nicht reichen."
fi
PACK_JSON="[$(printf '"%s",' "${RUN_IDS[@]}" | sed 's/,$//')]"

echo "====== JOB ======"
echo "Job ID:     $JOB_ID"
echo "Partition:  ${SLURM_JOB_PARTITION:-}"
echo "Node:       $(hostname)"
echo "Cluster:    $CLUSTER_ENV"
echo "Ressource:  $RESOURCE"
echo "Pack:       ${RUN_IDS[*]}"
echo ""

GPU_MONITOR_PID=""
if [[ "$RESOURCE" == "gpu" ]]; then
    nvidia-smi || die "Keine GPU sichtbar, GEN_RESOURCE=gpu braucht eine."
    nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu \
        --format=csv -l 60 > "$LOG_DIR/gpu_${JOB_ID}.csv" &
    GPU_MONITOR_PID=$!
    trap '[[ -n "$GPU_MONITOR_PID" ]] && kill "$GPU_MONITOR_PID" 2>/dev/null || true' EXIT
fi

# ---- Ein Run (läuft als Hintergrundprozess) -----------------------------------------------
# Exit: 0 = fertig/übersprungen, sonst Fehler. Schreibt eine Statuszeile nach $LOG_DIR/.status_<jobid>_<run_id>.
run_one() {
    local run_id="$1" log state started_at t0 wall rc=0
    eval "${RUN_SHELL[$run_id]}"
    log="$LOG_DIR/${run_id}_${JOB_ID}.log"
    status() { echo "$1" > "$LOG_DIR/.status_${JOB_ID}_${run_id}"; }

    state="$(python experiments/label_build_checks.py gen-state "$run_id" 2>>"$log")" || true
    case "$state" in
        done)    echo "[SKIP] $run_id: Generierung abgeschlossen."; status "skip"; return 0 ;;
        resume)  local resume=(--resume) ;;
        fresh)   local resume=() ;;
        partial) echo "FEHLER: $run_id: Teil-Outputs ohne Checkpoint, kein Fortsetzen möglich (siehe $log)." >&2
                 status "partial"; return 1 ;;
        *)       echo "FEHLER: $run_id: Generierungs-Zustand unbekannt ('$state', siehe $log)." >&2
                 status "error"; return 1 ;;
    esac

    local args=(
        --domain "$BENCH" --seed-set "$SEED_SET" --seed-group "$SEED_GROUP"
        --rouge-threshold "$ROUGE_THRESHOLD" --model "$GEN_MODEL" --prefix "$GEN_PREFIX"
        --max-concurrent-requests "$GEN_MAX_CONCURRENT" --env-file "$REPO/.env"
    )
    case "$ARM" in
        bb)     args+=(--n "$N_SYNTHETIC") ;;
        fg)     args+=(--n "$N_SYNTHETIC") ;;
        hybrid) args+=(--n-blackbox "$N_BLACKBOX" --n-feature-guided "$N_FEATURE_GUIDED") ;;
    esac
    if [[ "$ARM" == "fg" || "$ARM" == "hybrid" ]]; then
        args+=(--threshold "$ACTIVATION_THRESHOLD" --model-name "$BASE_MODEL" --sae-ckpt-path "$SAE_CKPT")
    fi
    args+=("${resume[@]}")

    echo "[INFO] $run_id: start ($state) -> $log"
    started_at="$(date -Iseconds)"; t0=$SECONDS
    {
        echo "run_id=$run_id job=$JOB_ID state=$state started_at=$started_at"
        echo "python data_synthesis/$GEN_ARM_DIR/run_generation.py ${args[*]}"
    } >> "$log"
    python "data_synthesis/$GEN_ARM_DIR/run_generation.py" "${args[@]}" >> "$log" 2>&1 || rc=$?
    wall=$(( SECONDS - t0 ))

    if (( rc != 0 )); then
        echo "FEHLER: $run_id: run_generation.py Exit $rc nach ${wall}s (siehe $log)." >&2
        status "failed:$rc:$wall"; return 1
    fi
    if ! python experiments/label_build_checks.py gen-done "$run_id" >> "$log" 2>&1; then
        echo "FEHLER: $run_id: Exit 0, aber Generierung nicht abgeschlossen (Call-Limit? siehe $log)." >&2
        status "incomplete:0:$wall"; return 1
    fi
    python experiments/run_meta.py "$run_id" --stage gen --started-at "$started_at" \
        --finished-at "$(date -Iseconds)" --wall-seconds "$wall" --extra-json "{\"pack\": $PACK_JSON}" >> "$log"
    echo "[INFO] $run_id: fertig nach $(( wall / 60 )) min $(( wall % 60 )) s."
    status "ok:0:$wall"
}

declare -A PIDS
for run_id in "${RUN_IDS[@]}"; do
    run_one "$run_id" &
    PIDS[$run_id]=$!
done

FAILED=()
for run_id in "${RUN_IDS[@]}"; do
    wait "${PIDS[$run_id]}" || FAILED+=("$run_id")
done

# ---- Übersicht -------------------------------------------------------------------------------
echo ""
echo "====== PACK-ÜBERSICHT ======"
printf '%-50s %-12s %8s %6s %8s\n' run_id status wall_s n_429 n_retry
for run_id in "${RUN_IDS[@]}"; do
    log="$LOG_DIR/${run_id}_${JOB_ID}.log"
    IFS=: read -r st _ wall < "$LOG_DIR/.status_${JOB_ID}_${run_id}" 2>/dev/null || st="?"
    n429="$(grep -c "HTTP Error 429" "$log" 2>/dev/null || true)"
    nretry="$(grep -c "OpenRouter chat request failed (attempt" "$log" 2>/dev/null || true)"
    printf '%-50s %-12s %8s %6s %8s\n' "$run_id" "$st" "${wall:--}" "${n429:-0}" "${nretry:-0}"
    rm -f "$LOG_DIR/.status_${JOB_ID}_${run_id}"
done
if [[ -n "$GPU_MONITOR_PID" ]]; then
    peak="$(awk -F', ' 'NR > 1 {gsub(/ MiB/, "", $3); if ($3 + 0 > max) max = $3 + 0} END {print max + 0}' "$LOG_DIR/gpu_${JOB_ID}.csv")"
    echo "GPU-Speicher peak (60-s-Samples): ${peak} MiB  ($LOG_DIR/gpu_${JOB_ID}.csv)"
fi

if (( ${#FAILED[@]} )); then
    die "Fehlgeschlagene Runs (${#FAILED[@]}): ${FAILED[*]}"
fi
echo "[INFO] Job finished successfully. Total: $(( SECONDS / 60 )) min $(( SECONDS % 60 )) s"
