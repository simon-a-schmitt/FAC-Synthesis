#!/bin/bash
#SBATCH --job-name=gen                       # Generierung eines Packs von bb/fg/hybrid-Runs
#SBATCH --partition=gpu_a100_short           # = GEN_PARTITION in cluster.env (#SBATCH kann keine Variablen lesen)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120gb
#SBATCH --time=00:30:00                      # = GEN_TIME; längere Runs laufen per Selbst-Requeue weiter
#SBATCH --requeue
#SBATCH --signal=B:USR1@120                  # 120 s vor dem Zeitlimit: Runs sauber beenden + requeue
#SBATCH --open-mode=append                   # Logs eines requeueten Jobs werden fortgeschrieben
#SBATCH --output=experiments/logs/%x_%j.out  # relativ zum Submit-Verzeichnis (= Repo-Root)
#SBATCH --error=experiments/logs/%x_%j.err

# ---------------------------------------------------------------------------
# Stage gen für einen oder mehrere Runs ("Pack") in EINEM Job, gesteuert nur über run_ids. Alle
# Pfade/Parameter kommen aus `experiments/runs.py show <run_id> --format shell`, alles
# Clusterspezifische aus experiments/config/cluster.env (Override: $CLUSTER_ENV).
#
# Aufruf (aus dem Repo-Root FAC-Synthesis/; normalerweise über experiments/submit.py); alle run_ids
# eines Aufrufs brauchen dieselbe GEN_RESOURCE (bb: cpu, fg/hybrid: gpu):
#   gpu (Header-Default = GEN_PARTITION/GEN_TIME; höchstens GEN_PACK_SIZE Runs pro Job):
#     sbatch experiments/slurm/gen_job.sh <fg/hybrid run_id> [<run_id> ...]
#     (Option: --partition=$GPU_LONG_PARTITION --time=... für lange Läufe ohne Requeue)
#   cpu (GEN_CPU_PARTITION, ohne GPU; höchstens GEN_PACK_SIZE_CPU Runs):
#     sbatch --partition=cpu --gres=none --time=$GEN_CPU_TIME experiments/slurm/gen_job.sh <bb run_id> [...]
#
# Selbst-Requeue: SLURM schickt 120 s vor dem Zeitlimit SIGUSR1 an dieses Skript. Es beendet die
# laufenden run_generation.py-Prozesse mit SIGTERM (sie schreiben keine halbe Wave: shared/run_io.py
# verzögert SIGTERM bis die Outputs einer Wave komplett sind) und ruft `scontrol requeue` auf,
# solange SLURM_RESTART_COUNT < GEN_MAX_REQUEUES; sonst Exit != 0. Der requeuete Job (gleiche
# Job-ID, abhängige Jobs warten weiter) setzt per --resume fort, fertige Runs werden übersprungen.
#
# Pro run_id (jeweils eigener Hintergrundprozess, Log experiments/logs/gen/<run_id>_<jobid>.log):
#   done     Generierung abgeschlossen (stages.py gen-done)    -> [SKIP]
#   resume   GEN_CHECKPOINT_JSON existiert                                 -> mit --resume fortsetzen
#   fresh    noch keine Outputs                                            -> neu starten
#   stale    Outputs/Checkpoint unter anderem gen_fingerprint              -> Gen-Outputs, Labeling-
#            TSV/-Logs und lf_data nach *.stale.<ts> verschieben (stages.py gen-move-stale), neu starten
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

case "$RESOURCE" in
    cpu) ALLOWED_PARTITIONS=("$GEN_CPU_PARTITION"); PACK_LIMIT="$GEN_PACK_SIZE_CPU" ;;
    gpu) ALLOWED_PARTITIONS=("$GEN_PARTITION" "$GPU_LONG_PARTITION"); PACK_LIMIT="$GEN_PACK_SIZE" ;;
    *)   die "Unbekannte GEN_RESOURCE '$RESOURCE'." ;;
esac
if [[ -n "${SLURM_JOB_PARTITION:-}" && " ${ALLOWED_PARTITIONS[*]} " != *" $SLURM_JOB_PARTITION "* ]]; then
    die "GEN_RESOURCE=$RESOURCE gehört auf Partition ${ALLOWED_PARTITIONS[*]}, Job läuft auf $SLURM_JOB_PARTITION" \
        "(cpu: sbatch --partition=$GEN_CPU_PARTITION --gres=none ...)."
fi
if (( ${#RUN_IDS[@]} > PACK_LIMIT )); then
    echo "WARNUNG: ${#RUN_IDS[@]} Runs > Pack-Limit $PACK_LIMIT für $RESOURCE (cluster.env)."
fi
RESTART_COUNT="${SLURM_RESTART_COUNT:-0}"
PACK_JSON="[$(printf '"%s",' "${RUN_IDS[@]}" | sed 's/,$//')]"

echo "====== JOB ======"
echo "Job ID:     $JOB_ID"
echo "Partition:  ${SLURM_JOB_PARTITION:-}"
echo "Node:       $(hostname)"
echo "Cluster:    $CLUSTER_ENV"
echo "Ressource:  $RESOURCE"
echo "Pack:       ${RUN_IDS[*]}"
echo "Requeues:   $RESTART_COUNT/$GEN_MAX_REQUEUES"
echo "Start:      $(date -Iseconds)"
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

    state="$(python experiments/stages.py gen-state "$run_id" 2>>"$log")" || true
    case "$state" in
        done)    echo "[SKIP] $run_id: Generierung abgeschlossen."; status "skip"; return 0 ;;
        resume)  local resume=(--resume) ;;
        fresh)   local resume=() ;;
        stale)   echo "[STALE] $run_id: Config geändert (gen_fingerprint, siehe $log) -> Artefakte beiseite, neu generieren."
                 python experiments/stages.py gen-move-stale "$run_id" | tee -a "$log"
                 local resume=() ;;
        partial) echo "FEHLER: $run_id: Teil-Outputs ohne Checkpoint, kein Fortsetzen möglich (siehe $log)." >&2
                 status "partial"; return 1 ;;
        *)       echo "FEHLER: $run_id: Generierungs-Zustand unbekannt ('$state', siehe $log)." >&2
                 status "error"; return 1 ;;
    esac

    local args=(
        --domain "$BENCH" --seed-set "$SEED_SET" --seed-group "$SEED_GROUP"
        --rouge-threshold "$ROUGE_THRESHOLD" --model "$GEN_MODEL" --prefix "$GEN_PREFIX"
        --temperature "$GEN_TEMPERATURE" --top-p "$GEN_TOP_P" --max-tokens "$GEN_MAX_TOKENS"
        --max-concurrent-requests "$GEN_MAX_CONCURRENT" --gen-fingerprint "$GEN_FINGERPRINT"
        --env-file "$REPO/.env"
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
    if ! python experiments/stages.py gen-done "$run_id" >> "$log" 2>&1; then
        echo "FEHLER: $run_id: Exit 0, aber Generierung nicht abgeschlossen (Call-Limit? siehe $log)." >&2
        status "incomplete:0:$wall"; return 1
    fi
    python experiments/run_meta.py "$run_id" --stage gen --started-at "$started_at" \
        --finished-at "$(date -Iseconds)" --wall-seconds "$wall" --extra-json "{\"pack\": $PACK_JSON}" >> "$log"
    echo "[INFO] $run_id: fertig nach $(( wall / 60 )) min $(( wall % 60 )) s."
    status "ok:0:$wall"
}

declare -A PIDS
REQUEUE_REQUESTED=0
# SIGUSR1 (Zeitlimit naht): run_generation.py-Prozesse (Kinder der run_one-Subshells) beenden.
on_usr1() {
    echo "[SIGNAL] $(date -Iseconds) USR1: Zeitlimit naht -> laufende Runs mit SIGTERM beenden, dann Requeue."
    REQUEUE_REQUESTED=1
    local pid
    for pid in "${PIDS[@]}"; do
        pkill -TERM -P "$pid" 2>/dev/null || true
    done
}
trap on_usr1 USR1

for run_id in "${RUN_IDS[@]}"; do
    run_one "$run_id" &
    PIDS[$run_id]=$!
done

FAILED=()
for run_id in "${RUN_IDS[@]}"; do
    # Ein Signal mit Trap unterbricht `wait`; weiterwarten, solange der Prozess lebt.
    while true; do
        rc=0; wait "${PIDS[$run_id]}" || rc=$?
        kill -0 "${PIDS[$run_id]}" 2>/dev/null || break
    done
    (( rc == 0 )) || FAILED+=("$run_id")
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

if (( REQUEUE_REQUESTED && ${#FAILED[@]} )); then
    [[ -n "${SLURM_JOB_ID:-}" ]] || die "USR1 ohne SLURM-Job - kein Requeue möglich."
    if (( RESTART_COUNT < GEN_MAX_REQUEUES )); then
        echo "[REQUEUE] $(date -Iseconds) Unfertig: ${FAILED[*]} -> scontrol requeue $SLURM_JOB_ID" \
             "($(( RESTART_COUNT + 1 ))/$GEN_MAX_REQUEUES)."
        scontrol requeue "$SLURM_JOB_ID"
        exit 0
    fi
    die "Requeue-Limit GEN_MAX_REQUEUES=$GEN_MAX_REQUEUES erreicht, unfertig: ${FAILED[*]}"
fi
if (( ${#FAILED[@]} )); then
    die "Fehlgeschlagene Runs (${#FAILED[@]}): ${FAILED[*]}"
fi
echo "[INFO] Job finished successfully. Total: $(( SECONDS / 60 )) min $(( SECONDS % 60 )) s"
