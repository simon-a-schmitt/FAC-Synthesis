#!/bin/bash
#SBATCH --job-name=ftb                       # Fine-Tuning + Benchmark eines Runs
#SBATCH --partition=gpu_a100_short           # = GPU_PARTITION in cluster.env (#SBATCH kann keine Variablen lesen)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=120gb
#SBATCH --time=00:30:00
#SBATCH --output=experiments/logs/%x_%j.out  # relativ zum Submit-Verzeichnis (= Repo-Root)
#SBATCH --error=experiments/logs/%x_%j.err

# ---------------------------------------------------------------------------
# Ein generischer Job für alle Arme (plain, icl, gold, bb, fg, hybrid), gesteuert nur über die
# run_id. Alle Pfade/Parameter kommen aus `experiments/runs.py show <run_id> --format shell`,
# alles Clusterspezifische aus experiments/config/cluster.env (Override: $CLUSTER_ENV).
#
# Aufruf (aus dem Repo-Root FAC-Synthesis/):
#   sbatch experiments/slurm/ft_bench_job.sh <run_id>
#   sbatch --time=01:00:00 --partition=<GPU_PARTITION> experiments/slurm/ft_bench_job.sh <run_id>
#   sbatch --export=ALL,CLUSTER_ENV=experiments/config/cluster_<name>.env experiments/slurm/ft_bench_job.sh <run_id>
#
# Phasen (jede in eigener Subshell mit eigener Umgebung; jede wird übersprungen, wenn erledigt):
#   FT     nur wenn "ft_bench" in STAGES. Erledigt: LORA_DIR/adapter_model.safetensors UND
#          FT_LOG_JSON existieren (FT_LOG_JSON wird als LETZTES geschrieben = Done-Marker) UND
#          dessen ft_fingerprint == FT_FINGERPRINT (runs.py); sonst [STALE] -> neu trainieren.
#          gold: Dataset via experiments/prepare_lf_dataset.py; bb/fg/hybrid: LF_DATASET_JSON
#          muss bereits existieren (Schritt label_build).
#          Vor dem Training werden BENCH_JSONL und bench.done nach *.stale.<timestamp> verschoben,
#          damit --resume nie Vorhersagen eines alten Adapters übernimmt.
#   Bench  Erledigt: RUN_DIR/bench.done existiert mit bench_fingerprint == BENCH_FINGERPRINT;
#          sonst [STALE]: bench.done + BENCH_JSONL nach *.stale.<timestamp>, neu rechnen.
#          Läuft mit --resume; ein unfertiges BENCH_JSONL wird nur fortgesetzt, wenn bench.started
#          denselben Fingerprint trägt. Marker ohne Fingerprint (ältere Läufe) gelten als stale.
#   Danach wird RUN_META_JSON gemergt (pro Phase Job-ID, Node, GPU, Zeiten, git commit;
#   einmalig sha256 von TEST_TSV und SEED_FILE).
#
# Voraussetzung: LLaMA-Factory-Patch in software/LLaMA-Factory/src/llamafactory/train/sft/
# workflow.py, Zeile 160 (create_modelcard_and_push).
# ---------------------------------------------------------------------------

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

die() { echo "FEHLER: $*" >&2; exit 1; }

[[ $# -eq 1 ]] || die "Aufruf: sbatch experiments/slurm/ft_bench_job.sh <run_id>"
RUN_ID_ARG="$1"

# Unter SLURM liegt das Skript in einer Spool-Kopie; das Repo ist das Submit-Verzeichnis.
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
[[ -f "$REPO/experiments/runs.py" ]] || die "$REPO ist nicht der Repo-Root (sbatch aus FAC-Synthesis/ aufrufen)."
cd "$REPO"

# ---- a) Cluster + Run auflösen ------------------------------------------------
CLUSTER_ENV="${CLUSTER_ENV:-$REPO/experiments/config/cluster.env}"
[[ -f "$CLUSTER_ENV" ]] || die "Cluster-Konfiguration $CLUSTER_ENV fehlt."
source "$CLUSTER_ENV"

# Orchestrierungs-Python (runs.py, prepare, Meta) = Benchmark-Umgebung; die Phasen aktivieren
# in ihren Subshells jeweils ihre eigene Umgebung.
set +u; source "$START_LLAMA_SH" >/dev/null; set -u

RUN_SHELL="$(python experiments/runs.py show "$RUN_ID_ARG" --format shell)" \
    || die "runs.py konnte run_id '$RUN_ID_ARG' nicht auflösen."
eval "$RUN_SHELL"
mkdir -p "$RUN_DIR"

has_stage() { [[ " ${STAGES[*]} " == *" $1 "* ]]; }
GIT_COMMIT="$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"
GIT_DIRTY="$([[ -n "$(git -C "$REPO" status --porcelain --untracked-files=no 2>/dev/null)" ]] && echo 1 || echo 0)"
BENCH_DONE="$RUN_DIR/bench.done"
BENCH_STARTED="$RUN_DIR/bench.started"   # Fingerprint der Config, unter der BENCH_JSONL begonnen wurde

echo "====== JOB ======"
echo "Job ID:     ${SLURM_JOB_ID:-<kein SLURM>}"
echo "Partition:  ${SLURM_JOB_PARTITION:-}"
echo "Node:       $(hostname)"
echo "Cluster:    $CLUSTER_ENV"
echo "Run:        $RUN_ID ($ARM, stages: ${STAGES[*]})"
echo "Run-Dir:    $RUN_DIR"
echo "Git:        $GIT_COMMIT (dirty=$GIT_DIRTY)"
echo ""

gpu_diagnostics() {
    echo "====== GPU DIAGNOSTICS ======"
    echo "Active python: $(command -v python)"
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<leer>}"
    nvidia-smi || echo "WARNUNG: nvidia-smi fehlgeschlagen."
    python -c "import torch; print('torch:', torch.__version__, '| cuda:', torch.version.cuda, '| available:', torch.cuda.is_available(), '| devices:', torch.cuda.device_count())"
    echo ""
}

# ---- d) RUN_META_JSON mergen (nach jeder ausgeführten Phase), siehe experiments/run_meta.py ----
# Args: <stage> <started_at> <finished_at> <wall_seconds>
merge_run_meta() {
    python experiments/run_meta.py "$RUN_ID" --stage "$1" --started-at "$2" --finished-at "$3" --wall-seconds "$4"
}

# Args: <stage> <function>
run_phase() {
    local stage="$1" fn="$2" started_at t0
    started_at="$(date -Iseconds)"; t0=$SECONDS
    ( "$fn" )
    local wall=$(( SECONDS - t0 ))
    echo "[TIME] $stage: $(( wall / 60 )) min $(( wall % 60 )) s"
    merge_run_meta "$stage" "$started_at" "$(date -Iseconds)" "$wall"
    echo ""
}

# Fingerprint, den ein Done-Marker (JSON) gespeichert hat; leer, wenn keiner/kein Marker.
# Args: <marker.json> <key>
marker_fingerprint() {
    python3 -c 'import json, sys
try:
    print(json.load(open(sys.argv[1])).get(sys.argv[2]) or "")
except (OSError, ValueError):
    print("")' "$1" "$2"
}

# Verschiebt vorhandene Dateien nach <datei>.stale.<timestamp> (nie löschen).
move_stale() {
    local ts f; ts="$(date +%Y%m%d-%H%M%S)"
    for f in "$@"; do
        if [[ -e "$f" ]]; then
            mv "$f" "$f.stale.$ts"
            echo "[INFO] Veraltet, verschoben: $f -> $f.stale.$ts"
        fi
    done
}

# ======================= PHASE FT ==========================================
ft_done() {
    [[ -f "$LORA_DIR/adapter_model.safetensors" && -f "$FT_LOG_JSON" ]] \
        && [[ "$(marker_fingerprint "$FT_LOG_JSON" ft_fingerprint)" == "$FT_FINGERPRINT" ]]
}

# Läuft in der Orchestrierungs-Umgebung (vor der Trainings-Subshell).
ft_prepare() {
    if [[ "$ARM" == "gold" ]]; then
        python experiments/prepare_lf_dataset.py "$RUN_ID"
    else
        [[ -f "$LF_DATASET_JSON" && -f "$LF_DATASET_INFO_JSON" ]] \
            || die "Trainingsdaten fehlen: $LF_DATASET_JSON (bzw. dataset_info.json). Für $ARM-Runs erzeugt sie der Schritt label_build (experiments/prepare_lf_dataset.py $RUN_ID)."
    fi

    # Neuer Adapter -> alte Vorhersagen/Marker beiseitelegen (nicht löschen).
    move_stale "$BENCH_JSONL" "$BENCH_DONE" "$FT_LOG_JSON"
}

phase_train() {
    export MAMBA_ROOT_PREFIX="$MAMBA_ROOT"
    [[ -f "$MAMBA_ROOT/etc/profile.d/mamba.sh" ]] || die "Mamba-Profilskript fehlt unter $MAMBA_ROOT."
    [[ -d "$FT_ENV_DIR" ]]                         || die "Conda-Umgebung fehlt unter $FT_ENV_DIR."
    set +u; source "$MAMBA_ROOT/etc/profile.d/mamba.sh"; mamba activate "$FT_ENV_DIR"; set -u
    [[ "$(realpath "$CONDA_PREFIX")" == "$(realpath "$FT_ENV_DIR")" ]] \
        || die "Falsche Umgebung aktiv: ${CONDA_PREFIX:-<leer>}"
    export HF_HUB_OFFLINE=1
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"   # GLIBCXX-Fix

    gpu_diagnostics

    echo "====== DATASET ======"
    echo "Dataset:   $LF_DATASET_NAME ($LF_DATASET_JSON)"
    python3 -c "import json, sys; print('Beispiele:', len(json.load(open(sys.argv[1]))))" "$LF_DATASET_JSON"
    echo "FT-Seed:   $FT_SEED"
    echo ""

    echo "[INFO] Starting SFT training..."
    llamafactory-cli train \
      --stage sft \
      --do_train \
      --model_name_or_path "$BASE_MODEL" \
      --dataset "$LF_DATASET_NAME" \
      --dataset_dir "$LF_DATA_DIR" \
      --template "$FT_TEMPLATE" \
      --cutoff_len "$FT_CUTOFF_LEN" \
      --finetuning_type lora \
      --lora_target "$FT_LORA_TARGET" \
      --lora_rank "$FT_LORA_RANK" \
      --lora_alpha "$FT_LORA_ALPHA" \
      --lora_dropout "$FT_LORA_DROPOUT" \
      --output_dir "$LORA_DIR" \
      --per_device_train_batch_size "$FT_BATCH_SIZE" \
      --gradient_accumulation_steps "$FT_GRAD_ACCUM" \
      --learning_rate "$FT_LEARNING_RATE" \
      --lr_scheduler_type "$FT_LR_SCHEDULER_TYPE" \
      --weight_decay "$FT_WEIGHT_DECAY" \
      --warmup_ratio "$FT_WARMUP_RATIO" \
      --num_train_epochs "$FT_EPOCHS" \
      --bf16 \
      --seed "$FT_SEED" \
      --data_seed "$FT_SEED" \
      --logging_steps 5 \
      --plot_loss \
      --overwrite_output_dir \
      --report_to none \
      --save_strategy no \
      --save_only_model

    [[ -f "$LORA_DIR/adapter_model.safetensors" ]] \
        || die "Training beendet, aber kein Adapter unter $LORA_DIR gefunden."
    echo "[INFO] Training done."

    write_ft_log
}

# ---- FT-Log: train_results.json + Hardware + Hyperparameter -> JSON (Done-Marker) ----
write_ft_log() {
    RUN_ID="$RUN_ID" LF_DATASET_NAME="$LF_DATASET_NAME" LF_DATASET_JSON="$LF_DATASET_JSON" \
    LORA_DIR="$LORA_DIR" FT_LOG_JSON="$FT_LOG_JSON" BASE_MODEL="$BASE_MODEL" FT_SEED="$FT_SEED" \
    GIT_COMMIT="$GIT_COMMIT" GIT_DIRTY="$GIT_DIRTY" FT_FINGERPRINT="$FT_FINGERPRINT" \
    FT_TEMPLATE="$FT_TEMPLATE" FT_EPOCHS="$FT_EPOCHS" FT_LORA_RANK="$FT_LORA_RANK" \
    FT_LORA_ALPHA="$FT_LORA_ALPHA" FT_LORA_DROPOUT="$FT_LORA_DROPOUT" FT_LORA_TARGET="$FT_LORA_TARGET" \
    FT_LEARNING_RATE="$FT_LEARNING_RATE" FT_LR_SCHEDULER_TYPE="$FT_LR_SCHEDULER_TYPE" \
    FT_WEIGHT_DECAY="$FT_WEIGHT_DECAY" FT_WARMUP_RATIO="$FT_WARMUP_RATIO" \
    FT_BATCH_SIZE="$FT_BATCH_SIZE" FT_GRAD_ACCUM="$FT_GRAD_ACCUM" FT_CUTOFF_LEN="$FT_CUTOFF_LEN" \
    python3 - <<'PYEOF'
import json, os, subprocess
from datetime import datetime

e = os.environ

def env_int(*keys):
    for k in keys:
        try:
            return int(e[k])
        except (KeyError, ValueError):
            pass
    return None

# ---- Hardware ----
gpus, driver = [], None
try:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True).stdout
    for line in out.strip().splitlines():
        name, mem, driver = (x.strip() for x in line.split(","))
        gpus.append({"name": name, "memory_total_mib": int(mem)})
except Exception as exc:
    print(f"WARNUNG: nvidia-smi-Abfrage fehlgeschlagen: {exc}")

node_ram_gb = None
try:
    with open("/proc/meminfo") as f:
        kb = int(next(l for l in f if l.startswith("MemTotal:")).split()[1])
    node_ram_gb = round(kb / 1024**2, 1)
except Exception:
    pass

mem_per_node_mb = env_int("SLURM_MEM_PER_NODE")
mem_per_cpu_mb = env_int("SLURM_MEM_PER_CPU")
cpus_on_node = env_int("SLURM_CPUS_ON_NODE")
if mem_per_node_mb is None and mem_per_cpu_mb and cpus_on_node:
    mem_per_node_mb = mem_per_cpu_mb * cpus_on_node

t_start, t_end = env_int("SLURM_JOB_START_TIME"), env_int("SLURM_JOB_END_TIME")
hardware = {
    "node": os.uname().nodename,
    "partition": e.get("SLURM_JOB_PARTITION"),
    "num_nodes": env_int("SLURM_JOB_NUM_NODES", "SLURM_NNODES"),
    "ntasks": env_int("SLURM_NTASKS"),
    "num_gpus": len(gpus) or None,
    "gpus_requested_per_node": e.get("SLURM_GPUS_ON_NODE") or e.get("SLURM_GPUS_PER_NODE"),
    "gpu_model": gpus[0]["name"] if gpus else None,
    "gpus": gpus,
    "gpu_driver": driver,
    "cpus_allocated": cpus_on_node or len(os.sched_getaffinity(0)),
    "mem_requested_gb": round(mem_per_node_mb / 1024, 1) if mem_per_node_mb else None,
    "node_ram_total_gb": node_ram_gb,
    "time_limit_min": (t_end - t_start) // 60 if t_start and t_end else None,
}
with open(os.path.join(e["LORA_DIR"], "train_results.json")) as f:
    tr = json.load(f)

with open(e["LF_DATASET_JSON"]) as f:
    n_data = len(json.load(f))

bs, ga = int(e["FT_BATCH_SIZE"]), int(e["FT_GRAD_ACCUM"])
log = {
    "run_id": e["RUN_ID"],
    "dataset": e["LF_DATASET_NAME"],
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "slurm_job_id": e.get("SLURM_JOB_ID"),
    "git_commit": e["GIT_COMMIT"],
    "git_dirty": e["GIT_DIRTY"] == "1",
    "ft_seed": int(e["FT_SEED"]),
    "ft_fingerprint": e["FT_FINGERPRINT"],
    "hardware": hardware,
    "model": e["BASE_MODEL"],
    "lora_dir": e["LORA_DIR"],
    "params": {
        "n_data": n_data,
        "epochs": float(e["FT_EPOCHS"]),
        "lora_rank": int(e["FT_LORA_RANK"]),
        "lora_alpha": int(e["FT_LORA_ALPHA"]),
        "lora_dropout": float(e["FT_LORA_DROPOUT"]),
        "lora_target": e["FT_LORA_TARGET"],
        "learning_rate": float(e["FT_LEARNING_RATE"]),
        "lr_scheduler_type": e["FT_LR_SCHEDULER_TYPE"],
        "weight_decay": float(e["FT_WEIGHT_DECAY"]),
        "warmup_ratio": float(e["FT_WARMUP_RATIO"]),
        "batch_size": bs,
        "grad_accum": ga,
        "effective_batch_size": bs * ga,
        "cutoff_len": int(e["FT_CUTOFF_LEN"]),
        "template": e["FT_TEMPLATE"],
        "seed": int(e["FT_SEED"]),
    },
    "train_results": {k: tr.get(k) for k in (
        "train_runtime",
        "train_samples_per_second",
        "train_steps_per_second",
        "total_flos",
        "train_loss",
    )},
}

# Done-Marker der FT-Phase: atomar und als letzter Schritt schreiben.
tmp = e["FT_LOG_JSON"] + ".tmp"
with open(tmp, "w") as f:
    json.dump(log, f, indent=2)
os.replace(tmp, e["FT_LOG_JSON"])
print(f"[INFO] FT-Log geschrieben: {e['FT_LOG_JSON']}")
print(json.dumps(log["train_results"], indent=2))
PYEOF
}

# ======================= PHASE BENCH =======================================
phase_benchmark() {
    echo "Loading environment from start_llama.sh..."
    set +u; source "$START_LLAMA_SH"; set -u
    gpu_diagnostics

    local args=(
        --model-path "$BASE_MODEL"
        --data-tsv "$TEST_TSV"
        --mode "$BENCH_MODE"
        --max-prompts "$MAX_PROMPTS"
        --device cuda
    )
    case "$BENCH_MODE" in
        fine_tuned)
            [[ -f "$LORA_DIR/adapter_model.safetensors" ]] || die "Kein Adapter unter $LORA_DIR."
            args+=(--lora-path "$LORA_DIR") ;;
        icl)
            args+=(--few-shot-tsv "$FEW_SHOT_TSV" --icl-k "$ICL_K")
            echo "[INFO] ICL: k=$ICL_K Few-Shots aus $FEW_SHOT_TSV" ;;
    esac
    args+=("${BENCH_EXTRA_ARGS[@]}" --output-jsonl "$BENCH_JSONL" --resume)

    echo "[INFO] Starting benchmark: $BENCH_SCRIPT ${args[*]}"
    python "$BENCH_SCRIPT" "${args[@]}"

    printf '{"run_id": "%s", "slurm_job_id": "%s", "finished_at": "%s", "bench_fingerprint": "%s"}\n' \
        "$RUN_ID" "${SLURM_JOB_ID:-}" "$(date -Iseconds)" "$BENCH_FINGERPRINT" > "$BENCH_DONE"
    echo "[INFO] Benchmark done: $BENCH_DONE"
}

# ============================== ABLAUF =====================================
# Schlägt eine Phase fehl, bricht set -e den Job ab (keine Marker, kein Meta-Eintrag).
if has_stage ft_bench; then
    if ft_done; then
        echo "[SKIP] FT: Adapter + $FT_LOG_JSON vorhanden (Fingerprint aktuell)."
    else
        if [[ -f "$FT_LOG_JSON" ]]; then
            echo "[STALE] FT: Config geändert (ft_fingerprint in $FT_LOG_JSON:" \
                 "'$(marker_fingerprint "$FT_LOG_JSON" ft_fingerprint)', aktuell: '$FT_FINGERPRINT') -> neu trainieren."
        fi
        ft_prepare
        run_phase ft phase_train
    fi
elif ! has_stage bench; then
    die "Run $RUN_ID hat weder Stage 'bench' noch 'ft_bench' (stages: ${STAGES[*]})."
fi

if [[ -f "$BENCH_DONE" && "$(marker_fingerprint "$BENCH_DONE" bench_fingerprint)" == "$BENCH_FINGERPRINT" ]]; then
    echo "[SKIP] Bench: $BENCH_DONE vorhanden (Fingerprint aktuell)."
else
    if [[ -f "$BENCH_DONE" ]]; then
        echo "[STALE] Bench: Config geändert (bench_fingerprint in $BENCH_DONE:" \
             "'$(marker_fingerprint "$BENCH_DONE" bench_fingerprint)', aktuell: '$BENCH_FINGERPRINT') -> neu rechnen."
        move_stale "$BENCH_DONE" "$BENCH_JSONL"
    fi
    # Abgebrochener Lauf unter anderer (oder unbekannter) Config: nicht per --resume fortsetzen.
    if [[ -f "$BENCH_JSONL" && "$(marker_fingerprint "$BENCH_STARTED" bench_fingerprint)" != "$BENCH_FINGERPRINT" ]]; then
        echo "[STALE] Bench: $BENCH_JSONL wurde unter anderer/unbekannter Config begonnen -> neu rechnen."
        move_stale "$BENCH_JSONL"
    fi
    printf '{"bench_fingerprint": "%s"}\n' "$BENCH_FINGERPRINT" > "$BENCH_STARTED"
    run_phase bench phase_benchmark
fi

echo "[INFO] Job finished successfully. Total: $(( SECONDS / 60 )) min $(( SECONDS % 60 )) s"
