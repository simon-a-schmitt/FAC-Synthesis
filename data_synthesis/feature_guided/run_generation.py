"""Generate synthetic feature-guided examples for one domain (currently toxicity_detection) by
prompting the OpenRouter API, and verify each candidate with the local Llama-3.1-8B + SAE.

Sibling of ../blackbox/run_generation.py. Everything that is shared with the blackbox branch -
context-example drawing (2 seed examples + 1 accepted-pool example, or 3 seeds while the pool is
still empty), ROUGE-L dedup, wave-based concurrency with a barrier, checkpoint/--resume - is
imported from there unchanged, so both branches behave identically apart from the feature
guidance below.

Feature guidance: every call is additionally given one SAE feature (its annotation as
{{FEATURE_EXPLANATION}} and its top-activating spans as {{FEATURE_SPANS}}) from
data/feature_scores/<domain>_feature_relevance_scores.jsonl. Only task-relevant features are
used, i.e. those whose label is in --feature-labels (default Yes, Probably, Maybe), ordered by
label priority (the order of --feature-labels) and then by file order.

Feature schedule:
  - Seed coverage: once at the start of a run, every seed example is run through Llama + SAE
    exactly as active_feature_identification/get_active_features/run_get_active_features.py
    does. A relevant feature counts as covered if its raw activation exceeds --threshold on at
    least one content token of at least one seed. This is computed only once (and stored in the
    checkpoint); it is NOT updated as the accepted pool grows.
  - Pass 0 iterates over the relevant features NOT covered by the seeds; every later pass
    (1, 2, ...) iterates over ALL relevant features. Passes repeat until --n are accepted.
  - Within a pass, each feature gets up to --attempts-per-feature calls (default 3) and is done
    as soon as one of them yields an accepted candidate. A call counts as an attempt if the API
    returned a response (failed API calls are retried without using up an attempt). A feature
    whose attempts in a pass all end in rejections (feature_inactive / rouge_duplicate / no
    candidate) is marked exhausted and skipped in every later pass.
  - A wave holds at most one call per feature (the first 2 * --max-concurrent-requests
    unresolved features of the current pass), so "stop after the first accept" holds exactly.
The run stops once --n candidates are accepted, when every relevant feature is exhausted, or on
--max-calls (then the checkpoint is kept).

SAE verification: each candidate is wrapped in the domain's classification prompt (system =
SYSTEM_PROMPT from labeling/<path>/prompt/prompt_step_2.py, user = candidate verbatim), run through
Llama-3.1-8B + the SAE, and the target feature counts as active if its RAW (not p95-normalised)
activation exceeds --threshold on at least one content token (= user-turn tokens, special tokens
excluded). A candidate enters the accepted pool only if it activates its target feature AND
passes the ROUGE-L dedup.

Scheduling of API calls vs. SAE checks: API calls are I/O-bound and run on the worker threads;
the SAE forward pass runs on the main thread (the only one touching the GPU) as soon as each
call of a wave completes (as_completed), while the rest of the wave is still in flight. The SAE
result depends only on (candidate, feature), not on the pool, so running it in completion order
keeps runs deterministic. Dedup and pool mutation still happen strictly in slot order after the
wave's barrier, exactly as in the blackbox branch.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).parent
DATA_SYNTHESIS_DIR = BASE_DIR.parent
# The step-2 (labeling) prompts live in labeling/<domain>/prompt/, shared with run_labeling.py.
LABELING_DIR = DATA_SYNTHESIS_DIR / "labeling"
BLACKBOX_SCRIPT = DATA_SYNTHESIS_DIR / "blackbox" / "run_generation.py"
SEED_GROUPS_DIR = DATA_SYNTHESIS_DIR / "data" / "seed_groups"
FEATURE_SCORES_DIR = DATA_SYNTHESIS_DIR / "data" / "feature_scores"
FAC_TEST_PIPELINE_DIR = DATA_SYNTHESIS_DIR.parent / "fac_test_pipeline"
GET_ACTIVE_FEATURES_SCRIPT = (
    DATA_SYNTHESIS_DIR.parent / "active_feature_identification" / "get_active_features" / "run_get_active_features.py"
)


def _load_blackbox_module():
    spec = importlib.util.spec_from_file_location("blackbox_run_generation", BLACKBOX_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Must be registered before exec: its @dataclass definitions look themselves up in sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# Shared generation/dedup machinery - imported, not copied, so it can never drift from the
# blackbox branch.
bb = _load_blackbox_module()

# ---------------------------------------------------------------------------
# Generator model - selected via --model, independently of the blackbox branch
# ---------------------------------------------------------------------------

# --model selects one of these by keyword: keyword -> (model id, provider), as in
# blackbox/run_generation.py, hybrid/run_generation.py and labeling/run_labeling.py.
MODEL_PRESETS = {
    "llama": ("meta-llama/llama-3.1-8b-instruct", "deepinfra/fp8"),
    "deepseek": ("deepseek/deepseek-v4-flash-0731", "baseten/fp8"),
}
DEFAULT_MODEL = "deepseek"

# Module-level defaults (DEFAULT_MODEL); main() overrides them via configure_model() from --model.
# hybrid/ imports this module and overwrites these globals itself.
MODEL_ID, PROVIDER = MODEL_PRESETS[DEFAULT_MODEL]

# If True, every request is pinned to PROVIDER (via provider.only/allow_fallbacks=False). If
# False, PROVIDER is only used as EXPECTED_PROVIDER_NAME's default and requests are left free to
# be routed by OpenRouter across any provider that satisfies provider.require_parameters.
PIN_PROVIDER = False

# The deepseek-v4-flash family is the only model the "reasoning": {"enabled": False} override is
# known to be needed/supported for.
DEEPSEEK_V4_FLASH_MODEL_ID_FRAGMENT = "deepseek-v4-flash"
LLAMA_3_1_8B_INSTRUCT_MODEL_ID_FRAGMENT = "llama-3.1-8b-instruct"

# For these two models, every request is additionally pinned to an fp8 quantization and opts
# out of OpenRouter's prompt data collection, regardless of PIN_PROVIDER above.
FP8_DATA_DENY_MODEL_ID_FRAGMENTS = (
    DEEPSEEK_V4_FLASH_MODEL_ID_FRAGMENT,
    LLAMA_3_1_8B_INSTRUCT_MODEL_ID_FRAGMENT,
)

# OpenRouter's response echoes back just the provider's display name (e.g. "DeepInfra"),
# not the "<provider>/<quantization>" slug used to pin it in the request.
EXPECTED_PROVIDER_NAME = PROVIDER.split("/")[0]

BASE_MODEL_PARAMS = {
    "temperature": 1.0,
    "top_p": 0.95,
    "max_tokens": 2048,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "usage": {"include": True},
}

if DEEPSEEK_V4_FLASH_MODEL_ID_FRAGMENT in MODEL_ID:
    BASE_MODEL_PARAMS["reasoning"] = {"enabled": False}

# The imported blackbox helpers (fetch_one, derive_seed, print_raw_first_call) read these as
# module globals at call time. `bb` is a private module instance loaded just for this script, so
# overriding them here only affects the feature-guided branch, never blackbox runs.
bb.MODEL_ID = MODEL_ID
bb.PROVIDER = PROVIDER
bb.PIN_PROVIDER = PIN_PROVIDER
bb.EXPECTED_PROVIDER_NAME = EXPECTED_PROVIDER_NAME
bb.BASE_MODEL_PARAMS = BASE_MODEL_PARAMS


def configure_model(model_keyword: str) -> None:
    """Sets the generator model from --model, here and in the private `bb` instance. Only called
    from this script's main(), so importing this module (hybrid/) is unaffected."""
    global MODEL_ID, PROVIDER, EXPECTED_PROVIDER_NAME
    MODEL_ID, PROVIDER = MODEL_PRESETS[model_keyword]
    EXPECTED_PROVIDER_NAME = PROVIDER.split("/")[0]
    BASE_MODEL_PARAMS.pop("reasoning", None)
    if DEEPSEEK_V4_FLASH_MODEL_ID_FRAGMENT in MODEL_ID:
        BASE_MODEL_PARAMS["reasoning"] = {"enabled": False}
    bb.MODEL_ID = MODEL_ID
    bb.PROVIDER = PROVIDER
    bb.EXPECTED_PROVIDER_NAME = EXPECTED_PROVIDER_NAME
    bb.BASE_MODEL_PARAMS = BASE_MODEL_PARAMS

# Identifies this script's generation branch, so its per-slot RNG never collides with the
# blackbox branch's even under a shared run_id/seed_group.
ARM = "feature_guided"

DEFAULT_ATTEMPTS_PER_FEATURE = 3
# Task-relevant labels, in scheduling priority order.
DEFAULT_FEATURE_LABELS = ["Yes", "Probably", "Maybe"]


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def resolve_domain_dir(path_arg: str) -> Path:
    candidate = Path(path_arg)
    if candidate.is_absolute() and candidate.is_dir():
        return candidate
    under_base = BASE_DIR / candidate
    if under_base.is_dir():
        return under_base
    if candidate.is_dir():
        return candidate
    raise SystemExit(
        f"--path '{path_arg}' does not resolve to a directory (tried '{under_base}' and '{candidate}')."
    )


def _load_module(path: Path, module_name: str, required_attrs: tuple[str, ...]):
    if not path.exists():
        raise SystemExit(f"Expected prompt module at {path}, but it does not exist.")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    for attr in required_attrs:
        if not hasattr(module, attr):
            raise SystemExit(f"{path} is missing required attribute '{attr}'.")
    return module


def load_prompts(domain_dir: Path) -> tuple[str, str, str]:
    """Returns (generation system prompt, generation template, SAE classification system prompt)."""
    step_1 = _load_module(
        domain_dir / "prompts" / "prompt_step_1.py",
        f"fg_prompt_step_1_{domain_dir.name}",
        ("SYSTEM_PROMPT", "STEP_1_FG_PROMPT_TEMPLATE"),
    )
    step_2 = _load_module(
        LABELING_DIR / domain_dir.name / "prompt" / "prompt_step_2.py",
        f"fg_prompt_step_2_{domain_dir.name}",
        ("SYSTEM_PROMPT",),
    )
    return step_1.SYSTEM_PROMPT, step_1.STEP_1_FG_PROMPT_TEMPLATE, step_2.SYSTEM_PROMPT


def find_seed_file(domain_name: str, seed_group: str) -> Path:
    seed_dir = SEED_GROUPS_DIR / domain_name
    matches = sorted(seed_dir.glob(f"*_seed_group_{seed_group}.tsv"))
    if not matches:
        raise SystemExit(f"No seed file matching '*_seed_group_{seed_group}.tsv' found in {seed_dir}.")
    if len(matches) > 1:
        raise SystemExit(f"Multiple seed files match '*_seed_group_{seed_group}.tsv' in {seed_dir}: {matches}")
    return matches[0]


# ---------------------------------------------------------------------------
# Text cleaning (seeds, feature annotations, feature spans)
# ---------------------------------------------------------------------------

# Mojibake repair: text whose UTF-8 bytes were once mis-decoded as cp1252/latin-1 (e.g. "Letâ€™s"
# instead of "Let’s", "Â\xa0" instead of a no-break space). Every such mis-decoded character maps
# back to exactly one byte, so runs of those characters are re-encoded to bytes, and every valid
# UTF-8 multi-byte sequence inside them is decoded again. Bytes that do not form valid UTF-8 (e.g.
# a genuine "é" or "’") are left as they were, so correctly encoded text passes through unchanged.
_CP1252_TO_BYTE = {}
for _byte in range(0x80, 0xA0):
    try:
        _CP1252_TO_BYTE[bytes([_byte]).decode("cp1252")] = _byte
    except UnicodeDecodeError:
        pass  # undefined in cp1252 - such bytes show up as the raw C1 control character instead
_MOJIBAKE_RUN_RE = re.compile("[\x80-\xff" + re.escape("".join(_CP1252_TO_BYTE)) + "]+")
_UTF8_SEQUENCE_RE = re.compile(rb"[\xc2-\xdf][\x80-\xbf]|[\xe0-\xef][\x80-\xbf]{2}|[\xf0-\xf4][\x80-\xbf]{3}")


def _repair_mojibake_run(match: re.Match) -> str:
    run = match.group()
    raw = bytes(_CP1252_TO_BYTE.get(ch, ord(ch)) for ch in run)
    parts, pos = [], 0
    for seq in _UTF8_SEQUENCE_RE.finditer(raw):
        parts.append(run[pos:seq.start()])
        try:
            parts.append(seq.group().decode("utf-8"))
        except UnicodeDecodeError:
            parts.append(run[seq.start():seq.end()])
        pos = seq.end()
    parts.append(run[pos:])
    return "".join(parts)


def fix_mojibake(text: str) -> str:
    # A few rounds, in case text was mis-decoded more than once.
    for _ in range(3):
        fixed = _MOJIBAKE_RUN_RE.sub(_repair_mojibake_run, text)
        if fixed == text:
            break
        text = fixed
    return text


_ZERO_WIDTH_RE = re.compile("[\u200b\u200c\u200d\u2060\ufeff]")
_NEWLINE_RUN_RE = re.compile(r"\s*\n\s*")
_SPACE_RUN_RE = re.compile(r"[ \t\r\f\v\u00a0\u202f]+")

TRUNCATION_MARKER = "… "
LINE_BREAK_MARKER = " / "


def clean_text(text: str) -> str:
    """Encoding cleanup shared by seeds, annotations and spans: mojibake repaired, zero-width
    characters dropped, no-break spaces turned into plain spaces. Backslashes are deliberately
    NOT unescaped here: in the feature spans they are genuine source text (code like
    'cout<<"x\\t"', Windows paths like 'T:\\Lab11data\\rain', LaTeX like '\\( e^{-x} \\)')."""
    text = fix_mojibake(text)
    text = _ZERO_WIDTH_RE.sub("", text)
    return text.replace("\u00a0", " ").replace("\u202f", " ")


def to_single_line(text: str) -> str:
    """Line breaks become the visible " / " marker, other whitespace runs a single space."""
    text = _NEWLINE_RUN_RE.sub(LINE_BREAK_MARKER, text.strip())
    return _SPACE_RUN_RE.sub(" ", text).strip()


def assert_clean(text: str, where: str) -> None:
    """Final guard: no repairable mojibake left in text that goes into a prompt."""
    if fix_mojibake(text) != text:
        raise SystemExit(f"{where}: text still contains mojibake after cleaning: {text[:160]!r}")


# Seed-specific escape artifacts that must never survive into a prompt: CSV-doubled quotes,
# backslash-escaped quotes, literal "\n"/"\t" escape sequences, and raw line breaks / carriage
# returns (a seed is rendered as one line of the numbered SEEDS list).
_SEED_ARTIFACT_RE = re.compile(r'""|\\"|\\n|\\t|[\r\n]')


def clean_seed_text(text: str) -> str:
    """Unescapes one seed text as read by csv.reader: encoding cleanup, then both the literal "\\n"
    escape sequences the seed TSVs use for line breaks and any real line breaks become " / "."""
    return to_single_line(clean_text(text).replace("\\n", "\n"))


def load_seed_examples(seed_file: Path) -> list[str]:
    """Seed texts (first column), fully unescaped.

    Parsed with csv.reader rather than a plain split on tabs, because the seed TSVs are written
    CSV-style: a field containing quotes is wrapped in quotes with its inner quotes doubled
    ("...like ""I look around"" and..."). A plain split would pass those quote artifacts straight
    into the prompt. Fails fast if any escape artifact is still left after unescaping.
    """
    examples = []
    with open(seed_file, "r", encoding="utf-8", newline="") as f:
        for line_no, row in enumerate(csv.reader(f, delimiter="\t"), start=1):
            text = clean_seed_text(row[0]) if row else ""
            if not text:
                continue
            artifact = _SEED_ARTIFACT_RE.search(text)
            if artifact:
                raise SystemExit(
                    f"{seed_file}:{line_no}: seed still contains the escape artifact {artifact.group()!r} "
                    f"after unescaping: {text[:120]!r}"
                )
            assert_clean(text, f"{seed_file}:{line_no}")
            examples.append(text)
    if not examples:
        raise SystemExit(f"No seed examples found in {seed_file}")
    return examples


def load_features(feature_scores_path: Path, labels: list[str]) -> list[dict]:
    """Features whose relevance label is in `labels`, ordered by label priority (= order of
    `labels`), then by file order."""
    features = []
    with open(feature_scores_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            label = row.get("label")
            if label not in labels:
                continue
            feature_id = int(row["feature_id"])
            annotation = to_single_line(clean_text(row["annotation"]))
            spans = [clean_text(span) for span in row["spans"]]
            assert_clean(annotation, f"feature {feature_id} annotation")
            for span in spans:
                assert_clean(span, f"feature {feature_id} span")
            features.append({"feature_id": feature_id, "label": label, "annotation": annotation, "spans": spans})
    if not features:
        raise SystemExit(f"No features with a label in {labels!r} found in {feature_scores_path}.")
    features.sort(key=lambda feature: labels.index(feature["label"]))  # stable: file order within a label
    return features


# ---------------------------------------------------------------------------
# Feature span rendering
# ---------------------------------------------------------------------------

# A (possibly cut-off) Llama-3 chat header: "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n",
# or just its tail, e.g. "user<|end_header_id|>\n\n" / "<|end_header_id|>\n\n". Whitespace before the
# header is left alone, so a line break before a mid-span header still renders as " / ".
_HEADER_RE = re.compile(
    r"(?:<\|eot_id\|>)?(?:<\|start_header_id\|>)?(user|assistant|system)?<\|end_header_id\|>\s*"
)
_SPECIAL_TOKEN_RE = re.compile(r"<\|[a-z_]+\|>")


def _header_replacement(match: re.Match) -> str:
    role = match.group(1)
    return f" [start of {role} turn] " if role else " [start of turn] "


def format_span(span: str) -> str:
    """One feature span as a single line: chat headers become "[start of <role> turn]", other
    special tokens are dropped, internal line breaks become " / ", and a span that does not start
    at a turn header (i.e. is cut off mid-text) gets a leading "… "."""
    starts_at_header = _HEADER_RE.match(span.lstrip()) is not None
    text = _HEADER_RE.sub(_header_replacement, span)
    text = to_single_line(_SPECIAL_TOKEN_RE.sub("", text))
    return text if starts_at_header else TRUNCATION_MARKER + text


def render_feature_spans(spans: list[str]) -> str:
    return "\n".join(f"- {format_span(span)}" for span in spans)


def build_user_prompt(template: str, context_examples: list[dict], feature: dict) -> str:
    user_prompt = (
        template.replace("{{SEED_EXAMPLES}}", bb.render_seed_examples_block(context_examples))
        .replace("{{FEATURE_EXPLANATION}}", feature["annotation"])
        .replace("{{FEATURE_SPANS}}", render_feature_spans(feature["spans"]))
    )
    assert_clean(user_prompt, f"prompt for feature {feature['feature_id']}")
    return user_prompt


# ---------------------------------------------------------------------------
# SAE verification
# ---------------------------------------------------------------------------

@dataclass
class SaeContext:
    fs: Any  # fac_test_pipeline/run_fac_test_pipeline_feature_stats module
    model: Any
    collector: Any
    sae: Any
    system_prompt: str


def load_sae_context(args: argparse.Namespace, system_prompt: str) -> SaeContext:
    """Loads Llama + SAE exactly as fac_test_pipeline/feature_coverage/*/run_feature_coverage.py does.

    Imported lazily: HF offline env vars must be set before transformers is imported.
    """
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    if args.hf_cache_dir:
        os.environ["TRANSFORMERS_CACHE"] = args.hf_cache_dir
        os.makedirs(args.hf_cache_dir, exist_ok=True)
    if args.device == "cuda":
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if str(FAC_TEST_PIPELINE_DIR) not in sys.path:
        sys.path.insert(0, str(FAC_TEST_PIPELINE_DIR))
    import run_fac_test_pipeline_feature_stats as fs  # noqa: E402

    model_path = os.path.abspath(args.model_name)
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Local model directory not found: {model_path}")
    sae_ckpt = fs.resolve_sae_checkpoint(local_path=args.sae_ckpt_path or None)

    model = fs.UnifiedGenerator(
        model_path,
        device=args.device,
        dtype=args.dtype,
        cache_dir=args.hf_cache_dir or fs._default_cache_dir(),
        local_files_only=True,
        strict_local_paths=True,
    )
    collector = fs.Collector(args.layer)
    fs.mount_function(model._model, "llama", args.layer, collector)
    collector.early_stop = True

    sae = fs.TopKSAE.from_disk(sae_ckpt, device=args.device)
    sae.topk = fs.TOP_K
    sae.eval()
    return SaeContext(fs=fs, model=model, collector=collector, sae=sae, system_prompt=system_prompt)


def compute_target_activation(text: str, feature_id: int, sae_ctx: SaeContext) -> dict:
    """Raw SAE activation of `feature_id` on the content (user-turn) tokens of the classification
    prompt for `text`. Mirrors compute_feature_activation_for_description() in
    fac_test_pipeline/run_synthetic_feature_activation_check.py, without the CVSS wrapping and
    without p95 normalisation."""
    fs = sae_ctx.fs
    tc = fs.tc
    model, collector = sae_ctx.model, sae_ctx.collector
    tokenizer = model._tokenizer  # noqa: SLF001

    with tc.no_grad():
        collector.cache = None
        _, token_ids, tokens = fs._encode_prompt_tokens(text, model, system=sae_ctx.system_prompt)
        try:
            model.get_activates(text, system=sae_ctx.system_prompt)
        except RuntimeError:
            pass  # early_stop raises right after the hooked layer
        if collector.cache is None:
            raise RuntimeError("Collector cache is empty. Hook may not be mounted correctly.")

        hidden_seq = collector.cache.to(tc.float32)[0]
        if hidden_seq.shape[0] != len(tokens):
            seq_len = min(hidden_seq.shape[0], len(tokens))
            hidden_seq = hidden_seq[:seq_len]
            token_ids = token_ids[:seq_len]
            tokens = tokens[:seq_len]
        sparse_features = sae_ctx.sae.encode(hidden_seq).detach().cpu()

    special_ids = fs._special_token_id_set(tokenizer)
    content_start = fs._find_user_content_start(token_ids, tokens, tokenizer)
    content_positions = [
        idx for idx, token_id in enumerate(token_ids) if idx >= content_start and token_id not in special_ids
    ]
    result = {
        "n_content_tokens": len(content_positions),
        "max_raw_activation": 0.0,
        "n_active_tokens": 0,
        "top_token": None,
    }
    if not content_positions or not 0 <= feature_id < sparse_features.shape[1]:
        return result

    raw_col = sparse_features[content_positions, feature_id]
    n_active = int((raw_col > 0).sum().item())
    if n_active == 0:
        return result
    best_local = int(raw_col.argmax().item())
    best_pos = content_positions[best_local]
    result.update(
        max_raw_activation=round(float(raw_col[best_local].item()), 6),
        n_active_tokens=n_active,
        top_token={"token_index": best_pos, "token": tokens[best_pos]},
    )
    return result


def compute_seed_peak_activations(seed_file: Path, sae_ctx: SaeContext) -> dict[int, float]:
    """{feature_id: max raw SAE activation over the content tokens of all seed rows}.

    Reuses active_feature_identification/get_active_features/run_get_active_features.py as is
    (TSV loading, per-row task detection from the label column, task-specific prompt/opening
    phrase, content-token masking), so "covered by the seeds" means exactly what that script
    reports as active. Must be called after load_sae_context() (offline env vars, sys.path)."""
    spec = importlib.util.spec_from_file_location("fg_get_active_features", GET_ACTIVE_FEATURES_SCRIPT)
    gaf = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(gaf)

    peaks: dict[int, float] = {}
    with sae_ctx.fs.tc.no_grad():
        for text, label in gaf.load_seed_group_tsv(str(seed_file)):
            task = gaf.detect_task(label)
            if task is None:
                raise SystemExit(f"{seed_file}: cannot determine task from seed label {label!r}.")
            system, user_content, opening_phrase = gaf.build_prompt_for_row(task, text)
            row_peaks = gaf.compute_max_raw_activation_per_feature(
                user_content, sae_ctx.model, sae_ctx.collector, sae_ctx.sae,
                opening_phrase=opening_phrase, system=system,
            )
            for feature_id, peak in row_peaks.items():
                if peak > peaks.get(feature_id, float("-inf")):
                    peaks[feature_id] = peak
    return peaks


# ---------------------------------------------------------------------------
# Feature schedule
# ---------------------------------------------------------------------------

def new_schedule(features: list[dict], seed_peaks: dict[int, float], threshold: float) -> dict:
    """Initial (JSON-serialisable, checkpointed) schedule state: pass 0 = relevant features not
    covered by the seeds, in priority order."""
    covered = [f["feature_id"] for f in features if seed_peaks.get(f["feature_id"], 0.0) > threshold]
    covered_set = set(covered)
    return {
        "seed_covered": covered,
        "seed_uncovered": [f["feature_id"] for f in features if f["feature_id"] not in covered_set],
        "seed_peak_activations": {
            str(f["feature_id"]): round(seed_peaks[f["feature_id"]], 6)
            for f in features if f["feature_id"] in seed_peaks
        },
        "pass_idx": 0,
        "pass_queue": [f["feature_id"] for f in features if f["feature_id"] not in covered_set],
        "pass_attempts": {},
        "exhausted": [],
        "exhausted_in_pass": {},
    }


def advance_pass_if_done(schedule: dict, features: list[dict]) -> None:
    """Starts the next pass (over ALL non-exhausted relevant features) once the current pass's
    queue is empty. Leaves the queue empty if every relevant feature is exhausted."""
    exhausted = set(schedule["exhausted"])
    while not schedule["pass_queue"]:
        next_queue = [f["feature_id"] for f in features if f["feature_id"] not in exhausted]
        if not next_queue:
            return
        schedule["pass_idx"] += 1
        schedule["pass_queue"] = next_queue
        schedule["pass_attempts"] = {}
        print(f"[schedule] Starting pass {schedule['pass_idx']} over {len(next_queue)} relevant feature(s).")


def update_schedule(schedule: dict, feature_id: int, outcome: str, attempts_per_feature: int) -> None:
    """Applies one call's outcome (see process_call_result) to the current pass."""
    if outcome not in ("accepted", "rejected"):
        return  # API failure / unchecked candidate: not an attempt, the feature stays queued as is
    key = str(feature_id)
    schedule["pass_attempts"][key] = schedule["pass_attempts"].get(key, 0) + 1
    if outcome == "accepted":
        schedule["pass_queue"].remove(feature_id)
    elif schedule["pass_attempts"][key] >= attempts_per_feature:
        schedule["pass_queue"].remove(feature_id)
        schedule["exhausted"].append(feature_id)
        schedule["exhausted_in_pass"][key] = schedule["pass_idx"]
        print(
            f"  [exhausted] f={feature_id}: no accepted candidate after {attempts_per_feature} attempt(s) "
            f"in pass {schedule['pass_idx']}; skipped from now on."
        )


# ---------------------------------------------------------------------------
# Per-call processing
# ---------------------------------------------------------------------------

@dataclass
class SlotInfo:
    feature: dict
    pass_idx: int
    feature_attempt: int  # 0-based attempt index for this feature within the pass


def process_call_result(
    result,
    slot_info: SlotInfo,
    sae_results: list[dict],
    *,
    run_id: str,
    args: argparse.Namespace,
    seed_tokens: list[list[str]],
    accepted_pool: list[dict],
    pool_tokens: list[list[str]],
    rejected_entries: list[dict],
    failed_entries: list[dict],
    call_records: list[dict],
    counters: dict,
    feature_stats: dict,
    endpoint_catalog,
) -> str:
    """Applies one call's result to the shared pool. Main-thread-only, called strictly in slot
    order for a wave (see blackbox process_call_result). `sae_results` holds the already computed
    SAE check for each of result.candidates.

    Returns the call's outcome for the feature schedule: "failed" (API error), "accepted" (at
    least one candidate entered the pool), "rejected" (no candidate, or all rejected by the
    activation / dedup check) or "target_reached" (candidate not checked, --n already reached)."""
    context_texts = [c["text"] for c in result.context_examples]
    feature_id = slot_info.feature["feature_id"]
    stats = feature_stats.setdefault(
        str(feature_id), {"label": slot_info.feature["label"], "attempts": 0, "accepted": 0, "feature_inactive": 0, "rouge_duplicate": 0,
         "no_candidate": 0, "failed": 0}
    )
    stats["attempts"] += 1
    base_entry = {
        "run_id": run_id,
        "call_id": result.call_id,
        "call_number": result.call_number,
        "wave_idx": result.wave_idx,
        "slot_index": result.slot_index,
        "feature_id": feature_id,
        "feature_label": slot_info.feature["label"],
        "pass_idx": slot_info.pass_idx,
        "feature_attempt": slot_info.feature_attempt,
    }

    if not result.ok:
        print(f"  [error] {result.error}", file=sys.stderr)
        counters["n_failed_calls"] += 1
        stats["failed"] += 1
        failed_entries.append(
            {
                **base_entry,
                "seed_sent": result.derived_seed,
                "context_examples": context_texts,
                "error": result.error,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        return "failed"

    counters["total_prompt_tokens"] += result.prompt_tokens
    counters["total_completion_tokens"] += result.completion_tokens

    sampling_verification = bb.build_sampling_verification(
        result.requested_params, result.openrouter_meta.get("provider"), endpoint_catalog
    )
    if not sampling_verification["verified"]:
        print(
            f"  [warn] Call {result.call_id} could not be verified against the endpoint catalog "
            f"(provider={sampling_verification['routed_provider']!r}, "
            f"endpoint_tags={sampling_verification['endpoint_tags']})",
            file=sys.stderr,
        )
    # One entry per parsed candidate, pointing at the accepted/rejected entry (same call_id).
    call_outcomes: list[dict] = []
    call_records.append(
        {
            "call_id": result.call_id,
            "call_number": result.call_number,
            "wave_idx": result.wave_idx,
            "slot_index": result.slot_index,
            "seed_sent": result.derived_seed,
            "feature_id": feature_id,
            "feature_label": slot_info.feature["label"],
            "pass_idx": slot_info.pass_idx,
            "feature_attempt": slot_info.feature_attempt,
            "context_examples": result.context_examples,
            "n_parsed_candidates": len(result.candidates),
            "outcomes": call_outcomes,
            "openrouter_response": result.openrouter_meta,
            "sampling_verification": sampling_verification,
        }
    )
    generation_id = result.openrouter_meta.get("id")

    if not result.candidates:
        stats["no_candidate"] += 1
    outcome = "rejected"
    for candidate_text, sae_result in zip(result.candidates, sae_results):
        sae_fields = {
            "sae_max_raw_activation": sae_result["max_raw_activation"],
            "sae_n_active_tokens": sae_result["n_active_tokens"],
            "sae_n_content_tokens": sae_result["n_content_tokens"],
            "sae_top_token": sae_result["top_token"],
        }

        if counters["n_accepted_this_run"] >= args.n:
            rejected_entries.append(
                {
                    **base_entry,
                    "generation_id": generation_id,
                    "rejected_text": candidate_text,
                    "rejected_reason": "target_reached",
                    **sae_fields,
                    "context_examples": context_texts,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
            counters["n_rejected_this_run"] += 1
            call_outcomes.append({"status": "rejected", "rejected_reason": "target_reached"})
            if outcome == "rejected":
                outcome = "target_reached"  # never checked, so not an attempt for the schedule
            continue

        candidate_tokens = bb.rouge_tokenize(candidate_text)
        best_score = -1.0
        best_match: dict | None = None
        for i, entry_tokens in enumerate(seed_tokens):
            score = bb.rouge_l_fmeasure(candidate_tokens, entry_tokens)
            if score > best_score:
                best_score = score
                best_match = {"type": "seed", "id": i}
        for i, entry_tokens in enumerate(pool_tokens):
            score = bb.rouge_l_fmeasure(candidate_tokens, entry_tokens)
            if score > best_score:
                best_score = score
                best_match = {"type": "synthetic", "id": accepted_pool[i]["id"], "text": accepted_pool[i]["text"]}

        feature_active = sae_result["max_raw_activation"] > args.threshold
        if not feature_active or best_score > args.rouge_threshold:
            reason = "feature_inactive" if not feature_active else "rouge_duplicate"
            rejected_entries.append(
                {
                    **base_entry,
                    "generation_id": generation_id,
                    "rejected_text": candidate_text,
                    "rejected_reason": reason,
                    **sae_fields,
                    "rouge_l_score": best_score,
                    "matched_pool_id": best_match["id"],
                    "matched_pool_type": best_match["type"],
                    "context_examples": context_texts,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
            counters["n_rejected_this_run"] += 1
            counters[f"n_{reason}"] += 1
            stats[reason] += 1
            call_outcomes.append({"status": "rejected", "rejected_reason": reason})
            print(
                f"  [reject:{reason}] f={feature_id} act={sae_result['max_raw_activation']:.3f} "
                f"rouge={best_score:.3f} vs {best_match['type']} id={best_match['id']}: {candidate_text[:80]!r}"
            )
        else:
            accepted_pool.append(
                {
                    "id": counters["next_id"],
                    "type": "synthetic",
                    "text": candidate_text,
                    "feature_id": feature_id,
                    "feature_label": slot_info.feature["label"],
                    "pass_idx": slot_info.pass_idx,
                    "feature_attempt": slot_info.feature_attempt,
                    **sae_fields,
                    "context_examples": context_texts,
                    "run_id": run_id,
                    "call_id": result.call_id,
                    "generation_id": generation_id,
                    "call_number": result.call_number,
                    "wave_idx": result.wave_idx,
                    "slot_index": result.slot_index,
                    "added_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            pool_tokens.append(candidate_tokens)
            call_outcomes.append({"status": "accepted", "accepted_id": counters["next_id"]})
            counters["next_id"] += 1
            counters["n_accepted_this_run"] += 1
            stats["accepted"] += 1
            outcome = "accepted"
            print(
                f"  [accept] ({counters['n_accepted_this_run']}/{args.n}) f={feature_id} "
                f"act={sae_result['max_raw_activation']:.3f} rouge={best_score:.3f}: {candidate_text[:80]!r}"
            )
    return outcome


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate synthetic feature-guided examples for one domain via the OpenRouter API, "
        "verified by Llama-3.1-8B + SAE and ROUGE-L deduped against an accepted pool."
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL, choices=sorted(MODEL_PRESETS),
        help="Generator model keyword, mapped to an OpenRouter model id + provider via "
        f"MODEL_PRESETS ({', '.join(f'{k}={v[0]}' for k, v in MODEL_PRESETS.items())}; default: %(default)s).",
    )
    parser.add_argument(
        "--path", type=str, required=True,
        help="Domain subfolder (e.g. 'toxicity_detection'), either a name under feature_guided/ or a path to it.",
    )
    parser.add_argument(
        "--seed-group", type=str, required=True, choices=sorted(bb.VALID_SEED_GROUPS),
        help="Seed group id (01-05); selects data/seed_groups/<domain>/*_seed_group_<id>.tsv.",
    )
    parser.add_argument("--n", type=int, required=True, help="Number of ACCEPTED samples to generate before stopping.")
    parser.add_argument(
        "--prefix", type=str, required=True,
        help="Filename prefix for output ('<prefix>_accepted.json' / '_rejected.json' / '_failed.json') "
        "and log ('<prefix>_log.json') files.",
    )
    parser.add_argument(
        "--rouge-threshold", type=float, required=True,
        help="Reject a candidate if its highest ROUGE-L F-measure against seeds + accepted pool "
        "exceeds this value. Set explicitly per run for the feature-guided arm (no default).",
    )

    # Feature guidance
    parser.add_argument(
        "--feature-scores", type=Path, default=None,
        help="Feature relevance JSONL (default: data/feature_scores/<domain>_feature_relevance_scores.jsonl).",
    )
    parser.add_argument(
        "--feature-labels", type=str, nargs="+", default=DEFAULT_FEATURE_LABELS,
        choices=["Yes", "Probably", "Maybe", "No"],
        help="Task-relevant feature labels, in scheduling priority order (default: %(default)s).",
    )
    parser.add_argument(
        "--attempts-per-feature", type=int, default=DEFAULT_ATTEMPTS_PER_FEATURE,
        help="Max. generation calls per feature and pass; a feature without an accepted candidate "
        "after that many attempts is skipped in all later passes (default: %(default)s).",
    )
    parser.add_argument(
        "--threshold", type=float, required=True,
        help="A candidate activates its target feature (and a seed covers a feature) if the RAW SAE "
        "activation exceeds this value on at least one content token.",
    )

    # SAE verification model
    parser.add_argument("--model-name", type=str, required=True, help="Local Llama-3.1-8B-Instruct directory.")
    parser.add_argument("--sae-ckpt-path", type=str, required=True)
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--device-id", type=str, default="0")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--hf-cache-dir", type=str, default=os.environ.get("TRANSFORMERS_CACHE", ""))

    # Generation API (same as blackbox)
    parser.add_argument("--temperature", type=float, default=None, help="Override BASE_MODEL_PARAMS['temperature'].")
    parser.add_argument("--top-p", type=float, default=None, help="Override BASE_MODEL_PARAMS['top_p'].")
    parser.add_argument("--max-tokens", type=int, default=None, help="Override BASE_MODEL_PARAMS['max_tokens'].")
    parser.add_argument(
        "--extra-params", type=str, default=None,
        help="Additional OpenRouter request body parameters as a JSON object string.",
    )
    parser.add_argument(
        "--max-calls", type=int, default=None,
        help="Safety cap on total API calls (default: unlimited).",
    )
    parser.add_argument("--env-file", type=Path, default=bb.DEFAULT_ENV_FILE, help="Path to a .env file providing OPENROUTER_API_KEY.")
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument(
        "--max-retry-seconds", type=float, default=600.0,
        help="Total wall-clock budget (seconds) to keep retrying a single failed HTTP request (default: %(default)s).",
    )
    parser.add_argument(
        "--max-concurrent-requests", type=int, default=1,
        help="Number of worker threads (default: 1). Requests run in waves of 2 * --max-concurrent-requests "
        "prompts, as in the blackbox branch.",
    )
    parser.add_argument(
        "--requests-per-second", type=float, default=None,
        help="Optional cap on request START rate shared across all concurrent workers (default: unlimited).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume an interrupted run from its checkpoint ('<prefix>_checkpoint.json' in <path>/output).",
    )
    return parser


def build_model_params(args: argparse.Namespace) -> dict:
    """Same request-parameter/provider setup as blackbox run_generation.main()."""
    model_params = dict(BASE_MODEL_PARAMS)
    if args.temperature is not None:
        model_params["temperature"] = args.temperature
    if args.top_p is not None:
        model_params["top_p"] = args.top_p
    if args.max_tokens is not None:
        model_params["max_tokens"] = args.max_tokens
    if args.extra_params:
        model_params.update(json.loads(args.extra_params))
    if PIN_PROVIDER:
        model_params["provider"] = {
            **model_params.get("provider", {}),
            "only": [PROVIDER],
            "allow_fallbacks": False,
            "require_parameters": True,
        }
    else:
        model_params["provider"] = {
            **model_params.get("provider", {}),
            "require_parameters": True,
        }
    if any(fragment in MODEL_ID for fragment in FP8_DATA_DENY_MODEL_ID_FRAGMENTS):
        model_params["provider"] = {
            **model_params["provider"],
            "quantizations": ["fp8"],
            "allow_fallbacks": False,
            "require_parameters": True,
            "data_collection": "deny",
        }
    return model_params


# Changing any of these on --resume would invalidate the checkpointed feature schedule (seed
# coverage, pass queue, exhausted features).
SCHEDULE_ARGS = ("seed_group", "feature_scores", "feature_labels", "attempts_per_feature", "threshold")
# Changing any of these on --resume is allowed, but noted.
NOTED_ARGS = ("n", "rouge_threshold", "max_calls", "max_concurrent_requests", "requests_per_second")


def checkpoint_resolved_args(domain_dir: Path, args: argparse.Namespace, feature_scores: Path, max_calls: int | None) -> dict:
    return {
        "model": args.model,
        "path": str(domain_dir),
        "seed_group": args.seed_group,
        "feature_scores": str(feature_scores),
        "feature_labels": list(args.feature_labels),
        "attempts_per_feature": args.attempts_per_feature,
        "n": args.n,
        "rouge_threshold": args.rouge_threshold,
        "threshold": args.threshold,
        "max_calls": max_calls,
        "max_concurrent_requests": args.max_concurrent_requests,
        "requests_per_second": args.requests_per_second,
    }


def save_checkpoint(path: Path, **fields) -> None:
    bb.save_json(path, {**fields, "saved_at": datetime.now(timezone.utc).isoformat()})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_arg_parser().parse_args()
    configure_model(args.model)
    if args.attempts_per_feature < 1:
        raise SystemExit("--attempts-per-feature must be >= 1.")

    bb.load_dotenv(args.env_file)
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit(f"OPENROUTER_API_KEY is not set. Set it in the environment or in {args.env_file}.")

    domain_dir = resolve_domain_dir(args.path)
    domain_name = domain_dir.name
    system_prompt, prompt_template, sae_system_prompt = load_prompts(domain_dir)
    seed_file = find_seed_file(domain_name, args.seed_group)
    seed_examples = load_seed_examples(seed_file)
    n_seeds = len(seed_examples)
    if n_seeds < bb.MIN_SEED_EXAMPLES:
        raise SystemExit(
            f"Need at least {bb.MIN_SEED_EXAMPLES} seed examples for the context-drawing scheme, "
            f"but {seed_file} only has {n_seeds}."
        )

    feature_scores = args.feature_scores or FEATURE_SCORES_DIR / f"{domain_name}_feature_relevance_scores.jsonl"
    features = load_features(feature_scores, args.feature_labels)
    features_by_id = {feature["feature_id"]: feature for feature in features}

    model_params = build_model_params(args)
    max_calls = args.max_calls

    output_dir = domain_dir / "output"
    log_dir = domain_dir / "log"
    accepted_path = output_dir / f"{args.prefix}_accepted.json"
    rejected_path = output_dir / f"{args.prefix}_rejected.json"
    failed_path = output_dir / f"{args.prefix}_failed.json"
    checkpoint_path = output_dir / f"{args.prefix}_checkpoint.json"
    log_path = log_dir / f"{args.prefix}_log.json"

    resolved_args = checkpoint_resolved_args(domain_dir, args, feature_scores, max_calls)
    checkpoint = bb.load_json_dict(checkpoint_path)

    if args.resume:
        if checkpoint is None:
            raise SystemExit(f"--resume was given but no checkpoint found at {checkpoint_path}; nothing to resume.")
        # Checkpoints written before --model existed carry no "model" key; those are not checked.
        checkpoint_model = checkpoint["resolved_args"].get("model")
        if checkpoint_model is not None and checkpoint_model != args.model:
            raise SystemExit(
                f"Checkpoint at {checkpoint_path} was created with --model {checkpoint_model!r}, "
                f"but this invocation passed --model {args.model!r}. Resuming with a different generator "
                "model would mix models within one run - use the original --model, or delete the "
                "checkpoint file to discard it and start fresh."
            )
        for key in SCHEDULE_ARGS:
            old_value = checkpoint["resolved_args"].get(key)
            if old_value != resolved_args[key]:
                raise SystemExit(
                    f"Checkpoint at {checkpoint_path} was created with {key}={old_value!r}, but this invocation "
                    f"has {key}={resolved_args[key]!r}. That would change the context/feature schedule - use the "
                    "original value, or delete the checkpoint file to start fresh."
                )
        for key in NOTED_ARGS:
            old_value = checkpoint["resolved_args"].get(key)
            if old_value != resolved_args[key]:
                print(f"[resume] Note: --{key.replace('_', '-')} changed since the checkpoint was saved ({old_value!r} -> {resolved_args[key]!r}).")
        run_id = checkpoint["run_id"]
        started_at = checkpoint["started_at"]
        wave_idx = checkpoint["wave_idx"]
        counters = checkpoint["counters"]
        call_records = checkpoint["call_records"]
        feature_stats = checkpoint["feature_stats"]
        schedule = checkpoint.get("schedule")
        if schedule is None:
            raise SystemExit(
                f"Checkpoint at {checkpoint_path} predates the seed-coverage feature schedule and cannot be "
                "resumed by this version; delete it to start a fresh run."
            )
        print(f"[resume] Continuing run {run_id} from wave {wave_idx} (checkpoint saved at {checkpoint['saved_at']}).")
    else:
        if checkpoint is not None:
            raise SystemExit(
                f"Found an incomplete checkpoint at {checkpoint_path} (run_id={checkpoint['run_id']}, "
                f"next wave {checkpoint['wave_idx']}, {checkpoint['counters']['n_accepted_this_run']} accepted "
                f"so far). Pass --resume to continue it, or delete the file to discard it and start a fresh run."
            )
        run_id = uuid.uuid4().hex
        started_at = datetime.now(timezone.utc).isoformat()
        wave_idx = 0
        counters = None
        call_records = []
        feature_stats = {}
        schedule = None  # computed from the seeds once Llama + SAE are loaded

    print(f"Domain: {domain_dir} | seed group: {args.seed_group} | seed file: {seed_file} ({n_seeds} seeds)")
    label_counts = {label: sum(f["label"] == label for f in features) for label in args.feature_labels}
    print(
        f"Relevant features: {len(features)} {label_counts} from {feature_scores}, "
        f"up to {args.attempts_per_feature} attempt(s) per feature and pass"
    )
    print(
        f"Run id: {run_id} | model: {MODEL_ID} | target accepted: {args.n} | max calls: {max_calls} "
        f"| SAE raw threshold: {args.threshold}"
    )

    # Seed examples vs. accepted pool: same split/merge as the blackbox branch.
    loaded_pool = bb.load_json_list(accepted_path)
    stored_seed_entries = [e for e in loaded_pool if e.get("type") == "seed"]
    accepted_pool = [e for e in loaded_pool if e.get("type") == "synthetic"]
    # Stored seed entries are only reused if their texts match the (unescaped) seed file exactly;
    # otherwise - e.g. an older accepted.json written before seeds were unescaped - they are
    # regenerated, so dedup never runs against stale, still-escaped seed texts.
    if [e["text"] for e in stored_seed_entries] == seed_examples:
        seed_pool_entries = stored_seed_entries
    else:
        if stored_seed_entries:
            print(f"[warn] Seed entries in {accepted_path} differ from {seed_file}; replacing them.")
        seed_pool_entries = [
            {"id": i, "type": "seed", "text": text, "seed_example_id": i, "run_id": None, "added_at": started_at}
            for i, text in enumerate(seed_examples)
        ]
    seed_tokens = [bb.rouge_tokenize(e["text"]) for e in seed_pool_entries]
    pool_tokens = [bb.rouge_tokenize(entry["text"]) for entry in accepted_pool]
    next_id = max((entry["id"] for entry in accepted_pool), default=len(seed_pool_entries) - 1) + 1
    print(
        f"Loaded {len(seed_pool_entries)} seed example(s) and {len(accepted_pool)} accepted "
        f"synthetic entrie(s) from {accepted_path} (ROUGE-L threshold: {args.rouge_threshold})"
    )

    rejected_entries = bb.load_json_list(rejected_path)
    failed_entries = bb.load_json_list(failed_path)

    if counters is None:
        counters = {
            "n_calls": 0,
            "n_failed_calls": 0,
            "n_accepted_this_run": 0,
            "n_rejected_this_run": 0,
            "n_feature_inactive": 0,
            "n_rouge_duplicate": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "next_id": next_id,
        }
    else:
        counters["next_id"] = max(counters["next_id"], next_id)

    print("Loading Llama + SAE for feature verification...")
    sae_ctx = load_sae_context(args, sae_system_prompt)

    if schedule is None:
        print(f"Computing seed coverage of the relevant features on {seed_file} (threshold {args.threshold})...")
        schedule = new_schedule(features, compute_seed_peak_activations(seed_file, sae_ctx), args.threshold)
    covered_set = set(schedule["seed_covered"])
    for label in args.feature_labels:
        n_label = label_counts[label]
        n_covered = sum(f["label"] == label and f["feature_id"] in covered_set for f in features)
        print(f"  [seed coverage] {label:<9} covered {n_covered:>4} / not covered {n_label - n_covered:>4} / total {n_label:>4}")
    print(
        f"[schedule] pass {schedule['pass_idx']}: {len(schedule['pass_queue'])} feature(s) queued, "
        f"{len(schedule['exhausted'])} exhausted"
    )
    advance_pass_if_done(schedule, features)

    ctx = bb.WorkerContext(
        api_key=api_key,
        model_params=model_params,
        system_prompt=system_prompt,
        max_retry_seconds=args.max_retry_seconds,
        request_timeout=args.request_timeout,
        rate_limiter=bb.RateLimiter(args.requests_per_second),
    )
    endpoint_catalog = bb.EndpointCatalog(MODEL_ID, args.request_timeout)
    endpoint_catalog.refresh()

    n_workers = args.max_concurrent_requests
    wave_batch_size = 2 * n_workers
    stop_reason = None

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        while counters["n_accepted_this_run"] < args.n:
            remaining_budget = None if max_calls is None else max_calls - counters["n_calls"]
            if remaining_budget is not None and remaining_budget <= 0:
                stop_reason = f"reached --max-calls ({max_calls})"
                break
            if not schedule["pass_queue"]:
                stop_reason = f"all {len(features)} relevant features are exhausted"
                break

            # One call per feature and wave: the first unresolved features of the current pass.
            wave_feature_ids = schedule["pass_queue"][:wave_batch_size]
            if remaining_budget is not None:
                wave_feature_ids = wave_feature_ids[:remaining_budget]
            slots_this_wave = len(wave_feature_ids)
            pool_snapshot = list(accepted_pool)  # frozen for this wave's prompt construction

            print(
                f"[wave {wave_idx}] pass {schedule['pass_idx']}: dispatching {slots_this_wave} call(s) "
                f"({len(schedule['pass_queue'])} feature(s) left in pass, {len(schedule['exhausted'])} exhausted; "
                f"accepted {counters['n_accepted_this_run']}/{args.n}, pool snapshot size {len(pool_snapshot)})..."
            )

            futures: dict = {}
            slot_infos: dict[int, SlotInfo] = {}
            for slot_index, feature_id in enumerate(wave_feature_ids):
                counters["n_calls"] += 1
                call_number = counters["n_calls"]
                feature = features_by_id[feature_id]
                slot_info = SlotInfo(
                    feature=feature,
                    pass_idx=schedule["pass_idx"],
                    feature_attempt=schedule["pass_attempts"].get(str(feature_id), 0),
                )
                slot_infos[slot_index] = slot_info

                slot_rng = bb.derive_slot_rng(run_id, args.seed_group, ARM, wave_idx, slot_index)
                context_examples = bb.pick_context_examples(slot_rng, seed_examples, pool_snapshot)
                user_prompt = build_user_prompt(prompt_template, context_examples, feature)
                derived_seed = bb.derive_seed(run_id, call_number)
                call_id = bb.make_call_id(run_id, call_number)

                if call_number == 1:
                    bb.print_raw_first_call(system_prompt, user_prompt, model_params, derived_seed)

                context_desc = ", ".join(
                    f"{c['type']}:{c.get('seed_example_id', c.get('accepted_id'))}" for c in context_examples
                )
                print(
                    f"  [wave {wave_idx} slot {slot_index}] call {call_number} ({call_id}) feature={feature['feature_id']} "
                    f"({feature['label']}) attempt={slot_info.feature_attempt} context=[{context_desc}]"
                )

                future = executor.submit(
                    bb.fetch_one, ctx, call_id, call_number, wave_idx, slot_index, user_prompt, context_examples, derived_seed
                )
                futures[future] = slot_index

            # SAE-verify each call as soon as it completes, while the rest of the wave is still in
            # flight. This is also the wave's barrier: the loop only ends once every call is done.
            results_by_slot = {}
            sae_by_slot: dict[int, list[dict]] = {}
            for future in as_completed(futures):
                slot_index = futures[future]
                result = future.result()
                results_by_slot[slot_index] = result
                feature_id = slot_infos[slot_index].feature["feature_id"]
                sae_by_slot[slot_index] = [
                    compute_target_activation(text, feature_id, sae_ctx) for text in (result.candidates if result.ok else [])
                ]

            outcomes = []
            for slot_index in range(slots_this_wave):
                outcome = process_call_result(
                    results_by_slot[slot_index],
                    slot_infos[slot_index],
                    sae_by_slot[slot_index],
                    run_id=run_id,
                    args=args,
                    seed_tokens=seed_tokens,
                    accepted_pool=accepted_pool,
                    pool_tokens=pool_tokens,
                    rejected_entries=rejected_entries,
                    failed_entries=failed_entries,
                    call_records=call_records,
                    counters=counters,
                    feature_stats=feature_stats,
                    endpoint_catalog=endpoint_catalog,
                )
                outcomes.append(outcome)
                update_schedule(schedule, slot_infos[slot_index].feature["feature_id"], outcome, args.attempts_per_feature)
            advance_pass_if_done(schedule, features)

            wave_idx += 1

            bb.save_json(accepted_path, seed_pool_entries + accepted_pool)
            bb.save_json(rejected_path, rejected_entries)
            bb.save_json(failed_path, failed_entries)
            save_checkpoint(
                checkpoint_path,
                run_id=run_id,
                started_at=started_at,
                wave_idx=wave_idx,
                counters=counters,
                call_records=call_records,
                feature_stats=feature_stats,
                schedule=schedule,
                resolved_args=resolved_args,
            )
            print(f"[checkpoint] Saved progress after wave {wave_idx - 1} ({counters['n_accepted_this_run']}/{args.n} accepted so far).")

            if all(outcome == "failed" for outcome in outcomes):
                # Failed calls do not use up attempts, so a persistent API outage would otherwise loop forever.
                stop_reason = f"every API call of wave {wave_idx - 1} failed"
                break

    n_calls = counters["n_calls"]
    n_accepted_this_run = counters["n_accepted_this_run"]
    total_prompt_tokens = counters["total_prompt_tokens"]
    total_completion_tokens = counters["total_completion_tokens"]
    target_reached = n_accepted_this_run >= args.n

    bb.save_json(accepted_path, seed_pool_entries + accepted_pool)
    print(f"Wrote {len(seed_pool_entries)} seed + {len(accepted_pool)} synthetic entrie(s) to {accepted_path}")
    bb.save_json(rejected_path, rejected_entries)
    print(f"Wrote {len(rejected_entries)} rejected entrie(s) to {rejected_path}")
    bb.save_json(failed_path, failed_entries)
    print(f"Wrote {len(failed_entries)} failed call entrie(s) to {failed_path}")

    if not target_reached:
        # Log entry deferred until the run completes (see blackbox main() for why).
        print(
            f"[warn] Stopped without reaching target ({stop_reason}; {n_accepted_this_run}/{args.n} accepted, "
            f"{n_calls} call(s) made); checkpoint retained at {checkpoint_path}. Run again with --resume "
            "(e.g. with a higher --max-calls, if non-exhausted features remain) to continue.",
            file=sys.stderr,
        )
        return

    finished_at = datetime.now(timezone.utc).isoformat()
    run_entry = {
        "run_id": run_id,
        "arm": ARM,
        "model": args.model,
        "model_id": MODEL_ID,
        "path": str(domain_dir),
        "seed_group": args.seed_group,
        "seed_file": str(seed_file),
        "n_seeds": n_seeds,
        "feature_scores": str(feature_scores),
        "feature_labels": list(args.feature_labels),
        "attempts_per_feature": args.attempts_per_feature,
        "n_relevant_features": len(features),
        "n_seed_covered": len(schedule["seed_covered"]),
        "n_seed_uncovered": len(schedule["seed_uncovered"]),
        "seed_covered": schedule["seed_covered"],
        "seed_uncovered": schedule["seed_uncovered"],
        "seed_peak_activations": schedule["seed_peak_activations"],
        "n_passes_started": schedule["pass_idx"] + 1,
        "n_exhausted": len(schedule["exhausted"]),
        "exhausted": schedule["exhausted"],
        "exhausted_in_pass": schedule["exhausted_in_pass"],
        "n_features_used": len(feature_stats),
        "threshold": args.threshold,
        "sae_model": os.path.abspath(args.model_name),
        "sae_ckpt_path": os.path.abspath(args.sae_ckpt_path),
        "sae_layer": args.layer,
        "n_requested": args.n,
        "rouge_threshold": args.rouge_threshold,
        "n_calls": n_calls,
        "n_failed_calls": counters["n_failed_calls"],
        "n_accepted": n_accepted_this_run,
        "n_rejected": counters["n_rejected_this_run"],
        "n_feature_inactive": counters["n_feature_inactive"],
        "n_rouge_duplicate": counters["n_rouge_duplicate"],
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "total_tokens": total_prompt_tokens + total_completion_tokens,
        "model_params": model_params,
        "sampling_verification_summary": bb.summarize_sampling_verification(call_records),
        "endpoint_catalog": endpoint_catalog.snapshot(),
        "endpoint_catalog_fetched_at": endpoint_catalog.fetched_at,
        "accepted_file": str(accepted_path),
        "rejected_file": str(rejected_path),
        "failed_file": str(failed_path),
        "started_at": started_at,
        "finished_at": finished_at,
        "wall_clock_time": bb.format_wall_clock_slurm(started_at, finished_at),
        "feature_stats": feature_stats,
        "calls": call_records,
    }

    existing_log = bb.load_json_dict(log_path) or {}
    cumulative_prompt_tokens = existing_log.get("cumulative_prompt_tokens", 0) + total_prompt_tokens
    cumulative_completion_tokens = existing_log.get("cumulative_completion_tokens", 0) + total_completion_tokens
    runs = existing_log.get("runs", [])
    runs.append(run_entry)
    bb.save_json(
        log_path,
        {
            "prefix": args.prefix,
            "path": str(domain_dir),
            "cumulative_prompt_tokens": cumulative_prompt_tokens,
            "cumulative_completion_tokens": cumulative_completion_tokens,
            "cumulative_total_tokens": cumulative_prompt_tokens + cumulative_completion_tokens,
            "runs": runs,
        },
    )
    print(f"Wrote run log to {log_path}")

    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"Target reached; removed checkpoint {checkpoint_path}")

    print(
        f"Done: {n_accepted_this_run} accepted / {counters['n_rejected_this_run']} rejected "
        f"({counters['n_feature_inactive']} feature inactive, {counters['n_rouge_duplicate']} ROUGE duplicate) "
        f"over {n_calls} call(s) ({counters['n_failed_calls']} failed); "
        f"prompt_tokens={total_prompt_tokens}, completion_tokens={total_completion_tokens}"
    )


if __name__ == "__main__":
    main()
