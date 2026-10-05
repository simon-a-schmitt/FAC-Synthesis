"""Feature guidance shared by the feature-guided arm and hybrid's phase 2.

Features: every call is given one task-relevant SAE feature - its annotation as
{{FEATURE_EXPLANATION}} and its top-activating spans as {{FEATURE_SPANS}} - from
benchmarks/<domain>/feature_scores/<domain>_feature_relevance_scores.jsonl. Relevant = label in
--feature-labels (default Yes, Probably, Maybe), ordered by label priority, then file order.

SAE check: one function (sae_forward) runs a text through Llama-3.1-8B + SAE wrapped in the
domain's classification prompt (prompts/<domain>/labeling.py: system = SYSTEM_PROMPT, user =
USER_PROMPT_PREFIX + text) and keeps the content tokens (user turn, special tokens and the prefix
excluded). It is used for the seed coverage, hybrid's blackbox-example coverage and the candidate
check alike. A feature is active if its RAW activation exceeds --threshold on at least one content
token. A candidate is accepted only if it activates its target feature AND passes ROUGE-L dedup.

Schedule:
  - Coverage, computed once per run (checkpointed): a relevant feature is covered if a seed (as
    stored in the seed TSV) - or, in hybrid, an accepted blackbox example - activates it.
  - Pass 0 iterates over the uncovered relevant features, every later pass over ALL relevant
    features; passes repeat until the target is reached.
  - Within a pass each feature gets up to --attempts-per-feature calls and is done after its first
    accepted candidate. A call is an attempt if the API returned a response and it was checked
    (failed calls / discarded candidates are not). A feature whose attempts in a pass all end
    in rejections is exhausted and skipped in every later pass.
  - A wave holds at most one call per feature, so "done after the first accept" holds exactly.

SAE compute: every forward pass is timed (wall clock between two torch.cuda.synchronize() calls)
and its token count recorded, in counters["sae_coverage_check"] / ["sae_candidate_check"].
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prompts import LabelingPrompt, load_labeling_prompt
from shared.benchmarks import PROJECT_DIR, feature_scores_path, load_raw_seed_texts
from shared.generation import render_examples_block
from shared.text_cleaning import TRUNCATION_MARKER, assert_clean, clean_text, to_single_line

FAC_TEST_PIPELINE_DIR = PROJECT_DIR / "fac_test_pipeline"
FEATURE_LABEL_CHOICES = ("Yes", "Probably", "Maybe", "No")
DEFAULT_FEATURE_LABELS = ["Yes", "Probably", "Maybe"]
DEFAULT_ATTEMPTS_PER_FEATURE = 3
# Changing any of these on --resume would invalidate the checkpointed coverage / schedule.
STRICT_RESUME_ARGS = ("feature_scores", "feature_labels", "attempts_per_feature", "threshold")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def add_feature_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("feature guidance")
    group.add_argument("--feature-scores", type=Path, default=None,
                       help="Feature relevance JSONL (default: benchmarks/<domain>/feature_scores/...).")
    group.add_argument("--feature-labels", type=str, nargs="+", default=DEFAULT_FEATURE_LABELS,
                       choices=FEATURE_LABEL_CHOICES, help="Relevant labels, in scheduling priority order (default: %(default)s).")
    group.add_argument("--attempts-per-feature", type=int, default=DEFAULT_ATTEMPTS_PER_FEATURE,
                       help="Max. calls per feature and pass before it is exhausted (default: %(default)s).")
    group.add_argument("--threshold", type=float, required=True,
                       help="A feature is active if its RAW SAE activation exceeds this on >= 1 content token.")
    group = parser.add_argument_group("SAE verification model")
    group.add_argument("--model-name", type=str, required=True, help="Local Llama-3.1-8B-Instruct directory.")
    group.add_argument("--sae-ckpt-path", type=str, required=True)
    group.add_argument("--layer", type=int, default=16)
    group.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    group.add_argument("--device-id", type=str, default="0",
                       help="CUDA_VISIBLE_DEVICES to use if it is not already set (e.g. by SLURM).")
    group.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    group.add_argument("--hf-cache-dir", type=str, default=os.environ.get("TRANSFORMERS_CACHE", ""))


def validate_feature_args(args: argparse.Namespace) -> None:
    """Fails before any API call (hybrid would otherwise only notice after its blackbox phase)."""
    if args.attempts_per_feature < 1:
        raise SystemExit("--attempts-per-feature must be >= 1.")
    if not os.path.isdir(os.path.abspath(args.model_name)):
        raise SystemExit(f"--model-name: local model directory not found: {os.path.abspath(args.model_name)}")
    if not os.path.exists(args.sae_ckpt_path):
        raise SystemExit(f"--sae-ckpt-path: {args.sae_ckpt_path} does not exist.")


def resolve_feature_scores(args: argparse.Namespace) -> Path:
    return args.feature_scores or feature_scores_path(args.domain)


def feature_resolved_args(args: argparse.Namespace) -> dict:
    return {
        "feature_scores": str(resolve_feature_scores(args)),
        "feature_labels": list(args.feature_labels),
        "attempts_per_feature": args.attempts_per_feature,
        "threshold": args.threshold,
    }


# ---------------------------------------------------------------------------
# Features + prompt
# ---------------------------------------------------------------------------

def load_features(feature_scores: Path, labels: list[str]) -> list[dict]:
    """Features whose relevance label is in `labels`, ordered by label priority, then file order."""
    features = []
    with open(feature_scores, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("label") not in labels:
                continue
            feature_id = int(row["feature_id"])
            annotation = to_single_line(clean_text(row["annotation"]))
            spans = [clean_text(span) for span in row["spans"]]
            for text in (annotation, *spans):
                assert_clean(text, f"feature {feature_id}")
            features.append({"feature_id": feature_id, "label": row["label"], "annotation": annotation, "spans": spans})
    if not features:
        raise SystemExit(f"No features with a label in {labels!r} found in {feature_scores}.")
    features.sort(key=lambda feature: labels.index(feature["label"]))  # stable: file order within a label
    return features


# A (possibly cut-off) Llama-3 chat header, e.g. "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
# or just its tail "<|end_header_id|>\n\n".
_HEADER_RE = re.compile(r"(?:<\|eot_id\|>)?(?:<\|start_header_id\|>)?(user|assistant|system)?<\|end_header_id\|>\s*")
_SPECIAL_TOKEN_RE = re.compile(r"<\|[a-z_]+\|>")


def format_span(span: str) -> str:
    """One feature span as a single line: chat headers become "[start of <role> turn]", other
    special tokens are dropped, line breaks become " / ", and a span cut off mid-text gets "… "."""
    starts_at_header = _HEADER_RE.match(span.lstrip()) is not None
    text = _HEADER_RE.sub(lambda m: f" [start of {m.group(1)} turn] " if m.group(1) else " [start of turn] ", span)
    text = to_single_line(_SPECIAL_TOKEN_RE.sub("", text))
    return text if starts_at_header else TRUNCATION_MARKER + text


def build_user_prompt(template: str, context_examples: list[dict], feature: dict) -> str:
    user_prompt = (
        template.replace("{{SEED_EXAMPLES}}", render_examples_block(context_examples))
        .replace("{{FEATURE_EXPLANATION}}", feature["annotation"])
        .replace("{{FEATURE_SPANS}}", "\n".join(f"- {format_span(span)}" for span in feature["spans"]))
    )
    assert_clean(user_prompt, f"prompt for feature {feature['feature_id']}")
    return user_prompt


# ---------------------------------------------------------------------------
# SAE
# ---------------------------------------------------------------------------

@dataclass
class SaeContext:
    fs: Any  # fac_test_pipeline/run_fac_test_pipeline_feature_stats
    model: Any
    collector: Any
    sae: Any
    prompt: LabelingPrompt


def load_sae_context(args: argparse.Namespace) -> SaeContext:
    """Llama + SAE as in fac_test_pipeline. Imported lazily: HF offline env vars must be set first."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    if args.hf_cache_dir:
        os.environ["TRANSFORMERS_CACHE"] = args.hf_cache_dir
        os.makedirs(args.hf_cache_dir, exist_ok=True)
    if args.device == "cuda":
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        # Never override an existing assignment (SLURM sets CUDA_VISIBLE_DEVICES to the job's GPUs);
        # --device-id only applies outside such an environment.
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            print(f"Using CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} from the environment "
                  f"(--device-id {args.device_id} ignored).")
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    if str(FAC_TEST_PIPELINE_DIR) not in sys.path:
        sys.path.insert(0, str(FAC_TEST_PIPELINE_DIR))
    import run_fac_test_pipeline_feature_stats as fs  # noqa: E402

    print("Loading Llama + SAE...")
    model = fs.UnifiedGenerator(
        os.path.abspath(args.model_name),
        device=args.device,
        dtype=args.dtype,
        cache_dir=args.hf_cache_dir or fs._default_cache_dir(),  # noqa: SLF001
        local_files_only=True,
        strict_local_paths=True,
    )
    collector = fs.Collector(args.layer)
    fs.mount_function(model._model, "llama", args.layer, collector)  # noqa: SLF001
    collector.early_stop = True
    sae = fs.TopKSAE.from_disk(fs.resolve_sae_checkpoint(local_path=args.sae_ckpt_path), device=args.device)
    sae.topk = fs.TOP_K
    sae.eval()
    return SaeContext(fs=fs, model=model, collector=collector, sae=sae, prompt=load_labeling_prompt(args.domain))


def new_sae_stats() -> dict:
    return {"n_forward_passes": 0, "n_forward_tokens": 0, "gpu_seconds": 0.0}


def add_sae_pass(stats: dict, n_forward_tokens: int, seconds: float) -> None:
    stats["n_forward_passes"] += 1
    stats["n_forward_tokens"] += n_forward_tokens
    stats["gpu_seconds"] = round(stats["gpu_seconds"] + seconds, 6)


@dataclass
class SaeForward:
    content_features: Any  # [n_content_tokens, n_features] raw SAE activations
    content_positions: list[int]
    tokens: list[str]
    n_forward_tokens: int  # all prompt tokens run through Llama up to the hooked layer
    seconds: float  # net GPU time of the pass


def sae_forward(text: str, sae_ctx: SaeContext) -> SaeForward:
    """Runs `text` in the domain's classification prompt through Llama + SAE."""
    fs, tc = sae_ctx.fs, sae_ctx.fs.tc
    model, collector, prompt = sae_ctx.model, sae_ctx.collector, sae_ctx.prompt
    tokenizer = model._tokenizer  # noqa: SLF001
    user_content = prompt.user_content(text.strip())

    def cuda_sync() -> None:
        if tc.cuda.is_available():
            tc.cuda.synchronize()

    cuda_sync()
    start = time.perf_counter()
    with tc.no_grad():
        collector.cache = None
        _, token_ids, tokens = fs._encode_prompt_tokens(user_content, model, system=prompt.system)  # noqa: SLF001
        try:
            model.get_activates(user_content, system=prompt.system)
        except RuntimeError:
            pass  # early_stop raises right after the hooked layer
        if collector.cache is None:
            raise RuntimeError("Collector cache is empty. Hook may not be mounted correctly.")
        hidden_seq = collector.cache.to(tc.float32)[0]
        n_forward_tokens = int(hidden_seq.shape[0])
        seq_len = min(hidden_seq.shape[0], len(tokens))
        hidden_seq, token_ids, tokens = hidden_seq[:seq_len], token_ids[:seq_len], tokens[:seq_len]
        sparse_features = sae_ctx.sae.encode(hidden_seq).detach().cpu()
    cuda_sync()
    seconds = time.perf_counter() - start

    special_ids = fs._special_token_id_set(tokenizer)  # noqa: SLF001
    content_start = fs._find_user_content_start(token_ids, tokens, tokenizer)  # noqa: SLF001
    if prompt.user_prefix.strip():
        content_start = fs._advance_past_opening_phrase(token_ids, content_start, tokenizer, prompt.user_prefix.strip())  # noqa: SLF001
    positions = [idx for idx, token_id in enumerate(token_ids) if idx >= content_start and token_id not in special_ids]
    return SaeForward(sparse_features[positions], positions, tokens, n_forward_tokens, seconds)


def peak_activations(forward: SaeForward) -> dict[int, float]:
    """{feature_id: max raw activation over the content tokens} for every feature active on them."""
    features = forward.content_features
    if features.numel() == 0:
        return {}
    active_ids = (features > 0).any(dim=0).nonzero(as_tuple=False)[:, 0]
    peaks = features[:, active_ids].max(dim=0).values
    return {int(fid): float(val) for fid, val in zip(active_ids.tolist(), peaks.tolist())}


def target_activation(forward: SaeForward, feature_id: int) -> dict:
    """Raw activation of one feature on the content tokens, plus the pass's compute."""
    result = {
        "max_raw_activation": 0.0,
        "n_active_tokens": 0,
        "n_content_tokens": len(forward.content_positions),
        "top_token": None,
        "n_forward_tokens": forward.n_forward_tokens,
        "seconds": round(forward.seconds, 6),
    }
    if not forward.content_positions or not 0 <= feature_id < forward.content_features.shape[1]:
        return result
    column = forward.content_features[:, feature_id]
    n_active = int((column > 0).sum().item())
    if n_active:
        best = int(column.argmax().item())
        position = forward.content_positions[best]
        result.update(max_raw_activation=round(float(column[best].item()), 6), n_active_tokens=n_active,
                      top_token={"token_index": position, "token": forward.tokens[position]})
    return result


def text_peaks(texts: list[str], sae_ctx: SaeContext, stats: dict) -> list[dict[int, float]]:
    peaks = []
    for text in texts:
        forward = sae_forward(text, sae_ctx)
        add_sae_pass(stats, forward.n_forward_tokens, forward.seconds)
        peaks.append(peak_activations(forward))
    return peaks


# ---------------------------------------------------------------------------
# Coverage + schedule
# ---------------------------------------------------------------------------

def _merge_peaks(per_text: list[dict[int, float]]) -> dict[int, float]:
    merged: dict[int, float] = {}
    for peaks in per_text:
        for fid, peak in peaks.items():
            if peak > merged.get(fid, float("-inf")):
                merged[fid] = peak
    return merged


def new_schedule(features: list[dict], threshold: float, seed_peaks: list[dict[int, float]],
                 blackbox_entries: list[dict] | None, blackbox_peaks: list[dict[int, float]] | None) -> dict:
    """Checkpointed coverage + schedule: pass 0 = the relevant features covered neither by the
    seeds nor (hybrid) by the blackbox examples, in priority order."""
    relevant = [f["feature_id"] for f in features]
    seeds = _merge_peaks(seed_peaks)
    seed_covered = {fid for fid in relevant if seeds.get(fid, 0.0) > threshold}
    schedule = {
        "seed_covered": [fid for fid in relevant if fid in seed_covered],
        "seed_uncovered": [fid for fid in relevant if fid not in seed_covered],
        "seed_peak_activations": {str(fid): round(seeds[fid], 6) for fid in relevant if fid in seeds},
    }
    covered = set(seed_covered)
    if blackbox_entries is not None:
        bb = _merge_peaks(blackbox_peaks)
        bb_covered = {fid for fid in relevant if bb.get(fid, 0.0) > threshold}
        covered |= bb_covered
        relevant_set = set(relevant)
        schedule.update(
            blackbox_covered=[fid for fid in relevant if fid in bb_covered],
            blackbox_only_covered=[fid for fid in relevant if fid in bb_covered - seed_covered],
            blackbox_peak_activations={str(fid): round(bb[fid], 6) for fid in relevant if fid in bb},
            # Relevant features each accepted blackbox example activates above the threshold.
            blackbox_entry_active_features={
                str(entry["id"]): sorted((fid for fid, peak in peaks.items() if fid in relevant_set and peak > threshold),
                                         key=relevant.index)
                for entry, peaks in zip(blackbox_entries, blackbox_peaks)
            },
        )
    uncovered = [fid for fid in relevant if fid not in covered]
    schedule.update(
        covered=[fid for fid in relevant if fid in covered],
        uncovered=uncovered,
        pass_idx=0,
        pass_queue=list(uncovered),
        pass_attempts={},
        exhausted=[],
        exhausted_in_pass={},
    )
    return schedule


def coverage_by_label(features: list[dict], schedule: dict, labels: list[str]) -> dict:
    sets = {key: set(schedule[key]) for key in ("seed_covered", "blackbox_covered", "covered") if key in schedule}
    by_label = {}
    for label in labels:
        ids = [f["feature_id"] for f in features if f["label"] == label]
        counts = {"total": len(ids), **{key: sum(fid in s for fid in ids) for key, s in sets.items()}}
        counts["uncovered"] = counts["total"] - counts["covered"]
        by_label[label] = counts
    return by_label


# ---------------------------------------------------------------------------
# Guidance of one feature-guided phase (hooks of shared.generation.run_phase)
# ---------------------------------------------------------------------------

@dataclass
class SlotInfo:
    feature: dict
    pass_idx: int
    feature_attempt: int  # 0-based attempt index for this feature within the pass


class FeatureGuidance:
    def __init__(self, args: argparse.Namespace, features: list[dict], sae_ctx: SaeContext, state: dict):
        self.args = args
        self.features = features
        self.features_by_id = {f["feature_id"]: f for f in features}
        self.sae_ctx = sae_ctx
        self.state = state
        self.schedule = state["schedule"]

    # -- schedule -------------------------------------------------------------

    def advance_pass_if_done(self) -> None:
        """Starts the next pass over ALL non-exhausted relevant features once the current pass's
        queue is empty; leaves it empty if every relevant feature is exhausted."""
        schedule = self.schedule
        exhausted = set(schedule["exhausted"])
        while not schedule["pass_queue"]:
            next_queue = [f["feature_id"] for f in self.features if f["feature_id"] not in exhausted]
            if not next_queue:
                return
            schedule["pass_idx"] += 1
            schedule["pass_queue"] = next_queue
            schedule["pass_attempts"] = {}
            print(f"[schedule] Starting pass {schedule['pass_idx']} over {len(next_queue)} relevant feature(s).")

    def _update_schedule(self, feature_id: int, outcome: str) -> None:
        if outcome not in ("accepted", "rejected"):
            return  # API failure / discarded (never checked): not an attempt
        schedule, key = self.schedule, str(feature_id)
        schedule["pass_attempts"][key] = schedule["pass_attempts"].get(key, 0) + 1
        if outcome == "accepted":
            schedule["pass_queue"].remove(feature_id)
        elif schedule["pass_attempts"][key] >= self.args.attempts_per_feature:
            schedule["pass_queue"].remove(feature_id)
            schedule["exhausted"].append(feature_id)
            schedule["exhausted_in_pass"][key] = schedule["pass_idx"]
            print(f"  [exhausted] f={feature_id}: no accepted candidate after {self.args.attempts_per_feature} "
                  f"attempt(s) in pass {schedule['pass_idx']}; skipped from now on.")

    # -- run_phase hooks ------------------------------------------------------

    def wave_slots(self, n: int) -> list[SlotInfo]:
        schedule = self.schedule
        return [
            SlotInfo(self.features_by_id[fid], schedule["pass_idx"], schedule["pass_attempts"].get(str(fid), 0))
            for fid in schedule["pass_queue"][:n]
        ]

    def wave_info(self) -> str:
        s = self.schedule
        return f"[pass {s['pass_idx']}: {len(s['pass_queue'])} feature(s) left, {len(s['exhausted'])} exhausted]"

    def describe(self, slot: SlotInfo) -> str:
        return f"feature={slot.feature['feature_id']} ({slot.feature['label']}) attempt={slot.feature_attempt}"

    def build_prompt(self, template: str, context_examples: list[dict], slot: SlotInfo) -> str:
        return build_user_prompt(template, context_examples, slot.feature)

    def slot_fields(self, slot: SlotInfo) -> dict:
        return {
            "feature_id": slot.feature["feature_id"],
            "feature_label": slot.feature["label"],
            "pass_idx": slot.pass_idx,
            "feature_attempt": slot.feature_attempt,
        }

    def check(self, result, slot: SlotInfo) -> list[dict]:
        """SAE check of every candidate of a returned call (main thread, completion order - the
        result depends only on (candidate, feature), so the run stays deterministic)."""
        checks = []
        feature_id = slot.feature["feature_id"]
        for text in result.candidates if result.ok else []:
            act = target_activation(sae_forward(text, self.sae_ctx), feature_id)
            checks.append({
                "active": act["max_raw_activation"] > self.args.threshold,
                "summary": f"f={feature_id} act={act['max_raw_activation']:.3f}",
                "fields": {
                    "sae_max_raw_activation": act["max_raw_activation"],
                    "sae_n_active_tokens": act["n_active_tokens"],
                    "sae_n_content_tokens": act["n_content_tokens"],
                    "sae_top_token": act["top_token"],
                    "sae_n_forward_tokens": act["n_forward_tokens"],
                    "sae_seconds": act["seconds"],
                },
            })
        return checks

    def call_fields(self, checks: list[dict]) -> dict:
        stats = new_sae_stats()
        for check in checks:
            add_sae_pass(stats, check["fields"]["sae_n_forward_tokens"], check["fields"]["sae_seconds"])
        return {f"sae_{key}": value for key, value in stats.items()}

    def after_call(self, slot: SlotInfo, outcome: str, call_outcomes: list[dict], checks: list[dict]) -> None:
        feature_id = slot.feature["feature_id"]
        stats = self.state["feature_stats"].setdefault(str(feature_id), {
            "label": slot.feature["label"], "attempts": 0, "accepted": 0, "feature_inactive": 0,
            "rouge_duplicate": 0, "no_candidate": 0, "failed": 0,
        })
        stats["attempts"] += 1
        if outcome == "failed":
            stats["failed"] += 1
        elif not call_outcomes:
            stats["no_candidate"] += 1
        for call_outcome in call_outcomes:
            if call_outcome["status"] == "discarded":
                continue  # never checked
            reason = "accepted" if call_outcome["status"] == "accepted" else call_outcome["rejected_reason"]
            if reason in stats:
                stats[reason] += 1
        # Every SAE pass of the call ran (also those of later discarded candidates).
        for check in checks:
            add_sae_pass(self.state["counters"]["sae_candidate_check"],
                         check["fields"]["sae_n_forward_tokens"], check["fields"]["sae_seconds"])
        self._update_schedule(feature_id, outcome)

    def after_wave(self) -> None:
        self.advance_pass_if_done()

    # -- reporting ------------------------------------------------------------

    def sae_totals(self) -> dict:
        counters = self.state["counters"]
        total = new_sae_stats()
        for key in ("sae_coverage_check", "sae_candidate_check"):
            for stat in total:
                total[stat] += counters[key][stat]
        total["gpu_seconds"] = round(total["gpu_seconds"], 6)
        return total

    def log_fields(self) -> dict:
        args, schedule, counters = self.args, self.schedule, self.state["counters"]
        fields = {
            "feature_scores": str(resolve_feature_scores(args)),
            "feature_labels": list(args.feature_labels),
            "attempts_per_feature": args.attempts_per_feature,
            "threshold": args.threshold,
            "n_relevant_features": len(self.features),
            "coverage_by_label": coverage_by_label(self.features, schedule, args.feature_labels),
        }
        for key in ("seed_covered", "seed_uncovered", "blackbox_covered", "blackbox_only_covered", "covered", "uncovered"):
            if key in schedule:
                fields[f"n_{key}"] = len(schedule[key])
        for key in ("seed_covered", "seed_uncovered", "blackbox_covered", "blackbox_only_covered", "covered",
                    "uncovered", "seed_peak_activations", "blackbox_peak_activations", "blackbox_entry_active_features"):
            if key in schedule:
                fields[key] = schedule[key]
        fields.update(
            n_passes_started=schedule["pass_idx"] + 1,
            n_exhausted=len(schedule["exhausted"]),
            exhausted=schedule["exhausted"],
            exhausted_in_pass=schedule["exhausted_in_pass"],
            n_features_used=len(self.state["feature_stats"]),
            sae_model=os.path.abspath(args.model_name),
            sae_ckpt_path=os.path.abspath(args.sae_ckpt_path),
            sae_layer=args.layer,
            sae_device=args.device,
            sae_coverage_check=counters["sae_coverage_check"],
            sae_candidate_check=counters["sae_candidate_check"],
            sae_total=self.sae_totals(),
        )
        return fields

    def print_sae_summary(self) -> None:
        counters, total = self.state["counters"], self.sae_totals()
        print(f"SAE: {total['n_forward_passes']} forward pass(es), {total['n_forward_tokens']} token(s), "
              f"{total['gpu_seconds']:.2f} s net GPU time (coverage check {counters['sae_coverage_check']['gpu_seconds']:.2f} s, "
              f"candidate check {counters['sae_candidate_check']['gpu_seconds']:.2f} s)")


def start_feature_guidance(args: argparse.Namespace, state: dict, seed_file: Path,
                           blackbox_entries: list[dict] | None = None) -> FeatureGuidance:
    """Loads features + Llama/SAE and, unless already checkpointed in `state`, computes the
    coverage of the seeds (and hybrid's blackbox examples) and the initial schedule."""
    feature_scores = resolve_feature_scores(args)
    features = load_features(feature_scores, args.feature_labels)
    print(f"Relevant features: {len(features)} "
          f"{ {label: sum(f['label'] == label for f in features) for label in args.feature_labels} } from {feature_scores}, "
          f"up to {args.attempts_per_feature} attempt(s) per feature and pass, SAE raw threshold {args.threshold}")
    sae_ctx = load_sae_context(args)
    if "schedule" not in state:
        counters = state["counters"]
        counters["sae_coverage_check"] = new_sae_stats()
        counters["sae_candidate_check"] = new_sae_stats()
        seed_texts = load_raw_seed_texts(seed_file)
        n_bb = len(blackbox_entries) if blackbox_entries is not None else 0
        print(f"Computing coverage of the relevant features on {len(seed_texts)} seed(s)"
              + (f" + {n_bb} blackbox example(s)" if blackbox_entries is not None else "") + "...")
        seed_peaks = text_peaks(seed_texts, sae_ctx, counters["sae_coverage_check"])
        bb_peaks = (text_peaks([e["text"] for e in blackbox_entries], sae_ctx, counters["sae_coverage_check"])
                    if blackbox_entries is not None else None)
        state["schedule"] = new_schedule(features, args.threshold, seed_peaks, blackbox_entries, bb_peaks)
        state["feature_stats"] = {}
    guidance = FeatureGuidance(args, features, sae_ctx, state)
    for label, c in coverage_by_label(features, state["schedule"], args.feature_labels).items():
        print(f"  [coverage] {label:<9} seeds {c['seed_covered']:>4}"
              + (f" / blackbox {c['blackbox_covered']:>4}" if "blackbox_covered" in c else "")
              + f" / covered {c['covered']:>4} / uncovered {c['uncovered']:>4} / total {c['total']:>4}")
    guidance.advance_pass_if_done()
    return guidance
