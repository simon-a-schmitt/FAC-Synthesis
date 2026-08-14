#!/usr/bin/env python3
"""
Contamination check: synthetic (deepseek_02) CVE descriptions vs. gold (test_500)
CVE descriptions.

Metrics:
  1a) Exact match (on normalized text) of synthetic samples against gold.
  1b) Near-exact match via normalized Levenshtein similarity
      (rapidfuzz.distance.Levenshtein.normalized_similarity - true
      Levenshtein/edit distance with substitutions, NOT indel/LCS-based
      fuzz.ratio). Distribution for synth-vs-gold and a leave-one-out
      reference distribution computed on gold-500 alone. Percentiles for
      both + how many synthetic samples would be flagged at the gold
      leave-one-out p95 / p99 thresholds.
  2)  n=8 word-gram overlap:
      2a) C_max  - overlap fraction with the single most similar gold item.
      2b) C_corp - fraction of the sample's n-grams hit anywhere in the gold
                   corpus.
      2c) for every matched n-gram, how many gold documents contain it
          (document frequency).
      2d) how many of the sample's n-grams are shared with the gold corpus
          but occur in exactly one gold document (df == 1, i.e. rare/highly
          specific overlap).
      2e) hard flag: does the sample share a 13-gram with the gold corpus.
      All of 2a/2b/2d/2e are additionally computed as a leave-one-out
      reference on gold-500 alone.
  3)  Longest common contiguous token run shared with any single gold item
      (again with a leave-one-out reference computed on gold-500 alone).

Requires: numpy, rapidfuzz (pip install rapidfuzz).

Usage:
    python3 contamination_check.py \
        --gold ../cti_vsp_benchmark_test_500.tsv \
        --synth ../cti_vsp_bb_deepseek_02.tsv \
        --out-dir .
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

CVE_MARKER = "CVE Description: "
NGRAM_N = 8
LONG_NGRAM_N = 13
PERCENTILES = [50, 75, 90, 95, 99, 99.9, 100]


# --------------------------------------------------------------------------
# Loading & normalization
# --------------------------------------------------------------------------

def load_prompts(path: Path) -> List[str]:
    """Read a TSV file and return the raw first column ('prompt' field)."""
    prompts: List[str] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            if not row or not row[0].strip():
                continue
            prompts.append(row[0])
    return prompts


def extract_cve_description(prompt: str) -> str:
    """Keep only the text behind 'CVE Description: ' if present, else the
    prompt as-is (deepseek_02 prompts are already bare descriptions)."""
    if CVE_MARKER in prompt:
        return prompt.split(CVE_MARKER, 1)[1].strip()
    return prompt.strip()


_WS_RE = re.compile(r"\s+")


def norm(text: str) -> str:
    """NFKC -> lowercase -> strip punctuation (keep '.' inside digit
    sequences, e.g. version numbers like '1.5.3') -> collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()

    chars = []
    for i, ch in enumerate(text):
        if ch == ".":
            prev_digit = i > 0 and text[i - 1].isdigit()
            next_digit = i + 1 < len(text) and text[i + 1].isdigit()
            if prev_digit and next_digit:
                chars.append(ch)
            continue  # drop '.' that is not between two digits
        if unicodedata.category(ch).startswith("P"):
            continue  # drop all other punctuation
        chars.append(ch)
    text = "".join(chars)

    text = _WS_RE.sub(" ", text).strip()
    return text


@dataclass
class Item:
    idx: int
    raw_prompt: str
    description: str
    norm_text: str
    tokens: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.tokens:
            self.tokens = self.norm_text.split()


def build_items(prompts: Sequence[str]) -> List[Item]:
    items = []
    for i, p in enumerate(prompts):
        desc = extract_cve_description(p)
        items.append(Item(idx=i, raw_prompt=p, description=desc, norm_text=norm(desc)))
    return items


# --------------------------------------------------------------------------
# 1a) exact match / 1b) near-exact match (normalized Levenshtein)
# --------------------------------------------------------------------------

def exact_match_analysis(synth: List[Item], gold: List[Item]) -> dict:
    gold_by_text: Dict[str, List[int]] = defaultdict(list)
    for g in gold:
        gold_by_text[g.norm_text].append(g.idx)

    matches = []
    for s in synth:
        if s.norm_text in gold_by_text:
            matches.append({
                "synth_idx": s.idx,
                "gold_idx": gold_by_text[s.norm_text],
                "synth_description": s.description,
            })
    return {
        "n_synth": len(synth),
        "n_exact_matches": len(matches),
        "fraction": len(matches) / len(synth) if synth else 0.0,
        "matches": matches,
    }


def percentiles(values: np.ndarray, ps: Sequence[float] = PERCENTILES) -> Dict[str, float]:
    if len(values) == 0:
        return {f"p{p}": float("nan") for p in ps}
    return {f"p{p}": float(np.percentile(values, p)) for p in ps}


def levenshtein_analysis(synth: List[Item], gold: List[Item]) -> dict:
    synth_texts = [s.norm_text for s in synth]
    gold_texts = [g.norm_text for g in gold]

    cross = process.cdist(synth_texts, gold_texts, scorer=Levenshtein.normalized_similarity, workers=-1)
    cross_max = cross.max(axis=1)
    cross_argmax = cross.argmax(axis=1)

    gold_mat = process.cdist(gold_texts, gold_texts, scorer=Levenshtein.normalized_similarity, workers=-1)
    np.fill_diagonal(gold_mat, -1.0)
    ref_max = gold_mat.max(axis=1)
    ref_argmax = gold_mat.argmax(axis=1)

    ref_p95 = float(np.percentile(ref_max, 95))
    ref_p99 = float(np.percentile(ref_max, 99))

    flagged_p95 = [int(i) for i in np.where(cross_max >= ref_p95)[0]]
    flagged_p99 = [int(i) for i in np.where(cross_max >= ref_p99)[0]]

    return {
        "cross_distribution": {
            "n": len(cross_max),
            "percentiles": percentiles(cross_max),
            "mean": float(np.mean(cross_max)),
        },
        "reference_loo_gold_distribution": {
            "n": len(ref_max),
            "percentiles": percentiles(ref_max),
            "mean": float(np.mean(ref_max)),
        },
        "reference_p95_threshold": ref_p95,
        "reference_p99_threshold": ref_p99,
        "n_flagged_at_p95": len(flagged_p95),
        "frac_flagged_at_p95": len(flagged_p95) / len(synth) if synth else 0.0,
        "n_flagged_at_p99": len(flagged_p99),
        "frac_flagged_at_p99": len(flagged_p99) / len(synth) if synth else 0.0,
        "flagged_p95_idx": flagged_p95,
        "flagged_p99_idx": flagged_p99,
        "_cross_max": cross_max,
        "_cross_argmax": cross_argmax,
        "_ref_max": ref_max,
        "_ref_argmax": ref_argmax,
    }


# --------------------------------------------------------------------------
# 2) n-gram overlap (n=8) + 13-gram hard flag
# --------------------------------------------------------------------------

def get_ngrams(tokens: Sequence[str], n: int) -> List[Tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def build_ngram_index(gold: List[Item], n: int) -> Dict[Tuple[str, ...], Set[int]]:
    index: Dict[Tuple[str, ...], Set[int]] = defaultdict(set)
    for g in gold:
        for ng in set(get_ngrams(g.tokens, n)):
            index[ng].add(g.idx)
    return index


@dataclass
class NgramResult:
    idx: int
    n_ngrams: int
    c_max: float
    c_max_doc: Optional[int]
    c_max_shared: int
    c_corp: float
    n_matched_ngrams: int
    n_unique_doc_ngrams: int  # matched ngrams whose gold df == 1
    unique_doc_examples: List[Tuple[Tuple[str, ...], int]]
    matched_ngram_dfs: List[int]
    flagged_13gram: bool
    flagged_13gram_doc: Optional[int]
    flagged_13gram_example: Optional[Tuple[str, ...]]


def analyze_ngrams_for_item(
    item: Item,
    index8: Dict[Tuple[str, ...], Set[int]],
    index13: Dict[Tuple[str, ...], Set[int]],
    exclude_idx: Optional[int] = None,
) -> NgramResult:
    ngrams8 = set(get_ngrams(item.tokens, NGRAM_N))
    doc_hits: Counter = Counter()
    matched_ngrams: Set[Tuple[str, ...]] = set()
    unique_doc_examples: List[Tuple[Tuple[str, ...], int]] = []
    matched_ngram_dfs: List[int] = []

    for ng in ngrams8:
        docs = index8.get(ng)
        if not docs:
            continue
        if exclude_idx is not None and exclude_idx in docs:
            docs = docs - {exclude_idx}
            if not docs:
                continue
        matched_ngrams.add(ng)
        matched_ngram_dfs.append(len(docs))
        for d in docs:
            doc_hits[d] += 1
        if len(docs) == 1:
            unique_doc_examples.append((ng, next(iter(docs))))

    if doc_hits:
        c_max_doc, c_max_shared = doc_hits.most_common(1)[0]
    else:
        c_max_doc, c_max_shared = None, 0

    n_ngrams = len(ngrams8)
    c_max = c_max_shared / n_ngrams if n_ngrams else 0.0
    c_corp = len(matched_ngrams) / n_ngrams if n_ngrams else 0.0

    # 13-gram hard flag
    flagged_doc = None
    flagged_example = None
    for ng in set(get_ngrams(item.tokens, LONG_NGRAM_N)):
        docs = index13.get(ng)
        if not docs:
            continue
        if exclude_idx is not None:
            docs = docs - {exclude_idx}
            if not docs:
                continue
        flagged_doc = next(iter(docs))
        flagged_example = ng
        break

    return NgramResult(
        idx=item.idx,
        n_ngrams=n_ngrams,
        c_max=c_max,
        c_max_doc=c_max_doc,
        c_max_shared=c_max_shared,
        c_corp=c_corp,
        n_matched_ngrams=len(matched_ngrams),
        n_unique_doc_ngrams=len(unique_doc_examples),
        unique_doc_examples=unique_doc_examples,
        matched_ngram_dfs=matched_ngram_dfs,
        flagged_13gram=flagged_doc is not None,
        flagged_13gram_doc=flagged_doc,
        flagged_13gram_example=flagged_example,
    )


def ngram_analysis(synth: List[Item], gold: List[Item]) -> dict:
    index8 = build_ngram_index(gold, NGRAM_N)
    index13 = build_ngram_index(gold, LONG_NGRAM_N)

    cross_results = [analyze_ngrams_for_item(s, index8, index13, exclude_idx=None) for s in synth]
    ref_results = [analyze_ngrams_for_item(g, index8, index13, exclude_idx=g.idx) for g in gold]

    def summarize(results: List[NgramResult]) -> dict:
        c_max_arr = np.array([r.c_max for r in results])
        c_corp_arr = np.array([r.c_corp for r in results])
        unique_arr = np.array([r.n_unique_doc_ngrams for r in results])
        flagged13 = sum(1 for r in results if r.flagged_13gram)
        all_dfs = [df for r in results for df in r.matched_ngram_dfs]
        df_hist = Counter()
        for df in all_dfs:
            if df == 1:
                df_hist["df=1"] += 1
            elif df <= 3:
                df_hist["df=2-3"] += 1
            elif df <= 5:
                df_hist["df=4-5"] += 1
            else:
                df_hist["df>5"] += 1
        return {
            "n": len(results),
            "c_max_percentiles": percentiles(c_max_arr),
            "c_max_mean": float(np.mean(c_max_arr)) if len(c_max_arr) else 0.0,
            "c_corp_percentiles": percentiles(c_corp_arr),
            "c_corp_mean": float(np.mean(c_corp_arr)) if len(c_corp_arr) else 0.0,
            "unique_doc_ngram_count_percentiles": percentiles(unique_arr),
            "unique_doc_ngram_count_mean": float(np.mean(unique_arr)) if len(unique_arr) else 0.0,
            "n_flagged_13gram": flagged13,
            "frac_flagged_13gram": flagged13 / len(results) if results else 0.0,
            "matched_ngram_df_histogram": dict(df_hist),
        }

    cross_summary = summarize(cross_results)
    ref_summary = summarize(ref_results)

    ref_c_max_arr = np.array([r.c_max for r in ref_results])
    cross_c_max_arr = np.array([r.c_max for r in cross_results])
    ref_p95 = float(np.percentile(ref_c_max_arr, 95)) if len(ref_c_max_arr) else float("nan")
    ref_p99 = float(np.percentile(ref_c_max_arr, 99)) if len(ref_c_max_arr) else float("nan")
    flagged_p95 = int(np.sum(cross_c_max_arr >= ref_p95)) if len(cross_c_max_arr) else 0
    flagged_p99 = int(np.sum(cross_c_max_arr >= ref_p99)) if len(cross_c_max_arr) else 0

    return {
        "cross": cross_summary,
        "reference_loo_gold": ref_summary,
        "c_max_reference_p95_threshold": ref_p95,
        "c_max_reference_p99_threshold": ref_p99,
        "n_flagged_c_max_at_ref_p95": flagged_p95,
        "n_flagged_c_max_at_ref_p99": flagged_p99,
        "_cross_results": cross_results,
        "_ref_results": ref_results,
    }


# --------------------------------------------------------------------------
# 3) longest common contiguous token run
# --------------------------------------------------------------------------

def build_all_length_index(gold: List[Item]) -> Dict[Tuple[str, ...], Set[int]]:
    """Map every contiguous token substring (any length) occurring in the
    gold corpus to the set of gold doc indices containing it."""
    index: Dict[Tuple[str, ...], Set[int]] = defaultdict(set)
    for g in gold:
        toks = g.tokens
        n = len(toks)
        for length in range(1, n + 1):
            for i in range(n - length + 1):
                index[tuple(toks[i:i + length])].add(g.idx)
    return index


def longest_common_run(
    tokens: Sequence[str],
    index: Dict[Tuple[str, ...], Set[int]],
    exclude_idx: Optional[int] = None,
) -> Tuple[int, Optional[int]]:
    """Binary search (existence of a common run is monotonic in length) for
    the longest contiguous token run shared with the (excl.-adjusted) index."""
    lo, hi = 1, len(tokens)
    best_len, best_doc = 0, None
    while lo <= hi:
        mid = (lo + hi) // 2
        found_doc = None
        for i in range(len(tokens) - mid + 1):
            docs = index.get(tuple(tokens[i:i + mid]))
            if not docs:
                continue
            if exclude_idx is not None:
                docs = docs - {exclude_idx}
                if not docs:
                    continue
            found_doc = next(iter(docs))
            break
        if found_doc is not None:
            best_len, best_doc = mid, found_doc
            lo = mid + 1
        else:
            hi = mid - 1
    return best_len, best_doc


def longest_run_analysis(synth: List[Item], gold: List[Item]) -> dict:
    index = build_all_length_index(gold)

    cross = [longest_common_run(s.tokens, index, exclude_idx=None) for s in synth]
    ref = [longest_common_run(g.tokens, index, exclude_idx=g.idx) for g in gold]

    cross_lens = np.array([c[0] for c in cross])
    ref_lens = np.array([r[0] for r in ref])

    ref_p95 = float(np.percentile(ref_lens, 95)) if len(ref_lens) else float("nan")
    ref_p99 = float(np.percentile(ref_lens, 99)) if len(ref_lens) else float("nan")
    flagged_p95 = int(np.sum(cross_lens >= ref_p95)) if len(cross_lens) else 0
    flagged_p99 = int(np.sum(cross_lens >= ref_p99)) if len(cross_lens) else 0

    return {
        "cross_percentiles": percentiles(cross_lens),
        "cross_mean": float(np.mean(cross_lens)) if len(cross_lens) else 0.0,
        "reference_loo_gold_percentiles": percentiles(ref_lens),
        "reference_loo_gold_mean": float(np.mean(ref_lens)) if len(ref_lens) else 0.0,
        "reference_p95_threshold": ref_p95,
        "reference_p99_threshold": ref_p99,
        "n_flagged_at_p95": flagged_p95,
        "frac_flagged_at_p95": flagged_p95 / len(synth) if synth else 0.0,
        "n_flagged_at_p99": flagged_p99,
        "frac_flagged_at_p99": flagged_p99 / len(synth) if synth else 0.0,
        "_cross": cross,
        "_ref": ref,
    }


# --------------------------------------------------------------------------
# Log file: highest overlaps per metric, with actual text + overlap type
# --------------------------------------------------------------------------

def shorten(text: str, limit: int = 300) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + " [...]"


def write_log_file(
    path: Path,
    synth: List[Item],
    gold: List[Item],
    lev: dict,
    ngram: dict,
    run: dict,
    top_k: int,
) -> None:
    lines: List[str] = []

    def hdr(title: str) -> None:
        lines.append("")
        lines.append("=" * 100)
        lines.append(title)
        lines.append("=" * 100)

    # ---- exact matches (1a) ----
    hdr("1a) EXACT MATCHES (normalized text, synthetic == gold)")
    exact = exact_match_analysis(synth, gold)
    if not exact["matches"]:
        lines.append("(none found)")
    for m in exact["matches"]:
        lines.append(f"synth #{m['synth_idx']}  <->  gold #{m['gold_idx']}")
        lines.append(f"  text: {shorten(m['synth_description'])}")

    # ---- 1b) near-exact match: top-K by normalized Levenshtein similarity ----
    hdr(f"1b) TOP {top_k} NEAR-EXACT MATCHES (normalized Levenshtein similarity)")
    cross_max = lev["_cross_max"]
    cross_argmax = lev["_cross_argmax"]
    order = np.argsort(-cross_max)[:top_k]
    for rank, si in enumerate(order, 1):
        si = int(si)
        gi = int(cross_argmax[si])
        lines.append(f"#{rank}  score={cross_max[si]:.4f}  synth #{si}  <->  gold #{gi}")
        lines.append(f"  overlap type: near-exact match (normalized Levenshtein similarity)")
        lines.append(f"  synth: {shorten(synth[si].description)}")
        lines.append(f"  gold : {shorten(gold[gi].description)}")

    # ---- 2) n-gram overlap: top-K by C_max ----
    hdr(f"2) TOP {top_k} N-GRAM OVERLAPS (n={NGRAM_N}, C_max = shared n-grams / synth n-grams)")
    cross_ngram: List[NgramResult] = ngram["_cross_results"]
    ranked = sorted(cross_ngram, key=lambda r: r.c_max, reverse=True)[:top_k]
    for rank, r in enumerate(ranked, 1):
        if r.c_max_doc is None:
            continue
        lines.append(
            f"#{rank}  C_max={r.c_max:.4f}  ({r.c_max_shared}/{r.n_ngrams} shared 8-grams)"
            f"  synth #{r.idx}  <->  gold #{r.c_max_doc}"
        )
        lines.append(
            f"  overlap type: n-gram overlap (n={NGRAM_N}); C_corp={r.c_corp:.4f}; "
            f"unique-to-one-gold-doc n-grams={r.n_unique_doc_ngrams}; "
            f"13-gram flag={'YES (gold #' + str(r.flagged_13gram_doc) + ')' if r.flagged_13gram else 'no'}"
        )
        lines.append(f"  synth: {shorten(synth[r.idx].description)}")
        lines.append(f"  gold : {shorten(gold[r.c_max_doc].description)}")

    # ---- 2e) 13-gram hard flags, full list ----
    hdr(f"2e) ALL 13-GRAM HARD-FLAGGED SAMPLES (n={LONG_NGRAM_N})")
    flagged13 = [r for r in cross_ngram if r.flagged_13gram]
    if not flagged13:
        lines.append("(none found)")
    for r in flagged13:
        lines.append(f"synth #{r.idx}  <->  gold #{r.flagged_13gram_doc}")
        lines.append(f"  shared 13-gram: {' '.join(r.flagged_13gram_example)}")
        lines.append(f"  synth: {shorten(synth[r.idx].description)}")
        lines.append(f"  gold : {shorten(gold[r.flagged_13gram_doc].description)}")

    # ---- 3) longest common contiguous token run: top-K ----
    hdr(f"3) TOP {top_k} LONGEST COMMON CONTIGUOUS TOKEN RUNS")
    cross_run = run["_cross"]
    ranked_idx = sorted(range(len(cross_run)), key=lambda i: cross_run[i][0], reverse=True)[:top_k]
    for rank, si in enumerate(ranked_idx, 1):
        length, gi = cross_run[si]
        if gi is None:
            continue
        # recover the actual matched token run (first occurrence) for context
        s_tokens = synth[si].tokens
        g_tokens = gold[gi].tokens
        g_ngrams_at_len = set(get_ngrams(g_tokens, length))
        matched_text = None
        for i in range(len(s_tokens) - length + 1):
            cand = tuple(s_tokens[i:i + length])
            if cand in g_ngrams_at_len:
                matched_text = " ".join(cand)
                break
        lines.append(f"#{rank}  run_length={length} tokens  synth #{si}  <->  gold #{gi}")
        lines.append(f"  overlap type: longest common contiguous token run")
        if matched_text:
            lines.append(f"  shared run: {shorten(matched_text)}")
        lines.append(f"  synth: {shorten(synth[si].description)}")
        lines.append(f"  gold : {shorten(gold[gi].description)}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def make_json_safe(obj):
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def print_and_collect(lines: List[str], text: str = "") -> None:
    print(text)
    lines.append(text)


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", type=Path, default=here.parent / "cti_vsp_benchmark_test_500.tsv")
    ap.add_argument("--synth", type=Path, default=here.parent / "cti_vsp_bb_deepseek_02.tsv")
    ap.add_argument("--out-dir", type=Path, default=here)
    ap.add_argument("--top-k", type=int, default=20)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    gold = build_items(load_prompts(args.gold))
    synth = build_items(load_prompts(args.synth))

    print(f"Loaded {len(gold)} gold items from {args.gold}")
    print(f"Loaded {len(synth)} synthetic items from {args.synth}")

    exact = exact_match_analysis(synth, gold)
    lev = levenshtein_analysis(synth, gold)
    ngram = ngram_analysis(synth, gold)
    run = longest_run_analysis(synth, gold)

    report = {
        "n_gold": len(gold),
        "n_synth": len(synth),
        "1a_exact_match": {k: v for k, v in exact.items() if k != "matches"} | {
            "n_matches_listed_in_log": len(exact["matches"])
        },
        "1b_near_exact_levenshtein": {k: v for k, v in lev.items() if not k.startswith("_")},
        "2_ngram_overlap": {k: v for k, v in ngram.items() if not k.startswith("_")},
        "3_longest_common_run": {k: v for k, v in run.items() if not k.startswith("_")},
    }

    report_path = args.out_dir / "contamination_report.json"
    report_path.write_text(json.dumps(make_json_safe(report), indent=2), encoding="utf-8")

    summary_lines: List[str] = []
    print_and_collect(summary_lines, "")
    print_and_collect(summary_lines, "=" * 80)
    print_and_collect(summary_lines, "CONTAMINATION CHECK: deepseek_02 (synthetic) vs. test_500 (gold)")
    print_and_collect(summary_lines, "=" * 80)

    print_and_collect(summary_lines, "\n--- 1a) Exact match ---")
    print_and_collect(
        summary_lines,
        f"{exact['n_exact_matches']}/{exact['n_synth']} synthetic samples are exact "
        f"(normalized) matches of a gold item ({exact['fraction']*100:.2f}%).",
    )

    print_and_collect(summary_lines, "\n--- 1b) Near-exact match (normalized Levenshtein similarity) ---")
    print_and_collect(summary_lines, "Cross distribution (synth vs gold-500) percentiles:")
    for k, v in lev["cross_distribution"]["percentiles"].items():
        print_and_collect(summary_lines, f"  {k}: {v:.4f}")
    print_and_collect(summary_lines, "Reference distribution (gold-500 leave-one-out) percentiles:")
    for k, v in lev["reference_loo_gold_distribution"]["percentiles"].items():
        print_and_collect(summary_lines, f"  {k}: {v:.4f}")
    print_and_collect(
        summary_lines,
        f"Reference p95={lev['reference_p95_threshold']:.4f} -> "
        f"{lev['n_flagged_at_p95']}/{exact['n_synth']} synth samples flagged "
        f"({lev['frac_flagged_at_p95']*100:.2f}%)",
    )
    print_and_collect(
        summary_lines,
        f"Reference p99={lev['reference_p99_threshold']:.4f} -> "
        f"{lev['n_flagged_at_p99']}/{exact['n_synth']} synth samples flagged "
        f"({lev['frac_flagged_at_p99']*100:.2f}%)",
    )

    print_and_collect(summary_lines, f"\n--- 2) n-gram overlap (n={NGRAM_N}) ---")
    c = ngram["cross"]
    r = ngram["reference_loo_gold"]
    print_and_collect(summary_lines, f"C_max  cross mean={c['c_max_mean']:.4f}  ref(loo) mean={r['c_max_mean']:.4f}")
    print_and_collect(summary_lines, f"C_corp cross mean={c['c_corp_mean']:.4f}  ref(loo) mean={r['c_corp_mean']:.4f}")
    print_and_collect(
        summary_lines,
        f"C_max reference p95={ngram['c_max_reference_p95_threshold']:.4f} -> "
        f"{ngram['n_flagged_c_max_at_ref_p95']}/{exact['n_synth']} flagged",
    )
    print_and_collect(
        summary_lines,
        f"C_max reference p99={ngram['c_max_reference_p99_threshold']:.4f} -> "
        f"{ngram['n_flagged_c_max_at_ref_p99']}/{exact['n_synth']} flagged",
    )
    print_and_collect(
        summary_lines,
        f"Unique-to-single-gold-doc n-gram matches (2d): cross mean={c['unique_doc_ngram_count_mean']:.2f}  "
        f"ref(loo) mean={r['unique_doc_ngram_count_mean']:.2f}",
    )
    print_and_collect(summary_lines, f"Matched n-gram gold-doc-frequency histogram (cross): {c['matched_ngram_df_histogram']}")
    print_and_collect(
        summary_lines,
        f"13-gram hard flag (2e): cross {c['n_flagged_13gram']}/{exact['n_synth']} "
        f"({c['frac_flagged_13gram']*100:.2f}%)  vs. ref(loo) {r['n_flagged_13gram']}/{len(gold)} "
        f"({r['frac_flagged_13gram']*100:.2f}%)",
    )

    print_and_collect(summary_lines, "\n--- 3) Longest common contiguous token run ---")
    print_and_collect(summary_lines, f"Cross percentiles: {run['cross_percentiles']}")
    print_and_collect(summary_lines, f"Reference (loo) percentiles: {run['reference_loo_gold_percentiles']}")
    print_and_collect(
        summary_lines,
        f"Reference p95={run['reference_p95_threshold']:.1f} tokens -> "
        f"{run['n_flagged_at_p95']}/{exact['n_synth']} flagged "
        f"({run['frac_flagged_at_p95']*100:.2f}%)",
    )
    print_and_collect(
        summary_lines,
        f"Reference p99={run['reference_p99_threshold']:.1f} tokens -> "
        f"{run['n_flagged_at_p99']}/{exact['n_synth']} flagged "
        f"({run['frac_flagged_at_p99']*100:.2f}%)",
    )

    summary_path = args.out_dir / "contamination_report.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    log_path = args.out_dir / "contamination_log.txt"
    write_log_file(log_path, synth, gold, lev, ngram, run, top_k=args.top_k)

    print(f"\nWrote: {report_path}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {log_path}")


if __name__ == "__main__":
    main()
