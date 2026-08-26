"""
Select --n examples from a CLAUDETTE-TOS pool TSV under one of three --mode
strategies. Conceptually analogous to
feature_coverage_filter/cti_vsp/run_feature_coverage_filter_select.py, but
for the CLAUDETTE-TOS 8-way binary unfairness-type vector instead of the
CVSS categorical vector, and with a positive-label quota instead of
distribution matching as the greedy tie-break/side constraint.

Input
-----
- --pool-tsv: headerless TSV, column 0 = ToS sentence, column 1 = unfairness
  vector "LTD:Y|TER:N|CH:N|CR:N|USE:N|LAW:N|J:N|ARB:N|"
  (feature_coverage_filter/claudette_tos/input), e.g. claudette_tos_train.tsv.
  Each row's 1-based line number is its pool_id, matching the convention used
  by run_feature_coverage_filter.py.
- --pool-features-jsonl: output of run_feature_coverage_filter.py, one JSON
  object per line: {"pool_id": ..., "n_content_tokens": ..., "features":
  {feature_id: magnitude, ...}}.

Processing
----------
A feature counts as "active" on an example if its magnitude > --threshold.
--mode selects the strategy:

  - max_features (default): greedily maximise the number of distinct active
    features covered by the selection.
  - min_features: greedily minimise it (0-gain examples first).
  - random: --n examples are drawn uniformly at random from the pool
    (--seed for reproducibility). Feature coverage plays no role.

For max_features/min_features, a side constraint applies: at least
--min-positive-fraction (default 0.11, i.e. 11%) of the n selected examples
must have at least one positive ("Y") label in their unfairness vector. This
is enforced with a two-phase greedy: first, ceil(n * min_positive_fraction)
picks are made greedily (by the same coverage-gain criterion, max or min)
restricted to examples with >=1 positive label; then the remaining picks are
made greedily over the whole remaining pool. This guarantees the quota is
met (or, if the pool doesn't contain enough positive examples, that every
positive example is used) while still optimising coverage. Ties in coverage
gain are broken by the smaller pool_id. --mode random draws uniformly at
random and is not subject to this constraint.

Output
------
- A TSV of the n selected examples (original 2-column pool format, original
  pool order) written to feature_coverage_filter/claudette_tos/output.
- A summary JSON with:
  - the distribution of result-vector "signatures" (the set of positive
    labels, e.g. "{}" or "{ARB, J}") among the selection,
  - the marginal Y/N distribution of each of the 8 submetrics, and
  - the number of unique active features covered,
  reported for all three modes.
"""

import argparse
import collections
import csv
import json
import math
import os
import random
from typing import Any, Dict, List, Set, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_POOL_TSV = os.path.join(_SCRIPT_DIR, "input", "claudette_tos_train.tsv")
_DEFAULT_POOL_FEATURES_JSONL = os.path.join(
    _SCRIPT_DIR, "output", "claudette_tos_train_fc_filter.jsonl"
)

METRICS = ["LTD", "TER", "CH", "CR", "USE", "LAW", "J", "ARB"]
CLASSES = {m: ["Y", "N"] for m in METRICS}


# ---------------------------------------------------------------------------
# Unfairness-vector parsing
# ---------------------------------------------------------------------------

def parse_vector(vec: str) -> Dict[str, str]:
    """'LTD:Y|TER:N|CH:N|CR:N|USE:N|LAW:N|J:N|ARB:N|' -> {'LTD': 'Y', ...}"""
    out: Dict[str, str] = {}
    for part in vec.strip().split("|"):
        if not part or ":" not in part:
            continue
        k, v = part.split(":", 1)
        if k in CLASSES:
            if v not in CLASSES[k]:
                raise ValueError(f"unbekannte Klasse {k}:{v} in {vec!r}")
            out[k] = v
    missing = [m for m in METRICS if m not in out]
    if missing:
        raise ValueError(f"fehlende Metriken {missing} in {vec!r}")
    return out


def signature_str(label: Dict[str, str]) -> str:
    """{'LTD': 'N', 'J': 'Y', 'ARB': 'Y', ...} -> '{ARB, J}' (sorted, '{}' if none)."""
    positives = sorted(m for m in METRICS if label[m] == "Y")
    return "{" + ", ".join(positives) + "}"


def has_positive(label: Dict[str, str]) -> bool:
    return any(label[m] == "Y" for m in METRICS)


def marginals(labels: List[Dict[str, str]]) -> Dict[str, Dict[str, float]]:
    n = len(labels)
    out: Dict[str, Dict[str, float]] = {}
    for m in METRICS:
        cnt = collections.Counter(lab[m] for lab in labels)
        out[m] = {c: cnt.get(c, 0) / n for c in CLASSES[m]}
    return out


def signature_distribution(labels: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    n = len(labels)
    counter = collections.Counter(signature_str(lab) for lab in labels)
    entries = [
        {"signature": sig, "count": cnt, "fraction": cnt / n}
        for sig, cnt in counter.items()
    ]
    entries.sort(key=lambda e: (-e["count"], e["signature"]))
    return entries


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_pool_rows(pool_tsv: str) -> List[Dict[str, Any]]:
    """Read {pool_id, text, vec, label} from a headerless <sentence>\\t<vector> TSV.

    pool_id is the 1-based line number, identical to run_feature_coverage_filter.py's
    load_pool_items(), so pool_ids line up with --pool-features-jsonl.
    """
    rows: List[Dict[str, Any]] = []
    with open(pool_tsv, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for line_no, row in enumerate(reader, start=1):
            if not row or all(not field.strip() for field in row):
                continue
            if len(row) < 2:
                raise ValueError(
                    f"{pool_tsv}:{line_no}: expected 2 tab-separated columns "
                    f"(sentence, unfairness_vector), got {len(row)}"
                )
            text = row[0]
            vec = row[1].strip()
            if not text.strip():
                continue
            rows.append({"pool_id": line_no, "text": text, "vec": vec, "label": parse_vector(vec)})

    if not rows:
        raise ValueError(f"No valid pool rows (sentence + unfairness_vector) found in {pool_tsv}")
    return rows


def load_pool_features(pool_features_jsonl: str) -> Dict[int, Dict[str, float]]:
    """{pool_id: {feature_id_str: magnitude}} from run_feature_coverage_filter.py output."""
    pool_features: Dict[int, Dict[str, float]] = {}
    with open(pool_features_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            pool_id = rec.get("pool_id")
            if pool_id is None:
                continue
            pool_features[pool_id] = rec.get("features", {})
    if not pool_features:
        raise ValueError(f"No pool feature records found in {pool_features_jsonl}")
    return pool_features


def active_features_by_prompt(
    pool_features: Dict[int, Dict[str, float]],
    threshold: float,
) -> Dict[int, Set[int]]:
    """{pool_id: {feature_id, ...}} restricted to magnitude > threshold."""
    return {
        pool_id: {int(fid) for fid, mag in features.items() if mag > threshold}
        for pool_id, features in pool_features.items()
    }


# ---------------------------------------------------------------------------
# Quota-constrained greedy selection
# ---------------------------------------------------------------------------

def min_required_positive(n: int, fraction: float) -> int:
    """ceil(n * fraction), computed with a small epsilon to avoid float overshoot."""
    return max(0, math.ceil(n * fraction - 1e-9))


def greedy_select(
    pool_rows: List[Dict[str, Any]],
    active_features: Dict[int, Set[int]],
    n: int,
    mode: str,
    min_positive_fraction: float,
) -> Tuple[List[int], Set[int]]:
    """Greedily pick up to n pool_ids, maximising (mode="max_features") or
    minimising (mode="min_features") distinct active-feature coverage,
    subject to at least ceil(n * min_positive_fraction) of the picks having
    >=1 positive label.

    Returns (selected_pool_ids_in_pick_order, covered_features).
    """
    assert mode in ("max_features", "min_features")
    sign = -1 if mode == "max_features" else 1

    label_by_pid = {row["pool_id"]: row["label"] for row in pool_rows}
    positive_pids = {pid for pid, lbl in label_by_pid.items() if has_positive(lbl)}
    remaining: Set[int] = set(label_by_pid.keys())
    covered: Set[int] = set()
    selected: List[int] = []

    n_steps = min(n, len(remaining))
    k_min = min_required_positive(n_steps, min_positive_fraction)
    if k_min > len(positive_pids):
        print(
            f"WARNING: pool only has {len(positive_pids)} positive-label examples, "
            f"fewer than the {k_min} required for an {min_positive_fraction:.0%} quota "
            f"over {n_steps} picks. Using all available positive examples."
        )
        k_min = len(positive_pids)

    def pick_best(candidates: Set[int]) -> int:
        best_pid = None
        best_key = None
        for pid in sorted(candidates):
            feats = active_features.get(pid, set())
            gain = len(feats - covered)
            key = (sign * gain, pid)
            if best_key is None or key < best_key:
                best_key = key
                best_pid = pid
        return best_pid

    # Phase 1: fill the positive-label quota.
    for _ in range(k_min):
        candidates = remaining & positive_pids
        if not candidates:
            break
        pid = pick_best(candidates)
        selected.append(pid)
        remaining.remove(pid)
        covered |= active_features.get(pid, set())

    # Phase 2: fill the rest of the selection unconstrained.
    for _ in range(n_steps - len(selected)):
        pid = pick_best(remaining)
        selected.append(pid)
        remaining.remove(pid)
        covered |= active_features.get(pid, set())

    return selected, covered


def random_select(
    pool_rows: List[Dict[str, Any]],
    active_features: Dict[int, Set[int]],
    n: int,
    seed: int = None,
) -> Tuple[List[int], Set[int]]:
    """Uniformly sample up to n pool_ids at random (--seed for reproducibility).

    covered_features is computed post hoc for reporting only (it plays no
    role in the selection itself).
    """
    pool_ids = [row["pool_id"] for row in pool_rows]
    n_pick = min(n, len(pool_ids))
    selected = random.Random(seed).sample(pool_ids, n_pick)

    covered: Set[int] = set()
    for pid in selected:
        covered |= active_features.get(pid, set())

    return selected, covered


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_selected_tsv(pool_rows: List[Dict[str, Any]], selected_pool_ids: Set[int], output_tsv: str) -> None:
    out_dir = os.path.dirname(os.path.abspath(output_tsv))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_tsv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for row in pool_rows:
            if row["pool_id"] in selected_pool_ids:
                writer.writerow([row["text"], row["vec"]])


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select n examples from a CLAUDETTE-TOS pool TSV, either greedily "
            "maximising (--mode max_features) or minimising (--mode "
            "min_features) distinct SAE feature coverage (features counted "
            "only above --threshold), subject to a minimum fraction of "
            "positive-label picks (--min-positive-fraction), or by uniform "
            "random sampling (--mode random)."
        )
    )
    parser.add_argument("--pool-tsv", type=str, default=_DEFAULT_POOL_TSV)
    parser.add_argument("--pool-features-jsonl", type=str, default=_DEFAULT_POOL_FEATURES_JSONL)

    parser.add_argument(
        "--mode",
        type=str,
        default="max_features",
        choices=["max_features", "min_features", "random"],
        help=(
            "Selection strategy: max_features (greedily maximise distinct "
            "active-feature coverage), min_features (greedily minimise it), "
            "or random (uniform random sample, --seed for reproducibility)."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.5,
        help="Minimum (exclusive) magnitude for a feature to count as active.",
    )
    parser.add_argument("--n", type=int, required=True, help="Number of examples to select.")
    parser.add_argument(
        "--min-positive-fraction",
        type=float,
        default=0.11,
        help=(
            "Only applies to --mode max_features/min_features. At least this "
            "fraction of the n selected examples must have >=1 positive "
            "('Y') label in their unfairness vector."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed, only used by --mode random.",
    )

    parser.add_argument(
        "--output-tsv",
        type=str,
        default="",
        help="Destination TSV. Defaults to "
        "feature_coverage_filter/claudette_tos/output/<pool-tsv-stem>_fc_select_<mode>_n<n>_t<threshold>.tsv.",
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        default="",
        help="Destination summary JSON. Defaults to "
        "feature_coverage_filter/claudette_tos/output/<pool-tsv-stem>_fc_select_<mode>_n<n>_t<threshold>_summary.json.",
    )
    args = parser.parse_args()

    if args.mode == "random" and args.min_positive_fraction != 0.11:
        parser.error("--min-positive-fraction is not compatible with --mode random.")

    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.pool_tsv):
        raise FileNotFoundError(f"Pool TSV not found: {args.pool_tsv}")
    if not os.path.isfile(args.pool_features_jsonl):
        raise FileNotFoundError(f"Pool-features JSONL not found: {args.pool_features_jsonl}")

    pool_rows = load_pool_rows(args.pool_tsv)
    pool_features = load_pool_features(args.pool_features_jsonl)
    print(
        f"Loaded {len(pool_rows)} pool rows from {args.pool_tsv} and "
        f"{len(pool_features)} pool feature records from {args.pool_features_jsonl}"
    )

    active_features = active_features_by_prompt(pool_features, args.threshold)
    pool_marginals = marginals([row["label"] for row in pool_rows])

    stem = os.path.splitext(os.path.basename(args.pool_tsv))[0]
    threshold_str = str(args.threshold).replace(".", "p")
    suffix = f"_{args.mode}_n{args.n}_t{threshold_str}"

    output_tsv = args.output_tsv or os.path.join(_SCRIPT_DIR, "output", f"{stem}_fc_select{suffix}.tsv")
    summary_json = args.summary_json or os.path.join(
        _SCRIPT_DIR, "output", f"{stem}_fc_select{suffix}_summary.json"
    )

    if args.mode == "random":
        selected_pool_ids, covered = random_select(pool_rows, active_features, args.n, seed=args.seed)
    else:
        selected_pool_ids, covered = greedy_select(
            pool_rows, active_features, args.n, args.mode, args.min_positive_fraction
        )
    if len(selected_pool_ids) < args.n:
        print(
            f"WARNING: pool exhausted, selected only {len(selected_pool_ids)} / "
            f"{args.n} requested examples."
        )

    label_by_pid = {row["pool_id"]: row["label"] for row in pool_rows}
    selected_labels = [label_by_pid[pid] for pid in selected_pool_ids]
    n_selected = len(selected_pool_ids)
    n_positive = sum(1 for lbl in selected_labels if has_positive(lbl))
    selected_marginals = marginals(selected_labels)
    n_unique_features = len(covered)

    write_selected_tsv(pool_rows, set(selected_pool_ids), output_tsv)

    summary = {
        "pool_tsv": args.pool_tsv,
        "pool_features_jsonl": args.pool_features_jsonl,
        "mode": args.mode,
        "threshold": args.threshold,
        "n_requested": args.n,
        "n_selected": n_selected,
        "min_positive_fraction": args.min_positive_fraction,
        "n_selected_with_positive_label": n_positive,
        "positive_fraction_selected": n_positive / n_selected if n_selected else 0.0,
        "seed": args.seed,
        "selected_pool_ids": sorted(selected_pool_ids),
        "n_unique_features_covered": n_unique_features,
        "signature_distribution_selected": signature_distribution(selected_labels),
        "signature_distribution_pool": signature_distribution([row["label"] for row in pool_rows]),
        "marginals_selected": selected_marginals,
        "marginals_pool": pool_marginals,
    }
    out_dir = os.path.dirname(os.path.abspath(summary_json))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(
        f"Selected {n_selected} / {len(pool_rows)} pool examples "
        f"(mode={args.mode}, threshold={args.threshold})."
    )
    print(f"Aktivierte (unique) features mit magnitude > {args.threshold}: {n_unique_features}")
    print(
        f"Beispiele mit positivem Label: {n_positive}/{n_selected} "
        f"({n_positive / n_selected:.1%}, gefordert: {args.min_positive_fraction:.0%})"
    )
    print(f"Wrote {n_selected} rows to {output_tsv}")
    print(f"Wrote summary to {summary_json}")


if __name__ == "__main__":
    main()
