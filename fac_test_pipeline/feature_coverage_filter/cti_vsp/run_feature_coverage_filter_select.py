"""
Select --n examples from a CTI-VSP pool TSV under one of three --mode
strategies, and (optionally, for the two greedy modes) keep the marginal
distributions of the 8 CVSS metrics of the selection close to the full
pool's marginals.

Input
-----
- --pool-tsv: headerless TSV, column 0 = CVE description, column 1 = CVSS
  vector (feature_coverage_filter/input), e.g. cti_vsp_bb_deepseek_06_pool.tsv.
  Each row's 1-based line number is its pool_id, matching the convention used
  by run_feature_coverage_filter.py.
- --pool-features-jsonl: output of run_feature_coverage_filter.py, one JSON
  object per line: {"pool_id": ..., "n_content_tokens": ..., "features":
  {feature_id: peak_magnitude, ...}}.

Processing
----------
A feature counts as "active" on an example if its peak_magnitude >=
--threshold. --mode selects the strategy:

  - max_features (default): greedy maximum-coverage selection. Repeatedly
    pick the not-yet-selected example that adds the most not-yet-covered
    active features to the selection.
  - min_features: the mirror image. Repeatedly pick the not-yet-selected
    example that adds the *fewest* not-yet-covered active features (0-gain
    examples first), directly minimising the number of distinct active
    features touched by the final selection.
  - random: --n examples are drawn uniformly at random from the pool
    (--seed for reproducibility). Feature coverage and distribution
    matching play no role in the selection itself.

For max_features/min_features, ties in coverage gain (including the common
case where every remaining example has the same gain, e.g. 0 once coverage
has saturated) are broken:
  - without --keep-distribution: by the smaller pool_id.
  - with --keep-distribution: by whichever example would keep the
    selection's running mean total-variation distance (over the 8 CVSS
    metric marginals, against the full pool's marginals) lowest, and only
    then by the smaller pool_id.
--keep-distribution is not compatible with --mode random (there is no
greedy tie-break to steer there).

Output
------
- A TSV of the n selected examples (original 2-column pool format, original
  pool order) written to feature_coverage_filter/output.
- A summary JSON with the number of unique active features covered and the
  mean total-variation distance between the selection's and the full pool's
  CVSS metric marginals (reported in every case, for all three modes).
Both figures are also printed to the console.
"""

import argparse
import collections
import csv
import json
import os
import random
from typing import Any, Dict, List, Set, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_POOL_TSV = os.path.join(_SCRIPT_DIR, "input", "cti_vsp_bb_deepseek_06_pool.tsv")
_DEFAULT_POOL_FEATURES_JSONL = os.path.join(
    _SCRIPT_DIR, "output", "cti_vsp_bb_deepseek_06_pool_fc_filter.jsonl"
)

METRICS = ["AV", "AC", "PR", "UI", "S", "C", "I", "A"]
CLASSES = {
    "AV": ["N", "A", "L", "P"],
    "AC": ["L", "H"],
    "PR": ["N", "L", "H"],
    "UI": ["N", "R"],
    "S":  ["U", "C"],
    "C":  ["H", "L", "N"],
    "I":  ["H", "L", "N"],
    "A":  ["H", "L", "N"],
}


# ---------------------------------------------------------------------------
# CVSS-vector parsing (matches run_label_distribution_filter_select.py)
# ---------------------------------------------------------------------------

def parse_vector(vec: str) -> Dict[str, str]:
    """'CVSS:3.1/AV:N/AC:L/...' -> {'AV': 'N', 'AC': 'L', ...}"""
    out: Dict[str, str] = {}
    for part in vec.strip().split("/"):
        if ":" not in part:
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


def marginals(labels: List[Dict[str, str]]) -> Dict[str, Dict[str, float]]:
    n = len(labels)
    out: Dict[str, Dict[str, float]] = {}
    for m in METRICS:
        cnt = collections.Counter(lab[m] for lab in labels)
        out[m] = {c: cnt.get(c, 0) / n for c in CLASSES[m]}
    return out


def tv_per_metric(p: Dict[str, Dict[str, float]], q: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    return {
        m: 0.5 * sum(abs(p[m].get(c, 0.0) - q[m].get(c, 0.0)) for c in CLASSES[m])
        for m in METRICS
    }


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_pool_rows(pool_tsv: str) -> List[Dict[str, Any]]:
    """Read {pool_id, text, vec, label} from a headerless <description>\\t<cvss> TSV.

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
                    f"(description, cvss_vector), got {len(row)}"
                )
            text = row[0]
            vec = row[1].strip()
            if not text.strip():
                continue
            rows.append({"pool_id": line_no, "text": text, "vec": vec, "label": parse_vector(vec)})

    if not rows:
        raise ValueError(f"No valid pool rows (description + cvss_vector) found in {pool_tsv}")
    return rows


def load_pool_features(pool_features_jsonl: str) -> Dict[int, Dict[str, float]]:
    """{pool_id: {feature_id_str: peak_magnitude}} from run_feature_coverage_filter.py output."""
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
    """{pool_id: {feature_id, ...}} restricted to peak_magnitude >= threshold."""
    return {
        pool_id: {int(fid) for fid, mag in features.items() if mag >= threshold}
        for pool_id, features in pool_features.items()
    }


# ---------------------------------------------------------------------------
# Greedy max-coverage selection, optionally tie-broken by distribution match
# ---------------------------------------------------------------------------

def project_mean_tv(
    counts: Dict[str, "collections.Counter"],
    k: int,
    candidate_label: Dict[str, str],
    pool_marginals: Dict[str, Dict[str, float]],
) -> float:
    """Mean TVD (over the 8 metrics) if candidate_label were added to a
    selection of size k whose per-metric class counts are `counts`."""
    n = k + 1
    total = 0.0
    for m in METRICS:
        cls = candidate_label[m]
        cnt_m = counts[m]
        s = 0.0
        for c in CLASSES[m]:
            new_count = cnt_m.get(c, 0) + (1 if c == cls else 0)
            s += abs(new_count / n - pool_marginals[m].get(c, 0.0))
        total += 0.5 * s
    return total / len(METRICS)


def greedy_select(
    pool_rows: List[Dict[str, Any]],
    active_features: Dict[int, Set[int]],
    n: int,
    pool_marginals: Dict[str, Dict[str, float]],
    keep_distribution: bool,
    mode: str,
) -> Tuple[List[int], Set[int], Dict[str, "collections.Counter"]]:
    """Greedily pick up to n pool_ids, maximising (mode="max_features") or
    minimising (mode="min_features") distinct active-feature coverage.

    Returns (selected_pool_ids_in_pick_order, covered_features, final_counts).
    """
    assert mode in ("max_features", "min_features")
    sign = -1 if mode == "max_features" else 1

    label_by_pid = {row["pool_id"]: row["label"] for row in pool_rows}
    remaining: Set[int] = set(label_by_pid.keys())

    covered: Set[int] = set()
    counts: Dict[str, "collections.Counter"] = {m: collections.Counter() for m in METRICS}
    selected: List[int] = []

    n_steps = min(n, len(remaining))
    for _ in range(n_steps):
        k = len(selected)
        best_pid = None
        best_key = None
        for pid in sorted(remaining):
            feats = active_features.get(pid, set())
            gain = len(feats - covered)
            if keep_distribution:
                proj_tv = project_mean_tv(counts, k, label_by_pid[pid], pool_marginals)
                key = (sign * gain, proj_tv, pid)
            else:
                key = (sign * gain, pid)
            if best_key is None or key < best_key:
                best_key = key
                best_pid = pid

        selected.append(best_pid)
        remaining.remove(best_pid)
        covered |= active_features.get(best_pid, set())
        lbl = label_by_pid[best_pid]
        for m in METRICS:
            counts[m][lbl[m]] += 1

    return selected, covered, counts


def random_select(
    pool_rows: List[Dict[str, Any]],
    active_features: Dict[int, Set[int]],
    n: int,
    seed: int = None,
) -> Tuple[List[int], Set[int], Dict[str, "collections.Counter"]]:
    """Uniformly sample up to n pool_ids at random (--seed for reproducibility).

    Returns (selected_pool_ids, covered_features, final_counts), where
    covered_features/final_counts are computed post hoc for reporting only
    (they play no role in the selection itself).
    """
    label_by_pid = {row["pool_id"]: row["label"] for row in pool_rows}
    pool_ids = list(label_by_pid.keys())
    n_pick = min(n, len(pool_ids))
    selected = random.Random(seed).sample(pool_ids, n_pick)

    covered: Set[int] = set()
    counts: Dict[str, "collections.Counter"] = {m: collections.Counter() for m in METRICS}
    for pid in selected:
        covered |= active_features.get(pid, set())
        lbl = label_by_pid[pid]
        for m in METRICS:
            counts[m][lbl[m]] += 1

    return selected, covered, counts


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
            "Select n examples from a CTI-VSP pool TSV, either greedily "
            "maximising (--mode max_features) or minimising (--mode "
            "min_features) distinct SAE feature coverage (features counted "
            "only above --threshold), or by uniform random sampling (--mode "
            "random). --keep-distribution additionally tie-breaks the two "
            "greedy modes towards keeping the selection's CVSS metric "
            "marginals close to the full pool's."
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
        help="Minimum peak_magnitude for a feature to count as active.",
    )
    parser.add_argument("--n", type=int, required=True, help="Number of examples to select.")
    parser.add_argument(
        "--keep-distribution",
        action="store_true",
        help=(
            "Only compatible with --mode max_features/min_features. Secondary "
            "selection criterion: among examples tied on feature-coverage "
            "gain, prefer the one that keeps the selection's CVSS metric "
            "marginals closest to the full pool's marginals."
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
        "feature_coverage_filter/output/<pool-tsv-stem>_fc_select_<mode>_n<n>_t<threshold>[_keepdist].tsv.",
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        default="",
        help="Destination summary JSON. Defaults to "
        "feature_coverage_filter/output/<pool-tsv-stem>_fc_select_<mode>_n<n>_t<threshold>[_keepdist]_summary.json.",
    )
    args = parser.parse_args()

    if args.keep_distribution and args.mode == "random":
        parser.error("--keep-distribution is not compatible with --mode random.")

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
    suffix = f"_{args.mode}_n{args.n}_t{threshold_str}" + ("_keepdist" if args.keep_distribution else "")

    output_tsv = args.output_tsv or os.path.join(_SCRIPT_DIR, "output", f"{stem}_fc_select{suffix}.tsv")
    summary_json = args.summary_json or os.path.join(
        _SCRIPT_DIR, "output", f"{stem}_fc_select{suffix}_summary.json"
    )

    if args.mode == "random":
        selected_pool_ids, covered, counts = random_select(
            pool_rows, active_features, args.n, seed=args.seed
        )
    else:
        selected_pool_ids, covered, counts = greedy_select(
            pool_rows, active_features, args.n, pool_marginals, args.keep_distribution, args.mode
        )
    if len(selected_pool_ids) < args.n:
        print(
            f"WARNING: pool exhausted, selected only {len(selected_pool_ids)} / "
            f"{args.n} requested examples."
        )

    n_selected = len(selected_pool_ids)
    selected_marginals = {
        m: {c: counts[m].get(c, 0) / n_selected for c in CLASSES[m]} for m in METRICS
    }
    tv_sel = tv_per_metric(selected_marginals, pool_marginals)
    tv_mean = sum(tv_sel.values()) / len(METRICS)
    n_unique_features = len(covered)

    write_selected_tsv(pool_rows, set(selected_pool_ids), output_tsv)

    summary = {
        "pool_tsv": args.pool_tsv,
        "pool_features_jsonl": args.pool_features_jsonl,
        "mode": args.mode,
        "threshold": args.threshold,
        "n_requested": args.n,
        "n_selected": n_selected,
        "keep_distribution": args.keep_distribution,
        "seed": args.seed,
        "selected_pool_ids": sorted(selected_pool_ids),
        "n_unique_features_covered": n_unique_features,
        "tv_per_metric": tv_sel,
        "tv_mean": tv_mean,
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
    print(f"Aktivierte (unique) features mit Aktivierung >= {args.threshold}: {n_unique_features}")
    print(f"TV je Metrik (Teilset vs. Gesamt-Pool): {({k: round(v, 4) for k, v in tv_sel.items()})}")
    print(f"TV_mean (Teilset vs. Gesamt-Pool): {tv_mean:.4f}")
    print(f"Wrote {n_selected} rows to {output_tsv}")
    print(f"Wrote summary to {summary_json}")


if __name__ == "__main__":
    main()
