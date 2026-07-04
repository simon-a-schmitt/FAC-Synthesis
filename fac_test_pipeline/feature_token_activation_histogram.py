"""
Compute a histogram over how many tokens each feature is active on,
using the first entry of a token-matrix JSONL file.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_JSONL = (
    Path(__file__).parent
    / "output"
    / "results_prompts_seed_casehold_token_shard1.jsonl"
)


def main(jsonl_path: Path) -> None:
    with open(jsonl_path) as f:
        entry = json.loads(f.readline())

    rel_features = entry["rel_features"]          # list[int], length = num_features
    feature_matrix = entry["feature_matrix"]      # list[num_tokens] of list[num_features]
    num_tokens = entry["num_tokens"]
    num_features = entry["num_features"]

    print(f"Prompt index : {entry['index']}")
    print(f"Num tokens   : {num_tokens}")
    print(f"Num features : {num_features}")
    print()

    # For each feature column, count how many tokens have non-zero activation
    token_counts = []
    for feat_idx in range(num_features):
        count = sum(1 for tok in range(num_tokens) if feature_matrix[tok][feat_idx] != 0.0)
        token_counts.append(count)

    # Group features by their token-count
    bins: dict[int, list[int]] = defaultdict(list)
    for feat_idx, count in enumerate(token_counts):
        bins[count].append(rel_features[feat_idx])

    max_count = max(bins.keys())
    bar_max_width = 40

    print(f"{'Tokens':>7}  {'#Features':>9}  {'Bar':<{bar_max_width}}  Features")
    print("-" * 120)

    for n_tok in sorted(bins.keys()):
        feats = bins[n_tok]
        n_feats = len(feats)
        bar_len = max(1, round(n_feats / len(max(bins.values(), key=len)) * bar_max_width))
        bar = "#" * bar_len
        feat_str = ", ".join(str(f) for f in sorted(feats))
        print(f"{n_tok:>7}  {n_feats:>9}  {bar:<{bar_max_width}}  [{feat_str}]")

    print()
    print(f"Features active on 0 tokens : {len(bins.get(0, []))}")
    print(f"Features active on ≥1 token : {sum(len(v) for k, v in bins.items() if k >= 1)}")


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_JSONL
    main(path)
