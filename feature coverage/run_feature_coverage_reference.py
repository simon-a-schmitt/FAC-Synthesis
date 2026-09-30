"""Reference feature set of a (test) dataset: for every threshold t in --thresholds, the ids of all
SAE features that are active on at least one content token of at least one sample of --input-tsv
(raw activation > t; see coverage_common.py for the exact measurement).

Output: feature coverage/<path>/<input-tsv-stem>_reference_active_features.json - consumed by
run_feature_coverage.py as the reference for feature_coverage_test_at_t<t>.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import coverage_common as cc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Determine, per threshold, which SAE features are active on a reference (test) TSV."
    )
    cc.add_common_args(parser)
    args = parser.parse_args()

    if not args.input_tsv.is_file():
        raise SystemExit(f"Input TSV not found: {args.input_tsv}")
    _, active, metadata = cc.run_sae_on_tsv(args, args.input_tsv, cc.load_runner(args))

    out_path = cc.domain_output_dir(args.path) / f"{args.input_tsv.stem}_reference_active_features.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        **metadata,
        "thresholds": {key: {"n_active": len(ids), "feature_ids": ids} for key, ids in active.items()},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    for key, ids in active.items():
        print(f"  t={key:<8} {len(ids):>6} active feature(s)")
    print(f"Wrote reference active features to {out_path}")


if __name__ == "__main__":
    main()
