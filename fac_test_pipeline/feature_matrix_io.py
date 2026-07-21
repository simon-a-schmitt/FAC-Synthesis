"""
Reader/writer helpers for the dense prompt x feature statistics matrix produced
by run_fac_test_pipeline_feature_matrix.py.

On-disk layout (one directory, three stats each stored as their own .npy so
individual rows/columns can be memory-mapped without touching the others):

    <output-dir>/
        activation_density.npy   float32 [n_prompts, n_features], NaN = not active
        peak_magnitude.npy       float32 [n_prompts, n_features], NaN = not active
        log_odds.npy             float32 [n_prompts, n_features], NaN = not active
        prompt_ids.npy           int64   [n_prompts]  (index into the source prompts JSON)
        feature_ids.npy          int64   [n_features] (SAE feature id, sorted ascending)
        meta.json                small JSON with shape/threshold/source info

Rows are prompt-major and features are the fast (contiguous) axis, so
row(prompt_id) reads one contiguous disk block. n_prompts is typically small
relative to n_features, so column reads stay cheap even though they are
strided.
"""

import argparse
import json
import os
from typing import Dict, Optional

import numpy as np

STATS = ("activation_density", "peak_magnitude", "log_odds")


class FeatureMatrix:
    """Memory-mapped accessor for the prompt x feature statistics matrix."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        with open(os.path.join(output_dir, "meta.json"), "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        self.prompt_ids = np.load(os.path.join(output_dir, "prompt_ids.npy"))
        self.feature_ids = np.load(os.path.join(output_dir, "feature_ids.npy"))
        self._prompt_index: Dict[int, int] = {int(pid): i for i, pid in enumerate(self.prompt_ids)}
        self._feature_index: Dict[int, int] = {int(fid): j for j, fid in enumerate(self.feature_ids)}

        self._arrays = {
            stat: np.load(os.path.join(output_dir, f"{stat}.npy"), mmap_mode="r") for stat in STATS
        }

    def get(self, prompt_id: int, feature_id: int) -> Optional[Dict[str, float]]:
        """All three stats for one (prompt, feature) cell, or None if never active there."""
        i = self._prompt_index[prompt_id]
        j = self._feature_index[feature_id]
        vals = {stat: float(arr[i, j]) for stat, arr in self._arrays.items()}
        if np.isnan(vals["activation_density"]):
            return None
        return vals

    def row(self, prompt_id: int, stat: str = "activation_density") -> np.ndarray:
        """All values for one prompt across every feature (contiguous read)."""
        i = self._prompt_index[prompt_id]
        return np.array(self._arrays[stat][i, :])

    def col(self, feature_id: int, stat: str = "activation_density") -> np.ndarray:
        """All values for one feature across every prompt (strided read)."""
        j = self._feature_index[feature_id]
        return np.array(self._arrays[stat][:, j])

    def row_max(self, prompt_id: int, stat: str = "activation_density") -> Optional[float]:
        vals = self.row(prompt_id, stat)
        return float(np.nanmax(vals)) if np.any(~np.isnan(vals)) else None

    def row_min(self, prompt_id: int, stat: str = "activation_density") -> Optional[float]:
        vals = self.row(prompt_id, stat)
        return float(np.nanmin(vals)) if np.any(~np.isnan(vals)) else None

    def col_max(self, feature_id: int, stat: str = "activation_density") -> Optional[float]:
        vals = self.col(feature_id, stat)
        return float(np.nanmax(vals)) if np.any(~np.isnan(vals)) else None

    def col_min(self, feature_id: int, stat: str = "activation_density") -> Optional[float]:
        vals = self.col(feature_id, stat)
        return float(np.nanmin(vals)) if np.any(~np.isnan(vals)) else None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ad-hoc lookup CLI for a feature-matrix directory written by "
        "run_fac_test_pipeline_feature_matrix.py."
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--prompt-id", type=int, default=None)
    parser.add_argument("--feature-id", type=int, default=None)
    parser.add_argument(
        "--stat", type=str, default="activation_density", choices=list(STATS)
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    fm = FeatureMatrix(args.output_dir)
    print(f"Matrix: {fm.meta['n_prompts']} prompts x {fm.meta['n_features']} features")

    if args.prompt_id is not None and args.feature_id is not None:
        print(fm.get(args.prompt_id, args.feature_id))
    elif args.prompt_id is not None:
        print(f"row max ({args.stat}): {fm.row_max(args.prompt_id, args.stat)}")
        print(f"row min ({args.stat}): {fm.row_min(args.prompt_id, args.stat)}")
    elif args.feature_id is not None:
        print(f"col max ({args.stat}): {fm.col_max(args.feature_id, args.stat)}")
        print(f"col min ({args.stat}): {fm.col_min(args.feature_id, args.stat)}")
    else:
        print("Provide --prompt-id and/or --feature-id.")


if __name__ == "__main__":
    main()
