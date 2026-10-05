"""Feature coverage (FAC) of the training data of FT runs (gold / bb / fg / hybrid), via the existing
"feature coverage/run_feature_coverage.py" (Llama + SAE, loaded once for all runs of a call).

Input per run: its training TSV - GOLD_TSV (gold) or LABEL_TSV (bb/fg/hybrid), the source of the
LLaMA-Factory dataset - copied to experiments/runs/<run_id>/fac/<run_id>.tsv so the coverage output
is named after the run. Result per run: experiments/runs/<run_id>/fac.json = the coverage JSON plus
run_id, threshold and input_sha256 (sha256 of the training TSV), so experiments/collect_results.py
can tell whether the FAC belongs to the current data.

Threshold: --threshold (default 0, the activation threshold of the experiments); the domain's test
reference (feature coverage/<bench>/*_reference_active_features.json) must contain it.

Usage (GPU; normally via experiments/slurm/fac_job.sh):
    python experiments/run_fac.py <run_id> [<run_id> ...] [--threshold 0]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from runs import PROJECT_DIR, WS_PATH, file_sha256, load_config, parse_run_id, resolve

FAC_DIR = PROJECT_DIR / "feature coverage"


def training_tsv(r: dict) -> Path:
    if r["arm"] == "gold":
        return Path(r["gold_tsv"])
    if "label_tsv" in r:
        return Path(r["label_tsv"])
    raise SystemExit(f"error: {r['run_id']} has no training data (arm {r['arm']!r}).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_ids", nargs="+")
    parser.add_argument("--threshold", type=float, default=0.0)
    args = parser.parse_args()

    config = load_config()
    by_bench: dict[str, list[tuple[dict, Path]]] = {}
    for run_id in args.run_ids:
        r = resolve(parse_run_id(run_id), config)
        src = training_tsv(r)
        if not src.is_file():
            raise SystemExit(f"error: training TSV of {run_id} missing: {src}")
        dst = Path(r["run_dir"]) / "fac" / f"{run_id}.tsv"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        by_bench.setdefault(r["bench"], []).append((r, dst))

    key = f"{args.threshold:g}"
    for bench, items in by_bench.items():
        cmd = [sys.executable, str(FAC_DIR / "run_feature_coverage.py"), "--path", bench,
               "--input-tsv", *[str(dst) for _, dst in items], "--thresholds", str(args.threshold),
               "--model-name", str(WS_PATH / config["global"]["base_model"]),
               "--sae-ckpt-path", str(WS_PATH / config["global"]["sae_ckpt"])]
        print("[fac] " + " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=FAC_DIR, check=True)
        for r, dst in items:
            out = FAC_DIR / bench / "output" / f"{r['run_id']}_feature_coverage.json"
            result = json.loads(out.read_text(encoding="utf-8"))
            result.update(run_id=r["run_id"], threshold=key, input_sha256=file_sha256(dst))
            target = Path(r["run_dir"]) / "fac.json"
            target.write_text(json.dumps(result, indent=2), encoding="utf-8")
            cov = result["metrics"][f"feature_coverage_at_t{key}"]
            print(f"[fac] {r['run_id']}: relevant coverage {cov['n_covered']}/{cov['n_relevant']} "
                  f"({cov['fraction']:.3f}) -> {target}")


if __name__ == "__main__":
    main()
