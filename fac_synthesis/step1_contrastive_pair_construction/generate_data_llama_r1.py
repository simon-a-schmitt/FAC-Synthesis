import os
import re
import json
import time
import argparse
import random
from typing import Dict, Any, List, Tuple
from llama_wrapper import llama3_generate
import numpy as np
from tqdm import tqdm

from gen_utils import (
    _extract_json_block,
    _parse_transcript,
    _check_alternating_roles,
    _validate_multi_turn_pair,
    _validate_single_turn_pair,
    parse_instruction_input_pairs,
    _clean_ins_inp,
    _prepend_input_to_human,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--min_exchanges", type=int, default=1)
    parser.add_argument("--max_exchanges", type=int, default=3)
    parser.add_argument("--max_retry_per_question", type=int, default=10)
    parser.add_argument("--num_synthetic_samples", type=int, default=1)
    parser.add_argument("--ratio", type=float, default=0.2, help="Sampling ratio (e.g., 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    random.seed(args.seed)

    features = []
    num_per_feature = 1
    with open(args.features, "r", encoding="utf-8", errors="ignore") as f:
        header_skipped = False
        for line in f:
            line = line.strip()
            if not line:
                continue
            if not header_skipped:
                if "FeatureID" in line and "Summary" in line:
                    header_skipped = True
                    continue
            parts = line.split(None, 1)
            if len(parts) < 2:
                continue
            fid, summary = parts[0].strip(), parts[1].strip().strip('"').strip("'").replace('""', '"')
            if fid.isdigit():
                features.append({"FeatureID": fid, "Summary": summary})
    if not features:
        raise ValueError("No valid features found.")

    print(f"Loaded {len(features)} features.")

    if args.ratio < 1.0:
        sample_size = int(len(features) * args.ratio)
        features = random.sample(features, sample_size)
        print(f"[INFO] Subsampled to {len(features)} features (ratio={args.ratio}, seed={args.seed})")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    all_tasks, all_q_records = [], []
    error_fids = []
    for feat in tqdm(features, desc="Generating queries"):
        summary = feat.get("Summary", "").strip()
        fid = feat["FeatureID"]
        spans = feat.get("Words", "").strip()
        spans_clean = re.sub(r"\s+", " ", spans)
        if spans_clean:
            context_text = f"Feature Summary: {summary}\nExample Spans:\n{spans_clean}"
        else:
            context_text = f"Feature Summary: {summary}"
        for i in range(num_per_feature):
            feature_content = context_text.strip()
            user_msg = "Generate ONE explicitly toxic user query in the required format."
            response = llama3_generate(
            user_msg,
            temperature=args.temperature,
            num_return_sequences=args.num_synthetic_samples,
            feature_content=feature_content
            )
            if not response:
                print(f"[WARN] Empty response for feature {fid}")
                continue
            for text in response:
                if not text.strip():
                    continue
                text = text.strip()

                try:
                    label = 1

                    segs = re.findall(
                            r'(?:^|\n)\s*(Query-\d+\s*:\s*.*?)(?=(?:\n\s*Query-\d+\s*:)|(?:\t\s*[01]\s*$)|$)',
                            text,
                            flags=re.S
                    )

                    if not segs:
                        print(f"[WARN] No valid Query segments for feature {fid}")
                        error_fids.append(int(fid))
                        continue

                    tmp = []
                    for s in segs:
                        m_idx = re.match(r'\s*Query-(\d+)\s*:', s)
                        idx = int(m_idx.group(1)) if m_idx else 999999
                        tmp.append((idx, s.strip()))
                    tmp.sort(key=lambda x: x[0])

                    qtext = "\n".join(s for _, s in tmp)

                    all_tasks.append((qtext, str(label)))
                    all_q_records.append({
                        "FeatureID": fid,
                        "query": qtext,
                        "label": label,
                        "context_used": context_text[:4000]
                     })

                except Exception as e:
                    print(f"[skip {fid}] {e}")

            time.sleep(args.sleep)
    print(f"Features with no valid segments: {error_fids}")

    tsv_out = args.out + ".queries.tsv"
    with open(tsv_out, "w", encoding="utf-8") as tf:
        for rec in all_q_records:
            qline = (
                rec["query"]
                .replace("\r", "")
                .replace("\t", " ")
                .replace("\n", "\\n")
            )
            tf.write(f"{qline}\t{rec['label']}\n")

    print(f"[INFO] Saved {len(all_q_records)} queries to {tsv_out}")

    if not all_tasks:
        raise RuntimeError("No queries generated.")


if __name__ == "__main__":
    main()

