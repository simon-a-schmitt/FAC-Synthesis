import os
import re
import json
import time
import argparse
import random
from typing import Dict, Any, List, Tuple
from llama_wrapper_qwen import llama3_generate
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
    parser.add_argument("--max_retry_per_question", type=int, default=2)
    args = parser.parse_args()

    features = []
    num_per_feature = 1
    with open(args.features, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            features.append(data)
    if not features:
        raise ValueError("No valid features found.")

    print(f"Loaded {len(features)} features.")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    all_tasks, all_q_records = [], []
    error_fids = []
    for feat in features:
        for key in [
            "Feature Summary", "Good example", "Bad example",
            "Good Span Activated", "Bad Span Activated",
            "Good Activation score", "Bad Activation score"
        ]:
            if key in feat:
                val = feat[key]
                if not isinstance(val, str):
                    feat[key] = "" if val is None or (isinstance(val, float) and np.isnan(val)) else str(val)

    for feat in tqdm(features, desc="Generating queries"):
        summary = feat.get("Feature Summary", "").strip()
        fid = feat.get("feature_id")
        good_ex = feat.get("Good example", "").strip()
        bad_ex = feat.get("Bad example", "").strip()
        good_span = feat.get("Good Span Activated", "").strip()
        good_score = feat.get("Good Activation score", "")
        bad_span = feat.get("Bad Span Activated", "").strip()
        bad_score = feat.get("Bad Activation score", "")
        context_text = (
            f"Feature Summary: {summary}\n\n"
            f"Good Example:\n{good_ex}\n[Good Span Activated]: {good_span} (Good Score: {good_score})\n\n"
            f"Bad Example:\n{bad_ex}\n[Bad Span Activated]: {bad_span} (Bad Score: {bad_score})"
        )
        for i in range(num_per_feature):
            user_msg = f"{context_text.strip()}"
            response = llama3_generate(user_msg, temperature=args.temperature, num_return_sequences=2)

            if not response:
                print(f"[WARN] Empty response for feature {fid}")
                continue
            for text in response:
                if not text.strip():
                    continue
                text = text.strip()

                if not re.search(r'\t[01]\s*$', text):
                    text = text.rstrip("\n ") + "\t0"

                try:
                    label = 1

                    segs = re.findall(
                        r'(?:^|\n)\s*(Query-\d+\s*:\s*.*?)(?=(?:\n\s*Query-\d+\s*:)|(?:\t\s*[01]\s*$)|$)',
                        text,
                        flags=re.S
                    )

                    if not segs:
                        print(text)
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

