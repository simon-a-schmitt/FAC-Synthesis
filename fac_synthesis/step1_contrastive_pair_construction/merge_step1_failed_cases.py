import re
import json
import pandas as pd

FINAL_DECISION_FILE = "xxx.tsv"
TRIPLETS_FILE = "xxx.jsonl"
OUTPUT_FILE = "xxx.jsonl"

def split_summary_and_spans(text):
    text = text.replace("\\n", "\n")
    parts = text.split("\n\n", 1)
    summary_part = parts[0].strip()
    span_part = parts[1] if len(parts) > 1 else ""
    spans = re.findall(r"Span\s*\d+:\s*(.+)", span_part)
    top_spans = spans[:1] if spans else []
    span_text = "; ".join(s.strip() for s in top_spans)
    return summary_part, span_text

def is_activated_sample(s):
    try:
        span_ok = bool(str(s.get("span", "")).strip())
        sc = float(s.get("score", 0) or 0)
        return span_ok and sc > 0
    except:
        return False

def merge_jsonl():
    summary_map = {}
    span_map = {}
    df = pd.read_csv(FINAL_DECISION_FILE, sep="\t", header=0, dtype=str, keep_default_na=False)
    for _, row in df.iterrows():
        fid = int(row.iloc[0])
        summary_text = "\t".join(row.astype(str).tolist())
        summary, spans = split_summary_and_spans(summary_text)
        summary_clean = re.sub(rf"^{fid}\s+", "", summary)
        summary_map[fid] = summary_clean
        span_map[fid] = spans

    triplets = {}
    gen_map = {}
    with open(TRIPLETS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            fid = data["feature_id"]
            samples = data.get("samples", []) or []
            gen_map[fid] = samples
            activated = [s for s in samples if is_activated_sample(s)]
            if len(activated) >= 2:
                srt = sorted(activated, key=lambda x: float(x.get("score", 0) or 0), reverse=True)
                good, bad = srt[0], srt[1]
                triplets[fid] = {"good": good, "bad": bad}

    merged = []
    all_feature_ids = list(summary_map.keys())

    for fid in all_feature_ids:
        if fid in triplets:
            good = triplets[fid]["good"]
            bad = triplets[fid]["bad"]
            rec = {
                "feature_id": fid,
                "Feature Summary": summary_map.get(fid, ""),
                "Good example": good.get("text", ""),
                "Good Span Activated": good.get("span", ""),
                "Good Activation score": good.get("score", ""),
                "Bad example": bad.get("text", ""),
                "Bad Span Activated": bad.get("span", ""),
                "Bad Activation score": bad.get("score", "")
            }
        else:
            span_text = span_map.get(fid, "")
            bad_text = ""
            bad_span = ""
            bad_score = ""
            samples = gen_map.get(fid, [])
            if samples:
                srt = sorted(samples, key=lambda x: float(x.get("score", 0) or 0))
                b = srt[0]
                bad_text = b.get("text", "")
                bad_span = b.get("span", "")
                bad_score = b.get("score", "")
            rec = {
                "feature_id": fid,
                "Feature Summary": summary_map.get(fid, ""),
                "Good example": span_text,
                "Good Span Activated": span_text,
                "Good Activation score": 5.0,
                "Bad example": bad_text,
                "Bad Span Activated": bad_span,
                "Bad Activation score": bad_score
            }
        merged.append(rec)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        for item in merged:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    merge_jsonl()

