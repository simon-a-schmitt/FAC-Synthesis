import re
import json
import pandas as pd
from collections import defaultdict

FINAL_DECISION_FILE = 'xxx.tsv'
EXTSPANS_FILE = 'xxx.tsv'
YNTHETIC_QUERIES_FILE = 'xxx.tsv'
OUTPUT_JSONL = 'xxx.jsonl'

TARGET_SAMPLE_COUNT = 2

def analyze_data():
    feature_df = pd.read_csv(FINAL_DECISION_FILE, sep='\t', usecols=[0], header=0)
    feature_df.iloc[:, 0] = pd.to_numeric(feature_df.iloc[:, 0], errors='coerce').astype('Int64')
    feature_df.dropna(subset=[feature_df.columns[0]], inplace=True)
    feature_ids = set(feature_df.iloc[:, 0].tolist())
    ordered_feature_ids = feature_df.iloc[:, 0].tolist()
    print(f"Success: Extracted {len(feature_ids)} unique FeatureIDs from {FINAL_DECISION_FILE}.")

    spans_df = pd.read_csv(
        TEXTSPANS_FILE,
        sep='\t',
        header=0,
        usecols=[0, 1, 2, 3],
        skipinitialspace=True,
        engine="python",
        quoting=3,
        on_bad_lines="skip"
    )
    spans_df.columns = ['NeuronID', 'TextID', 'Score', 'Span']
    spans_df['NeuronID'] = pd.to_numeric(spans_df['NeuronID'], errors='coerce')
    spans_df['TextID'] = pd.to_numeric(spans_df['TextID'], errors='coerce')
    spans_df.dropna(subset=['NeuronID', 'TextID'], inplace=True)
    spans_df['NeuronID'] = spans_df['NeuronID'].astype(int)
    spans_df['TextID'] = spans_df['TextID'].astype(int)
    neuron_ids = set(spans_df['NeuronID'].tolist())
    matching_ids_set = feature_ids.intersection(neuron_ids)
    matching_ids = sorted(list(matching_ids_set))
    unmatched_ids = feature_ids - neuron_ids
    print(f"Total unmatched FeatureIDs: {len(unmatched_ids)}")
    print(sorted(list(unmatched_ids)))
    print(f"Success: Extracted {len(neuron_ids)} unique NeuronIDs from {TEXTSPANS_FILE}.")
    print(f"Result: Number of overlapping NeuronID/FeatureID is {len(matching_ids)}.")
    
    neuron_to_textid = defaultdict(set)
    for _, row in spans_df.iterrows():
        neuron_id = row['NeuronID']
        if neuron_id in matching_ids_set:
            neuron_to_textid[neuron_id].add(row['TextID'])

    warn_ids = ["xxx", "xxx", "xxx"]
    warn_counts = defaultdict(int)
    for fid in warn_ids:
        warn_counts[fid] += 1

    feature_to_text_ids = defaultdict(list)
    current_text_id_index = 0

    for fid in ordered_feature_ids:
        count = warn_counts[fid]
        
        num_successful = max(0, TARGET_SAMPLE_COUNT - count)
        
        for _ in range(num_successful):
            feature_to_text_ids[fid].append(current_text_id_index)
            current_text_id_index += 1

    expected_total_successful_samples = current_text_id_index
    print(f"Info: Total expected successful samples based on WARN list: {expected_total_successful_samples}")

    all_texts = {}
    with open(SYNTHETIC_QUERIES_FILE, 'r', encoding='utf-8', errors='ignore') as f:
        idx = 1
        for line in f:
            if re.match(r'^Query-\d+:', line):
                t = line.rstrip('\n')
                t = re.sub(r'\t[01]\s*$', '', t)
                all_texts[idx-1] = t
                idx += 1
        actual_query_count = idx - 1
    print(f"Success: Actual generated Query-1 samples count: {actual_query_count}")
    
    total_matched_text_ids = 0
    match_statistics = {fid: {'expected_ids': [], 'matched_ids': []} for fid in ordered_feature_ids}
    
    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as outj:
        for fid in ordered_feature_ids:
            expected_ids = feature_to_text_ids.get(fid, [])
            actual_ids_in_spans = neuron_to_textid.get(fid, set())
            
            chosen_ids = []
            for tid in expected_ids:
                if tid in all_texts:
                    chosen_ids.append(tid)
                if len(chosen_ids) == TARGET_SAMPLE_COUNT:
                    break
            
            samples = []
            for tid in chosen_ids:
                rows = spans_df[(spans_df['NeuronID'] == fid) & (spans_df['TextID'] == tid)]
                span_text = rows['Span'].iloc[0] if not rows.empty else ""
                score_val = float(rows['Score'].iloc[0]) if not rows.empty else 0.0
                samples.append({'text_id': tid, 'text': all_texts.get(tid, ''), 'span': span_text, 'score': score_val})
            
            if not samples and expected_ids:
                for tid in expected_ids[:TARGET_SAMPLE_COUNT]:
                    samples.append({'text_id': tid, 'text': all_texts.get(tid, ''), 'span': '', 'score': 0.0})
            
            if samples:
                matched_ids = [tid for tid in expected_ids if tid in actual_ids_in_spans]
                match_statistics[fid]['expected_ids'] = list(expected_ids)
                match_statistics[fid]['matched_ids'] = list(matched_ids)
                total_matched_text_ids += len(matched_ids)
                
                samples = sorted(samples, key=lambda x: x['score'], reverse=True)
                outj.write(json.dumps({'feature_id': fid, 'samples': samples}, ensure_ascii=False) + '\n')

    print("\n" + "--- Final Matching Statistics ---")
    print(f"Result: Among the {len(matching_ids)} overlapping NeuronIDs (FeatureIDs),")
    print(f"the total count of matched TextIDs (i.e., synthetic sample indices) is {total_matched_text_ids}.")
    print("\nDetailed Statistics (Matching NeuronID/FeatureID):")
    print(len(matching_ids))
    matched_features_count = 0
    for fid in ordered_feature_ids:
        stats = match_statistics[fid]
        matched_count = len(stats['matched_ids'])
        expected = stats['expected_ids']
        expected_str = f"{expected}" if expected else "[]"
        if matched_count > 0:
            print(f"FeatureID {fid}: Expected successful TextIDs {expected_str}, Actual Matched TextIDs count: {matched_count}.")
            matched_features_count += 1
        else:
            print(f"FeatureID {fid}: Expected successful TextIDs {expected_str}, Actual Matched TextIDs count: 0 (No overlap).")
    print(matched_features_count)

if __name__ == '__main__':
    analyze_data()
