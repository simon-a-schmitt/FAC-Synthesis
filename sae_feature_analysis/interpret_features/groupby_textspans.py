import time
import re
import os
import string
import json
import concurrent
import multiprocessing
import pandas as pd
import transformers as trf
import tqdm


CACHE_DIR = "../../"
TOTAL_FEATURES = 2 ** 16


class Reader:
    
    def __init__(self, fpath):
        self.df = pd.read_csv(fpath, engine="python", 
    on_bad_lines="skip", encoding="utf8", sep="\t")
        print("Loading success!")
        self.df.sort_values("NeuronID", inplace=True)
        self.tokenizer = trf.AutoTokenizer.from_pretrained(
                           "mistralai/Mistral-7B-Instruct-v0.2", # optional
                           use_fast=False, padding_side="right", 
                           cache_dir=CACHE_DIR)
    
    def select(self, idx, topK=5, key="Span"):
        i = self.df.NeuronID.searchsorted(idx, side="left")
        j = self.df.NeuronID.searchsorted(idx, side="right")
        if not i <= j - 1:
            return []
        df = self.df.iloc[i:j]
        df = df.sort_values(by="Score", ascending=False)
        return df[key].tolist()[:topK]

    def truncate(self, span, topN=10):
        if not isinstance(span, str):
            span = ''
        ids = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(span))[-topN:]
        return self.tokenizer.batch_decode([ids])[0]

    def get_neuron_spans(self, idx, topK, topN=10):
        spans = [self.truncate(_, topN) for _ in self.select(idx, topK)]
        return "\n".join("Span %d: %s" % pair
                         for pair in enumerate(spans, 1))


def build_deduplicated_file(input_path, dedup_path):
    if os.path.exists(dedup_path) and os.path.getmtime(dedup_path) >= os.path.getmtime(input_path):
        return dedup_path

    duplicated = set()
    with open(dedup_path, "w", encoding="utf8") as f:
        with open(input_path, encoding="utf8") as g:
            f.write(g.readline())
            for row in g:
                temp = row.split("\t")
                temp = (temp[0], temp[-1])
                if temp in duplicated:
                    continue
                f.write(row)
                duplicated.add(temp)
    return dedup_path


def load_checkpoint(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        return {"next_idx": 0, "activated": 0}
    with open(checkpoint_path, encoding="utf8") as f:
        state = json.load(f)
    return {
        "next_idx": int(state.get("next_idx", 0)),
        "activated": int(state.get("activated", 0)),
    }


def save_checkpoint(checkpoint_path, next_idx, activated, output_path):
    state = {
        "next_idx": int(next_idx),
        "activated": int(activated),
        "output_path": output_path,
    }
    with open(checkpoint_path, "w", encoding="utf8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
        

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Group and post-process activation text spans.")
    parser.add_argument("folder", help="Folder that contains full.tsv")
    parser.add_argument("--checkpoint-every", type=int, default=1000,
                        help="Write a checkpoint every N features.")
    parser.add_argument("--restart", action="store_true",
                        help="Ignore any existing checkpoint and start from scratch.")
    args = parser.parse_args()

    folder = args.folder
    print("Grouping By Files from: %s" % folder)

    input_path = os.path.join(folder, "full.tsv")
    dedup_path = os.path.join(folder, "full_deduplicated.tsv")
    source_path = build_deduplicated_file(input_path, dedup_path)

    reader = Reader(source_path)
    print("Loading %d deduplicated records." % len(reader.df))
    file = os.path.split(folder)[-1]

    output_path = os.path.join(".", "%s.tsv" % file.replace("textspans", "TopAct"))
    checkpoint_path = output_path + ".checkpoint.json"

    start_state = {"next_idx": 0, "activated": 0}
    if not args.restart:
        start_state = load_checkpoint(checkpoint_path)

    start_idx = max(0, min(TOTAL_FEATURES, start_state["next_idx"]))
    activated = start_state["activated"]

    if args.restart:
        if os.path.exists(output_path):
            os.remove(output_path)
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

    if start_idx > 0 and not os.path.exists(output_path):
        start_idx = 0
        activated = 0

    mode = "a" if start_idx > 0 and os.path.exists(output_path) and not args.restart else "w"

    print("Writing to %s" % output_path)
    bar = tqdm.tqdm(total=TOTAL_FEATURES, initial=start_idx)
    with open(output_path, mode, encoding="utf8") as f:
        if mode == "w":
            f.write("FeatureID\tWords\n")
        for idx in range(start_idx, TOTAL_FEATURES):
            span = reader.get_neuron_spans(idx, topK=10)
            if "Span" in span:
                activated += 1
            f.write("%d\t%s\n" % (idx, span.replace("\t", "\\t").replace("\n", "\\n").replace("\r", "")))
            bar.update(1)

            if (idx + 1) % args.checkpoint_every == 0:
                f.flush()
                save_checkpoint(checkpoint_path, idx + 1, activated, output_path)

        f.flush()
        save_checkpoint(checkpoint_path, TOTAL_FEATURES, activated, output_path)
    print("Totally %d neurons are activated." % activated)
        
        
        

