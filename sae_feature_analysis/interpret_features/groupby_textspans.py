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
EXPECTED_COLUMNS = {"NeuronID", "TextID", "Score", "Span"}


def find_and_repair_header(fpath):
    """Findet und repariert beschädigte Header (ANSI-Escape-Sequenzen gemischt mit Header)."""
    import re
    
    try:
        with open(fpath, encoding="utf8") as f:
            first_line = f.readline()
        
        # Überprüfe, ob die Zeile ANSI-Escape-Sequenzen enthält
        if "\x1b" in first_line or "\033" in first_line or "\x07" in first_line:
            # Versuche, den echten Header zu finden und zu extrahieren
            # Suche nach dem Muster "NeuronID\tTextID\tScore\tSpan"
            match = re.search(r"NeuronID[\t\s]+TextID[\t\s]+Score[\t\s]+Span", first_line)
            if match:
                header_start = match.start()
                # Finde das Ende der Header-Zeile (bis zum nächsten Tab und dann zur nächsten logischen Zeile)
                header_end = first_line.find("\n", match.start())
                if header_end == -1:
                    # Wenn kein \n, finde den nächsten echten Tabulator nach "Span"
                    span_end = match.end()
                    # Suche nach dem nächsten Non-Header-Zeichen (z.B. "0\t" für erste Datenzeile)
                    next_tab = first_line.find("\t", span_end)
                    if next_tab != -1:
                        # Der Header endet hier, Daten beginnen
                        header_end = next_tab
                    else:
                        header_end = span_end
                
                # Extrahiere den echten Header
                clean_header = first_line[header_start:header_end].strip()
                
                # Lese restliche Zeilen und schreibe Datei neu
                with open(fpath, encoding="utf8") as f:
                    f.readline()  # Überspringe die kaputte erste Zeile
                    remaining_lines = f.readlines()
                
                # Schreibe neu: Header + Rest
                with open(fpath, "w", encoding="utf8") as f:
                    f.write(clean_header + "\n")
                    f.writelines(remaining_lines)
                
                print(f"Repariert {fpath}: Extrahiert Header aus beschädigter Zeile")
                print(f"  Clean header: {clean_header[:80]}...")
                return True, "OK"
        
        # Wenn keine Escape-Sequenzen, suche nach gültigem Header an anderen Positionen
        with open(fpath, encoding="utf8") as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines[:100]):
            line_stripped = line.strip()
            if not line_stripped:
                continue
            columns = set(line_stripped.split("\t"))
            if EXPECTED_COLUMNS.issubset(columns):
                if i > 0:
                    print(f"Repariert {fpath}: Header gefunden an Zeile {i+1}, entferne {i} Zeilen")
                    with open(fpath, "w", encoding="utf8") as f:
                        f.writelines(lines[i:])
                return True, "OK"
        
        return False, "Kein gültiger Header gefunden"
        
    except Exception as e:
        return False, f"Fehler beim Reparieren: {e}"


def validate_header(fpath):
    """Überprüft, ob die erste Zeile ein gültiger Header ist."""
    try:
        with open(fpath, encoding="utf8") as f:
            header_line = f.readline().strip()
            # Überprüfe, ob die Zeile nur Kontrollzeichen oder ANSI-Escape-Sequenzen enthält
            if not header_line or "\x1b" in header_line or "\033" in header_line:
                return False, "Header enthält ANSI-Escape-Sequenzen oder ist leer"
            # Überprüfe, ob die erwarteten Spalten vorhanden sind
            columns = set(header_line.split("\t"))
            if not EXPECTED_COLUMNS.issubset(columns):
                missing = EXPECTED_COLUMNS - columns
                return False, f"Fehlende Spalten: {missing}. Header: {header_line[:100]}"
            return True, "OK"
    except Exception as e:
        return False, f"Fehler beim Lesen des Headers: {e}"


class Reader:
    
    def __init__(self, fpath):
        # Header validieren
        is_valid, msg = validate_header(fpath)
        if not is_valid:
            raise ValueError(f"Ungültiger CSV-Header in {fpath}: {msg}")
        
        # CSV mit Fehlerbehandlung laden
        try:
            self.df = pd.read_csv(fpath, engine="python", 
                on_bad_lines="skip", encoding="utf8", sep="\t")
            if "NeuronID" not in self.df.columns:
                raise KeyError(f"Spalte 'NeuronID' nicht gefunden. Verfügbare Spalten: {self.df.columns.tolist()}")
            print(f"Loading success! {len(self.df)} rows geladen.")
            self.df.sort_values("NeuronID", inplace=True)
        except Exception as e:
            raise RuntimeError(f"Fehler beim Laden von {fpath}: {e}")
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
    """Erstellt deduplizierte TSV-Datei mit Header-Validierung und Reparatur."""
    # Repariere Input-Datei falls nötig
    is_repaired, msg = find_and_repair_header(input_path)
    if not is_repaired:
        raise ValueError(f"Konnte Input-Datei nicht reparieren {input_path}: {msg}")
    
    # Überprüfe, ob deduplizierte Datei existiert und aktuell ist
    if os.path.exists(dedup_path) and os.path.getmtime(dedup_path) >= os.path.getmtime(input_path):
        is_valid, msg = validate_header(dedup_path)
        if is_valid:
            print(f"Verwende vorhandene deduplizierte Datei: {dedup_path}")
            return dedup_path
        else:
            print(f"Deduplizierte Datei ist beschädigt ({msg}). Erstelle neu...")
            try:
                os.remove(dedup_path)
            except Exception as e:
                print(f"Warnung: Konnte nicht löschen {dedup_path}: {e}")

    print(f"Erstelle deduplizierte Datei: {dedup_path}")
    duplicated = set()
    row_count = 0
    with open(dedup_path, "w", encoding="utf8") as f:
        with open(input_path, encoding="utf8") as g:
            header = g.readline()
            # Validiere Input-Header
            is_valid, msg = validate_header(input_path)
            if not is_valid:
                raise ValueError(f"Input-Datei {input_path} hat ungültigen Header: {msg}")
            f.write(header)
            
            for row in g:
                if not row.strip():
                    continue
                temp = row.split("\t")
                if len(temp) < 2:
                    continue
                key = (temp[0], temp[-1])
                if key in duplicated:
                    continue
                f.write(row)
                duplicated.add(key)
                row_count += 1
    
    print(f"Deduplizierung abgeschlossen: {row_count} eindeutige Zeilen geschrieben.")
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
        
        
        

