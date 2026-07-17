import csv
import json
import random
import sys

csv.field_size_limit(sys.maxsize)

INPUT_TSV = "casehold.tsv"
SAMPLE_TSV = "casehold_50.tsv"
OUTPUT_JSON = "casehold_train.json"
SAMPLE_SIZE = 50

# (1) Zufällig 50 Einträge aus casehold.tsv auswählen (Header-Zeile wird uebersprungen)
with open(INPUT_TSV, newline="", encoding="utf-8") as f:
    reader = csv.reader(f, delimiter="\t")
    header = next(reader)
    rows = list(reader)

sample = random.sample(rows, SAMPLE_SIZE)

with open(SAMPLE_TSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerows(sample)

print(f"{len(sample)} Zeilen zufaellig ausgewaehlt und nach {SAMPLE_TSV} geschrieben.")

# (2) Sample-Einträge in das Zielformat transformieren und als JSON schreiben
records = []
with open(SAMPLE_TSV, newline="", encoding="utf-8") as f:
    reader = csv.reader(f, delimiter="\t")
    for row in reader:
        prompt, label = row[0], row[1]
        records.append({
            "instruction": prompt.strip(),
            "input": "",
            "output": label.strip()
        })

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print(f"{len(records)} Beispiele konvertiert.")
