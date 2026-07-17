import csv
import random
import shutil
import sys

csv.field_size_limit(sys.maxsize)

INPUT_TSV = "casehold.tsv"
EVAL_TSV = "casehold_eval.tsv"
BACKUP_TSV = "casehold.tsv.bak"
HOLDOUT_SIZE = 1000

# Original vor dem Ueberschreiben sichern
shutil.copyfile(INPUT_TSV, BACKUP_TSV)

with open(INPUT_TSV, newline="", encoding="utf-8") as f:
    reader = csv.reader(f, delimiter="\t")
    header = next(reader)
    rows = list(reader)

eval_indices = set(random.sample(range(len(rows)), HOLDOUT_SIZE))
eval_rows = [row for i, row in enumerate(rows) if i in eval_indices]
remaining_rows = [row for i, row in enumerate(rows) if i not in eval_indices]

with open(EVAL_TSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(header)
    writer.writerows(eval_rows)

with open(INPUT_TSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(header)
    writer.writerows(remaining_rows)

print(f"{len(eval_rows)} Zeilen als Holdout nach {EVAL_TSV} verschoben.")
print(f"{len(remaining_rows)} Zeilen verbleiben in {INPUT_TSV}.")
print(f"Backup der Originaldatei liegt unter {BACKUP_TSV}.")
