import json
from pathlib import Path

IN_PATH  = Path("blackbox_accept.jsonl")     # <- anpassen
OUT_PATH = Path("blackbox_accept_filtered.tsv")

with IN_PATH.open("r", encoding="utf-8") as f_in, \
     OUT_PATH.open("w", encoding="utf-8", newline="\n") as f_out:

    f_out.write("description\ttarget_vector\n")

    n = 0
    for line in f_in:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if not obj.get("accept"):
            continue

        prompt = obj["description"]
        label  = obj["target_vector"]

        # Tabs entfernen, Zeilenumbrüche als \n escapen -> eine Zeile pro Sample
        prompt = prompt.replace("\t", " ").replace("\r\n", "\n").replace("\r", "\n")
        prompt = prompt.replace("\n", "\\n")

        # Führende/abschließende " entfernen
        prompt = prompt.strip('"')
        if isinstance(label, str):
            label = label.strip('"')

        f_out.write(f"{prompt}\t{label}\n")
        n += 1

print(f"{n} Zeilen nach {OUT_PATH} geschrieben.")