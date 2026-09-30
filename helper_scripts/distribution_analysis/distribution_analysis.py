"""Berechnet Label-Verteilungen für TSV-Dateien (zwei Spalten pro Zeile: Text, Label)
und die Total-Variation-Distanz (TV) jeder Datei zu einer Referenz-TSV (i.d.R. dem
Testset der Domain).

Die Träger (Wertemengen) der Verteilungen stammen direkt aus den Label-Definitionen in
    data_synthesis/labeling/<domain>/prompt/prompt_step_2.py
damit Analyse, Labeling-Prompt und Parser nicht auseinanderlaufen.

Pro Domain (--path):

- toxicity (Label "Answer: safe" / "Answer: toxic"):
    TV = |p_toxic - q_toxic|, also die Differenz der Positivraten (bei einer binären
    Verteilung ist 0.5 * sum |p - q| genau das).

- claudette_tos (Label "LTD: N|TER: N|CH: N|CR: N|USE: N|LAW: N|J: N|ARB: N"):
    * marginal: je Slot TV = |p_Y - q_Y| (Differenz der Positivraten), danach
      Mittelwert über die 8 Slots.
    * joint: TV über den ganzen Label-Vektor; Träger = alle 2^8 = 256 Y/N-Vektoren.

- cti_vsp (Label "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"):
    * marginal: je Metrik TV = 0.5 * sum_v |p(v) - q(v)| über die volle
      CVSS-v3.1-Wertemenge der Metrik, danach Mittelwert über die 8 Metriken.
      Der konstante "CVSS:3.1"-Versions-Präfix ist kein Slot.
    * joint: TV über den ganzen Vektor; Träger = kartesisches Produkt aller
      Metrik-Wertemengen (4*2*3*2*2*3*3*3 = 2592 Vektoren).

Labels werden kanonisiert ("AV: N" == "AV:N", Groß-/Kleinschreibung egal). Labels, die
nicht im Träger liegen (unparsbar, fehlender/zusätzlicher Slot, ungültiger Wert), werden
aus der Verteilung ausgeschlossen und in der JSON unter "invalid_labels" gezählt.

Nutzung:
    python distribution_analysis.py --path cti_vsp --tsv a.tsv b.tsv [...]
        [--reference ref.tsv] [--output-dir DIR]

Ohne --reference wird das Testset unter benchmarks/<domain>/tsv_files/ verwendet.
Für jede Eingabe-TSV wird
    <output-dir>/<input-dateiname-ohne-endung>_label_distribution.json
geschrieben (Default output-dir: output/<domain-ordner>/json_files neben diesem Skript).
"""

import argparse
import csv
import importlib.util
import json
import sys
from collections import Counter, OrderedDict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LABELING_ROOT = REPO_ROOT / "data_synthesis" / "labeling"
BENCHMARK_ROOT = REPO_ROOT / "benchmarks"
OUTPUT_DIR = Path(__file__).parent / "output"


def load_prompt_module(domain_dir: str):
    module_path = LABELING_ROOT / domain_dir / "prompt" / "prompt_step_2.py"
    spec = importlib.util.spec_from_file_location(f"{domain_dir}_prompt_step_2", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_domain_specs() -> dict:
    toxicity = load_prompt_module("toxicity_detection").TOXICITY_FRAGMENTS
    claudette = load_prompt_module("claudette_tos").CLAUDETTE_FRAGMENTS
    cvss = load_prompt_module("cti_vsp").CVSS_FRAGMENTS

    return {
        "toxicity": {
            "domain_dir": "toxicity_detection",
            "reference": BENCHMARK_ROOT / "toxicity_detection" / "tsv_files" / "toxicity_test_set.tsv",
            "kind": "binary",
            "prefix": toxicity["prefix"],
            "labels": tuple(toxicity["labels"]),
            "positive": "toxic",
        },
        "claudette_tos": {
            "domain_dir": "claudette_tos",
            "reference": BENCHMARK_ROOT / "claudette_tos" / "tsv_files" / "claudette_tos_test.tsv",
            "kind": "vector",
            "separator": "|",
            "strip_prefix": "",
            "fields": claudette["fields"],
            "field_order": tuple(claudette["field_order"]),
            "positive": "Y",
        },
        "cti_vsp": {
            "domain_dir": "cti_vsp",
            "reference": BENCHMARK_ROOT / "cti_vsp" / "tsv_files" / "cti_vsp_test.tsv",
            "kind": "vector",
            "separator": "/",
            "strip_prefix": "CVSS:3.1/",
            "fields": cvss["fields"],
            "field_order": tuple(cvss["field_order"]),
            "positive": None,
        },
    }


def canonical_value(value: str, allowed) -> "str | None":
    lookup = {option.lower(): option for option in allowed}
    return lookup.get(value.strip().lower())


def parse_binary_label(label: str, spec: dict) -> "str | None":
    key, sep, value = label.partition(":")
    if not sep or key.strip() != spec["prefix"]:
        return None
    return canonical_value(value, spec["labels"])


def parse_vector_label(label: str, spec: dict) -> "tuple | None":
    """Liefert die Slot-Werte in field_order, oder None, falls das Label nicht im Träger liegt."""
    label = label.replace(" ", "")
    prefix = spec["strip_prefix"].replace(" ", "")
    if prefix:
        if not label.startswith(prefix):
            return None
        label = label[len(prefix):]

    values = {}
    for part in label.split(spec["separator"]):
        key, sep, value = part.partition(":")
        if not sep or key not in spec["fields"] or key in values:
            return None
        canon = canonical_value(value, spec["fields"][key])
        if canon is None:
            return None
        values[key] = canon

    if set(values) != set(spec["field_order"]):
        return None
    return tuple(values[field] for field in spec["field_order"])


def read_labels(tsv_path: Path) -> list:
    labels = []
    with tsv_path.open(encoding="utf-8", newline="") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) < 2 or not row[1].strip():
                continue
            labels.append(row[1].strip())
    if not labels:
        raise ValueError(f"Keine Labels in {tsv_path} gefunden.")
    return labels


def proportions(counter: Counter, support, total: int) -> "OrderedDict[str, float]":
    return OrderedDict((value, counter.get(value, 0) / total) for value in support)


def tv_distance(p: dict, q: dict) -> float:
    """0.5 * sum |p - q| über die Vereinigung der Schlüssel (fehlende Werte = 0)."""
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in set(p) | set(q))


def analyze(tsv_path: Path, spec: dict) -> dict:
    labels = read_labels(tsv_path)
    invalid: Counter = Counter()

    if spec["kind"] == "binary":
        counter: Counter = Counter()
        for label in labels:
            parsed = parse_binary_label(label, spec)
            if parsed is None:
                invalid[label] += 1
            else:
                counter[parsed] += 1
        n_valid = sum(counter.values())
        if n_valid == 0:
            raise ValueError(f"Keine gültigen Labels in {tsv_path}.")
        distribution = {
            "support": list(spec["labels"]),
            "counts": {value: counter.get(value, 0) for value in spec["labels"]},
            "proportions": proportions(counter, spec["labels"], n_valid),
            "positive_label": spec["positive"],
            "positive_rate": counter.get(spec["positive"], 0) / n_valid,
        }
    else:
        vectors: Counter = Counter()
        for label in labels:
            parsed = parse_vector_label(label, spec)
            if parsed is None:
                invalid[label] += 1
            else:
                vectors[parsed] += 1
        n_valid = sum(vectors.values())
        if n_valid == 0:
            raise ValueError(f"Keine gültigen Labels in {tsv_path}.")

        marginals = OrderedDict()
        for i, field in enumerate(spec["field_order"]):
            slot_counter: Counter = Counter()
            for vector, count in vectors.items():
                slot_counter[vector[i]] += count
            support = spec["fields"][field]
            marginals[field] = {
                "counts": {value: slot_counter.get(value, 0) for value in support},
                "proportions": proportions(slot_counter, support, n_valid),
            }
            if spec["positive"] is not None:
                marginals[field]["positive_rate"] = slot_counter.get(spec["positive"], 0) / n_valid

        joint_support_size = 1
        for field in spec["field_order"]:
            joint_support_size *= len(spec["fields"][field])

        # Nur Vektoren mit Masse > 0 werden gelistet; alle übrigen Träger-Elemente haben Anteil 0.
        joint = OrderedDict(
            (render_vector(vector, spec), {"count": count, "proportion": count / n_valid})
            for vector, count in sorted(vectors.items(), key=lambda kv: (-kv[1], kv[0]))
        )
        distribution = {
            "marginal": marginals,
            "joint": {
                "support_size": joint_support_size,
                "n_observed_vectors": len(vectors),
                "vectors": joint,
            },
        }

    return {
        "file": str(tsv_path),
        "n_total": len(labels),
        "n_valid": n_valid,
        "n_invalid": sum(invalid.values()),
        "invalid_labels": dict(invalid.most_common()),
        "distribution": distribution,
    }


def render_vector(vector: tuple, spec: dict) -> str:
    sep = spec["separator"]
    body = sep.join(f"{field}:{value}" for field, value in zip(spec["field_order"], vector))
    return spec["strip_prefix"] + body


def compute_tv(result: dict, reference: dict, spec: dict) -> dict:
    dist, ref = result["distribution"], reference["distribution"]

    if spec["kind"] == "binary":
        return {
            "tv": abs(dist["positive_rate"] - ref["positive_rate"]),
            "positive_rate": dist["positive_rate"],
            "reference_positive_rate": ref["positive_rate"],
        }

    per_slot = OrderedDict()
    for field in spec["field_order"]:
        per_slot[field] = tv_distance(dist["marginal"][field]["proportions"], ref["marginal"][field]["proportions"])

    joint_p = {k: v["proportion"] for k, v in dist["joint"]["vectors"].items()}
    joint_q = {k: v["proportion"] for k, v in ref["joint"]["vectors"].items()}

    return {
        "marginal": {
            "mean": sum(per_slot.values()) / len(per_slot),
            "per_slot": per_slot,
        },
        "joint": tv_distance(joint_p, joint_q),
    }


def print_report(name: str, result: dict, spec: dict) -> None:
    print(f"\n=== {name} ===")
    print(f"n_total = {result['n_total']}, n_valid = {result['n_valid']}, n_invalid = {result['n_invalid']}")
    tv = result["tv_distance_to_reference"]
    if spec["kind"] == "binary":
        print(
            f"  positive_rate = {tv['positive_rate']:.4f} "
            f"(Referenz {tv['reference_positive_rate']:.4f}) -> TV = {tv['tv']:.4f}"
        )
        return
    for field, value in tv["marginal"]["per_slot"].items():
        print(f"  [{field}] TV = {value:.4f}")
    print(f"  marginal TV (Mittel) = {tv['marginal']['mean']:.4f}")
    print(f"  joint TV             = {tv['joint']:.4f}")


def main() -> None:
    specs = build_domain_specs()

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--path", required=True, choices=sorted(specs), help="Domain der Labels.")
    parser.add_argument("--tsv", nargs="+", required=True, type=Path, help="Zu analysierende TSV-Datei(en).")
    parser.add_argument(
        "--reference", type=Path, default=None, help="Referenz-TSV (Default: Testset der Domain unter benchmarks/)."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="Zielordner für die JSONs (Default: output/<domain>/json_files)."
    )
    args = parser.parse_args()

    spec = specs[args.path]
    reference_path = args.reference or spec["reference"]
    output_dir = args.output_dir or OUTPUT_DIR / spec["domain_dir"] / "json_files"
    output_dir.mkdir(parents=True, exist_ok=True)

    reference = analyze(reference_path, spec)
    if reference["n_invalid"]:
        print(f"Warnung: {reference['n_invalid']} ungültige Labels in der Referenz {reference_path}.", file=sys.stderr)

    for tsv_path in args.tsv:
        try:
            result = analyze(tsv_path, spec)
        except (OSError, ValueError) as exc:
            print(exc, file=sys.stderr)
            continue

        result["domain"] = args.path
        result["reference"] = {"file": str(reference_path), "n_valid": reference["n_valid"]}
        result["tv_distance_to_reference"] = compute_tv(result, reference, spec)
        print_report(str(tsv_path), result, spec)

        output_path = output_dir / f"{tsv_path.stem}_label_distribution.json"
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"-> geschrieben nach {output_path}")


if __name__ == "__main__":
    main()
