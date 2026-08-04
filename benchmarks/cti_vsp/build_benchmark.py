import csv

INPUT_PATH = "cti-vsp.tsv"
OUTPUT_PATH = "cti_vsp_benchmark.tsv"
COLUMNS = ["Prompt", "GT"]

OLD_PROMPT_PREFIX = (
    "Analyze the following CVE description and calculate the CVSS v3.1 Base Score. "
    "Determine the values for each base metric: AV, AC, PR, UI, S, C, I, and A. "
    "Summarize each metric's value and provide the final CVSS v3.1 vector string.   "
    "Valid options for each metric are as follows: "
    "- **Attack Vector (AV)**: Network (N), Adjacent (A), Local (L), Physical (P) "
    "- **Attack Complexity (AC)**: Low (L), High (H) "
    "- **Privileges Required (PR)**: None (N), Low (L), High (H) "
    "- **User Interaction (UI)**: None (N), Required (R) "
    "- **Scope (S)**: Unchanged (U), Changed (C) "
    "- **Confidentiality (C)**: None (N), Low (L), High (H) "
    "- **Integrity (I)**: None (N), Low (L), High (H) "
    "- **Availability (A)**: None (N), Low (L), High (H)  "
    "Summarize each metric's value and provide the final CVSS v3.1 vector string. "
    "Ensure the final line of your response contains only the CVSS v3 Vector String "
    "in the following format:  "
    "Example format: CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H  "
    "CVE Description:"
)

NEW_PROMPT_PREFIX = (
    "Analyze the following CVE description and output the CVSS v3.1 Base vector string. "
    "Do not explain your reasoning. Output only the vector string and nothing else.  "
    "Valid options for each metric: "
    "- Attack Vector (AV): N, A, L, P "
    "- Attack Complexity (AC): L, H "
    "- Privileges Required (PR): N, L, H "
    "- User Interaction (UI): N, R "
    "- Scope (S): U, C "
    "- Confidentiality (C): N, L, H "
    "- Integrity (I): N, L, H "
    "- Availability (A): N, L, H  "
    "Output format (exactly this, no other text): "
    "CVSS:3.1/AV:_/AC:_/PR:_/UI:_/S:_/C:_/I:_/A:_  "
    "CVE Description:"
)


def build_benchmark():
    with open(INPUT_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar=None)
        rows = list(reader)

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar=None)
        writer.writeheader()
        for row in rows:
            record = {col: row[col] for col in COLUMNS}
            record["Prompt"] = record["Prompt"].replace(OLD_PROMPT_PREFIX, NEW_PROMPT_PREFIX)
            writer.writerow(record)


def main():
    build_benchmark()


if __name__ == "__main__":
    main()
