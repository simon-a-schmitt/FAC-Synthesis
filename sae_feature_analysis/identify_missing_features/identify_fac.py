import pandas as pd
import sys
import re

def extract_final_decision(task_str):
    if not isinstance(task_str, str):
        return "-"
    return task_str.split("|||")[0].strip().lower()

def extract_analysis(task_text):
    if not isinstance(task_text, str):
        return ""
    m = re.search(r"Analysis\s*:\s*(.*?)\s*Final Decision", task_text, flags=re.S | re.I)
    if m:
        return m.group(1).strip()
    m = re.search(r"\|\|\|\s*(.*?)\s*Final Decision", task_text, flags=re.S | re.I)
    if m:
        return m.group(1).strip()
    m = re.search(r"\|\|\|\s*(.*)", task_text, flags=re.S | re.I)
    if m:
        return m.group(1).strip()
    return ""

def main():
    if len(sys.argv) < 3:
        print("Usage: python identify_fac.py <file1.tsv> <file2.tsv>")
        sys.exit(1)

    file1, file2 = sys.argv[1], sys.argv[2]

    print(f"[INFO] Loading: {file1}")
    df1 = pd.read_csv(file1, sep="\t", engine="python", on_bad_lines="skip")
    print(f"[INFO] Loading: {file2}")
    df2 = pd.read_csv(file2, sep="\t", engine="python", on_bad_lines="skip")

    df1["FinalDecision"] = df1["Task"].apply(extract_final_decision)
    df2["FinalDecision"] = df2["Task"].apply(extract_final_decision)

    df1["Analysis"] = df1["Task"].apply(extract_analysis)
    df2["Analysis"] = df2["Task"].apply(extract_analysis)
    extra_col1 = df1.columns[-5]
    mask1 = (
        (df1["FinalDecision"].str.lower().isin({"yes", "probably", "maybe"})) &
        (df1[extra_col1].astype(str).str.lower().isin({"yes", "probably", "maybe"}))
    )
    df1 = df1[mask1].copy()

    merged = pd.merge(df1, df2, on="FeatureID", suffixes=("_file1", "_file2"))
    print(merged["FinalDecision_file1"].value_counts())
    print(merged["FinalDecision_file2"].value_counts())    
    
    sel = ((merged["FinalDecision_file1"].str.lower().isin(["yes", "probably", "maybe"])) &
            (~merged["FinalDecision_file2"].str.lower().isin(["yes", "probably", "maybe"])))
    
    rows = []
    for _, row in merged[sel].iterrows():
        summary = (str(row.get("Analysis_file1", ""))).strip()
        words = (str(row.get("Words_file1", ""))).strip()
        if "cannot tell" in summary.lower():
            continue
        rows.append({
            "FeatureID": row["FeatureID"],
            "Summary": summary,
            "Words": words
        })
    
    selected = pd.DataFrame(rows)

    print(f"[RESULT] Total merged & selected samples: {len(selected)}")
    print(selected.head())

    output_file = "xxx.tsv"
    selected.to_csv(output_file, sep="\t", index=False)
    print(f"[INFO] Saved merged results to {output_file}")

if __name__ == "__main__":
    main()

