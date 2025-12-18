import json
import re
from pathlib import Path
import pandas as pd
import os
import math

def create_accuracy_df(root_dir: str, output_path: str = "./figures_tables"):
    root_dir = Path(root_dir)
    for model_dir in root_dir.iterdir():
        model_name = model_dir.name
        files = list(model_dir.glob("*.jsonl"))
        df_list = []
        for file in files:
            df = pd.read_json(file, lines=True).drop(columns=["input_text"], errors="ignore")
            if "_A" in file.name:
                exclude = {"condition", "comparison", "target", "log_prob", "pearsonr", "pred"}
                meta_cols = [c for c in df.columns if c not in exclude and c != "input_text"]
                if "superordinate_A" in file.name:
                    pairs=(("category1", "category3"), ("category1", "category4"))
                elif "cohyponym_A" in file.name:
                    pairs = (("cohyp1", "cohyp2"), ("cohyp1", "cohyp3"), ("cohyp1", "cohyp4"))

                def make_pair(a, b):
                    A = df[df["condition"] == a].copy()
                    B = df[df["condition"] == b].copy()

                    B = B.rename(columns={
                    "log_prob": "log_prob_b",
                    "pearsonr": "pearsonr_b",
                    "comparison": "comparison_b",
                    })

                    merged = A.merge(
                        B[meta_cols + ["log_prob_b", "pearsonr_b", "comparison_b"]],
                        on=meta_cols,
                        how="inner",
                        validate="one_to_one"
                    )

                    out = merged[meta_cols].copy()
                    out["condition"] = f"{a}-{b}"
                    out["comparison"] = (
                        merged["comparison"].astype(str).str.strip()
                        + "-"
                        + merged["comparison_b"].astype(str).str.strip()
                    )
                    out["target"] = out["comparison"]

                    if "negated" in merged["prompt_key"][0]:
                        out["LPAcc"]  = (merged["log_prob"] < merged["log_prob_b"]).astype(int)
                        out["EmbAcc"] = (merged["pearsonr"] < merged["pearsonr_b"]).astype(int)
                    else:
                        out["LPAcc"]  = (merged["log_prob"] > merged["log_prob_b"]).astype(int)
                        out["EmbAcc"] = (merged["pearsonr"] > merged["pearsonr_b"]).astype(int)

                    return out
            
            elif "_B" in file.name:
                df["target"] = df["target"].astype(str).str.strip()
                exclude = {"target", "log_prob", "pearsonr", "pred"}
                meta_cols = [c for c in df.columns if c not in exclude and c != "input_text"]
                pairs = (("Yes", "No"),)
                def make_pair(a, b):
                    A = df[df["target"] == a].copy()
                    B = df[df["target"] == b].copy()

                    B = B.rename(columns={
                        "log_prob": "log_prob_b",
                        "pearsonr": "pearsonr_b",
                        "target": "target_b",
                    })
                    merged = A.merge(
                        B[meta_cols + ["log_prob_b", "pearsonr_b", "target_b"]],
                        on=meta_cols,
                        how="inner",
                        validate="one_to_one"
                    )
                    out = merged[meta_cols].copy()
                    out["target"] = f"{a}-{b}"

                    out["LPAcc"]  = (merged["log_prob"] > merged["log_prob_b"]).astype(int)
                    out["EmbAcc"] = (merged["pearsonr"] > merged["pearsonr_b"]).astype(int)
                    
                    return out

            accuracy_df = pd.concat([make_pair(a, b) for a, b in pairs], ignore_index=True)
            df_list.append(accuracy_df)

        accuracy_df = pd.concat(df_list, ignore_index=True)
        accuracy_df.insert(0, "model", model_name)
        accuracy_df.to_csv(os.path.join(output_path, f"accuracy.csv"), index=False)
            


def main():
    output_path = "./figures_tables"
    Path(output_path).mkdir(exist_ok=True)
    root_dir = "./evaluation"
    create_accuracy_df(root_dir)

if __name__ == "__main__":
    main()