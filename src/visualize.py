import json
import re
from pathlib import Path
import pandas as pd
import os
import math

metaprompt_dict = {
    "superordinate": {"cloze": {"metaprompt1": "Please complete the following sentence about the category label for the word that is provided. Respond as concisely as possible. ",
                        "metaprompt2": "Please complete the following sentence naturally. ",
                        "metaprompt3": ""},
                        "verification": {"metaprompt1": """Please answer the following question about the whether the provided word belongs to the stated category. Respond by saying only "True" or "False". """,
                        "metaprompt2": """Please answer the following question. Respond by saying only "True" or "False". """,
                        "metaprompt3": ""}
                        },
    "cohyponym": {"cloze": {"metaprompt1": "Please complete the following sentence about words and whether they belong to the same category. Respond as concisely as possible. ",
                    "metaprompt2": "Please complete the following sentence naturally. ",
                    "metaprompt3": ""},
                    "verification": {"metaprompt1": """Please answer the following question about whether the two words belong to the same category. Respond by saying only "True" or "False". """,
                    "metaprompt2": """Please answer the following question. Respond by saying only "True" or "False". """,
                    "metaprompt3": ""}
                    },
}

prompt_dict = {
    "superordinate": {"cloze":{"prompt1": "is",
                        "prompt2": "is a kind of",
                        "prompt3": "is a type of ",
                        "prompt4": "belongs to the category.",
                        "prompt5": "is classified as.",
                        "negated_prompt1": "is not",
                        "negated_prompt2": "is not a kind of",
                        "negated_prompt3": "is not a type of",
                        "negated_prompt4": "does not belong to the category",
                        "negated_prompt5": "is not classified as",
                        "control_prompt1": "_",
                        "control_prompt2": ":.",
                        "control_prompt3": "->",
                        "control_prompt4": "—",
                        "control_prompt5": "and"},
                        "verification": {"prompt1": "is",
                        "prompt2": "is a kind of",
                        "prompt3": "is a type of ",
                        "prompt4": "belongs to the category.",
                        "prompt5": "is classified as.",
                        "negated_prompt1": "is not",
                        "negated_prompt2": "is not a kind of",
                        "negated_prompt3": "is not a type of",
                        "negated_prompt4": "does not belong to the category",
                        "negated_prompt5": "is not classified as",
                        "control_prompt1": "_",
                        "control_prompt2": ":.",
                        "control_prompt3": "->",
                        "control_prompt4": "—",
                        "control_prompt5": "and"},},

    "cohyponym": {"cloze": {"prompt1": "is like",
                    "prompt2": "is similar to",
                    "prompt3": "Two words that belong to the same category are X and Y. ",
                    "prompt4": "Another word that belongs to the same category as X is Y.",
                    "prompt5": "is the same type of thing as",
                    "negated_prompt1": "is not like",
                    "negated_prompt2": "is not similar to",
                    "negated_prompt3": "Two words that do not belong to the same category are X and Y. ",
                    "negated_prompt4": "Another word that does not belong to the same category as X is Y.",
                    "negated_prompt5": "is not the same type of thing as",
                    "control_prompt1": "_",
                    "control_prompt2": ":.",
                    "control_prompt3": "->",
                    "control_prompt4": "—",
                    "control_prompt5": "and"},
                    "verification": {"prompt1": "is like",
                    "prompt2": "is similar to",
                    "prompt3": "Two words that belong to the same category are X and Y. ",
                    "prompt4": "Another word that belongs to the same category as X is Y.",
                    "prompt5": "is the same type of thing as",
                    "negated_prompt1": "is not like",
                    "negated_prompt2": "is not similar to",
                    "negated_prompt3": "Two words that do not belong to the same category are X and Y. ",
                    "negated_prompt4": "Another word that does not belong to the same category as X is Y.",
                    "negated_prompt5": "is not the same type of thing as",
                    "control_prompt1": "_",
                    "control_prompt2": ":.",
                    "control_prompt3": "->",
                    "control_prompt4": "—",
                    "control_prompt5": "and"}},
}

def create_accuracy_df(root_dir: str, output_path: str = "./figures_tables"):
    root_dir = Path(root_dir)
    model_name = root_dir.name

    files = list(root_dir.glob("*.jsonl"))
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
            pairs = (("True", "False"),)
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
    accuracy_df["model"] = model_name

    def prompt_lookup(row):
        return (
            prompt_dict
            .get(row["relationship"], {})
            .get(row["task"], {})
            .get(row["prompt_key"])
        )
    
    def prompt_type(row):
        if "negated" in row["prompt_key"]:
            return "negated"
        elif "control" in row["prompt_key"]:
            return "control"
        else:
            return "positive"

    def metaprompt_lookup(row):
        return (
            metaprompt_dict
            .get(row["relationship"], {})
            .get(row["task"], {})
            .get(row["meta_prompt_key"])
        )
    
    columns_to_keep = ["model", "relationship", "task", "condition", "meta_prompt_key", "prompt_key",]
    tidy_df = (accuracy_df
               .melt(
                   id_vars=columns_to_keep,
                   value_vars=["LPAcc", "EmbAcc"],
                   var_name="metric",
                   value_name="accuracy",
                   )
                   .dropna(subset=["accuracy"])
                    )
    summary_df = (
            tidy_df
            .groupby(["model", "relationship", "task", "condition", "meta_prompt_key", "prompt_key", "metric"], as_index=False)
            .agg(mean_accuracy=("accuracy", "mean"))
        )
    prompt_col_idx = df.columns.get_loc("prompt_key")
    metaprompt_col_idx = df.columns.get_loc("meta_prompt_key")
    summary_df.insert(prompt_col_idx, "prompt", summary_df.apply(prompt_lookup, axis=1))
    summary_df.insert(prompt_col_idx + 1, "prompt_type", summary_df.apply(prompt_type, axis=1))
    summary_df.insert(metaprompt_col_idx, "metaprompt", summary_df.apply(metaprompt_lookup, axis=1))
    summary_df = summary_df.drop(columns=["prompt_key", "meta_prompt_key"])
    summary_df = summary_df.round(4)
    summary_df.to_csv(f"{output_path}/accuracy.csv", index=False)

    df_grouped_by_task = summary_df.groupby(["model", "relationship", "task", "prompt_type", "metric"], as_index=False).agg(mean_accuracy=("mean_accuracy", "mean"))
    print(df_grouped_by_task)
    df_grouped_by_task.to_csv(f"{output_path}/accuracy_task_summary.csv", index=False)
            

    
# def create_contingency_table(path):
#     rows = [json.loads(line) for line in open(path)]
#     df = pd.DataFrame(rows)
#     wc1_table = pd.crosstab(df["cc_wc1_logprob"], df["cc_wc1_embed"])
#     wc1_table = wc1_table.reindex(index=[True, False], columns=[True, False])
#     wc1_prop = wc1_table / wc1_table.values.sum()
#     wc2_table = pd.crosstab(df["cc_wc2_logprob"], df["cc_wc2_embed"])
#     wc2_table = wc2_table.reindex(index=[True, False], columns=[True, False])
#     wc2_prop = wc2_table / wc2_table.values.sum()

#     return wc1_prop.round(2), wc2_prop.round(2)


def main():
    output_path = "./figures_tables"
    Path(output_path).mkdir(exist_ok=True)
    root_dir = "./evaluation/gpt2"
    create_accuracy_df(root_dir)

if __name__ == "__main__":
    main()