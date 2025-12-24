from pathlib import Path
from typing import List, Union, Dict
import pandas as pd
import os

def create_accuracy_df(root_dir: str, output_path: str = "./figures_tables"):
    root_dir = Path(root_dir)
    models_accuracy_df_list = []
    for model_dir in root_dir.iterdir():
        model_name = model_dir.name
        print(f"Processing model: {model_name}")
        files = list(model_dir.glob("*results.jsonl"))
        df_list = []
        for file in files:
            df = pd.read_json(file, lines=True).drop(columns=["input_text"], errors="ignore")
            df["meta_prompt_key"] = df["meta_prompt_key"].str.replace("task_biased", "task specific",regex=False)
            if "_A" in file.name:
                exclude = {"condition", "comparison", "target", "mean_log_prob", "sum_log_prob", "pearsonr", "pred"}
                meta_cols = [c for c in df.columns if c not in exclude and c != "input_text"]
                if "superordinate_A" in file.name:
                    pairs=(("category1", "category3"), ("category1", "category4"))
                elif "cohyponym_A" in file.name:
                    pairs = (("cohyp1", "cohyp3"), ("cohyp1", "cohyp4"))

                def make_pair(a, b):
                    A = df[df["condition"] == a].copy()
                    B = df[df["condition"] == b].copy()

                    B = B.rename(columns={
                    "mean_log_prob": "mean_log_prob_b",
                    "sum_log_prob": "sum_log_prob_b",
                    "pearsonr": "pearsonr_b",
                    "comparison": "comparison_b",
                    })

                    merged = A.merge(
                        B[meta_cols + ["mean_log_prob_b", "sum_log_prob_b", "pearsonr_b", "comparison_b"]],
                        on=meta_cols,
                        how="inner",
                        validate="one_to_one"
                    )

                    out = merged[meta_cols].copy()
                    if "3" in b:
                        out["condition"] = f"hard"
                    else:
                        out["condition"] = f"easy"

                    out["target"] = merged["comparison"].astype(str).str.strip()
                    out["comparison"] = merged["comparison_b"].astype(str).str.strip()

                    out["target_mean_lp"] = merged["mean_log_prob"]
                    out["target_sum_lp"] = merged["sum_log_prob"]
                    out["comparison_mean_lp"] = merged["mean_log_prob_b"]
                    out["comparison_sum_lp"] = merged["sum_log_prob_b"]
                    out["target_pearsonr"] = merged["pearsonr"]
                    out["comparison_pearsonr"] = merged["pearsonr_b"]

                    if "negated" in merged["prompt_key"][0]:
                        out["LPAcc"]  = (merged["log_prob"] < merged["log_prob_b"]).astype(int)
                        out["EmbAcc"] = (merged["pearsonr"] < merged["pearsonr_b"]).astype(int)
                    else:
                        out["Mean_LPAcc"]  = (merged["mean_log_prob"] > merged["mean_log_prob_b"]).astype(int)
                        out["Sum_LPAcc"]  = (merged["sum_log_prob"] > merged["sum_log_prob_b"]).astype(int)
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
        models_accuracy_df_list.append(accuracy_df)

    raw_data = pd.read_excel("./data/ACL/LLM_Categories_stim.xlsx", sheet_name=None)
    category_dict = dict(zip(raw_data["probes"]["instance"], raw_data["probes"]["category"]))
    models_accuracy_df = pd.concat(models_accuracy_df_list, ignore_index=True)

    probe_idx = models_accuracy_df.columns.get_loc("probe") + 1
    probe_category = models_accuracy_df["probe"].str.strip().map(category_dict)
    models_accuracy_df.insert(probe_idx, "probe_category", probe_category)

    prompt_key_idx = models_accuracy_df.columns.get_loc("prompt_key") + 1
    models_accuracy_df.insert(prompt_key_idx,"prompt_type", models_accuracy_df["prompt_key"].str.replace(r"\d+$", "", regex=True))
    models_accuracy_df["prompt_type"] = models_accuracy_df["prompt_type"].str.replace("task_biased", "task specific",regex=False)
        
    prompt_key_mask = models_accuracy_df["prompt_key"].str.contains("task_biased", na=False)
    models_accuracy_df.loc[prompt_key_mask, "prompt_key"] = (
        models_accuracy_df.loc[prompt_key_mask, "relationship"] + models_accuracy_df.loc[prompt_key_mask, "prompt_key"]
        .str.replace("task_biased", "", regex=False))

    models_accuracy_df.to_csv(os.path.join(output_path, f"accuracy.csv"), index=False)

def melt_metrics(df: pd.DataFrame, metrics: Dict) -> pd.DataFrame:
    id_cols = [c for c in df.columns if c not in {x for v in metrics.values() for x in v}]
    long_parts = []
    for m, (tcol, ccol, acccol) in metrics.items():
        tmp = df[id_cols].copy()
        tmp["metric"] = m
        tmp["accuracy"] = df[acccol]
        tmp["value_target"] = df[tcol]
        tmp["value_comparison"] = df[ccol]
        long_parts.append(tmp)
    tidy = pd.concat(long_parts, ignore_index=True)
    return tidy

def apply_filters(df, filters=None):
    if not filters:
        return df

    mask = pd.Series(True, index=df.index)

    for col, rule in filters.items():
        s = df[col]

        if isinstance(rule, dict):
            if "contains" in rule:
                mask &= s.astype(str).str.contains(rule["contains"], na=False)
            elif "not_contains" in rule:
                mask &= ~s.astype(str).str.contains(rule["not_contains"], na=False)
            else:
                raise ValueError(f"Unknown filter rule: {rule}")

        elif isinstance(rule, (list, tuple, set)):
            mask &= s.isin(rule)

        else:
            mask &= s == rule

    return df[mask]

def group_and_aggregate(
    df: pd.DataFrame,
    groupby: List[str],
    metrics: List[str],
    agg: Union[str, dict] = "mean",
) -> pd.DataFrame:
    for c in groupby + metrics:
        if c not in df.columns:
            raise KeyError(f"Column '{c}' not found")

    df = df.copy()

    for m in metrics:
        df[m] = pd.to_numeric(df[m], errors="coerce")

    grouped = df.groupby(groupby, dropna=False)

    summary = grouped[metrics].agg(agg).reset_index()
    counts = grouped.size().rename("n").reset_index()

    summary = summary.merge(counts, on=groupby, how="left")

    return summary.reset_index(drop=True)

def main():
    output_path = "./figures_tables"
    Path(output_path).mkdir(exist_ok=True)
    root_dir = "./evaluation/"
    if not os.path.exists("./figures_tables/accuracy.csv"):
        create_accuracy_df(root_dir)
    df = pd.read_csv("./figures_tables/accuracy.csv")
    metrics = {
        "sumlp": ("target_sum_lp", "comparison_sum_lp", "Sum_LPAcc"),
        "meanlp": ("target_mean_lp", "comparison_mean_lp", "Mean_LPAcc"),
        "emb": ("target_pearsonr", "comparison_pearsonr", "EmbAcc"),
    }
    tidy_df = melt_metrics(df, metrics)
    for metric, sub_df in tidy_df.groupby("metric"):
        sub_df.to_csv(f"./figures_tables/{metric}.csv", index=False)
    # filtered_df = apply_filters(df, filters={"task": "cloze", 
    #                                 "prompt_key": {"not_contains": "negated"}})
    
    # summary_df = group_and_aggregate(filtered_df, 
    #                                  groupby=["relationship", "meta_prompt_key","prompt_type", "condition"], 
    #                                  metrics=["Mean_LPAcc"], 
    #                                  agg="mean").round(4)
    
    # summary_df.to_csv(os.path.join(output_path, f"averaged_over_models.csv"), index=False)

    # summary_df = group_and_aggregate(filtered_df, 
    #                                  groupby=["model","relationship", "meta_prompt_key","prompt_type", "condition"], 
    #                                  metrics=["Mean_LPAcc"], 
    #                                  agg="mean").round(4)
    # summary_df.to_csv(os.path.join(output_path, f"accuracy_by_model.csv"), index=False)



if __name__ == "__main__":
    main()