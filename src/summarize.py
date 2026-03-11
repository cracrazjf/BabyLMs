from pathlib import Path
from typing import List, Union, Dict
import pandas as pd
import os
import json
import numpy as np

def create_accuracy_df(save_path):
    root_dir = Path("./results")
    models_accuracy_df_list = []
    for model_dir in root_dir.iterdir():
        model_name = model_dir.name
        print(f"Processing model: {model_name}")
        file = list(model_dir.glob("eval_results.jsonl"))[0]
        df = pd.read_json(file, lines=True).drop(columns=["input_text", "task", "idx"])
        superordinate_df = df[df["relationship"] == "superordinate"].copy()
        superordinate_pairs = (("category1", "category3"), ("category1", "category4"))


        cohyponym_df = df[df["relationship"] == "cohyponym"].copy()
        cohyponym_pairs = (("cohyp1", "cohyp3"), ("cohyp1", "cohyp4"))

        exclude = {"condition", "comparison", "target", "sum_log_prob", "pearsonr", "pred"}
        include = [c for c in df.columns if c not in exclude]

        def make_pair(df, a, b):
            A = df[df["condition"] == a].copy()
            B = df[df["condition"] == b].copy()

            B = B.rename(columns={
                "sum_log_prob": "sum_log_prob_b",
                "pearsonr": "pearsonr_b",
                "comparison": "comparison_b",
                })

            merged = A.merge(
                B[include + ["sum_log_prob_b", "pearsonr_b", "comparison_b"]],
                on=include,
                how="inner",
                validate="one_to_one"
            )

            out = merged[include].copy()
            if "3" in b:
                out["condition"] = f"hard"
            else:
                out["condition"] = f"easy"

            out["target"] = merged["comparison"].astype(str).str.strip()
            out["comparison"] = merged["comparison_b"].astype(str).str.strip()

            out["target_sum_lp"] = merged["sum_log_prob"]
            out["comparison_sum_lp"] = merged["sum_log_prob_b"]
            out["target_pearsonr"] = merged["pearsonr"]
            out["comparison_pearsonr"] = merged["pearsonr_b"]

            out["Sum_LPAcc"]  = (merged["sum_log_prob"] > merged["sum_log_prob_b"]).astype(int)
            out["EmbAcc"] = (merged["pearsonr"] > merged["pearsonr_b"]).astype(int)

            return out

        superordinate_accuracy_df = [make_pair(superordinate_df, a, b) for a, b in superordinate_pairs]
        cohyponym_accuracy_df = [make_pair(cohyponym_df, a, b) for a, b in cohyponym_pairs]

        model_accuracy_df = pd.concat(superordinate_accuracy_df + cohyponym_accuracy_df, ignore_index=True)
        model_accuracy_df.insert(0, "model", model_name)
        models_accuracy_df_list.append(model_accuracy_df)

    raw_data = pd.read_excel("./data/ACL/LLM_Categories_stim.xlsx", sheet_name=None)
    category_dict = dict(zip(raw_data["probes"]["instance"], raw_data["probes"]["category"]))
    models_accuracy_df = pd.concat(models_accuracy_df_list, ignore_index=True)

    probe_idx = models_accuracy_df.columns.get_loc("probe") + 1
    probe_category = models_accuracy_df["probe"].str.strip().map(category_dict)
    models_accuracy_df.insert(probe_idx, "probe_category", probe_category)

    models_accuracy_df.to_csv(os.path.join(save_path, f"accuracy.csv"), index=False)

    return models_accuracy_df

def melt_metrics(df: pd.DataFrame, metrics: Dict) -> pd.DataFrame:
    id_cols = [c for c in df.columns if c not in {x for v in metrics.values() for x in v}]
    df_list = []
    for m, (tcol, ccol, acccol) in metrics.items():
        tmp = df[id_cols].copy()
        tmp["measure"] = m
        tmp["acc"] = df[acccol]
        tmp["value_target"] = df[tcol]
        tmp["value_comparison"] = df[ccol]
        df_list.append(tmp)
    df = pd.concat(df_list, ignore_index=True)
    df.to_csv(f"./summary/accuracy_melted.csv", index=False)
    return df

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
    save_path = "./summary"

    Path(save_path).mkdir(exist_ok=True)
    if not os.path.exists(os.path.join(save_path, "accuracy.csv")):
        df = create_accuracy_df(save_path)
    else:
        df = pd.read_csv(os.path.join(save_path, "accuracy.csv"))

    metrics = {
        "sum logprob": ("target_sum_lp", "comparison_sum_lp", "Sum_LPAcc"),
        "embed sim": ("target_pearsonr", "comparison_pearsonr", "EmbAcc"),
    }
    cleaned_df = melt_metrics(df, metrics)

    group_and_aggregate(
        cleaned_df,
        groupby=["model", "relationship", "condition", "measure"],
        metrics=["acc"],
        agg="mean",
    ).to_csv(os.path.join(save_path, f"accuracy_summary.csv"), index=False)

if __name__ == "__main__":
    main()