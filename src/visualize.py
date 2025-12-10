import json
import re
from pathlib import Path
import pandas as pd
import os
import math

def build_eval_tables_by_metric(
    root_dir,
    acc_keys_dict = {
        "cat_eval_A": ["cc_wc1_logprob", "cc_wc2_logprob", "cc_wc1_embed", "cc_wc2_embed"],
        "cat_eval_B": ["cc_yes_percent", "wc1_yes_percent", "wc2_yes_percent"],
        "cohypo_eval_A": ["c1_c2_logprob", "c1_c3_logprob", "c1_c4_logprob", "c1_c2_embed", "c1_c3_embed", "c1_c4_embed"],
        "cohypo_eval_B": ["c1_yes_percent", "c2_yes_percent", "c3_yes_percent", "c4_yes_percent"],
    },
    input_dict = {
        "cat_eval_A": ["is", "is a type of", "belongs to"],
        "cat_eval_B": ["is", "is a type of", "belongs to"],
        "cohypo_eval_A": ["like", "similar to", "equals"],
        "cohypo_eval_B": ["like", "similar to", "equals"],
    },
    metric_dict = {
        "cat_eval_A": "Category Membership Evaluation A",
        "cat_eval_B": "Category Membership Evaluation B",
        "cohypo_eval_A": "Co-hyponym Evaluation A",
        "cohypo_eval_B": "Co-hyponym Evaluation B", 
    }
):
    root_dir = Path(root_dir)

    filename_pattern = (
        r"^(?P<metric>.+)_promp?t(?P<prompt>\d+)_input_?(?P<input>\d+)_results\.jsonl$"
    )
    filename_re = re.compile(filename_pattern)

    def parse_filename(path: Path):
        m = filename_re.match(path.name)
        if not m:
            return None
        return {
            "metric_name": m.group("metric"),
            "prompt_idx": int(m.group("prompt")),
            "input_idx": int(m.group("input")),
        }

    def compute_accuracy(path: Path, acc_keys: list):
        accuracy_dict = {k: 0.0 for k in acc_keys}
        line_count = 0
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                for acc_key in acc_keys:
                    if acc_key in rec:
                        accuracy_dict[acc_key] += rec[acc_key]
                line_count += 1
            accuracy_dict = {k: v / line_count for k, v in accuracy_dict.items()}
        return accuracy_dict
    
    def create_contingency_table(path):
        rows = [json.loads(line) for line in open(path)]
        df = pd.DataFrame(rows)
        wc1_table = pd.crosstab(df["cc_wc1_logprob"], df["cc_wc1_embed"])
        wc1_table = wc1_table.reindex(index=[True, False], columns=[True, False])
        wc1_prop = wc1_table / wc1_table.values.sum()
        wc2_table = pd.crosstab(df["cc_wc2_logprob"], df["cc_wc2_embed"])
        wc2_table = wc2_table.reindex(index=[True, False], columns=[True, False])
        wc2_prop = wc2_table / wc2_table.values.sum()

        return wc1_prop.round(2), wc2_prop.round(2)

    rows = []
    contingency_tables = {}

    # Collect all rows
    for model_dir in root_dir.iterdir():
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name

        for jsonl_path in model_dir.glob("*.jsonl"):
            parsed = parse_filename(jsonl_path)
            if parsed is None:
                continue

            accuracy_dict = compute_accuracy(jsonl_path, acc_keys_dict[parsed["metric_name"]])
            if parsed["metric_name"] == "cat_eval_A":
                wc1_table, wc2_table = create_contingency_table(jsonl_path)
                contingency_tables[(model_name, parsed["prompt_idx"], input_dict[parsed["metric_name"]][parsed["input_idx"]-1])] = (wc1_table, wc2_table)


            for acc_key in accuracy_dict:
                rows.append({
                    "model": model_name,
                    "metric_name": metric_dict[parsed["metric_name"]],
                    "prompt_idx": parsed["prompt_idx"],
                    "input_idx": input_dict[parsed["metric_name"]][parsed["input_idx"]-1],
                    "acc_type": acc_key,
                    "value": round(accuracy_dict[acc_key], 2),
                })

    print("Contingency Tables for cat_eval_A:")
    for key, (wc1_table, wc2_table) in contingency_tables.items():
        print(f"{key}:")
        print("WC1 Table:")
        print(wc1_table)
        print("WC2 Table:")
        print(wc2_table)
        print()

    if not rows:
        return {}

    df = pd.DataFrame(rows)

    tables_by_metric = {}
    for metric_name, df_metric in df.groupby("metric_name"):
        df_metric["acc_type"] = pd.Categorical(df_metric["acc_type"], categories=df_metric["acc_type"].unique(), ordered=True)
        table = df_metric.pivot_table(
            index=["metric_name", "prompt_idx", "input_idx", "acc_type"],
            columns="model",
            values="value",
            observed=True
        ).sort_index()
        tables_by_metric[metric_name] = table

    return tables_by_metric, contingency_tables

def main():
    output_path = "./figures"
    Path(output_path).mkdir(exist_ok=True)
    tables_by_metric, contingency_tables = build_eval_tables_by_metric("./evaluation")

    for metric_name, table in tables_by_metric.items():
        save_path = Path(output_path) / f"{metric_name.replace(' ', '_')}_results_table.csv"
        table.to_csv(save_path)
        print(f"Saved results table for {metric_name} to {save_path}")

    contingency_rows = []
    for (model_name, prompt_idx, input_desc), (wc1, wc2) in contingency_tables.items():
        for wc_name, table in (("wc1", wc1), ("wc2", wc2)):
            table = table.reindex(index=[True, False], columns=[True, False])
            TT = table.loc[True,  True]
            TF = table.loc[True,  False]
            FT = table.loc[False, True]
            FF = table.loc[False, False]

            contingency_rows.append({
            "model_name": model_name,
            "prompt_idx": prompt_idx,
            "input_desc": input_desc,
            "wc": wc_name,
            "TT": TT,
            "TF": TF,
            "FT": FT,
            "FF": FF,
            })

    big_df = pd.DataFrame(contingency_rows)
    big_df.to_csv("./figures/all_contingency_tables_wide.csv", index=False)
    # df = pd.read_json("./data/childes/flores_stimuli_with_categories.jsonl", lines=True)
    # df.to_csv("./figures/flores_stimuli_with_categories.csv", index=False)

if __name__ == "__main__":
    main()