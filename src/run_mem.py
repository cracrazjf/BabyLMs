from dataclasses import dataclass
import json
from pathlib import Path
from datetime import datetime
import multiprocessing as mp

import pandas as pd
import bambi as bmb
import arviz as az

@dataclass
class AnalysisParams:
    analysis_name: str
    draws: int = 1000
    cores: int = 4
    tune: int = 1000
    chains: int = 4
    target_accept: float = 0.90
    out_dir: str = "./mem_analysis_small"
    random_seed: int = 0

@dataclass
class LogisticMEMParams(AnalysisParams):
    # put analysis-specific knobs here later if you want
    formula: str = ""


MEM_EMBED = LogisticMEMParams(
    analysis_name="mem_embed_v1",
    # sampling knobs...
    formula=(
        "acc ~ relationship * condition + relationship * model + condition * model"
        " + (1|probe_group) + (1|target) + (1|comparison)"
    ),
    cores=4,
    )

MEM_LP = LogisticMEMParams(
    analysis_name="mem_logprob_v1",
    # sampling knobs...
    formula=(
        # "acc ~ relationship * condition + relationship * model + condition * model"
        # " + meta_prompt_type * prompt_type"
        # " + relationship:meta_prompt_type + relationship:prompt_type"
        # " + (1|probe_group) + (1|prompt_group) + (1|target) + (1|comparison)"
        "acc ~ relationship * condition + relationship * model + condition * model + meta_prompt_type * prompt_type"
        " + relationship:meta_prompt_type + relationship:prompt_type"
        " + (1|probe_group) + (1|prompt_group) + (1|target) + (1|comparison)"
        
    ),
    cores=4,
)


def load_df(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_idata_and_meta(idata, meta: dict, out_dir: str, analysis_name: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{analysis_name}__{stamp()}"
    nc_path = out / f"{tag}.nc"
    js_path = out / f"{tag}.json"

    az.to_netcdf(idata, nc_path)
    js_path.write_text(json.dumps(meta, indent=2))
    print("saved:", nc_path)
    print("saved:", js_path)


# ----------------------------
# preprocessing
# ----------------------------
def set_reference_levels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures consistent reference levels (treatment coding) across runs.
    Adjust order if you want different references.
    """
    df = df.copy()
    df = df[~df["model"].isin([])]
    # Fixed factor refs
    df["model"] = pd.Categorical(
        df["model"],
        categories=["gpt2", "gemma2_9b_base","mistral_7b_base", "olmo3_7b_base", "olmo3_7b_instruct", "mistral_7b_instruct", "llama3.1_8b_base", "llama3.1_8b_instruct", "qwen3_8b_base", "qwen3_8b_instruct"],
        ordered=True
    )
    df["relationship"] = pd.Categorical(df["relationship"], categories=["cohyponym", "superordinate"], ordered=True)
    df["condition"] = pd.Categorical(df["condition"], categories=["easy", "hard"], ordered=True)
    df["meta_prompt_type"] = pd.Categorical(df["meta_prompt_type"], categories=["neutral", "none", "task specific"], ordered=True)
    df["prompt_type"] = pd.Categorical(df["prompt_type"], categories=["control", "task specific"], ordered=True)

    # measure is used only upstream to split; reference level isn’t critical afterward
    df["measure"] = pd.Categorical(df["measure"], categories=["embed sim", "sum logprob"], ordered=True)

    return df


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Collapse redundant embed-sim rows (prompts/metaprompts don't affect static embeddings).
    - Keep all logprob rows (prompts matter).
    - Build grouping keys for random effects.
    """
    df = df.copy()

    # probe_group: your "probe binned with category" (feel free to change)
    df["probe_group"] = df["probe_category"].astype(str) + ":" + df["probe"].astype(str)

    # prompt_group used only for logprob models (but we can compute it for all rows)
    # neutral prompts cross relationship; task-specific prompts are relationship-nested
    df["prompt_group"] = df.apply(
        lambda r: r["prompt_key"] if r["prompt_type"] == "control" else f'{r["relationship"]}:{r["prompt_key"]}',
        axis=1
    )

    # Collapse embed-sim rows that differ only by prompt/metaprompt
    embed = df[df["measure"] == "embed sim"].drop_duplicates(
        subset=["model", "relationship", "probe_group", "condition", "target", "comparison", "measure", "acc"]
    )
    lp = df[df["measure"] == "sum logprob"]

    df2 = pd.concat([lp, embed], ignore_index=True)
    return df2


def split_df(df2: pd.DataFrame):
    df_embed = df2[df2["measure"] == "embed sim"].copy()
    df_lp = df2[df2["measure"] == "sum logprob"].copy()
    return df_embed, df_lp


# ----------------------------
# model fitting
# ----------------------------
def fit_bambi_logit(formula: str, df: pd.DataFrame, params: LogisticMEMParams):
    model = bmb.Model(formula, df, family="bernoulli")

    # IMPORTANT: ask for pointwise log-likelihood so you can do LOO later
    # This is not guaranteed unless requested.  [oai_citation:2‡Bambinos](https://bambinos.github.io/bambi/notebooks/t_regression.html?utm_source=chatgpt.com)
    # idata = model.fit(
    #     draws=params.draws,
    #     tune=params.tune,
    #     chains=params.chains,
    #     target_accept=params.target_accept,
    #     random_seed=getattr(params, "random_seed", None),
    #     idata_kwargs={"log_likelihood": True},
    # )
    idata = model.fit(
        draws=params.draws,
        tune=params.tune,
        chains=params.chains,
        cores=params.cores,  # e.g., 2 or 4
        target_accept=params.target_accept,
        random_seed=params.random_seed,
        idata_kwargs={"log_likelihood": True},
        mp_ctx=mp.get_context("forkserver"),
    )

    return model, idata

def main():

    file_path = "./summary/accuracy_melted.csv"

    df = load_df(file_path)

    df = set_reference_levels(df)
    df2 = prepare_df(df)
    df_embed, df_lp = split_df(df2)

    print("rows total:", len(df2), "| embed:", len(df_embed), "| logprob:", len(df_lp))

    # ---- Model A: embeddings (no prompt/meta terms, no prompt random effect)
    params_embed = MEM_EMBED  # define in analysis_params.py
    formula_embed = params_embed.formula
    m_embed, idata_embed = fit_bambi_logit(formula_embed, df_embed, params_embed)
    save_idata_and_meta(
        idata_embed,
        meta={"analysis": params_embed.analysis_name, "formula": formula_embed, "n_rows": len(df_embed)},
        out_dir=params_embed.out_dir,
        analysis_name=params_embed.analysis_name
    )

    # ---- Model B: logprob (prompt/meta terms, prompt random effect)
    params_lp = MEM_LP  # define in analysis_params.py
    formula_lp = params_lp.formula
    m_lp, idata_lp = fit_bambi_logit(formula_lp, df_lp, params_lp)
    save_idata_and_meta(
        idata_lp,
        meta={"analysis": params_lp.analysis_name, "formula": formula_lp, "n_rows": len(df_lp)},
        out_dir=params_lp.out_dir,
        analysis_name=params_lp.analysis_name
    )


if __name__ == "__main__":
    main()

