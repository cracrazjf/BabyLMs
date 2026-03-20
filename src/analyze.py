#!/usr/bin/env python3
"""
Generate posterior predicted cell means from fitted bambi mixed-effects models,
by predicting on the *raw rows* and then averaging within analysis-defined cells.

Outputs probabilities (accuracy) + intervals computed from posterior draws.

Also optionally outputs raw (observed) cell means of acc for sanity checking,
with a Jeffreys interval (Beta(0.5, 0.5)).

Expected input columns in cleaned_accuracy.csv:
model, relationship, meta_prompt_type, prompt_key, prompt_type, combined_prompt,
probe, probe_category, condition, target, comparison, measure, acc
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import arviz as az
import bambi as bmb

from run_mem import MEM_EMBED, MEM_LP  # must define MEM_EMBED, MEM_LP


# ----------------------------
# Utilities
# ----------------------------

def timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def inv_logit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x))


def ensure_prob(x: np.ndarray, *, name: str = "pred") -> np.ndarray:
    """
    Ensure predictions are on probability scale.
    If values fall outside [0,1], assume logits and inverse-logit transform.
    After transform, enforce [0,1] (with small tolerance).
    """
    x = np.asarray(x, dtype=float)
    x_min = np.nanmin(x)
    x_max = np.nanmax(x)

    if x_min < -1e-6 or x_max > 1.0 + 1e-6:
        x = inv_logit(x)

    x_min2 = np.nanmin(x)
    x_max2 = np.nanmax(x)
    if x_min2 < -1e-6 or x_max2 > 1.0 + 1e-6:
        raise ValueError(
            f"{name} not in [0,1] after conversion. min={x_min2}, max={x_max2}. "
            "This suggests you're extracting the wrong variable from idata_pred."
        )
    return np.clip(x, 0.0, 1.0)


def interval_bounds(draws: np.ndarray, interval_prob: float) -> Tuple[float, float]:
    """
    Equal-tailed interval (ETI) bounds for 1D draws.
    """
    lo_q = (1.0 - interval_prob) / 2.0
    hi_q = 1.0 - lo_q
    lo = float(np.quantile(draws, lo_q))
    hi = float(np.quantile(draws, hi_q))
    return lo, hi


def set_reference_levels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Match the factor reference levels used when fitting.
    IMPORTANT: must match run_split_models.py
    """
    df = df.copy()

    df["model"] = pd.Categorical(
        df["model"],
        categories=["gpt2", "gemma2_9b_base","mistral_7b_base", "olmo3_7b_base", "olmo3_7b_instruct", "mistral_7b_instruct", "llama3.1_8b_base", "llama3.1_8b_instruct", "qwen3_8b_base", "qwen3_8b_instruct"],
        ordered=True
    )
    df["relationship"] = pd.Categorical(df["relationship"], categories=["cohyponym", "superordinate"], ordered=True)
    df["condition"] = pd.Categorical(df["condition"], categories=["easy", "hard"], ordered=True)
    df["meta_prompt_type"] = pd.Categorical(df["meta_prompt_type"], categories=["neutral", "none", "task specific"], ordered=True)
    df["prompt_type"] = pd.Categorical(df["prompt_type"], categories=["control", "task specific"], ordered=True)
    df["measure"] = pd.Categorical(df["measure"], categories=["embed sim", "sum logprob"], ordered=True)

    return df


def add_group_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["probe_group"] = df["probe_category"].astype(str) + ":" + df["probe"].astype(str)

    # prompt_group is meaningful for logprob only, but safe to create for all
    # control prompts: prompt_key
    # task-specific prompts: relationship:prompt_key
    df["prompt_group"] = np.where(
        df["prompt_type"].astype(str) == "control",
        df["prompt_key"].astype(str),
        df["relationship"].astype(str) + ":" + df["prompt_key"].astype(str),
    )
    return df


def add_grouping_vars_like_fit(df: pd.DataFrame, *, for_logprob: bool) -> pd.DataFrame:
    """
    Recreate grouping variables used during fitting.

    - probe_group: probe_category:probe
    - prompt_group (logprob only): nests task-specific prompts within relationship, per your description:
        control -> prompt_key
        task specific -> relationship:prompt_key
    """
    df = df.copy()

    required = {"probe", "probe_category", "relationship"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns for grouping vars: {sorted(missing)}")

    df["probe_group"] = df["probe_category"].astype(str) + ":" + df["probe"].astype(str)

    if for_logprob:
        if "prompt_key" not in df.columns or "prompt_type" not in df.columns:
            raise KeyError("Logprob dataset requires prompt_key and prompt_type to build prompt_group.")
        df["prompt_group"] = np.where(
            df["prompt_type"].astype(str) == "control",
            df["prompt_key"].astype(str),
            df["relationship"].astype(str) + ":" + df["prompt_key"].astype(str),
        )

    return df


def split_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_embed = df[df["measure"] == "embed sim"].copy()
    df_lp = df[df["measure"] == "sum logprob"].copy()
    return df_embed, df_lp


def collapse_embed_rows_like_fit(df_embed: pd.DataFrame) -> pd.DataFrame:
    """
    Mirror run_split_models.py behavior: embedding rows that differ only by prompt/metaprompt
    are redundant. Deduplicate so we don't overweight prompt variants when averaging.
    """
    subset = ["model", "relationship", "probe_group", "condition", "target", "comparison", "measure", "acc"]
    return df_embed.drop_duplicates(subset=subset).copy()


def collapse_embed_rows_like_fit_contrast(df_embed: pd.DataFrame) -> pd.DataFrame:
    """
    Replicate the 'collapse' used before fitting the embed model:
    drop duplicate rows that differ only in prompt/metaprompt fields.
    """
    df = df_embed.copy()

    # These columns can vary even though embed sims do not
    drop_cols = [c for c in ["prompt_type", "meta_prompt_type", "prompt_key", "combined_prompt"] if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Deduplicate on the columns that remain (after adding probe_group)
    df = df.drop_duplicates().copy()
    return df


def build_bambi_model(formula: str, df: pd.DataFrame) -> bmb.Model:
    return bmb.Model(formula, df, family="bernoulli")


def predict_row_probs(model: bmb.Model,
                      idata: az.InferenceData,
                      df_rows: pd.DataFrame,
                      *,
                      include_group_specific: bool) -> np.ndarray:
    """
    Predict per-row mean parameter (Bernoulli p) for each posterior draw.
    Returns array shape (n_draws_total, n_rows).
    """
    idata_pred = model.predict(
        idata=idata,
        data=df_rows,
        kind="mean",
        include_group_specific=include_group_specific,
        inplace=False,
    )

    # Find the predicted variable in idata_pred.posterior
    # We look for a variable with dims (chain, draw, obs_dim) where obs_dim matches n_rows.
    posterior = idata_pred.posterior
    n_rows = df_rows.shape[0]

    pred_da = None
    for var in posterior.data_vars:
        da = posterior[var]
        if "chain" in da.dims and "draw" in da.dims:
            other_dims = [d for d in da.dims if d not in ("chain", "draw")]
            if len(other_dims) == 1 and da.sizes[other_dims[0]] == n_rows:
                pred_da = da
                break

    if pred_da is None:
        # Sometimes bambi stores predictions under posterior_predictive; try that.
        if hasattr(idata_pred, "posterior_predictive") and idata_pred.posterior_predictive is not None:
            pp = idata_pred.posterior_predictive
            for var in pp.data_vars:
                da = pp[var]
                if "chain" in da.dims and "draw" in da.dims:
                    other_dims = [d for d in da.dims if d not in ("chain", "draw")]
                    if len(other_dims) == 1 and da.sizes[other_dims[0]] == n_rows:
                        pred_da = da
                        break

    if pred_da is None:
        raise RuntimeError(
            "Could not locate prediction variable in idata_pred. "
            f"posterior vars: {list(idata_pred.posterior.data_vars)}"
        )

    obs_dim = [d for d in pred_da.dims if d not in ("chain", "draw")][0]
    stacked = pred_da.stack(sample=("chain", "draw")).transpose("sample", obs_dim)
    samples = stacked.values  # (n_samples, n_rows)

    # Convert immediately to probabilities BEFORE any averaging
    samples = ensure_prob(samples, name="row_pred")

    return samples


def summarize_cells_from_row_draws(df_rows: pd.DataFrame,
                                  row_draws: np.ndarray,
                                  cell_cols: List[str],
                                  *,
                                  interval_prob: float,
                                  equal_weight_models: bool,
                                  model_col: str = "model") -> pd.DataFrame:
    """
    Summarize posterior row-level draws into cell means.
    If equal_weight_models=True and model_col not in cell_cols,
    we compute the cell mean per-model and then average equally over models.
    """
    if row_draws.shape[1] != df_rows.shape[0]:
        raise ValueError("row_draws second dimension must equal df_rows rows")

    # Construct cell keys
    cell_df = df_rows[cell_cols].astype(str).copy()
    cell_key = cell_df.agg("|".join, axis=1).values

    # Precompute unique cell mapping for output
    uniq_cells = pd.DataFrame(cell_df.drop_duplicates()).reset_index(drop=True)
    uniq_cells["__cell_key__"] = uniq_cells.astype(str).agg("|".join, axis=1)

    # Summarize per cell
    out_rows = []

    if equal_weight_models and (model_col in df_rows.columns) and (model_col not in cell_cols):
        # Two-stage: within each cell, average across rows per model, then equal-average across models.
        models = sorted(df_rows[model_col].astype(str).unique().tolist())
        for _, row in uniq_cells.iterrows():
            key = row["__cell_key__"]
            idx_cell = np.where(cell_key == key)[0]
            if idx_cell.size == 0:
                continue

            # per-model means for each draw
            per_model = []
            for m in models:
                idx = idx_cell[df_rows.iloc[idx_cell][model_col].astype(str).values == m]
                if idx.size == 0:
                    continue
                per_model.append(row_draws[:, idx].mean(axis=1))  # (n_draws,)
            if not per_model:
                continue

            per_model = np.vstack(per_model)            # (n_models_present, n_draws)
            cell_draws = per_model.mean(axis=0)         # equal-average over models (n_draws,)

            lo, hi = interval_bounds(cell_draws, interval_prob)
            d = {c: row[c] for c in cell_cols}
            d.update({
                "p_mean": float(cell_draws.mean()),
                "ci_prob": float(interval_prob),
                "ci_lo": lo,
                "ci_hi": hi,
                "n_rows": int(idx_cell.size),
            })
            out_rows.append(d)

    else:
        # Simple: average across all rows in the cell (weights by row counts).
        for _, row in uniq_cells.iterrows():
            key = row["__cell_key__"]
            idx = np.where(cell_key == key)[0]
            if idx.size == 0:
                continue

            cell_draws = row_draws[:, idx].mean(axis=1)  # (n_draws,)
            lo, hi = interval_bounds(cell_draws, interval_prob)

            d = {c: row[c] for c in cell_cols}
            d.update({
                "p_mean": float(cell_draws.mean()),
                "ci_prob": float(interval_prob),
                "ci_lo": lo,
                "ci_hi": hi,
                "n_rows": int(idx.size),
            })
            out_rows.append(d)

    return pd.DataFrame(out_rows)


def summarize_raw_cells(df_rows: pd.DataFrame,
                        cell_cols: List[str],
                        *,
                        interval_prob: float) -> pd.DataFrame:
    """
    Raw observed cell means of acc plus a Jeffreys interval:
      posterior for p is Beta(k+0.5, n-k+0.5)
    which gives a nice sanity-check interval.
    """
    out = []
    g = df_rows.groupby(cell_cols, dropna=False)
    alpha0 = 0.5
    beta0 = 0.5
    lo_q = (1.0 - interval_prob) / 2.0
    hi_q = 1.0 - lo_q

    for keys, sub in g:
        if not isinstance(keys, tuple):
            keys = (keys,)
        n = sub.shape[0]
        k = float(sub["acc"].sum())
        p_mean = k / n if n > 0 else np.nan

        a = k + alpha0
        b = (n - k) + beta0

        # Jeffreys posterior interval
        lo = float(az.stats.stats._quantile(np.random.beta(a, b, size=200000), lo_q))  # Monte Carlo quantile
        hi = float(az.stats.stats._quantile(np.random.beta(a, b, size=200000), hi_q))

        row = {col: val for col, val in zip(cell_cols, keys)}
        row.update({
            "raw_mean": p_mean,
            "ci_prob": float(interval_prob),
            "ci_lo": lo,
            "ci_hi": hi,
            "n_rows": int(n),
        })
        out.append(row)

    return pd.DataFrame(out)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _extract_prediction_da(idata_pred: az.InferenceData):
    """
    Find the prediction DataArray produced by bambi.Model.predict(inplace=False).
    We look across common groups and pick the variable with the largest observation dimension.
    """
    candidate_groups = []
    for g in ("posterior", "predictions", "posterior_predictive"):
        if hasattr(idata_pred, g):
            candidate_groups.append(g)

    if not candidate_groups:
        raise RuntimeError("No posterior/predictions groups found in prediction InferenceData.")

    best = None  # (group, varname, da, obs_dim, obs_size)
    for g in candidate_groups:
        ds = getattr(idata_pred, g)
        for v in ds.data_vars:
            da = ds[v]
            # Identify an observation dimension (anything other than chain/draw/sample)
            obs_dims = [d for d in da.dims if d not in ("chain", "draw", "sample")]
            if len(obs_dims) != 1:
                continue
            obs_dim = obs_dims[0]
            obs_size = da.sizes.get(obs_dim, 0)
            if obs_size <= 0:
                continue
            if best is None or obs_size > best[-1]:
                best = (g, v, da, obs_dim, obs_size)

    if best is None:
        # Helpful debug message
        groups = []
        for g in candidate_groups:
            ds = getattr(idata_pred, g)
            groups.append(f"{g}: vars={list(ds.data_vars)} dims={[tuple(ds[v].dims) for v in ds.data_vars]}")
        raise RuntimeError(
            "Could not find a prediction variable with exactly one observation dimension. "
            "Groups inspected:\n" + "\n".join(groups)
        )

    return best[2], best[3]  # da, obs_dim


def predict_row_probs_as_draws(model: bmb.Model, idata: az.InferenceData, df_rows: pd.DataFrame) -> np.ndarray:
    """
    Return row-level predicted probabilities as an array of shape (n_draws, n_rows).

    We request bambi's predictions, then:
    - extract the predicted mean per observation per posterior draw
    - ensure values are on probability scale (apply sigmoid if needed)
    """
    idata_pred = model.predict(idata=idata, data=df_rows, kind="mean", include_group_specific=False, inplace=False)
    da, obs_dim = _extract_prediction_da(idata_pred)

    # Convert to (sample, obs)
    if "chain" in da.dims and "draw" in da.dims:
        arr = da.transpose("chain", "draw", obs_dim).values
        arr = arr.reshape(arr.shape[0] * arr.shape[1], arr.shape[2])
    elif "sample" in da.dims:
        arr = da.transpose("sample", obs_dim).values
    else:
        raise RuntimeError(f"Unexpected prediction dims: {da.dims}")

    # Safety: ensure probability scale
    if (arr.min() < 0.0) or (arr.max() > 1.0):
        arr = _sigmoid(arr)

    return arr  # (n_draws, n_rows)


def interval_bounds_contrast(x: np.ndarray, *, ci_prob: float, interval_kind: str) -> tuple[float, float]:
    x = np.asarray(x)

    kind = interval_kind.lower()
    if kind == "hdi":
        h = az.hdi(x, hdi_prob=ci_prob)
        h = np.asarray(h)
        return float(h[0]), float(h[1])

    if kind == "eti":
        alpha = (1.0 - ci_prob) / 2.0
        return float(np.quantile(x, alpha)), float(np.quantile(x, 1.0 - alpha))

    raise ValueError("interval_kind must be 'hdi' or 'eti'.")


def summarize_delta(delta: np.ndarray, *, ci_prob: float, interval_kind: str) -> dict:
    lo, hi = interval_bounds_contrast(delta, ci_prob=ci_prob, interval_kind=interval_kind)
    return dict(
        delta_mean=float(np.mean(delta)),
        delta_ci_lo=lo,
        delta_ci_hi=hi,
        p_gt_0=float(np.mean(delta > 0)),
        p_lt_0=float(np.mean(delta < 0)),
    )


def mean_prob_over_rows(row_draws: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    row_draws: (n_draws, n_rows)
    mask: boolean of length n_rows
    returns: (n_draws,) mean over masked rows
    """
    if mask.dtype != bool:
        mask = mask.astype(bool)
    if mask.sum() == 0:
        raise ValueError("Mask selects 0 rows; cannot compute mean.")
    return row_draws[:, mask].mean(axis=1)


@dataclass(frozen=True)
class ContrastSpec:
    analysis: str
    model: str  # "avg" for analyses averaged over model, else concrete model name
    measure: str  # "logprob" or "embed"
    side_a: str
    side_b: str
    mask_a: np.ndarray
    mask_b: np.ndarray


def mask_task_specific(df: pd.DataFrame) -> np.ndarray:
    return (
        (df["prompt_type"].astype(str) == "task specific")
        & (df["meta_prompt_type"].astype(str) == "task specific")
    ).to_numpy()


def predict_average_mean(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    # df = df[~df["model"].isin(["olmo3_7b_base", "olmo3_7b_instruct"])]
    df = set_reference_levels(df)
    df = add_group_cols(df)

    df_embed, df_lp = split_df(df)

    # IMPORTANT: reset indices so row positions match draw columns
    df_lp = df_lp.reset_index(drop=True)

    df_embed = collapse_embed_rows_like_fit(df_embed)
    df_embed = df_embed.reset_index(drop=True)

    # Load idata
    idata_embed = az.from_netcdf(args.nc_embed)
    idata_lp = az.from_netcdf(args.nc_lp)

    # Build bambi model objects (needed to interpret the posterior for prediction)
    m_embed = build_bambi_model(MEM_EMBED.formula, df_embed)
    m_lp = build_bambi_model(MEM_LP.formula, df_lp)

    # Predict per-row probabilities
    row_probs_embed = predict_row_probs(
        m_embed, idata_embed, df_embed, include_group_specific=args.include_group_specific
    )
    row_probs_lp = predict_row_probs(
        m_lp, idata_lp, df_lp, include_group_specific=args.include_group_specific
    )

    ts = timestamp()

    # -----------------------
    # Analysis 1 (marginal over model): relationship x condition, separately by measure
    # Restrict logprob rows to task-specific prompt & metaprompt
    # Embedding does not depend on prompt; we use deduped df_embed as is.
    # -----------------------
    a1_mask_lp = mask_task_specific(df_lp)
    a1_lp_draws = row_probs_lp[:, a1_mask_lp]
    a1_lp = df_lp.loc[a1_mask_lp].reset_index(drop=True)

    a1_embed = df_embed  # already deduped; prompt doesn't matter
    a1_embed_draws = row_probs_embed  # already aligned to df_embed rows

    a1_cols = ["relationship", "condition"]

    a1_lp_summary = summarize_cells_from_row_draws(
        a1_lp.reset_index(drop=True),
        a1_lp_draws[:, :a1_lp.shape[0]],
        a1_cols,
        interval_prob=args.interval_prob,
        equal_weight_models=args.equal_weight_models,
    )
    a1_lp_summary.insert(0, "measure", "sum logprob")

    a1_embed_summary = summarize_cells_from_row_draws(
        a1_embed.reset_index(drop=True),
        a1_embed_draws,
        a1_cols,
        interval_prob=args.interval_prob,
        equal_weight_models=args.equal_weight_models,
    )
    a1_embed_summary.insert(0, "measure", "embed sim")

    analysis1 = pd.concat([a1_lp_summary, a1_embed_summary], ignore_index=True)
    analysis1_path = out_dir / f"analysis1_cell_means__{ts}.csv"
    analysis1.to_csv(analysis1_path, index=False)

    # -----------------------
    # Analysis 2 (logprob only, marginal over model):
    # relationship x condition x prompt_type x meta_prompt_type
    # -----------------------
    a2_cols = ["relationship", "condition", "prompt_type", "meta_prompt_type"]
    analysis2 = summarize_cells_from_row_draws(
        df_lp.reset_index(drop=True),
        row_probs_lp,
        a2_cols,
        interval_prob=args.interval_prob,
        equal_weight_models=args.equal_weight_models,
    )
    analysis2.insert(0, "measure", "sum logprob")
    analysis2_path = out_dir / f"analysis2_cell_means__{ts}.csv"
    analysis2.to_csv(analysis2_path, index=False)

    # -----------------------
    # Analysis 3 (task-specific prompt & metaprompt): relationship x condition x measure x model
    # Here we keep model as a cell factor.
    # -----------------------
    a3_mask_lp = mask_task_specific(df_lp)
    a3_lp_draws = row_probs_lp[:, a3_mask_lp]
    a3_lp = df_lp.loc[a3_mask_lp].reset_index(drop=True)

    a3_embed = df_embed
    a3_embed_draws = row_probs_embed

    a3_cols = ["model", "relationship", "condition"]

    a3_lp_summary = summarize_cells_from_row_draws(
        a3_lp.reset_index(drop=True),
        a3_lp_draws[:, :a3_lp.shape[0]],
        a3_cols,
        interval_prob=args.interval_prob,
        equal_weight_models=False,
    )
    a3_lp_summary.insert(0, "measure", "sum logprob")

    a3_embed_summary = summarize_cells_from_row_draws(
        a3_embed.reset_index(drop=True),
        a3_embed_draws,
        a3_cols,
        interval_prob=args.interval_prob,
        equal_weight_models=False,
    )
    a3_embed_summary.insert(0, "measure", "embed sim")

    analysis3 = pd.concat([a3_lp_summary, a3_embed_summary], ignore_index=True)
    analysis3_path = out_dir / f"analysis3_cell_means__{ts}.csv"
    analysis3.to_csv(analysis3_path, index=False)

    # -----------------------
    # Full cell means (optional convenience): model x relationship x condition x measure x prompt_type x meta_prompt_type
    # For embed, prompt/meta are meaningless; we output NAs by merging later if you want.
    # Here we only output logprob full cells, since embed doesn't vary.
    # -----------------------
    all_cols_lp = ["model", "relationship", "condition", "prompt_type", "meta_prompt_type"]
    all_lp = summarize_cells_from_row_draws(
        df_lp.reset_index(drop=True),
        row_probs_lp,
        all_cols_lp,
        interval_prob=args.interval_prob,
        equal_weight_models=False,
    )
    all_lp.insert(0, "measure", "sum logprob")
    all_lp_path = out_dir / f"all_logprob_cell_means__{ts}.csv"
    all_lp.to_csv(all_lp_path, index=False)

    # Raw sanity-check summaries (optional)
    if args.also_write_raw:
        raw_a1_lp = summarize_raw_cells(a1_lp, a1_cols, interval_prob=args.interval_prob)
        raw_a1_lp.insert(0, "measure", "sum logprob")
        raw_a1_embed = summarize_raw_cells(a1_embed, a1_cols, interval_prob=args.interval_prob)
        raw_a1_embed.insert(0, "measure", "embed sim")
        raw_a1 = pd.concat([raw_a1_lp, raw_a1_embed], ignore_index=True)
        raw_a1.to_csv(out_dir / f"analysis1_raw_cell_means__{ts}.csv", index=False)

        raw_a2 = summarize_raw_cells(df_lp, a2_cols, interval_prob=args.interval_prob)
        raw_a2.insert(0, "measure", "sum logprob")
        raw_a2.to_csv(out_dir / f"analysis2_raw_cell_means__{ts}.csv", index=False)

        raw_a3_lp = summarize_raw_cells(a3_lp, a3_cols, interval_prob=args.interval_prob)
        raw_a3_lp.insert(0, "measure", "sum logprob")
        raw_a3_embed = summarize_raw_cells(a3_embed, a3_cols, interval_prob=args.interval_prob)
        raw_a3_embed.insert(0, "measure", "embed sim")
        raw_a3 = pd.concat([raw_a3_lp, raw_a3_embed], ignore_index=True)
        raw_a3.to_csv(out_dir / f"analysis3_raw_cell_means__{ts}.csv", index=False)

    print("Wrote:")
    print(" ", analysis1_path)
    print(" ", analysis2_path)
    print(" ", analysis3_path)
    print(" ", all_lp_path)
    if args.also_write_raw:
        print(" (and raw sanity-check cell means files)")

def compute_contrast(args):
    stamp = timestamp()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(args.data)
    # df = df[~df["model"].isin(["olmo3_7b_base", "olmo3_7b_instruct"])]
    df = set_reference_levels(df)

    # Split and recreate grouping vars exactly once (then reset indices!)
    df_embed, df_lp = split_df(df)

    df_lp = add_grouping_vars_like_fit(df_lp, for_logprob=True)
    df_lp = set_reference_levels(df_lp)
    df_lp = df_lp.reset_index(drop=True)

    df_embed = add_grouping_vars_like_fit(df_embed, for_logprob=False)
    df_embed = set_reference_levels(df_embed)
    df_embed = collapse_embed_rows_like_fit_contrast(df_embed)
    df_embed = df_embed.reset_index(drop=True)

    # Build Bambi models (no fitting here; we load idata)
    # These formulas MUST match what you fit.
    FORMULA_EMBED = (
        "acc ~ relationship * condition + relationship * model + condition * model"
        " + (1|probe_group) + (1|target) + (1|comparison)"
    )
    FORMULA_LP = (
        "acc ~ relationship * condition + relationship * model + condition * model"
        " + meta_prompt_type * prompt_type"
        " + relationship:meta_prompt_type + relationship:prompt_type"
        " + (1|probe_group) + (1|prompt_group) + (1|target) + (1|comparison)"
    )

    m_embed = bmb.Model(FORMULA_EMBED, df_embed, family="bernoulli")
    m_lp = bmb.Model(FORMULA_LP, df_lp, family="bernoulli")

    # Load posterior draws
    idata_embed = az.from_netcdf(args.nc_embed)
    idata_lp = az.from_netcdf(args.nc_lp)

    # Predict row-level probabilities ONCE
    print("Predicting row-level probabilities (logprob model)...")
    lp_row_draws = predict_row_probs_as_draws(m_lp, idata_lp, df_lp)  # (draws, n_lp_rows)
    print("Predicting row-level probabilities (embed model)...")
    embed_row_draws = predict_row_probs_as_draws(m_embed, idata_embed, df_embed)  # (draws, n_embed_rows)

    # Convenience masks
    def lp_mask(**kw) -> np.ndarray:
        m = np.ones(len(df_lp), dtype=bool)
        for k, v in kw.items():
            m &= (df_lp[k].astype(str) == str(v))
        return m

    def emb_mask(**kw) -> np.ndarray:
        m = np.ones(len(df_embed), dtype=bool)
        for k, v in kw.items():
            m &= (df_embed[k].astype(str) == str(v))
        return m

    # -------------------------
    # Build contrast lists
    # -------------------------

    specs_a1: list[ContrastSpec] = []
    specs_a2: list[ContrastSpec] = []
    specs_a3a: list[ContrastSpec] = []
    specs_a3b: list[ContrastSpec] = []

    # Shared restrictions for analyses 1 and 3: task-specific prompt & metaprompt
    lp_best = dict(prompt_type="task specific", meta_prompt_type="task specific")

    # ---- Analysis 1 (avg over model): 8 contrasts, within-measure only
    def add_a1(measure: str, side_a: dict, side_b: dict, label_a: str, label_b: str):
        if measure == "logprob":
            ma = lp_mask(**lp_best, **side_a)
            mb = lp_mask(**lp_best, **side_b)
        else:
            # embed sim does not depend on prompt, but we keep the same filters for parity if present
            ma = emb_mask(**side_a)
            mb = emb_mask(**side_b)
        specs_a1.append(ContrastSpec("analysis1", "avg", measure, label_a, label_b, ma, mb))

    # logprob
    add_a1("logprob", dict(relationship="cohyponym", condition="easy"), dict(relationship="cohyponym", condition="hard"),
           "cohyponym/easy", "cohyponym/hard")
    add_a1("logprob", dict(relationship="cohyponym", condition="easy"), dict(relationship="superordinate", condition="easy"),
           "cohyponym/easy", "superordinate/easy")
    add_a1("logprob", dict(relationship="cohyponym", condition="hard"), dict(relationship="superordinate", condition="hard"),
           "cohyponym/hard", "superordinate/hard")
    add_a1("logprob", dict(relationship="superordinate", condition="easy"), dict(relationship="superordinate", condition="hard"),
           "superordinate/easy", "superordinate/hard")

    # embed
    add_a1("embed", dict(relationship="cohyponym", condition="easy"), dict(relationship="cohyponym", condition="hard"),
           "cohyponym/easy", "cohyponym/hard")
    add_a1("embed", dict(relationship="cohyponym", condition="easy"), dict(relationship="superordinate", condition="easy"),
           "cohyponym/easy", "superordinate/easy")
    add_a1("embed", dict(relationship="cohyponym", condition="hard"), dict(relationship="superordinate", condition="hard"),
           "cohyponym/hard", "superordinate/hard")
    add_a1("embed", dict(relationship="superordinate", condition="easy"), dict(relationship="superordinate", condition="hard"),
           "superordinate/easy", "superordinate/hard")

    # ---- Analysis 2 (avg over model): 28 contrasts within each relationship x condition block
    rel_cond_blocks = [(r, c) for r in ["cohyponym", "superordinate"] for c in ["easy", "hard"]]
    meta_levels = ["none", "neutral", "task specific"]
    prompt_levels = ["control", "task specific"]

    for rel, cond in rel_cond_blocks:
        base = dict(relationship=rel, condition=cond)

        # 1-3: control vs task-specific prompt within each metaprompt
        for meta in meta_levels:
            a = lp_mask(**base, meta_prompt_type=meta, prompt_type="control")
            b = lp_mask(**base, meta_prompt_type=meta, prompt_type="task specific")
            specs_a2.append(ContrastSpec(
                "analysis2", "avg", "logprob",
                f"{rel}/{cond} meta={meta} prompt=control",
                f"{rel}/{cond} meta={meta} prompt=task specific",
                a, b
            ))

        # 4-5: meta none vs neutral within each prompt_type
        for prompt in prompt_levels:
            a = lp_mask(**base, meta_prompt_type="none", prompt_type=prompt)
            b = lp_mask(**base, meta_prompt_type="neutral", prompt_type=prompt)
            specs_a2.append(ContrastSpec(
                "analysis2", "avg", "logprob",
                f"{rel}/{cond} prompt={prompt} meta=none",
                f"{rel}/{cond} prompt={prompt} meta=neutral",
                a, b
            ))

        # 6-7: meta neutral vs task-specific within each prompt_type
        for prompt in prompt_levels:
            a = lp_mask(**base, meta_prompt_type="neutral", prompt_type=prompt)
            b = lp_mask(**base, meta_prompt_type="task specific", prompt_type=prompt)
            specs_a2.append(ContrastSpec(
                "analysis2", "avg", "logprob",
                f"{rel}/{cond} prompt={prompt} meta=neutral",
                f"{rel}/{cond} prompt={prompt} meta=task specific",
                a, b
            ))

    # ---- Analysis 3a: replicate analysis 1 within each model (8 * 5 = 40)
    models = ["gpt2", "llama3.1_8b_base", "llama3.1_8b_instruct", "qwen3_8b_base", "qwen3_8b_instruct"]
    for mdl in models:
        # logprob (best prompting)
        specs_a3a.extend([
            ContrastSpec("analysis3a", mdl, "logprob", "cohyponym/easy", "cohyponym/hard",
                         lp_mask(model=mdl, **lp_best, relationship="cohyponym", condition="easy"),
                         lp_mask(model=mdl, **lp_best, relationship="cohyponym", condition="hard")),
            ContrastSpec("analysis3a", mdl, "logprob", "cohyponym/easy", "superordinate/easy",
                         lp_mask(model=mdl, **lp_best, relationship="cohyponym", condition="easy"),
                         lp_mask(model=mdl, **lp_best, relationship="superordinate", condition="easy")),
            ContrastSpec("analysis3a", mdl, "logprob", "cohyponym/hard", "superordinate/hard",
                         lp_mask(model=mdl, **lp_best, relationship="cohyponym", condition="hard"),
                         lp_mask(model=mdl, **lp_best, relationship="superordinate", condition="hard")),
            ContrastSpec("analysis3a", mdl, "logprob", "superordinate/easy", "superordinate/hard",
                         lp_mask(model=mdl, **lp_best, relationship="superordinate", condition="easy"),
                         lp_mask(model=mdl, **lp_best, relationship="superordinate", condition="hard")),
        ])

        # embed (no prompt vars in df_embed after collapse; just condition on model/rel/cond)
        specs_a3a.extend([
            ContrastSpec("analysis3a", mdl, "embed", "cohyponym/easy", "cohyponym/hard",
                         emb_mask(model=mdl, relationship="cohyponym", condition="easy"),
                         emb_mask(model=mdl, relationship="cohyponym", condition="hard")),
            ContrastSpec("analysis3a", mdl, "embed", "cohyponym/easy", "superordinate/easy",
                         emb_mask(model=mdl, relationship="cohyponym", condition="easy"),
                         emb_mask(model=mdl, relationship="superordinate", condition="easy")),
            ContrastSpec("analysis3a", mdl, "embed", "cohyponym/hard", "superordinate/hard",
                         emb_mask(model=mdl, relationship="cohyponym", condition="hard"),
                         emb_mask(model=mdl, relationship="superordinate", condition="hard")),
            ContrastSpec("analysis3a", mdl, "embed", "superordinate/easy", "superordinate/hard",
                         emb_mask(model=mdl, relationship="superordinate", condition="easy"),
                         emb_mask(model=mdl, relationship="superordinate", condition="hard")),
        ])

    # ---- Analysis 3b: planned model comparisons within each (relationship x condition x measure) cell
    model_contrasts = [
        ("gpt2", "llama3.1_8b_base"),
        ("gpt2", "qwen3_8b_base"),
        ("qwen3_8b_base", "llama3.1_8b_base"),
        ("qwen3_8b_instruct", "llama3.1_8b_instruct"),
        ("qwen3_8b_base", "qwen3_8b_instruct"),
        ("llama3.1_8b_base", "llama3.1_8b_instruct"),
    ]
    cells = [(rel, cond) for rel in ["superordinate", "cohyponym"] for cond in ["easy", "hard"]]

    for (mA, mB) in model_contrasts:
        for rel, cond in cells:
            # logprob cell (best prompting)
            a = lp_mask(model=mA, **lp_best, relationship=rel, condition=cond)
            b = lp_mask(model=mB, **lp_best, relationship=rel, condition=cond)
            specs_a3b.append(ContrastSpec(
                "analysis3b", f"{mA} - {mB}", "logprob",
                f"{mA} {rel}/{cond}", f"{mB} {rel}/{cond}",
                a, b
            ))

            # embed cell
            a = emb_mask(model=mA, relationship=rel, condition=cond)
            b = emb_mask(model=mB, relationship=rel, condition=cond)
            specs_a3b.append(ContrastSpec(
                "analysis3b", f"{mA} - {mB}", "embed",
                f"{mA} {rel}/{cond}", f"{mB} {rel}/{cond}",
                a, b
            ))

    # -------------------------
    # Compute and save
    # -------------------------

    def run_specs(specs: list[ContrastSpec]) -> pd.DataFrame:
        rows = []
        for s in specs:
            if s.measure == "logprob":
                a_draw = mean_prob_over_rows(lp_row_draws, s.mask_a)
                b_draw = mean_prob_over_rows(lp_row_draws, s.mask_b)
            else:
                a_draw = mean_prob_over_rows(embed_row_draws, s.mask_a)
                b_draw = mean_prob_over_rows(embed_row_draws, s.mask_b)

            delta = a_draw - b_draw  # probability difference, range [-1, 1]
            summ = summarize_delta(delta, ci_prob=args.interval_prob, interval_kind=args.interval_kind)

            rows.append(dict(
                analysis=s.analysis,
                model=s.model,
                measure=s.measure,
                side_a=s.side_a,
                side_b=s.side_b,
                ci_prob=args.interval_prob,
                interval_kind=args.interval_kind,
                **summ,
            ))
        return pd.DataFrame(rows)

    df_a1 = run_specs(specs_a1)
    df_a2 = run_specs(specs_a2)
    df_a3a = run_specs(specs_a3a)
    df_a3b = run_specs(specs_a3b)

    out_a1 = out_dir / f"analysis1_contrasts__{stamp}.csv"
    out_a2 = out_dir / f"analysis2_contrasts__{stamp}.csv"
    out_a3a = out_dir / f"analysis3a_contrasts__{stamp}.csv"
    out_a3b = out_dir / f"analysis3b_contrasts__{stamp}.csv"

    df_a1.to_csv(out_a1, index=False)
    df_a2.to_csv(out_a2, index=False)
    df_a3a.to_csv(out_a3a, index=False)
    df_a3b.to_csv(out_a3b, index=False)

    print("Wrote:")
    print(f"  {out_a1}")
    print(f"  {out_a2}")
    print(f"  {out_a3a}")
    print(f"  {out_a3b}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./summary/accuracy_melted.csv")
    ap.add_argument("--nc_embed", default="./mem_analysis/mem_embed_v1__20260314_191013.nc")
    ap.add_argument("--nc_lp", default="./mem_analysis/mem_logprob_v1__20260314_231128.nc")
    ap.add_argument("--out_dir", default="./outputs")
    ap.add_argument("--interval_prob", type=float, default=0.94)
    ap.add_argument("--interval_kind", type=str, default="hdi", choices=["hdi", "eti"])
    ap.add_argument("--include_group_specific", action="store_true",
                    help="Include random intercepts in row predictions (recommended for matching raw row averages).")
    ap.add_argument("--equal_weight_models", action="store_true",
                    help="When marginalizing over model, average cell means equally across models.")
    ap.add_argument("--also_write_raw", action="store_true",
                    help="Also write raw observed cell means of acc (sanity check).")
    args = ap.parse_args()

    # predict_average_mean(args)
    compute_contrast(args)

    

    


if __name__ == "__main__":
    main()