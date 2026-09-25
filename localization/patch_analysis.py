#!/usr/bin/env python3
"""Summaries and statistical analysis for activation-patching evaluations."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

RAW_SCORE_COLUMNS = [
    "baseline_log_prob",
    "reference_log_prob",
    "patched_log_prob",
]
TOKEN_COUNT_COLUMNS = [
    "baseline_token_count",
    "reference_token_count",
    "patched_token_count",
]
PER_TOKEN_SCORE_COLUMNS = [
    "baseline_log_prob_per_token",
    "reference_log_prob_per_token",
    "patched_log_prob_per_token",
]
PRIMARY_METRIC = "effect_score"
COMPARISON_METRICS = [
    "patched_metric_change",
    "effect_score",
    "normalized_effect",
]
ITEM_KEY_COLUMNS = [
    "particle",
    "eval_mode",
    "top_k",
    "train_fold",
    "eval_fold",
    "target_type",
    "source_row_index",
    "id",
]
OPTIONAL_CONTEXT_COLUMNS = [
    "component",
    "position_label",
    "corruption_mode",
]
MEAN_METRIC_COLUMNS = RAW_SCORE_COLUMNS + PER_TOKEN_SCORE_COLUMNS + COMPARISON_METRICS
FOLD_RATE_COLUMNS = [
    "effect_positive_fraction",
    "localized_stronger_than_random_mean_fraction",
    "localized_random_cdf_mean",
]


def _coerce_numeric_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    result = df.copy()
    for column in columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def _finite_array(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def bootstrap_ci(
    values: Sequence[float],
    rng: np.random.Generator,
    replicates: int,
) -> tuple[float, float]:
    """Return a percentile bootstrap confidence interval around the mean."""
    finite = _finite_array(values)
    if len(finite) == 0 or replicates <= 0:
        return (float("nan"), float("nan"))
    sample_idx = rng.integers(0, len(finite), size=(replicates, len(finite)))
    sample_means = finite[sample_idx].mean(axis=1)
    low, high = np.percentile(sample_means, [2.5, 97.5])
    return (float(low), float(high))


def _greater_sign_flip_pvalues(
    values: Sequence[float],
    rng: np.random.Generator,
    replicates: int,
) -> tuple[float, float]:
    """Return one-sided-greater and two-sided sign-flip p-values for a paired mean."""
    finite = _finite_array(values)
    if len(finite) == 0 or replicates <= 0:
        return (float("nan"), float("nan"))
    observed = float(np.mean(finite))
    signs = rng.choice(np.array([-1.0, 1.0], dtype=float), size=(replicates, len(finite)))
    sample_means = (signs * finite[None, :]).mean(axis=1)
    p_greater = float((np.sum(sample_means >= observed) + 1) / (replicates + 1))
    p_two_sided = float((np.sum(np.abs(sample_means) >= abs(observed)) + 1) / (replicates + 1))
    return (p_greater, p_two_sided)


def _numeric_mean(series: pd.Series) -> float:
    finite = _finite_array(pd.to_numeric(series, errors="coerce").to_numpy(dtype=float))
    return float(np.mean(finite)) if len(finite) else float("nan")


def _context_columns(df: pd.DataFrame) -> list[str]:
    """Return optional context columns present in a patching output table."""
    return [column for column in OPTIONAL_CONTEXT_COLUMNS if column in df.columns]


def _item_key_columns(df: pd.DataFrame) -> list[str]:
    """Return item identity columns, including optional context dimensions."""
    base = list(ITEM_KEY_COLUMNS)
    insert_at = 1
    for column in _context_columns(df):
        if column not in base:
            base.insert(insert_at, column)
            insert_at += 1
    return base


def _item_group_columns(df: pd.DataFrame) -> list[str]:
    return _item_key_columns(df) + ["site_set_type", "random_seed"]


def _fold_group_columns(df: pd.DataFrame) -> list[str]:
    context = _context_columns(df)
    return ["particle"] + context + ["eval_mode", "target_type", "top_k", "train_fold", "site_set_type"]


def _comparison_group_columns(df: pd.DataFrame) -> list[str]:
    context = _context_columns(df)
    return ["particle"] + context + ["eval_mode", "target_type", "top_k", "train_fold"]


def _condition_group_columns(df: pd.DataFrame) -> list[str]:
    context = _context_columns(df)
    return ["particle"] + context + ["eval_mode", "target_type", "top_k", "site_set_type"]


def _condition_comparison_columns(df: pd.DataFrame) -> list[str]:
    context = _context_columns(df)
    return ["particle"] + context + ["eval_mode", "target_type", "top_k"]


def _keys_to_record(columns: list[str], keys: object) -> dict:
    if not isinstance(keys, tuple):
        keys = (keys,)
    return dict(zip(columns, keys))


def build_item_summary(row_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate row-level patch metrics so source items receive equal weight."""
    item_group_columns = _item_group_columns(row_df)
    required_columns = set(
        item_group_columns
        + ["evaluation_id", "selected_site_count"]
        + TOKEN_COUNT_COLUMNS
        + MEAN_METRIC_COLUMNS
    )
    missing = sorted(required_columns - set(row_df.columns))
    if missing:
        raise KeyError("Missing columns for patch item summary: " + ", ".join(missing))

    df = _coerce_numeric_columns(
        row_df,
        ["top_k", "train_fold", "eval_fold", "source_row_index", "selected_site_count"]
        + TOKEN_COUNT_COLUMNS
        + MEAN_METRIC_COLUMNS,
    )
    agg_map: dict[str, str] = {
        "evaluation_id": "nunique",
        "selected_site_count": "mean",
    }
    for column in TOKEN_COUNT_COLUMNS + MEAN_METRIC_COLUMNS:
        agg_map[column] = "mean"

    item_df = (
        df.groupby(item_group_columns, dropna=False, sort=False)
        .agg(agg_map)
        .reset_index()
        .rename(columns={"evaluation_id": "n_member_targets"})
    )
    item_df["n_member_targets"] = pd.to_numeric(item_df["n_member_targets"], errors="coerce").fillna(0).astype(int)
    item_df["effect_positive"] = pd.to_numeric(item_df[PRIMARY_METRIC], errors="coerce") > 0
    return item_df


def build_random_mean_item_rows(item_df: pd.DataFrame) -> pd.DataFrame:
    """Average random-seed controls to one item row."""
    random_df = item_df.loc[item_df["site_set_type"] == "random"].copy()
    if random_df.empty:
        return pd.DataFrame(columns=item_df.columns)

    item_key_columns = _item_key_columns(item_df)
    value_columns = ["n_member_targets", "selected_site_count"] + TOKEN_COUNT_COLUMNS + MEAN_METRIC_COLUMNS
    random_mean_df = (
        random_df.groupby(item_key_columns, dropna=False, sort=False)[value_columns]
        .mean()
        .reset_index()
    )
    random_mean_df["n_member_targets"] = (
        pd.to_numeric(random_mean_df["n_member_targets"], errors="coerce").fillna(0).round().astype(int)
    )
    random_mean_df["site_set_type"] = "random_mean"
    random_mean_df["random_seed"] = ""
    random_mean_df["effect_positive"] = pd.to_numeric(random_mean_df[PRIMARY_METRIC], errors="coerce") > 0
    return random_mean_df.reindex(columns=item_df.columns)


def build_item_comparison(item_df: pd.DataFrame) -> pd.DataFrame:
    """Compare localized patching effects against the random-site distribution."""
    localized_df = item_df.loc[item_df["site_set_type"] == "localized"].copy()
    random_df = item_df.loc[item_df["site_set_type"] == "random"].copy()
    if localized_df.empty:
        return pd.DataFrame()
    if random_df.empty:
        return localized_df.rename(columns={"site_set_type": "comparison_source"})

    item_key_columns = _item_key_columns(item_df)
    random_summary_rows = []
    for keys, group in random_df.groupby(item_key_columns, dropna=False, sort=False):
        record = _keys_to_record(item_key_columns, keys)
        record["random_seed_count"] = int(group["random_seed"].astype(str).nunique())
        for metric in COMPARISON_METRICS:
            finite = _finite_array(pd.to_numeric(group[metric], errors="coerce").to_numpy(dtype=float))
            if len(finite) == 0:
                record[f"{metric}_random_mean"] = float("nan")
                record[f"{metric}_random_std"] = float("nan")
                record[f"{metric}_random_cdf_at_localized"] = float("nan")
            else:
                record[f"{metric}_random_mean"] = float(np.mean(finite))
                record[f"{metric}_random_std"] = float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
        random_summary_rows.append(record)
    random_summary_df = pd.DataFrame(random_summary_rows)

    keep_columns = item_key_columns + [
        "n_member_targets",
        "selected_site_count",
        "effect_positive",
    ] + TOKEN_COUNT_COLUMNS + MEAN_METRIC_COLUMNS
    comparison_df = localized_df.loc[:, keep_columns].merge(
        random_summary_df,
        on=item_key_columns,
        how="left",
        validate="one_to_one",
    )

    random_groups = {
        tuple(group_key if isinstance(group_key, tuple) else (group_key,)): group
        for group_key, group in random_df.groupby(item_key_columns, dropna=False, sort=False)
    }
    for metric in COMPARISON_METRICS:
        comparison_df[f"{metric}_localized_minus_random_mean"] = (
            comparison_df[metric] - comparison_df[f"{metric}_random_mean"]
        )
        comparison_df[f"{metric}_localized_random_cdf"] = np.nan
        for idx, row in comparison_df.iterrows():
            key = tuple(row[column] for column in item_key_columns)
            group = random_groups.get(key)
            if group is None or not np.isfinite(row[metric]):
                continue
            finite = _finite_array(pd.to_numeric(group[metric], errors="coerce").to_numpy(dtype=float))
            if len(finite) == 0:
                continue
            comparison_df.at[idx, f"{metric}_localized_random_cdf"] = float(np.mean(finite <= row[metric]))

    comparison_df["localized_stronger_than_random_mean"] = (
        pd.to_numeric(comparison_df[PRIMARY_METRIC], errors="coerce")
        > pd.to_numeric(comparison_df[f"{PRIMARY_METRIC}_random_mean"], errors="coerce")
    )
    comparison_df["localized_random_cdf"] = comparison_df[f"{PRIMARY_METRIC}_localized_random_cdf"]
    return comparison_df


def build_fold_summary(item_df: pd.DataFrame, comparison_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate item-level patch metrics to one row per fold and condition."""
    combined_df = pd.concat([item_df, build_random_mean_item_rows(item_df)], ignore_index=True, sort=False)
    combined_df = combined_df.loc[combined_df["site_set_type"] != "random"].copy()
    fold_group_columns = _fold_group_columns(combined_df)
    rows = []
    for keys, group in combined_df.groupby(fold_group_columns, dropna=False, sort=False):
        record = _keys_to_record(fold_group_columns, keys)
        record["n_items"] = int(len(group))
        record["mean_member_targets"] = _numeric_mean(group["n_member_targets"])
        record["mean_selected_sites"] = _numeric_mean(group["selected_site_count"])
        for metric in MEAN_METRIC_COLUMNS:
            record[f"{metric}_mean"] = _numeric_mean(group[metric])
        record["effect_positive_fraction"] = float(group["effect_positive"].astype(bool).mean())
        record["localized_stronger_than_random_mean_fraction"] = float("nan")
        record["localized_random_cdf_mean"] = float("nan")
        rows.append(record)
    fold_summary_df = pd.DataFrame(rows)

    if comparison_df.empty:
        return fold_summary_df

    comparison_group_columns = _comparison_group_columns(comparison_df)
    comparison_rows = []
    for keys, group in comparison_df.groupby(comparison_group_columns, dropna=False, sort=False):
        record = _keys_to_record(comparison_group_columns, keys)
        record["localized_stronger_than_random_mean_fraction"] = float(
            group["localized_stronger_than_random_mean"].astype(bool).mean()
        )
        record["localized_random_cdf_mean"] = _numeric_mean(group["localized_random_cdf"])
        comparison_rows.append(record)
    comparison_fold_df = pd.DataFrame(comparison_rows)
    if comparison_fold_df.empty:
        return fold_summary_df

    localized_mask = fold_summary_df["site_set_type"] == "localized"
    localized_df = fold_summary_df.loc[localized_mask].merge(
        comparison_fold_df,
        on=comparison_group_columns,
        how="left",
        validate="one_to_one",
        suffixes=("", "_comparison"),
    )
    for column in ["localized_stronger_than_random_mean_fraction", "localized_random_cdf_mean"]:
        comparison_column = f"{column}_comparison"
        if comparison_column in localized_df.columns:
            localized_df[column] = localized_df[comparison_column]
            localized_df.drop(columns=[comparison_column], inplace=True)
    other_df = fold_summary_df.loc[~localized_mask].copy()
    return pd.concat([other_df, localized_df], ignore_index=True, sort=False).sort_values(fold_group_columns)


def build_condition_summary(
    fold_summary_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    rng: np.random.Generator,
    bootstrap_replicates: int,
) -> pd.DataFrame:
    """Aggregate fold summaries and attach uncertainty estimates."""
    rows = []
    columns_to_summarize = [f"{metric}_mean" for metric in MEAN_METRIC_COLUMNS] + FOLD_RATE_COLUMNS
    condition_group_columns = _condition_group_columns(fold_summary_df)
    for keys, group in fold_summary_df.groupby(condition_group_columns, dropna=False, sort=False):
        record = _keys_to_record(condition_group_columns, keys)
        fold_ids = sorted(int(value) for value in pd.unique(group["train_fold"]))
        record["n_folds"] = int(len(group))
        record["train_folds"] = ",".join(str(value) for value in fold_ids)
        record["mean_items_per_fold"] = _numeric_mean(group["n_items"])
        record["mean_member_targets_per_item"] = _numeric_mean(group["mean_member_targets"])
        record["mean_selected_sites"] = _numeric_mean(group["mean_selected_sites"])
        for column in columns_to_summarize:
            finite = _finite_array(pd.to_numeric(group[column], errors="coerce").to_numpy(dtype=float))
            record[column] = float(np.mean(finite)) if len(finite) else float("nan")
            record[f"{column}_sd"] = float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan")
            ci_low, ci_high = bootstrap_ci(finite, rng, bootstrap_replicates)
            record[f"{column}_ci_low"] = ci_low
            record[f"{column}_ci_high"] = ci_high
        rows.append(record)
    summary_df = pd.DataFrame(rows)
    if summary_df.empty or comparison_df.empty:
        return summary_df

    condition_comparison_columns = _condition_comparison_columns(comparison_df)
    comparison_rows = []
    for keys, group in comparison_df.groupby(condition_comparison_columns, dropna=False, sort=False):
        record = _keys_to_record(condition_comparison_columns, keys)
        record["localized_stronger_than_random_mean_fraction_items"] = float(
            group["localized_stronger_than_random_mean"].astype(bool).mean()
        )
        for metric in COMPARISON_METRICS:
            diff_column = f"{metric}_localized_minus_random_mean"
            record[f"{diff_column}_mean"] = _numeric_mean(group[diff_column])
        comparison_rows.append(record)
    comparison_summary_df = pd.DataFrame(comparison_rows)
    if comparison_summary_df.empty:
        return summary_df

    localized_mask = summary_df["site_set_type"] == "localized"
    localized_df = summary_df.loc[localized_mask].merge(
        comparison_summary_df,
        on=condition_comparison_columns,
        how="left",
        validate="one_to_one",
    )
    other_df = summary_df.loc[~localized_mask].copy()
    return pd.concat([other_df, localized_df], ignore_index=True, sort=False).sort_values(condition_group_columns)


def build_significance_table(
    comparison_df: pd.DataFrame,
    rng: np.random.Generator,
    bootstrap_replicates: int,
    signflip_replicates: int,
) -> pd.DataFrame:
    """Test whether localized site sets outperform the random-site mean."""
    if comparison_df.empty:
        return pd.DataFrame()

    condition_comparison_columns = _condition_comparison_columns(comparison_df)
    rows = []
    for metric in COMPARISON_METRICS:
        diff_column = f"{metric}_localized_minus_random_mean"
        for keys, group in comparison_df.groupby(condition_comparison_columns, dropna=False, sort=False):
            record = _keys_to_record(condition_comparison_columns, keys)
            finite = _finite_array(pd.to_numeric(group[diff_column], errors="coerce").to_numpy(dtype=float))
            record["metric"] = metric
            record["n_item_units"] = int(len(finite))
            if len(finite) == 0:
                record["mean_difference"] = float("nan")
                record["median_difference"] = float("nan")
                record["difference_sd"] = float("nan")
                record["effect_size_dz"] = float("nan")
                record["ci_low"] = float("nan")
                record["ci_high"] = float("nan")
                record["p_value_greater"] = float("nan")
                record["p_value_two_sided"] = float("nan")
            else:
                record["mean_difference"] = float(np.mean(finite))
                record["median_difference"] = float(np.median(finite))
                record["difference_sd"] = float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
                record["effect_size_dz"] = (
                    record["mean_difference"] / record["difference_sd"]
                    if len(finite) > 1 and record["difference_sd"] > 0
                    else float("nan")
                )
                ci_low, ci_high = bootstrap_ci(finite, rng, bootstrap_replicates)
                record["ci_low"] = ci_low
                record["ci_high"] = ci_high
                p_greater, p_two_sided = _greater_sign_flip_pvalues(finite, rng, signflip_replicates)
                record["p_value_greater"] = p_greater
                record["p_value_two_sided"] = p_two_sided
            rows.append(record)
    return pd.DataFrame(rows).sort_values(condition_comparison_columns + ["metric"])


def build_text_summary(condition_summary_df: pd.DataFrame, significance_df: pd.DataFrame) -> str:
    """Create a compact human-readable summary of the key patching findings."""
    lines = [
        "=" * 80,
        "PATCHING EVALUATION SUMMARY",
        "=" * 80,
    ]
    if condition_summary_df.empty:
        lines.append("No patching summary rows were available.")
        lines.append("=" * 80)
        return "\n".join(lines)

    display_sort_columns = _condition_group_columns(condition_summary_df)
    display = condition_summary_df.sort_values(display_sort_columns)
    for _, row in display.iterrows():
        context_bits = []
        for column in OPTIONAL_CONTEXT_COLUMNS:
            if column in row.index and not pd.isna(row[column]):
                context_bits.append(str(row[column]))
        context_prefix = " | ".join(context_bits)
        if context_prefix:
            context_prefix = f" | {context_prefix}"
        lines.append(
            f"{row['particle']}{context_prefix} | {row['eval_mode']} | {row['target_type']} | "
            f"top_k={int(row['top_k'])} | "
            f"{row['site_set_type']}: effect={row.get('effect_score_mean', float('nan')):.4f}, "
            f"delta={row.get('patched_metric_change_mean', float('nan')):.4f}, "
            f"normalized={row.get('normalized_effect_mean', float('nan')):.4f}"
        )
        if row["site_set_type"] == "localized":
            localized_minus_random = row.get("effect_score_localized_minus_random_mean_mean", float("nan"))
            stronger_than_random = row.get("localized_stronger_than_random_mean_fraction_items", float("nan"))
            if np.isfinite(localized_minus_random):
                lines.append(
                    f"  localized-minus-random mean={localized_minus_random:.4f}, "
                    f"stronger_fraction={stronger_than_random:.3f}"
                )

    if not significance_df.empty:
        lines.append("")
        lines.append("--- Localized vs Random Mean Tests ---")
        primary_rows = significance_df.loc[significance_df["metric"] == PRIMARY_METRIC].copy()
        primary_rows.sort_values(_condition_comparison_columns(primary_rows), inplace=True)
        for _, row in primary_rows.iterrows():
            context_bits = []
            for column in OPTIONAL_CONTEXT_COLUMNS:
                if column in row.index and not pd.isna(row[column]):
                    context_bits.append(str(row[column]))
            context_prefix = " | ".join(context_bits)
            if context_prefix:
                context_prefix = f" | {context_prefix}"
            lines.append(
                f"{row['particle']}{context_prefix} | {row['eval_mode']} | {row['target_type']} | "
                f"top_k={int(row['top_k'])}: "
                f"mean_diff={row['mean_difference']:.4f}, "
                f"95% CI [{row['ci_low']:.4f}, {row['ci_high']:.4f}], "
                f"p_greater={row['p_value_greater']:.4f}, "
                f"n={int(row['n_item_units'])}"
            )

    lines.append("=" * 80)
    return "\n".join(lines)


def write_patch_analysis_outputs(
    patch_dir: Path,
    row_df: pd.DataFrame,
    bootstrap_replicates: int = 5000,
    bootstrap_seed: int = 0,
    signflip_replicates: int = 20000,
) -> dict[str, pd.DataFrame | str]:
    """Write item, fold, summary, significance, and text outputs for patching runs."""
    rng = np.random.default_rng(bootstrap_seed)
    item_df = build_item_summary(row_df)
    comparison_df = build_item_comparison(item_df)
    fold_summary_df = build_fold_summary(item_df, comparison_df)
    condition_summary_df = build_condition_summary(
        fold_summary_df=fold_summary_df,
        comparison_df=comparison_df,
        rng=rng,
        bootstrap_replicates=bootstrap_replicates,
    )
    significance_df = build_significance_table(
        comparison_df=comparison_df,
        rng=rng,
        bootstrap_replicates=bootstrap_replicates,
        signflip_replicates=signflip_replicates,
    )
    text_summary = build_text_summary(condition_summary_df, significance_df)

    outputs: dict[str, pd.DataFrame | str] = {
        "item": item_df,
        "item_comparison": comparison_df,
        "fold": fold_summary_df,
        "summary": condition_summary_df,
        "significance": significance_df,
        "text": text_summary,
    }
    output_paths = {
        "item": patch_dir / "eval_item_summary.tsv",
        "item_comparison": patch_dir / "eval_item_comparison.tsv",
        "fold": patch_dir / "eval_fold_summary.tsv",
        "summary": patch_dir / "eval_summary.tsv",
        "significance": patch_dir / "eval_significance.tsv",
    }
    for key, path in output_paths.items():
        value = outputs[key]
        assert isinstance(value, pd.DataFrame)
        value.to_csv(path, sep="\t", index=False)
    (patch_dir / "eval_summary.txt").write_text(text_summary + "\n", encoding="utf-8")
    return outputs
