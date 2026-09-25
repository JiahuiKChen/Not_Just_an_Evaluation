#!/usr/bin/env python3
"""Analyze P/P' sampled-response cross-scoring outputs."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


ITEM_COLS = ["source_dataset", "dataset_role", "source_row_index", "id", "word", "sense"]
ANALYSIS_DELTA_COL = "own_context_log_prob_advantage_per_token"
ANALYSIS_OWN_LOG_PROB_COL = "own_log_prob_per_token"
ANALYSIS_WIN_COL = "own_context_win_per_token"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize generated response cross-scoring results.")
    parser.add_argument("--candidates_path", required=True, help="Candidate TSV from generate_topk_responses.py")
    parser.add_argument(
        "--output_prefix",
        default=None,
        help="Prefix for summary outputs. Defaults to candidates path without _candidates.tsv.",
    )
    return parser.parse_args()


def default_output_prefix(candidates_path: Path) -> Path:
    name = candidates_path.name
    if name.endswith("_candidates.tsv"):
        name = name[: -len("_candidates.tsv")]
    elif name.endswith(".tsv"):
        name = name[:-4]
    return candidates_path.with_name(name)


def prepare_analysis_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    required_cols = [ANALYSIS_DELTA_COL, ANALYSIS_OWN_LOG_PROB_COL]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            "Candidate file is missing token-normalized columns required for analysis: "
            + ", ".join(missing)
        )

    df[ANALYSIS_DELTA_COL] = pd.to_numeric(df[ANALYSIS_DELTA_COL], errors="coerce")
    df[ANALYSIS_OWN_LOG_PROB_COL] = pd.to_numeric(df[ANALYSIS_OWN_LOG_PROB_COL], errors="coerce")
    df[ANALYSIS_WIN_COL] = df[ANALYSIS_DELTA_COL] > 0
    return df


def add_sampling_weights(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    group_cols = ["source_dataset", "source_row_index", "id", "generated_from"]
    group_max = df.groupby(group_cols, dropna=False)[ANALYSIS_OWN_LOG_PROB_COL].transform("max")
    df["candidate_weight_unnormalized"] = (df[ANALYSIS_OWN_LOG_PROB_COL] - group_max).apply(math.exp)
    group_sum = df.groupby(group_cols, dropna=False)["candidate_weight_unnormalized"].transform("sum")
    df["candidate_weight"] = df["candidate_weight_unnormalized"] / group_sum
    df["candidate_weight_basis"] = ANALYSIS_OWN_LOG_PROB_COL
    return df


def summarize_group(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    records = []
    grouped = df.groupby(group_cols, dropna=False)
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = dict(zip(group_cols, keys))
        weights = group["candidate_weight"]
        deltas = group[ANALYSIS_DELTA_COL]
        wins = group[ANALYSIS_WIN_COL].astype(float)
        top1 = group[group["unique_rank"] == group["unique_rank"].min()]
        weighted_delta = (weights * deltas).sum() / weights.sum()
        weighted_win = (weights * wins).sum() / weights.sum()
        record.update(
            {
                "analysis_metric": ANALYSIS_DELTA_COL,
                "n_candidates": len(group),
                "n_items": group[["source_dataset", "source_row_index", "id"]].drop_duplicates().shape[0],
                "mean_candidates_per_item": len(group)
                / max(group[["source_dataset", "source_row_index", "id"]].drop_duplicates().shape[0], 1),
                "mean_own_context_log_prob_advantage": deltas.mean(),
                "median_own_context_log_prob_advantage": deltas.median(),
                "mean_own_context_log_prob_advantage_per_token": deltas.mean(),
                "median_own_context_log_prob_advantage_per_token": deltas.median(),
                "own_context_win_rate": wins.mean(),
                "own_context_win_rate_per_token": wins.mean(),
                "weighted_own_context_win_rate": weighted_win,
                "weighted_own_context_win_rate_per_token": weighted_win,
                "weighted_own_context_log_prob_advantage": weighted_delta,
                "weighted_own_context_log_prob_advantage_per_token": weighted_delta,
                "top1_own_context_win_rate": top1[ANALYSIS_WIN_COL].astype(float).mean(),
                "top1_own_context_win_rate_per_token": top1[ANALYSIS_WIN_COL].astype(float).mean(),
                "top1_mean_own_context_log_prob_advantage": top1[ANALYSIS_DELTA_COL].mean(),
                "top1_mean_own_context_log_prob_advantage_per_token": top1[ANALYSIS_DELTA_COL].mean(),
                "mean_unique_samples_obtained": group["unique_samples_obtained"].mean(),
                "min_unique_samples_obtained": group["unique_samples_obtained"].min(),
                "mean_sample_draw": group["sample_draw"].mean(),
            }
        )
        records.append(record)
    return pd.DataFrame(records)


def item_condition_summary(df: pd.DataFrame) -> pd.DataFrame:
    return summarize_group(df, ITEM_COLS + ["generated_from"])


def paired_item_summary(df: pd.DataFrame) -> pd.DataFrame:
    condition_summary = item_condition_summary(df)
    records = []
    grouped = df.groupby(ITEM_COLS, dropna=False)
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = dict(zip(ITEM_COLS, keys))
        group_p = group[group["generated_from"] == "P"]
        group_p_prime = group[group["generated_from"] == "P_prime"]
        set_p = set(group_p["normalized_response"])
        set_p_prime = set(group_p_prime["normalized_response"])
        union = set_p | set_p_prime
        intersection = set_p & set_p_prime
        record.update(
            {
                "p_candidate_count": len(group_p),
                "p_prime_candidate_count": len(group_p_prime),
                "exact_overlap_count": len(intersection),
                "exact_overlap_jaccard": len(intersection) / len(union) if union else float("nan"),
            }
        )
        for label, source in [("p", "P"), ("p_prime", "P_prime")]:
            one = condition_summary[
                (condition_summary["source_dataset"] == record["source_dataset"])
                & (condition_summary["source_row_index"] == record["source_row_index"])
                & (condition_summary["id"] == record["id"])
                & (condition_summary["generated_from"] == source)
            ]
            if one.empty:
                continue
            one_row = one.iloc[0]
            record[f"{label}_win_rate"] = one_row["own_context_win_rate"]
            record[f"{label}_win_rate_per_token"] = one_row["own_context_win_rate_per_token"]
            record[f"{label}_weighted_win_rate"] = one_row["weighted_own_context_win_rate"]
            record[f"{label}_weighted_win_rate_per_token"] = one_row[
                "weighted_own_context_win_rate_per_token"
            ]
            record[f"{label}_mean_advantage"] = one_row["mean_own_context_log_prob_advantage"]
            record[f"{label}_mean_advantage_per_token"] = one_row[
                "mean_own_context_log_prob_advantage_per_token"
            ]
            record[f"{label}_weighted_advantage"] = one_row[
                "weighted_own_context_log_prob_advantage"
            ]
            record[f"{label}_weighted_advantage_per_token"] = one_row[
                "weighted_own_context_log_prob_advantage_per_token"
            ]
            record[f"{label}_top1_win"] = one_row["top1_own_context_win_rate"]
            record[f"{label}_top1_win_per_token"] = one_row["top1_own_context_win_rate_per_token"]
            record[f"{label}_top1_advantage"] = one_row[
                "top1_mean_own_context_log_prob_advantage"
            ]
            record[f"{label}_top1_advantage_per_token"] = one_row[
                "top1_mean_own_context_log_prob_advantage_per_token"
            ]
        records.append(record)
    paired = pd.DataFrame(records)
    if {"p_mean_advantage_per_token", "p_prime_mean_advantage_per_token"}.issubset(paired.columns):
        paired["both_mean_advantages_positive"] = (
            (paired["p_mean_advantage_per_token"] > 0) & (paired["p_prime_mean_advantage_per_token"] > 0)
        )
    if {"p_top1_win_per_token", "p_prime_top1_win_per_token"}.issubset(paired.columns):
        paired["both_top1_own_context_wins"] = (
            (paired["p_top1_win_per_token"] > 0) & (paired["p_prime_top1_win_per_token"] > 0)
        )
    return paired


def build_text_summary(df: pd.DataFrame, summaries: dict[str, pd.DataFrame]) -> str:
    lines = [
        "=" * 80,
        "GENERATION CROSS-SCORING SUMMARY",
        "=" * 80,
        f"Total candidates: {len(df)}",
        f"Total stimulus items: {df[['source_dataset', 'source_row_index', 'id']].drop_duplicates().shape[0]}",
        f"Sampling strategy: {', '.join(sorted(df['sampling_strategy'].dropna().astype(str).unique()))}",
        f"Top-p values: {', '.join(str(x) for x in sorted(df['top_p'].dropna().unique()))}",
        f"Candidate weight basis: {ANALYSIS_OWN_LOG_PROB_COL}",
        "",
        f"Main metric: {ANALYSIS_DELTA_COL}",
        "  P samples:        mean token log p(response | P) - mean token log p(response | P')",
        "  P' samples:       mean token log p(response | P') - mean token log p(response | P)",
        "  Positive values:  response is more likely under its source context",
        "",
    ]

    condition_summary = summaries["condition"]
    display_cols = [
        "source_dataset",
        "dataset_role",
        "generated_from",
        "n_candidates",
        "n_items",
        "own_context_win_rate",
        "weighted_own_context_win_rate",
        "mean_own_context_log_prob_advantage_per_token",
        "weighted_own_context_log_prob_advantage_per_token",
        "top1_own_context_win_rate",
        "min_unique_samples_obtained",
    ]
    lines.append("--- BY DATASET AND CONDITION ---")
    for _, row in condition_summary[display_cols].iterrows():
        lines.append(
            f"{row['source_dataset']} ({row['dataset_role']}), {row['generated_from']}: "
            f"win_per_token={row['own_context_win_rate']:.3f}, "
            f"weighted_win_per_token={row['weighted_own_context_win_rate']:.3f}, "
            f"mean_adv_per_token={row['mean_own_context_log_prob_advantage_per_token']:.3f}, "
            f"weighted_adv_per_token={row['weighted_own_context_log_prob_advantage_per_token']:.3f}, "
            f"top1_win_per_token={row['top1_own_context_win_rate']:.3f}, "
            f"candidates={int(row['n_candidates'])}, items={int(row['n_items'])}, "
            f"min_unique={int(row['min_unique_samples_obtained'])}"
        )

    paired = summaries["paired_item"]
    if "both_mean_advantages_positive" in paired.columns:
        lines.extend(
            [
                "",
                "--- ITEM-LEVEL PAIRING ---",
                f"Items where both P and P' sampled sets have positive mean own-context advantage per token: "
                f"{int(paired['both_mean_advantages_positive'].sum())}/{len(paired)}",
                f"Mean exact overlap count: {paired['exact_overlap_count'].mean():.2f}",
                f"Mean exact overlap Jaccard: {paired['exact_overlap_jaccard'].mean():.3f}",
            ]
        )

    lines.append("=" * 80)
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    candidates_path = Path(args.candidates_path)
    output_prefix = Path(args.output_prefix) if args.output_prefix else default_output_prefix(candidates_path)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(candidates_path, sep="\t")
    if df.empty:
        raise ValueError(f"No candidates found in {candidates_path}")
    df = prepare_analysis_frame(df)
    df = add_sampling_weights(df)

    summaries = {
        "condition": summarize_group(
            df,
            ["source_dataset", "dataset_role", "word", "generated_from"],
        ),
        "item_condition": item_condition_summary(df),
        "paired_item": paired_item_summary(df),
    }

    weighted_candidates_path = output_prefix.with_name(output_prefix.name + "_candidates_weighted.tsv")
    df.to_csv(weighted_candidates_path, sep="\t", index=False)

    for name, summary_df in summaries.items():
        summary_path = output_prefix.with_name(output_prefix.name + f"_{name}_summary.tsv")
        summary_df.to_csv(summary_path, sep="\t", index=False)
        print(f"Saved {name} summary to: {summary_path}")

    text_summary = build_text_summary(df, summaries)
    text_path = output_prefix.with_name(output_prefix.name + "_summary.txt")
    text_path.write_text(text_summary + "\n")
    print("\n" + text_summary + "\n")
    print(f"Saved weighted candidates to: {weighted_candidates_path}")
    print(f"Saved text summary to: {text_path}")


if __name__ == "__main__":
    main()
