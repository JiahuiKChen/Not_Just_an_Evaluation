#!/usr/bin/env python3
"""Build pooled hyper-sensitive generation sets for particle localization."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from localization.common import (
    DEFAULT_MODEL_NAME,
    DEFAULT_PARTICLES,
    DEFAULT_POOL_NUM_FOLDS,
    DEFAULT_POOL_TOP_K,
    assign_group_folds,
    build_prompt_columns,
    default_results_dir,
    find_generation_candidates_path,
    load_particle_source_rows,
    normalize_particle_name,
    particle_data_path,
)

SELECTION_MODE_TOP_POSITIVE_K = "top_positive_k"
SELECTION_MODE_ALL_P_GENERATIONS = "all_p_generations"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build pooled hyper-sensitive generation sets for particle localization."
    )
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME, help="Hugging Face model name.")
    parser.add_argument(
        "--particles",
        nargs="+",
        default=list(DEFAULT_PARTICLES),
        help="Particles to process.",
    )
    parser.add_argument(
        "--results_dir",
        default=str(REPO_ROOT / "results" / "experiment2"),
        help="Root experiment-2 results directory used to locate per-model candidate TSVs.",
    )
    parser.add_argument(
        "--candidate_path",
        default=None,
        help=(
            "Optional explicit candidate TSV. This may be a consolidated file containing "
            "multiple particles; rows are filtered by --particles."
        ),
    )
    parser.add_argument(
        "--candidate_id_prefix",
        default=None,
        help=(
            "Optional source-id prefix applied after particle filtering, for example "
            "'just_e_' to isolate exclusive-just items in a consolidated candidate TSV."
        ),
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=DEFAULT_POOL_TOP_K,
        help="Maximum number of positive-advantage generations to retain per particle.",
    )
    parser.add_argument(
        "--num_folds",
        type=int,
        default=DEFAULT_POOL_NUM_FOLDS,
        help="Number of cross-validation folds, assigned by source_row_index.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Seed used for deterministic fold assignment.",
    )
    parser.add_argument(
        "--selection_mode",
        choices=[SELECTION_MODE_TOP_POSITIVE_K, SELECTION_MODE_ALL_P_GENERATIONS],
        default=SELECTION_MODE_TOP_POSITIVE_K,
        help="How to build the per-particle generation pool.",
    )
    parser.add_argument(
        "--output_root",
        default=None,
        help="Optional output root. When omitted, uses results/experiment3/<model>/<particle>/.",
    )
    return parser.parse_args()


def resolve_run_dir(model_name: str, particle: str, output_root: str | None) -> Path:
    """Resolve the output directory for one particle."""
    if output_root:
        return Path(output_root) / particle
    return default_results_dir(model_name, dataset_name=particle)


def validate_joined_columns(df: pd.DataFrame) -> None:
    """Ensure the candidate TSV and source TSV agree on shared metadata."""
    column_pairs = (
        ("context", "source_context"),
        ("w_word", "source_w_word"),
        ("wo_word", "source_wo_word"),
    )
    mismatches = []
    for left_column, right_column in column_pairs:
        if left_column not in df.columns or right_column not in df.columns:
            continue
        mismatch_mask = df[right_column].notna() & (df[left_column].astype(str) != df[right_column].astype(str))
        if mismatch_mask.any():
            mismatches.append(f"{left_column} vs {right_column}: {int(mismatch_mask.sum())}")
    if mismatches:
        raise ValueError("Source metadata did not match candidate TSV values: " + ", ".join(mismatches))


def summarize_pool(
    selected_df: pd.DataFrame,
    particle: str,
    candidate_path: Path,
    candidate_id_prefix: str | None,
    source_data_path: str,
    selection_mode: str,
    positive_count: int,
    positive_source_count: int,
    requested_top_k: int,
    num_folds: int,
) -> pd.DataFrame:
    """Create a one-row summary table for one particle pool."""
    record = {
        "particle": particle,
        "candidate_path": str(candidate_path),
        "candidate_id_prefix": candidate_id_prefix or "",
        "source_data_path": source_data_path,
        "selection_mode": selection_mode,
        "selection_metric": "own_context_log_prob_advantage_per_token",
        "selection_requires_positive": selection_mode == SELECTION_MODE_TOP_POSITIVE_K,
        "requested_top_k": requested_top_k,
        "selected_generation_count": int(len(selected_df)),
        "selected_source_item_count": int(selected_df["source_row_index"].nunique()),
        "positive_generation_count": int(positive_count),
        "positive_source_item_count": int(positive_source_count),
        "score_min": float(selected_df["own_context_log_prob_advantage_per_token"].min()),
        "score_max": float(selected_df["own_context_log_prob_advantage_per_token"].max()),
        "score_mean": float(selected_df["own_context_log_prob_advantage_per_token"].mean()),
        "num_folds": int(num_folds),
    }
    for fold_id in sorted(selected_df["fold_id"].unique()):
        fold_mask = selected_df["fold_id"] == fold_id
        record[f"fold{fold_id}_generation_count"] = int(fold_mask.sum())
        record[f"fold{fold_id}_source_item_count"] = int(selected_df.loc[fold_mask, "source_row_index"].nunique())
    return pd.DataFrame([record])


def select_candidate_rows(
    relevant_df: pd.DataFrame,
    selection_mode: str,
    top_k: int,
    particle: str,
    candidate_path: Path,
) -> pd.DataFrame:
    """Select candidate rows according to the requested pool-building mode."""
    positive_df = relevant_df.loc[
        pd.to_numeric(relevant_df["own_context_log_prob_advantage_per_token"], errors="coerce") > 0
    ].copy()
    positive_df["own_context_log_prob_advantage_per_token"] = pd.to_numeric(
        positive_df["own_context_log_prob_advantage_per_token"],
        errors="coerce",
    )
    positive_df["own_context_log_prob_advantage"] = pd.to_numeric(
        positive_df["own_context_log_prob_advantage"],
        errors="coerce",
    )

    if selection_mode == SELECTION_MODE_ALL_P_GENERATIONS:
        selected_df = relevant_df.copy()
        if selected_df.empty:
            raise ValueError(f"No P generations found for {particle} in {candidate_path}")
        return selected_df

    positive_df.sort_values(
        ["own_context_log_prob_advantage_per_token", "own_context_log_prob_advantage"],
        ascending=[False, False],
        inplace=True,
        kind="mergesort",
    )
    selected_df = positive_df.head(top_k).copy()
    if selected_df.empty:
        raise ValueError(f"No positive-advantage P generations found for {particle} in {candidate_path}")
    return selected_df


def validate_all_source_prompts_covered(
    candidate_particle_df: pd.DataFrame,
    relevant_df: pd.DataFrame,
    particle: str,
    candidate_path: Path,
) -> None:
    """Ensure every particle-specific prompt in the candidate TSV has at least one P generation row."""
    source_rows = pd.Index(candidate_particle_df["source_row_index"].drop_duplicates().to_numpy(dtype=int))
    observed_rows = pd.Index(relevant_df["source_row_index"].drop_duplicates().to_numpy(dtype=int))
    missing_rows = source_rows.difference(observed_rows)
    if len(missing_rows) > 0:
        raise ValueError(
            f"{candidate_path} is missing P generations for {particle} source_row_index values: "
            + ", ".join(str(value) for value in missing_rows.tolist())
        )


def build_particle_pool(
    model_name: str,
    particle: str,
    results_dir: Path,
    candidate_path_override: str | None,
    candidate_id_prefix: str | None,
    top_k: int,
    num_folds: int,
    seed: int,
    selection_mode: str,
    output_root: str | None,
) -> None:
    """Build and save one pooled generation set."""
    candidate_path = (
        Path(candidate_path_override)
        if candidate_path_override
        else find_generation_candidates_path(
            model_name=model_name,
            particle=particle,
            results_dir=results_dir,
        )
    )
    if not candidate_path.exists():
        raise FileNotFoundError(f"Candidate TSV does not exist: {candidate_path}")
    run_dir = resolve_run_dir(model_name, particle, output_root=output_root)
    run_dir.mkdir(parents=True, exist_ok=True)

    candidate_df = pd.read_csv(candidate_path, sep="\t")
    required_columns = {
        "source_row_index",
        "generated_from",
        "word",
        "response",
        "own_context_log_prob_advantage",
        "own_context_log_prob_advantage_per_token",
        "id",
        "context",
        "w_word",
        "wo_word",
    }
    missing = sorted(required_columns - set(candidate_df.columns))
    if missing:
        raise KeyError(f"{candidate_path} is missing required columns: {', '.join(missing)}")

    normalized_particle = normalize_particle_name(particle)
    candidate_particle_df = candidate_df.loc[
        candidate_df["word"].fillna("").astype(str).str.strip().str.lower() == normalized_particle
    ].copy()
    if candidate_id_prefix:
        candidate_particle_df = candidate_particle_df.loc[
            candidate_particle_df["id"].fillna("").astype(str).str.startswith(candidate_id_prefix)
        ].copy()
        if candidate_particle_df.empty:
            raise ValueError(
                f"No {normalized_particle} rows in {candidate_path} matched source-id prefix "
                f"{candidate_id_prefix!r}."
            )
    relevant_df = candidate_particle_df.loc[
        candidate_particle_df["generated_from"].astype(str) == "P"
    ].copy()
    relevant_df["own_context_log_prob_advantage_per_token"] = pd.to_numeric(
        relevant_df["own_context_log_prob_advantage_per_token"],
        errors="coerce",
    )
    relevant_df["own_context_log_prob_advantage"] = pd.to_numeric(
        relevant_df["own_context_log_prob_advantage"],
        errors="coerce",
    )

    embedded_followup_column = "followup_metadata_only"
    has_embedded_followups = (
        embedded_followup_column in candidate_particle_df.columns
        and candidate_particle_df[embedded_followup_column].notna().all()
        and candidate_particle_df[embedded_followup_column].astype(str).str.strip().ne("").all()
    )
    source_data_path = (
        f"{candidate_path}::{embedded_followup_column}"
        if has_embedded_followups
        else particle_data_path(particle)
    )
    source_df = None if has_embedded_followups else load_particle_source_rows(particle, data_path=source_data_path)
    if selection_mode == SELECTION_MODE_ALL_P_GENERATIONS:
        validate_all_source_prompts_covered(
            candidate_particle_df=candidate_particle_df,
            relevant_df=relevant_df,
            particle=particle,
            candidate_path=candidate_path,
        )

    selected_df = select_candidate_rows(
        relevant_df=relevant_df,
        selection_mode=selection_mode,
        top_k=top_k,
        particle=particle,
        candidate_path=candidate_path,
    )
    positive_df = relevant_df.loc[
        pd.to_numeric(relevant_df["own_context_log_prob_advantage_per_token"], errors="coerce") > 0
    ].copy()
    if has_embedded_followups:
        selected_df["followup"] = selected_df[embedded_followup_column].astype(str).str.strip()
    else:
        assert source_df is not None
        source_meta = source_df.loc[
            :,
            ["source_row_index", "id", "context", "w_word", "wo_word", "followup"],
        ].rename(
            columns={
                "source_row_index": "raw_source_row_index",
                "context": "source_context",
                "w_word": "source_w_word",
                "wo_word": "source_wo_word",
                "followup": "source_followup",
            }
        )
        selected_df = selected_df.merge(source_meta, on="id", how="left", validate="many_to_one")
        if selected_df["source_followup"].isna().any():
            raise ValueError(
                f"Failed to recover source followups for {particle}: "
                f"{int(selected_df['source_followup'].isna().sum())} missing rows"
            )
        validate_joined_columns(selected_df)
        selected_df["followup"] = selected_df["source_followup"].astype(str).str.strip()

    selected_df["particle"] = normalized_particle
    selected_df["selection_score_per_token"] = selected_df["own_context_log_prob_advantage_per_token"]
    selected_df["selection_score_raw"] = selected_df["own_context_log_prob_advantage"]
    selected_df.reset_index(drop=True, inplace=True)
    selected_df.insert(0, "pool_row_index", range(len(selected_df)))
    selected_df.insert(1, "pool_rank", np.arange(1, len(selected_df) + 1))
    if selection_mode == SELECTION_MODE_ALL_P_GENERATIONS:
        fold_assignment_df = selected_df.loc[:, ["source_row_index", "id"]].drop_duplicates().copy()
        fold_assignment_df["fold_id"] = assign_group_folds(
            fold_assignment_df["source_row_index"].tolist(),
            num_folds=num_folds,
            seed=seed,
        )
        selected_df = selected_df.merge(
            fold_assignment_df,
            on=["source_row_index", "id"],
            how="left",
            validate="many_to_one",
        )
        if selected_df["fold_id"].isna().any():
            raise ValueError(
                f"Failed to assign folds for {particle}: "
                f"{int(selected_df['fold_id'].isna().sum())} rows were missing fold ids"
            )
        selected_df["fold_id"] = selected_df["fold_id"].astype(int)
    else:
        selected_df["fold_id"] = assign_group_folds(
            selected_df["source_row_index"].tolist(),
            num_folds=num_folds,
            seed=seed,
        )
    selected_df = build_prompt_columns(selected_df)

    helper_columns = [
        "raw_source_row_index",
        "source_context",
        "source_w_word",
        "source_wo_word",
        "source_followup",
    ]
    selected_df.drop(columns=[column for column in helper_columns if column in selected_df.columns], inplace=True)
    selected_df.to_csv(run_dir / "generation_pool.tsv", sep="\t", index=False)

    fold_df = (
        selected_df.loc[:, ["source_row_index", "id", "particle", "fold_id"]]
        .drop_duplicates()
        .sort_values(["fold_id", "source_row_index"])
        .reset_index(drop=True)
    )
    fold_df.to_csv(run_dir / "fold_assignment.tsv", sep="\t", index=False)

    summary_df = summarize_pool(
        selected_df=selected_df,
        particle=normalized_particle,
        candidate_path=candidate_path,
        candidate_id_prefix=candidate_id_prefix,
        source_data_path=source_data_path,
        selection_mode=selection_mode,
        positive_count=len(positive_df),
        positive_source_count=int(positive_df["source_row_index"].nunique()),
        requested_top_k=top_k,
        num_folds=num_folds,
    )
    summary_df.to_csv(run_dir / "pool_summary.tsv", sep="\t", index=False)

    print(f"Saved pool to {run_dir / 'generation_pool.tsv'}")
    if selection_mode == SELECTION_MODE_ALL_P_GENERATIONS:
        print(
            f"{normalized_particle}: selected all {len(selected_df)} P generations "
            f"across {selected_df['source_row_index'].nunique()} source items"
        )
    else:
        print(
            f"{normalized_particle}: selected {len(selected_df)} rows "
            f"from {len(positive_df)} positive P generations across "
            f"{selected_df['source_row_index'].nunique()} source items"
        )


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    for particle in [normalize_particle_name(value) for value in args.particles]:
        build_particle_pool(
            model_name=args.model_name,
            particle=particle,
            results_dir=results_dir,
            candidate_path_override=args.candidate_path,
            candidate_id_prefix=args.candidate_id_prefix,
            top_k=args.top_k,
            num_folds=args.num_folds,
            seed=args.seed,
            selection_mode=args.selection_mode,
            output_root=args.output_root,
        )


if __name__ == "__main__":
    main()
