#!/usr/bin/env python3
"""Localize component-level prompt-boundary patching sites for discourse particles."""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from localization.common import (
    DEFAULT_MODEL_NAME,
    build_encoded_sequence_batch,
    build_residual_patch_plan,
    default_results_dir,
    get_component_modules,
    get_device,
    load_model,
    model_short_name,
    normalize_particle_name,
    prepare_prompt_columns_for_model,
    resolve_utterance_final_lexical_positions,
    resolve_utterance_final_positions,
    run_component_sequence_model,
    score_target_log_probs_from_logits,
)

COMPONENTS = ("resid", "attn", "mlp")
CORRUPTION_MODE = "prompt_without"
POSITION_LABEL_PROMPT_BOUNDARY = "prompt_boundary"
POSITION_LABEL_UTTERANCE_FINAL = "utterance_final"
POSITION_LABEL_UTTERANCE_FINAL_LEXICAL = "utterance_final_lexical"
POSITION_LABELS = (
    POSITION_LABEL_PROMPT_BOUNDARY,
    POSITION_LABEL_UTTERANCE_FINAL,
    POSITION_LABEL_UTTERANCE_FINAL_LEXICAL,
)
POSITION_ORDER = 0
PATCH_DIR_NAME = "component_patching"
SUMMARY_FILES = ("generation_pool.tsv", "fold_assignment.tsv", "pool_summary.tsv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Localize component-level prompt-boundary sites for particle-sensitive generations."
    )
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME, help="Hugging Face model name.")
    parser.add_argument("--particles", nargs="+", default=["not"], help="Particles to localize.")
    parser.add_argument(
        "--components",
        default=",".join(COMPONENTS),
        help="Comma-separated components to patch: resid, attn, mlp.",
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for forward passes.")
    parser.add_argument(
        "--trace_rows_per_source",
        type=int,
        default=1,
        help="Rows retained per source item for localization scoring.",
    )
    parser.add_argument(
        "--max_source_items",
        type=int,
        default=None,
        help="Optional smoke-test cap on source items per training fold.",
    )
    parser.add_argument(
        "--position_label",
        choices=POSITION_LABELS,
        default=POSITION_LABEL_PROMPT_BOUNDARY,
        help="Prompt-side token location to patch.",
    )
    parser.add_argument(
        "--pool_root",
        default=None,
        help="Root containing per-particle generation_pool.tsv files. If omitted, uses default or latest patch run.",
    )
    parser.add_argument(
        "--output_root",
        default=None,
        help="Output root. Defaults to results/experiment3/localization/<timestamp>_<model>_<particles>_<position_label>.",
    )
    return parser.parse_args()


def parse_components(raw: object) -> list[str]:
    values = [part.strip().lower() for part in str(raw).split(",") if part.strip()]
    if not values:
        raise ValueError("At least one component is required.")
    invalid = sorted(set(values) - set(COMPONENTS))
    if invalid:
        raise ValueError(f"Unsupported components: {', '.join(invalid)}")
    return values


def default_output_root(model_name: str, particles: list[str], position_label: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    particle_slug = "_".join(particles)
    model_slug = model_short_name(model_name)
    return REPO_ROOT / "results" / "experiment3" / "localization" / (
        f"{timestamp}_{model_slug}_{particle_slug}_{position_label}"
    )


def _candidate_pool_dirs(model_name: str, particle: str) -> list[Path]:
    candidates = [default_results_dir(model_name, dataset_name=particle)]
    patch_root = REPO_ROOT / "results" / "experiment3" / "pools"
    if patch_root.exists():
        candidates.extend(sorted((path / particle for path in patch_root.iterdir() if path.is_dir()), reverse=True))
    return candidates


def resolve_pool_dir(model_name: str, particle: str, pool_root: str | None) -> Path:
    if pool_root:
        root = Path(pool_root)
        if (root / "generation_pool.tsv").exists():
            return root
        if (root / particle / "generation_pool.tsv").exists():
            return root / particle
        raise FileNotFoundError(f"Could not find generation_pool.tsv under {root} or {root / particle}")

    for candidate in _candidate_pool_dirs(model_name=model_name, particle=particle):
        if (candidate / "generation_pool.tsv").exists():
            return candidate
    searched = ", ".join(str(path) for path in _candidate_pool_dirs(model_name=model_name, particle=particle))
    raise FileNotFoundError(f"Could not find a generation_pool.tsv for {particle}; searched: {searched}")


def component_patching_dir(run_dir: Path) -> Path:
    return run_dir / PATCH_DIR_NAME


def copy_pool_files(pool_dir: Path, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    for file_name in SUMMARY_FILES:
        source_path = pool_dir / file_name
        if source_path.exists():
            shutil.copyfile(source_path, run_dir / file_name)


def prompt_boundary_positions(prompt_lens: np.ndarray) -> np.ndarray:
    prompt_array = np.asarray(prompt_lens, dtype=np.int64)
    positions = prompt_array - 1
    if np.any(positions < 0):
        raise ValueError("All prompts must contain at least one token for prompt-boundary patching.")
    return positions.reshape(-1, 1)


def resolve_prompt_positions(
    tokenizer,
    batch_df: pd.DataFrame,
    encoded_batch,
    position_label: str,
    prompt_column: str,
    utterance_column: str,
) -> np.ndarray:
    if position_label == POSITION_LABEL_PROMPT_BOUNDARY:
        return prompt_boundary_positions(encoded_batch.prompt_lens)
    if position_label == POSITION_LABEL_UTTERANCE_FINAL:
        return resolve_utterance_final_positions(
            tokenizer=tokenizer,
            prompts=batch_df[prompt_column].astype(str).tolist(),
            utterances=batch_df[utterance_column].astype(str).tolist(),
        )
    if position_label == POSITION_LABEL_UTTERANCE_FINAL_LEXICAL:
        return resolve_utterance_final_lexical_positions(
            tokenizer=tokenizer,
            prompts=batch_df[prompt_column].astype(str).tolist(),
            utterances=batch_df[utterance_column].astype(str).tolist(),
        )
    raise ValueError(f"Unsupported position_label: {position_label}")


def select_trace_rows(
    train_df: pd.DataFrame,
    trace_rows_per_source: int,
    max_source_items: int | None,
) -> pd.DataFrame:
    if trace_rows_per_source <= 0:
        raise ValueError("trace_rows_per_source must be positive.")
    ordered_df = train_df.sort_values(["source_row_index", "pool_rank", "pool_row_index"]).copy()
    if max_source_items is not None:
        source_ids = ordered_df["source_row_index"].drop_duplicates().head(max_source_items)
        ordered_df = ordered_df.loc[ordered_df["source_row_index"].isin(source_ids)].copy()
    return (
        ordered_df.groupby("source_row_index", dropna=False, sort=False)
        .head(trace_rows_per_source)
        .reset_index(drop=True)
    )


def group_site_rows(site_row_df: pd.DataFrame) -> pd.DataFrame:
    site_key_columns = [
        "particle",
        "train_fold",
        "component",
        "layer_index",
        "position_label",
        "position_order",
        "corruption_mode",
    ]
    value_columns = [
        "clean_log_prob_per_token",
        "corrupted_log_prob_per_token",
        "patched_log_prob_per_token",
        "raw_delta_per_token",
        "restoration",
        "restoration_clipped",
    ]
    item_df = (
        site_row_df.groupby(site_key_columns + ["source_row_index"], dropna=False, sort=False)[value_columns]
        .mean()
        .reset_index()
    )
    summary_df = (
        item_df.groupby(site_key_columns, dropna=False, sort=False)[value_columns]
        .mean()
        .reset_index()
        .rename(
            columns={
                "clean_log_prob_per_token": "mean_clean_log_prob_per_token",
                "corrupted_log_prob_per_token": "mean_corrupted_log_prob_per_token",
                "patched_log_prob_per_token": "mean_patched_log_prob_per_token",
                "raw_delta_per_token": "mean_raw_delta_per_token",
                "restoration": "mean_restoration",
                "restoration_clipped": "mean_restoration_clipped",
            }
        )
    )
    row_counts = site_row_df.groupby(site_key_columns, dropna=False, sort=False).size().reset_index(name="row_count")
    source_counts = item_df.groupby(site_key_columns, dropna=False, sort=False).size().reset_index(
        name="source_item_count"
    )
    summary_df = summary_df.merge(row_counts, on=site_key_columns, how="left", validate="one_to_one")
    summary_df = summary_df.merge(source_counts, on=site_key_columns, how="left", validate="one_to_one")
    summary_df["site_metric"] = "restoration"
    summary_df["ranking_score"] = pd.to_numeric(summary_df["mean_restoration"], errors="coerce")
    return summary_df.sort_values(
        ["train_fold", "component", "ranking_score", "layer_index"],
        ascending=[True, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def localize_particle(
    model,
    tokenizer,
    particle: str,
    pool_dir: Path,
    run_dir: Path,
    components: list[str],
    batch_size: int,
    trace_rows_per_source: int,
    max_source_items: int | None,
    position_label: str,
) -> None:
    pool_path = pool_dir / "generation_pool.tsv"
    if not pool_path.exists():
        raise FileNotFoundError(f"Missing generation pool: {pool_path}")
    copy_pool_files(pool_dir=pool_dir, run_dir=run_dir)
    pool_df = pd.read_csv(pool_path, sep="\t")
    pool_df = prepare_prompt_columns_for_model(pool_df, tokenizer, model.config._name_or_path)
    required_columns = {
        "pool_row_index",
        "pool_rank",
        "source_row_index",
        "fold_id",
        "response",
        "prompt_with",
        "prompt_without",
    }
    if position_label in {POSITION_LABEL_UTTERANCE_FINAL, POSITION_LABEL_UTTERANCE_FINAL_LEXICAL}:
        required_columns.update({"w_word", "wo_word"})
    missing = sorted(required_columns - set(pool_df.columns))
    if missing:
        raise KeyError(f"{pool_path} is missing required columns: {', '.join(missing)}")

    patch_dir = component_patching_dir(run_dir)
    patch_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(model)
    site_row_records = []

    for component in components:
        layer_count = len(get_component_modules(model, component))
        for fold_id in sorted(int(value) for value in pd.unique(pool_df["fold_id"])):
            train_df = pool_df.loc[pool_df["fold_id"] == fold_id].copy().reset_index(drop=True)
            train_df = select_trace_rows(
                train_df=train_df,
                trace_rows_per_source=trace_rows_per_source,
                max_source_items=max_source_items,
            )
            if train_df.empty:
                raise ValueError(f"Fold {fold_id} contained no localization rows for {particle}.")

            desc = f"{particle} {component} fold {fold_id} {position_label} sites"
            for start in tqdm(range(0, len(train_df), batch_size), desc=desc):
                batch_df = train_df.iloc[start : start + batch_size].copy().reset_index(drop=True)
                clean_batch = build_encoded_sequence_batch(
                    tokenizer=tokenizer,
                    prompts=batch_df["prompt_with"].astype(str).tolist(),
                    targets=batch_df["response"].astype(str).tolist(),
                    device=device,
                )
                corrupted_batch = build_encoded_sequence_batch(
                    tokenizer=tokenizer,
                    prompts=batch_df["prompt_without"].astype(str).tolist(),
                    targets=batch_df["response"].astype(str).tolist(),
                    device=device,
                )
                clean_positions = resolve_prompt_positions(
                    tokenizer=tokenizer,
                    batch_df=batch_df,
                    encoded_batch=clean_batch,
                    position_label=position_label,
                    prompt_column="prompt_with",
                    utterance_column="w_word",
                )
                corrupted_positions = resolve_prompt_positions(
                    tokenizer=tokenizer,
                    batch_df=batch_df,
                    encoded_batch=corrupted_batch,
                    position_label=position_label,
                    prompt_column="prompt_without",
                    utterance_column="wo_word",
                )

                clean_logits, clean_cache = run_component_sequence_model(
                    model=model,
                    batch=clean_batch,
                    component=component,
                    capture_positions=clean_positions,
                )
                corrupted_logits, _ = run_component_sequence_model(
                    model=model,
                    batch=corrupted_batch,
                    component=component,
                )
                assert clean_cache is not None
                _, _, clean_per_token = score_target_log_probs_from_logits(
                    logits=clean_logits,
                    input_ids=clean_batch.input_ids,
                    prompt_lens=clean_batch.prompt_lens,
                    target_lens=clean_batch.target_lens,
                    bos_token_id=tokenizer.bos_token_id,
                )
                _, _, corrupted_per_token = score_target_log_probs_from_logits(
                    logits=corrupted_logits,
                    input_ids=corrupted_batch.input_ids,
                    prompt_lens=corrupted_batch.prompt_lens,
                    target_lens=corrupted_batch.target_lens,
                    bos_token_id=tokenizer.bos_token_id,
                )
                denominator = np.maximum(clean_per_token - corrupted_per_token, 1e-6)

                for layer_index in range(layer_count):
                    patch_plan = build_residual_patch_plan(
                        layer_position_pairs=[(layer_index, 0)],
                        target_positions=corrupted_positions,
                        source_cache=clean_cache,
                    )
                    patched_logits, _ = run_component_sequence_model(
                        model=model,
                        batch=corrupted_batch,
                        component=component,
                        patch_plan=patch_plan,
                    )
                    _, _, patched_per_token = score_target_log_probs_from_logits(
                        logits=patched_logits,
                        input_ids=corrupted_batch.input_ids,
                        prompt_lens=corrupted_batch.prompt_lens,
                        target_lens=corrupted_batch.target_lens,
                        bos_token_id=tokenizer.bos_token_id,
                    )
                    raw_delta_per_token = patched_per_token - corrupted_per_token
                    restoration = raw_delta_per_token / denominator
                    restoration_clipped = np.clip(restoration, 0.0, 1.0)
                    for row_idx, row in batch_df.iterrows():
                        site_row_records.append(
                            {
                                "particle": particle,
                                "train_fold": fold_id,
                                "source_row_index": int(row["source_row_index"]),
                                "pool_row_index": int(row["pool_row_index"]),
                                "component": component,
                                "layer_index": layer_index,
                                "position_label": position_label,
                                "position_order": POSITION_ORDER,
                                "corruption_mode": CORRUPTION_MODE,
                                "clean_patch_position": int(clean_positions[row_idx, 0]),
                                "corrupted_patch_position": int(corrupted_positions[row_idx, 0]),
                                "clean_log_prob_per_token": float(clean_per_token[row_idx]),
                                "corrupted_log_prob_per_token": float(corrupted_per_token[row_idx]),
                                "patched_log_prob_per_token": float(patched_per_token[row_idx]),
                                "raw_delta_per_token": float(raw_delta_per_token[row_idx]),
                                "restoration": float(restoration[row_idx]),
                                "restoration_clipped": float(restoration_clipped[row_idx]),
                            }
                        )

    if not site_row_records:
        raise ValueError(f"No component patching site rows were generated for {particle}.")

    site_row_df = pd.DataFrame(site_row_records)
    site_row_df.to_csv(patch_dir / "site_rows.tsv", sep="\t", index=False)
    site_scores_df = group_site_rows(site_row_df)
    site_scores_df.to_csv(patch_dir / "site_scores.tsv", sep="\t", index=False)
    print(f"Saved component {position_label} site scores under {patch_dir}")


def main() -> None:
    args = parse_args()
    particles = [normalize_particle_name(value) for value in args.particles]
    components = parse_components(args.components)
    output_root = (
        Path(args.output_root)
        if args.output_root
        else default_output_root(args.model_name, particles, args.position_label)
    )

    print(f"Loading model: {args.model_name}")
    model, tokenizer = load_model(args.model_name)
    for particle in particles:
        pool_dir = resolve_pool_dir(model_name=args.model_name, particle=particle, pool_root=args.pool_root)
        run_dir = output_root / particle
        print(f"Using pool for {particle}: {pool_dir}")
        localize_particle(
            model=model,
            tokenizer=tokenizer,
            particle=particle,
            pool_dir=pool_dir,
            run_dir=run_dir,
            components=components,
            batch_size=args.batch_size,
            trace_rows_per_source=args.trace_rows_per_source,
            max_source_items=args.max_source_items,
            position_label=args.position_label,
        )


if __name__ == "__main__":
    main()
