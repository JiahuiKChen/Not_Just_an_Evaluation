#!/usr/bin/env python3
"""Evaluate component-level prompt-boundary patching sites."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from localization.common import (
    DEFAULT_MODEL_NAME,
    DEFAULT_RANDOM_MASK_SEEDS,
    build_encoded_sequence_batch,
    build_residual_patch_plan,
    get_device,
    load_model,
    normalize_particle_name,
    parse_random_seeds,
    prepare_prompt_columns_for_model,
    resolve_utterance_final_lexical_positions,
    resolve_utterance_final_positions,
    run_component_sequence_model,
    score_target_log_probs_from_logits,
    stable_int,
)
from localization.patch_analysis import write_patch_analysis_outputs

EVAL_MODE_SUFFICIENCY = "sufficiency"
EVAL_MODE_NECESSITY = "necessity"
SITE_SET_NONE = "none"
SITE_SET_LOCALIZED = "localized"
SITE_SET_RANDOM = "random"
PATCH_DIR_NAME = "component_patching"
POSITION_LABEL_PROMPT_BOUNDARY = "prompt_boundary"
POSITION_LABEL_UTTERANCE_FINAL = "utterance_final"
POSITION_LABEL_UTTERANCE_FINAL_LEXICAL = "utterance_final_lexical"
POSITION_LABELS = {
    POSITION_LABEL_PROMPT_BOUNDARY,
    POSITION_LABEL_UTTERANCE_FINAL,
    POSITION_LABEL_UTTERANCE_FINAL_LEXICAL,
}
CORRUPTION_MODE = "prompt_without"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate held-out component-level prompt-boundary sites for discourse particles."
    )
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME, help="Hugging Face model name.")
    parser.add_argument("--particles", nargs="+", default=["not"], help="Particles to evaluate.")
    parser.add_argument("--top_ks", default="1,3,5,10", help="Comma-separated top-k site-set sizes.")
    parser.add_argument(
        "--random_seeds",
        default=",".join(str(seed) for seed in DEFAULT_RANDOM_MASK_SEEDS),
        help="Comma-separated random seeds for same-size controls.",
    )
    parser.add_argument(
        "--eval_modes",
        default=f"{EVAL_MODE_SUFFICIENCY},{EVAL_MODE_NECESSITY}",
        help="Comma-separated evaluation modes: sufficiency, necessity.",
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for forward passes.")
    parser.add_argument(
        "--max_source_items",
        type=int,
        default=None,
        help="Optional smoke-test cap on held-out source items per train fold.",
    )
    parser.add_argument(
        "--output_root",
        default=None,
        help="Component patch run root. If omitted, uses the latest results/experiment3/localization run.",
    )
    parser.add_argument(
        "--analysis_bootstrap_replicates",
        type=int,
        default=5000,
        help="Bootstrap replicates for fold and paired-difference confidence intervals.",
    )
    parser.add_argument(
        "--analysis_bootstrap_seed",
        type=int,
        default=0,
        help="Random seed for summary bootstraps and sign-flip tests.",
    )
    parser.add_argument(
        "--analysis_signflip_replicates",
        type=int,
        default=20000,
        help="Monte Carlo replicates for localized-vs-random sign-flip tests.",
    )
    return parser.parse_args()


def parse_top_ks(raw: object) -> list[int]:
    values = [int(part.strip()) for part in str(raw).split(",") if part.strip()]
    if not values:
        raise ValueError("At least one top-k value is required.")
    if any(value <= 0 for value in values):
        raise ValueError("All top-k values must be positive.")
    return values


def parse_eval_modes(raw: object) -> list[str]:
    values = [part.strip() for part in str(raw).split(",") if part.strip()]
    allowed = {EVAL_MODE_SUFFICIENCY, EVAL_MODE_NECESSITY}
    invalid = sorted(set(values) - allowed)
    if invalid:
        raise ValueError(f"Unsupported eval modes: {', '.join(invalid)}")
    if not values:
        raise ValueError("At least one eval mode is required.")
    return values


def component_patching_dir(run_dir: Path) -> Path:
    return run_dir / PATCH_DIR_NAME


def resolve_run_dir(particle: str, output_root: str | None) -> Path:
    if output_root:
        root = Path(output_root)
        if (root / "generation_pool.tsv").exists():
            return root
        if (root / particle / "generation_pool.tsv").exists():
            return root / particle
        raise FileNotFoundError(f"Could not find generation_pool.tsv under {root} or {root / particle}")

    run_root = REPO_ROOT / "results" / "experiment3" / "localization"
    candidates = sorted(
        path / particle
        for path in run_root.iterdir()
        if path.is_dir() and (path / particle / "generation_pool.tsv").exists()
    ) if run_root.exists() else []
    if not candidates:
        raise FileNotFoundError(f"No component patch run found for {particle} under {run_root}")
    return candidates[-1]


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


def build_evaluation_targets(
    pool_df: pd.DataFrame,
    train_fold: int,
    max_source_items: int | None,
) -> pd.DataFrame:
    held_out_df = pool_df.loc[pool_df["fold_id"] != train_fold].copy()
    if held_out_df.empty:
        raise ValueError(f"No held-out rows remained for train fold {train_fold}")
    if max_source_items is not None:
        source_ids = held_out_df.sort_values(["source_row_index", "pool_row_index"])[
            "source_row_index"
        ].drop_duplicates().head(max_source_items)
        held_out_df = held_out_df.loc[held_out_df["source_row_index"].isin(source_ids)].copy()

    generated_df = held_out_df.copy()
    generated_df["target_type"] = "generated"
    generated_df["target_text"] = generated_df["response"].astype(str)
    generated_df["evaluation_id"] = generated_df["pool_row_index"].map(lambda value: f"generated:{value}")

    gold_df = held_out_df.sort_values("source_row_index").drop_duplicates("source_row_index").copy()
    gold_df["target_type"] = "gold"
    gold_df["target_text"] = gold_df["followup"].astype(str)
    gold_df["evaluation_id"] = gold_df["source_row_index"].map(lambda value: f"gold:{value}")

    eval_df = pd.concat([generated_df, gold_df], ignore_index=True, sort=False)
    eval_df["train_fold"] = train_fold
    eval_df["eval_fold"] = eval_df["fold_id"].astype(int)
    return eval_df


def load_site_scores(patch_dir: Path) -> pd.DataFrame:
    site_scores_path = patch_dir / "site_scores.tsv"
    if not site_scores_path.exists():
        raise FileNotFoundError(f"Missing site scores: {site_scores_path}")
    site_scores_df = pd.read_csv(site_scores_path, sep="\t")
    required_columns = {
        "train_fold",
        "component",
        "layer_index",
        "position_label",
        "position_order",
        "corruption_mode",
        "ranking_score",
    }
    missing = sorted(required_columns - set(site_scores_df.columns))
    if missing:
        raise KeyError(f"{site_scores_path} is missing required columns: {', '.join(missing)}")
    return site_scores_df


def infer_position_label(site_scores_df: pd.DataFrame, patch_dir: Path) -> str:
    labels = sorted(str(value) for value in pd.unique(site_scores_df["position_label"]))
    if len(labels) != 1:
        raise ValueError(f"Expected one position_label in {patch_dir / 'site_scores.tsv'}, found: {', '.join(labels)}")
    if labels[0] not in POSITION_LABELS:
        raise ValueError(f"Unsupported position_label in site_scores.tsv: {labels[0]}")
    return labels[0]


def select_top_sites(fold_component_scores: pd.DataFrame, top_k: int) -> list[dict[str, int | str]]:
    ranked_df = fold_component_scores.loc[
        pd.to_numeric(fold_component_scores["ranking_score"], errors="coerce").notna()
    ].copy()
    ranked_df.sort_values(
        ["ranking_score", "layer_index"],
        ascending=[False, True],
        inplace=True,
        kind="mergesort",
    )
    selected_df = ranked_df.head(top_k).copy()
    sites = []
    for row in selected_df.itertuples(index=False):
        sites.append(
            {
                "component": str(row.component),
                "layer_index": int(row.layer_index),
                "position_label": str(row.position_label),
                "position_index": 0,
            }
        )
    return sites


def sample_random_site_set(
    localized_sites: list[dict[str, int | str]],
    site_universe: list[dict[str, int | str]],
    seed: int,
) -> list[dict[str, int | str]]:
    if not localized_sites:
        return []
    localized_keys = {(str(site["component"]), int(site["layer_index"])) for site in localized_sites}
    available = [
        site
        for site in site_universe
        if (str(site["component"]), int(site["layer_index"])) not in localized_keys
    ]
    if len(available) < len(localized_sites):
        available = list(site_universe)
    rng = np.random.default_rng(seed)
    chosen_indices = rng.choice(len(available), size=len(localized_sites), replace=False)
    return [available[int(index)] for index in chosen_indices]


def build_site_set_requests(
    particle: str,
    component: str,
    fold_id: int,
    top_ks: list[int],
    eval_mode: str,
    localized_site_sets: dict[str, dict[int, dict[int, list[dict[str, int | str]]]]],
    site_universe: list[dict[str, int | str]],
    random_seeds: list[int],
) -> list[dict[str, object]]:
    requests = []
    for top_k in top_ks:
        localized_sites = localized_site_sets[component][top_k][fold_id]
        requests.append(
            {
                "particle": particle,
                "component": component,
                "top_k": top_k,
                "eval_mode": eval_mode,
                "site_set_type": SITE_SET_NONE,
                "selected_sites": [],
                "random_seed": "",
            }
        )
        requests.append(
            {
                "particle": particle,
                "component": component,
                "top_k": top_k,
                "eval_mode": eval_mode,
                "site_set_type": SITE_SET_LOCALIZED,
                "selected_sites": localized_sites,
                "random_seed": "",
            }
        )
        for seed in random_seeds:
            random_sites = sample_random_site_set(
                localized_sites=localized_sites,
                site_universe=site_universe,
                seed=stable_int(particle, component, fold_id, top_k, eval_mode, seed),
            )
            requests.append(
                {
                    "particle": particle,
                    "component": component,
                    "top_k": top_k,
                    "eval_mode": eval_mode,
                    "site_set_type": SITE_SET_RANDOM,
                    "selected_sites": random_sites,
                    "random_seed": str(seed),
                }
            )
    return requests


def build_site_overlap_rows(
    localized_site_sets: dict[str, dict[int, dict[int, list[dict[str, int | str]]]]],
    site_universes: dict[str, dict[int, list[dict[str, int | str]]]],
    position_label: str,
) -> pd.DataFrame:
    rows = []
    for component, top_k_sets in localized_site_sets.items():
        for top_k, fold_sets in top_k_sets.items():
            fold_ids = sorted(fold_sets)
            for left_fold in fold_ids:
                left_keys = {int(site["layer_index"]) for site in fold_sets[left_fold]}
                for right_fold in fold_ids:
                    right_keys = {int(site["layer_index"]) for site in fold_sets[right_fold]}
                    intersection = len(left_keys & right_keys)
                    union = len(left_keys | right_keys)
                    left_universe = len(site_universes[component][left_fold])
                    right_universe = len(site_universes[component][right_fold])
                    if left_universe == right_universe and left_universe > 0:
                        expected_intersection = (len(left_keys) * len(right_keys)) / float(left_universe)
                        expected_union = len(left_keys) + len(right_keys) - expected_intersection
                        expected_jaccard = expected_intersection / expected_union if expected_union > 0 else np.nan
                    else:
                        expected_intersection = np.nan
                        expected_jaccard = np.nan
                    rows.append(
                        {
                            "component": component,
                            "position_label": position_label,
                            "corruption_mode": CORRUPTION_MODE,
                            "top_k": top_k,
                            "fold_i": left_fold,
                            "fold_j": right_fold,
                            "intersection_count": intersection,
                            "jaccard": (intersection / union) if union > 0 else np.nan,
                            "selected_sites_i": len(left_keys),
                            "selected_sites_j": len(right_keys),
                            "site_universe_i": left_universe,
                            "site_universe_j": right_universe,
                            "expected_random_intersection": expected_intersection,
                            "expected_random_jaccard": expected_jaccard,
                        }
                    )
    return pd.DataFrame(rows)


def score_component_site_requests(
    model,
    tokenizer,
    eval_df: pd.DataFrame,
    site_requests: list[dict[str, object]],
    batch_size: int,
    position_label: str,
) -> pd.DataFrame:
    if eval_df.empty or not site_requests:
        return pd.DataFrame()

    device = get_device(model)
    component = str(site_requests[0]["component"])
    particle = str(site_requests[0]["particle"])
    eval_mode = str(site_requests[0]["eval_mode"])
    fold_id = int(eval_df["train_fold"].iloc[0])
    rows = []

    for start in tqdm(range(0, len(eval_df), batch_size), desc=f"{particle} {component} {eval_mode} fold {fold_id}"):
        batch_df = eval_df.iloc[start : start + batch_size].copy().reset_index(drop=True)
        clean_batch = build_encoded_sequence_batch(
            tokenizer=tokenizer,
            prompts=batch_df["prompt_with"].astype(str).tolist(),
            targets=batch_df["target_text"].astype(str).tolist(),
            device=device,
        )
        corrupted_batch = build_encoded_sequence_batch(
            tokenizer=tokenizer,
            prompts=batch_df["prompt_without"].astype(str).tolist(),
            targets=batch_df["target_text"].astype(str).tolist(),
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

        selected_layer_union = sorted(
            {
                int(site["layer_index"])
                for request in site_requests
                for site in request["selected_sites"]
            }
        )
        capture_sites = [(layer_index, 0) for layer_index in selected_layer_union]
        if eval_mode == EVAL_MODE_SUFFICIENCY:
            clean_logits, clean_cache = run_component_sequence_model(
                model=model,
                batch=clean_batch,
                component=component,
                capture_positions=clean_positions if capture_sites else None,
            )
            corrupted_logits, _ = run_component_sequence_model(
                model=model,
                batch=corrupted_batch,
                component=component,
            )
            baseline_logits = corrupted_logits
            baseline_batch = corrupted_batch
            target_positions = corrupted_positions
            source_cache = clean_cache
            baseline_condition = "corrupted"
            reference_condition = "clean"
        else:
            clean_logits, _ = run_component_sequence_model(
                model=model,
                batch=clean_batch,
                component=component,
            )
            corrupted_logits, corrupted_cache = run_component_sequence_model(
                model=model,
                batch=corrupted_batch,
                component=component,
                capture_positions=corrupted_positions if capture_sites else None,
            )
            baseline_logits = clean_logits
            baseline_batch = clean_batch
            target_positions = clean_positions
            source_cache = corrupted_cache
            baseline_condition = "clean"
            reference_condition = "corrupted"

        clean_totals, clean_counts, clean_per_token = score_target_log_probs_from_logits(
            logits=clean_logits,
            input_ids=clean_batch.input_ids,
            prompt_lens=clean_batch.prompt_lens,
            target_lens=clean_batch.target_lens,
            bos_token_id=tokenizer.bos_token_id,
        )
        corrupted_totals, corrupted_counts, corrupted_per_token = score_target_log_probs_from_logits(
            logits=corrupted_logits,
            input_ids=corrupted_batch.input_ids,
            prompt_lens=corrupted_batch.prompt_lens,
            target_lens=corrupted_batch.target_lens,
            bos_token_id=tokenizer.bos_token_id,
        )
        baseline_totals, baseline_counts, baseline_per_token = score_target_log_probs_from_logits(
            logits=baseline_logits,
            input_ids=baseline_batch.input_ids,
            prompt_lens=baseline_batch.prompt_lens,
            target_lens=baseline_batch.target_lens,
            bos_token_id=tokenizer.bos_token_id,
        )
        clean_minus_corrupted = np.maximum(clean_per_token - corrupted_per_token, 1e-6)

        patch_plan_cache: dict[tuple[int, ...], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for request in site_requests:
            selected_sites = list(request["selected_sites"])
            request_key = tuple(sorted(int(site["layer_index"]) for site in selected_sites))
            if request_key not in patch_plan_cache:
                if request_key:
                    assert source_cache is not None
                    patch_plan = build_residual_patch_plan(
                        layer_position_pairs=[(layer_index, 0) for layer_index in request_key],
                        target_positions=target_positions,
                        source_cache=source_cache,
                    )
                    patched_logits, _ = run_component_sequence_model(
                        model=model,
                        batch=baseline_batch,
                        component=component,
                        patch_plan=patch_plan,
                    )
                    patched_totals, patched_counts, patched_per_token = score_target_log_probs_from_logits(
                        logits=patched_logits,
                        input_ids=baseline_batch.input_ids,
                        prompt_lens=baseline_batch.prompt_lens,
                        target_lens=baseline_batch.target_lens,
                        bos_token_id=tokenizer.bos_token_id,
                    )
                else:
                    patched_totals = baseline_totals
                    patched_counts = baseline_counts
                    patched_per_token = baseline_per_token
                patch_plan_cache[request_key] = (patched_totals, patched_counts, patched_per_token)

            patched_totals, patched_counts, patched_per_token = patch_plan_cache[request_key]
            patched_metric_change = patched_per_token - baseline_per_token
            if eval_mode == EVAL_MODE_SUFFICIENCY:
                effect_score = patched_metric_change
            else:
                effect_score = baseline_per_token - patched_per_token
            normalized_effect = effect_score / clean_minus_corrupted

            for row_idx, row in batch_df.iterrows():
                rows.append(
                    {
                        "particle": str(request["particle"]),
                        "component": component,
                        "position_label": position_label,
                        "corruption_mode": CORRUPTION_MODE,
                        "top_k": int(request["top_k"]),
                        "train_fold": int(row["train_fold"]),
                        "eval_fold": int(row["eval_fold"]),
                        "target_type": row["target_type"],
                        "evaluation_id": row["evaluation_id"],
                        "source_row_index": int(row["source_row_index"]),
                        "id": row["id"],
                        "pool_row_index": row.get("pool_row_index", ""),
                        "target_text": row["target_text"],
                        "eval_mode": str(request["eval_mode"]),
                        "site_set_type": str(request["site_set_type"]),
                        "random_seed": str(request["random_seed"]),
                        "selected_site_count": int(len(selected_sites)),
                        "selected_layers": ",".join(str(layer_index) for layer_index in request_key),
                        "baseline_condition": baseline_condition,
                        "reference_condition": reference_condition,
                        "clean_log_prob": float(clean_totals[row_idx]),
                        "corrupted_log_prob": float(corrupted_totals[row_idx]),
                        "baseline_log_prob": float(baseline_totals[row_idx]),
                        "reference_log_prob": float(
                            clean_totals[row_idx]
                            if eval_mode == EVAL_MODE_SUFFICIENCY
                            else corrupted_totals[row_idx]
                        ),
                        "patched_log_prob": float(patched_totals[row_idx]),
                        "clean_token_count": int(clean_counts[row_idx]),
                        "corrupted_token_count": int(corrupted_counts[row_idx]),
                        "baseline_token_count": int(baseline_counts[row_idx]),
                        "reference_token_count": int(
                            clean_counts[row_idx]
                            if eval_mode == EVAL_MODE_SUFFICIENCY
                            else corrupted_counts[row_idx]
                        ),
                        "patched_token_count": int(patched_counts[row_idx]),
                        "clean_log_prob_per_token": float(clean_per_token[row_idx]),
                        "corrupted_log_prob_per_token": float(corrupted_per_token[row_idx]),
                        "baseline_log_prob_per_token": float(baseline_per_token[row_idx]),
                        "reference_log_prob_per_token": float(
                            clean_per_token[row_idx]
                            if eval_mode == EVAL_MODE_SUFFICIENCY
                            else corrupted_per_token[row_idx]
                        ),
                        "patched_log_prob_per_token": float(patched_per_token[row_idx]),
                        "patched_metric_change": float(patched_metric_change[row_idx]),
                        "effect_score": float(effect_score[row_idx]),
                        "normalized_effect": float(normalized_effect[row_idx]),
                    }
                )
    return pd.DataFrame(rows)


def evaluate_particle(
    model,
    tokenizer,
    particle: str,
    run_dir: Path,
    top_ks: list[int],
    random_seeds: list[int],
    eval_modes: list[str],
    batch_size: int,
    max_source_items: int | None,
    analysis_bootstrap_replicates: int,
    analysis_bootstrap_seed: int,
    analysis_signflip_replicates: int,
) -> None:
    pool_path = run_dir / "generation_pool.tsv"
    if not pool_path.exists():
        raise FileNotFoundError(f"Missing generation pool: {pool_path}")
    pool_df = pd.read_csv(pool_path, sep="\t")
    pool_df = prepare_prompt_columns_for_model(pool_df, tokenizer, model.config._name_or_path)
    required_columns = {
        "pool_row_index",
        "source_row_index",
        "fold_id",
        "id",
        "followup",
        "response",
        "prompt_with",
        "prompt_without",
    }
    missing = sorted(required_columns - set(pool_df.columns))
    if missing:
        raise KeyError(f"{pool_path} is missing required columns: {', '.join(missing)}")

    patch_dir = component_patching_dir(run_dir)
    patch_dir.mkdir(parents=True, exist_ok=True)
    site_scores_df = load_site_scores(patch_dir)
    position_label = infer_position_label(site_scores_df=site_scores_df, patch_dir=patch_dir)
    if position_label in {POSITION_LABEL_UTTERANCE_FINAL, POSITION_LABEL_UTTERANCE_FINAL_LEXICAL}:
        required_columns.update({"w_word", "wo_word"})
        missing = sorted(required_columns - set(pool_df.columns))
        if missing:
            raise KeyError(f"{pool_path} is missing required columns: {', '.join(missing)}")
    components = sorted(str(value) for value in pd.unique(site_scores_df["component"]))

    site_universes: dict[str, dict[int, list[dict[str, int | str]]]] = {component: {} for component in components}
    localized_site_sets: dict[str, dict[int, dict[int, list[dict[str, int | str]]]]] = {
        component: {top_k: {} for top_k in top_ks}
        for component in components
    }
    for component in components:
        component_scores = site_scores_df.loc[site_scores_df["component"] == component].copy()
        for fold_id in sorted(int(value) for value in pd.unique(component_scores["train_fold"])):
            fold_component_scores = component_scores.loc[component_scores["train_fold"] == fold_id].copy()
            universe = select_top_sites(fold_component_scores=fold_component_scores, top_k=len(fold_component_scores))
            site_universes[component][fold_id] = universe
            for top_k in top_ks:
                localized_site_sets[component][top_k][fold_id] = select_top_sites(
                    fold_component_scores=fold_component_scores,
                    top_k=top_k,
                )

    row_frames = []
    for fold_id in sorted(int(value) for value in pd.unique(pool_df["fold_id"])):
        eval_df = build_evaluation_targets(pool_df=pool_df, train_fold=fold_id, max_source_items=max_source_items)
        for component in components:
            for eval_mode in eval_modes:
                site_requests = build_site_set_requests(
                    particle=particle,
                    component=component,
                    fold_id=fold_id,
                    top_ks=top_ks,
                    eval_mode=eval_mode,
                    localized_site_sets=localized_site_sets,
                    site_universe=site_universes[component][fold_id],
                    random_seeds=random_seeds,
                )
                row_frames.append(
                    score_component_site_requests(
                        model=model,
                        tokenizer=tokenizer,
                        eval_df=eval_df,
                        site_requests=site_requests,
                        batch_size=batch_size,
                        position_label=position_label,
                    )
                )

    row_df = pd.concat(row_frames, ignore_index=True, sort=False)
    row_df.to_csv(patch_dir / "eval_rows.tsv", sep="\t", index=False)
    overlap_df = build_site_overlap_rows(
        localized_site_sets=localized_site_sets,
        site_universes=site_universes,
        position_label=position_label,
    )
    overlap_df.to_csv(patch_dir / "site_overlap.tsv", sep="\t", index=False)
    write_patch_analysis_outputs(
        patch_dir=patch_dir,
        row_df=row_df,
        bootstrap_replicates=analysis_bootstrap_replicates,
        bootstrap_seed=analysis_bootstrap_seed,
        signflip_replicates=analysis_signflip_replicates,
    )
    print(f"Saved component {position_label} evaluation outputs under {patch_dir}")


def main() -> None:
    args = parse_args()
    top_ks = parse_top_ks(args.top_ks)
    random_seeds = parse_random_seeds(args.random_seeds)
    eval_modes = parse_eval_modes(args.eval_modes)
    particles = [normalize_particle_name(value) for value in args.particles]

    print(f"Loading model: {args.model_name}")
    model, tokenizer = load_model(args.model_name)
    for particle in particles:
        run_dir = resolve_run_dir(particle=particle, output_root=args.output_root)
        print(f"Evaluating component patching run for {particle}: {run_dir}")
        evaluate_particle(
            model=model,
            tokenizer=tokenizer,
            particle=particle,
            run_dir=run_dir,
            top_ks=top_ks,
            random_seeds=random_seeds,
            eval_modes=eval_modes,
            batch_size=args.batch_size,
            max_source_items=args.max_source_items,
            analysis_bootstrap_replicates=args.analysis_bootstrap_replicates,
            analysis_bootstrap_seed=args.analysis_bootstrap_seed,
            analysis_signflip_replicates=args.analysis_signflip_replicates,
        )


if __name__ == "__main__":
    main()
