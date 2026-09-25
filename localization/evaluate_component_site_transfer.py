#!/usr/bin/env python3
"""Evaluate component sites localized on one group against another group's targets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from localization.common import (
    DEFAULT_MODEL_NAME,
    load_model,
    parse_random_seeds,
    prepare_prompt_columns_for_model,
)
from localization.evaluate_component_prompt_boundary import (
    PATCH_DIR_NAME,
    POSITION_LABEL_UTTERANCE_FINAL,
    POSITION_LABEL_UTTERANCE_FINAL_LEXICAL,
    build_evaluation_targets,
    build_site_overlap_rows,
    build_site_set_requests,
    infer_position_label,
    load_site_scores,
    parse_eval_modes,
    parse_top_ks,
    score_component_site_requests,
    select_top_sites,
)
from localization.patch_analysis import write_patch_analysis_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Patch source-group localized component sites into target-group held-out examples."
    )
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME, help="Hugging Face model name.")
    parser.add_argument(
        "--source_site_scores",
        required=True,
        help="Source localization component_patching/site_scores.tsv.",
    )
    parser.add_argument("--target_pool", required=True, help="Target generation_pool.tsv.")
    parser.add_argument(
        "--transfer_label",
        required=True,
        help="Stable identifier stored in the particle field, e.g. exclusive_just_to_only.",
    )
    parser.add_argument("--source_label", required=True, help="Human-readable source group label.")
    parser.add_argument("--target_label", required=True, help="Human-readable target group label.")
    parser.add_argument(
        "--output_root",
        required=True,
        help="Transfer run root; outputs go under TRANSFER_LABEL/component_patching.",
    )
    parser.add_argument("--top_ks", default="1,3,5,10")
    parser.add_argument("--random_seeds", default="0,1,2,3")
    parser.add_argument("--eval_modes", default="sufficiency,necessity")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_source_items", type=int, default=None)
    parser.add_argument("--analysis_bootstrap_replicates", type=int, default=5000)
    parser.add_argument("--analysis_bootstrap_seed", type=int, default=0)
    parser.add_argument("--analysis_signflip_replicates", type=int, default=20000)
    return parser.parse_args()


def load_target_pool(path: Path, position_label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Target pool does not exist: {path}")
    pool_df = pd.read_csv(path, sep="\t")
    required = {
        "pool_row_index",
        "source_row_index",
        "fold_id",
        "id",
        "followup",
        "response",
        "prompt_with",
        "prompt_without",
    }
    if position_label in {POSITION_LABEL_UTTERANCE_FINAL, POSITION_LABEL_UTTERANCE_FINAL_LEXICAL}:
        required.update({"w_word", "wo_word"})
    missing = sorted(required - set(pool_df.columns))
    if missing:
        raise KeyError(f"{path} is missing required columns: {', '.join(missing)}")
    return pool_df


def evaluate_transfer(
    model,
    tokenizer,
    site_scores_df: pd.DataFrame,
    target_pool_df: pd.DataFrame,
    transfer_label: str,
    source_label: str,
    target_label: str,
    output_dir: Path,
    top_ks: list[int],
    random_seeds: list[int],
    eval_modes: list[str],
    batch_size: int,
    max_source_items: int | None,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    signflip_replicates: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    position_label = infer_position_label(site_scores_df=site_scores_df, patch_dir=output_dir)
    components = sorted(str(value) for value in pd.unique(site_scores_df["component"]))
    source_folds = sorted(int(value) for value in pd.unique(site_scores_df["train_fold"]))
    target_folds = sorted(int(value) for value in pd.unique(target_pool_df["fold_id"]))
    if source_folds != target_folds:
        raise ValueError(
            f"Source-site folds {source_folds} do not match target-pool folds {target_folds}."
        )

    site_universes: dict[str, dict[int, list[dict[str, int | str]]]] = {
        component: {} for component in components
    }
    localized_site_sets: dict[str, dict[int, dict[int, list[dict[str, int | str]]]]] = {
        component: {top_k: {} for top_k in top_ks} for component in components
    }
    selected_records = []
    for component in components:
        component_scores = site_scores_df.loc[site_scores_df["component"] == component].copy()
        for fold_id in source_folds:
            fold_scores = component_scores.loc[component_scores["train_fold"] == fold_id].copy()
            universe = select_top_sites(fold_component_scores=fold_scores, top_k=len(fold_scores))
            site_universes[component][fold_id] = universe
            for top_k in top_ks:
                sites = select_top_sites(fold_component_scores=fold_scores, top_k=top_k)
                localized_site_sets[component][top_k][fold_id] = sites
                selected_records.append(
                    {
                        "transfer_label": transfer_label,
                        "site_source": source_label,
                        "evaluation_target": target_label,
                        "train_fold": fold_id,
                        "component": component,
                        "top_k": top_k,
                        "selected_layers": ",".join(str(site["layer_index"]) for site in sites),
                    }
                )

    row_frames = []
    for fold_id in source_folds:
        eval_df = build_evaluation_targets(
            pool_df=target_pool_df,
            train_fold=fold_id,
            max_source_items=max_source_items,
        )
        for component in components:
            for eval_mode in eval_modes:
                requests = build_site_set_requests(
                    particle=transfer_label,
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
                        site_requests=requests,
                        batch_size=batch_size,
                        position_label=position_label,
                    )
                )

    row_df = pd.concat(row_frames, ignore_index=True, sort=False)
    row_df.insert(1, "site_source", source_label)
    row_df.insert(2, "evaluation_target", target_label)
    row_df.to_csv(output_dir / "eval_rows.tsv", sep="\t", index=False)
    pd.DataFrame(selected_records).to_csv(output_dir / "selected_site_sets.tsv", sep="\t", index=False)
    site_scores_df.to_csv(output_dir / "source_site_scores.tsv", sep="\t", index=False)

    overlap_df = build_site_overlap_rows(
        localized_site_sets=localized_site_sets,
        site_universes=site_universes,
        position_label=position_label,
    )
    overlap_df.insert(0, "transfer_label", transfer_label)
    overlap_df.to_csv(output_dir / "source_site_overlap.tsv", sep="\t", index=False)

    pd.DataFrame(
        [
            {
                "transfer_label": transfer_label,
                "site_source": source_label,
                "evaluation_target": target_label,
                "position_label": position_label,
                "source_site_score_rows": len(site_scores_df),
                "target_pool_rows": len(target_pool_df),
                "target_source_items": target_pool_df["source_row_index"].nunique(),
                "top_ks": ",".join(str(value) for value in top_ks),
                "random_seeds": ",".join(str(value) for value in random_seeds),
                "eval_modes": ",".join(eval_modes),
            }
        ]
    ).to_csv(output_dir / "transfer_config.tsv", sep="\t", index=False)

    write_patch_analysis_outputs(
        patch_dir=output_dir,
        row_df=row_df,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
        signflip_replicates=signflip_replicates,
    )
    print(f"Saved component-site transfer outputs under {output_dir}")


def main() -> None:
    args = parse_args()
    top_ks = parse_top_ks(args.top_ks)
    random_seeds = parse_random_seeds(args.random_seeds)
    eval_modes = parse_eval_modes(args.eval_modes)
    output_dir = Path(args.output_root) / args.transfer_label / PATCH_DIR_NAME
    source_site_scores_path = Path(args.source_site_scores)
    site_scores_df = load_site_scores(source_site_scores_path.parent)
    position_label = infer_position_label(
        site_scores_df=site_scores_df,
        patch_dir=source_site_scores_path.parent,
    )
    target_pool_df = load_target_pool(Path(args.target_pool), position_label=position_label)

    print(f"Loading model: {args.model_name}")
    model, tokenizer = load_model(args.model_name)
    target_pool_df = prepare_prompt_columns_for_model(
        target_pool_df,
        tokenizer,
        args.model_name,
    )
    evaluate_transfer(
        model=model,
        tokenizer=tokenizer,
        site_scores_df=site_scores_df,
        target_pool_df=target_pool_df,
        transfer_label=args.transfer_label,
        source_label=args.source_label,
        target_label=args.target_label,
        output_dir=output_dir,
        top_ks=top_ks,
        random_seeds=random_seeds,
        eval_modes=eval_modes,
        batch_size=args.batch_size,
        max_source_items=args.max_source_items,
        bootstrap_replicates=args.analysis_bootstrap_replicates,
        bootstrap_seed=args.analysis_bootstrap_seed,
        signflip_replicates=args.analysis_signflip_replicates,
    )


if __name__ == "__main__":
    main()
