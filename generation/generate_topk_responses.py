#!/usr/bin/env python3
"""Generate plausible next-turn alternatives and cross-score them.

For each stimulus row, this script samples responses from:

    P  = context + w_word
    P' = context + wo_word

The human followup column is copied into the output as metadata only. It is not
included in the prompt. Responses are sampled with nucleus sampling by default
(`top_p=0.9`) and then scored under both P and P'.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generation.common import (  # noqa: E402
    build_chat_prompt_pair,
    build_prompt_pair,
    calculate_log_probability_chat,
    calculate_log_probability,
    chat_template_info,
    dataset_role_for_word,
    load_model,
    safe_div,
    sample_unique_responses_nucleus,
    sample_unique_responses_nucleus_chat,
    validate_stimulus_row,
)


def stable_seed(base_seed: int, *parts: object) -> int:
    joined = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    return (base_seed + int(digest[:8], 16)) % (2**31 - 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample generated responses from P/P' and cross-score them."
    )
    parser.add_argument(
        "--data_path",
        default="data/dailydialog_sense_subset.tsv",
        help="Stimulus TSV with context, w_word, wo_word, and followup columns.",
    )
    parser.add_argument(
        "--model_name",
        default="meta-llama/Meta-Llama-3-8B",
        help="Hugging Face model name.",
    )
    parser.add_argument(
        "--prompt_mode",
        choices=["base", "chat"],
        default="base",
        help="Use base speaker-label prompts or tokenizer chat-template prompts.",
    )
    parser.add_argument(
        "--system_prompt",
        default=None,
        help="Optional system prompt for chat prompt mode.",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        help="Output TSV path. Defaults to results/experiment2/<model>_context_<dataset>_generation_candidates.tsv.",
    )
    parser.add_argument(
        "--samples_per_condition",
        type=int,
        default=50,
        help="Number of unique sampled responses to keep for P and P' each.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling p value.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=64,
        help="Number of samples drawn per model.generate call.",
    )
    parser.add_argument(
        "--max_sample_draws",
        type=int,
        default=200,
        help="Maximum raw samples to draw per row and condition while seeking unique responses.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
        help="Maximum generated tokens per response before truncating to one turn.",
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=20260416,
        help="Base seed for reproducible nucleus samples.",
    )
    parser.add_argument(
        "--no_context",
        action="store_true",
        help="Condition only on w_word/wo_word, without previous context.",
    )
    parser.add_argument(
        "--row_limit",
        type=int,
        default=None,
        help="Optional number of rows to process after start_row filtering.",
    )
    parser.add_argument(
        "--start_row",
        type=int,
        default=0,
        help="Optional zero-based row offset into the input TSV.",
    )
    parser.add_argument(
        "--end_row",
        type=int,
        default=None,
        help="Optional exclusive zero-based end row into the input TSV.",
    )
    return parser.parse_args()


def default_output_path(model_name: str, data_path: str, include_context: bool) -> Path:
    model_short = model_name.split("/")[-1]
    context_tag = "context" if include_context else "noContext"
    data_stem = Path(data_path).stem
    return (
        REPO_ROOT
        / "results"
        / "experiment2"
        / f"{model_short}_{context_tag}_{data_stem}_generation_candidates.tsv"
    )


def score_response(model, tokenizer, prompt_pair, response: str, device) -> dict:
    log_prob_p, token_count_p = calculate_log_probability(
        model,
        tokenizer,
        prompt_pair.prompt_p,
        response,
        device,
    )
    log_prob_p_prime, token_count_p_prime = calculate_log_probability(
        model,
        tokenizer,
        prompt_pair.prompt_p_prime,
        response,
        device,
    )
    return {
        "log_prob_given_P": log_prob_p,
        "log_prob_given_P_prime": log_prob_p_prime,
        "log_prob_given_P_per_token": safe_div(log_prob_p, token_count_p),
        "log_prob_given_P_prime_per_token": safe_div(log_prob_p_prime, token_count_p_prime),
        "token_count_given_P": token_count_p,
        "token_count_given_P_prime": token_count_p_prime,
    }


def score_response_chat(model, tokenizer, prompt_pair, response: str, device) -> dict:
    target_messages = [{"role": "assistant", "content": response}]
    log_prob_p, token_count_p = calculate_log_probability_chat(
        model,
        tokenizer,
        prompt_pair.messages_p,
        target_messages,
        device,
    )
    log_prob_p_prime, token_count_p_prime = calculate_log_probability_chat(
        model,
        tokenizer,
        prompt_pair.messages_p_prime,
        target_messages,
        device,
    )
    return {
        "log_prob_given_P": log_prob_p,
        "log_prob_given_P_prime": log_prob_p_prime,
        "log_prob_given_P_per_token": safe_div(log_prob_p, token_count_p),
        "log_prob_given_P_prime_per_token": safe_div(log_prob_p_prime, token_count_p_prime),
        "token_count_given_P": token_count_p,
        "token_count_given_P_prime": token_count_p_prime,
    }


def process_dataset(args: argparse.Namespace) -> pd.DataFrame:
    include_context = not args.no_context
    data_path = Path(args.data_path)
    df = pd.read_csv(data_path, sep="\t")
    source_dataset = (
        str(df["__source_dataset"].iloc[0])
        if "__source_dataset" in df.columns and not df.empty
        else data_path.stem
    )
    if "__source_row_index" not in df.columns:
        df["__source_row_index"] = list(range(len(df)))

    if args.end_row is not None:
        df = df.iloc[: args.end_row]
    if args.start_row:
        df = df.iloc[args.start_row :]
    if args.row_limit is not None:
        df = df.iloc[: args.row_limit]

    print(f"Loaded {len(df)} rows from {data_path}", flush=True)
    print(
        f"Sampling {args.samples_per_condition} unique responses per condition "
        f"with nucleus top_p={args.top_p}",
        flush=True,
    )

    model, tokenizer = load_model(args.model_name)
    template_info = chat_template_info(args.model_name, tokenizer) if args.prompt_mode == "chat" else None
    if args.prompt_mode == "chat":
        if args.system_prompt and not template_info["supports_system"]:
            raise ValueError(f"{args.model_name} does not support system messages in its chat template.")
        print(f"Prompt mode: chat", flush=True)
    else:
        print(f"Prompt mode: base", flush=True)
    device = next(model.parameters()).device
    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Rows"):
        try:
            context, w_word, wo_word = validate_stimulus_row(row, include_context=include_context)
        except ValueError as exc:
            print(f"Skipping source row {row.get('__source_row_index')}: {exc}", flush=True)
            continue

        if args.prompt_mode == "chat":
            prompt_pair = build_chat_prompt_pair(
                context,
                w_word,
                wo_word,
                include_context=include_context,
                system_prompt=args.system_prompt,
                info=template_info,
            )
        else:
            prompt_pair = build_prompt_pair(context, w_word, wo_word)
        row_id = row.get("id")
        word = row.get("word")
        sense = row.get("sense", "")
        followup = row.get("followup", "")
        with_without_diff = row.get("with-without_diff", "")
        source_row_index = row.get("__source_row_index")
        dataset_role = dataset_role_for_word(word)

        if args.prompt_mode == "chat":
            condition_prompts = {
                "P": prompt_pair.messages_p,
                "P_prime": prompt_pair.messages_p_prime,
            }
        else:
            condition_prompts = {
                "P": prompt_pair.prompt_p,
                "P_prime": prompt_pair.prompt_p_prime,
            }
        condition_text = {
            "P": w_word,
            "P_prime": wo_word,
        }

        for generated_from, prompt in condition_prompts.items():
            seed = stable_seed(args.base_seed, source_dataset, source_row_index, row_id, generated_from)
            if args.prompt_mode == "chat":
                samples = sample_unique_responses_nucleus_chat(
                    model=model,
                    tokenizer=tokenizer,
                    messages=prompt,
                    device=device,
                    samples_per_condition=args.samples_per_condition,
                    top_p=args.top_p,
                    temperature=args.temperature,
                    sample_batch_size=args.sample_batch_size,
                    max_sample_draws=args.max_sample_draws,
                    max_new_tokens=args.max_new_tokens,
                    seed=seed,
                )
            else:
                samples = sample_unique_responses_nucleus(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    device=device,
                    samples_per_condition=args.samples_per_condition,
                    top_p=args.top_p,
                    temperature=args.temperature,
                    sample_batch_size=args.sample_batch_size,
                    max_sample_draws=args.max_sample_draws,
                    max_new_tokens=args.max_new_tokens,
                    seed=seed,
                )
            unique_count = len(samples)

            for sample in samples:
                if args.prompt_mode == "chat":
                    score = score_response_chat(model, tokenizer, prompt_pair, sample["response"], device)
                else:
                    score = score_response(model, tokenizer, prompt_pair, sample["response"], device)
                if generated_from == "P":
                    own_log_prob = score["log_prob_given_P"]
                    other_log_prob = score["log_prob_given_P_prime"]
                    own_per_token = score["log_prob_given_P_per_token"]
                    other_per_token = score["log_prob_given_P_prime_per_token"]
                else:
                    own_log_prob = score["log_prob_given_P_prime"]
                    other_log_prob = score["log_prob_given_P"]
                    own_per_token = score["log_prob_given_P_prime_per_token"]
                    other_per_token = score["log_prob_given_P_per_token"]

                rows.append(
                    {
                        "source_dataset": source_dataset,
                        "dataset_role": dataset_role,
                        "source_row_index": source_row_index,
                        "id": row_id,
                        "word": word,
                        "sense": sense,
                        "context": context if include_context else "[NO CONTEXT]",
                        "w_word": w_word,
                        "wo_word": wo_word,
                        "followup_metadata_only": followup,
                        "with_without_diff": with_without_diff,
                        "generated_from": generated_from,
                        "generated_from_text": condition_text[generated_from],
                        "sampling_strategy": "nucleus",
                        "prompt_mode": args.prompt_mode,
                        "top_p": args.top_p,
                        "temperature": args.temperature,
                        "base_seed": args.base_seed,
                        "condition_seed": seed,
                        "samples_requested": args.samples_per_condition,
                        "unique_samples_obtained": unique_count,
                        "max_sample_draws": args.max_sample_draws,
                        "sample_draw": sample["sample_draw"],
                        "unique_rank": sample["unique_rank"],
                        "response": sample["response"],
                        "normalized_response": sample["normalized_response"],
                        "raw_generation": sample["raw_generation"].replace("\r\n", "\\n").replace("\n", "\\n"),
                        **score,
                        "own_log_prob": own_log_prob,
                        "other_log_prob": other_log_prob,
                        "own_log_prob_per_token": own_per_token,
                        "other_log_prob_per_token": other_per_token,
                        "own_context_log_prob_advantage": own_log_prob - other_log_prob,
                        "own_context_log_prob_advantage_per_token": own_per_token - other_per_token,
                        "own_context_win": own_log_prob > other_log_prob,
                        "include_context": include_context,
                    }
                )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    include_context = not args.no_context
    output_path = Path(args.output_path) if args.output_path else default_output_path(
        args.model_name,
        args.data_path,
        include_context,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results_df = process_dataset(args)
    results_df.to_csv(output_path, sep="\t", index=False)
    print(f"Saved candidates to: {output_path}", flush=True)


if __name__ == "__main__":
    main()
