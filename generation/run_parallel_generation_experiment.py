#!/usr/bin/env python3
"""Run the generation/cross-scoring experiment across as many GPUs as possible."""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def _remove_empty_parent_dirs(path: Path, stop_at: Path) -> None:
    """Remove empty parent directories up to, but not including, stop_at."""
    current = path
    while current != stop_at and current.exists():
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent


def cleanup_shard_artifacts(shard_dir: Path, output_dir: Path) -> None:
    """Delete temporary shard files after a successful merged run."""
    if not shard_dir.exists():
        return
    shutil.rmtree(shard_dir)
    _remove_empty_parent_dirs(shard_dir.parent, output_dir)
    print(f"Removed temporary shard artifacts: {shard_dir}", flush=True)


def detect_usable_gpus(min_free_mem_mb: int) -> List[int]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.free",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(cmd, text=True)
    gpu_ids = []
    for line in output.strip().splitlines():
        gpu_idx_str, free_mem_str = [part.strip() for part in line.split(",", 1)]
        if int(free_mem_str) >= min_free_mem_mb:
            gpu_ids.append(int(gpu_idx_str))
    return gpu_ids


def split_tsv(data_path: Path, shard_count: int, shard_dir: Path) -> List[Path]:
    shard_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_path, sep="\t")
    if df.empty:
        raise ValueError(f"Dataset is empty: {data_path}")
    if "__source_dataset" not in df.columns:
        df.insert(0, "__source_dataset", data_path.stem)
    if "__source_row_index" not in df.columns:
        df.insert(0, "__source_row_index", range(len(df)))

    shard_count = max(1, min(shard_count, len(df)))
    rows_per_shard = math.ceil(len(df) / shard_count)
    shard_paths = []
    for shard_idx in range(shard_count):
        start = shard_idx * rows_per_shard
        end = min(len(df), (shard_idx + 1) * rows_per_shard)
        shard_df = df.iloc[start:end]
        if shard_df.empty:
            continue
        shard_path = shard_dir / f"{data_path.stem}.shard{shard_idx}.tsv"
        shard_df.to_csv(shard_path, sep="\t", index=False)
        shard_paths.append(shard_path)
    return shard_paths


def run_worker(
    model_name: str,
    shard_path: Path,
    output_path: Path,
    gpu_id: int,
    log_path: Path,
    include_context: bool,
    samples_per_condition: int,
    top_p: float,
    temperature: float,
    sample_batch_size: int,
    max_sample_draws: int,
    max_new_tokens: int,
    base_seed: int,
    prompt_mode: str,
    system_prompt: Optional[str],
) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "generation/generate_topk_responses.py",
        "--model_name",
        model_name,
        "--data_path",
        str(shard_path),
        "--output_path",
        str(output_path),
        "--samples_per_condition",
        str(samples_per_condition),
        "--top_p",
        str(top_p),
        "--temperature",
        str(temperature),
        "--sample_batch_size",
        str(sample_batch_size),
        "--max_sample_draws",
        str(max_sample_draws),
        "--max_new_tokens",
        str(max_new_tokens),
        "--base_seed",
        str(base_seed),
        "--prompt_mode",
        prompt_mode,
    ]
    if not include_context:
        cmd.append("--no_context")
    if system_prompt is not None:
        cmd.extend(["--system_prompt", system_prompt])

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    proc._codex_log_file = log_file
    return proc


def analyze_candidates(candidates_path: Path) -> None:
    cmd = [
        sys.executable,
        "generation/analyze_generation_results.py",
        "--candidates_path",
        str(candidates_path),
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def run_dataset_parallel(
    model_name: str,
    data_path: Path,
    output_dir: Path,
    gpu_ids: List[int],
    include_context: bool,
    samples_per_condition: int,
    top_p: float,
    temperature: float,
    sample_batch_size: int,
    max_sample_draws: int,
    max_new_tokens: int,
    base_seed: int,
    skip_analysis: bool,
    prompt_mode: str,
    system_prompt: Optional[str],
    keep_shards: bool,
    skip_existing: bool,
) -> Path:
    model_short = model_name.split("/")[-1]
    context_tag = "context" if include_context else "noContext"
    shard_dir = output_dir / "shards" / data_path.stem
    final_output_path = output_dir / (
        f"{model_short}_{context_tag}_{data_path.stem}_generation_candidates.tsv"
    )
    if skip_existing and final_output_path.exists():
        print(f"Skipping existing candidates: {final_output_path}", flush=True)
        if not keep_shards:
            cleanup_shard_artifacts(shard_dir, output_dir)
        return final_output_path

    shard_paths = split_tsv(data_path, len(gpu_ids), shard_dir)
    used_gpu_ids = gpu_ids[: len(shard_paths)]

    print(
        f"Processing {data_path} across GPUs {used_gpu_ids} "
        f"with {len(shard_paths)} shard(s)",
        flush=True,
    )

    processes = []
    shard_output_paths = []
    log_paths = []
    for shard_idx, (gpu_id, shard_path) in enumerate(zip(used_gpu_ids, shard_paths)):
        shard_output_path = shard_dir / (
            f"{model_short}_{context_tag}_{data_path.stem}_generation_shard{shard_idx}.tsv"
        )
        log_path = shard_dir / (
            f"{model_short}_{context_tag}_{data_path.stem}_generation_shard{shard_idx}.log"
        )
        proc = run_worker(
            model_name=model_name,
            shard_path=shard_path,
            output_path=shard_output_path,
            gpu_id=gpu_id,
            log_path=log_path,
            include_context=include_context,
            samples_per_condition=samples_per_condition,
            top_p=top_p,
            temperature=temperature,
            sample_batch_size=sample_batch_size,
            max_sample_draws=max_sample_draws,
            max_new_tokens=max_new_tokens,
            base_seed=base_seed,
            prompt_mode=prompt_mode,
            system_prompt=system_prompt,
        )
        processes.append(proc)
        shard_output_paths.append(shard_output_path)
        log_paths.append(log_path)

    try:
        for proc, log_path in zip(processes, log_paths):
            return_code = proc.wait()
            log_file = getattr(proc, "_codex_log_file", None)
            if log_file is not None:
                log_file.close()
            if return_code != 0:
                raise RuntimeError(
                    f"Shard worker failed with code {return_code}. See log: {log_path}"
                )

        merged_df = pd.concat(
            [pd.read_csv(path, sep="\t") for path in shard_output_paths],
            ignore_index=True,
        )
        merged_df.sort_values(
            ["source_dataset", "source_row_index", "generated_from", "unique_rank"],
            inplace=True,
        )
        merged_df.to_csv(final_output_path, sep="\t", index=False)
        print(f"Merged candidates saved to: {final_output_path}", flush=True)

        if not skip_analysis:
            analyze_candidates(final_output_path)
    finally:
        for proc in processes:
            log_file = getattr(proc, "_codex_log_file", None)
            if log_file is not None and not log_file.closed:
                log_file.close()

    if not keep_shards:
        cleanup_shard_artifacts(shard_dir, output_dir)
    return final_output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run nucleus-sampled generation/cross-scoring across GPUs."
    )
    parser.add_argument(
        "--model_name",
        default="meta-llama/Meta-Llama-3-8B",
        help="Hugging Face model name.",
    )
    parser.add_argument(
        "--data_paths",
        nargs="+",
        default=["data/dailydialog_sense_subset.tsv"],
        help="Stimulus TSVs to process.",
    )
    parser.add_argument(
        "--output_dir",
        default="results/experiment2/llama3",
        help="Directory for merged outputs, logs, and shards.",
    )
    parser.add_argument(
        "--gpu_ids",
        nargs="+",
        type=int,
        default=None,
        help="Explicit GPU ids. If omitted, use all GPUs over min_free_mem_mb.",
    )
    parser.add_argument(
        "--min_free_mem_mb",
        type=int,
        default=18000,
        help="Minimum free VRAM required for one full model replica.",
    )
    parser.add_argument("--samples_per_condition", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sample_batch_size", type=int, default=64)
    parser.add_argument("--max_sample_draws", type=int, default=200)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--base_seed", type=int, default=20260416)
    parser.add_argument(
        "--prompt_mode",
        choices=["base", "chat"],
        default="base",
        help="Use base speaker-label prompts or tokenizer chat-template prompts.",
    )
    parser.add_argument(
        "--system_prompt",
        default=None,
        help="Optional system prompt to pass to chat-template workers.",
    )
    parser.add_argument("--no_context", action="store_true")
    parser.add_argument("--skip_analysis", action="store_true")
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Reuse an existing merged candidates TSV instead of regenerating it.",
    )
    parser.add_argument(
        "--keep_shards",
        action="store_true",
        help="Keep parallel shard TSVs/logs instead of deleting them after merge.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.gpu_ids is not None:
        gpu_ids = args.gpu_ids
    else:
        gpu_ids = detect_usable_gpus(args.min_free_mem_mb)
    if not gpu_ids:
        raise RuntimeError(f"No GPUs have at least {args.min_free_mem_mb} MiB free.")

    include_context = not args.no_context
    print(f"Usable GPUs: {gpu_ids}", flush=True)
    print(
        f"Sampling strategy: nucleus top_p={args.top_p}, "
        f"samples_per_condition={args.samples_per_condition}, "
        f"prompt_mode={args.prompt_mode}",
        flush=True,
    )

    merged_paths = []
    for data_path_str in args.data_paths:
        merged_paths.append(
            run_dataset_parallel(
                model_name=args.model_name,
                data_path=Path(data_path_str),
                output_dir=output_dir,
                gpu_ids=gpu_ids,
                include_context=include_context,
                samples_per_condition=args.samples_per_condition,
                top_p=args.top_p,
                temperature=args.temperature,
                sample_batch_size=args.sample_batch_size,
                max_sample_draws=args.max_sample_draws,
                max_new_tokens=args.max_new_tokens,
                base_seed=args.base_seed,
                skip_analysis=args.skip_analysis,
                prompt_mode=args.prompt_mode,
                system_prompt=args.system_prompt,
                keep_shards=args.keep_shards,
                skip_existing=args.skip_existing,
            )
        )

    if len(merged_paths) > 1:
        model_short = args.model_name.split("/")[-1]
        context_tag = "context" if include_context else "noContext"
        combined_path = output_dir / (
            f"{model_short}_{context_tag}_stimulus_set_generation_candidates.tsv"
        )
        combined_df = pd.concat(
            [pd.read_csv(path, sep="\t") for path in merged_paths],
            ignore_index=True,
        )
        combined_df.sort_values(
            ["source_dataset", "source_row_index", "generated_from", "unique_rank"],
            inplace=True,
        )
        combined_df.to_csv(combined_path, sep="\t", index=False)
        print(f"Combined candidates saved to: {combined_path}", flush=True)
        if not args.skip_analysis:
            analyze_candidates(combined_path)

    print("Parallel generation batch complete.", flush=True)


if __name__ == "__main__":
    main()
