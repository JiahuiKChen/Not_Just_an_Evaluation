#!/usr/bin/env python3
"""Run followup-probability experiments in parallel across multiple GPUs.

This script shards each TSV across the GPUs that have enough free memory to host
an independent model replica, runs one worker per GPU, merges the shard outputs,
and recomputes global outlier statistics from the merged CSV.
"""

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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from particle_removal.calculate_followup_probabilities import removal_outlier_detection


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
    """Return GPU indices with enough currently free memory for a full replica."""
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.free",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(cmd, text=True)
    gpus = []
    for line in output.strip().splitlines():
        gpu_idx_str, free_mem_str = [part.strip() for part in line.split(",", 1)]
        gpu_idx = int(gpu_idx_str)
        free_mem_mb = int(free_mem_str)
        if free_mem_mb >= min_free_mem_mb:
            gpus.append(gpu_idx)
    return gpus


def split_tsv(data_path: Path, shard_count: int, shard_dir: Path) -> List[Path]:
    """Split a TSV into contiguous shards so merged outputs preserve row order."""
    shard_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_path, sep="\t")
    if df.empty:
        raise ValueError(f"Dataset is empty: {data_path}")

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


def build_summary(
    results_df: pd.DataFrame,
    include_context: bool,
    outlier_summary_lines: List[str],
    observer_mode: Optional[bool] = None,
) -> str:
    """Build the same summary text as the single-process experiment scripts."""
    raw_diff = results_df["log_prob_followup_with"] - results_df["log_prob_followup_without"]
    full_diff = results_df["log_prob_full_with"] - results_df["log_prob_full_without"]
    raw_diff_per_token = (
        results_df["log_prob_followup_with_per_token"]
        - results_df["log_prob_followup_without_per_token"]
    )
    full_diff_per_token = (
        results_df["log_prob_full_with_per_token"]
        - results_df["log_prob_full_without_per_token"]
    )

    increased = (raw_diff > 0).sum()
    decreased = (raw_diff < 0).sum()
    unchanged = (raw_diff == 0).sum()
    full_increased = (full_diff > 0).sum()
    full_decreased = (full_diff < 0).sum()
    full_unchanged = (full_diff == 0).sum()

    increased_pt = (raw_diff_per_token > 0).sum()
    decreased_pt = (raw_diff_per_token < 0).sum()
    unchanged_pt = (raw_diff_per_token == 0).sum()
    full_increased_pt = (full_diff_per_token > 0).sum()
    full_decreased_pt = (full_diff_per_token < 0).sum()
    full_unchanged_pt = (full_diff_per_token == 0).sum()

    summary_lines = [
        "=" * 80,
        "SUMMARY STATISTICS",
        "=" * 80,
        f"Total examples processed: {len(results_df)}",
        f"Include context: {include_context}",
    ]
    if observer_mode is not None:
        summary_lines.append(f"Observer mode: {observer_mode}")
    summary_lines.extend([
        "",
        "--- RAW FOLLOWUP LOG PROBABILITIES ---",
        f"Average log probability WITH particle: {results_df['log_prob_followup_with'].mean():.4f}",
        f"Average log probability WITHOUT particle: {results_df['log_prob_followup_without'].mean():.4f}",
        f"Average log probability difference: {raw_diff.mean():.4f}",
        "  (positive = particle increases followup probability)",
        "",
        f"Median log probability difference: {raw_diff.median():.4f}",
        f"Std dev of log probability difference: {raw_diff.std():.4f}",
        "",
        f"Particle INCREASED followup probability: {increased} ({increased/len(results_df)*100:.1f}%)",
        f"Particle DECREASED followup probability: {decreased} ({decreased/len(results_df)*100:.1f}%)",
        f"No change: {unchanged}",
        "",
        "--- RAW FOLLOWUP LOG PROBABILITIES (PER-TOKEN) ---",
        f"Average log probability WITH particle: {results_df['log_prob_followup_with_per_token'].mean():.4f}",
        f"Average log probability WITHOUT particle: {results_df['log_prob_followup_without_per_token'].mean():.4f}",
        f"Average log probability difference: {raw_diff_per_token.mean():.4f}",
        "  (positive = particle increases followup probability)",
        "",
        f"Median log probability difference: {raw_diff_per_token.median():.4f}",
        f"Std dev of log probability difference: {raw_diff_per_token.std():.4f}",
        "",
        f"Particle INCREASED followup probability: {increased_pt} ({increased_pt/len(results_df)*100:.1f}%)",
        f"Particle DECREASED followup probability: {decreased_pt} ({decreased_pt/len(results_df)*100:.1f}%)",
        f"No change: {unchanged_pt}",
        "",
        "--- FULL-SEQUENCE LOG PROBABILITIES ---",
        f"Average log probability WITH particle: {results_df['log_prob_full_with'].mean():.4f}",
        f"Average log probability WITHOUT particle: {results_df['log_prob_full_without'].mean():.4f}",
        f"Average log probability difference: {full_diff.mean():.4f}",
        "  (positive = particle increases full-sequence probability)",
        "",
        f"Median log probability difference: {full_diff.median():.4f}",
        f"Std dev of log probability difference: {full_diff.std():.4f}",
        "",
        f"Particle INCREASED full-sequence probability: {full_increased} ({full_increased/len(results_df)*100:.1f}%)",
        f"Particle DECREASED full-sequence probability: {full_decreased} ({full_decreased/len(results_df)*100:.1f}%)",
        f"No change: {full_unchanged}",
        "",
        "--- FULL-SEQUENCE LOG PROBABILITIES (PER-TOKEN) ---",
        f"Average log probability WITH particle: {results_df['log_prob_full_with_per_token'].mean():.4f}",
        f"Average log probability WITHOUT particle: {results_df['log_prob_full_without_per_token'].mean():.4f}",
        f"Average log probability difference: {full_diff_per_token.mean():.4f}",
        "  (positive = particle increases full-sequence probability)",
        "",
        f"Median log probability difference: {full_diff_per_token.median():.4f}",
        f"Std dev of log probability difference: {full_diff_per_token.std():.4f}",
        "",
        f"Particle INCREASED full-sequence probability: {full_increased_pt} ({full_increased_pt/len(results_df)*100:.1f}%)",
        f"Particle DECREASED full-sequence probability: {full_decreased_pt} ({full_decreased_pt/len(results_df)*100:.1f}%)",
        f"No change: {full_unchanged_pt}",
        *outlier_summary_lines,
        "=" * 80,
    ])
    return "\n".join(summary_lines)


def finalize_merged_results(
    merged_df: pd.DataFrame,
    include_context: bool,
    output_path: Path,
    observer_mode: Optional[bool] = None,
) -> None:
    """Recompute global outlier statistics and write summary files."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df, outlier_summary_lines = removal_outlier_detection(merged_df)
    summary = build_summary(
        results_df,
        include_context,
        outlier_summary_lines,
        observer_mode=observer_mode,
    )

    results_df.to_csv(output_path, sep="\t", index=False)
    rejected_iqr_path = output_path.with_name(output_path.stem + "_rejected_iqr.csv")
    results_df[results_df["removal_upper_iqr_outlier"]].to_csv(
        rejected_iqr_path,
        sep="\t",
        index=False,
    )
    summary_path = output_path.with_suffix(".txt")
    with open(summary_path, "w") as f:
        f.write(summary + "\n")
        f.write(f"IQR-rejected samples file: {rejected_iqr_path}\n")

    print("\n" + summary + "\n", flush=True)
    print(f"Results saved to: {output_path}", flush=True)
    print(f"IQR-rejected samples saved to: {rejected_iqr_path}", flush=True)
    print(f"Summary saved to: {summary_path}", flush=True)


def run_worker(
    model_name: str,
    shard_path: Path,
    shard_output_path: Path,
    gpu_id: int,
    include_context: bool,
    repo_root: Path,
    log_path: Path,
    experiment_script: str,
    observer_mode: bool,
    system_prompt: Optional[str],
) -> subprocess.Popen:
    """Launch one one-GPU worker process for a shard."""
    cmd = [
        sys.executable,
        experiment_script,
        "--model_name",
        model_name,
        "--data_path",
        str(shard_path),
        "--output_path",
        str(shard_output_path),
    ]
    if not include_context:
        cmd.append("--no_context")
    if observer_mode:
        cmd.append("--observer_mode")
    if system_prompt is not None:
        cmd.extend(["--system_prompt", system_prompt])

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONUNBUFFERED"] = "1"

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as log_file:
        return subprocess.Popen(
            cmd,
            cwd=repo_root,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )


def run_dataset_parallel(
    model_name: str,
    data_path: Path,
    output_dir: Path,
    gpu_ids: List[int],
    include_context: bool,
    repo_root: Path,
    experiment_script: str,
    observer_mode: bool,
    system_prompt: Optional[str],
) -> Path:
    """Run one dataset across multiple GPUs, then merge and finalize results."""
    shard_dir = output_dir / "shards" / data_path.stem
    shard_paths = split_tsv(data_path, len(gpu_ids), shard_dir)
    used_gpu_ids = gpu_ids[:len(shard_paths)]

    print(
        f"Processing {data_path} across GPUs {used_gpu_ids} with {len(shard_paths)} shard(s)",
        flush=True,
    )

    processes = []
    shard_output_paths = []
    log_paths = []
    model_name_short = model_name.split("/")[-1]

    for shard_idx, (gpu_id, shard_path) in enumerate(zip(used_gpu_ids, shard_paths)):
        shard_output_path = shard_dir / f"{model_name_short}_{data_path.stem}_shard{shard_idx}.csv"
        log_path = shard_dir / f"{model_name_short}_{data_path.stem}_shard{shard_idx}.log"
        proc = run_worker(
            model_name=model_name,
            shard_path=shard_path,
            shard_output_path=shard_output_path,
            gpu_id=gpu_id,
            include_context=include_context,
            repo_root=repo_root,
            log_path=log_path,
            experiment_script=experiment_script,
            observer_mode=observer_mode,
            system_prompt=system_prompt,
        )
        processes.append(proc)
        shard_output_paths.append(shard_output_path)
        log_paths.append(log_path)

    for proc, log_path in zip(processes, log_paths):
        return_code = proc.wait()
        if return_code != 0:
            raise RuntimeError(
                f"Shard worker failed with code {return_code}. See log: {log_path}"
            )

    merged_df = pd.concat(
        [pd.read_csv(path, sep="\t") for path in shard_output_paths],
        ignore_index=True,
    )
    context_tag = "context" if include_context else "noContext"
    model_prefix = f"{model_name_short}_Obs" if observer_mode else model_name_short
    final_output_path = output_dir / f"{model_prefix}_{context_tag}_{data_path.stem}_removal.csv"
    finalize_merged_results(
        merged_df,
        include_context=include_context,
        output_path=final_output_path,
        observer_mode=observer_mode if observer_mode else None,
    )
    cleanup_shard_artifacts(shard_dir, output_dir)
    return final_output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run followup probability experiments in parallel across GPUs."
    )
    parser.add_argument("--model_name", required=True, help="Hugging Face model name")
    parser.add_argument("--output_dir", required=True, help="Directory for merged outputs")
    parser.add_argument(
        "--experiment_script",
        default="particle_removal/calculate_followup_probabilities.py",
        help="Experiment entrypoint to run for each shard",
    )
    parser.add_argument(
        "--data_paths",
        nargs="+",
        required=True,
        help="One or more TSV datasets to process",
    )
    parser.add_argument(
        "--min_free_mem_mb",
        type=int,
        default=18000,
        help="Minimum free VRAM required to assign a full model replica to a GPU",
    )
    parser.add_argument(
        "--gpu_ids",
        nargs="+",
        type=int,
        default=None,
        help="Explicit GPU ids to use. If omitted, auto-detect based on free VRAM.",
    )
    parser.add_argument(
        "--no_context",
        action="store_true",
        help="Disable conversation context",
    )
    parser.add_argument(
        "--observer_mode",
        action="store_true",
        help="Pass observer mode through to the experiment script",
    )
    parser.add_argument(
        "--system_prompt",
        default=None,
        help="Optional system prompt to pass through to the experiment script",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.gpu_ids:
        gpu_ids = args.gpu_ids
    else:
        gpu_ids = detect_usable_gpus(args.min_free_mem_mb)
        if not gpu_ids:
            raise RuntimeError(
                f"No GPUs have at least {args.min_free_mem_mb} MiB free."
            )

    include_context = not args.no_context
    print(f"Usable GPUs: {gpu_ids}", flush=True)

    for data_path_str in args.data_paths:
        data_path = Path(data_path_str)
        run_dataset_parallel(
            model_name=args.model_name,
            data_path=data_path,
            output_dir=output_dir,
            gpu_ids=gpu_ids,
            include_context=include_context,
            repo_root=REPO_ROOT,
            experiment_script=args.experiment_script,
            observer_mode=args.observer_mode,
            system_prompt=args.system_prompt,
        )

    print("Parallel batch complete.", flush=True)


if __name__ == "__main__":
    main()
