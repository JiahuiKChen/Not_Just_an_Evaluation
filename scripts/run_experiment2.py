#!/usr/bin/env python3
"""Run Section 5 paired-context generation for the eight paper checkpoints."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.model_catalog import select_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["all"])
    parser.add_argument("--data-path", default="data/dailydialog_sense_subset.tsv")
    parser.add_argument("--output-root", default="results/experiment2")
    parser.add_argument("--gpu-ids", nargs="+", type=int, default=None)
    parser.add_argument("--min-free-mem-mb", type=int, default=18000)
    parser.add_argument("--samples-per-condition", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--base-seed", type=int, default=20260416)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = select_models(args.models)
    data_path = Path(args.data_path)
    if not data_path.is_absolute():
        data_path = REPO_ROOT / data_path
    if not args.dry_run and not data_path.exists():
        raise FileNotFoundError(data_path)

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root

    for spec in models:
        output_dir = output_root / spec.slug
        sample_batch_size = 32 if spec.prompt_mode == "chat" else 64
        max_sample_draws = 500 if spec.prompt_mode == "chat" else 200
        command = [
            sys.executable,
            "generation/run_parallel_generation_experiment.py",
            "--model_name",
            spec.checkpoint,
            "--output_dir",
            str(output_dir),
            "--prompt_mode",
            spec.prompt_mode,
            "--data_paths",
            str(data_path),
            "--samples_per_condition",
            str(args.samples_per_condition),
            "--top_p",
            str(args.top_p),
            "--temperature",
            str(args.temperature),
            "--sample_batch_size",
            str(sample_batch_size),
            "--max_sample_draws",
            str(max_sample_draws),
            "--max_new_tokens",
            str(args.max_new_tokens),
            "--base_seed",
            str(args.base_seed),
            "--min_free_mem_mb",
            str(args.min_free_mem_mb),
        ]
        if args.gpu_ids:
            command.extend(["--gpu_ids", *(str(value) for value in args.gpu_ids)])
        if args.skip_existing:
            command.append("--skip_existing")

        print(f"\n[{spec.slug}] {shlex.join(command)}", flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
