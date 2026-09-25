#!/usr/bin/env python3
"""Run Section 4 particle-removal scoring for the eight paper checkpoints."""

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
    parser.add_argument(
        "--data-path",
        default="data/dailydialog_sense_subset.tsv",
        help="The 180-item paper stimulus set.",
    )
    parser.add_argument("--output-root", default="results/experiment1")
    parser.add_argument("--gpu-ids", nargs="+", type=int, default=None)
    parser.add_argument("--min-free-mem-mb", type=int, default=18000)
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
        expected = output_dir / (
            f"{spec.checkpoint.rsplit('/', 1)[-1]}_context_{data_path.stem}_removal.csv"
        )
        if args.skip_existing and expected.exists():
            print(f"[skip] {expected.relative_to(REPO_ROOT)}", flush=True)
            continue

        scorer = (
            "particle_removal/calculate_followup_probabilities_instruct.py"
            if spec.prompt_mode == "chat"
            else "particle_removal/calculate_followup_probabilities.py"
        )
        command = [
            sys.executable,
            "particle_removal/run_parallel_probability_experiment.py",
            "--model_name",
            spec.checkpoint,
            "--output_dir",
            str(output_dir),
            "--experiment_script",
            scorer,
            "--data_paths",
            str(data_path),
            "--min_free_mem_mb",
            str(args.min_free_mem_mb),
        ]
        if args.gpu_ids:
            command.extend(["--gpu_ids", *(str(value) for value in args.gpu_ids)])

        print(f"\n[{spec.slug}] {shlex.join(command)}", flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
