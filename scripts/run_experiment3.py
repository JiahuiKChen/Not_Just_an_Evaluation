#!/usr/bin/env python3
"""Run Section 6 component localization and just/only sense transfer."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-path",
        default=None,
        help="Experiment 2 candidates for Llama 3 8B Instruct; inferred when omitted.",
    )
    parser.add_argument("--output-root", default="results/experiment3")
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--top-k-pool", type=int, default=500)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def infer_candidate_path() -> Path:
    result_dir = REPO_ROOT / "results" / "experiment2" / "llama3_instruct"
    expected = result_dir / (
        "Meta-Llama-3-8B-Instruct_context_"
        "dailydialog_sense_subset_generation_candidates.tsv"
    )
    if expected.exists():
        return expected
    matches = sorted(result_dir.glob("*_generation_candidates.tsv"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(
        f"Could not uniquely infer Experiment 2 candidates under {result_dir}. "
        "Pass --candidate-path explicitly."
    )


def run(
    command: list[str],
    *,
    marker: Path | None,
    args: argparse.Namespace,
    env: dict[str, str],
) -> None:
    if args.skip_existing and marker is not None and marker.exists():
        print(f"[skip] {marker.relative_to(REPO_ROOT)}", flush=True)
        return
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def main() -> None:
    args = parse_args()
    if args.candidate_path:
        candidates = Path(args.candidate_path)
        if not candidates.is_absolute():
            candidates = REPO_ROOT / candidates
    elif args.dry_run:
        candidates = (
            REPO_ROOT
            / "results/experiment2/llama3_instruct/"
            "Meta-Llama-3-8B-Instruct_context_dailydialog_sense_subset_generation_candidates.tsv"
        )
    else:
        candidates = infer_candidate_path()
    if not args.dry_run and not candidates.exists():
        raise FileNotFoundError(candidates)

    root = Path(args.output_root)
    if not root.is_absolute():
        root = REPO_ROOT / root
    pools = root / "pools"
    localizations = root / "localization"
    transfers = root / "transfers"

    env = os.environ.copy()
    env.setdefault("JUST_LLMS_DEVICE_MAP", "auto")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    if args.gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    groups = (
        ("native", ["not", "just", "only"], None),
        ("exclusive_just", ["just"], "just_e_"),
        ("nonexclusive_just", ["just"], "just_j_"),
    )

    for name, particles, id_prefix in groups:
        command = [
            sys.executable,
            "localization/build_particle_generation_pool.py",
            "--model_name",
            MODEL_NAME,
            "--particles",
            *particles,
            "--candidate_path",
            str(candidates),
            "--top_k",
            str(args.top_k_pool),
            "--num_folds",
            "2",
            "--seed",
            "13",
            "--selection_mode",
            "top_positive_k",
            "--output_root",
            str(pools / name),
        ]
        if id_prefix:
            command.extend(["--candidate_id_prefix", id_prefix])
        markers = [pools / name / particle / "generation_pool.tsv" for particle in particles]
        marker = markers[0] if len(markers) == 1 else None
        if args.skip_existing and all(path.exists() for path in markers):
            print(f"[skip] all pools for {name}", flush=True)
        else:
            run(command, marker=marker, args=args, env=env)

    for name, particles, _ in groups:
        output_dir = localizations / name
        localization_markers = [
            output_dir / particle / "component_patching/site_scores.tsv"
            for particle in particles
        ]
        command = [
            sys.executable,
            "localization/localize_component_prompt_boundary.py",
            "--model_name",
            MODEL_NAME,
            "--particles",
            *particles,
            "--components",
            "resid,attn,mlp",
            "--batch_size",
            str(args.batch_size),
            "--trace_rows_per_source",
            "1",
            "--position_label",
            "utterance_final",
            "--pool_root",
            str(pools / name),
            "--output_root",
            str(output_dir),
        ]
        if args.skip_existing and all(path.exists() for path in localization_markers):
            print(f"[skip] localization for {name}", flush=True)
        else:
            run(command, marker=None, args=args, env=env)

        evaluation_markers = [
            output_dir / particle / "component_patching/eval_summary.tsv"
            for particle in particles
        ]
        command = [
            sys.executable,
            "localization/evaluate_component_prompt_boundary.py",
            "--model_name",
            MODEL_NAME,
            "--particles",
            *particles,
            "--top_ks",
            "1,3,5,10",
            "--random_seeds",
            "0,1,2,3",
            "--eval_modes",
            "necessity",
            "--batch_size",
            str(args.batch_size),
            "--output_root",
            str(output_dir),
            "--analysis_bootstrap_replicates",
            "5000",
            "--analysis_bootstrap_seed",
            "0",
            "--analysis_signflip_replicates",
            "20000",
        ]
        if args.skip_existing and all(path.exists() for path in evaluation_markers):
            print(f"[skip] held-out evaluation for {name}", flush=True)
        else:
            run(command, marker=None, args=args, env=env)

    transfer_specs = (
        (
            "exclusive_just_to_only",
            "exclusive_just",
            "only",
            localizations / "exclusive_just/just/component_patching/site_scores.tsv",
            pools / "native/only/generation_pool.tsv",
        ),
        (
            "nonexclusive_just_to_only",
            "nonexclusive_just",
            "only",
            localizations / "nonexclusive_just/just/component_patching/site_scores.tsv",
            pools / "native/only/generation_pool.tsv",
        ),
        (
            "only_to_exclusive_just",
            "only",
            "exclusive_just",
            localizations / "native/only/component_patching/site_scores.tsv",
            pools / "exclusive_just/just/generation_pool.tsv",
        ),
        (
            "only_to_nonexclusive_just",
            "only",
            "nonexclusive_just",
            localizations / "native/only/component_patching/site_scores.tsv",
            pools / "nonexclusive_just/just/generation_pool.tsv",
        ),
    )
    for label, source_label, target_label, site_scores, target_pool in transfer_specs:
        command = [
            sys.executable,
            "localization/evaluate_component_site_transfer.py",
            "--model_name",
            MODEL_NAME,
            "--source_site_scores",
            str(site_scores),
            "--target_pool",
            str(target_pool),
            "--transfer_label",
            label,
            "--source_label",
            source_label,
            "--target_label",
            target_label,
            "--output_root",
            str(transfers),
            "--top_ks",
            "1,3,5,10",
            "--random_seeds",
            "0,1,2,3",
            "--eval_modes",
            "necessity",
            "--batch_size",
            str(args.batch_size),
            "--analysis_bootstrap_replicates",
            "5000",
            "--analysis_bootstrap_seed",
            "0",
            "--analysis_signflip_replicates",
            "20000",
        ]
        marker = transfers / label / "component_patching/eval_summary.tsv"
        run(command, marker=marker, args=args, env=env)


if __name__ == "__main__":
    main()
