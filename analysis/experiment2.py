#!/usr/bin/env python3
"""Analyze Section 5 paired-context generations and recreate the main summary."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.model_catalog import MODEL_SPECS

DELTA = "own_context_log_prob_advantage_per_token"
WORDS = ("not", "just", "only")
SOURCES = ("P", "P_prime")


EPILOG = (
    "Requires finished Experiment 2 candidates for all eight models, each "
    "covering all 180 items. See 'Running the analysis scripts' in the README."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, epilog=EPILOG)
    parser.add_argument("--results-root", default="results/experiment2")
    parser.add_argument("--data-stem", default="dailydialog_sense_subset")
    parser.add_argument("--output-dir", default="results/experiment2/analysis")
    parser.add_argument(
        "--expected-samples-per-condition",
        type=int,
        default=None,
        help=(
            "Candidates required per item and prompt condition. Defaults to the "
            "samples_requested value recorded in each candidate file. Use 0 to "
            "disable the candidate-count check."
        ),
    )
    return parser.parse_args()


def resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def candidate_path(results_root: Path, slug: str, data_stem: str) -> Path:
    matches = sorted((results_root / slug).glob(f"*_context_{data_stem}_generation_candidates.tsv"))
    if len(matches) != 1:
        found = ", ".join(str(path) for path in matches) or "<none>"
        raise FileNotFoundError(f"Expected one candidate file for {slug}; found {found}")
    return matches[0]


def requested_samples(frame: pd.DataFrame, path: Path) -> int:
    """Return the per-condition sample count recorded at generation time."""
    if "samples_requested" not in frame.columns:
        raise KeyError(
            f"{path} has no samples_requested column; "
            "pass --expected-samples-per-condition explicitly."
        )
    values = pd.to_numeric(frame["samples_requested"], errors="raise").unique()
    if len(values) != 1:
        raise ValueError(f"{path} mixes samples_requested values: {sorted(values)}.")
    return int(values[0])


def main() -> None:
    args = parse_args()
    results_root = resolve(args.results_root)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for model_order, spec in enumerate(MODEL_SPECS):
        path = candidate_path(results_root, spec.slug, args.data_stem)
        frame = pd.read_csv(path, sep="\t", dtype={"id": str})
        required = {"id", "word", "generated_from", DELTA}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise KeyError(f"{path} is missing: {', '.join(missing)}")
        if frame["id"].nunique() != 180:
            raise ValueError(f"Expected 180 stimulus items in {path}.")
        if set(frame["word"].dropna()) != set(WORDS):
            raise ValueError(f"Unexpected particle coverage in {path}.")
        if set(frame["generated_from"].dropna()) != set(SOURCES):
            raise ValueError(f"Unexpected source conditions in {path}.")
        expected = args.expected_samples_per_condition
        if expected is None:
            expected = requested_samples(frame, path)
        if expected:
            counts = frame.groupby(["id", "generated_from"]).size()
            if not (counts == expected).all():
                bad = counts.loc[counts != expected]
                raise ValueError(
                    f"{path} has {len(bad)} item/source groups with counts other than {expected}."
                )
        frame["model"] = spec.label
        frame["model_slug"] = spec.slug
        frame["family"] = spec.family
        frame["tuning"] = spec.tuning
        frame["model_order"] = model_order
        frame[DELTA] = pd.to_numeric(frame[DELTA], errors="raise")
        frames.append(frame)

    candidates = pd.concat(frames, ignore_index=True)
    summary = candidates.groupby(
        ["model_order", "model_slug", "model", "family", "tuning", "word", "generated_from"],
        as_index=False,
        sort=False,
    ).agg(
        n_candidates=(DELTA, "size"),
        mean_own_context_advantage_per_token=(DELTA, "mean"),
        sd_own_context_advantage_per_token=(DELTA, "std"),
    )
    summary["se_own_context_advantage_per_token"] = (
        summary["sd_own_context_advantage_per_token"] / np.sqrt(summary["n_candidates"])
    )
    win_rates = (
        candidates.assign(own_context_win=candidates[DELTA] > 0)
        .groupby(["model_slug", "word", "generated_from"], as_index=False)["own_context_win"]
        .mean()
        .rename(columns={"own_context_win": "own_context_win_rate"})
    )
    summary = summary.merge(win_rates, on=["model_slug", "word", "generated_from"], validate="one_to_one")
    summary.sort_values(["model_order", "word", "generated_from"], inplace=True)
    summary.to_csv(output_dir / "candidate_level_summary.tsv", sep="\t", index=False)

    item_summary = (
        candidates.groupby(
            ["model_order", "model_slug", "model", "family", "tuning", "word", "generated_from", "id"],
            as_index=False,
        )[DELTA]
        .mean()
        .rename(columns={DELTA: "item_mean_own_context_advantage_per_token"})
    )
    item_summary.to_csv(output_dir / "item_level_summary.tsv", sep="\t", index=False)

    print(f"Wrote Experiment 2 analyses to {output_dir}")


if __name__ == "__main__":
    main()
