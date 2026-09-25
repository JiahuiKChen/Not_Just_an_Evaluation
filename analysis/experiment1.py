#!/usr/bin/env python3
"""Analyze Section 4 human and language-model particle-removal preferences."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.model_catalog import MODEL_SPECS

WORDS = ("not", "just", "only")
DEFAULT_SCORE = "full-dialogue"
SCORE_COLUMN = "analysis_score"
SCORE_FORMULAS = {
    "full-dialogue": (
        "log_prob_full_with_per_token - log_prob_full_without_per_token"
    ),
    "sentence-removal": "removal_log_prob_delta_per_token",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results/experiment1")
    parser.add_argument("--human-data", default="data/human_annotations.csv")
    parser.add_argument("--stimuli", default="data/dailydialog_sense_subset.tsv")
    parser.add_argument("--output-dir", default="results/experiment1/analysis")
    parser.add_argument(
        "--score",
        choices=tuple(SCORE_FORMULAS),
        default=DEFAULT_SCORE,
        help=(
            "Model preference score. The default is the paper's token-normalized "
            "full-dialogue difference, log P(CSF) - log P(CS'F)."
        ),
    )
    return parser.parse_args()


def resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def zscore(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise").astype(float)
    sd = float(numeric.std(ddof=0))
    if sd == 0:
        raise ValueError("Cannot z-score a constant preference vector.")
    return (numeric - float(numeric.mean())) / sd


def find_score_file(results_root: Path, slug: str, data_stem: str) -> Path:
    matches = sorted((results_root / slug).glob(f"*_context_{data_stem}_removal.csv"))
    if len(matches) != 1:
        found = ", ".join(str(path) for path in matches) or "<none>"
        raise FileNotFoundError(f"Expected one score file for {slug}; found {found}")
    return matches[0]


def model_scores(frame: pd.DataFrame, score: str = DEFAULT_SCORE) -> pd.DataFrame:
    """Return item IDs, particles, and the selected model preference score."""
    if score == "full-dialogue":
        source_columns = {
            "log_prob_full_with_per_token",
            "log_prob_full_without_per_token",
        }
    elif score == "sentence-removal":
        source_columns = {"removal_log_prob_delta_per_token"}
    else:
        raise ValueError(f"Unknown Experiment 1 score: {score}")

    required = {"id", "word", *source_columns}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError("Model result file is missing: " + ", ".join(missing))

    scored = frame.loc[:, ["id", "word", *sorted(source_columns)]].copy()
    if score == "full-dialogue":
        scored[SCORE_COLUMN] = pd.to_numeric(
            scored["log_prob_full_with_per_token"], errors="raise"
        ) - pd.to_numeric(
            scored["log_prob_full_without_per_token"], errors="raise"
        )
    else:
        scored[SCORE_COLUMN] = pd.to_numeric(
            scored["removal_log_prob_delta_per_token"], errors="raise"
        )
    return scored.loc[:, ["id", "word", SCORE_COLUMN]]


def icc_absolute_average(matrix: np.ndarray) -> float:
    """McGraw-Wong ICC(A,k) for a complete target-by-rater matrix."""
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or np.isnan(values).any():
        raise ValueError("ICC requires a complete two-dimensional ratings matrix.")
    n_targets, n_raters = values.shape
    grand = values.mean()
    row_means = values.mean(axis=1)
    col_means = values.mean(axis=0)
    ms_rows = n_raters * np.square(row_means - grand).sum() / (n_targets - 1)
    ms_cols = n_targets * np.square(col_means - grand).sum() / (n_raters - 1)
    residual = values - row_means[:, None] - col_means[None, :] + grand
    ms_error = np.square(residual).sum() / ((n_targets - 1) * (n_raters - 1))
    return float((ms_rows - ms_error) / (ms_rows + (ms_cols - ms_error) / n_targets))


def load_human(path: Path, stimulus_ids: set[str]) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    human = pd.read_csv(path, dtype={"participant_id": str, "comparison_id": str})
    required = {
        "participant_id",
        "comparison_id",
        "prefers_w_word",
        "preference_strength",
        "word",
    }
    missing = sorted(required - set(human.columns))
    if missing:
        raise KeyError(f"{path} is missing: {', '.join(missing)}")

    expected_per_participant = human.groupby("participant_id").size()
    if len(expected_per_participant) != 9 or not (expected_per_participant == 240).all():
        raise ValueError("Expected 9 participants with 240 judgments each.")

    pivot = human.pivot(
        index="comparison_id",
        columns="participant_id",
        values="preference_strength",
    )
    icc = icc_absolute_average(pivot.to_numpy(dtype=float))

    target = human.loc[human["comparison_id"].isin(stimulus_ids)].copy()
    if len(target) != 9 * 180:
        raise ValueError(f"Expected 1,620 target-word human judgments, found {len(target)}.")
    target["preference_z"] = target.groupby("participant_id", group_keys=False)[
        "preference_strength"
    ].transform(zscore)
    item = (
        target.groupby(["comparison_id", "word"], as_index=False)
        .agg(
            human_preference=("preference_strength", "mean"),
            human_preference_z=("preference_z", "mean"),
        )
        .rename(columns={"comparison_id": "id"})
    )
    return human, item, icc


def preference_test(values: pd.Series) -> tuple[float, float]:
    array = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if not len(array) or np.allclose(array, 0):
        return float("nan"), float("nan")
    result = stats.wilcoxon(array, alternative="greater", zero_method="wilcox")
    return float(result.statistic), float(result.pvalue)


def main() -> None:
    args = parse_args()
    results_root = resolve(args.results_root)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stimuli = pd.read_csv(resolve(args.stimuli), sep="\t", dtype={"id": str})
    if len(stimuli) != 180 or stimuli["id"].duplicated().any():
        raise ValueError("The canonical stimulus file must contain 180 unique items.")
    if stimuli["word"].value_counts().to_dict() != {"just": 60, "only": 60, "not": 60}:
        raise ValueError("Expected 60 stimuli for each of just, only, and not.")

    human, human_items, icc = load_human(
        resolve(args.human_data), set(stimuli["id"].astype(str))
    )
    human_items["judge"] = "Human"
    human_items["judge_type"] = "human"
    human_items["preference_z"] = human_items["human_preference_z"]

    long_frames = [human_items[["id", "word", "judge", "judge_type", "preference_z"]]]
    correlations = []
    for spec in MODEL_SPECS:
        path = find_score_file(results_root, spec.slug, resolve(args.stimuli).stem)
        raw_frame = pd.read_csv(path, sep="\t", dtype={"id": str})
        try:
            frame = model_scores(raw_frame, score=args.score)
        except KeyError as error:
            raise KeyError(f"{path}: {error}") from error
        if len(frame) != 180 or frame["id"].duplicated().any():
            raise ValueError(f"Expected 180 unique model scores in {path}.")
        frame["preference_z"] = zscore(frame[SCORE_COLUMN])
        frame["judge"] = spec.label
        frame["judge_type"] = spec.tuning
        long_frames.append(frame[["id", "word", "judge", "judge_type", "preference_z"]])

        merged = human_items.merge(frame, on=["id", "word"], validate="one_to_one")
        for word in ("combined", *WORDS):
            subset = merged if word == "combined" else merged.loc[merged["word"] == word]
            rho, p_value = stats.spearmanr(subset["human_preference"], subset[SCORE_COLUMN])
            correlations.append(
                {
                    "model": spec.label,
                    "model_slug": spec.slug,
                    "tuning": spec.tuning,
                    "word": word,
                    "n_items": len(subset),
                    "spearman_rho": float(rho),
                    "p_value": float(p_value),
                }
            )

    preferences = pd.concat(long_frames, ignore_index=True)
    preferences.insert(0, "score", args.score)
    preferences.to_csv(output_dir / "preference_scores_z.tsv", sep="\t", index=False)
    correlation_df = pd.DataFrame(correlations)
    correlation_df.insert(0, "score", args.score)
    correlation_df.to_csv(output_dir / "human_model_correlations.tsv", sep="\t", index=False)

    summary_rows = []
    for keys, group in preferences.groupby(["judge", "judge_type", "word"], sort=False):
        statistic, p_value = preference_test(group["preference_z"])
        summary_rows.append(
            {
                "judge": keys[0],
                "judge_type": keys[1],
                "word": keys[2],
                "n_items": len(group),
                "mean_z": float(group["preference_z"].mean()),
                "se_z": float(group["preference_z"].std(ddof=1) / np.sqrt(len(group))),
                "wilcoxon_w": statistic,
                "p_value_greater": p_value,
                "p_value_bonferroni_3": min(1.0, p_value * 3) if np.isfinite(p_value) else np.nan,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.insert(0, "score", args.score)
    summary.to_csv(output_dir / "preference_summary.tsv", sep="\t", index=False)

    pd.DataFrame(
        [{"score": args.score, "formula": SCORE_FORMULAS[args.score]}]
    ).to_csv(output_dir / "score_metadata.tsv", sep="\t", index=False)

    raw_summary = (
        human.groupby("word", as_index=False)
        .agg(
            n=("preference_strength", "size"),
            mean=("preference_strength", "mean"),
            median=("preference_strength", "median"),
            sd=("preference_strength", "std"),
        )
    )
    raw_summary["icc_a_k_all_240_items"] = icc
    raw_summary.to_csv(output_dir / "human_annotation_summary.tsv", sep="\t", index=False)

    print(f"Human agreement ICC(A,k): {icc:.3f}")
    print(f"Model preference score: {args.score} ({SCORE_FORMULAS[args.score]})")
    print(f"Wrote Experiment 1 analyses to {output_dir}")


if __name__ == "__main__":
    main()
