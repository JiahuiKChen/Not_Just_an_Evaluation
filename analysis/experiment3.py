#!/usr/bin/env python3
"""Summarize Section 6 native localization and sense-transfer effects."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = ("attn", "mlp", "resid")
TOP_KS = (1, 3, 5, 10)
PARTICLES = ("not", "just", "only")
TRANSFER_LABELS = (
    "exclusive_just_to_only",
    "nonexclusive_just_to_only",
    "only_to_exclusive_just",
    "only_to_nonexclusive_just",
)


EPILOG = (
    "Requires every output from scripts/run_experiment3.py. "
    "See 'Running the analysis scripts' in the README."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, epilog=EPILOG)
    parser.add_argument("--results-root", default="results/experiment3")
    parser.add_argument("--output-dir", default="results/experiment3/analysis")
    return parser.parse_args()


def resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep="\t")


def filtered_summary(path: Path) -> pd.DataFrame:
    frame = read_tsv(path)
    required = {
        "component",
        "top_k",
        "eval_mode",
        "target_type",
        "site_set_type",
        "patched_metric_change_mean",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"{path} is missing: {', '.join(missing)}")
    frame = frame.loc[
        (frame["eval_mode"] == "necessity")
        & (frame["target_type"] == "generated")
        & frame["site_set_type"].isin(["localized", "random_mean"])
    ].copy()
    return frame


def native_results(root: Path) -> pd.DataFrame:
    rows = []
    for particle in PARTICLES:
        patch_dir = root / "localization" / "native" / particle / "component_patching"
        summary = filtered_summary(patch_dir / "eval_summary.tsv")
        significance = read_tsv(patch_dir / "eval_significance.tsv")
        significance = significance.loc[
            (significance["eval_mode"] == "necessity")
            & (significance["target_type"] == "generated")
            & (significance["metric"] == "effect_score")
        ].copy()
        sig_index = significance.set_index(["component", "top_k"])
        for component in COMPONENTS:
            for top_k in TOP_KS:
                subset = summary.loc[
                    (summary["component"] == component) & (summary["top_k"] == top_k)
                ].set_index("site_set_type")
                if set(subset.index) != {"localized", "random_mean"}:
                    raise ValueError(f"Incomplete native summary for {particle}/{component}/k={top_k}")
                p_value = float(sig_index.loc[(component, top_k), "p_value_greater"])
                rows.append(
                    {
                        "particle": particle,
                        "component": component,
                        "top_k": top_k,
                        "localized_delta_patch": float(
                            subset.loc["localized", "patched_metric_change_mean"]
                        ),
                        "random_delta_patch": float(
                            subset.loc["random_mean", "patched_metric_change_mean"]
                        ),
                        "localized_vs_random_p_value": p_value,
                        "localized_vs_random_p_bonferroni_36": min(1.0, p_value * 36),
                    }
                )
    return pd.DataFrame(rows)


def transfer_paths(root: Path, label: str) -> tuple[Path, Path]:
    patch_dir = root / "transfers" / label / "component_patching"
    return patch_dir / "eval_summary.tsv", patch_dir / "eval_item_summary.tsv"


def localized_items(path: Path) -> pd.DataFrame:
    frame = read_tsv(path)
    return frame.loc[
        (frame["eval_mode"] == "necessity")
        & (frame["target_type"] == "generated")
        & (frame["site_set_type"] == "localized")
    ].copy()


def transfer_results(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    means: dict[str, pd.Series] = {}
    items: dict[str, pd.DataFrame] = {}
    for label in TRANSFER_LABELS:
        summary_path, item_path = transfer_paths(root, label)
        summary = filtered_summary(summary_path)
        summary = summary.loc[summary["site_set_type"] == "localized"]
        means[label] = summary.set_index(["component", "top_k"])["patched_metric_change_mean"]
        items[label] = localized_items(item_path)

    joined = pd.concat(means, axis=1).reset_index()
    joined["just_to_only_delta_sense"] = (
        joined["nonexclusive_just_to_only"] - joined["exclusive_just_to_only"]
    )
    joined["only_to_just_delta_sense"] = (
        joined["only_to_nonexclusive_just"] - joined["only_to_exclusive_just"]
    )

    tests = []
    pair_keys = [
        "component",
        "top_k",
        "train_fold",
        "eval_fold",
        "target_type",
        "source_row_index",
        "id",
    ]
    for component in COMPONENTS:
        for top_k in TOP_KS:
            exclusive = items["exclusive_just_to_only"].loc[
                (items["exclusive_just_to_only"]["component"] == component)
                & (items["exclusive_just_to_only"]["top_k"] == top_k)
            ]
            nonexclusive = items["nonexclusive_just_to_only"].loc[
                (items["nonexclusive_just_to_only"]["component"] == component)
                & (items["nonexclusive_just_to_only"]["top_k"] == top_k)
            ]
            paired = nonexclusive.merge(
                exclusive,
                on=pair_keys,
                suffixes=("_nonexclusive", "_exclusive"),
                validate="one_to_one",
            )
            paired_difference = (
                paired["patched_metric_change_nonexclusive"]
                - paired["patched_metric_change_exclusive"]
            ).to_numpy(dtype=float)
            if np.allclose(paired_difference, 0):
                paired_statistic, paired_p_value = 0.0, 1.0
            else:
                paired_test = stats.wilcoxon(
                    paired_difference,
                    alternative="two-sided",
                    zero_method="wilcox",
                )
                paired_statistic = float(paired_test.statistic)
                paired_p_value = float(paired_test.pvalue)
            tests.append(
                {
                    "direction": "just_to_only",
                    "component": component,
                    "top_k": top_k,
                    "test": "paired_wilcoxon",
                    "n_nonexclusive": len(paired),
                    "n_exclusive": len(paired),
                    "statistic": paired_statistic,
                    "p_value_two_sided": paired_p_value,
                }
            )

            only_to_exclusive = items["only_to_exclusive_just"].loc[
                (items["only_to_exclusive_just"]["component"] == component)
                & (items["only_to_exclusive_just"]["top_k"] == top_k)
            ]["patched_metric_change"].to_numpy(dtype=float)
            only_to_nonexclusive = items["only_to_nonexclusive_just"].loc[
                (items["only_to_nonexclusive_just"]["component"] == component)
                & (items["only_to_nonexclusive_just"]["top_k"] == top_k)
            ]["patched_metric_change"].to_numpy(dtype=float)
            independent_test = stats.ttest_ind(
                only_to_nonexclusive,
                only_to_exclusive,
                equal_var=False,
                alternative="two-sided",
            )
            tests.append(
                {
                    "direction": "only_to_just",
                    "component": component,
                    "top_k": top_k,
                    "test": "welch_t",
                    "n_nonexclusive": len(only_to_nonexclusive),
                    "n_exclusive": len(only_to_exclusive),
                    "statistic": float(independent_test.statistic),
                    "p_value_two_sided": float(independent_test.pvalue),
                }
            )

    test_df = pd.DataFrame(tests)
    test_df["p_value_bonferroni_12_within_direction"] = np.minimum(
        1.0, test_df["p_value_two_sided"] * 12
    )
    return joined, test_df


def paper_table(native: pd.DataFrame, transfer: pd.DataFrame) -> pd.DataFrame:
    rows = []
    native_index = native.set_index(["particle", "component", "top_k"])
    transfer_index = transfer.set_index(["component", "top_k"])
    for component in COMPONENTS:
        for top_k in TOP_KS:
            row = {"component": component, "top_k": top_k}
            for particle in PARTICLES:
                values = native_index.loc[(particle, component, top_k)]
                row[f"{particle}_localized_x100"] = 100 * values["localized_delta_patch"]
                row[f"{particle}_random_x100"] = 100 * values["random_delta_patch"]
                row[f"{particle}_localized_vs_random_p_bonferroni_36"] = values[
                    "localized_vs_random_p_bonferroni_36"
                ]
            transfer_values = transfer_index.loc[(component, top_k)]
            row["just_to_only_delta_sense_x100"] = (
                100 * transfer_values["just_to_only_delta_sense"]
            )
            row["only_to_just_delta_sense_x100"] = (
                100 * transfer_values["only_to_just_delta_sense"]
            )
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    results_root = resolve(args.results_root)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    native = native_results(results_root)
    transfer, transfer_tests = transfer_results(results_root)
    table = paper_table(native, transfer)

    native.to_csv(output_dir / "native_localization_summary.tsv", sep="\t", index=False)
    transfer.to_csv(output_dir / "sense_transfer_summary.tsv", sep="\t", index=False)
    transfer_tests.to_csv(output_dir / "sense_transfer_significance.tsv", sep="\t", index=False)
    table.to_csv(output_dir / "table1_values.tsv", sep="\t", index=False, float_format="%.8g")
    (output_dir / "table1_values.tex").write_text(
        table.round(4).to_latex(index=False, escape=True), encoding="utf-8"
    )
    print(f"Wrote Experiment 3 analyses to {output_dir}")


if __name__ == "__main__":
    main()
