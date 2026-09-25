# Released data

This directory contains the two data files used by the published experiments.

## `dailydialog_sense_subset.tsv`

The 180 manually selected DailyDialog items used in Experiments 1–3. There are
60 rows each for `just`, `only`, and `not`. The `just` items contain 30
exclusive and 30 non-exclusive uses.

Columns:

- `id`: stable item ID. `just_e_` and `just_j_` identify exclusive and
  non-exclusive `just`, respectively.
- `word`: `just`, `only`, or the truth-conditional control `not`.
- `sense`: explicit sense/category label.
- `context`: up to three preceding turns, separated by `__eou__`.
- `w_word`: observed utterance containing the target word.
- `wo_word`: minimally edited utterance with the target word removed.
- `followup`: observed next turn.

Counts and SHA-256 checksums are verified by
`python scripts/validate_release_data.py`.

This file is derived from DailyDialog (Li et al., 2017). DailyDialog is
distributed as CC BY-NC-SA 4.0; the derivative subset is provided under the
same terms. See [DATA_LICENSE.md](../DATA_LICENSE.md).

## `human_annotations.csv`

The complete Section 4 preference data: 9 anonymized participants × 240 items
(60 `just`, 60 `only`, 60 `not`, and 60 word-order distractors), for 2,160
rows total.

Columns:

- `participant_id`: anonymized integer ID from 1 through 9.
- `comparison_id`: stimulus or distractor ID.
- `prefers_w_word`: `1` if the response favors the original passage, `-1` if
  it favors the edited passage, and `0` for a neutral judgment.
- `preference_strength`: the 1–7 Likert response, with 4 neutral and 7 the
  strongest preference for the original passage.
- `word`: `just`, `only`, `not`, or `control` for distractors.

The control passages themselves are not needed by the computational
experiments and are not included. Their judgments remain in the annotation
file so the reported annotator-validity and agreement checks are reproducible.

## Citation

Yanran Li, Hui Su, Xiaoyu Shen, Wenjie Li, Ziqiang Cao, and Shuzi Niu. 2017.
“DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset.” In *Proceedings
of IJCNLP 2017*, pages 986–995.
