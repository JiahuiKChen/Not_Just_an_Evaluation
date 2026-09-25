# Not Just an Evaluation

This repository accompanies the paper *Not Just an Evaluation: A Case Study
of Discourse Particle Sensitivity in Language Models*. It contains the
released dialogue stimuli, human preference annotations, and code for all
three experiments in the paper.

## Table of contents

- [Overview](#overview)
- [Released data](#released-data)
  - [Dialogue stimuli](#dialogue-stimuli)
  - [Human annotations](#human-annotations)
- [Repository layout](#repository-layout)
- [Setup](#setup)
- [Models](#models)
- [Reproducing the experiments](#reproducing-the-experiments)
  - [Experiment 1: Particle removal](#experiment-1-particle-removal)
  - [Experiment 2: Paired-context generation](#experiment-2-paired-context-generation)
  - [Experiment 3: Component localization](#experiment-3-component-localization)
  - [Running the analysis scripts](#running-the-analysis-scripts)
  - [Outputs and resuming](#outputs-and-resuming)
- [Data licensing](#data-licensing)
- [Citation](#citation)

## Overview

Discourse particles such as *just* and *only* subtly shape how a conversation
can continue, often without changing its truth-conditional content. The paper
studies whether language models are sensitive to these effects in natural
conversations from DailyDialog.

The repository supports the paper's three main experiments:

1. **Particle removal (Section 4):** compare the probability of an observed
   dialogue with and without `just`, `only`, or the control `not`, and compare
   language-model preferences with human judgments.
2. **Generation (Section 5):** sample continuations from both members of a
   minimal prompt pair and measure whether each response is more probable in
   the context from which it was generated.
3. **Localization (Section 6):** localize particle-sensitive attention, MLP,
   and residual-stream sites in Llama 3 8B Instruct, evaluate them on held-out
   items, and test sense-specific transfer between `just` and `only`.

The main results show that models and humans have similar preferences for
particle-bearing conversations, instruction tuning generally increases model
sensitivity, and the localized overlap between `just` and `only` is
asymmetric and partly organized by sense.

All commands below are run from the repository root.

## Released data

The [`data/`](data/) directory contains the study's dialogue stimuli and
complete anonymized human annotations. Full column definitions and file-level
details are available in [`data/README.md`](data/README.md).

| File | Contents | Size |
|---|---|---:|
| [`dailydialog_sense_subset.tsv`](data/dailydialog_sense_subset.tsv) | Sense-labeled dialogue stimuli used in Experiments 1–3 | 180 items |
| [`human_annotations.csv`](data/human_annotations.csv) | Human preferences for target items and word-order distractors | 2,160 judgments |

Run `python scripts/validate_release_data.py` to verify both files' schemas,
row counts, and checksums.

### Dialogue stimuli

The stimulus dataset contains 180 manually selected DailyDialog examples: 60
each for `just`, `only`, and the truth-conditional control `not`. The `just`
items are further labeled as 30 exclusive and 30 non-exclusive uses.

Each item includes up to three preceding dialogue turns, the original
utterance containing the target word, a minimally edited version with that
word removed, and the naturally occurring next turn. These fields provide the
context, minimal pair, and ground-truth continuation used throughout the
paper.

### Human annotations

Nine English-speaking participants recruited through Prolific each rated 240
paired conversations: the 180 target items and 60 word-order distractors. For
each pair, annotators used an implicit 7-point Likert scale to express a
preference between the original dialogue and its edited counterpart. This
produced 2,160 judgments collected over two annotation rounds.

The released annotation file contains anonymized participant IDs, item IDs,
preference direction, preference strength, and target-word category. It also
retains the distractor judgments used for annotation-validity and agreement
checks, enabling the paper's human preference analyses to be reproduced.

## Repository layout

```text
data/                 Released stimuli and human annotations
particle_removal/     Experiment 1 implementation
generation/           Experiment 2 implementation
localization/         Experiment 3 implementation
scripts/              Experiment launchers and data validator
analysis/             Numerical summaries and statistical tests
```

## Setup

Python 3.11 is recommended. Create a fresh environment and install the
dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/validate_release_data.py
```

The experiments download open-weight checkpoints from Hugging Face. The
published experiments used FP16 inference on NVIDIA A40 GPUs; no model was
trained or fine-tuned.

## Models

Experiments 1 and 2 use the following base/post-trained pairs:

| Family | Base | Instruction/chat tuned |
|---|---|---|
| Llama 3 | `meta-llama/Meta-Llama-3-8B` | `meta-llama/Meta-Llama-3-8B-Instruct` |
| OLMo-2 | `allenai/OLMo-2-1124-7B` | `allenai/OLMo-2-1124-7B-Instruct` |
| Qwen3.5 | `Qwen/Qwen3.5-9B-Base` | `Qwen/Qwen3.5-9B` |
| Gemma-2 | `google/gemma-2-9b` | `google/gemma-2-9b-it` |

Experiment 3 uses only `meta-llama/Meta-Llama-3-8B-Instruct`.

## Reproducing the experiments

### Experiment 1: Particle removal

Run all eight checkpoints on the 180 paper items:

```bash
python scripts/run_experiment1.py
```

Run a subset by stable slug, or inspect commands without launching models:

```bash
python scripts/run_experiment1.py --models llama3_base llama3_instruct
python scripts/run_experiment1.py --dry-run
```

Create the human/LM summaries, correlations, and agreement statistic:

```bash
python analysis/experiment1.py
```

The reported removal score is the token-normalized full-dialogue difference
`log P(C S F) - log P(C S' F)`. Positive values favor the observed dialogue.
IQR outliers are marked in the model output; raw rows are retained.
`analysis/experiment1.py` uses this full-dialogue score by default. The earlier
sentence-only removal score remains available explicitly with
`--score sentence-removal`.

### Experiment 2: Paired-context generation

Run the published nucleus-sampling setup (`top_p=0.9`, temperature 1.0, 50
unique responses from each prompt condition):

```bash
python scripts/run_experiment2.py
python analysis/experiment2.py
```

For a response sampled from source context `S_src`, own-context advantage is:

```text
mean_token_log_p(response | S_src) - mean_token_log_p(response | S_alt)
```

The observed DailyDialog `followup` is copied into candidate files as metadata
but is not part of the generation prompt.

### Experiment 3: Component localization

Experiment 3 consumes the Llama 3 8B Instruct candidates produced by
Experiment 2. It builds top-500 positive-advantage pools, assigns source items
to two folds, localizes sites on one fold, evaluates necessity on the other,
and runs all four sense-transfer directions.

```bash
python scripts/run_experiment3.py --skip-existing
python analysis/experiment3.py
```

The launcher reproduces the paper settings:

- final token of the target utterance as the intervention position;
- attention, MLP, and post-block residual components at all 32 layers;
- top `k` in `{1, 3, 5, 10}`;
- four matched random layer-set controls;
- generated-response targets and necessity interventions;
- 5,000 bootstrap replicates and 20,000 sign-flip replicates.

The four transfer directions are exclusive `just` → `only`, non-exclusive
`just` → `only`, `only` → exclusive `just`, and `only` → non-exclusive
`just`.

### Running the analysis scripts

The analysis scripts summarize the full paper setup, so they will only run
once every checkpoint has finished:

- `analysis/experiment1.py` needs Experiment 1 scores from all eight models,
  each covering all 180 items.
- `analysis/experiment2.py` needs Experiment 2 candidates from all eight
  models, each covering all 180 items. By default it expects the number of
  responses per prompt condition recorded in each candidate file (50 for the
  paper setup), so runs with a smaller `--samples-per-condition` are checked
  automatically. Pass `--expected-samples-per-condition N` to require a
  specific count, or `0` to turn the check off.
- `analysis/experiment3.py` needs every output from `scripts/run_experiment3.py`.

If a model is missing or incomplete, the script stops with an error naming
the missing model or file. Runs of a single model (`--models ...`) are useful for
debugging, but their per-model results are in the files under
`results/experiment{1,2}/<model_slug>/` (the `.txt` summary and the
`*_summary.tsv` files), not in the analysis outputs.

### Outputs and resuming

Generated artifacts are written beneath `results/` and ignored by Git.
Experiment launchers accept `--skip-existing`; all accept `--help`, and the
first two allow explicit model subsets. Use `--dry-run` to audit the exact
commands without loading a checkpoint.

## Data licensing

The code and anonymized human annotations are released under the GNU General
Public License v3.0. The DailyDialog-derived stimulus file is distributed
under CC BY-NC-SA 4.0. See [`DATA_LICENSE.md`](DATA_LICENSE.md) for attribution
and reuse terms.

## Citation

TODO: Add bibtex when arxiv/conference link is ready
