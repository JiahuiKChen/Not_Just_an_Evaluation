"""Model metadata shared by Experiments 1 and 2."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    slug: str
    checkpoint: str
    prompt_mode: str
    label: str
    family: str
    tuning: str


MODEL_SPECS = (
    ModelSpec(
        "llama3_base",
        "meta-llama/Meta-Llama-3-8B",
        "base",
        "Llama 3 8B",
        "Llama 3",
        "base",
    ),
    ModelSpec(
        "llama3_instruct",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "chat",
        "Llama 3 8B Instruct",
        "Llama 3",
        "instruct",
    ),
    ModelSpec(
        "olmo2_base",
        "allenai/OLMo-2-1124-7B",
        "base",
        "OLMo-2 7B",
        "OLMo-2",
        "base",
    ),
    ModelSpec(
        "olmo2_instruct",
        "allenai/OLMo-2-1124-7B-Instruct",
        "chat",
        "OLMo-2 7B Instruct",
        "OLMo-2",
        "instruct",
    ),
    ModelSpec(
        "qwen35_base",
        "Qwen/Qwen3.5-9B-Base",
        "base",
        "Qwen3.5 9B Base",
        "Qwen3.5",
        "base",
    ),
    ModelSpec(
        "qwen35_instruct",
        "Qwen/Qwen3.5-9B",
        "chat",
        "Qwen3.5 9B CT",
        "Qwen3.5",
        "instruct",
    ),
    ModelSpec(
        "gemma2_base",
        "google/gemma-2-9b",
        "base",
        "Gemma-2 9B",
        "Gemma-2",
        "base",
    ),
    ModelSpec(
        "gemma2_instruct",
        "google/gemma-2-9b-it",
        "chat",
        "Gemma-2 9B IT",
        "Gemma-2",
        "instruct",
    ),
)


def select_models(names: list[str]) -> list[ModelSpec]:
    """Resolve ``all`` or a list of stable model slugs."""
    by_slug = {spec.slug: spec for spec in MODEL_SPECS}
    if names == ["all"]:
        return list(MODEL_SPECS)
    unknown = sorted(set(names) - set(by_slug))
    if unknown:
        allowed = ", ".join(["all", *by_slug])
        raise ValueError(f"Unknown model slug(s): {', '.join(unknown)}. Choose from: {allowed}")
    return [by_slug[name] for name in names]
