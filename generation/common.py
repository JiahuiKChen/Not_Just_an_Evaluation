"""Shared utilities for response-generation particle experiments."""

from __future__ import annotations

import gc
import math
import os
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer


_WARNED_NO_BOS = False


@dataclass(frozen=True)
class PromptPair:
    """The two prompts used for one stimulus item."""

    prompt_p: str
    prompt_p_prime: str
    next_speaker: str


@dataclass(frozen=True)
class ChatPromptPair:
    """The two chat-template contexts used for one stimulus item."""

    messages_p: List[dict]
    messages_p_prime: List[dict]


def hf_load_kwargs(model_name: str) -> dict:
    """Return model-loading kwargs for checkpoints with custom HF code."""
    if model_name.startswith("allenai/OLMo") or model_name.startswith("Qwen/Qwen3.5"):
        return {"trust_remote_code": True}
    return {}


def uses_image_text_loader(model_name: str) -> bool:
    """Return True for multimodal checkpoints that support text-only scoring."""
    return model_name.startswith("Qwen/Qwen3.5")


def configured_device_map() -> Optional[str]:
    """Read an optional device_map override from the environment."""
    if not torch.cuda.is_available():
        return None
    raw_value = os.environ.get("JUST_LLMS_DEVICE_MAP", "auto").strip()
    if raw_value.lower() in {"", "none", "null"}:
        return None
    return raw_value


def load_model(model_name: str = "meta-llama/Meta-Llama-3-8B"):
    """Load a Hugging Face causal LM and tokenizer."""
    print(f"Loading model: {model_name}", flush=True)
    load_kwargs = hf_load_kwargs(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, **load_kwargs)
    model_kwargs = {
        "dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
        "device_map": configured_device_map(),
        **load_kwargs,
    }
    model_cls = AutoModelForImageTextToText if uses_image_text_loader(model_name) else AutoModelForCausalLM
    model = model_cls.from_pretrained(model_name, **model_kwargs)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def safe_div(total: float, count: int) -> float:
    if count and count > 0:
        return total / count
    return float("nan")


def capitalize_first_alpha(text: str) -> str:
    """Capitalize the first alphabetic character, preserving punctuation."""
    if not text:
        return text
    for i, ch in enumerate(text):
        if ch.isalpha():
            return text[:i] + ch.upper() + text[i + 1 :]
    return text


def parse_context_turns(context: str) -> List[str]:
    if not context:
        return []
    return [turn.strip() for turn in str(context).split("__eou__") if turn.strip()]


def speaker_label(idx: int) -> str:
    return "Speaker 1" if idx % 2 == 0 else "Speaker 2"


def format_conversation(context: str, utterance: str) -> str:
    """Format context plus the particle-bearing utterance with speaker labels."""
    turns = parse_context_turns(context)
    turns.append(utterance)
    formatted_turns = []
    for i, turn in enumerate(turns):
        formatted_turns.append(f"{speaker_label(i)}: {turn}")
    return "\n".join(formatted_turns) + "\n"


def build_prompt_pair(context: str, w_word: str, wo_word: str) -> PromptPair:
    """Build P and P' generation/scoring prompts ending at the next speaker."""
    turns = parse_context_turns(context)
    next_speaker = speaker_label(len(turns) + 1)
    prompt_p = format_conversation(context, w_word) + f"{next_speaker}: "
    prompt_p_prime = format_conversation(context, wo_word) + f"{next_speaker}: "
    return PromptPair(prompt_p=prompt_p, prompt_p_prime=prompt_p_prime, next_speaker=next_speaker)


# Legacy template string retained only for tokenizer-behavior detection.
# Qwen3/Qwen3.5 post-trained checkpoints are intentionally formatted with
# their chat template as-is; we do not inject a default system prompt.
QWEN25_DEFAULT_SYSTEM_PROMPT = (
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
)


def chat_template_info(model_name: str, tokenizer) -> dict:
    """Describe system-prompt and message-shape constraints for a chat template."""
    template = getattr(tokenizer, "chat_template", "") or ""
    if "System role not supported" in template:
        return {
            "supports_system": False,
            "implicit_system_prompt": None,
            "requires_user_message": False,
            "requires_user_first": True,
        }
    if QWEN25_DEFAULT_SYSTEM_PROMPT in template:
        return {
            "supports_system": True,
            "implicit_system_prompt": QWEN25_DEFAULT_SYSTEM_PROMPT,
            "requires_user_message": False,
            "requires_user_first": False,
        }
    if model_name.startswith("google/gemma-2"):
        return {
            "supports_system": False,
            "implicit_system_prompt": None,
            "requires_user_message": False,
            "requires_user_first": True,
        }
    if "No user query found in messages." in template:
        return {
            "supports_system": True,
            "implicit_system_prompt": None,
            "requires_user_message": True,
            "requires_user_first": False,
        }
    return {
        "supports_system": True,
        "implicit_system_prompt": None,
        "requires_user_message": False,
        "requires_user_first": False,
    }


def normalize_messages_for_chat_template(messages: List[dict], info: Optional[dict]) -> List[dict]:
    """Insert an empty bootstrap user turn when a tokenizer requires one."""
    if not messages or not info:
        return messages

    normalized = list(messages)
    insert_idx = 1 if normalized and normalized[0]["role"] == "system" else 0
    conversation_messages = normalized[insert_idx:]
    if not conversation_messages:
        return normalized

    needs_user_first = info.get("requires_user_first", False)
    needs_any_user = info.get("requires_user_message", False)

    if needs_user_first and conversation_messages[0]["role"] != "user":
        normalized.insert(insert_idx, {"role": "user", "content": ""})
        return normalized

    if needs_any_user and not any(msg["role"] == "user" for msg in conversation_messages):
        normalized.insert(insert_idx, {"role": "user", "content": ""})

    return normalized


def build_chat_messages(
    context: str,
    utterance: Optional[str] = None,
    include_context: bool = True,
    system_prompt: Optional[str] = None,
) -> List[dict]:
    """
    Build chat messages for instruction-tuned/chat-template generation.

    Roles are assigned so the particle-bearing utterance is a user message and
    the generated continuation is the assistant's next turn.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    turns = parse_context_turns(context) if include_context else []
    if turns:
        start_role = "user" if len(turns) % 2 == 0 else "assistant"
        for i, turn in enumerate(turns):
            role = start_role if i % 2 == 0 else ("assistant" if start_role == "user" else "user")
            messages.append({"role": role, "content": turn})

    if utterance is not None:
        messages.append({"role": "user", "content": utterance})

    return messages


def build_chat_prompt_pair(
    context: str,
    w_word: str,
    wo_word: str,
    include_context: bool = True,
    system_prompt: Optional[str] = None,
    info: Optional[dict] = None,
) -> ChatPromptPair:
    messages_p = build_chat_messages(
        context,
        utterance=w_word,
        include_context=include_context,
        system_prompt=system_prompt,
    )
    messages_p_prime = build_chat_messages(
        context,
        utterance=wo_word,
        include_context=include_context,
        system_prompt=system_prompt,
    )
    return ChatPromptPair(
        messages_p=normalize_messages_for_chat_template(messages_p, info),
        messages_p_prime=normalize_messages_for_chat_template(messages_p_prime, info),
    )


def encode_messages(tokenizer, messages: List[dict], add_generation_prompt: bool = False) -> List[int]:
    """Encode chat-template messages into a flat list of token ids."""
    if not messages:
        return []
    template = getattr(tokenizer, "chat_template", "") or ""
    template_kwargs = {}
    if add_generation_prompt and "enable_thinking" in template:
        template_kwargs["enable_thinking"] = False
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        **template_kwargs,
    )
    if isinstance(encoded, list):
        if not encoded:
            return []
        if isinstance(encoded[0], int):
            return encoded
        if isinstance(encoded[0], list):
            return encoded[0]
    if hasattr(encoded, "input_ids"):
        ids = encoded["input_ids"]
        if torch.is_tensor(ids):
            return ids[0].tolist()
        if isinstance(ids, list):
            if ids and isinstance(ids[0], list):
                return ids[0]
            return ids
    if hasattr(encoded, "ids"):
        return list(encoded.ids)
    if torch.is_tensor(encoded):
        return encoded.squeeze(0).tolist()
    raise TypeError(f"Unexpected chat template encoding type: {type(encoded)}")


def dataset_role_for_word(word: object) -> str:
    """Mark not as the meaning-control condition."""
    word_str = "" if pd.isna(word) else str(word).strip().lower()
    return "meaning_control" if word_str == "not" else "discourse_particle"


def validate_stimulus_row(row: pd.Series, include_context: bool = True) -> Tuple[str, str, str]:
    """Return normalized context, w_word, wo_word or raise ValueError."""
    w_word = row.get("w_word")
    wo_word = row.get("wo_word")
    context = row.get("context") if include_context else ""

    if pd.isna(w_word) or not isinstance(w_word, str) or not w_word.strip():
        raise ValueError("invalid w_word")
    if pd.isna(wo_word) or not isinstance(wo_word, str) or not wo_word.strip():
        raise ValueError("invalid wo_word")
    if include_context and (pd.isna(context) or not isinstance(context, str)):
        context = ""

    return str(context), capitalize_first_alpha(w_word), capitalize_first_alpha(wo_word)


def calculate_log_probability(model, tokenizer, context_text: str, target_text: str, device) -> Tuple[float, int]:
    """
    Calculate log P(target_text | context_text).

    This mirrors the particle-removal scorer: tokenize the full string once when
    possible so boundary-tokenization effects are minimized.
    """
    full_text = context_text + target_text

    if tokenizer.is_fast:
        enc = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
        full_tokens = enc["input_ids"]
        offsets = enc["offset_mapping"]
        context_char_len = len(context_text)
        context_token_count = 0
        for start, end in offsets:
            if start == end == 0:
                continue
            if end <= context_char_len:
                context_token_count += 1
            else:
                break
    else:
        context_tokens = tokenizer.encode(context_text, add_special_tokens=False)
        target_tokens = tokenizer.encode(target_text, add_special_tokens=False)
        full_tokens = context_tokens + target_tokens
        context_token_count = len(context_tokens)

    if tokenizer.bos_token_id is not None:
        full_tokens = [tokenizer.bos_token_id] + full_tokens
        context_len = context_token_count + 1
    else:
        context_len = context_token_count

    target_len = len(full_tokens) - context_len
    if target_len <= 0:
        return 0.0, 0

    input_ids = torch.tensor([full_tokens], device=device)
    with torch.no_grad():
        logits = model(input_ids).logits

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    total_log_prob = 0.0
    start_idx = max(context_len - 1, 0)
    end_idx = len(full_tokens) - 1
    for i in range(start_idx, end_idx):
        next_token_id = full_tokens[i + 1]
        total_log_prob += log_probs[0, i, next_token_id].item()

    scored_target_len = target_len
    global _WARNED_NO_BOS
    if tokenizer.bos_token_id is None and context_len == 0 and target_len > 0:
        scored_target_len = target_len - 1
        if not _WARNED_NO_BOS:
            print("Warning: no BOS token; first target token not scored.", flush=True)
            _WARNED_NO_BOS = True

    return total_log_prob, scored_target_len


def calculate_log_probability_chat(
    model,
    tokenizer,
    context_messages: List[dict],
    target_messages: List[dict],
    device,
) -> Tuple[float, int]:
    """Calculate log P(target_messages | context_messages) using chat templates."""
    context_ids = encode_messages(tokenizer, context_messages)
    full_ids = encode_messages(tokenizer, context_messages + target_messages)

    if len(full_ids) <= len(context_ids):
        return 0.0, 0

    input_ids = torch.tensor([full_ids], device=device)
    with torch.no_grad():
        logits = model(input_ids).logits

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    total_log_prob = 0.0
    start_idx = max(len(context_ids) - 1, 0)
    end_idx = len(full_ids) - 1
    for i in range(start_idx, end_idx):
        next_pos = i + 1
        if next_pos < len(context_ids):
            continue
        next_token_id = full_ids[next_pos]
        total_log_prob += log_probs[0, i, next_token_id].item()

    target_len = len(full_ids) - len(context_ids)
    global _WARNED_NO_BOS
    if len(context_ids) == 0 and tokenizer.bos_token_id is None and target_len > 0:
        target_len -= 1
        if not _WARNED_NO_BOS:
            print("Warning: no BOS token; first target token not scored.", flush=True)
            _WARNED_NO_BOS = True

    return total_log_prob, target_len


_LEADING_SPEAKER_RE = re.compile(r"^\s*(?:Speaker\s*[12]|User|Assistant)\s*:\s*", re.IGNORECASE)


def clean_generated_response(text: str) -> str:
    """Keep one next-turn utterance and remove accidental speaker labels."""
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    stop_patterns = [
        "__eou__",
        "<|eot_id|>",
        "<|end_of_text|>",
        "\nSpeaker 1:",
        "\nSpeaker 2:",
        "\nUser:",
        "\nAssistant:",
        "\n",
    ]
    stop_positions = [cleaned.find(pattern) for pattern in stop_patterns if cleaned.find(pattern) >= 0]
    if stop_positions:
        cleaned = cleaned[: min(stop_positions)]
    cleaned = _LEADING_SPEAKER_RE.sub("", cleaned)
    return cleaned.strip()


def normalize_response(text: str) -> str:
    """Normalize generated text for duplicate/overlap checks."""
    text = clean_generated_response(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def generate_unique_responses(
    model,
    tokenizer,
    prompt: str,
    device,
    top_k: int,
    num_beams: int,
    candidate_pool: Optional[int],
    max_new_tokens: int,
) -> List[dict]:
    """Generate top beam continuations and return the first top_k unique responses."""
    if num_beams < top_k:
        raise ValueError(f"num_beams ({num_beams}) must be >= top_k ({top_k})")
    if candidate_pool is None:
        candidate_pool = num_beams
    if candidate_pool < top_k:
        raise ValueError(f"candidate_pool ({candidate_pool}) must be >= top_k ({top_k})")

    num_return_sequences = min(candidate_pool, num_beams)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=False,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            max_new_tokens=max_new_tokens,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=False,
            use_cache=True,
        )

    seen = set()
    unique = []
    for beam_rank, sequence in enumerate(outputs.sequences.tolist(), start=1):
        continuation_ids = sequence[prompt_len:]
        raw_text = tokenizer.decode(continuation_ids, skip_special_tokens=True)
        response = clean_generated_response(raw_text)
        normalized = normalize_response(response)
        if not response or not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(
            {
                "beam_rank": beam_rank,
                "unique_rank": len(unique) + 1,
                "response": response,
                "normalized_response": normalized,
                "raw_generation": raw_text,
            }
        )
        if len(unique) >= top_k:
            break

    return unique


def sample_unique_responses_nucleus(
    model,
    tokenizer,
    prompt: str,
    device,
    samples_per_condition: int,
    top_p: float,
    temperature: float,
    sample_batch_size: int,
    max_sample_draws: int,
    max_new_tokens: int,
    seed: Optional[int] = None,
) -> List[dict]:
    """Sample unique next-turn continuations with nucleus sampling."""
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    return sample_unique_responses_nucleus_from_inputs(
        model=model,
        tokenizer=tokenizer,
        inputs=inputs,
        device=device,
        samples_per_condition=samples_per_condition,
        top_p=top_p,
        temperature=temperature,
        sample_batch_size=sample_batch_size,
        max_sample_draws=max_sample_draws,
        max_new_tokens=max_new_tokens,
        seed=seed,
    )


def sample_unique_responses_nucleus_chat(
    model,
    tokenizer,
    messages: List[dict],
    device,
    samples_per_condition: int,
    top_p: float,
    temperature: float,
    sample_batch_size: int,
    max_sample_draws: int,
    max_new_tokens: int,
    seed: Optional[int] = None,
) -> List[dict]:
    """Sample unique assistant continuations from a chat-template prompt."""
    prompt_ids = encode_messages(tokenizer, messages, add_generation_prompt=True)
    input_ids = torch.tensor([prompt_ids], device=device)
    inputs = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids, device=device),
    }
    return sample_unique_responses_nucleus_from_inputs(
        model=model,
        tokenizer=tokenizer,
        inputs=inputs,
        device=device,
        samples_per_condition=samples_per_condition,
        top_p=top_p,
        temperature=temperature,
        sample_batch_size=sample_batch_size,
        max_sample_draws=max_sample_draws,
        max_new_tokens=max_new_tokens,
        seed=seed,
    )


def sample_unique_responses_nucleus_from_inputs(
    model,
    tokenizer,
    inputs: dict,
    device,
    samples_per_condition: int,
    top_p: float,
    temperature: float,
    sample_batch_size: int,
    max_sample_draws: int,
    max_new_tokens: int,
    seed: Optional[int] = None,
) -> List[dict]:
    """Sample unique next-turn continuations from encoded prompt inputs."""
    if not 0.0 < top_p <= 1.0:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if samples_per_condition <= 0:
        raise ValueError("samples_per_condition must be positive")
    if sample_batch_size <= 0:
        raise ValueError("sample_batch_size must be positive")
    if max_sample_draws < samples_per_condition:
        raise ValueError("max_sample_draws must be >= samples_per_condition")

    prompt_len = inputs["input_ids"].shape[1]

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    seen = set()
    unique = []
    draws = 0
    current_sample_batch_size = sample_batch_size

    while len(unique) < samples_per_condition and draws < max_sample_draws:
        batch_size = min(current_sample_batch_size, max_sample_draws - draws)
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    do_sample=True,
                    top_p=top_p,
                    top_k=0,
                    temperature=temperature,
                    num_return_sequences=batch_size,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=False,
                    use_cache=True,
                )
        except torch.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            if current_sample_batch_size == 1:
                raise
            current_sample_batch_size = max(1, current_sample_batch_size // 2)
            print(
                "CUDA OOM during sampling; retrying with "
                f"sample_batch_size={current_sample_batch_size}",
                flush=True,
            )
            continue

        for sequence in outputs.sequences.tolist():
            draws += 1
            continuation_ids = sequence[prompt_len:]
            raw_text = tokenizer.decode(continuation_ids, skip_special_tokens=True)
            response = clean_generated_response(raw_text)
            normalized = normalize_response(response)
            if not response or not normalized or normalized in seen:
                continue
            seen.add(normalized)
            unique.append(
                {
                    "sample_draw": draws,
                    "unique_rank": len(unique) + 1,
                    "response": response,
                    "normalized_response": normalized,
                    "raw_generation": raw_text,
                }
            )
            if len(unique) >= samples_per_condition:
                break

    return unique


def logsumexp(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return float("nan")
    max_value = max(values)
    if math.isinf(max_value):
        return max_value
    return max_value + math.log(sum(math.exp(value - max_value) for value in values))
