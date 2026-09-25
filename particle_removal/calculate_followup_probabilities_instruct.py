"""
Calculate probabilities of follow-up utterances for instruction-tuned chat models.

This script mirrors particle_removal/calculate_followup_probabilities.py, but formats
inputs using the model's chat template (tokenizer.apply_chat_template).

DEFAULT CLI BEHAVIOR:
    Running `python calculate_followup_probabilities_instruct.py` uses:
        --data_path data/dailydialog_sense_subset.tsv
        --model_name meta-llama/Meta-Llama-3-8B-Instruct
        --include_context True
        --no_context False
        --system_prompt None
        --observer_mode False
        --observer_system_prompt "You are a conversation observer. Predict the next turn in the conversation."
        --output_path results/Meta-Llama-3-8B-Instruct_context_filtered_just_dd_removal.csv

    Run commands from the repository root so relative paths resolve consistently.

ROLE / SPEAKER ASSIGNMENT:
    Default mode (`observer_mode=False`):
        - LLM is treated as the conversation participant producing the
          follow-up: the followup is always scored as an `assistant` message.
        - The preceding utterance (`w_word` / `wo_word`) is always scored as a
          `user` message.
        - Conversation context (if included) is mapped to alternating
          `user`/`assistant` roles so that the next utterance is a `user`
          message.
        - There are no explicit "Speaker 1" / "Speaker 2" labels in this mode;
          speaker labels are represented through chat roles.

    Observer mode (`--observer_mode`):
        - The conversation is passed as a plain-text transcript with
          "Speaker 1" / "Speaker 2" labels inside a single user message.
        - The model is prompted as an observer and predicts the next turn as the
          assistant.

BASIC USAGE:
    python calculate_followup_probabilities_instruct.py --data_path ...

    python calculate_followup_probabilities_instruct.py --no_context

    python calculate_followup_probabilities_instruct.py --model_name meta-llama/Meta-Llama-3-8B-Instruct

    python calculate_followup_probabilities_instruct.py --observer_mode
"""

import argparse
import os
import pandas as pd
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
)
from tqdm import tqdm


_WARNED_NO_BOS = False
# Older Qwen chat templates may inject this prompt, but verified Qwen3 models do not.
QWEN_LEGACY_DEFAULT_SYSTEM_PROMPT = (
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
)


def _safe_div(total_log_prob, token_count):
    if token_count and token_count > 0:
        return total_log_prob / token_count
    return float('nan')


def removal_outlier_detection(results_df):
    """
    Compute removal-based outlier scores.

    The primary score is:
        log P(S | C) - log P(S' | C)
    where:
        - S  is the original sentence with the particle
        - S' is the sentence with the particle removed
        - C  is the preceding context

    Because S and S' can have different token lengths, this function computes
    a per-token delta and uses an upper IQR fence for outlier detection.

    Returns:
        (updated_results_df, summary_lines)
    """
    required_columns = {
        'log_prob_w_word',
        'log_prob_wo_word',
        'log_prob_w_word_per_token',
        'log_prob_wo_word_per_token',
    }
    missing = sorted(required_columns - set(results_df.columns))
    if missing:
        raise KeyError(
            "Missing columns for removal outlier detection: "
            + ", ".join(missing)
        )

    df = results_df.copy()
    df['removal_log_prob_delta_per_token'] = (
        df['log_prob_w_word_per_token'] - df['log_prob_wo_word_per_token']
    )

    per_token_delta = df['removal_log_prob_delta_per_token']
    median_pt = per_token_delta.median()

    q1 = per_token_delta.quantile(0.25)
    q3 = per_token_delta.quantile(0.75)
    iqr = q3 - q1
    upper_iqr = q3 + 1.5 * iqr if pd.notna(iqr) else float('nan')

    df['removal_upper_iqr_outlier'] = (
        per_token_delta > upper_iqr if pd.notna(upper_iqr) else False
    )
    mean_pt = per_token_delta.mean()
    q1_pt = q1
    q3_pt = q3
    std_pt = per_token_delta.std()

    summary_lines = [
        "",
        "--- REMOVAL OUTLIER DETECTION ---",
        "Score = log P(S | C) - log P(S' | C)",
        "  S  = original sentence with particle",
        "  S' = sentence with particle removed",
        "  C  = context",
        "",
        "Per-token delta statistics:",
        f"Average per-token delta: {mean_pt:.4f}",
        f"Median per-token delta: {median_pt:.4f}",
        f"Std dev per-token delta: {std_pt:.4f}",
        f"Q1 per-token delta: {q1_pt:.4f}",
        f"Q3 per-token delta: {q3_pt:.4f}",
        f"Q3 + 1.5*IQR upper fence: {upper_iqr:.4f}",
        "",
        f"Upper-tail IQR outliers: {int(df['removal_upper_iqr_outlier'].sum())}",
        "",
        "Threshold guidance:",
        "Use the per-token delta for filtering, since raw sentence logprobs are length-sensitive.",
        "Use the IQR upper fence if you want a simple rule that is easy to explain in a paper.",
        "Best practice is to inspect examples near the proposed cutoff and calibrate the final threshold on a small labeled set.",
    ]

    return df, summary_lines


def _hf_load_kwargs(model_name):
    """Return model-loading kwargs for checkpoints with custom HF implementations."""
    if model_name.startswith("allenai/OLMo") or model_name.startswith("Qwen/Qwen3.5"):
        return {"trust_remote_code": True}
    return {}


def _uses_image_text_loader(model_name):
    """Return True for multimodal checkpoints that still support text-only scoring."""
    return model_name.startswith("Qwen/Qwen3.5")


def _chat_template_info(model_name, tokenizer):
    """Describe system-prompt behavior for the tokenizer's chat template."""
    template = getattr(tokenizer, "chat_template", "") or ""
    if "System role not supported" in template or "System role not supported" in template:
        return {
            "supports_system": False,
            "implicit_system_prompt": None,
            "requires_user_message": False,
            "requires_user_first": True,
        }
    if QWEN_LEGACY_DEFAULT_SYSTEM_PROMPT in template:
        return {
            "supports_system": True,
            "implicit_system_prompt": QWEN_LEGACY_DEFAULT_SYSTEM_PROMPT,
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


def _log_chat_template_info(model_name, chat_template_info):
    """Print the chat-template/system-prompt behavior being used."""
    if model_name == "Qwen/Qwen3.5-9B":
        print(
            f"{model_name}: using the post-trained Qwen3.5 checkpoint for "
            "instruction-tuned scoring via tokenizer.apply_chat_template()."
        )
    elif model_name == "Qwen/Qwen3.5-9B-Base":
        print(
            f"{model_name}: this is the pre-trained-only base checkpoint; "
            "use the non-chat scoring script for base-model runs."
        )
    if not chat_template_info["supports_system"]:
        print(f"{model_name}: tokenizer chat template does not support system messages.")
    elif chat_template_info["implicit_system_prompt"]:
        print(
            f"{model_name}: tokenizer chat template injects the default system prompt: "
            f"{chat_template_info['implicit_system_prompt']}"
        )
    else:
        print(f"{model_name}: tokenizer chat template has no built-in default system prompt.")
    if chat_template_info["requires_user_first"]:
        print(f"{model_name}: chat template requires the conversation to start with a user message.")
    elif chat_template_info["requires_user_message"]:
        print(f"{model_name}: chat template requires at least one user message in every rendered prompt.")


def _normalize_messages_for_chat_template(messages, chat_template_info):
    """Insert a blank bootstrap user turn when the tokenizer template requires it."""
    if not messages or not chat_template_info:
        return messages

    normalized = list(messages)
    insert_idx = 1 if normalized and normalized[0]["role"] == "system" else 0
    conversation_messages = normalized[insert_idx:]
    if not conversation_messages:
        return normalized

    needs_user_first = chat_template_info.get("requires_user_first", False)
    needs_any_user = chat_template_info.get("requires_user_message", False)

    if needs_user_first and conversation_messages[0]["role"] != "user":
        normalized.insert(insert_idx, {"role": "user", "content": ""})
        return normalized

    if needs_any_user and not any(msg["role"] == "user" for msg in conversation_messages):
        normalized.insert(insert_idx, {"role": "user", "content": ""})

    return normalized


def load_model(model_name="meta-llama/Meta-Llama-3-8B-Instruct"):
    """Load the language model and tokenizer."""
    print(f"Loading model: {model_name}")
    load_kwargs = _hf_load_kwargs(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, **load_kwargs)
    if _uses_image_text_loader(model_name):
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            **load_kwargs,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            **load_kwargs,
        )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


def _capitalize_first_alpha(text: str) -> str:
    """Capitalize the first alphabetic character, preserving any leading punctuation."""
    if not text:
        return text
    for i, ch in enumerate(text):
        if ch.isalpha():
            return text[:i] + ch.upper() + text[i + 1:]
    return text


def _parse_context_turns(context: str):
    if not context:
        return []
    turns = context.split("__eou__")
    return [t.strip() for t in turns if t.strip()]


def _speaker_label(idx: int) -> str:
    return "Speaker 1" if idx % 2 == 0 else "Speaker 2"


def build_transcript(context, utterance=None):
    """
    Build a plain-text transcript with Speaker 1/2 labels.
    Optionally append the next utterance as the next speaker.
    """
    turns = _parse_context_turns(context)
    formatted_turns = []
    for i, turn in enumerate(turns):
        formatted_turns.append(f"{_speaker_label(i)}: {turn}")

    if utterance is not None:
        formatted_turns.append(f"{_speaker_label(len(turns))}: {utterance}")

    return "\n".join(formatted_turns)


def build_observer_prompt_messages(
    context,
    utterance=None,
    include_context=True,
    system_prompt=None,
    next_turn_prefix="Next turn:"
):
    """
    Build messages for observer mode.
    The conversation is provided as a transcript in a user message, and the model
    predicts the next turn as the assistant.
    """
    transcript = ""
    if include_context:
        transcript = build_transcript(context, utterance=utterance)
    elif utterance is not None:
        transcript = build_transcript("", utterance=utterance)

    if transcript:
        content = f"Conversation so far:\n{transcript}\n{next_turn_prefix}"
    else:
        content = next_turn_prefix

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content})
    return messages


def build_chat_messages(context, utterance=None, followup=None, include_context=True, system_prompt=None):
    """
    Build a list of chat messages for the instruction-tuned model.

    Roles are assigned so that:
        - utterance (w_word / wo_word) is always user
        - followup is always assistant
        - context alternates user/assistant so that the next utterance is user
    """
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    turns = _parse_context_turns(context) if include_context else []

    if turns:
        # Ensure the next (utterance) role is user.
        start_role = "user" if len(turns) % 2 == 0 else "assistant"
        for i, turn in enumerate(turns):
            role = start_role if i % 2 == 0 else ("assistant" if start_role == "user" else "user")
            messages.append({"role": role, "content": turn})

    if utterance is not None:
        messages.append({"role": "user", "content": utterance})

    if followup is not None:
        messages.append({"role": "assistant", "content": followup})

    return messages


def _encode_messages(tokenizer, messages):
    if not messages:
        return []
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False
    )
    # Normalize to a flat list[int] across tokenizer implementations.
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


def calculate_log_probability_chat(model, tokenizer, context_messages, target_messages, device):
    """
    Calculate log probability of target_messages given context_messages
    using the model's chat template.
    """
    context_ids = _encode_messages(tokenizer, context_messages)
    full_ids = _encode_messages(tokenizer, context_messages + target_messages)

    if len(full_ids) <= len(context_ids):
        return 0.0, 0

    input_ids = torch.tensor([full_ids]).to(device)

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
            print("Warning: no BOS token; first target token not scored.")
            _WARNED_NO_BOS = True

    return total_log_prob, target_len


def process_dataset(
    data_path,
    model,
    tokenizer,
    include_context=True,
    output_path=None,
    system_prompt=None,
    observer_mode=False,
    chat_template_info=None,
):
    """Process the dataset and calculate probabilities."""
    df = pd.read_csv(data_path, sep='\t')
    print(f"Loaded {len(df)} examples from {data_path}")

    device = next(model.parameters()).device
    results = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        try:
            w_word = row['w_word']
            wo_word = row['wo_word']
            followup = row['followup']
            context = row['context'] if include_context else ""

            if pd.isna(w_word) or not isinstance(w_word, str) or w_word.strip() == "":
                print(f"Skipping row {idx}: invalid w_word")
                continue
            if pd.isna(wo_word) or not isinstance(wo_word, str) or wo_word.strip() == "":
                print(f"Skipping row {idx}: invalid wo_word")
                continue
            if pd.isna(followup) or not isinstance(followup, str) or followup.strip() == "":
                print(f"Skipping row {idx}: invalid or empty followup")
                continue
            w_word = _capitalize_first_alpha(w_word)
            wo_word = _capitalize_first_alpha(wo_word)
            if include_context and (pd.isna(context) or not isinstance(context, str)):
                context = ""
        except (KeyError, TypeError) as e:
            print(f"Skipping row {idx}: error reading fields - {e}")
            continue

        if context:
            context_turns = _parse_context_turns(context)
            num_context_turns = len(context_turns)
        else:
            num_context_turns = 0

        utterance_speaker = _speaker_label(num_context_turns)
        followup_speaker = _speaker_label(num_context_turns + 1)

        utterance_with_label_w = f"{utterance_speaker}: {w_word}"
        utterance_with_label_wo = f"{utterance_speaker}: {wo_word}"
        followup_with_label = f"{followup_speaker}: {followup}"

        # Context-only messages (no utterance)
        if observer_mode:
            context_messages = build_observer_prompt_messages(
                context,
                include_context=include_context,
                system_prompt=system_prompt
            )
        else:
            context_messages = build_chat_messages(
                context,
                include_context=include_context,
                system_prompt=system_prompt
            )
            context_messages = _normalize_messages_for_chat_template(
                context_messages,
                chat_template_info,
            )

        # Logprobs for utterances
        if observer_mode:
            log_prob_w_word, w_word_tokens = calculate_log_probability_chat(
                model,
                tokenizer,
                context_messages,
                [{"role": "assistant", "content": utterance_with_label_w}],
                device
            )
            log_prob_wo_word, wo_word_tokens = calculate_log_probability_chat(
                model,
                tokenizer,
                context_messages,
                [{"role": "assistant", "content": utterance_with_label_wo}],
                device
            )
        else:
            log_prob_w_word, w_word_tokens = calculate_log_probability_chat(
                model,
                tokenizer,
                context_messages,
                [{"role": "user", "content": w_word}],
                device
            )
            log_prob_wo_word, wo_word_tokens = calculate_log_probability_chat(
                model,
                tokenizer,
                context_messages,
                [{"role": "user", "content": wo_word}],
                device
            )

        # Followup logprobs (assistant message) given context + utterance
        if observer_mode:
            context_with_w = build_observer_prompt_messages(
                context,
                utterance=w_word,
                include_context=include_context,
                system_prompt=system_prompt
            )
            context_with_wo = build_observer_prompt_messages(
                context,
                utterance=wo_word,
                include_context=include_context,
                system_prompt=system_prompt
            )
        else:
            context_with_w = build_chat_messages(
                context,
                utterance=w_word,
                include_context=include_context,
                system_prompt=system_prompt
            )
            context_with_w = _normalize_messages_for_chat_template(
                context_with_w,
                chat_template_info,
            )
            context_with_wo = build_chat_messages(
                context,
                utterance=wo_word,
                include_context=include_context,
                system_prompt=system_prompt
            )
            context_with_wo = _normalize_messages_for_chat_template(
                context_with_wo,
                chat_template_info,
            )

        log_prob_followup_with, followup_tokens_with = calculate_log_probability_chat(
            model,
            tokenizer,
            context_with_w,
            [{"role": "assistant", "content": followup_with_label if observer_mode else followup}],
            device
        )
        log_prob_followup_without, followup_tokens_without = calculate_log_probability_chat(
            model,
            tokenizer,
            context_with_wo,
            [{"role": "assistant", "content": followup_with_label if observer_mode else followup}],
            device
        )
        if followup_tokens_with != followup_tokens_without:
            print(
                f"Warning: followup token counts differ at row {idx} "
                f"({followup_tokens_with} vs {followup_tokens_without})"
            )

        # Full-sequence logprobs (system + context + utterance + followup)
        if observer_mode:
            full_with_messages = build_observer_prompt_messages(
                context,
                include_context=include_context,
                system_prompt=system_prompt
            )
            full_without_messages = build_observer_prompt_messages(
                context,
                include_context=include_context,
                system_prompt=system_prompt
            )
            full_with_target = f"{utterance_with_label_w}\n{followup_with_label}"
            full_without_target = f"{utterance_with_label_wo}\n{followup_with_label}"
        else:
            full_with_messages = build_chat_messages(
                context,
                utterance=w_word,
                followup=followup,
                include_context=include_context,
                system_prompt=system_prompt
            )
            full_with_messages = _normalize_messages_for_chat_template(
                full_with_messages,
                chat_template_info,
            )
            full_without_messages = build_chat_messages(
                context,
                utterance=wo_word,
                followup=followup,
                include_context=include_context,
                system_prompt=system_prompt
            )
            full_without_messages = _normalize_messages_for_chat_template(
                full_without_messages,
                chat_template_info,
            )
            full_with_target = None
            full_without_target = None

        log_prob_full_with, full_tokens_with = calculate_log_probability_chat(
            model,
            tokenizer,
            [],
            full_with_messages if not observer_mode else full_with_messages + [{"role": "assistant", "content": full_with_target}],
            device
        )
        log_prob_full_without, full_tokens_without = calculate_log_probability_chat(
            model,
            tokenizer,
            [],
            full_without_messages if not observer_mode else full_without_messages + [{"role": "assistant", "content": full_without_target}],
            device
        )

        results.append({
            'id': row['id'],
            'word': row['word'],
            'context': context if include_context else "[NO CONTEXT]",
            'w_word': w_word,
            'wo_word': wo_word,
            'followup': followup,
            'log_prob_w_word': log_prob_w_word,
            'log_prob_wo_word': log_prob_wo_word,
            'log_prob_w_word_per_token': _safe_div(log_prob_w_word, w_word_tokens),
            'log_prob_wo_word_per_token': _safe_div(log_prob_wo_word, wo_word_tokens),
            'w_word_token_count': w_word_tokens,
            'wo_word_token_count': wo_word_tokens,
            'log_prob_followup_with': log_prob_followup_with,
            'log_prob_followup_without': log_prob_followup_without,
            'log_prob_followup_with_per_token': _safe_div(log_prob_followup_with, followup_tokens_with),
            'log_prob_followup_without_per_token': _safe_div(log_prob_followup_without, followup_tokens_without),
            'followup_token_count_with': followup_tokens_with,
            'followup_token_count_without': followup_tokens_without,
            'log_prob_full_with': log_prob_full_with,
            'log_prob_full_without': log_prob_full_without,
            'log_prob_full_with_per_token': _safe_div(log_prob_full_with, full_tokens_with),
            'log_prob_full_without_per_token': _safe_div(log_prob_full_without, full_tokens_without),
            'full_token_count_with': full_tokens_with,
            'full_token_count_without': full_tokens_without,
            'include_context': include_context,
            'observer_mode': observer_mode
        })

    results_df = pd.DataFrame(results)

    results_df, outlier_summary_lines = removal_outlier_detection(results_df)

    raw_diff = results_df['log_prob_followup_with'] - results_df['log_prob_followup_without']
    full_diff = results_df['log_prob_full_with'] - results_df['log_prob_full_without']
    raw_diff_per_token = (
        results_df['log_prob_followup_with_per_token']
        - results_df['log_prob_followup_without_per_token']
    )
    full_diff_per_token = (
        results_df['log_prob_full_with_per_token']
        - results_df['log_prob_full_without_per_token']
    )

    increased = (raw_diff > 0).sum()
    decreased = (raw_diff < 0).sum()
    unchanged = (raw_diff == 0).sum()
    full_increased = (full_diff > 0).sum()
    full_decreased = (full_diff < 0).sum()
    full_unchanged = (full_diff == 0).sum()
    increased_pt = (raw_diff_per_token > 0).sum()
    decreased_pt = (raw_diff_per_token < 0).sum()
    unchanged_pt = (raw_diff_per_token == 0).sum()
    full_increased_pt = (full_diff_per_token > 0).sum()
    full_decreased_pt = (full_diff_per_token < 0).sum()
    full_unchanged_pt = (full_diff_per_token == 0).sum()

    summary_lines = [
        "=" * 80,
        "SUMMARY STATISTICS",
        "=" * 80,
        f"Total examples processed: {len(results_df)}",
        f"Include context: {include_context}",
        f"Observer mode: {observer_mode}",
        "",
        "--- RAW FOLLOWUP LOG PROBABILITIES ---",
        f"Average log probability WITH particle: {results_df['log_prob_followup_with'].mean():.4f}",
        f"Average log probability WITHOUT particle: {results_df['log_prob_followup_without'].mean():.4f}",
        f"Average log probability difference: {raw_diff.mean():.4f}",
        f"  (positive = particle increases followup probability)",
        "",
        f"Median log probability difference: {raw_diff.median():.4f}",
        f"Std dev of log probability difference: {raw_diff.std():.4f}",
        "",
        f"Particle INCREASED followup probability: {increased} ({increased/len(results_df)*100:.1f}%)",
        f"Particle DECREASED followup probability: {decreased} ({decreased/len(results_df)*100:.1f}%)",
        f"No change: {unchanged}",
        "",
        "--- RAW FOLLOWUP LOG PROBABILITIES (PER-TOKEN) ---",
        f"Average log probability WITH particle: {results_df['log_prob_followup_with_per_token'].mean():.4f}",
        f"Average log probability WITHOUT particle: {results_df['log_prob_followup_without_per_token'].mean():.4f}",
        f"Average log probability difference: {raw_diff_per_token.mean():.4f}",
        f"  (positive = particle increases followup probability)",
        "",
        f"Median log probability difference: {raw_diff_per_token.median():.4f}",
        f"Std dev of log probability difference: {raw_diff_per_token.std():.4f}",
        "",
        f"Particle INCREASED followup probability: {increased_pt} ({increased_pt/len(results_df)*100:.1f}%)",
        f"Particle DECREASED followup probability: {decreased_pt} ({decreased_pt/len(results_df)*100:.1f}%)",
        f"No change: {unchanged_pt}",
        "",
        "--- FULL-SEQUENCE LOG PROBABILITIES ---",
        f"Average log probability WITH particle: {results_df['log_prob_full_with'].mean():.4f}",
        f"Average log probability WITHOUT particle: {results_df['log_prob_full_without'].mean():.4f}",
        f"Average log probability difference: {full_diff.mean():.4f}",
        f"  (positive = particle increases full-sequence probability)",
        "",
        f"Median log probability difference: {full_diff.median():.4f}",
        f"Std dev of log probability difference: {full_diff.std():.4f}",
        "",
        f"Particle INCREASED full-sequence probability: {full_increased} ({full_increased/len(results_df)*100:.1f}%)",
        f"Particle DECREASED full-sequence probability: {full_decreased} ({full_decreased/len(results_df)*100:.1f}%)",
        f"No change: {full_unchanged}",
        "",
        "--- FULL-SEQUENCE LOG PROBABILITIES (PER-TOKEN) ---",
        f"Average log probability WITH particle: {results_df['log_prob_full_with_per_token'].mean():.4f}",
        f"Average log probability WITHOUT particle: {results_df['log_prob_full_without_per_token'].mean():.4f}",
        f"Average log probability difference: {full_diff_per_token.mean():.4f}",
        f"  (positive = particle increases full-sequence probability)",
        "",
        f"Median log probability difference: {full_diff_per_token.median():.4f}",
        f"Std dev of log probability difference: {full_diff_per_token.std():.4f}",
        "",
        f"Particle INCREASED full-sequence probability: {full_increased_pt} ({full_increased_pt/len(results_df)*100:.1f}%)",
        f"Particle DECREASED full-sequence probability: {full_decreased_pt} ({full_decreased_pt/len(results_df)*100:.1f}%)",
        f"No change: {full_unchanged_pt}",
        *outlier_summary_lines,
        "=" * 80,
    ]
    summary = "\n".join(summary_lines)

    print("\n" + summary + "\n")

    if output_path:
        results_df.to_csv(output_path, sep='\t', index=False)
        print(f"Results saved to: {output_path}")

        output_base = output_path.rsplit('.', 1)[0]
        rejected_iqr_path = output_base + '_rejected_iqr.csv'

        results_df[results_df['removal_upper_iqr_outlier']].to_csv(
            rejected_iqr_path,
            sep='\t',
            index=False
        )
        print(f"IQR-rejected samples saved to: {rejected_iqr_path}")

        summary_path = output_base + '.txt'
        with open(summary_path, 'w') as f:
            f.write(summary + "\n")
            f.write(f"IQR-rejected samples file: {rejected_iqr_path}\n")
        print(f"Summary saved to: {summary_path}")

    return results_df


def main():
    parser = argparse.ArgumentParser(
        description="Calculate followup probabilities using chat templates for instruction-tuned models"
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default='data/dailydialog_sense_subset.tsv',
        help='Path to the input TSV file'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='meta-llama/Meta-Llama-3-8B-Instruct',
        help='HuggingFace model name (default: meta-llama/Meta-Llama-3-8B-Instruct)'
    )
    parser.add_argument(
        '--include_context',
        action='store_true',
        default=True,
        help='Include conversation context in probability calculation'
    )
    parser.add_argument(
        '--no_context',
        action='store_true',
        help='Do NOT include conversation context (overrides --include_context)'
    )
    parser.add_argument(
        '--system_prompt',
        type=str,
        default=None,
        help='Optional system prompt to prepend (default: none). Ignored in observer mode.'
    )
    parser.add_argument(
        '--observer_mode',
        action='store_true',
        help='Treat the model as a conversation observer (transcript-in, next-turn-out)'
    )
    parser.add_argument(
        '--observer_system_prompt',
        type=str,
        default="You are a conversation observer. Predict the next turn in the conversation.",
        help='System prompt used in observer mode (always applied)'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help='Path to save results CSV (default: results/<model>_<context/noContext>_<datafile>_removal.csv)'
    )

    args = parser.parse_args()

    include_context = not args.no_context if args.no_context else args.include_context

    if args.output_path is None:
        if '/' in args.model_name:
            model_name_short = args.model_name.split('/')[-1]
        else:
            model_name_short = args.model_name

        context_str = "context" if include_context else "noContext"
        model_prefix = f"{model_name_short}_Obs" if args.observer_mode else model_name_short
        data_basename = os.path.splitext(os.path.basename(args.data_path))[0]
        output_dir = os.path.join("results", "experiment1")
        os.makedirs(output_dir, exist_ok=True)
        args.output_path = os.path.join(
            output_dir,
            f"{model_prefix}_{context_str}_{data_basename}_removal.csv"
        )

    model, tokenizer = load_model(args.model_name)
    chat_template_info = _chat_template_info(args.model_name, tokenizer)
    _log_chat_template_info(args.model_name, chat_template_info)

    system_prompt = args.observer_system_prompt if args.observer_mode else args.system_prompt
    if system_prompt and not chat_template_info["supports_system"]:
        raise ValueError(
            f"{args.model_name} does not support system messages in its tokenizer chat template."
        )

    process_dataset(
        args.data_path,
        model,
        tokenizer,
        include_context=include_context,
        output_path=args.output_path,
        system_prompt=system_prompt,
        observer_mode=args.observer_mode,
        chat_template_info=chat_template_info,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
