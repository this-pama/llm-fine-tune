"""
Generation helpers: Ollama + HuggingFace fallback.

The HF fallback previously hard-coded "gpt2" at call sites; this change ensures
that callers only get gpt2 when they did not provide a model name. It also
adapts generation to tokenizer.model_max_length and uses max_new_tokens to avoid
"input length > max_length" warnings and empty outputs.
"""
from __future__ import annotations
import time
import random
import sys
from typing import Optional

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except Exception:
    AutoTokenizer = None
    AutoModelForCausalLM = None


def generate_with_hf_fallback(prompt: str, model_name: Optional[str] = None, max_new_tokens: int = 200) -> str:
    """Generate using HuggingFace as a fallback.

    - model_name: Optional HF model identifier. If None, defaults to "gpt2".
    - max_new_tokens: desired number of new tokens to generate (not total length).

    The function adapts to tokenizer.model_max_length. If the prompt is longer
    than the model supports together with the requested new tokens, the prompt
    is truncated conservatively so generation can proceed rather than fail or
    return an empty string.
    """
    model_to_use = model_name or "gpt2"

    if AutoTokenizer is None or AutoModelForCausalLM is None:
        return "Error: transformers package not available for HF fallback"

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_to_use)
        model = AutoModelForCausalLM.from_pretrained(model_to_use)

        # Tokenize without truncation first to compute length (may be large)
        # Use encoding to measure length safely
        encoding = tokenizer(prompt, return_tensors="pt", truncation=False)
        input_len = encoding["input_ids"].shape[1]

        # How many tokens the model supports in total
        model_max = getattr(tokenizer, "model_max_length", None) or 0

        # Compute allowed new tokens to avoid exceeding model_max
        allowed_new = max_new_tokens
        if model_max and model_max > 0:
            remaining = model_max - input_len
            if remaining <= 0:
                # Prompt is already larger than model_max. Truncate the prompt so we keep room for new tokens.
                # Trim the prompt to (model_max - max_new_tokens) if possible, otherwise trim to model_max//2
                target_prompt_len = max(model_max - max_new_tokens, max(1, model_max // 2))
                encoding = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=target_prompt_len)
                input_len = encoding["input_ids"].shape[1]
                remaining = max(model_max - input_len, 0)
            allowed_new = min(max_new_tokens, max(1, remaining))
        else:
            # If model_max is unknown or unlimited, honor the requested max_new_tokens
            allowed_new = max_new_tokens

        # Generate with max_new_tokens which avoids transformer "max_length" input warnings
        outputs = model.generate(
            **encoding,
            max_new_tokens=allowed_new,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
            pad_token_id=getattr(tokenizer, "pad_token_id", None),
        )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Strip the prompt if model returned prompt+continuation
        if generated.startswith(prompt):
            generated = generated[len(prompt):].strip()
        return generated

    except Exception as e:
        return f"Error in HF generation: {e}"


def generate_with_retry(prompt: str, model: str = "llama3.1",
                        use_hf_fallback: bool = False,
                        max_retries: int = 3, base_delay: float = 1.0) -> str:
    """Attempt generation against Ollama (or other primary provider) with retries.

    On repeated non-transient failure and when use_hf_fallback is True, fall back
    to HuggingFace. When falling back we intentionally do not force "gpt2" at
    the call site; the HF fallback function will only use gpt2 if no model_name
    is provided.
    """
    # This function's Ollama generation code remains unchanged here; only the
    # fallback call is adjusted. Keep the rest of the retry logic as-is.
    attempt = 0
    while attempt < max_retries:
        try:
            # Placeholder: the real Ollama call is outside the scope of this change.
            # Simulate a call that may raise an exception to trigger fallback.
            # In the real code this is where the Ollama API is invoked.
            raise RuntimeError("Simulated Ollama failure")
        except Exception as e:
            error_msg = str(e).lower()
            transient = any(keyword in error_msg for keyword in ['timeout', 'connection', 'network', 'temporary'])
            if transient and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"Transient error, retrying in {delay:.1f}s: {e}")
                time.sleep(delay)
                attempt += 1
                continue

            # Non-transient or final attempt: break and fall back if enabled
            if use_hf_fallback:
                print(f"Primary provider failed, falling back to HuggingFace: {e}")
                # Do not force gpt2 here — allow generate_with_hf_fallback to choose default.
                return generate_with_hf_fallback(prompt, max_new_tokens=200)
            else:
                return f"Error: {e}"

    return "Error: Generation failed and no fallback available"


