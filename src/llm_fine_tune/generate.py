"""
Generation helpers: Ollama + HuggingFace fallback.

This module provides a HuggingFace fallback that selects the fallback model from
an explicit argument, the HF_FALLBACK_MODEL environment variable, or the
conservative default "gpt2" only when nothing else is provided. It also uses
tokenizer.model_max_length and max_new_tokens to avoid transformer warnings.
"""
from __future__ import annotations
import time
import random
import os
from typing import Optional

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except Exception:
    AutoTokenizer = None
    AutoModelForCausalLM = None


def generate_with_hf_fallback(prompt: str, model_name: Optional[str] = None, max_new_tokens: int = 200) -> str:
    """Generate using HuggingFace as a fallback.

    Selection priority for model_name:
      1) explicit model_name arg
      2) HF_FALLBACK_MODEL env var
      3) "gpt2"
    """
    model_to_use = model_name or os.getenv("HF_FALLBACK_MODEL") or "gpt2"

    if AutoTokenizer is None or AutoModelForCausalLM is None:
        return "Error: transformers package not available for HF fallback"

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_to_use)
        model = AutoModelForCausalLM.from_pretrained(model_to_use)

        # Tokenize without truncation first to compute length (may be large)
        encoding = tokenizer(prompt, return_tensors="pt", truncation=False)
        input_len = encoding["input_ids"].shape[1]

        model_max = getattr(tokenizer, "model_max_length", None) or 0

        allowed_new = max_new_tokens
        if model_max and model_max > 0:
            remaining = model_max - input_len
            if remaining <= 0:
                target_prompt_len = max(model_max - max_new_tokens, max(1, model_max // 2))
                encoding = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=target_prompt_len)
                input_len = encoding["input_ids"].shape[1]
                remaining = max(model_max - input_len, 0)
            allowed_new = min(max_new_tokens, max(1, remaining))

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
        if generated.startswith(prompt):
            generated = generated[len(prompt):].strip()
        return generated

    except Exception as e:
        return f"Error in HF generation: {e}"


def generate_with_retry(prompt: str, model: str = "llama3.1",
                        use_hf_fallback: bool = False,
                        max_retries: int = 3, base_delay: float = 1.0,
                        hf_fallback_model: Optional[str] = None) -> str:
    """Attempt generation against Ollama (or other primary provider) with retries.

    If use_hf_fallback is True, on final failure call generate_with_hf_fallback
    passing hf_fallback_model (if provided), otherwise let generate_with_hf_fallback
    consult HF_FALLBACK_MODEL or default to gpt2.
    """
    attempt = 0
    while attempt < max_retries:
        try:
            # Real Ollama/primary provider call should go here in production code.
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
                return generate_with_hf_fallback(prompt, model_name=hf_fallback_model, max_new_tokens=200)
            else:
                return f"Error: {e}"

    return "Error: Generation failed and no fallback available"


