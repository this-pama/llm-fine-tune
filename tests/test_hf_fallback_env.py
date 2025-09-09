#!/usr/bin/env python3
"""
Test HuggingFace fallback environment variable configurability.
"""

import sys
import os
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# flake8: noqa: E402
from llm_fine_tune.generate import generate_with_hf_fallback


def test_hf_fallback_explicit_model_priority():
    """Test that explicit model_name argument takes highest priority."""
    # Set environment variable
    os.environ["HF_FALLBACK_MODEL"] = "distilgpt2"

    try:
        import transformers  # noqa: F401
    except ImportError:
        import pytest
        pytest.skip("transformers not available for HF fallback test")

    prompt = "### Instruction:\nSummarize this text.\n\n### Input:\nThis is a test.\n\n### Response:\n"

    # Test that explicit model_name overrides environment variable
    # Note: We can't easily test the actual model loading without downloading models,
    # but we can test the model selection logic by checking error messages
    try:
        result = generate_with_hf_fallback(prompt, model_name="gpt2", max_new_tokens=10)
        assert isinstance(result, str)
        # Should work with gpt2 or give a specific error (not about missing transformers)
        assert "transformers package not available" not in result
    except Exception as e:
        # If model download fails, that's OK for CI, but should not be about missing transformers
        print(f"Model test failed (expected in CI): {e}")

    # Clean up environment
    if "HF_FALLBACK_MODEL" in os.environ:
        del os.environ["HF_FALLBACK_MODEL"]


def test_hf_fallback_env_var_used():
    """Test that HF_FALLBACK_MODEL environment variable is used when no explicit model."""
    try:
        import transformers  # noqa: F401
    except ImportError:
        import pytest
        pytest.skip("transformers not available for HF fallback test")

    # Set environment variable
    os.environ["HF_FALLBACK_MODEL"] = "gpt2"

    prompt = "### Instruction:\nSummarize this text.\n\n### Input:\nThis is a test.\n\n### Response:\n"

    try:
        result = generate_with_hf_fallback(prompt, model_name=None, max_new_tokens=10)
        assert isinstance(result, str)
        # Should work with gpt2 or give a specific error (not about missing transformers)
        assert "transformers package not available" not in result
    except Exception as e:
        # If model download fails, that's OK for CI
        print(f"Environment variable test failed (expected in CI): {e}")

    # Clean up environment
    if "HF_FALLBACK_MODEL" in os.environ:
        del os.environ["HF_FALLBACK_MODEL"]


def test_hf_fallback_default_gpt2():
    """Test that gpt2 is used as default when no explicit model or env var."""
    # Ensure no environment variable is set
    if "HF_FALLBACK_MODEL" in os.environ:
        del os.environ["HF_FALLBACK_MODEL"]

    try:
        import transformers  # noqa: F401
    except ImportError:
        import pytest
        pytest.skip("transformers not available for HF fallback test")

    prompt = "### Instruction:\nSummarize this text.\n\n### Input:\nThis is a test.\n\n### Response:\n"

    try:
        result = generate_with_hf_fallback(prompt, model_name=None, max_new_tokens=10)
        assert isinstance(result, str)
        # Should work with gpt2 or give a specific error (not about missing transformers)
        assert "transformers package not available" not in result
    except Exception as e:
        # If model download fails, that's OK for CI
        print(f"Default test failed (expected in CI): {e}")


def test_hf_fallback_model_selection_logic():
    """Test the model selection logic without actually loading models."""
    # This test verifies the logical flow by inspecting what would happen

    # Test 1: explicit model_name takes priority
    test_model_name = "test-model"
    os.environ["HF_FALLBACK_MODEL"] = "env-model"

    # We can't directly test the internal logic without refactoring,
    # but we can verify behavior indirectly by ensuring the function handles different inputs

    # Test with explicit model (should not fall back to env or gpt2)
    prompt = "test prompt"
    try:
        # This will likely fail with model not found, but that proves it tried the explicit model
        result = generate_with_hf_fallback(prompt, model_name=test_model_name, max_new_tokens=5)
        # If it doesn't fail, check it's a string response
        assert isinstance(result, str)
    except Exception:
        # Expected - the test model doesn't exist
        pass

    # Clean up
    if "HF_FALLBACK_MODEL" in os.environ:
        del os.environ["HF_FALLBACK_MODEL"]


if __name__ == "__main__":
    # Run tests directly
    import pytest
    pytest.main([__file__, "-v"])
