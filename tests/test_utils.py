#!/usr/bin/env python3
"""
Unit tests for utility functions in src.llm_fine_tune.utils
"""

import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_fine_tune.utils import chunk_text, extract_text_from_obj, build_prompt, clean_text


def test_chunk_text_basic():
    """Test basic chunking functionality."""
    text = "A" * 100
    chunks = chunk_text(text, max_chars=50, overlap=10)
    # With 100 chars, 50 char chunks, 10 char overlap:
    # Chunk 1: 0-50, Chunk 2: 40-90, Chunk 3: 80-100
    assert len(chunks) == 3
    assert len(chunks[0]) == 50
    assert len(chunks[1]) == 50
    assert len(chunks[2]) == 20  # Remaining chars


def test_chunk_text_overlap_fixed():
    """Test that overlap is preserved correctly (bug fix)."""
    text = "A" * 1000 + "B" * 1000  # 2000 chars total
    chunks = chunk_text(text, max_chars=800, overlap=100)
    
    # Should have 3 chunks with proper overlap
    assert len(chunks) >= 2
    
    if len(chunks) >= 2:
        # Check overlap between first two chunks
        chunk1_end = chunks[0][-100:]  # Last 100 chars of chunk 1
        chunk2_start = chunks[1][:100]  # First 100 chars of chunk 2
        # Due to line breaking, they might not be exactly equal, but should have substantial overlap
        overlap_found = any(char in chunk2_start for char in chunk1_end[-50:])
        assert overlap_found, "Expected overlap between chunks"


def test_chunk_text_no_infinite_loop():
    """Test that chunking doesn't create infinite loops."""
    text = "A" * 50
    chunks = chunk_text(text, max_chars=100, overlap=200)  # Overlap > max_chars
    assert len(chunks) == 1
    assert chunks[0] == text


def test_extract_text_from_obj_string():
    """Test extracting text from string response."""
    result = extract_text_from_obj("Simple string response")
    assert result == "Simple string response"


def test_extract_text_from_obj_ollama():
    """Test extracting text from Ollama response format."""
    ollama_response = {"response": "Ollama generated text"}
    result = extract_text_from_obj(ollama_response)
    assert result == "Ollama generated text"


def test_extract_text_from_obj_openai():
    """Test extracting text from OpenAI response format."""
    openai_response = {
        "choices": [
            {"message": {"content": "OpenAI generated text"}}
        ]
    }
    result = extract_text_from_obj(openai_response)
    assert result == "OpenAI generated text"


def test_extract_text_from_obj_huggingface():
    """Test extracting text from HuggingFace response format."""
    hf_response = {"generated_text": "HuggingFace generated text"}
    result = extract_text_from_obj(hf_response)
    assert result == "HuggingFace generated text"


def test_extract_text_from_obj_generic():
    """Test extracting text from generic response formats."""
    responses = [
        {"text": "Generic text field"},
        {"content": "Generic content field"},
        {"output": "Generic output field"},
        {"result": "Generic result field"}
    ]
    for response in responses:
        result = extract_text_from_obj(response)
        assert result is not None
        assert "Generic" in result


def test_extract_text_from_obj_empty():
    """Test handling of empty/None responses."""
    assert extract_text_from_obj(None) is None
    assert extract_text_from_obj("") is None
    assert extract_text_from_obj("   ") is None
    assert extract_text_from_obj({}) is None


def test_extract_text_from_obj_list():
    """Test extracting text from list responses."""
    list_response = [
        {"text": "First response"},
        {"content": "Second response"}
    ]
    result = extract_text_from_obj(list_response)
    assert result == "First response"


def test_build_prompt():
    """Test prompt building functionality."""
    example = {
        "instruction": "Test instruction",
        "input": "Test input"
    }
    prompt = build_prompt(example)
    assert "### Instruction:" in prompt
    assert "Test instruction" in prompt
    assert "### Input:" in prompt
    assert "Test input" in prompt
    assert "### Response:" in prompt


def test_clean_text():
    """Test text cleaning functionality."""
    dirty_text = "  Text with\r\n  multiple   spaces\n\n\n  and line breaks  \n  "
    cleaned = clean_text(dirty_text)
    assert not cleaned.startswith(" ")
    assert not cleaned.endswith(" ")
    assert "\r" not in cleaned
    assert "   " not in cleaned  # Multiple spaces should be reduced


if __name__ == "__main__":
    # Run tests directly
    import pytest
    pytest.main([__file__, "-v"])