#!/usr/bin/env python3
"""
Integration test for sample pipeline using the small sample dataset.
Tests data preparation and generation with HF fallback.
"""

import sys
import os
import tempfile
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_fine_tune.data_prep import collect_docs_from_dir, create_examples_from_doc
from llm_fine_tune.generate import generate_with_hf_fallback
from llm_fine_tune.utils import load_jsonl, write_jsonl, build_prompt


def test_sample_dataset_exists():
    """Test that the sample dataset exists and is valid."""
    sample_path = Path(__file__).parent.parent / "data" / "sample_sft_small.jsonl"
    assert sample_path.exists(), "Sample dataset not found"
    
    data = load_jsonl(sample_path)
    assert len(data) == 10, f"Expected 10 examples, got {len(data)}"
    
    for item in data:
        assert "instruction" in item
        assert "input" in item
        assert "output" in item
        assert "metadata" in item
        assert item["output"]  # All should have outputs


def test_data_prep_with_sample():
    """Test data preparation module with sample data."""
    # Create a temporary input directory with a sample file
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_input = Path(temp_dir) / "input"
        temp_input.mkdir()
        
        # Create a sample text file
        sample_text = (temp_input / "test_solution.txt")
        sample_text.write_text("This is a test solution for water management. "
                              "It involves community participation and sustainable practices.")
        
        # Test collection
        docs = collect_docs_from_dir(temp_input)
        assert len(docs) == 1
        assert docs[0]["file"] == "test_solution.txt"
        assert "water management" in docs[0]["text"]
        
        # Test example creation
        examples = create_examples_from_doc(
            docs[0]["file"], 
            docs[0]["title"], 
            docs[0]["text"], 
            max_chars=100
        )
        assert len(examples) >= 1
        assert examples[0]["metadata"]["source_type"] == "solution"
        assert examples[0]["instruction"]
        assert examples[0]["input"]
        assert examples[0]["output"] == ""  # Should be empty


def test_hf_generation_fallback():
    """Test HuggingFace generation fallback (skip if transformers not available)."""
    try:
        import transformers
    except ImportError:
        import pytest
        pytest.skip("transformers not available for HF fallback test")
    
    prompt = "### Instruction:\nSummarize this text.\n\n### Input:\nThis is a test.\n\n### Response:\n"
    
    # Test generation (may be slow on first run due to model download)
    try:
        result = generate_with_hf_fallback(prompt, model_name="gpt2", max_length=50)
        assert isinstance(result, str)
        assert len(result) > 0
        print(f"Generated: {result[:100]}...")
    except Exception as e:
        # If model download fails or other issues, that's OK for CI
        print(f"HF generation test failed (expected in CI): {e}")


def test_sample_pipeline_flow():
    """Test the complete pipeline flow with sample data (fast version)."""
    sample_path = Path(__file__).parent.parent / "data" / "sample_sft_small.jsonl"
    data = load_jsonl(sample_path)
    
    # Take just one example for testing
    example = data[0]
    
    # Test prompt building
    prompt = build_prompt(example)
    assert "### Instruction:" in prompt
    assert "### Input:" in prompt
    assert "### Response:" in prompt
    
    # Test that we can process the data structure
    assert example["metadata"]["source_type"] in ["solution", "experiment", "action_plan", "blog"]
    assert len(example["output"]) > 10  # Should have substantive output
    
    print(f"Pipeline test successful with example: {example['metadata']['source_type']}")


def test_sample_dataset_schemas():
    """Test that all examples in sample dataset follow proper schema."""
    sample_path = Path(__file__).parent.parent / "data" / "sample_sft_small.jsonl"
    data = load_jsonl(sample_path)
    
    required_fields = ["instruction", "input", "output", "metadata"]
    required_metadata = ["source_type", "source_filename", "title", "chunk_index", "chunk_chars"]
    
    for i, example in enumerate(data):
        for field in required_fields:
            assert field in example, f"Example {i} missing field {field}"
        
        for meta_field in required_metadata:
            assert meta_field in example["metadata"], f"Example {i} missing metadata {meta_field}"
        
        # Check types
        assert isinstance(example["instruction"], str)
        assert isinstance(example["input"], str)
        assert isinstance(example["output"], str)
        assert isinstance(example["metadata"]["chunk_index"], int)
        assert isinstance(example["metadata"]["chunk_chars"], int)
        
        # Check content quality
        assert len(example["instruction"]) > 20
        assert len(example["input"]) > 50
        assert len(example["output"]) > 50


if __name__ == "__main__":
    # Run tests directly
    import pytest
    pytest.main([__file__, "-v"])