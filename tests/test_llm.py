import pytest
from ai_scientist.llm import extract_json_between_markers

def test_extract_json_unmarked():
    """Test parsing unmarked but valid JSON."""
    valid_json = '{"key": "value"}'
    result = extract_json_between_markers(valid_json, strict_mode=False)
    assert result == {"key": "value"}

def test_extract_json_marked():
    """Test parsing JSON with markers."""
    marked_json = '```json\n{"key": "value"}\n```'
    result = extract_json_between_markers(marked_json)
    assert result == {"key": "value"}

def test_extract_json_fallback():
    """Test fallback behavior with invalid marked JSON."""
    invalid_marked = '```json\nInvalid JSON\n```'
    valid_unmarked = '{"key": "value"}'
    combined = f"{invalid_marked}\n{valid_unmarked}"
    result = extract_json_between_markers(combined, strict_mode=False)
    assert result == {"key": "value"}

def test_extract_json_strict_mode_error():
    """Test strict mode error handling."""
    valid_json = '{"key": "value"}'
    with pytest.raises(ValueError, match="No JSON markers found in output"):
        extract_json_between_markers(valid_json, strict_mode=True)

def test_extract_json_invalid_content():
    """Test handling of invalid JSON content."""
    invalid_json = '{"key": value}'  # Missing quotes around value
    with pytest.raises(ValueError, match="Failed to parse any JSON content"):
        extract_json_between_markers(invalid_json, strict_mode=False)

def test_extract_json_no_content():
    """Test handling of empty or non-JSON content."""
    with pytest.raises(ValueError, match="Failed to parse any JSON content"):
        extract_json_between_markers("No JSON here", strict_mode=False)
