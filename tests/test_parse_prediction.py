"""
Unit tests for src/parse_prediction.py

Run with:  python -m pytest tests/test_parse_prediction.py -v
"""
import numpy as np
import pytest

from src.parse_prediction import (
    extract_first_inverse_matrix,
    normalize_prediction_text,
    parse_prediction,
    parse_to_matrix,
)


# ---------------------------------------------------------------------------
# normalize_prediction_text
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_none_returns_empty(self):
        assert normalize_prediction_text(None) == ''

    def test_special_tokens_replaced(self):
        text = '<|begin_of_text|>Inverse:\n1, 2\n3, 4<|end_of_text|>'
        result = normalize_prediction_text(text)
        assert '<|begin_of_text|>' not in result
        assert '<|end_of_text|>' not in result
        assert 'Inverse:' in result

    def test_inline_header_split_onto_own_line(self):
        # number immediately followed by Inverse: must be separated
        text = '-0.88582Inverse:\n1, 2\n3, 4'
        result = normalize_prediction_text(text)
        lines = result.splitlines()
        assert any(l.strip().lower() == 'inverse:' for l in lines)

    def test_triple_newlines_collapsed(self):
        text = 'Inverse:\n\n\n\n1, 2'
        result = normalize_prediction_text(text)
        assert '\n\n\n' not in result

    def test_crlf_normalized(self):
        text = 'Inverse:\r\n1, 2\r\n3, 4'
        result = normalize_prediction_text(text)
        assert '\r' not in result


# ---------------------------------------------------------------------------
# Happy-path extraction
# ---------------------------------------------------------------------------

class TestHappyPath:
    def test_clean_3x3(self):
        text = 'Inverse:\n1.0, 2.0, 3.0\n4.0, 5.0, 6.0\n7.0, 8.0, 9.0'
        rows = extract_first_inverse_matrix(normalize_prediction_text(text), 3)
        assert rows == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]

    def test_scientific_notation(self):
        text = 'Inverse:\n1.0e-3, -2.5E+1\n3.14e0, 0.0'
        rows = extract_first_inverse_matrix(normalize_prediction_text(text), 2)
        assert rows is not None
        assert abs(rows[0][0] - 1e-3) < 1e-10
        assert abs(rows[0][1] - (-25.0)) < 1e-10

    def test_case_insensitive_header(self):
        text = 'INVERSE:\n1.0, 2.0\n3.0, 4.0'
        rows = extract_first_inverse_matrix(normalize_prediction_text(text), 2)
        assert rows == [[1.0, 2.0], [3.0, 4.0]]

    def test_leading_junk_before_header_is_ignored(self):
        text = 'Some instruction text\nFormatting rules\nInverse:\n1.0, 2.0\n3.0, 4.0'
        rows = extract_first_inverse_matrix(normalize_prediction_text(text), 2)
        assert rows == [[1.0, 2.0], [3.0, 4.0]]

    def test_instruction_text_before_first_row_is_skipped(self):
        text = 'Inverse:\nHere is the matrix:\n1.0, 2.0\n3.0, 4.0'
        rows = extract_first_inverse_matrix(normalize_prediction_text(text), 2)
        assert rows == [[1.0, 2.0], [3.0, 4.0]]

    def test_special_tokens_around_header(self):
        text = '<|begin_of_text|>Inverse:\n1.0, 2.0\n3.0, 4.0<|end_of_text|>'
        rows = extract_first_inverse_matrix(normalize_prediction_text(text), 2)
        assert rows == [[1.0, 2.0], [3.0, 4.0]]

    def test_inline_header_on_number_line(self):
        # Model ran "...numberInverse:\nrow1\nrow2"
        text = '-0.12345Inverse:\n1.0, 2.0\n3.0, 4.0'
        rows = extract_first_inverse_matrix(normalize_prediction_text(text), 2)
        assert rows == [[1.0, 2.0], [3.0, 4.0]]


# ---------------------------------------------------------------------------
# Loop / repeated-block protection
# ---------------------------------------------------------------------------

class TestRepeatedBlocks:
    def test_only_first_block_used(self):
        text = (
            'Inverse:\n1.0, 2.0\n3.0, 4.0\n'
            'Inverse:\n9.0, 8.0\n7.0, 6.0\n'
        )
        rows = extract_first_inverse_matrix(normalize_prediction_text(text), 2)
        assert rows == [[1.0, 2.0], [3.0, 4.0]]

    def test_many_repeated_blocks_stops_after_first(self):
        block = 'Inverse:\n1.0, 0.0\n0.0, 1.0\n'
        text = block * 50          # simulates max_tokens loop
        rows = extract_first_inverse_matrix(normalize_prediction_text(text), 2)
        assert rows == [[1.0, 0.0], [0.0, 1.0]]

    def test_bos_tokens_between_blocks(self):
        text = (
            'Inverse:\n1.0, 2.0\n3.0, 4.0\n'
            '<|begin_of_text|>Inverse:\n9.0, 8.0\n7.0, 6.0'
        )
        rows = extract_first_inverse_matrix(normalize_prediction_text(text), 2)
        assert rows == [[1.0, 2.0], [3.0, 4.0]]


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------

class TestFailureModes:
    def test_no_header_returns_none(self):
        text = '1.0, 2.0\n3.0, 4.0'
        rows = extract_first_inverse_matrix(normalize_prediction_text(text), 2)
        assert rows is None

    def test_empty_string_returns_none(self):
        assert extract_first_inverse_matrix('', 2) is None

    def test_only_special_tokens_returns_none(self):
        text = '<|begin_of_text|><|end_of_text|>'
        assert extract_first_inverse_matrix(normalize_prediction_text(text), 2) is None

    def test_wrong_column_count_breaks_immediately(self):
        # First data row has 3 values but matrix_size=2 → should fail
        text = 'Inverse:\n1.0, 2.0, 3.0\n4.0, 5.0, 6.0'
        rows = extract_first_inverse_matrix(normalize_prediction_text(text), 2)
        assert rows is None

    def test_too_few_rows_returns_none(self):
        text = 'Inverse:\n1.0, 2.0'   # only 1 row for a 2×2
        rows = extract_first_inverse_matrix(normalize_prediction_text(text), 2)
        assert rows is None

    def test_blank_line_mid_block_stops_collection(self):
        # blank line after first row should stop; only 1 of 3 rows collected
        text = 'Inverse:\n1.0, 2.0, 3.0\n\n4.0, 5.0, 6.0\n7.0, 8.0, 9.0'
        rows = extract_first_inverse_matrix(normalize_prediction_text(text), 3)
        assert rows is None

    def test_instruction_text_mid_block_stops_collection(self):
        text = 'Inverse:\n1.0, 2.0\nDo not add extra lines\n3.0, 4.0'
        rows = extract_first_inverse_matrix(normalize_prediction_text(text), 2)
        assert rows is None

    def test_all_instruction_text_no_numbers(self):
        text = (
            'Inverse:\n'
            '- No extra lines before or after the line "Inverse:".\n'
            '- Respond with no explanation.\n'
        )
        rows = extract_first_inverse_matrix(normalize_prediction_text(text), 3)
        assert rows is None


# ---------------------------------------------------------------------------
# parse_prediction dict API
# ---------------------------------------------------------------------------

class TestParsePrediction:
    def test_success_keys(self):
        text = 'Inverse:\n1.0, 2.0\n3.0, 4.0'
        result = parse_prediction(text, 2)
        assert result['parse_success'] is True
        assert result['is_correct_shape'] is True
        assert result['extracted_values_count'] == 4
        assert result['parsed_matrix'] == [[1.0, 2.0], [3.0, 4.0]]
        assert 'normalized_text' in result

    def test_failure_keys(self):
        result = parse_prediction('no matrix here', 2)
        assert result['parse_success'] is False
        assert result['is_correct_shape'] is False
        assert result['extracted_values_count'] == 0
        assert result['parsed_matrix'] is None

    def test_none_input(self):
        result = parse_prediction(None, 2)
        assert result['parse_success'] is False


# ---------------------------------------------------------------------------
# parse_to_matrix legacy shim
# ---------------------------------------------------------------------------

class TestParseToMatrix:
    def test_returns_ndarray_on_success(self):
        text = 'Inverse:\n1.0, 2.0\n3.0, 4.0'
        matrix, meta = parse_to_matrix(text, 2)
        assert isinstance(matrix, np.ndarray)
        assert matrix.shape == (2, 2)
        assert meta['parse_success'] is True
        assert meta['is_correct_shape'] is True

    def test_returns_none_on_failure(self):
        matrix, meta = parse_to_matrix('no matrix', 2)
        assert matrix is None
        assert meta['parse_success'] is False

    def test_meta_has_legacy_keys(self):
        _, meta = parse_to_matrix('Inverse:\n1.0, 2.0\n3.0, 4.0', 2)
        assert 'raw_text' in meta
        assert 'extracted_values_count' in meta
