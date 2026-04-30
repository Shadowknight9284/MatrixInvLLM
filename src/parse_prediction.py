from __future__ import annotations

import re
import numpy as np
from typing import Any

_NUMBER_RE = r'[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?'
_ROW_RE = re.compile(rf'^\s*{_NUMBER_RE}\s*(?:,\s*{_NUMBER_RE}\s*)*$')
_HEADER_RE = re.compile(r'^\s*inverse\s*:\s*$', re.IGNORECASE)
_INLINE_HEADER_RE = re.compile(r'(?i)inverse\s*:')
_SPECIAL_TOKENS = [
    '<|begin_of_text|>',
    '<|end_of_text|>',
    '<s>',
    '</s>',
]


def normalize_prediction_text(text: str) -> str:
    if text is None:
        return ''
    normalized = str(text)
    for token in _SPECIAL_TOKENS:
        normalized = normalized.replace(token, '\n')
    normalized = normalized.replace('\r\n', '\n').replace('\r', '\n')
    normalized = _INLINE_HEADER_RE.sub('\nInverse:\n', normalized)
    normalized = re.sub(r'\n{3,}', '\n\n', normalized)
    return normalized.strip()


def is_numeric_row(line: str) -> bool:
    return bool(_ROW_RE.match(line.strip()))


def parse_numeric_row(line: str) -> list[float] | None:
    line = line.strip()
    if not is_numeric_row(line):
        return None
    parts = [p.strip() for p in line.split(',')]
    try:
        return [float(p) for p in parts]
    except ValueError:
        return None


def extract_first_inverse_matrix(normalized: str, matrix_size: int) -> list[list[float]] | None:
    """
    Parse a pre-normalized string and return the first complete n×n numeric
    block that follows an 'Inverse:' header.

    Stops collecting rows on any of:
      - blank line after at least one row has been collected
      - second 'Inverse:' header
      - non-numeric line once row collection has started
      - numeric row with wrong column count
    """
    lines = normalized.splitlines()

    started = False
    rows: list[list[float]] = []

    for raw in lines:
        line = raw.strip()
        if not started:
            if _HEADER_RE.match(line):
                started = True
            continue

        if not line:
            if rows:
                break
            continue

        if _HEADER_RE.match(line):
            break

        parsed = parse_numeric_row(line)
        if parsed is None:
            if rows:
                break
            continue

        if len(parsed) != matrix_size:
            break

        rows.append(parsed)
        if len(rows) == matrix_size:
            return rows

    return None


def parse_prediction(prediction: str, matrix_size: int) -> dict[str, Any]:
    """
    Normalize `prediction` and extract the first valid n×n inverse matrix.

    Returns a dict with:
      parse_success         bool
      is_correct_shape      bool
      extracted_values_count int
      parsed_matrix         list[list[float]] | None
      normalized_text       str
    """
    normalized = normalize_prediction_text(prediction)
    matrix = extract_first_inverse_matrix(normalized, matrix_size)

    if matrix is None:
        return {
            'parse_success': False,
            'is_correct_shape': False,
            'extracted_values_count': 0,
            'parsed_matrix': None,
            'normalized_text': normalized,
        }

    return {
        'parse_success': True,
        'is_correct_shape': True,   # guaranteed by extraction logic
        'extracted_values_count': matrix_size * matrix_size,
        'parsed_matrix': matrix,
        'normalized_text': normalized,
    }


# ---------------------------------------------------------------------------
# Backwards-compatibility shim — score_prediction.py imports this signature
# ---------------------------------------------------------------------------

def parse_to_matrix(text: str, size: int) -> tuple[np.ndarray | None, dict[str, Any]]:
    """
    Legacy interface used by score_prediction.py.
    Delegates to parse_prediction and converts the list matrix to ndarray.
    """
    result = parse_prediction(text, size)
    meta = {
        'parse_success': result['parse_success'],
        'is_correct_shape': result['is_correct_shape'],
        'extracted_values_count': result['extracted_values_count'],
        'raw_text': text,
    }
    if result['parsed_matrix'] is None:
        return None, meta
    return np.array(result['parsed_matrix'], dtype=float), meta
