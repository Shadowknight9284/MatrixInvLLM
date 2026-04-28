import re
import numpy as np
import logging

logger = logging.getLogger(__name__)

def extract_floats(text):
    """
    Extract all floating point numbers from a text string.
    Handles scientific notation as well.
    """
    # Regex for floats: handles -1.23, 10, .5, 1e-5, 1.2E+10
    pattern = r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)?"
    matches = re.finditer(pattern, text)
    floats = [float(m.group(0)) for m in matches]
    return floats

def parse_to_matrix(text, size):
    """
    Attempt to parse text into a square matrix of a given size.
    Returns (matrix, metadata).
    """
    metadata = {
        "parse_success": False,
        "is_correct_shape": False,
        "raw_text": text,
        "extracted_values_count": 0
    }
    
    # Pre-processing: Look for headers like "Inverse:" to narrow the search?
    # For now, let's just extract all floats and see if the count matches.
    # The user might have preamble or postamble.
    
    values = extract_floats(text)
    metadata["extracted_values_count"] = len(values)
    
    expected_count = size * size
    
    if len(values) == 0:
        return None, metadata
    
    metadata["parse_success"] = True
    
    if len(values) != expected_count:
        logger.debug(f"Shape mismatch: expected {expected_count} values, got {len(values)}")
        return None, metadata
    
    try:
        matrix = np.array(values).reshape((size, size))
        metadata["is_correct_shape"] = True
        return matrix, metadata
    except Exception as e:
        logger.error(f"Error reshaping matrix: {e}")
        return None, metadata
