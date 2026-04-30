from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


# --------------------------------------------------------------------
# Dummy structures so you can plug in real Tinker outputs later
# --------------------------------------------------------------------


@dataclass
class DummySequence:
    tokens: List[int]


@dataclass
class DummyResult:
    sequences: List[DummySequence]


class DummyTokenizer:
    """
    Minimal tokenizer stub.

    In your real code, replace this with your actual tokenizer; the
    interface we rely on is just: decode(List[int]) -> str.
    """

    def __init__(self, vocab=None):
        self.vocab = vocab or {}

    def decode(self, tokens: List[int]) -> str:
        # For diagnostics we assume tokens are already character codes.
        # In real usage, your tokenizer will map IDs to text.
        try:
            return "".join(chr(t) for t in tokens)
        except Exception:
            return ""


# --------------------------------------------------------------------
# Prompt building (diagnostic version of build_prompt)
# --------------------------------------------------------------------


def build_prompt(raw_prompt: str, matrix_size: int) -> str:
    """
    Build a strict, numeric-only prompt for inverse prediction.

    This mirrors the tightened instructions we discussed:
    - no explanations
    - no examples
    - concrete numeric format
    """
    n = matrix_size
    example_row = ", ".join(["0.00000"] * n)

    instruction = (
        f"\n\nYou are given MatrixSize and Matrix.\n"
        f"Compute the inverse of the matrix.\n\n"
        f"Respond with no explanation, no examples, and no commentary.\n"
        f"Respond in exactly the following format and nothing else:\n\n"
        f"Inverse:\n"
        f"{example_row}\n"
        f"...\n"
        f"{example_row}\n\n"
        f"Formatting rules:\n"
        f"- Exactly {n} rows after the line \"Inverse:\".\n"
        f"- Each row must contain exactly {n} comma-separated numbers.\n"
        f"- Use exactly 5 decimal places for every number.\n"
        f"- Do NOT repeat the Matrix.\n"
        f"- Do NOT include any words or sentences.\n"
        f"- No extra lines before or after the numeric rows.\n"
    )

    full_prompt = raw_prompt + instruction
    logger.info("Built prompt (first 400 chars): %r", full_prompt[:400])
    return full_prompt


# --------------------------------------------------------------------
# Model call diagnostics: decode & clean
# --------------------------------------------------------------------


def decode_and_clean(result: DummyResult, tokenizer: DummyTokenizer) -> str:
    """
    Decode a model result, log key stats, and strip obvious special tokens.

    This is where you check:
    - Did we get any tokens at all?
    - Is it only EOS/special tokens?
    - What does the first part of the raw text look like?
    """
    if not hasattr(result, "sequences") or not result.sequences:
        logger.error("No sequences returned from model.")
        return ""

    seq = result.sequences[0]
    logger.info("Generated token count: %d", len(seq.tokens))

    raw_text = tokenizer.decode(seq.tokens)
    logger.info("Raw decoded text (first 300 chars): %r", raw_text[:300])

    # Strip common special tokens and whitespace
    text = (
        raw_text.replace("<|end_of_text|>", "")
        .replace("<|eot_id|>", "")
        .replace("endoftext", "")
        .strip()
    )

    if not text:
        logger.warning("Model returned only EOS/special tokens after cleaning.")
        return ""

    logger.info("Cleaned text (first 300 chars): %r", text[:300])
    return text


# --------------------------------------------------------------------
# Numeric-only filtering (drops prose, keeps candidate rows)
# --------------------------------------------------------------------


def strip_to_numeric_block(text: str) -> str:
    """
    Keep only numeric-looking lines and drop prose.

    - Removes any line that contains letters.
    - Ignores the 'Inverse:' header.
    - Keeps lines that start with digit / '-' / '.'.
    """
    numeric_lines: List[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        # Skip header
        if stripped.lower().startswith("inverse:"):
            continue

        # If the line has any alphabetic characters, drop it
        if re.search(r"[A-Za-z]", stripped):
            logger.debug("Dropping non-numeric line: %r", stripped)
            continue

        # Heuristic: keep lines that look like rows of numbers
        if stripped[0] in "0123456789-.":
            numeric_lines.append(stripped)
        else:
            logger.debug("Dropping line that does not look numeric: %r", stripped)

    numeric_text = "\n".join(numeric_lines)
    logger.info("Numeric-only text (first 300 chars): %r", numeric_text[:300])
    return numeric_text


# --------------------------------------------------------------------
# Simple shape / parsing check
# --------------------------------------------------------------------


def parse_matrix(text: str, expected_size: int) -> Optional[List[List[float]]]:
    """
    Minimal parser to check if we got an n x n matrix.

    This is NOT meant to replace your production parser; it's for debugging:
    - Do we get exactly n lines?
    - Does each line have exactly n comma-separated numbers?
    """
    if not text.strip():
        logger.warning("No numeric content to parse.")
        return None

    rows = [ln for ln in text.splitlines() if ln.strip()]
    logger.info("Row count after filtering: %d", len(rows))

    if len(rows) != expected_size:
        logger.warning("Expected %d rows but got %d.", expected_size, len(rows))
        return None

    matrix: List[List[float]] = []

    for i, row in enumerate(rows):
        parts = [p.strip() for p in row.split(",") if p.strip()]
        logger.info("Row %d column count: %d", i, len(parts))

        if len(parts) != expected_size:
            logger.warning(
                "Row %d: expected %d columns but got %d.",
                i,
                expected_size,
                len(parts),
            )
            return None

        try:
            matrix.append([float(p) for p in parts])
        except ValueError as exc:
            logger.warning("Row %d: failed to parse float: %s", i, exc)
            return None

    logger.info("Parsed a valid %dx%d matrix.", expected_size, expected_size)
    return matrix


# --------------------------------------------------------------------
# End-to-end diagnostic helper
# --------------------------------------------------------------------


def diagnose_prediction(
    raw_prompt: str,
    matrix_size: int,
    result: DummyResult,
    tokenizer: DummyTokenizer,
) -> None:
    """
    Run the full diagnostic pipeline and log what's going wrong.

    Steps:
    1) Build the prompt (for inspection only).
    2) Decode and clean the model result.
    3) Strip to numeric-only lines.
    4) Try to parse an n x n matrix.

    In your real pipeline you can:
    - Replace DummyResult/DummyTokenizer with real types.
    - Call this right after each Tinker call to see where failures occur.
    """
    _ = build_prompt(raw_prompt, matrix_size)

    text = decode_and_clean(result, tokenizer)
    if not text:
        logger.error("No usable text from model; aborting diagnostics.")
        return

    numeric_text = strip_to_numeric_block(text)

    # Optional: very strict check – reject if any letters remain
    if re.search(r"[A-Za-z]", numeric_text):
        logger.warning(
            "Letters still present after numeric filtering; this should not happen."
        )

    matrix = parse_matrix(numeric_text, matrix_size)
    if matrix is None:
        logger.error(
            "Prediction did not yield a valid %dx%d matrix.",
            matrix_size,
            matrix_size,
        )
    else:
        logger.info("Matrix looks well-formed; ready for numerical scoring.")


# --------------------------------------------------------------------
# Example usage
# --------------------------------------------------------------------


if __name__ == "__main__":
    # Tiny demo with a fake 'good' prediction for a 2x2 case.
    demo_text = "Inverse:\n0.1, 0.2\n0.3, 0.4\n<|end_of_text|>"
    demo_tokens = [ord(c) for c in demo_text]
    demo_result = DummyResult(sequences=[DummySequence(tokens=demo_tokens)])
    demo_tokenizer = DummyTokenizer()

    diagnose_prediction(
        raw_prompt="MatrixSize 2 1.0 0.0 0.0 1.0",
        matrix_size=2,
        result=demo_result,
        tokenizer=demo_tokenizer,
    )