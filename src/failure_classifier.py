"""
failure_classifier.py
---------------------
Classify structured-output failures into four mutually exclusive buckets:

  EOS_ONLY         – Model emitted nothing (or only EOS/special tokens).
                     Root causes: prompt-template mismatch, EOS misconfiguration,
                     adapter load failure, or chat-template incompatibility.

  PROMPT_ECHO      – Model copied instruction text before (or instead of) the
                     numeric answer.  Root cause: weak schema anchoring.

  SHAPE_FORMAT     – Some numbers were emitted but the matrix shape is wrong
                     (wrong row count or column count).

  NUMERICALLY_WRONG – Shape is correct but the prediction fails all tolerance
                      levels.  Pure mathematical failure.

  SUCCESS          – Prediction passes at least the loose tolerance level.

Usage
-----
  from src.failure_classifier import classify_failure, summarize_failures

  mode = classify_failure(
      raw_text        = diag["raw_text"],
      cleaned_text    = pred_text,
      parse_meta      = parse_meta,   # dict from parse_to_matrix
      scores          = scores,       # dict from score_example
  )

  breakdown = summarize_failures(results)  # list of result dicts with "failure_mode"
"""

from __future__ import annotations

import re
import logging
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Instruction keywords used to detect prompt-echo
# ---------------------------------------------------------------------------
_ECHO_KEYWORDS = [
    "Formatting rules",
    "Respond with no explanation",
    "Do NOT repeat",
    "Do NOT include",
    "No extra lines",
    "comma-separated numbers",
    "decimal places for every",
    "exactly the following format",
]

# How many characters of cleaned_text to inspect for echo detection
_ECHO_SCAN_CHARS = 600


class FailureMode(str, Enum):
    """
    Ordered from 'most broken' to 'least broken'.

    Using str mixin so instances serialize cleanly as plain strings in JSON.
    """
    EOS_ONLY            = "EOS_ONLY"
    PROMPT_ECHO         = "PROMPT_ECHO"
    REPEATED_BLOCK_LOOP = "REPEATED_BLOCK_LOOP"
    SHAPE_FORMAT        = "SHAPE_FORMAT"
    NUMERICALLY_WRONG   = "NUMERICALLY_WRONG"
    SUCCESS             = "SUCCESS"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_eos_only(raw_text: str, cleaned_text: str) -> bool:
    """
    True when the model produced no usable content.

    This covers two scenarios:
    - raw_text is genuinely empty (nothing was decoded at all).
    - raw_text contains only special/EOS tokens so cleaned_text is empty
      after stripping them.
    """
    if not cleaned_text or not cleaned_text.strip():
        return True
    return False


def _is_prompt_echo(cleaned_text: str) -> bool:
    """
    True when the model copied instruction boilerplate into its output.

    Heuristic: if two or more known instruction keywords appear in the first
    _ECHO_SCAN_CHARS characters of the cleaned output, we call it an echo.
    A single match can be coincidental; two or more is diagnostic.
    """
    sample = cleaned_text[:_ECHO_SCAN_CHARS]
    hits = sum(1 for kw in _ECHO_KEYWORDS if kw.lower() in sample.lower())
    if hits >= 2:
        logger.debug("Prompt-echo detected: %d keyword hits in first %d chars.", hits, _ECHO_SCAN_CHARS)
        return True
    return False


def _is_repeated_block_loop(cleaned_text: str, raw_text: str, stop_reason: str) -> bool:
    """
    True when the model emitted multiple Inverse: blocks instead of stopping.

    Two independent signals, either one is sufficient:
    - Two or more 'Inverse:' headers in cleaned_text.
    - Two or more <|begin_of_text|> BOS tokens in raw_text while also hitting
      max_tokens (the model restarted sequence generation mid-output).
    """
    inverse_count = len(re.findall(r"(?im)^inverse\s*:?[ \t]*$", cleaned_text))
    if inverse_count >= 2:
        logger.debug("Repeated-block loop: %d Inverse: headers found.", inverse_count)
        return True
    bos_count = raw_text.count("<|begin_of_text|>")
    if bos_count >= 2 and stop_reason == "max_tokens":
        logger.debug("Repeated-block loop: %d BOS tokens + max_tokens stop.", bos_count)
        return True
    return False


def _has_any_numbers(parse_meta: dict[str, Any]) -> bool:
    return parse_meta.get("extracted_values_count", 0) > 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_failure(
    raw_text: str,
    cleaned_text: str,
    parse_meta: dict[str, Any],
    scores: dict[str, Any],
    stop_reason: str = "normal",
) -> FailureMode:
    """
    Classify a single prediction into one FailureMode.

    Parameters
    ----------
    raw_text:
        Decoded text **with** special tokens kept (from the raw decode pass).
    cleaned_text:
        Decoded text with special tokens stripped (what the parser sees).
    parse_meta:
        The metadata dict returned by ``parse_to_matrix``, containing at least:
          - parse_success        : bool
          - is_correct_shape     : bool
          - extracted_values_count: int
    scores:
        The scores dict returned by ``score_example``, used to check tolerance
        pass flags (pass_strict, pass_medium, pass_loose).

    Returns
    -------
    FailureMode
    """
    # ── Tier 1: nothing came back ──────────────────────────────────────────
    if _is_eos_only(raw_text, cleaned_text):
        logger.debug("classify_failure → EOS_ONLY (empty cleaned output)")
        return FailureMode.EOS_ONLY

    # ── Tier 2: instruction text echoed back ───────────────────────────────
    if _is_prompt_echo(cleaned_text):
        logger.debug("classify_failure → PROMPT_ECHO")
        return FailureMode.PROMPT_ECHO

    # ── Tier 3: model looped and emitted multiple Inverse: blocks ──────────
    if _is_repeated_block_loop(cleaned_text, raw_text, stop_reason):
        logger.debug("classify_failure → REPEATED_BLOCK_LOOP")
        return FailureMode.REPEATED_BLOCK_LOOP

    # ── Tier 4: shape / format problem ────────────────────────────────────
    correct_shape = parse_meta.get("is_correct_shape", False)
    if not correct_shape:
        logger.debug("classify_failure → SHAPE_FORMAT (extracted=%d)",
                     parse_meta.get("extracted_values_count", 0))
        return FailureMode.SHAPE_FORMAT

    # ── Tier 5: shape OK but numerics fail all tolerances ─────────────────
    passes_any = any([
        scores.get("pass_strict", False),
        scores.get("pass_medium", False),
        scores.get("pass_loose",  False),
    ])
    if not passes_any:
        logger.debug("classify_failure → NUMERICALLY_WRONG")
        return FailureMode.NUMERICALLY_WRONG

    # ── Tier 5: success ────────────────────────────────────────────────────
    logger.debug("classify_failure → SUCCESS")
    return FailureMode.SUCCESS


def summarize_failures(results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Aggregate failure-mode counts and percentages across an eval run.

    Parameters
    ----------
    results:
        List of per-example result dicts. Each must contain a ``"failure_mode"``
        key whose value is a ``FailureMode`` or its string representation.

    Returns
    -------
    dict with keys:
        total_count        : int
        counts             : {mode_name: int, ...}
        percentages        : {mode_name: float (0-1), ...}
    """
    total = len(results)
    counts: dict[str, int] = {mode.value: 0 for mode in FailureMode}

    for r in results:
        mode_raw = r.get("failure_mode", FailureMode.EOS_ONLY)
        # Accept both enum instances and plain strings
        mode_str = mode_raw.value if isinstance(mode_raw, FailureMode) else str(mode_raw)
        if mode_str in counts:
            counts[mode_str] += 1
        else:
            logger.warning("Unknown failure_mode value in results: %r", mode_str)
            counts[mode_str] = counts.get(mode_str, 0) + 1

    percentages = {
        k: (v / total if total > 0 else 0.0)
        for k, v in counts.items()
    }

    summary = {
        "total_count": total,
        "counts": counts,
        "percentages": percentages,
    }

    # Human-readable log
    logger.info("─── Failure Mode Breakdown (n=%d) ───", total)
    for mode in FailureMode:
        n = counts[mode.value]
        pct = percentages[mode.value]
        logger.info("  %-20s %4d  (%5.1f%%)", mode.value, n, pct * 100)

    return summary
