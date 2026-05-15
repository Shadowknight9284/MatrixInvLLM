"""
eval_diagnostics.py
-------------------
Pure helper functions for building rich per-example diagnostic fields in the
evaluation loop.  All outputs are JSON-serialisable (nested Python lists /
floats / strings / None).  No side-effects; no imports from other src modules.
"""
from __future__ import annotations

import csv
import logging
import os
import re
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_HEADER_RE = re.compile(r"(?im)^inverse\s*:?[ \t]*$")


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def to_nested_list(arr: np.ndarray | None) -> list[list[float]] | None:
    """Convert a 2-D numpy array to a nested list of plain Python floats."""
    if arr is None:
        return None
    return [[float(v) for v in row] for row in arr]


# ---------------------------------------------------------------------------
# Text diagnostics
# ---------------------------------------------------------------------------

def extract_first_block_text(text: str) -> str | None:
    """
    Return the substring from the first 'Inverse:' header through the last
    line of its numeric block.  Stops at a second 'Inverse:' header.
    Returns None when no header is found.
    """
    if not text:
        return None
    m = _HEADER_RE.search(text)
    if not m:
        return None
    after = text[m.end():]
    m2 = _HEADER_RE.search(after)
    block_body = after[: m2.start()] if m2 else after
    result = ("Inverse:\n" + block_body).rstrip()
    return result if result.strip() else None


def build_generation_fields(diag: dict[str, Any], pred_text: str) -> dict[str, Any]:
    """
    Return the three text-level diagnostic fields.

    `first_block_text` is extracted from cleaned_text (what the parser saw),
    not from raw_text, so it reflects the post-truncation signal.
    """
    cleaned = diag.get("cleaned_text", pred_text)
    return {
        "raw_text": diag.get("raw_text", ""),
        "cleaned_text": cleaned,
        "first_block_text": extract_first_block_text(cleaned),
    }


# ---------------------------------------------------------------------------
# Matrix diagnostics
# ---------------------------------------------------------------------------

def build_matrix_fields(
    pred_matrix: np.ndarray | None,
    gold_matrix: np.ndarray,
    a_matrix: np.ndarray,
) -> dict[str, Any]:
    """
    Compute structured matrix representations and extra math diagnostics.

    Always returns all keys; matrix-derived keys are None when pred_matrix
    is None (parse failure).

    Keys
    ----
    prompt_matrix               nested list — the input matrix A
    target_matrix               nested list — the gold inverse A^{-1}
    predicted_matrix            nested list | None
    error_matrix                predicted - target, element-wise | None

    product_left                A @ predicted | None
    product_right               predicted @ A | None
    identity_error_left         A @ predicted - I | None
    identity_error_right        predicted @ A - I | None

    frobenius_dist_target       ||predicted - target||_F   | None
    frobenius_identity_left     ||A @ predicted - I||_F    | None
    frobenius_identity_right    ||predicted @ A - I||_F    | None
    """
    n = gold_matrix.shape[0]
    identity = np.eye(n)

    base: dict[str, Any] = {
        "prompt_matrix": to_nested_list(a_matrix),
        "target_matrix": to_nested_list(gold_matrix),
        "predicted_matrix": None,
        "error_matrix": None,
        "product_left": None,
        "product_right": None,
        "identity_error_left": None,
        "identity_error_right": None,
        "frobenius_dist_target": None,
        "frobenius_identity_left": None,
        "frobenius_identity_right": None,
    }

    if pred_matrix is None:
        return base

    try:
        error = pred_matrix - gold_matrix
        prod_left = a_matrix @ pred_matrix
        prod_right = pred_matrix @ a_matrix
        id_err_left = prod_left - identity
        id_err_right = prod_right - identity

        base.update({
            "predicted_matrix": to_nested_list(pred_matrix),
            "error_matrix": to_nested_list(error),
            "product_left": to_nested_list(prod_left),
            "product_right": to_nested_list(prod_right),
            "identity_error_left": to_nested_list(id_err_left),
            "identity_error_right": to_nested_list(id_err_right),
            "frobenius_dist_target": float(np.linalg.norm(error, ord="fro")),
            "frobenius_identity_left": float(np.linalg.norm(id_err_left, ord="fro")),
            "frobenius_identity_right": float(np.linalg.norm(id_err_right, ord="fro")),
        })
    except Exception as exc:
        logger.warning("build_matrix_fields: computation error — %s", exc)

    return base


# ---------------------------------------------------------------------------
# Extended per-example diagnostics
# ---------------------------------------------------------------------------

def build_extended_matrix_fields(
    pred_matrix: np.ndarray | None,
    gold_matrix: np.ndarray,
    a_matrix: np.ndarray,
) -> dict[str, Any]:
    """
    Richer per-example diagnostics useful for a written report.

    All values are plain Python floats or None (JSON-serialisable).

    Keys
    ----
    det_input           det(A)
    det_expected        1/det(A)  — what det(predicted) should equal
    det_observed        det(predicted)
    det_ratio           det_observed / det_expected  (1.0 = perfect scale)

    trace_expected      tr(A^{-1})  = tr(gold)
    trace_observed      tr(predicted)
    trace_abs_error     |trace_observed - trace_expected|

    relative_frob_error ||predicted - gold||_F / ||gold||_F

    frob_vs_input       ||predicted - A||_F     — detects input-copy failure
    frob_vs_neg_input   ||predicted - (-A)||_F  — detects negation copy
    frob_vs_transpose   ||predicted - A^T||_F   — detects transpose copy

    sign_agreement      fraction of elements where sign(predicted) == sign(gold)
    spectral_error      largest singular value of (predicted - gold)
    """
    base: dict[str, Any] = {
        "det_input":           None,
        "det_expected":        None,
        "det_observed":        None,
        "det_ratio":           None,
        "trace_expected":      None,
        "trace_observed":      None,
        "trace_abs_error":     None,
        "relative_frob_error": None,
        "frob_vs_input":       None,
        "frob_vs_neg_input":   None,
        "frob_vs_transpose":   None,
        "sign_agreement":      None,
        "spectral_error":      None,
    }

    try:
        det_a = float(np.linalg.det(a_matrix))
        base["det_input"] = det_a
        if abs(det_a) > 1e-12:
            base["det_expected"] = float(1.0 / det_a)

        base["trace_expected"] = float(np.trace(gold_matrix))

        gold_frob = float(np.linalg.norm(gold_matrix, ord="fro"))
    except Exception as exc:
        logger.warning("build_extended_matrix_fields: input computation error — %s", exc)
        return base

    if pred_matrix is None:
        return base

    try:
        det_pred = float(np.linalg.det(pred_matrix))
        base["det_observed"] = det_pred
        if base["det_expected"] is not None and abs(base["det_expected"]) > 1e-12:
            base["det_ratio"] = float(det_pred / base["det_expected"])

        base["trace_observed"] = float(np.trace(pred_matrix))
        base["trace_abs_error"] = float(abs(base["trace_observed"] - base["trace_expected"]))

        err = pred_matrix - gold_matrix
        if gold_frob > 1e-12:
            base["relative_frob_error"] = float(np.linalg.norm(err, ord="fro") / gold_frob)

        base["frob_vs_input"]     = float(np.linalg.norm(pred_matrix - a_matrix,    ord="fro"))
        base["frob_vs_neg_input"] = float(np.linalg.norm(pred_matrix + a_matrix,    ord="fro"))
        base["frob_vs_transpose"] = float(np.linalg.norm(pred_matrix - a_matrix.T,  ord="fro"))

        total = pred_matrix.size
        base["sign_agreement"] = float(
            np.sum(np.sign(pred_matrix) == np.sign(gold_matrix)) / total
        )

        base["spectral_error"] = float(np.linalg.norm(err, ord=2))

    except Exception as exc:
        logger.warning("build_extended_matrix_fields: computation error — %s", exc)

    return base


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

_CSV_COLUMNS = [
    "id",
    "matrix_size",
    "parse_success",
    "failure_mode",
    "mae",
    "mse",
    "frobenius_dist_target",
    "frobenius_identity_left",
    "frobenius_identity_right",
    "residual_max",
    "relative_frob_error",
    "det_ratio",
    "trace_abs_error",
    "frob_vs_input",
    "frob_vs_neg_input",
    "frob_vs_transpose",
    "sign_agreement",
    "spectral_error",
    "token_count",
    "stop_reason",
    "adapter_status",
]


def save_diagnostics_csv(results: list[dict[str, Any]], output_dir: str) -> None:
    """
    Write diagnostics.csv — one row per example, scalar columns only.

    Pulls from four sub-dicts that evaluate_model.py attaches to each result:
      result["scores"]               — parse / numeric metrics
      result["matrix_diagnostics"]   — frobenius_dist_target, frobenius_identity_*
      result["extended_diagnostics"] — det_ratio, trace_abs_error, frob_vs_*, etc.
      result["inference_diagnostics"]— token_count, stop_reason, adapter_status
    """
    path = os.path.join(output_dir, "diagnostics.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            scores = r.get("scores", {})
            mat    = r.get("matrix_diagnostics", {})
            ext    = r.get("extended_diagnostics", {})
            inf    = r.get("inference_diagnostics", {})
            writer.writerow({
                "id":                      r.get("id"),
                "matrix_size":             r.get("matrix_size"),
                "parse_success":           scores.get("parse_success"),
                "failure_mode":            r.get("failure_mode"),
                "mae":                     scores.get("mae"),
                "mse":                     scores.get("mse"),
                "frobenius_dist_target":   mat.get("frobenius_dist_target"),
                "frobenius_identity_left": mat.get("frobenius_identity_left"),
                "frobenius_identity_right":mat.get("frobenius_identity_right"),
                "residual_max":            scores.get("residual_max"),
                "relative_frob_error":     ext.get("relative_frob_error"),
                "det_ratio":               ext.get("det_ratio"),
                "trace_abs_error":         ext.get("trace_abs_error"),
                "frob_vs_input":           ext.get("frob_vs_input"),
                "frob_vs_neg_input":       ext.get("frob_vs_neg_input"),
                "frob_vs_transpose":       ext.get("frob_vs_transpose"),
                "sign_agreement":          ext.get("sign_agreement"),
                "spectral_error":          ext.get("spectral_error"),
                "token_count":             inf.get("token_count"),
                "stop_reason":             inf.get("stop_reason"),
                "adapter_status":          inf.get("adapter_status"),
            })
    logger.info("Diagnostics CSV saved to %s", path)
