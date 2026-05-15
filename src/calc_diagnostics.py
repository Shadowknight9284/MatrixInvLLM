"""
calc_diagnostics.py
-------------------
Standalone diagnostic calculator.  Reads one or two predictions.jsonl files,
recomputes all extended metrics from the stored matrices, and prints a report.

Usage
-----
  # Single run
  python -m src.calc_diagnostics outputs/baseline

  # Side-by-side comparison
  python -m src.calc_diagnostics outputs/baseline outputs/finetuned

  # Named args (same behaviour)
  python -m src.calc_diagnostics --baseline outputs/baseline --finetuned outputs/finetuned
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Metric computation (no dependency on other src modules)
# ---------------------------------------------------------------------------

def _compute_extended(pred: list | None, gold: list, a: list) -> dict[str, Any]:
    """Compute all extended metrics from nested-list matrices."""
    none_fields: dict[str, Any] = {
        "det_input": None, "det_expected": None, "det_observed": None,
        "det_ratio": None, "trace_expected": None, "trace_observed": None,
        "trace_abs_error": None, "relative_frob_error": None,
        "frob_vs_input": None, "frob_vs_neg_input": None,
        "frob_vs_transpose": None, "sign_agreement": None,
        "spectral_error": None,
    }
    try:
        A = np.array(a, dtype=float)
        G = np.array(gold, dtype=float)
        det_a = float(np.linalg.det(A))
        none_fields["det_input"] = det_a
        if abs(det_a) > 1e-12:
            none_fields["det_expected"] = 1.0 / det_a
        none_fields["trace_expected"] = float(np.trace(G))
        gold_frob = float(np.linalg.norm(G, ord="fro"))
    except Exception:
        return none_fields

    if pred is None:
        return none_fields

    try:
        X = np.array(pred, dtype=float)
        det_x = float(np.linalg.det(X))
        none_fields["det_observed"] = det_x
        if none_fields["det_expected"] and abs(none_fields["det_expected"]) > 1e-12:
            none_fields["det_ratio"] = det_x / none_fields["det_expected"]
        none_fields["trace_observed"] = float(np.trace(X))
        none_fields["trace_abs_error"] = abs(none_fields["trace_observed"] - none_fields["trace_expected"])
        err = X - G
        if gold_frob > 1e-12:
            none_fields["relative_frob_error"] = float(np.linalg.norm(err, "fro") / gold_frob)
        none_fields["frob_vs_input"]     = float(np.linalg.norm(X - A,    "fro"))
        none_fields["frob_vs_neg_input"] = float(np.linalg.norm(X + A,    "fro"))
        none_fields["frob_vs_transpose"] = float(np.linalg.norm(X - A.T,  "fro"))
        none_fields["sign_agreement"]    = float(np.mean(np.sign(X) == np.sign(G)))
        none_fields["spectral_error"]    = float(np.linalg.norm(err, ord=2))
    except Exception:
        pass
    return none_fields


def _identity_residuals(pred: list | None, a: list) -> dict[str, float | None]:
    if pred is None:
        return {"frob_identity_left": None, "frob_identity_right": None}
    try:
        X = np.array(pred, dtype=float)
        A = np.array(a, dtype=float)
        n = A.shape[0]
        I = np.eye(n)
        return {
            "frob_identity_left":  float(np.linalg.norm(A @ X - I, "fro")),
            "frob_identity_right": float(np.linalg.norm(X @ A - I, "fro")),
        }
    except Exception:
        return {"frob_identity_left": None, "frob_identity_right": None}


# ---------------------------------------------------------------------------
# Load & enrich predictions
# ---------------------------------------------------------------------------

def load_predictions(path: str) -> list[dict]:
    jsonl = os.path.join(path, "predictions.jsonl")
    if not os.path.exists(jsonl):
        print(f"ERROR: {jsonl} not found.", file=sys.stderr)
        sys.exit(1)
    records = []
    with open(jsonl, encoding="utf-8") as fh:
        for line in fh:
            records.append(json.loads(line))
    return records


def enrich(records: list[dict]) -> list[dict]:
    """Add / refresh extended_diagnostics from stored matrix data."""
    for r in records:
        mat = r.get("matrix_diagnostics", {})
        pred  = mat.get("predicted_matrix")
        gold  = mat.get("target_matrix")
        a_mat = mat.get("prompt_matrix")
        if gold is None or a_mat is None:
            r["_ext"] = {}
            r["_idr"] = {}
            continue
        ext = r.get("extended_diagnostics") or _compute_extended(pred, gold, a_mat)
        idr = _identity_residuals(pred, a_mat)
        # Prefer freshly-stored values from matrix_diagnostics for identity residuals
        idr["frob_identity_left"]  = mat.get("frobenius_identity_left")  or idr["frob_identity_left"]
        idr["frob_identity_right"] = mat.get("frobenius_identity_right") or idr["frob_identity_right"]
        r["_ext"] = ext
        r["_idr"] = idr
    return records


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _avg(vals: list) -> float | None:
    v = [x for x in vals if x is not None]
    return sum(v) / len(v) if v else None


def aggregate(records: list[dict]) -> dict[str, Any]:
    total = len(records)
    parsed = [r for r in records if r.get("scores", {}).get("is_correct_shape")]

    def ep(key):   # extended, parsed only
        return [r["_ext"].get(key) for r in parsed]
    def ip(key):   # identity residuals, parsed only
        return [r["_idr"].get(key) for r in parsed]

    from collections import Counter
    modes = Counter(r.get("failure_mode", "UNKNOWN") for r in records)

    return {
        "total":            total,
        "parsed":           len(parsed),
        "parse_rate":       len(parsed) / total if total else 0,
        "failure_modes":    dict(modes),
        # identity residuals
        "avg_frob_id_left":  _avg(ip("frob_identity_left")),
        "avg_frob_id_right": _avg(ip("frob_identity_right")),
        # extended
        "avg_relative_frob": _avg(ep("relative_frob_error")),
        "avg_det_ratio":     _avg(ep("det_ratio")),
        "avg_trace_err":     _avg(ep("trace_abs_error")),
        "avg_sign_agree":    _avg(ep("sign_agreement")),
        "avg_spectral_err":  _avg(ep("spectral_error")),
        "avg_frob_vs_A":     _avg(ep("frob_vs_input")),
        "avg_frob_vs_AT":    _avg(ep("frob_vs_transpose")),
        "avg_frob_vs_negA":  _avg(ep("frob_vs_neg_input")),
    }


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

W = 70

def _f(v, w=9, d=4):
    return f"{v:{w}.{d}f}" if v is not None else f"{'N/A':>{w}}"

def _fd(b, f, w=9, d=4):
    return f"{f-b:>+{w}.{d}f}" if (b is not None and f is not None) else f"{'N/A':>{w}}"


def print_single(label: str, records: list[dict], agg: dict) -> None:
    print(f"\n{'='*W}")
    print(f"  DIAGNOSTICS: {label}")
    print(f"{'='*W}")
    print(f"  Total examples : {agg['total']}")
    print(f"  Parsed / shaped: {agg['parsed']} / {agg['total']}  ({agg['parse_rate']:.1%})")
    print()
    print("  Failure modes:")
    for mode, count in sorted(agg["failure_modes"].items(), key=lambda x: -x[1]):
        bar = "#" * count
        print(f"    {mode:<25} {count:>4}  {bar}")

    print()
    print(f"  {'id':<10} {'sz':>3}  {'mode':<22}  {'frob_id_L':>9}  {'frob_id_R':>9}  "
          f"{'rel_frob':>8}  {'det_ratio':>9}  {'sign_agr':>8}")
    print(f"  {'-'*10} {'-'*3}  {'-'*22}  {'-'*9}  {'-'*9}  {'-'*8}  {'-'*9}  {'-'*8}")
    for r in records:
        ext = r.get("_ext", {})
        idr = r.get("_idr", {})
        print(
            f"  {r.get('id','?'):<10} {r.get('matrix_size',0):>3}  "
            f"{r.get('failure_mode','?'):<22}  "
            f"{_f(idr.get('frob_identity_left'))}  "
            f"{_f(idr.get('frob_identity_right'))}  "
            f"{_f(ext.get('relative_frob_error'),8,4)}  "
            f"{_f(ext.get('det_ratio'))}  "
            f"{_f(ext.get('sign_agreement'),8,4)}"
        )

    print()
    print(f"  {'-'*W}")
    rows = [
        ("||AX-I||_F (avg)",            agg["avg_frob_id_left"]),
        ("||XA-I||_F (avg)",            agg["avg_frob_id_right"]),
        ("Relative ||X-A^-1||_F (avg)", agg["avg_relative_frob"]),
        ("Det ratio (avg, 1.0=perfect)",agg["avg_det_ratio"]),
        ("Trace abs error (avg)",       agg["avg_trace_err"]),
        ("Sign agreement (avg)",        agg["avg_sign_agree"]),
        ("Spectral error (avg)",        agg["avg_spectral_err"]),
        ("||X-A||_F  (input copy?)",    agg["avg_frob_vs_A"]),
        ("||X-A^T||_F (transpose?)",    agg["avg_frob_vs_AT"]),
        ("||X-(-A)||_F (neg copy?)",    agg["avg_frob_vs_negA"]),
    ]
    for lbl, val in rows:
        print(f"  {lbl:<35} {_f(val)}")
    print(f"{'='*W}\n")


def print_comparison(lbl_b: str, agg_b: dict, lbl_f: str, agg_f: dict) -> None:
    print(f"\n{'='*W}")
    print("  COMPARISON")
    print(f"{'='*W}")
    print(f"  {'Metric':<35} {'Baseline':>9}  {'Finetuned':>9}  {'Delta':>9}")
    print(f"  {'-'*35} {'-'*9}  {'-'*9}  {'-'*9}")

    pct_rows = [
        ("Parse / shape rate", "parse_rate"),
    ]
    for lbl, key in pct_rows:
        b, f = agg_b.get(key), agg_f.get(key)
        bs = f"{b:>9.2%}" if b is not None else f"{'N/A':>9}"
        fs = f"{f:>9.2%}" if f is not None else f"{'N/A':>9}"
        ds = f"{f-b:>+9.2%}" if (b is not None and f is not None) else f"{'N/A':>9}"
        print(f"  {lbl:<35} {bs}  {fs}  {ds}")

    print(f"  {'-'*W}")

    float_rows = [
        ("||AX-I||_F (avg)",             "avg_frob_id_left"),
        ("||XA-I||_F (avg)",             "avg_frob_id_right"),
        ("Relative ||X-A^-1||_F (avg)",  "avg_relative_frob"),
        ("Det ratio (avg, 1.0=perfect)", "avg_det_ratio"),
        ("Trace abs error (avg)",        "avg_trace_err"),
        ("Sign agreement (avg)",         "avg_sign_agree"),
        ("Spectral error (avg)",         "avg_spectral_err"),
        ("||X-A||_F  (input copy?)",     "avg_frob_vs_A"),
        ("||X-A^T||_F (transpose?)",     "avg_frob_vs_AT"),
        ("||X-(-A)||_F (neg copy?)",     "avg_frob_vs_negA"),
    ]
    for lbl, key in float_rows:
        b, f = agg_b.get(key), agg_f.get(key)
        print(f"  {lbl:<35} {_f(b)}  {_f(f)}  {_fd(b, f)}")

    print(f"{'='*W}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Compute diagnostics from predictions.jsonl")
    parser.add_argument("dirs", nargs="*",
                        help="One or two output directories (positional)")
    parser.add_argument("--baseline",  default=None)
    parser.add_argument("--finetuned", default=None)
    args = parser.parse_args()

    positional = args.dirs or []
    baseline_dir  = args.baseline  or (positional[0] if len(positional) > 0 else None)
    finetuned_dir = args.finetuned or (positional[1] if len(positional) > 1 else None)

    if baseline_dir is None:
        parser.print_help()
        sys.exit(0)

    recs_b = enrich(load_predictions(baseline_dir))
    agg_b  = aggregate(recs_b)
    print_single(baseline_dir, recs_b, agg_b)

    if finetuned_dir:
        recs_f = enrich(load_predictions(finetuned_dir))
        agg_f  = aggregate(recs_f)
        print_single(finetuned_dir, recs_f, agg_f)
        print_comparison(baseline_dir, agg_b, finetuned_dir, agg_f)


if __name__ == "__main__":
    main()
