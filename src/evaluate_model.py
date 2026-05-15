import json
import os
import re
import logging
from tqdm import tqdm
import numpy as np
from src.utils import load_config, setup_logging, set_seed
from src.inference import sample_model_with_diagnostics
from src.score_prediction import score_example
from src.failure_classifier import classify_failure, summarize_failures
from src.parse_prediction import parse_to_matrix
from src.eval_diagnostics import (
    build_generation_fields,
    build_matrix_fields,
    build_extended_matrix_fields,
    save_diagnostics_csv,
)

logger = logging.getLogger(__name__)

_INVERSE_HEADER_RE = re.compile(r'(?i)inverse\s*:')


def run_evaluation(dataset_path, output_dir, config, limit=None):
    """
    Run evaluation loop over a JSONL dataset.
    """
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found: {dataset_path}")
        return
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    records = []
    with open(dataset_path, "r") as f:
        for line in f:
            records.append(json.loads(line))
            
    if limit:
        records = records[:limit]
        
    logger.info(f"Starting evaluation on {len(records)} examples from {dataset_path}...")
    
    results = []
    
    for i, ex in enumerate(tqdm(records, desc="Evaluating")):
        # Inference
        prompt = ex["input_text"]
        pred_text, diag = sample_model_with_diagnostics(prompt, config)

        # The prompt now ends with "Inverse:\n", so a base model may return just
        # numeric rows without the header.  Prepend it for the parser only;
        # pred_text (the raw model output) is kept as-is for logging.
        pred_for_parsing = (
            "Inverse:\n" + pred_text
            if pred_text.strip() and not _INVERSE_HEADER_RE.search(pred_text)
            else pred_text
        )

        # Scoring
        # ex["matrix"] and ex["inverse"] were saved as lists in Phase 2, convert to numpy
        gold_inverse = np.array(ex["inverse"])
        a_matrix = np.array(ex["matrix"])

        scores = score_example(
            pred_for_parsing,
            gold_inverse,
            a_matrix,
            ex["matrix_size"],
            config
        )

        failure_mode = classify_failure(
            raw_text=diag["raw_text"],
            cleaned_text=pred_for_parsing,
            parse_meta=scores,
            scores=scores,
            stop_reason=diag.get("stop_reason", "normal"),
        )

        # Parse predicted matrix for matrix-level diagnostics.
        # parse_to_matrix is lightweight (no model call); double-parsing is fine.
        pred_matrix, _ = parse_to_matrix(pred_for_parsing, ex["matrix_size"])

        gen_fields = build_generation_fields(diag, pred_text)
        mat_fields = build_matrix_fields(pred_matrix, gold_inverse, a_matrix)
        ext_fields = build_extended_matrix_fields(pred_matrix, gold_inverse, a_matrix)

        # Store result
        result = {
            "id": ex["id"],
            "matrix_size": ex["matrix_size"],
            "prompt": prompt,
            "prediction": pred_text,
            "target": ex["target_text"],
            "scores": scores,
            "failure_mode": failure_mode.value if hasattr(failure_mode, 'value') else str(failure_mode),
            "matrix_diagnostics": mat_fields,
            "extended_diagnostics": ext_fields,
        }
        if config.get("diagnostics", {}).get("save_inference_diagnostics", True):
            # Extend the existing inference_diagnostics with first_block_text;
            # raw_text and cleaned_text are already present from sample_model_with_diagnostics.
            result["inference_diagnostics"] = {**diag, "first_block_text": gen_fields["first_block_text"]}
            
        results.append(result)
        
    # Aggregate
    agg_report = aggregate_results(results, config)
    agg_report["failure_mode_breakdown"] = summarize_failures(results)
    
    # Save
    save_evaluation_results(results, agg_report, output_dir)
    
    return agg_report

def aggregate_results(results, config):
    """
    Compute aggregate metrics across the evaluation run.
    """
    eval_cfg = config.get("evaluation", {})
    tols = eval_cfg.get("tolerance_bands", ["strict", "medium", "loose"])
    
    stats = {
        "total_count": len(results),
        "parse_success_rate": 0,
        "is_correct_shape_rate": 0,
        "by_size": {}
    }
    
    if not results:
        return stats
        
    parse_successes = [r["scores"]["parse_success"] for r in results]
    shape_successes = [r["scores"]["is_correct_shape"] for r in results]
    
    stats["parse_success_rate"] = sum(parse_successes) / len(results)
    stats["is_correct_shape_rate"] = sum(shape_successes) / len(results)
    
    # Pass rates for strictly correct shapes
    valid_results = [r for r in results if r["scores"]["is_correct_shape"]]

    for level in tols:
        if valid_results:
            pass_rate = sum([r["scores"][f"pass_{level}"] for r in valid_results]) / len(results)
        else:
            pass_rate = 0.0
        stats[f"pass_rate_{level}"] = pass_rate

    # Average Frobenius identity errors (||AX - I||_F and ||XA - I||_F)
    # Only over examples that were successfully parsed (have a predicted matrix).
    def _frob_vals(key):
        return [
            r["matrix_diagnostics"][key]
            for r in valid_results
            if r.get("matrix_diagnostics", {}).get(key) is not None
        ]

    frob_left = _frob_vals("frobenius_identity_left")
    frob_right = _frob_vals("frobenius_identity_right")
    stats["avg_frobenius_identity_left"] = sum(frob_left) / len(frob_left) if frob_left else None
    stats["avg_frobenius_identity_right"] = sum(frob_right) / len(frob_right) if frob_right else None

    # Extended metric averages (over parsed examples only)
    def _ext_vals(key):
        return [
            r["extended_diagnostics"][key]
            for r in valid_results
            if r.get("extended_diagnostics", {}).get(key) is not None
        ]

    def _avg(vals):
        return sum(vals) / len(vals) if vals else None

    stats["avg_relative_frob_error"] = _avg(_ext_vals("relative_frob_error"))
    stats["avg_det_ratio"]           = _avg(_ext_vals("det_ratio"))
    stats["avg_trace_abs_error"]     = _avg(_ext_vals("trace_abs_error"))
    stats["avg_frob_vs_input"]       = _avg(_ext_vals("frob_vs_input"))
    stats["avg_frob_vs_neg_input"]   = _avg(_ext_vals("frob_vs_neg_input"))
    stats["avg_frob_vs_transpose"]   = _avg(_ext_vals("frob_vs_transpose"))
    stats["avg_sign_agreement"]      = _avg(_ext_vals("sign_agreement"))
    stats["avg_spectral_error"]      = _avg(_ext_vals("spectral_error"))

    # Breakdown by size
    sizes = sorted(list(set([r["matrix_size"] for r in results])))
    for size in sizes:
        size_results = [r for r in results if r["matrix_size"] == size]
        valid_size_results = [r for r in size_results if r["scores"]["is_correct_shape"]]
        
        size_stats = {
            "count": len(size_results),
            "parse_success_rate": sum([r["scores"]["parse_success"] for r in size_results]) / len(size_results)
        }
        
        for level in tols:
            if valid_size_results:
                pr = sum([r["scores"][f"pass_{level}"] for r in valid_size_results]) / len(size_results)
            else:
                pr = 0.0
            size_stats[f"pass_rate_{level}"] = pr
            
        stats["by_size"][str(size)] = size_stats
        
    return stats

def save_evaluation_results(results, agg_report, output_dir):
    """
    Save detailed JSONL predictions and aggregate JSON summary.
    """
    # Detailed JSONL
    det_path = os.path.join(output_dir, "predictions.jsonl")
    with open(det_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
            
    # Aggregate JSON
    agg_path = os.path.join(output_dir, "metrics.json")
    with open(agg_path, "w") as f:
        json.dump(agg_report, f, indent=2)
        
    save_diagnostics_csv(results, output_dir)

    logger.info(f"Detailed results saved to {det_path}")
    logger.info(f"Aggregate report saved to {agg_path}")

if __name__ == "__main__":
    config = load_config()
    setup_logging(config.get("logging", {}).get("level", "INFO"))
    set_seed(config.get("project", {}).get("seed", 42))
    
    paths = config.get("paths", {})
    test_path = paths.get("test_file", "data/test.jsonl")
    output_dir = "outputs/baseline"
    
    import sys
    limit = None
    if "--limit" in sys.argv:
        idx = sys.argv.index("--limit")
        limit = int(sys.argv[idx+1])
        
    run_evaluation(test_path, output_dir, config, limit=limit)
