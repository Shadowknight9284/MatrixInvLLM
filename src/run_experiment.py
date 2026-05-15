import os
import json
import logging
import yaml
from pathlib import Path
import sys

# Add project root to path to ensure imports work if run from root
sys.path.append(os.getcwd())

from src.utils import load_config, setup_logging
from src.evaluate_model import run_evaluation

logger = logging.getLogger(__name__)

def load_file_content(path):
    """Simple helper to read a single value from a file."""
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return f.read().strip()

def compare_metrics(baseline, finetuned):
    """
    Compute deltas and performance improvements between baseline and fine-tuned metrics.
    """
    comparison = {
        "baseline": baseline,
        "finetuned": finetuned,
        "deltas": {}
    }
    
    # Global metrics
    for key in ["parse_success_rate", "is_correct_shape_rate"]:
        if key in baseline and key in finetuned:
            comparison["deltas"][key] = finetuned[key] - baseline[key]
        
    for key in baseline:
        if key.startswith("pass_rate_"):
            if key in finetuned:
                comparison["deltas"][key] = finetuned[key] - baseline[key]
            
    # Per-size metrics
    comparison["deltas"]["by_size"] = {}
    base_by_size = baseline.get("by_size", {})
    ft_by_size = finetuned.get("by_size", {})
    
    for size, b_stats in base_by_size.items():
        if size in ft_by_size:
            f_stats = ft_by_size[size]
            size_deltas = {}
            for k in b_stats:
                if k != "count":
                    size_deltas[k] = f_stats.get(k, 0) - b_stats.get(k, 0)
            comparison["deltas"]["by_size"][size] = size_deltas
            
    return comparison

def main():
    config = load_config()
    setup_logging(config.get("logging", {}).get("level", "INFO"))
    
    paths = config.get("paths", {})
    test_path = paths.get("test_file", "data/test.jsonl")
    outputs_dir = Path(paths.get("outputs_dir", "outputs"))
    
    # Handle limit from CLI for quick verification
    limit = None
    if "--limit" in sys.argv:
        idx = sys.argv.index("--limit")
        try:
            limit = int(sys.argv[idx+1])
            logger.info(f"Running with evaluation limit: {limit}")
        except (IndexError, ValueError):
            logger.warning("Invalid limit provided, running full set.")
        
    # 1. Baseline Evaluation
    logger.info("=== Phase 5.1: Baseline Evaluation ===")
    
    # Prepare baseline config (ensure base_model is used for inference)
    baseline_config = json.loads(json.dumps(config))
    base_model_id = config.get("model", {}).get("base_model")
    
    if "inference" not in baseline_config:
        baseline_config["inference"] = {}
    
    # Force the base model ID for the first run
    baseline_config["inference"]["base_model"] = base_model_id
    
    baseline_metrics = run_evaluation(
        test_path, 
        os.path.join(outputs_dir, "baseline"), 
        baseline_config, 
        limit=limit
    )
    
    # 2. Load Fine-tuned Model ID
    ft_id_path = outputs_dir / "finetuned_model_id.txt"
    ft_id = load_file_content(ft_id_path)
    
    if not ft_id:
        logger.error(f"Fine-tuned model ID not found at {ft_id_path}. Run Phase 4 (train_lora.py) first.")
        return
        
    logger.info(f"Detected fine-tuned model ID: {ft_id}")
    
    # 3. Post-train Evaluation
    logger.info("=== Phase 5.2: Post-train Evaluation ===")
    
    ft_config = json.loads(json.dumps(config))
    if "inference" not in ft_config:
        ft_config["inference"] = {}
        
    # Force the fine-tuned model ID for the second run
    ft_config["inference"]["base_model"] = ft_id
    
    ft_metrics = run_evaluation(
        test_path, 
        os.path.join(outputs_dir, "finetuned"), 
        ft_config, 
        limit=limit
    )
    
    # 4. Compare and Aggregate Results
    if baseline_metrics and ft_metrics:
        logger.info("=== Phase 5.3: Comparative Analysis ===")
        summary = compare_metrics(baseline_metrics, ft_metrics)
        
        # Save consolidated summary
        metrics_dir = Path(paths.get("metrics_dir", "outputs/metrics"))
        metrics_dir.mkdir(parents=True, exist_ok=True)
        
        summary_path = metrics_dir / "experiment_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"Experiment summary successfully generated at {summary_path}")
        
        # Log Summary Table Highlights
        W = 62
        print("\n" + "=" * W)
        print("              EXPERIMENT SUMMARY HIGHLIGHTS")
        print("=" * W)
        print(f"{'Metric':<30} | {'Baseline':>9} | {'Finetuned':>9} | {'Delta':>9}")
        print("-" * W)

        pct_metrics = [
            "parse_success_rate",
            "is_correct_shape_rate",
            "pass_rate_strict",
            "pass_rate_medium",
            "pass_rate_loose",
        ]
        for m in pct_metrics:
            b_val = baseline_metrics.get(m, 0) or 0
            f_val = ft_metrics.get(m, 0) or 0
            d_val = f_val - b_val
            print(f"{m:<30} | {b_val:>9.2%} | {f_val:>9.2%} | {d_val:>+9.2%}")

        print("-" * W)

        def _fmt(v, width=9, decimals=4):
            return f"{v:{width}.{decimals}f}" if v is not None else f"{'N/A':>{width}}"

        def _fmt_delta(b, f, width=9, decimals=4):
            return f"{f - b:>+{width}.{decimals}f}" if (b is not None and f is not None) else f"{'N/A':>{width}}"

        float_metrics = [
            ("avg_frobenius_identity_left",  "||AX-I||_F (avg)"),
            ("avg_frobenius_identity_right", "||XA-I||_F (avg)"),
            ("avg_frobenius_identity_left",  ""),   # spacer handled below
        ]

        # Identity residuals
        for key, label in [
            ("avg_frobenius_identity_left",  "||AX-I||_F (avg)"),
            ("avg_frobenius_identity_right", "||XA-I||_F (avg)"),
        ]:
            b_val = baseline_metrics.get(key)
            f_val = ft_metrics.get(key)
            print(f"{label:<30} | {_fmt(b_val)} | {_fmt(f_val)} | {_fmt_delta(b_val, f_val)}")

        print("-" * W)

        # Extended metrics
        ext_metrics = [
            ("avg_relative_frob_error", "Relative ||X-A^-1||_F (avg)"),
            ("avg_det_ratio",           "det ratio (avg, 1.0=perfect)"),
            ("avg_trace_abs_error",     "Trace abs error (avg)"),
            ("avg_sign_agreement",      "Sign agreement (avg)"),
            ("avg_spectral_error",      "Spectral error (avg)"),
        ]
        for key, label in ext_metrics:
            b_val = baseline_metrics.get(key)
            f_val = ft_metrics.get(key)
            print(f"{label:<30} | {_fmt(b_val)} | {_fmt(f_val)} | {_fmt_delta(b_val, f_val)}")

        print("-" * W)

        # Copy-detection metrics (lower = more copying)
        copy_metrics = [
            ("avg_frob_vs_input",     "||X-A||_F (copy detect)"),
            ("avg_frob_vs_transpose", "||X-A^T||_F (transpose detect)"),
            ("avg_frob_vs_neg_input", "||X-(-A)||_F (neg copy detect)"),
        ]
        for key, label in copy_metrics:
            b_val = baseline_metrics.get(key)
            f_val = ft_metrics.get(key)
            print(f"{label:<30} | {_fmt(b_val)} | {_fmt(f_val)} | {_fmt_delta(b_val, f_val)}")

        print("=" * W + "\n")
        
    else:
        logger.error("Evaluation failed to produce complete metrics for comparison.")

if __name__ == "__main__":
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    main()
