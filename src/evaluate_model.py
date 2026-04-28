import json
import os
import logging
from tqdm import tqdm
import numpy as np
from src.utils import load_config, setup_logging, set_seed
from src.inference import sample_model
from src.score_prediction import score_example

logger = logging.getLogger(__name__)

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
        pred_text = sample_model(prompt, config)
        
        # Scoring
        # ex["matrix"] and ex["inverse"] were saved as lists in Phase 2, convert to numpy
        gold_inverse = np.array(ex["inverse"])
        a_matrix = np.array(ex["matrix"])
        
        scores = score_example(
            pred_text, 
            gold_inverse, 
            a_matrix, 
            ex["matrix_size"], 
            config
        )
        
        # Store result
        result = {
            "id": ex["id"],
            "matrix_size": ex["matrix_size"],
            "prompt": prompt,
            "prediction": pred_text,
            "target": ex["target_text"],
            "scores": scores
        }
        results.append(result)
        
    # Aggregate
    agg_report = aggregate_results(results, config)
    
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
