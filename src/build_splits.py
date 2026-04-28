import json
import random
import os
import logging
import numpy as np
from src.utils import load_config, setup_logging, set_seed
from src.generate_matrices import generate_raw_dataset
from src.serialize_dataset import apply_serialization_to_dataset

logger = logging.getLogger(__name__)

def split_dataset(examples, ratios, seed=123):
    """
    Deterministically shuffle and split a list of examples.
    Ratios: (train, val, test)
    """
    random.seed(seed)
    random.shuffle(examples)
    
    n = len(examples)
    train_end = int(n * ratios[0])
    val_end = train_end + int(n * ratios[1])
    
    train_set = examples[:train_end]
    val_set = examples[train_end:val_end]
    test_set = examples[val_end:]
    
    return train_set, val_set, test_set

def save_jsonl(examples, output_path):
    """
    Save a list of dictionaries as a JSONL file.
    """
    directory = os.path.dirname(output_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        
    with open(output_path, "w") as f:
        for ex in examples:
            # We convert numpy arrays to lists for JSON serialization
            record = {
                "id": ex["id"],
                "matrix_size": ex["matrix_size"],
                "determinant": float(ex["determinant"]),
                "condition_number": float(ex["condition_number"]),
                "input_text": ex["input_text"],
                "target_text": ex["target_text"]
            }
            # Optional: include raw matrix data as lists if needed for later scoring
            if isinstance(ex["matrix"], np.ndarray):
                record["matrix"] = ex["matrix"].tolist()
            if isinstance(ex["inverse"], np.ndarray):
                record["inverse"] = ex["inverse"].tolist()
                
            f.write(json.dumps(record) + "\n")
    logger.info(f"Saved {len(examples)} records to {output_path}")

def run_dataset_pipeline(config):
    """
    Orchestrate full data pipeline: Gen -> Serialize -> Split -> Save
    """
    gen_cfg = config.get("matrix_generation", {})
    ds_cfg = config.get("dataset", {})
    paths_cfg = config.get("paths", {})
    
    # 1. Generate Raw
    sizes = gen_cfg.get("sizes", [3, 4, 5])
    samples_per_size = gen_cfg.get("samples_per_size", 100)
    
    dataset = generate_raw_dataset(sizes, samples_per_size, config)
    
    # 2. Serialize
    dataset = apply_serialization_to_dataset(dataset, config)
    
    # 3. Split
    ratios = (
        ds_cfg.get("train_ratio", 0.8),
        ds_cfg.get("val_ratio", 0.1),
        ds_cfg.get("test_ratio", 0.1)
    )
    split_seed = ds_cfg.get("split_seed", 123)
    
    train_set, val_set, test_set = split_dataset(dataset, ratios, split_seed)
    
    # 4. Save
    save_jsonl(train_set, paths_cfg.get("train_file", "data/train.jsonl"))
    save_jsonl(val_set, paths_cfg.get("val_file", "data/val.jsonl"))
    save_jsonl(test_set, paths_cfg.get("test_file", "data/test.jsonl"))

if __name__ == "__main__":
    config = load_config()
    setup_logging(config.get("logging", {}).get("level", "INFO"))
    set_seed(config.get("project", {}).get("seed", 42))
    
    # For MVP verification, we might want to override sample size if requested
    # But we'll stick to config or a small test run.
    import sys
    if "--test-run" in sys.argv:
        logger.info("Running small test run...")
        config["matrix_generation"]["samples_per_size"] = 10
        
    run_dataset_pipeline(config)
