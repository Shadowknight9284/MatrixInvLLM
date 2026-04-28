import yaml
import logging
import os
import numpy as np

def load_config(config_path="config.yaml"):
    """
    Load project configuration from a YAML file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def setup_logging(level_name="INFO"):
    """
    Configure standard Python logging.
    """
    numeric_level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger("MatrixInverse-LLM")

def set_seed(seed):
    """
    Set random seeds for reproducibility.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    # If other seeds (torch etc.) are needed, add them here
