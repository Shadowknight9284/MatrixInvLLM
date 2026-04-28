import numpy as np
import logging
from src.utils import load_config, setup_logging, set_seed

logger = logging.getLogger(__name__)

def generate_random_matrix(n, low=-1.0, high=1.0, distribution="uniform"):
    """
    Sample a dense square matrix from a given distribution.
    """
    if distribution == "uniform":
        return np.random.uniform(low, high, size=(n, n))
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")

def is_stable(matrix, determinant_abs_min=0.05, condition_number_max=50.0):
    """
    Check if a matrix is numerically stable and non-singular.
    """
    # Rank check
    rank = np.linalg.matrix_rank(matrix)
    if rank < matrix.shape[0]:
        logger.debug(f"Matrix is singular (rank {rank} < {matrix.shape[0]})")
        return False, "singular"

    # Determinant check
    det = np.linalg.det(matrix)
    if abs(det) < determinant_abs_min:
        logger.debug(f"Matrix determinant is too small: {det}")
        return False, "small_det"

    # Condition number check
    cond = np.linalg.cond(matrix)
    if cond > condition_number_max:
        logger.debug(f"Matrix condition number is too high: {cond}")
        return False, "high_cond"

    return True, None

def compute_inverse(matrix):
    """
    Compute the inverse of a matrix using NumPy.
    """
    return np.linalg.inv(matrix)

def verify_residual(matrix, inverse, tolerance=1e-8):
    """
    Verify the functional correctness of an inverse using the residual norm ||AX - I||.
    """
    n = matrix.shape[0]
    identity = np.eye(n)
    residual = np.matmul(matrix, inverse) - identity
    residual_norm = np.linalg.norm(residual, ord='fro')  # Frobenius norm
    
    is_valid = residual_norm < tolerance
    return is_valid, residual_norm

def generate_stable_example(n, config):
    """
    Generate a single stable matrix and its validated inverse.
    """
    gen_cfg = config.get("matrix_generation", {})
    low = gen_cfg.get("uniform_low", -1.0)
    high = gen_cfg.get("uniform_high", 1.0)
    det_min = gen_cfg.get("determinant_abs_min", 0.05)
    cond_max = gen_cfg.get("condition_number_max", 50.0)
    res_tol = gen_cfg.get("residual_tolerance_gold", 1e-8)
    max_attempts = gen_cfg.get("max_generation_attempts_per_size", 1000)

    for attempt in range(max_attempts):
        matrix = generate_random_matrix(n, low, high)
        stable, reason = is_stable(matrix, det_min, cond_max)
        
        if stable:
            inverse = compute_inverse(matrix)
            is_valid, res_norm = verify_residual(matrix, inverse, res_tol)
            
            if is_valid:
                return {
                    "matrix": matrix,
                    "inverse": inverse,
                    "determinant": np.linalg.det(matrix),
                    "condition_number": np.linalg.cond(matrix),
                    "residual_norm": res_norm,
                    "attempts": attempt + 1
                }
            else:
                logger.debug(f"Attempt {attempt}: Inverse failed residual check (norm: {res_norm})")
    
    raise RuntimeError(f"Failed to generate stable {n}x{n} matrix after {max_attempts} attempts")

if __name__ == "__main__":
    # Internal Verification for Phase 1
    config = load_config()
    setup_logging(config.get("logging", {}).get("level", "INFO"))
    set_seed(config.get("project", {}).get("seed", 42))
    
    sizes = config.get("matrix_generation", {}).get("sizes", [3, 4, 5])
    print(f"Generating stable matrices for sizes: {sizes}")
    
    for size in sizes:
        try:
            example = generate_stable_example(size, config)
            print(f"\nSize {size}x{size}:")
            print(f"  Attempts: {example['attempts']}")
            print(f"  Det: {example['determinant']:.5f}")
            print(f"  Cond: {example['condition_number']:.2f}")
            print(f"  Residual Norm: {example['residual_norm']:.1e}")
        except Exception as e:
            print(f"  Error generating size {size}: {e}")
