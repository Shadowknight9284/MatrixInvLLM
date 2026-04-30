import numpy as np
import logging
from src.utils import load_config, setup_logging

logger = logging.getLogger(__name__)

def compute_distance_metrics(pred_matrix, gold_matrix):
    """
    Compute element-wise distance metrics between predicted and gold matrices.
    """
    diff = pred_matrix - gold_matrix
    
    metrics = {
        "mae": float(np.mean(np.abs(diff))),
        "mse": float(np.mean(np.square(diff))),
        "frobenius_dist": float(np.linalg.norm(diff, ord='fro')),
        "max_abs_error": float(np.max(np.abs(diff)))
    }
    return metrics

def compute_residual_metrics(pred_matrix, a_matrix):
    """
    Compute functional residual norms: ||I - AX||_F and ||I - XA||_F.
    """
    n = a_matrix.shape[0]
    identity = np.eye(n)
    
    # Left residual: I - AX
    left_res = identity - np.matmul(a_matrix, pred_matrix)
    left_res_norm = float(np.linalg.norm(left_res, ord='fro'))
    
    # Right residual: I - XA
    right_res = identity - np.matmul(pred_matrix, a_matrix)
    right_res_norm = float(np.linalg.norm(right_res, ord='fro'))
    
    return {
        "residual_left": left_res_norm,
        "residual_right": right_res_norm,
        "residual_max": max(left_res_norm, right_res_norm)
    }

def check_tolerances(residual_norm, config):
    """
    Check if the residual norm passes at different tolerance levels.
    """
    eval_cfg = config.get("evaluation", {})
    
    # Try to get from specific keys first (Phase 5 alignment)
    tolerances = {
        "strict": eval_cfg.get("strict_tolerance", 1e-5),
        "medium": eval_cfg.get("medium_tolerance", 1e-3),
        "loose": eval_cfg.get("loose_tolerance", 1e-1)
    }
    
    # Fallback to tolerance_bands if explicitly defined
    if "tolerance_bands" in eval_cfg:
        tolerances.update(eval_cfg["tolerance_bands"])
    
    results = {}
    for level, tol in tolerances.items():
        results[f"pass_{level}"] = bool(residual_norm < tol)
        
    return results

def score_example(pred_text, gold_matrix, a_matrix, matrix_size, config):
    """
    Full scoring pipeline for a single example.

    Returns a flat dict with:
      parse_success, is_correct_shape, extracted_values_count,
      mae, mse, frobenius_dist, max_abs_error,
      residual_left, residual_right, residual_max,
      pass_strict, pass_medium, pass_loose
    """
    from src.parse_prediction import parse_to_matrix

    matrix, parse_meta = parse_to_matrix(pred_text, matrix_size)

    full_metrics = {
        "parse_success": parse_meta["parse_success"],
        "is_correct_shape": parse_meta["is_correct_shape"],
        "extracted_values_count": parse_meta["extracted_values_count"],
    }

    if not parse_meta["is_correct_shape"]:
        # Fill all numeric fields with sentinel values for schema consistency
        full_metrics.update({
            "mae": None,
            "mse": None,
            "frobenius_dist": None,
            "max_abs_error": None,
            "residual_left": None,
            "residual_right": None,
            "residual_max": None,
            "pass_strict": False,
            "pass_medium": False,
            "pass_loose": False,
        })
        return full_metrics

    # Shape is valid — run full numeric scoring
    logger.debug(
        f"score_example: valid {matrix_size}x{matrix_size} prediction — computing numerics."
    )
    dist_metrics = compute_distance_metrics(matrix, gold_matrix)
    res_metrics = compute_residual_metrics(matrix, a_matrix)
    tol_results = check_tolerances(res_metrics["residual_max"], config)

    full_metrics.update(dist_metrics)
    full_metrics.update(res_metrics)
    full_metrics.update(tol_results)

    return full_metrics

if __name__ == "__main__":
    # Internal Verification for Phase 3
    config = load_config()
    setup_logging("DEBUG")
    
    # Mock Example: 3x3
    size = 3
    A = np.array([[1, 2, 3], [0, 1, 4], [5, 6, 0]], dtype=float)
    X_gold = np.linalg.inv(A)
    
    # Correct Parse Case
    mock_response = "Inverse:\n" + "\n".join([", ".join([f"{val:.5f}" for val in row]) for row in X_gold])
    print(f"Mock Response:\n{mock_response}")
    
    scores = score_example(mock_response, X_gold, A, size, config)
    print("\nCorrect Parse Scores:")
    for k, v in scores.items():
        print(f"  {k}: {v}")
        
    # Malformed Case (wrong shape)
    mock_bad = "Inverse:\n1.0, 2.0, 3.0"
    scores_bad = score_example(mock_bad, X_gold, A, size, config)
    print("\nMalformed Parse Scores:")
    for k, v in scores_bad.items():
        print(f"  {k}: {v}")
