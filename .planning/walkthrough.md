# Walkthrough: Phase 1 - Foundation & Data Engine

## Summary
Phase 1 of the MatrixInverse-LLM project successfully established the numerical foundation. We implemented a robust data generation engine that produces stable, invertible dense square matrices and validates their numerical inverses using residual checks.

## Changes Made
- **`src/utils.py`**: Implemented configuration loading (YAML), logging setup, and reproducibility seeding.
- **`src/generate_matrices.py`**: Implemented the core numerical logic:
    - Random matrix sampling from a uniform distribution.
    - Stability filtering based on matrix rank, determinant (min 0.05), and condition number (max 50.0).
    - Trusted inverse computation using NumPy/SciPy.
    - Rigorous functional validation via Frobenius norm residual checks ($||AX - I||$).

## Verification Results
Executed `python src/generate_matrices.py` and confirmed successful generation for configured sizes:
- **3x3**: Residual norm ~ $10^{-16}$.
- **4x4**: Residual norm ~ $10^{-16}$.
- **5x5**: Residual norm ~ $10^{-16}$.

All gold inverses meet the required tolerance ($10^{-8}$) by several orders of magnitude.

## Phase 2: Serialization & Splitting - Complete
Phase 2 established the data conversion and splitting logic, effectively turning raw numeric matrices into training-ready JSONL files.

### Changes Made
- **`src/generate_matrices.py`**: Added `generate_raw_dataset` to orchestrate multi-size matrix generation.
- **`src/serialize_dataset.py`**: Implemented strict text conversion using headers (`MatrixSize:`, `Matrix:`, `Inverse:`) and fixed decimal precision (5).
- **`src/build_splits.py`**: Implemented deterministic shuffling and 80/10/10 splitting.

### Verification Results
Ran `python src/build_splits.py --test-run` and confirmed:
- `data/train.jsonl`, `data/val.jsonl`, and `data/test.jsonl` were created.
- JSON structure includes `input_text`, `target_text`, and numeric metadata (`determinant`, `condition_number`).
- Text formatting exactly matches the project spec.

## Next Steps
Phase 3 will focus on implementation the evaluation engine (parser and scorers) and running the first baseline evaluation using the Tinker Sampling API on the `test.jsonl` set.
