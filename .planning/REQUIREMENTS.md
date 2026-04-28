# Requirements: MatrixInverse-LLM

**Defined:** 2026-04-28
**Core Value:** Demonstrate a clear performance delta between a baseline LLM and a LoRA-tuned LLM on a well-defined numerical task (3x3 to 5x5 matrix inversion) using the Tinker API.

## v1 Requirements

### Data Generation
- [ ] **DATA-01**: Generate random dense $n \times n$ matrices from uniform distribution.
- [ ] **DATA-02**: Compute gold inverse targets using NumPy/SciPy.
- [ ] **DATA-03**: Validate gold inverses using residual check ($||AX - I|| < 10^{-8}$).
- [ ] **DATA-04**: Filter matrices by determinant (min 0.05) and condition number (max 50).

### Serialization & Splitting
- [ ] **SER-01**: Serialize matrices into strict CSV-style text format with fixed decimal precision.
- [ ] **SER-02**: Create reproducible train, val, and test splits (80/10/10).
- [ ] **SER-03**: Export splits as JSONL for Tinker ingestion.

### Evaluation Engine
- [ ] **EVAL-01**: Implement robust regex-based parser for model's text outputs.
- [ ] **EVAL-02**: Compute per-example metrics (MAE, Frobenius norm, Residual norms).
- [ ] **EVAL-03**: Aggregate metrics by matrix size and tolerance buckets (Strict/Med/Loose).

### Model Operations (Tinker)
- [ ] **TINK-01**: Initialize `tinker.ServiceClient` and handle API key authentication.
- [ ] **TINK-02**: Run baseline inference on test set using `SamplingClient`.
- [ ] **TINK-03**: Execute LoRA fine-tuning using `LoraTrainingClient` on synthetic train set.
- [ ] **TINK-04**: Save and load fine-tuned weights for post-train assessment.
- [ ] **TINK-05**: Run post-train inference on test set.

### Orchestration
- [ ] **ORCH-01**: Single CLI command to run the full baseline -> train -> eval pipeline.
- [ ] **ORCH-02**: Export a final comparison report (markdown/json).

## v2 Requirements
- **STRET-01**: Support larger matrix sizes (8x8, 10x10).
- **STRET-02**: Implement symbolic inversion comparison.
- **STRET-03**: Web dashboard for real-time training progress and metric visualization.

## Out of Scope
| Feature | Reason |
|---------|--------|
| Sparse Matrices | High complexity, not needed for core research question. |
| Complex Valued Matrices | Outside current scope. |
| Automatic HPO | Time constraint (2-day MVP). |

## Traceability
(To be populated in ROADMAP)

---
*Requirements defined: 2026-04-28*
