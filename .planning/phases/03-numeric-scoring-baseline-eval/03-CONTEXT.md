# Phase 3: Numeric Scoring & Metrics - Context

**Gathered:** 2026-04-28
**Status:** Implementation-only (Phase 3 tightly scoped)

<domain>
## Phase Boundary
This phase focuses on the evaluation engine. It implements the logic to convert raw LLM text output back into NumPy matrices and compute all required scientific metrics.

- [ ] `src/parse_prediction.py`
- [ ] `src/score_prediction.py`
</domain>

<decisions>
## Implementation Decisions

### Parsing Logic
- **Decision 1**: Handle malformed outputs by searching for the "Inverse:" header or just searching for rows of floats.
- **Decision 2**: Enforce exactly `matrix_size` rows and columns.
- **Decision 3**: Store parsing state: `parse_success` (bool), `is_correct_shape` (bool).

### Scoring Metrics
- **Decision 4**: Compute element-wise MAE and MSE.
- **Decision 5**: Compute Frobenius norm of distance $||X_{pred} - X_{gold}||_F$.
- **Decision 6**: Compute left and right residuals ($||I - AX||_F$, $||I - XA||_F$).
- **Decision 7**: Pass/Fail against strict (1e-5), medium (1e-3), and loose (1e-1) tolerances from `evaluation` section of `config.yaml`.

### the agent's Discretion
- Regex patterns for float extraction.
- Structuring the final metrics output object (e.g., Pydantic model or dict).
</decisions>

<canonical_refs>
## Canonical References
- [README.md](file:///c:/dev/MatrixInvLLM/README.md) - Metric definitions and scientific framing.
- [config.yaml](file:///c:/dev/MatrixInvLLM/config.yaml) - Evaluation tolerances and precision settings.
- [.planning/REQUIREMENTS.md](file:///c:/dev/MatrixInvLLM/.planning/REQUIREMENTS.md) - EVAL-01, EVAL-02, EVAL-03 IDs.
</canonical_refs>

<deferred>
## Deferred Ideas
- Tinker API integration.
- Full dataset evaluation loops.
</deferred>

---
*Phase: 03-numeric-scoring-baseline-eval*
