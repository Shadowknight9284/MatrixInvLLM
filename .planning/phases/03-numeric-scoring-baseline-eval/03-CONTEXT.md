# Phase 3: Numeric Scoring & Metrics - Context

**Gathered:** 2026-04-28
**Status:** Implementation-only (Phase 3 tightly scoped)

<domain>
## Phase Boundary
This phase focuses on the evaluation engine. It implements the logic to convert raw LLM text output back into NumPy matrices and compute all required scientific metrics.

- [x] `src/parse_prediction.py` (Part 1)
- [x] `src/score_prediction.py` (Part 1)
- [ ] `src/inference.py` (Part 2)
- [ ] `src/evaluate_model.py` (Part 2 - Baseline portion)
</domain>

<decisions>
## Implementation Decisions

### Tinker Integration
- **Decision 8**: Use `tinker.ServiceClient()` for all API interactions.
- **Decision 9**: Use the base model (e.g., Llama 3.2 3B) as configured in `config.yaml`.
- **Decision 10**: Implement a `SamplingClient` wrapper for robust inference.

### Baseline Evaluation
- **Decision 11**: Evaluation loop iterates over `data/test.jsonl`.
- **Decision 12**: Per-example metrics stored in a list for aggregation.
- **Decision 13**: Final report includes aggregate pass rates by tolerance band and matrix size.
- **Decision 14**: Save raw LLM responses alongside numeric scores for auditing.

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
