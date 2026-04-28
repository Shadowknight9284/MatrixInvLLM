# Phase 2: Serialization & Splitting - Context

**Gathered:** 2026-04-28
**Status:** Implementation-only (Phase 2 small and focused)

<domain>
## Phase Boundary
This phase focuses on converting the numeric matrix data into the strict machine-readable text format required for LLM fine-tuning and establishing reproducible dataset splits.

- [ ] `src/serialize_dataset.py`
- [ ] `src/build_splits.py`
</domain>

<decisions>
## Implementation Decisions

### Serialization
- **Decision 1**: Exact headers: `MatrixSize:`, `Matrix:`, `Inverse:`.
- **Decision 2**: Format: One row per line, comma-separated values.
- **Decision 3**: Use `fixed_decimal_precision` from `config.yaml` (default 5).
- **Decision 4**: Input/Output headers from `serialization` section of `config.yaml`.
- **Decision 5**: Strict: No extra text or prose.

### Splitting
- **Decision 6**: Reproducible splits using `train_ratio`, `val_ratio`, `test_ratio`, and `split_seed` from `config.yaml`.
- **Decision 7**: Output JSONL files: `train.jsonl`, `val.jsonl`, `test.jsonl`.
- **Decision 8**: Each JSONL record includes:
    - `input_text`, `target_text`.
    - Numeric metadata: `matrix_size`, `determinant`, `condition_number`.

### the agent's Discretion
- Internal function structure for `serialize_dataset.py`.
- Methods for ensuring deterministic shuffling in `build_splits.py`.
</decisions>

<canonical_refs>
## Canonical References
- [README.md](file:///c:/dev/MatrixInvLLM/README.md) - Exact formatting rules.
- [config.yaml](file:///c:/dev/MatrixInvLLM/config.yaml) - Serialization and dataset split parameters.
- [.planning/PROJECT.md](file:///c:/dev/MatrixInvLLM/.planning/PROJECT.md) - Core values and requirement IDs.
</canonical_refs>

<deferred>
## Deferred Ideas
- Tinker API integration.
- Numeric scoring or base model evaluation.
</deferred>

---
*Phase: 02-serialization-splitting*
