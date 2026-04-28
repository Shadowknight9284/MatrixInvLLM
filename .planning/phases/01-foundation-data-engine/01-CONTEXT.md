# Phase 1: Foundation & Data Engine - Context

**Gathered:** 2026-04-28
**Status:** Implementation-only (Phase 1 reduced scope)

<domain>
## Phase Boundary
This phase focuses exclusively on the core numerical utilities and basic matrix generation logic. It establishes the "ground truth" for the research pipeline.

- [ ] `src/utils.py`
- [ ] `src/generate_matrices.py`
- [ ] Minimal directory setup strictly required for these files.
</domain>

<decisions>
## Implementation Decisions

### Matrix Generation
- **Decision 1**: Generate dense square matrices for sizes 3, 4, and 5 only (configured via `config.yaml`).
- **Decision 2**: Reject matrices that are:
    - Singular (rank check).
    - Near-singular (determinant < 0.05).
    - Poorly conditioned (condition number > 50).
- **Decision 3**: Use NumPy/SciPy for inverse computation.
- **Decision 4**: Functional validation: Verify $||AX - I|| < 10^{-8}$.

### Metadata Storage
- **Decision 5**: Store accepted examples with:
    - `id`, `matrix_size`, `matrix`, `inverse`, `determinant`, `condition_number`.
    - Ready for later serialization.

### the agent's Discretion
- Module structure for `utils.py` (logging, IO helpers).
- Naming conventions for internal generation functions.
</decisions>

<canonical_refs>
## Canonical References

- [README.md](file:///c:/dev/MatrixInvLLM/README.md) - System overview and system behavior specs.
- [config.yaml](file:///c:/dev/MatrixInvLLM/config.yaml) - Target sizes and stability thresholds.
- [.planning/research/STACK.md](file:///c:/dev/MatrixInvLLM/.planning/research/STACK.md) - Version constraints for NumPy/SciPy.
- [.planning/research/PITFALLS.md](file:///c:/dev/MatrixInvLLM/.planning/research/PITFALLS.md) - Warning about condition numbers.

</canonical_refs>

<deferred>
## Deferred Ideas
- Dataset serialization (`serialize_dataset.py`).
- Splitting logic (`build_splits.py`).
- Tinker API integration.
- Inference and Training modules.
</deferred>

---
*Phase: 01-foundation-data-engine*
