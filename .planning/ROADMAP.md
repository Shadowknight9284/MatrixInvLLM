# Roadmap: MatrixInverse-LLM

## Overview
This 2-day MVP pipeline takes the project from synthetic data generation to a full quantitative comparison between a baseline and a LoRA-fine-tuned model using the Tinker API.

## Phases

- [ ] **Phase 1: Foundation & Data Engine** - Generate stable synthetic matrix datasets.
- [ ] **Phase 2: Serialization & Splitting** - Create reproducible dataset splits.
- [ ] **Phase 3: Numeric Scoring & Baseline Eval** - Establish baseline performance with Tinker sampling.
- [ ] **Phase 4: Tinker LoRA Training** - Fine-tune the model specifically for inversion.
- [ ] **Phase 5: Final Comparison & Orchestration** - Run complete end-to-end experiment and report.

## Phase Details

### Phase 1: Foundation & Data Engine
**Goal**: Generate and serialize a high-quality synthetic dataset of invertible 3x3 to 5x5 matrices.
**Depends on**: Nothing
**Requirements**: DATA-01, DATA-02, DATA-03, DATA-04, SER-01, SER-02, SER-03
**Success Criteria**:
  1. Matrices generated are dense, square, and invertible.
  2. Gold inverses pass residual checks ($||AX - I|| < 10^{-8}$).
  3. `train.jsonl`, `val.jsonl`, and `test.jsonl` files are produced with strict formatting.
**Plans**: 1 plan

Plans:
- [x] 01-01: Implement `src/utils.py` and `src/generate_matrices.py`.

### Phase 2: Serialization & Splitting
**Goal**: Convert numeric matrices to machine-readable text and build train/val/test splits.
**Depends on**: Phase 1
**Requirements**: SER-01, SER-02, SER-03
**Success Criteria**:
  1. `train.jsonl`, `val.jsonl`, and `test.jsonl` files exist in `data/`.
  2. Formatting matches the strict spec (MatrixSize, Matrix:, Inverse:).
  3. Every record contains both `input_text`/`target_text` and numeric metadata.
**Plans**: 1 plan

Plans:
- [x] 02-01: Implement `src/serialize_dataset.py` and `src/build_splits.py`.

### Phase 3: Numeric Scoring & Baseline Eval
**Goal**: Establish a baseline performance level and implement robust evaluation logic.
**Depends on**: Phase 2
**Requirements**: EVAL-01, EVAL-02, EVAL-03, TINK-01, TINK-02
**Success Criteria**:
  1. Model responses can be parsed into numeric NumPy arrays reliably.
  2. Evaluation reports include MAE and Residual metrics.
  3. A baseline evaluation run using Tinker `SamplingClient` completes on the test set.
**Plans**: 2 plans

Plans:
- [x] 03-01: Implement `src/parse_prediction.py` and `src/score_prediction.py`.
- [x] 03-02: Implement `src/inference.py` (Tinker) and run baseline evaluation.

### Phase 4: Tinker LoRA Training
**Goal**: Fine-tune the model using the Tinker API's LoRA primitives.
**Depends on**: Phase 3
**Requirements**: TINK-03
**Success Criteria**:
  1. Tinker LoRA training job starts and completes successfully.
  2. Model weights are saved for post-train inference.
**Plans**: 1 plan

Plans:
- [ ] 04-01: Implement `src/train_lora.py` and execute the single-run fine-tuning job.

### Phase 5: Final Comparison & Orchestration
**Goal**: Orchestrate the full pipeline and quantify improvement post-training.
**Depends on**: Phase 4
**Requirements**: TINK-04, TINK-05, ORCH-01, ORCH-02
**Success Criteria**:
  1. Post-train evaluation completes on the same test set as baseline.
  2. A comparison report shows the performance delta by matrix size and tolerance.
  3. `src/run_experiment.py` can trigger the full flow.
**Plans**: 1 plan

Plans:
- [ ] 05-01: Implement `src/run_experiment.py` and generate final comparison reports.

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Foundation | 1/1 | Complete | 2026-04-28 |
| 2. Serialization | 1/1 | Complete | 2026-04-28 |
| 3. Eval & Baseline | 2/2 | Complete | 2026-04-28 |
| 4. LoRA Training | 0/1 | Not started | - |
| 5. Orchestration | 0/1 | Not started | - |

---
*Roadmap defined: 2026-04-28*
