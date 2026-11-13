# Roadmap

The night builder (Pranav's homelab) takes the first unchecked item each night, does it on a branch `agent/<slug>`,
and opens a PR. Pranav reviews and merges or closes it; one open PR at a time.
Tags: `[gpu]` may use the local RTX 5060 Ti overnight (3 h cap). `[you]` is Pranav's own work; the builder skips it.
Edit this file freely: reorder, reword, add items, or delete them.

- [ ] repo-hygiene: Stop tracking build output and caches
  Untrack LaTeX build files under report/ and slides/ (.aux .log .fls .fdb_latexmk .synctex.gz .out .toc .nav .snm .vrb)
  and every __pycache__/*.pyc; add them to .gitignore. Keep report.pdf, slides.pdf and all .tex sources.
- [ ] readme: Results-first README
  Move the current spec in README.md to docs/SPEC.md and drop the [cite:N] markers. New README: the question, the
  method in 4-5 lines, a results table built from outputs/metrics/experiment_summary.json (baseline vs fine-tuned),
  the main failure modes, how to reproduce, and links to report/report.pdf and slides/slides.pdf.
- [ ] tests-ci: Unit tests and CI
  pytest for generate_matrices (invertible, condition-number filter), serialize/parse round trip, score_prediction,
  and failure_classifier. GitHub Actions runs them on push and PR, CPU only, no API keys.
- [ ] local-backend: Local training and inference backend
  Add model.provider: local (Hugging Face transformers + PEFT LoRA) next to Tinker, using the same config keys.
  A CPU smoke test with a tiny ungated model runs in CI.
- [ ] local-baseline: [gpu] Reproduce the experiment locally
  Run baseline eval, LoRA fine-tune and post-train eval on an ungated ~3B model (e.g. Qwen2.5-3B) with the existing
  data and config. Save metrics under outputs/local/ and add a README table next to the Tinker Llama-3.2-3B numbers.
- [ ] compact-format: Simpler output format and length-controlled decoding
  Optional compact serialization (no headers, configurable decimals) and decoding that stops once n rows of n numbers
  are parsed, with max_new_tokens derived from n. Tests for both. (The report names format and looping as the main bottlenecks.)
- [ ] format-ablation: [gpu] Fine-tune with the compact format
  Same model and budget as local-baseline; compare parse rate, loop rate, sign agreement, relative error and
  residual pass rate against the original format.
- [ ] overfit-test: [gpu] Can the model memorize 32 inverses?
  Fine-tune on 32 training matrices until training loss plateaus; evaluate on those same 32. Separates model
  capacity from format and generalization problems.
- [ ] seeds: [gpu] Variance across 3 seeds
  Re-run the best configuration with 3 seeds; report mean and standard deviation for the main metrics.
- [ ] step-supervision: [you] Intermediate-step supervision
  Design how to supervise the computation (Gauss-Jordan steps, cofactors, or Newton-Schulz iterations), not only
  the final inverse. This is the research idea; once you write a spec here, untag it and the builder implements it.
- [ ] report-update: Update the report and slides with the new experiments
  Only after the experiment items above are merged. New results section and figures; keep the honest framing.
