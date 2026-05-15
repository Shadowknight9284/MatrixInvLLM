# MatrixInvLLM Research Report

This directory contains a complete LaTeX research report on the MatrixInvLLM project, structured as an academic paper (3-5 pages).

## Structure

```
report/
├── report.tex                 # Main document (entry point)
├── sections/
│   ├── abstract.tex          # Paper abstract
│   ├── introduction.tex       # Research motivation and contribution
│   ├── methodology.tex        # Experimental framework and design
│   ├── experiments.tex        # Results and key findings
│   ├── discussion.tex         # Interpretation and implications
│   └── conclusion.tex         # Summary and future work
├── tables/
│   ├── stability_criteria.tex # Matrix generation parameters
│   ├── training_config.tex    # LoRA and training hyperparameters
│   └── results.tex            # Baseline vs. fine-tuned metrics
├── figures/
│   ├── pipeline.tex           # 6-stage pipeline diagram (TikZ)
│   └── results_chart.tex      # Results comparison chart (TikZ)
└── README.md                  # This file
```

## Compilation

### Requirements
- **LaTeX distribution**: TeX Live, MiKTeX, or MacTeX
- **Packages**: Standard (`amsmath`, `graphicx`, `booktabs`, `tikz`)

### Linux/macOS
```bash
cd report
pdflatex report.tex
pdflatex report.tex  # Run twice to resolve references and generate TOC
```

### Windows
```cmd
cd report
pdflatex report.tex
pdflatex report.tex
```

Or use an IDE like TeXworks, Overleaf, or VS Code with LaTeX Workshop extension.

### Online (No Installation)
Upload `report/` to [Overleaf](https://overleaf.com) and compile directly in the browser.

## Output

After compilation, you'll have:
- **report.pdf** — The final formatted research paper (4-5 pages)

## Content Overview

### Abstract
- Research question: Can LLMs learn matrix inversion via fine-tuning?
- Key finding: Model learned output **format** (87.5% parse success) but not **values** (0% numeric accuracy)
- Contribution: Demonstrates the syntax-semantics gap in LLM arithmetic reasoning

### Introduction
- Motivation: Why test LLM arithmetic reasoning?
- Research question and methodology overview
- Contribution to LLM understanding

### Methodology (Largest Section)
- **Matrix generation**: Stability filters (determinant, condition number, residual validation)
- **Data serialization**: Deterministic text format for reproducible parsing
- **Training**: LoRA fine-tuning on Llama-3.2-3B with 80 examples, 100 steps
- **Evaluation**: Syntactic (parse success, shape), numeric (MAE, residuals, tolerance-based accuracy), and failure modes

### Experiments & Results
- **Baseline**: 0% output (EOS_ONLY failure mode—untrained model produces nothing)
- **Fine-tuned**: 87.5% parse success, 0% numeric accuracy (all outputs are NUMERICALLY_WRONG)
- **Critical finding**: Format ≠ Function—the model learned to produce "inverse-shaped" outputs, not actual inverses

### Discussion
- Interpretation: What does this gap reveal about LLM arithmetic reasoning?
- Implications: Fine-tuning alone may be insufficient for precise computation
- Limitations: Small test set, limited scope to 3×3 matrices, no hyperparameter tuning
- Future directions: Hyperparameter search, larger matrices, alternative training paradigms

### Conclusion
- Summary: Syntax learned, semantics not learned
- Significance: Positioned as a diagnostic (what doesn't work) rather than a failure
- Next steps: Investigate architectural or training changes

## Key Figures & Tables

- **Table 1**: Stability criteria (determinant > 0.05, κ < 50, residual < 10⁻⁸)
- **Table 2**: Training config (LoRA rank 8, α 16, 100 steps, batch size 16)
- **Table 3**: Results comparison showing baseline (0%) vs. fine-tuned (87.5% parse, 0% numeric)
- **Figure 1**: 6-stage pipeline diagram (generation → serialization → split → baseline eval → train → re-eval)
- **Figure 2**: Results chart comparing baseline and fine-tuned performance across metrics

## Citation

If you reference this work:

```bibtex
@techreport{matrixinvllm2026,
  title={Can Small Language Models Learn Matrix Inversion?},
  subtitle={Exploring the Gap Between Syntactic and Semantic Understanding},
  author={MatrixInvLLM Project},
  year={2026},
  note={Experimental Report}
}
```

## Notes

1. **Page count**: The report is designed to fit 4-5 pages (typical for a research conference paper).
2. **Figures use TikZ**: All figures are vector graphics, compiled inline—no external images needed.
3. **Reproducibility**: Configuration and methodology are documented to enable reproduction; all hyperparameters are specified.
4. **Tone**: Academic but accessible; positions intermediate results as diagnostic rather than negative.

## Viewing Without Compilation

If you don't have LaTeX installed, you can:
- Use [Overleaf](https://overleaf.com) (free online LaTeX editor)
- Use [Papeeria](https://papeeria.com) or other online LaTeX services
- View a pre-compiled PDF (if provided separately)

---

**Report Generated**: May 14, 2026  
**Project**: MatrixInvLLM (Synthetic data LoRA fine-tuning for matrix inverse computation)
