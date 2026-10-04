# Results

Main run: **run 1**, `~/objectcore/runs/2026-10-01_2301_all` (1–2 Oct 2026, about 5 h 10 min on an A10G). It covers:
- Phase A, no detector fine-tuning
- Phase B, fine-tuning exactly as in the paper
- 5 categories × k = 1, 2, 4 × 3 random support draws

## Headline: MVTec LOCO I-AUROC (mean ± std over 3 draws)

| k | PatchCore (paper) | PromptAD (paper) | **ObjectCore (paper)** | **Ours, Phase B** | Ours, Phase A |
|---|---|---|---|---|---|
| 1 | 54.7 | 61.9 | 72.5 ± 0.2 | 60.83 ± 1.38 | 57.75 |
| 2 | 58.3 | 63.8 | 74.6 ± 0.6 | 66.25 ± 2.11 | 58.80 |
| 4 | 63.5 | 70.9 | 80.8 ± 1.3 | 63.48 ± 3.37 | 61.38 |

Pixel-level results (P-AUROC, P-F1-max) are close to the paper, and P-F1 is above the paper at every k. The full numbers are in `tables/table1_comparison_long.csv` from the AWS report.

## k = 4 per category (Phase B)

| Category | I-AUROC all | logical | structural | only M_S | only M_L |
|---|---|---|---|---|---|
| breakfast_box | 62.58 ± 5.3 | 69.50 | 56.21 | **76.7** | 60.1 |
| juice_bottle | 80.53 ± 10.1 | 79.99 | 81.35 | **83.8** | 70.6 |
| pushpins | 61.62 ± 1.8 | 68.10 | 54.34 | **63.9** | 61.3 |
| screw_bag | 50.92 ± 3.6 | 49.79 | 52.82 | **62.1** | 50.8 |
| splicing_connectors | 61.72 ± 2.6 | 58.60 | 65.67 | **64.9** | 59.9 |
| **Average** | **63.48 ± 3.4** | 65.20 | 62.08 | **70.3** (paper 68.7) | 60.5 (paper 70.6) |

## Contents

| Path | What |
|---|---|
| `metrics/run1_iauroc_summary.csv` | I-AUROC per phase and k, compared with the paper |
| `metrics/run1_phaseB_k4_per_category.csv` | the table above |
| `metrics/diag_finetune_summary.csv` | detector-collapse diagnostic (V1 paper loss vs V2 phrase-token fix) |
| `metrics/detector_collapse_run1_log_examples.csv` | objects before and after fine-tuning, taken from the run 1 logs |
| `tables/table1_comparison.csv` | paper Table 1 layout with our rows |
| `tables/ablation_comparison.csv` | paper Table 3 rows we can reproduce |
| `figures/summary/` | I-AUROC vs k, k = 4 branch comparison, collapse diagnostic (PNG + PDF) |
| `make_summary_figures.py` | rebuilds `figures/summary/` and `01_methodology/methodology_diagram.png` from the CSVs |
| `aws_report/` | **to add:** the full `report/` folder from the AWS run (see below) |
| `viz_steps/` | **to add:** step-by-step images for one test image (`viz_steps.py` output) |

## Files still on the AWS workspace
The following were produced on AWS and are added to this folder before the first commit:
- `runs/2026-10-01_2301_all/report/`, which holds:
  - `tables/`: `table1_comparison(_long).csv`, `ablation_comparison.csv`, `per_category.csv`, `per_draw.csv`, `context_fullshot.csv`
  - `graphs/`: g1–g6
  - `figures/`: `fig2_detections`, `fig3_qualitative_best` / `_random`, `per_category/*_examples`
  - `report.html`
- `runs/2026-10-01_2301_all/run_record.txt` and `logs/` (console logs and `summary_after_phase*.txt`)
- `~/objectcore/viz_steps/<category>_<image>/` (00–10 step images + `summary.png`)
- `~/objectcore/logs/diag_finetune.txt`, plus the fusion-check and objects-check outputs
- the Phase B phrase-fix rerun folder, if it has been run

Raw anomaly maps (`maps/*.npz`, 1–2 GB) are not committed.
