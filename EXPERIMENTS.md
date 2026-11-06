# Experiment inventory

Dates below describe the original experiment files, not the date they were consolidated into this workspace.

| Date | Experiment | Main notebook | Outputs | Status |
|---|---|---|---|---|
| 2026-07-20 | Annotation agreement / IRR | `Data_exploration/IRR-reapp-annotations.ipynb` | Notebook outputs | Existing analysis |
| 2026-07-31 | LLM prompting comparison | `Code/LLM/llm_model_comparison_baseline.ipynb` | `Code/LLM/outputs/` | Partial run: cached raw scores and one failed-response record were recovered; final comparison artifacts are not present |
| 2026-07-31 | High-ROI multitask heads | `Code/multitask/high_roi_multitask_heads_evaluation.ipynb` | `Code/multitask/outputs/` | Experimental scaffold; training is disabled and no recovered result artifacts were present |
| 2026-08-03 | TF-IDF ridge baseline | `Code/TF_IDF/TFIDF_Ridge_Baseline.ipynb` | `Code/TF_IDF/outputs/` | Baseline experiment and outputs present |
| 2026-08-03 | Initial model comparison | `Code/comparison/Model_comparison.ipynb` | Notebook outputs | Comparison notebook present |
| 2026-08-09 | BERT soft-label experiment suite | `Code/BERT_models/notebooks/training/BERT_Soft_Label_Experiment_Runner.ipynb` | `Code/BERT_models/experiments/baseline/` | Results present for BERT Base, BERT Large, and ModernBERT Base with seeds 13 and 42 |
| 2026-08-17 | BERT result summary and visualization | `Code/BERT_models/notebooks/analysis/Important_Results_Summary.ipynb`; `Code/BERT_models/notebooks/analysis/BERT_Visualisations.ipynb` | Notebook figures and summaries | Present |
| 2026-10-25 | Learning-rate and epoch tuning | `Code/BERT_models/notebooks/training/BERT_Soft_Label_LR_Epoch_Tuning.ipynb` | `Code/BERT_models/experiments/lr_epoch_tuning/` | Tuning artifacts present |
| 2026-10-26–27 | RoBERTa task-adaptive pretraining (TAPT) | `Code/BERT_models/notebooks/training/RoBERTa_Masked_Language_Modelling_TAPT.ipynb` | `Code/BERT_models/experiments/tapt/masked_language_modeling/` | Trained artifacts and manifests present |
| 2026-10-26–27 | RoBERTa TAPT classification comparison | `Code/BERT_models/notebooks/training/RoBERTa_TAPT_Comparison_Seed13.ipynb`; `Code/BERT_models/notebooks/analysis/RoBERTa_TAPT_Results_Comparison_Seed13.ipynb` | `Code/BERT_models/experiments/tapt/downstream_comparison/` | Latest experiment series; checkpoints, predictions, metrics, and comparison tables present |

## Recovered legacy material

The following unique items were copied into this workspace without altering their originals:

- LLM comparison notebook and partial outputs from `~/Documents/Reappraisal`
- High-ROI multitask notebook from `~/Documents/Reappraisal`
- Extracted research-paper text from `~/Documents/Reappraisal/tmp/pdfs`
- Reference-library export from `~/Downloads/Reappraisal`

The imported notebook paths were changed from machine-specific absolute paths to project-relative discovery. No existing BERT, TF-IDF, comparison, data, or result files were moved.

## Suggested next steps

1. Treat `Code/BERT_models/experiments/tapt/downstream_comparison/` as the latest completed experiment series.
2. Review the cached LLM scores before resuming API calls; the notebook deliberately keeps `RUN_API = False`.
3. Add the missing auxiliary emotion and intensity labels before enabling multitask training; the notebook deliberately keeps `RUN_TRAINING = False`.
4. Keep model checkpoints and large generated artifacts out of version control if this folder is later turned into a Git repository.
