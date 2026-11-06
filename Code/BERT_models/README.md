# Transformer experiment workspace

This directory contains the BERT, ModernBERT, and RoBERTa experiments. Notebooks are separated from generated artifacts so it is clear which files are executable sources and which are results.

## Layout

```text
notebooks/
  training/                   Training, tuning, TAPT, and comparison notebooks
  analysis/                   Result-summary and visualization notebooks
experiments/
  baseline/
    summaries/                Cross-model baseline summary tables
    bert_base_uncased/
      seed13/                 Checkpoint, metrics, predictions, and histories
      seed42/
    bert_large_uncased/
      seed13/
      seed42/
    modernbert_base/
      seed13/
      seed42/
  lr_epoch_tuning/            Learning-rate, scheduler, and epoch-tuning runs
  tapt/
    masked_language_modeling/ Masked-language TAPT encoders and manifests
    downstream_comparison/    Standard versus TAPT RoBERTa classification runs
```

## Main entry points

- `notebooks/training/BERT_Soft_Label_Experiment_Runner.ipynb`: original soft-label baseline suite
- `notebooks/training/BERT_Soft_Label_LR_Epoch_Tuning.ipynb`: learning-rate, scheduler, and epoch tuning
- `notebooks/training/RoBERTa_Masked_Language_Modelling_TAPT.ipynb`: task-adaptive masked-language pretraining
- `notebooks/training/RoBERTa_TAPT_Comparison_Seed13.ipynb`: downstream standard-versus-TAPT comparison
- `notebooks/analysis/Important_Results_Summary.ipynb`: consolidated important results
- `notebooks/analysis/BERT_Visualisations.ipynb`: plots and result tables
- `notebooks/analysis/RoBERTa_TAPT_Results_Comparison_Seed13.ipynb`: final TAPT comparison report

## Output conventions

New baseline runs are written to `experiments/baseline/<model>/<run>/`, with cross-run tables in `experiments/baseline/summaries/`. Tuning and TAPT notebooks write to their corresponding experiment directories.

All notebooks discover the project root from `data/annotate-lime.csv`; they no longer depend on being launched from the former flat `Code/BERT_models` directory. Historical output text may describe the machine on which a run completed, while executable paths and result metadata point to the current organized locations.

## Safe execution

Training is opt-in. Running every cell with the default configuration loads existing summaries and histories but does not train or overwrite checkpoints.

- Baselines: set `RUN_BASELINE_EXPERIMENTS = True`; completed runs are skipped while `SKIP_COMPLETED_BASELINE = True`.
- LR/epoch tuning: set `RUN_SELECTED_TUNING = True`; completed runs are skipped while `SKIP_COMPLETED_TUNING = True`.
- Masked-language TAPT: set `RUN_TAPT = True`. The default is a smoke test. For a full run, deliberately also set `SMOKE_TEST = False`. Existing artifacts are protected by `SKIP_COMPLETED_TAPT` and `ALLOW_TAPT_OVERWRITE`.
- TAPT downstream comparison: set `RUN_TAPT_COMPARISON = True`; completed runs are skipped while `SKIP_COMPLETED_COMPARISON = True`.

Keep the skip flags enabled unless a deliberate rerun is required. For TAPT, prefer a new run name over overwriting an existing encoder.
