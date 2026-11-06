# Reappraisal Classification

This folder is the canonical workspace for the reappraisal-classification experiments. It now contains the newer transformer experiments together with the unique LLM and multitask notebooks recovered from the older `Documents/Reappraisal` workspace.

## Project layout

```text
Code/
  BERT_models/       Organized transformer notebooks and experiment artifacts
  TF_IDF/            TF-IDF ridge baseline and outputs
  LLM/               LLM prompting comparison and its outputs
  multitask/         Experimental shared-encoder multitask model
  comparison/        Cross-model comparison notebook
Data_exploration/    Annotation and inter-rater-reliability analysis
data/                Input datasets, annotation material, and study data
Papers/
  library/            Imported reference-library export
  extracted_text/     Text extracted from papers for research notes/search
tmp/                 Temporary working files
```

See [EXPERIMENTS.md](EXPERIMENTS.md) for the experiment inventory, dates, status, and output locations.

## Repository publication policy

Git tracks the code, notebooks, small result tables, metrics, manifests, and documentation. Large model weights are excluded and should be published through dedicated artifact storage; see [MODEL_ARTIFACTS.md](MODEL_ARTIFACTS.md).

Raw research data is excluded by default pending a data-sharing and de-identification review. See [data/README.md](data/README.md). Third-party research PDFs are also kept outside the repository unless redistribution rights are confirmed.

## Running the imported notebooks

The imported notebooks locate this project automatically when Jupyter is started from this directory or one of its subdirectories. Optional environment variables can override their paths:

- `REAPPRAISAL_PROJECT_DIR`: project root
- `REAPPRAISAL_DATA_DIR`: directory containing `annotate-lime.csv`
- `REAPPRAISAL_LLM_OUTPUT_DIR`: LLM output directory
- `REAPPRAISAL_MULTITASK_OUTPUT_DIR`: multitask output directory

The LLM notebook has external API calls disabled by default. The multitask notebook has model training disabled by default. Review their configuration cells before enabling either operation.

## Consolidation note

The source folders in `~/Documents/Reappraisal` and `~/Downloads/Reappraisal` were not deleted or modified. They can remain as temporary backups until this consolidated workspace has been fully verified.
