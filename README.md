# Reappraisal Classification

This repository studies **multilabel classification of cognitive reappraisal strategies** in short text. Given a situation and a written reappraisal, the models estimate how strongly annotators would associate the response with each of eight strategies.

The repository contains the experiment notebooks and compact result artifacts for classical machine-learning, transformer, task-adaptive pretraining, large-language-model, and multitask approaches. 
## Classification task

Each reappraisal can express zero, one, or several strategies:

| Code | Strategy |
|---|---|
| `GM` | Growth mindset |
| `PR` | Positive repurposing |
| `A` | Aberration |
| `EJ` | External justification |
| `N` | Normalizing |
| `UC` | Unwarranted conclusions |
| `V` | Validation |
| `H` | Humor |

The main supervised experiments use **soft labels**: the proportion of valid annotators who marked each strategy as present. Binary metrics are secondary views of the same predictions using three interpretations:

- **Weak presence:** at least one annotator selected the strategy.
- **Majority:** more than half of valid annotators selected it.
- **Strong presence:** at least 80% of valid annotators selected it.

## Where to start

If you are new to the project, use this route:

1. Read [EXPERIMENTS.md](EXPERIMENTS.md) for a chronological inventory and the status of every experiment family.
2. Open [Model_comparison.ipynb](Code/comparison/Model_comparison.ipynb) for the cross-family comparison.
3. Open [Important_Results_Summary.ipynb](Code/BERT_models/notebooks/analysis/Important_Results_Summary.ipynb) for the main transformer result tables.
4. Read [Code/BERT_models/README.md](Code/BERT_models/README.md) before rerunning or extending a transformer experiment.

The latest experiment series is the RoBERTa task-adaptive pretraining comparison under `Code/BERT_models/experiments/tapt/`. TAPT uses masked-language modeling as its adaptation stage, which is why its encoder artifacts are nested under `tapt/masked_language_modeling/`.

## Repository map

```text
Code/
  TF_IDF/
    TFIDF_Ridge_Baseline.ipynb       Fast classical baseline
    outputs/                          Aggregate metrics and tuning results
  BERT_models/
    notebooks/training/               Training and tuning entry points
    notebooks/analysis/               Result summaries and visualizations
    experiments/baseline/             BERT, BERT Large, and ModernBERT runs
    experiments/lr_epoch_tuning/      Learning-rate and scheduler experiments
    experiments/tapt/
      masked_language_modeling/       RoBERTa TAPT stage
      downstream_comparison/          Standard vs. TAPT classifier results
  LLM/
    llm_model_comparison_baseline.ipynb
    outputs/                          Cached prompting results
  multitask/
    high_roi_multitask_heads_evaluation.ipynb
  comparison/
    Model_comparison.ipynb            Cross-family comparison
Data_exploration/
  IRR-reapp-annotations.ipynb         Inter-rater reliability analysis
data/
  README.md                           Data requirements and sharing policy
EXPERIMENTS.md                        Experiment inventory and chronology
MODEL_ARTIFACTS.md                    Model-weight storage notes
```

In general, `notebooks/` contains executable research code and `experiments/` or `outputs/` contains generated metrics, histories, manifests, and figures. The public result artifacts are intentionally small enough to inspect without downloading model checkpoints.

## Experiment families

| Family | Purpose | Main entry point |
|---|---|---|
| TF-IDF + Ridge | Lightweight reference baseline | `Code/TF_IDF/TFIDF_Ridge_Baseline.ipynb` |
| Transformer baselines | Soft-label BERT, BERT Large, and ModernBERT comparison | `Code/BERT_models/notebooks/training/BERT_Soft_Label_Experiment_Runner.ipynb` |
| Hyperparameter tuning | Learning-rate, scheduler, and epoch studies | `Code/BERT_models/notebooks/training/BERT_Soft_Label_LR_Epoch_Tuning.ipynb` |
| RoBERTa TAPT | Domain adaptation through masked-language modeling | `Code/BERT_models/notebooks/training/RoBERTa_Masked_Language_Modelling_TAPT.ipynb` |
| TAPT downstream comparison | Standard RoBERTa versus the adapted encoder | `Code/BERT_models/notebooks/training/RoBERTa_TAPT_Comparison_Seed13.ipynb` |
| LLM prompting | Prompt-based classification comparison | `Code/LLM/llm_model_comparison_baseline.ipynb` |
| Multitask prototype | Shared encoder with additional prediction heads | `Code/multitask/high_roi_multitask_heads_evaluation.ipynb` |
| Annotation analysis | Agreement and reliability of strategy annotations | `Data_exploration/IRR-reapp-annotations.ipynb` |



## Data required for rerunning experiments

The public repository does **not** include the raw research data. The training notebooks expect the following approved files in `data/`:

```text
data/
  annotate-lime.csv
  annotations-only-[GM].csv
  annotations-only-[PR].csv
  annotations-only-[A].csv
  annotations-only-[EJ].csv
  annotations-only-[N].csv
  annotations-only-[UC].csv
  annotations-only-[V].csv
  annotations-only-[H].csv
```

`annotate-lime.csv` supplies the text examples. Each semicolon-separated annotation matrix supplies the individual judgments used to construct vote counts and soft targets for one strategy. See [data/README.md](data/README.md) before adding or sharing any data.

Without the private data, a new reader can still inspect the notebooks and reproduce many of the published tables and visualizations from the committed aggregate result files. Full training or row-level evaluation requires authorized access to the dataset.

## Model artifacts

Trained PyTorch and Hugging Face weight files are excluded from Git. The repository retains compact configurations, tokenizer metadata, training histories, aggregate metrics, and figures where appropriate. See [MODEL_ARTIFACTS.md](MODEL_ARTIFACTS.md) for the excluded artifact inventory and recommended storage approach.

Historical metric files can refer to a checkpoint path even when that checkpoint is not present in the public clone. This does not affect reading the recorded metrics, but loading the trained model requires obtaining the corresponding artifact separately.


