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

## Results at a glance

The table below uses the best recorded configuration for each model architecture in the committed summaries. Macro MAE and mean Pearson evaluate the predicted soft labels. Lower MAE and higher Pearson are better. Binary macro F1 is shown with its target definition because the TF-IDF and transformer experiments do not use exactly the same binary cutoff.

| Approach | Selected configuration | Seeds | Macro MAE ↓ | Mean Pearson ↑ | Binary macro F1 ↑ | Binary target |
|---|---|---:|---:|---:|---:|---|
| Prevalence baseline | Constant label prevalence | — | — | — | 0.084 | Vote fraction ≥ 0.60 |
| TF-IDF + tuned Ridge | Tuned word 1–2 grams | One split | 0.163 | 0.492 | 0.400 | Vote fraction ≥ 0.60, validation-tuned |
| ModernBERT Base | Original runner, LR `1e-5` | 13, 42 | 0.163 ± 0.001 | 0.443 ± 0.001 | 0.401 ± 0.032 | Majority |
| BERT Base | Linear schedule, LR `3e-5` | 42 | 0.145 | 0.593 | 0.443 | Majority |
| BERT Large | Linear schedule, LR `1e-5` | 13, 42 | 0.142 ± 0.006 | 0.566 ± 0.035 | 0.436 ± 0.038 | Majority |
| RoBERTa Large | Linear schedule, LR `1e-5` | 42 | **0.128** | **0.700** | **0.570** | Majority |
| RoBERTa Large + TAPT | Linear schedule, LR `1e-5` | 13 | 0.134 | 0.589 | 0.454 | Majority |

The tuned RoBERTa Large run has the strongest recorded result on all three headline metrics. It is a single-seed result, so it should be confirmed with additional seeds. Among configurations evaluated with two seeds, BERT Large has the lowest soft-label error, while its majority macro F1 is close to the selected BERT Base result. TF-IDF is a useful lightweight baseline, but its binary F1 is not directly interchangeable with the majority F1 because it uses a stricter target definition.

The source tables are the [TF-IDF model summary](Code/TF_IDF/outputs/tfidf_model_summary.csv), [ranked transformer configuration summary](Code/BERT_models/experiments/lr_epoch_tuning/important_key_results_by_config.csv), and [TAPT downstream comparison](Code/BERT_models/experiments/tapt/downstream_comparison/roberta_large_tapt_comparison_seed13_by_lr.csv). Values above are rounded to three decimals; `±` values are standard deviations across the two listed seeds.

### Does TAPT improve RoBERTa?

The controlled seed-13 comparison gives a mixed result:

| Variant | Macro MAE ↓ | Mean Pearson ↑ | Weak-presence macro F1 ↑ | Majority macro F1 ↑ | Strong-presence macro F1 ↑ |
|---|---:|---:|---:|---:|---:|
| Standard RoBERTa Large | **0.133** | **0.603** | **0.712** | 0.453 | **0.380** |
| RoBERTa Large + TAPT | 0.134 | 0.589 | 0.696 | **0.454** | 0.328 |

TAPT lowers validation loss and raises majority macro F1 by about 0.002, but it slightly worsens soft-label MAE and Pearson correlation and reduces the weak- and strong-presence F1 scores. On this one-seed comparison, TAPT does not provide a consistent overall improvement. The detailed metrics are in the [TAPT comparison table](Code/BERT_models/experiments/tapt/downstream_comparison/roberta_large_tapt_comparison_seed13_by_lr.csv).

### Approaches without a completed aggregate comparison

- **LLM prompting:** the repository contains 61 cached responses from DeepSeek V4 Pro and GLM 5.2 across zero-shot and few-shot conditions. The committed artifacts do not contain a completed ground-truth join or aggregate evaluation table, so no comparable performance conclusion is available yet. See the [cached LLM scores](Code/LLM/outputs/multi_model_prompting_raw_scores.csv).
- **Multitask model:** the notebook is an experimental scaffold. Training is disabled and no completed result artifacts were recovered, so its performance is not yet known.
- **Annotation reliability:** the IRR notebook evaluates annotation agreement rather than predictive performance and therefore does not belong in the model leaderboard.

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
