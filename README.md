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
  paraphrasing/
    Paraphrase_Augmentation_Experiment.ipynb
    Final_AH_One_Per_Source_Augmentation.ipynb
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

## Experimental approach

The project does not test only one model family. It moves from an interpretable classical baseline to pretrained language models, then asks whether model adaptation or data adaptation adds value. All supervised approaches predict the same eight soft labels and are evaluated on held-out data.

```mermaid
flowchart LR
    A[Annotated reappraisals] --> B[Soft-label construction]
    B --> C[Fixed train / validation / test split]
    C --> D1[TF-IDF + Ridge]
    C --> D2[Transformer fine-tuning]
    C --> D3[LLM prompting]
    D2 --> E1[Hyperparameter tuning]
    D2 --> E2[Task-adaptive pretraining]
    C --> E3[Rare-label paraphrase augmentation]
    D1 --> F[Common metrics and comparison]
    E1 --> F
    E2 --> F
    E3 --> F
    D3 --> F
```

### Approaches and research questions

| Stage | Approach | Question being tested | Evidence in this repository | Status |
|---:|---|---|---|---|
| 1 | Annotation reliability | How consistently do annotators identify each strategy? | Agreement analysis with Gwet's AC1 and Krippendorff's alpha | Analysis present |
| 2 | TF-IDF + Ridge | How far can a fast, interpretable bag-of-words baseline go? | Tuned word n-grams, soft-label regression, and validation-tuned binary thresholds | Completed |
| 3 | Transformer baselines | Do contextual encoders improve on the classical baseline? | BERT Base, BERT Large, and ModernBERT runs across seeds 13 and 42 | Completed |
| 4 | Hyperparameter tuning | How sensitive are the transformers to learning rate, scheduler, and training duration? | Controlled sweeps with early stopping and stored per-epoch histories | Completed |
| 5 | RoBERTa + TAPT | Does additional masked-language-model training on in-domain text improve downstream classification? | Standard and task-adapted RoBERTa compared with the same split and training setup | Completed for the recorded seeds |
| 6 | Paraphrase augmentation | Can label-preserving paraphrases improve the rare `A` and `H` strategies? | Controlled original-versus-augmented protocol with unchanged validation and test sets | Pipeline prepared; final comparison pending |
| 7 | LLM prompting | Can instruction-following models classify the strategies without fine-tuning? | Cached zero-shot and few-shot responses | Partial; aggregate evaluation pending |
| 8 | Multitask learning | Can auxiliary prediction heads improve the shared representation? | Shared-encoder prototype | Scaffold only; training pending |

The main entry points are:

- [TF-IDF baseline](Code/TF_IDF/TFIDF_Ridge_Baseline.ipynb)
- [Transformer experiment runner](Code/BERT_models/notebooks/training/BERT_Soft_Label_Experiment_Runner.ipynb)
- [Learning-rate, scheduler, and epoch tuning](Code/BERT_models/notebooks/training/BERT_Soft_Label_LR_Epoch_Tuning.ipynb)
- [RoBERTa task-adaptive pretraining](Code/BERT_models/notebooks/training/RoBERTa_Masked_Language_Modelling_TAPT.ipynb) and [controlled downstream comparison](Code/BERT_models/notebooks/training/RoBERTa_TAPT_Comparison_Seed13.ipynb)
- [Rare-label paraphrase augmentation](Code/paraphrasing/Final_AH_One_Per_Source_Augmentation.ipynb)
- [LLM prompting comparison](Code/LLM/llm_model_comparison_baseline.ipynb)
- [Multitask prototype](Code/multitask/high_roi_multitask_heads_evaluation.ipynb)
- [Annotation reliability analysis](Data_exploration/IRR-reapp-annotations.ipynb)

## Experiment tracking and visualisation

The neural experiments produce reproducible local artifacts rather than relying only on notebook output:

- each run records its model configuration, learning rate, scheduler, random seed, best epoch, and evaluation metrics;
- per-epoch training and validation losses are saved as `*_training_history.csv` files;
- early stopping restores the checkpoint with the lowest validation loss;
- the analysis notebooks turn the saved histories into learning curves and cross-run comparison tables; and
- metrics and compact result artifacts remain inspectable without downloading full model checkpoints.

Use [BERT Visualisations](Code/BERT_models/notebooks/analysis/BERT_Visualisations.ipynb) for the full visual analysis and [Important Results Summary](Code/BERT_models/notebooks/analysis/Important_Results_Summary.ipynb) for the condensed comparison. The TAPT workflow additionally exports an MLM loss curve for each run.

> **Possible next improvement — Weights & Biases:** W&B is not currently connected to these notebooks. A useful extension would be to mirror the existing configuration, loss, and metric logs to W&B for live run comparison and optional completion/failure notifications. The committed CSV and JSON files should remain the reproducible source of truth.

## Results at a glance

The table below uses the best recorded configuration for each model architecture in the committed summaries. Macro MAE and mean Pearson evaluate the predicted soft labels. Lower MAE and higher Pearson are better. Binary macro F1 is shown with its target definition because the TF-IDF and transformer experiments do not use exactly the same binary cutoff.

### Current leaders

<table>
  <tr>
    <td align="center" width="33%">
      <strong>Best recorded overall</strong><br><br>
      <strong>RoBERTa Large</strong><br>
      <code>MAE 0.128 ↓</code><br>
      <code>Pearson 0.700 ↑</code><br>
      <code>Majority F1 0.570 ↑</code><br><br>
      <sub>Best on all three headline metrics · 1 seed</sub>
    </td>
    <td align="center" width="33%">
      <strong>Best replicated soft-label result</strong><br><br>
      <strong>BERT Large</strong><br>
      <code>MAE 0.142 ± 0.006 ↓</code><br>
      <code>Pearson 0.566 ± 0.035 ↑</code><br>
      <code>Majority F1 0.436 ± 0.038 ↑</code><br><br>
      <sub>Strongest MAE among configurations run with 2 seeds</sub>
    </td>
    <td align="center" width="33%">
      <strong>Best lightweight approach</strong><br><br>
      <strong>TF-IDF + tuned Ridge</strong><br>
      <code>MAE 0.163 ↓</code><br>
      <code>Pearson 0.492 ↑</code><br>
      <code>Binary F1 0.400 ↑</code><br><br>
      <sub>Fast classical baseline · binary target differs</sub>
    </td>
  </tr>
</table>

These labels summarize the evidence currently committed to the repository, not a final model ranking. In particular, the overall leader is a single-seed result, and TF-IDF's binary F1 is not directly comparable with transformer majority F1.

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

